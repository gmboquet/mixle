"""Train, calibrate, and serve a task model from an existing teacher.

``solve`` converts a callable teacher into a deployable :class:`Solution`.
The teacher may be a rule cascade, legacy scoring routine, API client, or any
other callable that currently performs the task. The solver labels example
inputs with that teacher, trains a student matched to the input shape, calibrates
an answer-or-escalate rule on held-out data, verifies agreement against the
teacher, and returns a callable object that answers locally when calibrated
confidence is sufficient.

Escalated requests remain useful after deployment. They are teacher-labeled
examples from the part of the input space where the student abstained.
``Solution.improve()`` re-distills with those harvested labels and promotes the
new student only when it preserves the verified agreement and escalation gates.

    def route(ticket): ...                      # existing production rule or service
    sol = solve(route, tickets)                 # dataset <- route(t) for t in tickets; train; calibrate
    sol(ticket)                                 # answer locally or escalate to route()
    sol.report()                                # agreement, escalation rate, realized cost
    sol.improve()                               # fold escalations back in; promote only if better

``solve`` is deterministic given ``seed``. Only student training requires the
optional neural dependency; the teacher remains an external callable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mixle.task.calibrate import CalibratedTaskModel
from mixle.task.cascade import Cascade
from mixle.task.density import DensityGate
from mixle.task.distill import agreement, distill_from_labels, distill_records_from_labels
from mixle.task.model import HashedNGram, HashedRecord, TaskModel
from mixle.task.tune import RecipeSpace


def _label_with(teacher: Callable[..., Any], items: list) -> list:
    """Label ``items`` with either a per-item or batched teacher callable."""
    try:
        out = teacher(items)
        if isinstance(out, (list, tuple)) and len(out) == len(items):
            return list(out)
    except Exception:
        pass
    return [teacher(x) for x in items]


def _batch_view(teacher: Callable[..., Any]) -> Callable[[list], list]:
    """Return a strict ``list -> list`` teacher view for cascade probes."""

    def batched(batch: list) -> list:
        return _label_with(teacher, list(batch))

    return batched


def load_harvested(path: str) -> tuple[list, list]:
    """Read harvested serving feedback into ``(inputs, answers)``.

    Two JSONL formats are supported: ``{"input": ..., "label": ...}`` for
    classification feedback and ``{"input": ..., "answer": ...}`` for solution
    feedback. Classification labels are string-coerced; solution answers keep
    their JSON shape. Input JSON lists are restored as tuples so record-shaped
    examples can be passed back into solve/distillation workflows.
    """
    inputs: list = []
    answers: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            x = row["input"]
            inputs.append(tuple(x) if isinstance(x, list) else x)
            answers.append(str(row["label"]) if "label" in row else row["answer"])
    return inputs, answers


def _input_kind(x: Any) -> str:
    """Infer whether one input should use the text or record student path."""
    if isinstance(x, str):
        return "text"
    if isinstance(x, (dict, tuple, list)):
        return "record"
    raise TypeError(
        "solve() handles text or record (tuple/dict) inputs; got %r. Pass kind='text'|'record' to override."
        % type(x).__name__
    )


def _fit_student(kind: str, inputs: list, labels: list, distill_kw: dict) -> TaskModel:
    kw = dict(distill_kw)
    student = kw.pop("student", "mlp")
    if student == "generative":
        if kind == "text":
            from mixle.task.generative_text import distill_text_generative_from_labels

            keep = {k: v for k, v in kw.items() if k in ("labels", "pseudo_count", "min_count", "task")}
            return distill_text_generative_from_labels(inputs, labels, **keep)
        from mixle.task.distill import distill_structured_from_labels

        keep = {
            k: v
            for k, v in kw.items()
            if k in ("labels", "n_components", "min_gain", "n_bins", "max_its", "seed", "task")
        }
        return distill_structured_from_labels(inputs, labels, **keep)
    fit = distill_from_labels if kind == "text" else distill_records_from_labels
    if kind != "text":  # the n-gram order is a text-only knob
        kw = {k: v for k, v in kw.items() if k != "n"}
    return fit(inputs, labels, **kw)


def _fit_gate(kind: str, inputs: list, alpha: float, seed: int, dim: int = 256) -> DensityGate:
    """Fit the p(x) OOD gate over the training inputs (text n-grams or hashed records)."""
    feat = HashedNGram(n=3, dim=dim, seed=seed) if kind == "text" else HashedRecord(dim=dim, seed=seed)
    return DensityGate(feat).fit(inputs, alpha=alpha, seed=seed)


def _synthesize_inputs(real_inputs: list, n: int, seed: int) -> list:
    """Sample ``n`` fresh synthetic inputs from a generative model fit to the real inputs.

    mixle's home turf: the input space is heterogeneous records, so infer a generative model of it
    (:func:`mixle.utils.automatic.get_estimator`), fit, and sample. Dedup against the real inputs and
    within the draw -- the synthetic inputs are only ever *teacher-labeled*, so labels stay real."""
    from mixle.inference import optimize
    from mixle.utils.automatic import get_estimator

    gen = optimize(real_inputs, get_estimator(real_inputs), max_its=25, out=None, rng=np.random.RandomState(seed))
    draws = gen.sampler(seed=seed).sample(max(n + n // 2, n))  # oversample; dedup below
    seen = {repr(x) for x in real_inputs}
    out: list = []
    for x in draws:
        r = repr(x)
        if r not in seen:
            seen.add(r)
            out.append(x)
        if len(out) >= n:
            break
    return out


def _tune_recipe(kind: str, inputs: list, labels: list, distill_kw: dict, budget: int, seed: int) -> dict:
    """Teacher-free recipe search: BO over (dim, hidden, epochs, lr) maximizing agreement on a val slice.

    The labels are already computed, so candidates cost only student fits — the teacher is never re-called.
    The val slice is carved from the *training* inputs; the calibration slice stays untouched, so the
    conformal guarantee downstream is unaffected by selection."""
    from mixle.doe import minimize

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(inputs))
    n_val = max(2, len(inputs) // 4)
    val_idx, fit_idx = order[:n_val], order[n_val:]
    fit_in, fit_lab = [inputs[i] for i in fit_idx], [labels[i] for i in fit_idx]
    val_in, val_lab = [inputs[i] for i in val_idx], [labels[i] for i in val_idx]

    space = RecipeSpace()
    trials: list[tuple[float, dict]] = []

    def objective(point: np.ndarray) -> float:
        recipe = {**distill_kw, **space.decode(point), "seed": seed}
        student = _fit_student(kind, fit_in, fit_lab, recipe)
        score = agreement(student, val_lab, val_in)
        trials.append((score, recipe))
        return score

    n_init = min(3, max(1, budget // 2))
    minimize(objective, space.bounds(), n_init=n_init, n_iter=max(0, budget - n_init), seed=seed, maximize=True)
    return max(trials, key=lambda t: t[0])[1]


@dataclass
class Solution:
    """A deployed task: a calibrated student in front of the teacher, plus the loop to improve it.

    Call it like the original function. ``promoted`` says whether the student passed verification --
    when False the callable simply runs the teacher instead of deploying an unverified student.
    """

    cascade: Cascade
    teacher: Callable[..., Any]
    kind: str
    train_inputs: list
    train_labels: list
    cal_inputs: list
    cal_labels: list
    holdout_agreement: float
    escalation_rate: float
    promoted: bool
    target_agreement: float | None
    distill_kw: dict = field(default_factory=dict)
    ood: float | None = None  # OOD-floor quantile the gate was fit with (None = no gate)
    seed: int = 0
    synthesized: int = 0  # synthetic (generative-sampled, teacher-labeled) inputs in the training set
    gate_inputs: list = field(default_factory=list)  # real inputs only -- what the p(x) gate is fit on
    edge: Any = None  # EdgeDistillResult when solve() ran under a DeviceSpec (footprint, pareto, design)

    def __call__(self, x: Any) -> Any:
        if not self.promoted:
            return _label_with(self.teacher, [x])[0]
        return self.cascade(x)

    def report(self) -> dict:
        """What you would want on a dashboard: verification, live escalation, realized cost."""
        stats = self.cascade.stats
        out = {
            "promoted": self.promoted,
            "holdout_agreement": round(self.holdout_agreement, 4),
            "holdout_escalation_rate": round(self.escalation_rate, 4),
            "requests": stats.n_requests,
            "live_escalated": stats.n_escalated,
            "harvested_labels": len(stats.escalated_labels),
            "synthesized_inputs": self.synthesized,
        }
        if self.edge is not None:
            out["device"] = {
                "feasible": self.edge.feasible,
                "family": self.edge.family,
                "bytes": self.edge.footprint.bytes,
                "ops": self.edge.footprint.ops,
                "torch_free": self.edge.footprint.torch_free,
            }
        return out

    def improve(self) -> bool:
        """Re-distill with the harvested (escalated) labels; promote only if it verifies at least as well.

        Returns True when a better student was promoted. The calibration slice is never trained on, so the
        conformal guarantee and the agreement comparison remain valid across rounds.
        """
        if not self.cal_inputs:
            raise RuntimeError(
                "this Solution was loaded from an artifact and has no training/calibration data; "
                "collect cascade.harvested() and re-solve(real + harvested inputs) to improve."
            )
        new_inputs, new_labels = self.cascade.harvested()
        if not new_inputs:
            return False
        inputs = self.train_inputs + list(new_inputs)
        labels = self.train_labels + [str(y) for y in new_labels]
        student = _fit_student(self.kind, inputs, labels, self.distill_kw)
        alpha = self.cascade.model.alpha
        # the gate stays real-inputs-only: harvested escalations are real, synthetic training rows are not
        gate_inputs = self.gate_inputs + list(new_inputs)
        gate = _fit_gate(self.kind, gate_inputs, self.ood, self.seed) if self.ood is not None else None
        cal = CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(self.cal_inputs, self.cal_labels)
        agree = agreement(student, self.cal_labels, self.cal_inputs)
        esc = cal.escalation_rate(self.cal_inputs)
        if agree < self.holdout_agreement or esc > self.escalation_rate:
            return False  # anti-regression: keep the current student
        self.cascade.model = cal
        self.train_inputs, self.train_labels = inputs, labels
        self.gate_inputs = gate_inputs
        self.holdout_agreement, self.escalation_rate = agree, esc
        self.promoted = self.promoted or self._passes_target(agree)
        self.cascade.stats.escalated_texts.clear()
        self.cascade.stats.escalated_labels.clear()
        return True

    def _passes_target(self, agree: float) -> bool:
        return self.target_agreement is None or agree >= self.target_agreement

    def health(self, recent_inputs: Any = None, *, p_threshold: float = 0.01) -> dict[str, Any]:
        """Check whether live escalation behavior has drifted from calibration.

        The conformal answer-or-escalate rule is calibrated under an
        exchangeability assumption. When the input distribution shifts, the live
        escalation rate may move away from the verified baseline. This method
        compares the live rate with the baseline using an exact binomial test
        and, when ``recent_inputs`` and an OOD gate are available, compares the
        gate hit rate with its design quantile.

        Returns a dictionary with ``drifted``, live and baseline rates, and
        p-values where enough observations are available. A drift alarm means
        traffic has changed and retraining or review may be needed; abstained
        inputs still route to the teacher.
        """
        from scipy.stats import binomtest

        stats = self.cascade.stats
        out: dict[str, Any] = {
            "requests": stats.n_requests,
            "live_escalation_rate": (stats.n_escalated / stats.n_requests) if stats.n_requests else float("nan"),
            "baseline_escalation_rate": self.escalation_rate,
            "drifted": False,
        }
        if stats.n_requests >= 20 and np.isfinite(self.escalation_rate):
            p = float(binomtest(stats.n_escalated, stats.n_requests, max(min(self.escalation_rate, 1.0), 1e-9)).pvalue)
            out["escalation_p_value"] = p
            out["drifted"] = p < p_threshold
        gate = self.cascade.model.density_gate
        if recent_inputs is not None and gate is not None and self.ood is not None:
            rows = list(recent_inputs)
            if rows:
                hit = float(np.mean(gate.ood_mask(rows)))
                out["live_ood_rate"] = hit
                out["design_ood_rate"] = float(self.ood)
                p_ood = float(binomtest(int(round(hit * len(rows))), len(rows), max(self.ood, 1e-9)).pvalue)
                out["ood_p_value"] = p_ood
                out["drifted"] = bool(out["drifted"] or p_ood < p_threshold)
        return out

    def save(self, path: str) -> str:
        """Persist the calibrated student as a load-anywhere artifact, with its verification record.

        Every deployed artifact carries how it was verified — held-out agreement with the teacher, the
        escalation rate, the conformal alpha, and how much of its training data was synthetic — so "is
        this model trustworthy" is answerable from the artifact alone."""
        task = self.cascade.model.task
        task.meta = {
            **task.meta,
            "solve": {
                "kind": self.kind,
                "ood": self.ood,
                "verification": {
                    "holdout_agreement": self.holdout_agreement,
                    "holdout_escalation_rate": self.escalation_rate,
                    "alpha": self.cascade.model.alpha,
                    "promoted": self.promoted,
                    "n_train": len(self.train_inputs),
                    "n_calibration": len(self.cal_inputs),
                    "synthesized_inputs": self.synthesized,
                    "verified_at": time.time(),
                },
            },
        }
        return self.cascade.model.save(path)

    def deploy(self, name: str, root: str = "./mixle_data/registry") -> str:
        """Save into the serving layout — ``{root}/tasks/{name}`` — the directory the mixle-mlops
        ``/v1/tasks`` routes serve from. Returns the artifact path."""
        return self.save(str(Path(root) / "tasks" / name))

    @classmethod
    def load(cls, path: str, teacher: Callable[..., Any], *, cost: Any = None, device: str = "cpu") -> Solution:
        """Reconstitute a *serving* Solution from a saved artifact — the deploy path for a fresh process.

        The loaded Solution answers locally / escalates to ``teacher`` and harvests labels exactly like
        the original. It carries no training or calibration data, so :meth:`improve` raises — collect the
        harvested pairs and re-``solve`` (real + harvested inputs) to train the next round."""
        cal = CalibratedTaskModel.load(path, device=device)
        meta = (cal.task.meta or {}).get("solve", {})
        return cls(
            cascade=Cascade(cal, _batch_view(teacher), cost=cost),
            teacher=teacher,
            kind=str(meta.get("kind", "text")),
            train_inputs=[],
            train_labels=[],
            cal_inputs=[],
            cal_labels=[],
            holdout_agreement=float("nan"),
            escalation_rate=float("nan"),
            promoted=True,  # only verified solutions should be saved; loading one serves it
            target_agreement=None,
            ood=meta.get("ood"),
        )


def solve(
    teacher: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    alpha: float = 0.1,
    target_agreement: float | None = None,
    holdout: float = 0.25,
    kind: str | None = None,
    ood: float | None = 0.02,
    propose: str | None = None,
    propose_budget: int = 8,
    synthesize: int = 0,
    prelabeled: tuple[Sequence[Any], Sequence[Any]] | None = None,
    device: Any = None,
    device_space: Any = None,
    cost: Any = None,
    seed: int = 0,
    **distill_kw: Any,
) -> Solution:
    """Replace ``teacher`` (the code currently doing the job) with a calibrated, self-improving model.

    Args:
        teacher: The callable performing the task today (per-item or batched). It labels the dataset and
            remains the fallback for inputs the student does not handle confidently.
        inputs: Example inputs (text, or tuple/dict records) covering the task. The teacher labels them.
        alpha: Escalation honesty -- answer locally only when a single label is conformally covered at
            ``>= 1 - alpha``; otherwise fall back to the teacher.
        target_agreement: Optional gate. If the student's held-out agreement with the teacher misses it,
            the returned Solution routes *everything* to the teacher (``promoted=False``).
        holdout: Fraction reserved for calibration + verification (never trained on).
        kind: Force the student path, ``'text'`` or ``'record'``; default sniffs the first input.
        ood: Fit a ``p(x)`` gate over the training inputs and escalate inputs whose ``log p(x)`` falls
            below this quantile floor — so a wildly novel input escalates even when the softmax looks
            confident. On by default (0.02); ``None`` disables.
        propose: ``"auto"`` searches the student recipe (dim/hidden/epochs/lr, Bayesian-optimized on a
            val slice carved from the training split) instead of using the defaults. Teacher-free — the
            labels are already computed, so candidates cost only student fits.
        propose_budget: Total candidate recipes tried when ``propose="auto"``.
        synthesize: When example inputs are scarce, sample this many *synthetic* inputs from a generative
            model fit to the real training inputs (record inputs only) and have the teacher label them.
            Labels are always real (teacher-produced); the calibration slice and the OOD gate stay
            real-inputs-only, so the conformal guarantee and the p(x) floor reflect the true distribution.
        prelabeled: Already-teacher-labeled ``(inputs, labels)`` pairs — typically
            ``load_harvested("harvested.jsonl")`` from a serving deployment — folded into the TRAINING
            split (and the OOD gate: they are real traffic) but never into calibration, which stays a
            fresh split of ``inputs``. This is the re-solve half of the serving loop.
        device: A :class:`~mixle.task.edge.DeviceSpec` makes this "give me this capability on that
            device": the student is found by :func:`~mixle.task.edge.distill_for_edge` — a structure x
            precision x recipe search under the device's hard byte/ops/torch-free budget (reusing the
            already-computed labels; the teacher is not re-called) — and the result's footprint,
            Pareto front, and design ledger land on ``Solution.edge``. If nothing fits the budget the
            Solution is demoted (everything routes to the teacher). Incompatible
            with ``propose="auto"`` (the device search subsumes it). A plain string (e.g. ``"cpu"``)
            keeps its old meaning: the torch training device.
        device_space: Optional :class:`~mixle.task.edge.EdgeSpace` constraining the device search
            (families, size ranges, precisions); default spans the standard space.
        cost: Optional :class:`~mixle.task.economics.CostModel` for realized-savings reporting.
        seed: Split + fit determinism.
        **distill_kw: Student knobs forwarded to distillation (``dim``, ``hidden``, ``epochs``, ``lr``, …).
            ``student="generative"`` swaps the hashed-feature MLP for mixle's generative student —
            per-class token models for text (:mod:`mixle.task.generative_text`) or the structure-learned
            joint for records (:func:`~mixle.task.distill.distill_structured_from_labels`): exact
            posteriors, no torch needed at inference, and a built-in ``log p(x)``.

    Returns:
        A :class:`Solution` -- call it like the original function; ``report()`` / ``improve()`` / ``save()``.
    """
    if isinstance(device, str):  # back-compat: solve(..., device="cpu") is the torch training device
        distill_kw["device"] = device
        device = None
    if device is not None and propose == "auto":
        raise ValueError("device= runs its own structure x recipe search; drop propose='auto'")

    items = list(inputs)
    if len(items) < 8:
        raise ValueError("solve() needs at least 8 example inputs to train and calibrate honestly")
    k = kind or _input_kind(items[0])
    labels = [str(y) for y in _label_with(teacher, items)]

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_cal = max(2, int(round(len(items) * holdout)))
    cal_idx, train_idx = order[:n_cal], order[n_cal:]
    train_inputs = [items[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    cal_inputs = [items[i] for i in cal_idx]
    cal_labels = [labels[i] for i in cal_idx]

    if prelabeled is not None:
        pre_in, pre_lab = prelabeled
        if len(pre_in) != len(pre_lab):
            raise ValueError("prelabeled inputs and labels must have equal length")
        train_inputs = train_inputs + list(pre_in)
        train_labels = train_labels + [str(y) for y in pre_lab]

    n_synth = 0
    gate = _fit_gate(k, train_inputs, ood, seed) if ood is not None else None  # real inputs only
    if synthesize:
        if k == "text":
            raise ValueError(
                "synthesize= samples a generative model of the inputs, which needs record inputs; "
                "for text, provide more examples (or synthesize upstream with an LLM) instead."
            )
        synth = _synthesize_inputs(train_inputs, int(synthesize), seed)
        if synth:
            train_inputs = train_inputs + synth
            train_labels = train_labels + [str(y) for y in _label_with(teacher, synth)]
            n_synth = len(synth)

    distill_kw.setdefault("seed", seed)
    edge_result = None
    if device is not None:
        # "this capability on that device": structure x precision x recipe search under the hard
        # budget, on the labels already computed -- the teacher is never re-called.
        from mixle.task.edge import distill_for_edge

        n_val = max(2, len(train_inputs) // 4)
        val_order = np.random.RandomState(seed).permutation(len(train_inputs))
        v_idx, f_idx = val_order[:n_val], val_order[n_val:]
        edge_result = distill_for_edge(
            None,
            [train_inputs[i] for i in f_idx],
            [train_inputs[i] for i in v_idx],
            device,
            train_labels=[train_labels[i] for i in f_idx],
            val_labels=[train_labels[i] for i in v_idx],
            labels=sorted(set(train_labels)),
            space=device_space,
            n_init=min(4, max(2, propose_budget // 2)),
            n_iter=max(1, propose_budget - min(4, max(2, propose_budget // 2))),
            seed=seed,
        )
        student = edge_result.model
    else:
        if propose == "auto":
            distill_kw = _tune_recipe(k, train_inputs, train_labels, distill_kw, propose_budget, seed)
        student = _fit_student(k, train_inputs, train_labels, distill_kw)
    cal = CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(cal_inputs, cal_labels)
    agree = agreement(student, cal_labels, cal_inputs)
    esc = cal.escalation_rate(cal_inputs)
    promoted = target_agreement is None or agree >= target_agreement
    if edge_result is not None and not edge_result.feasible:
        promoted = False  # nothing fit the device: serve the teacher, never a budget-busting student

    return Solution(
        cascade=Cascade(cal, _batch_view(teacher), cost=cost),
        teacher=teacher,
        kind=k,
        train_inputs=train_inputs,
        train_labels=train_labels,
        cal_inputs=cal_inputs,
        cal_labels=cal_labels,
        holdout_agreement=float(agree),
        escalation_rate=float(esc),
        promoted=bool(promoted),
        target_agreement=target_agreement,
        distill_kw=dict(distill_kw),
        ood=ood,
        seed=seed,
        synthesized=n_synth,
        gate_inputs=list(train_inputs[: len(train_inputs) - n_synth]),
        edge=edge_result,
    )
