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
from hashlib import sha256
from typing import Any

import numpy as np

from mixle.task._ledger import CLASSIFICATION_LEDGER, read_ledger, write_ledger
from mixle.task._teacher import TeacherCaller, as_batch_view
from mixle.task.calibrate import CalibratedTaskModel
from mixle.task.cascade import Cascade
from mixle.task.density import DensityGate
from mixle.task.distill import agreement, distill_from_labels, distill_records_from_labels
from mixle.task.model import HashedNGram, HashedRecord, TaskModel
from mixle.task.tune import RecipeSpace
from mixle.utils.paths import contained_path


def _label_with(teacher: Callable[..., Any], items: list) -> list:
    """Label ``items`` with either a per-item or batched teacher callable.

    Prefer holding a :class:`TeacherCaller` where the same teacher is used more than once: this
    shim re-discovers the calling convention on every call, which costs the teacher an extra
    invocation each time.
    """
    return _batch_view(teacher)(list(items))


def _batch_view(teacher: Callable[..., Any]) -> TeacherCaller:
    """Return a strict ``list -> list`` teacher view for cascade probes."""
    return as_batch_view(teacher)


def _evidence_digest(inputs: Sequence[Any], labels: Sequence[Any]) -> str:
    """Bind a verification decision to the exact ordered evidence rows without persisting those rows."""
    return sha256(repr(list(zip(inputs, labels))).encode("utf-8")).hexdigest()


def _split_holdout_roles(count: int) -> int:
    """How many of ``count`` reserved rows take the CONFORMAL role; the rest take the SELECTION role.

    ``holdout=`` is documented as the fraction reserved for "calibration + verification" -- two roles.
    They were one set, so :meth:`Solution.improve` chose the student with the same rows that then set its
    conformal threshold, and the reported agreement was a running maximum over one fixed set rather than
    a held-out estimate (MXR-080-1891). Conformal takes the larger half because its quantile is the part
    with a coverage claim attached; the split is deterministic in the count alone, so the two roles are
    fixed the moment the holdout is drawn and are never re-drawn or swapped.
    """
    return max(1, (count + 1) // 2)


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


_INPUT_KINDS = ("text", "record")


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


def _validated_kind(kind: Any) -> str:
    """Check the student-path discriminator BEFORE anything dispatches on it (MXR-080-1893).

    Every dispatch site spells the discriminator ``"text" if kind == "text" else <record>``, so an
    unrecognized kind was never refused -- it silently meant "record". ``kind="txt"`` fitted a record
    student over text inputs, stored ``kind="txt"`` in the artifact, and reloaded to the same wrong
    path, with no error anywhere in the chain.
    """
    if kind not in _INPUT_KINDS:
        raise ValueError(f"kind must be one of {list(_INPUT_KINDS)}, got {kind!r}")
    return str(kind)


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


def _fit_edge_student(
    inputs: list,
    labels: list,
    device: Any,
    device_space: Any,
    propose_budget: int,
    seed: int,
    design: Any = None,
) -> Any:
    """Structure x recipe search for a student that fits ``device`` -- the shared core of ``solve()``'s
    initial edge search and :meth:`Solution.improve`'s re-search under the SAME budget. Passing the
    prior search's ``design`` (a :class:`~mixle.task.edge.DesignModel`) warm-starts the surrogate from
    what it already learned, so re-searching after harvesting new labels isn't a cold restart."""
    from mixle.task.edge import distill_for_edge

    n_val = max(2, len(inputs) // 4)
    val_order = np.random.RandomState(seed).permutation(len(inputs))
    v_idx, f_idx = val_order[:n_val], val_order[n_val:]
    n_init = min(4, max(2, propose_budget // 2))
    return distill_for_edge(
        None,
        [inputs[i] for i in f_idx],
        [inputs[i] for i in v_idx],
        device,
        train_labels=[labels[i] for i in f_idx],
        val_labels=[labels[i] for i in v_idx],
        labels=sorted(set(labels)),
        space=device_space,
        design=design,
        n_init=n_init,
        n_iter=max(1, propose_budget - n_init),
        seed=seed,
    )


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
    device: Any = None  # DeviceSpec solve() searched under (edge != None) -- improve() re-searches the SAME budget
    device_space: Any = None  # EdgeSpace the edge search used (None = default); reused to warm-start from edge.design
    propose_budget: int = 8  # edge-search effort improve() reuses when re-searching under device
    verification_digest: str | None = None
    # MXR-080-1891: the reserved holdout carries two IMMUTABLE roles. ``cal_*`` sets the conformal
    # threshold and is never read by a selection decision; ``sel_*`` produces holdout_agreement /
    # escalation_rate and is the only evidence improve() promotes on. ``selection_uses`` is the one-use
    # receipt: 1 means the reported agreement is a genuine single-use held-out estimate, >1 means the
    # same rows have now decided more than one promotion and the number is selection-contaminated.
    sel_inputs: list = field(default_factory=list)
    sel_labels: list = field(default_factory=list)
    selection_uses: int = 0
    # Which regime certified the CURRENT conformal threshold: "solve-split" (the initial holdout
    # calibration role), "fresh-evidence" (an ``evidence_inputs`` batch), or
    # "reused-after-adaptive-harvest" -- a no-argument promotion recalibrated on the solve-time
    # rows even though the candidate trained on escalations that the OLD threshold selected
    # per-query, which is STAT-RR12-1's leak in per-query form: the finite-sample statement for
    # such a threshold is not restored by the role split, and this field says so in report().
    calibration_evidence: str = "solve-split"
    selection_receipt: list[dict] = field(default_factory=list)

    @property
    def selection_evidence_is_single_use(self) -> bool:
        """True while ``holdout_agreement`` is still an untouched-once held-out number."""
        return self.selection_uses <= 1

    def _teacher_call(self) -> TeacherCaller:
        """The teacher view this Solution routes through, resolved once and kept.

        A demoted Solution runs the teacher on *every* request, so rediscovering the calling
        convention per call would cost an extra teacher invocation per request forever.
        """
        caller = self.__dict__.get("_teacher_caller")
        if caller is None:
            # the cascade already holds a resolved view of the same teacher when there is one
            escalate = getattr(self.cascade, "teacher", None)
            caller = escalate if isinstance(escalate, TeacherCaller) else TeacherCaller(self.teacher)
            self.__dict__["_teacher_caller"] = caller
        return caller

    def __call__(self, x: Any) -> Any:
        if not self.promoted:
            return self._teacher_call().one(x)
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
            # MXR-080-1891: how many promotion decisions the reported agreement's own rows have now made.
            # 1 = a single-use held-out number; >1 = the same rows picked more than one student, so read
            # it as a selection score, not a generalization estimate.
            "selection_uses": self.selection_uses,
            "selection_evidence_is_single_use": self.selection_evidence_is_single_use,
            # which rows certified the CURRENT threshold; "reused-after-adaptive-harvest" means
            # the finite-sample coverage statement is voided for this artifact (STAT-RR12-1)
            "calibration_evidence": self.calibration_evidence,
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

    def improve(self, evidence_inputs: Sequence[Any] | None = None) -> bool:
        """Re-distill with the harvested (escalated) labels; promote only if it verifies at least as well.

        Returns True when a better student was promoted. Neither reserved role is ever trained on.

        **Roles (MXR-080-1891).** The promote/reject decision reads only the SELECTION rows and the
        conformal threshold is set only from the CALIBRATION rows, so the promoted student is no longer
        chosen with the same data that certifies its coverage. Passing ``evidence_inputs`` supplies a
        fresh, teacher-labeled batch that REPLACES both roles, which is the only way to keep the reported
        agreement a genuine single-use held-out number across rounds; without it the existing selection
        rows decide again and ``selection_uses`` records that the number is now a selection score. The
        reuse is recorded rather than refused because the no-argument loop is the documented serving
        workflow and a caller with no fresh traffic still needs the anti-regression gate.

        **What reuse does to the threshold's guarantee (STAT-RR12-1).** The role split does NOT
        restore the finite-sample coverage statement for a no-argument promotion: the harvested
        escalations that train the candidate were selected per-query by the OLD threshold, itself a
        function of the calibration rows -- so those rows helped construct the candidate and cannot
        also certify it (the regression loop's all-or-none version of this leak measured 0.8857
        against a claimed 0.90; the per-query selection here is the stronger channel, and unlike
        regression the escalated slice is threshold-selected, so it cannot serve as fresh calibration
        either). After such a promotion ``report()['calibration_evidence']`` says
        ``"reused-after-adaptive-harvest"`` and the deployed threshold's nominal statement should be
        treated as empirical; a promotion fed by ``evidence_inputs`` records ``"fresh-evidence"`` and
        keeps the exact statement.

        When the original ``solve()`` ran under a device budget (``self.edge`` is set), the harvested
        labels are re-searched under that SAME ``DeviceSpec``/``EdgeSpace`` -- warm-started from the
        prior search's design ledger -- instead of being refit with a generic, unconstrained student. A
        candidate that no longer fits the device is rejected exactly like any other anti-regression
        failure, so a promoted student never silently exceeds the budget ``device=`` promised.
        """
        if not self.cal_inputs or not self.sel_inputs:
            raise RuntimeError(
                "this Solution was loaded from an artifact and has no training/calibration data; "
                "collect cascade.harvested() and re-solve(real + harvested inputs) to improve."
            )
        new_inputs, new_labels = self.cascade.harvested()
        if not new_inputs:
            return False
        cal_in, cal_lab, sel_in, sel_lab, fresh = self._evidence_roles(evidence_inputs)
        inputs = self.train_inputs + list(new_inputs)
        labels = self.train_labels + [str(y) for y in new_labels]
        new_edge = None
        if self.edge is not None:
            new_edge = _fit_edge_student(
                inputs,
                labels,
                self.device,
                self.device_space,
                self.propose_budget,
                self.seed,
                design=self.edge.design,
            )
            if not new_edge.feasible:
                return False  # nothing this round fits the device budget: keep the current, compliant student
            student = new_edge.model
        else:
            student = _fit_student(self.kind, inputs, labels, self.distill_kw)
        alpha = self.cascade.model.alpha
        # the gate stays real-inputs-only: harvested escalations are real, synthetic training rows are not
        gate_inputs = self.gate_inputs + list(new_inputs)
        gate = _fit_gate(self.kind, gate_inputs, self.ood, self.seed) if self.ood is not None else None
        # calibrate on the CALIBRATION role, decide on the SELECTION role -- the two never cross.
        cal = CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(cal_in, cal_lab)
        agree = agreement(student, sel_lab, sel_in)
        esc = cal.escalation_rate(sel_in)
        # On fresh evidence the incumbent's stored numbers came from different rows, so re-measure it on
        # the same rows the candidate is judged on; otherwise the stored numbers already are that.
        if fresh:
            incumbent_agree = agreement(self.cascade.model.task, sel_lab, sel_in)
            incumbent_esc = self.cascade.model.escalation_rate(sel_in)
        else:
            incumbent_agree, incumbent_esc = self.holdout_agreement, self.escalation_rate
        promoted_now = not (agree < incumbent_agree or esc > incumbent_esc)
        uses = 1 if fresh else self.selection_uses + 1
        self.selection_receipt.append(
            {
                "round": len(self.selection_receipt) + 1,
                "fresh_evidence": fresh,
                "selection_uses": uses,
                "n_calibration": len(cal_in),
                "n_selection": len(sel_in),
                "evidence_sha256": _evidence_digest([*cal_in, *sel_in], [*cal_lab, *sel_lab]),
                "incumbent_agreement": float(incumbent_agree),
                "candidate_agreement": float(agree),
                "promoted": bool(promoted_now),
            }
        )
        if not promoted_now:
            # A rejected candidate still consumed a look at the selection rows when the evidence was
            # fresh (those rows are spent either way); reusing rows only counts once they decide.
            if fresh:
                self.cal_inputs, self.cal_labels = cal_in, cal_lab
                self.sel_inputs, self.sel_labels = sel_in, sel_lab
                self.selection_uses = uses
                self.holdout_agreement, self.escalation_rate = incumbent_agree, incumbent_esc
            else:
                self.selection_uses = uses
            return False  # anti-regression: keep the current student
        self.cascade.model = cal
        self.train_inputs, self.train_labels = inputs, labels
        self.gate_inputs = gate_inputs
        self.cal_inputs, self.cal_labels = cal_in, cal_lab
        self.sel_inputs, self.sel_labels = sel_in, sel_lab
        self.selection_uses = uses
        self.calibration_evidence = "fresh-evidence" if fresh else "reused-after-adaptive-harvest"
        self.verification_digest = _evidence_digest([*cal_in, *sel_in], [*cal_lab, *sel_lab])
        self.holdout_agreement, self.escalation_rate = agree, esc
        self.promoted = self.promoted or self._passes_target(agree)
        self.cascade.stats.escalated_texts.clear()
        self.cascade.stats.escalated_labels.clear()
        if new_edge is not None:
            self.edge = new_edge  # keep report()'s device fields honest about what's actually deployed
        return True

    def _evidence_roles(self, evidence_inputs: Sequence[Any] | None) -> tuple[list, list, list, list, bool]:
        """Resolve the (calibration, selection) rows this improve() round judges on.

        ``None`` reuses the roles fixed by ``solve()``. A fresh batch is teacher-labeled and split with
        the same deterministic role rule, which is what restores a one-use receipt (MXR-080-1891).
        """
        if evidence_inputs is None:
            return list(self.cal_inputs), list(self.cal_labels), list(self.sel_inputs), list(self.sel_labels), False
        rows = list(evidence_inputs)
        if len(rows) < 2:
            raise ValueError("fresh evidence needs at least two rows to fill both the calibration and selection roles")
        fresh_labels = [str(y) for y in self._teacher_call()(rows)]
        n_conf = _split_holdout_roles(len(rows))
        return rows[:n_conf], fresh_labels[:n_conf], rows[n_conf:], fresh_labels[n_conf:], True

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
                    # MXR-080-1891: the artifact says which rows certified what, and how many promotion
                    # decisions the reported agreement's own rows have made. selection_uses > 1 means
                    # read holdout_agreement as a selection score, not a generalization estimate.
                    "n_selection": len(self.sel_inputs),
                    # the honesty ledger is registry-driven: save and load iterate the SAME
                    # declaration, so a claim field cannot be written without being restored
                    # (STAT-R1/RR13-1/RR13-2 were all hand-maintained-pair drift)
                    **write_ledger(self, CLASSIFICATION_LEDGER),
                    "selection_evidence_is_single_use": self.selection_evidence_is_single_use,
                    "synthesized_inputs": self.synthesized,
                    "verified_at": time.time(),
                    "evidence_sha256": self.verification_digest
                    or _evidence_digest([*self.cal_inputs, *self.sel_inputs], [*self.cal_labels, *self.sel_labels]),
                    # Task artifacts bind the entire manifest (including this decision) and payload with
                    # SHA-256. Loading may promote only when this marker and the strict evidence record
                    # survive that integrity check.
                    "artifact_binding": "manifest-integrity-v1",
                },
                "target_agreement": self.target_agreement,
            },
        }
        return self.cascade.model.save(path)

    def deploy(self, name: str, root: str = "./mixle_data/registry") -> str:
        """Save into the serving layout — ``{root}/tasks/{name}`` — the directory the mixle-mlops
        ``/v1/tasks`` routes serve from. Returns the artifact path.

        ``name`` must be a single path component. It was joined onto ``root`` unchecked, and both
        ``pathlib`` and ``os.path.join`` will leave a root when asked to: ``name="../../escaped"``
        traversed out of it and an absolute ``name`` discarded it entirely, so a deployment could
        write anywhere the process could (MXR-080-1910). A serving name arrives from an API request
        or a config file, so this is the ordinary case, not an exotic one."""
        return self.save(str(contained_path(root, "tasks", name, kind="deployment name")))

    @classmethod
    def load(cls, path: str, teacher: Callable[..., Any], *, cost: Any = None, device: str = "cpu") -> Solution:
        """Reconstitute a *serving* Solution from a saved artifact — the deploy path for a fresh process.

        The loaded Solution answers locally / escalates to ``teacher`` and harvests labels exactly like
        the original. It carries no training or calibration data, so :meth:`improve` raises — collect the
        harvested pairs and re-``solve`` (real + harvested inputs) to train the next round."""
        cal = CalibratedTaskModel.load(path, device=device)
        meta = (cal.task.meta or {}).get("solve", {})
        if not isinstance(meta, dict):
            raise ValueError("artifact solve metadata must be a dictionary")
        verification = meta.get("verification")
        # An artifact without a verification block carries no honesty ledger at all: its
        # calibration history is unknown, and unknown history must refuse rather than reload
        # under fresh-solve defaults (STAT-RR14-1). Every 0.8.0 writer produces the block.
        if not isinstance(verification, dict):
            raise ValueError(
                "artifact has no verification metadata, so its claim-bearing ledger is missing: "
                "its calibration history is unknown and cannot present as certified evidence -- "
                "re-solve to produce a current artifact"
            )
        saved_promoted = verification.get("promoted")
        if not isinstance(saved_promoted, bool):
            raise ValueError("artifact verification promoted state must be boolean")
        holdout_agreement = float(verification.get("holdout_agreement", float("nan")))
        escalation_rate = float(verification.get("holdout_escalation_rate", float("nan")))
        verification_digest = verification.get("evidence_sha256")
        valid_evidence = (
            isinstance(verification_digest, str)
            and len(verification_digest) == 64
            and verification.get("artifact_binding") == "manifest-integrity-v1"
            and np.isfinite(holdout_agreement)
            and 0.0 <= holdout_agreement <= 1.0
            and np.isfinite(escalation_rate)
            and 0.0 <= escalation_rate <= 1.0
        )
        promoted = bool(saved_promoted and valid_evidence)
        # The honesty ledger survives the round trip (STAT-RR13-1/2) and is REQUIRED
        # (STAT-RR14-1): the registry refuses missing fields, unrecognized regimes, and
        # negative counts -- never defaults them.
        ledger = read_ledger(verification, CLASSIFICATION_LEDGER)
        return cls(
            cascade=Cascade(cal, _batch_view(teacher), cost=cost),
            teacher=teacher,
            # MXR-080-1893: validate the stored discriminator before it dispatches. A serving process
            # that reloads an artifact carrying an unrecognized kind would otherwise silently take the
            # record path -- including for a text model -- on every improve()/re-fit.
            kind=_validated_kind(meta.get("kind", "text")),
            train_inputs=[],
            train_labels=[],
            cal_inputs=[],
            cal_labels=[],
            holdout_agreement=holdout_agreement,
            escalation_rate=escalation_rate,
            promoted=promoted,
            target_agreement=meta.get("target_agreement"),
            ood=meta.get("ood"),
            verification_digest=verification_digest,
            **ledger,
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
    teacher_mode: str = "auto",
    **distill_kw: Any,
) -> Solution:
    """Replace ``teacher`` (the code currently doing the job) with a calibrated, self-improving model.

    Args:
        teacher: The callable performing the task today (per-item or batched). It labels the dataset and
            remains the fallback for inputs the student does not handle confidently.
        inputs: Example inputs (text, or tuple/dict records) covering the task. The teacher labels them.
        alpha: Miscoverage budget for the conformal prediction sets. Under exchangeability of the
            calibration rows and incoming queries, a set covers the teacher's label with probability
            ``>= 1 - alpha`` MARGINALLY over queries; the student answers locally only when its set
            is a single label and escalates otherwise. Marginal set coverage is NOT a bound on the
            error of the locally-answered slice (answering conditions on the set being a singleton),
            so answered-slice agreement is reported as a measurement -- certify an answered-slice
            target separately with
            :meth:`mixle.task.calibrate.CalibratedTaskModel.calibrate_selective`. Both statements
            fail silently under distribution shift; the ``ood`` gate below mitigates, and drifted
            traffic calls for re-measurement. The statement covers the INITIAL solve and any
            ``improve(evidence_inputs=...)`` promotion; a no-argument promotion recalibrates on
            rows that already shaped the candidate through the harvest, voiding the finite-sample
            statement for that artifact -- ``report()['calibration_evidence']`` says which regime
            applies (see :meth:`Solution.improve`).
        target_agreement: Optional gate. If the student's held-out agreement with the teacher misses it,
            the returned Solution routes *everything* to the teacher (``promoted=False``).
        holdout: Fraction reserved for calibration + verification (never trained on). Those are two
            IMMUTABLE roles, split deterministically the moment the holdout is drawn: the calibration
            rows set the conformal threshold and the selection rows produce ``holdout_agreement`` /
            ``escalation_rate`` and decide every :meth:`Solution.improve` promotion. Nothing reads both.
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
    k = _validated_kind(kind) if kind is not None else _input_kind(items[0])
    # one view for the whole call: the calling convention is discovered once, not per labeling pass,
    # and the same resolved view is handed to the Cascade so escalation never re-probes either.
    call = as_batch_view(teacher, teacher_mode)
    labels = [str(y) for y in call(items)]

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_cal = max(2, int(round(len(items) * holdout)))
    cal_idx, train_idx = order[:n_cal], order[n_cal:]
    train_inputs = [items[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    # MXR-080-1891: the reserved rows carry two roles, fixed here and never re-drawn. Conformal
    # calibration reads cal_*; every selection/verification number reads sel_*. Nothing reads both.
    n_conf = _split_holdout_roles(n_cal)
    conf_idx, sel_idx = cal_idx[:n_conf], cal_idx[n_conf:]
    cal_inputs = [items[i] for i in conf_idx]
    cal_labels = [labels[i] for i in conf_idx]
    sel_inputs = [items[i] for i in sel_idx]
    sel_labels = [labels[i] for i in sel_idx]

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
            train_labels = train_labels + [str(y) for y in call(synth)]
            n_synth = len(synth)

    distill_kw.setdefault("seed", seed)
    edge_result = None
    if device is not None:
        # "this capability on that device": structure x precision x recipe search under the hard
        # budget, on the labels already computed -- the teacher is never re-called.
        edge_result = _fit_edge_student(train_inputs, train_labels, device, device_space, propose_budget, seed)
        student = edge_result.model
    else:
        if propose == "auto":
            distill_kw = _tune_recipe(k, train_inputs, train_labels, distill_kw, propose_budget, seed)
        student = _fit_student(k, train_inputs, train_labels, distill_kw)
    cal = CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(cal_inputs, cal_labels)
    # Verification reads the selection role only. Measuring it on the calibration rows made the
    # escalation rate mechanically equal to alpha (those rows DEFINED the threshold) and handed
    # improve() the calibration set as its selection set (MXR-080-1891).
    agree = agreement(student, sel_labels, sel_inputs)
    esc = cal.escalation_rate(sel_inputs)
    promoted = target_agreement is None or agree >= target_agreement
    if edge_result is not None and not edge_result.feasible:
        promoted = False  # nothing fit the device: serve the teacher, never a budget-busting student

    return Solution(
        cascade=Cascade(cal, call, cost=cost),
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
        # under device=, **distill_kw is never consulted for the winning student -- the edge search's
        # own recipe is -- so record THAT (what was actually fit) rather than the caller's unused kwargs
        distill_kw=dict(edge_result.recipe) if edge_result is not None else dict(distill_kw),
        ood=ood,
        seed=seed,
        synthesized=n_synth,
        gate_inputs=list(train_inputs[: len(train_inputs) - n_synth]),
        edge=edge_result,
        device=device,
        device_space=device_space,
        propose_budget=propose_budget,
        verification_digest=_evidence_digest([*cal_inputs, *sel_inputs], [*cal_labels, *sel_labels]),
        sel_inputs=sel_inputs,
        sel_labels=sel_labels,
        selection_uses=1,  # solve() itself is the selection role's first and, so far, only use
    )
