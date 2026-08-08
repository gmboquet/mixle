"""``solve_multilabel`` -- replace rigid code that returns a set of labels, with joint-set honesty.

The multi-label shape of the solve loop: ``teacher(x) -> list[str]`` (tags, flags, categories -- any
subset of a label universe). The student is one shared-feature net with a sigmoid head per label; the
calibration uses one joint nonconformity score: the largest binary-label error across the whole label
vector. Its split-conformal quantile covers the complete teacher set with probability at least
``1 - alpha`` under exchangeability. A request is answered locally only when that joint prediction set
contains exactly one label vector; multiple or zero admissible vectors escalate to the teacher.

**Scope of that statement.** The ``1 - alpha`` joint-set coverage is finite-sample and MARGINAL over
the calibration draw and the query jointly, under exchangeability of the calibration rows and
incoming traffic. It is NOT an accuracy guarantee conditional on answering locally: serving conditions
on the joint set being a singleton, and coverage conditional on that event is not controlled (the
classification side measured 9% marginal versus 47% answered-slice error from the same selection
effect). Answered-slice quality is a MEASUREMENT, never a guarantee, and ``report()`` carries two
distinct numbers that must not be conflated: ``holdout_set_agreement`` is the RAW route-independent
0.5-threshold exact-set agreement on the evaluation rows (the serving gate never runs in it), while
``answered_slice`` applies the real joint-qhat singleton rule to the same disjoint evaluation rows
and reports exact-set agreement among the rows the gate actually answered, with the answered
denominator and an exact 95% Clopper-Pearson interval (``None`` when it answered none). After an
``improve()`` promotion those numbers describe the winner of a 2-way comparison made on the same
rows, so they are post-selection measurements (the receipt records this). Distribution shift or any
other break of exchangeability voids the coverage statement silently; re-measure on drifted
traffic. ``report()`` carries the scope machine-readably in ``coverage_contract_scope``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np

from mixle.task._ledger import _clopper_pearson_interval, conformal_scope
from mixle.task.model import HashedNGram
from mixle.task.regress import (
    RecordRegressionFeaturizer,
    _validated_features,
    featurizer_from_spec,
    featurizer_spec,
)
from mixle.task.solve import _input_kind, _label_with


def _fit_multilabel_mlp(x: np.ndarray, y: np.ndarray, hidden: Sequence[int], epochs: int, lr: float, seed: int):
    import torch

    torch.manual_seed(seed)
    dims = [x.shape[1], *hidden, y.shape[1]]
    layers: list[Any] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float32))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    return net


def _quantile_upper(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample upper bar: the ceil((n+1)(1-alpha)) order statistic; ``inf`` when n is too small."""
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def _score_net(net: Any, feats: np.ndarray, n_labels: int) -> np.ndarray:
    import torch

    params = list(net.parameters())
    if not params:
        raise ValueError("multilabel network has no parameters")
    tensor = torch.as_tensor(feats, dtype=params[0].dtype, device=params[0].device)
    with torch.no_grad():
        scores = torch.sigmoid(net(tensor)).detach().to(device="cpu", dtype=torch.float64).numpy()
    if (
        scores.shape != (len(feats), n_labels)
        or not np.all(np.isfinite(scores))
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError("multilabel network returned invalid probabilities")
    return scores


def _indicator_matrix(label_sets: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
    index = {label: j for j, label in enumerate(labels)}
    matrix = np.zeros((len(label_sets), len(labels)), dtype=bool)
    for i, tags in enumerate(label_sets):
        for tag in tags:
            if tag not in index:
                raise ValueError(f"label {tag!r} is outside the declared multilabel support")
            matrix[i, index[tag]] = True
    return matrix


def _joint_decisions(scores: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(uniquely_decided, predicted_present)`` for a joint conformal set."""
    present_allowed = (1.0 - scores) <= qhat
    absent_allowed = scores <= qhat
    unique = np.all(np.logical_xor(present_allowed, absent_allowed), axis=1)
    return unique, present_allowed & ~absent_allowed


def _set_agreement(
    net: Any,
    featurizer: Any,
    inputs: Sequence[Any],
    label_sets: Sequence[Sequence[str]],
    labels: Sequence[str],
    qhat: float,
) -> float:
    if not inputs or len(inputs) != len(label_sets):
        raise ValueError("evaluation inputs and label sets must be nonempty and aligned")
    scores = _score_net(net, _validated_features(featurizer, inputs), len(labels))
    unique, predicted = _joint_decisions(scores, qhat)
    truth = _indicator_matrix(label_sets, labels)
    return float(np.mean(unique & np.all(predicted == truth, axis=1)))


def _raw_set_agreement(
    net: Any,
    featurizer: Any,
    inputs: Sequence[Any],
    label_sets: Sequence[Sequence[str]],
    labels: Sequence[str],
) -> float:
    if not inputs or len(inputs) != len(label_sets):
        raise ValueError("evaluation inputs and label sets must be nonempty and aligned")
    scores = _score_net(net, _validated_features(featurizer, inputs), len(labels))
    truth = _indicator_matrix(label_sets, labels)
    return float(np.mean(np.all((scores >= 0.5) == truth, axis=1)))


def _normalize_label_sets(values: Sequence[Any], *, name: str) -> list[list[str]]:
    result: list[list[str]] = []
    for row in values:
        if row is None:
            tags: list[Any] = []
        elif isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"{name} rows must be sequences of labels, not {type(row).__name__}")
        else:
            tags = list(row)
        normalized = [str(tag) for tag in tags]
        if any(not tag for tag in normalized):
            raise ValueError(f"{name} labels must be nonempty strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{name} rows must not contain duplicate labels")
        result.append(normalized)
    return result


@dataclass
class MultiLabelSolution:
    """A per-label-calibrated tagger in front of the routine it replaces."""

    net: Any
    featurizer: Any
    labels: list[str]
    teacher: Callable[..., Any]
    upper_absent: np.ndarray  # A_l: above -> confidently present
    lower_present: np.ndarray  # P_l: below -> confidently absent
    joint_qhat: float
    alpha: float
    holdout_set_agreement: float
    # answered-slice MEASUREMENT (STAT-RR16-2): the real joint-qhat singleton gate applied to the
    # disjoint evaluation rows -- how many rows were evaluated, how many the gate answered, and how
    # many answered rows matched the teacher's set exactly. Measurements, never guarantees.
    eval_rows: int = 0
    answered_eval_n: int = 0
    answered_eval_correct: int = 0
    train_inputs: list = field(default_factory=list)
    train_sets: list = field(default_factory=list)
    cal_inputs: list = field(default_factory=list)
    cal_sets: list = field(default_factory=list)
    eval_inputs: list = field(default_factory=list)
    eval_sets: list = field(default_factory=list)
    hidden: tuple = (64,)
    epochs: int = 300
    lr: float = 1e-2
    seed: int = 0
    n_requests: int = 0
    n_escalated: int = 0
    harvested_inputs: list = field(default_factory=list)
    harvested_sets: list = field(default_factory=list)
    calibration_receipt: dict[str, Any] = field(default_factory=dict)

    def _scores(self, xs: list) -> np.ndarray:
        return _score_net(self.net, _validated_features(self.featurizer, xs), len(self.labels))

    def try_local(self, x: Any) -> list[str] | None:
        """The decided label set, or ``None`` when any label is ambiguous (= must escalate)."""
        s = self._scores([x])[0]
        unique, present = _joint_decisions(s[None, :], self.joint_qhat)
        if bool(unique[0]):
            return [lab for lab, is_present in zip(self.labels, present[0]) if is_present]
        return None

    def decide(self, x: Any) -> list[str] | None:
        """Return the local multilabel decision, or ``None`` when the example should escalate."""
        return self.try_local(x)

    def __call__(self, x: Any) -> list[str]:
        self.n_requests += 1
        local = self.try_local(x)
        if local is not None:
            return local
        self.n_escalated += 1
        got = _label_with(self.teacher, [x])[0]
        tags = _normalize_label_sets([got], name="teacher output")[0]
        self.harvested_inputs.append(x)
        self.harvested_sets.append(tags)
        return tags

    def report(self) -> dict[str, Any]:
        """Return multi-label agreement, escalation, and harvest metrics."""
        return {
            "labels": len(self.labels),
            # RAW route-independent 0.5-threshold exact-set agreement on the evaluation rows; the
            # serving gate never runs in it, so it is NOT an answered-slice number
            "holdout_set_agreement": round(self.holdout_set_agreement, 4),
            # answered-slice MEASUREMENT: the real joint-qhat singleton rule on the disjoint
            # evaluation rows, with the answered denominator and an exact 95% Clopper-Pearson
            # interval; None when the gate answered none of them
            "answered_slice": (
                None
                if self.answered_eval_n == 0
                else {
                    "agreement": round(self.answered_eval_correct / self.answered_eval_n, 4),
                    "n_answered": self.answered_eval_n,
                    "n_evaluated": self.eval_rows,
                    "ci95": [
                        round(bound, 4)
                        for bound in _clopper_pearson_interval(self.answered_eval_correct, self.answered_eval_n, 0.95)
                    ],
                }
            ),
            "alpha": self.alpha,
            "coverage_contract": "joint_exact_set",
            "coverage_contract_scope": conformal_scope(
                "finite-sample marginal joint-set coverage of the complete teacher label vector"
            ),
            "joint_qhat": self.joint_qhat,
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "escalation_rate": (self.n_escalated / self.n_requests) if self.n_requests else 0.0,
            "harvested": len(self.harvested_sets),
        }

    def save(self, path: str) -> str:
        """Persist net + featurizer + per-label bars; :meth:`load` restores a serving tagger."""
        from mixle.task.artifact import save_module

        first = next(m for m in self.net.modules() if hasattr(m, "in_features"))
        return save_module(
            path,
            self.net,
            "mixle.mlp",
            {
                "input_dim": int(first.in_features),
                "hidden_dims": [int(h) for h in self.hidden],
                "output_dim": len(self.labels),
                "activation": "relu",
            },
            task="solve_multilabel student",
            io=featurizer_spec(self.featurizer),
            meta={
                "multilabel": {
                    "labels": list(self.labels),
                    "upper_absent": [float(v) for v in self.upper_absent],
                    "lower_present": [float(v) for v in self.lower_present],
                    "joint_qhat": self.joint_qhat,
                    "alpha": self.alpha,
                    "holdout_set_agreement": self.holdout_set_agreement,
                    "eval_rows": int(self.eval_rows),
                    "answered_eval_n": int(self.answered_eval_n),
                    "answered_eval_correct": int(self.answered_eval_correct),
                    "hidden": [int(h) for h in self.hidden],
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "seed": self.seed,
                    "calibration_receipt": self.calibration_receipt,
                }
            },
        )

    @classmethod
    def load(cls, path: str, teacher: Callable[..., Any], *, device: str = "cpu") -> MultiLabelSolution:
        """Reconstitute a serving MultiLabelSolution (no training/calibration data; improve() raises)."""
        from mixle.task.artifact import load_module

        net, manifest = load_module(path, device=device)
        m = manifest.meta["multilabel"]
        return cls(
            net=net,
            featurizer=featurizer_from_spec(manifest.io),
            labels=list(m["labels"]),
            teacher=teacher,
            upper_absent=np.asarray(m["upper_absent"], dtype=np.float64),
            lower_present=np.asarray(m["lower_present"], dtype=np.float64),
            joint_qhat=float(m["joint_qhat"]),
            alpha=float(m["alpha"]),
            holdout_set_agreement=float(m["holdout_set_agreement"]),
            # required, not defaulted: a live object's zero-initialization means "not yet
            # measured", and reusing it for an ABSENT member would present an artifact with an
            # unknown measurement as one that measured nothing (the STAT-RR14-1 mechanism); the
            # 0.8.0 format is the first to ship this artifact, so every artifact carries these
            eval_rows=int(m["eval_rows"]),
            answered_eval_n=int(m["answered_eval_n"]),
            answered_eval_correct=int(m["answered_eval_correct"]),
            hidden=tuple(m["hidden"]),
            epochs=int(m["epochs"]),
            lr=float(m["lr"]),
            seed=int(m["seed"]),
            calibration_receipt=dict(m.get("calibration_receipt", {})),
        )

    def improve(self, evidence_inputs: Sequence[Any] | None = None) -> bool:
        """Re-fit harvested sets and use a fresh, single-use evidence batch for promotion and calibration."""
        if not self.harvested_inputs:
            return False
        if not self.cal_inputs:
            raise RuntimeError(
                "this MultiLabelSolution was loaded from an artifact and has no calibration data; "
                "collect the harvested pairs and re-solve_multilabel() to improve."
            )
        if evidence_inputs is None:
            raise ValueError("improve() requires fresh evidence_inputs; prior calibration data cannot be reused")
        fresh_inputs = list(evidence_inputs)
        min_cal = int(np.ceil(1.0 / self.alpha)) - 1
        if len(fresh_inputs) < min_cal + 2:
            raise ValueError(f"fresh evidence needs at least {min_cal + 2} examples")
        fresh_sets = _normalize_label_sets(_label_with(self.teacher, fresh_inputs), name="fresh evidence")
        unknown = sorted({tag for tags in fresh_sets for tag in tags} - set(self.labels))
        if unknown:
            raise ValueError(f"fresh evidence contains labels outside the fitted support: {unknown}")
        order = np.random.RandomState(
            self.seed + len(self.calibration_receipt.get("improvements", [])) + 1
        ).permutation(len(fresh_inputs))
        cal_idx, eval_idx = order[:min_cal], order[min_cal:]
        fresh_cal_inputs = [fresh_inputs[i] for i in cal_idx]
        fresh_cal_sets = [fresh_sets[i] for i in cal_idx]
        fresh_eval_inputs = [fresh_inputs[i] for i in eval_idx]
        fresh_eval_sets = [fresh_sets[i] for i in eval_idx]
        inputs = self.train_inputs + list(self.harvested_inputs)
        sets = self.train_sets + [list(v) for v in self.harvested_sets]
        unknown_harvest = sorted({tag for tags in sets for tag in tags} - set(self.labels))
        if unknown_harvest:
            raise ValueError(f"harvested evidence contains labels outside the fitted support: {unknown_harvest}")
        cand = _fit_and_calibrate(
            inputs,
            sets,
            fresh_cal_inputs,
            fresh_cal_sets,
            fresh_eval_inputs,
            fresh_eval_sets,
            self.labels,
            self.featurizer,
            self.alpha,
            self.hidden,
            self.epochs,
            self.lr,
            self.seed,
        )
        incumbent_agreement = _raw_set_agreement(
            self.net, self.featurizer, fresh_eval_inputs, fresh_eval_sets, self.labels
        )
        if cand["agreement"] < incumbent_agreement - 1e-12:
            return False
        self.net = cand["net"]
        self.upper_absent, self.lower_present = cand["upper_absent"], cand["lower_present"]
        self.joint_qhat = float(cand["joint_qhat"])
        self.holdout_set_agreement = float(cand["agreement"])
        self.eval_rows = int(cand["eval_rows"])
        self.answered_eval_n = int(cand["answered_eval_n"])
        self.answered_eval_correct = int(cand["answered_eval_correct"])
        self.train_inputs, self.train_sets = inputs, sets
        self.cal_inputs, self.cal_sets = fresh_cal_inputs, fresh_cal_sets
        self.eval_inputs, self.eval_sets = fresh_eval_inputs, fresh_eval_sets
        self.calibration_receipt.setdefault("improvements", []).append(
            {
                "calibration_count": len(fresh_cal_inputs),
                "evaluation_count": len(fresh_eval_inputs),
                "evidence_sha256": sha256(repr(list(zip(fresh_inputs, fresh_sets))).encode("utf-8")).hexdigest(),
                "incumbent_agreement": incumbent_agreement,
                "candidate_agreement": cand["agreement"],
                # the promotion rule compared incumbent vs candidate on these same evaluation
                # rows, so the promoted numbers are post-selection (a 2-way pick); they are
                # measurements of the winner, not selection-free estimates
                "post_selection": True,
            }
        )
        self.harvested_inputs.clear()
        self.harvested_sets.clear()
        return True


def _fit_and_calibrate(
    train_inputs,
    train_sets,
    cal_inputs,
    cal_sets,
    eval_inputs,
    eval_sets,
    labels,
    featurizer,
    alpha,
    hidden,
    epochs,
    lr,
    seed,
) -> dict:
    if len(train_inputs) != len(train_sets) or len(cal_inputs) != len(cal_sets):
        raise ValueError("multilabel inputs and label sets must be aligned")
    y = _indicator_matrix(train_sets, labels).astype(np.float32)
    feats = _validated_features(featurizer, train_inputs)
    net = _fit_multilabel_mlp(feats, y, hidden, epochs, lr, seed)

    s = _score_net(net, _validated_features(featurizer, cal_inputs), len(labels))
    y_cal = _indicator_matrix(cal_sets, labels)
    joint_scores = np.max(np.where(y_cal, 1.0 - s, s), axis=1)
    qhat = _quantile_upper(joint_scores, alpha)
    if not np.isfinite(qhat):
        raise ValueError(
            f"{len(cal_inputs)} calibration examples are insufficient for finite {1.0 - alpha:.6g} joint coverage"
        )
    agree = _raw_set_agreement(net, featurizer, eval_inputs, eval_sets, labels)
    # The ANSWERED-SLICE measurement runs the REAL serving rule (joint_qhat singleton decision)
    # on the disjoint evaluation rows: how many the gate answers, and how many of those match
    # the teacher's exact set. The raw 0.5-threshold agreement above is route-independent and
    # must never be presented as an answered-slice number (STAT-RR16-2).
    eval_scores = _score_net(net, _validated_features(featurizer, eval_inputs), len(labels))
    unique, predicted = _joint_decisions(eval_scores, qhat)
    eval_truth = _indicator_matrix(eval_sets, labels)
    answered_n = int(np.sum(unique))
    answered_correct = int(np.sum(unique & np.all(predicted == eval_truth, axis=1)))
    # Retained for artifact/API compatibility; the serving decision uses joint_qhat.
    upper_absent = np.full(len(labels), 1.0 - qhat)
    lower_present = np.full(len(labels), qhat)
    return {
        "net": net,
        "upper_absent": upper_absent,
        "lower_present": lower_present,
        "joint_qhat": qhat,
        "agreement": agree,
        "eval_rows": len(eval_inputs),
        "answered_eval_n": answered_n,
        "answered_eval_correct": answered_correct,
    }


def solve_multilabel(
    teacher: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    alpha: float = 0.1,
    holdout: float = 0.25,
    kind: str | None = None,
    hidden: Sequence[int] = (64,),
    epochs: int = 300,
    lr: float = 1e-2,
    dim: int = 256,
    prelabeled: tuple[Sequence[Any], Sequence[Sequence[str]]] | None = None,
    seed: int = 0,
) -> MultiLabelSolution:
    """Replace a set-of-labels routine with a per-label-calibrated student (see module docstring).

    ``prelabeled`` — already-teacher-labeled ``(inputs, label_sets)``, typically harvested escalations
    from a serving deployment — folds into the TRAINING split only, never calibration (which stays a
    fresh split of ``inputs``, so the per-label bars keep their finite-sample rank guarantee). Labels
    seen only in ``prelabeled`` still enter the label space.
    """
    if not callable(teacher):
        raise TypeError("teacher must be callable")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not np.isfinite(holdout) or not 0.0 < holdout < 1.0:
        raise ValueError("holdout must be finite and in (0, 1)")
    if kind not in (None, "text", "record"):
        raise ValueError("kind must be None, 'text', or 'record'")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    hidden = tuple(hidden)
    if any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in hidden):
        raise ValueError("hidden widths must be positive integers")

    items = list(inputs)
    if len(items) < 12:
        raise ValueError("solve_multilabel needs at least 12 example inputs")
    k = kind or _input_kind(items[0])
    raw = _label_with(teacher, items)
    if len(raw) != len(items):
        raise ValueError("teacher must return exactly one label set per input")
    sets = _normalize_label_sets(raw, name="teacher output")
    pre_in: list = []
    pre_sets: list[list[str]] = []
    if prelabeled is not None:
        pre_in = list(prelabeled[0])
        pre_sets = _normalize_label_sets(prelabeled[1], name="prelabeled")
        if len(pre_in) != len(pre_sets):
            raise ValueError("prelabeled inputs and label sets must have equal length")
    labels = sorted({t for tags in sets for t in tags} | {t for tags in pre_sets for t in tags})
    if not labels:
        raise ValueError("the teacher produced no labels on the example inputs")

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_holdout = max(4, int(round(len(items) * holdout)))
    min_cal = int(np.ceil(1.0 / alpha)) - 1
    n_cal = max(min_cal, n_holdout // 2)
    n_eval = n_holdout - n_cal
    if n_eval < 2 or n_holdout >= len(items):
        raise ValueError(
            f"holdout must leave training data, at least {min_cal} calibration rows, and two evaluation rows"
        )
    cal_idx, eval_idx, train_idx = order[:n_cal], order[n_cal:n_holdout], order[n_holdout:]
    train_inputs = [items[i] for i in train_idx] + pre_in
    train_sets = [sets[i] for i in train_idx] + pre_sets
    cal_inputs = [items[i] for i in cal_idx]
    cal_sets = [sets[i] for i in cal_idx]
    eval_inputs = [items[i] for i in eval_idx]
    eval_sets = [sets[i] for i in eval_idx]

    # records: standardized numeric pass-through + hashed categoricals (HashedRecord's tanh squash
    # saturates and erases the magnitude signal threshold-flags like "high-value" depend on)
    featurizer = (
        HashedNGram(n=3, dim=dim, seed=seed)
        if k == "text"
        else RecordRegressionFeaturizer(dim=dim, seed=seed).fit(train_inputs)
    )
    cand = _fit_and_calibrate(
        train_inputs,
        train_sets,
        cal_inputs,
        cal_sets,
        eval_inputs,
        eval_sets,
        labels,
        featurizer,
        float(alpha),
        hidden,
        epochs,
        float(lr),
        int(seed),
    )
    return MultiLabelSolution(
        net=cand["net"],
        featurizer=featurizer,
        labels=labels,
        teacher=teacher,
        upper_absent=cand["upper_absent"],
        lower_present=cand["lower_present"],
        joint_qhat=float(cand["joint_qhat"]),
        alpha=float(alpha),
        holdout_set_agreement=float(cand["agreement"]),
        eval_rows=int(cand["eval_rows"]),
        answered_eval_n=int(cand["answered_eval_n"]),
        answered_eval_correct=int(cand["answered_eval_correct"]),
        train_inputs=train_inputs,
        train_sets=train_sets,
        cal_inputs=cal_inputs,
        cal_sets=cal_sets,
        eval_inputs=eval_inputs,
        eval_sets=eval_sets,
        hidden=hidden,
        epochs=epochs,
        lr=float(lr),
        seed=seed,
        calibration_receipt={
            "contract": "joint_exact_set",
            "calibration_indices": [int(i) for i in cal_idx],
            "evaluation_indices": [int(i) for i in eval_idx],
            "calibration_count": len(cal_inputs),
            "evaluation_count": len(eval_inputs),
            "evidence_sha256": sha256(repr(list(zip(items, sets))).encode("utf-8")).hexdigest(),
            "improvements": [],
        },
    )
