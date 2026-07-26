"""Disagreement gate: escalate where the student has historically diverged from the teacher.

This is distinct from escalating only where inputs look statistically atypical
(:class:`~mixle.task.density.DensityGate`) or where the conformal set itself is ambiguous
(:class:`~mixle.task.calibrate.CalibratedTaskModel`).

:func:`fit_disagreement_gate` turns a set of ``(text, student_label, teacher_label)`` triples into a compact
binary ``agree``/``disagree`` classifier over the student's own feature space (reusing
:func:`~mixle.task.distill.distill_from_labels` -- the disagreement gate is itself a distilled student,
just of a different target). The resulting :class:`DisagreementGate` exposes ``ood_mask`` with the exact
same duck-typed shape as :class:`~mixle.task.density.DensityGate`, so it plugs into
``CalibratedTaskModel(..., density_gate=...)`` directly -- or unions with a real density gate via
:func:`union_gate` -- with no changes needed to :mod:`mixle.task.calibrate`'s extension point.

:func:`measure_disagreement_mass` is the plain fraction-of-examples-where-student-differs-from-teacher
metric the active-labeling loop (:func:`~mixle.task.active.active_distill`) is measured against: label the
gate-flagged region with the teacher, re-distill including those labels, and confirm the region's mass
shrinks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np

from mixle.task.distill import distill_from_labels
from mixle.task.model import TaskModel


def measure_disagreement_mass(student: TaskModel, texts: Sequence[str], teacher_labels: Sequence[Any]) -> float:
    """Fraction of ``texts`` where the student's label differs from the teacher's."""
    items = list(texts)
    truth = list(teacher_labels)
    if not items:
        raise ValueError("texts must contain at least one example")
    if len(items) != len(truth):
        raise ValueError("texts and teacher_labels must have the same length")
    pred = list(student.batch(items))
    if len(pred) != len(items):
        raise ValueError("student must return exactly one label per input")
    tl = [str(t) for t in truth]
    return float(np.mean([p != t for p, t in zip(pred, tl)]))


@dataclass
class DisagreementGate:
    """A fitted agree/disagree classifier over the student's feature space, plus an escalation threshold."""

    classifier: TaskModel
    threshold: float = 0.5
    calibration_receipt: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")

    def disagreement_proba(self, texts: Sequence[str]) -> np.ndarray:
        """``P(disagree | x)`` under the fitted classifier."""
        items = list(texts)
        prob = np.asarray(self.classifier.adapter.proba_batch(self.classifier.model, items), dtype=np.float64)
        labels = list(self.classifier.adapter.labels)
        if "disagree" not in labels:
            raise ValueError("disagreement classifier does not expose a 'disagree' label")
        if (
            prob.shape != (len(items), len(labels))
            or not np.all(np.isfinite(prob))
            or np.any(prob < 0.0)
            or np.any(prob > 1.0)
            or (len(items) and not np.allclose(prob.sum(axis=1), 1.0, atol=1e-6))
        ):
            raise ValueError("disagreement classifier returned invalid probabilities")
        idx = labels.index("disagree")
        return prob[:, idx]

    def is_ood(self, text: str) -> bool:
        """Return whether one input is predicted to disagree with the teacher."""
        return bool(self.disagreement_proba([text])[0] > self.threshold)

    def ood_mask(self, texts: Sequence[str]) -> np.ndarray:
        """Same duck-typed shape as :meth:`mixle.task.density.DensityGate.ood_mask` -- drops straight into
        ``CalibratedTaskModel(..., density_gate=this)``."""
        return self.disagreement_proba(texts) > self.threshold


def fit_disagreement_gate(
    student: TaskModel,
    texts: Sequence[str],
    teacher_labels: Sequence[Any],
    *,
    dim: int = 256,
    hidden: Sequence[int] = (32,),
    epochs: int = 150,
    lr: float = 1e-2,
    seed: int = 0,
    alpha: float = 0.05,
    calibration_frac: float = 0.2,
    calibration_texts: Sequence[str] | None = None,
    calibration_teacher_labels: Sequence[Any] | None = None,
) -> DisagreementGate:
    """Fit a :class:`DisagreementGate` from a labeled sample: run ``student`` on ``texts``, label each
    example ``"disagree"`` where it differs from ``teacher_labels`` and ``"agree"`` otherwise, and distill a
    compact binary classifier of that target over the same hashed n-gram feature family the student itself
    uses (a different, wider/deeper recipe is fine -- what matters is the classifier learns a decision
    surface over the input text, not that it matches the student's exact recipe).
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    source = [str(t) for t in texts]
    truth = [str(t) for t in teacher_labels]
    if len(source) != len(truth):
        raise ValueError("texts and teacher_labels must have the same length")

    explicit_calibration = calibration_texts is not None or calibration_teacher_labels is not None
    if explicit_calibration:
        if calibration_texts is None or calibration_teacher_labels is None:
            raise ValueError("calibration_texts and calibration_teacher_labels must be provided together")
        train_texts, train_truth = source, truth
        cal_texts = [str(t) for t in calibration_texts]
        cal_truth = [str(t) for t in calibration_teacher_labels]
        if not train_texts or not cal_texts or len(cal_texts) != len(cal_truth):
            raise ValueError("explicit fitting and aligned calibration slices must both be nonempty")
        train_indices = cal_indices = None
        split_kind = "explicit"
    else:
        if not np.isfinite(calibration_frac) or not 0.0 < calibration_frac < 1.0:
            raise ValueError("calibration_frac must be finite and in (0, 1)")
        if len(source) < 2:
            raise ValueError("at least two examples are required for disjoint disagreement calibration")
        permutation = np.random.RandomState(seed).permutation(len(source))
        n_cal = min(len(source) - 1, max(1, int(round(len(source) * calibration_frac))))
        cal_indices = [int(i) for i in permutation[:n_cal]]
        train_indices = [int(i) for i in permutation[n_cal:]]
        train_texts = [source[i] for i in train_indices]
        train_truth = [truth[i] for i in train_indices]
        cal_texts = [source[i] for i in cal_indices]
        cal_truth = [truth[i] for i in cal_indices]
        split_kind = "seeded_internal"

    train_student_labels = list(student.batch(train_texts))
    if len(train_student_labels) != len(train_texts):
        raise ValueError("student must return exactly one label per fitting input")
    disagreement_labels = [
        "disagree" if str(student_label) != teacher_label else "agree"
        for student_label, teacher_label in zip(train_student_labels, train_truth)
    ]
    classifier = distill_from_labels(
        train_texts,
        disagreement_labels,
        labels=["agree", "disagree"],
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task="disagreement gate",
    )
    provisional = DisagreementGate(classifier)
    cal_student_labels = list(student.batch(cal_texts))
    if len(cal_student_labels) != len(cal_texts):
        raise ValueError("student must return exactly one label per calibration input")
    cal_scores = provisional.disagreement_proba(cal_texts)
    agree_mask = np.asarray(
        [str(student_label) == teacher_label for student_label, teacher_label in zip(cal_student_labels, cal_truth)]
    )
    if not np.any(agree_mask):
        raise ValueError("calibration slice must contain at least one student-teacher agreement")
    threshold = float(np.quantile(cal_scores[agree_mask], 1.0 - alpha, method="higher"))
    receipt = {
        "kind": split_kind,
        "seed": seed,
        "alpha": float(alpha),
        "fit_count": len(train_texts),
        "calibration_count": len(cal_texts),
        "calibration_agree_count": int(np.sum(agree_mask)),
        "fit_indices": train_indices,
        "calibration_indices": cal_indices,
        "fit_digest": sha256(repr(list(zip(train_texts, train_truth))).encode("utf-8")).hexdigest(),
        "calibration_digest": sha256(repr(list(zip(cal_texts, cal_truth))).encode("utf-8")).hexdigest(),
        "realized_agree_flag_rate": float(np.mean(cal_scores[agree_mask] > threshold)),
    }
    return DisagreementGate(classifier, threshold=threshold, calibration_receipt=receipt)


class UnionGate:
    """Escalate if ANY constituent gate flags an input -- composes a :class:`DisagreementGate` with a real
    :class:`~mixle.task.density.DensityGate` (or any other ``ood_mask``-exposing gate) with no changes to
    either gate's own code."""

    def __init__(self, *gates: Any) -> None:
        self.gates = gates

    def ood_mask(self, texts: Sequence[str]) -> np.ndarray:
        """Return the elementwise OR of all constituent gate masks."""
        masks = [np.asarray(g.ood_mask(texts), dtype=bool) for g in self.gates]
        return np.logical_or.reduce(masks) if masks else np.zeros(len(texts), dtype=bool)
