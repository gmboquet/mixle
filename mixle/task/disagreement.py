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
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.task.distill import distill_from_labels
from mixle.task.model import TaskModel


def measure_disagreement_mass(student: TaskModel, texts: Sequence[str], teacher_labels: Sequence[Any]) -> float:
    """Fraction of ``texts`` where the student's label differs from the teacher's."""
    if not texts:
        return 0.0
    pred = student.batch(list(texts))
    tl = [str(t) for t in teacher_labels]
    return float(np.mean([p != t for p, t in zip(pred, tl)]))


@dataclass
class DisagreementGate:
    """A fitted agree/disagree classifier over the student's feature space, plus an escalation threshold."""

    classifier: TaskModel
    threshold: float = 0.5

    def disagreement_proba(self, texts: Sequence[str]) -> np.ndarray:
        """``P(disagree | x)`` under the fitted classifier."""
        prob = self.classifier.adapter.proba_batch(self.classifier.model, list(texts))
        idx = self.classifier.adapter.labels.index("disagree")
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
    threshold: float = 0.5,
) -> DisagreementGate:
    """Fit a :class:`DisagreementGate` from a labeled sample: run ``student`` on ``texts``, label each
    example ``"disagree"`` where it differs from ``teacher_labels`` and ``"agree"`` otherwise, and distill a
    compact binary classifier of that target over the same hashed n-gram feature family the student itself
    uses (a different, wider/deeper recipe is fine -- what matters is the classifier learns a decision
    surface over the input text, not that it matches the student's exact recipe).
    """
    texts = [str(t) for t in texts]
    student_labels = student.batch(texts)
    tl = [str(t) for t in teacher_labels]
    disagreement_labels = ["disagree" if s != t else "agree" for s, t in zip(student_labels, tl)]
    classifier = distill_from_labels(
        texts,
        disagreement_labels,
        labels=["agree", "disagree"],
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task="disagreement gate",
    )
    return DisagreementGate(classifier, threshold=threshold)


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
