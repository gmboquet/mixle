"""Frontier -> mixle-native students: distill, LNS-compress, calibrate, cascade -- the loop closed.

The end-to-end pipeline the roadmap calls "J4": take a frontier/teacher model (large, expensive,
general-purpose), distill it into a SMALL, TASK-SPECIFIC student, re-execute that student's inference
in :class:`mixle.engines.lns.LogNumberSystem`'s integer log-space (compact, transcendental-free), wrap
it in :class:`~mixle.task.calibrate.CalibratedTaskModel` for an honest answer-or-escalate decision, and
compose it with the teacher into a :class:`~mixle.task.cascade.Cascade` for served, tiered inference.

This module is deliberately thin: every piece already exists --

  * :func:`mixle.task.distill.distill_structured` distills a teacher into a structured probabilistic
    student (a learned dependency network: kilobytes, torch-free, an exact posterior).
  * :func:`mixle.task.quantize.lns_classifier` re-executes that student's inference in the
    :class:`~mixle.engines.lns.LogNumberSystem` integer log-space (the same LNS ``task.quantize``
    already applies for compute quantization).
  * :class:`~mixle.task.calibrate.CalibratedTaskModel` calibrates a conformal answer/escalate
    threshold on held-out data (:class:`~mixle.task.calibrated_generator.CalibratedGenerator` is the
    generative sibling -- it also exposes ``decide()``, so it drops into :class:`Cascade` unmodified
    if the task is generative rather than classification).
  * :class:`~mixle.task.cascade.Cascade` serves the calibrated LNS student first, escalating only
    ambiguous/OOD requests to the teacher, and tracks realized cost.
  * :func:`mixle.task.edge.footprint` measures the LNS student's real deployment bytes.

:func:`distill_to_lns_student` / :func:`build_served_cascade` / :func:`measure_cascade_receipt` wire
those five subsystems together and report the two numbers the roadmap item's acceptance criteria ask
for: a served cascade cost/quality receipt, and the edge student's footprint + student-teacher
agreement rate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.task.calibrate import CalibratedTaskModel
from mixle.task.cascade import Cascade
from mixle.task.distill import _as_batched, distill_structured
from mixle.task.economics import CostModel
from mixle.task.edge import footprint
from mixle.task.model import TaskModel
from mixle.task.quantize import lns_classifier

__all__ = ["CascadeReceipt", "distill_to_lns_student", "build_served_cascade", "measure_cascade_receipt"]


def distill_to_lns_student(
    teacher: Callable[..., Any],
    task_data: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    n_components: int = 1,
    min_gain: float = 0.0,
    n_bins: int = 4,
    max_its: int = 30,
    step: float = 1e-2,
    seed: int = 0,
    task: str = "",
    n_jobs: int = 1,
) -> TaskModel:
    """Distill ``teacher`` into a small, task-specific structured student, then LNS-compress it.

    Reuses :func:`~mixle.task.distill.distill_structured` for the distillation (the teacher labels
    ``task_data`` once; the student discovers the joint dependency structure and classifies
    generatively) and :func:`~mixle.task.quantize.lns_classifier` for the LNS conversion -- no new
    quantization or distillation logic, just the existing rungs composed. The returned
    :class:`~mixle.task.model.TaskModel` runs inference as integer add/max/LUT above the leaf boundary
    (:mod:`mixle.engines.lns`) and needs no torch.
    """
    student = distill_structured(
        teacher,
        task_data,
        labels=labels,
        n_components=n_components,
        min_gain=min_gain,
        n_bins=n_bins,
        max_its=max_its,
        seed=seed,
        task=task,
        n_jobs=n_jobs,
    )
    return lns_classifier(student, step=step)


def build_served_cascade(
    lns_student: TaskModel,
    teacher: Callable[..., Any],
    cal_data: Sequence[Any],
    cal_labels: Sequence[Any] | None = None,
    *,
    alpha: float = 0.1,
    cost: CostModel | None = None,
) -> Cascade:
    """Calibrate the LNS student and compose it with ``teacher`` into a served :class:`Cascade`.

    ``cal_data`` is a held-out slice (disjoint from ``lns_student``'s training data) used to fit the
    conformal answer/escalate threshold (:meth:`~mixle.task.calibrate.CalibratedTaskModel.calibrate`).
    If ``cal_labels`` is omitted, the teacher labels ``cal_data`` itself (one batched call) -- the same
    "teacher is the ground truth for calibration" convention :func:`~mixle.task.distill.distill_for_routing`
    uses. The returned :class:`~mixle.task.cascade.Cascade` answers locally when the LNS student's
    conformal set is a confident singleton, and escalates to ``teacher`` otherwise.
    """
    cal_data = list(cal_data)
    if cal_labels is None:
        cal_labels = _as_batched(teacher)(cal_data)
    else:
        cal_labels = list(cal_labels)
    calibrated = CalibratedTaskModel(lns_student, alpha=alpha).calibrate(cal_data, cal_labels)
    return Cascade(calibrated, teacher, cost=cost)


@dataclass(frozen=True)
class CascadeReceipt:
    """The served cascade's cost/quality tradeoff, plus the edge student's footprint and agreement.

    ``*_cost_per_request`` are the per-request costs (student-only always local, teacher-only always
    escalates, cascade the realized mix); the whole point of cascading is that ``cascade_cost`` lands
    near ``student_cost`` while ``cascade_quality`` lands near (or measurably closer to)
    ``teacher_quality`` -- see :meth:`earns_its_complexity`. ``student_bytes``/``teacher_bytes`` are the
    real, measured deployment footprints (:func:`~mixle.task.edge.footprint` for the student); disk
    ``compression_ratio`` is ``teacher_bytes / student_bytes`` when a teacher footprint is supplied.
    ``agreement_rate`` is the fraction of the held-out test set where the LNS student's own answer
    (not the cascade's escalate-mediated answer) matches the teacher's.
    """

    n_requests: int
    n_escalated: int
    student_cost_per_request: float
    teacher_cost_per_request: float
    cascade_cost_per_request: float
    student_quality: float
    teacher_quality: float
    cascade_quality: float
    student_bytes: int
    teacher_bytes: int | None
    compression_ratio: float | None
    agreement_rate: float

    def earns_its_complexity(self, *, tol: float = 1e-9) -> bool:
        """Whether the cascade actually beats the extremes it sits between.

        Cost: cascading costs ``c_local + p_escalate * c_frontier`` per request, so it can never be
        cheaper than the pure-local student -- but it must be strictly cheaper than always paying the
        teacher (``tol`` allows the degenerate zero-escalation case, where cascade cost equals the
        student's exactly). Quality: the cascade must be at least as good as the student alone (the
        escalations it does pay for should be net-positive, not wasted spend).
        """
        cost_between = (
            self.student_cost_per_request - tol <= self.cascade_cost_per_request <= self.teacher_cost_per_request + tol
        )
        better_than_student = self.cascade_quality >= self.student_quality - tol
        return cost_between and better_than_student

    def summary(self) -> str:
        comp = f"{self.compression_ratio:.1f}x" if self.compression_ratio is not None else "n/a"
        return (
            f"served {self.n_requests} requests, escalated {self.n_escalated} "
            f"({self.n_escalated / self.n_requests:.0%})\n"
            f"  cost/req   student ${self.student_cost_per_request:.5f}  "
            f"cascade ${self.cascade_cost_per_request:.5f}  teacher ${self.teacher_cost_per_request:.5f}\n"
            f"  quality    student {self.student_quality:.3f}  "
            f"cascade {self.cascade_quality:.3f}  teacher {self.teacher_quality:.3f}\n"
            f"  footprint  student {self.student_bytes}B  teacher "
            f"{self.teacher_bytes if self.teacher_bytes is not None else 'n/a'}B  compression {comp}\n"
            f"  student-teacher agreement {self.agreement_rate:.3f}"
        )


def measure_cascade_receipt(
    cascade: Cascade,
    test_data: Sequence[Any],
    truth_labels: Sequence[Any],
    *,
    teacher_bytes: int | None = None,
) -> CascadeReceipt:
    """Serve ``test_data`` through ``cascade`` and measure the real cost/quality/footprint/agreement receipt.

    Reuses the machinery already built for this, rather than re-deriving it: :meth:`Cascade.serve` (real
    serving, so ``cascade``'s stats/realized cost are genuine, not simulated), :func:`~mixle.task.edge.footprint`
    for the student's measured deployment bytes, and plain accuracy-vs-``truth_labels`` for quality (the
    student and teacher are scored on the SAME held-out set the cascade was served, so all three numbers are
    directly comparable). ``teacher_bytes`` is the caller-supplied measured/declared footprint of the frontier
    model (opaque to mixle -- it is not a mixle artifact) used only for the reported compression ratio.
    """
    if cascade.cost is None:
        raise ValueError("measure_cascade_receipt needs a Cascade built with a CostModel (cascade.cost)")

    test_data = list(test_data)
    truth = [str(t) for t in truth_labels]
    if len(test_data) != len(truth):
        raise ValueError("test_data and truth_labels must be the same length")

    student = cascade.model.task  # the underlying (LNS-compressed) TaskModel wrapped by CalibratedTaskModel
    student_preds = [str(p) for p in student.batch(test_data)]
    teacher_preds = [str(p) for p in _as_batched(cascade.teacher)(test_data)]

    student_quality = float(np.mean([p == t for p, t in zip(student_preds, truth)]))
    teacher_quality = float(np.mean([p == t for p, t in zip(teacher_preds, truth)]))
    agreement_rate = float(np.mean([s == t for s, t in zip(student_preds, teacher_preds)]))

    cascade_preds = [str(p) for p in cascade.serve(test_data)]  # real serving: updates cascade.stats
    cascade_quality = float(np.mean([p == t for p, t in zip(cascade_preds, truth)]))

    n_requests = cascade.stats.n_requests
    cascade_cost_per_request = cascade.realized_cost() / n_requests if n_requests else 0.0

    student_bytes = footprint(student).bytes

    return CascadeReceipt(
        n_requests=n_requests,
        n_escalated=cascade.stats.n_escalated,
        student_cost_per_request=cascade.cost.c_local,
        teacher_cost_per_request=cascade.cost.c_frontier,
        cascade_cost_per_request=cascade_cost_per_request,
        student_quality=student_quality,
        teacher_quality=teacher_quality,
        cascade_quality=cascade_quality,
        student_bytes=student_bytes,
        teacher_bytes=teacher_bytes,
        compression_ratio=(teacher_bytes / student_bytes) if teacher_bytes else None,
        agreement_rate=agreement_rate,
    )
