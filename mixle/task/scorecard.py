"""``scorecard`` measures a deployed task route against teacher and task truth.

Point it at a deployed :class:`~mixle.task.solve.Solution` (or a Router), the
teacher it replaces, and a held-out test set. Optionally provide an independent
``task_truth`` callable or aligned label panel to measure real end-to-end
accuracy. Teacher agreement remains a separate distillation metric. Teacher
outputs are cached from the same calls used for latency, and realized cost
charges the student attempt on every request plus the teacher on escalations::

    card = scorecard(
        sol, teacher, test_inputs, task_truth=gold_labels,
        student_cost=0.0001, teacher_cost=0.03,
    )
    print(card.table())

    metric                     student      teacher
    end-to-end accuracy         0.982          —      (against task truth)
    served teacher agreement    0.991          —
    local teacher agreement     0.964          —
    escalation rate             0.11           —
    p50 latency                 0.08 ms      2.1 ms
    artifact size               210 KB         —
    cost / 1k requests          $3.41        $30.00

Without ``task_truth``, end-to-end accuracy is explicitly unmeasured rather
than declaring every teacher escalation correct by construction. ``teacher_agreement``
reports agreement of the served route (local answer or teacher fallback) with
the cached teacher output; ``local_agreement`` covers locally answered rows only.

Every solve shape gets receipts, with agreement meaning that shape's own promise: classification =
exact label match; :class:`~mixle.task.regress.RegressionSolution` = within the caller's ``tol``;
:class:`~mixle.task.multilabel.MultiLabelSolution` = exact set match;
:class:`~mixle.task.structured_out.StructuredSolution` = every categorical field exact and every
numeric field within its own ``tol``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@dataclass
class Scorecard:
    """Evaluation summary for a distilled task service."""

    task: str
    n_test: int
    end_to_end_accuracy: float | None
    teacher_agreement: float
    local_agreement: float
    local_accuracy: float | None
    truth_source: str
    escalation_rate: float
    student_p50_ms: float
    student_p95_ms: float
    teacher_p50_ms: float
    artifact_bytes: int | None
    student_cost_per_1k: float | None
    teacher_cost_per_1k: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return the scorecard fields as a plain dictionary."""
        return dict(self.__dict__)

    def table(self) -> str:
        """Render a compact comparison table for local and teacher service metrics."""
        accuracy = "not measured" if self.end_to_end_accuracy is None else f"{self.end_to_end_accuracy:.3f}"
        rows: list[tuple[str, str, str]] = [
            ("end-to-end accuracy (task truth)", accuracy, "—"),
            ("served teacher agreement", f"{self.teacher_agreement:.3f}", "—"),
            ("local teacher agreement", f"{self.local_agreement:.3f}", "—"),
            ("escalation rate", f"{self.escalation_rate:.3f}", "—"),
            ("p50 latency", f"{self.student_p50_ms:.2f} ms", f"{self.teacher_p50_ms:.2f} ms"),
            ("p95 latency", f"{self.student_p95_ms:.2f} ms", "—"),
        ]
        if self.local_accuracy is not None:
            rows.insert(3, ("local accuracy (task truth)", f"{self.local_accuracy:.3f}", "—"))
        if self.artifact_bytes is not None:
            rows.append(("artifact size", _fmt_bytes(float(self.artifact_bytes)), "—"))
        if self.student_cost_per_1k is not None and self.teacher_cost_per_1k is not None:
            rows.append(("cost / 1k requests", f"${self.student_cost_per_1k:.2f}", f"${self.teacher_cost_per_1k:.2f}"))
        w = max(len(r[0]) for r in rows)
        head = f"{'metric'.ljust(w)}   {'student':>12}   {'teacher':>12}   (task: {self.task}, n={self.n_test})"
        return "\n".join([head] + [f"{a.ljust(w)}   {b:>12}   {c:>12}" for a, b, c in rows])


def _local_decider(student: Any) -> Any:
    """The teacher-free half of ``student`` — the shape's local answer, or ``None`` = escalate."""
    from mixle.task.multilabel import MultiLabelSolution
    from mixle.task.regress import RegressionSolution
    from mixle.task.structured_out import StructuredSolution

    if isinstance(student, RegressionSolution):
        return lambda x: float(student._predict([x])[0]) if student.answers_locally else None
    if isinstance(student, (MultiLabelSolution, StructuredSolution)):
        return student.try_local
    model = student.cascade.model if hasattr(student, "cascade") else student
    return model.decide


def _agrees(student: Any, a: Any, y: Any) -> bool:
    """Does local answer ``a`` meet the shape's own promise against reference ``y``?"""
    from mixle.task.multilabel import MultiLabelSolution
    from mixle.task.regress import RegressionSolution
    from mixle.task.structured_out import StructuredSolution

    if isinstance(student, RegressionSolution):
        return abs(float(a) - float(y)) <= student.tol
    if isinstance(student, MultiLabelSolution):
        return sorted(a) == sorted(y)
    if isinstance(student, StructuredSolution):
        if any(abs(float(a[k]) - float(y[k])) > sub.tol for k, sub in student.fields_num.items()):
            return False
        return all(str(a[k]) == str(y[k]) for k in student.fields_cat)
    return a == y


def _artifact_bytes(student: Any, model: Any) -> int | None:
    try:
        from mixle.task.edge import footprint

        return int(footprint(model.task if hasattr(model, "task") else model).bytes)
    except Exception:  # noqa: BLE001 - fall through to the on-disk truth
        pass
    if hasattr(student, "save"):
        import tempfile
        from pathlib import Path

        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(student.save(str(Path(d) / "artifact")))
                return sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        except Exception:  # noqa: BLE001 - size is a nicety; never fail the receipts over it
            pass
    return None


def scorecard(
    student: Any,
    teacher: Any,
    test_inputs: Any,
    *,
    task_truth: Callable[[Any], Any] | Sequence[Any] | None = None,
    student_cost: float | None = None,
    teacher_cost: float | None = None,
    task: str = "task",
) -> Scorecard:
    """Measure a deployed route against its teacher and optional independent truth.

    Args:
        student: any solve shape — :class:`~mixle.task.solve.Solution`,
            :class:`~mixle.task.regress.RegressionSolution`, :class:`~mixle.task.multilabel.MultiLabelSolution`,
            :class:`~mixle.task.structured_out.StructuredSolution` (or anything exposing
            ``cascade.model.decide``) — the escalate-aware system under test.
        teacher: the callable being replaced; used only as the distillation-agreement reference.
        test_inputs: held-out inputs. The teacher is called exactly once per input; that call supplies
            both the cached reference output and its latency.
        task_truth: independent truth callable or an aligned sequence of gold outputs. When omitted,
            ``end_to_end_accuracy`` is ``None`` rather than a tautological teacher-fallback score.
        student_cost / teacher_cost: optional non-negative per-request costs. A served route costs
            ``student_cost`` on every attempted request plus ``teacher_cost`` on every escalation.
        task: a label for the table header.
    """
    if not callable(teacher):
        raise TypeError("teacher must be callable")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    if isinstance(test_inputs, (str, bytes)):
        raise TypeError("test_inputs must be a sequence of requests")
    xs = list(test_inputs)
    if not xs:
        raise ValueError("scorecard needs a non-empty test set")
    if (student_cost is None) != (teacher_cost is None):
        raise ValueError("student_cost and teacher_cost must be provided together")
    if student_cost is not None:
        for value, name in ((student_cost, "student_cost"), (teacher_cost, "teacher_cost")):
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        student_cost, teacher_cost = float(student_cost), float(teacher_cost)

    teacher_outputs: list[Any] = []
    t_lat: list[float] = []
    for x in xs:
        t0 = time.perf_counter()
        teacher_outputs.append(teacher(x))
        t_lat.append(time.perf_counter() - t0)

    gold: list[Any] | None
    if task_truth is None:
        gold = None
        truth_source = "not_measured"
    elif callable(task_truth):
        gold = [task_truth(x) for x in xs]
        truth_source = "callable"
    else:
        if isinstance(task_truth, (str, bytes)):
            raise TypeError("task_truth must be a callable or an aligned output sequence")
        gold = list(task_truth)
        if len(gold) != len(xs):
            raise ValueError("task_truth must contain exactly one output per test input")
        truth_source = "provided_panel"

    model = student.cascade.model if hasattr(student, "cascade") else student
    decide = _local_decider(student)
    if not callable(decide):
        raise TypeError("student must expose a callable local decision surface")
    local: list[Any] = []
    lat: list[float] = []
    for x in xs:
        t0 = time.perf_counter()
        local.append(decide(x))
        lat.append(time.perf_counter() - t0)

    escalated = np.asarray([a is None for a in local])
    answered = ~escalated
    agree = (
        float(
            np.mean(
                [
                    _agrees(student, answer, reference)
                    for answer, reference, is_answered in zip(
                        local,
                        teacher_outputs,
                        answered,
                        strict=True,
                    )
                    if is_answered
                ]
            )
        )
        if answered.any()
        else float("nan")
    )
    served = [
        reference if is_escalated else answer
        for answer, reference, is_escalated in zip(
            local,
            teacher_outputs,
            escalated,
            strict=True,
        )
    ]
    teacher_agreement = float(
        np.mean(
            [_agrees(student, output, reference) for output, reference in zip(served, teacher_outputs, strict=True)]
        )
    )
    if gold is None:
        end_to_end = None
        local_accuracy = None
    else:
        end_to_end = float(
            np.mean([_agrees(student, output, target) for output, target in zip(served, gold, strict=True)])
        )
        local_accuracy = (
            float(
                np.mean(
                    [
                        _agrees(student, answer, target)
                        for answer, target, is_answered in zip(
                            local,
                            gold,
                            answered,
                            strict=True,
                        )
                        if is_answered
                    ]
                )
            )
            if answered.any()
            else float("nan")
        )
    esc_rate = float(escalated.mean())

    artifact_bytes = _artifact_bytes(student, model)

    s_1k = t_1k = None
    if student_cost is not None and teacher_cost is not None:
        blended = student_cost + esc_rate * teacher_cost
        s_1k, t_1k = 1000.0 * blended, 1000.0 * teacher_cost

    lat_ms = 1e3 * np.asarray(lat)
    return Scorecard(
        task=task,
        n_test=len(xs),
        end_to_end_accuracy=end_to_end,
        teacher_agreement=teacher_agreement,
        local_agreement=agree,
        local_accuracy=local_accuracy,
        truth_source=truth_source,
        escalation_rate=esc_rate,
        student_p50_ms=float(np.percentile(lat_ms, 50)),
        student_p95_ms=float(np.percentile(lat_ms, 95)),
        teacher_p50_ms=float(np.percentile(1e3 * np.asarray(t_lat), 50)),
        artifact_bytes=artifact_bytes,
        student_cost_per_1k=s_1k,
        teacher_cost_per_1k=t_1k,
    )
