"""``scorecard`` measures a deployed task route against its teacher.

Point it at a deployed :class:`~mixle.task.solve.Solution` (or a Router), the
teacher it replaces, and a held-out test set. The result measures end-to-end
accuracy against the teacher, local-answer agreement, escalation rate,
wall-clock latency for both sides, artifact size, and, when costs are supplied,
realized dollars per thousand requests::

    card = scorecard(sol, teacher, test_inputs, student_cost=0.0001, teacher_cost=0.03)
    print(card.table())

    metric                     student      teacher
    end-to-end accuracy         1.000          —      (escalations answered by the teacher)
    local agreement             0.964          —
    escalation rate             0.11           —
    p50 latency                 0.08 ms      2.1 ms
    artifact size               210 KB         —
    cost / 1k requests          $3.41        $30.00

"End-to-end accuracy" counts escalated requests as correct because the teacher
answered them. "Local agreement" is the student alone on requests it chose to
answer. Reporting both avoids hiding local-model errors behind the fallback.

Every solve shape gets receipts, with agreement meaning that shape's own promise: classification =
exact label match; :class:`~mixle.task.regress.RegressionSolution` = within the caller's ``tol``;
:class:`~mixle.task.multilabel.MultiLabelSolution` = exact set match;
:class:`~mixle.task.structured_out.StructuredSolution` = every categorical field exact and every
numeric field within its own ``tol``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    end_to_end_accuracy: float
    local_agreement: float
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
        rows: list[tuple[str, str, str]] = [
            ("end-to-end accuracy", f"{self.end_to_end_accuracy:.3f}", "—"),
            ("local agreement", f"{self.local_agreement:.3f}", "—"),
            ("escalation rate", f"{self.escalation_rate:.3f}", "—"),
            ("p50 latency", f"{self.student_p50_ms:.2f} ms", f"{self.teacher_p50_ms:.2f} ms"),
            ("p95 latency", f"{self.student_p95_ms:.2f} ms", "—"),
        ]
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
    student_cost: float | None = None,
    teacher_cost: float | None = None,
    task: str = "task",
) -> Scorecard:
    """Measure a deployed student against the teacher it replaces on held-out inputs (see module docstring).

    Args:
        student: any solve shape — :class:`~mixle.task.solve.Solution`,
            :class:`~mixle.task.regress.RegressionSolution`, :class:`~mixle.task.multilabel.MultiLabelSolution`,
            :class:`~mixle.task.structured_out.StructuredSolution` (or anything exposing
            ``cascade.model.decide``) — the escalate-aware system under test.
        teacher: the callable being replaced; also the accuracy reference.
        test_inputs: held-out inputs (the teacher is called once per input for the reference labels).
        student_cost / teacher_cost: optional per-request costs for the $/1k rows. The blended student
            cost prices escalated requests at ``teacher_cost``.
        task: a label for the table header.
    """
    xs = list(test_inputs)
    if not xs:
        raise ValueError("scorecard needs a non-empty test set")

    truth = [teacher(x) for x in xs]

    model = student.cascade.model if hasattr(student, "cascade") else student
    decide = _local_decider(student)
    local: list[Any] = []
    lat: list[float] = []
    for x in xs:
        t0 = time.perf_counter()
        local.append(decide(x))
        lat.append(time.perf_counter() - t0)

    t_lat: list[float] = []
    for x in xs[: min(len(xs), 50)]:
        t0 = time.perf_counter()
        teacher(x)
        t_lat.append(time.perf_counter() - t0)

    escalated = np.asarray([a is None for a in local])
    answered = ~escalated
    agree = (
        float(np.mean([_agrees(student, a, y) for a, y, m in zip(local, truth, answered) if m]))
        if answered.any()
        else float("nan")
    )
    end_to_end = float(np.mean([True if e else _agrees(student, a, y) for a, y, e in zip(local, truth, escalated)]))
    esc_rate = float(escalated.mean())

    artifact_bytes = _artifact_bytes(student, model)

    s_1k = t_1k = None
    if student_cost is not None and teacher_cost is not None:
        blended = (1.0 - esc_rate) * student_cost + esc_rate * teacher_cost
        s_1k, t_1k = 1000.0 * blended, 1000.0 * teacher_cost

    lat_ms = 1e3 * np.asarray(lat)
    return Scorecard(
        task=task,
        n_test=len(xs),
        end_to_end_accuracy=end_to_end,
        local_agreement=agree,
        escalation_rate=esc_rate,
        student_p50_ms=float(np.percentile(lat_ms, 50)),
        student_p95_ms=float(np.percentile(lat_ms, 95)),
        teacher_p50_ms=float(np.percentile(1e3 * np.asarray(t_lat), 50)),
        artifact_bytes=artifact_bytes,
        student_cost_per_1k=s_1k,
        teacher_cost_per_1k=t_1k,
    )
