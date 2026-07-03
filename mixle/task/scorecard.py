"""``scorecard`` -- the receipts: tiny model vs frontier on YOUR task, measured, in one table.

The artifact that wins arguments. Point it at a deployed :class:`~mixle.task.solve.Solution` (or a
Router), the teacher it replaces, and a held-out test set; it MEASURES — never projects — end-to-end
accuracy against the teacher, local-answer agreement, escalation rate, wall-clock latency for both
sides, artifact size, and (given per-request costs) realized $/1k::

    card = scorecard(sol, teacher, test_inputs, student_cost=0.0001, teacher_cost=0.03)
    print(card.table())

    metric                     student      teacher
    end-to-end accuracy         1.000          —      (escalations answered BY the teacher)
    local agreement             0.964          —
    escalation rate             0.11           —
    p50 latency                 0.08 ms      2.1 ms
    artifact size               210 KB         —
    cost / 1k requests          $3.41        $30.00

"End-to-end accuracy" counts escalated requests as correct-by-construction (the teacher answered
them); "local agreement" is the student alone on the requests it chose to answer — both are reported
so the honest story is visible: the system is never worse than the teacher, and here is exactly how
much of the work the tiny model absorbed.
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
        return dict(self.__dict__)

    def table(self) -> str:
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
        student: a :class:`~mixle.task.solve.Solution` (or anything exposing ``cascade.model.decide``
            and ``__call__``) — the escalate-aware system under test.
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
    local: list[Any] = []
    lat: list[float] = []
    for x in xs:
        t0 = time.perf_counter()
        local.append(model.decide(x))
        lat.append(time.perf_counter() - t0)

    t_lat: list[float] = []
    for x in xs[: min(len(xs), 50)]:
        t0 = time.perf_counter()
        teacher(x)
        t_lat.append(time.perf_counter() - t0)

    escalated = np.asarray([a is None for a in local])
    answered = ~escalated
    agree = float(np.mean([a == y for a, y, m in zip(local, truth, answered) if m])) if answered.any() else float("nan")
    end_to_end = float(np.mean([(y if e else a) == y for a, y, e in zip(local, truth, escalated)]))
    esc_rate = float(escalated.mean())

    artifact_bytes: int | None = None
    try:
        from mixle.task.edge import footprint

        artifact_bytes = int(footprint(model.task if hasattr(model, "task") else model).bytes)
    except Exception:  # noqa: BLE001 - size is a nicety; never fail the receipts over it
        artifact_bytes = None

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
