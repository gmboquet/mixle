"""Pool jobs, results, and backend protocol.

A :class:`PoolJob` describes a runnable unit of work, its input manifest, the
placement reason, estimated cost, and budget. A :class:`Backend` executes the
job and returns a :class:`PoolResult` whose ``artifact`` can be used by the
submitter locally.

The included :class:`LocalBackend` runs jobs in-process and is useful for tests
or systems without a remote pool. Billable backends must require explicit
confirmation, and jobs whose estimated cost exceeds their budget are rejected
before execution.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PoolJob:
    """A runnable unit of work destined for local-or-pool execution, with its reason and budget.

    ``run`` is any callable returning an artifact (a fitted model, an index, a dataset). ``est_cost``
    is the estimated dollar cost the economics assigned; ``budget`` is the ceiling the submitter set.
    ``reason`` is the placement justification the planner produced ("8.2 TFLOP gradient residual").
    """

    run: Callable[[], Any]
    kind: str = "block"  # 'block' | 'verb' | 'index' -- what sort of work this is (telemetry label)
    reason: str = ""  # why this is pool-eligible (from the estimation planner's placement report)
    est_cost: float = 0.0  # estimated dollar cost (economics); 0.0 for the free local backend
    budget: float = float("inf")  # the submitter's cost ceiling
    inputs: dict[str, Any] = field(default_factory=dict)  # a manifest of inputs (paths/hashes), not the data
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class PoolResult:
    """The outcome of a pool job: the artifact that round-trips home, plus realized cost/timing."""

    job_id: str
    status: str  # 'done' | 'rejected' | 'error'
    artifact: Any = None
    cost: float = 0.0
    duration_s: float = 0.0
    reason: str = ""  # rejection/error explanation when status != 'done'

    @property
    def ok(self) -> bool:
        """Whether the job completed successfully."""
        return self.status == "done"


class Backend(Protocol):
    """Executes a :class:`PoolJob`. Real backends set ``billable=True`` and honor the confirm gate."""

    billable: bool

    def submit(self, job: PoolJob) -> PoolResult:
        """Execute ``job`` and return a :class:`PoolResult`."""
        ...


class LocalBackend:
    """The pool degraded to this machine: runs the job in-process, free, no confirm needed.

    This is what a user with no remote pool configured gets: the abstraction
    works end-to-end and every result is a real local artifact. ``clock`` is
    injectable for deterministic timing in tests.
    """

    billable = False

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock

    def submit(self, job: PoolJob) -> PoolResult:
        """Run ``job`` in this process and wrap the outcome as a pool result."""
        t0 = self._tick()
        try:
            artifact = job.run()
        except Exception as exc:  # noqa: BLE001 - a failed job is a result, not a crash of the submitter
            return PoolResult(job.id, "error", reason=f"{type(exc).__name__}: {exc}")
        return PoolResult(job.id, "done", artifact=artifact, cost=0.0, duration_s=self._tick() - t0)

    def _tick(self) -> float:
        if self._clock is not None:
            return float(self._clock())
        import time

        return time.perf_counter()


def submit(
    job: PoolJob,
    backend: Backend | None = None,
    *,
    confirm: bool = False,
    telemetry: Any = None,
) -> PoolResult:
    """Submit ``job`` to ``backend`` (default :class:`LocalBackend`), enforcing budget + confirm rails.

    A job whose ``est_cost`` exceeds its ``budget`` is REJECTED before running. A BILLABLE backend
    (a real GPU pool) additionally requires ``confirm=True`` -- spend is never incurred implicitly.
    Every submission emits a ``pool_job`` telemetry event (features + realized outcome).
    """
    backend = backend or LocalBackend()

    if job.est_cost > job.budget:
        result = PoolResult(job.id, "rejected", reason=f"estimated cost {job.est_cost} exceeds budget {job.budget}")
    elif getattr(backend, "billable", False) and not confirm:
        result = PoolResult(
            job.id,
            "rejected",
            reason="billable backend requires confirm=True (dry-run + explicit confirm; spend is never implicit)",
        )
    else:
        result = backend.submit(job)

    _emit(telemetry, job, backend, result)
    return result


def _emit(telemetry: Any, job: PoolJob, backend: Backend, result: PoolResult) -> None:
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        rec(
            "pool_job",
            features={
                "kind": job.kind,
                "reason": job.reason,
                "est_cost": job.est_cost,
                "budget": job.budget if job.budget != float("inf") else None,
                "backend": type(backend).__name__,
                "billable": bool(getattr(backend, "billable", False)),
            },
            choice=result.status,
            outcome={"cost": result.cost, "duration_s": round(result.duration_s, 6), "ok": result.ok},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break a submission
        pass
