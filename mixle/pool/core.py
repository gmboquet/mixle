"""Pool jobs, results, and backend protocol.

A :class:`PoolJob` describes a runnable unit of work, its input manifest, the
placement reason, estimated cost, and budget. A :class:`Backend` executes the
job and returns a :class:`PoolResult` whose ``artifact`` can be used by the
submitter locally.

The included :class:`LocalBackend` runs jobs in-process and is useful for tests
or systems without a remote pool. Billable backends must require explicit
confirmation, and jobs whose estimated cost exceeds their budget are rejected
before execution. A backend is third-party code, so its response is not taken on
trust either: :func:`submit` settles it against the job it was given (identity,
status vocabulary, finite economics, realized cost within budget) before the
caller ever sees ``ok == True``.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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


POOL_STATUSES = frozenset({"done", "rejected", "error"})
"""The closed status vocabulary a :class:`PoolResult` may use.

:func:`submit` refuses anything outside this set rather than letting an unrecognized status ride
through: :attr:`PoolResult.ok` only tests equality with ``"done"``, so an unknown status would read as
"not ok" while carrying none of the error handling a real failure gets.
"""


@dataclass
class PoolResult:
    """The outcome of a pool job: the artifact that round-trips home, plus realized cost/timing."""

    job_id: str
    status: str  # one of POOL_STATUSES: 'done' | 'rejected' | 'error'
    artifact: Any = None
    cost: float = 0.0
    duration_s: float = 0.0
    reason: str = ""  # rejection/error explanation when status != 'done'
    telemetry_error: str = ""  # why the promised ``pool_job`` event was not emitted; "" when it was

    @property
    def ok(self) -> bool:
        """Whether the job completed successfully."""
        return self.status == "done"

    @property
    def telemetry_recorded(self) -> bool:
        """Whether this submission's ``pool_job`` event actually reached the telemetry sink.

        Separate from :attr:`ok`: the job itself can succeed while its record is lost, and the two
        failures need different responses -- a dropped event means the spend and duration behind this
        result are missing from whatever ledger reconciles them.
        """
        return not self.telemetry_error


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


def _is_finite_nonneg(value: Any) -> bool:
    """Whether ``value`` is a real number that is finite and ``>= 0`` (non-numbers are simply not)."""
    try:
        return bool(math.isfinite(value)) and value >= 0
    except TypeError:
        return False


def _settle(job: PoolJob, result: Any) -> PoolResult:
    """Check a backend's response against ``job`` before the submitter is allowed to trust it.

    A :class:`Backend` is third-party code: it can be buggy, out of date, or (for a remote pool)
    answering across a network where responses can be misrouted. Without this check ``submit()``
    returned whatever came back, so a response naming a *different* job, carrying an unrecognized
    status, reporting a NaN duration, or billing far past the ceiling the submitter set was handed
    to the caller as an ordinary successful result with ``ok == True``.

    Any violation is converted into an ``"error"`` result for *this* job rather than raised: a bad
    response is an outcome of the submission, the same way a job that raised is (see
    :meth:`LocalBackend.submit`), and the caller's ``.ok`` check is then the single place that decides
    whether the artifact may be used. The realized ``cost`` is preserved on the budget-overrun path
    specifically -- that spend really happened and still needs to be reconciled and telemetered, even
    though the artifact it produced must not be consumed as though the ceiling had held.

    Note the ordering this can and cannot promise: ``est_cost <= budget`` is checked *before*
    dispatch, so an over-estimate never runs at all, but the realized-cost check below necessarily
    happens after the backend already did the work. Preventing over-spend (rather than detecting it)
    requires a ceiling the backend itself enforces -- an authorization reserved before dispatch and
    cancelled when it is exhausted -- which is a property of the backend protocol, not something this
    in-process wrapper can supply on a backend's behalf.
    """
    if not isinstance(result, PoolResult):
        return PoolResult(job.id, "error", reason=f"backend returned {type(result).__name__}, not a PoolResult")
    if result.job_id != job.id:
        return PoolResult(
            job.id, "error", reason=f"backend returned a result for job {result.job_id!r}, not {job.id!r}"
        )
    if result.status not in POOL_STATUSES:
        return PoolResult(
            job.id,
            "error",
            reason=f"backend returned unknown status {result.status!r}, expected one of {sorted(POOL_STATUSES)}",
        )
    if not _is_finite_nonneg(result.cost):
        return PoolResult(job.id, "error", reason=f"backend reported a non-finite or negative cost: {result.cost!r}")
    if not _is_finite_nonneg(result.duration_s):
        return PoolResult(
            job.id, "error", reason=f"backend reported a non-finite or negative duration_s: {result.duration_s!r}"
        )
    if result.cost > job.budget:
        return PoolResult(
            job.id,
            "error",
            cost=result.cost,
            duration_s=result.duration_s,
            reason=f"realized cost {result.cost} exceeds budget {job.budget}",
        )
    return result


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
    Every submission emits a ``pool_job`` telemetry event (features + realized outcome). A telemetry
    sink that fails does not fail the job -- but it does not disappear either: the reason lands on
    ``result.telemetry_error`` (and ``result.telemetry_recorded`` goes false), so a caller
    reconciling spend against that event stream can see which rows are missing from it.

    The confirm gate demands the literal Boolean ``True``, not merely a truthy value: ``confirm`` is
    routinely threaded through JSON payloads, CLI flags and form fields, where the *string* ``"false"``
    (or ``"0"``, or an empty-but-present marker object) is a perfectly ordinary way to spell "no" --
    and every one of those is truthy in Python. Anything other than ``True`` therefore leaves the job
    unconfirmed and rejected rather than silently authorizing billable execution.

    ``job.est_cost`` must be finite and non-negative and ``job.budget`` must be non-negative (``inf``
    allowed, meaning "no ceiling") -- raises :class:`ValueError` otherwise. Costs are a dollar estimate,
    never a credit: a NaN or negative value would defeat the comparison below rather than be caught by
    it (NaN compares false against everything, so ``est_cost > budget`` silently passes; a negative
    value can "refund" budget that was never spent), so both are rejected outright rather than routed
    through the ordinary rejected-:class:`PoolResult` path.

    The backend's *response* is checked too, by :func:`_settle`: it must be a :class:`PoolResult` for
    this exact job, with a status in :data:`POOL_STATUSES` and a finite non-negative ``cost`` and
    ``duration_s`` that settles within ``job.budget``. A response failing any of those is returned as
    an ``"error"`` result instead of being passed through, so ``.ok`` is never true for work whose
    identity or economics do not check out. A backend that *raises* rather than responding is
    likewise an ``"error"`` result naming the backend and the exception, not an exception out of
    ``submit``: the return contract and the telemetry guarantee both have to hold on the path where
    the pool is down, which is the path an operator most needs a record of.

    Submission is NOT idempotent and deliberately keeps no durable job state: submitting the same
    :class:`PoolJob` twice runs it twice, even when its ``id`` is a caller-supplied constant rather
    than the default fresh UUID (``id`` is a correlation handle for telemetry and settlement, not a
    deduplication key). A caller whose work is irreversible or billable -- and any orchestrator that
    retries after a crash or a lost response -- must therefore carry its own idempotency key and
    committed-result lookup; that needs durable storage, and this library layer does no I/O.
    """
    if not math.isfinite(job.est_cost) or job.est_cost < 0:
        raise ValueError(f"est_cost must be finite and non-negative, got {job.est_cost!r}")
    if math.isnan(job.budget) or job.budget < 0:
        raise ValueError(f"budget must be non-negative (inf allowed for no ceiling), got {job.budget!r}")

    backend = backend or LocalBackend()

    if job.est_cost > job.budget:
        result = PoolResult(job.id, "rejected", reason=f"estimated cost {job.est_cost} exceeds budget {job.budget}")
    elif getattr(backend, "billable", False) and confirm is not True:
        result = PoolResult(
            job.id,
            "rejected",
            reason=(
                "billable backend requires the literal confirm=True (dry-run + explicit confirm; spend "
                f"is never implicit), got {confirm!r}"
            ),
        )
    else:
        try:
            response = backend.submit(job)
        except Exception as exc:  # noqa: BLE001 - a backend that fails is an outcome, not a crash of the submitter
            # LocalBackend already turns a raising *job* into an error result; a raising *backend*
            # -- an unreachable GPU pool, an expired credential, a driver fault -- had no such path,
            # so it escaped submit() entirely. Two documented guarantees broke at once: callers were
            # promised a PoolResult to check .ok on and got an exception instead, and "every
            # submission emits a pool_job telemetry event" silently excluded the failures most worth
            # recording -- including a billable dispatch that may already have incurred spend.
            result = PoolResult(
                job.id, "error", reason=f"backend {type(backend).__name__} raised {type(exc).__name__}: {exc}"
            )
        else:
            result = _settle(job, response)

    failure = _emit(telemetry, job, backend, result)
    if failure:
        result = replace(result, telemetry_error=failure)
    return result


def _emit(telemetry: Any, job: PoolJob, backend: Backend, result: PoolResult) -> str:
    """Emit the ``pool_job`` event. Returns "" on success, or why it failed -- never raises.

    A telemetry outage must not turn a completed job into a failure, but it must not vanish either:
    the docstring above promises *every* submission emits an event, and a caller reconciling spend
    against that ledger has no way to notice a silently dropped row. The reason travels back on the
    result so the broken guarantee is visible exactly where the job's cost is.
    """
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
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a submission
        return f"pool_job telemetry event not recorded: {type(exc).__name__}: {exc}"
    return ""
