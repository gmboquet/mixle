"""Time-to-target traces and negative-control failure receipts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from typing import Any

from mixle.experimental.typed_runtime.contracts import CounterSemantics
from mixle.utils.immutable import freeze_receipt_container


def _nonnegative_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _finite_real(value: Any, name: str, *, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative.")


class TargetDirection(StrEnum):
    """Whether reaching a target requires increasing or decreasing a metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class ObjectiveTarget:
    """Explicit quality threshold for a time-to-target benchmark."""

    name: str
    direction: TargetDirection
    threshold: float
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("target name must be non-empty.")
        if not isinstance(self.direction, TargetDirection):
            raise TypeError("target direction must be TargetDirection.")
        _finite_real(self.threshold, "target threshold")
        _finite_real(self.tolerance, "target tolerance", nonnegative=True)

    def reached(self, value: float) -> bool:
        """Whether ``value`` reaches this target within tolerance."""

        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            return False
        if self.direction is TargetDirection.MAXIMIZE:
            return value + self.tolerance >= self.threshold
        return value - self.tolerance <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible target declaration."""

        return {
            "name": self.name,
            "direction": self.direction.value,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class BenchmarkPoint:
    """One cumulative quality/work observation from an actual run."""

    step: int
    objective: float
    elapsed_seconds: float
    operation_count: int = 0
    model_evaluations: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    collective_bytes: int = 0
    peak_memory_bytes: int = 0
    maximum_staleness_steps: int = 0
    accepted_updates: int = 0
    rejected_updates: int = 0
    work_counter_semantics: CounterSemantics = CounterSemantics.CUMULATIVE
    peak_memory_semantics: CounterSemantics = CounterSemantics.HIGH_WATER_MARK
    staleness_semantics: CounterSemantics = CounterSemantics.HIGH_WATER_MARK

    def __post_init__(self) -> None:
        _nonnegative_integer(self.step, "benchmark step")
        _finite_real(self.objective, "benchmark objective")
        _finite_real(self.elapsed_seconds, "benchmark elapsed_seconds", nonnegative=True)
        for name in (
            "operation_count",
            "model_evaluations",
            "bytes_read",
            "bytes_written",
            "collective_bytes",
            "peak_memory_bytes",
            "maximum_staleness_steps",
            "accepted_updates",
            "rejected_updates",
        ):
            _nonnegative_integer(getattr(self, name), f"benchmark {name}")
        expected = (
            (self.work_counter_semantics, CounterSemantics.CUMULATIVE, "work counters"),
            (self.peak_memory_semantics, CounterSemantics.HIGH_WATER_MARK, "peak memory"),
            (self.staleness_semantics, CounterSemantics.HIGH_WATER_MARK, "maximum staleness"),
        )
        for actual, required, label in expected:
            if actual is not required:
                raise ValueError(f"benchmark {label} semantics must be {required.value}.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible benchmark observation."""

        return {
            "step": self.step,
            "objective": self.objective,
            "elapsed_seconds": self.elapsed_seconds,
            "operation_count": self.operation_count,
            "model_evaluations": self.model_evaluations,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "collective_bytes": self.collective_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "maximum_staleness_steps": self.maximum_staleness_steps,
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "counter_semantics": {
                "elapsed_seconds": CounterSemantics.CUMULATIVE.value,
                "operation_count": self.work_counter_semantics.value,
                "model_evaluations": self.work_counter_semantics.value,
                "bytes_read": self.work_counter_semantics.value,
                "bytes_written": self.work_counter_semantics.value,
                "collective_bytes": self.work_counter_semantics.value,
                "accepted_updates": self.work_counter_semantics.value,
                "rejected_updates": self.work_counter_semantics.value,
                "peak_memory_bytes": self.peak_memory_semantics.value,
                "maximum_staleness_steps": self.staleness_semantics.value,
            },
        }


class TimeToTargetTrace:
    """Monotone cumulative work trace for one strategy and one target.

    Not a dataclass any more (MXR-080-1905): ``points`` was a public list, so the chronology and
    cumulative-counter contract that :meth:`record` enforces could be walked straight past --
    ``trace.points.append(BenchmarkPoint(0, 0.95, 0.0))`` after a step-1 point produced a trace whose
    steps ran 1, 0 and whose elapsed times ran 1.0, 0.0, and ``achieved`` then reported True from
    that out-of-order point. :attr:`points` is a read-only view now and :meth:`record` is the only
    way in. The constructor signature is unchanged.
    """

    def __init__(
        self,
        benchmark_id: str,
        strategy: str,
        target: ObjectiveTarget,
        points: Sequence[BenchmarkPoint] = (),
    ) -> None:
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise ValueError("benchmark_id and strategy must be non-empty.")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("benchmark_id and strategy must be non-empty.")
        if not isinstance(target, ObjectiveTarget):
            raise TypeError("time-to-target trace requires an ObjectiveTarget.")
        if isinstance(points, (str, bytes)) or not isinstance(points, (list, tuple)):
            raise TypeError("time-to-target points must be a list of BenchmarkPoint values.")
        self.benchmark_id = benchmark_id
        self.strategy = strategy
        self.target = target
        self._points: list[BenchmarkPoint] = []
        for point in points:
            self.record(point)

    def __repr__(self) -> str:
        return "TimeToTargetTrace(benchmark_id=%r, strategy=%r, target=%r, points=%d)" % (
            self.benchmark_id,
            self.strategy,
            self.target,
            len(self._points),
        )

    @property
    def points(self) -> tuple[BenchmarkPoint, ...]:
        """The recorded observations, oldest first, as a detached tuple."""

        return tuple(self._points)

    def record(self, point: BenchmarkPoint) -> None:
        """Append a cumulative point after checking chronology and counters."""

        if not isinstance(point, BenchmarkPoint):
            raise TypeError("time-to-target traces accept BenchmarkPoint values.")
        if self._points:
            previous = self._points[-1]
            if point.step <= previous.step or point.elapsed_seconds < previous.elapsed_seconds:
                raise ValueError("benchmark points must advance in step and elapsed time.")
            cumulative = (
                "operation_count",
                "model_evaluations",
                "bytes_read",
                "bytes_written",
                "collective_bytes",
                "peak_memory_bytes",
                "maximum_staleness_steps",
                "accepted_updates",
                "rejected_updates",
            )
            if any(getattr(point, name) < getattr(previous, name) for name in cumulative):
                raise ValueError("cumulative benchmark counters cannot decrease.")
        self._points.append(point)

    @property
    def first_target_point(self) -> BenchmarkPoint | None:
        """First observed point that reaches the declared quality target."""

        return next((point for point in self._points if self.target.reached(point.objective)), None)

    @property
    def achieved(self) -> bool:
        """Whether this trace reaches its declared target."""

        return self.first_target_point is not None

    def as_dict(self) -> dict[str, Any]:
        """Return the complete trace without collapsing time and operations."""

        first = self.first_target_point
        return {
            "benchmark_id": self.benchmark_id,
            "strategy": self.strategy,
            "target": self.target.as_dict(),
            "achieved": self.achieved,
            "time_to_target_seconds": first.elapsed_seconds if first is not None else None,
            "operations_to_target": first.operation_count if first is not None else None,
            "model_evaluations_to_target": first.model_evaluations if first is not None else None,
            "points": [point.as_dict() for point in self.points],
        }


class FailureKind(StrEnum):
    """Failure families used by Stage-0 negative controls."""

    NUMERICAL = "numerical"
    OBJECTIVE_REGRESSION = "objective_regression"
    REPLAY_MISMATCH = "replay_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    QUALITY_REGRESSION = "quality_regression"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"


@dataclass(frozen=True)
class FailureReceipt:
    """Outcome of one expected or naturally occurring failure case."""

    benchmark_id: str
    case_id: str
    kind: FailureKind
    oracle: str
    expected_failure: bool
    detected: bool
    observed: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Detached and sealed: these were caller-owned containers stored by reference on a
        # frozen dataclass, so a mutation after construction rewrote evidence that had already
        # been recorded (MXR-080-1876).
        object.__setattr__(self, "details", freeze_receipt_container(self.details))
        if not self.benchmark_id or not self.case_id or not self.oracle or not self.observed:
            raise ValueError("failure receipt identifiers, oracle, and observation must be non-empty.")

    @property
    def oracle_passed(self) -> bool:
        """Whether the detector behaved as expected for this case."""

        return self.detected if self.expected_failure else not self.detected

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible negative-control receipt."""

        return {
            "benchmark_id": self.benchmark_id,
            "case_id": self.case_id,
            "kind": self.kind.value,
            "oracle": self.oracle,
            "expected_failure": self.expected_failure,
            "detected": self.detected,
            "oracle_passed": self.oracle_passed,
            "observed": self.observed,
            "details": dict(self.details),
        }


class FailureLedger:
    """Append-only in-memory ledger for benchmark failure oracles.

    Append-only in fact, not only in the docstring (MXR-080-1905). ``receipts`` was a public list on
    a dataclass, so a ledger holding one failed oracle answered ``all_oracles_passed: False`` until
    ``ledger.receipts.clear()``, after which it answered ``True`` -- the negative control it was
    recording had been erased through the very attribute that reports it, and the duplicate-case
    check in :meth:`record` was bypassable the same way. :attr:`receipts` is a detached tuple now.
    """

    def __init__(self, receipts: Sequence[FailureReceipt] = ()) -> None:
        self._receipts: list[FailureReceipt] = []
        for receipt in receipts:
            self.record(receipt)

    def __repr__(self) -> str:
        return "FailureLedger(receipts=%d)" % len(self._receipts)

    @property
    def receipts(self) -> tuple[FailureReceipt, ...]:
        """Every recorded failure-oracle receipt, in the order recorded."""

        return tuple(self._receipts)

    def record(self, receipt: FailureReceipt) -> None:
        """Record one uniquely identified case."""

        if not isinstance(receipt, FailureReceipt):
            raise TypeError("failure ledgers accept FailureReceipt values.")
        key = (receipt.benchmark_id, receipt.case_id)
        if any((row.benchmark_id, row.case_id) == key for row in self._receipts):
            raise ValueError("failure case %s/%s is already recorded." % key)
        self._receipts.append(receipt)

    @property
    def failed_oracles(self) -> tuple[FailureReceipt, ...]:
        """Expected failures missed or clean controls falsely flagged."""

        return tuple(receipt for receipt in self._receipts if not receipt.oracle_passed)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible ledger."""

        return {
            "all_oracles_passed": not self.failed_oracles,
            "receipts": [receipt.as_dict() for receipt in self._receipts],
        }


__all__ = [
    "BenchmarkPoint",
    "FailureKind",
    "FailureLedger",
    "FailureReceipt",
    "ObjectiveTarget",
    "TargetDirection",
    "TimeToTargetTrace",
]
