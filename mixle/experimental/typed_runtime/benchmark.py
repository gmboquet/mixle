"""Time-to-target traces and negative-control failure receipts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


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
        if not self.name:
            raise ValueError("target name must be non-empty.")
        if not math.isfinite(self.threshold) or not math.isfinite(self.tolerance):
            raise ValueError("target threshold and tolerance must be finite.")
        if self.tolerance < 0.0:
            raise ValueError("target tolerance must be non-negative.")

    def reached(self, value: float) -> bool:
        """Whether ``value`` reaches this target within tolerance."""

        if not math.isfinite(value):
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

    def __post_init__(self) -> None:
        if self.step < 0 or not math.isfinite(self.objective):
            raise ValueError("benchmark step must be non-negative and objective finite.")
        counts = (
            self.elapsed_seconds,
            self.operation_count,
            self.model_evaluations,
            self.bytes_read,
            self.bytes_written,
            self.collective_bytes,
            self.peak_memory_bytes,
            self.maximum_staleness_steps,
            self.accepted_updates,
            self.rejected_updates,
        )
        if any(value < 0 for value in counts):
            raise ValueError("benchmark work counters must be non-negative.")

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
        }


@dataclass
class TimeToTargetTrace:
    """Monotone cumulative work trace for one strategy and one target."""

    benchmark_id: str
    strategy: str
    target: ObjectiveTarget
    points: list[BenchmarkPoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.strategy:
            raise ValueError("benchmark_id and strategy must be non-empty.")

    def record(self, point: BenchmarkPoint) -> None:
        """Append a cumulative point after checking chronology and counters."""

        if self.points:
            previous = self.points[-1]
            if point.step <= previous.step or point.elapsed_seconds < previous.elapsed_seconds:
                raise ValueError("benchmark points must advance in step and elapsed time.")
            cumulative = (
                "operation_count",
                "model_evaluations",
                "bytes_read",
                "bytes_written",
                "collective_bytes",
                "accepted_updates",
                "rejected_updates",
            )
            if any(getattr(point, name) < getattr(previous, name) for name in cumulative):
                raise ValueError("cumulative benchmark counters cannot decrease.")
        self.points.append(point)

    @property
    def first_target_point(self) -> BenchmarkPoint | None:
        """First observed point that reaches the declared quality target."""

        return next((point for point in self.points if self.target.reached(point.objective)), None)

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


@dataclass
class FailureLedger:
    """Append-only in-memory ledger for benchmark failure oracles."""

    receipts: list[FailureReceipt] = field(default_factory=list)

    def record(self, receipt: FailureReceipt) -> None:
        """Record one uniquely identified case."""

        key = (receipt.benchmark_id, receipt.case_id)
        if any((row.benchmark_id, row.case_id) == key for row in self.receipts):
            raise ValueError("failure case %s/%s is already recorded." % key)
        self.receipts.append(receipt)

    @property
    def failed_oracles(self) -> tuple[FailureReceipt, ...]:
        """Expected failures missed or clean controls falsely flagged."""

        return tuple(receipt for receipt in self.receipts if not receipt.oracle_passed)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible ledger."""

        return {
            "all_oracles_passed": not self.failed_oracles,
            "receipts": [receipt.as_dict() for receipt in self.receipts],
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
