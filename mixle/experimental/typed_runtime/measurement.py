"""Measured work catalogs and effective-context vocabulary for runtime planning."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from mixle.experimental.typed_runtime.contracts import CostEstimate, CounterSemantics, UpdateKind


def _nonnegative_integer(value: Any, name: str) -> int:
    """Validate and return ``value`` as a builtin ``int``.

    Returning the canonical type is the point, for the same reason :func:`_finite_real` returns a
    builtin float: an ``np.int64`` satisfies ``Integral`` and was then left on the frozen record, so
    the advertised JSON-compatible ``as_dict()`` raised inside ``json.dumps`` (MXR-080-1868). The
    previous fix canonicalized the three float counters and left every integer counter alone.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite_real(value: Any, name: str, *, nonnegative: bool = False) -> float:
    """Validate and return ``value`` as a builtin ``float``.

    Returning the canonical type is the point: this used to validate in place and leave, say, an
    ``np.float32`` on the frozen record, which then made the advertised JSON-compatible ``as_dict()``
    raise inside ``json.dumps`` -- the receipt claimed serializability it did not have
    (MXR-080-1864).
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return float(value)


def _frozen_json_value(value: Any, path: str) -> Any:
    """Return an immutable, JSON-expressible copy of caller-supplied receipt metadata.

    ``extra`` is caller-owned and was stored by reference on a frozen dataclass, so a mutation after
    construction rewrote evidence that had already been recorded (MXR-080-1864). NumPy scalars are
    canonicalized here for the same reason ``_finite_real`` returns a builtin float.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"work-measurement {path} must be finite to serialize, got {value!r}.")
        return value
    if isinstance(value, Real):  # numpy scalar
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"work-measurement {path} must be finite to serialize, got {value!r}.")
        return int(value) if float(numeric).is_integer() and isinstance(value, Integral) else numeric
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"work-measurement {path} keys must be strings, got {key!r}.")
            frozen[key] = _frozen_json_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_json_value(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(
        f"work-measurement {path} holds {type(value).__name__}, which is neither immutable nor "
        "JSON-expressible; a receipt that cannot serialize is not evidence."
    )


def _plain_json(value: Any) -> Any:
    """Undo :func:`_frozen_json_value`'s containers for ``as_dict``'s JSON-compatible output."""
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True)
class WorkMeasurement:
    """One measured execution of a typed node operation."""

    node_type: str
    update_kind: UpdateKind
    backend: str
    wall_time_seconds: float
    compute_units: float = 0.0
    communication_bytes: int = 0
    peak_memory_bytes: int = 0
    observations: float = 0.0
    tokens: int = 0
    model_evaluations: int = 0
    operation_count: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    collective_bytes: int = 0
    staleness_steps: int = 0
    run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    work_counter_semantics: CounterSemantics = CounterSemantics.INCREMENTAL
    peak_memory_semantics: CounterSemantics = CounterSemantics.HIGH_WATER_MARK
    staleness_semantics: CounterSemantics = CounterSemantics.HIGH_WATER_MARK

    def __post_init__(self) -> None:
        if not isinstance(self.node_type, str) or not self.node_type:
            raise ValueError("node_type and backend must be non-empty.")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("node_type and backend must be non-empty.")
        if not isinstance(self.update_kind, UpdateKind):
            raise TypeError("work-measurement update_kind must be UpdateKind.")
        for name in ("wall_time_seconds", "compute_units", "observations"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name, nonnegative=True))
        for name in (
            "communication_bytes",
            "peak_memory_bytes",
            "tokens",
            "model_evaluations",
            "operation_count",
            "bytes_read",
            "bytes_written",
            "collective_bytes",
            "staleness_steps",
        ):
            object.__setattr__(self, name, _nonnegative_integer(getattr(self, name), name))
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id):
            raise ValueError("run_id must be a non-empty string when supplied.")
        if not isinstance(self.extra, Mapping):
            raise TypeError("work-measurement extra metadata must be a dictionary.")
        object.__setattr__(self, "extra", _frozen_json_value(dict(self.extra), "extra"))
        expected = (
            (self.work_counter_semantics, CounterSemantics.INCREMENTAL, "work counters"),
            (self.peak_memory_semantics, CounterSemantics.HIGH_WATER_MARK, "peak memory"),
            (self.staleness_semantics, CounterSemantics.HIGH_WATER_MARK, "staleness"),
        )
        for actual, required, label in expected:
            if actual is not required:
                raise ValueError(f"work-measurement {label} semantics must be {required.value}.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible receipt."""

        return {
            "node_type": self.node_type,
            "update_kind": self.update_kind.value,
            "backend": self.backend,
            "wall_time_seconds": self.wall_time_seconds,
            "compute_units": self.compute_units,
            "communication_bytes": self.communication_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "observations": self.observations,
            "tokens": self.tokens,
            "model_evaluations": self.model_evaluations,
            "operation_count": self.operation_count,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "collective_bytes": self.collective_bytes,
            "staleness_steps": self.staleness_steps,
            "run_id": self.run_id,
            "extra": _plain_json(self.extra),
            "counter_semantics": {
                "wall_time_seconds": self.work_counter_semantics.value,
                "compute_units": self.work_counter_semantics.value,
                "communication_bytes": self.work_counter_semantics.value,
                "observations": self.work_counter_semantics.value,
                "tokens": self.work_counter_semantics.value,
                "model_evaluations": self.work_counter_semantics.value,
                "operation_count": self.work_counter_semantics.value,
                "bytes_read": self.work_counter_semantics.value,
                "bytes_written": self.work_counter_semantics.value,
                "collective_bytes": self.work_counter_semantics.value,
                "peak_memory_bytes": self.peak_memory_semantics.value,
                "staleness_steps": self.staleness_semantics.value,
            },
        }


@dataclass
class MeasurementCatalog:
    """In-memory measured-cost catalog used by the experimental compiler.

    Medians make a small catalog robust to one noisy timing observation. A
    persistent/versioned catalog is deliberately deferred until the receipt
    schema has survived real backends.
    """

    _records: list[WorkMeasurement] = field(default_factory=list, repr=False)

    def __init__(self, records: Iterable[WorkMeasurement] | None = None) -> None:
        rows = list(records or ())
        if any(not isinstance(row, WorkMeasurement) for row in rows):
            raise TypeError("measurement catalog records must be WorkMeasurement values.")
        self._records = rows

    @property
    def records(self) -> tuple[WorkMeasurement, ...]:
        """The measurements, as a detached read-only view.

        This was a public ``list``, so a caller could append a fabricated measurement, delete a real
        one, or rewrite the catalog the compiler reads its cost estimates from -- all without going
        through :meth:`record`, which is the only place the type is checked (MXR-080-1878). The
        catalog is append-only evidence; ``record``/``extend`` are the way in.
        """
        return tuple(self._records)

    def record(self, measurement: WorkMeasurement) -> None:
        """Append one immutable measurement."""

        if not isinstance(measurement, WorkMeasurement):
            raise TypeError("measurement catalogs accept WorkMeasurement values.")
        self._records.append(measurement)

    def extend(self, measurements: Iterable[WorkMeasurement]) -> None:
        """Append several measurements."""

        for measurement in measurements:
            self.record(measurement)

    def matching(self, node_type: str, update_kind: UpdateKind, backend: str) -> tuple[WorkMeasurement, ...]:
        """Return exact-key measurements."""

        return tuple(
            record
            for record in self.records
            if record.node_type == node_type and record.update_kind is update_kind and record.backend == backend
        )

    def estimate(self, node_type: str, update_kind: UpdateKind, backend: str) -> CostEstimate | None:
        """Return a median measured cost, or ``None`` when no matching evidence exists."""

        rows = self.matching(node_type, update_kind, backend)
        if not rows:
            return None
        return CostEstimate(
            compute_units=float(statistics.median(row.compute_units for row in rows)),
            wall_time_seconds=float(statistics.median(row.wall_time_seconds for row in rows)),
            # ``int()`` TRUNCATES, and a median over an even number of samples is a midpoint: two
            # measurements of 0 and 1 bytes estimated 0 bytes of communication, i.e. free
            # (MXR-080-1878). This is a cost estimate a scheduler divides gain by, so rounding the
            # wrong way makes work look cheaper than anything ever measured. Round upward: an
            # estimate that overstates cost by under a byte is conservative, one that understates it
            # to zero is not.
            communication_bytes=math.ceil(statistics.median(row.communication_bytes for row in rows)),
            peak_memory_bytes=int(max(row.peak_memory_bytes for row in rows)),
            source="measurement_catalog",
            sample_count=len(rows),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible catalog."""

        return {"records": [record.as_dict() for record in self.records]}


@dataclass(frozen=True)
class EffectiveContextMeasurement:
    """Receipt separating source horizon from bounded active computation.

    ``source_horizon_tokens`` may be unknown for graph/database/tool sources.
    It is never inferred from attended tokens. Generated context is counted
    separately so it cannot be mislabeled as retrieved evidence.
    """

    source_horizon_tokens: int | None = None
    materialized_tokens: int = 0
    attended_tokens: int = 0
    evidence_nodes: int = 0
    evidence_edges: int = 0
    context_actions: int = 0
    retrieval_actions: int = 0
    generation_actions: int = 0
    verification_actions: int = 0
    tool_calls: int = 0
    latency_seconds: float = 0.0
    monetary_cost: float = 0.0
    verified_claim_fraction: float | None = None
    stopped_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "materialized_tokens",
            "attended_tokens",
            "evidence_nodes",
            "evidence_edges",
            "context_actions",
            "retrieval_actions",
            "generation_actions",
            "verification_actions",
            "tool_calls",
        ):
            # Canonicalized, not merely checked: every counter here had the same np.int64 /
            # np.float32 leak WorkMeasurement did -- validated in place, left on the frozen record,
            # and then rejected by json.dumps out of an as_dict() that advertises JSON compatibility
            # (MXR-080-1868).
            object.__setattr__(self, name, _nonnegative_integer(getattr(self, name), f"effective-context {name}"))
        if self.source_horizon_tokens is not None:
            object.__setattr__(
                self, "source_horizon_tokens", _nonnegative_integer(self.source_horizon_tokens, "source_horizon_tokens")
            )
            if self.source_horizon_tokens < self.materialized_tokens:
                raise ValueError("source horizon cannot be smaller than materialized context.")
        # The counts nest physically -- attended is a subset of materialized, which is a subset of the
        # source horizon -- and only the outer relation was enforced (MXR-080-1878). A receipt could
        # report attending to more tokens than it ever materialized, which makes
        # ``active_to_source_ratio`` and every attention-cost figure derived from it describe a run
        # that cannot have happened.
        if self.attended_tokens > self.materialized_tokens:
            raise ValueError(
                f"effective-context attended_tokens ({self.attended_tokens}) cannot exceed "
                f"materialized_tokens ({self.materialized_tokens}); attention is over materialized "
                "context, so it is a subset of it."
            )
        for name in ("latency_seconds", "monetary_cost"):
            object.__setattr__(
                self, name, _finite_real(getattr(self, name), f"effective-context {name}", nonnegative=True)
            )
        if self.verified_claim_fraction is not None:
            object.__setattr__(
                self, "verified_claim_fraction", _finite_real(self.verified_claim_fraction, "verified_claim_fraction")
            )
            if not 0.0 <= self.verified_claim_fraction <= 1.0:
                raise ValueError("verified_claim_fraction must be in [0, 1].")
        if self.stopped_reason is not None and (not isinstance(self.stopped_reason, str) or not self.stopped_reason):
            raise ValueError("stopped_reason must be a non-empty string when supplied.")
        classified_actions = self.retrieval_actions + self.generation_actions + self.verification_actions
        if classified_actions > self.context_actions:
            raise ValueError("classified context actions cannot exceed total context_actions.")

    @property
    def active_to_source_ratio(self) -> float | None:
        """Materialized/source ratio when the source horizon is known and nonzero."""

        if not self.source_horizon_tokens:
            return None
        return self.materialized_tokens / self.source_horizon_tokens

    def as_dict(self) -> dict[str, Any]:
        """Return the complete measurement vocabulary as JSON-compatible data."""

        return {
            "source_horizon_tokens": self.source_horizon_tokens,
            "materialized_tokens": self.materialized_tokens,
            "attended_tokens": self.attended_tokens,
            "evidence_nodes": self.evidence_nodes,
            "evidence_edges": self.evidence_edges,
            "context_actions": self.context_actions,
            "retrieval_actions": self.retrieval_actions,
            "generation_actions": self.generation_actions,
            "verification_actions": self.verification_actions,
            "tool_calls": self.tool_calls,
            "latency_seconds": self.latency_seconds,
            "monetary_cost": self.monetary_cost,
            "verified_claim_fraction": self.verified_claim_fraction,
            "stopped_reason": self.stopped_reason,
            "active_to_source_ratio": self.active_to_source_ratio,
            "counter_semantics": {
                "all_counts": CounterSemantics.INCREMENTAL.value,
                "latency_seconds": CounterSemantics.INCREMENTAL.value,
                "monetary_cost": CounterSemantics.INCREMENTAL.value,
            },
        }


__all__ = ["EffectiveContextMeasurement", "MeasurementCatalog", "WorkMeasurement"]
