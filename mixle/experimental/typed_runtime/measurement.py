"""Measured work catalogs and effective-context vocabulary for runtime planning."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from mixle.experimental.typed_runtime.contracts import CostEstimate, UpdateKind


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

    def __post_init__(self) -> None:
        values = (
            self.wall_time_seconds,
            self.compute_units,
            self.communication_bytes,
            self.peak_memory_bytes,
            self.observations,
            self.tokens,
            self.model_evaluations,
            self.operation_count,
            self.bytes_read,
            self.bytes_written,
            self.collective_bytes,
            self.staleness_steps,
        )
        if any(value < 0 for value in values):
            raise ValueError("work measurements must be non-negative.")
        if not self.node_type or not self.backend:
            raise ValueError("node_type and backend must be non-empty.")

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
            "extra": dict(self.extra),
        }


@dataclass
class MeasurementCatalog:
    """In-memory measured-cost catalog used by the experimental compiler.

    Medians make a small catalog robust to one noisy timing observation. A
    persistent/versioned catalog is deliberately deferred until the receipt
    schema has survived real backends.
    """

    records: list[WorkMeasurement] = field(default_factory=list)

    def record(self, measurement: WorkMeasurement) -> None:
        """Append one immutable measurement."""

        self.records.append(measurement)

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
            communication_bytes=int(statistics.median(row.communication_bytes for row in rows)),
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
        counts = (
            self.materialized_tokens,
            self.attended_tokens,
            self.evidence_nodes,
            self.evidence_edges,
            self.context_actions,
            self.retrieval_actions,
            self.generation_actions,
            self.verification_actions,
            self.tool_calls,
        )
        if any(value < 0 for value in counts):
            raise ValueError("effective-context counts must be non-negative.")
        if self.source_horizon_tokens is not None:
            if self.source_horizon_tokens < 0:
                raise ValueError("source_horizon_tokens must be non-negative when supplied.")
            if self.source_horizon_tokens < self.materialized_tokens:
                raise ValueError("source horizon cannot be smaller than materialized context.")
        if self.latency_seconds < 0.0 or self.monetary_cost < 0.0:
            raise ValueError("effective-context costs must be non-negative.")
        if self.verified_claim_fraction is not None and not 0.0 <= self.verified_claim_fraction <= 1.0:
            raise ValueError("verified_claim_fraction must be in [0, 1].")
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
        }


__all__ = ["EffectiveContextMeasurement", "MeasurementCatalog", "WorkMeasurement"]
