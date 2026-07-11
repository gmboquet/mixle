"""Semantic contracts for the experimental statistically typed training runtime.

These types describe *what an update means*. They intentionally contain no
execution code and do not import torch, distributed runtimes, or concrete model
families. That makes a compiled plan inspectable before any model is sampled,
scored, mutated, placed, or communicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ObjectiveKind(StrEnum):
    """Objective whose value gives an update its statistical meaning."""

    MLE = "mle"
    MAP = "map"
    ELBO = "elbo"
    CONTRASTIVE = "contrastive"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    USER_SURROGATE = "user_surrogate"
    UNKNOWN = "unknown"


class UpdateKind(StrEnum):
    """Algorithmic family used to propose a node update."""

    EXACT_CLOSED_FORM = "exact_closed_form"
    GENERALIZED_EM = "generalized_em"
    COORDINATE = "coordinate"
    FIRST_ORDER = "first_order"
    PRECONDITIONED = "preconditioned"
    PROXIMAL = "proximal"
    MESSAGE_PASSING = "message_passing"
    MONTE_CARLO = "monte_carlo"
    DISCRETE_SEARCH = "discrete_search"
    FROZEN = "frozen"
    UNKNOWN = "unknown"


class MergeLaw(StrEnum):
    """Algebra available for combining work from separate data/model shards."""

    ADDITIVE = "additive"
    ASSOCIATIVE_MONOID = "associative_monoid"
    INVERTIBLE_GROUP = "invertible_group"
    WEIGHTED_SKETCH = "weighted_sketch"
    LOW_RANK = "low_rank"
    NON_MERGEABLE = "non_mergeable"
    REPLICATED = "replicated"


class StateSemantics(StrEnum):
    """Kinds of mutable or replay-relevant state touched by an update."""

    IMMUTABLE_RESULT = "immutable_result"
    MUTABLE_PARAMETERS = "mutable_parameters"
    MUTABLE_OPTIMIZER = "mutable_optimizer"
    STOCHASTIC_RNG = "stochastic_rng"
    EXTERNAL_STATE = "external_state"


class ConsistencyRequirement(StrEnum):
    """Strongest distributed consistency mode known to preserve semantics."""

    STRICT_SYNCHRONOUS = "strict_synchronous"
    BOUNDED_STALE = "bounded_stale"
    CORRECTED_EVENTUAL = "corrected_eventual"
    LOCAL_ONLY = "local_only"


class CurvatureKind(StrEnum):
    """Curvature information available to a geometry-aware optimizer."""

    EXACT_HESSIAN = "exact_hessian"
    FISHER = "fisher"
    DIAGONAL = "diagonal"
    KRONECKER = "kronecker"
    LOW_RANK = "low_rank"
    UNAVAILABLE = "unavailable"


class ArtifactKind(StrEnum):
    """Versionable artifacts read or written by update nodes."""

    OBSERVATIONS = "observations"
    PARAMETERS = "parameters"
    OPTIMIZER_STATE = "optimizer_state"
    SUFFICIENT_STATISTICS = "sufficient_statistics"
    POSTERIORS = "posteriors"
    MESSAGES = "messages"
    SCORES = "scores"
    ACTIVATIONS = "activations"
    KV_BLOCKS = "kv_blocks"
    CONTEXT_SUMMARIES = "context_summaries"
    GRAPH_STATE = "graph_state"
    CALIBRATION_STATE = "calibration_state"
    RNG_STATE = "rng_state"
    EXTERNAL_STATE = "external_state"


@dataclass(frozen=True)
class CostEstimate:
    """Planning cost with provenance separating measured data from a proxy."""

    compute_units: float = 0.0
    wall_time_seconds: float | None = None
    communication_bytes: int = 0
    peak_memory_bytes: int = 0
    source: str = "structural_proxy"
    sample_count: int = 0

    def __post_init__(self) -> None:
        numeric = (self.compute_units, self.communication_bytes, self.peak_memory_bytes, self.sample_count)
        if any(value < 0 for value in numeric):
            raise ValueError("cost estimates must be non-negative.")
        if self.wall_time_seconds is not None and self.wall_time_seconds < 0.0:
            raise ValueError("wall_time_seconds must be non-negative when supplied.")

    @property
    def measured(self) -> bool:
        """Whether this estimate came from runtime observations."""

        return self.sample_count > 0 and self.source == "measurement_catalog"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "compute_units": self.compute_units,
            "wall_time_seconds": self.wall_time_seconds,
            "communication_bytes": self.communication_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "source": self.source,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class UpdateContract:
    """Complete semantic declaration for one independently schedulable node."""

    objective_kind: ObjectiveKind
    update_kind: UpdateKind
    merge_law: MergeLaw
    state_semantics: frozenset[StateSemantics] = field(
        default_factory=lambda: frozenset({StateSemantics.IMMUTABLE_RESULT})
    )
    consistency: ConsistencyRequirement = ConsistencyRequirement.STRICT_SYNCHRONOUS
    curvature_kind: CurvatureKind = CurvatureKind.UNAVAILABLE
    decomposition_axes: tuple[str, ...] = ()
    reads: frozenset[ArtifactKind] = field(
        default_factory=lambda: frozenset({ArtifactKind.OBSERVATIONS, ArtifactKind.PARAMETERS})
    )
    writes: frozenset[ArtifactKind] = field(
        default_factory=lambda: frozenset({ArtifactKind.SUFFICIENT_STATISTICS, ArtifactKind.PARAMETERS})
    )
    outer_objective_compatible: bool = True
    exact: bool = True
    declared_by: str = "compiler_default"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.update_kind is UpdateKind.FROZEN and self.writes:
            raise ValueError("a frozen update contract cannot write artifacts.")
        if self.update_kind is UpdateKind.UNKNOWN and self.exact:
            raise ValueError("an unknown update kind cannot claim exactness.")
        if self.objective_kind is ObjectiveKind.UNKNOWN and self.outer_objective_compatible:
            raise ValueError("an unknown objective cannot claim outer-objective compatibility.")
        if StateSemantics.IMMUTABLE_RESULT in self.state_semantics and len(self.state_semantics) > 1:
            raise ValueError("immutable_result cannot be combined with mutable state semantics.")

    @property
    def is_mutable(self) -> bool:
        """Whether committing/rejecting this update requires state management."""

        return StateSemantics.IMMUTABLE_RESULT not in self.state_semantics

    @property
    def shard_mergeable(self) -> bool:
        """Whether independent shard payloads have a declared merge operation."""

        return self.merge_law not in (MergeLaw.NON_MERGEABLE, MergeLaw.REPLICATED)

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "objective_kind": self.objective_kind.value,
            "update_kind": self.update_kind.value,
            "merge_law": self.merge_law.value,
            "state_semantics": sorted(value.value for value in self.state_semantics),
            "consistency": self.consistency.value,
            "curvature_kind": self.curvature_kind.value,
            "decomposition_axes": list(self.decomposition_axes),
            "reads": sorted(value.value for value in self.reads),
            "writes": sorted(value.value for value in self.writes),
            "outer_objective_compatible": self.outer_objective_compatible,
            "exact": self.exact,
            "declared_by": self.declared_by,
            "notes": list(self.notes),
        }


__all__ = [
    "ArtifactKind",
    "ConsistencyRequirement",
    "CostEstimate",
    "CurvatureKind",
    "MergeLaw",
    "ObjectiveKind",
    "StateSemantics",
    "UpdateContract",
    "UpdateKind",
]
