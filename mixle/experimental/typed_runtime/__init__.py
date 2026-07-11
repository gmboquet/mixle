"""Experimental statistically typed optimization and context-runtime foundation.

The package currently implements the side-effect-free semantic compiler and
Stage-0 measurement vocabulary, and deterministic typed-node scheduling.
Execution, proposal/commit, distributed placement, and context actions are added
only after their dependency gates pass.
"""

from mixle.experimental.typed_runtime.benchmark import (
    BenchmarkPoint,
    FailureKind,
    FailureLedger,
    FailureReceipt,
    ObjectiveTarget,
    TargetDirection,
    TimeToTargetTrace,
)
from mixle.experimental.typed_runtime.cache import CachedArtifact, InvalidationReceipt, VersionedArtifactCache
from mixle.experimental.typed_runtime.compiler import ContractRegistry, compile_update_graph, infer_update_contract
from mixle.experimental.typed_runtime.contracts import (
    ArtifactKind,
    ConsistencyRequirement,
    CostEstimate,
    CurvatureKind,
    MergeLaw,
    ObjectiveKind,
    StateSemantics,
    UpdateContract,
    UpdateKind,
)
from mixle.experimental.typed_runtime.graph import DependencyEdge, UpdateGraph, UpdateGraphError, UpdateNode
from mixle.experimental.typed_runtime.local import (
    GainProvider,
    TypedMixtureRoundReceipt,
    TypedMixtureRun,
    run_typed_mixture_em,
)
from mixle.experimental.typed_runtime.measurement import (
    EffectiveContextMeasurement,
    MeasurementCatalog,
    WorkMeasurement,
)
from mixle.experimental.typed_runtime.scheduler import (
    GainEvidence,
    GainPerCostScheduler,
    NodeScheduleState,
    SchedulerConfig,
    ScheduleReceipt,
)
from mixle.experimental.typed_runtime.validation import (
    IssueSeverity,
    UpdateGraphValidationError,
    ValidationIssue,
    validate_update_graph,
)

__all__ = [
    "ArtifactKind",
    "BenchmarkPoint",
    "CachedArtifact",
    "ConsistencyRequirement",
    "ContractRegistry",
    "CostEstimate",
    "CurvatureKind",
    "DependencyEdge",
    "EffectiveContextMeasurement",
    "FailureKind",
    "FailureLedger",
    "FailureReceipt",
    "GainEvidence",
    "GainProvider",
    "GainPerCostScheduler",
    "IssueSeverity",
    "InvalidationReceipt",
    "MeasurementCatalog",
    "MergeLaw",
    "ObjectiveKind",
    "ObjectiveTarget",
    "NodeScheduleState",
    "ScheduleReceipt",
    "SchedulerConfig",
    "StateSemantics",
    "TargetDirection",
    "TimeToTargetTrace",
    "TypedMixtureRoundReceipt",
    "TypedMixtureRun",
    "UpdateContract",
    "UpdateGraph",
    "UpdateGraphError",
    "UpdateGraphValidationError",
    "UpdateKind",
    "UpdateNode",
    "ValidationIssue",
    "VersionedArtifactCache",
    "WorkMeasurement",
    "compile_update_graph",
    "infer_update_contract",
    "validate_update_graph",
    "run_typed_mixture_em",
]
