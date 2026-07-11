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
from mixle.experimental.typed_runtime.clocks import (
    ClockDecision,
    ClockProgress,
    ClockTrigger,
    MultiRateUpdateClocks,
    UpdateCadence,
)
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
from mixle.experimental.typed_runtime.proposal import (
    PayloadMerger,
    ProposalBatch,
    ProposalConflict,
    ProposalPacket,
    merge_same_node_proposals,
    payload_fingerprint,
    proposal_conflicts,
)
from mixle.experimental.typed_runtime.replay import (
    ReplayEntry,
    ReplayLog,
    ReplayMode,
    ReplayReport,
    ReplayStepReceipt,
    StateProbe,
    replay_log,
)
from mixle.experimental.typed_runtime.scheduler import (
    GainEvidence,
    GainPerCostScheduler,
    NodeScheduleState,
    SchedulerConfig,
    ScheduleReceipt,
)
from mixle.experimental.typed_runtime.transaction import (
    ApplyProposalFn,
    CanaryFn,
    CanaryVerdict,
    CommitReceipt,
    CommitStatus,
    FingerprintFn,
    RestoreFn,
    RuntimeVersions,
    SnapshotFn,
    TransactionalCoordinator,
    TransactionParticipant,
)
from mixle.experimental.typed_runtime.validation import (
    IssueSeverity,
    UpdateGraphValidationError,
    ValidationIssue,
    validate_update_graph,
)

__all__ = [
    "ArtifactKind",
    "ApplyProposalFn",
    "BenchmarkPoint",
    "CachedArtifact",
    "CanaryFn",
    "CanaryVerdict",
    "ClockDecision",
    "ClockProgress",
    "ClockTrigger",
    "CommitReceipt",
    "CommitStatus",
    "ConsistencyRequirement",
    "ContractRegistry",
    "CostEstimate",
    "CurvatureKind",
    "DependencyEdge",
    "EffectiveContextMeasurement",
    "FailureKind",
    "FailureLedger",
    "FailureReceipt",
    "FingerprintFn",
    "GainEvidence",
    "GainProvider",
    "GainPerCostScheduler",
    "IssueSeverity",
    "InvalidationReceipt",
    "MeasurementCatalog",
    "MergeLaw",
    "MultiRateUpdateClocks",
    "ObjectiveKind",
    "ObjectiveTarget",
    "PayloadMerger",
    "ProposalBatch",
    "ProposalConflict",
    "ProposalPacket",
    "RestoreFn",
    "ReplayEntry",
    "ReplayLog",
    "ReplayMode",
    "ReplayReport",
    "ReplayStepReceipt",
    "RuntimeVersions",
    "NodeScheduleState",
    "ScheduleReceipt",
    "SchedulerConfig",
    "SnapshotFn",
    "StateSemantics",
    "StateProbe",
    "TargetDirection",
    "TimeToTargetTrace",
    "TypedMixtureRoundReceipt",
    "TypedMixtureRun",
    "TransactionParticipant",
    "TransactionalCoordinator",
    "UpdateContract",
    "UpdateCadence",
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
    "merge_same_node_proposals",
    "payload_fingerprint",
    "proposal_conflicts",
    "replay_log",
    "validate_update_graph",
    "run_typed_mixture_em",
]
