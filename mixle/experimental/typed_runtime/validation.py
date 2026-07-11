"""Static validation for compiled statistically typed update graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mixle.experimental.typed_runtime.contracts import (
    MergeLaw,
    ObjectiveKind,
    StateSemantics,
    UpdateKind,
)
from mixle.experimental.typed_runtime.graph import UpdateGraph


class IssueSeverity(StrEnum):
    """Validation issue severity."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable contract or planning defect."""

    severity: IssueSeverity
    code: str
    node_id: str
    message: str


class UpdateGraphValidationError(ValueError):
    """Raised when strict validation finds one or more errors."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        super().__init__("\n".join("%s %s: %s" % (issue.node_id, issue.code, issue.message) for issue in issues))


def validate_update_graph(graph: UpdateGraph, *, strict: bool = False) -> tuple[ValidationIssue, ...]:
    """Return semantic issues and optionally raise when any are errors."""

    issues: list[ValidationIssue] = []
    for node in graph.nodes:
        contract = node.contract
        if contract.objective_kind is ObjectiveKind.UNKNOWN:
            issues.append(
                ValidationIssue(IssueSeverity.ERROR, "unknown-objective", node.node_id, "declare an objective kind")
            )
        if contract.update_kind is UpdateKind.UNKNOWN:
            issues.append(
                ValidationIssue(IssueSeverity.ERROR, "unknown-update", node.node_id, "declare an update kind")
            )
        if contract.is_mutable and StateSemantics.STOCHASTIC_RNG in contract.state_semantics:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    "transaction-required",
                    node.node_id,
                    "rollback/replay must include mutable parameters and RNG state",
                )
            )
        if contract.decomposition_axes and contract.merge_law in (MergeLaw.NON_MERGEABLE, MergeLaw.REPLICATED):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    "unmergeable-shard",
                    node.node_id,
                    "declared decomposition has no cross-shard merge law",
                )
            )
        if not contract.outer_objective_compatible:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    "surrogate-objective",
                    node.node_id,
                    "outer-objective convergence and best-state selection are not valid for this update",
                )
            )
        if node.cost.measured and node.cost.wall_time_seconds is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    "invalid-measurement",
                    node.node_id,
                    "measured cost is missing wall time",
                )
            )

    result = tuple(issues)
    errors = tuple(issue for issue in result if issue.severity is IssueSeverity.ERROR)
    if strict and errors:
        raise UpdateGraphValidationError(errors)
    return result


__all__ = [
    "IssueSeverity",
    "UpdateGraphValidationError",
    "ValidationIssue",
    "validate_update_graph",
]
