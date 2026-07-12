"""Compiled update graph and dependency-aware invalidation operations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from mixle.experimental.typed_runtime.contracts import ArtifactKind, CostEstimate, UpdateContract


@dataclass(frozen=True)
class UpdateNode:
    """One independently inspectable and potentially schedulable model node."""

    node_id: str
    path: str
    model_type: str
    estimator_type: str | None
    contract: UpdateContract
    cost: CostEstimate
    parameter_count: int
    model: Any = field(default=None, repr=False, compare=False)
    estimator: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return metadata without runtime model or estimator objects."""

        return {
            "node_id": self.node_id,
            "path": self.path,
            "model_type": self.model_type,
            "estimator_type": self.estimator_type,
            "contract": self.contract.as_dict(),
            "cost": self.cost.as_dict(),
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class DependencyEdge:
    """A source-node artifact whose change invalidates a dependent target."""

    source_node: str
    target_node: str
    artifact: ArtifactKind = ArtifactKind.PARAMETERS
    reason: str = "child update invalidates parent computation"

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""

        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "artifact": self.artifact.value,
            "reason": self.reason,
        }


class UpdateGraphError(ValueError):
    """Raised when a compiled update graph violates structural invariants."""


@dataclass(frozen=True)
class UpdateGraph:
    """Directed acyclic graph of typed updates and invalidation dependencies.

    Edges point from an updated source to a dependent computation. Consequently,
    ``invalidated_by(node)`` walks forward from that node toward ancestors and
    any other consumers of shared state.
    """

    nodes: tuple[UpdateNode, ...]
    edges: tuple[DependencyEdge, ...]
    root_node: str
    compiler_version: int = 1

    def __post_init__(self) -> None:
        identifiers = [node.node_id for node in self.nodes]
        if len(identifiers) != len(set(identifiers)):
            raise UpdateGraphError("update graph node identifiers must be unique.")
        known = set(identifiers)
        if self.root_node not in known:
            raise UpdateGraphError("root_node does not identify a graph node.")
        for edge in self.edges:
            if edge.source_node not in known or edge.target_node not in known:
                raise UpdateGraphError("dependency edge refers to an unknown node.")
        self.topological_order()  # cycle check

    def node(self, node_id: str) -> UpdateNode:
        """Return a node by id."""

        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def node_at(self, path: str) -> UpdateNode:
        """Return the first node compiled at ``path``."""

        for node in self.nodes:
            if node.path == path:
                return node
        raise KeyError(path)

    def dependents(self, node_id: str) -> tuple[str, ...]:
        """Return direct computations invalidated by an update to ``node_id``."""

        return tuple(edge.target_node for edge in self.edges if edge.source_node == node_id)

    def invalidated_by(self, node_id: str, *, include_self: bool = True) -> tuple[str, ...]:
        """Return the transitive invalidation closure in dependency order."""

        self.node(node_id)  # validate id
        seen = {node_id} if include_self else set()
        frontier = [node_id]
        while frontier:
            source = frontier.pop(0)
            for target in self.dependents(source):
                if target not in seen:
                    seen.add(target)
                    frontier.append(target)
        order = self.topological_order()
        return tuple(node for node in order if node in seen)

    def topological_order(self) -> tuple[str, ...]:
        """Return child-before-dependent order, raising on a dependency cycle."""

        known = {node.node_id for node in self.nodes}
        incoming = {node_id: 0 for node_id in known}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in known}
        for edge in self.edges:
            incoming[edge.target_node] += 1
            outgoing[edge.source_node].append(edge.target_node)
        ready = sorted(node_id for node_id, degree in incoming.items() if degree == 0)
        result: list[str] = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for target in sorted(outgoing[current]):
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(result) != len(known):
            raise UpdateGraphError("update graph contains a dependency cycle.")
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph without runtime object references."""

        return {
            "compiler_version": self.compiler_version,
            "root_node": self.root_node,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
        }

    @property
    def convergence_certificate(self):
        """Tree-level guarantee: the weakest node certificate (guarantees compose by minimum)."""
        from mixle.experimental.typed_runtime.contracts import weakest_certificate

        return weakest_certificate(node.contract.convergence_certificate for node in self.nodes)

    def explain(self) -> str:
        """Render a compact human-readable execution-plan explanation."""

        lines = [
            "Statistically typed update graph",
            "root=%s nodes=%d dependencies=%d" % (self.root_node, len(self.nodes), len(self.edges)),
            "convergence certificate (weakest link): %s" % self.convergence_certificate.value,
        ]
        by_id = {node.node_id: node for node in self.nodes}
        for node_id in self.topological_order():
            node = by_id[node_id]
            contract = node.contract
            axes = ",".join(contract.decomposition_axes) or "replicated"
            state = ",".join(sorted(item.value for item in contract.state_semantics))
            lines.append(
                "- %s %s: %s/%s merge=%s state=%s axes=%s cost=%s cert=%s"
                % (
                    node.node_id,
                    node.path,
                    contract.objective_kind.value,
                    contract.update_kind.value,
                    contract.merge_law.value,
                    state,
                    axes,
                    node.cost.source,
                    contract.convergence_certificate.value,
                )
            )
        if self.edges:
            lines.append("Invalidation dependencies")
            for edge in self.edges:
                lines.append("- %s -> %s (%s)" % (edge.source_node, edge.target_node, edge.artifact.value))
        return "\n".join(lines)

    @classmethod
    def from_parts(
        cls,
        nodes: Iterable[UpdateNode],
        edges: Iterable[DependencyEdge],
        *,
        root_node: str,
    ) -> UpdateGraph:
        """Build and validate an immutable graph from iterables."""

        return cls(tuple(nodes), tuple(edges), root_node)


__all__ = ["DependencyEdge", "UpdateGraph", "UpdateGraphError", "UpdateNode"]
