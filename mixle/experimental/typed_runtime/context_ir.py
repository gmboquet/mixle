"""Revisitable effective-context graph, provenance, and context action IR."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from mixle.experimental.typed_runtime.proposal import payload_fingerprint


class ContextNodeKind(StrEnum):
    """Semantic role of one bounded context artifact."""

    SOURCE_CHUNK = "source_chunk"
    CLAIM = "claim"
    GENERATED_HYPOTHESIS = "generated_hypothesis"
    GENERATED_QUERY = "generated_query"
    SUMMARY = "summary"
    TOOL_RESULT = "tool_result"
    ENTITY = "entity"
    MEMORY = "memory"


class EvidenceStatus(StrEnum):
    """Verification status kept separate from model confidence."""

    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class ContextEdgeKind(StrEnum):
    """Typed relation between context artifacts."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    GENERATED_FROM = "generated_from"
    REFERS_TO = "refers_to"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    EXPANDS = "expands"


@dataclass(frozen=True)
class Provenance:
    """Stable source identity and exact locator for one context artifact."""

    source_id: str
    source_version: str
    locator: str
    content_hash: str
    provider: str | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        if any(not value for value in (self.source_id, self.source_version, self.locator, self.content_hash)):
            raise ValueError("provenance source, version, locator, and content hash must be non-empty.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible provenance record."""

        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "locator": self.locator,
            "content_hash": self.content_hash,
            "provider": self.provider,
            "uri": self.uri,
        }


@dataclass(frozen=True)
class ContextNode:
    """One source, claim, generated artifact, summary, result, or memory unit."""

    node_id: str
    kind: ContextNodeKind
    text: str
    token_count: int
    provenance: tuple[Provenance, ...] = ()
    evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    confidence: float | None = None
    generated: bool = False
    source_horizon_position: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id or not self.text:
            raise ValueError("context node id and text must be non-empty.")
        if self.token_count < 0:
            raise ValueError("context node token_count must be non-negative.")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("context node confidence must be in [0, 1].")
        if self.source_horizon_position is not None and self.source_horizon_position < 0:
            raise ValueError("source_horizon_position must be non-negative.")
        generated_kinds = {
            ContextNodeKind.GENERATED_HYPOTHESIS,
            ContextNodeKind.GENERATED_QUERY,
            ContextNodeKind.SUMMARY,
        }
        if self.kind in generated_kinds and not self.generated:
            raise ValueError("generated hypotheses, queries, and summaries must set generated=True.")
        if self.evidence_status is EvidenceStatus.SUPPORTED and not self.provenance:
            raise ValueError("supported context nodes require source provenance.")

    @property
    def content_hash(self) -> str:
        """Deterministic node-content fingerprint."""

        return payload_fingerprint(
            (
                self.kind.value,
                self.text,
                self.token_count,
                tuple(item.as_dict() for item in self.provenance),
                self.evidence_status.value,
                self.generated,
                self.source_horizon_position,
                self.metadata,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible context node."""

        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "text": self.text,
            "token_count": self.token_count,
            "provenance": [item.as_dict() for item in self.provenance],
            "evidence_status": self.evidence_status.value,
            "confidence": self.confidence,
            "generated": self.generated,
            "source_horizon_position": self.source_horizon_position,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ContextEdge:
    """One versionable typed relation in the context graph."""

    edge_id: str
    source_node: str
    target_node: str
    kind: ContextEdgeKind
    confidence: float | None = None
    provenance: tuple[Provenance, ...] = ()

    def __post_init__(self) -> None:
        if any(not value for value in (self.edge_id, self.source_node, self.target_node)):
            raise ValueError("context edge identity must be non-empty.")
        if self.source_node == self.target_node:
            raise ValueError("context edges cannot be self-loops.")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("context edge confidence must be in [0, 1].")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible context edge."""

        return {
            "edge_id": self.edge_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "kind": self.kind.value,
            "confidence": self.confidence,
            "provenance": [item.as_dict() for item in self.provenance],
        }


class ContextGraph:
    """Mutable, versioned evidence graph with deterministic snapshot/restore."""

    def __init__(self) -> None:
        self.nodes: dict[str, ContextNode] = {}
        self.edges: dict[str, ContextEdge] = {}
        self.version = 0

    def add_node(self, node: ContextNode) -> None:
        """Add an immutable node or accept an idempotent duplicate."""

        existing = self.nodes.get(node.node_id)
        if existing is not None:
            if existing.content_hash != node.content_hash:
                raise ValueError("context node id collision with different content: %s" % node.node_id)
            return
        self.nodes[node.node_id] = node
        self.version += 1

    def add_edge(self, edge: ContextEdge) -> None:
        """Add an immutable edge after endpoint validation."""

        if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
            raise KeyError("context edge endpoints must exist before the edge is added.")
        existing = self.edges.get(edge.edge_id)
        if existing is not None:
            if existing != edge:
                raise ValueError("context edge id collision with different content: %s" % edge.edge_id)
            return
        self.edges[edge.edge_id] = edge
        self.version += 1

    def verify(
        self,
        node_id: str,
        status: EvidenceStatus,
        *,
        provenance: tuple[Provenance, ...] = (),
        confidence: float | None = None,
    ) -> ContextNode:
        """Replace verification fields without erasing generated provenance."""

        node = self.nodes[node_id]
        combined = tuple(dict.fromkeys(node.provenance + provenance))
        updated = replace(node, evidence_status=status, provenance=combined, confidence=confidence)
        self.nodes[node_id] = updated
        self.version += 1
        return updated

    def neighbors(self, node_id: str, *, kinds: tuple[ContextEdgeKind, ...] | None = None) -> tuple[str, ...]:
        """Return both incoming and outgoing neighbors for revisitation."""

        if node_id not in self.nodes:
            raise KeyError(node_id)
        selected = set(kinds) if kinds is not None else None
        result = set()
        for edge in self.edges.values():
            if selected is not None and edge.kind not in selected:
                continue
            if edge.source_node == node_id:
                result.add(edge.target_node)
            elif edge.target_node == node_id:
                result.add(edge.source_node)
        return tuple(sorted(result))

    def unresolved_nodes(self) -> tuple[ContextNode, ...]:
        """Generated hypotheses/claims that still need verification."""

        return tuple(
            node
            for node in self.nodes.values()
            if node.kind in (ContextNodeKind.CLAIM, ContextNodeKind.GENERATED_HYPOTHESIS)
            and node.evidence_status is EvidenceStatus.UNVERIFIED
        )

    def snapshot(self) -> tuple[int, dict[str, ContextNode], dict[str, ContextEdge]]:
        """Capture graph state for transactional context actions."""

        return self.version, copy.deepcopy(self.nodes), copy.deepcopy(self.edges)

    def restore(self, snapshot: tuple[int, dict[str, ContextNode], dict[str, ContextEdge]]) -> None:
        """Restore an action snapshot after failure."""

        self.version, self.nodes, self.edges = copy.deepcopy(snapshot)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph sorted by stable ids."""

        return {
            "version": self.version,
            "nodes": [self.nodes[node_id].as_dict() for node_id in sorted(self.nodes)],
            "edges": [self.edges[edge_id].as_dict() for edge_id in sorted(self.edges)],
        }


class ContextActionKind(StrEnum):
    """Operation that may create, expand, verify, or materialize context."""

    RETRIEVE = "retrieve"
    EXPAND_SOURCE = "expand_source"
    GENERATE_HYPOTHESIS = "generate_hypothesis"
    GENERATE_QUERY = "generate_query"
    SUMMARIZE = "summarize"
    VERIFY = "verify"
    TOOL_CALL = "tool_call"
    LINK = "link"
    PRUNE = "prune"
    MATERIALIZE = "materialize"
    STOP = "stop"


@dataclass(frozen=True)
class ContextAction:
    """Inspectable proposal for one context-construction operation."""

    action_id: str
    kind: ContextActionKind
    input_nodes: tuple[str, ...] = ()
    query: str | None = None
    source_scope: tuple[str, ...] = ()
    expected_information_gain: float = 0.0
    gain_standard_error: float = 0.0
    gain_sample_count: int = 0
    expected_latency_seconds: float = 0.0
    expected_tokens: int = 0
    expected_monetary_cost: float = 0.0
    expected_tool_calls: int = 0
    maximum_outputs: int = 1
    generated_output: bool = False

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("context action id must be non-empty.")
        numeric = (
            self.expected_information_gain,
            self.gain_standard_error,
            self.expected_latency_seconds,
            self.expected_monetary_cost,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("context action gain and costs must be finite.")
        if self.gain_standard_error < 0.0 or self.expected_latency_seconds < 0.0:
            raise ValueError("context action uncertainty and latency must be non-negative.")
        if (
            self.gain_sample_count < 0
            or self.expected_tokens < 0
            or self.expected_tool_calls < 0
            or self.maximum_outputs < 1
        ):
            raise ValueError("context action work counts must be non-negative and outputs positive.")
        generation = self.kind in (
            ContextActionKind.GENERATE_HYPOTHESIS,
            ContextActionKind.GENERATE_QUERY,
            ContextActionKind.SUMMARIZE,
        )
        if generation and not self.generated_output:
            raise ValueError("generative context actions must disclose generated_output=True.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible context action."""

        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "input_nodes": list(self.input_nodes),
            "query": self.query,
            "source_scope": list(self.source_scope),
            "expected_information_gain": self.expected_information_gain,
            "gain_standard_error": self.gain_standard_error,
            "gain_sample_count": self.gain_sample_count,
            "expected_latency_seconds": self.expected_latency_seconds,
            "expected_tokens": self.expected_tokens,
            "expected_monetary_cost": self.expected_monetary_cost,
            "expected_tool_calls": self.expected_tool_calls,
            "maximum_outputs": self.maximum_outputs,
            "generated_output": self.generated_output,
        }


@dataclass(frozen=True)
class ContextActionReceipt:
    """Actual graph transition and cost for one context action."""

    action: ContextAction
    graph_version_before: int
    graph_version_after: int
    output_nodes: tuple[str, ...]
    output_edges: tuple[str, ...]
    latency_seconds: float
    materialized_tokens: int
    tool_calls: int
    monetary_cost: float
    measured_information_gain: float | None
    outcome: str
    rolled_back: bool = False

    def __post_init__(self) -> None:
        if self.graph_version_before < 0 or self.graph_version_after < 0:
            raise ValueError("context action graph versions must be non-negative.")
        if self.latency_seconds < 0.0 or self.materialized_tokens < 0 or self.tool_calls < 0:
            raise ValueError("context action actual work must be non-negative.")
        if self.monetary_cost < 0.0 or not self.outcome:
            raise ValueError("context action cost/outcome must be valid.")
        if self.measured_information_gain is not None and not math.isfinite(self.measured_information_gain):
            raise ValueError("measured_information_gain must be finite when supplied.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible context action receipt."""

        return {
            "action": self.action.as_dict(),
            "graph_version_before": self.graph_version_before,
            "graph_version_after": self.graph_version_after,
            "output_nodes": list(self.output_nodes),
            "output_edges": list(self.output_edges),
            "latency_seconds": self.latency_seconds,
            "materialized_tokens": self.materialized_tokens,
            "tool_calls": self.tool_calls,
            "monetary_cost": self.monetary_cost,
            "measured_information_gain": self.measured_information_gain,
            "outcome": self.outcome,
            "rolled_back": self.rolled_back,
        }


__all__ = [
    "ContextAction",
    "ContextActionKind",
    "ContextActionReceipt",
    "ContextEdge",
    "ContextEdgeKind",
    "ContextGraph",
    "ContextNode",
    "ContextNodeKind",
    "EvidenceStatus",
    "Provenance",
]
