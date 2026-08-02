"""Revisitable effective-context graph, provenance, and context action IR."""

from __future__ import annotations

import copy
import math
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMetadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "context metadata values must be JSON scalars, mappings, lists, or tuples; found %s." % type(value).__name__
    )


def _thaw_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata_value(item) for item in value]
    return value


class FrozenMetadata(Mapping[str, Any]):
    """Deterministic, recursively immutable context metadata."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        values = values or {}
        if any(not isinstance(key, str) or not key for key in values):
            raise TypeError("context metadata keys must be non-empty strings.")
        self._items = tuple((key, _freeze_metadata_value(value)) for key, value in sorted(values.items()))
        payload_fingerprint(self._items)

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenMetadata:
        return self


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
class EvidenceTransition:
    """One immutable, graph-versioned evidentiary state."""

    graph_version: int
    status: EvidenceStatus
    provenance: tuple[Provenance, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.graph_version < 0:
            raise ValueError("evidence transition graph_version must be non-negative.")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("evidence transition confidence must be in [0, 1].")
        if self.status is EvidenceStatus.SUPPORTED and not self.provenance:
            raise ValueError("supported evidence transitions require source provenance.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible evidence transition."""

        return {
            "graph_version": self.graph_version,
            "status": self.status.value,
            "provenance": [item.as_dict() for item in self.provenance],
            "confidence": self.confidence,
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
    metadata: Mapping[str, Any] = field(default_factory=FrozenMetadata)
    evidence_history: tuple[EvidenceTransition, ...] = ()

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
        object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        history = self.evidence_history
        if not history:
            history = (
                EvidenceTransition(
                    0,
                    self.evidence_status,
                    self.provenance,
                    self.confidence,
                ),
            )
            object.__setattr__(self, "evidence_history", history)
        latest = history[-1]
        if (
            latest.status is not self.evidence_status
            or latest.provenance != self.provenance
            or latest.confidence != self.confidence
        ):
            raise ValueError("context node evidence fields must match its latest evidence history entry.")
        if any(
            later.graph_version <= earlier.graph_version for earlier, later in zip(history, history[1:], strict=False)
        ):
            raise ValueError("context node evidence history versions must increase strictly.")

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
                self.confidence,
                self.generated,
                self.source_horizon_position,
                self.metadata,
                tuple(item.as_dict() for item in self.evidence_history),
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
            "metadata": _thaw_metadata_value(self.metadata),
            "evidence_history": [item.as_dict() for item in self.evidence_history],
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
        # (source, target, kind) -> edge_id. An edge's IDENTITY is the relation it asserts, not the
        # label the caller happened to mint for it; confidence and provenance are payload. Without
        # this index the same claim could enter the graph any number of times under fresh ids, and
        # every downstream count of that relation -- support tallies, degree, traversal weight --
        # would multiply by however many times it was restated (MXR-080-0643).
        self._relations: dict[tuple[str, str, ContextEdgeKind], str] = {}
        self.version = 0
        self._lock = threading.RLock()

    def add_node(self, node: ContextNode) -> None:
        """Add an immutable node or accept an idempotent duplicate."""

        with self._lock:
            existing = self.nodes.get(node.node_id)
            if existing is not None:
                if existing.content_hash != node.content_hash:
                    raise ValueError("context node id collision with different content: %s" % node.node_id)
                return
            if node.evidence_history[-1].graph_version > self.version:
                raise ValueError("context node evidence history is ahead of the target graph.")
            self.nodes[node.node_id] = node
            self.version += 1

    def add_edge(self, edge: ContextEdge) -> None:
        """Add an immutable edge after endpoint validation."""

        with self._lock:
            if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
                raise KeyError("context edge endpoints must exist before the edge is added.")
            existing = self.edges.get(edge.edge_id)
            if existing is not None:
                if existing != edge:
                    raise ValueError("context edge id collision with different content: %s" % edge.edge_id)
                return
            relation = (edge.source_node, edge.target_node, edge.kind)
            held_by = self._relations.get(relation)
            if held_by is not None:
                raise ValueError(
                    "context edge %s restates the relation %s --%s--> %s already held by edge %s. "
                    "Merge the provenance into the existing edge instead of adding a second id: two "
                    "edges asserting one relation double-count it in every downstream tally."
                    % (edge.edge_id, edge.source_node, edge.kind.value, edge.target_node, held_by)
                )
            self.edges[edge.edge_id] = edge
            self._relations[relation] = edge.edge_id
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

        with self._lock:
            node = self.nodes[node_id]
            combined = tuple(dict.fromkeys(node.provenance + provenance))
            transition = EvidenceTransition(
                self.version + 1,
                status,
                combined,
                confidence,
            )
            updated = replace(
                node,
                evidence_status=status,
                provenance=combined,
                confidence=confidence,
                evidence_history=node.evidence_history + (transition,),
            )
            self.nodes[node_id] = updated
            self.version += 1
            return updated

    def _reindex_relations(self) -> None:
        """Rebuild the (source, target, kind) index from ``self.edges``.

        Every path that replaces ``self.edges`` wholesale -- remove_node, restore,
        replace_if_unchanged -- must call this. Refreshing the container and leaving the derived
        index behind is the same stale-cache defect as MXR-080-1192: the guard would keep rejecting
        relations that a restore had already removed, and would stop rejecting ones it reinstated.

        This also enforces the bijection it is rebuilding (MXR-080-0643). A dict comprehension kept
        only the LAST edge for a repeated relation while ``self.edges`` kept both, so a snapshot
        handed to ``restore`` or ``replace_if_unchanged`` could reinstate exactly what ``add_edge``
        refuses: two edge ids asserting one relation, double-counted in every downstream tally and
        invisible in the index. Those two are the untrusted paths -- any state built through
        ``add_edge`` already satisfies this, so nothing the graph produces is rejected here.
        """
        rebuilt: dict[tuple[str, str, ContextEdgeKind], str] = {}
        for edge_id, edge in self.edges.items():
            relation = (edge.source_node, edge.target_node, edge.kind)
            held_by = rebuilt.get(relation)
            if held_by is not None:
                raise ValueError(
                    "context graph state holds two edges for one relation %s --%s--> %s: %s and %s. "
                    "A restored or replaced graph must satisfy the same one-edge-per-relation rule "
                    "add_edge enforces; merge their provenance into a single edge."
                    % (edge.source_node, edge.kind.value, edge.target_node, held_by, edge_id)
                )
            rebuilt[relation] = edge_id
        self._relations = rebuilt

    def remove_node(self, node_id: str) -> None:
        """Remove one node and all incident edges as one versioned mutation."""

        with self._lock:
            if node_id not in self.nodes:
                raise KeyError(node_id)
            self.edges = {
                edge_id: edge
                for edge_id, edge in self.edges.items()
                if edge.source_node != node_id and edge.target_node != node_id
            }
            self._reindex_relations()
            del self.nodes[node_id]
            self.version += 1

    def neighbors(self, node_id: str, *, kinds: tuple[ContextEdgeKind, ...] | None = None) -> tuple[str, ...]:
        """Return both incoming and outgoing neighbors for revisitation."""

        with self._lock:
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

        with self._lock:
            return tuple(
                node
                for node in self.nodes.values()
                if node.kind in (ContextNodeKind.CLAIM, ContextNodeKind.GENERATED_HYPOTHESIS)
                and node.evidence_status is EvidenceStatus.UNVERIFIED
            )

    def snapshot(self) -> tuple[int, dict[str, ContextNode], dict[str, ContextEdge]]:
        """Capture graph state for transactional context actions."""

        with self._lock:
            return self.version, copy.deepcopy(self.nodes), copy.deepcopy(self.edges)

    def restore(self, snapshot: tuple[int, dict[str, ContextNode], dict[str, ContextEdge]]) -> None:
        """Restore an action snapshot after failure."""

        with self._lock:
            self.version, self.nodes, self.edges = copy.deepcopy(snapshot)
            self._reindex_relations()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Exclude concurrent graph mutations while a staged transaction is prepared."""

        with self._lock:
            yield

    def replace_if_unchanged(
        self,
        expected: tuple[int, dict[str, ContextNode], dict[str, ContextEdge]],
        replacement: tuple[int, dict[str, ContextNode], dict[str, ContextEdge]],
    ) -> None:
        """Install a staged state only when the complete base snapshot still matches."""

        with self._lock:
            current = (self.version, self.nodes, self.edges)
            if current != expected:
                raise RuntimeError("context graph changed while the action adapter was running.")
            self.version, self.nodes, self.edges = copy.deepcopy(replacement)
            self._reindex_relations()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph sorted by stable ids."""

        with self._lock:
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
class ContextActionLimits:
    """Finite worst-case resources an adapter is permitted to consume."""

    latency_seconds: float
    materialized_tokens: int
    monetary_cost: float
    tool_calls: int

    def __post_init__(self) -> None:
        numeric = (self.latency_seconds, self.monetary_cost)
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("context action latency and monetary limits must be finite and non-negative.")
        counts = (self.materialized_tokens, self.tool_calls)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("context action token and tool-call limits must be non-negative integers.")

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible action resource limits."""

        return {
            "latency_seconds": self.latency_seconds,
            "materialized_tokens": self.materialized_tokens,
            "monetary_cost": self.monetary_cost,
            "tool_calls": self.tool_calls,
        }


@dataclass(frozen=True)
class ContextAction:
    """Inspectable proposal for one context-construction operation."""

    action_id: str
    kind: ContextActionKind
    expected_graph_version: int | None = None
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
    resource_limits: ContextActionLimits | None = None

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("context action id must be non-empty.")
        if self.expected_graph_version is not None and self.expected_graph_version < 0:
            raise ValueError("expected_graph_version must be non-negative when supplied.")
        numeric = (
            self.expected_information_gain,
            self.gain_standard_error,
            self.expected_latency_seconds,
            self.expected_monetary_cost,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("context action gain and costs must be finite.")
        if self.gain_standard_error < 0.0 or self.expected_latency_seconds < 0.0 or self.expected_monetary_cost < 0.0:
            raise ValueError("context action uncertainty and expected costs must be non-negative.")
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
            "expected_graph_version": self.expected_graph_version,
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
            "resource_limits": self.resource_limits.as_dict() if self.resource_limits is not None else None,
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
        if not math.isfinite(self.latency_seconds) or not math.isfinite(self.monetary_cost):
            raise ValueError("context action actual work and cost must be finite.")
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
    "ContextActionLimits",
    "ContextActionReceipt",
    "ContextEdge",
    "ContextEdgeKind",
    "ContextGraph",
    "ContextNode",
    "ContextNodeKind",
    "EvidenceTransition",
    "EvidenceStatus",
    "FrozenMetadata",
    "Provenance",
]
