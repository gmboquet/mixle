"""Transactional execution adapters for context graph actions."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mixle.experimental.typed_runtime.context_ir import (
    ContextAction,
    ContextActionKind,
    ContextActionReceipt,
    ContextEdge,
    ContextGraph,
    ContextNode,
    EvidenceStatus,
    Provenance,
)


@dataclass(frozen=True)
class VerificationUpdate:
    """Proposed evidence-status transition for an existing graph node."""

    node_id: str
    status: EvidenceStatus
    provenance: tuple[Provenance, ...] = ()
    confidence: float | None = None


@dataclass(frozen=True)
class ContextGraphView:
    """Detached, read-only graph snapshot supplied to action adapters."""

    version: int
    nodes: Mapping[str, ContextNode]
    edges: Mapping[str, ContextEdge]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: tuple[int, dict[str, ContextNode], dict[str, ContextEdge]],
    ) -> ContextGraphView:
        """Create a view whose mappings and values are detached from the live graph."""

        version, nodes, edges = snapshot
        return cls(
            version,
            MappingProxyType(copy.deepcopy(nodes)),
            MappingProxyType(copy.deepcopy(edges)),
        )

    def neighbors(self, node_id: str) -> tuple[str, ...]:
        """Return both incoming and outgoing neighbors from this fixed snapshot."""

        if node_id not in self.nodes:
            raise KeyError(node_id)
        neighbors = set()
        for edge in self.edges.values():
            if edge.source_node == node_id:
                neighbors.add(edge.target_node)
            elif edge.target_node == node_id:
                neighbors.add(edge.source_node)
        return tuple(sorted(neighbors))


@dataclass(frozen=True)
class ContextActionResult:
    """Pure graph mutations and actual external work returned by an adapter."""

    nodes: tuple[ContextNode, ...] = ()
    edges: tuple[ContextEdge, ...] = ()
    verifications: tuple[VerificationUpdate, ...] = ()
    remove_nodes: tuple[str, ...] = ()
    external_latency_seconds: float = 0.0
    materialized_tokens: int = 0
    tool_calls: int = 0
    monetary_cost: float = 0.0
    measured_information_gain: float | None = None
    outcome: str = "completed"

    def __post_init__(self) -> None:
        numeric = (
            self.external_latency_seconds,
            self.monetary_cost,
            self.measured_information_gain,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("context action result measurements must be finite.")
        if self.external_latency_seconds < 0.0 or self.materialized_tokens < 0 or self.tool_calls < 0:
            raise ValueError("context action result work must be non-negative.")
        if self.monetary_cost < 0.0 or not self.outcome:
            raise ValueError("context action result cost/outcome must be valid.")


ContextActionAdapter = Callable[[ContextAction, ContextGraphView], ContextActionResult]


class ContextActionExecutor:
    """Apply adapter results atomically to a versioned context graph."""

    def __init__(
        self,
        graph: ContextGraph,
        adapters: Mapping[ContextActionKind, ContextActionAdapter] | None = None,
    ) -> None:
        self.graph = graph
        self.adapters = dict(adapters or {})
        self.receipts: list[ContextActionReceipt] = []
        self._receipts_by_action_id: dict[str, ContextActionReceipt] = {}
        self._lock = threading.RLock()

    def register(self, kind: ContextActionKind, adapter: ContextActionAdapter) -> None:
        """Register or deliberately replace one action adapter."""

        if not callable(adapter):
            raise TypeError("context action adapter must be callable.")
        self.adapters[kind] = adapter

    @staticmethod
    def _apply_result(graph: ContextGraph, action: ContextAction, result: ContextActionResult) -> None:
        if len(result.nodes) > action.maximum_outputs:
            raise ValueError("context action produced more nodes than maximum_outputs.")
        if action.generated_output and any(not node.generated for node in result.nodes):
            raise ValueError("generative context action returned an undisclosed non-generated node.")
        if not action.generated_output and action.kind in (
            ContextActionKind.RETRIEVE,
            ContextActionKind.EXPAND_SOURCE,
            ContextActionKind.TOOL_CALL,
        ):
            if any(node.generated for node in result.nodes):
                raise ValueError("retrieval/tool action returned generated content without disclosure.")
        for node in result.nodes:
            graph.add_node(node)
        for edge in result.edges:
            graph.add_edge(edge)
        for update in result.verifications:
            graph.verify(
                update.node_id,
                update.status,
                provenance=update.provenance,
                confidence=update.confidence,
            )
        for node_id in result.remove_nodes:
            graph.remove_node(node_id)

    def execute(self, action: ContextAction) -> ContextActionReceipt:
        """Execute one context action and return success or rollback receipt."""

        with self._lock, self.graph.transaction():
            previous = self._receipts_by_action_id.get(action.action_id)
            if previous is not None:
                if previous.action != action:
                    raise ValueError("context action id was reused for a different action.")
                return previous
            missing = sorted(set(action.input_nodes) - set(self.graph.nodes))
            if missing:
                raise KeyError("context action inputs are missing: %s" % ", ".join(missing))
            version_before = self.graph.version
            if (
                action.expected_graph_version is not None
                and action.expected_graph_version != version_before
            ):
                raise RuntimeError(
                    "context action expected graph version %d, found %d."
                    % (action.expected_graph_version, version_before)
                )
            if action.kind is ContextActionKind.STOP:
                receipt = ContextActionReceipt(
                    action,
                    version_before,
                    version_before,
                    (),
                    (),
                    0.0,
                    0,
                    0,
                    0.0,
                    None,
                    "stopped",
                )
                self._record(receipt)
                return receipt
            if action.expected_graph_version is None:
                raise ValueError("non-stop context actions require expected_graph_version.")
            if action.resource_limits is None:
                raise ValueError("non-stop context actions require finite resource_limits.")
            if action.kind not in self.adapters:
                raise KeyError("no context action adapter registered for %s" % action.kind.value)

            snapshot = self.graph.snapshot()
            view = ContextGraphView.from_snapshot(snapshot)
            started = time.perf_counter()
            result: ContextActionResult | None = None
            try:
                result = self.adapters[action.kind](action, view)
                if not isinstance(result, ContextActionResult):
                    raise TypeError("context action adapters must return ContextActionResult.")
                if self.graph.snapshot() != snapshot:
                    self.graph.restore(snapshot)
                    raise RuntimeError("context action adapter mutated the live graph.")
                staged = ContextGraph()
                staged.restore(snapshot)
                self._apply_result(staged, action, result)
                wall = time.perf_counter() - started
                actual_latency = max(wall, result.external_latency_seconds)
                limits = action.resource_limits
                violations = []
                if actual_latency > limits.latency_seconds:
                    violations.append("latency")
                if result.materialized_tokens > limits.materialized_tokens:
                    violations.append("materialized_tokens")
                if result.monetary_cost > limits.monetary_cost:
                    violations.append("monetary_cost")
                if result.tool_calls > limits.tool_calls:
                    violations.append("tool_calls")
                if violations:
                    raise RuntimeError(
                        "context action exceeded declared resource limits: %s."
                        % ", ".join(violations)
                    )
                receipt = ContextActionReceipt(
                    action,
                    version_before,
                    staged.version,
                    tuple(node.node_id for node in result.nodes),
                    tuple(edge.edge_id for edge in result.edges),
                    actual_latency,
                    result.materialized_tokens,
                    result.tool_calls,
                    result.monetary_cost,
                    result.measured_information_gain,
                    result.outcome,
                )
                self.graph.replace_if_unchanged(snapshot, staged.snapshot())
            except Exception as error:  # noqa: BLE001 - failures become durable action receipts
                if self.graph.snapshot() != snapshot:
                    self.graph.restore(snapshot)
                wall = time.perf_counter() - started
                external_latency = result.external_latency_seconds if result is not None else 0.0
                materialized_tokens = result.materialized_tokens if result is not None else 0
                tool_calls = result.tool_calls if result is not None else 0
                monetary_cost = result.monetary_cost if result is not None else 0.0
                receipt = ContextActionReceipt(
                    action,
                    version_before,
                    self.graph.version,
                    (),
                    (),
                    max(wall, external_latency),
                    materialized_tokens,
                    tool_calls,
                    monetary_cost,
                    None,
                    "error:%s:%s" % (type(error).__name__, error),
                    rolled_back=True,
                )
            self._record(receipt)
            return receipt

    def _record(self, receipt: ContextActionReceipt) -> None:
        """Record one terminal receipt before the action can be invoked again."""

        self.receipts.append(receipt)
        self._receipts_by_action_id[receipt.action.action_id] = receipt

    def as_dict(self) -> dict[str, Any]:
        """Return graph and action ledger."""

        return {
            "graph": self.graph.as_dict(),
            "receipts": [receipt.as_dict() for receipt in self.receipts],
        }


__all__ = [
    "ContextActionAdapter",
    "ContextActionExecutor",
    "ContextActionResult",
    "ContextGraphView",
    "VerificationUpdate",
]
