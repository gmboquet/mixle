"""Transactional execution adapters for context graph actions."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
        if self.external_latency_seconds < 0.0 or self.materialized_tokens < 0 or self.tool_calls < 0:
            raise ValueError("context action result work must be non-negative.")
        if self.monetary_cost < 0.0 or not self.outcome:
            raise ValueError("context action result cost/outcome must be valid.")


ContextActionAdapter = Callable[[ContextAction, ContextGraph], ContextActionResult]


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

    def register(self, kind: ContextActionKind, adapter: ContextActionAdapter) -> None:
        """Register or deliberately replace one action adapter."""

        if not callable(adapter):
            raise TypeError("context action adapter must be callable.")
        self.adapters[kind] = adapter

    def _apply_result(self, action: ContextAction, result: ContextActionResult) -> None:
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
            self.graph.add_node(node)
        for edge in result.edges:
            self.graph.add_edge(edge)
        for update in result.verifications:
            self.graph.verify(
                update.node_id,
                update.status,
                provenance=update.provenance,
                confidence=update.confidence,
            )
        for node_id in result.remove_nodes:
            if node_id not in self.graph.nodes:
                raise KeyError(node_id)
            self.graph.edges = {
                edge_id: edge
                for edge_id, edge in self.graph.edges.items()
                if edge.source_node != node_id and edge.target_node != node_id
            }
            del self.graph.nodes[node_id]
            self.graph.version += 1

    def execute(self, action: ContextAction) -> ContextActionReceipt:
        """Execute one context action and return success or rollback receipt."""

        missing = sorted(set(action.input_nodes) - set(self.graph.nodes))
        if missing:
            raise KeyError("context action inputs are missing: %s" % ", ".join(missing))
        version_before = self.graph.version
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
            self.receipts.append(receipt)
            return receipt
        if action.kind not in self.adapters:
            raise KeyError("no context action adapter registered for %s" % action.kind.value)

        snapshot = self.graph.snapshot()
        started = time.perf_counter()
        result: ContextActionResult | None = None
        error: Exception | None = None
        try:
            result = self.adapters[action.kind](action, self.graph)
            if not isinstance(result, ContextActionResult):
                raise TypeError("context action adapters must return ContextActionResult.")
            self._apply_result(action, result)
        except Exception as caught:  # noqa: BLE001 - action failures become rollback receipts
            error = caught
            self.graph.restore(snapshot)
        wall = time.perf_counter() - started
        if error is not None:
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
                wall + external_latency,
                materialized_tokens,
                tool_calls,
                monetary_cost,
                None,
                "error:%s:%s" % (type(error).__name__, error),
                rolled_back=True,
            )
        else:
            receipt = ContextActionReceipt(
                action,
                version_before,
                self.graph.version,
                tuple(node.node_id for node in result.nodes),
                tuple(edge.edge_id for edge in result.edges),
                wall + result.external_latency_seconds,
                result.materialized_tokens,
                result.tool_calls,
                result.monetary_cost,
                result.measured_information_gain,
                result.outcome,
            )
        self.receipts.append(receipt)
        return receipt

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
    "VerificationUpdate",
]
