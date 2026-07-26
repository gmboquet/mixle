"""Provenance-preserving bounded materialization from an effective-context graph."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mixle.experimental.typed_runtime.context_ir import (
    ContextEdgeKind,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    EvidenceStatus,
)
from mixle.experimental.typed_runtime.measurement import EffectiveContextMeasurement


@dataclass(frozen=True)
class MaterializationPolicy:
    """Hard active-context bounds and evidence admission policy."""

    token_budget: int
    attended_token_budget: int | None = None
    maximum_nodes: int = 128
    minimum_relevance: float = 0.0
    require_supported_claims: bool = True
    exclude_contradicted: bool = True

    def __post_init__(self) -> None:
        if self.token_budget <= 0 or self.maximum_nodes <= 0:
            raise ValueError("materialization token and node budgets must be positive.")
        if self.attended_token_budget is not None and (
            self.attended_token_budget <= 0 or self.attended_token_budget > self.token_budget
        ):
            raise ValueError("attended_token_budget must be positive and no larger than token_budget.")
        if not math.isfinite(self.minimum_relevance):
            raise ValueError("minimum_relevance must be finite.")


@dataclass(frozen=True)
class ContextTokenizer:
    """Named deterministic tokenizer used for hard rendered-prompt bounds."""

    tokenizer_id: str
    encode: Callable[[str], Sequence[int]]

    def __post_init__(self) -> None:
        if not self.tokenizer_id or not callable(self.encode):
            raise ValueError("context tokenizer requires a non-empty identity and callable encoder.")

    def token_ids(self, text: str) -> tuple[int, ...]:
        """Encode twice to reject stateful tokenizers and validate stable integer ids."""

        first = tuple(self.encode(text))
        second = tuple(self.encode(text))
        if first != second:
            raise ValueError("context tokenizer must return deterministic token ids.")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in first):
            raise TypeError("context tokenizer must return non-negative integer token ids.")
        return first


@dataclass(frozen=True)
class MaterializedContext:
    """Bounded prompt text, selected evidence subgraph, and honest context receipt."""

    text: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    tokenizer_id: str
    token_ids: tuple[int, ...]
    token_count: int
    attended_token_indices: tuple[int, ...]
    attended_tokens: int
    excluded: dict[str, str]
    measurement: EffectiveContextMeasurement

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible materialization receipt."""

        return {
            "text": self.text,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "tokenizer_id": self.tokenizer_id,
            "token_ids": list(self.token_ids),
            "token_count": self.token_count,
            "attended_token_indices": list(self.attended_token_indices),
            "attended_tokens": self.attended_tokens,
            "excluded": dict(self.excluded),
            "measurement": self.measurement.as_dict(),
        }


def _admission_reason(node: ContextNode, policy: MaterializationPolicy, relevance: float) -> str | None:
    if relevance < policy.minimum_relevance:
        return "below-minimum-relevance"
    if policy.exclude_contradicted and node.evidence_status is EvidenceStatus.CONTRADICTED:
        return "contradicted"
    claim_like = node.kind in (ContextNodeKind.CLAIM, ContextNodeKind.GENERATED_HYPOTHESIS)
    if policy.require_supported_claims and claim_like and node.evidence_status is not EvidenceStatus.SUPPORTED:
        return "claim-not-supported"
    if node.generated and node.evidence_status is not EvidenceStatus.SUPPORTED:
        return "generated-content-not-verified"
    return None


def _support_bundle(graph: ContextGraph, node_id: str) -> set[str]:
    bundle: set[str] = set()
    pending = [node_id]
    while pending:
        current = pending.pop()
        if current in bundle:
            continue
        bundle.add(current)
        for edge in graph.edges.values():
            if edge.kind is ContextEdgeKind.SUPPORTS and edge.target_node == current:
                pending.append(edge.source_node)
            if edge.kind is ContextEdgeKind.DERIVED_FROM and edge.source_node == current:
                pending.append(edge.target_node)
    return bundle


def _render_node(node: ContextNode) -> str:
    sources = ",".join(sorted({provenance.source_id for provenance in node.provenance})) or "none"
    return "[%s|%s|status=%s|sources=%s] %s" % (
        node.node_id,
        node.kind.value,
        node.evidence_status.value,
        sources,
        node.text,
    )


def materialize_context(
    graph: ContextGraph,
    relevance: dict[str, float],
    policy: MaterializationPolicy,
    tokenizer: ContextTokenizer,
    *,
    source_horizon_tokens: int | None = None,
    required_node_ids: tuple[str, ...] = (),
    context_actions: int = 0,
    retrieval_actions: int = 0,
    generation_actions: int = 0,
    verification_actions: int = 0,
    tool_calls: int = 0,
    latency_seconds: float = 0.0,
    monetary_cost: float = 0.0,
    stopped_reason: str | None = None,
) -> MaterializedContext:
    """Greedily pack relevance-per-token evidence bundles under hard bounds."""

    unknown_relevance = sorted(set(relevance) - set(graph.nodes))
    if unknown_relevance:
        raise KeyError("relevance refers to unknown context nodes: %s" % ", ".join(unknown_relevance))
    unknown_required = sorted(set(required_node_ids) - set(graph.nodes))
    if unknown_required:
        raise KeyError("required context nodes are missing: %s" % ", ".join(unknown_required))
    excluded: dict[str, str] = {}
    admitted: dict[str, float] = {}
    for node_id, node in graph.nodes.items():
        score = float(relevance.get(node_id, 0.0))
        if not math.isfinite(score):
            raise ValueError("context relevance must be finite.")
        reason = _admission_reason(node, policy, score)
        if reason is None:
            admitted[node_id] = score
        else:
            excluded[node_id] = reason

    for node_id in required_node_ids:
        if node_id not in admitted:
            raise ValueError("required context node %s is not admissible: %s" % (node_id, excluded[node_id]))

    selected: set[str] = set()

    def ordered(node_ids: set[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                node_ids,
                key=lambda node_id: (-admitted.get(node_id, 0.0), node_id),
            )
        )

    def render(node_ids: set[str]) -> tuple[str, tuple[int, ...]]:
        text = "\n".join(_render_node(graph.nodes[node_id]) for node_id in ordered(node_ids))
        return text, tokenizer.token_ids(text)

    def add_bundle(bundle: set[str], *, required: bool) -> bool:
        combined = selected | bundle
        _, combined_tokens = render(combined)
        if len(combined) > policy.maximum_nodes or len(combined_tokens) > policy.token_budget:
            if required:
                raise ValueError("required context bundle exceeds materialization budget.")
            return False
        selected.update(combined)
        return True

    for node_id in required_node_ids:
        bundle = _support_bundle(graph, node_id)
        inadmissible = sorted(member for member in bundle if member not in admitted)
        if inadmissible:
            reasons = ", ".join("%s:%s" % (member, excluded[member]) for member in inadmissible)
            raise ValueError("required context support bundle is not admissible: %s" % reasons)
        add_bundle(bundle, required=True)

    candidates = []
    for node_id, score in admitted.items():
        if node_id in selected:
            continue
        bundle = _support_bundle(graph, node_id)
        if any(member not in admitted for member in bundle):
            excluded[node_id] = "support-bundle-not-admissible"
            continue
        _, bundle_token_ids = render(selected | bundle)
        _, selected_token_ids = render(selected)
        bundle_tokens = len(bundle_token_ids) - len(selected_token_ids)
        status_bonus = 0.1 if graph.nodes[node_id].evidence_status is EvidenceStatus.SUPPORTED else 0.0
        candidates.append((-(score + status_bonus) / max(bundle_tokens, 1), -score, node_id, bundle))
    for _, _, node_id, bundle in sorted(candidates):
        if not add_bundle(bundle, required=False):
            excluded.setdefault(node_id, "materialization-budget")

    selected_order = ordered(selected)
    selected_edges = tuple(
        sorted(
            edge.edge_id
            for edge in graph.edges.values()
            if edge.source_node in selected and edge.target_node in selected
        )
    )
    text, token_ids = render(selected)
    tokens = len(token_ids)
    attended_limit = policy.attended_token_budget or tokens
    attended_token_indices = tuple(range(min(tokens, attended_limit)))
    attended = len(attended_token_indices)
    claim_nodes = [
        graph.nodes[node_id]
        for node_id in selected
        if graph.nodes[node_id].kind in (ContextNodeKind.CLAIM, ContextNodeKind.GENERATED_HYPOTHESIS)
    ]
    verified_fraction = (
        sum(node.evidence_status is EvidenceStatus.SUPPORTED for node in claim_nodes) / len(claim_nodes)
        if claim_nodes
        else None
    )
    measurement = EffectiveContextMeasurement(
        source_horizon_tokens=source_horizon_tokens,
        materialized_tokens=tokens,
        attended_tokens=attended,
        evidence_nodes=len(selected),
        evidence_edges=len(selected_edges),
        context_actions=context_actions,
        retrieval_actions=retrieval_actions,
        generation_actions=generation_actions,
        verification_actions=verification_actions,
        tool_calls=tool_calls,
        latency_seconds=latency_seconds,
        monetary_cost=monetary_cost,
        verified_claim_fraction=verified_fraction,
        stopped_reason=stopped_reason,
    )
    return MaterializedContext(
        text,
        selected_order,
        selected_edges,
        tokenizer.tokenizer_id,
        token_ids,
        tokens,
        attended_token_indices,
        attended,
        excluded,
        measurement,
    )


__all__ = [
    "ContextTokenizer",
    "MaterializationPolicy",
    "MaterializedContext",
    "materialize_context",
]
