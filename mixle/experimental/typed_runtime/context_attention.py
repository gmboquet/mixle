"""Bounded exact-near plus retrieved-far attention over context graph nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.context_ir import ContextGraph, EvidenceStatus


@dataclass(frozen=True)
class ContextAttentionConfig:
    """Hard exact-near, retrieval-count, and total active-token bounds."""

    exact_near_tokens: int = 4_096
    retrieved_nodes: int = 32
    maximum_active_tokens: int = 8_192
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.exact_near_tokens < 0 or self.retrieved_nodes < 0 or self.maximum_active_tokens <= 0:
            raise ValueError("context attention bounds must be non-negative and active tokens positive.")
        if self.exact_near_tokens > self.maximum_active_tokens:
            raise ValueError("exact_near_tokens cannot exceed maximum_active_tokens.")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("attention temperature must be finite and positive.")


@dataclass(frozen=True)
class AttentionCandidate:
    """One graph node's retrieval key, attention value, and source position."""

    node_id: str
    key: np.ndarray
    value: np.ndarray
    position: int

    def __post_init__(self) -> None:
        key = np.asarray(self.key)
        value = np.asarray(self.value)
        if not self.node_id or key.ndim != 1 or value.ndim != 1 or self.position < 0:
            raise ValueError("attention candidates require id, vector key/value, and non-negative position.")
        if not np.all(np.isfinite(key)) or not np.all(np.isfinite(value)):
            raise ValueError("attention candidate vectors must be finite.")


@dataclass(frozen=True)
class ContextAttentionReceipt:
    """Selected exact/retrieved nodes and bounded active work."""

    source_nodes: int
    source_horizon_tokens: int | None
    exact_node_ids: tuple[str, ...]
    retrieved_node_ids: tuple[str, ...]
    active_tokens: int
    excluded_unverified_generated: tuple[str, ...]
    similarities: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible attention receipt."""

        return {
            "source_nodes": self.source_nodes,
            "source_horizon_tokens": self.source_horizon_tokens,
            "exact_node_ids": list(self.exact_node_ids),
            "retrieved_node_ids": list(self.retrieved_node_ids),
            "active_tokens": self.active_tokens,
            "excluded_unverified_generated": list(self.excluded_unverified_generated),
            "similarities": dict(self.similarities),
        }


@dataclass(frozen=True)
class ContextAttentionResult:
    """Attended value and selection receipt."""

    value: np.ndarray
    receipt: ContextAttentionReceipt


def bounded_context_attention(
    query: Any,
    candidates: tuple[AttentionCandidate, ...],
    graph: ContextGraph,
    config: ContextAttentionConfig | None = None,
    *,
    source_horizon_tokens: int | None = None,
) -> ContextAttentionResult:
    """Attend exactly to recent nodes and retrieve similar old nodes under one hard token cap."""

    config = config or ContextAttentionConfig()
    query = np.asarray(query, dtype=np.float64)
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError("context attention query must be a finite vector.")
    if len({candidate.node_id for candidate in candidates}) != len(candidates):
        raise ValueError("attention candidate node ids must be unique.")
    missing = sorted({candidate.node_id for candidate in candidates} - set(graph.nodes))
    if missing:
        raise KeyError("attention candidates refer to missing graph nodes: %s" % ", ".join(missing))
    if candidates and any(np.asarray(candidate.key).shape != query.shape for candidate in candidates):
        raise ValueError("attention candidate key shape must match query.")
    value_shapes = {np.asarray(candidate.value).shape for candidate in candidates}
    if len(value_shapes) > 1:
        raise ValueError("attention candidate values must have one shared shape.")

    admissible = []
    excluded = []
    for candidate in candidates:
        node = graph.nodes[candidate.node_id]
        if node.generated and node.evidence_status is not EvidenceStatus.SUPPORTED:
            excluded.append(candidate.node_id)
        else:
            admissible.append(candidate)

    selected: list[AttentionCandidate] = []
    exact: list[str] = []
    tokens = 0
    for candidate in sorted(admissible, key=lambda row: (-row.position, row.node_id)):
        node_tokens = graph.nodes[candidate.node_id].token_count
        if tokens + node_tokens > config.exact_near_tokens:
            continue
        selected.append(candidate)
        exact.append(candidate.node_id)
        tokens += node_tokens

    selected_ids = set(exact)
    query_norm = float(np.linalg.norm(query))
    similarities: dict[str, float] = {}
    ranked_far = []
    for candidate in admissible:
        if candidate.node_id in selected_ids:
            continue
        key = np.asarray(candidate.key, dtype=np.float64)
        denominator = query_norm * float(np.linalg.norm(key))
        similarity = float(np.dot(query, key) / denominator) if denominator > 0.0 else 0.0
        similarities[candidate.node_id] = similarity
        ranked_far.append((-similarity, -candidate.position, candidate.node_id, candidate))
    retrieved = []
    for _, _, node_id, candidate in sorted(ranked_far):
        if len(retrieved) >= config.retrieved_nodes:
            break
        node_tokens = graph.nodes[node_id].token_count
        if tokens + node_tokens > config.maximum_active_tokens:
            continue
        selected.append(candidate)
        retrieved.append(node_id)
        tokens += node_tokens

    if selected:
        keys = np.stack([np.asarray(candidate.key, dtype=np.float64) for candidate in selected])
        values = np.stack([np.asarray(candidate.value, dtype=np.float64) for candidate in selected])
        logits = keys @ query / config.temperature
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()
        attended = weights @ values
    else:
        value_shape = next(iter(value_shapes), query.shape)
        attended = np.zeros(value_shape, dtype=np.float64)
    receipt = ContextAttentionReceipt(
        len(candidates),
        source_horizon_tokens,
        tuple(exact),
        tuple(retrieved),
        tokens,
        tuple(sorted(excluded)),
        similarities,
    )
    return ContextAttentionResult(attended, receipt)


__all__ = [
    "AttentionCandidate",
    "ContextAttentionConfig",
    "ContextAttentionReceipt",
    "ContextAttentionResult",
    "bounded_context_attention",
]
