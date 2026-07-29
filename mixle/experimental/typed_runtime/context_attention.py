"""Node-, token-, and compute-bounded attention over materialized context tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.context_ir import ContextGraph, EvidenceStatus


def _nonnegative_integer(value: Any, name: str, *, positive: bool = False) -> None:
    lower = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer.")


@dataclass(frozen=True)
class ContextAttentionConfig:
    """Hard bounds and the tokenizer identity used by materialized token sequences."""

    tokenizer_id: str
    exact_near_tokens: int = 4_096
    exact_near_nodes: int = 64
    retrieved_nodes: int = 32
    maximum_active_tokens: int = 8_192
    maximum_active_nodes: int = 96
    maximum_scored_nodes: int = 4_096
    maximum_attention_elements: int = 1_000_000
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_id, str) or not self.tokenizer_id:
            raise ValueError("context attention requires a non-empty tokenizer_id.")
        for name in ("exact_near_tokens", "exact_near_nodes", "retrieved_nodes"):
            _nonnegative_integer(getattr(self, name), name)
        for name in ("maximum_active_tokens", "maximum_active_nodes", "maximum_scored_nodes"):
            _nonnegative_integer(getattr(self, name), name, positive=True)
        _nonnegative_integer(self.maximum_attention_elements, "maximum_attention_elements", positive=True)
        if self.exact_near_tokens > self.maximum_active_tokens:
            raise ValueError("exact_near_tokens cannot exceed maximum_active_tokens.")
        if self.exact_near_nodes > self.maximum_active_nodes:
            raise ValueError("exact_near_nodes cannot exceed maximum_active_nodes.")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0.0
        ):
            raise ValueError("attention temperature must be finite and positive.")


@dataclass(frozen=True)
class AttentionCandidate:
    """One graph node with retrieval vectors and content-bound materialized token IDs."""

    node_id: str
    key: np.ndarray
    value: np.ndarray
    position: int
    token_ids: tuple[int, ...]
    tokenizer_id: str
    materialized_content_hash: str

    def __post_init__(self) -> None:
        key = np.asarray(self.key)
        value = np.asarray(self.value)
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("attention candidates require a non-empty node_id.")
        if key.ndim != 1 or not key.size or value.ndim != 1 or not value.size:
            raise ValueError("attention candidates require non-empty vector key/value.")
        _nonnegative_integer(self.position, "attention candidate position")
        real_numeric = (np.integer, np.floating)
        if not any(np.issubdtype(key.dtype, kind) for kind in real_numeric) or not any(
            np.issubdtype(value.dtype, kind) for kind in real_numeric
        ):
            raise TypeError("attention candidate key/value arrays must be real numeric vectors.")
        if not np.all(np.isfinite(key)) or not np.all(np.isfinite(value)):
            raise ValueError("attention candidate vectors must be finite.")
        if (
            not isinstance(self.token_ids, tuple)
            or not self.token_ids
            or any(isinstance(token, bool) or not isinstance(token, Integral) or token < 0 for token in self.token_ids)
        ):
            raise ValueError("attention candidates require a non-empty sequence of non-negative token IDs.")
        if not isinstance(self.tokenizer_id, str) or not self.tokenizer_id:
            raise ValueError("attention candidates require a non-empty tokenizer_id.")
        if not isinstance(self.materialized_content_hash, str) or not self.materialized_content_hash:
            raise ValueError("attention candidates require a materialized content hash.")
        key = key.copy()
        value = value.copy()
        key.setflags(write=False)
        value.setflags(write=False)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "token_ids", tuple(int(token) for token in self.token_ids))

    @property
    def measured_token_count(self) -> int:
        """Token work derived from the materialized token-ID sequence."""

        return len(self.token_ids)


@dataclass(frozen=True)
class ContextAttentionReceipt:
    """Selected nodes plus measured, independently bounded token and vector work."""

    tokenizer_id: str
    source_nodes: int
    source_horizon_tokens: int | None
    maximum_scored_nodes: int
    maximum_attention_elements: int
    maximum_active_nodes: int
    maximum_active_tokens: int
    scored_nodes: int
    scoring_elements: int
    exact_node_ids: tuple[str, ...]
    retrieved_node_ids: tuple[str, ...]
    active_nodes: int
    active_tokens: int
    active_token_counts: tuple[tuple[str, int], ...]
    excluded_unverified_generated: tuple[str, ...]
    similarities: dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_id, str) or not self.tokenizer_id:
            raise ValueError("context attention receipts require tokenizer identity.")
        for name in (
            "source_nodes",
            "scored_nodes",
            "scoring_elements",
            "active_nodes",
            "active_tokens",
        ):
            _nonnegative_integer(getattr(self, name), f"attention receipt {name}")
        for name in (
            "maximum_scored_nodes",
            "maximum_attention_elements",
            "maximum_active_nodes",
            "maximum_active_tokens",
        ):
            _nonnegative_integer(getattr(self, name), f"attention receipt {name}", positive=True)
        if (
            self.source_nodes > self.maximum_scored_nodes
            or self.scoring_elements > self.maximum_attention_elements
            or self.active_nodes > self.maximum_active_nodes
            or self.active_tokens > self.maximum_active_tokens
        ):
            raise ValueError("attention receipt observed work exceeds a declared hard bound.")
        if self.source_horizon_tokens is not None:
            _nonnegative_integer(self.source_horizon_tokens, "source_horizon_tokens")
            if self.source_horizon_tokens < self.active_tokens:
                raise ValueError("source horizon cannot be smaller than active measured tokens.")
        selected = self.exact_node_ids + self.retrieved_node_ids
        if self.active_nodes != len(selected) or len(set(selected)) != len(selected):
            raise ValueError("attention receipt active-node count must match unique selected nodes.")
        if (
            not isinstance(self.active_token_counts, tuple)
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
                or isinstance(row[1], bool)
                or not isinstance(row[1], Integral)
                or row[1] < 1
                for row in self.active_token_counts
            )
            or len({node_id for node_id, _ in self.active_token_counts}) != len(self.active_token_counts)
        ):
            raise ValueError("attention receipt token counts must be unique positive node measurements.")
        if set(dict(self.active_token_counts)) != set(selected):
            raise ValueError("attention receipt token counts must identify every selected node.")
        if sum(count for _, count in self.active_token_counts) != self.active_tokens:
            raise ValueError("attention receipt active token count must equal its per-node measurements.")
        if any(not math.isfinite(value) for value in self.similarities.values()):
            raise ValueError("attention receipt similarities must be finite.")
        if len(self.similarities) != self.scored_nodes or set(self.similarities) & set(self.exact_node_ids):
            raise ValueError("attention receipt similarities must identify exactly the scored far nodes.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible attention receipt."""

        return {
            "tokenizer_id": self.tokenizer_id,
            "source_nodes": self.source_nodes,
            "source_horizon_tokens": self.source_horizon_tokens,
            "maximum_scored_nodes": self.maximum_scored_nodes,
            "maximum_attention_elements": self.maximum_attention_elements,
            "maximum_active_nodes": self.maximum_active_nodes,
            "maximum_active_tokens": self.maximum_active_tokens,
            "scored_nodes": self.scored_nodes,
            "scoring_elements": self.scoring_elements,
            "exact_node_ids": list(self.exact_node_ids),
            "retrieved_node_ids": list(self.retrieved_node_ids),
            "active_nodes": self.active_nodes,
            "active_tokens": self.active_tokens,
            "active_token_counts": dict(self.active_token_counts),
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
    config: ContextAttentionConfig,
    *,
    source_horizon_tokens: int | None = None,
) -> ContextAttentionResult:
    """Attend to recent/retrieved nodes under hard node, token, and vector-work caps."""

    if not isinstance(candidates, tuple) or any(
        not isinstance(candidate, AttentionCandidate) for candidate in candidates
    ):
        raise TypeError("context attention candidates must be a tuple of AttentionCandidate values.")
    if not isinstance(graph, ContextGraph) or not isinstance(config, ContextAttentionConfig):
        raise TypeError("context attention requires a ContextGraph and ContextAttentionConfig.")
    raw_query = np.asarray(query)
    if (
        raw_query.ndim != 1
        or not raw_query.size
        or not any(np.issubdtype(raw_query.dtype, kind) for kind in (np.integer, np.floating))
        or not np.all(np.isfinite(raw_query))
    ):
        raise ValueError("context attention query must be a non-empty finite vector.")
    query = raw_query.astype(np.float64, copy=False)
    if len(candidates) > config.maximum_scored_nodes:
        raise ValueError("attention candidates exceed maximum_scored_nodes.")
    if len({candidate.node_id for candidate in candidates}) != len(candidates):
        raise ValueError("attention candidate node ids must be unique.")
    missing = sorted({candidate.node_id for candidate in candidates} - set(graph.nodes))
    if missing:
        raise KeyError("attention candidates refer to missing graph nodes: %s" % ", ".join(missing))
    if candidates and any(candidate.key.shape != query.shape for candidate in candidates):
        raise ValueError("attention candidate key shape must match query.")
    value_shapes = {candidate.value.shape for candidate in candidates}
    if len(value_shapes) > 1:
        raise ValueError("attention candidate values must have one shared shape.")
    for candidate in candidates:
        node = graph.nodes[candidate.node_id]
        if candidate.tokenizer_id != config.tokenizer_id:
            raise ValueError("attention candidate tokenizer does not match the configured tokenizer.")
        if candidate.materialized_content_hash != node.content_hash:
            raise ValueError("attention candidate tokens are not bound to the current graph-node content.")
        if candidate.measured_token_count != node.token_count:
            raise ValueError("declared node token_count does not match materialized token IDs.")
    measured_source_tokens = sum(candidate.measured_token_count for candidate in candidates)
    if source_horizon_tokens is not None:
        _nonnegative_integer(source_horizon_tokens, "source_horizon_tokens")
        if source_horizon_tokens < measured_source_tokens:
            raise ValueError("source_horizon_tokens cannot be smaller than measured candidate tokens.")

    admissible = []
    excluded = []
    for candidate in candidates:
        node = graph.nodes[candidate.node_id]
        if node.generated and node.evidence_status is not EvidenceStatus.SUPPORTED:
            excluded.append(candidate.node_id)
        else:
            admissible.append(candidate)
    value_size = next(iter(value_shapes), query.shape)[0]
    worst_case_elements = len(admissible) * query.size + config.maximum_active_nodes * (query.size + value_size)
    if worst_case_elements > config.maximum_attention_elements:
        raise ValueError("attention request exceeds maximum_attention_elements before scoring.")

    selected: list[AttentionCandidate] = []
    exact: list[str] = []
    tokens = 0
    for candidate in sorted(admissible, key=lambda row: (-row.position, row.node_id)):
        if len(exact) >= config.exact_near_nodes or len(selected) >= config.maximum_active_nodes:
            break
        node_tokens = candidate.measured_token_count
        if tokens + node_tokens > config.exact_near_tokens:
            continue
        selected.append(candidate)
        exact.append(candidate.node_id)
        tokens += node_tokens

    selected_ids = set(exact)
    query_norm = float(np.linalg.norm(query))
    similarities: dict[str, float] = {}
    ranked_far = []
    scored_nodes = 0
    for candidate in admissible:
        if candidate.node_id in selected_ids:
            continue
        denominator = query_norm * float(np.linalg.norm(candidate.key))
        similarity = float(np.dot(query, candidate.key) / denominator) if denominator > 0.0 else 0.0
        similarities[candidate.node_id] = similarity
        ranked_far.append((-similarity, -candidate.position, candidate.node_id, candidate))
        scored_nodes += 1
    retrieved = []
    for _, _, node_id, candidate in sorted(ranked_far):
        if len(retrieved) >= config.retrieved_nodes or len(selected) >= config.maximum_active_nodes:
            break
        node_tokens = candidate.measured_token_count
        if tokens + node_tokens > config.maximum_active_tokens:
            continue
        selected.append(candidate)
        retrieved.append(node_id)
        tokens += node_tokens

    if selected:
        keys = np.stack([candidate.key for candidate in selected]).astype(np.float64, copy=False)
        values = np.stack([candidate.value for candidate in selected]).astype(np.float64, copy=False)
        logits = keys @ query / config.temperature
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()
        attended = weights @ values
    else:
        attended = np.zeros(next(iter(value_shapes), query.shape), dtype=np.float64)
    scoring_elements = scored_nodes * query.size + len(selected) * (query.size + value_size)
    if scoring_elements > config.maximum_attention_elements:
        raise RuntimeError("measured attention work exceeded its preflight bound.")
    active_counts = tuple((candidate.node_id, candidate.measured_token_count) for candidate in selected)
    receipt = ContextAttentionReceipt(
        config.tokenizer_id,
        len(candidates),
        source_horizon_tokens,
        config.maximum_scored_nodes,
        config.maximum_attention_elements,
        config.maximum_active_nodes,
        config.maximum_active_tokens,
        scored_nodes,
        scoring_elements,
        tuple(exact),
        tuple(retrieved),
        len(selected),
        tokens,
        active_counts,
        tuple(sorted(excluded)),
        similarities,
    )
    attended.setflags(write=False)
    return ContextAttentionResult(attended, receipt)


__all__ = [
    "AttentionCandidate",
    "ContextAttentionConfig",
    "ContextAttentionReceipt",
    "ContextAttentionResult",
    "bounded_context_attention",
]
