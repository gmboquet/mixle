"""Bounded near-field plus retrieved-far effective-context tests."""

import json
from dataclasses import replace

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    AttentionCandidate,
    ContextAttentionConfig,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    bounded_context_attention,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _fixture(count=500, target_index=40):
    rng = np.random.default_rng(9)
    dimension = 16
    query = rng.normal(size=dimension)
    graph = ContextGraph()
    candidates = []
    for index in range(count):
        node_id = "n%04d" % index
        node = ContextNode(node_id, ContextNodeKind.MEMORY, "memory %d" % index, 1)
        graph.add_node(node)
        key = rng.normal(size=dimension)
        value = np.zeros(1)
        if index == target_index:
            key = 10.0 * query
            value = np.ones(1)
        candidates.append(
            AttentionCandidate(
                node_id,
                key,
                value,
                index,
                (index,),
                "test-tokenizer-v1",
                node.content_hash,
            )
        )
    return query, graph, tuple(candidates)


def test_far_needle_is_retrieved_while_active_tokens_stay_bounded():
    query, graph, candidates = _fixture()
    config = ContextAttentionConfig(
        "test-tokenizer-v1",
        exact_near_tokens=8,
        exact_near_nodes=8,
        retrieved_nodes=4,
        maximum_active_tokens=12,
        maximum_active_nodes=12,
    )
    result = bounded_context_attention(
        query,
        candidates,
        graph,
        config,
        source_horizon_tokens=1_000_000_000_000,
    )
    assert "n0040" in result.receipt.retrieved_node_ids
    assert result.receipt.active_tokens <= 12
    assert result.receipt.active_nodes <= 12
    assert result.receipt.scoring_elements <= config.maximum_attention_elements
    assert len(result.receipt.exact_node_ids) == 8
    assert result.receipt.source_nodes == 500
    assert result.receipt.source_horizon_tokens == 1_000_000_000_000
    assert float(result.value[0]) > 0.9
    json.dumps(result.receipt.as_dict(), allow_nan=False)


def test_quality_improves_when_source_horizon_expands_but_active_memory_does_not():
    query, graph, candidates = _fixture()
    config = ContextAttentionConfig(
        "test-tokenizer-v1",
        exact_near_tokens=8,
        exact_near_nodes=8,
        retrieved_nodes=4,
        maximum_active_tokens=12,
        maximum_active_nodes=12,
    )
    short = bounded_context_attention(query, candidates[100:], graph, config, source_horizon_tokens=400)
    long = bounded_context_attention(query, candidates, graph, config, source_horizon_tokens=500)
    assert float(short.value[0]) == 0.0
    assert float(long.value[0]) > 0.9
    assert short.receipt.active_tokens <= 12
    assert long.receipt.active_tokens <= 12


def test_unverified_generated_node_is_excluded_even_with_perfect_key():
    query, graph, candidates = _fixture(count=40, target_index=5)
    generated = ContextNode(
        "generated",
        ContextNodeKind.GENERATED_HYPOTHESIS,
        "Unverified answer-shaped text",
        1,
        generated=True,
    )
    graph.add_node(generated)
    candidates = candidates + (
        AttentionCandidate(
            "generated",
            100.0 * query,
            np.ones(1) * 100.0,
            100,
            (999,),
            "test-tokenizer-v1",
            generated.content_hash,
        ),
    )
    result = bounded_context_attention(
        query,
        candidates,
        graph,
        ContextAttentionConfig(
            "test-tokenizer-v1",
            exact_near_tokens=4,
            exact_near_nodes=4,
            retrieved_nodes=2,
            maximum_active_tokens=6,
            maximum_active_nodes=6,
        ),
    )
    assert result.receipt.excluded_unverified_generated == ("generated",)
    assert "generated" not in result.receipt.exact_node_ids + result.receipt.retrieved_node_ids


def test_candidate_key_shape_must_match_query():
    query, graph, candidates = _fixture(count=2, target_index=0)
    bad = AttentionCandidate(
        "n0000",
        np.ones(3),
        np.zeros(1),
        0,
        (0,),
        "test-tokenizer-v1",
        graph.nodes["n0000"].content_hash,
    )
    with pytest.raises(ValueError, match="key shape"):
        bounded_context_attention(query, (bad, candidates[1]), graph, ContextAttentionConfig("test-tokenizer-v1"))


def test_zero_declared_tokens_cannot_bypass_node_or_materialized_token_bounds():
    query = np.ones(2)
    graph = ContextGraph()
    candidates = []
    for index in range(20):
        node = ContextNode(f"zero-{index}", ContextNodeKind.MEMORY, f"nonempty {index}", 0)
        graph.add_node(node)
        candidates.append(
            AttentionCandidate(
                node.node_id,
                np.ones(2),
                np.ones(1),
                index,
                (index,),
                "test-tokenizer-v1",
                node.content_hash,
            )
        )
    config = ContextAttentionConfig(
        "test-tokenizer-v1",
        exact_near_tokens=10,
        exact_near_nodes=2,
        retrieved_nodes=2,
        maximum_active_tokens=10,
        maximum_active_nodes=4,
    )
    with pytest.raises(ValueError, match="does not match materialized token IDs"):
        bounded_context_attention(query, tuple(candidates), graph, config)


def test_source_node_and_vector_work_bounds_fail_before_attention():
    query, graph, candidates = _fixture(count=5, target_index=0)
    with pytest.raises(ValueError, match="maximum_scored_nodes"):
        bounded_context_attention(
            query,
            candidates,
            graph,
            ContextAttentionConfig("test-tokenizer-v1", maximum_scored_nodes=4),
        )
    with pytest.raises(ValueError, match="maximum_attention_elements"):
        bounded_context_attention(
            query,
            candidates,
            graph,
            ContextAttentionConfig(
                "test-tokenizer-v1",
                maximum_active_nodes=1,
                exact_near_nodes=1,
                maximum_attention_elements=10,
            ),
        )


def test_materialized_tokens_are_content_and_tokenizer_bound_and_horizon_validated():
    query, graph, candidates = _fixture(count=2, target_index=0)
    config = ContextAttentionConfig("test-tokenizer-v1")
    with pytest.raises(ValueError, match="configured tokenizer"):
        bounded_context_attention(
            query,
            (replace(candidates[0], tokenizer_id="other"), candidates[1]),
            graph,
            config,
        )
    with pytest.raises(ValueError, match="current graph-node content"):
        bounded_context_attention(
            query,
            (replace(candidates[0], materialized_content_hash="stale"), candidates[1]),
            graph,
            config,
        )
    with pytest.raises(ValueError, match="measured candidate tokens"):
        bounded_context_attention(query, candidates, graph, config, source_horizon_tokens=1)
