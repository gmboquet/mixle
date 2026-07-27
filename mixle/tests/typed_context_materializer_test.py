"""Bounded active-context and provenance-complete materialization tests."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ContextEdge,
    ContextEdgeKind,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    ContextTokenizer,
    EvidenceStatus,
    MaterializationPolicy,
    Provenance,
    materialize_context,
)

pytestmark = [pytest.mark.experimental]


def _provenance():
    return Provenance("paper", "2026-01", "section=results", "sha256:paper")


def _tokenizer():
    return ContextTokenizer("whitespace-v1", lambda text: tuple(range(len(text.split()))))


def _graph():
    graph = ContextGraph()
    graph.add_node(
        ContextNode(
            "source",
            ContextNodeKind.SOURCE_CHUNK,
            "The measured source result.",
            4,
            provenance=(_provenance(),),
            evidence_status=EvidenceStatus.SUPPORTED,
        )
    )
    graph.add_node(
        ContextNode(
            "claim",
            ContextNodeKind.CLAIM,
            "The supported claim.",
            5,
            provenance=(_provenance(),),
            evidence_status=EvidenceStatus.SUPPORTED,
            confidence=0.95,
        )
    )
    graph.add_edge(ContextEdge("support", "source", "claim", ContextEdgeKind.SUPPORTS))
    graph.add_node(
        ContextNode(
            "generated",
            ContextNodeKind.GENERATED_HYPOTHESIS,
            "An attractive but unverified generated claim.",
            3,
            generated=True,
        )
    )
    graph.add_node(ContextNode("large", ContextNodeKind.MEMORY, "Large low-value memory.", 100))
    return graph


def test_trillion_token_source_horizon_materializes_small_verified_support_bundle():
    graph = _graph()
    result = materialize_context(
        graph,
        {"claim": 10.0, "generated": 100.0, "large": 1.0, "source": 0.0},
        MaterializationPolicy(token_budget=9, attended_token_budget=6),
        _tokenizer(),
        source_horizon_tokens=1_000_000_000_000,
        context_actions=7,
        retrieval_actions=3,
        generation_actions=2,
        verification_actions=2,
        stopped_reason="expected-value-below-cost",
    )

    assert set(result.node_ids) == {"source", "claim"}
    assert result.edge_ids == ("support",)
    assert result.token_count == 9
    assert result.attended_tokens == 6
    assert result.tokenizer_id == "whitespace-v1"
    assert len(result.token_ids) == 9
    assert result.attended_token_indices == tuple(range(6))
    assert result.excluded["generated"] == "claim-not-supported"
    assert result.excluded["large"] == "materialization-budget"
    assert "sources=paper" in result.text
    assert "generated" not in result.text
    assert result.measurement.active_to_source_ratio == pytest.approx(9.0e-12)
    assert result.measurement.verified_claim_fraction == 1.0
    json.dumps(result.as_dict(), allow_nan=False)


def test_required_unverified_generated_claim_is_rejected_not_smuggled_into_prompt():
    with pytest.raises(ValueError, match="not admissible"):
        materialize_context(
            _graph(),
            {"generated": 100.0},
            MaterializationPolicy(token_budget=100),
            _tokenizer(),
            required_node_ids=("generated",),
        )


def test_required_support_bundle_must_fit_as_a_unit():
    with pytest.raises(ValueError, match="exceeds"):
        materialize_context(
            _graph(),
            {"claim": 10.0, "source": 1.0},
            MaterializationPolicy(token_budget=8),
            _tokenizer(),
            required_node_ids=("claim",),
        )


def test_contradicted_source_excludes_dependent_claim_bundle():
    graph = _graph()
    graph.verify("source", EvidenceStatus.CONTRADICTED, provenance=(_provenance(),), confidence=0.9)
    result = materialize_context(
        graph,
        {"claim": 10.0, "source": 1.0},
        MaterializationPolicy(token_budget=20),
        _tokenizer(),
    )
    assert "claim" not in result.node_ids
    assert result.excluded["source"] == "contradicted"
    assert result.excluded["claim"] == "support-bundle-not-admissible"


def test_required_claim_cannot_smuggle_in_an_inadmissible_support_node():
    graph = ContextGraph()
    graph.add_node(
        ContextNode(
            "generated-support",
            ContextNodeKind.GENERATED_HYPOTHESIS,
            "Unverified support",
            2,
            generated=True,
        )
    )
    graph.add_node(
        ContextNode(
            "claim",
            ContextNodeKind.CLAIM,
            "Supported claim",
            2,
            provenance=(_provenance(),),
            evidence_status=EvidenceStatus.SUPPORTED,
        )
    )
    graph.add_edge(
        ContextEdge(
            "support",
            "generated-support",
            "claim",
            ContextEdgeKind.SUPPORTS,
        )
    )

    with pytest.raises(ValueError, match="support bundle is not admissible"):
        materialize_context(
            graph,
            {"claim": 10.0, "generated-support": 1.0},
            MaterializationPolicy(token_budget=100),
            _tokenizer(),
            required_node_ids=("claim",),
        )


def test_rendered_headers_are_included_in_the_hard_prompt_budget():
    graph = ContextGraph()
    graph.add_node(ContextNode("memory", ContextNodeKind.MEMORY, "one", 1))
    result = materialize_context(
        graph,
        {"memory": 1.0},
        MaterializationPolicy(token_budget=1),
        _tokenizer(),
    )

    assert result.text == ""
    assert result.token_ids == ()
    assert result.token_count == 0
    assert result.excluded == {"memory": "materialization-budget"}


def test_stateful_or_invalid_tokenizers_are_rejected():
    state = 0

    def stateful(text):
        nonlocal state
        state += 1
        return (state,)

    with pytest.raises(ValueError, match="deterministic"):
        ContextTokenizer("stateful", stateful).token_ids("text")
    with pytest.raises(TypeError, match="integer"):
        ContextTokenizer("invalid", lambda text: ("token",)).token_ids("text")
