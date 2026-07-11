"""Bounded active-context and provenance-complete materialization tests."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ContextEdge,
    ContextEdgeKind,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    EvidenceStatus,
    MaterializationPolicy,
    Provenance,
    materialize_context,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _provenance():
    return Provenance("paper", "2026-01", "section=results", "sha256:paper")


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
            required_node_ids=("generated",),
        )


def test_required_support_bundle_must_fit_as_a_unit():
    with pytest.raises(ValueError, match="exceeds"):
        materialize_context(
            _graph(),
            {"claim": 10.0, "source": 1.0},
            MaterializationPolicy(token_budget=8),
            required_node_ids=("claim",),
        )


def test_contradicted_source_excludes_dependent_claim_bundle():
    graph = _graph()
    graph.verify("source", EvidenceStatus.CONTRADICTED, provenance=(_provenance(),), confidence=0.9)
    result = materialize_context(
        graph,
        {"claim": 10.0, "source": 1.0},
        MaterializationPolicy(token_budget=20),
    )
    assert "claim" not in result.node_ids
    assert result.excluded["source"] == "contradicted"
    assert result.excluded["claim"] == "support-bundle-not-admissible"
