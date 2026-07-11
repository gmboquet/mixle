"""Provenance, generation disclosure, revisitation, and rollback tests for context IR."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ContextAction,
    ContextActionKind,
    ContextActionReceipt,
    ContextEdge,
    ContextEdgeKind,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    EvidenceStatus,
    Provenance,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _provenance(source="paper"):
    return Provenance(source, "v1", "page=3#paragraph=2", "sha256:abc", uri="https://example.test/paper")


def test_generated_hypothesis_stays_generated_after_source_verification():
    graph = ContextGraph()
    source = ContextNode(
        "source",
        ContextNodeKind.SOURCE_CHUNK,
        "Measured result from the source.",
        7,
        provenance=(_provenance(),),
        evidence_status=EvidenceStatus.SUPPORTED,
        source_horizon_position=999_999_999_999,
    )
    hypothesis = ContextNode(
        "hypothesis",
        ContextNodeKind.GENERATED_HYPOTHESIS,
        "Perhaps the effect is conditional.",
        6,
        generated=True,
    )
    graph.add_node(source)
    graph.add_node(hypothesis)
    graph.add_edge(ContextEdge("derived", "hypothesis", "source", ContextEdgeKind.GENERATED_FROM))
    assert graph.unresolved_nodes() == (hypothesis,)

    verified = graph.verify(
        "hypothesis",
        EvidenceStatus.SUPPORTED,
        provenance=(_provenance(),),
        confidence=0.9,
    )
    assert verified.generated is True
    assert verified.evidence_status is EvidenceStatus.SUPPORTED
    assert verified.provenance == (_provenance(),)
    assert graph.unresolved_nodes() == ()
    assert graph.neighbors("source") == ("hypothesis",)
    json.dumps(graph.as_dict(), allow_nan=False)


def test_supported_node_requires_provenance_and_generated_kind_requires_disclosure():
    with pytest.raises(ValueError, match="provenance"):
        ContextNode("claim", ContextNodeKind.CLAIM, "Claim", 1, evidence_status=EvidenceStatus.SUPPORTED)
    with pytest.raises(ValueError, match="generated=True"):
        ContextNode("query", ContextNodeKind.GENERATED_QUERY, "What evidence?", 2)

    with pytest.raises(ValueError, match="generated_output"):
        ContextAction("generate", ContextActionKind.GENERATE_HYPOTHESIS)


def test_graph_snapshot_restores_failed_context_expansion_exactly():
    graph = ContextGraph()
    graph.add_node(ContextNode("memory", ContextNodeKind.MEMORY, "Known context", 2))
    snapshot = graph.snapshot()
    graph.add_node(ContextNode("query", ContextNodeKind.GENERATED_QUERY, "Expand this source", 3, generated=True))
    graph.add_edge(ContextEdge("expand", "query", "memory", ContextEdgeKind.EXPANDS))
    assert graph.version > snapshot[0]
    graph.restore(snapshot)
    assert graph.version == snapshot[0]
    assert tuple(graph.nodes) == ("memory",)
    assert graph.edges == {}


def test_node_id_collision_cannot_rewrite_provenance():
    graph = ContextGraph()
    original = ContextNode("source", ContextNodeKind.SOURCE_CHUNK, "Original", 1, provenance=(_provenance(),))
    graph.add_node(original)
    graph.add_node(original)
    with pytest.raises(ValueError, match="collision"):
        graph.add_node(ContextNode("source", ContextNodeKind.SOURCE_CHUNK, "Rewritten", 1, provenance=(_provenance(),)))


def test_context_action_and_receipt_keep_expected_and_actual_work_separate():
    action = ContextAction(
        "retrieve-1",
        ContextActionKind.RETRIEVE,
        query="conditional effects",
        source_scope=("papers", "graph"),
        expected_information_gain=0.4,
        gain_standard_error=0.1,
        expected_latency_seconds=0.2,
        expected_tokens=1_000,
        expected_monetary_cost=0.01,
        maximum_outputs=5,
    )
    receipt = ContextActionReceipt(
        action,
        2,
        5,
        ("source-a", "source-b"),
        ("edge-a",),
        0.3,
        800,
        0,
        0.012,
        0.35,
        "retrieved-two-sources",
    )
    assert receipt.action.expected_tokens == 1_000
    assert receipt.materialized_tokens == 800
    json.dumps(receipt.as_dict(), allow_nan=False)
