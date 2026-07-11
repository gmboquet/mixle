"""Transactional retrieval, generation, verification, and failure tests."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ContextAction,
    ContextActionExecutor,
    ContextActionKind,
    ContextActionResult,
    ContextEdge,
    ContextEdgeKind,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    EvidenceStatus,
    Provenance,
    VerificationUpdate,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _provenance():
    return Provenance("source", "v1", "row=10", "sha256:10")


def test_retrieve_generate_and_verify_are_separate_receipted_actions():
    graph = ContextGraph()

    def retrieve(action, current):
        return ContextActionResult(
            nodes=(
                ContextNode(
                    "source",
                    ContextNodeKind.SOURCE_CHUNK,
                    "Observed source evidence",
                    3,
                    provenance=(_provenance(),),
                    evidence_status=EvidenceStatus.SUPPORTED,
                ),
            ),
            external_latency_seconds=0.1,
            materialized_tokens=3,
            measured_information_gain=0.4,
            outcome="retrieved",
        )

    def generate(action, current):
        return ContextActionResult(
            nodes=(
                ContextNode(
                    "hypothesis",
                    ContextNodeKind.GENERATED_HYPOTHESIS,
                    "A testable generated hypothesis",
                    4,
                    generated=True,
                ),
            ),
            edges=(ContextEdge("generated", "hypothesis", "source", ContextEdgeKind.GENERATED_FROM),),
            materialized_tokens=4,
            outcome="generated",
        )

    def verify(action, current):
        return ContextActionResult(
            verifications=(
                VerificationUpdate(
                    "hypothesis",
                    EvidenceStatus.SUPPORTED,
                    provenance=(_provenance(),),
                    confidence=0.8,
                ),
            ),
            measured_information_gain=0.2,
            outcome="verified",
        )

    executor = ContextActionExecutor(
        graph,
        {
            ContextActionKind.RETRIEVE: retrieve,
            ContextActionKind.GENERATE_HYPOTHESIS: generate,
            ContextActionKind.VERIFY: verify,
        },
    )
    retrieval = executor.execute(ContextAction("retrieve", ContextActionKind.RETRIEVE, query="evidence"))
    generation = executor.execute(
        ContextAction(
            "generate",
            ContextActionKind.GENERATE_HYPOTHESIS,
            input_nodes=("source",),
            generated_output=True,
        )
    )
    verification = executor.execute(ContextAction("verify", ContextActionKind.VERIFY, input_nodes=("hypothesis",)))

    assert retrieval.output_nodes == ("source",)
    assert generation.output_nodes == ("hypothesis",)
    assert verification.output_nodes == ()
    assert graph.nodes["hypothesis"].generated
    assert graph.nodes["hypothesis"].evidence_status is EvidenceStatus.SUPPORTED
    assert len(executor.receipts) == 3
    json.dumps(executor.as_dict(), allow_nan=False)


def test_malformed_generated_output_rolls_back_graph_but_charges_actual_work():
    graph = ContextGraph()

    def malformed(action, current):
        return ContextActionResult(
            nodes=(ContextNode("bad", ContextNodeKind.CLAIM, "Undisclosed", 1),),
            external_latency_seconds=0.2,
            materialized_tokens=10,
            tool_calls=1,
            monetary_cost=0.5,
        )

    executor = ContextActionExecutor(graph, {ContextActionKind.GENERATE_HYPOTHESIS: malformed})
    receipt = executor.execute(
        ContextAction("bad-action", ContextActionKind.GENERATE_HYPOTHESIS, generated_output=True)
    )
    assert receipt.rolled_back
    assert graph.nodes == {}
    assert graph.version == 0
    assert receipt.materialized_tokens == 10
    assert receipt.tool_calls == 1
    assert receipt.monetary_cost == pytest.approx(0.5)
    assert receipt.latency_seconds >= 0.2


def test_invalid_edge_rolls_back_nodes_added_earlier_in_same_action():
    graph = ContextGraph()

    def invalid_edge(action, current):
        return ContextActionResult(
            nodes=(ContextNode("new", ContextNodeKind.MEMORY, "Temporary", 1),),
            edges=(ContextEdge("bad-edge", "new", "missing", ContextEdgeKind.SEMANTIC),),
        )

    executor = ContextActionExecutor(graph, {ContextActionKind.LINK: invalid_edge})
    receipt = executor.execute(ContextAction("link", ContextActionKind.LINK))
    assert receipt.rolled_back
    assert graph.nodes == {}
    assert graph.edges == {}


def test_stop_action_requires_no_adapter_and_does_not_change_graph():
    graph = ContextGraph()
    executor = ContextActionExecutor(graph)
    receipt = executor.execute(ContextAction("stop", ContextActionKind.STOP))
    assert receipt.outcome == "stopped"
    assert receipt.graph_version_before == receipt.graph_version_after == 0
