"""Closed-loop retrieve/generate/verify/stop and materialization tests."""

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
    ContextSchedulerConfig,
    EffectiveContextRuntime,
    EvidenceStatus,
    MaterializationPolicy,
    Provenance,
    ValueOfInformationScheduler,
    VerificationUpdate,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _provenance():
    return Provenance("corpus", "snapshot-1", "document=42", "sha256:42")


def test_runtime_creates_then_verifies_context_before_explicit_voi_stop():
    graph = ContextGraph()

    def retrieve(action, current):
        return ContextActionResult(
            nodes=(
                ContextNode(
                    "source",
                    ContextNodeKind.SOURCE_CHUNK,
                    "Retrieved source evidence",
                    4,
                    provenance=(_provenance(),),
                    evidence_status=EvidenceStatus.SUPPORTED,
                ),
            ),
            materialized_tokens=4,
            measured_information_gain=0.8,
            outcome="retrieved",
        )

    def generate(action, current):
        return ContextActionResult(
            nodes=(
                ContextNode(
                    "hypothesis",
                    ContextNodeKind.GENERATED_HYPOTHESIS,
                    "Generated hypothesis requiring verification",
                    5,
                    generated=True,
                ),
            ),
            edges=(ContextEdge("origin", "hypothesis", "source", ContextEdgeKind.GENERATED_FROM),),
            materialized_tokens=5,
            measured_information_gain=0.5,
            outcome="generated",
        )

    def verify(action, current):
        return ContextActionResult(
            verifications=(
                VerificationUpdate(
                    "hypothesis",
                    EvidenceStatus.SUPPORTED,
                    provenance=(_provenance(),),
                    confidence=0.9,
                ),
            ),
            measured_information_gain=0.4,
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
    scheduler = ValueOfInformationScheduler(
        config=ContextSchedulerConfig(confidence_z=0.0, latency_cost=1.0, token_cost=0.0)
    )

    def provider(current, receipts):
        if "source" not in current.nodes:
            return (ContextAction("retrieve", ContextActionKind.RETRIEVE, expected_information_gain=1.0),)
        if "hypothesis" not in current.nodes:
            return (
                ContextAction(
                    "generate",
                    ContextActionKind.GENERATE_HYPOTHESIS,
                    input_nodes=("source",),
                    expected_information_gain=0.8,
                    generated_output=True,
                ),
            )
        if current.nodes["hypothesis"].evidence_status is EvidenceStatus.UNVERIFIED:
            return (
                ContextAction(
                    "verify",
                    ContextActionKind.VERIFY,
                    input_nodes=("hypothesis",),
                    expected_information_gain=0.6,
                ),
            )
        return (
            ContextAction(
                "not-worth-it",
                ContextActionKind.RETRIEVE,
                input_nodes=("source",),
                expected_information_gain=0.01,
                expected_latency_seconds=1.0,
            ),
        )

    runtime = EffectiveContextRuntime(graph, scheduler, executor, provider)
    run = runtime.run(maximum_iterations=10)
    assert run.stopping_reason == "expected-value-below-cost"
    assert len(run.completed_actions) == 3
    assert run.action_receipts[-1].action.kind is ContextActionKind.STOP
    assert graph.nodes["hypothesis"].generated
    assert graph.nodes["hypothesis"].evidence_status is EvidenceStatus.SUPPORTED

    materialized = runtime.materialize(
        {"hypothesis": 10.0, "source": 5.0},
        MaterializationPolicy(token_budget=9),
        source_horizon_tokens=1_000_000_000_000,
        required_node_ids=("hypothesis",),
    )
    assert set(materialized.node_ids) == {"source", "hypothesis"}
    assert materialized.measurement.context_actions == 3
    assert materialized.measurement.retrieval_actions == 1
    assert materialized.measurement.generation_actions == 1
    assert materialized.measurement.verification_actions == 1
    assert materialized.measurement.stopped_reason == "expected-value-below-cost"
    json.dumps(run.as_dict(), allow_nan=False)


def test_hard_maximum_iteration_guard_emits_stop_receipt():
    graph = ContextGraph()
    executor = ContextActionExecutor(
        graph,
        {ContextActionKind.TOOL_CALL: lambda action, current: ContextActionResult(outcome="tool")},
    )
    scheduler = ValueOfInformationScheduler(
        config=ContextSchedulerConfig(confidence_z=0.0, latency_cost=0.0, token_cost=0.0)
    )

    def provider(current, receipts):
        index = len(receipts)
        return (
            ContextAction(
                "tool-%d" % index,
                ContextActionKind.TOOL_CALL,
                expected_information_gain=1.0,
            ),
        )

    run = EffectiveContextRuntime(graph, scheduler, executor, provider).run(maximum_iterations=2)
    assert run.stopping_reason == "maximum-iterations"
    assert len(run.completed_actions) == 2
    assert run.action_receipts[-1].action.kind is ContextActionKind.STOP


def test_materialize_requires_completed_run():
    graph = ContextGraph()
    runtime = EffectiveContextRuntime(
        graph,
        ValueOfInformationScheduler(),
        ContextActionExecutor(graph),
        lambda current, receipts: (),
    )
    with pytest.raises(RuntimeError, match="run context"):
        runtime.materialize({}, MaterializationPolicy(token_budget=1))
