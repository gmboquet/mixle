"""Integrated graph-context MoE pilot and recovery acceptance receipt."""

import json

import pytest

pytest.importorskip("torch")

from mixle.experimental.typed_runtime import (  # noqa: E402
    ClaimKind,
    GateStatus,
    ScaleRunReceipt,
    assess_frontier_claims,
    run_graph_memory_pilot,
)

pytestmark = [pytest.mark.experimental, pytest.mark.torch]


def test_graph_context_improves_quality_under_bounded_active_memory_and_recovers_bitwise():
    receipt = run_graph_memory_pilot(
        seed=17,
        source_nodes=128,
        train_examples=128,
        test_examples=64,
        updates=60,
        microbatch_size=16,
        accumulation_steps=2,
        target_accuracy=0.9,
    )

    assert receipt.local_adamw.test_accuracy < 0.7
    assert receipt.graph_adamw.test_accuracy >= 0.95
    assert receipt.graph_routed.test_accuracy >= 0.95
    assert receipt.graph_quality_gain >= 0.25
    assert receipt.active_context_bounded
    assert receipt.graph_adamw.maximum_active_context_tokens == 9
    assert receipt.graph_adamw.source_horizon_tokens == 128
    assert receipt.local_adamw.batch.optimizer_updates == 60
    assert receipt.graph_adamw.batch.optimizer_updates == 60
    assert receipt.graph_routed.batch.optimizer_updates == 60
    assert receipt.graph_routed.batch.examples_per_microbatch == 16
    assert receipt.graph_routed.batch.accumulation_steps == 2
    assert receipt.graph_routed.batch.effective_global_examples == 32
    assert set(receipt.graph_routed.optimizer_families) == {"adamw"}
    assert receipt.recovery.passed
    assert all(row.oracle_passed for row in receipt.failure_receipts)

    assessment = assess_frontier_claims(receipt)
    assert not assessment.frontier_training_allowed
    assert not assessment.effective_trillion_context_allowed
    assert assessment.claim_allowed(ClaimKind.FRONTIER_TRAINING) is False
    assert any(gate.status is GateStatus.NOT_RUN for gate in assessment.gates)

    # Keep optimizer evidence honest: the routed reference may or may not beat AdamW.
    assert receipt.graph_adamw.time_to_target_updates is not None
    assert receipt.graph_routed.time_to_target_updates is not None
    assert receipt.graph_adamw.time_to_target_seconds is not None
    assert receipt.graph_routed.time_to_target_seconds is not None
    assert receipt.graph_adamw.time_to_target_seconds <= receipt.graph_adamw.elapsed_seconds
    assert receipt.graph_routed.time_to_target_seconds <= receipt.graph_routed.elapsed_seconds
    json.dumps(receipt.as_dict(), allow_nan=False)
    json.dumps(assessment.as_dict(), allow_nan=False)


def test_claim_gate_requires_real_scale_evidence_and_accepts_complete_receipts():
    receipt = run_graph_memory_pilot(
        seed=17,
        source_nodes=64,
        train_examples=64,
        test_examples=32,
        updates=50,
        microbatch_size=16,
        accumulation_steps=2,
        target_accuracy=0.9,
    )
    scale = ScaleRunReceipt(
        run_id="acceptance-fixture",
        accelerator_count=8,
        host_count=2,
        real_distributed_transport=True,
        model_parameters=1_000_000_000,
        source_horizon_tokens=1_000_000_000_000,
        maximum_active_context_tokens=131_072,
        baseline_time_to_target_seconds=120.0,
        candidate_time_to_target_seconds=100.0,
        baseline_peak_memory_bytes=80_000_000_000,
        candidate_peak_memory_bytes=60_000_000_000,
        quality_target_achieved=True,
        worker_loss_recovered=True,
        replay_verified=True,
        provenance_complete=True,
        evidence_uri="fixture://typed-frontier-acceptance",
    )
    assessment = assess_frontier_claims(receipt, scale)
    assert assessment.frontier_training_allowed
    assert assessment.effective_trillion_context_allowed
    assert all(gate.status is GateStatus.PASSED for gate in assessment.gates)
