"""Bitwise and explicit-tolerance proposal replay tests."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    CanaryVerdict,
    CostEstimate,
    MergeLaw,
    ObjectiveKind,
    ProposalBatch,
    ProposalPacket,
    ReplayLog,
    ReplayMode,
    StateSemantics,
    TransactionalCoordinator,
    TransactionParticipant,
    UpdateContract,
    UpdateGraph,
    UpdateKind,
    UpdateNode,
    payload_fingerprint,
    replay_log,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _graph():
    contract = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.FIRST_ORDER,
        merge_law=MergeLaw.NON_MERGEABLE,
        state_semantics=frozenset({StateSemantics.MUTABLE_PARAMETERS}),
        exact=False,
    )
    node = UpdateNode("node", "root", "Fixture", "FixtureEstimator", contract, CostEstimate(1.0), 2)
    return UpdateGraph((node,), (), "node")


def _proposal(proposal_id, base, dependency, delta):
    return ProposalPacket(
        proposal_id=proposal_id,
        run_id="run",
        model_id="model",
        node_id="node",
        shard_id="worker",
        base_model_version=base,
        dependency_versions={"node": dependency},
        update_kind=UpdateKind.FIRST_ORDER,
        objective_kind=ObjectiveKind.MLE,
        payload={"delta": np.asarray(delta, dtype=np.float64)},
    )


def _coordinator(*, noise=0.0):
    state = {"parameters": np.zeros(2, dtype=np.float64)}
    participant = TransactionParticipant(
        "parameters",
        frozenset({StateSemantics.MUTABLE_PARAMETERS}),
        lambda: state["parameters"].copy(),
        lambda value: state.__setitem__("parameters", value.copy()),
        lambda: payload_fingerprint(state["parameters"]),
    )

    def apply(proposal):
        state["parameters"] += proposal.payload["delta"] + noise

    def canary(batch):
        proposed_gain = sum(float(np.sum(proposal.payload["delta"])) for proposal in batch.proposals)
        after = float(np.sum(state["parameters"]))
        return CanaryVerdict(True, "deterministic canary", after - proposed_gain, after, sample_count=8)

    return state, TransactionalCoordinator(_graph(), apply, canary, participants=(participant,))


def test_two_commit_log_replays_bitwise_with_identical_versions_and_state():
    state, coordinator = _coordinator()
    log = ReplayLog()
    batches = (
        ProposalBatch("batch-1", (_proposal("p1", 0, 0, [1.0, 2.0]),)),
        ProposalBatch("batch-2", (_proposal("p2", 1, 1, [0.5, 0.5]),)),
    )
    for batch in batches:
        receipt = coordinator.commit(batch)
        log.record(batch, receipt, expected_state=state)

    replay_state, replay_coordinator = _coordinator()
    report = replay_log(log, replay_coordinator, state_probe=lambda: replay_state)

    assert report.matched
    assert all(step.matched for step in report.steps)
    np.testing.assert_array_equal(replay_state["parameters"], state["parameters"])
    assert replay_coordinator.versions.as_dict() == coordinator.versions.as_dict()
    json.dumps(log.as_dict(), allow_nan=False)
    json.dumps(report.as_dict(), allow_nan=False)


def test_tolerance_replay_accepts_declared_numerical_drift_but_bitwise_does_not():
    state, coordinator = _coordinator()
    batch = ProposalBatch("batch", (_proposal("p1", 0, 0, [1.0, 2.0]),))
    receipt = coordinator.commit(batch)
    log = ReplayLog()
    log.record(batch, receipt, expected_state=state)

    bitwise_state, bitwise_coordinator = _coordinator(noise=1.0e-10)
    bitwise = replay_log(log, bitwise_coordinator, state_probe=lambda: bitwise_state)
    assert not bitwise.matched
    assert "participant-state-fingerprint" in bitwise.steps[0].mismatches

    tolerance_state, tolerance_coordinator = _coordinator(noise=1.0e-10)
    tolerance = replay_log(
        log,
        tolerance_coordinator,
        mode=ReplayMode.TOLERANCE,
        state_probe=lambda: tolerance_state,
        absolute_tolerance=1.0e-8,
        relative_tolerance=1.0e-8,
    )
    assert tolerance.matched


def test_mutated_logged_payload_is_detected_before_apply():
    state, coordinator = _coordinator()
    batch = ProposalBatch("batch", (_proposal("p1", 0, 0, [1.0, 2.0]),))
    receipt = coordinator.commit(batch)
    log = ReplayLog()
    log.record(batch, receipt, expected_state=state)
    log.entries[0].batch.proposals[0].payload["delta"][0] = 999.0

    replay_state, replay_coordinator = _coordinator()
    report = replay_log(log, replay_coordinator, state_probe=lambda: replay_state)
    assert not report.matched
    assert report.steps[0].actual_receipt is None
    assert report.steps[0].mismatches == ("proposal-payload-mutated:p1",)
    np.testing.assert_array_equal(replay_state["parameters"], [0.0, 0.0])


def test_tolerance_replay_requires_state_probe():
    with pytest.raises(ValueError, match="state_probe"):
        replay_log(ReplayLog(), _coordinator()[1], mode=ReplayMode.TOLERANCE)
