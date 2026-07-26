"""Transactional commit, versioning, canary, and rollback tests."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    ArtifactKind,
    CanaryVerdict,
    CommitStatus,
    CostEstimate,
    DependencyEdge,
    MergeLaw,
    ObjectiveGateEvidence,
    ObjectiveKind,
    ProposalBatch,
    ProposalPacket,
    StateSemantics,
    TransactionParticipant,
    UpdateContract,
    UpdateGraph,
    UpdateKind,
    UpdateNode,
    payload_fingerprint,
)
from mixle.experimental.typed_runtime import (
    TransactionalCoordinator as RuntimeTransactionalCoordinator,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def TransactionalCoordinator(*args, **kwargs):
    """Test fixture coordinator bound to the fixture proposal identity."""

    return RuntimeTransactionalCoordinator(
        *args,
        run_id="run",
        model_id="model",
        **kwargs,
    )


def _mutable_contract(update=UpdateKind.FIRST_ORDER):
    return UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=update,
        merge_law=MergeLaw.NON_MERGEABLE,
        state_semantics=frozenset(
            {
                StateSemantics.MUTABLE_PARAMETERS,
                StateSemantics.MUTABLE_OPTIMIZER,
                StateSemantics.STOCHASTIC_RNG,
            }
        ),
        reads=frozenset({ArtifactKind.PARAMETERS, ArtifactKind.OPTIMIZER_STATE, ArtifactKind.RNG_STATE}),
        writes=frozenset({ArtifactKind.PARAMETERS, ArtifactKind.OPTIMIZER_STATE, ArtifactKind.RNG_STATE}),
        exact=False,
        declared_by="test",
    )


def _single_node_graph():
    node = UpdateNode(
        "node",
        "root",
        "MutableFixture",
        "MutableEstimator",
        _mutable_contract(),
        CostEstimate(compute_units=1.0),
        2,
    )
    return UpdateGraph((node,), (), "node")


def _packet(*, base_version=0, node_id="node", proposal_id="p1", update=UpdateKind.FIRST_ORDER):
    return ProposalPacket(
        proposal_id=proposal_id,
        run_id="run",
        model_id="model",
        node_id=node_id,
        shard_id="worker-0",
        base_model_version=base_version,
        dependency_versions={node_id: 0},
        update_kind=update,
        objective_kind=ObjectiveKind.MLE,
        payload={"delta": np.array([1.0, -1.0])},
        writes=frozenset({ArtifactKind.PARAMETERS, ArtifactKind.OPTIMIZER_STATE, ArtifactKind.RNG_STATE}),
    )


def _state_and_participants(*, broken_restore=False):
    state = {
        "parameters": np.array([0.0, 0.0]),
        "optimizer": {"step": 0, "moment": np.array([0.0, 0.0])},
        "rng": {"counter": 7},
    }

    def participant(name, semantics):
        artifacts = {
            StateSemantics.MUTABLE_PARAMETERS: ArtifactKind.PARAMETERS,
            StateSemantics.MUTABLE_OPTIMIZER: ArtifactKind.OPTIMIZER_STATE,
            StateSemantics.STOCHASTIC_RNG: ArtifactKind.RNG_STATE,
        }

        def snapshot():
            return copy.deepcopy(state[name])

        def restore(value):
            if not broken_restore:
                state[name] = copy.deepcopy(value)

        return TransactionParticipant(
            name,
            frozenset({semantics}),
            snapshot,
            restore,
            lambda: payload_fingerprint(state[name]),
            frozenset({artifacts[semantics]}),
        )

    participants = (
        participant("parameters", StateSemantics.MUTABLE_PARAMETERS),
        participant("optimizer", StateSemantics.MUTABLE_OPTIMIZER),
        participant("rng", StateSemantics.STOCHASTIC_RNG),
    )
    return state, participants


def _apply(state, proposal):
    state["parameters"] += proposal.payload["delta"]
    state["optimizer"]["step"] += 1
    state["optimizer"]["moment"] += proposal.payload["delta"]
    state["rng"]["counter"] += 1


class CommitPreflightTest:
    def test_coordinator_rejects_nonfinite_tolerance_and_foreign_identity(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        with pytest.raises(ValueError, match="finite"):
            RuntimeTransactionalCoordinator(
                graph,
                lambda proposal: _apply(state, proposal),
                lambda batch: CanaryVerdict(True, "ok", 0.0, 1.0),
                run_id="run",
                model_id="model",
                participants=participants,
                objective_tolerance=float("nan"),
            )

        coordinator = RuntimeTransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(True, "ok", 0.0, 1.0),
            run_id="other-run",
            model_id="model",
            participants=participants,
        )
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.REJECTED
        assert receipt.reason == "run-id-mismatch:p1"

    def test_stale_model_version_never_calls_apply_or_canary(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        calls = []
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: calls.append("apply"),
            lambda batch: calls.append("canary"),
            participants=participants,
        )

        receipt = coordinator.commit(_packet(base_version=1))
        assert receipt.status is CommitStatus.REJECTED
        assert receipt.reason == "base-model-version-mismatch:node"
        assert calls == []
        assert coordinator.versions.model_version == 0
        assert np.array_equal(state["parameters"], [0.0, 0.0])

    def test_missing_declared_rng_state_rejects_before_mutation(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(True, "ok", 0.0, 1.0),
            participants=participants[:-1],
        )
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.REJECTED
        assert receipt.reason == "missing-transaction-state:stochastic_rng"
        assert np.array_equal(state["parameters"], [0.0, 0.0])

    def test_every_mutated_artifact_requires_explicit_snapshot_coverage(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        uncovered_rng = replace(
            participants[-1],
            artifacts=frozenset({ArtifactKind.PARAMETERS}),
        )
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(True, "ok", 0.0, 1.0),
            participants=participants[:-1] + (uncovered_rng,),
        )
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.REJECTED
        assert receipt.reason == "missing-transaction-artifact:rng_state"

    def test_preflight_rejection_does_not_consume_safe_retry_identity(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(True, "ok", 0.0, 1.0),
            participants=participants,
        )
        mutated = _packet()
        mutated.payload["delta"][0] = 99.0
        first = coordinator.commit(mutated)
        second = coordinator.commit(_packet())

        assert first.status is CommitStatus.REJECTED
        assert first.reason == "proposal-payload-mutated:p1"
        assert second.status is CommitStatus.ACCEPTED
        np.testing.assert_array_equal(state["parameters"], [1.0, -1.0])

    def test_commit_ids_cannot_be_reused(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(True, "ok", 0.0, 1.0),
            participants=participants,
        )
        coordinator.commit(_packet(), commit_id="fixed")
        with pytest.raises(ValueError, match="already been used"):
            coordinator.commit(_packet(proposal_id="p2", base_version=1), commit_id="fixed")


class RollbackTest:
    def test_verified_canary_rejection_allows_safe_same_proposal_retry(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        attempts = 0

        def canary(batch):
            nonlocal attempts
            attempts += 1
            return CanaryVerdict(
                attempts > 1,
                "transient probe",
                0.0,
                1.0 if attempts > 1 else -1.0,
            )

        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            canary,
            participants=participants,
        )
        first = coordinator.commit(_packet())
        second = coordinator.commit(_packet())
        assert first.status is CommitStatus.ROLLED_BACK
        assert second.status is CommitStatus.ACCEPTED
        np.testing.assert_array_equal(state["parameters"], [1.0, -1.0])

    def test_canary_rejection_restores_parameters_optimizer_and_rng(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        before = payload_fingerprint(state)
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(False, "quality floor failed", 0.0, -1.0, sample_count=32),
            participants=participants,
        )

        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ROLLED_BACK
        assert receipt.rollback_verified is True
        assert payload_fingerprint(state) == before
        assert receipt.versions_before == receipt.versions_after
        assert coordinator.versions.model_version == 0
        json.dumps(receipt.as_dict(), allow_nan=False)

    def test_apply_exception_rolls_back_all_tentative_state(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        before = payload_fingerprint(state)

        def exploding_apply(proposal):
            _apply(state, proposal)
            raise RuntimeError("injected apply failure")

        coordinator = TransactionalCoordinator(
            graph,
            exploding_apply,
            lambda batch: CanaryVerdict(True, "unreachable", 0.0, 1.0),
            participants=participants,
        )
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ROLLED_BACK
        assert receipt.reason == "apply-or-canary-error"
        assert receipt.error_type == "RuntimeError"
        assert payload_fingerprint(state) == before

    def test_unverified_rollback_poison_blocks_later_commits(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants(broken_restore=True)
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(False, "reject", 0.0, -1.0),
            participants=participants,
        )
        first = coordinator.commit(_packet())
        second = coordinator.commit(_packet(proposal_id="p2"))

        assert first.status is CommitStatus.ROLLBACK_FAILED
        assert first.rollback_verified is False
        assert coordinator.poisoned
        assert second.status is CommitStatus.REJECTED
        assert second.reason == "coordinator-poisoned-by-unverified-rollback"


class AcceptedCommitTest:
    def test_accepted_commit_advances_versions_once(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(True, "held-out improved", 0.0, 2.0, lower_confidence_gain=0.5),
            participants=participants,
        )

        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ACCEPTED
        assert coordinator.versions.model_version == 1
        assert coordinator.versions.node_versions == {"node": 1}
        assert receipt.invalidated_nodes == ("node",)
        np.testing.assert_array_equal(state["parameters"], [1.0, -1.0])
        assert state["optimizer"]["step"] == 1
        assert state["rng"]["counter"] == 8

        stale = replace(_packet(proposal_id="p2"), dependency_versions={"node": 1})
        stale_receipt = coordinator.commit(stale)
        assert stale_receipt.status is CommitStatus.REJECTED
        assert stale_receipt.reason == "base-model-version-mismatch:node"

    def test_strict_objective_regression_rolls_back_even_if_callback_says_accept(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(True, "raw detector passed", 3.0, 2.0),
            participants=participants,
        )
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ROLLED_BACK
        assert receipt.reason == "strict-objective-regression"
        np.testing.assert_array_equal(state["parameters"], [0.0, 0.0])

    def test_negative_confidence_bound_rolls_back_an_improved_point_estimate(self):
        graph = _single_node_graph()
        state, participants = _state_and_participants()
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(
                True,
                "mean improved but uncertainty is too large",
                0.0,
                1.0,
                lower_confidence_gain=-0.1,
            ),
            participants=participants,
        )
        receipt = coordinator.commit(_packet())
        assert receipt.status is CommitStatus.ROLLED_BACK
        assert receipt.reason == "negative-canary-lower-bound"

    def test_surrogate_batch_member_cannot_disable_strict_proposal_gate(self):
        strict = replace(_mutable_contract(UpdateKind.COORDINATE), objective_kind=ObjectiveKind.MLE)
        surrogate = replace(
            strict,
            objective_kind=ObjectiveKind.USER_SURROGATE,
            outer_objective_compatible=False,
            exact=False,
        )
        graph = UpdateGraph(
            (
                UpdateNode("strict", "root", "Strict", "E", strict, CostEstimate(1.0), 1),
                UpdateNode("surrogate", "surrogate", "Surrogate", "E", surrogate, CostEstimate(1.0), 1),
            ),
            (),
            "strict",
        )
        state, participants = _state_and_participants()
        coordinator = TransactionalCoordinator(
            graph,
            lambda proposal: _apply(state, proposal),
            lambda batch: CanaryVerdict(
                True,
                "surrogate improved overall",
                0.0,
                10.0,
                proposal_objectives={
                    "strict": ObjectiveGateEvidence(3.0, 2.0),
                },
            ),
            participants=participants,
        )
        strict_proposal = replace(
            _packet(
                node_id="strict",
                proposal_id="strict",
                update=UpdateKind.COORDINATE,
            ),
            dependency_versions={"strict": 0},
        )
        surrogate_proposal = replace(
            _packet(
                node_id="surrogate",
                proposal_id="surrogate",
                update=UpdateKind.COORDINATE,
            ),
            dependency_versions={"surrogate": 0},
            objective_kind=ObjectiveKind.USER_SURROGATE,
            surrogate_disclosed=True,
        )
        receipt = coordinator.commit(ProposalBatch("mixed", (strict_proposal, surrogate_proposal)))
        assert receipt.status is CommitStatus.ROLLED_BACK
        assert receipt.reason == "strict-objective-regression:strict"
        np.testing.assert_array_equal(state["parameters"], [0.0, 0.0])


def test_independent_siblings_commit_one_global_version_and_one_parent_invalidation():
    mutable = _mutable_contract(update=UpdateKind.COORDINATE)
    frozen = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.FROZEN,
        merge_law=MergeLaw.REPLICATED,
        writes=frozenset(),
    )
    nodes = (
        UpdateNode("a", "root -> a", "A", "AE", mutable, CostEstimate(1.0), 1),
        UpdateNode("b", "root -> b", "B", "BE", mutable, CostEstimate(1.0), 1),
        UpdateNode("root", "root", "Root", None, frozen, CostEstimate(), 0),
    )
    graph = UpdateGraph(nodes, (DependencyEdge("a", "root"), DependencyEdge("b", "root")), "root")
    state, participants = _state_and_participants()

    def apply_sibling(proposal):
        state["parameters"] += proposal.payload["delta"]
        state["optimizer"]["step"] += 1
        state["rng"]["counter"] += 1

    coordinator = TransactionalCoordinator(
        graph,
        apply_sibling,
        lambda batch: CanaryVerdict(
            True,
            "joint canary passed",
            0.0,
            1.0,
            proposal_objectives={
                "p1": ObjectiveGateEvidence(0.0, 0.5),
                "p2": ObjectiveGateEvidence(0.0, 0.5),
            },
        ),
        participants=participants,
    )
    left = replace(_packet(node_id="a", update=UpdateKind.COORDINATE), dependency_versions={"a": 0})
    right = replace(_packet(node_id="b", proposal_id="p2", update=UpdateKind.COORDINATE), dependency_versions={"b": 0})
    receipt = coordinator.commit(ProposalBatch("siblings", (left, right)))

    assert receipt.status is CommitStatus.ACCEPTED
    assert receipt.invalidated_nodes == ("a", "b", "root")
    assert coordinator.versions.model_version == 1
    assert coordinator.versions.node_versions == {"a": 1, "b": 1, "root": 1}
