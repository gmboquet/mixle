"""Island proposal admission, merge, and outer commit tests."""

import copy
import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    CanaryVerdict,
    ConsistencyRequirement,
    CostEstimate,
    HierarchicalProposalCoordinator,
    MergeLaw,
    ObjectiveKind,
    ProposalPacket,
    RuntimeVersions,
    StalenessPolicy,
    StateSemantics,
    TransactionalCoordinator,
    TransactionParticipant,
    UpdateContract,
    UpdateGraph,
    UpdateKind,
    UpdateNode,
    compile_update_graph,
    payload_fingerprint,
)
from mixle.stats import GaussianDistribution, GaussianEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _exact_proposal(proposal_id, payload, *, base=0, dependency=0, observations=1.0):
    return ProposalPacket(
        proposal_id,
        "run",
        "model",
        "n0000",
        proposal_id.replace("proposal", "island"),
        base,
        {"n0000": dependency},
        UpdateKind.EXACT_CLOSED_FORM,
        ObjectiveKind.MLE,
        payload,
        observations=observations,
    )


def test_exact_shard_statistics_merge_then_commit_once():
    graph = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
    state = {"statistics": None}
    coordinator = TransactionalCoordinator(
        graph,
        lambda proposal: state.__setitem__("statistics", proposal.payload),
        lambda batch: CanaryVerdict(True, "global probe improved", 0.0, 1.0),
    )
    hierarchical = HierarchicalProposalCoordinator(coordinator)
    left = _exact_proposal("proposal-left", {"sum": np.array([1.0, 2.0]), "count": 2}, observations=2)
    right = _exact_proposal("proposal-right", {"sum": np.array([3.0, 4.0]), "count": 5}, observations=5)

    receipt = hierarchical.submit("round-0", (right, left))
    assert receipt.commit is not None and receipt.commit.accepted
    assert len(receipt.merged_proposals) == 1
    merged_id = next(iter(receipt.merged_proposals))
    assert receipt.merged_proposals[merged_id] == ("proposal-left", "proposal-right")
    np.testing.assert_array_equal(state["statistics"]["sum"], [4.0, 6.0])
    assert state["statistics"]["count"] == 7
    assert coordinator.versions.model_version == 1
    json.dumps(receipt.as_dict(), allow_nan=False)


def test_stale_exact_statistics_are_rejected_before_merge():
    graph = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
    coordinator = TransactionalCoordinator(
        graph,
        lambda proposal: pytest.fail("stale exact proposal was applied"),
        lambda batch: CanaryVerdict(True, "unreachable", 0.0, 1.0),
        versions=RuntimeVersions(1, {"n0000": 1}),
    )
    receipt = HierarchicalProposalCoordinator(
        coordinator,
        default_staleness_policy=StalenessPolicy(max_model_lag=2, max_node_lag=2),
    ).submit("round", (_exact_proposal("proposal", {"sum": 1.0}, base=0, dependency=0),))
    assert receipt.commit is None
    assert receipt.rejected == {"proposal": "exact-update-requires-current-version"}


def _bounded_graph():
    contract = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.FIRST_ORDER,
        merge_law=MergeLaw.NON_MERGEABLE,
        state_semantics=frozenset({StateSemantics.MUTABLE_PARAMETERS}),
        consistency=ConsistencyRequirement.BOUNDED_STALE,
        exact=False,
    )
    return UpdateGraph(
        (UpdateNode("node", "root", "Neural", "NeuralEstimator", contract, CostEstimate(1.0), 2),),
        (),
        "node",
    )


def test_bounded_stale_neural_delta_is_shrunk_rebased_and_committed():
    graph = _bounded_graph()
    state = {"parameters": np.zeros(2)}
    participant = TransactionParticipant(
        "parameters",
        frozenset({StateSemantics.MUTABLE_PARAMETERS}),
        lambda: state["parameters"].copy(),
        lambda value: state.__setitem__("parameters", value.copy()),
        lambda: payload_fingerprint(state["parameters"]),
    )
    coordinator = TransactionalCoordinator(
        graph,
        lambda proposal: state.__setitem__("parameters", state["parameters"] + proposal.payload["delta"]),
        lambda batch: CanaryVerdict(True, "probe improved", 0.0, 0.1),
        participants=(participant,),
        versions=RuntimeVersions(2, {"node": 2}),
    )
    proposal = ProposalPacket(
        "stale",
        "run",
        "model",
        "node",
        "remote-island",
        1,
        {"node": 1},
        UpdateKind.FIRST_ORDER,
        ObjectiveKind.MLE,
        {"delta": np.array([2.0, -2.0])},
    )
    hierarchical = HierarchicalProposalCoordinator(
        coordinator,
        default_staleness_policy=StalenessPolicy(max_model_lag=1, max_node_lag=1, shrink_decay=1.0),
    )
    receipt = hierarchical.submit("round", (proposal,))

    assert receipt.commit is not None and receipt.commit.accepted
    np.testing.assert_allclose(state["parameters"], np.array([2.0, -2.0]) * np.exp(-2.0))
    assert coordinator.versions.model_version == 3


def test_nonmergeable_same_node_proposals_are_rejected_without_apply():
    graph = _bounded_graph()
    state = {"parameters": np.zeros(2)}
    participant = TransactionParticipant(
        "parameters",
        frozenset({StateSemantics.MUTABLE_PARAMETERS}),
        lambda: state["parameters"].copy(),
        lambda value: state.__setitem__("parameters", copy.deepcopy(value)),
        lambda: payload_fingerprint(state["parameters"]),
    )
    coordinator = TransactionalCoordinator(
        graph,
        lambda proposal: pytest.fail("nonmergeable proposals were applied"),
        lambda batch: CanaryVerdict(True, "unreachable", 0.0, 1.0),
        participants=(participant,),
    )

    def proposal(proposal_id):
        return ProposalPacket(
            proposal_id,
            "run",
            "model",
            "node",
            proposal_id,
            0,
            {"node": 0},
            UpdateKind.FIRST_ORDER,
            ObjectiveKind.MLE,
            {"delta": np.ones(2)},
        )

    receipt = HierarchicalProposalCoordinator(coordinator).submit("round", (proposal("a"), proposal("b")))
    assert receipt.commit is None
    assert set(receipt.rejected) == {"a", "b"}
    assert all(reason.startswith("merge-failed") for reason in receipt.rejected.values())
