"""Immutable proposal, fingerprint, conflict, and merge tests."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    ObjectiveKind,
    ProposalBatch,
    ProposalPacket,
    UpdateKind,
    compile_update_graph,
    merge_same_node_proposals,
    payload_fingerprint,
    proposal_conflicts,
)
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _proposal(
    proposal_id,
    node_id,
    payload,
    *,
    update=UpdateKind.EXACT_CLOSED_FORM,
    objective=ObjectiveKind.MLE,
    shard="s0",
):
    return ProposalPacket(
        proposal_id=proposal_id,
        run_id="run-1",
        model_id="model-1",
        node_id=node_id,
        shard_id=shard,
        base_model_version=0,
        dependency_versions={node_id: 0},
        update_kind=update,
        objective_kind=objective,
        payload=payload,
        observations=5,
        predicted_gain=1.0,
        data_fingerprint="data-%s" % shard,
    )


class DeterministicPayloadTest:
    def test_mapping_order_and_array_layout_do_not_change_fingerprint(self):
        left = {"b": [2, 3], "a": np.arange(6, dtype=np.float64).reshape(2, 3)}
        right = {"a": np.asfortranarray(left["a"]), "b": [2, 3]}
        assert payload_fingerprint(left) == payload_fingerprint(right)

    def test_nonfinite_and_object_payloads_fail_before_transport(self):
        with pytest.raises(ValueError, match="non-finite"):
            payload_fingerprint(np.array([1.0, np.nan]))
        with pytest.raises(TypeError, match="unsupported deterministic"):
            payload_fingerprint(object())

    def test_packet_receipt_excludes_payload_and_checks_declared_hash(self):
        packet = _proposal("p1", "n0", {"sum": np.array([1.0, 2.0])})
        payload = packet.as_dict()
        assert "payload" not in payload
        assert payload["payload_hash"] == packet.payload_hash
        json.dumps(payload, allow_nan=False)

        with pytest.raises(ValueError, match="does not match"):
            ProposalPacket(
                proposal_id="bad",
                run_id="run-1",
                model_id="model-1",
                node_id="n0",
                shard_id="s0",
                base_model_version=0,
                dependency_versions={"n0": 0},
                update_kind=UpdateKind.EXACT_CLOSED_FORM,
                objective_kind=ObjectiveKind.MLE,
                payload=1.0,
                payload_hash="wrong",
            )

    def test_surrogate_disclosure_is_mandatory(self):
        with pytest.raises(ValueError, match="surrogate"):
            _proposal("p1", "n0", 1.0, objective=ObjectiveKind.USER_SURROGATE)


class CompositionTest:
    def test_additive_shards_merge_in_canonical_order(self):
        first = _proposal("p-z", "n0", {"sum": np.array([1.0, 2.0]), "count": 3}, shard="s1")
        second = _proposal("p-a", "n0", {"sum": np.array([4.0, 8.0]), "count": 7}, shard="s0")

        left = merge_same_node_proposals(
            [first, second],
            merged_proposal_id="merged",
            merge_law=compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
            .node("n0000")
            .contract.merge_law,
        )
        right = merge_same_node_proposals(
            [second, first],
            merged_proposal_id="merged",
            merge_law=compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
            .node("n0000")
            .contract.merge_law,
        )

        assert left.payload_hash == right.payload_hash
        np.testing.assert_array_equal(left.payload["sum"], [5.0, 10.0])
        assert left.payload["count"] == 10
        assert left.observations == 10
        assert left.predicted_gain == pytest.approx(2.0)

    def test_graph_conflicts_parent_child_but_not_siblings(self):
        model = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        )
        graph = compile_update_graph(model, MixtureEstimator([GaussianEstimator(), GaussianEstimator()]))
        children = [node for node in graph.nodes if node.node_id != graph.root_node]
        siblings = ProposalBatch(
            "siblings",
            (
                _proposal("left", children[0].node_id, 1.0),
                _proposal("right", children[1].node_id, 2.0, shard="s1"),
            ),
        )
        assert proposal_conflicts(graph, siblings) == ()

        dependent = ProposalBatch(
            "dependent",
            (
                _proposal("child", children[0].node_id, 1.0),
                _proposal(
                    "root",
                    graph.root_node,
                    2.0,
                    update=graph.node(graph.root_node).contract.update_kind,
                    shard="s1",
                ),
            ),
        )
        conflicts = proposal_conflicts(graph, dependent)
        assert len(conflicts) == 1
        assert conflicts[0].reason == "dependency-version-order"
