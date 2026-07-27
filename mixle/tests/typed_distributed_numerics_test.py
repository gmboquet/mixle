"""Distributed plans separate model exactness from observed collective numerics."""

import pytest

from mixle.experimental.typed_runtime import (
    ArtifactKind,
    CollectiveNumericsEvidence,
    CostEstimate,
    MergeLaw,
    ObjectiveKind,
    UpdateContract,
    UpdateKind,
    plan_distributed_updates,
)
from mixle.experimental.typed_runtime.graph import UpdateGraph, UpdateNode
from mixle.utils.parallel.training_contracts import (
    CollectiveKind,
    ParallelAxis,
    ParallelPlan,
)

pytestmark = [pytest.mark.experimental]


def _graph():
    contract = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.EXACT_CLOSED_FORM,
        merge_law=MergeLaw.ADDITIVE,
        writes=frozenset({ArtifactKind.SUFFICIENT_STATISTICS}),
        exact=True,
        declared_by="test",
    )
    return UpdateGraph(
        nodes=(UpdateNode("node", "root", "Model", "Estimator", contract, CostEstimate(), 10),),
        edges=(),
        root_node="node",
    )


def test_floating_collective_does_not_inherit_model_contract_exactness():
    update = plan_distributed_updates(_graph(), ParallelPlan(dp_replicate=2))[0]
    assert update.collective is CollectiveKind.ALL_REDUCE
    assert update.contract_exact
    assert not update.exact
    assert update.determinism_observed is None
    assert "not guaranteed" in update.notes[-1]


def test_observed_determinism_and_error_are_bound_to_collective_identity():
    evidence = CollectiveNumericsEvidence(
        "evidence-1",
        "node",
        CollectiveKind.ALL_REDUCE,
        (ParallelAxis.DP_REPLICATE,),
        "gloo",
        "float64",
        True,
        1.0e-12,
        2.0e-13,
        20,
        "sha256:ordering",
    )
    update = plan_distributed_updates(
        _graph(),
        ParallelPlan(dp_replicate=2),
        numerics_evidence={"node": evidence},
    )[0]

    assert not update.exact
    assert update.determinism_observed is True
    assert update.maximum_absolute_error == pytest.approx(1.0e-12)
    assert update.maximum_relative_error == pytest.approx(2.0e-13)
    assert update.numerics_evidence_id == "evidence-1"
    assert update.numerics_sample_count == 20


def test_local_exact_update_remains_exact_without_a_collective():
    update = plan_distributed_updates(_graph(), ParallelPlan())[0]
    assert update.collective is CollectiveKind.NONE
    assert update.contract_exact
    assert update.exact


def test_mismatched_or_unused_collective_evidence_is_rejected():
    evidence = CollectiveNumericsEvidence(
        "evidence-1",
        "other",
        CollectiveKind.ALL_REDUCE,
        (ParallelAxis.DP_REPLICATE,),
        "gloo",
        "float64",
        True,
        0.0,
        0.0,
        1,
        "sha256:ordering",
    )
    with pytest.raises(ValueError, match="did not match"):
        plan_distributed_updates(
            _graph(),
            ParallelPlan(dp_replicate=2),
            numerics_evidence={"other": evidence},
        )
