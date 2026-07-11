"""Behavioral tests for deterministic typed-node scheduling."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    CostEstimate,
    DependencyEdge,
    GainEvidence,
    GainPerCostScheduler,
    MergeLaw,
    ObjectiveKind,
    SchedulerConfig,
    UpdateContract,
    UpdateGraph,
    UpdateKind,
    UpdateNode,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _contract(
    *,
    objective: ObjectiveKind = ObjectiveKind.MLE,
    update: UpdateKind = UpdateKind.COORDINATE,
    compatible: bool = True,
) -> UpdateContract:
    writes = (
        frozenset() if update is UpdateKind.FROZEN else UpdateContract.__dataclass_fields__["writes"].default_factory()
    )
    return UpdateContract(
        objective_kind=objective,
        update_kind=update,
        merge_law=MergeLaw.REPLICATED if update is UpdateKind.FROZEN else MergeLaw.ADDITIVE,
        writes=writes,
        outer_objective_compatible=compatible,
        exact=update is not UpdateKind.UNKNOWN and compatible,
        declared_by="test",
    )


def _node(
    node_id: str,
    cost: float,
    *,
    objective: ObjectiveKind = ObjectiveKind.MLE,
    update: UpdateKind = UpdateKind.COORDINATE,
    compatible: bool = True,
) -> UpdateNode:
    return UpdateNode(
        node_id=node_id,
        path="root -> %s" % node_id,
        model_type="Fixture",
        estimator_type="FixtureEstimator",
        contract=_contract(objective=objective, update=update, compatible=compatible),
        cost=CostEstimate(compute_units=cost),
        parameter_count=1,
    )


def _two_leaf_graph(*, root_cost: float = 1.0, edge_from_a: bool = False) -> UpdateGraph:
    nodes = (
        _node("a", 1.0),
        _node("b", 1.0),
        _node("root", root_cost, update=UpdateKind.FROZEN),
    )
    edges = (DependencyEdge("a", "root"),) if edge_from_a else ()
    return UpdateGraph(nodes, edges, "root")


def _evidence(node_id: str, gain: float, *, error: float = 0.0, samples: int = 4) -> GainEvidence:
    return GainEvidence(node_id, ObjectiveKind.MLE, gain, standard_error=error, sample_count=samples)


class GainPerCostTest:
    def test_lower_confidence_bound_not_raw_mean_controls_ranking(self):
        graph = _two_leaf_graph()
        scheduler = GainPerCostScheduler(
            SchedulerConfig(budget_fraction=0.5, confidence_z=2.0, lambda_invalidation=0.0)
        )
        receipt = scheduler.schedule(
            graph,
            {
                "a": _evidence("a", 10.0, error=6.0),
                "b": _evidence("b", 2.0, error=0.1),
            },
        )

        assert receipt.selected_nodes == ("b",)
        assert receipt.lower_confidence_bounds == {"a": pytest.approx(-2.0), "b": pytest.approx(1.8)}

    def test_measured_invalidation_changes_the_preferred_cheap_block(self):
        graph = _two_leaf_graph(root_cost=100.0, edge_from_a=True)
        scheduler = GainPerCostScheduler(
            SchedulerConfig(budget_fraction=0.5, confidence_z=0.0, lambda_invalidation=1.0)
        )
        receipt = scheduler.schedule(graph, {"a": _evidence("a", 10.0), "b": _evidence("b", 2.0)})

        assert receipt.effective_costs == {"a": pytest.approx(101.0), "b": pytest.approx(1.0)}
        assert receipt.selected_nodes == ("b",)

    def test_soft_budget_selects_at_least_one_node(self):
        graph = _two_leaf_graph()
        scheduler = GainPerCostScheduler(
            SchedulerConfig(budget_fraction=0.0, confidence_z=0.0, lambda_invalidation=0.0)
        )
        receipt = scheduler.schedule(graph, {"a": _evidence("a", 2.0), "b": _evidence("b", 1.0)})
        assert receipt.selected_nodes == ("a",)
        assert receipt.budget == 0.0
        assert receipt.budget_overrun == pytest.approx(1.0)


class ObjectiveCompatibilityTest:
    def test_surrogate_gain_is_skipped_until_explicitly_normalized(self):
        graph = _two_leaf_graph()
        config = SchedulerConfig(budget_fraction=0.5, confidence_z=0.0, lambda_invalidation=0.0)
        unnormalized = GainEvidence("a", ObjectiveKind.USER_SURROGATE, 100.0, sample_count=5)
        baseline = _evidence("b", 1.0)

        first = GainPerCostScheduler(config).schedule(graph, {"a": unnormalized, "b": baseline})
        assert first.selected_nodes == ("b",)
        assert first.skipped["a"] == "incompatible-objective:user_surrogate"

        normalized = GainEvidence(
            "a",
            ObjectiveKind.USER_SURROGATE,
            100.0,
            sample_count=5,
            normalized_to=ObjectiveKind.MLE,
        )
        second = GainPerCostScheduler(config).schedule(graph, {"a": normalized, "b": baseline})
        assert second.selected_nodes == ("a",)


class FairnessAndReplayTest:
    def test_starvation_bound_forces_low_gain_node(self):
        graph = _two_leaf_graph()
        scheduler = GainPerCostScheduler(
            SchedulerConfig(
                budget_fraction=0.5,
                confidence_z=0.0,
                lambda_invalidation=0.0,
                max_skip_rounds=1,
            )
        )
        evidence = {"a": _evidence("a", 10.0), "b": _evidence("b", 1.0)}

        first = scheduler.schedule(graph, evidence)
        second = scheduler.schedule(graph, evidence)

        assert first.selected_nodes == ("a",)
        assert second.selected_nodes == ("b",)
        assert second.forced_starvation == ("b",)
        assert scheduler.states["b"].last_selected_round == 1

    def test_bootstrap_and_receipt_are_json_safe_and_deterministic(self):
        graph = _two_leaf_graph()
        config = SchedulerConfig(budget_fraction=0.5, lambda_invalidation=0.0)
        left = GainPerCostScheduler(config).schedule(graph)
        right = GainPerCostScheduler(config).schedule(graph)

        assert left == right
        assert left.bootstrap_nodes == ("a", "b")
        assert left.selected_nodes == ("a",)
        json.dumps(left.as_dict(), allow_nan=False)

    def test_unknown_evidence_and_round_rewind_fail_before_state_changes(self):
        graph = _two_leaf_graph()
        scheduler = GainPerCostScheduler()
        with pytest.raises(KeyError, match="unknown nodes"):
            scheduler.schedule(graph, {"missing": _evidence("missing", 1.0)})
        assert scheduler.states == {}

        scheduler.schedule(graph)
        with pytest.raises(ValueError, match="backwards"):
            scheduler.schedule(graph, round_index=0)
