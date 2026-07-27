"""Value-of-information ranking, budget, and stopping tests."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ContextAction,
    ContextActionKind,
    ContextActionLimits,
    ContextActionReceipt,
    ContextBudget,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    ContextSchedulerConfig,
    ValueOfInformationScheduler,
)

pytestmark = [pytest.mark.experimental]


def _action(
    action_id,
    gain,
    error,
    *,
    latency=0.1,
    tokens=0,
    tools=0,
    inputs=(),
    maximum_tokens=None,
):
    return ContextAction(
        action_id,
        ContextActionKind.RETRIEVE,
        input_nodes=inputs,
        query=action_id,
        expected_information_gain=gain,
        gain_standard_error=error,
        gain_sample_count=4,
        expected_latency_seconds=latency,
        expected_tokens=tokens,
        expected_tool_calls=tools,
        resource_limits=ContextActionLimits(
            1.0,
            tokens if maximum_tokens is None else maximum_tokens,
            1.0,
            tools,
        ),
    )


def test_lower_confidence_value_not_raw_mean_selects_action():
    graph = ContextGraph()
    scheduler = ValueOfInformationScheduler(
        config=ContextSchedulerConfig(confidence_z=1.0, latency_cost=1.0, token_cost=0.0)
    )
    uncertain = _action("uncertain", 1.0, 0.6)
    reliable = _action("reliable", 0.6, 0.0)
    decision = scheduler.choose((uncertain, reliable), graph)

    assert decision.selected.action_id == reliable.action_id
    assert decision.selected.expected_graph_version == graph.version
    assert decision.lower_confidence_gains == {"uncertain": pytest.approx(0.4), "reliable": 0.6}
    assert decision.net_values["reliable"] > decision.net_values["uncertain"]
    json.dumps(decision.as_dict(), allow_nan=False)


def test_scheduler_stops_when_every_action_has_value_below_cost():
    scheduler = ValueOfInformationScheduler(
        config=ContextSchedulerConfig(confidence_z=0.0, latency_cost=1.0, token_cost=0.0)
    )
    decision = scheduler.choose((_action("bad", 0.05, 0.0, latency=0.1),), ContextGraph())
    assert decision.stopped
    assert decision.stopping_reason == "expected-value-below-cost"
    assert decision.selected.kind is ContextActionKind.STOP


def test_expected_budget_filters_actions_and_actual_receipt_debits_once():
    graph = ContextGraph()
    scheduler = ValueOfInformationScheduler(
        ContextBudget(latency_seconds=1.0, materialized_tokens=100, monetary_cost=1.0, tool_calls=1, maximum_actions=1),
        ContextSchedulerConfig(confidence_z=0.0, latency_cost=0.0, token_cost=0.0, tool_call_cost=0.0),
    )
    too_many_tokens = _action("large", 10.0, 0.0, tokens=101)
    allowed = _action("small", 1.0, 0.0, tokens=80, tools=1, maximum_tokens=100)
    decision = scheduler.choose((too_many_tokens, allowed), graph)
    assert decision.selected.action_id == allowed.action_id
    assert decision.selected.expected_graph_version == graph.version
    assert decision.inadmissible == {"large": "token-budget"}

    receipt = ContextActionReceipt(
        decision.selected,
        0,
        1,
        (),
        (),
        0.8,
        90,
        1,
        0.5,
        0.7,
        "done",
    )
    scheduler.record(receipt)
    stopped = scheduler.choose((_action("next", 100.0, 0.0),), graph)
    assert stopped.stopped
    assert stopped.inadmissible == {"next": "action-budget"}
    assert scheduler.as_dict()["tokens_spent"] == 90
    with pytest.raises(ValueError, match="already recorded"):
        scheduler.record(receipt)


def test_missing_input_node_is_inadmissible_not_silently_ignored():
    graph = ContextGraph()
    graph.add_node(ContextNode("known", ContextNodeKind.MEMORY, "Known", 1))
    action = _action("expand", 1.0, 0.0, inputs=("missing",))
    decision = ValueOfInformationScheduler().choose((action,), graph)
    assert decision.stopped
    assert decision.inadmissible == {"expand": "missing-input:missing"}


def test_stale_prebound_action_is_inadmissible():
    graph = ContextGraph()
    graph.add_node(ContextNode("new", ContextNodeKind.MEMORY, "New", 1))
    stale = ContextAction(
        "stale",
        ContextActionKind.RETRIEVE,
        expected_graph_version=0,
        expected_information_gain=1.0,
        gain_sample_count=1,
        resource_limits=ContextActionLimits(1.0, 0, 0.0, 0),
    )
    decision = ValueOfInformationScheduler().choose((stale,), graph)
    assert decision.stopped
    assert decision.inadmissible == {"stale": "stale-graph-version"}


def test_missing_gain_evidence_and_missing_resource_limits_are_inadmissible():
    no_samples = ContextAction(
        "no-samples",
        ContextActionKind.RETRIEVE,
        expected_information_gain=1.0,
        resource_limits=ContextActionLimits(1.0, 0, 0.0, 0),
    )
    no_limits = ContextAction(
        "no-limits",
        ContextActionKind.RETRIEVE,
        expected_information_gain=1.0,
        gain_sample_count=1,
    )
    decision = ValueOfInformationScheduler().choose((no_samples, no_limits), ContextGraph())
    assert decision.stopped
    assert decision.inadmissible == {
        "no-limits": "missing-resource-limits",
        "no-samples": "missing-gain-evidence",
    }


def test_reservations_use_worst_case_limits_and_actual_overrun_closes_scheduler():
    graph = ContextGraph()
    scheduler = ValueOfInformationScheduler(
        ContextBudget(
            latency_seconds=1.0,
            materialized_tokens=10,
            monetary_cost=1.0,
            tool_calls=1,
            maximum_actions=2,
        ),
        ContextSchedulerConfig(confidence_z=0.0, latency_cost=0.0, token_cost=0.0),
    )
    selected = scheduler.choose((_action("selected", 1.0, 0.0, tokens=10, tools=1),), graph)
    blocked = scheduler.choose((_action("concurrent", 2.0, 0.0),), graph)

    assert selected.selected.action_id == "selected"
    assert blocked.stopped
    assert blocked.inadmissible == {"concurrent": "latency-budget"}
    receipt = ContextActionReceipt(
        selected.selected,
        0,
        0,
        (),
        (),
        1.1,
        11,
        1,
        0.0,
        0.0,
        "error:resource-limit",
        rolled_back=True,
    )
    scheduler.record(receipt)
    assert scheduler.budget_breached
    stopped = scheduler.choose((_action("after-breach", 3.0, 0.0),), graph)
    assert stopped.inadmissible == {"after-breach": "actual-budget-breached"}


def test_context_budget_rejects_nan_and_negative_expected_money():
    with pytest.raises(ValueError, match="not NaN"):
        ContextBudget(latency_seconds=float("nan"))
    with pytest.raises(ValueError, match="expected costs"):
        ContextAction(
            "negative",
            ContextActionKind.RETRIEVE,
            expected_monetary_cost=-1.0,
        )
