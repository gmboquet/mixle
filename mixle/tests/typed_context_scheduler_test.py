"""Value-of-information ranking, budget, and stopping tests."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ContextAction,
    ContextActionKind,
    ContextActionReceipt,
    ContextBudget,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    ContextSchedulerConfig,
    ValueOfInformationScheduler,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _action(action_id, gain, error, *, latency=0.1, tokens=0, tools=0, inputs=()):
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
    )


def test_lower_confidence_value_not_raw_mean_selects_action():
    graph = ContextGraph()
    scheduler = ValueOfInformationScheduler(
        config=ContextSchedulerConfig(confidence_z=1.0, latency_cost=1.0, token_cost=0.0)
    )
    uncertain = _action("uncertain", 1.0, 0.6)
    reliable = _action("reliable", 0.6, 0.0)
    decision = scheduler.choose((uncertain, reliable), graph)

    assert decision.selected is reliable
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
    allowed = _action("small", 1.0, 0.0, tokens=80, tools=1)
    decision = scheduler.choose((too_many_tokens, allowed), graph)
    assert decision.selected is allowed
    assert decision.inadmissible == {"large": "token-budget"}

    receipt = ContextActionReceipt(allowed, 0, 1, (), (), 0.8, 90, 1, 0.5, 0.7, "done")
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
