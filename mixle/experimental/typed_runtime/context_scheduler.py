"""Budgeted value-of-information scheduling for context construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from mixle.experimental.typed_runtime.context_ir import (
    ContextAction,
    ContextActionKind,
    ContextActionReceipt,
    ContextGraph,
)


@dataclass(frozen=True)
class ContextBudget:
    """Hard cumulative context-construction resource limits."""

    latency_seconds: float = math.inf
    materialized_tokens: int = 2**63 - 1
    monetary_cost: float = math.inf
    tool_calls: int = 2**31 - 1
    maximum_actions: int = 2**31 - 1

    def __post_init__(self) -> None:
        if any(
            math.isnan(value) or value < 0.0
            for value in (self.latency_seconds, self.monetary_cost)
        ):
            raise ValueError("context latency and monetary budgets must be non-negative and not NaN.")
        counts = (self.materialized_tokens, self.tool_calls, self.maximum_actions)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("context count budgets must be non-negative integers.")


@dataclass(frozen=True)
class ContextSchedulerConfig:
    """Confidence and resource exchange rates for value-of-information."""

    confidence_z: float = 1.645
    latency_cost: float = 1.0
    token_cost: float = 1.0e-5
    monetary_cost: float = 1.0
    tool_call_cost: float = 0.01
    minimum_net_value: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.confidence_z,
            self.latency_cost,
            self.token_cost,
            self.monetary_cost,
            self.tool_call_cost,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("context scheduler confidence/cost weights must be finite and non-negative.")
        if not math.isfinite(self.minimum_net_value):
            raise ValueError("minimum_net_value must be finite.")


@dataclass(frozen=True)
class ContextScheduleDecision:
    """One selected action or explicit stopping action with ranked evidence."""

    selected: ContextAction
    ranked_action_ids: tuple[str, ...]
    lower_confidence_gains: dict[str, float]
    expected_costs: dict[str, float]
    net_values: dict[str, float]
    inadmissible: dict[str, str]
    stopping_reason: str | None

    @property
    def stopped(self) -> bool:
        """Whether the scheduler selected its explicit STOP action."""

        return self.selected.kind is ContextActionKind.STOP

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible decision receipt."""

        return {
            "selected": self.selected.as_dict(),
            "ranked_action_ids": list(self.ranked_action_ids),
            "lower_confidence_gains": dict(self.lower_confidence_gains),
            "expected_costs": dict(self.expected_costs),
            "net_values": dict(self.net_values),
            "inadmissible": dict(self.inadmissible),
            "stopping_reason": self.stopping_reason,
            "stopped": self.stopped,
        }


class ValueOfInformationScheduler:
    """Stateful context-action scheduler debited by actual execution receipts."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        config: ContextSchedulerConfig | None = None,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.config = config or ContextSchedulerConfig()
        self.latency_spent = 0.0
        self.tokens_spent = 0
        self.money_spent = 0.0
        self.tool_calls_spent = 0
        self.actions_completed = 0
        self._completed_action_ids: set[str] = set()
        self._reservations: dict[str, ContextAction] = {}
        self._reserved_latency = 0.0
        self._reserved_tokens = 0
        self._reserved_money = 0.0
        self._reserved_tool_calls = 0
        self.budget_breached = False

    def _expected_cost(self, action: ContextAction) -> float:
        return (
            self.config.latency_cost * action.expected_latency_seconds
            + self.config.token_cost * action.expected_tokens
            + self.config.monetary_cost * action.expected_monetary_cost
            + self.config.tool_call_cost * action.expected_tool_calls
        )

    def _admissible(self, action: ContextAction) -> str | None:
        if self.budget_breached:
            return "actual-budget-breached"
        if action.action_id in self._completed_action_ids:
            return "already-completed"
        if action.action_id in self._reservations:
            return "already-reserved"
        limits = action.resource_limits
        if limits is None:
            return "missing-resource-limits"
        if self.actions_completed + len(self._reservations) + 1 > self.budget.maximum_actions:
            return "action-budget"
        if (
            self.latency_spent + self._reserved_latency + limits.latency_seconds
            > self.budget.latency_seconds
        ):
            return "latency-budget"
        if (
            self.tokens_spent + self._reserved_tokens + limits.materialized_tokens
            > self.budget.materialized_tokens
        ):
            return "token-budget"
        if (
            self.money_spent + self._reserved_money + limits.monetary_cost
            > self.budget.monetary_cost
        ):
            return "monetary-budget"
        if (
            self.tool_calls_spent + self._reserved_tool_calls + limits.tool_calls
            > self.budget.tool_calls
        ):
            return "tool-call-budget"
        return None

    def _reserve(self, action: ContextAction) -> None:
        limits = action.resource_limits
        if limits is None:
            raise ValueError("cannot reserve an action without resource_limits.")
        self._reservations[action.action_id] = action
        self._reserved_latency += limits.latency_seconds
        self._reserved_tokens += limits.materialized_tokens
        self._reserved_money += limits.monetary_cost
        self._reserved_tool_calls += limits.tool_calls

    def stop_decision(self, graph: ContextGraph, reason: str) -> ContextScheduleDecision:
        """Create an explicit zero-work scheduling decision for termination."""

        if not reason:
            raise ValueError("context stopping reason must be non-empty.")
        stop = ContextAction(
            action_id="stop:%s:v%d:a%d" % (reason, graph.version, self.actions_completed),
            kind=ContextActionKind.STOP,
            expected_graph_version=graph.version,
        )
        return ContextScheduleDecision(stop, (), {}, {}, {}, {}, reason)

    def choose(self, actions: tuple[ContextAction, ...], graph: ContextGraph) -> ContextScheduleDecision:
        """Choose the highest positive net VOI action or stop explicitly."""

        if len({action.action_id for action in actions}) != len(actions):
            raise ValueError("candidate context action ids must be unique.")
        lower: dict[str, float] = {}
        costs: dict[str, float] = {}
        net: dict[str, float] = {}
        inadmissible: dict[str, str] = {}
        for action in actions:
            if (
                action.expected_graph_version is not None
                and action.expected_graph_version != graph.version
            ):
                inadmissible[action.action_id] = "stale-graph-version"
                continue
            missing = sorted(set(action.input_nodes) - set(graph.nodes))
            if missing:
                inadmissible[action.action_id] = "missing-input:%s" % ",".join(missing)
                continue
            reason = self._admissible(action)
            if reason is not None:
                inadmissible[action.action_id] = reason
                continue
            if action.gain_sample_count <= 0:
                inadmissible[action.action_id] = "missing-gain-evidence"
                continue
            lower[action.action_id] = action.expected_information_gain - (
                self.config.confidence_z * action.gain_standard_error
            )
            costs[action.action_id] = self._expected_cost(action)
            net[action.action_id] = lower[action.action_id] - costs[action.action_id]

        ranked = tuple(
            sorted(
                net,
                key=lambda action_id: (
                    -net[action_id],
                    -(lower[action_id] / max(costs[action_id], 1.0e-12)),
                    action_id,
                ),
            )
        )
        selected = next(
            (action for action_id in ranked for action in actions if action.action_id == action_id),
            None,
        )
        stopping_reason = None
        if selected is None:
            stopping_reason = "no-admissible-context-action"
        elif net[selected.action_id] <= self.config.minimum_net_value:
            stopping_reason = "expected-value-below-cost"
        if stopping_reason is not None:
            decision = self.stop_decision(graph, stopping_reason)
            return replace(
                decision,
                ranked_action_ids=ranked,
                lower_confidence_gains=lower,
                expected_costs=costs,
                net_values=net,
                inadmissible=inadmissible,
            )
        if selected.expected_graph_version is None:
            selected = replace(selected, expected_graph_version=graph.version)
        self._reserve(selected)
        return ContextScheduleDecision(selected, ranked, lower, costs, net, inadmissible, None)

    def record(self, receipt: ContextActionReceipt) -> None:
        """Debit actual work once; rejected/rolled-back work still costs resources."""

        action_id = receipt.action.action_id
        if action_id in self._completed_action_ids:
            raise ValueError("context action receipt was already recorded: %s" % action_id)
        reserved = self._reservations.pop(action_id, None)
        if reserved is None:
            raise ValueError("context action receipt has no matching resource reservation: %s" % action_id)
        if reserved != receipt.action:
            self._reservations[action_id] = reserved
            raise ValueError("context action receipt does not match its resource reservation.")
        limits = reserved.resource_limits
        self._reserved_latency -= limits.latency_seconds
        self._reserved_tokens -= limits.materialized_tokens
        self._reserved_money -= limits.monetary_cost
        self._reserved_tool_calls -= limits.tool_calls
        self._completed_action_ids.add(action_id)
        self.latency_spent += receipt.latency_seconds
        self.tokens_spent += receipt.materialized_tokens
        self.money_spent += receipt.monetary_cost
        self.tool_calls_spent += receipt.tool_calls
        self.actions_completed += 1
        self.budget_breached = (
            self.latency_spent > self.budget.latency_seconds
            or self.tokens_spent > self.budget.materialized_tokens
            or self.money_spent > self.budget.monetary_cost
            or self.tool_calls_spent > self.budget.tool_calls
            or receipt.latency_seconds > limits.latency_seconds
            or receipt.materialized_tokens > limits.materialized_tokens
            or receipt.monetary_cost > limits.monetary_cost
            or receipt.tool_calls > limits.tool_calls
        )

    def as_dict(self) -> dict[str, Any]:
        """Return cumulative actual context-construction spend."""

        return {
            "latency_spent": self.latency_spent,
            "tokens_spent": self.tokens_spent,
            "money_spent": self.money_spent,
            "tool_calls_spent": self.tool_calls_spent,
            "actions_completed": self.actions_completed,
            "completed_action_ids": sorted(self._completed_action_ids),
            "reserved_action_ids": sorted(self._reservations),
            "reserved_latency_seconds": self._reserved_latency,
            "reserved_tokens": self._reserved_tokens,
            "reserved_monetary_cost": self._reserved_money,
            "reserved_tool_calls": self._reserved_tool_calls,
            "budget_breached": self.budget_breached,
        }


__all__ = [
    "ContextBudget",
    "ContextScheduleDecision",
    "ContextSchedulerConfig",
    "ValueOfInformationScheduler",
]
