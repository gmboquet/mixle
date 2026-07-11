"""Deterministic gain-per-cost scheduling over a typed update graph."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from mixle.experimental.typed_runtime.contracts import ObjectiveKind, UpdateKind
from mixle.experimental.typed_runtime.graph import UpdateGraph, UpdateNode
from mixle.experimental.typed_runtime.validation import validate_update_graph


@dataclass(frozen=True)
class GainEvidence:
    """Objective-gain estimate for one update node.

    ``normalized_to`` is required when a surrogate gain has been transformed to
    the scheduler's reporting objective. Merely setting
    ``outer_objective_compatible`` on a contract does not make unlike numeric
    scales comparable.
    """

    node_id: str
    objective_kind: ObjectiveKind
    expected_gain: float
    standard_error: float = 0.0
    sample_count: int = 0
    normalized_to: ObjectiveKind | None = None
    staleness_risk: float = 0.0
    model_version: int | None = None

    def __post_init__(self) -> None:
        numeric = (self.expected_gain, self.standard_error, self.staleness_risk)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("gain evidence values must be finite.")
        if self.standard_error < 0.0 or self.staleness_risk < 0.0:
            raise ValueError("gain uncertainty and staleness risk must be non-negative.")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative.")
        if self.model_version is not None and self.model_version < 0:
            raise ValueError("model_version must be non-negative when supplied.")

    @property
    def effective_objective(self) -> ObjectiveKind:
        """Objective scale on which ``expected_gain`` is expressed."""

        return self.normalized_to or self.objective_kind

    def lower_confidence_bound(self, z_value: float) -> float:
        """Return the one-sided normal-approximation lower bound."""

        return self.expected_gain - z_value * self.standard_error

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "node_id": self.node_id,
            "objective_kind": self.objective_kind.value,
            "expected_gain": self.expected_gain,
            "standard_error": self.standard_error,
            "sample_count": self.sample_count,
            "normalized_to": self.normalized_to.value if self.normalized_to is not None else None,
            "staleness_risk": self.staleness_risk,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    """Resource weights and fairness policy for typed greedy scheduling."""

    budget_fraction: float = 0.5
    confidence_z: float = 1.645
    lambda_network: float = 0.0
    lambda_memory: float = 0.0
    lambda_stale: float = 1.0
    lambda_invalidation: float = 0.1
    minimum_gain_lcb: float = 0.0
    max_skip_rounds: int = 2
    bootstrap_unmeasured: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in [0, 1].")
        values = (
            self.confidence_z,
            self.lambda_network,
            self.lambda_memory,
            self.lambda_stale,
            self.lambda_invalidation,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("scheduler confidence and cost weights must be finite and non-negative.")
        if not math.isfinite(self.minimum_gain_lcb):
            raise ValueError("minimum_gain_lcb must be finite.")
        if self.max_skip_rounds < 0:
            raise ValueError("max_skip_rounds must be non-negative.")


@dataclass(frozen=True)
class NodeScheduleState:
    """Persistent fairness clock for one eligible node."""

    selected_count: int = 0
    skip_rounds: int = 0
    last_selected_round: int | None = None


@dataclass(frozen=True)
class ScheduleReceipt:
    """Complete, replayable record of one deterministic scheduling decision."""

    round_index: int
    target_objective: ObjectiveKind
    selected_nodes: tuple[str, ...]
    ranked_nodes: tuple[str, ...]
    eligible_nodes: tuple[str, ...]
    forced_starvation: tuple[str, ...]
    bootstrap_nodes: tuple[str, ...]
    lower_confidence_bounds: dict[str, float | None]
    effective_costs: dict[str, float]
    priorities: dict[str, float | None]
    invalidation_costs: dict[str, float]
    skipped: dict[str, str]
    budget: float
    spent: float

    @property
    def budget_overrun(self) -> float:
        """Amount by which a fairness-forced decision exceeds its soft budget."""

        return max(0.0, self.spent - self.budget)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible receipt."""

        return {
            "round_index": self.round_index,
            "target_objective": self.target_objective.value,
            "selected_nodes": list(self.selected_nodes),
            "ranked_nodes": list(self.ranked_nodes),
            "eligible_nodes": list(self.eligible_nodes),
            "forced_starvation": list(self.forced_starvation),
            "bootstrap_nodes": list(self.bootstrap_nodes),
            "lower_confidence_bounds": dict(self.lower_confidence_bounds),
            "effective_costs": dict(self.effective_costs),
            "priorities": dict(self.priorities),
            "invalidation_costs": dict(self.invalidation_costs),
            "skipped": dict(self.skipped),
            "budget": self.budget,
            "spent": self.spent,
            "budget_overrun": self.budget_overrun,
        }


def _base_cost(node: UpdateNode) -> float:
    measured_time = node.cost.wall_time_seconds
    if measured_time is not None and measured_time > 0.0:
        return measured_time
    return max(node.cost.compute_units, 1.0e-12)


class GainPerCostScheduler:
    """Stateful deterministic scheduler with bounded starvation.

    The resource budget is soft because fairness-forced nodes are always run.
    This is deliberate: a budget may delay a reachable coordinate, but cannot
    remove it from the optimization unless the caller freezes that node.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._states: dict[str, NodeScheduleState] = {}
        self._next_round = 0

    @property
    def states(self) -> dict[str, NodeScheduleState]:
        """Return a copy of the fairness clocks."""

        return dict(self._states)

    def reset(self) -> None:
        """Reset round and starvation state without changing configuration."""

        self._states.clear()
        self._next_round = 0

    def schedule(
        self,
        graph: UpdateGraph,
        evidence: dict[str, GainEvidence] | None = None,
        *,
        target_objective: ObjectiveKind | None = None,
        round_index: int | None = None,
        candidate_nodes: Collection[str] | None = None,
    ) -> ScheduleReceipt:
        """Select a budgeted set of nodes and advance their fairness clocks."""

        validate_update_graph(graph, strict=True)
        evidence = dict(evidence or {})
        graph_nodes = {node.node_id for node in graph.nodes}
        unknown_evidence = sorted(set(evidence) - graph_nodes)
        if unknown_evidence:
            raise KeyError("gain evidence refers to unknown nodes: %s" % ", ".join(unknown_evidence))
        for node_id, row in evidence.items():
            if row.node_id != node_id:
                raise ValueError("gain evidence key and node_id differ for %s." % node_id)
        candidates = graph_nodes if candidate_nodes is None else set(candidate_nodes)
        unknown_candidates = sorted(candidates - graph_nodes)
        if unknown_candidates:
            raise KeyError("candidate set refers to unknown nodes: %s" % ", ".join(unknown_candidates))

        if round_index is None:
            round_index = self._next_round
        if round_index < self._next_round:
            raise ValueError("round_index cannot move backwards.")
        self._next_round = round_index + 1
        target = target_objective or graph.node(graph.root_node).contract.objective_kind

        skipped: dict[str, str] = {}
        eligible: list[str] = []
        bootstrap: set[str] = set()
        lower_bounds: dict[str, float | None] = {}
        priorities: dict[str, float | None] = {}
        effective_costs: dict[str, float] = {}
        invalidation_costs: dict[str, float] = {}

        for node in graph.nodes:
            node_id = node.node_id
            if node_id not in candidates:
                skipped[node_id] = "not-candidate"
                continue
            if node.contract.update_kind is UpdateKind.FROZEN:
                skipped[node_id] = "frozen"
                continue
            row = evidence.get(node_id)
            if row is not None and row.effective_objective is not target:
                skipped[node_id] = "incompatible-objective:%s" % row.effective_objective.value
                continue
            if (
                row is None
                and node.contract.objective_kind is not target
                and not node.contract.outer_objective_compatible
            ):
                skipped[node_id] = "unnormalized-surrogate:%s" % node.contract.objective_kind.value
                continue

            invalidated = graph.invalidated_by(node_id, include_self=False)
            invalidation_cost = sum(_base_cost(graph.node(dependent)) for dependent in invalidated)
            base = _base_cost(node)
            stale = row.staleness_risk if row is not None else 0.0
            effective = (
                base
                + self.config.lambda_network * node.cost.communication_bytes
                + self.config.lambda_memory * node.cost.peak_memory_bytes
                + self.config.lambda_stale * stale
                + self.config.lambda_invalidation * invalidation_cost
            )
            invalidation_costs[node_id] = invalidation_cost
            effective_costs[node_id] = max(effective, 1.0e-12)
            eligible.append(node_id)

            if row is None or row.sample_count == 0:
                if self.config.bootstrap_unmeasured:
                    bootstrap.add(node_id)
                    lower_bounds[node_id] = None
                    priorities[node_id] = None
                    continue
                row = row or GainEvidence(node_id, target, 0.0)
            lower_bound = row.lower_confidence_bound(self.config.confidence_z)
            lower_bounds[node_id] = lower_bound
            priorities[node_id] = lower_bound / effective_costs[node_id]

        total_cost = sum(effective_costs.values())
        budget = self.config.budget_fraction * total_cost
        old_states = {node_id: self._states.get(node_id, NodeScheduleState()) for node_id in eligible}
        forced = {node_id for node_id, state in old_states.items() if state.skip_rounds >= self.config.max_skip_rounds}

        ranked = sorted(
            eligible,
            key=lambda node_id: (
                node_id not in forced,
                node_id not in bootstrap,
                -(priorities[node_id] if priorities[node_id] is not None else 0.0),
                graph.node(node_id).path,
                node_id,
            ),
        )
        selected: set[str] = set(forced)
        spent = sum(effective_costs[node_id] for node_id in selected)
        for node_id in ranked:
            if node_id in selected:
                continue
            lower_bound = lower_bounds[node_id]
            if node_id not in bootstrap and lower_bound is not None and lower_bound < self.config.minimum_gain_lcb:
                skipped[node_id] = "lower-confidence-bound-below-threshold"
                continue
            node_cost = effective_costs[node_id]
            if selected and spent + node_cost > budget:
                skipped[node_id] = "budget"
                continue
            selected.add(node_id)
            spent += node_cost

        topo = graph.topological_order()
        selected_order = tuple(node_id for node_id in topo if node_id in selected)
        for node_id in eligible:
            state = old_states[node_id]
            if node_id in selected:
                self._states[node_id] = NodeScheduleState(
                    selected_count=state.selected_count + 1,
                    skip_rounds=0,
                    last_selected_round=round_index,
                )
            else:
                self._states[node_id] = NodeScheduleState(
                    selected_count=state.selected_count,
                    skip_rounds=state.skip_rounds + 1,
                    last_selected_round=state.last_selected_round,
                )

        return ScheduleReceipt(
            round_index=round_index,
            target_objective=target,
            selected_nodes=selected_order,
            ranked_nodes=tuple(ranked),
            eligible_nodes=tuple(node_id for node_id in topo if node_id in eligible),
            forced_starvation=tuple(node_id for node_id in topo if node_id in forced),
            bootstrap_nodes=tuple(node_id for node_id in topo if node_id in bootstrap),
            lower_confidence_bounds=lower_bounds,
            effective_costs=effective_costs,
            priorities=priorities,
            invalidation_costs=invalidation_costs,
            skipped=skipped,
            budget=budget,
            spent=spent,
        )


__all__ = [
    "GainEvidence",
    "GainPerCostScheduler",
    "NodeScheduleState",
    "ScheduleReceipt",
    "SchedulerConfig",
]
