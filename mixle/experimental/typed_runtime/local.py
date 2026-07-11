"""First executable typed local path: budgeted component-coordinate EM."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.cache import InvalidationReceipt, VersionedArtifactCache
from mixle.experimental.typed_runtime.compiler import compile_update_graph
from mixle.experimental.typed_runtime.contracts import ObjectiveKind, UpdateKind
from mixle.experimental.typed_runtime.graph import UpdateGraph
from mixle.experimental.typed_runtime.measurement import WorkMeasurement
from mixle.experimental.typed_runtime.scheduler import (
    GainEvidence,
    GainPerCostScheduler,
    SchedulerConfig,
    ScheduleReceipt,
)
from mixle.inference.freeze_rollup import (
    FreezeRollupCache,
    _combine,
    _component_log_density_matrix,
    _log_density_from_matrix,
    _m_step,
    _resolve_payload,
)
from mixle.stats.latent.mixture import MixtureDistribution, MixtureEstimator


@dataclass
class _RunningGain:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def standard_error(self) -> float:
        if self.count < 2:
            return abs(self.mean) * 0.5
        variance = self.m2 / (self.count - 1)
        return math.sqrt(max(variance, 0.0) / self.count)


@dataclass(frozen=True)
class TypedMixtureRoundReceipt:
    """Selection, objective gate, invalidation, and work for one local round."""

    round_index: int
    schedule: ScheduleReceipt
    coordinator_nodes: tuple[str, ...]
    objective_before: float
    candidate_objective: float
    committed_objective: float
    accepted: bool
    realized_gain: float
    gain_attribution: str
    active_components: tuple[int, ...]
    invalidation: InvalidationReceipt | None
    work: WorkMeasurement

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible execution receipt."""

        return {
            "round_index": self.round_index,
            "schedule": self.schedule.as_dict(),
            "coordinator_nodes": list(self.coordinator_nodes),
            "objective_before": self.objective_before,
            "candidate_objective": self.candidate_objective,
            "committed_objective": self.committed_objective,
            "accepted": self.accepted,
            "realized_gain": self.realized_gain,
            "gain_attribution": self.gain_attribution,
            "active_components": list(self.active_components),
            "invalidation": self.invalidation.as_dict() if self.invalidation is not None else None,
            "work": self.work.as_dict(),
        }


@dataclass(frozen=True)
class TypedMixtureRun:
    """Final model and immutable evidence from a typed local mixture fit."""

    model: MixtureDistribution = field(repr=False)
    graph: UpdateGraph = field(repr=False)
    rounds: tuple[TypedMixtureRoundReceipt, ...]

    @property
    def objective_trace(self) -> tuple[float, ...]:
        """Committed observed-data objective after every round."""

        return tuple(receipt.committed_objective for receipt in self.rounds)

    @property
    def total_model_evaluations(self) -> int:
        """Total real component log-density calls across the fit."""

        return sum(receipt.work.model_evaluations for receipt in self.rounds)

    def as_dict(self) -> dict[str, Any]:
        """Return run metadata and receipts without serializing the model graph objects."""

        return {
            "graph": self.graph.as_dict(),
            "objective_trace": list(self.objective_trace),
            "total_model_evaluations": self.total_model_evaluations,
            "rounds": [receipt.as_dict() for receipt in self.rounds],
        }


GainProvider = Callable[
    [int, MixtureDistribution, UpdateGraph, tuple[TypedMixtureRoundReceipt, ...]],
    Mapping[str, GainEvidence],
]


def _component_node_map(graph: UpdateGraph, model: MixtureDistribution) -> dict[str, int]:
    by_identity: dict[int, list[int]] = {}
    for index, component in enumerate(model.components):
        by_identity.setdefault(id(component), []).append(index)
    shared = {ident: indices for ident, indices in by_identity.items() if len(indices) > 1}
    if shared:
        raise NotImplementedError(
            "typed local mixture EM does not yet split a shared component into a joint proposal; "
            "use the full-tree estimator until proposal composition is enabled."
        )
    result: dict[str, int] = {}
    for node in graph.nodes:
        indices = by_identity.get(id(node.model))
        if indices:
            result[node.node_id] = indices[0]
    if len(result) != model.num_components:
        raise RuntimeError("compiled graph did not preserve every mixture component identity.")
    return result


def run_typed_mixture_em(
    enc_data: Any,
    estimator: MixtureEstimator,
    initial_model: MixtureDistribution,
    *,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    stall_patience: int = 3,
    accept_tolerance: float = 1.0e-9,
    scheduler: GainPerCostScheduler | None = None,
    gain_provider: GainProvider | None = None,
) -> TypedMixtureRun:
    """Run objective-gated component EM selected by the typed scheduler.

    The root mixture-weight update is an explicit always-on coordinator. Only
    component nodes enter the scheduler's budget. Internal gain evidence comes
    from committed global-objective improvement and is therefore labelled
    ``joint_with_coordinator`` rather than misrepresented as an isolated causal
    contribution. A caller with better per-block probes may supply
    ``gain_provider``; objective compatibility is still enforced by the
    scheduler.
    """

    if not isinstance(initial_model, MixtureDistribution) or not isinstance(estimator, MixtureEstimator):
        raise TypeError("run_typed_mixture_em requires a MixtureDistribution and MixtureEstimator.")
    if getattr(estimator, "has_conj_prior", False) and estimator.fixed_weights is None:
        raise NotImplementedError("typed local mixture EM currently supports the observed-data MLE path only.")
    if max_its < 1:
        raise ValueError("max_its must be at least one.")
    if delta is not None and delta < 0.0:
        raise ValueError("delta must be non-negative when supplied.")

    graph = compile_update_graph(initial_model, estimator)
    if graph.node(graph.root_node).contract.objective_kind is not ObjectiveKind.MLE:
        raise NotImplementedError("typed local mixture EM currently supports MLE objective semantics only.")
    node_to_component = _component_node_map(graph, initial_model)
    component_to_node = {index: node_id for node_id, index in node_to_component.items()}
    candidate_nodes = tuple(node_to_component)
    if scheduler is None:
        scheduler = GainPerCostScheduler(
            SchedulerConfig(
                budget_fraction=min(0.5, 1.0 / max(1, len(candidate_nodes))),
                confidence_z=0.5,
                lambda_invalidation=0.0,
                max_skip_rounds=max(1, len(candidate_nodes) - 1),
            )
        )

    payload = _resolve_payload(enc_data)
    density_cache = FreezeRollupCache()
    artifact_cache = VersionedArtifactCache(graph)
    gain_stats = {node_id: _RunningGain() for node_id in candidate_nodes}
    model = initial_model
    history: list[TypedMixtureRoundReceipt] = []
    stall_streak = 0

    for round_index in range(max_its):
        internal_evidence = {
            node_id: GainEvidence(
                node_id,
                ObjectiveKind.MLE,
                stats.mean,
                standard_error=stats.standard_error,
                sample_count=stats.count,
                model_version=artifact_cache.generation(node_id),
            )
            for node_id, stats in gain_stats.items()
            if stats.count > 0
        }
        if gain_provider is not None:
            internal_evidence.update(gain_provider(round_index, model, graph, tuple(history)))
        schedule = scheduler.schedule(
            graph,
            internal_evidence,
            target_objective=ObjectiveKind.MLE,
            round_index=round_index,
            candidate_nodes=candidate_nodes,
        )
        active = tuple(sorted(node_to_component[node_id] for node_id in schedule.selected_nodes))
        inactive = set(range(model.num_components)) - set(active)
        inactive.update(index for index in range(model.num_components) if model.zw[index])

        started = time.perf_counter()
        ll_matrix, evals_before = _component_log_density_matrix(model, payload, density_cache, inactive)
        log_density, responsibilities = _combine(ll_matrix, model.log_w)
        objective_before = float(np.sum(log_density))
        candidate = _m_step(payload, estimator, model, responsibilities, inactive)
        candidate_matrix, evals_after = _component_log_density_matrix(candidate, payload, density_cache, inactive)
        candidate_density = _log_density_from_matrix(candidate_matrix, candidate.log_w)
        candidate_objective = float(np.sum(candidate_density))
        elapsed = time.perf_counter() - started

        accepted = np.isfinite(candidate_objective) and candidate_objective + accept_tolerance >= objective_before
        if accepted:
            model = candidate
            committed_objective = candidate_objective
            written_nodes = tuple(schedule.selected_nodes) + (graph.root_node,)
            invalidation = artifact_cache.invalidate_many(written_nodes)
        else:
            committed_objective = objective_before
            invalidation = None
            for index in active:
                density_cache.invalidate(index)
        realized_gain = max(0.0, committed_objective - objective_before)
        if schedule.selected_nodes:
            attributed_gain = realized_gain / len(schedule.selected_nodes)
            for node_id in schedule.selected_nodes:
                gain_stats[node_id].update(attributed_gain)

        nobs = float(len(log_density))
        model_evaluations = evals_before + evals_after
        receipt = TypedMixtureRoundReceipt(
            round_index=round_index,
            schedule=schedule,
            coordinator_nodes=(graph.root_node,),
            objective_before=objective_before,
            candidate_objective=candidate_objective,
            committed_objective=committed_objective,
            accepted=accepted,
            realized_gain=realized_gain,
            gain_attribution="joint_with_coordinator",
            active_components=active,
            invalidation=invalidation,
            work=WorkMeasurement(
                node_type=type(model).__name__,
                update_kind=UpdateKind.GENERALIZED_EM,
                backend="typed_local",
                wall_time_seconds=elapsed,
                compute_units=nobs * model_evaluations,
                observations=nobs,
                model_evaluations=model_evaluations,
                operation_count=model_evaluations + len(active) + 1,
                extra={"accepted": accepted, "active_components": list(active)},
            ),
        )
        history.append(receipt)

        if delta is not None and realized_gain < delta:
            stall_streak += 1
            if stall_streak >= max(1, stall_patience):
                break
        else:
            stall_streak = 0

    return TypedMixtureRun(model, graph, tuple(history))


__all__ = ["GainProvider", "TypedMixtureRoundReceipt", "TypedMixtureRun", "run_typed_mixture_em"]
