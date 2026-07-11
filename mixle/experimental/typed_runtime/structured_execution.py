"""Typed adapter for the stable exact model-parallel sufficient-statistic fold."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.compiler import compile_update_graph
from mixle.experimental.typed_runtime.measurement import WorkMeasurement
from mixle.experimental.typed_runtime.proposal import payload_fingerprint
from mixle.experimental.typed_runtime.topology import (
    ClusterTopology,
    StructuredPlacementPlan,
    plan_structured_placement,
)
from mixle.stats.compute.pdist import ParameterEstimator, ProbabilityDistribution
from mixle.utils.parallel.model_parallel import _parallel_ids, model_parallel_fold
from mixle.utils.parallel.planner import _global_key_merge


@dataclass(frozen=True)
class StructuredEstimationReceipt:
    """Placement, work, sufficient-statistic, and model parity for one M-step."""

    placement: StructuredPlacementPlan
    observations: float
    num_workers: int
    parallel_node_ids: tuple[str, ...]
    parallel_statistics_hash: str
    reference_statistics_hash: str | None
    parallel_model_hash: str
    reference_model_hash: str | None
    exact_parity: bool | None
    work: WorkMeasurement
    reference_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible structured-execution receipt."""

        return {
            "placement": self.placement.as_dict(),
            "observations": self.observations,
            "num_workers": self.num_workers,
            "parallel_node_ids": list(self.parallel_node_ids),
            "parallel_statistics_hash": self.parallel_statistics_hash,
            "reference_statistics_hash": self.reference_statistics_hash,
            "parallel_model_hash": self.parallel_model_hash,
            "reference_model_hash": self.reference_model_hash,
            "exact_parity": self.exact_parity,
            "work": self.work.as_dict(),
            "reference_seconds": self.reference_seconds,
        }


@dataclass(frozen=True)
class StructuredEstimationResult:
    """Estimated model and receipt; runtime model excluded from serialization."""

    model: ProbabilityDistribution = field(repr=False)
    receipt: StructuredEstimationReceipt


def _encoded_payload_and_size(encoded_data: Any, weights: np.ndarray | None) -> tuple[Any, np.ndarray, float]:
    if (
        isinstance(encoded_data, list)
        and len(encoded_data) == 1
        and isinstance(encoded_data[0], tuple)
        and len(encoded_data[0]) == 2
        and isinstance(encoded_data[0][0], (int, float, np.integer, np.floating))
    ):
        size, payload = encoded_data[0]
        if weights is None:
            weights = np.ones(int(size), dtype=np.float64)
    else:
        payload = encoded_data
        if weights is None:
            raise ValueError("bare encoded payloads require explicit weights.")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("structured estimation weights must be a finite non-negative vector.")
    return payload, weights, float(weights.sum())


def _model_hash(model: ProbabilityDistribution) -> str:
    to_json = getattr(model, "to_json", None)
    return payload_fingerprint(to_json() if callable(to_json) else str(model))


def run_structured_estimation_step(
    encoded_data: Any,
    estimator: ParameterEstimator,
    model: ProbabilityDistribution,
    topology: ClusterTopology,
    *,
    weights: np.ndarray | None = None,
    num_workers: int | None = None,
    verify_reference: bool = True,
) -> StructuredEstimationResult:
    """Execute one exact typed model-axis E/M step and optionally verify serial parity."""

    payload, weights, nobs = _encoded_payload_and_size(encoded_data, weights)
    graph = compile_update_graph(model, estimator, nobs=nobs)
    placement = plan_structured_placement(graph, topology, n_data=int(len(weights)))
    worker_count = num_workers or len(topology.devices_in(placement.primary_island))
    if worker_count < 1:
        raise ValueError("num_workers must be positive.")

    parallel_accumulator = estimator.accumulator_factory().make()
    started = time.perf_counter()
    model_parallel_fold(parallel_accumulator, model, payload, weights, worker_count)
    _global_key_merge(parallel_accumulator)
    parallel_statistics = parallel_accumulator.value()
    parallel_model = estimator.estimate(nobs, parallel_statistics)
    elapsed = time.perf_counter() - started
    parallel_statistics_hash = payload_fingerprint(parallel_statistics)
    parallel_model_hash = _model_hash(parallel_model)

    reference_statistics_hash: str | None = None
    reference_model_hash: str | None = None
    reference_seconds: float | None = None
    parity: bool | None = None
    if verify_reference:
        reference_accumulator = estimator.accumulator_factory().make()
        reference_started = time.perf_counter()
        reference_accumulator.seq_update(payload, weights, model)
        _global_key_merge(reference_accumulator)
        reference_statistics = reference_accumulator.value()
        reference_model = estimator.estimate(nobs, reference_statistics)
        reference_seconds = time.perf_counter() - reference_started
        reference_statistics_hash = payload_fingerprint(reference_statistics)
        reference_model_hash = _model_hash(reference_model)
        parity = parallel_statistics_hash == reference_statistics_hash and parallel_model_hash == reference_model_hash
        if graph.node(graph.root_node).contract.exact and not parity:
            raise RuntimeError("exact structured estimation did not match the serial reference.")

    selected_ids = _parallel_ids(model, worker_count)
    parallel_nodes = tuple(node.node_id for node in graph.nodes if id(node.model) in selected_ids)
    receipt = StructuredEstimationReceipt(
        placement=placement,
        observations=nobs,
        num_workers=worker_count,
        parallel_node_ids=parallel_nodes,
        parallel_statistics_hash=parallel_statistics_hash,
        reference_statistics_hash=reference_statistics_hash,
        parallel_model_hash=parallel_model_hash,
        reference_model_hash=reference_model_hash,
        exact_parity=parity,
        work=WorkMeasurement(
            node_type=type(model).__name__,
            update_kind=graph.node(graph.root_node).contract.update_kind,
            backend="typed_model_parallel",
            wall_time_seconds=elapsed,
            compute_units=graph.node(graph.root_node).cost.compute_units,
            observations=nobs,
            operation_count=1,
            extra={"num_workers": worker_count, "parallel_node_ids": list(parallel_nodes)},
        ),
        reference_seconds=reference_seconds,
    )
    return StructuredEstimationResult(parallel_model, receipt)


__all__ = ["StructuredEstimationReceipt", "StructuredEstimationResult", "run_structured_estimation_step"]
