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
    PlacementScope,
    StructuredPlacementPlan,
    plan_structured_placement,
)
from mixle.stats.compute.pdist import ParameterEstimator, ProbabilityDistribution
from mixle.utils.parallel.model_parallel import model_parallel_fold
from mixle.utils.parallel.planner import _global_key_merge


@dataclass(frozen=True)
class StructuredEstimationReceipt:
    """Placement, work, sufficient-statistic, and model parity for one M-step."""

    placement: StructuredPlacementPlan
    observations: float
    num_workers: int
    worker_device_ids: tuple[str, ...]
    execution_backend: str
    parallel_node_ids: tuple[str, ...]
    parallel_statistics_hash: str
    reference_statistics_hash: str | None
    parallel_model_hash: str
    reference_model_hash: str | None
    exact_parity: bool | None
    work: WorkMeasurement
    reference_seconds: float | None = None

    def __post_init__(self) -> None:
        """Bind ``exact_parity`` to the hashes it claims to summarize (MXR-080-0647).

        ``exact_parity`` is the receipt's headline claim: that the parallel M-step reproduced the
        serial reference exactly. It was previously a free-standing boolean, so a receipt could
        report ``exact_parity=True`` while carrying ``parallel_statistics_hash != reference_statistics_hash``
        -- asserting a parity its own evidence refutes, and doing so in the artifact a reader would
        consult to confirm it. The producer derives the flag from exactly this comparison, so
        checking it here rejects only receipts no run can produce.

        Deliberately NOT checked: ``num_workers == len(worker_device_ids)``. A caller may pass
        ``num_workers`` below the placement capacity, so a worker count smaller than the device list
        is a legitimate plan, not a forged one.
        """
        has_reference = self.reference_statistics_hash is not None or self.reference_model_hash is not None
        if self.exact_parity is None:
            if has_reference:
                raise ValueError(
                    "structured-estimation receipt carries reference hashes but leaves exact_parity "
                    "unset; a reference run either establishes parity or refutes it."
                )
        else:
            if not has_reference:
                raise ValueError(
                    f"structured-estimation receipt claims exact_parity={self.exact_parity} with no "
                    "reference hashes; parity is a comparison and needs both sides."
                )
            matches = (
                self.parallel_statistics_hash == self.reference_statistics_hash
                and self.parallel_model_hash == self.reference_model_hash
            )
            if bool(self.exact_parity) != matches:
                raise ValueError(
                    f"structured-estimation receipt reports exact_parity={self.exact_parity} but its "
                    f"hashes say otherwise: statistics {self.parallel_statistics_hash!r} vs "
                    f"{self.reference_statistics_hash!r}, model {self.parallel_model_hash!r} vs "
                    f"{self.reference_model_hash!r}."
                )
        if not (isinstance(self.observations, (int, float)) and self.observations >= 0.0):
            raise ValueError(
                f"structured-estimation receipt observations must be non-negative, got {self.observations!r}."
            )
        if self.reference_seconds is not None and not self.reference_seconds >= 0.0:
            raise ValueError(
                f"structured-estimation receipt reference_seconds must be non-negative, got {self.reference_seconds!r}."
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible structured-execution receipt."""

        return {
            "placement": self.placement.as_dict(),
            "observations": self.observations,
            "num_workers": self.num_workers,
            "worker_device_ids": list(self.worker_device_ids),
            "execution_backend": self.execution_backend,
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
    if not callable(to_json):
        raise TypeError("structured model hashing requires deterministic to_json().")
    return payload_fingerprint(to_json())


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
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be positive.")
    if any(row.scope is PlacementScope.CROSS_ISLAND_PROPOSAL for row in placement.placements):
        raise NotImplementedError("the local structured executor cannot execute cross-island proposal placement.")
    planned_parallel = tuple(row for row in placement.placements if any(shard.axis != "none" for shard in row.shards))
    unsupported_axes = sorted(
        {shard.axis for row in planned_parallel for shard in row.shards if shard.axis not in {"component", "factor"}}
    )
    if unsupported_axes:
        raise NotImplementedError(
            "the local structured executor does not implement planned axes: %s." % ", ".join(unsupported_axes)
        )
    worker_capacity = (
        min(len({shard.device_id for shard in row.shards}) for row in planned_parallel) if planned_parallel else 1
    )
    worker_count = worker_capacity if num_workers is None else num_workers
    if worker_count > worker_capacity:
        raise ValueError("num_workers=%d exceeds the enforced placement capacity %d." % (worker_count, worker_capacity))
    parallel_node_ids = tuple(row.node_id for row in planned_parallel)
    parallel_model_ids = frozenset(id(graph.node(node_id).model) for node_id in parallel_node_ids)
    worker_devices = tuple(
        sorted(
            {
                shard.device_id
                for row in (planned_parallel if planned_parallel else (placement.placement(graph.root_node),))
                for shard in row.shards
            }
        )
    )[:worker_count]
    topology_devices = {device.device_id: device for device in topology.devices}
    if any(
        topology_devices[device_id].spec.kind != "cpu"
        or topology_devices[device_id].spec.engine != "numpy"
        or topology_devices[device_id].provider != "local"
        for device_id in worker_devices
    ):
        raise NotImplementedError("the local structured executor supports only declared local numpy CPU worker slots.")
    if len({topology_devices[device_id].host for device_id in worker_devices}) != 1:
        raise ValueError("local structured worker slots must belong to one declared host.")

    parallel_accumulator = estimator.accumulator_factory().make()
    started = time.perf_counter()
    model_parallel_fold(
        parallel_accumulator,
        model,
        payload,
        weights,
        worker_count,
        parallel_ids=parallel_model_ids,
    )
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

    receipt = StructuredEstimationReceipt(
        placement=placement,
        observations=nobs,
        num_workers=worker_count,
        worker_device_ids=worker_devices,
        execution_backend="local_numpy_thread_pool",
        parallel_node_ids=parallel_node_ids,
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
            extra={
                "num_workers": worker_count,
                "worker_device_ids": list(worker_devices),
                "parallel_node_ids": list(parallel_node_ids),
                "placement_enforced": True,
            },
        ),
        reference_seconds=reference_seconds,
    )
    return StructuredEstimationResult(parallel_model, receipt)


__all__ = ["StructuredEstimationReceipt", "StructuredEstimationResult", "run_structured_estimation_step"]
