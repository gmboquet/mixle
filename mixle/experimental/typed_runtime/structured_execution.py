"""Typed adapter for the stable exact model-parallel sufficient-statistic fold."""

from __future__ import annotations

import math
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
        is a legitimate plan, not a forged one. Above it is a different matter, and was unchecked:
        ``num_workers=999`` with one placed device constructed, claiming workers no placement
        admitted. That direction is now refused, along with blank or repeated ids and empty hashes.

        Note what ``worker_device_ids`` does and does not attest. The executor admits work against
        these slots -- capacity, device kind and host are all checked before the fold -- but the fold
        itself runs on a ``ThreadPoolExecutor``, which has no API to pin a task to a device. The list
        names the slots the work was admitted against, not devices it provably ran on, and the work
        measurement carries ``placement_admitted`` / ``device_affinity_enforced`` separately so the
        two are not read as one claim.
        """
        # The typed evidence itself, checked first: a receipt whose placement or work measurement is
        # None carries no evidence at all, yet every field below reads as a claim about a run that
        # produced them (MXR-080-0647). Node identity is bound to the plan for the same reason -- an
        # invented node id or an empty parallel set describes no execution the plan could have
        # scheduled.
        if not isinstance(self.placement, StructuredPlacementPlan):
            raise TypeError(
                "structured-estimation receipt placement must be a StructuredPlacementPlan, got "
                f"{type(self.placement).__name__}: the device and worker claims below are only "
                "meaningful against a real plan."
            )
        if not isinstance(self.work, WorkMeasurement):
            raise TypeError(
                f"structured-estimation receipt work must be a WorkMeasurement, got {type(self.work).__name__}."
            )
        # An EMPTY parallel set is deliberately allowed: a graph with no shardable axis executes
        # atomically, and the executor produces exactly that receipt. Requiring a node here rejected
        # a real run rather than a forged one.
        planned_nodes = {row.node_id for row in self.placement.placements}
        unplanned = sorted(set(self.parallel_node_ids) - planned_nodes)
        if unplanned:
            raise ValueError(
                f"structured-estimation receipt claims parallel node(s) {unplanned} that its own "
                "placement plan does not contain."
            )
        if len(set(self.parallel_node_ids)) != len(self.parallel_node_ids):
            raise ValueError("structured-estimation receipt repeats a parallel node id.")
        placed_devices = {shard.device_id for row in self.placement.placements for shard in row.shards}
        invented = sorted(set(self.worker_device_ids) - placed_devices)
        if invented:
            raise ValueError(
                f"structured-estimation receipt names worker device(s) {invented} that its own "
                "placement plan never placed."
            )
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 1:
            raise ValueError(
                f"structured-estimation receipt num_workers must be a positive integer, got {self.num_workers!r}."
            )
        if len(set(self.worker_device_ids)) != len(self.worker_device_ids):
            raise ValueError(
                f"structured-estimation receipt repeats a worker device id: {list(self.worker_device_ids)}. "
                "One slot cannot be two workers."
            )
        if self.num_workers > len(self.worker_device_ids):
            raise ValueError(
                f"structured-estimation receipt claims {self.num_workers} workers against "
                f"{len(self.worker_device_ids)} placed device slot(s). A worker count cannot exceed the "
                "placement it was admitted against."
            )
        for name, value in (
            ("execution_backend", self.execution_backend),
            ("parallel_statistics_hash", self.parallel_statistics_hash),
            ("parallel_model_hash", self.parallel_model_hash),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"structured-estimation receipt {name} must name something, got {value!r}.")
        # Internal consistency, not just per-field plausibility (MXR-080-0647). A genuine placement
        # and a genuine work measurement could be paired with contradictory scalars: observations=10
        # beside work.observations=40, a backend naming one executor beside work.backend naming
        # another, and a non-Boolean parity. Each field passed on its own and the receipt as a whole
        # described two different runs.
        if float(self.observations) != float(self.work.observations):
            raise ValueError(
                f"structured-estimation receipt reports observations={self.observations} while its own "
                f"work measurement reports {self.work.observations}; one run processed one row count."
            )
        # The receipt and its work measurement each name a backend, and nothing tied them together:
        # ``execution_backend="contradictory-backend"`` sat beside ``work.backend="typed_model_parallel"``
        # and constructed, so the artifact a reader consults to learn what ran named two things
        # (MXR-080-1871). They are deliberately NOT required to be equal -- they answer different
        # questions, and the real producer pairs the executor "local_numpy_thread_pool" with the typed
        # node backend "typed_model_parallel" -- so the binding is that the measurement records the
        # executor it measured and the two agree. Every scalar the receipt copies out of its own work
        # measurement is checked the same way, for the same reason.
        recorded_backend = self.work.extra.get("execution_backend")
        if recorded_backend is None:
            raise ValueError(
                "structured-estimation receipt work measurement does not record the executor it "
                f"measured; add extra={{'execution_backend': {self.execution_backend!r}}}. Without it "
                "the receipt's own execution_backend is bound to nothing and can name anything."
            )
        if recorded_backend != self.execution_backend:
            raise ValueError(
                f"structured-estimation receipt reports execution_backend={self.execution_backend!r} "
                f"while its own work measurement recorded {recorded_backend!r}; one run had one "
                "executor."
            )
        for name, claimed in (
            ("num_workers", self.num_workers),
            ("worker_device_ids", tuple(self.worker_device_ids)),
            ("parallel_node_ids", tuple(self.parallel_node_ids)),
        ):
            measured = self.work.extra.get(name)
            if measured is None:
                continue
            if isinstance(claimed, tuple):
                measured = tuple(measured)
            if measured != claimed:
                raise ValueError(
                    f"structured-estimation receipt reports {name}={claimed!r} while its own work "
                    f"measurement recorded {measured!r}; one run produced one answer."
                )
        if self.exact_parity is not None and not isinstance(self.exact_parity, bool):
            raise TypeError(
                f"structured-estimation receipt exact_parity must be a Boolean verdict or None, got "
                f"{self.exact_parity!r}."
            )
        for name in ("reference_statistics_hash", "reference_model_hash"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"structured-estimation receipt {name} must name something when set, got {value!r}.")
        blank_devices = [item for item in self.worker_device_ids if not isinstance(item, str) or not item.strip()]
        if blank_devices:
            raise ValueError(
                f"structured-estimation receipt worker_device_ids contains an id that names nothing: {blank_devices!r}."
            )
        present = (self.reference_statistics_hash is not None, self.reference_model_hash is not None)
        if any(present) and not all(present):
            raise ValueError(
                "structured-estimation receipt carries a half reference: statistics "
                f"{self.reference_statistics_hash!r}, model {self.reference_model_hash!r}. Parity is "
                "the conjunction of both comparisons, so one side alone cannot establish it and "
                "cannot refute it either -- the missing hash would compare unequal purely by being "
                "absent. The reference run computes both together or runs not at all."
            )
        has_reference = all(present)
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
        # ``>= 0.0`` alone admits ``inf``: NaN fails it (NaN compares false against everything) but
        # infinity passes, so a receipt could report having processed infinitely many observations in
        # infinite reference time. Both are measured quantities from a run that finished, so both are
        # finite by construction; isfinite rejects the impossible magnitude the old bound let through.
        if not (isinstance(self.observations, (int, float)) and math.isfinite(self.observations)):
            raise ValueError(
                f"structured-estimation receipt observations must be a finite number, got {self.observations!r}."
            )
        if self.observations < 0.0:
            raise ValueError(
                f"structured-estimation receipt observations must be non-negative, got {self.observations!r}."
            )
        if self.reference_seconds is not None:
            if not math.isfinite(self.reference_seconds):
                raise ValueError(
                    "structured-estimation receipt reference_seconds must be finite, got "
                    f"{self.reference_seconds!r}: a reference run that produced hashes also took a "
                    "measurable amount of time."
                )
            if self.reference_seconds < 0.0:
                raise ValueError(
                    "structured-estimation receipt reference_seconds must be non-negative, got "
                    f"{self.reference_seconds!r}."
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
                # Records the executor this measurement was taken on, so the receipt's own
                # execution_backend is bound to measured evidence rather than free-floating
                # (MXR-080-1871).
                "execution_backend": "local_numpy_thread_pool",
                "num_workers": worker_count,
                "worker_device_ids": list(worker_devices),
                "parallel_node_ids": list(parallel_node_ids),
                # This said "placement_enforced": True, which was not true of anything that happens
                # below (MXR-080-0647). What IS enforced is admission: the slots were checked against
                # the plan's capacity, kind and host before the fold started, and a request exceeding
                # them is refused. What is NOT enforced is affinity -- the fold runs on a
                # ThreadPoolExecutor, which offers no API to pin a task to a device, so
                # worker_device_ids names the slots the work was admitted against, not the devices it
                # provably ran on. Reporting both separately keeps a reader from taking the device
                # list as placement evidence it cannot be.
                "placement_admitted": True,
                "device_affinity_enforced": False,
            },
        ),
        reference_seconds=reference_seconds,
    )
    return StructuredEstimationResult(parallel_model, receipt)


__all__ = ["StructuredEstimationReceipt", "StructuredEstimationResult", "run_structured_estimation_step"]
