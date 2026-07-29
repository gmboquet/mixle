"""Optional cluster launchers for distributed-gradient sessions.

Ray Train and Lightning Fabric manage worker lifecycle. They do not alter the
statistical update or pretend to implement model-parallel kernels; each worker
still selects ``torch_native`` or ``megatron`` and materializes its plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from numbers import Integral, Real
from typing import Any

from mixle.utils.parallel.training_contracts import ParallelAxis, ParallelPlan


@dataclass(frozen=True)
class WorkerTopologyAttestation:
    """Runtime proof binding one launched rank to one planned mesh coordinate."""

    rank: int
    world_size: int
    coordinate: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "coordinate": dict(self.coordinate),
        }


def attest_worker_coordinate(plan: ParallelPlan, *, actual_world_size: int, rank: int) -> WorkerTopologyAttestation:
    """Validate a runtime rank and deterministically bind it to the plan mesh."""
    if not isinstance(plan, ParallelPlan):
        raise TypeError("plan must be a ParallelPlan.")
    plan.validate_world_size(actual_world_size)
    actual_world_size = int(actual_world_size)
    if isinstance(rank, bool) or not isinstance(rank, Integral) or not 0 <= rank < actual_world_size:
        raise ValueError("rank must be an exact integer in the runtime world-size range.")
    rank = int(rank)
    physical_axes = (
        ParallelAxis.DP_REPLICATE,
        ParallelAxis.DP_SHARD,
        ParallelAxis.TP,
        ParallelAxis.PP,
        ParallelAxis.CP,
    )
    remainder = rank
    coordinates = [0] * len(physical_axes)
    for position in range(len(physical_axes) - 1, -1, -1):
        size = plan.size(physical_axes[position])
        coordinates[position] = remainder % size
        remainder //= size
    if remainder:
        raise ValueError("rank cannot be represented by the planned mesh.")
    return WorkerTopologyAttestation(
        rank=rank,
        world_size=actual_world_size,
        coordinate=tuple((axis.value, coordinates[position]) for position, axis in enumerate(physical_axes)),
    )


def _ray_attested_loop(train_loop_per_worker, plan: ParallelPlan, config: dict[str, Any]) -> Any:
    from ray import train

    context = train.get_context()
    attestation = attest_worker_coordinate(
        plan,
        actual_world_size=context.get_world_size(),
        rank=context.get_world_rank(),
    )
    worker_config = dict(config)
    worker_config["mixle_worker_topology_attestation"] = attestation.as_dict()
    return train_loop_per_worker(worker_config)


class _TopologyBoundFabric:
    """Proxy that refuses to claim a Fabric rank until ``launch`` proves it."""

    def __init__(self, fabric: Any, plan: ParallelPlan) -> None:
        self._fabric = fabric
        self._plan = plan
        self.topology_attestation: WorkerTopologyAttestation | None = None

    def launch(self, *args: Any, **kwargs: Any) -> Any:
        result = self._fabric.launch(*args, **kwargs)
        self.topology_attestation = attest_worker_coordinate(
            self._plan,
            actual_world_size=self._fabric.world_size,
            rank=self._fabric.global_rank,
        )
        return result

    def attest_topology(self) -> WorkerTopologyAttestation:
        if self.topology_attestation is None:
            raise RuntimeError("Fabric topology is not attested; call launch() on every worker first.")
        return self.topology_attestation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fabric, name)


def _validate_resources_per_worker(resources_per_worker, *, use_gpu: bool) -> dict[str, float] | None:
    if resources_per_worker is None:
        return None
    if not isinstance(resources_per_worker, dict):
        raise TypeError("resources_per_worker must be a dictionary or None.")
    result = {}
    for name, value in resources_per_worker.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("resource names must be non-empty strings.")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0.0:
            raise ValueError("resource quantities must be finite non-negative real numbers.")
        result[name] = float(value)
    gpu = result.get("GPU")
    if use_gpu and gpu is not None and gpu <= 0.0:
        raise ValueError("use_gpu=True conflicts with a non-positive GPU resource request.")
    if not use_gpu and gpu is not None and gpu > 0.0:
        raise ValueError("use_gpu=False conflicts with a positive GPU resource request.")
    return result


class RayTrainLauncher:
    """Launch one worker per physical plan rank with ``ray.train.torch``."""

    def launch(
        self,
        train_loop_per_worker: Any,
        *,
        plan: ParallelPlan,
        train_loop_config: dict[str, Any] | None = None,
        use_gpu: bool = True,
        resources_per_worker: dict[str, float] | None = None,
        run_config: Any = None,
    ) -> Any:
        if not callable(train_loop_per_worker):
            raise TypeError("train_loop_per_worker must be callable.")
        if not isinstance(plan, ParallelPlan):
            raise TypeError("plan must be a ParallelPlan.")
        if not isinstance(use_gpu, bool):
            raise TypeError("use_gpu must be boolean.")
        if train_loop_config is not None and not isinstance(train_loop_config, dict):
            raise TypeError("train_loop_config must be a dictionary or None.")
        resources_per_worker = _validate_resources_per_worker(resources_per_worker, use_gpu=use_gpu)
        config = dict(train_loop_config or {})
        reserved = {"mixle_parallel_plan", "mixle_worker_topology_attestation"}.intersection(config)
        if reserved:
            raise ValueError("train_loop_config contains reserved Mixle topology keys.")
        config["mixle_parallel_plan"] = plan.as_dict()
        try:
            from ray.train import RunConfig, ScalingConfig
            from ray.train.torch import TorchTrainer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Ray training launch requires the ray extra.") from error

        scaling = ScalingConfig(
            num_workers=plan.world_size,
            use_gpu=use_gpu,
            resources_per_worker=resources_per_worker,
        )
        trainer = TorchTrainer(
            train_loop_per_worker=partial(_ray_attested_loop, train_loop_per_worker, plan),
            train_loop_config=config,
            scaling_config=scaling,
            run_config=run_config if run_config is not None else RunConfig(),
        )
        return trainer.fit()


class LightningFabricLauncher:
    """Create Fabric for pure DDP or FSDP jobs.

    Lightning is capability-gated to data parallelism here. Model parallel
    dimensions remain the responsibility of the native DeviceMesh or Megatron
    backend rather than being accepted and discarded by Fabric.
    """

    def create(
        self,
        *,
        plan: ParallelPlan,
        accelerator: str = "auto",
        devices: int | str = "auto",
        num_nodes: int = 1,
        precision: str = "32-true",
        strategy: Any = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(plan, ParallelPlan):
            raise TypeError("plan must be a ParallelPlan.")
        model_axes = {
            ParallelAxis.TP,
            ParallelAxis.PP,
            ParallelAxis.CP,
            ParallelAxis.EP,
            ParallelAxis.ETP,
        }
        active = model_axes.intersection(plan.active_axes)
        if active:
            raise NotImplementedError(
                "Lightning Fabric launcher only owns data parallelism; requested: %s"
                % ", ".join(sorted(axis.value for axis in active))
            )
        if plan.dp_replicate > 1 and plan.dp_shard > 1 and strategy is None:
            raise NotImplementedError("hybrid sharded data parallelism requires an explicit Fabric strategy.")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral) or num_nodes < 1:
            raise ValueError("num_nodes must be an exact positive integer.")
        num_nodes = int(num_nodes)
        if devices == "auto":
            if num_nodes != 1:
                raise ValueError("multi-node launch requires an explicit per-node device count.")
            devices = plan.world_size
        elif isinstance(devices, bool) or not isinstance(devices, Integral) or devices < 1:
            raise ValueError("devices must be 'auto' or an exact positive integer.")
        else:
            devices = int(devices)
        if devices * num_nodes != plan.world_size:
            raise ValueError(
                f"devices * num_nodes must equal plan.world_size ({devices} * {num_nodes} != {plan.world_size})."
            )
        try:
            from lightning.fabric import Fabric
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Lightning launch requires the lightning extra.") from error
        if strategy is None:
            strategy = "fsdp" if plan.dp_shard > 1 else "ddp"
        return _TopologyBoundFabric(
            Fabric(
                accelerator=accelerator,
                devices=devices,
                num_nodes=num_nodes,
                precision=precision,
                strategy=strategy,
                **kwargs,
            ),
            plan,
        )


__all__ = [
    "LightningFabricLauncher",
    "RayTrainLauncher",
    "WorkerTopologyAttestation",
    "attest_worker_coordinate",
]
