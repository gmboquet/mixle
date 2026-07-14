"""Optional cluster launchers for distributed-gradient sessions.

Ray Train and Lightning Fabric manage worker lifecycle. They do not alter the
statistical update or pretend to implement model-parallel kernels; each worker
still selects ``torch_native`` or ``megatron`` and materializes its plan.
"""

from __future__ import annotations

from typing import Any

from mixle.utils.parallel.training_contracts import ParallelAxis, ParallelPlan


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
        try:
            from ray.train import RunConfig, ScalingConfig
            from ray.train.torch import TorchTrainer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Ray training launch requires the ray extra.") from error

        config = dict(train_loop_config or {})
        config["mixle_parallel_plan"] = plan.as_dict()
        scaling = ScalingConfig(
            num_workers=plan.world_size,
            use_gpu=use_gpu,
            resources_per_worker=resources_per_worker,
        )
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
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
        try:
            from lightning.fabric import Fabric
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Lightning launch requires the lightning extra.") from error
        if strategy is None:
            strategy = "fsdp" if plan.dp_shard > 1 else "ddp"
        if devices == "auto" and num_nodes == 1:
            devices = plan.world_size
        return Fabric(
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            strategy=strategy,
            **kwargs,
        )


__all__ = ["LightningFabricLauncher", "RayTrainLauncher"]
