"""Executable PyTorch distributed-gradient backend.

This is the native path for arbitrary Mixle ``nn.Module`` objects.  It uses
public PyTorch APIs: DeviceMesh/DTensor tensor parallelism, FSDP2 (including
HSDP), DDP, distributed checkpoint-compatible optimizer state, and the
experimental context-parallel SDPA context manager.  Pipeline and expert
parallel transformer training belong to the Megatron backend, whose model
partitioner owns those semantics.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from mixle.models.optimizer_routing import (
    plan_neural_optimizer,
    resolve_neural_optimizer,
    shard_safe_neural_optimizer_plan,
)
from mixle.utils.parallel.training_contracts import (
    BackendCapabilities,
    ParallelAxis,
    ParallelPlan,
    ParameterLayout,
    StateLayout,
    StepReceipt,
)


def _is_dtensor(value: Any) -> bool:
    return type(value).__name__ == "DTensor" and hasattr(value, "placements")


def _is_sharded_on(value: Any, axis: ParallelAxis) -> bool:
    if not _is_dtensor(value):
        return False
    names = tuple(getattr(value.device_mesh, "mesh_dim_names", ()) or ())
    for name, placement in zip(names, value.placements):
        if name == axis.value and type(placement).__name__ == "Shard":
            return True
    return False


def describe_parameter_layouts(module: Any, plan: ParallelPlan) -> tuple[ParameterLayout, ...]:
    """Capture stable logical identities before distributed wrappers rewrite parameters."""

    try:
        named = module.named_parameters(remove_duplicate=False)
    except TypeError:  # older/custom Module-compatible implementations
        named = module.named_parameters()
    occurrences: dict[int, list[str]] = {}
    records = list(named)
    for name, parameter in records:
        occurrences.setdefault(id(parameter), []).append(str(name))
    layouts: list[ParameterLayout] = []
    seen: set[int] = set()
    for name, parameter in records:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        logical_id = str(name)
        placements: list[tuple[str, str]] = []
        if plan.dp_shard > 1:
            placements.append((ParallelAxis.DP_SHARD.value, "fsdp_flat_shard"))
        if plan.tp > 1 and (".mlp.0." in logical_id or ".mlp.2." in logical_id):
            dimension = "output" if ".mlp.0." in logical_id else "input"
            placements.append((ParallelAxis.TP.value, "%s_feature_shard" % dimension))
        shared = occurrences[id(parameter)]
        layouts.append(
            ParameterLayout(
                logical_id=logical_id,
                global_shape=tuple(int(size) for size in parameter.shape),
                placements=tuple(placements),
                shared_group="=".join(shared) if len(shared) > 1 else None,
                optimizer_state=StateLayout.SHARDED if plan.dp_shard > 1 else StateLayout.REPLICATED,
            )
        )
    return tuple(layouts)


def _submesh(mesh: Any, names: tuple[str, ...]) -> Any:
    if len(names) == 1:
        return mesh[names[0]]
    return mesh[names]


@dataclass(frozen=True)
class TorchSessionState:
    """Serializable non-tensor clocks owned by a native training session."""

    step: int
    pending_accumulation: int
    optimizer_receipt: dict[str, Any]


class TorchDistributedSession:
    """A transformed module plus optimizer and explicit process-mesh semantics."""

    def __init__(
        self,
        module: Any,
        *,
        plan: ParallelPlan,
        capabilities: BackendCapabilities,
        mesh: Any,
        device: Any,
        optimizer: Any,
        optimizer_receipt: dict[str, Any],
        parameter_layouts: tuple[ParameterLayout, ...],
        precision: str,
        max_grad_norm: float | None,
        scheduler: Any = None,
        manual_data_parallel: bool = False,
    ) -> None:
        self.module = module
        self.plan = plan
        self.capabilities = capabilities
        self.mesh = mesh
        self.device = device
        self.optimizer = optimizer
        self.optimizer_receipt = optimizer_receipt
        self.parameter_layouts = parameter_layouts
        self.precision = precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.manual_data_parallel = manual_data_parallel
        self.step = 0
        self._pending = 0
        self._pending_examples = 0
        self._pending_tokens = 0
        self._pending_loss_sum = 0.0
        self._closed = False
        self.optimizer.zero_grad(set_to_none=True)

        coordinate = mesh.get_coordinate() if mesh is not None else None
        names = mesh.mesh_dim_names if mesh is not None else ()
        coords = dict(zip(names, coordinate or ()))
        self.data_parallel_rank = coords.get(ParallelAxis.DP_REPLICATE.value, 0) * plan.dp_shard + coords.get(
            ParallelAxis.DP_SHARD.value, 0
        )
        self.data_parallel_size = plan.data_parallel_size
        self.is_logging_rank = (
            all(
                coords.get(axis.value, 0) == 0
                for axis in (ParallelAxis.TP, ParallelAxis.PP, ParallelAxis.CP, ParallelAxis.EP, ParallelAxis.ETP)
            )
            and self.data_parallel_rank == 0
        )

    def _axis_group(self, axis: ParallelAxis) -> Any:
        if self.mesh is None or self.plan.size(axis) == 1:
            return None
        return self.mesh[axis.value].get_group()

    def _sync_model_axis_gradients(self) -> int:
        """Average gradients of replicated parameters across TP/CP axes.

        Parameters sharded on the current model axis already carry
        placement-aware gradients. Replicas (embeddings, norms, attention in
        the native partial TP plan) require an explicit reduction, including
        FSDP local shards replicated over CP.
        """

        import torch.distributed as dist

        communicated = 0
        for axis in (ParallelAxis.TP, ParallelAxis.CP):
            size = self.plan.size(axis)
            if size == 1:
                continue
            group = self._axis_group(axis)
            for parameter in self.module.parameters():
                gradient = parameter.grad
                if gradient is None or _is_sharded_on(parameter, axis) or _is_sharded_on(gradient, axis):
                    continue
                local_gradient = gradient.to_local() if _is_dtensor(gradient) else gradient
                dist.all_reduce(local_gradient, group=group)
                local_gradient.div_(size)
                communicated += local_gradient.numel() * local_gradient.element_size() * 2
        return communicated

    def _sync_manual_data_parallel_gradients(self) -> int:
        if not self.manual_data_parallel:
            return 0
        import torch.distributed as dist

        group = self._axis_group(ParallelAxis.DP_REPLICATE)
        communicated = 0
        for parameter in self.module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            local_gradient = gradient.to_local() if _is_dtensor(gradient) else gradient
            dist.all_reduce(local_gradient, group=group)
            local_gradient.div_(self.plan.dp_replicate)
            communicated += local_gradient.numel() * local_gradient.element_size() * 2
        return communicated

    def _autocast(self) -> Any:
        import torch

        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _context_parallel(self, inputs: Any, targets: Any, position_ids: Any) -> Any:
        if self.plan.cp == 1:
            return nullcontext()
        try:
            from torch.distributed.tensor.experimental import context_parallel
        except ImportError as error:  # pragma: no cover - depends on the installed torch build
            raise RuntimeError("context parallelism requires torch.distributed.tensor.experimental.") from error
        cp_mesh = self.mesh[ParallelAxis.CP.value]
        return context_parallel(
            cp_mesh,
            buffers=(inputs, targets, position_ids),
            buffer_seq_dims=(1, 1, 1),
            no_restore_buffers=frozenset({inputs, targets, position_ids}),
        )

    def _loss(self, inputs: Any, targets: Any) -> Any:
        import torch

        position_ids = torch.arange(inputs.shape[1], device=inputs.device).expand(inputs.shape[0], -1).clone()
        with self._context_parallel(inputs, targets, position_ids), self._autocast():
            logits = self.module(inputs, position_ids=position_ids, return_all_logits=True)
            return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    def _skipped_receipt(self, loss: float, examples: int, tokens: int) -> StepReceipt:
        return StepReceipt(
            step=self.step,
            loss=loss,
            local_examples=examples,
            local_tokens=tokens,
            microbatches=self.plan.microbatches,
            accumulation_steps=self._pending,
            data_parallel_size=self.plan.data_parallel_size,
            optimizer=str(self.optimizer_receipt["name"]),
            precision=self.precision,
            skipped=True,
            extra={"reason": "gradient_accumulation"},
        )

    def train_batch(self, inputs: Any, targets: Any) -> StepReceipt:
        """Accumulate one local batch and commit when the configured clock fires."""

        import torch

        if self._closed:
            raise RuntimeError("distributed training session is closed.")
        inputs = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        if inputs.shape != targets.shape or inputs.ndim != 2:
            raise ValueError("native language-model training expects equal (batch, sequence) inputs and targets.")
        input_chunks = torch.chunk(inputs, min(self.plan.microbatches, inputs.shape[0]), dim=0)
        target_chunks = torch.chunk(targets, len(input_chunks), dim=0)
        total_targets = max(int(targets.numel()), 1)
        detached_loss = 0.0
        for input_chunk, target_chunk in zip(input_chunks, target_chunks):
            loss = self._loss(input_chunk, target_chunk)
            weight = target_chunk.numel() / total_targets
            (loss * weight / self.plan.gradient_accumulation_steps).backward()
            detached_loss += float(loss.detach()) * weight
        self._pending += 1
        self._pending_examples += int(inputs.shape[0])
        self._pending_tokens += int(targets.numel())
        self._pending_loss_sum += detached_loss * int(targets.numel())
        if self._pending < self.plan.gradient_accumulation_steps:
            return self._skipped_receipt(detached_loss, int(inputs.shape[0]), int(targets.numel()))
        return self._commit()

    def _commit(self, *, partial: bool = False) -> StepReceipt:
        import torch

        if partial and self._pending:
            correction = self.plan.gradient_accumulation_steps / self._pending
            for parameter in self.module.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        collective_bytes = self._sync_model_axis_gradients()
        collective_bytes += self._sync_manual_data_parallel_gradients()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        receipt = StepReceipt(
            step=self.step,
            loss=self._pending_loss_sum / max(self._pending_tokens, 1),
            local_examples=self._pending_examples,
            local_tokens=self._pending_tokens,
            microbatches=self.plan.microbatches,
            accumulation_steps=self._pending,
            data_parallel_size=self.plan.data_parallel_size,
            optimizer=str(self.optimizer_receipt["name"]),
            precision=self.precision,
            collective_bytes=collective_bytes,
            extra={"partial_accumulation": partial, "parameter_layouts": len(self.parameter_layouts)},
        )
        self._pending = 0
        self._pending_examples = 0
        self._pending_tokens = 0
        self._pending_loss_sum = 0.0
        return receipt

    def finish_accumulation(self) -> StepReceipt | None:
        return self._commit(partial=True) if self._pending else None

    def state(self) -> TorchSessionState:
        return TorchSessionState(self.step, self._pending, dict(self.optimizer_receipt))

    def save_checkpoint(
        self,
        path: str,
        *,
        loader_state: Any = None,
        typed_scheduler_state: Any = None,
        extra: dict[str, Any] | None = None,
        asynchronous: bool = False,
    ) -> Any:
        """Save all state needed to resume this session."""

        from mixle.utils.parallel.dcp_checkpoint import async_save_training_state, save_training_state

        if self._pending:
            raise RuntimeError("finish gradient accumulation before checkpointing; pending gradients are not state.")
        kwargs = {
            "step": self.step,
            "scheduler": self.scheduler,
            "loader_state": loader_state,
            "parallel_plan": self.plan,
            "typed_scheduler_state": typed_scheduler_state,
            "extra": {"optimizer_receipt": self.optimizer_receipt, **(extra or {})},
        }
        if asynchronous:
            return async_save_training_state(self.module, self.optimizer, path, **kwargs)
        save_training_state(self.module, self.optimizer, path, **kwargs)
        return None

    def load_checkpoint(self, path: str, *, allow_world_size_change: bool = False) -> dict[str, Any]:
        """Restore this transformed model/optimizer and resume its step clock."""

        from mixle.utils.parallel.dcp_checkpoint import load_training_state

        payload = load_training_state(
            self.module,
            self.optimizer,
            path,
            scheduler=self.scheduler,
            allow_world_size_change=allow_world_size_change,
        )
        self.step = int(payload["step"])
        return payload

    def reduce_sums(self, *values: float) -> tuple[float, ...]:
        """Sum scalar metrics over the two-dimensional data-parallel domain."""

        import torch
        import torch.distributed as dist

        result = torch.tensor(values, dtype=torch.float64, device=self.device)
        for axis in (ParallelAxis.DP_SHARD, ParallelAxis.DP_REPLICATE):
            if self.plan.size(axis) > 1:
                dist.all_reduce(result, group=self._axis_group(axis))
        return tuple(float(value) for value in result.cpu())

    def close(self) -> None:
        self.finish_accumulation()
        self._closed = True


class TorchDistributedBackend:
    """Native backend for Mixle modules that remain ordinary PyTorch modules."""

    capabilities = BackendCapabilities(
        name="torch_native",
        axes=frozenset(
            {
                ParallelAxis.DP_REPLICATE,
                ParallelAxis.DP_SHARD,
                ParallelAxis.TP,
                ParallelAxis.CP,
            }
        ),
        precisions=frozenset({"fp32", "fp16", "bf16"}),
        distributed_optimizer=True,
        reshardable_checkpoint=True,
        elastic_restart=False,
        requirements=("torch>=2.4",),
        incompatible_axis_sets=(frozenset({ParallelAxis.TP, ParallelAxis.CP}),),
    )

    @staticmethod
    def _initialize(plan: ParallelPlan, device: str | Any | None) -> tuple[Any, Any]:
        import torch
        import torch.distributed as dist

        requested = torch.device(device) if device is not None else None
        if requested is None:
            requested = (
                torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        if requested.type == "cuda":
            torch.cuda.set_device(requested)
        if plan.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if requested.type == "cuda" else "gloo")
        actual = dist.get_world_size() if dist.is_initialized() else 1
        plan.validate_world_size(actual)
        mesh = None
        if plan.world_size > 1:
            from torch.distributed.device_mesh import init_device_mesh

            names, shape = plan.mesh
            mesh = init_device_mesh(requested.type, shape, mesh_dim_names=names)
        return requested, mesh

    @staticmethod
    def _apply_tensor_parallel(module: Any, mesh: Any, plan: ParallelPlan) -> Any:
        if plan.tp == 1:
            return module
        if not hasattr(module, "blocks"):
            raise TypeError("torch-native tensor parallelism currently requires a module.blocks transformer.")
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

        tp_mesh = mesh[ParallelAxis.TP.value]
        for block in module.blocks:
            parallelize_module(
                block,
                tp_mesh,
                {
                    "mlp.0": ColwiseParallel(),
                    "mlp.2": RowwiseParallel(),
                },
            )
        return module

    @staticmethod
    def _apply_data_parallel(
        module: Any, mesh: Any, plan: ParallelPlan, device: Any, precision: str
    ) -> tuple[Any, bool]:
        if plan.dp_shard > 1:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            names = tuple(
                axis.value for axis in (ParallelAxis.DP_REPLICATE, ParallelAxis.DP_SHARD) if plan.size(axis) > 1
            )
            dp_mesh = _submesh(mesh, names)
            policy = None
            if precision != "fp32":
                import torch

                dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
            if hasattr(module, "blocks"):
                for block in module.blocks:
                    fully_shard(block, mesh=dp_mesh, mp_policy=policy)
            fully_shard(module, mesh=dp_mesh, mp_policy=policy)
            return module, False
        if plan.dp_replicate > 1:
            if plan.tp > 1 or plan.cp > 1:
                return module, True
            from torch.nn.parallel import DistributedDataParallel

            kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
            return (
                DistributedDataParallel(
                    module,
                    process_group=mesh[ParallelAxis.DP_REPLICATE.value].get_group(),
                    **kwargs,
                ),
                False,
            )
        return module, False

    def prepare(self, module: Any, *, plan: ParallelPlan, **kwargs: Any) -> TorchDistributedSession:
        """Materialize mesh, transforms, optimizer, and resumable session."""

        import torch

        precision = str(kwargs.pop("precision", "fp32")).lower()
        self.capabilities.validate(plan, precision=precision)
        device, mesh = self._initialize(plan, kwargs.pop("device", None))
        if plan.cp > 1 and getattr(module, "block", 0) % plan.cp:
            raise ValueError("the transformer block length must be divisible by cp.")
        if plan.cp > 1 and device.type != "cuda":
            raise RuntimeError("PyTorch context parallelism requires CUDA/NCCL; use Megatron on GPU clusters.")
        if plan.tp > 1 and getattr(module, "d_model", 0) * 4 % plan.tp:
            raise ValueError("the transformer MLP width must be divisible by tp.")
        parameter_layouts = describe_parameter_layouts(module, plan)
        optimizer_spec = kwargs.pop("optimizer", None)
        automatic = optimizer_spec is None or (isinstance(optimizer_spec, str) and optimizer_spec.lower() == "auto")
        optimizer_plan = plan_neural_optimizer(module, sign_stable=False) if automatic else None
        if optimizer_plan is not None and (plan.dp_shard > 1 or plan.tp > 1):
            optimizer_plan = shard_safe_neural_optimizer_plan(optimizer_plan)
        module = module.to(device)
        module = self._apply_tensor_parallel(module, mesh, plan)
        module, manual_data_parallel = self._apply_data_parallel(module, mesh, plan, device, precision)
        if kwargs.pop("compile", False):
            module = torch.compile(module)
        optimizer, optimizer_receipt = resolve_neural_optimizer(
            module,
            optimizer=optimizer_plan if optimizer_plan is not None else optimizer_spec,
            lr=float(kwargs.pop("lr", 1.0e-3)),
            sign_stable=False,
        )
        scheduler_factory = kwargs.pop("scheduler", None)
        scheduler = scheduler_factory(optimizer) if callable(scheduler_factory) else scheduler_factory
        max_grad_norm = kwargs.pop("max_grad_norm", None)
        if kwargs:
            raise TypeError("unexpected torch-native backend options: %s" % ", ".join(sorted(kwargs)))
        module.train()
        return TorchDistributedSession(
            module,
            plan=plan,
            capabilities=self.capabilities,
            mesh=mesh,
            device=device,
            optimizer=optimizer,
            optimizer_receipt=optimizer_receipt,
            parameter_layouts=parameter_layouts,
            precision=precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            manual_data_parallel=manual_data_parallel,
        )


__all__ = [
    "TorchDistributedBackend",
    "TorchDistributedSession",
    "TorchSessionState",
    "describe_parameter_layouts",
]
