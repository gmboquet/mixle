"""Distributed STREAMING neural-training handle -- the inverted sibling of :class:`TorchRunEncodedData`.

``TorchRunEncodedData`` gathers each rank's sufficient statistic to root, runs the M-step on root, and pickle-
broadcasts the model back -- structurally the wrong shape for a sharded gradient update (it cannot scale past a
model that fits, and is folded on, one rank). :class:`StreamingTokenEncodedData` inverts the reduction: each rank
keeps its OWN streamed token shard resident and trains the estimator's module SPMD, and the SOLE cross-rank
collective is the in-backward gradient all-reduce (DDP) / reduce-scatter (FSDP2) -- never a gather-to-root + a
pickle-broadcast. Each rank ends with the same (DDP/FSDP2-consistent) model, so ``seq_estimate`` needs no gather.

It plugs into the same duck-typed dispatch (``seq_estimate`` calls ``handle.pysp_seq_estimate``). Run under
torchrun (RANK/WORLD_SIZE set); the process group is initialized from the environment. On CUDA, swap the DDP wrap
for ``torch.distributed.fsdp.fully_shard`` (ZeRO-3) -- the architecture is identical, that is the one-line change.
"""

from __future__ import annotations

import os
from numbers import Integral
from typing import Any

import numpy as np

from mixle.utils.parallel.planner import EncodedDataHandle
from mixle.utils.vector import validate_initialization_probability


def _torch_dist() -> tuple[Any, Any]:
    import torch

    if not torch.distributed.is_available():
        raise ImportError("StreamingTokenEncodedData requires torch.distributed.")
    return torch, torch.distributed


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _rank_window_schedule(
    n_tokens: int,
    world_size: int,
    *,
    block: int,
    batch_size: int,
    shard_by_rank: bool,
) -> tuple[tuple[int, ...], int]:
    """Return real target counts per rank and one equal step count for every rank."""
    if shard_by_rank and world_size > 1:
        shard_lengths = tuple(
            ((rank + 1) * n_tokens // world_size) - (rank * n_tokens // world_size)
            for rank in range(world_size)
        )
    else:
        shard_lengths = (n_tokens,) * world_size
    windows = tuple(max(0, length - block) for length in shard_lengths)
    steps = max(((count + batch_size - 1) // batch_size for count in windows), default=0)
    return windows, steps


def _global_targets_for_step(window_counts: tuple[int, ...], step: int, batch_size: int) -> int:
    return sum(min(batch_size, max(0, count - step * batch_size)) for count in window_counts)


class StreamingTokenEncodedData(EncodedDataHandle):
    """Per-rank streamed token shard; ``pysp_seq_estimate`` trains the module SPMD with no gather-to-root.

    Pass a token-id array; with ``shard_by_rank`` each rank keeps a disjoint slice. The estimator is a
    :class:`~mixle.models.streaming_transformer_leaf.StreamingTransformerLeafEstimator` carrying the module + lr.
    All ranks execute the same globally derived step schedule. Short and partial shards use masked padding,
    so no rank can leave a DDP/FSDP collective early; :attr:`schedule_receipt` discloses real, masked, and
    shard-boundary-excluded target counts.
    """

    def __init__(
        self,
        token_ids: Any,
        *,
        block: int,
        batch_size: int,
        epochs: int = 1,
        shuffle: bool = True,
        shard_by_rank: bool = True,
        init_process_group: bool = True,
        backend: str | None = None,
        parallel: str = "auto",
        precision: str = "fp32",
        activation_checkpointing: bool = False,
        tp_size: int = 1,
        pp_size: int = 1,
        cp_size: int = 1,
    ) -> None:
        self.torch, self.dist = _torch_dist()
        self._owns_pg = False
        # parallel: "auto" -> FSDP2 (ZeRO-3) on CUDA, DDP on CPU; precision: "fp32"|"bf16" (fp8 = torchao, vendored);
        # activation_checkpointing: re-materialize block activations in backward (memory for compute) -- all of this
        # is the CUDA scale-out path, off by default so the validated CPU/gloo (DDP, fp32) path is unchanged.
        self.parallel = parallel
        self.precision = precision
        self.activation_checkpointing = bool(activation_checkpointing)
        # tp_size/pp_size/cp_size: the F1 N-D-parallelism knobs, ORTHOGONAL to this handle's data-parallel
        # axis (DDP/FSDP2 above). The plan is validated against the real module in `pysp_seq_estimate`
        # (fails fast on a bad plan); wiring it into real per-axis NCCL process groups alongside FSDP2 is
        # the multi-GPU piece that this CPU/gloo-validated handle does not execute -- see
        # `mixle/experimental/tensor_pipeline_context_parallel.py` for the sharding/reconstruction
        # mechanism itself, which IS implemented and tested at small scale (kept under
        # mixle.experimental: its own NotImplementedError gate below is the reason why).
        self.tp_size = int(tp_size)
        self.pp_size = int(pp_size)
        self.cp_size = int(cp_size)
        self._maybe_init_process_group(init_process_group, backend)
        self.rank = self.dist.get_rank() if self.dist.is_initialized() else 0
        self.world = self.dist.get_world_size() if self.dist.is_initialized() else 1
        self.block = _positive_int(block, "block")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.epochs = _positive_int(epochs, "epochs", allow_zero=True)
        ids = np.asarray(token_ids)
        if ids.ndim != 1:
            raise ValueError(f"token_ids must be one-dimensional, got shape {ids.shape}")
        n_tokens = len(ids)
        if shard_by_rank and self.world > 1:  # each rank keeps its own disjoint slice resident
            ids = ids[self.rank * n_tokens // self.world : (self.rank + 1) * n_tokens // self.world]
        self._ids = ids
        self.shuffle = bool(shuffle)
        self._window_counts, self.steps_per_epoch = _rank_window_schedule(
            n_tokens,
            self.world,
            block=self.block,
            batch_size=self.batch_size,
            shard_by_rank=shard_by_rank,
        )
        self.local_target_tokens = self._window_counts[self.rank]
        self.masked_target_tokens = (
            self.steps_per_epoch * self.batch_size - self.local_target_tokens
        ) * self.epochs
        global_unsharded_targets = max(0, n_tokens - self.block)
        self.excluded_target_tokens = (
            max(0, global_unsharded_targets - sum(self._window_counts)) * self.epochs
            if shard_by_rank
            else 0
        )
        self.schedule_receipt = {
            "world_size": self.world,
            "window_counts_by_rank": self._window_counts,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "local_masked_target_tokens": self.masked_target_tokens,
            "global_real_target_tokens_per_epoch": sum(self._window_counts),
            "excluded_target_tokens": self.excluded_target_tokens,
            "policy": "pad-and-mask-to-global-max-steps",
        }

    def _maybe_init_process_group(self, init_process_group: bool, backend: str | None) -> None:
        if self.dist.is_initialized() or not init_process_group:
            return
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            return  # not launched under torchrun -- run single-process
        if backend is None:
            backend = "nccl" if self.torch.cuda.is_available() else "gloo"
        if backend == "nccl" and self.torch.cuda.is_available():
            self.torch.cuda.set_device(int(os.environ.get("LOCAL_RANK") or 0))
        self.dist.init_process_group(backend=backend)
        self._owns_pg = True

    def _resolve_parallel(self, device: str) -> str:
        use_dist = self.dist.is_initialized() and self.world > 1
        if self.parallel != "auto":
            return self.parallel
        if not use_dist:
            return "none"
        return "fsdp" if device.startswith("cuda") else "ddp"

    def _wrap_for_scale(self, module: Any, device: str) -> Any:
        """Wrap for distributed training: FSDP2 (ZeRO-3) on CUDA, DDP on CPU. The validated CPU path is DDP/fp32.

        The CUDA branch (FSDP2 ``fully_shard`` per block + root, bf16 ``MixedPrecisionPolicy``, activation
        checkpointing) is the cluster path -- correct per the torch 2.4+ APIs but only exercised on multi-GPU.
        """
        torch = self.torch
        mode = self._resolve_parallel(device)
        if self.activation_checkpointing and hasattr(module, "blocks"):
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

            for i, blk in enumerate(module.blocks):
                module.blocks[i] = checkpoint_wrapper(blk)
        if mode == "fsdp":  # CUDA: per-parameter sharded ZeRO-3 (the reduce-scatter happens in backward)
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            kw = {"mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16)} if self.precision == "bf16" else {}
            if hasattr(module, "blocks"):
                for blk in module.blocks:
                    fully_shard(blk, **kw)
            fully_shard(module, **kw)
            return module  # FSDP2 shards in place
        if mode == "ddp":  # CPU/gloo: replicate + in-backward gradient all-reduce (the validated path)
            from torch.nn.parallel import DistributedDataParallel as DDP

            return DDP(module)
        return module

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """SPMD: stream this rank's shard, train the module; the only collective is the in-backward all-reduce."""
        torch = self.torch
        from mixle.data.stream_token_source import stream_token_source
        from mixle.models.streaming_transformer_leaf import StreamingTransformerLeaf

        device = getattr(estimator, "device", "cpu")
        lr = float(getattr(estimator, "lr", 3e-3))
        module = estimator.module.to(device)
        if self.tp_size > 1 or self.pp_size > 1 or self.cp_size > 1:
            # This module is stable-namespace, so it must not pull mixle.experimental into the stable
            # import graph via a static import (enforced by experimental_boundary_test) -- resolve the
            # validator dynamically through importlib on a string instead, same bridge mixle/program.py
            # uses to reach mixle.experimental.program.
            import importlib

            validate_tp_pp_cp_plan = importlib.import_module(
                "mixle.experimental.tensor_pipeline_context_parallel"
            ).validate_tp_pp_cp_plan

            validate_tp_pp_cp_plan(module, self.tp_size, self.pp_size, self.cp_size)
            # The plan is validated but per-axis process groups are NOT integrated into training here:
            # execution below is data-parallel (DDP) only. Running silently would replicate a model the
            # caller asked to shard -- OOM or a wrong scaling claim, not what they requested. Fail loudly
            # unless the caller explicitly opts into the validated-plan / data-parallel-only path.
            if os.environ.get("MIXLE_EXPERIMENTAL_NDPARALLEL") != "1":
                raise NotImplementedError(
                    "tp_size/pp_size/cp_size > 1 validates the N-D-parallel plan, but per-axis process "
                    "groups are not integrated into training -- execution would be data-parallel only, "
                    "silently ignoring the requested tensor/pipeline/context sharding. Set "
                    "MIXLE_EXPERIMENTAL_NDPARALLEL=1 to proceed data-parallel-only with the plan "
                    "validated, or leave tp_size/pp_size/cp_size at 1."
                )
        wrapped = self._wrap_for_scale(module, device)
        opt = torch.optim.AdamW(wrapped.parameters(), lr=lr)
        autocast_dev = "cuda" if device.startswith("cuda") else "cpu"
        distributed_scale = float(self.world if self.dist.is_initialized() and self.world > 1 else 1)
        for epoch in range(self.epochs):
            source = iter(
                stream_token_source(
                    self._ids,
                    block=self.block,
                    batch_size=self.batch_size,
                    epochs=1,
                    shuffle=self.shuffle,
                    seed=epoch,
                )
            )
            for step in range(self.steps_per_epoch):
                try:
                    x, y = next(source)
                except StopIteration:
                    x = np.empty((0, self.block), dtype=np.int64)
                    y = np.empty((0,), dtype=np.int64)
                real_targets = len(y)
                padded_x = np.zeros((self.batch_size, self.block), dtype=np.int64)
                padded_y = np.zeros((self.batch_size,), dtype=np.int64)
                mask = np.zeros((self.batch_size,), dtype=np.float32)
                if real_targets:
                    padded_x[:real_targets] = x
                    padded_y[:real_targets] = y
                    mask[:real_targets] = 1.0
                global_targets = _global_targets_for_step(self._window_counts, step, self.batch_size)
                if global_targets <= 0:
                    raise RuntimeError("distributed token schedule produced an empty synchronized step")

                opt.zero_grad()
                with torch.autocast(
                    device_type=autocast_dev,
                    dtype=torch.bfloat16,
                    enabled=(self.precision == "bf16"),
                ):
                    logits = wrapped(torch.as_tensor(padded_x, dtype=torch.float32).to(device))
                    per_target = torch.nn.functional.cross_entropy(
                        logits,
                        torch.as_tensor(padded_y, dtype=torch.long).to(device),
                        reduction="none",
                    )
                    local_sum = (
                        per_target * torch.as_tensor(mask, dtype=per_target.dtype).to(device)
                    ).sum()
                    # DDP/FSDP averages rank gradients. Scaling each local sum by world/global
                    # count makes the post-collective gradient the exact global target mean.
                    loss = local_sum * (distributed_scale / float(global_targets))
                loss.backward()
                opt.step()
        return StreamingTransformerLeaf(module, device)  # consistent across ranks -- no gather, no broadcast

    def pysp_seq_initialize(self, estimator: Any, rng: Any, p: float) -> Any:
        """Return an initial streaming transformer leaf without distributed fitting."""
        from mixle.models.streaming_transformer_leaf import StreamingTransformerLeaf

        validate_initialization_probability(p)
        return StreamingTransformerLeaf(estimator.module, getattr(estimator, "device", "cpu"))

    def close(self) -> None:
        """Destroy the owned process group, if this handle created one."""
        if self._owns_pg and self.dist.is_initialized():
            self.dist.destroy_process_group()
            self._owns_pg = False
