"""Sharded distributed checkpoints via ``torch.distributed.checkpoint`` (DCP) -- replaces pickle-broadcast at scale.

.. note::

   This is a provisional API with experimental multi-node behavior.
   Complete-state and failure semantics are covered on CPU; resharding and
   throughput at multi-node GPU scale still require retained hardware receipts.

The gather-to-root + ``pickle``-broadcast that :class:`TorchRunEncodedData` uses to move a model cannot save a
model that does not fit (and is not folded on) one rank. DCP saves each rank's shard of the (FSDP2-sharded) model
+ optimizer state in parallel to a checkpoint directory, and loads it back sharded -- the standard frontier
checkpoint, and the resume hook for :class:`~mixle.utils.parallel.torch_neural.StreamingTokenEncodedData`.

CUDA / multi-GPU path: implemented against the torch 2.4+ DCP and
distributed-state-dict APIs, but not verified without retained hardware receipts.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT_VERSION = 1


def _rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sidecar_payload(
    *,
    step: int,
    scheduler: Any,
    scaler: Any,
    loader_state: Any,
    parallel_plan: Any,
    typed_scheduler_state: Any,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": _FORMAT_VERSION,
        "step": int(step),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "loader_state": loader_state.to_dict() if hasattr(loader_state, "to_dict") else loader_state,
        "parallel_plan": parallel_plan.as_dict() if hasattr(parallel_plan, "as_dict") else parallel_plan,
        "typed_scheduler_state": typed_scheduler_state,
        "rng": _rng_state(),
        "extra": dict(extra or {}),
    }


def _write_sidecar(path: Path, rank: int, payload: dict[str, Any]) -> None:
    import torch

    temporary = path / ("rank-%05d.pt.tmp" % rank)
    target = path / ("rank-%05d.pt" % rank)
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _finalize_checkpoint(path: Path, *, rank: int, world_size: int) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        manifest = {
            "format_version": _FORMAT_VERSION,
            "world_size": world_size,
            "rank_sidecars": ["rank-%05d.pt" % index for index in range(world_size)],
        }
        temporary = path / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path / "manifest.json")
        success_temporary = path / "_SUCCESS.tmp"
        success_temporary.write_text("complete\n", encoding="ascii")
        os.replace(success_temporary, path / "_SUCCESS")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _dcp_state(module: Any, optimizer: Any) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(module, optimizer)
    return {"model": model_state, "optimizer": optimizer_state}


def save_sharded(module: Any, optimizer: Any, path: str) -> None:
    """Save a sharded ``(model, optimizer)`` checkpoint to ``path`` -- every rank writes its own shard in parallel."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_sd, optim_sd = get_state_dict(module, optimizer)
    dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(path))


def load_sharded(module: Any, optimizer: Any, path: str) -> None:
    """Load a sharded checkpoint from ``path`` into ``module`` + ``optimizer`` in place (resumable training)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_sd, optim_sd = get_state_dict(module, optimizer)  # templates with the right sharded shapes
    dcp.load({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(path))
    set_state_dict(module, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)


def save_training_state(
    module: Any,
    optimizer: Any,
    path: str,
    *,
    step: int,
    scheduler: Any = None,
    scaler: Any = None,
    loader_state: Any = None,
    parallel_plan: Any = None,
    typed_scheduler_state: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Synchronously save complete, reshardable training state."""

    import torch.distributed.checkpoint as dcp

    rank, world_size = _rank_world()
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    success = destination / "_SUCCESS"
    if rank == 0 and success.exists():
        success.unlink()
    payload = _sidecar_payload(
        step=step,
        scheduler=scheduler,
        scaler=scaler,
        loader_state=loader_state,
        parallel_plan=parallel_plan,
        typed_scheduler_state=typed_scheduler_state,
        extra=extra,
    )
    dcp.save(_dcp_state(module, optimizer), checkpoint_id=str(destination))
    _write_sidecar(destination, rank, payload)
    _finalize_checkpoint(destination, rank=rank, world_size=world_size)


def load_training_state(
    module: Any,
    optimizer: Any,
    path: str,
    *,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
    allow_world_size_change: bool = False,
) -> dict[str, Any]:
    """Restore complete training state and return its non-tensor metadata."""

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    source = Path(path)
    if not (source / "_SUCCESS").is_file():
        raise RuntimeError("checkpoint is incomplete: %s has no _SUCCESS marker." % source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("format_version", -1)) != _FORMAT_VERSION:
        raise RuntimeError("unsupported training checkpoint format version.")
    rank, world_size = _rank_world()
    saved_world_size = int(manifest["world_size"])
    changed = world_size != saved_world_size
    if changed and not allow_world_size_change:
        raise RuntimeError(
            "checkpoint world_size changed from %d to %d; pass allow_world_size_change=True to reshard "
            "model state and rebuild rank-local loader state." % (saved_world_size, world_size)
        )
    model_state, optimizer_state = get_state_dict(module, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=str(source))
    set_state_dict(
        module,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    sidecar_rank = rank if not changed else rank % saved_world_size
    payload = torch.load(source / ("rank-%05d.pt" % sidecar_rank), weights_only=False)
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and not changed:
        _restore_rng_state(payload["rng"])
    payload = dict(payload)
    payload["world_size_changed"] = changed
    payload["saved_world_size"] = saved_world_size
    return payload


@dataclass
class AsyncTrainingCheckpoint:
    """DCP asynchronous save plus atomic sidecar/manifest finalization."""

    future: Any
    path: Path
    rank: int
    world_size: int
    payload: dict[str, Any]
    _complete: bool = False

    def wait(self) -> None:
        if self._complete:
            return
        wait = getattr(self.future, "wait", None)
        if callable(wait):
            wait()
        else:
            self.future.result()
        _write_sidecar(self.path, self.rank, self.payload)
        _finalize_checkpoint(self.path, rank=self.rank, world_size=self.world_size)
        self._complete = True

    @property
    def done(self) -> bool:
        # I/O completion is not checkpoint completion until sidecars and the
        # manifest marker have been committed by ``wait``.
        return self._complete


def async_save_training_state(
    module: Any,
    optimizer: Any,
    path: str,
    *,
    step: int,
    scheduler: Any = None,
    scaler: Any = None,
    loader_state: Any = None,
    parallel_plan: Any = None,
    typed_scheduler_state: Any = None,
    extra: dict[str, Any] | None = None,
) -> AsyncTrainingCheckpoint:
    """Start DCP's native async save; :meth:`wait` surfaces write failures."""

    import torch.distributed.checkpoint as dcp

    rank, world_size = _rank_world()
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    success = destination / "_SUCCESS"
    if rank == 0 and success.exists():
        success.unlink()
    payload = _sidecar_payload(
        step=step,
        scheduler=scheduler,
        scaler=scaler,
        loader_state=loader_state,
        parallel_plan=parallel_plan,
        typed_scheduler_state=typed_scheduler_state,
        extra=extra,
    )
    future = dcp.async_save(_dcp_state(module, optimizer), checkpoint_id=str(destination))
    return AsyncTrainingCheckpoint(future, destination, rank, world_size, payload)


__all__ = [
    "AsyncTrainingCheckpoint",
    "async_save_training_state",
    "load_sharded",
    "load_training_state",
    "save_sharded",
    "save_training_state",
]
