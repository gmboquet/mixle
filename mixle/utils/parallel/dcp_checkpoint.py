"""Sharded distributed checkpoints via ``torch.distributed.checkpoint`` (DCP) -- replaces pickle-broadcast at scale.

.. warning::

   **Experimental frontier-training prototype.** The DCP save/load mechanics are exact, but genuine
   multi-rank sharded checkpointing needs a real process group across real devices; at single-process or
   simulated-rank scale this exercises the mechanism, not a production multi-node checkpoint. Not a
   supported production checkpointing API.

The gather-to-root + ``pickle``-broadcast that :class:`TorchRunEncodedData` uses to move a model cannot save a
model that does not fit (and is not folded on) one rank. DCP saves each rank's shard of the (FSDP2-sharded) model
+ optimizer state in parallel to a checkpoint directory, and loads it back sharded -- the standard frontier
checkpoint, and the resume hook for :class:`~mixle.utils.parallel.torch_neural.StreamingTokenEncodedData`.

CUDA / multi-GPU path: correct per the torch 2.4+ DCP + distributed-state-dict APIs, exercised on the cluster.
"""

from __future__ import annotations

from typing import Any


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
