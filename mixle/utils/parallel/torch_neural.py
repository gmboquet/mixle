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
from typing import Any

import numpy as np

from mixle.utils.parallel.planner import EncodedDataHandle


def _torch_dist() -> tuple[Any, Any]:
    import torch

    if not torch.distributed.is_available():
        raise ImportError("StreamingTokenEncodedData requires torch.distributed.")
    return torch, torch.distributed


class StreamingTokenEncodedData(EncodedDataHandle):
    """Per-rank streamed token shard; ``pysp_seq_estimate`` trains the module SPMD with no gather-to-root.

    Pass a token-id array; with ``shard_by_rank`` each rank keeps a disjoint slice. The estimator is a
    :class:`~mixle.models.streaming_transformer_leaf.StreamingTransformerLeafEstimator` carrying the module + lr.
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
    ) -> None:
        self.torch, self.dist = _torch_dist()
        self._owns_pg = False
        self._maybe_init_process_group(init_process_group, backend)
        self.rank = self.dist.get_rank() if self.dist.is_initialized() else 0
        self.world = self.dist.get_world_size() if self.dist.is_initialized() else 1
        ids = np.asarray(token_ids)
        if shard_by_rank and self.world > 1:  # each rank keeps its own disjoint slice resident
            n = len(ids)
            ids = ids[self.rank * n // self.world : (self.rank + 1) * n // self.world]
        self._ids = ids
        self.block = int(block)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.shuffle = bool(shuffle)

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

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """SPMD: stream this rank's shard, train the module; the only collective is the in-backward all-reduce."""
        torch, dist = self.torch, self.dist
        from mixle.data.stream_token_source import stream_token_source
        from mixle.models.streaming_transformer_leaf import StreamingTransformerLeaf

        device = getattr(estimator, "device", "cpu")
        lr = float(getattr(estimator, "lr", 3e-3))
        module = estimator.module.to(device)
        if dist.is_initialized() and self.world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            wrapped = DDP(module)  # in-backward gradient all-reduce; on CUDA: fully_shard(module) for ZeRO-3
        else:
            wrapped = module
        opt = torch.optim.AdamW(wrapped.parameters(), lr=lr)
        ce = torch.nn.CrossEntropyLoss()
        src = stream_token_source(
            self._ids, block=self.block, batch_size=self.batch_size, epochs=self.epochs, shuffle=self.shuffle
        )
        for x, y in src:
            opt.zero_grad()
            loss = ce(
                wrapped(torch.as_tensor(x, dtype=torch.float32).to(device)),
                torch.as_tensor(y, dtype=torch.long).to(device),
            )
            loss.backward()  # <-- the SOLE cross-rank collective (gradient all-reduce / reduce-scatter)
            opt.step()
        return StreamingTransformerLeaf(module, device)  # consistent across ranks -- no gather, no broadcast

    def pysp_seq_initialize(self, estimator: Any, rng: Any, p: float) -> Any:
        from mixle.models.streaming_transformer_leaf import StreamingTransformerLeaf

        return StreamingTransformerLeaf(estimator.module, getattr(estimator, "device", "cpu"))

    def close(self) -> None:
        if self._owns_pg and self.dist.is_initialized():
            self.dist.destroy_process_group()
            self._owns_pg = False
