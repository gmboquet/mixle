"""torchrun SPMD backend for distributed estimation.

The handle mirrors ``MPIEncodedData`` using ``torch.distributed`` collectives:
each rank encodes its local data shard once, accumulates sufficient statistics
against that resident encoding, rank 0 folds/key-ties/runs the M-step, and the
result is broadcast back to every rank.
"""

import io
import os
import pickle
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.utils.parallel.planner import EncodedDataHandle

__all__ = ["TorchRunEncodedData", "torchrun_out"]

_PROTO = pickle.HIGHEST_PROTOCOL


def _torch_dist():
    try:
        import torch
    except ImportError as e:
        raise ImportError("TorchRunEncodedData requires torch.") from e
    if not torch.distributed.is_available():
        raise ImportError("TorchRunEncodedData requires torch.distributed.")
    return torch, torch.distributed


def torchrun_out(root: int = 0):
    """sys.stdout on the root rank, a discarded buffer elsewhere."""
    try:
        _, dist = _torch_dist()
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = int(os.environ.get("RANK") or 0)
    except ImportError:
        rank = 0
    return sys.stdout if rank == root else io.StringIO()


class TorchRunEncodedData(EncodedDataHandle):
    """Encoded-data handle sharded across torchrun ranks.

    Args:
        data (Optional[Sequence]): Raw observations. With ``root_only=False``
            every rank passes the same full dataset and keeps ``data[rank::world]``.
            With ``root_only=True`` only the root rank needs real data.
        estimator: Used to build the encoder when ``encoder`` is not given.
        model: Optional model used to build the encoder.
        encoder: Explicit data encoder.
        sub_chunks (int): Encoded sub-chunks per rank.
        group: Optional torch.distributed process group.
        root (int): Rank that folds statistics and runs the M-step.
        root_only (bool): Root-only data input mode.
        init_process_group (bool): Initialize ``torch.distributed`` from the
            torchrun environment when needed.
        backend (Optional[str]): Process-group backend. Defaults to ``nccl`` on
            CUDA and ``gloo`` otherwise.
    """

    def __init__(
        self,
        data: Sequence[Any] | None,
        estimator=None,
        model=None,
        encoder=None,
        sub_chunks: int = 1,
        group: Any | None = None,
        root: int = 0,
        root_only: bool = False,
        init_process_group: bool = True,
        backend: str | None = None,
    ):
        self.torch, self.dist = _torch_dist()
        self.group = group
        self.root = int(root)
        self._owns_process_group = False

        if encoder is None:
            if model is not None and callable(getattr(model, "dist_to_encoder", None)):
                encoder = model.dist_to_encoder()
            elif estimator is not None:
                encoder = estimator.accumulator_factory().make().acc_to_encoder()
        if encoder is None:
            raise ValueError("TorchRunEncodedData requires an encoder, model, or estimator.")

        self._maybe_init_process_group(init_process_group, backend)
        self.rank = self.dist.get_rank(group=self.group) if self.dist.is_initialized() else 0
        self.world = self.dist.get_world_size(group=self.group) if self.dist.is_initialized() else 1

        shard = self._local_shard(data, root_only)
        nobs = len(shard)
        part_count = max(1, min(int(sub_chunks), nobs)) if nobs else 1
        self._enc_chunks = []
        for i in range(part_count):
            part = [shard[j] for j in range(i, nobs, part_count)]
            if part:
                self._enc_chunks.append((len(part), encoder.seq_encode(part)))

        self.size = float(nobs)
        if self.world > 1:
            self.size = self._all_reduce_pair(float(nobs), 0.0)[0]

    def _maybe_init_process_group(self, init_process_group: bool, backend: str | None) -> None:
        if self.dist.is_initialized() or not init_process_group:
            return
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            return
        if backend is None:
            backend = "nccl" if self.torch.cuda.is_available() else "gloo"
        if backend == "nccl" and self.torch.cuda.is_available():
            self.torch.cuda.set_device(int(os.environ.get("LOCAL_RANK") or 0))
        self.dist.init_process_group(backend=backend)
        self._owns_process_group = True

    def _local_shard(self, data: Sequence[Any] | None, root_only: bool) -> Sequence[Any]:
        if self.world == 1:
            if data is None:
                raise ValueError("TorchRunEncodedData requires data in single-rank mode.")
            return data
        if root_only:
            if self.rank == self.root:
                if data is None:
                    raise ValueError("root_only=True requires data on the root rank.")
                shards = [[data[j] for j in range(i, len(data), self.world)] for i in range(self.world)]
            else:
                shards = None
            return self._scatter_object(shards)
        if data is None:
            raise ValueError("every rank must pass data when root_only=False.")
        return [data[j] for j in range(self.rank, len(data), self.world)]

    def _scatter_object(self, objects: Sequence[Any] | None) -> Any:
        if hasattr(self.dist, "scatter_object_list"):
            out = [None]
            in_list = list(objects) if self.rank == self.root else None
            self.dist.scatter_object_list(out, in_list, src=self.root, group=self.group)
            return out[0]
        box = [list(objects) if self.rank == self.root else None]
        self.dist.broadcast_object_list(box, src=self.root, group=self.group)
        return box[0][self.rank]

    def _broadcast_object(self, value: Any) -> Any:
        if self.world == 1:
            return value
        box = [pickle.dumps(value, protocol=_PROTO) if self.rank == self.root else None]
        self.dist.broadcast_object_list(box, src=self.root, group=self.group)
        return pickle.loads(box[0])

    def _gather_object(self, value: Any) -> Sequence[Any] | None:
        if self.world == 1:
            return [value]
        gathered = [None for _ in range(self.world)]
        self.dist.all_gather_object(gathered, pickle.dumps(value, protocol=_PROTO), group=self.group)
        if self.rank != self.root:
            return None
        return [pickle.loads(raw) for raw in gathered]

    def _all_reduce_pair(self, count: float, total: float) -> tuple[float, float]:
        if self.world == 1:
            return count, total
        backend = self.dist.get_backend(group=self.group)
        device = "cuda" if backend == "nccl" and self.torch.cuda.is_available() else "cpu"
        values = self.torch.tensor([count, total], dtype=self.torch.float64, device=device)
        self.dist.all_reduce(values, op=self.dist.ReduceOp.SUM, group=self.group)
        return float(values[0].cpu().item()), float(values[1].cpu().item())

    def _local_update(self, estimator, model) -> tuple[float, Any]:
        accumulator = estimator.accumulator_factory().make()
        count = 0.0
        for sz, enc in self._enc_chunks:
            count += sz
            accumulator.seq_update(enc, np.ones(sz, dtype=np.float64), model)
        return count, accumulator.value()

    def _fold_value_and_share(self, estimator, local: tuple[float, Any]) -> tuple[float, Any]:
        gathered = self._gather_object(local)
        if self.rank == self.root:
            accumulator = estimator.accumulator_factory().make()
            nobs = 0.0
            for count, stats in gathered:
                nobs += count
                accumulator.combine(stats)
            stats_dict = {}
            accumulator.key_merge(stats_dict)
            accumulator.key_replace(stats_dict)
            payload = nobs, accumulator.value()
        else:
            payload = None
        return self._broadcast_object(payload)

    def _fold_model_and_share(self, estimator, local: tuple[float, Any]):
        nobs, value = self._fold_value_and_share(estimator, local)
        if self.rank == self.root:
            model = estimator.estimate(nobs, value)
        else:
            model = None
        return self._broadcast_object(model)

    def pysp_seq_estimate(self, estimator, prev_estimate):
        """One distributed EM step; every rank returns the identical model."""
        model = self._broadcast_object(prev_estimate)
        return self._fold_model_and_share(estimator, self._local_update(estimator, model))

    def pysp_seq_initialize(self, estimator, rng: np.random.RandomState, p: float):
        """Distributed randomized initialization; identical model on all ranks."""
        if self.rank == self.root:
            seeds = [int(s) for s in rng.randint(2**31, size=self.world)]
        else:
            seeds = None
        seed = self._scatter_object(seeds) if self.world > 1 else int(rng.randint(2**31))
        rng_loc = np.random.RandomState(seed)
        rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))
        accumulator = estimator.accumulator_factory().make()
        count = 0.0
        for sz, enc in self._enc_chunks:
            weights = np.zeros(sz, dtype=np.float64)
            weights[rng_w.rand(sz) <= p] = 1.0
            count += float(weights.sum())
            accumulator.seq_initialize(enc, weights, rng_loc)
        return self._fold_model_and_share(estimator, (count, accumulator.value()))

    def pysp_seq_log_density_sum(self, estimate) -> tuple[float, float]:
        """Allreduced count and summed log density."""
        count = 0.0
        total = 0.0
        for sz, enc in self._enc_chunks:
            count += sz
            total += float(np.asarray(estimate.seq_log_density(enc), dtype=np.float64).sum())
        return self._all_reduce_pair(count, total)

    def pysp_stream_accumulate(self, estimator, model) -> tuple[float, Any]:
        """Globally folded batch sufficient statistics for streaming EM."""
        model = self._broadcast_object(model)
        return self._fold_value_and_share(estimator, self._local_update(estimator, model))

    def __len__(self) -> int:
        return int(self.size)

    def close(self) -> None:
        """Destroy the owned torch distributed process group, if any."""
        if self._owns_process_group and self.dist.is_initialized():
            self.dist.destroy_process_group(self.group)
            self._owns_process_group = False
