"""Lightning-backed encoded-data handle for mini-batch / stochastic EM (WS-C2).

``LightningEncodedData`` plugs PyTorch Lightning's data tooling into mixle's encoded-data backend
registry (``planner.encoded_data(..., backend="lightning")``). Full-data EM operations delegate to a
resident :class:`~mixle.utils.parallel.planner.LocalEncodedData` (identical results to ``backend="local"``); the
Lightning-specific value is **mini-batch iteration** via a :class:`lightning.pytorch.LightningDataModule`
+ ``DataLoader`` (shuffling, batching, multi-worker collation), which drives stochastic / mini-batch EM
through :class:`~mixle.inference.streaming.StreamingEstimator`.

Lightning is an optional dependency: this module is imported only when the ``"lightning"`` backend is
requested, so the rest of mixle (and CI without Lightning installed) is unaffected.
"""

from __future__ import annotations

from collections.abc import Iterator
from operator import index
from typing import Any

from numpy.random import RandomState

from mixle.stats.compute.pdist import DataSequenceEncoder
from mixle.utils.exact import require_exact_bool
from mixle.utils.parallel.planner import EncodedDataHandle, LocalEncodedData


def _resolve_encoder(estimator: Any, model: Any, encoder: DataSequenceEncoder | None) -> DataSequenceEncoder:
    if encoder is not None:
        return encoder
    if model is not None and callable(getattr(model, "dist_to_encoder", None)):
        return model.dist_to_encoder()
    if estimator is not None:
        return estimator.accumulator_factory().make().acc_to_encoder()
    raise ValueError("LightningEncodedData requires an encoder, model, or estimator.")


def _epoch_seed(seed: int, epoch: int) -> int:
    """Derive a stable, distinct PyTorch seed for a data epoch."""
    return (seed + epoch) % (2**63 - 1)


class _IndexDataset:
    """Pickle-safe row-index dataset for spawned DataLoader workers."""

    def __init__(self, num_rows: int):
        self.num_rows = num_rows

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, i: int) -> int:
        return int(i)


def _collate_indices(batch) -> list[int]:
    """Convert a DataLoader index batch into plain Python integers."""
    return [int(i) for i in batch]


def _make_datamodule(num_rows: int, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    """Return a LightningDataModule whose train DataLoader yields shuffled row-index batches."""
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader

    class _EncodedDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self._epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self._epoch = epoch

        def train_dataloader(self) -> DataLoader:
            epoch = self._epoch
            self._epoch += 1
            generator = torch.Generator().manual_seed(_epoch_seed(seed, epoch))
            return DataLoader(
                _IndexDataset(num_rows),
                batch_size=int(batch_size),
                shuffle=require_exact_bool(shuffle, "shuffle"),
                generator=generator,
                num_workers=num_workers,
                collate_fn=_collate_indices,
            )

    return _EncodedDataModule()


class LightningEncodedData(EncodedDataHandle):
    """Encoded-data handle that mini-batches via a Lightning ``DataModule`` for stochastic EM."""

    def __init__(
        self,
        data: Any,
        estimator: Any | None = None,
        model: Any | None = None,
        encoder: DataSequenceEncoder | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        num_workers: int = 0,
        sub_chunks: int = 1,
        **_: Any,
    ) -> None:
        rows = list(data)
        if not rows:
            raise ValueError("LightningEncodedData requires non-empty data.")
        self.encoder = _resolve_encoder(estimator, model, encoder)
        self._rows = rows
        self.size = len(rows)
        if batch_size is None:
            self.batch_size = max(1, self.size // 10)
        else:
            try:
                self.batch_size = index(batch_size)
            except TypeError as exc:
                raise ValueError("batch_size must be a positive integer or None.") from exc
            if isinstance(batch_size, bool) or self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer or None.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool.")
        try:
            self.seed = index(seed)
        except TypeError as exc:
            raise ValueError("seed must be a nonnegative integer.") from exc
        if isinstance(seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        try:
            self.num_workers = index(num_workers)
        except TypeError as exc:
            raise ValueError("num_workers must be a nonnegative integer.") from exc
        if isinstance(num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        self.shuffle = shuffle
        # Full-data EM operations reuse the local handle (so backend="lightning" matches "local").
        self._local = LocalEncodedData(
            rows, estimator=estimator, model=model, encoder=self.encoder, sub_chunks=sub_chunks
        )
        self._datamodule = _make_datamodule(self.size, self.batch_size, self.shuffle, self.seed, self.num_workers)

    # -- full-data orchestrator contract: delegate to the resident local handle ------------
    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Delegate full-data log-density summation to the resident local handle."""
        return self._local.pysp_seq_log_density_sum(estimate)

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Delegate full-data estimation to the resident local handle."""
        return self._local.pysp_seq_estimate(estimator, prev_estimate)

    def pysp_seq_initialize(self, estimator: Any, rng: RandomState, p: float) -> Any:
        """Delegate full-data initialization to the resident local handle."""
        return self._local.pysp_seq_initialize(estimator, rng, p)

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Delegate full-data streaming accumulation to the resident local handle."""
        return self._local.pysp_stream_accumulate(estimator, model)

    # -- Lightning-specific mini-batch iteration -------------------------------------------
    @property
    def datamodule(self):
        """Return the underlying ``lightning.pytorch.LightningDataModule``."""
        return self._datamodule

    def minibatches(self, *, epoch: int | None = None) -> Iterator[list[Any]]:
        """Yield one reproducibly shuffled epoch of raw-observation mini-batches."""
        if epoch is not None:
            try:
                epoch = index(epoch)
            except TypeError as exc:
                raise ValueError("epoch must be a nonnegative integer or None.") from exc
            if isinstance(epoch, bool) or epoch < 0:
                raise ValueError("epoch must be a nonnegative integer or None.")
            self._datamodule.set_epoch(epoch)
        for index_batch in self._datamodule.train_dataloader():
            yield [self._rows[i] for i in index_batch]

    def stochastic_em(
        self, estimator: Any, *, epochs: int = 5, schedule: Any | None = None, init_p: float = 0.2, seed: int = 0
    ) -> Any:
        """Fit ``estimator`` by mini-batch stochastic EM over the Lightning DataLoader batches.

        Runs ``epochs`` passes, feeding each DataLoader mini-batch to a
        :class:`~mixle.inference.streaming.StreamingEstimator` (decayed accumulator + M-step). Returns the
        fitted model.
        """
        from mixle.inference.streaming import StreamingEstimator

        stream = StreamingEstimator(estimator, schedule=schedule, init_p=init_p, rng=RandomState(seed))
        model = None
        for epoch in range(int(epochs)):
            for batch in self.minibatches(epoch=epoch):
                model = stream.update(batch)
        return model

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        """Release resources owned by the resident local handle."""
        self._local.close()
