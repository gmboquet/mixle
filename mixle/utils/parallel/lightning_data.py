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
from typing import Any

from numpy.random import RandomState

from mixle.stats.compute.pdist import DataSequenceEncoder
from mixle.utils.parallel.planner import EncodedDataHandle, LocalEncodedData


def _resolve_encoder(estimator: Any, model: Any, encoder: DataSequenceEncoder | None) -> DataSequenceEncoder:
    if encoder is not None:
        return encoder
    if model is not None and callable(getattr(model, "dist_to_encoder", None)):
        return model.dist_to_encoder()
    if estimator is not None:
        return estimator.accumulator_factory().make().acc_to_encoder()
    raise ValueError("LightningEncodedData requires an encoder, model, or estimator.")


def _make_datamodule(num_rows: int, batch_size: int, shuffle: bool, seed: int):
    """Return a LightningDataModule whose train DataLoader yields shuffled row-index batches."""
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _IndexDataset(Dataset):
        def __len__(self) -> int:
            return num_rows

        def __getitem__(self, i: int) -> int:
            return int(i)

    class _EncodedDataModule(pl.LightningDataModule):
        def train_dataloader(self) -> DataLoader:
            generator = torch.Generator().manual_seed(int(seed))
            return DataLoader(
                _IndexDataset(),
                batch_size=int(batch_size),
                shuffle=bool(shuffle),
                generator=generator,
                collate_fn=lambda batch: [int(i) for i in batch],
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
        sub_chunks: int = 1,
        **_: Any,
    ) -> None:
        rows = list(data)
        if not rows:
            raise ValueError("LightningEncodedData requires non-empty data.")
        self.encoder = _resolve_encoder(estimator, model, encoder)
        self._rows = rows
        self.size = len(rows)
        self.batch_size = int(batch_size) if batch_size else max(1, self.size // 10)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        # Full-data EM operations reuse the local handle (so backend="lightning" matches "local").
        self._local = LocalEncodedData(
            rows, estimator=estimator, model=model, encoder=self.encoder, sub_chunks=sub_chunks
        )
        self._datamodule = _make_datamodule(self.size, self.batch_size, self.shuffle, self.seed)

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

    def minibatches(self) -> Iterator[list[Any]]:
        """Yield one epoch of raw-observation mini-batches via the Lightning DataLoader."""
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
        for _epoch in range(int(epochs)):
            for batch in self.minibatches():
                model = stream.update(batch)
        return model

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        """Release resources owned by the resident local handle."""
        self._local.close()
