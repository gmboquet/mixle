"""Ray-backed encoded-data handle for distributed sufficient-statistic folding (WS-C2).

``RayEncodedData`` plugs Ray into mixle's encoded-data backend registry
(``planner.encoded_data(..., backend="ray")``). The data is encoded once, split into partitions, and
each partition is placed in the Ray object store; the orchestrator-contract methods
(``pysp_seq_log_density_sum`` / ``pysp_seq_estimate`` / ``pysp_seq_initialize`` /
``pysp_stream_accumulate``) map a Ray remote task over the partitions and reduce the per-partition
sufficient statistics on the driver -- the same map/fold the Spark and dask backends do, on a Ray
cluster.

Ray is an optional dependency: this module is imported only when the ``"ray"`` backend is requested,
so the rest of mixle (and CI without Ray installed) is unaffected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import DataSequenceEncoder
from mixle.utils.parallel.planner import EncodedDataHandle, _global_key_merge, _split_range


def _resolve_encoder(estimator: Any, model: Any, encoder: DataSequenceEncoder | None) -> DataSequenceEncoder:
    if encoder is not None:
        return encoder
    if model is not None and callable(getattr(model, "dist_to_encoder", None)):
        return model.dist_to_encoder()
    if estimator is not None:
        return estimator.accumulator_factory().make().acc_to_encoder()
    raise ValueError("RayEncodedData requires an encoder, model, or estimator.")


# Remote task bodies (plain functions; wrapped with ray.remote at call time so importing this module
# does not require Ray). Each receives a resident encoded chunk and returns reducible statistics.
def _score_chunk(enc: Any, size: int, estimate: Any) -> tuple[float, float]:
    return float(size), float(np.asarray(estimate.seq_log_density(enc), dtype=np.float64).sum())


def _accumulate_chunk(enc: Any, size: int, estimator: Any, model: Any) -> tuple[float, Any]:
    acc = estimator.accumulator_factory().make()
    acc.seq_update(enc, np.ones(size, dtype=np.float64), model)
    return float(size), acc.value()


def _initialize_chunk(enc: Any, size: int, estimator: Any, seed: int, p: float) -> tuple[float, Any]:
    rng = np.random.RandomState(int(seed))
    weights = np.zeros(size, dtype=np.float64)
    weights[rng.rand(size) <= p] = 1.0
    acc = estimator.accumulator_factory().make()
    acc.seq_initialize(enc, weights, rng)
    return float(weights.sum()), acc.value()


class RayEncodedData(EncodedDataHandle):
    """Encoded-data handle that folds sufficient statistics over Ray object-store partitions."""

    def __init__(
        self,
        data: Any,
        estimator: Any | None = None,
        model: Any | None = None,
        encoder: DataSequenceEncoder | None = None,
        num_partitions: int | None = None,
        num_workers: int | None = None,
        address: str | None = None,
        **_: Any,
    ) -> None:
        import ray

        rows = list(data)
        if not rows:
            raise ValueError("RayEncodedData requires non-empty data.")
        self.encoder = _resolve_encoder(estimator, model, encoder)
        self.size = len(rows)
        self._owns_ray = not ray.is_initialized()
        if self._owns_ray:
            ray.init(
                address=address,
                ignore_reinit_error=True,
                include_dashboard=False,
                configure_logging=False,
                log_to_driver=False,
                num_cpus=num_workers,
            )
        nparts = int(num_partitions or num_workers or 4)
        self._chunks: list[tuple[int, Any]] = []
        for start, stop in _split_range(0, self.size, max(1, nparts)):
            if stop <= start:
                continue
            enc = self.encoder.seq_encode(rows[start:stop])
            self._chunks.append((stop - start, ray.put(enc)))

    def _map(self, fn: Any, *args: Any) -> list[Any]:
        import ray

        remote = ray.remote(fn)
        futures = [remote.remote(ref, size, *args) for size, ref in self._chunks]
        return ray.get(futures)

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Return total observation count and summed log-density over Ray partitions."""
        results = self._map(_score_chunk, estimate)
        return float(sum(r[0] for r in results)), float(sum(r[1] for r in results))

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Fold sufficient statistics over Ray partitions and run the estimator M-step."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        results = self._map(_accumulate_chunk, estimator, prev_estimate)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        for size, value in results:
            nobs += size
            accumulator.combine(value)
        _global_key_merge(accumulator)
        return estimator.estimate(nobs, accumulator.value())

    def pysp_seq_initialize(self, estimator: Any, rng: RandomState, p: float) -> Any:
        """Initialize an estimate from Bernoulli-subsampled Ray partitions."""
        import ray

        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        seeds = rng.randint(2**31, size=max(1, len(self._chunks)))
        remote = ray.remote(_initialize_chunk)
        futures = [remote.remote(ref, size, estimator, int(seeds[i]), p) for i, (size, ref) in enumerate(self._chunks)]
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        for weight, value in ray.get(futures):
            nobs += weight
            accumulator.combine(value)
        _global_key_merge(accumulator)
        return estimator.estimate(nobs, accumulator.value())

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Return accumulated streaming statistics for all Ray partitions."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        results = self._map(_accumulate_chunk, estimator, model)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        for size, value in results:
            nobs += size
            accumulator.combine(value)
        _global_key_merge(accumulator)
        return nobs, accumulator.value()

    @property
    def num_chunks(self) -> int:
        """Number of encoded Ray partitions held by this handle."""
        return len(self._chunks)

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        """No-op: the Ray runtime is left running (shared across handles); the process tears it down."""
        return None
