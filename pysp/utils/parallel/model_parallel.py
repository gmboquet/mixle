"""Model-parallel encoded-data handle (component C3 of the model-parallel design).

The inversion of the data-parallel backends: there the model is replicated and the data sharded; here
the **data is replicated** and the **model's shardable axis is distributed across workers**. It satisfies
the duck-typed ``EncodedDataHandle`` contract, so ``optimize(..., backend="model_parallel")`` reaches it
through the unchanged dispatch path -- no edits to ``optimize`` / ``_em_loop`` / ``sequence.py``.

This first cut handles the universal, exact case: a **FACTOR**-decomposable model (Composite / Record).
Its sufficient statistic is a tuple of *independent* per-factor stats (``CompositeAccumulator.seq_update``
updates each child with its own data slice), so distributing the per-factor accumulations across a thread
pool is **bit-identical** to the single-node fold (each factor writes its own accumulator -- no races, no
new reduction algebra). Any other model (mixtures, leaves, HMMs, ...) falls back to ordinary replicated
accumulation, so the backend is correct for *every* family and never worse than ``backend="local"``.

The COMPONENT axis (mixtures) -- which couples components through the responsibility logsumexp -- is the
next milestone, reusing ``pysp.stats.compute.stacked``. See ``~/codex/notes/model-parallel-design.md``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from pysp.stats.compute.decomposition import DecompAxis, decomposition_for
from pysp.utils.parallel.planner import EncodedDataHandle, _global_key_merge, register_encoded_data_backend


class ModelParallelEncodedData(EncodedDataHandle):
    """Replicate the data, distribute the model's shardable axis across workers (exact, in-process)."""

    def __init__(
        self,
        data: Any,
        *,
        estimator: Any | None = None,
        model: Any | None = None,
        encoder: Any | None = None,
        num_workers: int | None = None,
        **_: Any,
    ) -> None:
        if encoder is None:
            if model is not None and callable(getattr(model, "dist_to_encoder", None)):
                encoder = model.dist_to_encoder()
            elif estimator is not None:
                encoder = estimator.accumulator_factory().make().acc_to_encoder()
        if encoder is None:
            raise ValueError("ModelParallelEncodedData requires an encoder, model, or estimator.")
        data = list(data)
        if not data:
            raise ValueError("ModelParallelEncodedData requires non-empty data.")
        self.encoder = encoder
        self.size = len(data)
        self.enc = encoder.seq_encode(data)
        self.num_workers = num_workers

    # --- the model-parallel E-step --------------------------------------------------------------
    def _factor_parallel(self, acc: Any, model: Any) -> bool:
        """True when ``model`` splits along independent factors and ``acc`` mirrors that structure."""
        dc = decomposition_for(model)
        accs = getattr(acc, "accumulators", None)
        dists = getattr(model, "dists", None)
        if dc.axis is not DecompAxis.FACTOR or accs is None or dists is None:
            return False
        return (
            len(accs) == dc.num_units == len(dists)
            and isinstance(self.enc, (tuple, list))
            and len(self.enc) == len(accs)
        )

    def _fold(self, estimator: Any, model: Any, weights: np.ndarray) -> Any:
        """Build and run the E-step accumulator -- factor-parallel when possible, else replicated."""
        acc = estimator.accumulator_factory().make()
        if self._factor_parallel(acc, model):
            accs = acc.accumulators  # CompositeAccumulator.seq_update does exactly this, per factor

            def do(i: int) -> None:
                accs[i].seq_update(self.enc[i], weights, model.dists[i])

            workers = self.num_workers or min(len(accs), max(1, os.cpu_count() or 1))
            if workers > 1 and len(accs) > 1:
                with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                    list(pool.map(do, range(len(accs))))  # order-independent: disjoint per-factor accumulators
            else:
                for i in range(len(accs)):
                    do(i)
        else:
            acc.seq_update(self.enc, weights, model)
        return acc

    # --- EncodedDataHandle contract -------------------------------------------------------------
    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        ll = np.asarray(estimate.seq_log_density(self.enc), dtype=np.float64)
        return float(self.size), float(ll.sum())

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        from pysp.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        acc = self._fold(estimator, prev_estimate, np.ones(self.size, dtype=np.float64))
        _global_key_merge(acc)
        return estimator.estimate(float(self.size), acc.value())

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        from pysp.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        acc = estimator.accumulator_factory().make()
        rng_w = np.random.RandomState(seed=rng.randint(2**31))
        weights = np.zeros(self.size, dtype=np.float64)
        weights[rng_w.rand(self.size) <= p] = 1.0
        acc.seq_initialize(self.enc, weights, rng)
        _global_key_merge(acc)
        return estimator.estimate(float(weights.sum()), acc.value())

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        from pysp.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        acc = self._fold(estimator, model, np.ones(self.size, dtype=np.float64))
        _global_key_merge(acc)
        return float(self.size), acc.value()


def _model_parallel_backend(
    data: Any,
    *,
    estimator: Any = None,
    model: Any = None,
    encoder: Any = None,
    num_workers: int | None = None,
    **_: Any,
) -> ModelParallelEncodedData:
    return ModelParallelEncodedData(data, estimator=estimator, model=model, encoder=encoder, num_workers=num_workers)


register_encoded_data_backend("model_parallel", _model_parallel_backend, aliases=("mp_model",))

__all__ = ["ModelParallelEncodedData", "_model_parallel_backend"]
