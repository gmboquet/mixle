"""Model-parallel encoded-data handle (component C3 of the model-parallel design).

The inversion of the data-parallel backends: there the model is replicated and the data sharded; here
the **data is replicated** and the **model's shardable axis is distributed across workers**. It satisfies
the duck-typed ``EncodedDataHandle`` contract, so ``optimize(..., backend="model_parallel")`` reaches it
through the unchanged dispatch path -- no edits to ``optimize`` / ``_em_loop`` / ``sequence.py``.

Two model-parallel axes are handled, both **bit-identical** to the single-node fold:

* **FACTOR** (Composite / Record) -- the sufficient statistic is a tuple of *independent* per-factor
  stats (``CompositeAccumulator.seq_update`` updates each child with its own data slice), so distributing
  the per-factor accumulations across a thread pool is exact (each factor writes its own accumulator --
  no races, no new reduction algebra).
* **COMPONENT** (mixtures) -- the responsibility ``logsumexp`` couples the components, so the (cheap)
  normalization runs centrally on the gathered score matrix while the *expensive* per-component emission
  scoring and the per-component accumulation are distributed -- an exact mirror of
  ``MixtureAccumulator.seq_update``.

Any other model (leaves, HMMs, sequences, ...) falls back to ordinary replicated accumulation, so the
backend is correct for *every* family and never worse than ``backend="local"``. See
``~/codex/notes/model-parallel-design.md``.
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
    def _map(self, fn: Any, items: Any) -> None:
        """Run ``fn`` over ``items`` across the worker pool (the model-axis units run in parallel)."""
        items = list(items)
        workers = self.num_workers or min(len(items), max(1, os.cpu_count() or 1))
        if workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                list(pool.map(fn, items))  # order-independent: each unit writes its own disjoint state
        else:
            for it in items:
                fn(it)

    def _factor_ok(self, acc: Any, model: Any, dc: Any) -> bool:
        accs = getattr(acc, "accumulators", None)
        dists = getattr(model, "dists", None)
        return (
            dc.axis is DecompAxis.FACTOR
            and accs is not None
            and dists is not None
            and len(accs) == dc.num_units == len(dists)
            and isinstance(self.enc, (tuple, list))
            and len(self.enc) == len(accs)
        )

    def _component_ok(self, acc: Any, model: Any, dc: Any) -> bool:
        return (
            dc.axis is DecompAxis.COMPONENT
            and hasattr(acc, "comp_counts")
            and getattr(model, "num_components", None) == dc.num_units
            and len(getattr(acc, "accumulators", ())) == dc.num_units
            and hasattr(model, "log_w")
            and hasattr(model, "zw")
        )

    def _fold_component(self, acc: Any, model: Any, weights: np.ndarray) -> None:
        """Mixture component-parallel E-step: distribute the per-component scoring + accumulation while
        the (cheap) responsibility normalization runs centrally -- a bit-identical mirror of
        ``MixtureAccumulator.seq_update`` (the logsumexp couples the components, so it cannot be sharded)."""
        from pysp.stats.latent.mixture import _component_enc

        k = int(model.num_components)
        enc = self.enc
        log_w = np.asarray(model.log_w, dtype=np.float64)
        zw = model.zw
        ll_mat = np.zeros((self.size, k), dtype=np.float64)
        ll_mat.fill(-np.inf)

        def score(i: int) -> None:  # distributed: the expensive per-component emission scoring
            if not zw[i]:
                ll_mat[:, i] = model.components[i].seq_log_density(_component_enc(enc, i)) + log_w[i]

        self._map(score, range(k))

        # central, exact normalization (identical buffer reuse to MixtureAccumulator.seq_update)
        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = log_w.copy()
        ll_max[bad_rows] = np.max(log_w)
        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        np.divide(weights[:, None], ll_max, out=ll_max)
        ll_mat *= ll_max  # ll_mat[:, i] is now responsibility_i * weight

        def accum(i: int) -> None:  # distributed: disjoint per-component sufficient statistics
            w_loc = ll_mat[:, i]
            acc.comp_counts[i] += w_loc.sum()
            acc.accumulators[i].seq_update(_component_enc(enc, i), w_loc, model.components[i])

        self._map(accum, range(k))

    def _fold(self, estimator: Any, model: Any, weights: np.ndarray) -> Any:
        """Build and run the E-step accumulator -- model-parallel along the declared axis, else replicated."""
        acc = estimator.accumulator_factory().make()
        dc = decomposition_for(model)
        if self._factor_ok(acc, model, dc):
            accs = acc.accumulators  # CompositeAccumulator.seq_update does exactly this, per factor
            self._map(lambda i: accs[i].seq_update(self.enc[i], weights, model.dists[i]), range(len(accs)))
        elif self._component_ok(acc, model, dc):
            self._fold_component(acc, model, weights)
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
