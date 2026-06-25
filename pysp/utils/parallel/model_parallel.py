"""Model-parallel estimation: distribute a model's shardable axis across workers (component C3).

The inversion of the data-parallel backends: there the *model* is replicated and the *data* sharded.
Here the model's shardable axis is distributed. Two entry points share one recursive fold:

* ``optimize(..., backend="model_parallel")`` -- the :class:`ModelParallelEncodedData` handle: the data
  is replicated in-process and the model axis is threaded. Simplest, single-machine.
* ``optimize(ModelParallelEstimator(est), backend="spark"|"mpi"|"mp"|"local")`` -- the estimator wrapper:
  composes with **any** data backend, so the *data* is sharded by that backend (Spark partitions, MPI
  ranks, mp pool) while the *model* axis is distributed inside each partition's accumulator. This is the
  data x model composition -- both axes at once.

The fold (:func:`model_parallel_fold`) is **recursive**: it walks the model tree and distributes the
per-unit work at the single *widest* shardable axis, recursing serially elsewhere (bounded threads, no
nested pools). Each recursive case reproduces the corresponding accumulator's ``seq_update`` exactly:

* **FACTOR** (Composite/Record) -- the per-factor accumulators are independent, so the per-factor
  ``seq_update`` calls are distributed (bit-identical).
* **COMPONENT** (mixtures) -- the responsibility ``logsumexp`` couples the components, so the cheap
  normalization runs centrally on the gathered score matrix while the expensive per-component scoring and
  accumulation are distributed -- a bit-identical mirror of ``MixtureAccumulator.seq_update``.
* atomic / unknown -- the replicated base case ``acc.seq_update(enc, weights, model)``.

So the whole fold is bit-identical to the single-node path (the data-axis reduce across partitions is the
usual additive ``combine``, exact up to float reassociation like every data-parallel backend). Correct
for *every* family; never worse than ``backend="local"``. See ``~/codex/notes/model-parallel-design.md``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from pysp.stats.compute.decomposition import DecompAxis, decomposition_for
from pysp.stats.compute.pdist import (
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.parallel.planner import EncodedDataHandle, _global_key_merge, register_encoded_data_backend


# --- the recursive model-parallel fold (module-level so both entry points share it) ---------------
def _run(parallel: bool, fn: Any, items: Any, num_workers: int | None) -> None:
    """Run ``fn`` over ``items`` -- across a thread pool when ``parallel``, else serially."""
    items = list(items)
    workers = num_workers or min(len(items), max(1, os.cpu_count() or 1))
    if parallel and workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            list(pool.map(fn, items))  # order-independent: each unit writes its own disjoint state
    else:
        for it in items:
            fn(it)


def _spine_units(model: Any) -> int:
    """The widest shardable ``num_units`` anywhere in the model tree -- the one axis we parallelize."""
    from pysp.utils.parallel.model_decomposition import shard_children

    dc = decomposition_for(model)
    best = dc.num_units if dc.is_shardable else 1
    for child in shard_children(model, dc):
        if child is not None:
            best = max(best, _spine_units(child))
    return best


def _factor_ok(acc: Any, model: Any, enc: Any, dc: Any) -> bool:
    accs = getattr(acc, "accumulators", None)
    dists = getattr(model, "dists", None)
    return (
        dc.axis is DecompAxis.FACTOR
        and accs is not None
        and dists is not None
        and len(accs) == dc.num_units == len(dists)
        and isinstance(enc, (tuple, list))
        and len(enc) == len(accs)
    )


def _component_ok(acc: Any, model: Any, dc: Any) -> bool:
    return (
        dc.axis is DecompAxis.COMPONENT
        and hasattr(acc, "comp_counts")
        and getattr(model, "num_components", None) == dc.num_units
        and len(getattr(acc, "accumulators", ())) == dc.num_units
        and hasattr(model, "log_w")
        and hasattr(model, "zw")
    )


def _fold_component_into(
    acc: Any, model: Any, enc: Any, weights: np.ndarray, parallel: bool, nxt: int, num_workers: int | None
) -> None:
    """Mixture component E-step: distribute per-component scoring + accumulation, normalize centrally."""
    from pysp.stats.latent.mixture import _component_enc

    k = int(model.num_components)
    log_w = np.asarray(model.log_w, dtype=np.float64)
    zw = model.zw
    ll_mat = np.zeros((len(weights), k), dtype=np.float64)
    ll_mat.fill(-np.inf)

    def score(i: int) -> None:  # distributed: the expensive per-component emission scoring
        if not zw[i]:
            ll_mat[:, i] = model.components[i].seq_log_density(_component_enc(enc, i)) + log_w[i]

    _run(parallel, score, range(k), num_workers)

    ll_max = ll_mat.max(axis=1, keepdims=True)  # central, exact (identical buffer reuse to the serial path)
    bad_rows = np.isinf(ll_max.flatten())
    ll_mat[bad_rows, :] = log_w.copy()
    ll_max[bad_rows] = np.max(log_w)
    ll_mat -= ll_max
    np.exp(ll_mat, out=ll_mat)
    np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
    np.divide(weights[:, None], ll_max, out=ll_max)
    ll_mat *= ll_max  # ll_mat[:, i] is now responsibility_i * weight

    def accum(i: int) -> None:  # distributed: disjoint per-component statistics, recursing into the child
        w_loc = ll_mat[:, i]
        acc.comp_counts[i] += w_loc.sum()
        _fold_into(acc.accumulators[i], model.components[i], _component_enc(enc, i), w_loc, nxt, num_workers)

    _run(parallel, accum, range(k), num_workers)


def _fold_into(acc: Any, model: Any, enc: Any, weights: np.ndarray, target: int, num_workers: int | None) -> None:
    """Recursively accumulate ``model``'s E-step into ``acc``, distributing the per-unit work at the single
    widest shardable axis (``num_units == target``) and recursing serially below it."""
    dc = decomposition_for(model)
    if _factor_ok(acc, model, enc, dc):
        accs = acc.accumulators
        parallel = dc.num_units == target
        nxt = -1 if parallel else target
        _run(
            parallel,
            lambda i: _fold_into(accs[i], model.dists[i], enc[i], weights, nxt, num_workers),
            range(len(accs)),
            num_workers,
        )
    elif _component_ok(acc, model, dc):
        parallel = dc.num_units == target
        _fold_component_into(acc, model, enc, weights, parallel, -1 if parallel else target, num_workers)
    else:
        acc.seq_update(enc, weights, model)  # atomic / unknown node: the replicated base case


def model_parallel_fold(acc: Any, model: Any, enc: Any, weights: np.ndarray, num_workers: int | None = None) -> None:
    """Run a model-parallel E-step of ``model`` over encoded ``enc`` into accumulator ``acc`` (in place)."""
    _fold_into(acc, model, enc, weights, _spine_units(model), num_workers)


# --- entry point 1: the in-process handle (data replicated, model distributed) --------------------
class ModelParallelEncodedData(EncodedDataHandle):
    """Replicate the data, distribute the model's shardable axis across threads (single machine)."""

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

    def _fold(self, estimator: Any, model: Any, weights: np.ndarray) -> Any:
        acc = estimator.accumulator_factory().make()
        model_parallel_fold(acc, model, self.enc, weights, self.num_workers)
        return acc

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


# --- entry point 2: the estimator wrapper (composes with any data backend -> data x model) --------
class ModelParallelAccumulator(SequenceEncodableStatisticAccumulator):
    """Wrap an accumulator so its E-step ``seq_update`` runs the recursive model-parallel fold.

    All sufficient-statistic methods delegate to the wrapped (``inner``) accumulator unchanged, so the
    value/combine/from_value/key-merge contract -- and thus every data backend's reduce -- is preserved;
    only ``seq_update`` is replaced with the distributed fold. Holding ``inner`` in ``vars()`` keeps the
    accumulator key-validator's recursion transparent.
    """

    def __init__(self, inner: SequenceEncodableStatisticAccumulator, num_workers: int | None = None) -> None:
        self.inner = inner
        self.num_workers = num_workers
        self.keys = getattr(inner, "keys", None)

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        self.inner.update(x, weight, estimate)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        self.inner.initialize(x, weight, rng)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: Any) -> None:
        model_parallel_fold(self.inner, estimate, x, weights, self.num_workers)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: Any) -> None:
        self.inner.seq_initialize(x, weights, rng)

    def combine(self, suff_stat: Any) -> ModelParallelAccumulator:
        self.inner.combine(suff_stat)
        return self

    def value(self) -> Any:
        return self.inner.value()

    def from_value(self, x: Any) -> ModelParallelAccumulator:
        self.inner.from_value(x)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        self.inner.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        self.inner.key_replace(stats_dict)

    def acc_to_encoder(self) -> Any:
        return self.inner.acc_to_encoder()


class ModelParallelAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, inner_factory: Any, num_workers: int | None = None) -> None:
        self.inner_factory = inner_factory
        self.num_workers = num_workers

    def make(self) -> ModelParallelAccumulator:
        return ModelParallelAccumulator(self.inner_factory.make(), self.num_workers)


class ModelParallelEstimator(ParameterEstimator):
    """Wrap an estimator so EM distributes the model axis -- composing with any ``backend=`` for the data.

    ``optimize(ModelParallelEstimator(est), backend="spark"|"mpi"|"mp"|"local")`` shards the data through
    that backend while each partition's E-step distributes the model axis. The M-step
    (``estimate``) and the accumulator's value/combine contract are the wrapped estimator's, unchanged.
    """

    def __init__(self, inner: ParameterEstimator, num_workers: int | None = None) -> None:
        self.inner = inner
        self.num_workers = num_workers
        self.keys = getattr(inner, "keys", None)

    def accumulator_factory(self) -> ModelParallelAccumulatorFactory:
        return ModelParallelAccumulatorFactory(self.inner.accumulator_factory(), self.num_workers)

    def estimate(self, nobs: float | None, suff_stat: Any) -> Any:
        return self.inner.estimate(nobs, suff_stat)


register_encoded_data_backend("model_parallel", _model_parallel_backend, aliases=("mp_model",))

__all__ = [
    "ModelParallelEncodedData",
    "ModelParallelEstimator",
    "ModelParallelAccumulator",
    "ModelParallelAccumulatorFactory",
    "model_parallel_fold",
    "_model_parallel_backend",
]
