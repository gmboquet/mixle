"""Model-parallel estimation: distribute a model's shardable axis across workers.

The inversion of the data-parallel backends: there the *model* is replicated and the *data* sharded.
Here the model's shardable axis is distributed. Two entry points share one recursive fold:

* ``optimize(..., backend="model_parallel")`` -- the :class:`ModelParallelEncodedData` handle: the data
  is replicated in-process and the model axis is threaded. Simplest, single-machine.
* ``optimize(ModelParallelEstimator(est), backend="spark"|"mpi"|"mp"|"local")`` -- the estimator wrapper:
  composes with **any** data backend, so the *data* is sharded by that backend (Spark partitions, MPI
  ranks, mp pool) while the *model* axis is distributed inside each partition's accumulator. This is the
  data x model composition -- both axes at once.

The fold (:func:`model_parallel_fold`) is **recursive**: it walks the whole model tree and threads the
axes that a per-node **compute-cost** model says save the most wall-time (``_parallel_ids``, consistent
with the structural planner -- a narrow batch of heavy MVGaussians beats a wider batch of low-cost leaves),
recursing serially below any threaded node so no two pools ever nest. Each recursive case reproduces the
corresponding accumulator's ``seq_update`` exactly:

* **FACTOR** (Composite/Record) -- the per-factor accumulators are independent, so the per-factor
  ``seq_update`` calls are distributed (bit-identical).
* **COMPONENT** (mixtures) -- the responsibility ``logsumexp`` couples the components, so the low-cost
  normalization runs centrally on the gathered score matrix while the expensive per-component scoring and
  accumulation are distributed -- a bit-identical mirror of ``MixtureAccumulator.seq_update``.
* atomic / unknown -- the replicated base case ``acc.seq_update(enc, weights, model)``.

So the whole fold is bit-identical to the single-node path (the data-axis reduce across partitions is the
usual additive ``combine``, exact up to float reassociation like every data-parallel backend). Correct
for *every* family; never worse than ``backend="local"``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from mixle.stats.compute.decomposition import DecompAxis, decomposition_for
from mixle.stats.compute.pdist import (
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.parallel.planner import EncodedDataHandle, _global_key_merge, register_encoded_data_backend


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


def _parallel_ids(model: Any, num_workers: int | None) -> frozenset[int]:
    """The set of tree nodes to thread, chosen by COMPUTE COST (not unit count), consistent with the C2
    planner -- so the executor parallelizes the genuinely heaviest axes, e.g. a narrow batch of D*D
    MVGaussians over a wider batch of low-cost categoricals.

    Every shardable node is scored with the planner's benefit = total_work - max(max_unit_work,
    total_work / P) (greedy-schedule time saved; a fat bottleneck unit caps it). We thread every node tied
    at the maximum benefit, which picks up several independent comparable axes (e.g. sibling mixtures of
    equal cost) -- the recursion below disables nested selection, so no two chosen nodes are ever
    ancestor/descendant and at most one pool is ever live (no nested pools, no oversubscription). Because
    the choice only reorders disjoint writes, the fold stays bit-identical regardless of what is selected.
    """
    from mixle.utils.parallel.model_decomposition import shard_children, subtree_work

    benefits: dict[int, float] = {}
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        dc = decomposition_for(node)
        kids = shard_children(node, dc)
        threadable = dc.axis in (DecompAxis.FACTOR, DecompAxis.COMPONENT)  # the only axes _fold threads via _run
        if threadable and dc.is_shardable and len(kids) == dc.num_units and dc.num_units >= 2:
            works = [subtree_work(k) for k in kids if k is not None]
            if works:
                total = float(sum(works))
                p = dc.num_units if not num_workers else min(num_workers, dc.num_units)
                benefits[id(node)] = total - max(max(works), total / max(1, p))
        if threadable:  # don't descend a STATE/SEQUENCE node's children (e.g. an HMM's 1000s of emission
            for child in kids:  # states): the executor can't thread inside them, and walking them is pure cost
                if child is not None:
                    walk(child)

    walk(model)
    if not benefits:
        return frozenset()
    best = max(benefits.values())
    if best <= 0.0:
        return frozenset()
    tol = 1e-9 * max(1.0, abs(best))
    return frozenset(nid for nid, v in benefits.items() if v >= best - tol)


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


def _component_encs(model: Any, enc: Any, k: int) -> list[Any]:
    """Per-component encodings, routing for BOTH mixture kinds: a homogeneous ``MixtureDistribution``
    shares one encoding across components (``_component_enc``), while a ``HeterogeneousMixtureDistribution``
    encodes as ``(tag_list, enc_data)`` and routes component ``i`` to the encoding of its distribution
    *type* ``enc_data[tag]`` (one tag per family, possibly shared by several components)."""
    from mixle.stats.latent.heterogeneous_mixture import HeterogeneousMixtureDistribution

    if isinstance(model, HeterogeneousMixtureDistribution):
        tag_list, enc_data = enc
        out: list[Any] = [None] * k
        for tag, tag_idxs in enumerate(tag_list):
            for i in tag_idxs:
                out[i] = enc_data[tag]
        return out

    from mixle.stats.latent.mixture import _component_enc

    return [_component_enc(enc, i) for i in range(k)]


def _fold_component_into(
    acc: Any, model: Any, enc: Any, weights: np.ndarray, parallel: bool, sub: frozenset[int], num_workers: int | None
) -> None:
    """Mixture component E-step: distribute per-component scoring + accumulation, normalize centrally.

    Works for both the homogeneous ``MixtureDistribution`` and the ``HeterogeneousMixtureDistribution``
    (which share this exact responsibility arithmetic and differ only in per-component encoding routing)."""
    k = int(model.num_components)
    cenc = _component_encs(model, enc, k)
    log_w = np.asarray(model.log_w, dtype=np.float64)
    zw = model.zw
    ll_mat = np.zeros((len(weights), k), dtype=np.float64)
    ll_mat.fill(-np.inf)

    def score(i: int) -> None:  # distributed: the expensive per-component emission scoring
        if not zw[i]:
            ll_mat[:, i] = model.components[i].seq_log_density(cenc[i]) + log_w[i]

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
        _fold_into(acc.accumulators[i], model.components[i], cenc[i], w_loc, sub, num_workers)

    _run(parallel, accum, range(k), num_workers)


def _fold_into(
    acc: Any, model: Any, enc: Any, weights: np.ndarray, pset: frozenset[int], num_workers: int | None
) -> None:
    """Recursively accumulate ``model``'s E-step into ``acc``, threading a node iff it is one of the
    cost-chosen axes (``pset``) and disabling selection below it so no two pools ever nest."""
    dc = decomposition_for(model)
    parallel = id(model) in pset
    sub: frozenset[int] = frozenset() if parallel else pset  # below a threaded node, recurse serially
    if _factor_ok(acc, model, enc, dc):
        accs = acc.accumulators
        _run(
            parallel,
            lambda i: _fold_into(accs[i], model.dists[i], enc[i], weights, sub, num_workers),
            range(len(accs)),
            num_workers,
        )
    elif _component_ok(acc, model, dc):
        _fold_component_into(acc, model, enc, weights, parallel, sub, num_workers)
    else:
        # base case: an accumulator that is not suff-stat-separable (a leaf, or an HMM whose forward-backward
        # couples all states). If it opts into internal state-parallelism (``_state_workers``, e.g. an HMM's
        # per-state emission scoring/accumulation), hand it the worker budget; otherwise it runs replicated.
        if hasattr(acc, "_state_workers"):
            acc._state_workers = num_workers
        acc.seq_update(enc, weights, model)


def model_parallel_fold(acc: Any, model: Any, enc: Any, weights: np.ndarray, num_workers: int | None = None) -> None:
    """Run a model-parallel E-step of ``model`` over encoded ``enc`` into accumulator ``acc`` (in place)."""
    _fold_into(acc, model, enc, weights, _parallel_ids(model, num_workers), num_workers)


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
        """Return the encoded-data size and total log likelihood under ``estimate``."""
        ll = np.asarray(estimate.seq_log_density(self.enc), dtype=np.float64)
        return float(self.size), float(ll.sum())

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """Run one model-parallel E/M update from ``prev_estimate``."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        acc = self._fold(estimator, prev_estimate, np.ones(self.size, dtype=np.float64))
        _global_key_merge(acc)
        return estimator.estimate(float(self.size), acc.value())

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        """Initialize a model by randomly selecting observations with probability ``p``."""
        from mixle.stats import validate_estimator_keys

        validate_estimator_keys(estimator)
        acc = estimator.accumulator_factory().make()
        rng_w = np.random.RandomState(seed=rng.randint(2**31))
        weights = np.zeros(self.size, dtype=np.float64)
        weights[rng_w.rand(self.size) <= p] = 1.0
        acc.seq_initialize(self.enc, weights, rng)
        _global_key_merge(acc)
        return estimator.estimate(float(weights.sum()), acc.value())

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Accumulate model-parallel sufficient statistics for streaming backends."""
        from mixle.stats import validate_estimator_keys

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
        """Delegate scalar accumulation to the wrapped accumulator."""
        self.inner.update(x, weight, estimate)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """Delegate scalar initialization to the wrapped accumulator."""
        self.inner.initialize(x, weight, rng)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: Any) -> None:
        """Run the wrapped accumulator's sequence update through the model-parallel fold."""
        model_parallel_fold(self.inner, estimate, x, weights, self.num_workers)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: Any) -> None:
        """Delegate encoded initialization to the wrapped accumulator."""
        self.inner.seq_initialize(x, weights, rng)

    def combine(self, suff_stat: Any) -> ModelParallelAccumulator:
        """Merge sufficient statistics into the wrapped accumulator."""
        self.inner.combine(suff_stat)
        return self

    def value(self) -> Any:
        """Return the wrapped accumulator's sufficient-statistic value."""
        return self.inner.value()

    def from_value(self, x: Any) -> ModelParallelAccumulator:
        """Replace the wrapped accumulator from a sufficient-statistic value."""
        self.inner.from_value(x)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic merging to the wrapped accumulator."""
        self.inner.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic replacement to the wrapped accumulator."""
        self.inner.key_replace(stats_dict)

    def acc_to_encoder(self) -> Any:
        """Return the wrapped accumulator's compatible data encoder."""
        return self.inner.acc_to_encoder()


class ModelParallelAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory that wraps another accumulator factory with model-parallel updates."""

    def __init__(self, inner_factory: Any, num_workers: int | None = None) -> None:
        self.inner_factory = inner_factory
        self.num_workers = num_workers

    def make(self) -> ModelParallelAccumulator:
        """Create a fresh model-parallel accumulator wrapper."""
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
        """Return a factory that wraps the inner estimator's accumulator factory."""
        return ModelParallelAccumulatorFactory(self.inner.accumulator_factory(), self.num_workers)

    def estimate(self, nobs: float | None, suff_stat: Any) -> Any:
        """Delegate the M-step to the wrapped estimator."""
        return self.inner.estimate(nobs, suff_stat)


register_encoded_data_backend("model_parallel", _model_parallel_backend, aliases=("mp_model",))


# --- C2 -> C3 wiring: let the planner choose the axis and size the model split --------------------
def auto_parallel_estimator(
    estimator: Any, model: Any, resources: Any = None, *, n_data: int | None = None, min_components_per_shard: int = 1
) -> tuple[Any, Any]:
    """Consult the C2 planner (:func:`decompose_model`) and return ``(estimator, decomposition)``.

    When the planner picks model-parallelism for ``model`` on ``resources``, the estimator is wrapped in
    :class:`ModelParallelEstimator` sized to the planner's cuts; otherwise the plain estimator is returned
    (replicate the model, shard the data -- already optimal when N dominates). Either way, run the returned
    estimator through ``optimize(data, est, backend=<data backend>)``: the data axis is handled by
    ``backend`` and the model axis, if any, by the wrapper -- composing into the data x model split. The
    ``decomposition``'s ``rationale`` explains the choice. ``resources`` defaults to the local CPU slots
    (use ``Resources.from_spark(sc)`` / ``Resources.from_mpi()`` to size the split to a real cluster)."""
    from mixle.utils.parallel.model_decomposition import decompose_model
    from mixle.utils.parallel.planner import Resources

    resources = Resources.local() if resources is None else resources
    dec = decompose_model(model, resources, n_data=n_data, min_components_per_shard=min_components_per_shard)
    if dec.is_model_parallel:
        return ModelParallelEstimator(estimator, num_workers=len(dec.cuts)), dec
    return estimator, dec


__all__ = [
    "ModelParallelEncodedData",
    "ModelParallelEstimator",
    "ModelParallelAccumulator",
    "ModelParallelAccumulatorFactory",
    "model_parallel_fold",
    "auto_parallel_estimator",
    "_model_parallel_backend",
]
