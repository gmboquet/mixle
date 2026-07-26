"""In-process component-parallel estimation over a fully resident model.

This runtime parallelizes independent model *work* across threads. It does not
partition model storage or execute a planner's device placements: every caller
must be able to hold the full model and statistic state in one process.

* ``optimize(..., backend="component_parallel")`` uses a replicated in-process
  data handle and threads independent factors/components.
* ``optimize(ComponentParallelEstimator(est), backend=...)`` composes that
  threading with a data backend. Each backend process still owns the full model.

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
from mixle.utils.vector import ImpossibleEvidenceError


class UnrealizedModelPlacementError(RuntimeError):
    """A planner selected model shards for which this runtime has no placement executor."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan
        super().__init__(
            "the planner selected distributed model shards, but the in-process "
            "component-parallel runtime keeps the full model resident and cannot "
            "execute device placement; use an executor with an explicit placement "
            "contract or revise the resources/workload"
        )


# --- the recursive component-parallel fold (module-level so both entry points share it) -----------
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
    from mixle.utils.parallel.model_decomposition import cost_children, shard_children, subtree_work

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
        # Axis discovery follows ownership, not shardability. An atomic wrapper
        # can own a separable mixture/composite whose axis is still executable
        # through the wrapper adapters in _fold_into.
        for child in cost_children(node):
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


def _optional_ok(acc: Any, model: Any, enc: Any, dc: Any) -> bool:
    return (
        dc.axis is DecompAxis.NONE
        and hasattr(acc, "accumulator")
        and hasattr(acc, "weights")
        and hasattr(model, "dist")
        and isinstance(enc, (tuple, list))
        and len(enc) == 4
    )


def _sequence_ok(acc: Any, model: Any, enc: Any, dc: Any) -> bool:
    return (
        dc.axis is DecompAxis.SEQUENCE
        and hasattr(acc, "accumulator")
        and hasattr(acc, "len_accumulator")
        and hasattr(model, "dist")
        and hasattr(model, "len_dist")
        and isinstance(enc, (tuple, list))
        and len(enc) == 5
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


def _validated_mixture_state(model: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a consistent simplex/log-simplex pair for a component fold."""
    components = getattr(model, "components", ())
    if k <= 0 or len(components) != k:
        raise ValueError("component fold requires a non-empty component list matching num_components")
    weights = np.asarray(getattr(model, "w", None), dtype=np.float64)
    log_weights = np.asarray(getattr(model, "log_w", None), dtype=np.float64)
    zero = np.asarray(getattr(model, "zw", None))
    if weights.shape != (k,) or log_weights.shape != (k,) or zero.shape != (k,):
        raise ValueError("mixture weights, log weights, and zero mask must match num_components")
    if zero.dtype.kind != "b":
        raise ValueError("mixture zero-weight mask must be boolean")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("mixture weights must be finite and non-negative")
    if not np.isclose(float(weights.sum()), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("mixture weights must sum to one before component folding")
    expected_zero = weights == 0.0
    if not np.array_equal(zero, expected_zero):
        raise ValueError("mixture zero-weight mask is inconsistent with its weights")
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError("mixture log weights may contain only finite values or -inf")
    expected_log = np.full(k, -np.inf, dtype=np.float64)
    np.log(weights, out=expected_log, where=~expected_zero)
    if not np.array_equal(np.isneginf(log_weights), expected_zero) or not np.allclose(
        log_weights[~expected_zero],
        expected_log[~expected_zero],
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("mixture log weights are inconsistent with its simplex weights")
    return log_weights, zero


def _weighted_component_responsibilities(log_joint: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalize component log-joints with explicit non-finite semantics."""
    scores = np.asarray(log_joint, dtype=np.float64).copy()
    obs_weights = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("component log-joints must be a non-empty two-dimensional matrix")
    if obs_weights.shape != (scores.shape[0],):
        raise ValueError("observation weights must align with component log-joint rows")
    if np.any(~np.isfinite(obs_weights)) or np.any(obs_weights < 0.0):
        raise ValueError("observation weights must be finite and non-negative")
    if np.any(np.isnan(scores)):
        raise ValueError("component log-likelihoods must not contain NaN")

    positive = np.isposinf(scores)
    has_positive = positive.any(axis=1)
    all_negative = np.isneginf(scores).all(axis=1)
    if np.any(all_negative):
        rows = np.flatnonzero(all_negative).tolist()
        raise ImpossibleEvidenceError("mixture evidence has zero probability at row(s) %s" % rows)

    # A +inf log-joint dominates every finite entry. Multiple +inf branches
    # split the limiting mass equally, matching a stable softmax's explicit
    # extended-real convention.
    if np.any(has_positive):
        scores[has_positive] = np.where(positive[has_positive], 0.0, -np.inf)

    maxima = scores.max(axis=1, keepdims=True)
    scores -= maxima
    np.exp(scores, out=scores)
    np.sum(scores, axis=1, keepdims=True, out=maxima)
    np.divide(obs_weights[:, None], maxima, out=maxima)
    scores *= maxima
    return scores


def _fold_component_into(
    acc: Any, model: Any, enc: Any, weights: np.ndarray, parallel: bool, sub: frozenset[int], num_workers: int | None
) -> None:
    """Mixture component E-step: distribute per-component scoring + accumulation, normalize centrally.

    Works for both the homogeneous ``MixtureDistribution`` and the ``HeterogeneousMixtureDistribution``
    (which share this exact responsibility arithmetic and differ only in per-component encoding routing)."""
    k = int(model.num_components)
    cenc = _component_encs(model, enc, k)
    log_w, zw = _validated_mixture_state(model, k)
    ll_mat = np.zeros((len(weights), k), dtype=np.float64)
    ll_mat.fill(-np.inf)

    def score(i: int) -> None:  # distributed: the expensive per-component emission scoring
        if not zw[i]:
            component_scores = np.asarray(model.components[i].seq_log_density(cenc[i]), dtype=np.float64)
            if component_scores.shape != (len(weights),):
                raise ValueError("component %d returned log-likelihood shape %r" % (i, component_scores.shape))
            if np.any(np.isnan(component_scores)):
                raise ValueError("component %d returned NaN log-likelihoods" % i)
            ll_mat[:, i] = component_scores + log_w[i]

    _run(parallel, score, range(k), num_workers)
    ll_mat = _weighted_component_responsibilities(ll_mat, weights)

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
    elif _optional_ok(acc, model, enc, dc):
        _size, zero_indices, nonzero_indices, child_encoding = enc
        nonzero_weights = weights[nonzero_indices]
        acc.weights[0] += float(np.sum(weights[zero_indices]))
        acc.weights[1] += float(np.sum(nonzero_weights))
        _fold_into(acc.accumulator, model.dist, child_encoding, nonzero_weights, pset, num_workers)
    elif _sequence_ok(acc, model, enc, dc):
        indices, inverse_counts, _nonzero, element_encoding, length_encoding = enc
        element_weights = (
            weights[indices] * inverse_counts[indices]
            if getattr(acc, "len_normalized", False)
            else weights[indices]
        )
        _fold_into(acc.accumulator, model.dist, element_encoding, element_weights, pset, num_workers)
        if not getattr(acc, "null_len_accumulator", False):
            _fold_into(acc.len_accumulator, model.len_dist, length_encoding, weights, pset, num_workers)
    else:
        # base case: an accumulator that is not suff-stat-separable (a leaf, or an HMM whose forward-backward
        # couples all states). If it opts into internal state-parallelism (``_state_workers``, e.g. an HMM's
        # per-state emission scoring/accumulation), hand it the worker budget; otherwise it runs replicated.
        if hasattr(acc, "_state_workers"):
            acc._state_workers = num_workers
        acc.seq_update(enc, weights, model)


def model_parallel_fold(
    acc: Any,
    model: Any,
    enc: Any,
    weights: np.ndarray,
    num_workers: int | None = None,
    *,
    parallel_ids: frozenset[int] | None = None,
) -> None:
    """Run a component-parallel E-step over a fully resident model."""

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("fold weights must be a finite, non-negative one-dimensional array")
    selected = _parallel_ids(model, num_workers) if parallel_ids is None else parallel_ids
    _fold_into(acc, model, enc, weights, selected, num_workers)


# --- entry point 1: the in-process handle (data replicated, model distributed) --------------------
class ModelParallelEncodedData(EncodedDataHandle):
    """Compatibility name for a fully resident, component-threaded data handle."""

    execution_kind = "component_parallel_threads"

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

        try:
            p = float(p)
        except (TypeError, ValueError) as exc:
            raise TypeError("initialization probability must be a real scalar") from exc
        if not np.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError("initialization probability must be finite and in [0, 1]")
        validate_estimator_keys(estimator)
        acc = estimator.accumulator_factory().make()
        rng_w = np.random.RandomState(seed=rng.randint(2**31))
        weights = np.zeros(self.size, dtype=np.float64)
        weights[rng_w.rand(self.size) <= p] = 1.0
        if not np.any(weights):
            raise ImpossibleEvidenceError("initialization selected no observations")
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
    """Wrap an accumulator so its E-step threads independent component work.

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
    """Compatibility wrapper for fully resident component-parallel EM.

    ``optimize(ModelParallelEstimator(est), backend="spark"|"mpi"|"mp"|"local")`` shards the data through
    that backend while each partition's E-step threads independent model work.
    The full model remains resident in every process. The M-step
    (``estimate``) and the accumulator's value/combine contract are the wrapped estimator's, unchanged.
    """

    execution_kind = "component_parallel_threads"

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


ComponentParallelEncodedData = ModelParallelEncodedData
ComponentParallelAccumulator = ModelParallelAccumulator
ComponentParallelAccumulatorFactory = ModelParallelAccumulatorFactory
ComponentParallelEstimator = ModelParallelEstimator
component_parallel_fold = model_parallel_fold


register_encoded_data_backend(
    "component_parallel",
    _model_parallel_backend,
    aliases=("model_parallel", "mp_model"),
)


# --- C2 -> C3 wiring: let the planner choose the axis and size the model split --------------------
def auto_parallel_estimator(
    estimator: Any, model: Any, resources: Any = None, *, n_data: int | None = None, min_components_per_shard: int = 1
) -> tuple[Any, Any]:
    """Return a plain estimator when no distributed model placement is required.

    The structural planner is advisory. If it selects cuts that require model
    storage to be partitioned across devices, this helper raises
    :class:`UnrealizedModelPlacementError`: the local component-thread runtime
    cannot truthfully realize those cuts. Callers may explicitly choose
    :class:`ComponentParallelEstimator` when the full model fits in each process.
    """
    from mixle.utils.parallel.model_decomposition import decompose_model
    from mixle.utils.parallel.planner import Resources

    resources = Resources.local() if resources is None else resources
    dec = decompose_model(model, resources, n_data=n_data, min_components_per_shard=min_components_per_shard)
    if dec.is_model_parallel:
        raise UnrealizedModelPlacementError(dec)
    return estimator, dec


__all__ = [
    "ModelParallelEncodedData",
    "ModelParallelEstimator",
    "ModelParallelAccumulator",
    "ModelParallelAccumulatorFactory",
    "ComponentParallelEncodedData",
    "ComponentParallelEstimator",
    "ComponentParallelAccumulator",
    "ComponentParallelAccumulatorFactory",
    "model_parallel_fold",
    "component_parallel_fold",
    "auto_parallel_estimator",
    "UnrealizedModelPlacementError",
    "_model_parallel_backend",
]
