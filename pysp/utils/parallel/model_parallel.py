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

The fold is **recursive**: it walks the whole model tree and distributes the per-unit work at the single
*widest* shardable axis (e.g. the 1000 components of a mixture nested inside a composite field), recursing
serially elsewhere -- so a nested model is parallelized along its dominant axis without spawning nested
thread pools. Any node that does not opt into the contract (leaves, HMMs, sequences, ...) is the
replicated base case, so the backend is correct for *every* family and never worse than
``backend="local"``. See ``~/codex/notes/model-parallel-design.md``.
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

    # --- the model-parallel E-step (recursive: parallelize the single widest shardable axis) --------
    def _run(self, parallel: bool, fn: Any, items: Any) -> None:
        """Run ``fn`` over ``items`` -- across the worker pool when ``parallel``, else serially."""
        items = list(items)
        workers = self.num_workers or min(len(items), max(1, os.cpu_count() or 1))
        if parallel and workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                list(pool.map(fn, items))  # order-independent: each unit writes its own disjoint state
        else:
            for it in items:
                fn(it)

    def _spine_units(self, model: Any) -> int:
        """The widest shardable ``num_units`` anywhere in the model tree -- the one axis we parallelize."""
        from pysp.utils.parallel.model_decomposition import shard_children

        dc = decomposition_for(model)
        best = dc.num_units if dc.is_shardable else 1
        for child in shard_children(model, dc):
            if child is not None:
                best = max(best, self._spine_units(child))
        return best

    def _factor_ok(self, acc: Any, model: Any, enc: Any, dc: Any) -> bool:
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

    def _component_ok(self, acc: Any, model: Any, dc: Any) -> bool:
        return (
            dc.axis is DecompAxis.COMPONENT
            and hasattr(acc, "comp_counts")
            and getattr(model, "num_components", None) == dc.num_units
            and len(getattr(acc, "accumulators", ())) == dc.num_units
            and hasattr(model, "log_w")
            and hasattr(model, "zw")
        )

    def _fold_into(self, acc: Any, model: Any, enc: Any, weights: np.ndarray, target: int) -> None:
        """Recursively accumulate ``model``'s E-step into ``acc``, distributing the per-unit work at the
        single widest shardable axis (``num_units == target``) and recursing serially elsewhere. Each
        recursive case reproduces the corresponding accumulator's ``seq_update`` exactly, so the whole
        fold is bit-identical to the single-node path -- just distributed."""
        dc = decomposition_for(model)
        if self._factor_ok(acc, model, enc, dc):
            accs = acc.accumulators
            parallel = dc.num_units == target  # parallelize this axis only if it is the widest
            nxt = -1 if parallel else target  # ...and then recurse serially below it
            self._run(
                parallel, lambda i: self._fold_into(accs[i], model.dists[i], enc[i], weights, nxt), range(len(accs))
            )
        elif self._component_ok(acc, model, dc):
            self._fold_component_into(
                acc, model, enc, weights, dc.num_units == target, -1 if dc.num_units == target else target
            )
        else:
            acc.seq_update(enc, weights, model)  # atomic / unknown node: the replicated base case

    def _fold_component_into(
        self, acc: Any, model: Any, enc: Any, weights: np.ndarray, parallel: bool, nxt: int
    ) -> None:
        """Mixture component E-step: distribute the per-component scoring + accumulation, normalize the
        coupling (responsibility logsumexp) centrally -- a bit-identical mirror of MixtureAccumulator."""
        from pysp.stats.latent.mixture import _component_enc

        k = int(model.num_components)
        log_w = np.asarray(model.log_w, dtype=np.float64)
        zw = model.zw
        ll_mat = np.zeros((len(weights), k), dtype=np.float64)
        ll_mat.fill(-np.inf)

        def score(i: int) -> None:  # distributed: the expensive per-component emission scoring
            if not zw[i]:
                ll_mat[:, i] = model.components[i].seq_log_density(_component_enc(enc, i)) + log_w[i]

        self._run(parallel, score, range(k))

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
            self._fold_into(acc.accumulators[i], model.components[i], _component_enc(enc, i), w_loc, nxt)

        self._run(parallel, accum, range(k))

    def _fold(self, estimator: Any, model: Any, weights: np.ndarray) -> Any:
        """Build the E-step accumulator, distributing work at the widest shardable axis of the model tree."""
        acc = estimator.accumulator_factory().make()
        self._fold_into(acc, model, self.enc, weights, self._spine_units(model))
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
