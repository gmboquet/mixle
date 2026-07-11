"""Distributed heterogeneous EM execution: sharded E-steps + k-way tree reduce of sufficient statistics.

The actual substance of 'distributed through heterogeneous compute' is backend-agnostic: split the data
into shards, run each shard's E-step (optionally at its own precision), and fold the fixed-size sufficient
statistics with a k-way tree of ``accumulator.combine()`` -- ``O(log W)`` depth, no single-root fan-in.
``combine`` is associative, so the tree result is bit-identical to a serial fold for integer/count
statistics and within float reassociation otherwise.

This module is the *executed, verifiable* core: a local executor that shards and tree-reduces in-process,
exactly matching a serial fit. The Spark (``RDD.treeReduce``), MPI (``comm.reduce``), and torchrun
transports are thin adapters that replace the local shard loop with cluster transport over the same
combine-tree -- they need a cluster to exercise, but the algorithm they run is the one verified here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def tree_reduce_values(values: list[Any], factory: Any, branch: int = 2) -> Any:
    """Fold accumulator ``value()`` payloads with a ``branch``-ary tree of ``combine()`` -- O(log n) depth.

    Each internal node makes a fresh accumulator (so a shared-reference ``value()`` is never mutated --
    the HMM-stat aliasing hazard), seeds it from the first child, and combines the rest. Bit-identical to
    a serial left fold for associative integer statistics.
    """
    if not values:
        raise ValueError("nothing to reduce")
    level = list(values)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), branch):
            group = level[i : i + branch]
            acc = factory.make().from_value(group[0])
            for v in group[1:]:
                acc.combine(v)
            nxt.append(acc.value())
        level = nxt
    return level[0]


def _shard_estep(estimator: Any, model: Any, shard: Any, compute_dtype: Any = None) -> tuple[int, Any]:
    """One shard's E-step -> (count, sufficient-statistic value). ``compute_dtype`` runs it in reduced
    precision via the fused kernel when the model is fusible (the per-worker precision band)."""
    n = len(shard)
    enc = model.dist_to_encoder().seq_encode(shard)
    weights = np.ones(n, dtype=np.float64)
    if compute_dtype is not None:
        try:
            from mixle.stats.compute.fused_codegen import fused_accumulate, fusible_estep

            if fusible_estep(model):
                return n, fused_accumulate(model, enc, weights, compute_dtype=compute_dtype)
        except Exception:  # noqa: BLE001
            pass  # fall back to the exact float64 accumulator path
    acc = estimator.accumulator_factory().make()
    acc.seq_update(enc, weights, model)
    return n, acc.value()


def _shard_task(payload: tuple[Any, Any, Any, Any]) -> tuple[int, Any]:
    """Picklable wrapper so a ProcessPoolExecutor can run a shard's E-step in a separate OS process."""
    estimator, model, shard, compute_dtype = payload
    return _shard_estep(estimator, model, shard, compute_dtype)


def _shard_bounds(n: int, sizes: list[int] | None, n_shards: int) -> list[tuple[int, int]]:
    if sizes is not None:
        bounds, off = [], 0
        for s in sizes:
            bounds.append((off, off + s))
            off += s
        return bounds
    edges = np.linspace(0, n, n_shards + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_shards)]


def heterogeneous_em_step(
    estimator: Any,
    model: Any,
    data: Any,
    n_shards: int = 1,
    shard_sizes: list[int] | None = None,
    shard_precisions: list[Any] | None = None,
    branch: int = 2,
    pool: Any = None,
) -> Any:
    """One distributed EM step: shard ``data``, E-step each shard (at its precision), tree-reduce, estimate.

    With one shard and no reduced precision this is byte-identical to a plain serial E-step; with many
    shards the tree-reduced result matches it up to float reassociation of ``combine()``. ``pool`` is an
    optional ``concurrent.futures``-style executor (e.g. ``ProcessPoolExecutor``) whose ``map`` runs the
    shard E-steps on real worker processes -- the sufficient-statistic payloads cross the process boundary
    by pickling, and ``combine`` operates on those freshly-unpickled copies (never a shared reference).
    """
    bounds = _shard_bounds(len(data), shard_sizes, n_shards)
    tasks = []
    for i, (lo, hi) in enumerate(bounds):
        shard = data[lo:hi]
        if not len(shard):
            continue
        cd = shard_precisions[i] if shard_precisions else None
        tasks.append((estimator, model, shard, cd))
    results = list(pool.map(_shard_task, tasks)) if pool is not None else [_shard_task(t) for t in tasks]
    values = [v for _, v in results]
    total = sum(c for c, _ in results)
    combined = tree_reduce_values(values, estimator.accumulator_factory(), branch)
    return estimator.estimate(float(total), combined)


def heterogeneous_fit(
    model: Any,
    data: Any,
    max_its: int = 20,
    n_shards: int = 4,
    shard_sizes: list[int] | None = None,
    shard_precisions: list[Any] | None = None,
    branch: int = 2,
    pool: Any = None,
) -> Any:
    """Run ``max_its`` EM iterations with the distributed heterogeneous executor; returns the fitted model."""
    estimator = model.estimator()
    current = model
    for _ in range(max_its):
        current = heterogeneous_em_step(estimator, current, data, n_shards, shard_sizes, shard_precisions, branch, pool)
    return current


def shards_from_plan(plan: Any) -> tuple[list[int], list[Any]]:
    """Translate a :class:`~mixle.engines.heterogeneous.HeterogeneousPlan` into (shard_sizes, precisions).

    Only ``float32`` is wired to the fused reduced-precision kernel here; other bands (fp8/bf16/dd) run on
    the exact float64 accumulator until their compute kernels exist -- so the executor stays correct.
    """
    sizes = [a.rows for a in plan.assignments]
    precisions = [np.float32 if a.precision == "float32" else None for a in plan.assignments]
    return sizes, precisions
