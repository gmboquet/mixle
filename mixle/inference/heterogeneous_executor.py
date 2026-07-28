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

from mixle.stats.compute.pdist import merge_accumulator_keys, validate_estimator_keys


def _reduction_branch(value: Any) -> int:
    """``branch`` as an exact integer of at least 2 -- the minimum that actually reduces a level.

    ``branch=1`` groups every level into singletons, so each pass rebuilds a level of the same length
    and the ``while len(level) > 1`` loop never terminates: the public EM step and fit hang forever
    rather than failing. ``branch <= 0`` fails deep inside ``range`` with an opaque message. Neither is
    a tuning choice a caller can recover from at runtime, so both are rejected up front.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"branch must be an exact integer >= 2, got {value!r}")
    branch = int(value)
    if branch < 2:
        raise ValueError(f"branch must be >= 2 to reduce a level, got {branch!r} (branch=1 never terminates)")
    return branch


def tree_reduce_values(values: list[Any], factory: Any, branch: int = 2) -> Any:
    """Fold accumulator ``value()`` payloads with a ``branch``-ary tree of ``combine()`` -- O(log n) depth.

    Each internal node makes a fresh accumulator (so a shared-reference ``value()`` is never mutated --
    the HMM-stat aliasing hazard), seeds it from the first child, and combines the rest. Bit-identical to
    a serial left fold for associative integer statistics.

    Raises:
        TypeError: if ``branch`` is not an exact integer.
        ValueError: if ``values`` is empty, or ``branch < 2`` when a reduction is actually needed.
    """
    if not values:
        raise ValueError("nothing to reduce")
    level = list(values)
    if len(level) > 1:
        # Validated only when a reduction has to happen: a single payload is returned untouched, so a
        # one-shard run must not be failed by a branch it never uses.
        branch = _reduction_branch(branch)
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
        sizes = list(sizes)
        if not sizes:
            raise ValueError("shard_sizes must name at least one shard.")
        # Checking only the sum is not enough to establish a partition. Python slicing reinterprets a
        # negative bound as an offset from the end, so [-2, -2, 9] over five rows -- which does sum to
        # five -- yields the slices [0:-2], [-2:-4], [-4:5]: rows 1 and 2 are processed TWICE, row 0
        # once, and the E-step reports seven observations for five rows. Fractional sizes pass the sum
        # test too and only fail later, deep inside the slice. Each size must therefore be an exact
        # nonnegative integer in its own right, so the cumulative bounds are a genuine monotone
        # partition starting at 0 and ending exactly at n.
        #
        # Zero IS allowed: a WorkerAssignment may legitimately carry zero rows (plan_heterogeneous
        # validates rows as nonnegative, and a slow worker can round to nothing), shards_from_plan
        # passes those straight through, and an empty shard is already skipped by the loop below. Only
        # negative sizes -- which no plan can produce and which silently duplicate rows -- are rejected.
        for i, s in enumerate(sizes):
            if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
                raise TypeError(f"shard_sizes[{i}] must be an exact nonnegative integer, got {s!r}")
            if int(s) < 0:
                raise ValueError(
                    f"shard_sizes[{i}] must be nonnegative, got {int(s)!r}; a negative size is read by "
                    "Python slicing as an offset from the end and silently duplicates rows across shards"
                )
        total = sum(int(s) for s in sizes)
        if total != n:
            # a mismatch (e.g. a precomputed HeterogeneousPlan sized against a different sample)
            # used to silently slice off whichever rows fell outside the last bound -- caller-
            # supplied shard_sizes must cover the data exactly, not less or more.
            raise ValueError(f"shard_sizes must sum to len(data) ({n}), got {total}.")
        bounds, off = [], 0
        for s in sizes:
            bounds.append((off, off + int(s)))
            off += int(s)
        return bounds
    if isinstance(n_shards, bool) or not isinstance(n_shards, (int, np.integer)) or int(n_shards) < 1:
        # A zero or negative shard count produces no bounds at all, so every row is silently dropped
        # and the reduce below is handed an empty list.
        raise ValueError(f"n_shards must be an exact positive integer, got {n_shards!r}")
    edges = np.linspace(0, n, int(n_shards) + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(int(n_shards))]


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

    Keyed (tied) estimators are handled exactly like every serial driver: keys are validated up front
    and the ``key_merge``/``key_replace`` pooling pass runs ONCE on the fully tree-reduced statistics
    (the MPI/multiprocessing drivers' driver-side contract), never per shard.
    """
    validate_estimator_keys(estimator)
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
    if total != len(data):
        # Independent of the bounds arithmetic above: the count the E-step actually processed is what
        # the M-step is weighted by, so a shard loop that double-counted or dropped rows must not reach
        # estimate() carrying a sample size the data never had.
        raise ValueError(
            f"sharded E-step processed {total} observation(s) for {len(data)} row(s) of data; "
            "shards must partition the data exactly once."
        )
    combined = tree_reduce_values(values, estimator.accumulator_factory(), branch)
    accumulator = estimator.accumulator_factory().make().from_value(combined)
    merge_accumulator_keys(accumulator)
    return estimator.estimate(float(total), accumulator.value())


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
    """Run ``max_its`` EM iterations with the distributed heterogeneous executor; returns the fitted model.

    Raises:
        TypeError: if ``max_its`` is not an exact integer.
        ValueError: if ``max_its`` is not positive.
    """
    # A zero or negative iteration count ran no EM at all and returned the *unfitted* input model, with
    # nothing in the return value to distinguish it from a completed fit.
    if isinstance(max_its, bool) or not isinstance(max_its, (int, np.integer)):
        raise TypeError(f"max_its must be an exact positive integer, got {max_its!r}")
    if int(max_its) < 1:
        raise ValueError(f"max_its must be positive, got {int(max_its)!r}; a fit that runs no iteration is not a fit")
    estimator = model.estimator()
    current = model
    for _ in range(int(max_its)):
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
