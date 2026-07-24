"""Parallel quantization and distributed enumeration for the count-semiring index.

Two capabilities, both built on the existing resource/chunking layer
(:mod:`mixle.utils.parallel.planner` ``Resources`` and ``_split_range``):

  - **Parallel quantization** (:class:`ConvolutionExecutor`): the count-DP's cost is dominated
    by big-integer convolutions of count histograms (sequence power chains, composite suffix
    convolutions). Those are pure-data and embarrassingly parallel over the output bucket range,
    so a worker pool computes disjoint output slices and the result is concatenated. Output
    ``out[k] = sum_i a[i]*b[k-i]`` is independent of the chunk boundaries, so the parallel result
    equals the serial one exactly. Attach an executor to a ``Quantizer`` and the DP routes its
    convolutions through it (see :meth:`mixle.enumeration.quantization.Quantizer.convolve`).

  - **Distributed unranking** (:func:`distributed_unrank`): unranking a contiguous rank range is
    embarrassingly parallel. The structural unrankers are closures (not picklable), so each worker
    *rebuilds* the index from the (picklable) distribution and unranks only its assigned rank
    sub-range; the per-worker rebuild is duplicated but the unranking is parallelized. Works on a
    local process pool and on a Spark context.

Use parallel quantization when histograms are large (deep budgets / high oversample); for small
problems the serial path is used automatically to avoid pickling overhead.
"""

import os
from typing import Any

import numpy as np

from mixle.utils.parallel.planner import _split_range

# Recognized multiprocessing start methods a caller may request via `start_method`. A closed set,
# matching the `_VALID_BACKENDS` validation style elsewhere in this file: an unrecognized spelling
# is rejected outright rather than handed to `multiprocessing.get_context` unvalidated (MXR-080-0210).
_VALID_START_METHODS = ("spawn", "fork", "forkserver")

# Default start method for every worker pool in this module (MXR-080-0210). Forking the current
# process is not made safe by the fact that *this module's own* worker arithmetic is pure: fork()
# clones the parent's entire memory image at the instant it is called, including the internal
# state of whatever lock/mutex each of the parent's THREADS currently holds -- not just the
# calling thread. In a notebook/ML host with NumPy/BLAS/logging/accelerator background threads
# already running, a fork that lands while one of those threads holds a lock copies that lock into
# the child already locked; the thread that would have released it does not exist in the child
# (only the forking thread survives a fork), so the lock can never be released. That can deadlock
# or otherwise corrupt process-pool startup before any worker function even runs -- the hazard is
# in the fork itself, not in what the worker computes afterward, so "the workers only do pure
# arithmetic" does not defend against it.
#
# 'spawn' starts each worker from a fresh interpreter with no inherited memory or lock state, which
# sidesteps the hazard entirely, needs no per-platform fallback (unlike 'forkserver' below, it is
# available everywhere Python's multiprocessing is), and is already what the rest of this
# codebase's own parallel-execution utilities use unconditionally (see
# `mixle.utils.parallel.multiprocessing.MPEncodedData` and `...resilient_em`'s
# `mp.get_context("spawn")`) -- so this module no longer stands alone in taking on fork risk.
# 'forkserver' would also be safe (it forks fresh workers from a dedicated, kept-clean,
# single-threaded server process) and is cheaper per `Pool()` call than 'spawn' since it skips
# re-importing every module, which would suit `distributed_unrank`'s per-call pool (a fresh Pool
# per invocation) better than `ConvolutionExecutor`'s single pool reused across many convolutions
# -- but it is POSIX-only, so defaulting to it here would need the exact per-platform fallback
# branch 'spawn' avoids. A caller who has confirmed their environment is fork-safe (e.g. a
# genuinely single-threaded batch script) or wants 'forkserver's lower per-call startup cost can
# still opt in explicitly via `start_method`.
_DEFAULT_START_METHOD = "spawn"


def _resolve_start_method(start_method: str | None) -> str:
    """Substitute the safe default for ``None``; reject anything outside ``_VALID_START_METHODS``.

    Raises rather than silently falling back to the platform's raw default context (``'fork'`` on
    Linux -- the exact hazard this guards against, MXR-080-0210) or passing an unrecognized
    spelling through to ``multiprocessing`` unvalidated.
    """
    method = _DEFAULT_START_METHOD if start_method is None else start_method
    if method not in _VALID_START_METHODS:
        raise ValueError(f"start_method must be one of {_VALID_START_METHODS!r}, got {method!r}.")
    return method


def _mp_context(start_method: str | None = None):
    """Resolve a multiprocessing context using a safe, explicit start method (MXR-080-0210).

    ``start_method=None`` resolves to :data:`_DEFAULT_START_METHOD` (``'spawn'``) instead of the
    platform's raw default context, which is ``'fork'`` on Linux. See the ``_DEFAULT_START_METHOD``
    comment above for why 'spawn' is the default and when a caller might reasonably opt into
    another start method via the ``start_method`` argument threaded through from
    ``ConvolutionExecutor`` and ``distributed_unrank``.
    """
    import multiprocessing as mp

    return mp.get_context(_resolve_start_method(start_method))


def _require_exact_positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    """Validate ``value`` is an exact integer ``>= minimum``; raise instead of truncating/coercing.

    Local copy of the ``mixle.doe.designs._require_exact_positive_int`` convention -- not imported
    directly, since this is a distribution-kernel module and a two-line validator does not justify a
    new ``mixle.enumeration -> mixle.doe`` package dependency. Every worker/rank/count control in this
    module previously either clamped a non-positive value up to 1 (``max(1, ...)``) or truncated a
    fractional one (``int(...)``), silently changing a caller's request instead of rejecting it -- a
    caller passing ``workers=-5`` or ``start=2.5`` almost certainly has a bug, and clamping/truncating
    hides that bug rather than surfacing it (MXR-080-0209). Accepts a genuine Python/numpy integer, or
    a float/numpy float that is finite and has no fractional part (``5.0`` is fine, ``5.5`` is not);
    rejects ``bool`` (never a meaningful count) and non-numeric types.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool.")
    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        if float(value) != int(value):
            raise ValueError(f"{name} must be an exact integer, got {value!r}.")
        ivalue = int(value)
    else:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if ivalue < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {ivalue}.")
    return ivalue


# Closed set of recognized `backend` spellings for `distributed_unrank`. A caller's selection must be
# honored or rejected outright, never silently reinterpreted -- see `distributed_unrank` (MXR-080-0209).
_VALID_BACKENDS = ("local", "spark")


def resolve_workers(num_workers: int | None = None) -> int:
    """Resolve a worker count from an explicit request, Resources, or the CPU count.

    An explicit ``num_workers`` must be an exact positive integer: non-positive or fractional values
    are rejected rather than silently clamped to 1 or truncated (MXR-080-0209). ``None`` defers to
    auto-detection from ``Resources`` or the CPU count, which are not caller-supplied values and so
    are still floored at 1 the way they always were (a sandboxed ``os.cpu_count()`` can return 0/None).
    """
    if num_workers is not None:
        return _require_exact_positive_int(num_workers, "num_workers")
    try:
        from mixle.utils.parallel.planner import Resources

        n = len(Resources.local().devices)
        if n >= 1:
            return n
    except Exception:  # noqa: BLE001
        pass
    return max(1, os.cpu_count() or 1)


# --- Parallel quantization: chunked convolution -----------------------------------------------


def _conv_chunk(a_data: list[int], b_data: list[int], lo: int, hi: int) -> list[int]:
    """Compute output buckets [lo, hi) of conv(a_data, b_data) (0-indexed in the result)."""
    na, nb = len(a_data), len(b_data)
    out = [0] * (hi - lo)
    for k in range(lo, hi):
        i_lo = 0 if k < nb else k - (nb - 1)
        i_hi = k if k < na else na - 1
        acc = 0
        for i in range(i_lo, i_hi + 1):
            ai = a_data[i]
            if ai:
                bj = b_data[k - i]
                if bj:
                    acc += ai * bj
        out[k - lo] = acc
    return out


class ConvolutionExecutor:
    """Process-pool executor for big-integer histogram convolutions (parallel quantization).

    A context manager holding a reusable pool. ``convolve(a, b, max_fine_bucket)`` returns a
    :class:`mixle.enumeration.quantization.CountHistogram` equal to the serial convolution. Falls back to
    serial when the output is small or only one worker is available.

    ``start_method`` selects the pool's multiprocessing start method (MXR-080-0210): ``None`` (the
    default) resolves to the safe ``'spawn'`` default rather than forking this process, which is
    not safe when other threads may be active (a NumPy/BLAS/logging/accelerator thread pool, in
    the notebook/ML hosts this class targets) -- see the ``_DEFAULT_START_METHOD`` module comment
    for the full hazard. Pass ``'fork'`` or ``'forkserver'`` explicitly to opt into cheaper pool
    startup in an environment you have confirmed is fork-safe.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        min_parallel_width: int = 2048,
        start_method: str | None = None,
    ) -> None:
        self.num_workers = resolve_workers(num_workers)
        self.min_parallel_width = _require_exact_positive_int(min_parallel_width, "min_parallel_width", minimum=0)
        self.start_method = _resolve_start_method(start_method)
        self._pool = None

    def __enter__(self) -> "ConvolutionExecutor":
        if self.num_workers > 1:
            self._pool = _mp_context(self.start_method).Pool(self.num_workers)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Close the worker pool, if one is active."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def convolve(self, a, b, max_fine_bucket: int | None = None):
        """Convolve two histograms, optionally using worker processes."""
        from mixle.enumeration.quantization.core import CountHistogram

        if not a.data or not b.data:
            return CountHistogram.empty()
        base = a.base + b.base
        width = len(a.data) + len(b.data) - 1
        if max_fine_bucket is not None:
            cap = int(max_fine_bucket) - base + 1
            if cap <= 0:
                return CountHistogram.empty()
            width = min(width, cap)
        if self._pool is None or self.num_workers <= 1 or width < self.min_parallel_width:
            return a.convolve(b, max_fine_bucket=max_fine_bucket)
        ranges = _split_range(0, width, self.num_workers)
        tasks = [(a.data, b.data, lo, hi) for lo, hi in ranges]
        parts = self._pool.starmap(_conv_chunk, tasks)
        data: list[int] = []
        for part in parts:
            data.extend(part)
        return CountHistogram(base, data)


# --- Distributed unranking --------------------------------------------------------------------


def _unrank_chunk(
    dist, budget_bits: float, bin_width_bits: float, oversample: int, lo: int, hi: int
) -> list[tuple[Any, float]]:
    """Rebuild the budget index on the worker and unrank ranks [lo, hi)."""
    index = dist.count_budget_index(budget_bits, bin_width_bits=bin_width_bits, oversample=oversample)
    top = min(hi, index.total_count)
    return [index.get(i) for i in range(lo, top)]


def distributed_unrank(
    dist,
    budget_bits: float,
    start: int = 0,
    count: int | None = None,
    bin_width_bits: float = 1.0,
    oversample: int = 8,
    num_workers: int | None = None,
    backend: str = "local",
    spark_context=None,
    start_method: str | None = None,
) -> list[tuple[Any, float]]:
    """Unrank the rank range [start, start+count) in parallel, returning items in rank order.

    Each worker rebuilds the index from ``dist`` (picklable) and unranks its assigned sub-range.
    ``count=None`` unranks to the end of the index. ``backend`` must be exactly ``'local'`` (process
    pool) or ``'spark'`` (requires ``spark_context``); any other spelling -- including a near-miss
    typo like ``'Spark'``, ``'spark '``, or ``'pyspark'`` -- is rejected rather than silently falling
    back to local execution, which could otherwise run cluster-sized work on the caller's own machine
    with no indication the requested backend was ignored (MXR-080-0209). ``start`` and an explicit
    ``count`` must be exact non-negative integers, validated before they reach the range splitter.
    The result equals the serial enumeration order.

    For ``backend='local'``, ``start_method`` selects the worker pool's multiprocessing start
    method (MXR-080-0210): ``None`` (the default) resolves to the safe ``'spawn'`` default rather
    than forking this process, which is not safe when other threads may be active -- see the
    ``_DEFAULT_START_METHOD`` module comment in this file for the full hazard. Validated eagerly
    even for ``backend='spark'``, where it has no effect (no local pool is started), so a typo'd
    ``start_method`` is rejected rather than silently ignored.
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS!r}, got {backend!r}.")
    workers = resolve_workers(num_workers)
    start = _require_exact_positive_int(start, "start", minimum=0)
    if count is not None:
        count = _require_exact_positive_int(count, "count", minimum=0)
    start_method = _resolve_start_method(start_method)
    if count is None:
        # Build once to learn the size (the workers rebuild independently).
        index = dist.count_budget_index(budget_bits, bin_width_bits=bin_width_bits, oversample=oversample)
        count = max(0, index.total_count - start)
    stop = start + count
    ranges = _split_range(start, stop, workers)

    if backend == "spark":
        if spark_context is None:
            raise ValueError("backend='spark' requires spark_context")
        bb, bw, ov = budget_bits, bin_width_bits, oversample
        rdd = spark_context.parallelize(ranges, len(ranges))
        pairs = rdd.flatMap(lambda r: _unrank_chunk(dist, bb, bw, ov, r[0], r[1])).collect()
        return pairs

    # backend == "local": the only other member of _VALID_BACKENDS, guaranteed by the check above.
    if workers <= 1 or len(ranges) <= 1:
        out: list[tuple[Any, float]] = []
        for lo, hi in ranges:
            out.extend(_unrank_chunk(dist, budget_bits, bin_width_bits, oversample, lo, hi))
        return out

    with _mp_context(start_method).Pool(workers) as pool:
        tasks = [(dist, budget_bits, bin_width_bits, oversample, lo, hi) for lo, hi in ranges]
        parts = pool.starmap(_unrank_chunk, tasks)
    out = []
    for part in parts:
        out.extend(part)
    return out
