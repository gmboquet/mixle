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

from mixle.utils.parallel.planner import _split_range


def _mp_context():
    """Pick a process-pool start method for CPU-bound work.

    Prefer 'fork': it needs no ``if __name__ == '__main__'`` guard (works from notebooks and
    unguarded scripts) and the workers here run pure big-integer arithmetic, so the
    multi-threaded-fork hazard does not apply. Fall back to the platform default (spawn on
    Windows), which requires the caller's entry module to be import-guarded.
    """
    import multiprocessing as mp

    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def resolve_workers(num_workers: int | None = None) -> int:
    """Resolve a worker count from an explicit request, Resources, or the CPU count."""
    if num_workers is not None:
        return max(1, int(num_workers))
    try:
        from mixle.utils.parallel.planner import Resources

        n = len(Resources.local().devices)
        if n >= 1:
            return n
    except Exception:
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
    """

    def __init__(self, num_workers: int | None = None, min_parallel_width: int = 2048) -> None:
        self.num_workers = resolve_workers(num_workers)
        self.min_parallel_width = int(min_parallel_width)
        self._pool = None

    def __enter__(self) -> "ConvolutionExecutor":
        if self.num_workers > 1:
            self._pool = _mp_context().Pool(self.num_workers)
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
) -> list[tuple[Any, float]]:
    """Unrank the rank range [start, start+count) in parallel, returning items in rank order.

    Each worker rebuilds the index from ``dist`` (picklable) and unranks its assigned sub-range.
    ``count=None`` unranks to the end of the index. ``backend`` is ``'local'`` (process pool) or
    ``'spark'`` (requires ``spark_context``). The result equals the serial enumeration order.
    """
    workers = resolve_workers(num_workers)
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

    if workers <= 1 or len(ranges) <= 1:
        out: list[tuple[Any, float]] = []
        for lo, hi in ranges:
            out.extend(_unrank_chunk(dist, budget_bits, bin_width_bits, oversample, lo, hi))
        return out

    with _mp_context().Pool(workers) as pool:
        tasks = [(dist, budget_bits, bin_width_bits, oversample, lo, hi) for lo, hi in ranges]
        parts = pool.starmap(_unrank_chunk, tasks)
    out = []
    for part in parts:
        out.extend(part)
    return out
