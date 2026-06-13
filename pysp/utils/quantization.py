"""Structural quantized enumeration: count the support without enumerating it.

The smart-enumeration index (see :mod:`pysp.utils.enumeration` and
:class:`pysp.stats.pdist.DistributionEnumerator`) is a *seek structure* over a
distribution's exact descending-probability enumeration: it precomputes how many
support values fall in each quantized log-probability bin so that an arbitrary rank
can be resolved and unranked without walking the prefix.

For exponential-support families (sequences, Markov chains, ...) the number of
values within a probability budget is astronomically large, so the index must be
built from per-bin **counts computed structurally** -- by quantizing and binning the
*complete* model probability ``Q(ln p(x))`` and lifting the model's own likelihood
recursion into a count/histogram semiring -- never by materializing the domain.

This module provides that semiring:

  - :class:`Quantizer` maps an exact ``log_prob`` to a fine integer bucket of
    accumulated bits and a coarse output bin.
  - :class:`CountHistogram` is the semiring value: counts indexed by fine bucket,
    with ``shift`` (add a constant log-prob term), ``convolve`` (independent additive
    composition), ``power`` (L-fold self-convolution), and ``add`` (pool alternatives).
    Counts are exact Python integers and may exceed 2**128.
  - :class:`CountIndex` pairs a histogram with a structural *unranker*
    ``get_in_bucket(fine_bucket, offset) -> (value, exact_log_prob)``.
  - :func:`leaf_count_index` builds a :class:`CountIndex` from any exact enumerator,
    bounded by depth (cheap for small/closed-form leaves).
  - :func:`convolve_indices` composes child indices (the Composite reference case),
    and :func:`build_budget_index` accumulates coarse bins until the cumulative count
    reaches the requested ``2**budget_bits`` budget, returning a
    :class:`pysp.utils.enumeration.LazyQuantizedEnumerationIndex`.

The bin assignment is the only approximation (intermediate fine-bucket rounding shifts
items by at most ``num_terms / oversample`` coarse bins); the *value set*, *total
count*, and the *exact log probability* of every unranked item are exact, because the
unranker returns the value and the index re-evaluates ``log_density`` on it.
"""
import bisect
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

_LOG2 = math.log(2.0)
_TOL = 1.0e-9


class Quantizer(object):
    """Maps exact log probabilities to fine buckets and coarse bins.

    bits(x) = -log2 p(x) >= 0. The fine bucket is floor(bits * oversample / bin_width);
    the coarse bin is fine_bucket // oversample, which equals floor(bits / bin_width) for
    a single value (so leaf coarse bins match the existing QuantizedEnumerationIndex
    convention). ``oversample`` (R) controls how finely accumulated log-probabilities are
    tracked through convolutions, bounding the requantization error.
    """

    __slots__ = ('bin_width_bits', 'oversample', 'executor')

    def __init__(self, bin_width_bits: float = 1.0, oversample: int = 8, executor=None) -> None:
        if bin_width_bits <= 0:
            raise ValueError('bin_width_bits must be positive.')
        if int(oversample) < 1:
            raise ValueError('oversample must be a positive integer.')
        self.bin_width_bits = float(bin_width_bits)
        self.oversample = int(oversample)
        # Optional convolution executor (see pysp.utils.quantization_parallel). Lives only in the
        # building process; the count-DP routes its heavy convolutions through it when present.
        self.executor = executor

    def convolve(self, a: 'CountHistogram', b: 'CountHistogram',
                 max_fine_bucket: Optional[int] = None) -> 'CountHistogram':
        """Convolve two histograms, using the attached parallel executor when present."""
        if self.executor is not None:
            return self.executor.convolve(a, b, max_fine_bucket)
        return a.convolve(b, max_fine_bucket=max_fine_bucket)

    def bits(self, log_prob: float) -> float:
        """Information content -log2 p in bits (>= 0)."""
        return max(0.0, -float(log_prob) / _LOG2)

    def fine_bucket(self, log_prob: float) -> int:
        """Fine integer bucket of a log probability."""
        b = self.bits(log_prob) * self.oversample / self.bin_width_bits
        return int(math.floor(b + _TOL))

    def coarse_bin(self, fine_bucket: int) -> int:
        """Coarse output bin containing a fine bucket."""
        return int(fine_bucket) // self.oversample

    def fine_per_bit(self) -> float:
        """Fine buckets per bit of information."""
        return self.oversample / self.bin_width_bits


class CountHistogram(object):
    """Counts of support values indexed by fine bucket of accumulated bits.

    ``data[i]`` is the (exact, possibly huge) number of values whose fine bucket is
    ``base + i``. The histogram is dense over ``[base, base + len(data))`` with implicit
    zeros outside. This is the value type of the count semiring.
    """

    __slots__ = ('base', 'data')

    def __init__(self, base: int, data: List[int]) -> None:
        self.base = int(base)
        self.data = list(data)
        self._normalize()

    def _normalize(self) -> None:
        d = self.data
        # Trim leading zeros (advancing base) and trailing zeros.
        lo = 0
        n = len(d)
        while lo < n and d[lo] == 0:
            lo += 1
        if lo == n:
            self.base = 0
            self.data = []
            return
        hi = n
        while hi > lo and d[hi - 1] == 0:
            hi -= 1
        if lo != 0 or hi != n:
            self.base += lo
            self.data = d[lo:hi]

    @classmethod
    def empty(cls) -> 'CountHistogram':
        return cls(0, [])

    @classmethod
    def delta(cls, fine_bucket: int, count: int = 1) -> 'CountHistogram':
        """A single bucket with the given count (the multiplicative identity when count=1, bucket=0)."""
        return cls(fine_bucket, [int(count)])

    def is_empty(self) -> bool:
        return not self.data

    def total(self) -> int:
        return sum(self.data)

    def max_bucket(self) -> Optional[int]:
        return None if not self.data else self.base + len(self.data) - 1

    def count_at(self, fine_bucket: int) -> int:
        i = int(fine_bucket) - self.base
        return self.data[i] if 0 <= i < len(self.data) else 0

    def shift(self, k: int) -> 'CountHistogram':
        """Return a copy with every bucket moved by k (adds a constant log-prob term)."""
        if not self.data:
            return CountHistogram.empty()
        return CountHistogram(self.base + int(k), list(self.data))

    def truncate(self, max_fine_bucket: int) -> 'CountHistogram':
        """Drop buckets strictly beyond ``max_fine_bucket`` (depth bound)."""
        if not self.data:
            return CountHistogram.empty()
        hi = int(max_fine_bucket) - self.base + 1
        if hi <= 0:
            return CountHistogram.empty()
        if hi >= len(self.data):
            return CountHistogram(self.base, list(self.data))
        return CountHistogram(self.base, self.data[:hi])

    def add(self, other: 'CountHistogram') -> 'CountHistogram':
        """Pointwise sum (pool of mutually exclusive alternatives, e.g. different lengths)."""
        if not self.data:
            return CountHistogram(other.base, list(other.data))
        if not other.data:
            return CountHistogram(self.base, list(self.data))
        base = min(self.base, other.base)
        end = max(self.base + len(self.data), other.base + len(other.data))
        out = [0] * (end - base)
        for i, c in enumerate(self.data):
            if c:
                out[self.base + i - base] += c
        for i, c in enumerate(other.data):
            if c:
                out[other.base + i - base] += c
        return CountHistogram(base, out)

    def convolve(self, other: 'CountHistogram', max_fine_bucket: Optional[int] = None) -> 'CountHistogram':
        """Discrete convolution: counts of sums of two independent additive log-prob terms.

        Optionally drop output buckets beyond ``max_fine_bucket`` during accumulation so a
        depth bound keeps the histogram width fixed regardless of how large the counts grow.
        """
        a, b = self.data, other.data
        if not a or not b:
            return CountHistogram.empty()
        base = self.base + other.base
        width = len(a) + len(b) - 1
        if max_fine_bucket is not None:
            cap = int(max_fine_bucket) - base + 1
            if cap <= 0:
                return CountHistogram.empty()
            width = min(width, cap)
        out = [0] * width
        for i, ai in enumerate(a):
            if not ai:
                continue
            hi = width - i
            if hi <= 0:
                break
            for j in range(min(len(b), hi)):
                bj = b[j]
                if bj:
                    out[i + j] += ai * bj
        return CountHistogram(base, out)


class CountIndex(object):
    """A fine-bucket count histogram paired with a structural unranker.

    ``get_in_bucket(fine_bucket, offset)`` returns ``(value, exact_log_prob)`` for the
    ``offset``-th value (0-based) whose fine bucket is ``fine_bucket``. The ordering within
    a bucket is deterministic but otherwise unspecified.
    """

    __slots__ = ('hist', '_getter')

    def __init__(self, hist: CountHistogram,
                 getter: Callable[[int, int], Tuple[Any, float]]) -> None:
        self.hist = hist
        self._getter = getter

    def total(self) -> int:
        return self.hist.total()

    def get_in_bucket(self, fine_bucket: int, offset: int) -> Tuple[Any, float]:
        if offset < 0 or offset >= self.hist.count_at(fine_bucket):
            raise IndexError('offset %d outside fine bucket %d' % (offset, fine_bucket))
        return self._getter(int(fine_bucket), int(offset))


def leaf_count_index(enum: Iterator[Tuple[Any, float]], quantizer: Quantizer,
                     max_fine_bucket: int, max_items: Optional[int] = None) -> Tuple[CountIndex, bool]:
    """Build a CountIndex from an exact descending-probability enumerator, bounded by depth.

    Pulls items until their fine bucket exceeds ``max_fine_bucket`` (or, if ``max_items`` is given,
    until that many items have been taken). Cheap for closed-form or small-support leaves (a
    geometric/Poisson has ~depth items within a depth bound); the ``max_items`` cap is the brake for
    the enumerate-and-bin fallback over exponential-support families that cannot count structurally.
    Returns ``(index, truncated)`` where ``truncated`` is True if in-bound items were left untaken.
    """
    by_bucket: Dict[int, List[Tuple[Any, float]]] = {}
    truncated = False
    taken = 0
    for value, log_prob in enum:
        if log_prob == -math.inf:
            continue
        fb = quantizer.fine_bucket(log_prob)
        if fb > max_fine_bucket:
            truncated = True
            break
        by_bucket.setdefault(fb, []).append((value, float(log_prob)))
        taken += 1
        if max_items is not None and taken >= max_items:
            truncated = True
            break

    if not by_bucket:
        return CountIndex(CountHistogram.empty(), lambda fb, off: (_ for _ in ()).throw(IndexError())), truncated

    lo = min(by_bucket)
    hi = max(by_bucket)
    data = [0] * (hi - lo + 1)
    for fb, items in by_bucket.items():
        data[fb - lo] = len(items)
    hist = CountHistogram(lo, data)

    def getter(fb: int, off: int) -> Tuple[Any, float]:
        return by_bucket[fb][off]

    return CountIndex(hist, getter), truncated


def child_count_index(child, path: str, quantizer: Quantizer,
                      max_fine_bucket: int) -> Tuple[CountIndex, bool]:
    """Build child.quantized_count_index(...), annotating EnumerationError with the child's path."""
    from pysp.stats.pdist import EnumerationError
    try:
        return child.quantized_count_index(quantizer, max_fine_bucket)
    except EnumerationError as e:
        new_path = path if not e.path else '%s -> %s' % (path, e.path)
        raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None


def convolve_indices(children: Sequence[CountIndex], quantizer: Quantizer,
                     max_fine_bucket: int) -> CountIndex:
    """Compose independent child indices into their additive (convolution) product.

    The joint histogram is the convolution of the child histograms (capped at the depth
    bound). Unranking resolves the per-child fine buckets that sum to the target, then the
    per-child offsets via mixed-radix decomposition using suffix-convolution counts. This is
    the Composite reference case; the empty product is the single empty tuple at bucket 0.
    """
    n = len(children)
    if n == 0:
        empty_hist = CountHistogram.delta(0, 1)
        return CountIndex(empty_hist, lambda fb, off: ((), 0.0))

    # Suffix convolutions: suffix[i] = conv(children[i].hist, ..., children[n-1].hist).
    suffix: List[CountHistogram] = [None] * (n + 1)  # type: ignore
    suffix[n] = CountHistogram.delta(0, 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = quantizer.convolve(children[i].hist, suffix[i + 1], max_fine_bucket=max_fine_bucket)

    joint = suffix[0]

    def getter(fb: int, off: int) -> Tuple[Any, float]:
        values: List[Any] = []
        log_prob = 0.0
        remaining = int(fb)
        o = int(off)
        for i in range(n):
            child = children[i]
            tail = suffix[i + 1]
            # Iterate this child's buckets in increasing order; completions for the
            # remaining children at (remaining - b) come from the suffix histogram.
            chosen = None
            for b in range(child.hist.base, child.hist.base + len(child.hist.data)):
                ci = child.hist.count_at(b)
                if not ci:
                    continue
                rem = remaining - b
                m = tail.count_at(rem)
                if not m:
                    continue
                block = ci * m
                if o < block:
                    local = o // m       # index of this child's item within bucket b
                    o = o % m            # offset into the remaining-children block
                    cval, clp = child.get_in_bucket(b, local)
                    values.append(cval)
                    log_prob += clp
                    remaining = rem
                    chosen = b
                    break
                o -= block
            if chosen is None:
                raise IndexError('offset outside convolution bucket %d' % fb)
        return tuple(values), float(log_prob)

    return CountIndex(joint, getter)


# ---------------------------------------------------------------------------
# Budget-driven coarse index (the new count-budget mode).
# ---------------------------------------------------------------------------

def build_budget_index(index: CountIndex, quantizer: Quantizer, budget_bits: float,
                       value_combine: Optional[Callable[[Any], Any]] = None,
                       exact_log_density: Optional[Callable[[Any], float]] = None,
                       truncated: bool = False):
    """Wrap a CountIndex as a budget-bounded LazyQuantizedEnumerationIndex.

    Accumulates coarse bins (fine buckets grouped by ``quantizer.oversample``) in
    descending-probability order until the cumulative count reaches ``2**budget_bits``, then
    stops. The returned getter maps a coarse (bin, offset) to a fine (bucket, offset),
    unranks the structural value, optionally maps it with ``value_combine`` (e.g. tuple->list),
    and reports the exact log density (recomputed via ``exact_log_density`` when supplied,
    otherwise the structurally accumulated log probability).
    """
    from pysp.utils.enumeration import LazyQuantizedEnumerationIndex

    R = quantizer.oversample
    bw = quantizer.bin_width_bits
    hist = index.hist
    budget = None if budget_bits is None else _two_pow(budget_bits)

    # Group fine buckets into coarse bins, in increasing depth (descending probability).
    coarse_counts: Dict[int, int] = {}
    # For each coarse bin: ordered list of (fine_bucket, count) and the cumulative offset
    # boundaries, so a within-bin offset maps to a fine bucket in O(log #fine).
    coarse_layout: Dict[int, Tuple[List[int], List[int], List[int]]] = {}
    cumulative = 0
    covered_truncated = truncated
    if hist.data:
        # Walk coarse bins in order; within a coarse bin walk its fine buckets in order.
        by_coarse: Dict[int, List[Tuple[int, int]]] = {}
        for i, c in enumerate(hist.data):
            if not c:
                continue
            fb = hist.base + i
            cb = fb // R
            by_coarse.setdefault(cb, []).append((fb, c))
        stop = False
        for cb in sorted(by_coarse):
            if stop:
                covered_truncated = True
                break
            fine_buckets = []
            fine_counts = []
            starts = []
            running = 0
            bin_total = 0
            for fb, c in by_coarse[cb]:
                fine_buckets.append(fb)
                fine_counts.append(c)
                starts.append(running)
                running += c
                bin_total += c
            coarse_counts[cb] = bin_total
            coarse_layout[cb] = (fine_buckets, starts, fine_counts)
            cumulative += bin_total
            if budget is not None and cumulative >= budget:
                stop = True  # budget met; include this bin, then stop adding deeper bins.
        else:
            # Loop finished without breaking: did we exhaust the histogram?
            covered_truncated = covered_truncated  # leave as-is (depth bound handled by caller)

    def getter(bin_id: int, offset: int) -> Tuple[Any, float]:
        layout = coarse_layout.get(bin_id)
        if layout is None:
            raise IndexError('coarse bin %d not indexed' % bin_id)
        fine_buckets, starts, fine_counts = layout
        # Find the fine bucket whose [start, start+count) range contains offset.
        j = bisect.bisect_right(starts, offset) - 1
        if j < 0 or offset >= starts[j] + fine_counts[j]:
            raise IndexError('offset %d outside coarse bin %d' % (offset, bin_id))
        fb = fine_buckets[j]
        value, lp = index.get_in_bucket(fb, offset - starts[j])
        if value_combine is not None:
            value = value_combine(value)
        if exact_log_density is not None:
            lp = float(exact_log_density(value))
        return value, lp

    max_bin = max(coarse_counts) if coarse_counts else 0
    return LazyQuantizedEnumerationIndex(
        coarse_counts, bin_width_bits=bw, max_bits=float(max_bin) * bw,
        truncated=covered_truncated, getter=getter)


def count_budget_index(dist, budget_bits: float, bin_width_bits: float = 1.0,
                       oversample: int = 8, max_depth_bits: float = 4096.0,
                       num_workers: Optional[int] = None):
    """Driver for the count-budget mode: deepen until the budget is covered, then build the index.

    Calls ``dist.quantized_count_index(quantizer, max_fine_bucket)`` at geometrically increasing
    depths until the structural count reaches ``2**budget_bits`` or the support is exhausted (no
    further truncation), then wraps the result with :func:`build_budget_index`. The exact log
    density of each unranked value is reported via ``dist.log_density``.

    When ``num_workers`` is greater than 1, the heavy count-histogram convolutions are computed on
    a process pool (parallel quantization); the parallel result is identical to the serial one.
    """
    executor = None
    if num_workers is not None and int(num_workers) > 1:
        from pysp.utils.quantization_parallel import ConvolutionExecutor
        executor = ConvolutionExecutor(num_workers=num_workers)
    try:
        if executor is not None:
            executor.__enter__()
        q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample, executor=executor)
        budget = _two_pow(budget_bits)
        depth_bits = max(float(bin_width_bits), float(budget_bits))
        index = None
        truncated = True
        fine_per_bit = q.fine_per_bit()
        while True:
            max_fb = int(math.ceil(depth_bits * fine_per_bit))
            index, truncated = dist.quantized_count_index(q, max_fb)
            if index.total() >= budget or not truncated:
                break
            if depth_bits >= max_depth_bits:
                break
            depth_bits = min(depth_bits * 2.0, max_depth_bits)
        return build_budget_index(index, q, budget_bits,
                                  exact_log_density=dist.log_density, truncated=truncated)
    finally:
        if executor is not None:
            executor.close()


def distinct_budget_stream(dist, budget_bits: float, bin_width_bits: float = 1.0,
                           oversample: int = 8, dedup: str = 'canonical',
                           start: int = 0, stop: Optional[int] = None,
                           max_entries: int = 1 << 16,
                           num_workers: Optional[int] = None) -> Iterator[Tuple[Any, float]]:
    """Yield DISTINCT (value, exact_log_prob) from a count-budget index, in approx descending order.

    Builds the budget index and removes repeats by one of two modes (see
    ``ProbabilityDistribution.count_budget_distinct`` for the full contract):

      - ``dedup='canonical'``: stateless per-item predicate (``dist.is_canonical_copy``), O(1)
        memory and random-accessible -- ``start``/``stop`` choose a STRUCTURAL rank range, so the
        distinct enumeration can begin anywhere and partition across workers with no shared state.
      - ``dedup='window'``: a bounded O(max_entries) LRU over the stream (sequential; ``start`` must
        be 0).

    Exact-count families never duplicate, so either mode is a pass-through.
    """
    index = count_budget_index(dist, budget_bits, bin_width_bits=bin_width_bits,
                               oversample=oversample, num_workers=num_workers)
    n = index.total_count
    stop = n if stop is None else min(int(stop), n)
    start = max(0, int(start))

    if dedup == 'window':
        if start != 0:
            raise ValueError("dedup='window' is sequential; start must be 0 (use 'canonical' to seek)")
        from pysp.utils.quantization_semiring import bounded_dedup_stream
        raw = (index.get(i) for i in range(start, stop))
        return bounded_dedup_stream(raw, max_entries=max_entries)

    if dedup != 'canonical':
        raise ValueError("dedup must be 'canonical' or 'window'")
    quantizer = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)

    def gen():
        for i in range(start, stop):
            coarse_bin, _off = index.bin_for_index(i)
            value, lp = index.get(i)
            if dist.is_canonical_copy(value, coarse_bin, quantizer):
                yield value, lp
    return gen()


def _two_pow(bits: float) -> int:
    """2**bits as an exact integer ceiling (budget is a count threshold)."""
    b = float(bits)
    if b <= 0:
        return 1
    fl = int(math.floor(b))
    frac = b - fl
    base = 1 << fl
    if frac <= 0:
        return base
    return int(math.ceil(base * (2.0 ** frac)))
