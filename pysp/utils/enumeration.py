"""Shared infrastructure for smart enumeration.

Smart enumeration lets a distribution lazily iterate its support in non-increasing
probability order, yielding (value, log_prob) pairs. The helpers here implement the
generic algorithms used by the combinator distributions:

  - BufferedStream: random access by rank into a lazy sorted stream.
  - freeze: canonical hashable keys for de-duplication of support values.
  - merge_enumerators: k-way merge of sorted streams with disjoint supports.
  - ProductEnumerator: best-first search over a Cartesian product of sorted streams.
  - best_first_union / best_first_union_max: union of sorted streams with possibly
    overlapping supports, re-scored exactly and emitted in provably correct order.

See pysp.stats.compute.pdist.DistributionEnumerator for the enumeration contract.
"""

import bisect
import heapq
import itertools
import math
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any

import numpy as np

from pysp.utils.vector import log_sum

__all__ = [
    "BufferedStream",
    "freeze",
    "merge_enumerators",
    "ProductEnumerator",
    "LengthFrontierMerge",
    "best_first_union",
    "best_first_union_max",
    "sound_top_k",
    "bounded_best_first_union_index",
    "QuantizedEnumerationIndex",
    "LazyQuantizedEnumerationIndex",
    "QuantizedCrossIndex",
    "quantized_index",
    "supports_enumeration",
]


_NAN_SENTINEL = ("__pysp_nan__",)


def freeze(x: Any) -> Hashable:
    """Return a canonical hashable key for x, for de-duplicating support values.

    Lists/tuples freeze element-wise to tuples, dicts to frozensets of (key, value)
    pairs, sets to frozensets, numpy arrays to (shape, bytes), numpy scalars to their
    python equivalents, and NaN to a shared sentinel (so nan == nan for dedup purposes).
    Raises TypeError for values that cannot be canonicalized.
    """
    if isinstance(x, (list, tuple)):
        return tuple(freeze(u) for u in x)
    if isinstance(x, dict):
        return frozenset((freeze(k), freeze(v)) for k, v in x.items())
    if isinstance(x, (set, frozenset)):
        return frozenset(freeze(u) for u in x)
    if isinstance(x, np.ndarray):
        return (x.shape, x.tobytes())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return _NAN_SENTINEL
    try:
        hash(x)
    except TypeError:
        raise TypeError("Cannot compute an enumeration dedup key for value of type %s" % type(x).__name__)
    return x


def supports_enumeration(dist) -> bool:
    """Return True if dist.enumerator() can be constructed."""
    from pysp.stats.compute.pdist import EnumerationError

    try:
        dist.enumerator()
        return True
    except EnumerationError:
        return False


class BufferedStream:
    """Random access by rank into a lazy stream of (value, log_prob) pairs.

    get(i) extends an internal buffer as needed and returns the i-th item, or None
    if the stream has fewer than i+1 items. The underlying stream is consumed at
    most once regardless of how many consumers share this object.
    """

    def __init__(self, it: Iterator[tuple[Any, float]]) -> None:
        self._it = iter(it)
        self._buf: list[tuple[Any, float]] = []
        self._done = False

    def get(self, i: int) -> tuple[Any, float] | None:
        while not self._done and len(self._buf) <= i:
            try:
                self._buf.append(next(self._it))
            except StopIteration:
                self._done = True
        return self._buf[i] if i < len(self._buf) else None


class QuantizedEnumerationIndex:
    """Bounded, indexable view of an exact probability-ordered enumeration.

    Items are grouped into log-probability bins measured in bits:

        bits(x) = -log2(p(x))
        bin(x)  = floor(bits(x) / bin_width_bits)

    Only items whose bit cost is at most max_bits are indexed. The returned
    (value, log_prob) pairs always carry the original exact log probability; the
    quantized bins are used only for counting and rank lookup.

    The generic implementation can build from an exact enumerator stream, so it works
    for every currently enumerable discrete stats model. Simple distributions may also
    build directly from scored support items, and distribution-specific dynamic programs
    can produce the same bin layout without relying on an exact global stream.
    """

    def __init__(
        self,
        bins: Sequence[tuple[int, list[tuple[Any, float]]]],
        bin_width_bits: float,
        max_bits: float,
        truncated: bool,
    ) -> None:
        self._bins = [(int(b), list(items)) for b, items in bins]
        self.bin_width_bits = float(bin_width_bits)
        self.max_bits = float(max_bits)
        self.truncated = bool(truncated)
        self.counts: dict[int, int] = {b: len(items) for b, items in self._bins}
        self._bin_lookup: dict[int, list[tuple[Any, float]]] = {b: items for b, items in self._bins}
        self._starts: dict[int, int] = {}
        self._cum_starts: list[int] = []
        self._cum_bins: list[int] = []
        pos = 0
        for b, items in self._bins:
            self._starts[b] = pos
            self._cum_starts.append(pos)
            self._cum_bins.append(b)
            pos += len(items)
        self.total_count = pos

    @staticmethod
    def bin_for_log_prob(log_prob: float, bin_width_bits: float = 1.0) -> int:
        """Return the quantized bit bin for a log probability."""
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")
        bits = max(0.0, -float(log_prob) / math.log(2.0))
        return int(math.floor(bits / bin_width_bits + 1.0e-12))

    @classmethod
    def from_enumerator(
        cls, enum: Iterator[tuple[Any, float]], max_bits: float, bin_width_bits: float = 1.0
    ) -> "QuantizedEnumerationIndex":
        """Build an index from an exact non-increasing-probability enumerator.

        Args:
            enum: Iterator yielding (value, log_prob) pairs in exact descending order.
            max_bits: Include only values with -log2(p) <= max_bits.
            bin_width_bits: Width of each quantized probability bin in bits.

        Returns:
            QuantizedEnumerationIndex over the bounded prefix/domain slice.

        """
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        bins: dict[int, list[tuple[Any, float]]] = {}
        truncated = False
        limit = float(max_bits) + 1.0e-12

        for value, log_prob in enum:
            if log_prob == -np.inf:
                continue
            bits = max(0.0, -float(log_prob) / math.log(2.0))
            if bits > limit:
                truncated = True
                break
            b = int(math.floor(bits / bin_width_bits + 1.0e-12))
            bins.setdefault(b, []).append((value, float(log_prob)))

        return cls([(b, bins[b]) for b in sorted(bins)], bin_width_bits, max_bits, truncated)

    @classmethod
    def from_items(
        cls,
        items: Sequence[tuple[Any, float]],
        max_bits: float,
        bin_width_bits: float = 1.0,
        sorted_items: bool = False,
        truncated: bool | None = None,
    ) -> "QuantizedEnumerationIndex":
        """Build an index from known support items and exact log probabilities.

        Args:
            items: Finite collection of (value, log_prob) pairs. Zero-probability
                items with log_prob == -inf are ignored.
            max_bits: Include only values with -log2(p) <= max_bits.
            bin_width_bits: Width of each quantized probability bin in bits.
            sorted_items: If True, preserve the input order. Otherwise, stable-sort
                included items by descending log probability.
            truncated: Optional explicit truncation flag. If omitted, it is True
                when any finite-probability item was outside the bit bound.

        Returns:
            QuantizedEnumerationIndex over the bounded finite item set.

        """
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        scored = []
        excluded = False
        limit = float(max_bits) + 1.0e-12
        for value, log_prob in items:
            lp = float(log_prob)
            if lp == -np.inf:
                continue
            bits = max(0.0, -lp / math.log(2.0))
            if bits > limit:
                excluded = True
                continue
            scored.append((value, lp))

        if not sorted_items:
            scored.sort(key=lambda u: -u[1])

        bins: dict[int, list[tuple[Any, float]]] = {}
        for value, lp in scored:
            b = cls.bin_for_log_prob(lp, bin_width_bits)
            bins.setdefault(b, []).append((value, lp))

        is_truncated = excluded if truncated is None else bool(truncated)
        return cls([(b, bins[b]) for b in sorted(bins)], bin_width_bits, max_bits, is_truncated)

    def __len__(self) -> int:
        return self.total_count

    def bin_for_index(self, index: int) -> tuple[int, int]:
        """Return (bin_id, offset_within_bin) for a bounded quantized rank."""
        if index < 0:
            raise IndexError("index must be non-negative.")
        if index >= self.total_count:
            raise IndexError("index %d outside indexed range of %d items." % (index, self.total_count))
        j = bisect.bisect_right(self._cum_starts, index) - 1
        return self._cum_bins[j], index - self._cum_starts[j]

    def get(self, index: int) -> tuple[Any, float]:
        """Return the indexed (value, exact_log_prob) pair."""
        b, offset = self.bin_for_index(index)
        return self._bin_lookup[b][offset]

    def slice(self, start: int, k: int) -> list[tuple[Any, float]]:
        """Return up to k indexed pairs starting at start."""
        if start < 0:
            raise IndexError("start must be non-negative.")
        if k < 0:
            raise ValueError("k must be non-negative.")
        return list(itertools.islice(self.iter_from(start), k))

    def iter_from(self, start: int = 0) -> Iterator[tuple[Any, float]]:
        """Iterate indexed pairs from start to the end of the bounded index."""
        if start < 0:
            raise IndexError("start must be non-negative.")
        if start >= self.total_count:
            return
        pos = 0
        for _, items in self._bins:
            n = len(items)
            if start >= pos + n:
                pos += n
                continue
            local = max(0, start - pos)
            yield from items[local:]
            pos += n

    def bin_items(self, bin_id: int) -> list[tuple[Any, float]]:
        """Return all indexed items in a quantized probability bin."""
        return list(self._bin_lookup.get(bin_id, []))

    def summary(self) -> dict[str, Any]:
        """Return a compact description of the bounded index."""
        return {
            "max_bits": self.max_bits,
            "bin_width_bits": self.bin_width_bits,
            "total_count": self.total_count,
            "num_bins": len(self._bins),
            "truncated": self.truncated,
            "counts": dict(self.counts),
        }


def quantized_index(
    enum: Iterator[tuple[Any, float]], max_bits: float, bin_width_bits: float = 1.0
) -> QuantizedEnumerationIndex:
    """Convenience wrapper for QuantizedEnumerationIndex.from_enumerator."""
    return QuantizedEnumerationIndex.from_enumerator(enum, max_bits=max_bits, bin_width_bits=bin_width_bits)


class LazyQuantizedEnumerationIndex(QuantizedEnumerationIndex):
    """Quantized index whose bin counts are precomputed but items are unranked lazily.

    This supports compositional distributions where a dynamic program can count how
    many values fall in each quantized log-density bin without materializing every
    Cartesian-product value. The getter receives a bin id and offset within that bin
    and returns the exact (value, log_prob) pair for that quantized rank.
    """

    def __init__(
        self,
        counts: dict[int, int],
        bin_width_bits: float,
        max_bits: float,
        truncated: bool,
        getter: Callable[[int, int], tuple[Any, float]],
    ) -> None:
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        self.bin_width_bits = float(bin_width_bits)
        self.max_bits = float(max_bits)
        self.truncated = bool(truncated)
        self.counts: dict[int, int] = {int(b): int(n) for b, n in sorted(counts.items()) if int(n) > 0}
        self._getter = getter
        self._bins = [(b, self.counts[b]) for b in sorted(self.counts)]
        self._starts: dict[int, int] = {}
        self._cum_starts: list[int] = []
        self._cum_bins: list[int] = []
        pos = 0
        for b, n in self._bins:
            self._starts[b] = pos
            self._cum_starts.append(pos)
            self._cum_bins.append(b)
            pos += n
        self.total_count = pos

    def bin_for_index(self, index: int) -> tuple[int, int]:
        """Return (bin_id, offset_within_bin) for a bounded quantized rank."""
        if index < 0:
            raise IndexError("index must be non-negative.")
        if index >= self.total_count:
            raise IndexError("index %d outside indexed range of %d items." % (index, self.total_count))
        j = bisect.bisect_right(self._cum_starts, index) - 1
        return self._cum_bins[j], index - self._cum_starts[j]

    def get(self, index: int) -> tuple[Any, float]:
        """Return the indexed (value, exact_log_prob) pair."""
        b, offset = self.bin_for_index(index)
        return self._getter(b, offset)

    def iter_from(self, start: int = 0) -> Iterator[tuple[Any, float]]:
        """Iterate indexed pairs from start to the end of the bounded index."""
        if start < 0:
            raise IndexError("start must be non-negative.")
        if start >= self.total_count:
            return
        pos = 0
        for b, n in self._bins:
            if start >= pos + n:
                pos += n
                continue
            local = max(0, start - pos)
            for offset in range(local, n):
                yield self._getter(b, offset)
            pos += n

    def bin_items(self, bin_id: int) -> list[tuple[Any, float]]:
        """Return all indexed items in a quantized probability bin."""
        n = self.counts.get(bin_id, 0)
        return [self._getter(bin_id, i) for i in range(n)]


class QuantizedCrossIndex:
    """Aligned support rows for multiple distributions under quantized bit bounds.

    Each row is (value, log_probs), where log_probs[j] is the exact log-density of
    value under component j, or -inf when that component assigns zero mass. Rows are
    included when any component's bit cost is within its requested bound. The joint
    bin counts expose the cross-bin alignment that mixtures need and marginal bin
    counts cannot provide.
    """

    def __init__(
        self,
        items: Sequence[tuple[Any, Sequence[float]]],
        max_bits: Sequence[float],
        bin_width_bits: float,
        truncated: bool = False,
    ) -> None:
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")
        self.items = [(v, tuple(float(lp) for lp in lps)) for v, lps in items]
        self.max_bits = tuple(float(b) for b in max_bits)
        self.bin_width_bits = float(bin_width_bits)
        self.truncated = bool(truncated)
        self.num_components = len(self.max_bits)
        self.counts: dict[tuple[int | None, ...], int] = {}
        for _, lps in self.items:
            bins = tuple(
                None if lp == -np.inf else QuantizedEnumerationIndex.bin_for_log_prob(lp, self.bin_width_bits)
                for lp in lps
            )
            self.counts[bins] = self.counts.get(bins, 0) + 1
        self.total_count = len(self.items)

    @staticmethod
    def bits_for_log_prob(log_prob: float) -> float:
        """Return -log2(p), using inf for zero probability."""
        lp = float(log_prob)
        return np.inf if lp == -np.inf else max(0.0, -lp / math.log(2.0))

    @classmethod
    def from_items(
        cls,
        items: Sequence[tuple[Any, Sequence[float]]],
        max_bits: Sequence[float],
        bin_width_bits: float = 1.0,
        truncated: bool = False,
    ) -> "QuantizedCrossIndex":
        """Build a cross index from exact aligned support rows."""
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = (float(max_bits),)
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        filtered = []
        excluded = False
        limits = tuple(b + 1.0e-12 for b in max_bits_tuple)
        for value, log_probs in items:
            lps = tuple(float(lp) for lp in log_probs)
            if len(lps) != len(max_bits_tuple):
                raise ValueError("log_probs length does not match max_bits length.")
            bits = tuple(cls.bits_for_log_prob(lp) for lp in lps)
            if any(bits[i] <= limits[i] for i in range(len(bits))):
                filtered.append((value, lps))
            else:
                excluded = True

        filtered.sort(key=lambda u: min(cls.bits_for_log_prob(lp) for lp in u[1]))
        return cls(filtered, max_bits_tuple, bin_width_bits, truncated=truncated or excluded)

    def iter_items(self) -> Iterator[tuple[Any, tuple[float, ...]]]:
        """Iterate aligned support rows."""
        return iter(self.items)

    def summary(self) -> dict[str, Any]:
        """Return a compact description of the cross index."""
        return {
            "max_bits": self.max_bits,
            "bin_width_bits": self.bin_width_bits,
            "total_count": self.total_count,
            "num_joint_bins": len(self.counts),
            "truncated": self.truncated,
            "counts": dict(self.counts),
        }


def merge_enumerators(
    streams: Sequence[Iterator[tuple[Any, float]]], offsets: Sequence[float]
) -> Iterator[tuple[Any, float]]:
    """Lazy k-way merge of sorted (value, log_prob) streams with per-stream offsets.

    Stream k's log probs are shifted by offsets[k]. Correct only when the streams
    have pairwise disjoint supports (no de-duplication or re-scoring is performed).
    """
    counter = itertools.count()
    heap = []
    its = [iter(s) for s in streams]
    for k, it in enumerate(its):
        if offsets[k] == -np.inf:
            continue
        for v, lp in it:
            heapq.heappush(heap, (-(lp + offsets[k]), next(counter), v, k))
            break
    while heap:
        neg_lp, _, v, k = heapq.heappop(heap)
        yield (v, -neg_lp)
        for v2, lp2 in its[k]:
            heapq.heappush(heap, (-(lp2 + offsets[k]), next(counter), v2, k))
            break


class ProductEnumerator:
    """Best-first enumeration of the Cartesian product of sorted child streams.

    Yields (combine(values), log_prob) with log_prob = offset + sum of child log
    probs, in non-increasing order. Standard k-best lattice search: a max-heap over
    index tuples, with successors advancing one coordinate; correctness follows from
    each child stream being sorted (coordinate-wise monotonicity).
    """

    def __init__(
        self, streams: Sequence[BufferedStream], combine: Callable[[tuple[Any, ...]], Any] = tuple, offset: float = 0.0
    ) -> None:
        self.streams = list(streams)
        self.combine = combine
        self.offset = offset
        self._counter = itertools.count()
        self._heap: list[tuple[float, int, tuple[int, ...]]] = []
        self._visited = set()
        n = len(self.streams)
        if n == 0:
            # Empty product: the single empty tuple with probability one.
            self._heap.append((-offset, next(self._counter), ()))
            self._visited.add(())
        else:
            heads = [s.get(0) for s in self.streams]
            if all(h is not None for h in heads):
                root = (0,) * n
                score = offset + sum(h[1] for h in heads)
                self._heap.append((-score, next(self._counter), root))
                self._visited.add(root)

    def __iter__(self) -> "ProductEnumerator":
        return self

    def _score(self, idx: tuple[int, ...]) -> float:
        return self.offset + sum(self.streams[k].get(i)[1] for k, i in enumerate(idx))

    def __next__(self) -> tuple[Any, float]:
        if not self._heap:
            raise StopIteration
        _, _, idx = heapq.heappop(self._heap)
        if len(idx) == 0:
            return (self.combine(()), self.offset)
        # Recompute the score from per-coordinate log probs to avoid float drift.
        score = self._score(idx)
        value = self.combine(tuple(self.streams[k].get(i)[0] for k, i in enumerate(idx)))
        for k in range(len(idx)):
            succ = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
            if succ not in self._visited and self.streams[k].get(idx[k] + 1) is not None:
                self._visited.add(succ)
                heapq.heappush(self._heap, (-self._score(succ), next(self._counter), succ))
        return (value, score)


class LengthFrontierMerge:
    """Merge per-length sorted streams, instantiating lengths lazily from a sorted length stream.

    len_stream yields (length, log_prob_of_length) in descending order. make_stream(length,
    lp_len) returns a sorted iterator of (value, log_prob) whose log probs already include
    lp_len and never exceed it (true whenever per-element contributions are log probs <= 0).
    The next un-instantiated length's lp_len is then a valid upper bound on anything its
    stream could produce, so lengths are instantiated only when they can beat the best
    instantiated head. Supports of distinct lengths must be disjoint (no de-duplication).
    """

    def __init__(
        self, len_stream: BufferedStream, make_stream: Callable[[int, float], Iterator[tuple[Any, float]]]
    ) -> None:
        self._len_stream = len_stream
        self._make_stream = make_stream
        self._next_len_rank = 0
        self._counter = itertools.count()
        self._heap: list[tuple[float, int, int]] = []  # (-head_lp, counter, stream id)
        self._heads = {}
        self._streams = {}

    def __iter__(self) -> "LengthFrontierMerge":
        return self

    def _pop(self) -> tuple[Any, float]:
        _, _, sid = heapq.heappop(self._heap)
        value, lp = self._heads.pop(sid)
        try:
            nxt = next(self._streams[sid])
            self._heads[sid] = nxt
            heapq.heappush(self._heap, (-nxt[1], next(self._counter), sid))
        except StopIteration:
            del self._streams[sid]
        return (value, lp)

    def __next__(self) -> tuple[Any, float]:
        while True:
            frontier = self._len_stream.get(self._next_len_rank)
            if frontier is None:
                if self._heap:
                    return self._pop()
                raise StopIteration
            if self._heap and -self._heap[0][0] >= frontier[1]:
                return self._pop()
            length, lp_len = frontier
            sid = self._next_len_rank
            self._next_len_rank += 1
            if not isinstance(length, (int, np.integer)) or length < 0:
                continue
            stream = self._make_stream(int(length), lp_len)
            try:
                head = next(stream)
            except StopIteration:
                continue
            self._streams[sid] = stream
            self._heads[sid] = head
            heapq.heappush(self._heap, (-head[1], next(self._counter), sid))


def _best_first_union(
    streams: Sequence[BufferedStream],
    log_offsets: Sequence[float],
    exact_log_density: Callable[[Any], float],
    bound_fn: Callable[[np.ndarray], float],
    tol: float,
) -> Iterator[tuple[Any, float]]:
    counter = itertools.count()
    # Per-stream head ranks; heads heap holds (-(offset + head_lp), counter, k, rank).
    heads: list[tuple[float, int, int, int]] = []
    live = {}
    for k, s in enumerate(streams):
        if log_offsets[k] == -np.inf:
            continue
        item = s.get(0)
        if item is not None:
            heapq.heappush(heads, (-(log_offsets[k] + item[1]), next(counter), k, 0))
            live[k] = 0
    seen = set()
    buffer: list[tuple[float, int, Any]] = []

    def compute_bound() -> float:
        if not live:
            return -np.inf
        return bound_fn(np.asarray([log_offsets[k] + streams[k].get(r)[1] for k, r in live.items()]))

    # The bound depends only on the live heads, which change only when a head is consumed;
    # cache it and recompute after each head pop instead of on every buffer-drain iteration.
    bound = compute_bound()
    while True:
        if buffer and -buffer[0][0] >= bound - tol:
            neg_lp, _, v = heapq.heappop(buffer)
            yield (v, -neg_lp)
            continue
        if not heads:
            if buffer:
                neg_lp, _, v = heapq.heappop(buffer)
                yield (v, -neg_lp)
                continue
            return
        _, _, k, rank = heapq.heappop(heads)
        v = streams[k].get(rank)[0]
        key = freeze(v)
        if key not in seen:
            seen.add(key)
            lp = exact_log_density(v)
            if lp > -np.inf:
                heapq.heappush(buffer, (-lp, next(counter), v))
        nxt = streams[k].get(rank + 1)
        if nxt is not None:
            live[k] = rank + 1
            heapq.heappush(heads, (-(log_offsets[k] + nxt[1]), next(counter), k, rank + 1))
        else:
            del live[k]
        bound = compute_bound()


def best_first_union(
    streams: Sequence[BufferedStream],
    log_offsets: Sequence[float],
    exact_log_density: Callable[[Any], float],
    tol: float = 1.0e-10,
) -> Iterator[tuple[Any, float]]:
    """Enumerate the union of sorted streams with overlapping supports.

    Candidate values are pulled from the streams (stream k shifted by log_offsets[k]),
    de-duplicated via freeze, re-scored exactly with exact_log_density, and buffered
    until their exact score is at least the upper bound on any not-yet-seen value:
    logsumexp_k(log_offsets[k] + head_lp_k). This is the mixture algorithm: any unseen
    x satisfies p_k(x) <= head_k for every k, hence sum_k w_k p_k(x) <= exp(bound).
    """
    return _best_first_union(streams, log_offsets, exact_log_density, log_sum, tol)


def _bounded_best_first_union_index_with_contributions(
    streams: Sequence[BufferedStream],
    log_offsets: Sequence[float],
    component_log_density: Callable[[int, Any], float],
    max_bits: float,
    bin_width_bits: float,
    tol: float,
) -> QuantizedEnumerationIndex:
    """Component-aware bounded union index with lazy mixture scoring."""
    log2 = math.log(2.0)
    threshold = -float(max_bits) * log2
    bit_limit = float(max_bits) + 1.0e-12
    n = len(streams)

    def within_bound(log_prob: float) -> bool:
        return max(0.0, -float(log_prob) / log2) <= bit_limit

    def log_sum_known(values: Sequence[float | None]) -> float:
        finite = [float(v) for v in values if v is not None and v > -np.inf]
        return log_sum(np.asarray(finite)) if finite else -np.inf

    counter = itertools.count()
    heads: list[tuple[float, int, int, int]] = []
    live = {}
    for k, s in enumerate(streams):
        if log_offsets[k] == -np.inf:
            continue
        item = s.get(0)
        if item is not None:
            heapq.heappush(heads, (-(log_offsets[k] + item[1]), next(counter), k, 0))
            live[k] = 0

    states: dict[Hashable, dict[str, Any]] = {}
    state_order: list[Hashable] = []

    def live_bound() -> float:
        if not live:
            return -np.inf
        return log_sum(np.asarray([log_offsets[k] + streams[k].get(r)[1] for k, r in live.items()]))

    def advance(k: int, rank: int) -> None:
        nxt = streams[k].get(rank + 1)
        if nxt is not None:
            live[k] = rank + 1
            heapq.heappush(heads, (-(log_offsets[k] + nxt[1]), next(counter), k, rank + 1))
        elif k in live:
            del live[k]

    def add_contribution(state: dict[str, Any], k: int, weighted_lp: float) -> None:
        contribs = state["contribs"]
        if contribs[k] is None or weighted_lp > contribs[k]:
            contribs[k] = float(weighted_lp)

    def state_for(value: Any) -> tuple[Hashable, dict[str, Any], bool]:
        key = freeze(value)
        state = states.get(key)
        if state is not None:
            return key, state, False
        state = {"value": value, "contribs": [None] * n}
        states[key] = state
        state_order.append(key)
        return key, state, True

    def upper_bound(state: dict[str, Any]) -> float:
        vals: list[float | None] = []
        contribs = state["contribs"]
        for k in range(n):
            if contribs[k] is not None:
                vals.append(contribs[k])
            elif k in live:
                item = streams[k].get(live[k])
                vals.append(log_offsets[k] + item[1] if item is not None else -np.inf)
            else:
                vals.append(-np.inf)
        return log_sum_known(vals)

    def exact_state_log_density(state: dict[str, Any]) -> float:
        value = state["value"]
        contribs = state["contribs"]
        for k in range(n):
            if contribs[k] is None:
                lp = float(component_log_density(k, value))
                contribs[k] = -np.inf if lp == -np.inf else float(log_offsets[k] + lp)
        return log_sum_known(contribs)

    def drain_seen_heads() -> bool:
        """Consume duplicate live heads; return True if a distinct tail exists."""
        while heads:
            _, _, k, rank = heapq.heappop(heads)
            item = streams[k].get(rank)
            if item is None:
                if k in live:
                    del live[k]
                continue
            value, lp = item
            key = freeze(value)
            state = states.get(key)
            if state is None:
                return True
            add_contribution(state, k, log_offsets[k] + lp)
            advance(k, rank)
        return False

    while True:
        if live_bound() < threshold - tol:
            truncated = drain_seen_heads()
            break
        if not heads:
            truncated = False
            break

        _, _, k, rank = heapq.heappop(heads)
        item = streams[k].get(rank)
        if item is None:
            if k in live:
                del live[k]
            continue
        value, lp = item
        _, state, _ = state_for(value)
        add_contribution(state, k, log_offsets[k] + lp)
        advance(k, rank)

    items: list[tuple[Any, float]] = []
    for key in state_order:
        state = states[key]
        if upper_bound(state) < threshold - tol:
            truncated = True
            continue
        lp = exact_state_log_density(state)
        if within_bound(lp):
            items.append((state["value"], lp))
        else:
            truncated = True

    return QuantizedEnumerationIndex.from_items(
        items, max_bits=max_bits, bin_width_bits=bin_width_bits, truncated=truncated
    )


def bounded_best_first_union_index(
    streams: Sequence[BufferedStream],
    log_offsets: Sequence[float],
    exact_log_density: Callable[[Any], float],
    max_bits: float,
    bin_width_bits: float = 1.0,
    tol: float = 1.0e-10,
    component_log_density: Callable[[int, Any], float] | None = None,
) -> QuantizedEnumerationIndex:
    """Build a quantized index from a globally bounded best-first union.

    This is the index-building counterpart of ``best_first_union``. It is useful
    for mixtures: if each stream head bounds one component contribution, then the
    log-sum of live heads bounds every unseen value. Once that global frontier is
    below the requested probability threshold, all remaining qualifying values must
    already be in the exact-score buffer, so the index can stop without enumerating
    a looser per-component candidate prefix.
    """
    if max_bits < 0:
        raise ValueError("max_bits must be non-negative.")
    if bin_width_bits <= 0:
        raise ValueError("bin_width_bits must be positive.")

    if component_log_density is not None:
        return _bounded_best_first_union_index_with_contributions(
            streams, log_offsets, component_log_density, max_bits=max_bits, bin_width_bits=bin_width_bits, tol=tol
        )

    log2 = math.log(2.0)
    threshold = -float(max_bits) * log2
    bit_limit = float(max_bits) + 1.0e-12

    def within_bound(log_prob: float) -> bool:
        return max(0.0, -float(log_prob) / log2) <= bit_limit

    counter = itertools.count()
    heads: list[tuple[float, int, int, int]] = []
    live = {}
    for k, s in enumerate(streams):
        if log_offsets[k] == -np.inf:
            continue
        item = s.get(0)
        if item is not None:
            heapq.heappush(heads, (-(log_offsets[k] + item[1]), next(counter), k, 0))
            live[k] = 0

    seen = set()
    buffer: list[tuple[float, int, Any]] = []
    items: list[tuple[Any, float]] = []
    truncated = False

    def live_bound() -> float:
        if not live:
            return -np.inf
        return log_sum(np.asarray([log_offsets[k] + streams[k].get(r)[1] for k, r in live.items()]))

    def drain_to_unseen_live_value() -> bool:
        """Skip already-seen live heads; return True if a distinct tail value exists."""
        while heads:
            _, _, k, rank = heapq.heappop(heads)
            item = streams[k].get(rank)
            if item is None:
                if k in live:
                    del live[k]
                continue
            if freeze(item[0]) not in seen:
                return True
            nxt = streams[k].get(rank + 1)
            if nxt is not None:
                live[k] = rank + 1
                heapq.heappush(heads, (-(log_offsets[k] + nxt[1]), next(counter), k, rank + 1))
            elif k in live:
                del live[k]
        return False

    # The live frontier bound changes only when a head is consumed; cache it and recompute
    # after each head pop rather than rebuilding the log-sum on every buffer-drain iteration.
    bound = live_bound()
    while True:
        if buffer and -buffer[0][0] >= bound - tol:
            neg_lp, _, value = heapq.heappop(buffer)
            lp = -neg_lp
            if within_bound(lp):
                items.append((value, lp))
                continue
            truncated = True
            break

        if bound < threshold - tol:
            while buffer:
                neg_lp, _, value = heapq.heappop(buffer)
                lp = -neg_lp
                if within_bound(lp):
                    items.append((value, lp))
                else:
                    truncated = True
                    break
            if drain_to_unseen_live_value():
                truncated = True
            break

        if not heads:
            while buffer:
                neg_lp, _, value = heapq.heappop(buffer)
                lp = -neg_lp
                if within_bound(lp):
                    items.append((value, lp))
                else:
                    truncated = True
                    break
            break

        _, _, k, rank = heapq.heappop(heads)
        value = streams[k].get(rank)[0]
        key = freeze(value)
        if key not in seen:
            seen.add(key)
            lp = exact_log_density(value)
            if lp > -np.inf:
                heapq.heappush(buffer, (-float(lp), next(counter), value))

        nxt = streams[k].get(rank + 1)
        if nxt is not None:
            live[k] = rank + 1
            heapq.heappush(heads, (-(log_offsets[k] + nxt[1]), next(counter), k, rank + 1))
        elif k in live:
            del live[k]
        bound = live_bound()

    return QuantizedEnumerationIndex.from_items(
        items, max_bits=max_bits, bin_width_bits=bin_width_bits, sorted_items=True, truncated=truncated
    )


def best_first_union_max(
    streams: Sequence[BufferedStream],
    log_offsets: Sequence[float],
    exact_log_density: Callable[[Any], float],
    tol: float = 1.0e-10,
) -> Iterator[tuple[Any, float]]:
    """Like best_first_union, but for a max-scored union (bound = max over heads).

    Used to enumerate a deduped symbol pool ordered by max-over-states emission
    probability for markov/HMM enumerators.
    """
    return _best_first_union(streams, log_offsets, exact_log_density, np.max, tol)


def sound_top_k(
    dist: Any,
    k: int,
    start: int = 0,
    budget_bits: float = 40.0,
    oversample: int = 8,
    bin_width_bits: float = 1.0,
    max_budget_bits: float = 512.0,
    total_mass: float = 1.0,
    tol: float = 1.0e-12,
) -> list[tuple[Any, float]]:
    """Exact true-descending observations ranked ``[start, start+k)`` for ANY normalized model.

    Mass-threshold certificate, correct regardless of the seek stream's ordering (so it holds for
    deep/nested mixtures and HMMs where the tropical order is badly displaced): pull distinct
    ``(value, log_prob)`` from the count-budget index, accumulate exact probability mass, and keep
    the best ``start+k`` by probability. Since every unpulled item's probability is at most the
    remaining mass ``total_mass - accumulated``, once ``remaining`` drops below the (start+k)-th best
    probability the heap is exactly the true top ``start+k``; return ``sorted_desc[start:start+k]``.
    If the in-budget stream is exhausted first, the budget is doubled (up to ``max_budget_bits``).
    ``total_mass`` (default 1.0) may be a known upper bound for sub-normalized models.

    Cost is ``O(number pulled)`` -- small for peaked distributions, large for flat ones, so for
    top-k from rank 0 ``dist.enumerator()`` is usually leaner; this adds the soundness certificate
    and the arbitrary start offset the sequential enumerator lacks.
    """
    from pysp.utils.quantization.core import count_budget_index

    if k < 1:
        raise ValueError("k must be a positive integer.")
    if start < 0:
        raise ValueError("start must be non-negative.")
    need = start + k
    budget = float(budget_bits)
    while True:
        # Consume the RAW (over-counted) seek index and de-duplicate here. The canonical-dedup
        # distinct stream is lossy -- it can drop a valid value whose minimal tropical fine bucket
        # is computed inconsistently with the convolution binning -- which would make the mass
        # certificate unreachable. The raw index emits every (value, path/component) copy, so the
        # self-dedup below recovers the complete distinct set (verified: accumulated mass -> 1).
        index = count_budget_index(dist, budget_bits=budget, oversample=oversample, bin_width_bits=bin_width_bits)
        heap: list[tuple[float, int, Any]] = []  # min-heap by log_prob, the best `need` so far
        counter = itertools.count()
        seen: set[Hashable] = set()
        accumulated = 0.0
        certified = False
        for i in range(index.total_count):
            value, lp = index.get(i)
            key = freeze(value)
            if key in seen:
                continue
            seen.add(key)
            accumulated += math.exp(lp)
            if len(heap) < need:
                heapq.heappush(heap, (lp, next(counter), value))
            elif lp > heap[0][0]:
                heapq.heapreplace(heap, (lp, next(counter), value))
            if len(heap) >= need:
                remaining = max(0.0, total_mass - accumulated)
                if remaining < math.exp(heap[0][0]) - tol:
                    certified = True
                    break
        ordered = sorted(heap, key=lambda t: -t[0])
        # Certified sound; or the whole in-budget index was consumed and it was NOT truncated, so
        # the full support was seen and the heap is exact; or fewer than `need` distinct values exist.
        if certified or not index.truncated or len(seen) < need:
            return [(v, lp) for lp, _, v in ordered][start:need]
        if budget >= max_budget_bits:
            return [(v, lp) for lp, _, v in ordered][start:need]  # best effort at the budget cap
        budget = min(budget * 2.0, max_budget_bits)
