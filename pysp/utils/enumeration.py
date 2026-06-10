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

See pysp.stats.pdist.DistributionEnumerator for the enumeration contract.
"""
import heapq
import itertools
import math
import numpy as np
from typing import Any, Callable, Hashable, Iterator, List, Optional, Sequence, Tuple

from pysp.utils.vector import log_sum

__all__ = ['BufferedStream', 'freeze', 'merge_enumerators', 'ProductEnumerator',
           'LengthFrontierMerge', 'best_first_union', 'best_first_union_max',
           'supports_enumeration']


_NAN_SENTINEL = ('__pysp_nan__',)


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
        raise TypeError('Cannot compute an enumeration dedup key for value of type %s' % type(x).__name__)
    return x


def supports_enumeration(dist) -> bool:
    """Return True if dist.enumerator() can be constructed."""
    from pysp.stats.pdist import EnumerationError
    try:
        dist.enumerator()
        return True
    except EnumerationError:
        return False


class BufferedStream(object):
    """Random access by rank into a lazy stream of (value, log_prob) pairs.

    get(i) extends an internal buffer as needed and returns the i-th item, or None
    if the stream has fewer than i+1 items. The underlying stream is consumed at
    most once regardless of how many consumers share this object.
    """

    def __init__(self, it: Iterator[Tuple[Any, float]]) -> None:
        self._it = iter(it)
        self._buf: List[Tuple[Any, float]] = []
        self._done = False

    def get(self, i: int) -> Optional[Tuple[Any, float]]:
        while not self._done and len(self._buf) <= i:
            try:
                self._buf.append(next(self._it))
            except StopIteration:
                self._done = True
        return self._buf[i] if i < len(self._buf) else None


def merge_enumerators(streams: Sequence[Iterator[Tuple[Any, float]]],
                      offsets: Sequence[float]) -> Iterator[Tuple[Any, float]]:
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


class ProductEnumerator(object):
    """Best-first enumeration of the Cartesian product of sorted child streams.

    Yields (combine(values), log_prob) with log_prob = offset + sum of child log
    probs, in non-increasing order. Standard k-best lattice search: a max-heap over
    index tuples, with successors advancing one coordinate; correctness follows from
    each child stream being sorted (coordinate-wise monotonicity).
    """

    def __init__(self, streams: Sequence[BufferedStream],
                 combine: Callable[[Tuple[Any, ...]], Any] = tuple,
                 offset: float = 0.0) -> None:
        self.streams = list(streams)
        self.combine = combine
        self.offset = offset
        self._counter = itertools.count()
        self._heap: List[Tuple[float, int, Tuple[int, ...]]] = []
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

    def __iter__(self) -> 'ProductEnumerator':
        return self

    def _score(self, idx: Tuple[int, ...]) -> float:
        return self.offset + sum(self.streams[k].get(i)[1] for k, i in enumerate(idx))

    def __next__(self) -> Tuple[Any, float]:
        if not self._heap:
            raise StopIteration
        _, _, idx = heapq.heappop(self._heap)
        if len(idx) == 0:
            return (self.combine(()), self.offset)
        # Recompute the score from per-coordinate log probs to avoid float drift.
        score = self._score(idx)
        value = self.combine(tuple(self.streams[k].get(i)[0] for k, i in enumerate(idx)))
        for k in range(len(idx)):
            succ = idx[:k] + (idx[k] + 1,) + idx[k + 1:]
            if succ not in self._visited and self.streams[k].get(idx[k] + 1) is not None:
                self._visited.add(succ)
                heapq.heappush(self._heap, (-self._score(succ), next(self._counter), succ))
        return (value, score)


class LengthFrontierMerge(object):
    """Merge per-length sorted streams, instantiating lengths lazily from a sorted length stream.

    len_stream yields (length, log_prob_of_length) in descending order. make_stream(length,
    lp_len) returns a sorted iterator of (value, log_prob) whose log probs already include
    lp_len and never exceed it (true whenever per-element contributions are log probs <= 0).
    The next un-instantiated length's lp_len is then a valid upper bound on anything its
    stream could produce, so lengths are instantiated only when they can beat the best
    instantiated head. Supports of distinct lengths must be disjoint (no de-duplication).
    """

    def __init__(self, len_stream: BufferedStream,
                 make_stream: Callable[[int, float], Iterator[Tuple[Any, float]]]) -> None:
        self._len_stream = len_stream
        self._make_stream = make_stream
        self._next_len_rank = 0
        self._counter = itertools.count()
        self._heap: List[Tuple[float, int, int]] = []  # (-head_lp, counter, stream id)
        self._heads = {}
        self._streams = {}

    def __iter__(self) -> 'LengthFrontierMerge':
        return self

    def _pop(self) -> Tuple[Any, float]:
        _, _, sid = heapq.heappop(self._heap)
        value, lp = self._heads.pop(sid)
        try:
            nxt = next(self._streams[sid])
            self._heads[sid] = nxt
            heapq.heappush(self._heap, (-nxt[1], next(self._counter), sid))
        except StopIteration:
            del self._streams[sid]
        return (value, lp)

    def __next__(self) -> Tuple[Any, float]:
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


def _best_first_union(streams: Sequence[BufferedStream],
                      log_offsets: Sequence[float],
                      exact_log_density: Callable[[Any], float],
                      bound_fn: Callable[[np.ndarray], float],
                      tol: float) -> Iterator[Tuple[Any, float]]:
    counter = itertools.count()
    # Per-stream head ranks; heads heap holds (-(offset + head_lp), counter, k, rank).
    heads: List[Tuple[float, int, int, int]] = []
    live = {}
    for k, s in enumerate(streams):
        if log_offsets[k] == -np.inf:
            continue
        item = s.get(0)
        if item is not None:
            heapq.heappush(heads, (-(log_offsets[k] + item[1]), next(counter), k, 0))
            live[k] = 0
    seen = set()
    buffer: List[Tuple[float, int, Any]] = []

    while True:
        if live:
            bound = bound_fn(np.asarray([log_offsets[k] + streams[k].get(r)[1] for k, r in live.items()]))
        else:
            bound = -np.inf
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


def best_first_union(streams: Sequence[BufferedStream],
                     log_offsets: Sequence[float],
                     exact_log_density: Callable[[Any], float],
                     tol: float = 1.0e-10) -> Iterator[Tuple[Any, float]]:
    """Enumerate the union of sorted streams with overlapping supports.

    Candidate values are pulled from the streams (stream k shifted by log_offsets[k]),
    de-duplicated via freeze, re-scored exactly with exact_log_density, and buffered
    until their exact score is at least the upper bound on any not-yet-seen value:
    logsumexp_k(log_offsets[k] + head_lp_k). This is the mixture algorithm: any unseen
    x satisfies p_k(x) <= head_k for every k, hence sum_k w_k p_k(x) <= exp(bound).
    """
    return _best_first_union(streams, log_offsets, exact_log_density, log_sum, tol)


def best_first_union_max(streams: Sequence[BufferedStream],
                         log_offsets: Sequence[float],
                         exact_log_density: Callable[[Any], float],
                         tol: float = 1.0e-10) -> Iterator[Tuple[Any, float]]:
    """Like best_first_union, but for a max-scored union (bound = max over heads).

    Used to enumerate a deduped symbol pool ordered by max-over-states emission
    probability for markov/HMM enumerators.
    """
    return _best_first_union(streams, log_offsets, exact_log_density, np.max, tol)
