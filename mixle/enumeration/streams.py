"""Generic stream primitives for smart enumeration.

The building blocks shared by the best-first algorithms and the combinator enumerators:
``freeze`` (canonical hashable keys for de-duplicating support values), ``BufferedStream`` (random
access by rank into a lazy descending-probability stream), ``merge_enumerators`` (k-way merge of
sorted streams with disjoint supports), and ``supports_enumeration`` (the capability probe). See
:class:`mixle.stats.compute.pdist.DistributionEnumerator` for the enumeration contract.
"""

import heapq
import itertools
import math
from collections.abc import Hashable, Iterator, Sequence
from typing import Any

import numpy as np

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
    from mixle.stats.compute.pdist import EnumerationError

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
        """Return buffered item ``i``, pulling from the stream if needed."""
        buf = self._buf
        # Fast path: already buffered (the common case -- coordinates are re-read every pop).
        if i < len(buf):
            return buf[i]
        while not self._done and len(buf) <= i:
            try:
                buf.append(next(self._it))
            except StopIteration:
                self._done = True
        return buf[i] if i < len(buf) else None


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
