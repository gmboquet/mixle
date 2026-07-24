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
    pairs, sets to frozensets, numpy arrays to (shape, dtype string, bytes), numpy
    scalars to their python equivalents, and NaN to a shared sentinel (so nan == nan
    for dedup purposes). Raises TypeError for values that cannot be canonicalized.

    Cross-type equality policy (deliberate, not incidental): type is part of a value's
    identity for dedup purposes, so this never falls back to Python's native cross-type
    equality/hashing.

      - ndarrays are keyed by (shape, dtype.str, bytes). Bytes alone are ambiguous --
        e.g. an int8 array holding -56 and a uint8 array holding 200 have identical
        ``tobytes()`` output but represent different numbers -- so omitting dtype would
        silently merge genuinely distinct support values.
      - Every other value is keyed by (type(x), x) rather than bare ``x``, after numpy
        scalars are unwrapped to their python equivalent via ``.item()``. Plain ``x``
        would let Python's own cross-type equality collide unrelated values that
        happen to compare equal -- ``True == 1``, ``False == 0``, ``1 == 1.0`` -- even
        though a typed distribution's support can legitimately contain both a bool and
        an int (or an int and a float) as distinct outcomes. Note ``.item()`` collapses
        numpy integer width (int8/uint8/int64 all become python ``int``), which is safe
        because ``.item()`` -- unlike an array's raw ``.tobytes()`` -- already applies
        each dtype's own signed/unsigned interpretation, so equal post-``.item()``
        values really are the same number.
      - The one exception: every NaN float collapses to a single shared sentinel
        regardless of which NaN produced it, since IEEE 754 ``nan != nan`` would
        otherwise make each NaN a unique, never-deduplicated key.
    """
    if isinstance(x, (list, tuple)):
        return tuple(freeze(u) for u in x)
    if isinstance(x, dict):
        return frozenset((freeze(k), freeze(v)) for k, v in x.items())
    if isinstance(x, (set, frozenset)):
        return frozenset(freeze(u) for u in x)
    if isinstance(x, np.ndarray):
        return (x.shape, x.dtype.str, x.tobytes())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return _NAN_SENTINEL
    try:
        hash(x)
    except TypeError:
        raise TypeError("Cannot compute an enumeration dedup key for value of type %s" % type(x).__name__)
    return (type(x), x)


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
        """Return buffered item ``i``, pulling from the stream if needed.

        ``i`` is an absolute, 0-based rank -- not a Python sequence index -- so it must
        be an exact, non-negative integer. This is checked before the buffer is touched
        at all, regardless of how much has been buffered so far: without this check,
        ``get(-1)`` would return the last *currently buffered* item via ordinary Python
        negative indexing once anything has been pulled, but raise an incidental
        ``IndexError`` from indexing an empty list otherwise -- making a negative rank's
        observable behavior depend on prior access history instead of always meaning
        "invalid." ``bool`` is rejected too despite being an ``int`` subclass, and a
        whole-valued ``float`` such as ``2.0`` is rejected as well: a rank is always
        exactly integral, never merely numerically equal to one.
        """
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise TypeError("rank must be an int, got %s: %r" % (type(i).__name__, i))
        if i < 0:
            raise ValueError("rank must be non-negative, got %r" % (i,))
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

    Validated eagerly at call time, before any stream is touched: ``offsets`` must have
    exactly one entry per stream, and every offset must be finite or exactly ``-inf`` --
    the sentinel for "this stream contributes nothing" (e.g. a zero-weight mixture
    component), which excludes that stream entirely without ever opening its iterator.
    NaN and ``+inf`` offsets are rejected outright, since neither has a sensible reading
    as a log-probability shift. This function is a thin eager-validation wrapper; the
    actual merge runs in the generator it returns, so a malformed call raises immediately
    instead of on first iteration.

    Each stream's own items are validated lazily -- one pull at a time, as they are
    actually consumed, so this is safe for infinite streams and never reads ahead: a
    non-finite log_prob (including ``-inf`` -- a properly-formed descending stream should
    already exclude impossible/zero-probability items, matching the convention used
    elsewhere in this package, e.g. best_first.py's ``if lp > -np.inf`` guards) raises
    immediately, and so does a log_prob that exceeds the previous log_prob pulled from
    that SAME stream -- i.e. the stream failing the k-way merge's fundamental
    precondition of actually being sorted descending.
    """
    n = len(streams)
    if len(offsets) != n:
        raise ValueError(
            "offsets must have exactly one entry per stream (got %d offsets for %d streams)" % (len(offsets), n)
        )
    checked_offsets: list[float] = []
    for k, o in enumerate(offsets):
        try:
            of = float(o)
        except (TypeError, ValueError):
            raise ValueError("offsets[%d] must be a real number, got %r" % (k, o))
        if math.isnan(of) or of == math.inf:
            raise ValueError("offsets[%d] must be finite or -inf (the exclude-this-stream sentinel), got %r" % (k, o))
        checked_offsets.append(of)
    return _merge_enumerators_body(streams, checked_offsets)


def _merge_enumerators_body(
    streams: Sequence[Iterator[tuple[Any, float]]], offsets: Sequence[float]
) -> Iterator[tuple[Any, float]]:
    """Generator body for :func:`merge_enumerators`.

    Trusts that its only caller already validated arity (``len(offsets) == len(streams)``)
    and that every offset is finite or ``-inf``. Still validates every per-item score as
    it is pulled -- lazily, one item at a time -- since sortedness and score finiteness can
    only be checked against the stream's actual contents, not the call-time arguments.
    """
    counter = itertools.count()
    heap = []
    its = [iter(s) for s in streams]
    last_lp: dict[int, float] = {}

    def _next_checked(k: int) -> tuple[Any, float] | None:
        for v, lp in its[k]:
            lpf = float(lp)
            if not math.isfinite(lpf):
                raise ValueError("stream %d yielded a non-finite log_prob (%r); scores must be finite" % (k, lp))
            prev = last_lp.get(k)
            if prev is not None and lpf > prev:
                raise ValueError("stream %d is not sorted descending: log_prob %r followed %r" % (k, lp, prev))
            last_lp[k] = lpf
            return v, lpf
        return None

    for k in range(len(its)):
        if offsets[k] == -np.inf:
            continue
        item = _next_checked(k)
        if item is not None:
            v, lp = item
            heapq.heappush(heap, (-(lp + offsets[k]), next(counter), v, k))
    while heap:
        neg_lp, _, v, k = heapq.heappop(heap)
        yield (v, -neg_lp)
        item = _next_checked(k)
        if item is not None:
            v2, lp2 = item
            heapq.heappush(heap, (-(lp2 + offsets[k]), next(counter), v2, k))
