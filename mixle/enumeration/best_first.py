"""Best-first / A*-style search over sorted enumeration streams.

The k-best engines used by the combinator distributions: ``ProductEnumerator`` (best-first over a
Cartesian product of child streams), ``LengthFrontierMerge`` / ``frontier_merge`` (length-indexed
frontier merge), and the ``best_first_union`` family (union of overlapping-support streams, re-scored
exactly and emitted in provably correct descending-probability order). See
:class:`mixle.stats.compute.pdist.DistributionEnumerator` for the enumeration contract.
"""

import heapq
import itertools
import math
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any

import numpy as np

from mixle.enumeration.quantization.seek import QuantizedEnumerationIndex
from mixle.enumeration.streams import BufferedStream, freeze
from mixle.utils.vector import log_sum


class ProductEnumerator:
    """Best-first enumeration of the Cartesian product of sorted child streams.

    Yields (combine(values), log_prob) with log_prob = offset + sum of child log
    probs, in non-increasing order. Standard k-best lattice search: a max-heap over
    index tuples, with successors advancing one coordinate; correctness follows from
    each child stream being sorted (coordinate-wise monotonicity).

    Duplicate-free successor rule: each heap entry carries the lowest coordinate it is
    allowed to advance, and a node only spawns successors for coordinates at or above
    that index. Every tuple is then reached by exactly one path -- the one that
    increments coordinates in non-decreasing index order -- so no ``visited`` set (and
    no O(n) per-successor tuple hashing) is needed. Best-first order is preserved
    because each successor's score is <= its generating node's (a sorted coordinate
    only decreases when advanced), so canonical edges are score-monotone.
    """

    def __init__(
        self, streams: Sequence[BufferedStream], combine: Callable[[tuple[Any, ...]], Any] = tuple, offset: float = 0.0
    ) -> None:
        self.streams = list(streams)
        self.combine = combine
        self.offset = offset
        self._counter = itertools.count()
        # heap entries: (-score, tie_counter, index_tuple, min_coord_allowed_to_advance)
        self._heap: list[tuple[float, int, tuple[int, ...], int]] = []
        self._n = n = len(self.streams)
        if n == 0:
            # Empty product: the single empty tuple with probability one.
            self._heap.append((-offset, next(self._counter), (), 0))
        else:
            heads = [s.get(0) for s in self.streams]
            if all(h is not None for h in heads):
                root = (0,) * n
                score = offset + sum(h[1] for h in heads)
                self._heap.append((-score, next(self._counter), root, 0))

    def __iter__(self) -> "ProductEnumerator":
        return self

    def __next__(self) -> tuple[Any, float]:
        if not self._heap:
            raise StopIteration
        _, _, idx, min_coord = heapq.heappop(self._heap)
        n = self._n
        if n == 0:
            return (self.combine(()), self.offset)
        # Fetch each coordinate's (value, log_prob) once and reuse it for the combined value, the
        # exact reported score, and the successor keys -- the previous O(n) re-sum per successor
        # (and a second O(n) fetch for the value) made __next__ quadratic in the number of fields.
        streams = self.streams
        items = [streams[k].get(idx[k]) for k in range(n)]
        value = self.combine(tuple(it[0] for it in items))
        score = self.offset + sum(it[1] for it in items)
        # Only advance coordinates at or above ``min_coord`` (canonical, duplicate-free generation).
        for k in range(min_coord, n):
            nxt = streams[k].get(idx[k] + 1)
            if nxt is None:
                continue
            succ = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
            # Advancing one coordinate only changes that coordinate's term, so the successor score
            # is the (exact, freshly re-summed) parent score plus that single delta. Re-basing on the
            # exact ``score`` every pop keeps each heap key within one ULP of exact -- no accumulating
            # drift -- so the pop order matches exact descending order except among true near-ties.
            succ_key = score + (nxt[1] - items[k][1])
            heapq.heappush(self._heap, (-succ_key, next(self._counter), succ, k))
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


def frontier_merge(
    outer_stream: BufferedStream, make_stream: Callable[[Any, float], Iterator[tuple[Any, float]]]
) -> Iterator[tuple[Any, float]]:
    """Merge per-outer-key inner streams, instantiating outer keys lazily from a sorted outer stream.

    The general form of :class:`LengthFrontierMerge` for arbitrary (not necessarily integer) outer
    keys: ``outer_stream`` yields ``(key, lp_key)`` in descending ``lp_key`` order, and
    ``make_stream(key, lp_key)`` returns a sorted ``(value, log_prob)`` iterator whose log-probs are
    ``<= lp_key`` (true whenever the inner contribution is ``<= 0`` and ``lp_key`` is folded in as an
    offset). The next un-instantiated key's ``lp_key`` is then an upper bound on anything its stream
    could produce, so keys are instantiated only when they can beat the best instantiated head.
    Supports of distinct keys must be disjoint (no de-duplication). Used for conditional products where
    the inner distribution depends on the outer value (e.g. hidden-association S2 given S1).
    """
    counter = itertools.count()
    heap: list[tuple[float, int, int]] = []
    heads: dict[int, tuple[Any, float]] = {}
    streams: dict[int, Iterator[tuple[Any, float]]] = {}
    next_rank = 0

    def pop() -> tuple[Any, float]:
        _, _, sid = heapq.heappop(heap)
        value, lp = heads.pop(sid)
        try:
            nxt = next(streams[sid])
            heads[sid] = nxt
            heapq.heappush(heap, (-nxt[1], next(counter), sid))
        except StopIteration:
            del streams[sid]
        return (value, lp)

    while True:
        frontier = outer_stream.get(next_rank)
        if frontier is None:
            if heap:
                yield pop()
                continue
            return
        if heap and -heap[0][0] >= frontier[1]:
            yield pop()
            continue
        key, lp_key = frontier
        sid = next_rank
        next_rank += 1
        if lp_key == -np.inf:
            continue
        stream = iter(make_stream(key, lp_key))
        try:
            head = next(stream)
        except StopIteration:
            continue
        streams[sid] = stream
        heads[sid] = head
        heapq.heappush(heap, (-head[1], next(counter), sid))


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
    # Live head scores keyed by stream index, maintained incrementally (only the consumed head
    # changes per pop). The bound reads these directly instead of re-fetching each head from its
    # stream, so a recompute is a single bound_fn over the cached scores -- no per-head get() calls.
    head_scores: dict[int, float] = {}
    for k, s in enumerate(streams):
        if log_offsets[k] == -np.inf:
            continue
        item = s.get(0)
        if item is not None:
            score = log_offsets[k] + item[1]
            heapq.heappush(heads, (-score, next(counter), k, 0))
            head_scores[k] = score
    seen = set()
    buffer: list[tuple[float, int, Any]] = []

    def exact_bound() -> float:
        return bound_fn(np.fromiter(head_scores.values(), dtype=float, count=len(head_scores)))

    # A buffered item is releasable once its probability is >= the bound on every unreleased item,
    # ``bound_fn`` over the live head scores. Both callers' bounds satisfy
    # ``max(scores) <= bound_fn(scores) <= max(scores) + log(#scores)`` (logsumexp and max), so we
    # avoid the O(#components) exact ``bound_fn`` on most pops: the heap top is ``max(scores)``, so
    # ``btop >= max + logK`` certifies release and ``btop < max`` certifies non-release outright --
    # the exact bound is only needed inside the ``logK``-wide uncertain band.
    while True:
        if buffer:
            btop = -buffer[0][0]
            if not heads:
                release = True
            else:
                mh = -heads[0][0]
                if btop >= mh + math.log(len(head_scores)) - tol:
                    release = True  # low-overhead upper bound on the frontier certifies release
                elif btop < mh - tol:
                    release = False  # frontier (>= mh) strictly exceeds btop: cannot release yet
                else:
                    release = btop >= exact_bound() - tol  # uncertain band: exact frontier needed
            if release:
                neg_lp, _, v = heapq.heappop(buffer)
                yield (v, -neg_lp)
                continue
        if not heads:
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
            score = log_offsets[k] + nxt[1]
            head_scores[k] = score
            heapq.heappush(heads, (-score, next(counter), k, rank + 1))
        else:
            del head_scores[k]


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
    from mixle.enumeration.quantization.core import count_budget_index

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
