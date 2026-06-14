"""The two enumeration primitives and the bridges between them (step-1 contract).

The enumeration system has two orthogonal computational modes:

  Axis A -- aggregate computation: fold a distribution's complete-probability *reduction* into a
    summary. Density, structural counts, and bounds all live here. This is captured by
    :class:`DecomposableSemiring`: a family expresses its reduction with ``leaf``/``plus``/``times``
    and the carrier decides what is computed. The carrier must be *witness-retaining* -- its
    elements keep enough structure to invert a rank back to a value (unranking) -- which a plain
    scalar semiring cannot do. :class:`CountSemiring` is the witness-retaining carrier over
    :class:`pysp.utils.quantization.CountIndex` that yields the count-budget seek index.

  Axis B -- ordered access: produce values in strict descending probability order, lazily. This is
    a best-first *search*, not a fold, and a semiring cannot generate it; it is the existing
    ``enumerator()`` machinery (:class:`pysp.stats.pdist.DistributionEnumerator`), aliased here as
    :class:`OrderedStream` for symmetry.

The two axes trade in one currency -- ``(value, log_prob)`` pairs under a shared
:class:`pysp.utils.quantization.Quantizer` -- connected by two coercion bridges:

  - :func:`enumerate_and_bin`  (Axis B -> Axis A): the universal fallback. Any ``OrderedStream``
    can be tabulated into a bounded count index; O(count), so only viable for small budgets.
  - :func:`ordered_stream_from_count_index`  (Axis A -> Axis B): unrank in bin order to get an
    approximately-ordered stream when exact best-first is too expensive.

So a family implements whichever axis is natural and the bridges synthesize the other (lossily).
This module is the contract; the witness-retaining semiring is the engine for everything
count-shaped, and a future tropical carrier will add Viterbi bounds (BoundedCount) for the
non-decomposable families (HMM/Mixture) by swapping only the carrier.
"""

import abc
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from pysp.utils.quantization import CountHistogram, CountIndex, Quantizer, convolve_indices, leaf_count_index

E = TypeVar("E")


class DecomposableSemiring(abc.ABC):
    """A witness-retaining semiring over which a likelihood reduction is evaluated.

    ``zero``/``one`` are the additive/multiplicative identities; ``plus`` pools mutually exclusive
    alternatives (e.g. different sequence lengths or Markov end-states); ``times`` composes
    independent factors (whose log-probabilities add); ``product`` is n-ary ``times``. ``leaf``
    lifts a single scored atom into the carrier. Implementations choose the carrier ``E``; for
    counting it is :class:`CountIndex`, whose per-bucket structure is the witness needed to unrank.
    """

    @abc.abstractmethod
    def zero(self) -> E: ...

    @abc.abstractmethod
    def one(self) -> E: ...

    @abc.abstractmethod
    def leaf(self, value: Any, log_prob: float, quantizer: Quantizer) -> E: ...

    @abc.abstractmethod
    def plus(self, a: E, b: E) -> E: ...

    @abc.abstractmethod
    def times(self, a: E, b: E, quantizer: Quantizer, max_fine_bucket: int) -> E: ...

    def product(self, elements: Sequence[E], quantizer: Quantizer, max_fine_bucket: int) -> E:
        """n-ary ``times``. Default folds ``times``; carriers may override for efficiency/order."""
        if not elements:
            return self.one()
        acc = elements[0]
        for nxt in elements[1:]:
            acc = self.times(acc, nxt, quantizer, max_fine_bucket)
        return acc


class _CNode:
    """A reified carrier element: a node in the composition tree with an eager count histogram.

    Reifying ``plus``/``scale``/``map_values``/``leaf`` (instead of nesting Python closures) lets a
    single *iterative* interpreter (:func:`_unrank`) descend a chain of these by looping over
    heap-allocated nodes, so a length-L trellis or length-fold unranks with O(1) call-stack depth
    instead of O(L). ``times``/``product`` are kept as the flat :class:`CountIndex` from
    ``convolve_indices`` (their unranker is already a flat loop over operands); a node references
    such a child opaquely through its ``get_in_bucket``.
    """

    __slots__ = ("kind", "hist", "a", "b", "child", "lp", "shift", "fn", "value")

    def __init__(self, kind, hist, a=None, b=None, child=None, lp=0.0, shift=0, fn=None, value=None):
        self.kind = kind
        self.hist = hist
        self.a = a
        self.b = b
        self.child = child
        self.lp = lp
        self.shift = shift
        self.fn = fn
        self.value = value

    def total(self) -> int:
        return self.hist.total()

    def get_in_bucket(self, fine_bucket: int, offset: int) -> tuple[Any, float]:
        return _unrank(self, fine_bucket, offset)


def _unrank(node, fb: int, off: int) -> tuple[Any, float]:
    """Iterative interpreter for a carrier node chain (no Python recursion over the chain depth).

    Descends linear ``scale``/``map_values``/``plus`` nodes in a loop, accumulating the log-prob
    shift and the value transforms, and bottoms out at a ``leaf`` or an opaque child (a flat
    ``CountIndex`` from ``times``/``product``), then applies the transforms outermost-last.
    """
    transforms: list = []
    lp_add = 0.0
    cur = node
    while True:
        if not isinstance(cur, _CNode):
            value, lp = cur.get_in_bucket(fb, off)
            break
        k = cur.kind
        if k == "leaf":
            value, lp = cur.value, cur.lp
            break
        if k == "scale":
            fb -= cur.shift
            lp_add += cur.lp
            cur = cur.child
        elif k == "mapv":
            transforms.append(cur.fn)
            cur = cur.child
        elif k == "plus":
            na = cur.a.hist.count_at(fb)
            if off < na:
                cur = cur.a
            else:
                off -= na
                cur = cur.b
        else:
            raise IndexError("offset outside carrier node")
    for fn in reversed(transforms):
        value = fn(value)
    return value, lp + lp_add


class CountSemiring(DecomposableSemiring):
    """Witness-retaining carrier over reified nodes / CountIndex: structural counting + unranking.

    ``leaf``/``plus``/``scale``/``map_values`` build :class:`_CNode` trees unranked by the iterative
    :func:`_unrank` (bounded call-stack regardless of chain depth); ``times``/``product`` convolve
    via :func:`pysp.utils.quantization.convolve_indices` (flat unranker -- identical bin counts and
    within-bucket order to the previous path). Both element kinds expose ``.hist`` and
    ``.get_in_bucket``, so they interoperate freely.
    """

    def zero(self) -> _CNode:
        return _CNode("empty", CountHistogram.empty())

    def one(self) -> _CNode:
        return _CNode("leaf", CountHistogram.delta(0, 1), value=(), lp=0.0)

    def leaf(self, value: Any, log_prob: float, quantizer: Quantizer) -> _CNode:
        fb = quantizer.fine_bucket(log_prob)
        return _CNode("leaf", CountHistogram.delta(fb, 1), value=value, lp=float(log_prob))

    def from_enumerator(
        self, enum: Iterator[tuple[Any, float]], quantizer: Quantizer, max_fine_bucket: int
    ) -> tuple[CountIndex, bool]:
        """Lift an atomic distribution's exact enumerator into a carrier element (depth-bounded)."""
        return leaf_count_index(enum, quantizer, max_fine_bucket)

    def plus(self, a, b) -> _CNode:
        return _CNode("plus", a.hist.add(b.hist), a=a, b=b)

    def times(self, a, b, quantizer: Quantizer, max_fine_bucket: int) -> CountIndex:
        return convolve_indices([a, b], quantizer, max_fine_bucket)

    def product(self, elements: Sequence[Any], quantizer: Quantizer, max_fine_bucket: int) -> CountIndex:
        # Flat n-ary convolution (suffix-histogram unranker) -- identical bin counts and within-bucket
        # order to the previous hand-written composite path.
        return convolve_indices(list(elements), quantizer, max_fine_bucket)

    def scale(self, a, log_prob: float, quantizer: Quantizer, max_fine_bucket: int | None = None) -> _CNode:
        """Multiply by a constant probability factor: shift buckets by the factor, value unchanged.

        This is the action of the base log-prob monoid on the carrier (e.g. a sequence length term,
        a Markov transition/initial term). Optionally truncate the shifted histogram to a depth bound.
        """
        shift = quantizer.fine_bucket(log_prob)
        hist = a.hist.shift(shift)
        if max_fine_bucket is not None:
            hist = hist.truncate(max_fine_bucket)
        return _CNode("scale", hist, child=a, lp=float(log_prob), shift=shift)

    def map_values(self, a, fn: Callable[[Any], Any]) -> _CNode:
        """Relabel values (pushforward) without touching counts or buckets -- e.g. tuple -> list."""
        return _CNode("mapv", a.hist, child=a, fn=fn)

    def power_prefix(
        self, a: CountIndex, max_k: int, quantizer: Quantizer, max_fine_bucket: int
    ) -> Sequence[CountIndex]:
        """Return [a^(times 0), ..., a^(times K)] (K <= max_k), the k-fold self-products.

        The histograms are built incrementally (O(max_k) convolutions, shared), so this is the
        count side of an iid sequence. Each element's unranker is the *flat* product of k copies,
        built lazily and cached -- identical bin counts and within-bucket order to calling
        ``product([a]*k)`` directly, but without the O(k^2) eager cost. Stops early if the element
        mass is exhausted within the depth bound.
        """
        prefix: list = [self.one()]
        hist_k = CountHistogram.delta(0, 1)
        cache: dict = {}
        for k in range(1, int(max_k) + 1):
            hist_k = quantizer.convolve(hist_k, a.hist, max_fine_bucket=max_fine_bucket)
            if hist_k.is_empty():
                break

            def make_getter(kk: int):
                def getter(fb: int, off: int) -> tuple[Any, float]:
                    ci = cache.get(kk)
                    if ci is None:
                        ci = convolve_indices([a] * kk, quantizer, max_fine_bucket)
                        cache[kk] = ci
                    return ci.get_in_bucket(fb, off)

                return getter

            prefix.append(CountIndex(hist_k, make_getter(k)))
        return prefix


# --- Axis B: ordered search (the existing enumerator), named for symmetry -------------------

OrderedStream = Iterator[tuple[Any, float]]
"""Exact strict-descending lazy stream of ``(value, log_prob)`` -- produced by ``enumerator()``."""


# --- Bridges between the axes ---------------------------------------------------------------


def enumerate_and_bin(stream: OrderedStream, quantizer: Quantizer, max_fine_bucket: int) -> tuple[CountIndex, bool]:
    """Axis B -> Axis A: tabulate an ordered stream into a bounded count index.

    The universal fallback for distributions that can only enumerate. O(number of in-bound values),
    so feasible only for small budgets. Returns ``(CountIndex, truncated)``.
    """
    return leaf_count_index(stream, quantizer, max_fine_bucket)


def bounded_dedup_stream(
    stream: OrderedStream, max_entries: int = 1 << 16, key: Callable[[Any], Any] | None = None
) -> OrderedStream:
    """Deduplicate an (approximately) descending ``(value, log_prob)`` stream in O(max_entries) memory.

    The structural BoundedCount index for a MARGINAL family (Mixture / HMM) emits a value once per
    contributing component / state-path; every copy reports the *same* exact ``log_density`` (it is
    path/component-independent), so the duplicates are exact repeats of the value. This wrapper keeps
    a least-recently-seen window of at most ``max_entries`` keys and suppresses repeats within it.

    Memory is hard-capped at ``max_entries`` -- never the full (up to 2**M) support. The trade: a
    duplicate whose two occurrences are more than ``max_entries`` distinct values apart in the stream
    (i.e. its second copy is far deeper / effectively outside the bound of interest) may survive. Set
    ``max_entries`` to bound how far apart duplicates can be and still be removed.
    """
    from collections import OrderedDict

    if key is None:
        from pysp.utils.enumeration import freeze as key
    seen: OrderedDict = OrderedDict()
    for value, lp in stream:
        k = key(value)
        if k in seen:
            seen.move_to_end(k)
            continue
        seen[k] = None
        if len(seen) > max_entries:
            seen.popitem(last=False)
        yield value, lp


def ordered_stream_from_count_index(index, max_items: int | None = None) -> OrderedStream:
    """Axis A -> Axis B: unrank a built count index in coarse-bin order (approximately descending).

    ``index`` is a built LazyQuantizedEnumerationIndex (from ``count_budget_index``). The order is
    exact across coarse bins but unspecified within a bin -- a good-enough stream when exact
    best-first enumeration is too expensive.
    """
    n = index.total_count if max_items is None else min(max_items, index.total_count)
    for i in range(n):
        yield index.get(i)
