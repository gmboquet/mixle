"""A persistent count-budget enumeration index: precompute once, then unrank/count/slice many times.

The one-shot drivers (:func:`~mixle.enumeration.quantization.core.count_budget_index`,
:func:`~mixle.enumeration.density_rank.count_dp_seek`, the :class:`AutoregressiveEnumerable` convenience
methods) rebuild the structural count DP on **every** query -- fine for a single lookup, wasteful for the
common pattern of many queries against one model (seek a thousand ranks, sweep thresholds, page through the
support). For a transformer LM every rebuild re-walks the whole live prefix tree in Python even when the
forward passes are memoized.

:class:`SeekIndex` is the reusable precomputation structure:

* **built once** -- the fine count histogram and its structural unranker are computed at a depth (bit budget)
  and kept;
* **queried many times** -- ``unrank`` / ``slice`` / ``iter_from`` / ``count`` / ``threshold`` /
  ``rank_bracket`` all read the stored tables (a query costs an O(log #bins) bisect plus one structural
  unrank walk, never a DP rebuild);
* **deepened in place** -- a query past the built depth triggers one rebuild at a geometrically deeper
  budget (models that memoize their evaluations, e.g. the autoregressive adapter's forward cache, make the
  re-walk much cheaper than the first build).

Approximation contract: identical to the underlying count DP -- the bin *assignment* is quantized (ordering
is exact up to the fine-bucket width) and, for the tropical Mixture/HMM indices, counts are the structural
upper bound documented on those families. With ``count_mode='float'`` the counts themselves are float64
(exact below 2**53, ~1e-16 relative error per operation beyond); every unranked value still carries its
**exact** ``log_density``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

from mixle.enumeration.quantization.core import CountIndex, Quantizer, build_budget_index

__all__ = ["SeekIndex"]


class SeekIndex:
    """Build a model's count-budget index once and answer many enumeration queries against it.

    ``model`` is anything implementing the count-index contract -- every mixle distribution with a
    ``quantized_count_index`` (Composite/Sequence/Markov exactly; Mixture/HMM as the documented tropical
    bound) and :class:`~mixle.enumeration.autoregressive.AutoregressiveEnumerable` (so LLMs plug in).

    Queries auto-deepen: asking for a rank (or a log-prob threshold) beyond the built depth rebuilds the
    DP at a geometrically larger budget, reusing whatever the model itself memoizes. ``max_depth_bits``
    caps the deepening; past it, queries raise ``IndexError`` for out-of-range ranks.
    """

    def __init__(
        self,
        model: Any,
        *,
        bin_width_bits: float = 1.0,
        oversample: int = 8,
        count_mode: str = "exact",
        max_depth_bits: float = 4096.0,
    ) -> None:
        self.model = model
        self.quantizer = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample, count_mode=count_mode)
        self.max_depth_bits = float(max_depth_bits)
        self._count_index: CountIndex | None = None
        self._budget_index = None  # LazyQuantizedEnumerationIndex over the built histogram
        self._built_bits: float = 0.0
        self._truncated: bool = True
        self._builds: int = 0  # DP rebuild counter (observable: tests assert queries reuse the build)

    # -- build / deepen ----------------------------------------------------------------------------------

    @property
    def built_bits(self) -> float:
        """The depth (bit budget) the index is currently built to (0 before the first build)."""
        return self._built_bits

    @property
    def truncated(self) -> bool:
        """True when values beyond the built depth exist (deepening can still grow the index)."""
        return self._truncated

    @property
    def builds(self) -> int:
        """How many times the structural DP has been (re)built -- the cost a persistent index amortizes."""
        return self._builds

    @property
    def dropped_upper(self) -> float:
        """Certified upper bound on in-budget values excluded by an approximation knob (e.g. ``branch_cap``).

        0.0 for exhaustive indices. The true in-budget count lies in ``[len(self), len(self) + dropped_upper]``;
        deepening does not recover these (that is what distinguishes it from ``truncated``).
        """
        return float(getattr(self._count_index, "dropped_upper", 0.0)) if self._count_index is not None else 0.0

    def _build(self, depth_bits: float) -> None:
        depth_bits = min(float(depth_bits), self.max_depth_bits)
        max_fb = int(math.ceil(depth_bits * self.quantizer.fine_per_bit()))
        self._count_index, self._truncated = self.model.quantized_count_index(self.quantizer, max_fb)
        self._built_bits = depth_bits
        self._builds += 1
        self._budget_index = build_budget_index(
            self._count_index,
            self.quantizer,
            None,  # no count cutoff: expose everything built; depth is managed here
            exact_log_density=self.model.log_density,
            truncated=self._truncated,
        )

    def ensure_bits(self, depth_bits: float) -> SeekIndex:
        """Make the index cover at least ``depth_bits`` of information (build or deepen if needed)."""
        needed = max(float(depth_bits), self.quantizer.bin_width_bits)
        if self._budget_index is None:
            self._build(needed)
        elif needed > self._built_bits and self._truncated:
            self._build(min(max(needed, self._built_bits * 2.0), self.max_depth_bits))
        return self

    def ensure_count(self, n: int) -> SeekIndex:
        """Make the index hold at least ``n`` values (geometric deepening from the current depth).

        Every rebuild at least doubles the built depth, so an ascending sequence of queries costs
        O(log) rebuilds total and the final (deepest) build dominates the work -- the amortization a
        persistent index exists for.
        """
        if self._budget_index is not None and (self._budget_index.total_count >= n or not self._truncated):
            return self
        depth = max(self.quantizer.bin_width_bits, math.log2(max(int(n), 1) + 1) + 1.0)
        if self._budget_index is not None:
            depth = max(depth, self._built_bits * 2.0)
        while True:
            self._build(depth)
            if self._budget_index.total_count >= n or not self._truncated or depth >= self.max_depth_bits:
                return self
            depth = min(depth * 2.0, self.max_depth_bits)

    # -- queries (all reuse the built tables) --------------------------------------------------------------

    def __len__(self) -> int:
        if self._budget_index is None:
            self.ensure_bits(self.quantizer.bin_width_bits)
        return self._budget_index.total_count

    def unrank(self, i: int) -> tuple[Any, float]:
        """The ``i``-th most probable value (0-based, quantized order) and its exact log-probability.

        Order is exact **between** fine buckets (width ``bin_width_bits / oversample`` bits) but
        unspecified **within** one: two values whose ``log_density`` differs by less than one bucket's
        width can be returned in either relative order (see :meth:`slice`'s note -- narrow that window
        by raising ``oversample`` or lowering ``bin_width_bits``). The returned ``log_density`` is always
        exact regardless.
        """
        if i < 0:
            raise IndexError("rank must be >= 0")
        self.ensure_count(int(i) + 1)
        if i >= self._budget_index.total_count:
            raise IndexError("rank %d beyond the enumerable support (size %d)" % (i, self._budget_index.total_count))
        return self._budget_index.get(int(i))

    def slice(self, start: int, k: int) -> list[tuple[Any, float]]:
        """Up to ``k`` values starting at rank ``start`` (one deepen at most, then table reads).

        Not a strict sort by ``log_density``: values are ordered by quantized fine bucket (bucket width
        ``bin_width_bits / oversample`` bits), and within a shared bucket the order follows the structural
        enumeration (token/branch order), not the exact log-density. Two values less than one bucket-width
        apart can therefore come back in either order -- this is a documented property of the quantization,
        not a bug, and it is independent of ``branch_cap``/pruning. If a caller needs a strict top-k sort
        (e.g. asserting descending order across close candidates), either re-sort the slice by its returned
        ``log_density`` values, or shrink the ambiguity window by raising ``oversample`` / lowering
        ``bin_width_bits`` on the model/index -- it cannot be eliminated, only made arbitrarily small.
        """
        if start < 0:
            raise IndexError("start must be non-negative")
        if k < 0:
            raise ValueError("k must be non-negative")
        self.ensure_count(int(start) + int(k))
        return self._budget_index.slice(int(start), int(k))

    def iter_from(self, start: int = 0) -> Iterator[tuple[Any, float]]:
        """Iterate values from rank ``start`` through the end of the *built* index (no auto-deepen)."""
        self.ensure_count(int(start) + 1)
        return self._budget_index.iter_from(int(start))

    def count(self, min_log_prob: float) -> int | float:
        """How many values have ``log_density >= min_log_prob`` -- read off the fine histogram.

        Exact-carrier families return an ``int``; ``count_mode='float'`` (or a float-carrying model like
        the deep-budget autoregressive fast path) returns a float with the documented relative error.
        Mixture/HMM counts are the structural (tropical) upper bound those families document.
        """
        self.ensure_bits(self.quantizer.bits(min_log_prob) + self.quantizer.bin_width_bits)
        fb = self.quantizer.fine_bucket(min_log_prob)
        hist = self._count_index.hist
        total = 0
        for j, c in enumerate(hist.data):
            if hist.base + j > fb:
                break
            total = total + c
        return total

    def threshold(self, rank: int) -> float:
        """Log-probability of the ``rank``-th most probable value (the top-``rank`` boundary)."""
        if rank < 1:
            raise ValueError("rank must be >= 1")
        _value, lp = self.unrank(rank - 1)
        return lp

    def rank_bracket(self, value: Any) -> tuple[int, int]:
        """A ``[lo, hi]`` bracket on ``value``'s quantized rank, from its structural fine bucket.

        ``lo`` counts every value in strictly shallower buckets; ``hi`` adds the rest of the value's own
        bucket. For Mixture/HMM the bucket is the documented tropical projection, so the bracket carries
        that projection's semantics (see the family docstrings).
        """
        lp = self.model.log_density(value)
        fb = (
            self.model.structural_fine_bucket(value, self.quantizer)
            if hasattr(self.model, "structural_fine_bucket")
            else self.quantizer.fine_bucket(lp)
        )
        self.ensure_bits((fb + 1) / self.quantizer.fine_per_bit())
        hist = self._count_index.hist
        lo = 0
        for j, c in enumerate(hist.data):
            if hist.base + j >= fb:
                break
            lo = lo + c
        in_bucket = hist.count_at(fb)
        lo_i = int(lo)
        return lo_i, lo_i + max(int(in_bucket) - 1, 0)

    def fine_histogram(self, min_depth_bits: float | None = None):
        """The built fine :class:`CountIndex` (ensuring at least ``min_depth_bits``) -- for mass/rank math."""
        self.ensure_bits(self.quantizer.bin_width_bits if min_depth_bits is None else float(min_depth_bits))
        return self._count_index
