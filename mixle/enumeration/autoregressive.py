"""Make any autoregressive model count-/threshold-/unrank-able by mixle's enumeration machinery.

:mod:`~mixle.enumeration.model_enumeration` already *lists* an autoregressive model's sequences in
descending probability (``best_first_decode``). That is the right tool for the top handful, but it does not
scale: to reach the k-th most probable sequence it must expand ~k prefixes, so a rank like 1e8 is hopeless.

This module adds the **count / threshold / unrank** surface for the *same* ``next_logprobs(prefix)`` callback,
so you can answer the questions that do *not* require listing:

* **count(min_log_prob)** -- how many sequences are at least this probable (without listing them),
* **threshold(rank)** -- the log-probability of the k-th most probable sequence (the top-k boundary),
* **unrank(i)** -- the i-th most probable sequence, by random access (one model query per step), and
* **mass_above(min_log_prob)** -- a bracket on the cumulative probability of that head.

The trick (see ``notes/enumerating-a-language-model.md``): the number of model forward passes is bounded by
the number of distinct *prefixes* (<= V^(L-1)), **not** by the rank k. We build a count histogram per prefix
and compose them up the prefix tree -- but because each step ``p(x_t | prefix)`` is a *distinct* function of
the prefix, the children are **not** independent, so this is a tree recursion (sum of per-token *shifted*
child histograms), not the independent-factor convolution that :func:`convolve_indices` does for ``Composite``.

The bridge is a thin adapter, :class:`AutoregressiveEnumerable`, that implements just enough of the
distribution count-index contract (:meth:`~AutoregressiveEnumerable.quantized_count_index`,
:meth:`~AutoregressiveEnumerable.log_density`, :meth:`~AutoregressiveEnumerable.structural_fine_bucket`) that
the existing drivers -- :func:`~mixle.enumeration.quantization.core.count_budget_index` and the
:mod:`~mixle.enumeration.density_rank` seek/rank/cumulative/nucleus functions -- work on it unchanged.

Example (transformer-style next-token decoding)::

    import numpy as np
    def next_logprobs(prefix):
        logits = my_transformer(prefix)                 # (vocab,) -> numpy
        lp = logits - logsumexp(logits)                 # log_softmax (<= 0)
        return list(enumerate(lp))                       # [(token_id, log_prob), ...]

    ar = AutoregressiveEnumerable(next_logprobs, max_len=2)
    ar.threshold(10**8)        # log-prob of the 100,000,000-th most probable length-2 sequence
    ar.count(min_log_prob)     # how many length-2 sequences are at least that probable
    ar.unrank(10**6)           # the millionth most probable sequence, without listing the first 1e6
    ar.top_k(5)                # the 5 most probable (exact best-first; for small k)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from mixle.enumeration.model_enumeration import best_first_decode
from mixle.enumeration.quantization.core import (
    CountHistogram,
    CountIndex,
    Quantizer,
    count_budget_index,
)

_NEG_INF = -math.inf


def autoregressive_count_index(
    steps: Callable[[tuple], list[tuple[Any, float]]],
    prefix: tuple,
    depth: int,
    quantizer: Quantizer,
    max_fine_bucket: int,
    eos: Any = None,
) -> tuple[CountIndex, bool]:
    """Tree-recursive count index over completions of ``prefix`` up to ``depth`` more tokens.

    Returns ``(CountIndex, truncated)``. The histogram counts completions by fine bucket of total bits;
    ``CountIndex.get_in_bucket(fb, offset)`` unranks the structural ``(token, ...)`` sequence and its exact
    log-probability. ``truncated`` is True if any completion was dropped at the ``max_fine_bucket`` depth
    bound (so a caller can deepen).

    Each step's bits ``-log2 p(x_t | prefix)`` are added to every completion bucket via
    :meth:`CountHistogram.shift`; the per-token children are pooled with :meth:`CountHistogram.add`. Because
    ``steps`` are taken in descending probability, once a token's own bits exceed the remaining budget every
    later token does too, so the loop can stop -- this is what bounds the work to the live prefixes.
    """
    # Complete: the empty completion sits in bucket 0 with log-prob 0 (multiplicative identity).
    if depth <= 0 or (eos is not None and prefix and prefix[-1] == eos):
        return CountIndex(CountHistogram.delta(0, 1), lambda fb, off: ((), 0.0)), False

    truncated = False
    by_token: list[tuple[Any, float, int, CountIndex]] = []
    acc: dict[int, int] = {}  # fine_bucket -> count, pooled across tokens (avoids O(V) array rebuilds)
    for token, step_lp in steps(prefix):
        sb = quantizer.fine_bucket(step_lp)
        if sb > max_fine_bucket:
            truncated = True  # steps are descending, so all remaining tokens also exceed the budget
            break
        child, child_trunc = autoregressive_count_index(
            steps, prefix + (token,), depth - 1, quantizer, max_fine_bucket - sb, eos
        )
        truncated = truncated or child_trunc
        h = child.hist
        if h.is_empty():
            continue
        for i, c in enumerate(h.data):  # shift the child by this step's bits and pool it in
            if c:
                fb = h.base + i + sb
                acc[fb] = acc.get(fb, 0) + c
        by_token.append((token, step_lp, sb, child))

    if not acc:
        return CountIndex(CountHistogram.empty(), lambda fb, off: (_ for _ in ()).throw(IndexError())), truncated

    lo, hi = min(acc), max(acc)
    data = [0] * (hi - lo + 1)
    for fb, c in acc.items():
        data[fb - lo] = c
    joint = CountHistogram(lo, data)

    def getter(fb: int, off: int) -> tuple[Any, float]:
        o = int(off)
        for token, step_lp, sb, child in by_token:
            cfb = int(fb) - sb
            c = child.hist.count_at(cfb)
            if o < c:
                cval, clp = child.get_in_bucket(cfb, o)
                return (token,) + cval, step_lp + clp
            o -= c
        raise IndexError("offset %d outside autoregressive bucket %d" % (off, fb))

    return CountIndex(joint, getter), truncated


class _ARSampler:
    """Ancestral sampler over the model -- token by token from ``next_logprobs`` (for the rank tail fallback)."""

    def __init__(self, model: AutoregressiveEnumerable, seed: int | None) -> None:
        import numpy as np

        self._model = model
        self._rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        import numpy as np

        n = 1 if size is None else int(size)
        out = []
        for _ in range(n):
            prefix: tuple = ()
            for _ in range(self._model.max_len):
                items = self._model._steps(prefix)
                toks = [t for t, _ in items]
                lps = np.array([lp for _, lp in items], dtype=float)
                p = np.exp(lps - np.max(lps))
                p /= p.sum()
                j = int(self._rng.choice(len(toks), p=p))
                prefix = prefix + (toks[j],)
                if self._model.eos is not None and toks[j] == self._model.eos:
                    break
            out.append(prefix)
        return out[0] if size is None else out


class AutoregressiveEnumerable:
    """Adapter: an autoregressive ``next_logprobs(prefix)`` model as a count-/rank-/unrank-able object.

    Args:
        next_logprobs: ``next_logprobs(prefix) -> [(token, log_prob), ...]`` -- the next-token log-probabilities
            (``<= 0``) given a prefix tuple, e.g. the ``log_softmax`` of a transformer's next-token logits.
        max_len: sequence length to enumerate (every path is completed at this many tokens; with ``eos`` a
            path may complete earlier).
        eos: optional end-of-sequence token; a prefix ending in ``eos`` is complete and not extended.
        bin_width_bits, oversample: quantization resolution of the count index (finer = exacter ordering,
            more memory). The defaults match the distribution count-DP.

    The model is queried lazily and **memoized by prefix**, so deepening the index (or recomputing a
    log-density) never re-runs a forward pass it has already seen.
    """

    def __init__(
        self,
        next_logprobs: Callable[[tuple], Iterable[tuple[Any, float]]],
        max_len: int,
        eos: Any = None,
        bin_width_bits: float = 1.0,
        oversample: int = 8,
    ) -> None:
        if max_len is None or int(max_len) < 1:
            raise ValueError("max_len must be a positive integer (the sequence length to enumerate).")
        self.next_logprobs = next_logprobs
        self.max_len = int(max_len)
        self.eos = eos
        self.bin_width_bits = float(bin_width_bits)
        self.oversample = int(oversample)
        self._cache: dict[tuple, list[tuple[Any, float]]] = {}

    # -- the model oracle, sorted-descending and memoized -------------------------------------------------
    def _steps(self, prefix: tuple) -> list[tuple[Any, float]]:
        cached = self._cache.get(prefix)
        if cached is None:
            cached = sorted(
                ((t, float(lp)) for t, lp in self.next_logprobs(prefix) if lp != _NEG_INF),
                key=lambda u: -u[1],
            )
            self._cache[prefix] = cached
        return cached

    # -- the count-index contract (this is all the existing drivers need) ---------------------------------
    def quantized_count_index(self, quantizer: Quantizer, max_fine_bucket: int) -> tuple[CountIndex, bool]:
        """Count index over all length-``max_len`` (or eos-terminated) sequences, bounded by depth bits."""
        return autoregressive_count_index(self._steps, (), self.max_len, quantizer, max_fine_bucket, self.eos)

    def log_density(self, sequence: Iterable[Any]) -> float:
        """Exact total log-probability of a sequence (``-inf`` if any token is off-support given its prefix)."""
        lp = 0.0
        prefix: tuple = ()
        for token in sequence:
            table = dict(self._steps(prefix))
            if token not in table:
                return _NEG_INF
            lp += table[token]
            prefix = prefix + (token,)
        return lp

    def structural_fine_bucket(self, sequence: Iterable[Any], quantizer: Quantizer) -> int:
        return quantizer.fine_bucket(self.log_density(tuple(sequence)))

    def sampler(self, seed: int | None = None) -> _ARSampler:
        return _ARSampler(self, seed)

    def enumerator(self):
        """A descending-probability iterator (lazy best-first); use ``top_k`` for the head."""
        return best_first_decode(lambda prefix: self._steps(prefix), eos=self.eos, max_len=self.max_len)

    # -- convenience surface (self-contained, via the core count-budget driver) ---------------------------
    def _quantizer(self) -> Quantizer:
        return Quantizer(bin_width_bits=self.bin_width_bits, oversample=self.oversample)

    def budget_index(self, budget_bits: float, max_depth_bits: float = 4096.0):
        """The count-budget seek index covering at least ``2**budget_bits`` sequences (for unrank/iterate)."""
        return count_budget_index(
            self,
            budget_bits=budget_bits,
            bin_width_bits=self.bin_width_bits,
            oversample=self.oversample,
            max_depth_bits=max_depth_bits,
        )

    def top_k(self, k: int) -> list[tuple[tuple, float]]:
        """The ``k`` most probable sequences, exact, by best-first listing (use for small ``k``)."""
        out = []
        for seq, lp in self.enumerator():
            out.append((seq, lp))
            if len(out) >= k:
                break
        return out

    def count(self, min_log_prob: float) -> int:
        """How many sequences have ``log_density >= min_log_prob`` -- computed from counts, not listed."""
        q = self._quantizer()
        index, _truncated = self.quantized_count_index(q, q.fine_bucket(min_log_prob))
        return index.total()

    def unrank(self, i: int) -> tuple[tuple, float]:
        """The ``i``-th most probable sequence (0-based) and its exact log-probability, by random access."""
        if i < 0:
            raise IndexError("rank must be >= 0")
        budget_bits = max(self.bin_width_bits, math.log2(i + 2) + 1.0)
        index = self.budget_index(budget_bits)
        if i >= len(index):
            raise IndexError("rank %d beyond the enumerable support (size %d)" % (i, len(index)))
        return index.get(i)

    def threshold(self, rank: int) -> float:
        """Log-probability of the ``rank``-th most probable sequence -- the boundary of the top-``rank`` set."""
        if rank < 1:
            raise ValueError("rank must be >= 1")
        _seq, lp = self.unrank(rank - 1)
        return lp

    def mass_above(self, min_log_prob: float) -> tuple[float, float]:
        """A ``(lower, upper)`` bracket on the total probability of sequences with ``log_density >= min_log_prob``.

        Computed from the count histogram alone (no enumeration): each fine bucket of ``c`` sequences
        contributes between ``c * 2**(-hi_bits)`` and ``c * 2**(-lo_bits)``, where the bucket spans
        ``[lo_bits, hi_bits)`` of information. Tighten by raising ``oversample``.
        """
        q = self._quantizer()
        index, _truncated = self.quantized_count_index(q, q.fine_bucket(min_log_prob))
        hist = index.hist
        lo = hi = 0.0
        per_bit = q.fine_per_bit()
        # A joint fine bucket is the SUM of per-step floor-quantized buckets, so accumulated rounding can put a
        # sequence's exact information anywhere in [fb / per_bit, (fb + max_len) / per_bit) bits -- the spread
        # grows with the number of steps, not 1/oversample. Bound the bucket's probability over that range.
        for j, c in enumerate(hist.data):
            if not c:
                continue
            fb = hist.base + j
            lo_bits = fb / per_bit  # least information in the bucket -> most probable edge
            hi_bits = (fb + self.max_len) / per_bit  # most information after up to max_len roundings
            hi += c * 2.0 ** (-lo_bits)
            lo += c * 2.0 ** (-hi_bits)
        return lo, hi

    # -- the full enumerator surface, delegated to the shared density-rank machinery ----------------------
    def seek(self, index: int):
        """:class:`~mixle.enumeration.density_rank.CountDPSeekResult` at descending ``index`` (with a bracket)."""
        from mixle.enumeration.density_rank import count_dp_seek

        return count_dp_seek(self, index)

    def rank(self, sequence: Iterable[Any]):
        """:class:`~mixle.enumeration.density_rank.DensityRankResult` -- rank + cumulative mass of a sequence."""
        from mixle.enumeration.density_rank import density_rank

        return density_rank(self, tuple(sequence))

    def cumulative(self, sequence: Iterable[Any]):
        """``G(seq) = P(p(Y) >= p(seq))`` -- total mass of sequences at least as probable as ``seq``."""
        from mixle.enumeration.density_rank import cumulative_probability

        return cumulative_probability(self, tuple(sequence))

    def nucleus_size(self, p: float):
        """Size of the minimal ``>= p``-mass set (:class:`CountDPTopPResult`), without materializing it."""
        from mixle.enumeration.density_rank import count_dp_top_p

        return count_dp_top_p(self, p)
