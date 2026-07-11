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

The key accounting fact is that the number of model forward passes is bounded by the number of
distinct *prefixes* (<= V^(L-1)), **not** by the rank k. We build a count histogram per prefix and
compose them up the prefix tree -- but because each step ``p(x_t | prefix)`` is a *distinct* function
of the prefix, the children are **not** independent, so this is a tree recursion (sum of per-token
*shifted* child histograms), not the independent-factor convolution that :func:`convolve_indices` does
for ``Composite``.

The bridge is a thin adapter, :class:`AutoregressiveEnumerable`, that implements the parts of the
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

    ar = AutoregressiveEnumerable(next_logprobs, max_len=2)   # fixed-length: support = all length-2 sequences
    ar.threshold(10**8)        # log-prob of the 100,000,000-th most probable length-2 sequence
    ar.count(min_log_prob)     # how many length-2 sequences are at least that probable
    ar.unrank(10**6)           # the millionth most probable sequence, without listing the first 1e6
    ar.top_k(5)                # the 5 most probable (exact best-first; for small k)

    ar = AutoregressiveEnumerable(next_logprobs, eos=EOS)    # terminating: support = ONLY eos-terminated
    ar.unrank(10**6)           # the millionth most probable COMPLETE sequence (ends in eos), of any length

Support: a fixed-length model (``max_len``) has support on every length-``max_len`` sequence; a terminating
model (``eos``) has support ONLY on eos-terminated sequences, of any length, bounded by the probability budget
rather than a length cap. An un-terminated truncation has zero mass as an output and is never counted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from mixle.enumeration.model_enumeration import best_first_decode
from mixle.enumeration.quantization.core import (
    _LOG2,
    _TOL,
    CountHistogram,
    CountIndex,
    Quantizer,
    count_budget_index,
)

_NEG_INF = -math.inf
# The numpy fast path accumulates counts in int64. The number of sequences within a probability budget of B
# bits is <= 2**B (their probabilities sum to <= 1), so the fast path is exact while the requested budget stays
# below this; deeper budgets fall back to the arbitrary-precision Python recursion (identical results).
_INT64_SAFE_BITS = 60.0


def _raise_index(fb: int, off: int) -> tuple[Any, float]:
    raise IndexError("empty autoregressive count index")


def autoregressive_count_index(
    steps: Callable[[tuple], list[tuple[Any, float]]],
    prefix: tuple,
    depth: int,
    quantizer: Quantizer,
    max_fine_bucket: int,
    eos: Any = None,
    branch_cap: int | None = None,
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

    ``branch_cap`` recurses into only the top-``cap`` in-budget tokens per node -- the certified
    approximation for wide vocabularies. Each skipped token's subtree is bounded soundly (completions
    within ``r`` remaining bits number at most ``2**r``, since their conditional probabilities sum to at
    most 1) and the total accumulates in ``CountIndex.dropped_upper``: the true in-budget count lies in
    ``[total(), total() + dropped_upper]``. Dropped tokens do NOT set ``truncated`` (deepening cannot
    recover them; raising ``branch_cap`` can).
    """
    # A sequence ending in eos is a complete element of a terminating model's support (bucket 0, log-prob 0).
    if eos is not None and prefix and prefix[-1] == eos:
        return CountIndex(CountHistogram.delta(0, 1), lambda fb, off: ((), 0.0)), False
    if depth <= 0:
        # Depth bound reached without terminating. For a fixed-length model (eos is None) the length IS the
        # support, so this is a complete sequence; for a terminating model the truncation is NOT in the support
        # -- contribute nothing, and flag truncation so the caller raises the bit budget (not the length).
        if eos is None:
            return CountIndex(CountHistogram.delta(0, 1), lambda fb, off: ((), 0.0)), False
        return CountIndex(CountHistogram.empty(), _raise_index), True

    truncated = False
    dropped = 0.0
    kept = 0
    per_bit = quantizer.fine_per_bit()
    by_token: list[tuple[Any, float, int, CountIndex]] = []
    acc: dict[int, int] = {}  # fine_bucket -> count, pooled across tokens (avoids O(V) array rebuilds)
    for token, step_lp in steps(prefix):
        sb = quantizer.fine_bucket(step_lp)
        if sb > max_fine_bucket:
            truncated = True  # steps are descending, so all remaining tokens also exceed the budget
            break
        if branch_cap is not None and kept >= branch_cap:
            # skipped in-budget token: its subtree holds at most 2**remaining_bits completions
            dropped += 2.0 ** ((max_fine_bucket - sb) / per_bit)
            continue
        kept += 1
        child, child_trunc = autoregressive_count_index(
            steps, prefix + (token,), depth - 1, quantizer, max_fine_bucket - sb, eos, branch_cap
        )
        truncated = truncated or child_trunc
        dropped += child.dropped_upper
        h = child.hist
        if h.is_empty():
            continue
        for i, c in enumerate(h.data):  # shift the child by this step's bits and pool it in
            if c:
                fb = h.base + i + sb
                acc[fb] = acc.get(fb, 0) + c
        by_token.append((token, step_lp, sb, child))

    if not acc:
        empty = CountIndex(
            CountHistogram.empty(), lambda fb, off: (_ for _ in ()).throw(IndexError()), dropped_upper=dropped
        )
        return empty, truncated

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

    return CountIndex(joint, getter, dropped_upper=dropped), truncated


def _ar_count_index_fast(
    steps_np: Callable[[tuple], tuple[np.ndarray, np.ndarray]],
    prefix: tuple,
    depth: int,
    quantizer: Quantizer,
    max_fine_bucket: int,
    eos: Any = None,
    dtype: type = np.int64,
    branch_cap: int | None = None,
) -> tuple[CountIndex, bool]:
    """numpy-vectorized :func:`autoregressive_count_index` (int64 or float64 counts).

    Identical results to the reference implementation, but the per-prefix work is vectorized: the V step
    log-probs are binned with one :func:`numpy.floor` + :func:`numpy.bincount` instead of a Python loop over
    the vocabulary, and child histograms are pooled with numpy slice-adds. ``steps_np(prefix)`` returns
    ``(tokens, log_probs)`` as numpy arrays sorted by descending log-prob. With ``dtype=int64`` counts are
    exact while the budget stays below ~``2**62`` (see :data:`_INT64_SAFE_BITS`); with ``dtype=float64``
    the same recursion carries **approximate** counts at any depth -- exact below 2**53, ~1e-16 relative
    error per pooling beyond -- so deep budgets keep numpy speed instead of falling back to the
    arbitrary-precision Python path.
    """
    if eos is not None and prefix and prefix[-1] == eos:
        return CountIndex(CountHistogram.delta(0, 1), lambda fb, off: ((), 0.0)), False
    if depth <= 0:
        if eos is None:  # fixed-length model: the length is the support, so this completes
            return CountIndex(CountHistogram.delta(0, 1), lambda fb, off: ((), 0.0)), False
        return CountIndex(CountHistogram.empty(), _raise_index), True  # terminating: truncation not in support

    tokens, lps = steps_np(prefix)
    sb = np.floor(np.maximum(0.0, -lps / _LOG2) * (quantizer.oversample / quantizer.bin_width_bits) + _TOL).astype(
        np.int64
    )
    keep = sb <= max_fine_bucket
    truncated = not bool(keep.all())
    tokens, lps, sb = tokens[keep], lps[keep], sb[keep]
    if tokens.size == 0:
        return CountIndex(CountHistogram.empty(), _raise_index), truncated

    dropped = 0.0
    if branch_cap is not None and tokens.size > branch_cap:
        tail_sb = sb[branch_cap:]
        if depth == 1 and eos is None:
            dropped = float(tail_sb.size)  # each dropped leaf token is exactly one completion
        else:
            # each skipped subtree holds at most 2**remaining_bits completions (probabilities sum <= 1)
            dropped = float(np.sum(2.0 ** ((max_fine_bucket - tail_sb) / quantizer.fine_per_bit())))
        tokens, lps, sb = tokens[:branch_cap], lps[:branch_cap], sb[:branch_cap]

    if depth == 1 and eos is None:
        # Fixed-length leaf: each kept token is a length-1 completion in fine bucket sb; bincount is the
        # histogram. (A terminating model has no fixed leaf depth -- only eos completes -- so it falls through
        # to the general recursion, where the eos base case supplies the variable-depth leaves.)
        order = np.argsort(sb, kind="stable")  # group by bucket, descending-lp order preserved within a bucket
        sb_s, tok_s, lp_s = sb[order], tokens[order], lps[order]
        base = int(sb_s[0])
        hist = CountHistogram(base, np.bincount(sb_s - base).tolist())

        def leaf_getter(fb: int, off: int, _sb=sb_s, _tok=tok_s, _lp=lp_s) -> tuple[Any, float]:
            start = int(np.searchsorted(_sb, int(fb), side="left"))
            j = start + int(off)
            if off < 0 or j >= _sb.size or int(_sb[j]) != int(fb):
                raise IndexError("offset %d outside leaf bucket %d" % (off, fb))
            return (_tok[j].item(),), float(_lp[j])

        return CountIndex(hist, leaf_getter, dropped_upper=dropped), truncated

    # depth > 1: recurse into each token's subtree, then pool the shifted child histograms with numpy.
    by_token: list[tuple[Any, float, int, CountIndex]] = []
    shifted: list[tuple[int, np.ndarray]] = []
    for tok, lp, s in zip(tokens.tolist(), lps.tolist(), sb.tolist()):
        child, child_trunc = _ar_count_index_fast(
            steps_np, prefix + (tok,), depth - 1, quantizer, max_fine_bucket - s, eos, dtype, branch_cap
        )
        truncated = truncated or child_trunc
        dropped += child.dropped_upper
        if not child.hist.data:
            continue
        shifted.append((child.hist.base + s, np.asarray(child.hist.data, dtype=dtype)))
        by_token.append((tok, float(lp), int(s), child))

    if not shifted:
        return CountIndex(CountHistogram.empty(), _raise_index, dropped_upper=dropped), truncated
    lo = min(s for s, _ in shifted)
    hi = max(s + d.size - 1 for s, d in shifted)
    buf = np.zeros(hi - lo + 1, dtype=dtype)
    for s, d in shifted:
        buf[s - lo : s - lo + d.size] += d
    joint = CountHistogram(lo, buf.tolist())

    def getter(fb: int, off: int) -> tuple[Any, float]:
        o = int(off)
        for tok, lp, s, child in by_token:
            cfb = int(fb) - s
            c = child.hist.count_at(cfb)
            if o < c:
                cval, clp = child.get_in_bucket(cfb, o)
                return (tok,) + cval, lp + clp
            o -= c
        raise IndexError("offset %d outside autoregressive bucket %d" % (off, fb))

    return CountIndex(joint, getter, dropped_upper=dropped), truncated


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
            for _ in range(self._model._depth):
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

    The support depends on the model. A **fixed-length** model (``max_len`` set, no ``eos``) has support on
    every length-``max_len`` sequence. A **terminating** model (``eos`` set) has support *only* on sequences
    that end in ``eos`` -- so the enumeration counts/ranks/unranks exactly those, of any length, bounded by the
    probability budget (a tight terminating model has finitely many sequences above any threshold). A length
    bound is NOT the support boundary there: an un-terminated truncation has zero mass as an output and is
    never counted. ``max_depth`` is only a safety cap on recursion for non-tight models.

    Args:
        next_logprobs: ``next_logprobs(prefix) -> [(token, log_prob), ...]`` -- the next-token log-probabilities
            (``<= 0``) given a prefix tuple, e.g. the ``log_softmax`` of a transformer's next-token logits. For
            speed it may instead return the ``(tokens, log_probs)`` numpy-array pair (skips per-token boxing).
            For a terminating model ``eos`` must be one of the tokens it can return.
        max_len: for a fixed-length model, the sequence length (the support is all length-``max_len`` sequences).
            Omit for a terminating model (or pass it as a hard length cap, but un-terminated truncations are
            still dropped).
        eos: end-of-sequence token. When given, the model is terminating: only sequences ending in ``eos`` are
            in the support.
        max_depth: safety bound on recursion depth for a terminating model (the probability budget is the real
            bound). Raise it if a tight model legitimately produces very long sequences.
        bin_width_bits, oversample: quantization resolution of the count index (finer = exacter ordering,
            more memory). The defaults match the distribution count-DP. Ordering from ``unrank``/``slice``
            (via :meth:`seek_index`) is exact between fine buckets (width ``bin_width_bits / oversample``
            bits) but NOT guaranteed within one -- see :meth:`~mixle.enumeration.seek_index.SeekIndex.slice`.
            Sequences whose ``log_density`` differs by less than one bucket's width can surface in either
            relative order; this is most visible when several near-tied candidates cluster at the head of
            a small/short-sequence model. Raise ``oversample`` or lower ``bin_width_bits`` to shrink it.
        batch_next_logprobs: optional ``batch_next_logprobs([prefix, ...]) -> [result, ...]`` scoring many
            prefixes in one (padded) forward. When given, the count index warms its forward cache breadth-first
            in ``batch_size`` chunks -- the large speed-up for transformers, where one-at-a-time forwards
            dominate (e.g. distilGPT-2 length-2 to rank 1e5: ~25 s one-at-a-time -> ~1 s batched).
        batch_size: prefixes per batched forward.
        count_mode: how counts are carried past the int64-exact regime (budgets over ~2**60 sequences).
            ``'auto'`` (default) switches the numpy fast path to float64 there -- **approximate** counts
            (exact below 2**53, ~1e-16 relative error per pooling beyond) at full numpy speed. ``'exact'``
            preserves arbitrary-precision counts by falling back to the slow Python recursion. ``'float'``
            forces float64 everywhere.
        branch_cap: recurse into only the top-``branch_cap`` in-budget tokens per prefix -- the certified
            approximation for wide (LLM-sized) vocabularies, shrinking the tree by ~V/cap per level. The
            skipped remainder is soundly bounded (``count_bracket``/``dropped_upper``: a skipped subtree
            with ``r`` remaining budget bits holds at most ``2**r`` completions); enumeration covers the
            sub-support of sequences whose every token is among its context's top-``branch_cap``.
        batch_score_sequences: optional teacher-forcing scorer ``[sequence, ...] -> array of total log-probs``
            -- ONE forward per sequence (all positions score in parallel) instead of one forward per token.
            Used by :meth:`score_sequences` and, when a sequence's prefixes are not already cached, by
            :meth:`log_density`; the substrate for draft-rescored (speculative) enumeration.
        all_position_logprobs: optional ``sequence -> [next_logprobs result for seq[:d], d in 0..len-1]`` --
            one forward yields the full next-token distribution at EVERY position; harvested into the
            forward cache by :meth:`harvest`. Makes corpus-calibrated envelopes ~L-times cheaper.

    The model is queried lazily and **memoized by prefix**, so deepening the index (or recomputing a
    log-density) never re-runs a forward pass it has already seen. With integer tokens the histogram build
    is the numpy fast path (int64 counts below ~``2**60`` budgets; float64 beyond, per ``count_mode``).
    """

    def __init__(
        self,
        next_logprobs: Callable[[tuple], Iterable[tuple[Any, float]]],
        max_len: int | None = None,
        eos: Any = None,
        max_depth: int = 1024,
        bin_width_bits: float = 1.0,
        oversample: int = 8,
        batch_next_logprobs: Callable[[list[tuple]], list[Any]] | None = None,
        batch_size: int = 256,
        count_mode: str = "auto",
        branch_cap: int | None = None,
        batch_score_sequences: Callable[[list[tuple]], Any] | None = None,
        all_position_logprobs: Callable[[tuple], list[Any]] | None = None,
    ) -> None:
        if eos is None and max_len is None:
            raise ValueError("give max_len (a fixed-length model) or eos (a terminating model).")
        if max_len is not None and int(max_len) < 1:
            raise ValueError("max_len must be a positive integer.")
        if count_mode not in ("auto", "exact", "float"):
            raise ValueError("count_mode must be 'auto', 'exact', or 'float'")
        if branch_cap is not None and int(branch_cap) < 1:
            raise ValueError("branch_cap must be a positive integer (or None for no cap)")
        self.next_logprobs = next_logprobs
        self.eos = eos
        self.terminating = eos is not None
        self.max_len = None if max_len is None else int(max_len)
        self.max_depth = int(max_depth)
        # depth bound passed to the recursion: a fixed-length model completes at max_len; a terminating model
        # completes only at eos and uses max_len (if given) or max_depth purely as a safety cap.
        self._depth = self.max_len if not self.terminating else (self.max_len or self.max_depth)
        self.bin_width_bits = float(bin_width_bits)
        self.oversample = int(oversample)
        self.batch_next_logprobs = batch_next_logprobs
        self.batch_size = int(batch_size)
        self.count_mode = count_mode
        self.branch_cap = None if branch_cap is None else int(branch_cap)
        self.batch_score_sequences = batch_score_sequences
        self.all_position_logprobs = all_position_logprobs
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # prefix -> (tokens, log_probs), desc by lp
        self._fast: bool | None = None
        self._seek = None  # cached SeekIndex: built once, reused by unrank/count/threshold/mass_above

    # -- the model oracle, descending by log-prob and memoized (one forward per prefix) -------------------
    def _parse_steps(self, raw: Any) -> tuple[np.ndarray, np.ndarray]:
        # Accept the fast ``(tokens, log_probs)`` numpy form or a ``[(token, log_prob), ...]`` list.
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], np.ndarray):
            tokens, lps = np.asarray(raw[0]), np.asarray(raw[1], dtype=float)
        else:
            items = [(t, lp) for t, lp in raw if lp != _NEG_INF]
            tokens = np.array([t for t, _ in items])
            lps = np.array([float(lp) for _, lp in items], dtype=float)
        finite = np.isfinite(lps)
        if not finite.all():
            tokens, lps = tokens[finite], lps[finite]
        order = np.argsort(-lps, kind="stable")  # descending by log-prob
        return tokens[order], lps[order]

    def _steps_np(self, prefix: tuple) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(prefix)
        if cached is None:
            cached = self._parse_steps(self.next_logprobs(prefix))
            self._cache[prefix] = cached
        return cached

    def _steps(self, prefix: tuple) -> list[tuple[Any, float]]:
        tokens, lps = self._steps_np(prefix)
        return list(zip(tokens.tolist(), lps.tolist()))

    def _use_fast(self) -> bool:
        # The fast path needs integer tokens; int64-safety is enforced per call by the bit budget in
        # quantized_count_index (the count within a B-bit budget is <= 2**B), so it does not depend on depth.
        if self._fast is None:
            try:
                tokens, _ = self._steps_np(())
                self._fast = bool(tokens.dtype.kind in "iu" and tokens.size > 0)
            except (TypeError, ValueError):
                self._fast = False
        return self._fast

    def _prefetch(self, quantizer: Quantizer, max_fine_bucket: int, frontier_cap: int = 500_000) -> None:
        """Warm the forward cache breadth-first, scoring whole levels of live prefixes in batched forwards.

        The count index needs a forward for every live prefix (length 0..max_len-1); doing them one at a time
        is the transformer bottleneck. With ``batch_next_logprobs`` we score each level's uncached prefixes in
        ``batch_size`` chunks (one padded forward each), pruning to prefixes whose cumulative bits stay within
        ``max_fine_bucket``. If a level grows past ``frontier_cap`` we stop prefetching and let the recursion
        fetch the deep remainder lazily -- so deep/wide trees degrade gracefully instead of materializing.
        """
        if self.batch_next_logprobs is None:
            return
        scale = quantizer.oversample / quantizer.bin_width_bits
        frontier: list[tuple[tuple, int]] = [((), 0)]
        for length in range(self._depth):
            need = [
                pfx
                for pfx, _ in frontier
                if pfx not in self._cache and not (self.eos is not None and pfx and pfx[-1] == self.eos)
            ]
            if not need and length > 0:
                break  # budget pruned the frontier to nothing -- no deeper forwards needed
            for i in range(0, len(need), self.batch_size):
                chunk = need[i : i + self.batch_size]
                for pfx, raw in zip(chunk, self.batch_next_logprobs(chunk)):
                    if pfx not in self._cache:
                        self._cache[pfx] = self._parse_steps(raw)
            if length == self._depth - 1:
                break  # deepest forward done; no further expansion needed
            nxt: list[tuple[tuple, int]] = []
            for pfx, cum in frontier:
                if self.eos is not None and pfx and pfx[-1] == self.eos:
                    continue
                tokens, lps = self._steps_np(pfx)
                sb = np.floor(np.maximum(0.0, -lps / _LOG2) * scale + _TOL).astype(np.int64)
                live = (cum + sb) <= max_fine_bucket
                live_toks, live_cum = tokens[live].tolist(), (cum + sb[live]).tolist()
                if self.branch_cap is not None:  # the recursion only descends the top-cap tokens
                    live_toks, live_cum = live_toks[: self.branch_cap], live_cum[: self.branch_cap]
                for tok, s in zip(live_toks, live_cum):
                    nxt.append((pfx + (tok,), int(s)))
                if len(nxt) > frontier_cap:
                    return  # too wide to prefetch; the recursion forwards the rest lazily
            frontier = nxt

    # -- the count-index contract (this is all the existing drivers need) ---------------------------------
    def quantized_count_index(self, quantizer: Quantizer, max_fine_bucket: int) -> tuple[CountIndex, bool]:
        """Count index over the model's support (length-``max_len`` sequences, or all eos-terminated
        sequences), bounded by the bit budget ``max_fine_bucket``."""
        budget_bits = max_fine_bucket * quantizer.bin_width_bits / quantizer.oversample
        if self._use_fast():
            int64_safe = budget_bits < _INT64_SAFE_BITS
            if int64_safe and self.count_mode != "float":
                self._prefetch(quantizer, max_fine_bucket)
                return _ar_count_index_fast(
                    self._steps_np, (), self._depth, quantizer, max_fine_bucket, self.eos, np.int64, self.branch_cap
                )
            if self.count_mode in ("auto", "float"):
                # Deep budget (or forced float): carry counts as float64 -- approximate past 2**53, but the
                # build keeps numpy speed instead of dropping to the arbitrary-precision Python recursion.
                self._prefetch(quantizer, max_fine_bucket)
                return _ar_count_index_fast(
                    self._steps_np, (), self._depth, quantizer, max_fine_bucket, self.eos, np.float64, self.branch_cap
                )
        return autoregressive_count_index(
            self._steps, (), self._depth, quantizer, max_fine_bucket, self.eos, self.branch_cap
        )

    def log_density(self, sequence: Iterable[Any]) -> float:
        """Exact total log-probability of a sequence (``-inf`` if any token is off-support given its prefix).

        When a ``batch_score_sequences`` scorer is configured and any of the sequence's prefixes is not
        already cached, the score comes from ONE teacher-forcing forward instead of one forward per token.
        """
        seq = tuple(sequence)
        if self.batch_score_sequences is not None and any(seq[:d] not in self._cache for d in range(len(seq))):
            return float(np.asarray(self.batch_score_sequences([seq]), dtype=float).reshape(-1)[0])
        lp = 0.0
        prefix: tuple = ()
        for token in seq:
            table = dict(self._steps(prefix))
            if token not in table:
                return _NEG_INF
            lp += table[token]
            prefix = prefix + (token,)
        return lp

    def score_sequences(self, sequences: list[Any]) -> np.ndarray:
        """Exact total log-probabilities of many sequences -- batched teacher forcing when available.

        With ``batch_score_sequences`` this is one call (one forward per sequence, all positions in
        parallel); otherwise it falls back to per-sequence :meth:`log_density` over the cached walk. The
        rescoring primitive for draft-based (speculative) enumeration.
        """
        seqs = [tuple(s) for s in sequences]
        if not seqs:
            return np.zeros(0, dtype=float)
        if self.batch_score_sequences is not None:
            return np.asarray(self.batch_score_sequences(seqs), dtype=float).reshape(len(seqs))
        return np.array([self.log_density(s) for s in seqs], dtype=float)

    def harvest(self, sequence: Iterable[Any]) -> None:
        """Cache the next-token distribution at every prefix of ``sequence`` from one forward.

        Requires ``all_position_logprobs``; a no-op without it. Feeding typical sequences (a corpus, a
        provider's fast generations) through this warms the same memo cache the count index and the
        envelope read -- L cache entries per model call.
        """
        if self.all_position_logprobs is None:
            return
        seq = tuple(sequence)
        need = [d for d in range(len(seq)) if seq[:d] not in self._cache]
        if not need:
            return
        results = self.all_position_logprobs(seq)
        for d, raw in enumerate(results[: len(seq)]):
            prefix = seq[:d]
            if prefix not in self._cache:
                self._cache[prefix] = self._parse_steps(raw)

    def structural_fine_bucket(self, sequence: Iterable[Any], quantizer: Quantizer) -> int:
        """Return the quantized fine bucket for a sequence log density."""
        return quantizer.fine_bucket(self.log_density(tuple(sequence)))

    def sampler(self, seed: int | None = None) -> _ARSampler:
        """Return an autoregressive sampler."""
        return _ARSampler(self, seed)

    def enumerator(self):
        """A descending-probability iterator over the support (lazy best-first); use ``top_k`` for the head."""
        stream = best_first_decode(lambda prefix: self._steps(prefix), eos=self.eos, max_len=self._depth)
        if self.terminating:  # only eos-terminated sequences are in a terminating model's support
            return ((s, lp) for s, lp in stream if s and s[-1] == self.eos)
        return stream

    # -- convenience surface (persistent: one cached SeekIndex serves every query) --------------------------
    def _quantizer(self) -> Quantizer:
        return Quantizer(bin_width_bits=self.bin_width_bits, oversample=self.oversample)

    def seek_index(self, *, max_depth_bits: float = 4096.0):
        """The cached persistent :class:`~mixle.enumeration.seek_index.SeekIndex` over this model.

        Built lazily on first use and **reused by every convenience query** (``unrank`` / ``count`` /
        ``threshold`` / ``mass_above``), deepening in place when a query needs more depth -- so a sweep of
        a thousand unranks pays for one tree build, not a thousand. The forward cache is shared with it,
        so deepening only runs new forwards for newly-live prefixes.
        """
        if self._seek is None:
            from mixle.enumeration.seek_index import SeekIndex

            self._seek = SeekIndex(
                self,
                bin_width_bits=self.bin_width_bits,
                oversample=self.oversample,
                max_depth_bits=max_depth_bits,
            )
        return self._seek

    def budget_index(self, budget_bits: float, max_depth_bits: float = 4096.0):
        """The count-budget seek index covering at least ``2**budget_bits`` sequences (for unrank/iterate)."""
        return count_budget_index(
            self,
            budget_bits=budget_bits,
            bin_width_bits=self.bin_width_bits,
            oversample=self.oversample,
            max_depth_bits=max_depth_bits,
        )

    def envelope_index(self, *, n_paths: int = 64, seed: int = 0, budget_bits: float = 64.0):
        """An :class:`~mixle.enumeration.envelope.AREnvelopeIndex` over this model -- **approximate**
        enumeration at depths the exact tree index cannot reach (O(L) forwards per unrank instead of
        Theta(count) tree expansion; exact for iid-step models, mean-field estimate otherwise)."""
        from mixle.enumeration.envelope import AREnvelopeIndex

        return AREnvelopeIndex(self, n_paths=n_paths, seed=seed, budget_bits=budget_bits)

    def top_k(self, k: int) -> list[tuple[tuple, float]]:
        """The ``k`` most probable sequences, exact, by best-first listing (use for small ``k``)."""
        out = []
        for seq, lp in self.enumerator():
            out.append((seq, lp))
            if len(out) >= k:
                break
        return out

    def count(self, min_log_prob: float) -> int | float:
        """How many sequences have ``log_density >= min_log_prob`` -- computed from counts, not listed.

        With ``branch_cap`` set this is the count over the capped sub-support (a sound lower bound);
        :meth:`count_bracket` adds the certified upper bound including the skipped remainder.
        """
        return self.seek_index().count(min_log_prob)

    def count_bracket(self, min_log_prob: float) -> tuple[float, float]:
        """A sound ``[lo, hi]`` bracket on the number of sequences with ``log_density >= min_log_prob``.

        ``lo`` counts the (exactly enumerated) kept sub-support; ``hi`` adds ``dropped_upper`` -- the
        certified bound on completions excluded by ``branch_cap`` (identical to ``lo`` when no cap is set).
        """
        si = self.seek_index()
        lo = float(si.count(min_log_prob))
        return lo, lo + float(si.dropped_upper)

    def unrank(self, i: int) -> tuple[tuple, float]:
        """The ``i``-th most probable sequence (0-based) and its exact log-probability, by random access."""
        return self.seek_index().unrank(i)

    def threshold(self, rank: int) -> float:
        """Log-probability of the ``rank``-th most probable sequence -- the boundary of the top-``rank`` set."""
        return self.seek_index().threshold(rank)

    def mass_above(self, min_log_prob: float) -> tuple[float, float]:
        """A ``(lower, upper)`` bracket on the total probability of sequences with ``log_density >= min_log_prob``.

        Computed from the count histogram alone (no enumeration): each fine bucket of ``c`` sequences
        contributes between ``c * 2**(-hi_bits)`` and ``c * 2**(-lo_bits)``, where the bucket spans
        ``[lo_bits, hi_bits)`` of information. Tighten by raising ``oversample``.
        """
        q = self.seek_index().quantizer
        index = self.seek_index().fine_histogram(q.bits(min_log_prob) + q.bin_width_bits)
        hist = index.hist
        lo = hi = 0.0
        per_bit = q.fine_per_bit()
        # A joint fine bucket is the SUM of per-step floor-quantized buckets, so accumulated rounding can put a
        # sequence's exact information anywhere in [fb / per_bit, (fb + L) / per_bit) bits, where L is the
        # number of steps. Bound L by the deepest sequence the index could hold (the upper bound is tight; the
        # lower bound loosens for long terminating sequences -- sum the head exactly if you need tight mass).
        steps_bound = self._depth if self.terminating else self.max_len
        cutoff = q.fine_bucket(min_log_prob)  # the shared index may be built deeper than this query's bound
        for j, c in enumerate(hist.data):
            fb = hist.base + j
            if fb > cutoff:
                break
            if not c:
                continue
            lo_bits = fb / per_bit  # least information in the bucket -> most probable edge
            hi_bits = (fb + steps_bound) / per_bit  # most information after up to steps_bound roundings
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
