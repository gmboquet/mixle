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
# Float-roundoff slack for validating a step's raw scores at the model boundary (MXR-080-0221): a
# log-probability must be <= 0 (_SIGN_TOL) and the kept per-step probabilities must sum to ~1 (_NORM_TOL) --
# both generous enough for float32-derived log_softmax output, tight enough to catch a genuinely broken model.
_SIGN_TOL = 1.0e-9
_NORM_TOL = 1.0e-4
# Bounded restarts before a terminating model's ancestral sampler gives up (MXR-080-0222): each restart is a
# fresh, independent ancestral draw (proper rejection sampling for "terminates within max_depth"), so this
# only matters for a model/max_depth combination where termination in-cap is itself rare.
_MAX_RESAMPLE_ATTEMPTS = 1000


def _raise_index(fb: int, off: int) -> tuple[Any, float]:
    raise IndexError("empty autoregressive count index")


def _require_positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    """Validate ``value`` is an EXACT positive integer for a size/cap constructor parameter (MXR-080-0224).

    Accepts a Python/numpy integer, or a float that is exactly integer-valued (e.g. ``2.0``) -- never
    silently truncated with ``int()`` (a fractional ``2.9`` used to become ``2`` with no error at all).
    ``allow_none`` lets ``None`` through unchanged (this file's spelling of "no explicit cap", used by
    ``max_len``/``branch_cap``). Raises ``ValueError`` naming the parameter otherwise.
    """
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a positive integer (got %r)." % (name, value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value) or value != math.floor(value):
            raise ValueError("%s must be a whole number, not a fractional value (got %r)." % (name, value))
    ivalue = int(value)
    if ivalue < 1:
        raise ValueError("%s must be a positive integer (got %d)." % (name, ivalue))
    return ivalue


def _require_exact_cardinality(actual: int, expected: int, source: str) -> None:
    """Raise a clear error if a callback returned the wrong number of results (MXR-080-0223).

    Positional pairing (``zip``, index assignment) between a batch of requests and a batch of results is
    only safe when the counts match exactly: a short result would otherwise silently drop or misalign the
    tail, and a long one would silently ignore the extra -- either way masking what is likely a caller-side
    bug, and risking writing a result into the shared forward cache under the WRONG prefix/sequence key.
    """
    if actual != expected:
        raise ValueError(
            "%s returned %d result(s) for %d input(s); exact cardinality is required (a mismatch would "
            "silently drop, ignore, or misalign results)." % (source, actual, expected)
        )


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
    """Ancestral sampler over the model -- token by token from ``next_logprobs`` (for the rank tail fallback).

    For a terminating model, one ancestral draw can exhaust the ``max_depth`` safety cap without ever
    emitting ``eos``. That truncated prefix is NOT in the model's declared support -- see
    :meth:`AutoregressiveEnumerable._in_declared_support`, the identical contract :meth:`~
    AutoregressiveEnumerable.log_density` and the count index enforce -- so it is never handed back as a
    sample (MXR-080-0222). Instead the draw is restarted from scratch: since a restart is a fresh,
    independent ancestral draw from the SAME generative process, accepting only eos-terminated attempts is
    proper rejection sampling for "terminates within max_depth", not an ad hoc patch. Restarts are bounded
    by :data:`_MAX_RESAMPLE_ATTEMPTS`; if every attempt is exhausted without termination, :meth:`sample`
    raises rather than ever returning a sample the scorer/index would declare impossible.
    """

    def __init__(self, model: AutoregressiveEnumerable, seed: int | None) -> None:
        self._model = model
        self._rng = np.random.RandomState(seed)

    def _draw_once(self) -> tuple:
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
        return prefix

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        n = 1 if size is None else int(size)
        out = []
        for _ in range(n):
            prefix = self._draw_once()
            attempts = 1
            while self._model.terminating and not self._model._in_declared_support(prefix):
                if attempts >= _MAX_RESAMPLE_ATTEMPTS:
                    raise RuntimeError(
                        "ancestral sampling could not draw an eos-terminated sequence within max_depth=%d "
                        "after %d attempts -- the model's termination probability is too low for this depth "
                        "cap (raise max_depth, or check the model's eos mass)." % (self._model._depth, attempts)
                    )
                prefix = self._draw_once()
                attempts += 1
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
            still dropped). An exact positive integer (or ``None``) -- a fractional value raises rather than
            silently truncating.
        eos: end-of-sequence token. When given, the model is terminating: only sequences ending in ``eos`` are
            in the support.
        max_depth: safety bound on recursion depth for a terminating model (the probability budget is the real
            bound). Raise it if a tight model legitimately produces very long sequences. An exact positive
            integer -- a fractional or non-positive value raises.
        bin_width_bits, oversample: quantization resolution of the count index (finer = exacter ordering,
            more memory). The defaults match the distribution count-DP. Ordering from ``unrank``/``slice``
            (via :meth:`seek_index`) is exact between fine buckets (width ``bin_width_bits / oversample``
            bits) but NOT guaranteed within one -- see :meth:`~mixle.enumeration.seek_index.SeekIndex.slice`.
            Sequences whose ``log_density`` differs by less than one bucket's width can surface in either
            relative order; this is most visible when several near-tied candidates cluster at the head of
            a small/short-sequence model. Raise ``oversample`` or lower ``bin_width_bits`` to shrink it.
            ``oversample`` is an exact positive integer -- a fractional or non-positive value raises.
        batch_next_logprobs: optional ``batch_next_logprobs([prefix, ...]) -> [result, ...]`` scoring many
            prefixes in one (padded) forward. When given, the count index warms its forward cache breadth-first
            in ``batch_size`` chunks -- the large speed-up for transformers, where one-at-a-time forwards
            dominate (e.g. distilGPT-2 length-2 to rank 1e5: ~25 s one-at-a-time -> ~1 s batched). Each
            chunk's results must exactly match the chunk's own size -- a short or long result raises rather
            than silently dropping/ignoring the mismatch (MXR-080-0223).
        batch_size: prefixes per batched forward. An exact positive integer -- a fractional or non-positive
            value raises.
        count_mode: how counts are carried past the int64-exact regime (budgets over ~2**60 sequences).
            ``'auto'`` (default) switches the numpy fast path to float64 there -- **approximate** counts
            (exact below 2**53, ~1e-16 relative error per pooling beyond) at full numpy speed. ``'exact'``
            preserves arbitrary-precision counts by falling back to the slow Python recursion. ``'float'``
            forces float64 everywhere.
        branch_cap: recurse into only the top-``branch_cap`` in-budget tokens per prefix -- the certified
            approximation for wide (LLM-sized) vocabularies, shrinking the tree by ~V/cap per level. The
            skipped remainder is soundly bounded (``count_bracket``/``dropped_upper``: a skipped subtree
            with ``r`` remaining budget bits holds at most ``2**r`` completions); enumeration covers the
            sub-support of sequences whose every token is among its context's top-``branch_cap``. An exact
            positive integer (or ``None``) -- a fractional value raises rather than silently truncating.
        batch_score_sequences: optional teacher-forcing scorer ``[sequence, ...] -> array of total log-probs``
            -- ONE forward per sequence (all positions score in parallel) instead of one forward per token.
            Used by :meth:`score_sequences` and, when a sequence's prefixes are not already cached, by
            :meth:`log_density`; the substrate for draft-rescored (speculative) enumeration. Must return
            exactly one score per (in-support) sequence requested -- a short or long result raises
            (MXR-080-0223).
        all_position_logprobs: optional ``sequence -> [next_logprobs result for seq[:d], d in 0..len-1]`` --
            one forward yields the full next-token distribution at EVERY position; harvested into the
            forward cache by :meth:`harvest`. Makes corpus-calibrated envelopes ~L-times cheaper. Must
            return exactly one result per position -- a short result raises rather than silently leaving
            some positions uncached (MXR-080-0223).

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
        # Exact positive integers, not silently int()-truncated (MXR-080-0224): a fractional max_len=2.9 used
        # to become max_len=2 with no error, and max_depth had no validation at all (a negative value was
        # silently accepted and stored, later producing a degenerate always-empty/truncated index).
        max_len = _require_positive_int(max_len, "max_len", allow_none=True)
        max_depth = _require_positive_int(max_depth, "max_depth")
        oversample = _require_positive_int(oversample, "oversample")
        batch_size = _require_positive_int(batch_size, "batch_size")
        branch_cap = _require_positive_int(branch_cap, "branch_cap", allow_none=True)
        if count_mode not in ("auto", "exact", "float"):
            raise ValueError("count_mode must be 'auto', 'exact', or 'float'")
        self.next_logprobs = next_logprobs
        self.eos = eos
        self.terminating = eos is not None
        self.max_len = max_len
        self.max_depth = max_depth
        # depth bound passed to the recursion: a fixed-length model completes at max_len; a terminating model
        # completes only at eos and uses max_len (if given) or max_depth purely as a safety cap.
        self._depth = self.max_len if not self.terminating else (self.max_len or self.max_depth)
        self.bin_width_bits = float(bin_width_bits)
        self.oversample = oversample
        self.batch_next_logprobs = batch_next_logprobs
        self.batch_size = batch_size
        self.count_mode = count_mode
        self.branch_cap = branch_cap
        self.batch_score_sequences = batch_score_sequences
        self.all_position_logprobs = all_position_logprobs
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # prefix -> (tokens, log_probs), desc by lp
        self._fast: bool | None = None
        self._seek = None  # cached SeekIndex: built once, reused by unrank/count/threshold/mass_above

    # -- the model oracle, descending by log-prob and memoized (one forward per prefix) -------------------
    def _parse_steps(self, raw: Any) -> tuple[np.ndarray, np.ndarray]:
        """Parse one step's raw ``next_logprobs`` output into ``(tokens, log_probs)``, descending by log-prob.

        THE model boundary: every step -- fetched on demand, prefetched in a batch, or harvested from an
        all-position forward -- is parsed here exactly once before entering the shared cache, so this is
        where a malformed step table is caught once and for all (MXR-080-0221): mismatched ``(tokens,
        log_probs)`` array lengths, a NaN score, a score ``> 0`` (an impossible log-probability -- this also
        catches ``+inf``), or a duplicate token are all REJECTED outright rather than silently laundered.
        A duplicate token is particularly dangerous silently: the tree-recursive count index and the
        ancestral sampler would count/sample it as two distinct branches, while :meth:`log_density`'s
        ``dict(steps)`` walk would keep only the last -- the same nominal sequence scored differently
        depending on which code path touched it. ``-inf`` remains the one legitimate non-finite score (an
        explicit "this token has zero probability") and is dropped, not rejected, exactly as before. Once
        cleaned, the kept probabilities must sum to ~1 -- this is meant to be an actual categorical
        distribution over the next token (e.g. a ``log_softmax``), not an arbitrary/truncated subset.
        """
        # Accept the fast ``(tokens, log_probs)`` numpy form or a ``[(token, log_prob), ...]`` list.
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], np.ndarray):
            tokens, lps = np.asarray(raw[0]), np.asarray(raw[1], dtype=float)
            if tokens.shape != lps.shape:
                raise ValueError(
                    "next_logprobs returned mismatched (tokens, log_probs) arrays: shapes %r vs %r."
                    % (tokens.shape, lps.shape)
                )
        else:
            items = list(raw)
            tokens = np.array([t for t, _ in items])
            lps = np.array([float(lp) for _, lp in items], dtype=float)

        if np.isnan(lps).any():
            raise ValueError("next_logprobs returned a NaN log-probability score; the model output is malformed.")
        over_zero = lps > _SIGN_TOL
        if over_zero.any():
            raise ValueError(
                "next_logprobs returned invalid log-probability score(s) > 0 (%r) -- log-probabilities must "
                "be <= 0 (this also catches +inf)." % (lps[over_zero][:5].tolist(),)
            )
        tok_list = tokens.tolist()
        if len(set(tok_list)) != len(tok_list):
            counts: dict[Any, int] = {}
            for t in tok_list:
                counts[t] = counts.get(t, 0) + 1
            dupes = sorted(t for t, c in counts.items() if c > 1)
            raise ValueError(
                "next_logprobs returned duplicate token(s) %r in one step's table -- each token must appear "
                "at most once." % (dupes,)
            )

        # -inf is the one legitimate non-finite score (an explicit "this token has zero probability"); drop
        # it here, same as before -- unlike NaN/positive scores above, it is not a sign of a broken model.
        finite = np.isfinite(lps)
        if not finite.all():
            tokens, lps = tokens[finite], lps[finite]

        if lps.size:
            total = float(np.sum(np.exp(lps)))
            if abs(total - 1.0) > _NORM_TOL:
                raise ValueError(
                    "next_logprobs returned a non-normalized distribution (kept probabilities sum to %.6g, "
                    "expected ~1.0) -- check the model's log_softmax / normalization." % total
                )

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

        Each chunk's result count is validated against the chunk's own size before anything is cached
        (MXR-080-0223): ``zip(chunk, results)`` alone would silently drop the tail of ``chunk`` on a short
        result (leaving those prefixes to the lazy per-prefix path with no warning) or silently ignore the
        extra on a long one -- either way masking a caller-side ``batch_next_logprobs`` bug instead of
        surfacing it.
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
                raw_results = list(self.batch_next_logprobs(chunk))
                _require_exact_cardinality(len(raw_results), len(chunk), "batch_next_logprobs")
                for pfx, raw in zip(chunk, raw_results):
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

    def _in_declared_support(self, seq: tuple) -> bool:
        """Whether ``seq`` satisfies this model's DECLARED STRUCTURAL support -- independent of whether each
        token is actually locally available given its prefix (that per-step check still runs separately, in
        :meth:`log_density`'s walk). This centralizes exactly what :func:`autoregressive_count_index`'s
        recursion already enforces structurally (a terminating branch that hits the depth bound without
        emitting ``eos`` contributes zero mass, not partial credit; an ``eos``-terminated branch stops right
        there, so nothing can follow it) -- so a candidate sequence handed directly to :meth:`log_density` /
        :meth:`score_sequences` is held to exactly the same contract as the count index built from the same
        model, instead of only the weaker per-step check.

        Fixed-length (``eos`` is None): exactly ``self._depth`` (== ``max_len``) tokens.
        Terminating (``eos`` set): non-empty, ends in ``eos``, with ``eos`` nowhere before the last position
        (so nothing -- including a repeated ``eos`` -- follows the first one), and total length not
        exceeding ``self._depth`` (== ``max_len`` when given as a hard cap, else the ``max_depth`` bound).
        """
        if not self.terminating:
            return len(seq) == self._depth
        if not seq or seq[-1] != self.eos or self.eos in seq[:-1]:
            return False
        return len(seq) <= self._depth

    def log_density(self, sequence: Iterable[Any]) -> float:
        """Exact total log-probability of a sequence -- ``-inf`` if it is off the model's DECLARED support
        (wrong length, missing/misplaced ``eos``, or past the depth cap; see :meth:`_in_declared_support`)
        or if any token along the way is locally unavailable given its prefix.

        When a ``batch_score_sequences`` scorer is configured and any of the sequence's prefixes is not
        already cached, the score comes from ONE teacher-forcing forward instead of one forward per token.
        The scorer's returned cardinality is validated (exactly one score for the one sequence requested)
        before it is trusted -- see :meth:`score_sequences`' own note (MXR-080-0223).
        """
        seq = tuple(sequence)
        if not self._in_declared_support(seq):
            return _NEG_INF
        if self.batch_score_sequences is not None and any(seq[:d] not in self._cache for d in range(len(seq))):
            scored = np.asarray(self.batch_score_sequences([seq]), dtype=float).reshape(-1)
            _require_exact_cardinality(scored.size, 1, "batch_score_sequences")
            return float(scored[0])
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

        With ``batch_score_sequences`` this is one call scoring every structurally in-support sequence (one
        forward per sequence, all positions in parallel); a sequence off the declared support (see
        :meth:`_in_declared_support`) scores ``-inf`` directly, without ever reaching the external scorer.
        The scorer's returned cardinality is then validated -- exactly one score per in-support sequence
        sent to it -- before it is trusted (MXR-080-0223); a short or long result raises rather than
        silently misaligning or dropping scores. Otherwise falls back to per-sequence :meth:`log_density`
        over the cached walk, which applies the identical structural check. The rescoring primitive for
        draft-based (speculative) enumeration.
        """
        seqs = [tuple(s) for s in sequences]
        if not seqs:
            return np.zeros(0, dtype=float)
        if self.batch_score_sequences is not None:
            out = np.full(len(seqs), _NEG_INF, dtype=float)
            idx = [i for i, s in enumerate(seqs) if self._in_declared_support(s)]
            if idx:
                scored = np.asarray(self.batch_score_sequences([seqs[i] for i in idx]), dtype=float).reshape(-1)
                _require_exact_cardinality(scored.size, len(idx), "batch_score_sequences")
                out[idx] = scored
            return out
        return np.array([self.log_density(s) for s in seqs], dtype=float)

    def harvest(self, sequence: Iterable[Any]) -> None:
        """Cache the next-token distribution at every prefix of ``sequence`` from one forward.

        Requires ``all_position_logprobs``; a no-op without it. Feeding typical sequences (a corpus, a
        provider's fast generations) through this warms the same memo cache the count index and the
        envelope read -- L cache entries per model call.

        The result must carry exactly one entry per position (MXR-080-0223): a short result used to be
        silently accepted by slicing to ``results[:len(seq)]``, leaving the un-covered tail of prefixes
        uncached with no signal that coverage was incomplete. That is validated -- and any cache mutation
        withheld -- before anything is written.
        """
        if self.all_position_logprobs is None:
            return
        seq = tuple(sequence)
        need = [d for d in range(len(seq)) if seq[:d] not in self._cache]
        if not need:
            return
        results = list(self.all_position_logprobs(seq))
        _require_exact_cardinality(len(results), len(seq), "all_position_logprobs")
        for d, raw in enumerate(results):
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

        ``max_depth_bits`` is reconciled on every call, not just honored on the first (MXR-080-0225): a
        later call requesting a GENUINELY DEEPER cap raises the cached index's own ceiling in place --
        ``SeekIndex`` already deepens its build lazily on demand, so widening the ceiling is enough; nothing
        already built is discarded or rebuilt. A later call requesting a smaller-or-equal cap is a no-op:
        the cached index already satisfies it, and lowering an already-built ceiling would only forbid
        further deepening the index could otherwise still do, for no benefit. Previously the FIRST call's
        cap was silently permanent -- a later, larger cap returned the same object with the old, too-small
        ceiling still in effect, so query results silently depended on unrelated call order.
        """
        if self._seek is None:
            from mixle.enumeration.seek_index import SeekIndex

            self._seek = SeekIndex(
                self,
                bin_width_bits=self.bin_width_bits,
                oversample=self.oversample,
                max_depth_bits=max_depth_bits,
            )
        elif float(max_depth_bits) > self._seek.max_depth_bits:
            self._seek.max_depth_bits = float(max_depth_bits)
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
        """The ``k`` most probable sequences, exact, by best-first listing (use for small ``k``).

        ``k <= 0`` returns ``[]`` without starting the best-first listing at all (MXR-080-0224 -- checked
        before any iteration, not after the loop body has already appended one item and only THEN noticed
        the bound was already satisfied, which used to make ``top_k(0)`` return one item instead of none).
        """
        out: list[tuple[tuple, float]] = []
        if k <= 0:
            return out
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
        """The ``i``-th most probable sequence (0-based, QUANTIZED order) and its exact log-probability.

        Random access through the count index, so the ordering carries the index's quantization
        (:meth:`~mixle.enumeration.seek_index.SeekIndex.unrank`): exact **between** fine buckets
        (width ``bin_width_bits / oversample`` bits), unspecified **within** one -- two sequences
        whose log-probabilities differ by less than a bucket's width can come back in either relative
        order, so ``rank(unrank(i))`` can differ from ``i`` for near-tied neighbours. The returned
        log-probability is always exact regardless. Shrink the ambiguity window by raising
        ``oversample`` or lowering ``bin_width_bits``; for a strictly-sorted head use :meth:`top_k`
        (exact best-first), and for a sequence's exact rank use :meth:`rank`.
        """
        return self.seek_index().unrank(i)

    def threshold(self, rank: int) -> float:
        """Log-probability of the ``rank``-th most probable sequence -- the boundary of the top-``rank`` set.

        Read off :meth:`unrank`, so it inherits the same fine-bucket quantization of the ordering
        (exact between buckets, unspecified within one); the returned log-probability itself is exact.
        """
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
        """Size of the minimal ``>= p``-mass set (:class:`~mixle.enumeration.density_rank.CountDPTopPResult`),
        without materializing it.

        Deliberately does NOT delegate to the generic
        :func:`~mixle.enumeration.density_rank.count_dp_top_p`: that function assumes its exact per-item
        mass histogram (bucketed by the floor of the EXACT total log-density) and its count index share one
        bucket numbering, which holds for Composite/Record/Sequence but not here -- :meth:`quantized_count_index`
        buckets a sequence by the SUM of its per-step floor-quantized buckets, and floor(a) + floor(b) <=
        floor(a + b), so the structural bucket is systematically <= the exact one (same discrepancy
        :meth:`mass_above` documents). Looking counts up at the exact-mass bucket key lands on the wrong
        (usually empty) slot in the count histogram -- confirmed: it returns a nucleus size of 0 against a
        brute-force ground truth of 5. This mirrors :meth:`mass_above`'s fix instead: every quantity is
        derived from the SAME structural count histogram, with each bucket's per-item probability bounded
        by ``[2**(-(b + steps_bound) * bpb), 2**(-b * bpb)]`` (identical bound, same derivation), so the
        result is a provable bracket rather than an exact mass -- the same contract shape
        ``count_dp_top_p`` documents, just derived from bounds instead of a wrongly-keyed exact histogram.
        """
        from mixle.enumeration.density_rank import CountDPTopPResult

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        if p <= 0.0:
            return CountDPTopPResult(0, 0, 0.0, float("inf"), p, False, self.oversample)

        tol = 1.0e-9
        si = self.seek_index()
        q = si.quantizer
        bits_per_bucket = q.bin_width_bits / q.oversample
        # steps_bound is a GLOBAL worst case (mirrors mass_above), so it can be far looser than the actual
        # sequences a given crossing needs -- for a terminating model with a generous max_depth safety cap,
        # forcing the pessimistic bound below to provably cross p can require capturing near-100% of the
        # true mass (see mass_above's own docstring: "the lower bound loosens for long terminating
        # sequences"), which is exponentially expensive to search for. So deepening below is driven by the
        # cheap OPTIMISTIC bound (matching count_dp_top_p's own fast deepening loop) -- size_upper is then
        # only certified (truncated=False) when the pessimistic bound independently confirms it at that
        # depth; otherwise it is honestly reported as a floor, not a cover (matching CountDPTopPResult's
        # own documented truncated=True contract), rather than searching indefinitely for a certificate.
        steps_bound = self._depth if self.terminating else self.max_len

        depth_bits = max(q.bin_width_bits, 64.0 * bits_per_bucket)
        while True:
            hist = si.fine_histogram(depth_bits).hist
            hi_total = sum(
                hist.data[j] * 2.0 ** (-(hist.base + j) * bits_per_bucket)
                for j in range(len(hist.data))
                if hist.data[j]
            )
            if hi_total >= p - tol or not si.truncated or depth_bits >= si.max_depth_bits:
                break
            depth_bits = min(depth_bits * 2.0, si.max_depth_bits)

        # Upper bound on size: include whole structural buckets, most-probable first, until even the
        # pessimistic (least-probable-edge) cumulative mass provably reaches p -- so the true nucleus is
        # no larger. covered_mass is that same sound lower bound on the true covered mass, reported
        # honestly rather than as a false-precision exact value. If the pessimistic sum never gets there
        # within the built depth (only possible once the optimistic bound above already confirmed the true
        # mass exists), size_upper falls back to everything seen -- a floor, not a certified cover, exactly
        # exactly like count_dp_top_p's own "never crosses" fallback, and truncated stays True to say so.
        cum_count = 0
        cum_lo = 0.0
        size_upper = 0
        boundary_bucket = hist.base + len(hist.data) - 1 if hist.data else 0
        covered_mass = 0.0
        truncated = True
        for j in range(len(hist.data)):
            c = hist.data[j]
            if not c:
                continue
            b = hist.base + j
            cum_count += c
            cum_lo += c * 2.0 ** (-(b + steps_bound) * bits_per_bucket)
            if cum_lo >= p - tol:
                size_upper = cum_count
                boundary_bucket = b
                covered_mass = cum_lo
                truncated = False
                break
        else:
            size_upper = cum_count
            covered_mass = cum_lo

        # Lower bound on size: cap every item in bucket b at its maximum possible probability
        # 2**(-b*bpb) (mirrors count_dp_top_p's own, already bucket-correct, size_lower derivation).
        cap_mass = 0.0
        cap_count = 0
        size_lower = cum_count
        for j in range(len(hist.data)):
            c = hist.data[j]
            if not c:
                continue
            b = hist.base + j
            cap_here = 2.0 ** (-b * bits_per_bucket)
            if cap_mass + c * cap_here >= p - tol:
                residual = p - cap_mass
                need = math.ceil(residual / cap_here - tol)
                size_lower = cap_count + max(0, min(need, c))
                break
            cap_mass += c * cap_here
            cap_count += c

        log_prob_threshold = -float(boundary_bucket) * bits_per_bucket * math.log(2.0)
        return CountDPTopPResult(
            int(size_lower), int(size_upper), float(covered_mass), log_prob_threshold, p, truncated, self.oversample
        )
