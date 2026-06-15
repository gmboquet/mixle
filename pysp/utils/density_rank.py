"""Rank and cumulative probability of an observation under the descending-probability order.

For an observation ``x``, two natural "where does x sit" queries are:

  - **rank**: how many observations are strictly more probable than ``x`` (its 0-based position
    in the descending-probability enumeration), and
  - **cumulative probability**: the total probability mass of all observations at least as
    probable as ``x`` -- ``G(x) = P_{Y~p}(p(Y) >= p(x)) = sum_{y: p(y) >= p(x)} p(y)``.

Both are exact and cheap for the *head* of the distribution (the most-probable values) via the
existing best-first ``enumerator()``: walk descending until the score drops below ``p(x)``, summing
mass and counting. But for an ``x`` deep in the tail the head is astronomically large, so exact
enumeration is infeasible -- and there a single Monte-Carlo pass is reliable, because ``G(x)`` is
then large (low relative error). Conversely sampling fails for the head (``G(x)`` tiny -> almost no
samples exceed it). The two regimes are exactly complementary, so this module's estimator is a
hybrid: exact enumeration up to a budget, then a sampling fallback. The sampling fallback works for
*any* samplable, density-evaluable model -- mixtures, HMMs, and other non-decomposable families
whose exact count-DP is intractable.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_LN2 = math.log(2.0)


@dataclass
class DensityRankResult:
    """Outcome of a rank / cumulative-probability query.

    Attributes:
        cumulative_probability: ``G(x) = sum_{y: p(y) >= p(x)} p(y)`` (the descending-order CDF at x).
        rank: number of observations strictly more probable than x (0-based position), or ``None``
            when only the sampling estimate was used (sampling estimates mass, not the integer count).
        exact: True when the head enumeration resolved the query exactly; False for a sampling estimate.
        stderr: standard error of ``cumulative_probability`` (0.0 when exact).
        log_prob: ``log p(x)``.
        method: ``"exact-head"``, ``"exact-exhausted"``, or ``"sampling"``.
    """

    cumulative_probability: float
    rank: int | None
    exact: bool
    stderr: float
    log_prob: float
    method: str


def density_rank(
    dist: Any,
    value: Any,
    max_exact: int = 100_000,
    n_samples: int = 20_000,
    seed: int = 0,
    tol: float = 1.0e-9,
) -> DensityRankResult:
    """Rank and cumulative probability of ``value`` under ``dist``'s descending-probability order.

    Strategy:
      1. If ``dist`` supports enumeration, walk the exact descending stream, accumulating the mass of
         every item at least as probable as ``value`` and counting those strictly more probable. If
         the stream drops below ``p(value)`` (or is exhausted) within ``max_exact`` items, the rank
         and cumulative probability are returned EXACTLY.
      2. Otherwise (``value`` is deeper than ``max_exact``, or enumeration is unsupported), estimate
         the cumulative probability by Monte Carlo: ``G_hat = mean_i 1[log p(Y_i) >= log p(value)]``
         with ``Y_i ~ dist``. Reliable here precisely because ``G`` is large in the tail.

    Args:
        dist: A distribution exposing ``log_density`` and ``sampler``; optionally ``enumerator``.
        value: The observation to locate.
        max_exact: Cap on items pulled from the exact enumerator before falling back to sampling.
        n_samples: Monte-Carlo sample count for the fallback.
        seed: Sampler seed for the fallback (reproducible).
        tol: Log-probability tolerance for the ``>=`` comparison (ties).

    Returns:
        DensityRankResult.
    """
    t = float(dist.log_density(value))
    if t == -np.inf:
        return DensityRankResult(0.0, None, True, 0.0, t, "exact-head")

    enumerator = _try_enumerator(dist)
    if enumerator is not None:
        mass = 0.0
        strictly_more = 0
        seen = 0
        for _v, lp in enumerator:
            lp = float(lp)
            if lp < t - tol:
                # Descending order: everything from here on is strictly less probable than value.
                return DensityRankResult(mass, strictly_more, True, 0.0, t, "exact-head")
            mass += math.exp(lp)
            if lp > t + tol:
                strictly_more += 1
            seen += 1
            if seen >= max_exact:
                break
        else:
            # Enumerator exhausted (finite support) without dropping below value's level:
            # value is among the least probable, and the accumulated mass is exact.
            return DensityRankResult(min(1.0, mass), strictly_more, True, 0.0, t, "exact-exhausted")

    # Exact analytic cumulative when the family provides one (e.g. the multivariate Gaussian's
    # chi-square-of-Mahalanobis highest-density-region mass): no enumeration, no sampling.
    exact_cumulative = getattr(dist, "density_cumulative", None)
    if callable(exact_cumulative):
        g = float(exact_cumulative(value))
        return DensityRankResult(min(1.0, max(0.0, g)), None, True, 0.0, t, "exact-analytic")

    # Sampling fallback: estimate G(value) = P(log p(Y) >= t).
    samples = dist.sampler(seed).sample(n_samples)
    hits = 0
    for y in samples:
        if float(dist.log_density(y)) >= t - tol:
            hits += 1
    g = hits / n_samples
    stderr = math.sqrt(max(g * (1.0 - g), 0.0) / n_samples)
    return DensityRankResult(g, None, False, stderr, t, "sampling")


def _try_enumerator(dist: Any):
    """Return ``dist.enumerator()`` if supported, else ``None``."""
    from pysp.stats.compute.pdist import EnumerationError

    enum = getattr(dist, "enumerator", None)
    if enum is None:
        return None
    try:
        return iter(enum())
    except EnumerationError:
        return None


@dataclass
class CountDPRankResult:
    """Approximate rank of an observation from the count DP, for decomposable families.

    Attributes:
        rank: estimated number of observations strictly more probable than the query.
        window_lower: count in buckets safely below the query's smear window (a conservative-ish
            floor; see note on quantization smear below).
        window_upper: ``window_lower`` plus the count inside the smear window.
        log_prob: ``log p(x)``.
        oversample: the quantizer oversample used (higher -> finer bucket -> smaller error).

    The estimate is *quantization-approximate*, not exact: the count DP bins each item by a sum of
    floored per-factor buckets, so an item near the query's probability can land one or two buckets
    to either side of the boundary. The error shrinks as ``oversample`` grows (empirically mean
    well under 1 rank at oversample 64 on small products) and the smear window is resolved exactly
    when small, but a guaranteed integer rank is not promised.
    """

    rank: int
    window_lower: int
    window_upper: int
    log_prob: float
    oversample: int


def _window_bracket(index, fb, smear, t, resolve_max, tol):
    """Smear-window rank bracket around fine bucket ``fb`` for log-prob ``t``.

    Returns ``(window_lower, window_upper, exact_rank)``: the count safely below the smear window,
    that plus the in-window count, and -- when the window holds at most ``resolve_max`` items -- the
    exact count strictly more probable than ``t`` (else ``None``). Shared by rank and seek so the two
    stay consistent.
    """
    hist = index.hist
    lo_b, hi_b = fb - smear, fb + smear
    window_lower = sum(hist.count_at(b) for b in range(hist.base, lo_b))
    window_count = sum(hist.count_at(b) for b in range(lo_b, hi_b + 1))
    window_upper = window_lower + window_count
    if window_count <= resolve_max:
        strictly_more = sum(
            1
            for b in range(lo_b, hi_b + 1)
            for off in range(hist.count_at(b))
            if float(index.get_in_bucket(b, off)[1]) > t + tol
        )
        return window_lower, window_upper, window_lower + strictly_more
    return window_lower, window_upper, None


def count_dp_rank(
    dist: Any,
    value: Any,
    oversample: int = 64,
    bin_width_bits: float = 1.0,
    smear: int | None = None,
    resolve_max: int = 8192,
    tol: float = 1.0e-9,
) -> CountDPRankResult:
    """Approximate rank of ``value`` via the structural count DP -- for decomposable families.

    The count DP convolves the per-factor log-probability histograms, so its histogram **is** the
    distribution of the total log-probability over the whole (astronomically large) support. The
    rank of ``value`` is then the cumulative count of more-probable buckets -- a prefix sum, no
    enumeration -- so it works for arbitrarily *deep* ranks that head enumeration and sampling
    cannot reach. Items within a few buckets of the query may straddle the boundary (quantization
    smear from the floored per-factor buckets), so a smear window around the query's bucket is
    resolved exactly (unranked and compared) when it holds at most ``resolve_max`` items; the result
    is an estimate whose error shrinks with ``oversample``.

    For the NON-decomposable marginal families (mixture, HMM) the count DP bins by the tropical
    (dominant-path) cost and over-counts, so this returns a *tropical* rank, not the true-marginal
    rank -- use :func:`density_rank` (head enumeration + sampling) for those.
    """
    from pysp.stats.compute.pdist import EnumerationError
    from pysp.utils.quantization.core import Quantizer

    t = float(dist.log_density(value))
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    fb = q.fine_bucket(t)
    smear = oversample if smear is None else int(smear)  # ~one bit of boundary uncertainty
    index, _truncated = dist.quantized_count_index(q, max_fine_bucket=fb + smear)
    if index is None:
        raise EnumerationError(dist, reason="no structural count index for rank")

    wl, wu, exact_rank = _window_bracket(index, fb, smear, t, resolve_max, tol)
    rank = exact_rank if exact_rank is not None else (wl + wu) // 2
    return CountDPRankResult(rank, wl, wu, t, oversample)


def _locate_bucket(hist, target_index):
    """Return ``(fine_bucket, offset)`` of the value at descending-probability ``target_index``.

    Buckets are scanned in increasing order (= descending probability). Returns ``None`` if
    ``target_index`` is at or beyond the histogram's total count.
    """
    cum = 0
    for b in range(hist.base, hist.base + len(hist.data)):
        c = hist.count_at(b)
        if cum + c > target_index:
            return b, target_index - cum
        cum += c
    return None


@dataclass
class CountDPSeekResult:
    """The observation at an arbitrary descending-probability index -- the inverse of a rank query.

    Attributes:
        value: the observation at (approximately) descending-probability ``index``.
        log_prob: ``log p(value)``.
        index: the requested 0-based index.
        rank_lower, rank_upper: the TRUE rank of ``value`` is bracketed in ``[rank_lower, rank_upper]``
            by the smear window. A *tight* bracket means the model is in the separated / near-exact
            regime here, so the seek is trustworthy; a *wide* one means many near-ties around ``value``.
        exact: the smear window was resolved exactly (then ``rank_lower == rank_upper`` is the true
            rank). For decomposable families (composite/sequence/markov) the count index is exact, so
            seek is exact up to quantization. For mixtures/HMMs the index is the TROPICAL projection
            (dominant component/path), so the bracket is the tropical error envelope -- a guaranteed
            bound only when ``smear`` covers the <= log2(K)-bit tropical displacement; otherwise it is
            the approximate tropical bracket (raise ``smear`` for rigor; see :func:`count_dp_rank`).
        oversample: the quantizer oversample used (higher -> finer buckets -> tighter bracket).
    """

    value: Any
    log_prob: float
    index: int
    rank_lower: int
    rank_upper: int
    exact: bool
    oversample: int


@dataclass
class CountDPTopPResult:
    """How many of the most-probable outcomes cover probability mass ``target`` -- the nucleus SIZE.

    The structural / at-depth counterpart of ``DistributionEnumerator.top_p``: where that materializes
    the nucleus (fine when it is small), this reports its size for decomposable families WITHOUT
    enumerating it, so it scales to huge supports where the nucleus itself is too large to list.

    Attributes:
        size_lower, size_upper: provable bracket on the nucleus size -- the number of most-probable
            outcomes whose summed mass first reaches ``target``. Both bounds hold regardless of the
            within-bucket ordering: ``size_upper`` includes whole probability buckets until the mass
            covers ``target`` (a valid covering set, so the true nucleus is no larger), and
            ``size_lower`` caps every item in bucket ``b`` at its maximum possible probability
            ``2**(-b * bits_per_bucket)`` (an over-estimate of coverage, so fewer items provably
            cannot reach ``target``).
        covered_mass: exact mass of the ``size_upper`` whole-bucket cover (``>= target`` unless truncated).
        log_prob_threshold: approximate log-prob at the cover boundary.
        target: the requested cumulative-probability target ``p``.
        truncated: the depth bound was hit before the mass reached ``target`` (then ``size_upper`` is a
            floor on the true size, not a cover).
        oversample: the quantizer oversample used (higher -> finer buckets -> tighter bracket).
    """

    size_lower: int
    size_upper: int
    covered_mass: float
    log_prob_threshold: float
    target: float
    truncated: bool
    oversample: int


def count_dp_seek(
    dist: Any,
    index: int,
    oversample: int = 64,
    bin_width_bits: float = 1.0,
    smear: int | None = None,
    resolve_max: int = 8192,
    tol: float = 1.0e-9,
    max_fine_bucket_cap: int = 1 << 30,
) -> CountDPSeekResult:
    """Seek the observation at descending-probability ``index`` -- the inverse of :func:`count_dp_rank`.

    Walks the structural count histogram in ascending-bucket (= descending-probability) order until
    the cumulative count passes ``index``, then unranks within that bucket. No prefix enumeration, so
    arbitrary *deep* indices are reachable directly. The depth bound is grown geometrically until the
    index (plus a smear margin) is covered or the support is exhausted.

    Returns the value together with a provable bracket ``[rank_lower, rank_upper]`` on its true rank
    (the smear window): a tight bracket certifies the seek is in the separated / near-exact regime.
    For decomposable families this is exact up to quantization; for mixtures/HMMs it seeks into the
    tropical (dominant-component/path) projection and the bracket is that projection's error envelope.
    """
    from pysp.stats.compute.pdist import EnumerationError
    from pysp.utils.quantization.core import Quantizer

    if index < 0:
        raise IndexError("index must be non-negative")
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    smear = oversample if smear is None else int(smear)

    # Deepen the depth bound until the located bucket's smear window is fully built, or the
    # structural count stops growing (support exhausted). The per-family ``truncated`` flag is not
    # reliable across all families, so drive the loop off coverage + total() growth instead.
    mfb = max(2 * smear, 64)
    idx = None
    prev_total = -1
    while True:
        idx, _truncated = dist.quantized_count_index(q, max_fine_bucket=mfb)
        if idx is None:
            raise EnumerationError(dist, reason="no structural count index for seek")
        total = idx.total()
        located = _locate_bucket(idx.hist, index)
        if located is not None:
            built_top = idx.hist.base + len(idx.hist.data) - 1
            if located[0] + smear <= built_top:
                break  # value found and its smear window is fully built
        if (total == prev_total and total > 0) or mfb >= max_fine_bucket_cap:
            break  # deepening adds nothing further (support exhausted) or runaway guard hit
        prev_total = total
        mfb *= 2

    located = _locate_bucket(idx.hist, index)
    if located is None:
        raise IndexError("index %d is beyond the structural support count %d" % (index, idx.total()))
    b_star, offset = located
    value, lp = idx.get_in_bucket(b_star, offset)
    lp = float(lp)
    fb_v = q.fine_bucket(lp)
    wl, wu, exact_rank = _window_bracket(idx, fb_v, smear, lp, resolve_max, tol)
    if exact_rank is not None:
        return CountDPSeekResult(value, lp, index, exact_rank, exact_rank, True, oversample)
    return CountDPSeekResult(value, lp, index, wl, wu, False, oversample)


def _mass_histogram(dist, quantizer, max_fine_bucket):
    """Probability MASS per fine bucket of bits -- ``{bucket: sum of p(y) over y in that bucket}``.

    Unlike the count histogram (which would need ``count x 2^-bits`` and is biased O(#factors/R)
    because structural bits under-estimate true bits), this carries the EXACT summed probability, so a
    bulk prefix sum is the exact cumulative mass of all strictly-more-probable buckets. Mass multiplies
    and bits add, so it convolves exactly like the count histogram: composites convolve their fields,
    sequences pool the per-length L-fold self-convolution shifted/scaled by the length term, leaves
    sum probabilities over their own support. Used by :func:`cumulative_probability`.
    """
    from pysp.stats.combinator.composite import CompositeDistribution
    from pysp.stats.combinator.record import RecordDistribution
    from pysp.stats.combinator.sequence import SequenceDistribution
    from pysp.stats.compute.pdist import EnumerationError

    def convolve(a, b):
        out: dict[int, float] = {}
        for ba, ma in a.items():
            for bb, mb in b.items():
                k = ba + bb
                if k <= max_fine_bucket:
                    out[k] = out.get(k, 0.0) + ma * mb
        return out

    # Composite and Record are both products of independent fields, so the joint mass histogram is
    # the convolution of the per-field histograms (mass multiplies, bits add) regardless of whether
    # fields are addressed by position or by name.
    if isinstance(dist, (CompositeDistribution, RecordDistribution)):
        joint = {0: 1.0}
        for f in range(dist.count):
            joint = convolve(joint, _mass_histogram(dist.dists[f], quantizer, max_fine_bucket))
        return joint

    if isinstance(dist, SequenceDistribution):
        from pysp.stats.compute.pdist import child_enumerator

        if dist.null_len_dist:
            raise EnumerationError(dist, reason="no length distribution is modeled")
        elem = _mass_histogram(dist.dist, quantizer, max_fine_bucket)
        total: dict[int, float] = {}
        powers = {0: {0: 1.0}}  # L-fold self-convolution of the element mass histogram
        max_len = 0
        for length, lp_len in child_enumerator(dist.len_dist, "SequenceDistribution.len_dist"):
            if not isinstance(length, (int, np.integer)) or length < 0 or lp_len == -np.inf:
                continue
            shift = quantizer.fine_bucket(float(lp_len))
            if shift > max_fine_bucket:
                break
            length = int(length)
            while max_len < length:
                powers[max_len + 1] = convolve(powers[max_len], elem)
                max_len += 1
            plen = math.exp(float(lp_len))
            for b, m in powers[length].items():
                k = b + shift
                if k <= max_fine_bucket:
                    total[k] = total.get(k, 0.0) + m * plen
        return total

    # Leaf: sum probabilities over the distribution's own support.
    enum = getattr(dist, "enumerator", None)
    if enum is None:
        raise EnumerationError(dist, reason="leaf does not support enumeration for mass histogram")
    hist: dict[int, float] = {}
    for v, lp in enum():
        lp = float(lp)
        if lp == -np.inf:
            continue
        b = quantizer.fine_bucket(lp)
        if b > max_fine_bucket:
            break
        hist[b] = hist.get(b, 0.0) + math.exp(lp)
    return hist


def cumulative_probability(dist, value, oversample: int = 64, bin_width_bits: float = 1.0, smear: int | None = None):
    """Exact cumulative probability ``G(x) = sum_{y: p(y) >= p(x)} p(y)`` for decomposable families.

    Structural and at arbitrary depth (no enumeration, no sampling): the bulk mass of all buckets
    strictly below the query's smear band comes from the exact :func:`_mass_histogram` prefix, and the
    band itself is resolved item-by-item (true ``log_density``) via the count index. Because each
    bucket's mass is the EXACT sum of its items' probabilities and the band absorbs the floored-bucket
    smear (within ``#factors`` buckets), the result is exact up to floating-point roundoff -- verified
    to 1e-16 on a 12-factor product, where the count-times-representative-probability shortcut returns
    G > 1. Deterministic, so it complements :func:`density_rank` (whose deep path is Monte-Carlo).
    """
    from pysp.utils.quantization.core import Quantizer

    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    t = float(dist.log_density(value))
    bx = dist.structural_fine_bucket(value, q)
    smear = oversample if smear is None else int(smear)
    mass = _mass_histogram(dist, q, bx + smear)
    bulk = sum(m for b, m in mass.items() if b < bx - smear)
    index, _truncated = dist.quantized_count_index(q, max_fine_bucket=bx + smear)
    band = 0.0
    for b in range(bx - smear, bx + smear + 1):
        for off in range(index.hist.count_at(b)):
            _v, lp = index.get_in_bucket(b, off)
            if float(lp) >= t - 1.0e-9:
                band += math.exp(float(lp))
    return min(1.0, bulk + band)


def count_dp_top_p(
    dist: Any,
    p: float,
    oversample: int = 64,
    bin_width_bits: float = 1.0,
    tol: float = 1.0e-9,
    max_fine_bucket_cap: int = 1 << 30,
) -> CountDPTopPResult:
    """Nucleus SIZE: how many most-probable outcomes cover mass ``p``, for decomposable families.

    The structural counterpart of ``dist.enumerator().top_p(p)`` -- it returns the *size* of the
    minimal high-probability set (and its boundary), computed from the exact per-bucket mass
    (:func:`_mass_histogram`) and per-bucket counts (the count index) WITHOUT enumerating the nucleus,
    so it works when the nucleus is far too large to list. The size is returned as a provable bracket
    ``[size_lower, size_upper]`` (see :class:`CountDPTopPResult`); the bracket is tight when the mass
    is concentrated and widens with quantization smear (raise ``oversample`` to tighten).

    For mixtures/HMMs the mass histogram has no exact decomposition -- use the enumerator's ``top_p``
    (exact for a small nucleus) there instead.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    from pysp.stats.compute.pdist import EnumerationError
    from pysp.utils.quantization.core import Quantizer

    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    bits_per_bucket = bin_width_bits / oversample
    if p <= 0.0:
        return CountDPTopPResult(0, 0, 0.0, float("inf"), p, False, oversample)

    # Deepen the depth bound until the whole-bucket cover reaches p, or the covered mass stalls.
    mfb = 64
    prev_mass = -1.0
    mass: dict[int, float] = {}
    while True:
        mass = _mass_histogram(dist, q, mfb)
        total = sum(mass.values())
        if total >= p - tol or (abs(total - prev_mass) <= tol and total > 0.0) or mfb >= max_fine_bucket_cap:
            break
        prev_mass = total
        mfb *= 2

    index, _truncated = dist.quantized_count_index(q, max_fine_bucket=mfb)
    if index is None:
        raise EnumerationError(dist, reason="no structural count index for top_p")
    counts = index.hist
    buckets = sorted(mass)

    # Upper bound: include whole probability buckets (descending prob = ascending bucket) until the
    # exact cumulative mass reaches p. The set of all items in those buckets covers p, so the true
    # nucleus is no larger than its size.
    cum_mass = 0.0
    cum_count = 0
    size_upper = 0
    boundary_bucket = buckets[-1] if buckets else 0
    covered_mass = 0.0
    truncated = True
    for b in buckets:
        cum_mass += mass[b]
        cum_count += counts.count_at(b)
        if cum_mass >= p - tol:
            size_upper = cum_count
            boundary_bucket = b
            covered_mass = cum_mass
            truncated = False
            break
    else:
        size_upper = cum_count
        covered_mass = cum_mass

    # Lower bound: cap each item in bucket b at its maximum possible probability 2**(-b*bits_per_bucket)
    # (a structural bucket is a sum of floored child buckets <= the true floored bits, so the true
    # probability never exceeds this cap). If even these caps cannot reach p with c items, the true
    # nucleus needs more than c, so this is a provable floor.
    cap_mass = 0.0
    cap_count = 0
    size_lower = cum_count  # if caps never reach p (only when truncated), the floor is everything seen
    for b in buckets:
        c = counts.count_at(b)
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
        int(size_lower), int(size_upper), float(covered_mass), log_prob_threshold, p, truncated, oversample
    )


def _joint_bucket_histogram(components, quantizer, max_fine_bucket):
    """Joint K-dim count histogram of (bucket(log p_1(y)), ..., bucket(log p_K(y))) over the support.

    Built structurally for homogeneous (same-structure) components, with NO enumeration of the joint
    support: composites convolve the per-field joint histograms (bucket tuples add, counts multiply);
    leaves enumerate their own (small) support and key by the K-tuple of per-component buckets.
    Returns ``{(b_1, ..., b_K): count}``. Raises EnumerationError for component structures not
    handled here (only Composite and atomic/enumerable leaves are supported).
    """
    from pysp.stats.combinator.composite import CompositeDistribution
    from pysp.stats.compute.pdist import EnumerationError
    from pysp.utils.enumeration import freeze

    head = components[0]
    if isinstance(head, CompositeDistribution):
        arity = head.count
        joint = {(0,) * len(components): 1}
        for f in range(arity):
            field = _joint_bucket_histogram([c.dists[f] for c in components], quantizer, max_fine_bucket)
            nxt: dict[tuple[int, ...], int] = {}
            for ka, ca in joint.items():
                for kb, cb in field.items():
                    key = tuple(ka[j] + kb[j] for j in range(len(ka)))
                    if max(key) <= max_fine_bucket:
                        nxt[key] = nxt.get(key, 0) + ca * cb
            joint = nxt
        return joint
    # Leaf: enumerate the union of component supports, key by the per-component bucket tuple.
    values: dict[Any, Any] = {}
    for comp in components:
        try:
            enum = comp.enumerator()
        except EnumerationError as e:
            raise EnumerationError(comp, reason="component does not support cross-rank: %s" % e.reason) from None
        for v, _lp in enum:
            values.setdefault(freeze(v), v)
    hist: dict[tuple[int, ...], int] = {}
    for v in values.values():
        key = tuple(quantizer.fine_bucket(float(c.log_density(v))) for c in components)
        if all(b <= max_fine_bucket for b in key):
            hist[key] = hist.get(key, 0) + 1
    return hist


def mixture_cross_rank(mixture, value, oversample: int = 64, bin_width_bits: float = 1.0, depth_bits: float = 64.0):
    """True-marginal rank of ``value`` under a homogeneous mixture, at arbitrary depth.

    ``count_dp_rank`` on a mixture gives only the TROPICAL (dominant-component) rank -- it bins by the
    best single component, so a value built from several components is badly mis-ranked. This computes
    the true rank against the actual marginal ``p = sum_k w_k p_k`` by building the JOINT K-dimensional
    count histogram of the per-component log-prob buckets (structurally, no enumeration of the joint
    support -- see :func:`_joint_bucket_histogram`) and counting joint bins whose representative
    marginal probability exceeds ``p(value)``.

    Quantization-approximate: a joint bin's marginal probability is evaluated at the bucket midpoints,
    so bins straddling the threshold may be mis-counted; the error shrinks as ``oversample`` grows.
    Cost is EXPONENTIAL in the number of components K (the histogram is K-dimensional), so this is for
    SMALL-K mixtures (a few components) of same-structured decomposable components; it needs no
    enumeration, so it scales to deep ranks. For non-mixtures use :func:`count_dp_rank`; for the head
    of any model use :func:`density_rank`.
    """
    from pysp.utils.quantization.core import Quantizer

    comps = [c for c, w in zip(mixture.components, mixture.w, strict=False) if w > 0.0]
    log_w = [float(lw) for lw, w in zip(mixture.log_w, mixture.w, strict=False) if w > 0.0]
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    max_fb = int(math.ceil(depth_bits * oversample / bin_width_bits))
    joint = _joint_bucket_histogram(comps, q, max_fb)

    t = float(mixture.log_density(value))
    px = math.exp(t)
    bits_per_bucket = bin_width_bits / oversample
    rank = 0
    for key, cnt in joint.items():
        # representative marginal probability of this joint bin (per-component bucket midpoints)
        p = sum(math.exp(lw) * 2.0 ** (-(key[j] + 0.5) * bits_per_bucket) for j, lw in enumerate(log_w))
        if p > px * (1.0 + 1.0e-9):
            rank += cnt
    return rank
