"""Rank and cumulative probability of an observation under the descending-probability order.

For an observation ``x``, two natural "where does x sit" queries are:

  - **rank**: how many observations are strictly more probable than ``x`` (its 0-based position
    in the descending-probability enumeration), and
  - **cumulative probability**: the total probability mass of all observations at least as
    probable as ``x`` -- ``G(x) = P_{Y~p}(p(Y) >= p(x)) = sum_{y: p(y) >= p(x)} p(y)``.

Both are exact and efficient for the *head* of the distribution (the most-probable values) via the
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

# How far a value is allowed to sit outside its mathematically valid range ([0, 1] for a probability
# mass or CDF) before it is treated as a real defect (duplicate counting, a normalization bug, an
# unaccounted truncation) rather than ordinary floating-point roundoff. Deliberately generous relative
# to float64 epsilon (~2.2e-16): a sum over many terms (a large composite/sequence support) can
# accumulate roundoff well beyond one ULP, but a genuine defect is typically orders of magnitude larger
# still. Shared so every roundoff-vs-defect judgment in this module uses one consistent budget.
_ROUNDOFF_TOL = 1.0e-6

# The tie-tolerance convention `count_dp_rank`/`count_dp_seek`/`marginal_seek` already document ("must
# match the 1e-12 convention the true rank is defined by"): two log-probabilities within this of each
# other are a tie. `density_rank` and `cumulative_probability` now default to the SAME convention
# (previously 1e-9), so "exact" means the same thing everywhere in this module by default.
_EXACT_TIE_TOL = 1.0e-12


@dataclass
class DensityRankResult:
    """Outcome of a rank / cumulative-probability query.

    Attributes:
        cumulative_probability: ``G(x) = sum_{y: p(y) >= p(x)} p(y)`` (the descending-order CDF at x).
        rank: number of observations strictly more probable than x (0-based position). Exact when the
            head enumeration resolved it; in ``"sampling"`` mode it is the rounded unbiased Monte-Carlo
            estimate (see ``rank_stderr``); ``None`` when ``log p(x) = -inf``, an exact-analytic CDF (no
            count) was used, or the sampling estimate itself overflowed (see ``rank_stderr``, which is
            still finite -- or ``inf`` -- even when ``rank`` is ``None``).
        exact: True when the head enumeration resolved the query exactly WITH RESPECT TO the ``tol``-
            equivalence relation ties are defined by (see ``density_rank``'s ``tol`` parameter); False
            for a sampling estimate.
        stderr: standard error of ``cumulative_probability`` (0.0 when exact).
        log_prob: ``log p(x)``.
        method: ``"exact-head"``, ``"exact-exhausted"``, ``"exact-analytic"``, or ``"sampling"``.
        rank_stderr: standard error of the Monte-Carlo ``rank`` estimate (0.0 when exact). Large
            relative to ``rank`` exactly in the deep-tail / high-entropy band where exact marginal
            rank is provably hard -- treat the estimate as unreliable there.
    """

    cumulative_probability: float
    rank: int | None
    exact: bool
    stderr: float
    log_prob: float
    method: str
    rank_stderr: float = 0.0


def density_rank(
    dist: Any,
    value: Any,
    max_exact: int = 100_000,
    n_samples: int = 20_000,
    seed: int = 0,
    tol: float = _EXACT_TIE_TOL,
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
        n_samples: Monte-Carlo sample count for the fallback; must be a positive integer (validated
            eagerly, even on a call that never reaches the sampling path -- a bad ``n_samples`` is a
            caller bug regardless of whether this particular query happens to exercise it).
        seed: Sampler seed for the fallback (reproducible).
        tol: Log-probability tolerance for the ``>=``/``>`` comparisons (ties). Defaults to this
            module's ``1e-12`` "true rank" convention (also used by ``count_dp_rank``/``count_dp_seek``/
            ``marginal_seek``). ``exact=True`` is exact WITH RESPECT TO this tolerance's equivalence
            relation: two scores within ``tol`` of each other are treated as tied. Widening ``tol``
            trades a stricter mathematical comparison for robustness against floating-point jitter in
            ``log_density`` -- know that a wide ``tol`` can change which values compare as tied.

    Returns:
        DensityRankResult.
    """
    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
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
        # A CDF must lie in [0, 1]; clamping unconditionally would silently paper over a broken
        # `density_cumulative` implementation and still call the (wrong) result exact. Only a
        # roundoff-scale excursion is clamped-and-blessed; anything larger is a real defect and is
        # surfaced instead of hidden.
        if g < -_ROUNDOFF_TOL or g > 1.0 + _ROUNDOFF_TOL:
            raise ValueError(
                "%s.density_cumulative(value) returned %.17g, which is outside [0, 1] by more than the "
                "%.1e roundoff budget -- this is a defect in density_cumulative, not floating-point "
                "roundoff, so it is not silently clamped" % (type(dist).__name__, g, _ROUNDOFF_TOL)
            )
        return DensityRankResult(min(1.0, max(0.0, g)), None, True, 0.0, t, "exact-analytic")

    # Sampling fallback. Two unbiased Monte-Carlo estimators from the SAME draws Y_i ~ dist:
    #   * mass  G(x) = P[log p(Y) >= t]                      ~ mean_i 1[lp_i >= t]
    #   * rank  #{y: p(y) > p(x)} = E_Y[1[p(Y) > p(x)] / p(Y)] ~ mean_i 1[lp_i > t] * exp(-lp_i)
    # The rank identity holds for ANY discrete samplable model (finite or infinite support): the
    # importance weight 1/p(Y) under Y ~ p telescopes the indicator set to its cardinality. This is the
    # count analogue of the mass estimate -- the marginal rank that the exact head and the tropical
    # count index cannot give for non-decomposable families (mixtures/HMMs), where exact marginal rank
    # is provably hard. Its variance is driven by the least-probable counted point (~1/p(x)), so the
    # estimate is reliable in the body and noisy in the deep tail (reported via rank_stderr).
    samples = dist.sampler(seed).sample(n_samples)
    n_drawn = len(samples)
    if n_drawn != n_samples:
        raise ValueError(
            "sampler produced %d samples for n_samples=%d -- sampler cardinality does not match the "
            "request" % (n_drawn, n_samples)
        )
    try:
        # vectorized: one seq_log_density pass instead of n_samples per-sample log_density calls
        lp = np.asarray(dist.seq_log_density(dist.dist_to_encoder().seq_encode(samples)), dtype=float)
    except Exception:  # noqa: BLE001
        lp = np.asarray([float(dist.log_density(y)) for y in samples], dtype=float)
    if lp.shape != (n_samples,):
        raise ValueError(
            "encoded log-density scores have shape %r, expected (%d,) -- one score per drawn sample"
            % (lp.shape, n_samples)
        )
    g = float(np.count_nonzero(lp >= t - tol)) / n_samples
    stderr = math.sqrt(max(g * (1.0 - g), 0.0) / n_samples)

    # rank weights: 1/p(y) for the strictly-more-probable draws (matching the exact path's `> t`), else 0.
    # Computed in LOG-SPACE via a stable logsumexp rather than materializing exp(-lp) per sample: for an
    # ordinary deep-tail query (t a few hundred nats or more -- unremarkable for a long sequence or a
    # many-factor composite) some strictly-more-probable draws already push -lp past float64's ~709.78
    # exp overflow threshold, long before anything about the samples is actually malformed.
    from mixle.utils.vector import log_sum

    neg_lp = -lp[lp > t + tol]
    if neg_lp.size == 0:
        rank_mean, rank_var = 0.0, 0.0
    else:
        log_n = math.log(n_samples)
        log_mean = float(log_sum(neg_lp)) - log_n
        log_second_moment = float(log_sum(2.0 * neg_lp)) - log_n
        # exp() overflows float64 past ~709.78; a mean/second-moment that large is a genuine property
        # of an astronomically deep rank, not a numerical artifact of the accumulation -- report it as
        # inf (a legitimate value for these fields) instead of letting int(round(...)) raise below.
        rank_mean = math.exp(log_mean) if log_mean < 709.0 else float("inf")
        second_moment = math.exp(log_second_moment) if log_second_moment < 709.0 else float("inf")
        rank_var = max(second_moment - rank_mean * rank_mean, 0.0) if math.isfinite(second_moment) else float("inf")
    # sqrt(rank_var / (n-1)) is algebraically weights.std(ddof=1)/sqrt(n) (the previous convention),
    # just accumulated in log-space above instead of building `weights` in linear scale first.
    rank_stderr = math.sqrt(rank_var / (n_samples - 1)) if n_samples > 1 and math.isfinite(rank_var) else float("inf")
    rank_est = int(round(rank_mean)) if math.isfinite(rank_mean) else None
    return DensityRankResult(g, rank_est, False, stderr, t, "sampling", rank_stderr=rank_stderr)


def _try_enumerator(dist: Any):
    """Return ``dist.enumerator()`` if supported, else ``None``."""
    from mixle.stats.compute.pdist import EnumerationError

    enum = getattr(dist, "enumerator", None)
    if enum is None:
        return None
    try:
        return iter(enum())
    except EnumerationError:
        return None


@dataclass
class TruncatedSumBound:
    """Bounds on a descending-probability sum truncated after the top ``num_enumerated`` items.

    Attributes:
        num_enumerated: how many top items were enumerated (``< k`` means the support was exhausted).
        enumerated_mass: exact summed probability of the enumerated items (a *lower* bound on the total).
        last_log_prob: ``log p`` of the smallest enumerated item; every un-enumerated item is ``<= `` this.
        support_size: the distribution's support cardinality (``None`` if infinite/unknown), or the number
            enumerated when the support was exhausted.
        exhausted: the enumerator ran dry within ``k`` -- the enumerated items ARE the whole support, so
            the tail is exactly zero and the bounds are exact.
        tail_upper_bound: provable upper bound on the un-enumerated mass: ``(support_size - num_enumerated)
            * exp(last_log_prob)`` (0 when exhausted; ``None`` when the support size is unknown).
        total_upper_bound: ``enumerated_mass + tail_upper_bound`` -- an upper bound on the full sum
            (``<= 1`` for a normalized model; bounds the partition function for an unnormalized one).
    """

    num_enumerated: int
    enumerated_mass: float
    last_log_prob: float
    support_size: int | None
    exhausted: bool
    tail_upper_bound: float | None
    total_upper_bound: float | None


def truncated_sum_bound(dist: Any, k: int) -> TruncatedSumBound:
    """Bound a distribution's descending-probability sum by truncating after the top ``k`` items.

    Enumerates the ``k`` most probable outcomes (descending), so any un-enumerated outcome has
    probability ``<= p_k`` (the smallest enumerated). With the support cardinality ``N =
    dist.support_size()`` the un-enumerated mass is then bounded by ``(N - k) * p_k`` -- a finite,
    low-overhead upper bound on the truncated tail that uses only ``k`` evaluations and the support size. If
    the enumerator is exhausted within ``k`` the bounds are exact (the tail is zero). Requires an
    enumerable family; raises EnumerationError otherwise.

    This is the truncation-based distribution upper bound: e.g. it certifies how much mass a top-``k``
    summary misses, or upper-bounds an unnormalized model's partition function.
    """
    from mixle.stats.compute.pdist import EnumerationError

    if k < 0:
        raise ValueError("k must be non-negative.")
    enumerator = _try_enumerator(dist)
    if enumerator is None:
        raise EnumerationError(dist, reason="truncated_sum_bound requires an enumerable support")

    if k == 0:
        # An explicit zero-prefix bound WITHOUT consuming the stream: `k=0` means "enumerate nothing",
        # not "enumerate one item and stop" (the previous behavior -- the loop below always pulled at
        # least one value before checking `n >= k`). Every item's log-probability is <= 0.0 (a
        # probability is <= 1), so 0.0 is the tightest universally-valid `last_log_prob` when nothing
        # has actually been enumerated, giving the loosest-but-still-sound `exp(0.0) == 1.0` per-item
        # cap on the (entirely un-enumerated) tail.
        support_size = dist.support_size()
        exhausted = support_size == 0
        tail_upper = None if support_size is None else float(support_size)
        return TruncatedSumBound(
            num_enumerated=0,
            enumerated_mass=0.0,
            last_log_prob=0.0,
            support_size=support_size,
            exhausted=exhausted,
            tail_upper_bound=tail_upper,
            total_upper_bound=tail_upper,
        )

    mass = 0.0
    last_lp = float("inf")
    n = 0
    exhausted = False
    for value, lp in enumerator:
        mass += math.exp(float(lp))
        last_lp = float(lp)
        n += 1
        if n >= k:
            # Peek whether the support continues beyond k.
            if next(enumerator, None) is None:
                exhausted = True
            break
    else:
        exhausted = True

    support_size = dist.support_size() if not exhausted else n
    if exhausted:
        tail_upper: float | None = 0.0
    elif support_size is None:
        tail_upper = None
    else:
        tail_upper = max(0, support_size - n) * math.exp(last_lp)
    total_upper = None if tail_upper is None else (mass + tail_upper)
    return TruncatedSumBound(
        num_enumerated=n,
        enumerated_mass=mass,
        last_log_prob=(last_lp if n > 0 else float("-inf")),
        support_size=support_size,
        exhausted=exhausted,
        tail_upper_bound=tail_upper,
        total_upper_bound=total_upper,
    )


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


def _window_bracket(index, fb, smear, t, resolve_max, tol, built_top, exhaustive):
    """Smear-window rank bracket around fine bucket ``fb`` for log-prob ``t``.

    Returns ``(window_lower, window_upper, exact_rank, complete)``. ``window_lower`` is the count in
    buckets safely below the smear window; ``window_upper`` adds the in-window count; ``exact_rank`` is
    the count strictly more probable than ``t``, computed only when the window is CERTIFIED complete
    and holds at most ``resolve_max`` items (else ``None``).

    ``complete`` says whether ``[fb - smear, fb + smear]`` is provably covered by ``index``: either
    ``exhaustive`` is True (the distribution's OWN exhaustion certificate -- ``quantized_count_index``'s
    ``truncated`` flag was False for the build that produced ``index``, so the histogram holds the
    ENTIRE support, not just whatever fits below ``built_top``), or the window's upper edge ``fb +
    smear`` does not exceed ``built_top``.

    ``built_top`` MUST be the ``max_fine_bucket`` depth ``index`` was actually BUILT WITH (what the
    caller passed to ``quantized_count_index``) -- NOT ``index.hist.base + len(index.hist.data) - 1``
    (the deepest OCCUPIED bucket). Those differ whenever the true distribution has zero mass in a run
    of buckets right below the depth cap: ``CountHistogram`` trims trailing zero entries, so the
    occupied range can be strictly shallower than what was actually built and verified empty. The DP
    contract guarantees every bucket ``<= max_fine_bucket`` is correctly represented, zero or not, so
    using the occupied range there would misreport a genuinely-complete window as unbuilt.

    When ``complete`` is False, ``window_upper`` (and any resolved ``exact_rank``) must NOT be treated
    as a guaranteed bound: buckets beyond ``built_top`` are simply ABSENT from the histogram (not
    confirmed empty), so the window may in truth hold more than reported. ``window_lower`` remains a
    valid proven lower bound regardless -- every bucket it sums is strictly below the window's lower
    edge, hence always within the built range whenever the query bucket ``fb`` itself was locatable in
    ``index`` at all. Shared by rank and seek so the two stay consistent.
    """
    hist = index.hist
    lo_b, hi_b = fb - smear, fb + smear
    window_lower = sum(hist.count_at(b) for b in range(hist.base, lo_b))
    window_count = sum(hist.count_at(b) for b in range(lo_b, hi_b + 1))
    window_upper = window_lower + window_count
    complete = exhaustive or hi_b <= built_top
    if complete and window_count <= resolve_max:
        strictly_more = sum(
            1
            for b in range(lo_b, hi_b + 1)
            for off in range(hist.count_at(b))
            if float(index.get_in_bucket(b, off)[1]) > t + tol
        )
        return window_lower, window_upper, window_lower + strictly_more, True
    return window_lower, window_upper, None, complete


def count_dp_rank(
    dist: Any,
    value: Any,
    oversample: int = 64,
    bin_width_bits: float = 1.0,
    smear: int | None = None,
    resolve_max: int = 8192,
    tol: float = 1.0e-12,  # rank tie threshold; must match the 1e-12 convention the true rank is defined by
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
    from mixle.enumeration.quantization.core import Quantizer
    from mixle.stats.compute.pdist import EnumerationError

    t = float(dist.log_density(value))
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    fb = q.fine_bucket(t)
    smear = oversample if smear is None else int(smear)  # ~one bit of boundary uncertainty
    index, truncated = dist.quantized_count_index(q, max_fine_bucket=fb + smear)
    if index is None:
        raise EnumerationError(dist, reason="no structural count index for rank")

    # The REQUESTED depth, not the deepest occupied bucket: the DP contract guarantees every bucket
    # <= max_fine_bucket is correctly represented (whether zero or nonzero), so a trailing run of
    # genuinely-empty buckets right below the cap must not read as "incomplete" (see _window_bracket).
    built_top = fb + smear
    wl, wu, exact_rank, _complete = _window_bracket(index, fb, smear, t, resolve_max, tol, built_top, not truncated)
    rank = exact_rank if exact_rank is not None else (wl + wu) // 2
    return CountDPRankResult(rank=rank, window_lower=wl, window_upper=wu, log_prob=t, oversample=oversample)


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
        truncated: True when the search hit ``max_fine_bucket_cap`` before ``value``'s smear window
            could be fully built AND before the distribution's own exhaustion certificate could rule
            out more (deeper, possibly gapped) support -- then ``exact`` is always False, and only
            ``rank_lower`` remains a PROVEN bound; ``rank_upper`` is a best-effort figure from a
            partially-built window, not a guarantee (buckets beyond the built depth are simply absent
            from the histogram, not confirmed empty). False (the normal case) means ``rank_lower``/
            ``rank_upper`` are both certified, whether or not they collapsed to an exact point.
    """

    value: Any
    log_prob: float
    index: int
    rank_lower: int
    rank_upper: int
    exact: bool
    oversample: int
    truncated: bool = False


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
    tol: float = 1.0e-12,  # rank tie threshold; must match the 1e-12 convention the true rank is defined by
    max_fine_bucket_cap: int = 1 << 30,
) -> CountDPSeekResult:
    """Seek the observation at descending-probability ``index`` -- the inverse of :func:`count_dp_rank`.

    Walks the structural count histogram in ascending-bucket (= descending-probability) order until
    the cumulative count passes ``index``, then unranks within that bucket. No prefix enumeration, so
    arbitrary *deep* indices are reachable directly. The depth bound is grown geometrically until the
    index (plus a smear margin) is covered or the distribution's own exhaustion certificate proves no
    more support exists -- see :meth:`~mixle.stats.compute.pdist.ProbabilityDistribution.quantized_count_index`'s
    ``truncated`` return value, which is False exactly when its histogram holds the ENTIRE support. A
    plain "the cumulative count stopped growing between two depths" plateau is deliberately NOT used to
    conclude exhaustion: a discrete model can have zero support across a wide run of probability buckets
    and resume having support below it (e.g. disjoint support clusters, or a component reachable only
    through a tiny-weight path), and a plateau there is indistinguishable from genuine exhaustion by
    count alone.

    Returns the value together with a provable bracket ``[rank_lower, rank_upper]`` on its true rank
    (the smear window): a tight bracket certifies the seek is in the separated / near-exact regime.
    For decomposable families this is exact up to quantization; for mixtures/HMMs it seeks into the
    tropical (dominant-component/path) projection and the bracket is that projection's error envelope.
    If the search reaches ``max_fine_bucket_cap`` before the window is fully built and before
    exhaustion is certified, the result is returned with ``truncated=True`` (``exact=False``, and only
    ``rank_lower`` remains a proven bound) rather than silently passed off as a certified bracket. If
    ``index`` itself is never located within the cap and exhaustion is not certified either, raises
    ``EnumerationError`` -- NOT ``IndexError``, since ``IndexError`` would falsely claim the index is
    proven out of range, when in fact the search simply ran out of budget; a genuine out-of-range
    ``index`` (proven via the exhaustion certificate) still raises ``IndexError``, unchanged.
    """
    from mixle.enumeration.quantization.core import Quantizer
    from mixle.stats.compute.pdist import EnumerationError

    if index < 0:
        raise IndexError("index must be non-negative")
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    smear = oversample if smear is None else int(smear)

    # Deepen the depth bound until the located bucket's smear window is fully built, or
    # `quantized_count_index`'s own `truncated` flag proves the ENTIRE support is already captured
    # (not a count plateau -- see the docstring above for why a plateau alone is not proof).
    mfb = max(2 * smear, 64)
    idx = None
    located = None
    truncated = True
    while True:
        idx, truncated = dist.quantized_count_index(q, max_fine_bucket=mfb)
        if idx is None:
            raise EnumerationError(dist, reason="no structural count index for seek")
        located = _locate_bucket(idx.hist, index)
        # The REQUESTED depth (mfb), not the deepest occupied bucket: the DP contract guarantees every
        # bucket <= mfb is correctly represented (whether zero or nonzero), so a trailing run of
        # genuinely-empty buckets right below the cap must not read as "not yet built" (see
        # _window_bracket's docstring for the full argument).
        if located is not None and located[0] + smear <= mfb:
            break  # value found and its smear window is fully built -- certified
        if not truncated:
            break  # exhaustion certificate: the histogram already holds the entire support
        if mfb >= max_fine_bucket_cap:
            break  # resource guard -- give up, but do NOT claim exhaustion (more support may be deeper)
        mfb *= 2

    built_top = mfb
    if located is None:
        if not truncated:
            raise IndexError("index %d is beyond the structural support count %d" % (index, idx.total()))
        raise EnumerationError(
            dist,
            reason=(
                "descending-probability index %d was not reached within max_fine_bucket_cap=%d bits; "
                "the support is not proven exhausted (more values may exist at lower probability) -- "
                "raise max_fine_bucket_cap to search deeper" % (index, mfb)
            ),
        )
    b_star, offset = located
    value, lp = idx.get_in_bucket(b_star, offset)
    lp = float(lp)
    fb_v = q.fine_bucket(lp)
    wl, wu, exact_rank, complete = _window_bracket(idx, fb_v, smear, lp, resolve_max, tol, built_top, not truncated)
    # The bucket/bracket math above uses the structural cost ``lp`` the index is built on (the tropical
    # dominant-path/component cost for HMM/mixture). The reported ``log_prob`` must be the documented
    # ``log p(value)``: for decomposable families this equals ``lp``, but for the tropical projection the
    # true marginal is a logsumexp over paths/components, so recompute it from the value.
    log_prob = float(dist.log_density(value))
    if not complete:
        # The cap was hit before value's smear window was fully built, and exhaustion was never
        # certified either: `wl` is still a sound proven lower bound (see _window_bracket), but `wu`
        # is a best-effort figure from a partial window, not a guaranteed upper bound -- mark the
        # whole bracket uncertified instead of letting a small visible fragment pass as a proof.
        return CountDPSeekResult(
            value=value,
            log_prob=log_prob,
            index=index,
            rank_lower=wl,
            rank_upper=wu,
            exact=False,
            oversample=oversample,
            truncated=True,
        )
    if exact_rank is not None:
        return CountDPSeekResult(
            value=value,
            log_prob=log_prob,
            index=index,
            rank_lower=exact_rank,
            rank_upper=exact_rank,
            exact=True,
            oversample=oversample,
            truncated=False,
        )
    return CountDPSeekResult(
        value=value,
        log_prob=log_prob,
        index=index,
        rank_lower=wl,
        rank_upper=wu,
        exact=False,
        oversample=oversample,
        truncated=False,
    )


@dataclass
class MarginalSeekResult:
    """The value at a descending index, with a *guaranteed* bracket on its TRUE marginal rank.

    :func:`count_dp_seek` brackets only the *tropical* rank for a marginal family (mixture): its count
    index bins by the dominant-component cost ``M(x)`` and over-counts values shared by several
    components, so neither the cost gap nor the multiplicity is accounted for. This result closes both
    gaps soundly:

      * **cost gap** -- the window is widened by the family's ``tropical_displacement_bits`` (``log2(K)``
        for a ``K``-component mixture, since ``M(x) <= log p(x) <= M(x) + log K``), so every value below
        the window is provably more probable than ``value`` and every value above it provably less.
      * **multiplicity gap** -- a shared value is counted at most ``K`` times, so the count strictly
        below the window over-states the distinct rank by at most ``K``; dividing by ``K`` restores a
        sound lower bound.

    Hence ``[true_rank_lower, true_rank_upper]`` provably contains ``#{u : log p(u) > log p(value)}``.
    Two regimes pin it exactly (``exact``): a decomposable / provably-disjoint family (no displacement,
    no over-count -> the structural count IS the distinct rank), or a shallow index whose whole prefix
    fits the resolve budget (unranked and de-duplicated against the true ``log_density``). Otherwise the
    bracket is the certified, provable envelope -- the #P-hard core (de-duplicating an arbitrarily deep
    overlapping prefix) is exactly what cannot be done efficiently.

    Attributes:
        value: the observation at tropical descending ``index``.
        log_prob: exact ``log p(value)`` (the true marginal, re-evaluated -- not the tropical cost).
        index: the requested 0-based (tropical) index.
        true_rank_lower, true_rank_upper: guaranteed bracket on ``value``'s TRUE marginal rank.
        exact: the bracket collapsed to the exact true rank (then the two bounds are equal).
        oversample: the quantizer oversample used (higher -> finer buckets -> tighter bracket).
        truncated: True when the search hit ``max_fine_bucket_cap`` before ``value``'s (displacement-
            widened) rank window could be fully built AND before the distribution's own exhaustion
            certificate ruled out more support -- then ``exact`` is always False, and only
            ``true_rank_lower`` remains a PROVEN bound; ``true_rank_upper`` is a best-effort figure from
            a partially-built window, not a guarantee. False (the normal case) means both bounds are
            certified, whether or not they collapsed to an exact point.
    """

    value: Any
    log_prob: float
    index: int
    true_rank_lower: int
    true_rank_upper: int
    exact: bool
    oversample: int
    truncated: bool = False

    @property
    def semantics(self):
        """``DensitySemantics.EXACT`` when the rank is pinned, else ``ESTIMATE`` (a provable bracket)."""
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.EXACT if self.exact else DensitySemantics.ESTIMATE


def marginal_seek(
    dist: Any,
    index: int,
    oversample: int = 64,
    bin_width_bits: float = 1.0,
    resolve_max: int = 8192,
    tol: float = 1.0e-12,
    max_fine_bucket_cap: int = 1 << 30,
) -> MarginalSeekResult:
    """Seek descending ``index`` with a GUARANTEED bracket on the value's true marginal rank.

    For a decomposable family this matches :func:`count_dp_seek` (zero displacement, exact count). For
    a marginal family (mixture) the structural count index is the *tropical* projection; this widens the
    rank window by the family's :meth:`~mixle.stats.compute.pdist.ProbabilityDistribution.tropical_displacement_bits`
    so the bracket provably bounds the TRUE marginal rank, divides the below-window count by the
    component multiplicity to stay sound against over-counting, and -- when the whole prefix is small or
    the family is decomposable/disjoint -- resolves the window against the true ``log_density`` to pin
    the exact rank in ``O(window)`` rather than ``O(index)``.

    Raises ``EnumerationError`` when no structural count index exists (e.g. a continuous-component
    mixture) -- both exactly like :func:`count_dp_seek`, whose depth-deepening loop this shares, INCLUDING
    its exhaustion certificate: deepening is driven by ``quantized_count_index``'s own ``truncated`` flag
    (False exactly when the histogram already holds the ENTIRE support), never by a plateau in the total
    count. A large probability *gap* (a value separated from the rest by many bits, e.g. an outcome that
    only a ~1e-6-weight component can emit) no longer causes a false-exhaustion conclusion: the loop
    deepens straight through the gap, since a plateau in that region is not by itself proof that nothing
    lies beyond it. ``IndexError`` is raised only once exhaustion is actually certified and ``index`` is
    still unreached -- a proof, not a guess. If the search instead gives up at ``max_fine_bucket_cap``
    with ``index`` still unreached and exhaustion uncertified, it raises ``EnumerationError`` (budget
    exhausted, existence unproven) rather than a falsely-certain ``IndexError``; if ``index`` WAS reached
    but its rank window could not be fully verified within the cap, it returns a result with
    ``truncated=True`` instead of a certified bracket. The returned ``value`` sits at the *tropical*
    index; for the value at a true descending index use :func:`mixle.enumeration.best_first.sound_top_k`.
    """
    from mixle.enumeration.quantization.core import Quantizer
    from mixle.enumeration.streams import freeze
    from mixle.stats.compute.pdist import EnumerationError

    if index < 0:
        raise IndexError("index must be non-negative")
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)

    # Smear covers (a) the ~1-bit quantization boundary and (b) the tropical cost gap log2(K) between
    # the structural count cost and the true marginal log-density. With smear >= the displacement,
    # every value whose tropical bucket is strictly below the window is provably MORE probable than the
    # located value, and every value strictly above it is provably LESS probable (see the class doc).
    disp_bits = 0.0
    disp_fn = getattr(dist, "tropical_displacement_bits", None)
    if callable(disp_fn):
        disp_bits = float(disp_fn())
    mult_cap = max(1, round(2.0**disp_bits))  # worst-case copies of a shared value (K)
    smear = oversample + math.ceil(disp_bits * oversample / bin_width_bits)

    # Deepen the depth bound until the located bucket's widened window is fully built, or
    # `quantized_count_index`'s own `truncated` flag proves the ENTIRE support is already captured --
    # the identical exhaustion-certificate loop as count_dp_seek (see its docstring for why a count
    # plateau is not proof: a discrete model can have zero support across a wide run of buckets and
    # resume having support below it, so "the count stopped growing" is never by itself conclusive).
    mfb = max(2 * smear, 64)
    idx = None
    located = None
    truncated = True
    while True:
        idx, truncated = dist.quantized_count_index(q, max_fine_bucket=mfb)
        if idx is None:
            raise EnumerationError(dist, reason="no structural count index for seek")
        located = _locate_bucket(idx.hist, index)
        # The REQUESTED depth (mfb), not the deepest occupied bucket -- see count_dp_seek's identical
        # note: the DP contract guarantees every bucket <= mfb is correctly represented (whether zero
        # or nonzero), so a trailing run of genuinely-empty buckets right below the cap must not read
        # as "not yet built".
        if located is not None and located[0] + smear <= mfb:
            break  # value found and its displacement-widened window is fully built -- certified
        if not truncated:
            break  # exhaustion certificate: the histogram already holds the entire support
        if mfb >= max_fine_bucket_cap:
            break  # resource guard -- give up, but do NOT claim exhaustion (more support may be deeper)
        mfb *= 2

    built_top = mfb
    if located is None:
        if not truncated:
            raise IndexError("index %d is beyond the structural support count %d" % (index, idx.total()))
        raise EnumerationError(
            dist,
            reason=(
                "descending-probability index %d was not reached within max_fine_bucket_cap=%d bits; "
                "the support is not proven exhausted (more values may exist at lower probability) -- "
                "raise max_fine_bucket_cap to search deeper" % (index, mfb)
            ),
        )
    b_star, offset = located
    value, _trop_lp = idx.get_in_bucket(b_star, offset)
    t = float(dist.log_density(value))  # TRUE marginal log-density of the located value

    # Center the rank window on ``value``'s TRUE bucket, not the located copy's bucket ``b_star``. A
    # value can surface as a low-weight *non-dominant* copy far down the tropical order while being
    # genuinely probable (true rank small): all of its copies sit at buckets >= fine_bucket(M(value)) >=
    # fine_bucket(log p(value)) = fb_t, so b_star >= fb_t. The displacement theorem brackets the rank
    # only when the window is centered on fb_t; ``b_star >= fb_t`` guarantees the built depth still
    # covers ``fb_t + smear``.
    hist = idx.hist
    fb_t = q.fine_bucket(t)
    lo_b, hi_b = fb_t - smear, fb_t + smear
    raw_below = sum(hist.count_at(b) for b in range(hist.base, lo_b))
    raw_within = sum(hist.count_at(b) for b in range(lo_b, hi_b + 1))
    window_built = (not truncated) or hi_b <= built_top

    if not window_built:
        # The cap was hit before value's displacement-widened window could be fully built, and
        # exhaustion was never certified either. ``raw_below`` remains a sound proven bound regardless
        # -- the identical argument count_dp_seek's ``window_lower`` relies on (see its docstring and
        # ``_window_bracket``'s): every bucket it sums is strictly below the window's lower edge, hence
        # always <= built_top whenever ``b_star`` itself was locatable at all. Concretely here: b_star
        # <= built_top by construction of ``_locate_bucket``, and fb_t <= b_star (the comment above),
        # so lo_b - 1 <= b_star - smear <= built_top, and raw_below sums only buckets < lo_b. Dividing
        # by ``mult_cap`` keeps that bound sound against the multiplicity over-count. ``raw_within`` is
        # only a best-effort figure from a partial window -- buckets beyond ``built_top`` are simply
        # ABSENT from the histogram (not confirmed empty) -- so it is NOT a guaranteed upper bound; the
        # whole bracket is marked uncertified instead of letting an incomplete window pass as a proof.
        lower = -(-raw_below // mult_cap)  # ceil division
        upper = raw_below + raw_within
        return MarginalSeekResult(
            value=value,
            log_prob=t,
            index=index,
            true_rank_lower=lower,
            true_rank_upper=upper,
            exact=False,
            oversample=oversample,
            truncated=True,
        )

    # ``tol`` is the rank's TIE THRESHOLD: a window value counts as strictly more probable only when its
    # true log-density exceeds ``t`` by more than ``tol``. It must match the convention the true rank is
    # defined by (1e-12) -- a looser value (e.g. 1e-9) would drop a genuinely-more-probable value whose
    # gap is a non-dominant component's tiny ~1e-9-nat contribution, corrupting the "exact" rank.

    # Exact path 1 -- no displacement and no over-count (decomposable, or a provably-disjoint mixture
    # reporting disp_bits == 0): raw_below is already the exact distinct count strictly below the window
    # (all provably more probable), so resolving the window against the true log_density pins the rank.
    # (window_built is guaranteed True here -- the incomplete case already returned above.)
    if mult_cap == 1 and raw_within <= resolve_max:
        seen: set = set()
        strictly_more = 0
        for b in range(lo_b, hi_b + 1):
            for off in range(hist.count_at(b)):
                u, _ = idx.get_in_bucket(b, off)
                key = freeze(u)
                if key in seen:
                    continue
                seen.add(key)
                if float(dist.log_density(u)) > t + tol:
                    strictly_more += 1
        rank = raw_below + strictly_more
        return MarginalSeekResult(
            value=value,
            log_prob=t,
            index=index,
            true_rank_lower=rank,
            true_rank_upper=rank,
            exact=True,
            oversample=oversample,
            truncated=False,
        )

    # Exact path 2 -- shallow index: the whole prefix up to the window is small, so unrank and
    # de-duplicate [base, hi_b] against the true log_density. Every value more probable than ``value``
    # has its dominant copy at a bucket <= b_star <= hi_b, so this distinct set is complete.
    if (raw_below + raw_within) <= resolve_max:
        seen = set()
        strictly_more = 0
        for b in range(hist.base, hi_b + 1):
            for off in range(hist.count_at(b)):
                u, _ = idx.get_in_bucket(b, off)
                key = freeze(u)
                if key in seen:
                    continue
                seen.add(key)
                if float(dist.log_density(u)) > t + tol:
                    strictly_more += 1
        return MarginalSeekResult(
            value=value,
            log_prob=t,
            index=index,
            true_rank_lower=strictly_more,
            true_rank_upper=strictly_more,
            exact=True,
            oversample=oversample,
            truncated=False,
        )

    # Bracket fallback -- the deep overlapping case (window_built is True here, so both bounds are
    # certified). raw_below over-counts the distinct rank by at most mult_cap (each shared value
    # appears once per component), so ceil(raw_below / mult_cap) is a sound floor; raw_below +
    # raw_within is a sound ceiling (every more-probable value has a copy <= hi_b).
    lower = -(-raw_below // mult_cap)  # ceil division
    upper = raw_below + raw_within
    return MarginalSeekResult(
        value=value,
        log_prob=t,
        index=index,
        true_rank_lower=lower,
        true_rank_upper=upper,
        exact=(lower == upper),
        oversample=oversample,
        truncated=False,
    )


def _mass_histogram(dist, quantizer, max_fine_bucket):
    """Probability MASS per fine bucket of bits -- ``{bucket: sum of p(y) over y in that bucket}``.

    Unlike the count histogram (which would need ``count x 2^-bits`` and is biased O(#factors/R)
    because structural bits under-estimate true bits), this carries the EXACT summed probability, so a
    bulk prefix sum is the exact cumulative mass of all strictly-more-probable buckets. Mass multiplies
    and bits add, so it convolves exactly like the count histogram: composites convolve their fields,
    sequences pool the per-length L-fold self-convolution shifted/scaled by the length term, leaves
    sum probabilities over their own support. Used by :func:`cumulative_probability`.
    """
    from mixle.stats.combinator.composite import CompositeDistribution
    from mixle.stats.combinator.record import RecordDistribution
    from mixle.stats.combinator.sequence import SequenceDistribution
    from mixle.stats.compute.pdist import EnumerationError

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
        from mixle.stats.compute.pdist import child_enumerator

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


@dataclass
class CumulativeProbabilityResult:
    """Outcome of :func:`cumulative_probability`: G(x) plus what is, and isn't, certified about it.

    Attributes:
        probability: ``G(x) = sum_{y: p(y) >= p(x)} p(y)``, from the exact bulk mass-histogram prefix
            plus the item-resolved tie band (see :func:`cumulative_probability`'s docstring). Clamped
            into ``[0, 1]`` only when the raw bulk+band sum landed outside that range by no more than
            this module's ``_ROUNDOFF_TOL`` roundoff budget -- see ``sum_defect``; a larger excursion
            raises instead of being silently clamped (a real defect, not roundoff).
        exact: True when nothing this function can detect compromises ``probability``: the count index
            behind the tie band carries no approximation-search drop (``dropped_upper == 0.0``, true
            for every exact-count family -- Composite/Sequence/Record/MarkovChain/leaf/Mixture/HMM --
            and false only for an approximate-search index, e.g. an autoregressive ``branch_cap``
            build). False means ``probability`` is a best-effort figure that may UNDERSTATE the truth
            -- see ``dropped_upper``.
        dropped_upper: a proven upper bound on the COUNT of in-band items an approximate-search
            family's count index may have pruned before ever reaching this function -- 0.0 for every
            exact-count family. Nonzero means ``probability`` may understate the true ``G(x)`` by up to
            that many items' worth of (each individually small) probability.
        index_truncated: the count index build's own ``truncated`` flag (True when the distribution's
            support extends beyond the query's tie band -- the common case for any non-trivial query).
            Exposed for transparency, but on its own it does NOT compromise ``probability``: the tie
            band only touches buckets within the requested depth, which the count-DP contract
            guarantees are correctly represented regardless of what lies beyond it. ``exact``/
            ``dropped_upper`` are the actual soundness signals; this one is diagnostic only.
        sum_defect: how far the raw bulk+band sum landed outside ``[0, 1]`` before any clamping -- 0.0
            when the sum was already a valid probability, otherwise a value ``<= _ROUNDOFF_TOL``
            (anything larger raises instead of reaching this result).
    """

    probability: float
    exact: bool
    dropped_upper: float
    index_truncated: bool
    sum_defect: float


def cumulative_probability(
    dist,
    value,
    oversample: int = 64,
    bin_width_bits: float = 1.0,
    smear: int | None = None,
    tol: float = _EXACT_TIE_TOL,
) -> CumulativeProbabilityResult:
    """Cumulative probability ``G(x) = sum_{y: p(y) >= p(x)} p(y)`` for decomposable families.

    Structural and at arbitrary depth (no enumeration, no sampling): the bulk mass of all buckets
    strictly below the query's smear band comes from the exact :func:`_mass_histogram` prefix, and the
    band itself is resolved item-by-item (true ``log_density``) via the count index. Because each
    bucket's mass is the EXACT sum of its items' probabilities and the band absorbs the floored-bucket
    smear (within ``#factors`` buckets), the result is exact up to floating-point roundoff for every
    exact-count family -- verified to 1e-16 on a 12-factor product, where the count-times-
    representative-probability shortcut returns G > 1. Deterministic, so it complements
    :func:`density_rank` (whose deep path is Monte-Carlo).

    Returns a :class:`CumulativeProbabilityResult`, not a bare float: an approximate-search family's
    count index (e.g. an autoregressive ``branch_cap`` build) can silently prune in-band items WITHOUT
    tripping the index's own ``truncated`` flag (that flag means something different and, for this
    function, largely benign -- see the result's ``index_truncated`` doc), so the honest completeness
    signal is the index's ``dropped_upper``, surfaced via ``result.exact``/``result.dropped_upper``
    rather than folded into a single number the caller has no way to audit.

    Args:
        tol: Log-probability tie tolerance for the band's ``>=`` comparison. Defaults to this module's
            ``1e-12`` "true rank" convention (see :func:`density_rank`'s ``tol``) -- NOT the previous
            hardcoded ``1e-9``, which is 1000x looser and can pull in values that are genuinely, if only
            slightly, less probable than ``value``.
    """
    from mixle.enumeration.quantization.core import Quantizer

    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    t = float(dist.log_density(value))
    bx = dist.structural_fine_bucket(value, q)
    smear = oversample if smear is None else int(smear)
    mass = _mass_histogram(dist, q, bx + smear)
    bulk = sum(m for b, m in mass.items() if b < bx - smear)
    index, truncated = dist.quantized_count_index(q, max_fine_bucket=bx + smear)
    band = 0.0
    for b in range(bx - smear, bx + smear + 1):
        for off in range(index.hist.count_at(b)):
            _v, lp = index.get_in_bucket(b, off)
            if float(lp) >= t - tol:
                band += math.exp(float(lp))

    raw = bulk + band
    # A cumulative probability must lie in [0, 1]; clamping unconditionally would silently paper over
    # a broken _mass_histogram/count-index pairing (duplicate counting, mismatched bucketing between
    # the two independent structural computations, an unaccounted truncation) and still call the
    # (wrong) result exact. Only a roundoff-scale excursion is clamped-and-blessed -- the same
    # roundoff-vs-defect distinction density_rank's analytic-CDF path uses, sharing its _ROUNDOFF_TOL
    # budget; anything larger is a real defect and is surfaced via `sum_defect` rather than hidden.
    sum_defect = max(0.0, -raw, raw - 1.0)
    if sum_defect > _ROUNDOFF_TOL:
        raise ValueError(
            "cumulative_probability's bulk+band sum is %.17g, outside [0, 1] by %.3e -- more than the "
            "%.1e roundoff budget, so this is a defect (duplicate counting / a normalization bug / a "
            "bucketing mismatch), not floating-point roundoff, and is not silently clamped"
            % (raw, sum_defect, _ROUNDOFF_TOL)
        )
    dropped_upper = float(getattr(index, "dropped_upper", 0.0))
    return CumulativeProbabilityResult(
        probability=min(1.0, max(0.0, raw)),
        exact=(dropped_upper == 0.0),
        dropped_upper=dropped_upper,
        index_truncated=truncated,
        sum_defect=sum_defect,
    )


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
    from mixle.enumeration.quantization.core import Quantizer
    from mixle.stats.compute.pdist import EnumerationError

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


def _joint_bucket_histogram(components, quantizer, max_fine_bucket, max_leaf_support):
    """Joint K-dim count histogram of (bucket(log p_1(y)), ..., bucket(log p_K(y))) over the support.

    Built structurally for homogeneous (same-structure) components, with NO enumeration of the JOINT
    (cross-component) support -- that union would be exponential in K. Composites instead convolve the
    per-field joint histograms (bucket tuples add, counts multiply); leaves enumerate their OWN support
    (the union of what each component individually assigns positive probability to) and key by the
    K-tuple of per-component buckets. That per-leaf enumeration is real, though, so a leaf's support
    must be finite and no larger than ``max_leaf_support`` -- see the ``EnumerationError`` raised below,
    which is exactly what stands between this function and hanging forever on an infinite-support leaf
    (e.g. a Poisson/geometric component) or burning unbounded time on a merely huge one.

    Returns ``{(b_1, ..., b_K): count}``. Raises EnumerationError for component structures not handled
    here (only Composite and atomic/enumerable leaves are supported), for a leaf with unknown/infinite
    support (``support_size()`` is ``None``), or for a leaf whose support exceeds ``max_leaf_support``.
    """
    from mixle.enumeration.streams import freeze
    from mixle.stats.combinator.composite import CompositeDistribution
    from mixle.stats.compute.pdist import EnumerationError

    head = components[0]
    if isinstance(head, CompositeDistribution):
        arity = head.count
        joint = {(0,) * len(components): 1}
        for f in range(arity):
            field = _joint_bucket_histogram(
                [c.dists[f] for c in components], quantizer, max_fine_bucket, max_leaf_support
            )
            nxt: dict[tuple[int, ...], int] = {}
            for ka, ca in joint.items():
                for kb, cb in field.items():
                    key = tuple(ka[j] + kb[j] for j in range(len(ka)))
                    if max(key) <= max_fine_bucket:
                        nxt[key] = nxt.get(key, 0) + ca * cb
            joint = nxt
        return joint
    # Leaf: enumerate the union of component supports, key by the per-component bucket tuple. Each
    # component's support must be finite and bounded BEFORE enumerating it -- support_size() is the
    # cheap, non-enumerating cardinality primitive this module already uses for the same purpose in
    # truncated_sum_bound (see mixle.stats.compute.pdist.ProbabilityDistribution.support_size).
    for comp in components:
        size = comp.support_size()
        if size is None:
            raise EnumerationError(
                comp,
                reason=(
                    "mixture_cross_rank requires every component to have a finite, bounded support; "
                    "support_size() is None (infinite or unknown) -- this component would be enumerated "
                    "without ever terminating"
                ),
            )
        if size > max_leaf_support:
            raise EnumerationError(
                comp,
                reason=(
                    "component support_size()=%d exceeds max_leaf_support=%d -- raise max_leaf_support "
                    "if this is really intended, or restrict to smaller leaf families; mixture_cross_rank's "
                    "K-dimensional joint histogram is for SMALL, bounded leaf supports, not a substitute "
                    "for enumerating a large one" % (size, max_leaf_support)
                ),
            )
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


def mixture_cross_rank(
    mixture,
    value,
    oversample: int = 64,
    bin_width_bits: float = 1.0,
    depth_bits: float = 64.0,
    max_leaf_support: int = 100_000,
):
    """True-marginal rank of ``value`` under a homogeneous mixture, at arbitrary depth.

    ``count_dp_rank`` on a mixture gives only the TROPICAL (dominant-component) rank -- it bins by the
    best single component, so a value built from several components is badly mis-ranked. This computes
    the true rank against the actual marginal ``p = sum_k w_k p_k`` by building the JOINT K-dimensional
    count histogram of the per-component log-prob buckets (structurally, with no enumeration of the
    JOINT/cross-component support -- see :func:`_joint_bucket_histogram`) and counting joint bins whose
    representative marginal probability exceeds ``p(value)``.

    Quantization-approximate: a joint bin's marginal probability is evaluated at the bucket midpoints,
    so bins straddling the threshold may be mis-counted; the error shrinks as ``oversample`` grows. The
    returned rank carries no explicit quantization-error or truncation metadata beyond that -- treat it
    as an approximate count, tightened by raising ``oversample``, not as a certified bound.

    Cost is EXPONENTIAL in the number of components K (the histogram is K-dimensional), so this is for
    SMALL-K mixtures (a few components); it needs no enumeration of the joint support, so it scales to
    deep ranks along that axis. It is NOT free along every axis, though: each LEAF component's OWN
    support (e.g. one field of a composite, or an atomic leaf) IS enumerated once (the union of what
    every component assigns positive probability to), so every leaf must have a finite support of at
    most ``max_leaf_support`` -- an infinite-support leaf (Poisson, geometric, an unbounded sequence, ...)
    or a merely huge finite one raises ``EnumerationError`` rather than hanging or silently defeating the
    advertised scaling. For non-mixtures use :func:`count_dp_rank`; for the head of any model use
    :func:`density_rank`.
    """
    from mixle.enumeration.quantization.core import Quantizer

    comps = [c for c, w in zip(mixture.components, mixture.w, strict=False) if w > 0.0]
    log_w = [float(lw) for lw, w in zip(mixture.log_w, mixture.w, strict=False) if w > 0.0]
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    max_fb = int(math.ceil(depth_bits * oversample / bin_width_bits))
    joint = _joint_bucket_histogram(comps, q, max_fb, max_leaf_support)

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
