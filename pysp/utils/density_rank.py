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
    from pysp.stats.pdist import EnumerationError

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
    from pysp.stats.pdist import EnumerationError
    from pysp.utils.quantization.core import Quantizer

    t = float(dist.log_density(value))
    q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)
    fb = q.fine_bucket(t)
    smear = oversample if smear is None else int(smear)  # ~one bit of boundary uncertainty
    index, _truncated = dist.quantized_count_index(q, max_fine_bucket=fb + smear)
    if index is None:
        raise EnumerationError(dist, reason="no structural count index for rank")
    hist = index.hist

    lo_b = fb - smear
    hi_b = fb + smear
    window_lower = sum(hist.count_at(b) for b in range(hist.base, lo_b))
    window_count = sum(hist.count_at(b) for b in range(lo_b, hi_b + 1))
    window_upper = window_lower + window_count

    if window_count <= resolve_max:
        # Resolve the smear window exactly: count its items strictly more probable than value.
        strictly_more = 0
        for b in range(lo_b, hi_b + 1):
            for off in range(hist.count_at(b)):
                _v, lp = index.get_in_bucket(b, off)
                if float(lp) > t + tol:
                    strictly_more += 1
        return CountDPRankResult(window_lower + strictly_more, window_lower, window_upper, t, oversample)
    return CountDPRankResult((window_lower + window_upper) // 2, window_lower, window_upper, t, oversample)
