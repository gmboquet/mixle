"""Multiple-testing correction and evidence combination.

When you run many tests at once, the chance that *some* cross a fixed threshold by luck grows with the
number of tests, so a raw 0.05 cutoff no longer means what it says. Two notions of "control" answer
this, trading power for stringency:

  * **Family-wise error rate (FWER)** -- the probability of *any* false rejection. Controlled by
    :func:`bonferroni` (single-step, always valid) and the uniformly more powerful step-wise
    :func:`holm` (always valid) and :func:`hochberg` (valid under independence / positive dependence).
  * **False discovery rate (FDR)** -- the *expected fraction* of rejections that are false. Less
    stringent, more power for screening many hypotheses. :func:`benjamini_hochberg` (valid under
    independence / PRDS) and :func:`benjamini_yekutieli` (valid under arbitrary dependence).

Each returns adjusted p-values (a.k.a. q-values for FDR) in the original order plus the rejection mask
at level ``alpha``; :func:`adjust_pvalues` is the unified dispatcher.

For the complementary problem -- combining evidence for the *same* hypothesis across independent
replications/strata -- :func:`fisher_combine`, :func:`stouffer_combine` (optionally weighted), and
:func:`tippett_combine` give a single pooled p-value (lightweight fixed-effect meta-analysis).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

_METHODS = ("bonferroni", "holm", "hochberg", "bh", "by")


def _prep(pvals: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float array of p-values and validate the range."""
    p = np.asarray(pvals, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("pvals must be non-empty.")
    if np.any((p < 0.0) | (p > 1.0)) or np.any(~np.isfinite(p)):
        raise ValueError("pvals must be finite and in [0, 1].")
    return p


def _check_alpha(alpha: float) -> None:
    """Validate a target error-rate threshold (FWER for bonferroni/holm/hochberg, FDR for bh/by).

    An out-of-range or NaN ``alpha`` does not raise inside any of the correction routines below --
    ``adjusted <= alpha`` is just a boolean comparison, so e.g. ``alpha=nan`` silently rejects nothing
    (a NaN comparison is always False) and ``alpha=1.5`` silently rejects everything -- both a
    plausible-looking but meaningless ``reject``/``n_reject`` instead of an error.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")


def _result(pvals: np.ndarray, adjusted: np.ndarray, alpha: float) -> dict[str, np.ndarray | int | float]:
    """Package adjusted p-values + rejection mask in the input order."""
    reject = adjusted <= alpha
    return {
        "reject": reject,
        "pvals_adjusted": adjusted,
        "n_reject": int(reject.sum()),
        "alpha": float(alpha),
    }


def bonferroni(pvals: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Bonferroni single-step FWER control: adjusted ``p_i = min(1, m p_i)``.

    The simplest and most conservative correction; always valid regardless of dependence.

    Args:
        pvals: ``(m,)`` raw p-values.
        alpha: target family-wise error rate, in ``(0, 1)``.

    Returns:
        ``{'reject', 'pvals_adjusted', 'n_reject', 'alpha'}``.
    """
    _check_alpha(alpha)
    p = _prep(pvals)
    adjusted = np.minimum(1.0, p * p.size)
    return _result(p, adjusted, alpha)


def holm(pvals: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Holm step-down FWER control -- uniformly more powerful than Bonferroni, equally valid.

    Sorts p-values ascending and applies the running factor ``m - k`` with a cumulative maximum so the
    adjusted values stay monotone.
    """
    _check_alpha(alpha)
    p = _prep(pvals)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    # step-down: factor (m - k) for the k-th smallest (0-indexed), then enforce monotone non-decreasing
    factors = m - np.arange(m)
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, factors * ranked))
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return _result(p, adjusted, alpha)


def hochberg(pvals: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Hochberg step-up FWER control (valid under independence / positive dependence).

    Same per-step factor as Holm but applied step-up (from the largest p-value down), giving more
    rejections; requires the independence/PRDS assumption that Holm does not.
    """
    _check_alpha(alpha)
    p = _prep(pvals)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    factors = m - np.arange(m)
    # step-up: cumulative minimum from the largest p-value downward
    adj_sorted = np.minimum.accumulate((factors * ranked)[::-1])[::-1]
    adj_sorted = np.minimum(1.0, adj_sorted)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return _result(p, adjusted, alpha)


def benjamini_hochberg(pvals: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Benjamini--Hochberg FDR control; adjusted values are q-values.

    Controls the expected false-discovery proportion at ``alpha`` under independence or positive
    regression dependence (PRDS). The standard choice for screening many hypotheses.
    """
    _check_alpha(alpha)
    p = _prep(pvals)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    # q_(i) = min_{k>=i} ( m/k * p_(k) ), enforced monotone via cumulative min from the top
    adj_sorted = np.minimum.accumulate((m / ranks * ranked)[::-1])[::-1]
    adj_sorted = np.minimum(1.0, adj_sorted)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return _result(p, adjusted, alpha)


def benjamini_yekutieli(pvals: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Benjamini--Yekutieli FDR control, valid under *arbitrary* dependence.

    Like :func:`benjamini_hochberg` but inflated by the harmonic factor ``c(m) = sum_{i=1}^m 1/i``, so
    it holds for any dependence structure at the cost of power.
    """
    _check_alpha(alpha)
    p = _prep(pvals)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    c_m = float(np.sum(1.0 / ranks))
    adj_sorted = np.minimum.accumulate((c_m * m / ranks * ranked)[::-1])[::-1]
    adj_sorted = np.minimum(1.0, adj_sorted)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return _result(p, adjusted, alpha)


def adjust_pvalues(pvals: np.ndarray, *, method: str = "bh", alpha: float = 0.05) -> dict:
    """Unified dispatcher over the correction methods.

    Args:
        pvals: ``(m,)`` raw p-values.
        method: one of ``"bonferroni"``, ``"holm"``, ``"hochberg"``, ``"bh"`` (Benjamini--Hochberg),
            ``"by"`` (Benjamini--Yekutieli).
        alpha: target error rate (FWER for the first three, FDR for the last two), in ``(0, 1)``.

    Returns:
        ``{'reject', 'pvals_adjusted', 'n_reject', 'alpha'}``.
    """
    dispatch = {
        "bonferroni": bonferroni,
        "holm": holm,
        "hochberg": hochberg,
        "bh": benjamini_hochberg,
        "by": benjamini_yekutieli,
    }
    if method not in dispatch:
        raise ValueError(f"method must be one of {_METHODS}.")
    return dispatch[method](pvals, alpha=alpha)


def fisher_combine(pvals: np.ndarray) -> dict[str, float]:
    """Fisher's method: combine independent p-values via ``-2 sum log p ~ chi^2_{2k}``.

    Sensitive to a few very small p-values. For combining evidence *for the same hypothesis* across
    independent tests.

    Returns:
        ``{'statistic', 'pvalue', 'df'}``.
    """
    p = _prep(pvals)
    p = np.clip(p, np.finfo(float).tiny, 1.0)
    stat = float(-2.0 * np.sum(np.log(p)))
    df = 2 * p.size
    return {"statistic": stat, "pvalue": float(chi2.sf(stat, df)), "df": df}


def stouffer_combine(pvals: np.ndarray, *, weights: np.ndarray | None = None) -> dict[str, float]:
    """Stouffer's Z method: combine p-values on the z-scale, optionally weighted.

    ``Z = sum w_i Phi^{-1}(1 - p_i) / sqrt(sum w_i^2)``. Weights let more-precise studies (e.g. larger
    samples) count more; equal weights recover the unweighted combination. A weight of exactly ``0``
    for one study while the others are positive is a legitimate "exclude this study" case (mirrors
    :func:`weighted_conformal`'s zero-weight convention) and stays valid; every weight being
    non-positive or non-finite together is degenerate and raises.

    Args:
        pvals: ``(k,)`` raw p-values from independent studies/strata.
        weights: optional ``(k,)`` weights, non-negative with a positive total; ``None`` weights every
            study equally.

    Returns:
        ``{'z', 'pvalue'}`` (one-sided combined p-value).
    """
    p = _prep(pvals)
    p = np.clip(p, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != p.shape:
        raise ValueError("weights must match pvals in length.")
    if not np.isfinite(w).all():
        # checked before the sign check below: a NaN comparison is always False, so `NaN < 0.0` would
        # otherwise silently pass as "non-negative" and propagate straight through the sum below into
        # a silent nan z/pvalue instead of raising.
        raise ValueError(f"weights must be finite, got {w!r}.")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be non-negative, got {w!r}.")
    total_weight = float(np.sum(w))
    if total_weight <= 0.0:
        # w is already known finite and non-negative here, so the only way to land here is every
        # weight being exactly zero -- no study actually contributing evidence -- which used to divide
        # sqrt(sum(w*w)) == 0 into 0/0 = NaN below instead of raising. A single zero weight among
        # otherwise-positive weights is fine (see the docstring); only ALL-zero is rejected.
        raise ValueError(f"stouffer_combine requires at least one positive weight, got total {total_weight!r}.")
    z = float(np.sum(w * norm.isf(p)) / np.sqrt(np.sum(w * w)))
    return {"z": z, "pvalue": float(norm.sf(z))}


def tippett_combine(pvals: np.ndarray) -> dict[str, float]:
    """Tippett's method (Sidak min-p): combined ``p = 1 - (1 - min p)^k``.

    Most powerful when a single strong signal exists among the tests.

    Evaluated as ``-expm1(k * log1p(-min_p))`` rather than literally (MXR-080-1604). Forming
    ``1 - min_p`` first rounds the evidence away before it is ever exponentiated: below the float64
    epsilon the subtraction returns exactly ``1.0``, so a minimum p-value of ``1e-17``, ``1e-20`` or
    ``1e-100`` combined to exactly ``0.0`` -- an impossible result for finite nonzero inputs, and one
    that overstates certainty without bound. Even ``1e-16`` was inflated to ``2.220446049250313e-16``
    (two epsilons) instead of the correct ``2e-16``. The stable form is accurate all the way down to
    the smallest subnormal.

    Returns:
        ``{'min_p', 'pvalue'}``.
    """
    p = _prep(pvals)
    min_p = float(p.min())
    if min_p >= 1.0:
        # log1p(-1) is -inf; the limit is 1, so return it directly rather than through the -inf path
        return {"min_p": min_p, "pvalue": 1.0}
    return {"min_p": min_p, "pvalue": float(-np.expm1(p.size * np.log1p(-min_p)))}


__all__ = [
    "bonferroni",
    "holm",
    "hochberg",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "adjust_pvalues",
    "fisher_combine",
    "stouffer_combine",
    "tippett_combine",
]
