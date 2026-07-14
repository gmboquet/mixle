"""Risk / tail metrics over a Monte-Carlo outcome distribution (e.g. J2's NPV samples).

A distribution's mean says nothing about how bad the bad outcomes are. This module turns a plain
sample array -- typically :class:`~mixle.analysis.valuation.NPVDistribution`'s ``samples``, but any
array of scalar outcomes works -- into the two standard tail-risk summaries plus a scenario ranking:

  * :func:`value_at_risk` -- the loss not exceeded with probability ``alpha`` (a quantile).
  * :func:`conditional_value_at_risk` -- the expected loss *given* that the VaR threshold is
    breached (expected shortfall); always at least as large as VaR. When the exceedance tail is
    sparse, refines the raw tail mean with a fitted Generalized Pareto tail
    (:func:`mixle.analysis.extreme.peaks_over_threshold`) rather than trusting a handful of points.
  * :func:`stress_rank` -- named stress scenarios (low-grade, price-crash, carbon-spike, ...) ranked
    from worst to least-bad loss.

Throughout, *loss* is ``-outcome`` -- a positive loss means a bad (low/negative NPV) draw, so VaR and
CVaR come back as positive numbers when the distribution has meaningful downside.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.analysis.extreme import peaks_over_threshold


def _as_samples(samples: Any) -> np.ndarray:
    x = np.asarray(samples, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("samples must be non-empty.")
    return x


def value_at_risk(samples: Any, alpha: float = 0.95) -> float:
    """Value-at-Risk at confidence ``alpha``.

    ``VaR_alpha = -quantile(samples, 1 - alpha)``: losses are ``-outcome``, so the ``(1-alpha)``
    lower tail of the outcome distribution (rare, bad draws) becomes the loss exceeded only
    ``(1-alpha)`` of the time.

    Args:
        samples: array-like of scalar outcomes (e.g. an NPV Monte-Carlo sample array).
        alpha: confidence level in ``(0, 1)``; ``0.95`` means "loss exceeded 5% of the time".

    Returns:
        The VaR as a loss (positive when the distribution has meaningful downside).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    x = _as_samples(samples)
    return float(-np.quantile(x, 1.0 - alpha))


def conditional_value_at_risk(samples: Any, alpha: float = 0.95, *, min_tail: int = 20) -> float:
    """Conditional Value-at-Risk (expected shortfall) at confidence ``alpha``.

    ``CVaR_alpha = -mean(samples[samples <= -VaR_alpha])`` -- the mean loss in the tail beyond VaR.
    Because the tail mean is at least as extreme as the threshold that bounds it, ``CVaR >= VaR``
    always holds.

    When the tail has fewer than ``min_tail`` observations the raw sample mean is noisy (a handful
    of points), so the exceedances are refined with a fitted Generalized Pareto tail
    (:func:`~mixle.analysis.extreme.peaks_over_threshold`, on the *loss* scale, thresholded at VaR)
    and the analytic GPD tail mean ``VaR + scale / (1 - shape)`` is returned instead, falling back to
    the raw tail mean if the fit is unavailable (too few exceedances) or the GPD tail mean is
    undefined (``shape >= 1``).

    Args:
        samples: array-like of scalar outcomes (same array passed to :func:`value_at_risk`).
        alpha: confidence level in ``(0, 1)``.
        min_tail: tail sample count below which the GPD refinement is attempted.

    Returns:
        The CVaR as a loss; always ``>= value_at_risk(samples, alpha)``.
    """
    x = _as_samples(samples)
    var = value_at_risk(x, alpha)
    tail = x[x <= -var]
    if tail.size == 0:
        tail = np.array([x.min()])
    raw_cvar = float(-tail.mean())
    if tail.size < min_tail:
        losses = -x
        try:
            fit = peaks_over_threshold(losses, threshold=var)
        except ValueError:
            fit = None
        if fit is not None and fit.shape < 1.0:
            refined = var + fit.scale / (1.0 - fit.shape)
            return float(max(refined, var))
    return raw_cvar


def stress_rank(scenarios: dict[str, Any]) -> list[tuple[str, float]]:
    """Rank named stress scenarios (e.g. low-grade, price-crash, carbon-spike) by loss.

    Each scenario's value may be a scalar outcome (e.g. a single stressed NPV) or an array of
    outcomes (e.g. NPV samples drawn under that stress); the ranking key is ``-mean(value)`` either
    way, so a bare float and a 1-sample array behave identically.

    Args:
        scenarios: mapping of scenario name -> scalar outcome or array of outcomes.

    Returns:
        ``[(name, loss), ...]`` sorted from worst (largest loss) to least-bad, ties broken by the
        input mapping's iteration order.
    """
    if not scenarios:
        raise ValueError("scenarios must be non-empty.")
    ranked = [(name, float(-np.asarray(value, dtype=float).ravel().mean())) for name, value in scenarios.items()]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked
