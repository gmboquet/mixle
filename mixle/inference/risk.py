"""Risk / tail metrics over a Monte-Carlo outcome distribution (e.g. J2's NPV samples).

A distribution's mean says nothing about how bad the bad outcomes are. This module turns a plain
sample array -- typically :class:`~mixle.analysis.valuation.NPVDistribution`'s ``samples``, but any
array of scalar outcomes works -- into the two standard tail-risk summaries plus a scenario ranking:

  * :func:`value_at_risk` -- the loss not exceeded with probability ``alpha`` (a quantile).
  * :func:`conditional_value_at_risk` -- the expected loss *given* that the VaR threshold is
    breached (expected shortfall); always at least as large as VaR. A Generalized Pareto tail
    extrapolation is available by OPT-IN only (``gpd_tail=True``): measured on sparse tails it
    was strictly noisier than the raw tail mean it replaced (audit R-1), so sparse tails default
    to the honest sample statistic.
  * :func:`stress_rank` -- named stress scenarios (low-grade, price-crash, carbon-spike, ...) ranked
    from worst to least-bad loss.

Throughout, *loss* is ``-outcome`` -- a positive loss means a bad (low/negative NPV) draw, so VaR and
CVaR come back as positive numbers when the distribution has meaningful downside.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def _as_samples(samples: Any) -> np.ndarray:
    x = np.asarray(samples, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("samples must be non-empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite (no NaN or Inf).")
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


def conditional_value_at_risk(
    samples: Any, alpha: float = 0.95, *, min_tail: int = 20, gpd_tail: bool = False
) -> float:
    """Conditional Value-at-Risk (expected shortfall) at confidence ``alpha``.

    The empirical estimator is the mass-weighted mean over the worst ``1 - alpha`` probability
    mass (NOT the docstring formula ``-mean(samples <= -VaR)``, which overweights observations
    tied at VaR -- audit R-3 pinned the mismatch between the stated and computed estimator).
    Because the tail mean is at least as extreme as the threshold bounding it, ``CVaR >= VaR``.

    ``gpd_tail=True`` opts in to a Generalized Pareto refinement of a sparse tail (fewer than
    ``min_tail`` exceedances). It is OFF by default because it is measurably WORSE exactly where
    it fires (audit R-1): fitting a two-parameter heavy-tail MLE to the same 5-19 points it is
    meant to rescue, then plugging the estimates into ``VaR + scale/(1 - shape)``, multiplied
    the estimator's spread instead of shrinking it -- the estimand of that measurement is the
    estimator's own sampling uncertainty: on lognormal losses at n = 100 (true ES 8.54), the raw
    tail mean's sampling sd was 2.36 while the GPD path's was 46.4 with a worst case of 663. Use the refinement only with an exogenous reason to trust a GPD tail (e.g. a
    threshold chosen from much more data), and treat its output as a model extrapolation, not a
    sample statistic; its parameters are plug-in MLEs with no interval attached (audit R-2).

    Args:
        samples: array-like of scalar outcomes (same array passed to :func:`value_at_risk`).
        alpha: confidence level in ``(0, 1)``.
        min_tail: tail sample count below which the (opt-in) GPD refinement is attempted.
        gpd_tail: opt in to the GPD tail extrapolation described above.

    Returns:
        The CVaR as a loss; always ``>= value_at_risk(samples, alpha)``.
    """
    if isinstance(min_tail, bool) or not isinstance(min_tail, Integral):
        raise TypeError("min_tail must be a positive integer")
    min_tail = int(min_tail)
    if min_tail < 1:
        raise ValueError("min_tail must be a positive integer")
    x = _as_samples(samples)
    var = value_at_risk(x, alpha)
    losses = np.sort(-x)[::-1]
    # Exact empirical expected shortfall is the average over the worst ``1-alpha`` probability mass.
    # When that mass cuts through observations tied at VaR, include only the fractional boundary mass
    # required instead of overweighting every tied row.
    tail_mass = len(losses) * (1.0 - alpha)
    full = int(np.floor(tail_mass))
    fraction = tail_mass - full
    weighted_loss = float(np.sum(losses[:full]))
    if fraction > 0.0:
        weighted_loss += fraction * float(losses[full])
    raw_cvar = weighted_loss / tail_mass
    empirical_tail_count = int(np.ceil(tail_mass))
    if gpd_tail and empirical_tail_count < min_tail:
        # Import lazily: importing a submodule executes ``mixle.analysis``'s
        # package initializer, which itself reaches inference through valuation.
        # Risk is re-exported while Dirichlet may still be initializing.
        from mixle.analysis.extreme import peaks_over_threshold

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
    # Route through _as_samples (rather than a bare np.asarray) so a NaN/Inf entry in any scenario's
    # outcome(s) raises here instead of silently producing a NaN loss that sorts unpredictably.
    ranked = [(name, float(-_as_samples(value).mean())) for name, value in scenarios.items()]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked
