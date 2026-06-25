"""Proper scoring rules for probabilistic forecasts.

A scoring rule ``S(F, y)`` measures how well a predictive distribution ``F`` matched the value ``y``
that actually occurred; it is *proper* when the forecaster minimises the expected score by reporting
their true belief (and *strictly* proper when that minimiser is unique). These give a single, fair
currency for comparing probabilistic forecasts and interval methods *at matched coverage* -- a 90%
interval that is wide everywhere and a 90% interval that is tight where it can be both cover 90% of
the time, but the second wins on the interval score.

Lower is better for every score here (they are penalties / losses):

  * :func:`log_score` -- the log loss ``-log p(y)`` (local, strictly proper).
  * :func:`brier_score` / :func:`brier_decomposition` -- squared-error score for categorical
    forecasts, with the reliability/resolution/uncertainty decomposition.
  * :func:`crps_ensemble` / :func:`crps_gaussian` -- the Continuous Ranked Probability Score, the
    CDF-space analogue of absolute error; reduces to absolute error for a point forecast.
  * :func:`interval_score` (a.k.a. :func:`winkler_score`) -- the proper score for a central interval
    forecast: width plus an out-of-bounds penalty scaled by the miscoverage level.
  * :func:`pinball_loss` -- the check loss whose expectation is minimised by the true quantile.
  * :func:`energy_score` -- the multivariate generalisation of CRPS.
  * :func:`skill_score` -- a score expressed as fractional improvement over a reference.

All functions are pure NumPy/SciPy and operate on plain arrays; by default they return the mean score
over observations, with ``mean=False`` returning the per-observation vector for paired comparisons
(see :func:`pysp.inference.resampling` for CIs on score differences).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_SQRT_PI = float(np.sqrt(np.pi))


def _reduce(values: np.ndarray, mean: bool) -> np.ndarray | float:
    """Return the per-observation array or its mean."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if mean else values


def log_score(prob: np.ndarray, *, mean: bool = True) -> np.ndarray | float:
    """Logarithmic score (log loss) ``-log p(y)`` of the probability assigned to the outcome.

    The log score is local (it depends only on the density at the realised value) and strictly
    proper. Pass the predictive probability/density evaluated at the observed outcome.

    Args:
        prob: array of predictive probabilities (or densities) at the realised outcomes. Values are
            clipped away from zero before the log so a single zero-probability event yields a large
            but finite penalty rather than ``inf``.
        mean: if True (default) return the mean log loss; otherwise the per-observation vector.

    Returns:
        Mean log loss (float) or the per-observation array.
    """
    p = np.asarray(prob, dtype=float)
    p = np.clip(p, np.finfo(float).tiny, None)
    return _reduce(-np.log(p), mean)


def brier_score(prob: np.ndarray, outcome: np.ndarray, *, mean: bool = True) -> np.ndarray | float:
    """Brier score (mean squared error of probabilistic classification).

    For binary forecasts pass 1-D ``prob`` (predicted probability of the positive class) and 0/1
    ``outcome``. For ``K``-class forecasts pass ``prob`` shaped ``(n, K)`` and ``outcome`` either as
    integer class labels (length ``n``) or as a one-hot ``(n, K)`` matrix; the multiclass score sums
    the squared error across classes, so it ranges in ``[0, 2]``.

    Args:
        prob: ``(n,)`` positive-class probabilities, or ``(n, K)`` class probabilities.
        outcome: ``(n,)`` 0/1 or integer labels, or ``(n, K)`` one-hot.
        mean: if True return the mean; otherwise the per-observation vector.

    Returns:
        Mean Brier score (float) or the per-observation array.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome)
    if p.ndim == 1:
        yb = y.astype(float)
        return _reduce((p - yb) ** 2, mean)
    n, k = p.shape
    if y.ndim == 1:
        onehot = np.zeros((n, k), dtype=float)
        onehot[np.arange(n), y.astype(int)] = 1.0
    else:
        onehot = y.astype(float)
    return _reduce(np.sum((p - onehot) ** 2, axis=1), mean)


def brier_decomposition(prob: np.ndarray, outcome: np.ndarray, *, bins: int = 10) -> dict[str, float]:
    """Murphy's reliability--resolution--uncertainty decomposition of the (binary) Brier score.

    Bins the forecast probabilities into ``bins`` equal-width bins on ``[0, 1]`` and computes

        Brier = reliability - resolution + uncertainty,

    where *reliability* (lower is better) is the mean squared gap between forecast probability and
    observed frequency within a bin, *resolution* (higher is better) is how far bin frequencies move
    from the base rate, and *uncertainty* is the base-rate variance ``p_bar (1 - p_bar)`` (a property
    of the data, not the forecaster).

    Args:
        prob: ``(n,)`` predicted probabilities of the positive class.
        outcome: ``(n,)`` 0/1 outcomes.
        bins: number of equal-width probability bins.

    Returns:
        ``{'reliability', 'resolution', 'uncertainty', 'brier'}``. The identity
        ``reliability - resolution + uncertainty == brier`` holds up to binning of the score's
        in-bin variance term.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    n = p.shape[0]
    p_bar = float(y.mean())
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    reliability = 0.0
    resolution = 0.0
    for b in range(bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        p_mean = float(p[mask].mean())
        o_mean = float(y[mask].mean())
        reliability += nb * (p_mean - o_mean) ** 2
        resolution += nb * (o_mean - p_bar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = p_bar * (1.0 - p_bar)
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": reliability - resolution + uncertainty,
    }


def _crps_sample(sorted_x: np.ndarray, y: float, fair: bool) -> float:
    """CRPS of a single sorted ensemble ``sorted_x`` against scalar ``y``."""
    m = sorted_x.shape[0]
    term1 = np.abs(sorted_x - y).mean()
    # sum_{i<j}(x_j - x_i) for ascending x equals sum_k x_k * (2k - m + 1).
    k = np.arange(m)
    pair_sum = float(np.dot(sorted_x, 2 * k - m + 1))
    denom = m * (m - 1) if fair else m * m
    return float(term1) - pair_sum / denom


def crps_ensemble(forecasts: np.ndarray, y: np.ndarray, *, fair: bool = False, mean: bool = True) -> np.ndarray | float:
    """Continuous Ranked Probability Score from a finite predictive ensemble (sample).

    Uses the energy form ``CRPS = E|X - y| - 1/2 E|X - X'|`` estimated from the ensemble draws. The
    CRPS generalises absolute error to distributional forecasts (a point forecast recovers ``|x-y|``)
    and is reported in the same units as ``y``.

    Args:
        forecasts: predictive draws. Either ``(n, m)`` (``m`` draws for each of ``n`` observations) or
            ``(m,)`` for a single observation. Ragged ensembles are not supported -- pad or call per
            observation.
        y: ``(n,)`` realised values (or a scalar for the ``(m,)`` case).
        fair: if True use the unbiased ``1/(m(m-1))`` spread estimator (the "fair"/almost-unbiased
            CRPS); if False (default) the standard ``1/m^2`` estimator.
        mean: if True return the mean CRPS; otherwise the per-observation vector.

    Returns:
        Mean CRPS (float) or the per-observation array.
    """
    f = np.asarray(forecasts, dtype=float)
    if f.ndim == 1:
        f = f[None, :]
        y = np.asarray([y], dtype=float)
    else:
        y = np.asarray(y, dtype=float)
    if f.shape[1] < 2 and fair:
        raise ValueError("fair CRPS needs at least two ensemble members.")
    out = np.empty(f.shape[0], dtype=float)
    for i in range(f.shape[0]):
        out[i] = _crps_sample(np.sort(f[i]), float(y[i]), fair)
    return _reduce(out, mean)


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, *, mean: bool = True) -> np.ndarray | float:
    """Closed-form CRPS for a Gaussian predictive distribution ``N(mu, sigma^2)``.

    ``CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]`` with ``z = (y - mu) / sigma``
    (Gneiting & Raftery 2007). Exact, so it is the right reference when the forecast is Gaussian.

    Args:
        mu: predictive means, broadcastable with ``y``.
        sigma: predictive standard deviations (> 0), broadcastable with ``y``.
        y: realised values.
        mean: if True return the mean; otherwise the per-observation vector.

    Returns:
        Mean CRPS (float) or the per-observation array.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive.")
    z = (y - mu) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / _SQRT_PI)
    return _reduce(crps, mean)


def interval_score(
    lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float, *, mean: bool = True
) -> np.ndarray | float:
    """Winkler interval score for a central ``(1 - alpha)`` prediction interval.

    ``IS = (u - l) + (2/alpha)(l - y) 1[y < l] + (2/alpha)(y - u) 1[y > u]`` (Gneiting & Raftery
    2007). It rewards narrow intervals but penalises misses by ``2/alpha`` times the shortfall, so at
    matched nominal coverage the tighter, better-placed interval scores lower. This is the proper way
    to rank interval methods (conformal, quantile-regression, GP credible bands) head to head.

    Args:
        lower: ``(n,)`` interval lower endpoints.
        upper: ``(n,)`` interval upper endpoints.
        y: ``(n,)`` realised values.
        alpha: miscoverage level (e.g. ``0.1`` for a 90% interval), in ``(0, 1)``.
        mean: if True return the mean; otherwise the per-observation vector.

    Returns:
        Mean interval score (float) or the per-observation array.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)
    width = hi - lo
    below = np.where(y < lo, lo - y, 0.0)
    above = np.where(y > hi, y - hi, 0.0)
    score = width + (2.0 / alpha) * (below + above)
    return _reduce(score, mean)


def winkler_score(
    lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float, *, mean: bool = True
) -> np.ndarray | float:
    """Alias for :func:`interval_score` (the score is due to Winkler 1972)."""
    return interval_score(lower, upper, y, alpha, mean=mean)


def pinball_loss(pred: np.ndarray, y: np.ndarray, tau: float | np.ndarray, *, mean: bool = True) -> np.ndarray | float:
    """Pinball (quantile / check) loss for a quantile forecast.

    ``rho_tau(u) = max(tau * u, (tau - 1) * u)`` with ``u = y - pred``; its expectation is minimised
    by the true ``tau``-quantile, so it is the proper score for quantile forecasts and the loss that
    quantile regression and conformalised quantiles optimise.

    Args:
        pred: predicted ``tau``-quantiles, shape ``(n,)`` (single level) or ``(n, q)`` (several
            levels, one per column of ``tau``).
        y: ``(n,)`` realised values.
        tau: quantile level(s) in ``(0, 1)`` -- a scalar for ``(n,)`` ``pred``, or a length-``q``
            array matching the columns of a ``(n, q)`` ``pred``.
        mean: if True return the mean over all entries; otherwise the per-observation (or
            per-observation-per-level) array.

    Returns:
        Mean pinball loss (float) or the per-observation array.
    """
    p = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    if np.any((tau_arr <= 0.0) | (tau_arr >= 1.0)):
        raise ValueError("tau must be in (0, 1).")
    if p.ndim == 1:
        u = y - p
    else:
        u = y[:, None] - p
    loss = np.maximum(tau_arr * u, (tau_arr - 1.0) * u)
    return _reduce(loss, mean)


def energy_score(forecasts: np.ndarray, y: np.ndarray, *, fair: bool = False, mean: bool = True) -> np.ndarray | float:
    """Energy score: the multivariate generalisation of CRPS for vector-valued forecasts.

    ``ES = E||X - y|| - 1/2 E||X - X'||`` with Euclidean norms; for scalar ``y`` it equals
    :func:`crps_ensemble`.

    Args:
        forecasts: predictive draws shaped ``(n, m, d)`` (``m`` ``d``-vectors per observation) or
            ``(m, d)`` for a single observation.
        y: ``(n, d)`` realised vectors (or ``(d,)`` for the single-observation case).
        fair: use the unbiased ``1/(m(m-1))`` spread estimator if True.
        mean: if True return the mean; otherwise the per-observation vector.

    Returns:
        Mean energy score (float) or the per-observation array.
    """
    f = np.asarray(forecasts, dtype=float)
    y = np.asarray(y, dtype=float)
    if f.ndim == 2:
        f = f[None, :, :]
        y = y[None, :]
    n, m, _ = f.shape
    if m < 2 and fair:
        raise ValueError("fair energy score needs at least two ensemble members.")
    out = np.empty(n, dtype=float)
    denom = m * (m - 1) if fair else m * m
    for i in range(n):
        xi = f[i]
        term1 = np.linalg.norm(xi - y[i], axis=1).mean()
        diff = xi[:, None, :] - xi[None, :, :]
        pair = np.linalg.norm(diff, axis=2).sum()
        out[i] = float(term1) - 0.5 * pair / denom
    return _reduce(out, mean)


def skill_score(score: float, reference: float, *, perfect: float = 0.0) -> float:
    """Skill score: fractional improvement of ``score`` over a ``reference`` score.

    ``skill = (reference - score) / (reference - perfect)`` -- 1 means the forecast hit the perfect
    score, 0 means it only matched the reference (e.g. climatology), and negative means it did worse
    than the reference. Works for any negatively-oriented score (lower is better) such as those above.

    Args:
        score: the forecast's score (lower is better).
        reference: the baseline/reference forecast's score.
        perfect: the score of a perfect forecast (``0.0`` for the rules in this module).

    Returns:
        The skill score (float); ``nan`` if the reference already equals the perfect score.
    """
    denom = reference - perfect
    if denom == 0.0:
        return float("nan")
    return float((reference - score) / denom)


__all__ = [
    "log_score",
    "brier_score",
    "brier_decomposition",
    "crps_ensemble",
    "crps_gaussian",
    "interval_score",
    "winkler_score",
    "pinball_loss",
    "energy_score",
    "skill_score",
]
