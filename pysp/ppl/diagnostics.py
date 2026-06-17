"""Predictive model-comparison diagnostics for the pysp PPL: WAIC and PSIS-LOO.

Both estimate the expected log pointwise predictive density (elpd) -- how well a fitted Bayesian model
predicts new data -- from the pointwise log-likelihood matrix ``loglik`` of shape ``(n_draws, n_obs)``
(the log-density of each observation under each posterior draw of the parameters).

* ``waic`` -- the Widely Applicable Information Criterion (Watanabe): ``elpd = lppd - p_waic`` with the
  effective parameter count ``p_waic`` the per-observation posterior variance of the log-likelihood.
* ``psis_loo`` -- Pareto-Smoothed Importance-Sampling Leave-One-Out cross-validation (Vehtari, Gelman &
  Gabry 2017). Importance-reweights the full-data posterior to each leave-one-out posterior, smoothing
  the heavy importance-weight tail with a generalized-Pareto fit; it reports the diagnostic shape
  ``khat`` (values above ~0.7 flag unreliable estimates).

Both return results on the deviance scale (``waic``/``loo`` = ``-2 * elpd``, lower is better) with a
standard error, matching the conventions of Stan / ArviZ / the R ``loo`` package.
"""

from __future__ import annotations

import numpy as np


def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    if axis is None:
        return float(np.squeeze(out))
    return np.squeeze(out, axis=axis)


def _lppd_pointwise(loglik: np.ndarray) -> np.ndarray:
    """Log pointwise predictive density per observation: log mean_s exp(loglik[s, i])."""
    s = loglik.shape[0]
    return _logsumexp(loglik, axis=0) - np.log(s)


def waic(loglik: np.ndarray) -> dict:
    """Return the WAIC of a ``(n_draws, n_obs)`` pointwise log-likelihood matrix."""
    loglik = np.atleast_2d(np.asarray(loglik, dtype=float))
    s, n = loglik.shape
    lppd_i = _lppd_pointwise(loglik)
    p_waic_i = np.var(loglik, axis=0, ddof=1) if s > 1 else np.zeros(n)
    elpd_i = lppd_i - p_waic_i
    elpd = float(np.sum(elpd_i))
    se = float(2.0 * np.sqrt(n * np.var(elpd_i, ddof=1))) if n > 1 else float("nan")
    return {
        "elpd_waic": elpd,
        "p_waic": float(np.sum(p_waic_i)),
        "waic": -2.0 * elpd,
        "se": se,
        "n_draws": s,
        "pointwise": elpd_i,
    }


def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Fit a generalized Pareto distribution to positive exceedances ``x`` (Zhang & Stephens 2009)."""
    x = np.sort(x)
    n = len(x)
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= prior_bs * x[int(np.ceil(n / 4.0)) - 1]
    bs += 1.0 / x[-1]
    ks = np.mean(np.log1p(-bs[:, None] * x[None, :]), axis=1)
    log_lik = n * (np.log(-bs / ks) - ks - 1.0)
    weights = np.exp(log_lik - _logsumexp(log_lik))
    b = float(np.sum(bs * weights))
    k = float(np.mean(np.log1p(-b * x)))
    sigma = -k / b
    # weakly informative prior shrinking k toward 0.5
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    return k, sigma


def _gpd_quantile(p: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """Quantile function of the generalized Pareto distribution (shape k, scale sigma)."""
    if abs(k) < 1.0e-8:
        return sigma * -np.log1p(-p)
    return sigma / k * (np.power(1.0 - p, -k) - 1.0)


def _psis_smooth(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smooth a 1-D array of log importance weights; return (smoothed log weights, khat)."""
    lw = np.asarray(log_weights, dtype=float).copy()
    s = len(lw)
    lw -= np.max(lw)  # stabilize
    m = int(min(0.2 * s, 3.0 * np.sqrt(s)))
    if m < 5 or s < 25:
        return lw, float("nan")  # too few draws to estimate a tail reliably

    order = np.argsort(lw)
    tail_idx = order[-m:]
    cutoff = lw[order[-m - 1]]  # log threshold below the tail
    exceedances = np.exp(lw[tail_idx]) - np.exp(cutoff)
    if np.any(exceedances <= 0.0):
        return lw, float("nan")

    k, sigma = _gpdfit(exceedances)
    # replace the tail by the smoothed expected order statistics from the fitted GPD
    probs = (np.arange(m) + 0.5) / m
    smoothed = np.log(_gpd_quantile(probs, k, sigma) + np.exp(cutoff))
    # tail_idx is ordered by ascending lw; smoothed is ascending -> assign in that order
    lw[tail_idx] = np.minimum(smoothed, 0.0)  # truncate at the (stabilized) max weight of 0
    return lw, k


def psis_loo(loglik: np.ndarray) -> dict:
    """Return PSIS-LOO of a ``(n_draws, n_obs)`` pointwise log-likelihood matrix."""
    loglik = np.atleast_2d(np.asarray(loglik, dtype=float))
    s, n = loglik.shape
    if s < 2:
        elpd_i = loglik[0].copy()
        return {
            "elpd_loo": float(np.sum(elpd_i)),
            "p_loo": 0.0,
            "loo": -2.0 * float(np.sum(elpd_i)),
            "se": float("nan"),
            "khat_max": float("nan"),
            "n_draws": s,
            "pointwise": elpd_i,
        }

    elpd_i = np.empty(n)
    khat = np.empty(n)
    for i in range(n):
        ll = loglik[:, i]
        lw, k = _psis_smooth(-ll)  # LOO importance weights are proportional to 1 / p(y_i | theta)
        elpd_i[i] = _logsumexp(lw + ll) - _logsumexp(lw)
        khat[i] = k

    elpd = float(np.sum(elpd_i))
    p_loo = float(np.sum(_lppd_pointwise(loglik)) - elpd)
    se = float(2.0 * np.sqrt(n * np.var(elpd_i, ddof=1))) if n > 1 else float("nan")
    return {
        "elpd_loo": elpd,
        "p_loo": p_loo,
        "loo": -2.0 * elpd,
        "se": se,
        "khat_max": float(np.nanmax(khat)) if np.any(np.isfinite(khat)) else float("nan"),
        "n_draws": s,
        "pointwise": elpd_i,
    }
