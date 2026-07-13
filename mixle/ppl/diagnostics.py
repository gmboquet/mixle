"""Predictive model-comparison diagnostics for the mixle PPL: WAIC and PSIS-LOO.

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

from collections.abc import Sequence

import numpy as np

from mixle.utils.special import logsumexp as _logsumexp


def _lppd_pointwise(loglik: np.ndarray) -> np.ndarray:
    """Log pointwise predictive density per observation: log mean_s exp(loglik[s, i])."""
    s = loglik.shape[0]
    return _logsumexp(loglik, axis=0) - np.log(s)


# ------------------------------------------- convergence diagnostics (Vehtari et al. 2021, Bayesian Analysis)
# Rank-normalized split-R-hat and bulk/tail effective sample size -- the modern Stan/ArviZ standard. The
# inputs are an ``(n_chains, n_draws)`` array of draws for one scalar parameter.


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance of a 1-D series at all lags via FFT."""
    n = x.size
    c = x - x.mean()
    m = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(c, n=m)
    ac = np.fft.irfft(f * np.conjugate(f), n=m)[:n].real
    return ac / n


def _ess_chains(x: np.ndarray) -> float:
    """Stan effective sample size for one parameter, ``x`` of shape ``(n_chains, n_draws)``."""
    m, n = x.shape
    if n < 4:
        return float(m * n)
    acov = np.array([_autocov(x[c]) for c in range(m)])
    mean_acov = acov.mean(axis=0)
    chain_var = acov[:, 0] * n / (n - 1.0)
    w = float(chain_var.mean())
    if not np.isfinite(w) or w <= 0:
        return float(m * n)
    b = n * float(np.var(x.mean(axis=1), ddof=1)) if m > 1 else 0.0
    var_plus = (n - 1.0) / n * w + b / n
    rho = 1.0 - (w - mean_acov) / var_plus  # rho[0] == 1
    # Geyer initial monotone positive sequence on paired autocorrelations.
    pairs = []
    k = 0
    while 2 * k + 2 < n:
        p = rho[2 * k + 1] + rho[2 * k + 2]
        if p < 0:
            break
        pairs.append(p)
        k += 1
    for i in range(1, len(pairs)):
        pairs[i] = min(pairs[i], pairs[i - 1])  # enforce monotone decreasing
    tau = max(1.0 + 2.0 * float(sum(pairs)), 1.0)
    return float(m * n / tau)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Blom rank-normalization to normal scores: ``Phi^{-1}((rank - 3/8) / (N - 1/4))``."""
    from scipy.stats import norm, rankdata

    r = rankdata(x).reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size - 0.25))


def _classic_rhat(x: np.ndarray) -> float:
    """Potential scale reduction from chains ``x`` of shape ``(n_chains, n_draws)``."""
    m, n = x.shape
    if m < 2 or n < 2:
        return float("nan")
    w = float(np.var(x, axis=1, ddof=1).mean())
    b = n * float(np.var(x.mean(axis=1), ddof=1))
    if w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1.0) / n * w + b / n) / w))


def split_rhat(draws: np.ndarray) -> float:
    """Rank-normalized split-R-hat for one parameter (``draws`` is ``(n_chains, n_draws)``).

    Splits each chain in half (catching within-chain non-stationarity), rank-normalizes, then takes the
    potential scale reduction. Values within ~0.01 of 1.0 indicate convergence; > 1.01 is a warning.
    """
    x = np.atleast_2d(np.asarray(draws, dtype=float))
    half = x.shape[1] // 2
    if half < 2:
        return float("nan")
    split = np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)
    return _classic_rhat(_rank_normalize(split).reshape(split.shape))


def bulk_ess(draws: np.ndarray) -> float:
    """Bulk effective sample size: ESS of the rank-normalized draws (efficiency in the distribution body)."""
    x = np.atleast_2d(np.asarray(draws, dtype=float))
    return _ess_chains(_rank_normalize(x).reshape(x.shape))


def tail_ess(draws: np.ndarray) -> float:
    """Tail effective sample size: the smaller of the 5% and 95% quantile-indicator ESS (tail efficiency)."""
    x = np.atleast_2d(np.asarray(draws, dtype=float))
    q05, q95 = np.quantile(x, 0.05), np.quantile(x, 0.95)
    lower = _ess_chains((x <= q05).astype(float))
    upper = _ess_chains((x >= q95).astype(float))
    return float(min(lower, upper))


def convergence_diagnostics(draws: np.ndarray) -> dict:
    """Return ``{'split_rhat', 'bulk_ess', 'tail_ess'}`` for one parameter's ``(n_chains, n_draws)`` draws."""
    return {"split_rhat": split_rhat(draws), "bulk_ess": bulk_ess(draws), "tail_ess": tail_ess(draws)}


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


def loo(loglik: np.ndarray) -> dict:
    """PSIS-LOO under its conventional short name (see :func:`psis_loo`)."""
    return psis_loo(loglik)


def loo_stacking_weights(pointwise_lpd: np.ndarray, iters: int = 2000, tol: float = 1.0e-10) -> np.ndarray:
    """Return LOO stacking weights (Yao, Vehtari, Simpson & Gelman, 2018).

    ``pointwise_lpd`` is an ``(n_obs, K)`` matrix of per-model pointwise LOO log-predictive
    densities (each column is ``psis_loo(model_k)["pointwise"]``). The returned simplex weights
    ``w`` maximize the LOO log-score of the weighted predictive distribution,
    ``sum_i log(sum_k w_k * exp(lpd_ik))``. This is concave in ``w`` and solved here by the standard
    mixture-weight EM update (no external optimizer), which respects the simplex by construction.
    """
    lpd = np.atleast_2d(np.asarray(pointwise_lpd, dtype=float))
    n, k = lpd.shape
    if k == 1:
        return np.ones(1)
    shift = lpd.max(axis=1, keepdims=True)
    p = np.exp(lpd - shift)  # per-row rescaled predictive densities (the row constant cancels)
    w = np.full(k, 1.0 / k)
    prev = -np.inf
    for _ in range(int(iters)):
        num = w[None, :] * p
        denom = num.sum(axis=1)
        score = float(np.sum(np.log(denom) + shift[:, 0]))
        w = (num / denom[:, None]).mean(axis=0)
        if score - prev < tol * (abs(prev) + 1.0):
            break
        prev = score
    return w


def loo_stack(logliks: Sequence[np.ndarray]) -> dict:
    """Stack K candidate models by LOO predictive performance.

    ``logliks`` is a sequence of ``(n_draws_k, n_obs)`` pointwise log-likelihood matrices over the
    same, aligned observations. Returns the stacking ``weights``, the ``(n_obs, K)`` per-model
    pointwise LOO densities, each model's ``elpd_loo``, and the ``stacked_elpd_loo`` of the weighted
    predictive (which is >= the best single-model elpd_loo, since a one-hot weight is feasible).
    """
    pointwise = np.column_stack([psis_loo(np.asarray(ll, dtype=float))["pointwise"] for ll in logliks])
    weights = loo_stacking_weights(pointwise)
    shift = pointwise.max(axis=1, keepdims=True)
    stacked = float(np.sum(np.log((weights[None, :] * np.exp(pointwise - shift)).sum(axis=1)) + shift[:, 0]))
    return {
        "weights": weights,
        "pointwise": pointwise,
        "model_elpd_loo": [float(pointwise[:, j].sum()) for j in range(pointwise.shape[1])],
        "stacked_elpd_loo": stacked,
    }
