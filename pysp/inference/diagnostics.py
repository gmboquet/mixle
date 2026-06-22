"""Public MCMC convergence diagnostics over plain arrays.

These functions are deliberately free of any ``MCMCResult`` / ``_Slot`` dependency: they
operate on ordinary NumPy arrays so the draws of *any* sampler (pysp's or an external one)
can be diagnosed. They are the generic cores behind ``pysp.inference.mcmc.gelman_rubin`` and
``MCMCResult.effective_sample_size``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _as_chains(chains: Any) -> np.ndarray:
    """Coerce input to a ``(n_chains, n_draws, d)`` float array.

    Accepts ``(n_chains, n_draws)`` (scalar parameter) or ``(n_chains, n_draws, d)``.
    """
    arr = np.asarray(chains, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("chains must have shape (n_chains, n_draws) or (n_chains, n_draws, d).")
    return arr


def rhat(chains: Any) -> np.ndarray:
    """Gelman-Rubin potential scale reduction factor (R-hat) per parameter dimension.

    R-hat compares the variance *between* chains to the variance *within* chains. Values near
    1.0 indicate the chains have mixed and are sampling a common target; values above ~1.01-1.1
    flag non-convergence. Standard multi-chain check (Gelman & Rubin 1992).

    Args:
        chains: array shaped ``(n_chains, n_draws)`` or ``(n_chains, n_draws, d)``. Needs at
            least two chains and two draws.

    Returns:
        A length-``d`` array of R-hat values (one per parameter).
    """
    arr = _as_chains(chains)
    m, n, d = arr.shape
    if m < 2 or n < 2:
        return np.full(d, np.nan)
    chain_means = arr.mean(axis=1)  # (m, d)
    w = arr.var(axis=1, ddof=1).mean(axis=0)  # within-chain variance (d,)
    b = n * chain_means.var(axis=0, ddof=1)  # between-chain variance (d,)
    var_hat = (n - 1) / n * w + b / n
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(np.where(w > 0.0, np.maximum(var_hat / np.where(w > 0.0, w, 1.0), 0.0), 1.0))
    return np.where(w > 0.0, out, 1.0)


def ess(samples: Any, max_lag: int | None = None) -> np.ndarray:
    """Effective sample size per parameter from positive autocorrelation lags.

    Accepts a single chain ``(n_draws,)`` / ``(n_draws, d)`` or a stack of chains
    ``(n_chains, n_draws, d)`` (the chains are pooled into one autocorrelation estimate after
    centering each chain by its own mean). Uses the initial-positive-sequence truncation
    (Geyer): sum autocorrelations until the first non-positive lag.

    Returns:
        A length-``d`` array of ESS values.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :, None]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("samples must have shape (n_draws,), (n_draws, d), or (n_chains, n_draws, d).")
    m, n, d = arr.shape
    if n <= 1:
        return np.full(d, float(n) * m)
    centered = arr - arr.mean(axis=1, keepdims=True)  # center each chain
    var = np.mean(centered * centered, axis=(0, 1))  # (d,)
    n_total = m * n
    out = np.empty(d, dtype=float)
    nonzero = var > 0.0
    out[~nonzero] = float(n_total)
    if not np.any(nonzero):
        return out
    lag_limit = n - 1 if max_lag is None else min(int(max_lag), n - 1)
    tau = np.ones(int(nonzero.sum()), dtype=float)
    cvar = var[nonzero]
    cc = centered[:, :, nonzero]
    for lag in range(1, lag_limit + 1):
        rho = np.mean(cc[:, :-lag] * cc[:, lag:], axis=(0, 1)) / cvar
        positive = rho > 0.0
        if not np.any(positive):
            break
        tau += 2.0 * np.where(positive, rho, 0.0)
    out[nonzero] = np.maximum(1.0, n_total / tau)
    return out


def _split_chains(arr: np.ndarray) -> np.ndarray:
    """Split each chain into first/second halves -> ``(2*n_chains, n_draws//2, d)`` (drops an odd tail draw)."""
    m, n, d = arr.shape
    h = n // 2
    if h < 1:
        return arr
    return np.concatenate([arr[:, :h, :], arr[:, h : 2 * h, :]], axis=0)


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Pooled rank-normal-score transform per parameter (Blom plotting positions)."""
    m, n, d = arr.shape
    total = m * n
    out = np.empty_like(arr)
    for k in range(d):
        ranks = rankdata(arr[:, :, k].ravel(), method="average")
        out[:, :, k] = ndtri((ranks - 0.375) / (total - 0.25)).reshape(m, n)
    return out


def split_rhat(chains: Any) -> np.ndarray:
    """Rank-normalized split-R-hat (Vehtari et al. 2021) -- the robust, recommended R-hat.

    Splits each chain in half (doubling the chain count, so within-chain non-stationarity is caught),
    rank-normalizes the pooled draws (robust to heavy tails / non-normality), then applies the
    Gelman-Rubin R-hat. Convergence is typically declared at ``< 1.01``.
    """
    return rhat(_rank_normalize(_split_chains(_as_chains(chains))))


def ess_bulk(chains: Any) -> np.ndarray:
    """Bulk effective sample size: ESS of the rank-normalized split chains (mixing in the distribution body)."""
    return ess(_rank_normalize(_split_chains(_as_chains(chains))))


def ess_tail(chains: Any, prob: float = 0.05) -> np.ndarray:
    """Tail effective sample size: the smaller ESS of the lower-/upper-``prob`` tail indicators.

    Tail quantiles mix more slowly than the bulk; tail-ESS is ``min(ESS[1(x <= q_prob)], ESS[1(x >=
    q_{1-prob})])`` over the split chains (Vehtari et al. 2021), surfacing poor tail exploration that
    bulk-ESS misses.
    """
    s = _split_chains(_as_chains(chains))
    m, n, d = s.shape
    out = np.empty(d, dtype=float)
    for k in range(d):
        flat = s[:, :, k]
        q_lo, q_hi = np.quantile(flat, prob), np.quantile(flat, 1.0 - prob)
        lo = ess((flat <= q_lo).astype(float)[:, :, None])[0]
        hi = ess((flat >= q_hi).astype(float)[:, :, None])[0]
        out[k] = min(float(lo), float(hi))
    return out


def _fold(arr: np.ndarray) -> np.ndarray:
    """Fold each parameter about its pooled median: ``z = |x - median(x)|`` (surfaces scale drift)."""
    out = np.empty_like(arr)
    for k in range(arr.shape[2]):
        out[:, :, k] = np.abs(arr[:, :, k] - np.median(arr[:, :, k]))
    return out


def folded_split_rhat(chains: Any) -> np.ndarray:
    """Folded rank-normalized split-R-hat (Vehtari et al. 2021): split-R-hat on ``|x - median(x)|``.

    The plain :func:`split_rhat` compares chain *locations*; folding about the median makes it compare
    chain *scales/tails*, catching the case where chains share a mean but differ in spread. Use together
    with :func:`split_rhat` (or :func:`rhat_max`).
    """
    return rhat(_rank_normalize(_split_chains(_fold(_as_chains(chains)))))


def rhat_max(chains: Any) -> np.ndarray:
    """The recommended convergence R-hat: ``max(split_rhat, folded_split_rhat)`` (Vehtari et al. 2021).

    A single number that flags non-convergence in either the location (bulk) or the scale (folded) of
    the chains; declare convergence at ``< 1.01``.
    """
    arr = _as_chains(chains)
    return np.maximum(split_rhat(arr), folded_split_rhat(arr))


def mcse_mean(chains: Any) -> np.ndarray:
    """Monte Carlo standard error of the posterior mean: ``sd(x) / sqrt(ESS)`` per parameter.

    The sampling error in the estimated mean from autocorrelated draws -- the posterior standard
    deviation deflated by the (autocorrelation-based) effective sample size of the raw chains.
    """
    arr = _as_chains(chains)
    m, n, d = arr.shape
    sd = arr.reshape(m * n, d).std(axis=0, ddof=1)
    return sd / np.sqrt(ess(arr))
