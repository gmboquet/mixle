"""Public MCMC convergence diagnostics over plain arrays.

These functions are deliberately free of any ``MCMCResult`` / ``_Slot`` dependency: they
operate on ordinary NumPy arrays so the draws of *any* sampler (pysp's or an external one)
can be diagnosed. They are the generic cores behind ``pysp.inference.mcmc.gelman_rubin`` and
``MCMCResult.effective_sample_size``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
