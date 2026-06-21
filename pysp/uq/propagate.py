"""Forward uncertainty propagation: push an input distribution through a model to output statistics.

Given input uncertainty (a Gaussian or a sampler) and a model ``f``, report the induced uncertainty on
the output -- mean, standard deviation, and quantiles. Monte Carlo is general; the unscented transform
propagates the first two moments with ``2d+1`` deterministic sigma points (exact for a linear model, a
good cheap approximation for mild nonlinearity). The back half of the UQ loop, after sensitivity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["propagate", "unscented_transform"]


def propagate(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    n: int = 10000,
    method: str = "montecarlo",
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    seed: int = 0,
) -> dict[str, Any]:
    """Propagate a Gaussian input ``N(mean, cov)`` through ``func`` to output statistics.

    Args:
        func: a *vectorized* model ``f(X) -> y`` mapping ``(m, d)`` inputs to ``(m,)`` (or ``(m, k)``)
            outputs.
        mean: length-``d`` input mean. ``cov``: ``(d, d)`` input covariance (defaults to identity).
        n: Monte Carlo sample size (ignored by the unscented method).
        method: ``'montecarlo'`` (sample + summarize, gives quantiles) or ``'unscented'`` (sigma-point
            moment propagation, mean + std only).
        quantiles: output quantiles to report (Monte Carlo only).

    Returns:
        ``{'mean', 'std', 'quantiles' (mc only), 'samples' (mc only)}``.
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    d = len(mean)
    cov = np.eye(d) if cov is None else np.atleast_2d(np.asarray(cov, dtype=float))
    if method == "unscented":
        m, c = unscented_transform(func, mean, cov)
        std = np.sqrt(np.clip(np.diag(np.atleast_2d(c)), 0.0, None))
        return {"mean": m, "std": float(std[0]) if np.ndim(m) == 0 else std}
    if method != "montecarlo":
        raise ValueError("method must be 'montecarlo' or 'unscented'.")
    rng = np.random.RandomState(seed)
    x = rng.multivariate_normal(mean, cov, size=n)
    y = np.asarray(func(x), dtype=float)
    return {
        "mean": y.mean(axis=0),
        "std": y.std(axis=0),
        "quantiles": {q: np.quantile(y, q, axis=0) for q in quantiles},
        "samples": y,
    }


def unscented_transform(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate ``(mean, cov)`` through ``func`` with the unscented (sigma-point) transform.

    Returns the output ``(mean, covariance)``. Exact for an affine ``func``; for a nonlinear one it
    captures the mean and covariance to second order with only ``2d+1`` evaluations.
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    d = len(mean)
    lam = alpha**2 * (d + kappa) - d
    chol = np.linalg.cholesky((d + lam) * cov)
    sigma = np.vstack([mean, mean + chol.T, mean - chol.T])  # 2d+1 sigma points
    wm = np.full(2 * d + 1, 1.0 / (2.0 * (d + lam)))
    wc = wm.copy()
    wm[0] = lam / (d + lam)
    wc[0] = lam / (d + lam) + (1.0 - alpha**2 + beta)
    y = np.atleast_2d(np.asarray(func(sigma), dtype=float).reshape(2 * d + 1, -1))
    y_mean = wm @ y
    dy = y - y_mean
    y_cov = (wc[:, None] * dy).T @ dy
    out_dim = y.shape[1]
    return (y_mean[0], float(y_cov[0, 0])) if out_dim == 1 else (y_mean, y_cov)
