"""Forward uncertainty propagation: push an input distribution through a model to output statistics.

Given input uncertainty (a Gaussian or a sampler) and a model ``f``, report the induced uncertainty on
the output -- mean, standard deviation, and quantiles. Monte Carlo is general; the unscented transform
propagates the first two moments with ``2d+1`` deterministic sigma points (exact for a linear model, a
useful low-cost approximation for mild nonlinearity). The back half of the UQ loop, after sensitivity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["propagate", "register_propagator", "unscented_transform"]


def _safe_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Cholesky of a covariance, falling back to escalating jitter only if the plain decomposition
    fails; diagonal fallback if jitter alone can't rescue it.

    Tries the UNPERTURBED matrix first: this callsite's covariance can be at any scale (the unscented
    transform's own `(d + lambda) * cov` factor can shrink it to ~1e-6 for small `alpha`), so an
    always-on jitter sized relative to a `max(1.0, ...)` floor would be a large RELATIVE perturbation
    on a small-scale matrix -- degrading precision on the common (already positive-definite) case
    instead of only kicking in for the genuinely singular/near-singular case (a fixed/zero-variance
    input dimension, perfectly correlated inputs) this exists to rescue.
    """
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        pass
    d = sigma.shape[0]
    jit = 1e-10 * max(1e-12, float(np.trace(sigma)) / d)
    for _ in range(7):
        try:
            return np.linalg.cholesky(sigma + jit * np.eye(d))
        except np.linalg.LinAlgError:
            jit *= 10.0
    return np.diag(np.sqrt(np.maximum(np.diag(sigma), 0.0)))


#: Registry of forward-propagation methods, keyed by ``method`` name. Each entry is a callable
#: ``f(func, mean, cov, *, n, quantiles, seed) -> dict`` -- the "register, don't branch" pattern
#: shared with the doe acquisition/criterion registries.
_PROPAGATORS: dict[str, Callable[..., dict[str, Any]]] = {}


def register_propagator(name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Decorator registering a propagation method under ``name`` for :func:`propagate`.

    The decorated callable receives ``(func, mean, cov, *, n, quantiles, seed)`` (``mean``/``cov``
    already coerced to float arrays) and returns the output-statistics dict.
    """

    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        _PROPAGATORS[name] = fn
        return fn

    return decorator


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
            moment propagation, mean + std only). Looked up through the propagator registry.
        quantiles: output quantiles to report (Monte Carlo only).

    Returns:
        ``{'mean', 'std', 'quantiles' (mc only), 'samples' (mc only)}``.
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    d = len(mean)
    cov = np.eye(d) if cov is None else np.atleast_2d(np.asarray(cov, dtype=float))
    try:
        propagator = _PROPAGATORS[method]
    except KeyError:
        raise ValueError(
            "unknown propagation method %r; registered methods are %s."
            % (method, ", ".join(repr(name) for name in sorted(_PROPAGATORS)))
        ) from None
    return propagator(func, mean, cov, n=n, quantiles=quantiles, seed=seed)


@register_propagator("unscented")
def _propagate_unscented(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n: int,
    quantiles: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    m, c = unscented_transform(func, mean, cov)
    std = np.sqrt(np.clip(np.diag(np.atleast_2d(c)), 0.0, None))
    return {"mean": m, "std": float(std[0]) if np.ndim(m) == 0 else std}


@register_propagator("montecarlo")
def _propagate_montecarlo(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n: int,
    quantiles: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
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
    if d + lam <= 0:
        raise ValueError(
            f"unscented_transform requires d + lambda > 0, got {d + lam:.6g}; "
            f"choose kappa > -d (here d={d}) so the sigma-point spread is positive."
        )
    chol = _safe_cholesky((d + lam) * cov)
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
