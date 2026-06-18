"""Gaussian-process Bayesian optimization over a bounded input space (WS-E).

A minimal sequential model-based optimization loop: fit a GP surrogate to the observed points,
score Latin-hypercube candidates with the expected-improvement (EI) acquisition, and evaluate the
best candidate next. Reuses :class:`pysp.models.gaussian_process.GaussianProcessRegressor` (torch)
as the surrogate; only ``expected_improvement`` is torch-free.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import ndtr

from pysp.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


def expected_improvement(mean: Any, std: Any, best: float, xi: float = 0.0, *, maximize: bool = False) -> np.ndarray:
    """Return the expected-improvement acquisition at points with surrogate ``mean`` and ``std``.

    For minimization the improvement over the incumbent ``best`` is ``best - mean - xi``; for
    maximization it is ``mean - best - xi``. ``xi >= 0`` trades exploration for exploitation.
    Points with zero predictive ``std`` get zero EI.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    improve = (mean - best - xi) if maximize else (best - mean - xi)
    ei = np.zeros_like(std)
    pos = std > 1.0e-12
    z = np.zeros_like(std)
    z[pos] = improve[pos] / std[pos]
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei[pos] = improve[pos] * ndtr(z[pos]) + std[pos] * pdf[pos]
    return np.maximum(ei, 0.0)


@dataclass(frozen=True)
class BayesOptResult:
    """Outcome of a Bayesian-optimization run."""

    best_x: np.ndarray
    best_y: float
    x: np.ndarray
    y: np.ndarray


def _fit_surrogate(x: np.ndarray, y: np.ndarray, gp: Any | None, fit_kwargs: dict[str, Any] | None) -> Any:
    if gp is None:
        from pysp.models.gaussian_process import GaussianProcessRegressor

        scale = float(np.std(y)) or 1.0
        gp = GaussianProcessRegressor(lengthscale=1.0, amplitude=scale, noise=0.1 * scale + 1.0e-6)
    kwargs = {"out": None, **(fit_kwargs or {})}
    gp.fit(x, y, **kwargs)
    return gp


def propose_next(
    x: Any,
    y: Any,
    bounds: Bounds,
    n_candidates: int = 512,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    gp: Any | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    return_acquisition: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Propose the next point to evaluate by maximizing expected improvement.

    Fits a GP to ``(x, y)``, scores ``n_candidates`` Latin-hypercube points by EI, and returns the
    best candidate (a ``(d,)`` array), optionally with its EI value.
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations.")

    gp = _fit_surrogate(x, y, gp, fit_kwargs)
    candidates = latin_hypercube(b, n_candidates, rng)
    mean, cov = gp.predict(x, y, candidates, return_cov=True)
    std = np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=np.float64)), 0.0, None))
    best = float(np.max(y)) if maximize else float(np.min(y))
    ei = expected_improvement(mean, std, best, xi, maximize=maximize)
    idx = int(np.argmax(ei))
    if return_acquisition:
        return candidates[idx], float(ei[idx])
    return candidates[idx]


def minimize(
    objective: Callable[[np.ndarray], float],
    bounds: Bounds,
    n_init: int = 5,
    n_iter: int = 15,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    n_candidates: int = 512,
    fit_kwargs: dict[str, Any] | None = None,
) -> BayesOptResult:
    """Run sequential GP-EI Bayesian optimization of a scalar ``objective`` over ``bounds``.

    Seeds with an ``n_init``-point Latin-hypercube design, then runs ``n_iter`` EI-driven steps.
    Minimizes by default; set ``maximize=True`` to maximize. ``objective`` takes a ``(d,)`` point
    and returns a float.
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    if n_init <= 0:
        raise ValueError("n_init must be positive.")

    x = latin_hypercube(b, n_init, rng)
    y = np.array([float(objective(np.asarray(row, dtype=np.float64))) for row in x], dtype=np.float64)

    for _ in range(int(n_iter)):
        nxt = propose_next(
            x, y, b, n_candidates=n_candidates, seed=rng, maximize=maximize, xi=xi, fit_kwargs=fit_kwargs
        )
        nxt = np.asarray(nxt, dtype=np.float64)
        x = np.vstack([x, nxt[None, :]])
        y = np.append(y, float(objective(nxt)))

    best_idx = int(np.argmax(y)) if maximize else int(np.argmin(y))
    return BayesOptResult(best_x=x[best_idx], best_y=float(y[best_idx]), x=x, y=y)


__all__: Sequence[str] = ["expected_improvement", "propose_next", "minimize", "BayesOptResult"]
