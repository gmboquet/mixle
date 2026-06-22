"""Multi-objective Bayesian optimization via ParEGO (WS-E).

Optimizes several competing objectives at once, returning the Pareto-optimal set rather than a single
point. Uses the ParEGO scheme (Knowles, 2006): at each step the observed objective vectors are
min-max normalized and collapsed to a single scalar by a randomly drawn **augmented Tchebycheff**
weighting

    s_i = max_m ( w_m * yhat_{i,m} ) + rho * sum_m ( w_m * yhat_{i,m} ),   w ~ uniform on the simplex,

and the standard single-objective GP-EI step (:func:`pysp.doe.bayesopt.propose_next`) proposes the
next point for that scalarization. Sweeping the random weights across iterations traces out the whole
Pareto front with one surrogate. All objectives are minimized by convention.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.doe.bayesopt import OptimizationResult, propose_next
from pysp.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


@dataclass(frozen=True)
class MultiObjectiveResult(OptimizationResult):
    """Outcome of a multi-objective Bayesian-optimization run.

    ``y`` is the ``(N, M)`` matrix of observed objective vectors (all minimized); ``pareto_mask`` flags
    the non-dominated rows, and ``pareto_x`` / ``pareto_y`` are those points and their objective vectors.
    """

    pareto_mask: np.ndarray
    pareto_x: np.ndarray
    pareto_y: np.ndarray


def pareto_mask(y: Any) -> np.ndarray:
    """Return a boolean mask of the non-dominated rows of ``y`` (an ``(N, M)`` minimization objective).

    Row ``i`` is dominated when some other row is ``<=`` it on every objective and strictly ``<`` on at
    least one; the mask is ``True`` for the rows that survive (the Pareto-optimal set).
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n = y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # A point dominates i if it is <= on all objectives and < on some.
        dominated = np.all(y <= y[i], axis=1) & np.any(y < y[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def _scalarize(y: np.ndarray, weights: np.ndarray, rho: float) -> np.ndarray:
    """Augmented Tchebycheff scalarization of min-max normalized objective vectors."""
    lo = y.min(axis=0)
    span = np.where(y.max(axis=0) - lo > 1.0e-12, y.max(axis=0) - lo, 1.0)
    yhat = (y - lo) / span
    weighted = yhat * weights
    return np.max(weighted, axis=1) + rho * np.sum(weighted, axis=1)


def _draw_weights(m: int, rng: RandomState) -> np.ndarray:
    """Draw a weight vector uniformly from the ``m``-simplex (normalized exponentials)."""
    w = rng.exponential(size=m)
    total = float(np.sum(w))
    return w / total if total > 0 else np.full(m, 1.0 / m)


def multi_minimize(
    objectives: Sequence[Callable[[np.ndarray], float]],
    bounds: Bounds,
    n_init: int = 10,
    n_iter: int = 20,
    seed: int | RandomState | None = None,
    *,
    rho: float = 0.05,
    n_candidates: int = 512,
    fit_kwargs: dict[str, Any] | None = None,
) -> MultiObjectiveResult:
    """Multi-objective GP Bayesian optimization of ``objectives`` over ``bounds`` (ParEGO).

    Each callable in ``objectives`` maps a ``(d,)`` point to a scalar; all are **minimized**. Seeds with
    an ``n_init``-point Latin-hypercube design, then runs ``n_iter`` steps, each drawing a random
    Tchebycheff weighting, scalarizing the observed objectives, and taking one GP-EI step on that
    scalar. Returns the full evaluation history and the Pareto-optimal subset.
    """
    if len(objectives) < 2:
        raise ValueError("multi_minimize requires at least two objectives; use minimize otherwise.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    if n_init <= 0:
        raise ValueError("n_init must be positive.")
    m = len(objectives)

    def eval_objs(point: np.ndarray) -> list[float]:
        return [float(obj(point)) for obj in objectives]

    x = latin_hypercube(b, n_init, rng)
    y = np.array([eval_objs(np.asarray(row, dtype=np.float64)) for row in x], dtype=np.float64)

    for _ in range(int(n_iter)):
        weights = _draw_weights(m, rng)
        scalar = _scalarize(y, weights, rho)
        nxt = np.asarray(
            propose_next(x, scalar, b, n_candidates=n_candidates, seed=rng, fit_kwargs=fit_kwargs),
            dtype=np.float64,
        )
        x = np.vstack([x, nxt[None, :]])
        y = np.vstack([y, np.asarray(eval_objs(nxt), dtype=np.float64)[None, :]])

    mask = pareto_mask(y)
    return MultiObjectiveResult(x=x, y=y, pareto_mask=mask, pareto_x=x[mask], pareto_y=y[mask])


__all__: Sequence[str] = [
    "MultiObjectiveResult",
    "pareto_mask",
    "multi_minimize",
]
