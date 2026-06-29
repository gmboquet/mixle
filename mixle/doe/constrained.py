"""Constrained Bayesian optimization over a bounded input space (WS-E).

Extends the unconstrained GP-BO loop (:mod:`mixle.doe.bayesopt`) to problems with black-box
inequality constraints ``c_k(x) <= 0``. The objective and each constraint get their own GP
surrogate; candidates are scored by a **feasibility-weighted acquisition**

    merit(x) = acquisition(x) * prod_k P(c_k(x) <= 0)

where the per-constraint feasibility probability comes from that constraint's GP posterior,
``P(c_k <= 0) = Phi(-mean_k / std_k)`` (Gardner et al., 2014). The acquisition's incumbent is the
best *feasible* objective seen so far; until a feasible point is found the search is driven by
feasibility alone, then switches to improving the objective within the feasible region.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import ndtr

from mixle.doe._contracts import Acquisition, Surrogate
from mixle.doe.bayesopt import BayesOptResult, _fit_surrogate, _get_acquisition, _validate_xy
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


@dataclass(frozen=True)
class ConstrainedBayesOptResult(BayesOptResult):
    """Outcome of a constrained Bayesian-optimization run.

    ``c`` holds the ``(N, K)`` observed constraint values (feasible rows have all entries ``<= 0``)
    and ``feasible`` is the corresponding boolean mask. ``best_x`` / ``best_y`` are the best feasible
    point; if no feasible point was found they fall back to the least-infeasible observation.
    """

    c: np.ndarray
    feasible: np.ndarray


def probability_of_feasibility(mean: Any, std: Any) -> np.ndarray:
    """Return the per-point probability that all constraints are satisfied (``c_k <= 0``).

    ``mean`` and ``std`` are ``(n, K)`` posterior predictive moments of the ``K`` constraint
    surrogates. Returns an ``(n,)`` array, the product over constraints of ``Phi(-mean_k / std_k)``.
    Where a constraint's ``std`` is zero the feasibility is deterministic (1.0 if ``mean <= 0``).
    """
    mean = np.atleast_2d(np.asarray(mean, dtype=np.float64))
    std = np.atleast_2d(np.asarray(std, dtype=np.float64))
    pf = np.ones(mean.shape[0], dtype=np.float64)
    for k in range(mean.shape[1]):
        mk = mean[:, k]
        sk = std[:, k]
        pk = np.where(mk <= 0.0, 1.0, 0.0)
        pos = sk > 1.0e-12
        pk[pos] = ndtr(-mk[pos] / sk[pos])
        pf = pf * pk
    return pf


def _predict_std(gp: Surrogate, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and standard deviation of ``gp`` at ``candidates``."""
    mean, cov = gp.predict(x, y, candidates, return_cov=True)
    std = np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=np.float64)), 0.0, None))
    return np.asarray(mean, dtype=np.float64), std


def _best_feasible(y: np.ndarray, c: np.ndarray, maximize: bool = False) -> tuple[int, np.ndarray]:
    """Return (index of incumbent, feasibility mask). Incumbent = best feasible, else least-infeasible."""
    feasible = np.all(c <= 0.0, axis=1)
    if np.any(feasible):
        masked = np.where(feasible, y, -np.inf if maximize else np.inf)
        idx = int(np.argmax(masked) if maximize else np.argmin(masked))
    else:
        violation = np.sum(np.maximum(c, 0.0), axis=1)
        idx = int(np.argmin(violation))
    return idx, feasible


def propose_next_constrained(
    x: Any,
    y: Any,
    c: Any,
    bounds: Bounds,
    n_candidates: int = 512,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    return_acquisition: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Propose the next point under inequality constraints ``c_k(x) <= 0``.

    Fits a GP to the objective ``(x, y)`` and one GP per constraint column of ``c`` (an ``(N, K)``
    array), then maximizes the feasibility-weighted acquisition over ``n_candidates`` Latin-hypercube
    points. Until a feasible observation exists the acquisition factor is held at 1 so the search
    targets feasibility; afterwards the incumbent is the best feasible objective. Returns the chosen
    ``(d,)`` point, optionally with its merit.
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    x, y = _validate_xy(x, y)
    c = np.atleast_2d(np.asarray(c, dtype=np.float64))
    if c.shape[0] != x.shape[0]:
        raise ValueError("c must have one row of constraint values per observation.")
    acq_fn = _get_acquisition(acq)
    kw = {"xi": xi, **(acq_kwargs or {})}

    candidates = latin_hypercube(b, n_candidates, rng)

    obj_gp = _fit_surrogate(x, y, None, fit_kwargs)
    mean, std = _predict_std(obj_gp, x, y, candidates)

    idx, feasible = _best_feasible(y, c, maximize=maximize)
    if np.any(feasible):
        best = float(y[idx])
        acq_vals = np.asarray(acq_fn(mean, std, best, maximize=maximize, **kw), dtype=np.float64)
    else:
        # No feasible point yet: drive purely by probability of feasibility.
        acq_vals = np.ones(candidates.shape[0], dtype=np.float64)

    c_mean = np.empty((candidates.shape[0], c.shape[1]), dtype=np.float64)
    c_std = np.empty_like(c_mean)
    for k in range(c.shape[1]):
        ck = c[:, k]
        gp_k = _fit_surrogate(x, ck, None, fit_kwargs)
        c_mean[:, k], c_std[:, k] = _predict_std(gp_k, x, ck, candidates)

    merit = acq_vals * probability_of_feasibility(c_mean, c_std)
    pick = int(np.argmax(merit))
    if return_acquisition:
        return candidates[pick], float(merit[pick])
    return candidates[pick]


def constrained_minimize(
    objective: Callable[[np.ndarray], float],
    constraints: Sequence[Callable[[np.ndarray], float]],
    bounds: Bounds,
    n_init: int = 5,
    n_iter: int = 15,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    n_candidates: int = 512,
    fit_kwargs: dict[str, Any] | None = None,
) -> ConstrainedBayesOptResult:
    """Constrained GP Bayesian optimization of ``objective`` subject to ``constraints`` over ``bounds``.

    Each callable in ``constraints`` maps a ``(d,)`` point to a scalar that is feasible when ``<= 0``.
    Seeds with an ``n_init``-point Latin-hypercube design, then runs ``n_iter`` feasibility-weighted
    acquisition steps. Minimizes the objective by default; returns the best feasible point (or the
    least-infeasible one if none feasible) along with the full evaluation history.
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    if n_init <= 0:
        raise ValueError("n_init must be positive.")
    if len(constraints) == 0:
        raise ValueError("constrained_minimize requires at least one constraint; use minimize otherwise.")

    def eval_c(point: np.ndarray) -> list[float]:
        return [float(con(point)) for con in constraints]

    x = latin_hypercube(b, n_init, rng)
    y = np.array([float(objective(np.asarray(row, dtype=np.float64))) for row in x], dtype=np.float64)
    c = np.array([eval_c(np.asarray(row, dtype=np.float64)) for row in x], dtype=np.float64)

    for _ in range(int(n_iter)):
        nxt = np.asarray(
            propose_next_constrained(
                x,
                y,
                c,
                b,
                n_candidates=n_candidates,
                seed=rng,
                maximize=maximize,
                xi=xi,
                acq=acq,
                acq_kwargs=acq_kwargs,
                fit_kwargs=fit_kwargs,
            ),
            dtype=np.float64,
        )
        x = np.vstack([x, nxt[None, :]])
        y = np.append(y, float(objective(nxt)))
        c = np.vstack([c, np.asarray(eval_c(nxt), dtype=np.float64)[None, :]])

    idx, feasible = _best_feasible(y, c, maximize=maximize)
    return ConstrainedBayesOptResult(best_x=x[idx], best_y=float(y[idx]), x=x, y=y, c=c, feasible=feasible)


__all__: Sequence[str] = [
    "ConstrainedBayesOptResult",
    "probability_of_feasibility",
    "propose_next_constrained",
    "constrained_minimize",
]
