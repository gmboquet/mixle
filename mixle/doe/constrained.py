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
from mixle.doe.bayesopt import (
    BayesOptResult,
    _fit_surrogate,
    _get_acquisition,
    _select_index,
    _validate_prediction,
    _validate_xy,
)
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, _require_exact_positive_int, latin_hypercube


@dataclass(frozen=True)
class ConstrainedBayesOptResult(BayesOptResult):
    """Outcome of a constrained Bayesian-optimization run.

    ``c`` holds the ``(N, K)`` observed constraint values (feasible rows have all entries ``<= 0``)
    and ``feasible`` is the corresponding boolean mask. ``best_x`` / ``best_y`` are the best feasible
    point; if no feasible point was found they fall back to the least-infeasible observation.
    """

    c: np.ndarray
    feasible: np.ndarray


def _as_feasibility_matrix(name: str, arr: Any) -> np.ndarray:
    """Reshape a feasibility moment array to ``(n_points, n_constraints)``.

    A scalar or 1-D array of length ``n`` means ``n`` points under a *single* constraint, so it is
    reshaped to ``(n, 1)`` -- never to ``(1, n)`` via ``np.atleast_2d``, which would silently
    misinterpret it as one point under ``n`` constraints. Already-2-D input passes through unchanged.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim >= 3:
        raise ValueError(f"{name} must be 0-, 1-, or 2-dimensional (points x constraints), got {a.ndim} dimensions.")
    if a.ndim < 2:
        a = a.reshape(-1, 1)
    return a


def probability_of_feasibility(mean: Any, std: Any) -> np.ndarray:
    """Return the per-point probability that all constraints are satisfied (``c_k <= 0``).

    ``mean`` and ``std`` are ``(n, K)`` posterior predictive moments of the ``K`` constraint
    surrogates: row ``i`` holds the ``K`` constraint moments at point ``i``. A scalar or 1-D input of
    length ``n`` is treated as ``n`` points under a single constraint and reshaped to ``(n, 1)`` -- it
    is never reinterpreted as one point under ``n`` constraints. Returns an ``(n,)`` array, the product
    over constraints of ``Phi(-mean_k / std_k)``. Where a constraint's ``std`` is zero the feasibility
    is deterministic (1.0 if ``mean <= 0``).

    Raises:
        ValueError: ``mean`` and ``std`` do not have identical shape once normalized, either is not
            finite, ``std`` is negative anywhere, or there are zero constraint columns.
    """
    mean = _as_feasibility_matrix("mean", mean)
    std = _as_feasibility_matrix("std", std)
    if mean.shape != std.shape:
        raise ValueError(f"mean and std must have identical shape, got {mean.shape} and {std.shape}.")
    if mean.shape[1] == 0:
        raise ValueError("probability_of_feasibility requires at least one constraint column.")
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean must be finite.")
    if not np.all(np.isfinite(std)):
        raise ValueError("std must be finite.")
    if np.any(std < 0.0):
        raise ValueError(f"std must be nonnegative, got a minimum of {float(std.min())!r}.")

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
    mean, cov = _validate_prediction(mean, cov, candidates.shape[0], context="constrained BO")
    assert cov is not None
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return mean, std


def _best_feasible(y: np.ndarray, c: np.ndarray, maximize: bool = False) -> tuple[int, np.ndarray]:
    """Return (index of incumbent, feasibility mask). Incumbent = best feasible, else least-infeasible."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}.")
    c = _as_feasibility_matrix("c", c)
    if c.shape[0] != y.shape[0] or c.shape[1] == 0:
        raise ValueError("c must have one nonempty constraint row per objective observation.")
    if not np.all(np.isfinite(y)):
        raise ValueError("objective observations must be finite.")
    if not np.all(np.isfinite(c)):
        raise ValueError("constraint observations must be finite.")
    if type(maximize) is not bool:
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}.")
    if y.size == 0:
        # np.argmin on an empty violation/masked array crashes with an opaque "attempt to get argmin
        # of an empty sequence" ValueError. Same documented path as bayesopt._propose_one: there is no
        # incumbent to identify yet with zero observations, so name that clearly instead of a generic
        # numpy crash. propose_next_constrained is a public, directly-callable entry point -- unlike its
        # one internal caller (task/edge.py), which already special-cases len(self.X) < 2 -- so external
        # callers can reach this with zero observations too.
        raise ValueError("cannot determine a best-feasible incumbent with zero observations; call tell() first.")
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
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    if type(maximize) is not bool:
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}.")
    x, y = _validate_xy(x, y)
    if y.size == 0:
        # Same defect class as bayesopt._propose_one / batch.propose_qei_batch /
        # batch.propose_local_penalization: without this check, _fit_surrogate below is what actually
        # executes first with zero observations, not _best_feasible's own zero-size guard. In a
        # torch-free environment that means GaussianProcessRegressor's unrelated, opaque "requires
        # torch" ImportError fires instead of a clear, named error -- and even with torch installed,
        # fitting a GP to an empty dataset silently succeeds (it degenerates to the prior mean/std)
        # rather than failing fast, so the real error would only surface several lines later. Check
        # here, first, so every environment gets the same clear error.
        raise ValueError("cannot propose a next constrained point with zero observations; call tell() first.")
    c = _as_feasibility_matrix("c", c)
    if c.shape[0] != x.shape[0]:
        raise ValueError("c must have one row of constraint values per observation.")
    if c.shape[1] == 0:
        raise ValueError("c must contain at least one constraint column.")
    if not np.all(np.isfinite(y)):
        raise ValueError("objective observations must be finite.")
    if not np.all(np.isfinite(c)):
        raise ValueError("constraint observations must be finite.")
    acq_fn = _get_acquisition(acq)
    kw = {"xi": xi, **(acq_kwargs or {})}

    candidates = latin_hypercube(b, n_candidates, rng)

    obj_gp = _fit_surrogate(x, y, None, fit_kwargs)
    mean, std = _predict_std(obj_gp, x, y, candidates)

    idx, feasible = _best_feasible(y, c, maximize=maximize)
    if np.any(feasible):
        best = float(y[idx])
        acq_vals = np.asarray(acq_fn(mean, std, best, maximize=maximize, **kw), dtype=np.float64)
        if acq_vals.shape != (candidates.shape[0],):
            raise ValueError(
                "constrained acquisition must return exactly one score per candidate; "
                f"expected ({candidates.shape[0]},), got {acq_vals.shape}."
            )
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
    pick = _select_index(merit, candidates.shape[0], largest=True, context="constrained BO merit")
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
    n_init = _require_exact_positive_int(n_init, "n_init")
    n_iter = _require_exact_positive_int(n_iter, "n_iter", minimum=0)
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    if type(maximize) is not bool:
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}.")
    if len(constraints) == 0:
        raise ValueError("constrained_minimize requires at least one constraint; use minimize otherwise.")

    x_rows: list[np.ndarray] = []
    y_values: list[float] = []
    c_rows: list[np.ndarray] = []
    failed_evaluations: list[dict[str, Any]] = []
    n_evaluations = 0

    def evaluate(point: np.ndarray) -> tuple[float, np.ndarray] | None:
        nonlocal n_evaluations
        candidate = np.array(point, dtype=np.float64, copy=True)
        n_evaluations += 1
        objective_value = float(objective(candidate.copy()))
        constraint_values = np.asarray([float(con(candidate.copy())) for con in constraints], dtype=np.float64)
        if not np.isfinite(objective_value) or not np.all(np.isfinite(constraint_values)):
            failed_evaluations.append(
                {
                    "evaluation": n_evaluations,
                    "x": candidate,
                    "status": "nonfinite_observation",
                    "objective": objective_value,
                    "constraints": constraint_values,
                }
            )
            return None
        return objective_value, constraint_values

    def result(stopped_reason: str) -> ConstrainedBayesOptResult:
        x = np.asarray(x_rows, dtype=np.float64).reshape(-1, b.shape[0])
        y = np.asarray(y_values, dtype=np.float64)
        c = np.asarray(c_rows, dtype=np.float64).reshape(-1, len(constraints))
        if y.size:
            idx, feasible = _best_feasible(y, c, maximize=maximize)
            best_x: np.ndarray | None = x[idx].copy()
            best_y: float | None = float(y[idx])
        else:
            feasible = np.empty(0, dtype=bool)
            best_x = None
            best_y = None
        return ConstrainedBayesOptResult(
            best_x=best_x,
            best_y=best_y,
            x=x,
            y=y,
            n_evaluations=n_evaluations,
            failed_evaluations=tuple(failed_evaluations),
            stopped_reason=stopped_reason,
            c=c,
            feasible=feasible,
        )

    for row in latin_hypercube(b, n_init, rng):
        observation = evaluate(row)
        if observation is None:
            return result("objective_or_constraint_failed")
        objective_value, constraint_values = observation
        x_rows.append(np.array(row, dtype=np.float64, copy=True))
        y_values.append(objective_value)
        c_rows.append(constraint_values.copy())

    for _ in range(n_iter):
        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_values, dtype=np.float64)
        c = np.asarray(c_rows, dtype=np.float64)
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
        observation = evaluate(nxt)
        if observation is None:
            return result("objective_or_constraint_failed")
        objective_value, constraint_values = observation
        x_rows.append(nxt.copy())
        y_values.append(objective_value)
        c_rows.append(constraint_values.copy())

    return result("budget_exhausted")


__all__: Sequence[str] = [
    "ConstrainedBayesOptResult",
    "probability_of_feasibility",
    "propose_next_constrained",
    "constrained_minimize",
]
