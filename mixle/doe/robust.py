"""Noise-robust incumbent selection for Bayesian optimization (a correctness fix for noisy objectives).

:func:`mixle.doe.bayesopt.minimize` returns the incumbent as ``argmin`` of the OBSERVED objective values.
That is correct for a deterministic objective, but optimistically biased for a NOISY one: the lowest
observed value is often just the point that drew the luckiest negative noise, not the true optimum, so
``best_x`` chases noise and ``best_y`` is a downward-biased estimate.

The standard fix (Jones et al.; the "noisy BO" recommendation) is to report the incumbent by the
surrogate's POSTERIOR MEAN, which averages out observation noise: fit the GP over every evaluated point
and return the evaluated point whose posterior-mean prediction is best. :func:`posterior_incumbent` is
that primitive; :func:`noisy_minimize` runs the ordinary BO loop and then re-selects the incumbent this
way -- so the exploration is unchanged, only the *reported answer* is made robust.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.doe.bayesopt import BayesOptResult, _select_index, _validate_observations, _validate_prediction, minimize

Bounds = Any


@dataclass
class IncumbentResult:
    """The posterior-mean incumbent among evaluated points: its location, posterior mean, and index."""

    best_x: np.ndarray
    best_mean: float
    best_index: int


def _fit_gp(x: np.ndarray, y: np.ndarray, gp: Any, fit_kwargs: dict[str, Any] | None) -> Any:
    _validate_observations(x, y, context="posterior_incumbent")
    if gp is None:
        from mixle.models.gaussian_process import GaussianProcessRegressor

        scale = float(np.std(y)) or 1.0
        gp = GaussianProcessRegressor(lengthscale=1.0, amplitude=scale, noise=0.1 * scale + 1.0e-6)
    gp.fit(x, y, **{"out": None, **(fit_kwargs or {})})
    return gp


def posterior_incumbent(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maximize: bool = False,
    gp: Any = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> IncumbentResult:
    """Among the evaluated points ``x``, return the one with the best GP POSTERIOR MEAN (min by default).

    Fits a GP surrogate on ``(x, y)`` -- the same surrogate the BO loop uses -- and predicts the
    denoised mean at every evaluated point, so the reported optimum reflects the model's belief, not a
    single lucky noisy observation. For a deterministic objective the posterior mean at the observed
    points is ~the observations, so this reduces to the ordinary argmin/argmax.

    Raises ``ValueError`` if ``x`` contains non-finite values, or if the surrogate's predicted mean
    doesn't match the candidate contract (wrong length, non-finite, or entirely non-finite -- see
    :func:`mixle.doe.bayesopt._validate_prediction` / ``_select_index``), the same shared surrogate
    boundary the sequential/batch/knowledge-gradient proposal paths use (MXR-080-0170).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    if x.shape[0] == 0:
        raise ValueError("need at least one evaluated point.")
    fitted = _fit_gp(x, y, gp, fit_kwargs)
    mean, _ = _validate_prediction(
        fitted.predict(x, y, x, return_cov=False), None, x.shape[0], context="posterior_incumbent"
    )
    idx = _select_index(mean, x.shape[0], largest=maximize, context="posterior_incumbent")
    return IncumbentResult(best_x=x[idx].copy(), best_mean=float(mean[idx]), best_index=idx)


def noisy_minimize(
    objective: Callable[[np.ndarray], float],
    bounds: Bounds,
    n_init: int = 5,
    n_iter: int = 15,
    seed: Any = None,
    *,
    maximize: bool = False,
    gp: Any = None,
    fit_kwargs: dict[str, Any] | None = None,
    **minimize_kwargs: Any,
) -> BayesOptResult:
    """Bayesian optimization of a NOISY objective, reporting the posterior-mean incumbent.

    Runs :func:`mixle.doe.bayesopt.minimize` (same exploration), then replaces the raw ``argmin``-of-
    observations incumbent with the posterior-mean incumbent (:func:`posterior_incumbent`), so ``best_x``
    is the model's believed optimum rather than the luckiest noisy draw and ``best_y`` is its denoised
    estimate. The full ``(x, y)`` evaluation history is preserved on the result.
    """
    result = minimize(objective, bounds, n_init=n_init, n_iter=n_iter, seed=seed, maximize=maximize, **minimize_kwargs)
    if result.stopped_reason != "budget_exhausted":
        return result
    incumbent = posterior_incumbent(result.x, result.y, maximize=maximize, gp=gp, fit_kwargs=fit_kwargs)
    return BayesOptResult(
        best_x=incumbent.best_x,
        best_y=incumbent.best_mean,
        x=result.x,
        y=result.y,
        n_evaluations=result.n_evaluations,
        failed_evaluations=result.failed_evaluations,
        stopped_reason=result.stopped_reason,
    )
