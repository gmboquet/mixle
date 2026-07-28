"""Cost-aware multi-fidelity Bayesian optimization.

Many expensive objectives have lower-fidelity approximations -- a coarser mesh, fewer Monte-Carlo samples, a
shorter training run. Multi-fidelity BO exploits them: it spends low-cost low-fidelity evaluations to
locate good regions and reserves the expensive high-fidelity ones for refinement, reaching the optimum
of the true (target) objective for a fraction of the cost of optimizing it directly.

:func:`multi_fidelity_minimize` follows the BOCA idea (Kandasamy et al. 2017): a single GP over the
input augmented with a fidelity coordinate learns how fidelities correlate; each step picks the input by
Expected Improvement at the *target* fidelity, then picks the fidelity that buys the most target-variance
reduction *per unit cost*. It fits the torch GP surrogate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm

from mixle.doe.bayesopt import _fit_surrogate
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, _require_exact_positive_int, latin_hypercube


def _surrogate_fit_error_types() -> tuple[type[BaseException], ...]:
    """The well-defined numerical failure types a GP surrogate fit can raise -- never a bare ``Exception``.

    A singular/indefinite covariance during fitting raises ``numpy.linalg.LinAlgError`` (a numpy-backed
    surrogate) or ``torch.linalg.LinAlgError`` (the default torch-backed one, out of
    ``torch.linalg.cholesky`` inside ``GaussianProcessRegressor.log_marginal_likelihood``/``predict``) --
    both a genuine "this data is ill-conditioned" failure, not a programming error. Anything else (a bad
    ``fit_kwargs`` optimizer name, a missing torch install, an unrelated bug) must propagate instead of
    being swallowed. Torch is imported lazily here, matching
    :func:`mixle.doe.bayesopt._fit_surrogate`'s own lazy import, so merely importing this module never
    requires torch -- the tuple just narrows to numpy alone when torch is not installed.
    """
    errors: tuple[type[BaseException], ...] = (np.linalg.LinAlgError,)
    try:
        import torch
    except ImportError:
        return errors
    return (*errors, torch.linalg.LinAlgError)


def _validated_posterior(
    gp: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    points: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite, symmetric, PSD surrogate posterior with exact requested axes."""
    mean_raw, covariance_raw = gp.predict(x_train, y_train, points, return_cov=True)
    mean = np.asarray(mean_raw, dtype=np.float64)
    n_points = points.shape[0]
    if mean.shape == (n_points, 1):
        mean = mean[:, 0]
    elif mean.shape != (n_points,):
        raise ValueError(f"{label} posterior mean must have shape ({n_points},) or ({n_points}, 1), got {mean.shape}.")
    covariance = np.asarray(covariance_raw, dtype=np.float64)
    if covariance.shape != (n_points, n_points):
        raise ValueError(
            f"{label} posterior covariance must have shape ({n_points}, {n_points}), got {covariance.shape}."
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{label} posterior mean and covariance must be finite.")
    scale = max(1.0, float(np.linalg.norm(covariance, ord=np.inf)))
    tolerance = 1e-10 * scale
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{label} posterior covariance must be symmetric.")
    symmetric = 0.5 * (covariance + covariance.T)
    minimum_eigenvalue = float(np.linalg.eigvalsh(symmetric).min())
    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            f"{label} posterior covariance must be positive semidefinite; "
            f"minimum eigenvalue is {minimum_eigenvalue:.6g}."
        )
    return mean, covariance


def multi_fidelity_minimize(
    objective: Callable[[np.ndarray, float], float],
    bounds: Bounds,
    *,
    fidelities: tuple[float, ...] = (0.5, 1.0),
    costs: tuple[float, ...] | None = None,
    target: float | None = None,
    n_init: int | None = None,
    max_cost: float = 40.0,
    n_candidates: int = 256,
    maximize: bool = False,
    seed: int | RandomState | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cost-aware multi-fidelity Bayesian optimization of ``objective(x, s)``.

    ``objective(x, s)`` returns the response at input ``x`` and fidelity ``s`` (one of ``fidelities``);
    the largest fidelity (or ``target``) is the true objective. ``target``, if given, must itself be one
    of ``fidelities``: this function never evaluates a fidelity outside that set, so an out-of-set target
    could otherwise only ever be "satisfied" by silently substituting a lower-fidelity response for it.
    ``costs`` is the per-fidelity evaluation cost (default: the fidelity value itself) and, if given,
    must supply exactly one entry per fidelity. The loop fits a GP over ``[x, s]``, proposes ``x`` by
    Expected Improvement at the target fidelity, then evaluates at the fidelity maximizing target-variance
    reduction per unit cost. Every evaluation -- initial per-fidelity seeding and the sequential loop
    alike -- reserves its cost against ``max_cost`` before calling ``objective``, so cumulative cost never
    overshoots ``max_cost``; a ``max_cost`` too small to afford even one evaluation is rejected up front.

    Returns ``{'x', 'y', 'X', 'Y', 'cost', 'target_evaluated', 'stopped_reason', 'error',
    'failed_evaluations'}``. ``X``/``Y`` are the usable finite augmented evaluation history (fidelity
    in the last column of ``X``) and ``cost`` is the total actually spent, including failed calls.
    ``failed_evaluations`` separately preserves each unusable attempt's point, fidelity, cost, status,
    and error. ``x``/``y`` are the best *target-fidelity* point and response, but only when
    ``target_evaluated`` is true: if the budget ran out before any target-fidelity evaluation was
    affordable, ``x``/``y`` are ``None`` rather than silently standing in a lower-fidelity result for the
    target-fidelity answer the caller asked for. ``stopped_reason`` is ``"budget_exhausted"`` (the loop
    ran until no further evaluation fit in ``max_cost`` -- the normal, successful termination for this
    optimizer, which has no other stopping criterion) or ``"surrogate_fit_failed"`` (a well-defined
    numerical failure -- e.g. ``numpy``/``torch`` ``linalg.LinAlgError`` from a singular covariance --
    aborted the loop early; ``error`` then holds a diagnostic string, else ``None``). Any other exception
    out of surrogate fitting (a bad ``fit_kwargs`` optimizer name, a missing torch install, ...)
    propagates rather than being reported as a quiet early stop.
    """
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    b = _as_bounds(bounds)
    d = b.shape[0]
    rng = _as_rng(seed)

    fids = np.asarray(fidelities, dtype=np.float64).ravel()
    if fids.size == 0:
        raise ValueError("fidelities must not be empty.")
    if not np.all(np.isfinite(fids)):
        raise ValueError(f"fidelities must all be finite, got {fidelities!r}.")
    if np.unique(fids).size != fids.size:
        raise ValueError(f"fidelities must not contain duplicates, got {fidelities!r}.")

    target = float(fids.max()) if target is None else float(target)
    if not np.isfinite(target):
        raise ValueError(f"target must be finite, got {target!r}.")
    if not np.any(fids == target):
        # Without this, a target outside `fidelities` is never evaluated: the fidelity-selection loop
        # below only ever picks from `fids`, so `at_target` stays all-False and the final fallback used
        # to silently return the best observation at ANY fidelity, mislabeled as the target-fidelity
        # result (MXR-080-0181). Reject it here instead of ever manufacturing that false answer.
        raise ValueError(f"target fidelity {target} must be one of fidelities {tuple(fids.tolist())}.")

    if costs is None:
        cost_arr = fids
    else:
        cost_arr = np.asarray(costs, dtype=np.float64).ravel()
        if cost_arr.size != fids.size:
            raise ValueError(
                f"costs must have exactly one entry per fidelity: got {cost_arr.size} costs for {fids.size} fidelities."
            )
    cost_map = {float(s): float(c) for s, c in zip(fids, cost_arr)}
    if any(not np.isfinite(c) or c <= 0.0 for c in cost_map.values()):
        # a zero, negative, or non-finite cost breaks the budget loop's termination: `spent` never
        # advances for that fidelity (or moves the wrong way, or corrupts the per-cost score entirely),
        # and its per-cost score (variance_reduction / cost) can be +inf, so it wins every round -- an
        # unbounded hang, not a graceful "always prefer the free fidelity" outcome. A free fidelity is a
        # real modeling choice (e.g. a cheap proxy at cost 0), so surface it as a clear error rather than
        # silently freezing the caller.
        raise ValueError(f"multi_fidelity_bo requires every fidelity cost to be finite and > 0, got {cost_map}")

    max_cost = float(max_cost)
    if not np.isfinite(max_cost) or max_cost < 0.0:
        raise ValueError(f"max_cost must be finite and non-negative, got {max_cost!r}.")

    sign = -1.0 if maximize else 1.0
    n_init = 2 * d if n_init is None else _require_exact_positive_int(n_init, "n_init")

    cheapest = min(cost_map.values())
    if cheapest > max_cost:
        # Reserve before spend, taken to its logical conclusion: if not even the cheapest fidelity fits
        # in the budget, reject up front (MXR-080-0182) instead of letting initialization spend anyway
        # (the pre-fix bug: a max_cost=0 run still spent on unconditional initial seeding).
        raise ValueError(
            f"max_cost={max_cost} cannot afford a single evaluation; the cheapest fidelity "
            f"({cheapest}) alone exceeds it."
        )

    rows: list[np.ndarray] = []
    y: list[float] = []
    failed_evaluations: list[dict[str, Any]] = []
    spent = 0.0

    def evaluate(xx: np.ndarray, fidelity: float) -> float | None:
        """Charge one attempted call and retain explicit evidence if it does not yield a finite scalar."""
        nonlocal spent, error, stopped_reason
        cost = cost_map[fidelity]
        spent += cost
        failure = {
            "x": np.asarray(xx, dtype=np.float64).copy(),
            "fidelity": fidelity,
            "cost": cost,
        }
        try:
            raw = objective(np.asarray(xx, dtype=np.float64), fidelity)
        except Exception as exc:  # noqa: BLE001 - preserve operational call failure and its real cost
            error = f"{type(exc).__name__}: {exc}"
            failure.update({"status": "exception", "error": error})
            failed_evaluations.append(failure)
            stopped_reason = "objective_failed"
            return None
        raw_array = np.asarray(raw)
        if raw_array.shape != () or np.issubdtype(raw_array.dtype, np.bool_):
            error = f"objective returned {type(raw).__name__}; expected one finite numeric scalar"
            failure.update({"status": "invalid_observation", "error": error})
            failed_evaluations.append(failure)
            stopped_reason = "objective_failed"
            return None
        try:
            observation = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            failure.update({"status": "invalid_observation", "error": error})
            failed_evaluations.append(failure)
            stopped_reason = "objective_failed"
            return None
        if not np.isfinite(observation):
            error = f"objective returned non-finite observation {observation!r}"
            failure.update({"status": "nonfinite_observation", "error": error})
            failed_evaluations.append(failure)
            stopped_reason = "objective_failed"
            return None
        return sign * observation

    stopped_reason = "budget_exhausted"
    error: str | None = None
    objective_failed = False
    for s in fids:  # seed every fidelity, budget permitting
        c = cost_map[float(s)]
        for xx in latin_hypercube(b, n_init, rng):
            if spent + c > max_cost:
                break  # this fidelity's next seed point would overshoot the remaining budget
            observation = evaluate(xx, float(s))
            if observation is None:
                objective_failed = True
                break
            rows.append(np.append(xx, s))
            y.append(observation)
        if objective_failed:
            break
    x_aug = np.asarray(rows, dtype=np.float64).reshape(-1, d + 1)
    y_arr = np.asarray(y, dtype=np.float64)

    fit_error_types = _surrogate_fit_error_types()
    while not objective_failed and spent < max_cost:
        try:
            gp = _fit_surrogate(x_aug, y_arr, None, fit_kwargs)
        except fit_error_types as exc:
            stopped_reason = "surrogate_fit_failed"
            error = f"{type(exc).__name__}: {exc}"
            break
        cand = latin_hypercube(b, n_candidates, rng)
        cand_t = np.column_stack([cand, np.full(cand.shape[0], target)])
        try:
            mean, cov = _validated_posterior(gp, x_aug, y_arr, cand_t, label="target-candidate")
        except ValueError as exc:
            stopped_reason = "surrogate_prediction_failed"
            error = f"{type(exc).__name__}: {exc}"
            break
        variance = np.diag(cov)
        tolerance = 1e-10 * max(1.0, float(np.linalg.norm(cov, ord=np.inf)))
        variance = np.where((variance < 0.0) & (variance >= -tolerance), 0.0, variance)
        std = np.sqrt(variance)
        at_target = x_aug[:, -1] == target
        best_t = float(y_arr[at_target].min()) if at_target.any() else float(mean.min())
        improvement = best_t - mean
        ei = np.maximum(improvement, 0.0)
        uncertain = std > 0.0
        z = improvement[uncertain] / std[uncertain]
        ei[uncertain] = improvement[uncertain] * norm.cdf(z) + std[uncertain] * norm.pdf(z)
        if not np.all(np.isfinite(ei)):
            stopped_reason = "surrogate_prediction_failed"
            error = "ValueError: expected improvement is non-finite"
            break
        xstar = cand[int(np.argmax(ei))]

        # Pick the fidelity that most reduces the target's posterior variance per unit cost, among those
        # that still fit in the remaining budget. Observing (xstar, s) cuts var of f(xstar, target) by
        # cov_post(target, s)^2 / var_post(s).
        best_s, best_score = None, -np.inf
        for s in fids:
            c = cost_map[float(s)]
            if spent + c > max_cost:
                continue  # would overshoot the remaining budget; not eligible this round
            pts = np.array([np.append(xstar, target), np.append(xstar, float(s))])
            try:
                _, c2 = _validated_posterior(gp, x_aug, y_arr, pts, label=f"fidelity-{float(s):g}")
            except ValueError as exc:
                stopped_reason = "surrogate_prediction_failed"
                error = f"{type(exc).__name__}: {exc}"
                best_s = None
                break
            var_reduction = 0.0 if c2[1, 1] == 0.0 else c2[0, 1] ** 2 / c2[1, 1]
            score = var_reduction / c
            if not np.isfinite(score) or score < 0.0:
                stopped_reason = "surrogate_prediction_failed"
                error = f"ValueError: fidelity-{float(s):g} variance-reduction score is invalid"
                best_s = None
                break
            if score > best_score:
                best_score, best_s = score, float(s)
        if stopped_reason == "surrogate_prediction_failed":
            break
        if best_s is None:
            break  # nothing affordable in the remaining budget; stopped_reason stays "budget_exhausted"

        yn = evaluate(xstar, best_s)
        if yn is None:
            objective_failed = True
            break
        x_aug = np.vstack([x_aug, np.append(xstar, best_s)])
        y_arr = np.append(y_arr, yn)

    at_target = x_aug[:, -1] == target if x_aug.size else np.array([], dtype=bool)
    target_evaluated = bool(at_target.any())
    if target_evaluated:
        idx = int(np.where(at_target)[0][int(np.argmin(y_arr[at_target]))])
        best_x: np.ndarray | None = x_aug[idx, :d]
        best_y: float | None = sign * float(y_arr[idx])
    else:
        # The budget ran out (or the surrogate broke) before any target-fidelity evaluation was
        # affordable. Returning the best lower-fidelity observation here as `x`/`y` would silently
        # mislabel it as the answer to the target-fidelity question (MXR-080-0181); report honestly that
        # no target-fidelity result was obtained instead. `X`/`Y` still hold the full history for a
        # caller who wants to inspect what lower-fidelity information was gathered regardless.
        best_x, best_y = None, None
    return {
        "x": best_x,
        "y": best_y,
        "X": x_aug,
        "Y": sign * y_arr,
        "cost": spent,
        "target_evaluated": target_evaluated,
        "stopped_reason": stopped_reason,
        "error": error,
        "failed_evaluations": failed_evaluations,
    }


__all__ = ["multi_fidelity_minimize"]
