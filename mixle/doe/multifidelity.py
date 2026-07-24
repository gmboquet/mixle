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
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


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

    Returns ``{'x', 'y', 'X', 'Y', 'cost', 'target_evaluated'}`` -- ``X``/``Y`` are the full augmented
    history and ``x``/``y`` the best *target-fidelity* point and response, but only when
    ``target_evaluated`` is true: if the budget ran out before any target-fidelity evaluation was
    affordable, ``x``/``y`` are ``None`` rather than silently standing in a lower-fidelity result for the
    target-fidelity answer the caller asked for.
    """
    if int(n_candidates) <= 0:
        raise ValueError("n_candidates must be positive.")
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
    n_init = int(n_init) if n_init else 2 * d
    if n_init <= 0:
        raise ValueError("n_init must be positive.")

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
    spent = 0.0
    for s in fids:  # seed every fidelity, budget permitting
        c = cost_map[float(s)]
        for xx in latin_hypercube(b, n_init, rng):
            if spent + c > max_cost:
                break  # this fidelity's next seed point would overshoot the remaining budget
            rows.append(np.append(xx, s))
            y.append(sign * float(objective(np.asarray(xx, dtype=np.float64), float(s))))
            spent += c
    x_aug = np.asarray(rows)
    y_arr = np.asarray(y, dtype=np.float64)

    while spent < max_cost:
        try:
            gp = _fit_surrogate(x_aug, y_arr, None, fit_kwargs)
        except Exception:  # noqa: BLE001 -- GP fit can fail on ill-conditioned data; stop gracefully
            break
        cand = latin_hypercube(b, int(n_candidates), rng)
        cand_t = np.column_stack([cand, np.full(cand.shape[0], target)])
        mean, cov = gp.predict(x_aug, y_arr, cand_t, return_cov=True)
        mean = np.asarray(mean, dtype=np.float64).ravel()
        std = np.sqrt(np.clip(np.diag(np.atleast_2d(np.asarray(cov, dtype=np.float64))), 1e-18, None))
        at_target = x_aug[:, -1] == target
        best_t = float(y_arr[at_target].min()) if at_target.any() else float(mean.min())
        z = (best_t - mean) / std
        ei = (best_t - mean) * norm.cdf(z) + std * norm.pdf(z)  # EI at the target fidelity (minimization)
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
            _, c2 = gp.predict(x_aug, y_arr, pts, return_cov=True)
            c2 = np.atleast_2d(np.asarray(c2, dtype=np.float64))
            var_reduction = c2[0, 1] ** 2 / max(c2[1, 1], 1e-12)
            score = var_reduction / c
            if score > best_score:
                best_score, best_s = score, float(s)
        if best_s is None:
            break  # nothing affordable in the remaining budget

        yn = sign * float(objective(np.asarray(xstar, dtype=np.float64), best_s))
        x_aug = np.vstack([x_aug, np.append(xstar, best_s)])
        y_arr = np.append(y_arr, yn)
        spent += cost_map[best_s]

    at_target = x_aug[:, -1] == target
    target_evaluated = bool(at_target.any())
    if target_evaluated:
        idx = int(np.where(at_target)[0][int(np.argmin(y_arr[at_target]))])
        best_x: np.ndarray | None = x_aug[idx, :d]
        best_y: float | None = sign * float(y_arr[idx])
    else:
        # The budget ran out before any target-fidelity evaluation was affordable. Returning the best
        # lower-fidelity observation here as `x`/`y` would silently mislabel it as the answer to the
        # target-fidelity question (MXR-080-0181); report honestly that no target-fidelity result was
        # obtained instead. `X`/`Y` still hold the full history for a caller who wants to inspect what
        # lower-fidelity information was gathered regardless.
        best_x, best_y = None, None
    return {
        "x": best_x,
        "y": best_y,
        "X": x_aug,
        "Y": sign * y_arr,
        "cost": spent,
        "target_evaluated": target_evaluated,
    }


__all__ = ["multi_fidelity_minimize"]
