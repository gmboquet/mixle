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
    the largest fidelity (or ``target``) is the true objective. ``costs`` is the per-fidelity evaluation
    cost (default: the fidelity value itself). The loop fits a GP over ``[x, s]``, proposes ``x`` by
    Expected Improvement at the target fidelity, then evaluates at the fidelity maximizing target-variance
    reduction per unit cost, until the cumulative cost reaches ``max_cost``. Returns
    ``{'x', 'y', 'X', 'Y', 'cost'}`` -- the best *target-fidelity* point and the full augmented history.
    """
    if int(n_candidates) <= 0:
        raise ValueError("n_candidates must be positive.")
    b = _as_bounds(bounds)
    d = b.shape[0]
    rng = _as_rng(seed)
    fids = np.asarray(fidelities, dtype=np.float64).ravel()
    target = float(fids.max()) if target is None else float(target)
    cost_arr = fids if costs is None else np.asarray(costs, dtype=np.float64).ravel()
    cost_map = {float(s): float(c) for s, c in zip(fids, cost_arr)}
    if any(c <= 0.0 for c in cost_map.values()):
        # a zero (or negative) cost breaks the budget loop's termination: `spent` never advances for
        # that fidelity, and its per-cost score (variance_reduction / cost) is +inf, so it wins every
        # round -- an unbounded hang, not a graceful "always prefer the free fidelity" outcome. A free
        # fidelity is a real modeling choice (e.g. a cheap proxy at cost 0), so surface it as a clear
        # error rather than silently freezing the caller.
        raise ValueError(f"multi_fidelity_bo requires every fidelity cost > 0, got {cost_map}")
    sign = -1.0 if maximize else 1.0
    n_init = int(n_init) if n_init else 2 * d

    rows: list[np.ndarray] = []
    y: list[float] = []
    for s in fids:  # seed every fidelity
        for xx in latin_hypercube(b, n_init, rng):
            rows.append(np.append(xx, s))
            y.append(sign * float(objective(np.asarray(xx, dtype=np.float64), float(s))))
    x_aug = np.asarray(rows)
    y_arr = np.asarray(y, dtype=np.float64)
    spent = float(sum(cost_map[float(s)] for s in x_aug[:, -1]))

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

        # Pick the fidelity that most reduces the target's posterior variance per unit cost. Observing
        # (xstar, s) cuts var of f(xstar, target) by cov_post(target, s)^2 / var_post(s).
        best_s, best_score = target, -np.inf
        for s in fids:
            pts = np.array([np.append(xstar, target), np.append(xstar, float(s))])
            _, c2 = gp.predict(x_aug, y_arr, pts, return_cov=True)
            c2 = np.atleast_2d(np.asarray(c2, dtype=np.float64))
            var_reduction = c2[0, 1] ** 2 / max(c2[1, 1], 1e-12)
            score = var_reduction / cost_map[float(s)]
            if score > best_score:
                best_score, best_s = score, float(s)

        yn = sign * float(objective(np.asarray(xstar, dtype=np.float64), best_s))
        x_aug = np.vstack([x_aug, np.append(xstar, best_s)])
        y_arr = np.append(y_arr, yn)
        spent += cost_map[best_s]

    at_target = x_aug[:, -1] == target
    if at_target.any():
        idx = int(np.where(at_target)[0][int(np.argmin(y_arr[at_target]))])
    else:
        idx = int(np.argmin(y_arr))
    return {
        "x": x_aug[idx, :d],
        "y": sign * float(y_arr[idx]),
        "X": x_aug,
        "Y": sign * y_arr,
        "cost": spent,
    }


__all__ = ["multi_fidelity_minimize"]
