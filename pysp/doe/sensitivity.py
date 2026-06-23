"""Global sensitivity analysis: variance-based Sobol indices and Morris screening.

Which inputs actually drive a model's output? Sobol indices decompose the output variance into the
contribution of each input (first order) and of each input including all its interactions (total order).
Morris screening is a cheaper one-at-a-time elementary-effects method for an initial factor ranking.
These tell you which survey parameters / forcings to refine and which to fix -- the front of the UQ loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pysp.doe.designs import _qmc_unit

__all__ = ["sobol_indices", "morris_screening"]


def _scale(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Map points in the unit cube to the box ``bounds`` (d, 2)."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + unit * (hi - lo)


def _sobol_unit(n: int, d: int, seed: int) -> np.ndarray:
    """``n`` low-discrepancy points in ``[0,1]^d`` via the shared DoE QMC engine.

    Reuses ``designs._qmc_unit`` (scrambled Sobol', stratified-random fallback on older scipy) so
    the sensitivity sampler and the DoE designs draw from one source.
    """
    try:
        from scipy.stats import qmc

        return _qmc_unit(qmc.Sobol, d, n, True, np.random.RandomState(seed))
    except Exception:  # pragma: no cover - qmc fallback
        return np.random.RandomState(seed).random((n, d))


def sobol_indices(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    n: int = 4096,
    *,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """First- and total-order Sobol sensitivity indices (Saltelli sampling, Jansen estimators).

    Args:
        func: a *vectorized* model ``f(X) -> y`` mapping an ``(m, d)`` array of inputs to an ``(m,)``
            array of scalar outputs.
        bounds: ``[(lo, hi), ...]`` per input -- the input is taken uniform on the box.
        n: base sample size; the total number of model evaluations is ``n * (d + 2)``.
        seed: RNG seed for the (Sobol) base samples.
        names: optional input names for the returned dict.

    Returns:
        ``{'S1': (d,), 'ST': (d,), 'names': [...], 'var': float}`` -- first-order ``S1[i]`` is the
        fraction of output variance from input ``i`` alone; total-order ``ST[i]`` includes all
        interactions involving ``i`` (so ``ST[i] - S1[i]`` measures ``i``'s interaction strength, and
        ``ST[i] ~ 0`` means input ``i`` can be fixed).
    """
    bounds = np.asarray(bounds, dtype=float)
    d = len(bounds)
    a_unit = _sobol_unit(n, 2 * d, seed)  # split one 2d-dimensional Sobol block into A and B (independence)
    a, b = a_unit[:, :d], a_unit[:, d:]
    ya = np.asarray(func(_scale(a, bounds)), dtype=float).ravel()
    yb = np.asarray(func(_scale(b, bounds)), dtype=float).ravel()
    var = np.var(np.concatenate([ya, yb]))
    s1 = np.zeros(d)
    st = np.zeros(d)
    if var <= 0:  # constant output: every index is zero
        return {"S1": s1, "ST": st, "names": list(names) if names else [f"x{i}" for i in range(d)], "var": 0.0}
    for i in range(d):
        ab = a.copy()
        ab[:, i] = b[:, i]  # A with column i taken from B
        yab = np.asarray(func(_scale(ab, bounds)), dtype=float).ravel()
        s1[i] = np.mean(yb * (yab - ya)) / var  # Saltelli 2010 first-order estimator
        st[i] = 0.5 * np.mean((ya - yab) ** 2) / var  # Jansen total-order estimator
    return {
        "S1": np.clip(s1, 0.0, 1.0),
        "ST": np.clip(st, 0.0, None),
        "names": list(names) if names else [f"x{i}" for i in range(d)],
        "var": float(var),
    }


def morris_screening(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    *,
    trajectories: int = 20,
    levels: int = 4,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Morris elementary-effects screening -- a cheap first factor ranking.

    Walks ``trajectories`` one-factor-at-a-time paths on a ``levels``-grid; the mean absolute elementary
    effect ``mu_star[i]`` ranks influence and the spread ``sigma[i]`` flags nonlinearity/interactions.
    Cost: ``trajectories * (d + 1)`` evaluations -- far fewer than Sobol, for an initial screen.
    """
    bounds = np.asarray(bounds, dtype=float)
    d = len(bounds)
    rng = np.random.RandomState(seed)
    delta = levels / (2.0 * (levels - 1))  # the standard Morris step on the unit grid
    grid = np.linspace(0.0, 1.0, levels)
    effects = [[] for _ in range(d)]
    for _ in range(trajectories):
        base = rng.choice(grid[: levels // 2 + 1] if levels > 1 else grid, size=d)  # room to step up by delta
        order = rng.permutation(d)
        x = base.copy()
        y_prev = float(np.asarray(func(_scale(x[None, :], bounds))).ravel()[0])
        for i in order:
            x_next = x.copy()
            x_next[i] = min(x_next[i] + delta, 1.0)
            y_next = float(np.asarray(func(_scale(x_next[None, :], bounds))).ravel()[0])
            step = x_next[i] - x[i]
            if step != 0:
                effects[i].append((y_next - y_prev) / step)
            x, y_prev = x_next, y_next
    mu_star = np.array([np.mean(np.abs(e)) if e else 0.0 for e in effects])
    sigma = np.array([np.std(e) if e else 0.0 for e in effects])
    return {
        "mu_star": mu_star,
        "sigma": sigma,
        "names": list(names) if names else [f"x{i}" for i in range(d)],
    }
