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

from mixle.doe.designs import _qmc_unit

__all__ = ["sobol_indices", "morris_screening", "fast_indices", "dgsm"]


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
    except Exception:  # pragma: no cover - qmc fallback  # noqa: BLE001
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
    """Morris elementary-effects screening -- a low-cost first factor ranking.

    Walks ``trajectories`` one-factor-at-a-time paths on a ``levels``-grid; the mean absolute elementary
    effect ``mu_star[i]`` ranks influence and the spread ``sigma[i]`` flags nonlinearity/interactions.
    Cost: ``trajectories * (d + 1)`` evaluations -- far fewer than Sobol, for an initial screen.
    """
    if levels < 2:
        raise ValueError(f"morris_screening requires levels >= 2 (a single-level grid has no step), got {levels}")
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


def fast_indices(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    n: int = 600,
    *,
    harmonics: int = 6,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """First-order sensitivity indices via Random Balance Designs FAST (RBD-FAST).

    A Fourier alternative to :func:`sobol_indices` for the *first-order* indices: every input is driven
    along the same triangle-wave search curve but under an independent random permutation, the model is
    evaluated once over the ``n`` points, and for each input the output -- reordered along that input's
    curve -- has its variance concentrated at the base frequency's first ``harmonics`` harmonics. The
    ratio of that power to the total is the first-order index (with the Tarantola bias correction). Cost
    is a single batch of ``n`` evaluations, independent of dimension.

    Returns ``{'S1': (d,), 'names': [...], 'var': float}``.
    """
    bounds = np.asarray(bounds, dtype=float)
    d = len(bounds)
    rng = np.random.RandomState(seed)
    s = np.linspace(-np.pi, np.pi, int(n), endpoint=False)
    base = 0.5 + np.arcsin(np.sin(s)) / np.pi  # triangle wave, uniform on [0, 1]
    perms = [rng.permutation(int(n)) for _ in range(d)]
    x = np.column_stack([base[perms[i]] for i in range(d)])
    y = np.asarray(func(_scale(x, bounds)), dtype=float).ravel()
    s1 = np.zeros(d)
    var = float(np.var(y))
    out_names = list(names) if names else [f"x{i}" for i in range(d)]
    if var <= 0:
        return {"S1": s1, "names": out_names, "var": 0.0}
    m = int(harmonics)
    if 2.0 * m >= int(n) - 1:
        # the Tarantola correction's denominator (1 - 2m/(n-1)) must stay strictly positive; once
        # 2m/(n-1) reaches or exceeds 1, the correction denominator hits zero or goes negative,
        # flipping the sign of a subsequent nonsensical S1 rather than raising. clip(0, 1) at the
        # end would otherwise silently mask this as an ordinary "no sensitivity" result.
        raise ValueError(
            f"fast_indices requires 2*harmonics < n-1 for a well-posed Tarantola correction "
            f"(harmonics={m}, n={n}); increase n or lower harmonics."
        )
    for i in range(d):
        yi = y[np.argsort(perms[i])]  # reorder output along input i's search-curve coordinate
        spectrum = np.abs(np.fft.rfft(yi - yi.mean())) ** 2
        total = float(spectrum[1:].sum())
        raw = float(spectrum[1 : m + 1].sum()) / total if total > 0 else 0.0
        # Tarantola (2006) bias correction: an uninformative input has expected raw ~ 2m/(n-1).
        s1[i] = (raw - 2.0 * m / (int(n) - 1)) / (1.0 - 2.0 * m / (int(n) - 1))
    return {"S1": np.clip(s1, 0.0, 1.0), "names": out_names, "var": var}


def dgsm(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    n: int = 1024,
    *,
    seed: int = 0,
    rel_step: float = 1.0e-4,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Derivative-based global sensitivity measures (DGSM): mean squared partial derivatives.

    ``nu[i] = E[(df/dx_i)^2]`` over the input box, estimated by central finite differences at ``n``
    low-discrepancy points. Unlike the first-order Sobol index, a DGSM is nonzero whenever an input
    matters *anywhere* -- including purely through interactions -- so it is a low-cost, robust screen that
    upper-bounds the total Sobol index (Sobol & Kucherenko, via the Poincare inequality:
    ``ST[i] <= (L_i / pi)^2 * nu[i] / Var(y)`` for a uniform input of width ``L_i``). The reported
    ``importance`` is ``L_i^2 * nu[i]`` normalized to sum to one -- a dimensionless influence ranking.

    Returns ``{'nu': (d,), 'importance': (d,), 'names': [...]}``.
    """
    bounds = np.asarray(bounds, dtype=float)
    d = len(bounds)
    x = _scale(_sobol_unit(int(n), d, seed), bounds)
    span = bounds[:, 1] - bounds[:, 0]
    nu = np.zeros(d)
    for i in range(d):
        h = rel_step * span[i]
        xp = x.copy()
        xm = x.copy()
        xp[:, i] = np.minimum(x[:, i] + h, bounds[i, 1])
        xm[:, i] = np.maximum(x[:, i] - h, bounds[i, 0])
        step = xp[:, i] - xm[:, i]
        yp = np.asarray(func(xp), dtype=float).ravel()
        ym = np.asarray(func(xm), dtype=float).ravel()
        nu[i] = float(np.mean(((yp - ym) / np.where(step > 0, step, 1.0)) ** 2))
    weighted = span**2 * nu
    total = float(weighted.sum())
    importance = weighted / total if total > 0 else np.zeros(d)
    return {"nu": nu, "importance": importance, "names": list(names) if names else [f"x{i}" for i in range(d)]}
