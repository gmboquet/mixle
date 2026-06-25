"""Space-filling and classical experiment-design generators.

Every generator takes per-dimension ``bounds`` -- a sequence of ``(low, high)`` pairs -- and
returns a ``(n, d)`` numpy array of points scaled into those bounds. Random designs accept a
``seed`` (int or ``numpy.random.RandomState``) for reproducibility, matching the rest of pysp.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.stats import qmc

Bounds = Sequence[tuple[float, float]]


def _as_bounds(bounds: Bounds) -> np.ndarray:
    """Validate and return bounds as a ``(d, 2)`` float array with ``low < high`` per row."""
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (low, high) pairs.")
    if arr.shape[0] == 0:
        raise ValueError("bounds must have at least one dimension.")
    if not np.all(arr[:, 0] < arr[:, 1]):
        raise ValueError("each bound must satisfy low < high.")
    return arr


def _as_rng(seed: int | RandomState | None) -> RandomState:
    """Return a ``RandomState`` from an int seed, an existing ``RandomState``, or ``None``."""
    if isinstance(seed, RandomState):
        return seed
    return RandomState(seed)


def _scale_unit(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Scale points from the unit cube ``[0, 1]^d`` into ``bounds``."""
    low = bounds[:, 0]
    high = bounds[:, 1]
    return low + unit * (high - low)


def random_design(bounds: Bounds, n: int, seed: int | RandomState | None = None) -> np.ndarray:
    """Return ``n`` iid uniform points over ``bounds`` as an ``(n, d)`` array."""
    if n <= 0:
        raise ValueError("n must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    unit = rng.random_sample((int(n), b.shape[0]))
    return _scale_unit(unit, b)


def latin_hypercube(
    bounds: Bounds, n: int, seed: int | RandomState | None = None, *, center: bool = False
) -> np.ndarray:
    """Return an ``n``-point Latin hypercube design over ``bounds``.

    Each of the ``d`` axes is partitioned into ``n`` equal strata and exactly one sample falls in
    each stratum (the defining LHS property), with the per-axis stratum assignments independently
    permuted. With ``center=True`` each sample sits at its stratum midpoint; otherwise it is drawn
    uniformly within the stratum.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    d = b.shape[0]
    n = int(n)
    unit = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        offset = 0.5 if center else rng.random_sample(n)
        unit[:, j] = (perm + offset) / n
    return _scale_unit(unit, b)


def maximin_latin_hypercube(
    bounds: Bounds, n: int, seed: int | RandomState | None = None, *, trials: int = 32
) -> np.ndarray:
    """Return the best of ``trials`` Latin hypercube designs by the maximin criterion.

    Generates ``trials`` independent LHS designs and keeps the one whose minimum pairwise
    (Euclidean, bound-normalized) distance is largest, a simple way to improve space-fillingness.
    """
    if trials <= 0:
        raise ValueError("trials must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    span = b[:, 1] - b[:, 0]
    best_design: np.ndarray | None = None
    best_score = -np.inf
    for _ in range(int(trials)):
        design = latin_hypercube(b, n, rng)
        if n < 2:
            return design
        scaled = (design - b[:, 0]) / span
        diff = scaled[:, None, :] - scaled[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        iu = np.triu_indices(n, k=1)
        score = float(np.min(sq[iu]))
        if score > best_score:
            best_score = score
            best_design = design
    assert best_design is not None
    return best_design


_MAXPRO_TINY = 1e-12


def _maxpro_criterion(unit: np.ndarray) -> float:
    """MaxPro criterion of a unit design: ``sum_{i<j} 1 / prod_k (x_ik - x_jk)^2`` (lower is better)."""
    diff = unit[:, None, :] - unit[None, :, :]
    prod = np.prod(diff * diff + _MAXPRO_TINY, axis=2)
    iu = np.triu_indices(unit.shape[0], k=1)
    return float(np.sum(1.0 / prod[iu]))


def _maxpro_obj_grad(flat: np.ndarray, n: int, d: int) -> tuple[float, np.ndarray]:
    """Log MaxPro criterion ``log sum_{i<j} 1/prod_k diff^2`` and its gradient wrt the flattened design.

    Optimizing the log keeps the objective scale-free (the raw criterion spans many orders of magnitude).
    The gradient is analytic: ``dC/dx_il = -2 sum_{j!=i} w_ij / (x_il - x_jl)`` with ``w_ij = 1/prod_k
    (x_ik - x_jk)^2``, and ``d log C = dC / C``.
    """
    x = flat.reshape(n, d)
    diff = x[:, None, :] - x[None, :, :]  # (n, n, d)
    sq = diff * diff
    w = np.exp(-np.sum(np.log(sq + _MAXPRO_TINY), axis=2))  # (n, n), = 1/prod_k diff^2
    np.fill_diagonal(w, 0.0)
    total = 0.5 * float(np.sum(w))  # symmetric sum -> halve for i<j
    inv_diff = diff / (sq + _MAXPRO_TINY)  # ~ 1/diff, finite at diff=0 (diagonal -> 0)
    grad = -2.0 * np.einsum("ij,ijk->ik", w, inv_diff)  # (n, d); j=i excluded since w_ii=0
    return float(np.log(max(total, _MAXPRO_TINY))), (grad / max(total, _MAXPRO_TINY)).ravel()


def _maxpro_swap(unit: np.ndarray, rng: RandomState, iterations: int) -> np.ndarray:
    """MaxProLHD coordinate exchange: accept-if-better within-column swaps preserving the LHS structure."""
    best = _maxpro_criterion(unit)
    n, d = unit.shape
    for _ in range(int(iterations)):
        col = int(rng.randint(d))
        i, j = rng.choice(n, size=2, replace=False)
        unit[[i, j], col] = unit[[j, i], col]
        trial = _maxpro_criterion(unit)
        if trial < best:
            best = trial
        else:
            unit[[i, j], col] = unit[[j, i], col]  # revert
    return unit


def maxpro_design(
    bounds: Bounds,
    n: int,
    seed: int | RandomState | None = None,
    *,
    restarts: int = 3,
    swaps: int = 1500,
    maxiter: int = 300,
) -> np.ndarray:
    """Return a maximum-projection (MaxPro) space-filling design (Joseph, Gul & Ba 2015).

    MaxPro minimizes ``sum_{i<j} 1 / prod_k (x_ik - x_jk)^2``, which forces the points to spread out in
    *every* subspace projection, not just the full space -- the design of choice when only some inputs
    turn out to matter (factor screening / sloppy models), since the points stay space-filling even
    projected onto the active subset.

    Built by the paper's two-stage procedure, repeated from ``restarts`` Latin-hypercube starts: (1) a
    MaxProLHD coordinate-exchange phase (``swaps`` within-column swaps) finds a good basin while keeping
    the LHS structure, then (2) continuous L-BFGS-B optimization (analytic gradient, ``maxiter`` steps)
    moves the points off the grid to the true MaxPro optimum -- the refinement that cuts the criterion
    by far more than the swap phase alone. The best design over restarts is returned as an ``(n, d)`` array.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    from scipy.optimize import minimize

    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    d = b.shape[0]
    n = int(n)
    if n < 2 or d == 0:
        unit = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            unit[:, j] = (rng.permutation(n) + rng.random_sample(n)) / n
        return _scale_unit(unit, b)

    box = [(_MAXPRO_TINY, 1.0 - _MAXPRO_TINY)] * (n * d)
    best_unit: np.ndarray | None = None
    best_crit = np.inf
    for _ in range(max(1, int(restarts))):
        start = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            start[:, j] = (rng.permutation(n) + rng.random_sample(n)) / n
        start = _maxpro_swap(start, rng, swaps)  # stage 1: coordinate exchange (LHD basin)
        res = minimize(  # stage 2: continuous refinement off the grid
            _maxpro_obj_grad,
            start.ravel(),
            args=(n, d),
            method="L-BFGS-B",
            jac=True,
            bounds=box,
            options={"maxiter": int(maxiter)},
        )
        cand = np.clip(res.x.reshape(n, d), 0.0, 1.0)
        crit = _maxpro_criterion(cand)
        if crit < best_crit:
            best_crit, best_unit = crit, cand
    assert best_unit is not None
    return _scale_unit(best_unit, b)


def _qmc_unit(engine_cls: Any, d: int, n: int, scramble: bool, rng: RandomState) -> np.ndarray:
    """Draw ``n`` points in ``[0, 1]^d`` from a scipy ``qmc`` engine, seeded from ``rng``.

    Uses an integer seed drawn from ``rng`` so the draw is reproducible, and prefers the newer
    ``rng=`` keyword (falling back to the deprecated ``seed=`` on older scipy).
    """
    int_seed = int(rng.randint(2**31))
    try:
        engine = engine_cls(d=d, scramble=scramble, rng=int_seed)
    except TypeError:  # pragma: no cover - older scipy without the rng= alias
        engine = engine_cls(d=d, scramble=scramble, seed=int_seed)
    return np.asarray(engine.random(n), dtype=np.float64)


def sobol_design(bounds: Bounds, n: int, seed: int | RandomState | None = None, *, scramble: bool = True) -> np.ndarray:
    """Return an ``n``-point Sobol' low-discrepancy design over ``bounds``.

    Sobol' is a quasi-random sequence whose discrepancy (deviation from perfectly even coverage) is
    far lower than iid uniform sampling, so it fills the space more evenly for moderate ``n``.
    ``scramble=True`` applies Owen scrambling, which randomizes the sequence reproducibly from
    ``seed`` while preserving the low-discrepancy structure; ``scramble=False`` returns the raw
    deterministic sequence. Balance properties are best when ``n`` is a power of two.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    unit = _qmc_unit(qmc.Sobol, b.shape[0], int(n), scramble, rng)
    return _scale_unit(unit, b)


def halton_design(
    bounds: Bounds, n: int, seed: int | RandomState | None = None, *, scramble: bool = True
) -> np.ndarray:
    """Return an ``n``-point Halton low-discrepancy design over ``bounds``.

    Halton is a quasi-random sequence (coprime van der Corput sequences per axis) that, like Sobol',
    fills space more evenly than iid uniform sampling and works for any ``n`` (no power-of-two
    preference). ``scramble=True`` randomizes it reproducibly from ``seed``.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    unit = _qmc_unit(qmc.Halton, b.shape[0], int(n), scramble, rng)
    return _scale_unit(unit, b)


def full_factorial(bounds: Bounds, levels: int | Sequence[int]) -> np.ndarray:
    """Return a full-factorial grid design over ``bounds``.

    ``levels`` is the number of evenly-spaced levels per dimension (a scalar applied to all
    dimensions, or one entry per dimension, each ``>= 1``). The result has ``prod(levels)`` rows in
    row-major (last axis fastest) order. A dimension with one level is placed at its midpoint.
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    if isinstance(levels, int):
        level_list = [levels] * d
    else:
        level_list = list(levels)
    if len(level_list) != d:
        raise ValueError("levels must be a scalar or have one entry per dimension.")
    if any(k < 1 for k in level_list):
        raise ValueError("each level count must be >= 1.")

    axes = []
    for j, k in enumerate(level_list):
        low, high = b[j, 0], b[j, 1]
        axes.append(np.array([0.5 * (low + high)]) if k == 1 else np.linspace(low, high, k))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)
