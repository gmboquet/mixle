"""Space-filling and classical experiment-design generators.

Every generator takes per-dimension ``bounds`` -- a sequence of ``(low, high)`` pairs -- and
returns a ``(n, d)`` numpy array of points scaled into those bounds. Random designs accept a
``seed`` (int or ``numpy.random.RandomState``) for reproducibility, matching the rest of pysp.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.random import RandomState

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
