"""Optimal experimental design via information-matrix criteria (WS-E).

For a regression model with basis (model matrix) ``F = model(X)``, the information matrix is
``M = F.T @ F`` -- the Gaussian-noise Fisher information for the linear coefficients, up to the
noise variance. An "alphabetic" optimal design picks the ``n`` design points that optimize a scalar
functional of ``M``:

* **D-optimal** -- maximize ``log det M`` (shrink the joint confidence ellipsoid of the coefficients)
* **A-optimal** -- minimize ``trace(M^{-1})`` (shrink the average coefficient variance)
* **I-optimal** -- minimize the mean prediction variance over a reference set

Criteria are looked up through a registry (``register_criterion`` / ``criterion=`` name) following the
"register, don't branch" pattern; each returns a *merit* that is maximized. :func:`optimal_design`
selects points from a candidate pool (a Sobol design over the bounds, or a user-supplied array) by a
modified Fedorov exchange: from a random starting subset, repeatedly apply the single in-design /
candidate swap that most improves the criterion until no swap helps, across a few random restarts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations_with_replacement
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.doe.designs import Bounds, _as_bounds, _as_rng, sobol_design

ModelMatrix = Callable[[np.ndarray], np.ndarray]


def polynomial_features(degree: int = 1, *, bias: bool = True) -> ModelMatrix:
    """Return a model-matrix function building polynomial features up to ``degree`` (with interactions).

    The returned ``f(X)`` maps an ``(n, d)`` point array to an ``(n, p)`` model matrix: an optional
    intercept column, then every monomial ``prod(x_j)`` over index multisets of size ``1..degree``
    (so ``degree=1`` is linear, ``degree=2`` is a full quadratic response surface including the
    cross terms ``x_i x_j``).
    """
    if degree < 1:
        raise ValueError("degree must be >= 1.")

    def f(x: Any) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        n, d = x.shape
        cols = [np.ones(n)] if bias else []
        for deg in range(1, degree + 1):
            for combo in combinations_with_replacement(range(d), deg):
                col = np.ones(n)
                for j in combo:
                    col = col * x[:, j]
                cols.append(col)
        return np.column_stack(cols)

    return f


def d_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """D-optimality merit: ``log det M`` (``-inf`` if ``M`` is singular). Higher is better."""
    sign, logabsdet = np.linalg.slogdet(info)
    return float(logabsdet) if sign > 0 else -np.inf


def a_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """A-optimality merit: ``-trace(M^{-1})`` (``-inf`` if singular). Higher is better."""
    try:
        inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return -np.inf
    return float(-np.trace(inv))


def i_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """I-optimality merit: ``-mean`` prediction variance over ``ref`` (``-inf`` if singular).

    The prediction variance at a reference row ``g`` is ``g M^{-1} g``; this returns the negative
    mean over the reference model matrix ``ref`` so larger is better. Falls back to A-optimality when
    no reference set is supplied.
    """
    try:
        inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return -np.inf
    if ref is None:
        return float(-np.trace(inv))
    pred_var = np.einsum("ij,jk,ik->i", ref, inv, ref)
    return float(-np.mean(pred_var))


# --- criterion registry ("register, don't branch") ----------------------------------------------
# A criterion is ``fn(info, *, ref) -> merit`` where ``merit`` is maximized over candidate designs.
_CRITERIA: dict[str, Callable[..., float]] = {}


def register_criterion(name: str, fn: Callable[..., float], aliases: tuple[str, ...] = ()) -> None:
    """Register an optimality criterion ``fn`` under ``name`` (and any ``aliases``).

    ``fn`` is called as ``fn(info, *, ref)`` with the information matrix ``M = F.T @ F`` and must
    return a merit that :func:`optimal_design` maximizes. This is the extension point for new
    criteria -- registering is all that is needed, no edits to the exchange loop.
    """
    if not callable(fn):
        raise TypeError("criterion must be callable.")
    _CRITERIA[name.lower()] = fn
    for alias in aliases:
        _CRITERIA[alias.lower()] = fn


def available_criteria() -> list[str]:
    """Return the sorted names (and aliases) of all registered optimality criteria."""
    return sorted(_CRITERIA)


def _get_criterion(criterion: str | Callable[..., float]) -> Callable[..., float]:
    if callable(criterion):
        return criterion
    fn = _CRITERIA.get(str(criterion).lower())
    if fn is None:
        raise ValueError("unknown criterion %r; registered: %s" % (criterion, ", ".join(available_criteria())))
    return fn


register_criterion("d", d_criterion, aliases=("d_optimal", "d-optimal", "det"))
register_criterion("a", a_criterion, aliases=("a_optimal", "a-optimal", "trace"))
register_criterion("i", i_criterion, aliases=("i_optimal", "i-optimal", "iv"))


def _exchange(
    fmat: np.ndarray,
    n: int,
    crit: Callable[..., float],
    ref: np.ndarray | None,
    rng: RandomState,
    max_iter: int,
) -> tuple[list[int], float]:
    """One modified-Fedorov run from a random start; return (selected row indices, merit)."""
    p = fmat.shape[1]
    pool = fmat.shape[0]
    sel = list(rng.choice(pool, size=n, replace=False))
    in_design = set(sel)
    cur = crit(fmat[sel].T @ fmat[sel], ref=ref)

    for _ in range(int(max_iter)):
        best_gain = 1.0e-10
        best_swap: tuple[int, int, float] | None = None
        remaining = [c for c in range(pool) if c not in in_design]
        for pos in range(len(sel)):
            kept = sel[:pos] + sel[pos + 1 :]
            base = fmat[kept]
            for add in remaining:
                trial = np.vstack([base, fmat[add]])
                val = crit(trial.T @ trial, ref=ref)
                if val - cur > best_gain:
                    best_gain = val - cur
                    best_swap = (pos, add, val)
        if best_swap is None:
            break
        pos, add, val = best_swap
        in_design.discard(sel[pos])
        sel[pos] = add
        in_design.add(add)
        cur = val
    return sel, cur


def optimal_design(
    bounds: Bounds | None,
    n: int,
    *,
    candidates: np.ndarray | None = None,
    model: ModelMatrix | None = None,
    criterion: str | Callable[..., float] = "D",
    n_candidates: int = 256,
    n_restarts: int = 5,
    max_iter: int = 100,
    ref: np.ndarray | None = None,
    seed: int | RandomState | None = None,
) -> np.ndarray:
    """Return an ``n``-point optimal design selected from a candidate pool by Fedorov exchange.

    The pool is either generated as a Sobol design of ``n_candidates`` points over per-dimension
    ``bounds``, or supplied directly as an ``(P, d)`` ``candidates`` array (in which case ``bounds``
    may be ``None``). ``model`` is a model-matrix function (default :func:`polynomial_features` degree
    1, i.e. linear with intercept); ``criterion`` selects the optimality merit (``"D"`` / ``"A"`` /
    ``"I"`` or any registered name / callable). The best design over ``n_restarts`` random starts is
    returned as an ``(n, d)`` array. For ``"I"`` optimality, prediction variance is averaged over
    ``ref`` (a model matrix) when given, else over the candidate pool.

    Raises if ``n`` is below the number of model parameters (the information matrix would be singular).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    rng = _as_rng(seed)
    model = model or polynomial_features(1)

    if candidates is not None:
        pool = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    elif bounds is not None:
        # Round the Sobol pool up to a power of two for its balance properties (exact size is
        # not critical -- it is only the candidate set the exchange selects from).
        pool_n = 1 << int(np.ceil(np.log2(max(2, int(n_candidates)))))
        pool = sobol_design(_as_bounds(bounds), pool_n, rng)
    else:
        raise ValueError("provide either bounds (to generate a candidate pool) or an explicit candidates array.")
    if n > pool.shape[0]:
        raise ValueError("n cannot exceed the number of candidate points.")

    fmat = np.asarray(model(pool), dtype=np.float64)
    p = fmat.shape[1]
    if n < p:
        raise ValueError(f"n={n} is below the {p} model parameters; the information matrix is singular.")

    ref_mat = np.asarray(ref, dtype=np.float64) if ref is not None else fmat
    crit = _get_criterion(criterion)

    best_sel: list[int] | None = None
    best_val = -np.inf
    for _ in range(max(1, int(n_restarts))):
        sel, val = _exchange(fmat, int(n), crit, ref_mat, rng, max_iter)
        if val > best_val:
            best_val = val
            best_sel = sel
    assert best_sel is not None
    return pool[best_sel]


__all__: Sequence[str] = [
    "polynomial_features",
    "d_criterion",
    "a_criterion",
    "i_criterion",
    "register_criterion",
    "available_criteria",
    "optimal_design",
]
