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

from mixle.doe._contracts import Criterion
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, sobol_design

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


def _validate_info(info: np.ndarray) -> np.ndarray:
    """Validate ``info`` as a genuine information matrix: finite, symmetric, and PSD.

    A real information matrix is ``M = F.T @ F`` for some real model matrix ``F``, which is always
    finite, symmetric, and positive-semi-definite (PSD) by construction -- every eigenvalue is
    ``>= 0``. A *singular* (rank-deficient) ``M`` is still a legitimate, if undesirable, degenerate
    case -- it passes here, and criteria report ``-inf`` merit for it. A matrix that is not finite,
    not symmetric, or has a genuinely negative eigenvalue (e.g. ``-I``, which is negative-definite)
    could never arise from a real ``F.T @ F``, so it is rejected outright rather than silently scored.
    """
    info = np.asarray(info, dtype=np.float64)
    if info.ndim != 2 or info.shape[0] != info.shape[1]:
        raise ValueError(f"information matrix must be square 2-D; got shape {info.shape}.")
    if not np.all(np.isfinite(info)):
        raise ValueError("information matrix must be finite (no NaN/Inf entries).")
    if not np.allclose(info, info.T, atol=1e-8, rtol=1e-6):
        raise ValueError("information matrix must be symmetric (M = F.T @ F is always symmetric).")
    eigvals = np.linalg.eigvalsh(info)
    scale = max(1.0, float(np.max(np.abs(eigvals))))
    if float(eigvals[0]) < -1e-8 * scale:
        raise ValueError(
            "information matrix must be positive-semi-definite; got a minimum eigenvalue of "
            f"{float(eigvals[0]):.6g}, which cannot arise from a real F.T @ F."
        )
    return info


def _validate_ref(ref: np.ndarray, p: int) -> np.ndarray:
    """Validate a reference model matrix has shape ``(*, p)``, matching the information matrix."""
    ref = np.asarray(ref, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != p:
        raise ValueError(
            f"reference model matrix must be 2-D with {p} columns to match the information matrix "
            f"dimension; got shape {ref.shape}."
        )
    if not np.all(np.isfinite(ref)):
        raise ValueError("reference model matrix must be finite (no NaN/Inf entries).")
    return ref


def d_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """D-optimality merit: ``log det M`` (``-inf`` if ``M`` is singular). Higher is better.

    Raises ``ValueError`` if ``info`` is not a finite, symmetric, PSD matrix. ``log det`` is computed
    via a Cholesky factorization (PSD-aware: it fails cleanly on a singular ``M`` rather than a raw
    determinant, which would happily return a number for any square matrix regardless of definiteness).
    """
    info = _validate_info(info)
    try:
        chol = np.linalg.cholesky(info)
    except np.linalg.LinAlgError:
        return -np.inf
    return float(2.0 * np.sum(np.log(np.diag(chol))))


def a_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """A-optimality merit: ``-trace(M^{-1})`` (``-inf`` if singular). Higher is better.

    Raises ``ValueError`` if ``info`` is not a finite, symmetric, PSD matrix.
    """
    info = _validate_info(info)
    try:
        # trace(M^{-1}) via solving M X = I, avoiding an explicit inverse.
        inv = np.linalg.solve(info, np.eye(info.shape[0]))
    except np.linalg.LinAlgError:
        return -np.inf
    return float(-np.trace(inv))


def i_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """I-optimality merit: ``-mean`` prediction variance over ``ref`` (``-inf`` if singular).

    The prediction variance at a reference row ``g`` is ``g M^{-1} g``; this returns the negative
    mean over the reference model matrix ``ref`` so larger is better. Falls back to A-optimality when
    no reference set is supplied.

    Raises ``ValueError`` if ``info`` is not a finite, symmetric, PSD matrix, or if ``ref`` is given
    and its column count does not match ``info``'s dimension.
    """
    info = _validate_info(info)
    p = info.shape[0]
    if ref is not None:
        ref = _validate_ref(ref, p)
    try:
        if ref is None:
            # Fall back to A-optimality: trace(M^{-1}) via solving M X = I.
            return float(-np.trace(np.linalg.solve(info, np.eye(p))))
        # Prediction variance g M^{-1} g per reference row via solving M Y = ref.T.
        sol = np.linalg.solve(info, ref.T)
    except np.linalg.LinAlgError:
        return -np.inf
    pred_var = np.einsum("ij,ji->i", ref, sol)
    return float(-np.mean(pred_var))


def g_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """G-optimality merit: ``-max`` prediction variance over ``ref`` (``-inf`` if singular).

    Where I-optimality minimizes the *average* prediction variance, G-optimality minimizes its
    *worst case* over the reference region, so the fitted surface has a bounded error everywhere.
    Returns the negative maximum of ``g M^{-1} g`` over the reference rows (falls back to the largest
    coefficient variance ``max diag(M^{-1})`` when no reference set is given).

    Raises ``ValueError`` if ``info`` is not a finite, symmetric, PSD matrix, or if ``ref`` is given
    and its column count does not match ``info``'s dimension.
    """
    info = _validate_info(info)
    p = info.shape[0]
    if ref is not None:
        ref = _validate_ref(ref, p)
    try:
        if ref is None:
            return float(-np.max(np.diag(np.linalg.solve(info, np.eye(p)))))
        sol = np.linalg.solve(info, ref.T)
    except np.linalg.LinAlgError:
        return -np.inf
    return float(-np.max(np.einsum("ij,ji->i", ref, sol)))


def e_criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
    """E-optimality merit: the smallest eigenvalue of ``M`` (higher is better).

    Maximizing the minimum eigenvalue of the information matrix shrinks the variance along the
    *worst-determined* parameter contrast, so no direction in coefficient space is left poorly
    estimated. ``0`` for a singular (rank-deficient) design.

    Raises ``ValueError`` if ``info`` is not a finite, symmetric, PSD matrix.
    """
    info = _validate_info(info)
    return float(np.linalg.eigvalsh(info)[0])  # eigvalsh is ascending -> [0] is the smallest


def c_criterion(c: np.ndarray) -> Criterion:
    """Return a c-optimality criterion targeting the linear combination ``c'.beta`` of coefficients.

    c-optimality minimizes the variance of a *specific* quantity of interest ``c'.beta`` (e.g. a
    contrast or a prediction at one point). The returned criterion has merit ``-c' M^{-1} c`` (``-inf``
    if singular), so it plugs straight into :func:`optimal_design` or :func:`register_criterion`.

    Raises ``ValueError`` (eagerly, at construction) if ``c`` is not 1-D, and (per call) if ``info``
    is not a finite, symmetric, PSD matrix or if ``c``'s length does not match ``info``'s dimension.
    """
    cvec = np.asarray(c, dtype=np.float64)
    if cvec.ndim != 1:
        raise ValueError(f"contrast vector c must be 1-D; got shape {cvec.shape}.")

    def criterion(info: np.ndarray, *, ref: np.ndarray | None = None) -> float:
        info = _validate_info(info)
        if cvec.shape[0] != info.shape[0]:
            raise ValueError(
                f"contrast vector c has length {cvec.shape[0]}, expected {info.shape[0]} to match "
                "the information matrix dimension."
            )
        try:
            return float(-cvec @ np.linalg.solve(info, cvec))
        except np.linalg.LinAlgError:
            return -np.inf

    return criterion


# --- criterion registry ("register, don't branch") ----------------------------------------------
# A criterion is ``fn(info, *, ref) -> merit`` where ``merit`` is maximized over candidate designs.
_CRITERIA: dict[str, Criterion] = {}


def register_criterion(name: str, fn: Criterion, aliases: tuple[str, ...] = ()) -> None:
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


def _get_criterion(criterion: str | Criterion) -> Criterion:
    if callable(criterion):
        return criterion
    fn = _CRITERIA.get(str(criterion).lower())
    if fn is None:
        raise ValueError("unknown criterion %r; registered: %s" % (criterion, ", ".join(available_criteria())))
    return fn


register_criterion("d", d_criterion, aliases=("d_optimal", "d-optimal", "det"))
register_criterion("a", a_criterion, aliases=("a_optimal", "a-optimal", "trace"))
register_criterion("i", i_criterion, aliases=("i_optimal", "i-optimal", "iv"))
register_criterion("g", g_criterion, aliases=("g_optimal", "g-optimal", "minimax"))
register_criterion("e", e_criterion, aliases=("e_optimal", "e-optimal", "eigen"))


class InfeasibleDesignError(ValueError):
    """Raised when no finite-merit design exists for a candidate pool and model (rank-deficient)."""


def _exchange(
    fmat: np.ndarray,
    n: int,
    crit: Criterion,
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
    criterion: str | Criterion = "D",
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

    Raises ``ValueError`` if ``n`` is below the number of model parameters (the information matrix
    would be singular), and :class:`InfeasibleDesignError` (a ``ValueError`` subclass) if ``n`` is
    otherwise sufficient by count but the candidate pool's model matrix is rank-deficient -- e.g. a
    constant or otherwise degenerate pool, where ``n >= p`` rows can never make up for the pool
    itself spanning fewer than ``p`` independent directions -- or if, defensively, every exchange
    restart still fails to find a finite-merit design.
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
    # n >= p (row count) is necessary but not sufficient: a rank-deficient pool (e.g. constant or
    # otherwise degenerate candidates) cannot support a full-rank n-point design no matter how large
    # n is, since the rank of any row subset is bounded by the rank of the full pool. Catch that
    # up front rather than discovering it only after every restart returns -inf.
    pool_rank = int(np.linalg.matrix_rank(fmat))
    if pool_rank < p:
        raise InfeasibleDesignError(
            f"the candidate pool's model matrix has rank {pool_rank} < {p} model parameters; no "
            f"{n}-point subset of it can be estimable regardless of n. Supply a richer/less-"
            "degenerate candidate pool or a lower-degree model."
        )

    ref_mat = np.asarray(ref, dtype=np.float64) if ref is not None else fmat
    crit = _get_criterion(criterion)

    best_sel: list[int] | None = None
    best_val = -np.inf
    for _ in range(max(1, int(n_restarts))):
        sel, val = _exchange(fmat, int(n), crit, ref_mat, rng, max_iter)
        if val > best_val:
            best_val = val
            best_sel = sel
    if best_sel is None:
        # Defense in depth: the upfront rank check catches the common case, but stays a stable,
        # never-optimized-away error (unlike a bare `assert`, which `python -O` strips entirely)
        # for any other way every restart could fail to find a finite-merit design.
        raise InfeasibleDesignError(
            f"no restart of the exchange search found a finite-merit {n}-point design under "
            f"criterion {criterion!r} over {pool.shape[0]} candidates; the candidate pool may be "
            "degenerate for this model."
        )
    return pool[best_sel]


__all__: Sequence[str] = [
    "Criterion",
    "InfeasibleDesignError",
    "polynomial_features",
    "d_criterion",
    "a_criterion",
    "i_criterion",
    "register_criterion",
    "available_criteria",
    "optimal_design",
]
