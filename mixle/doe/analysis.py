"""Analysis of designed experiments: factorial effects and second-order response surfaces.

These turn the runs of a design (see :mod:`mixle.doe.factorial`) and their measured responses into the
quantities a practitioner reads off: the *effect* of each factor and interaction in a two-level
design, and -- for a response-surface design -- the fitted second-order model, its stationary point,
and the canonical (eigenvalue) analysis that says whether that point is a maximum, minimum, or saddle.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class FactorialEffects:
    """Estimated effects from a two-level factorial / fractional-factorial / Plackett-Burman design.

    Attributes:
        terms: term names (``"intercept"``, ``"x0"``, ``"x0:x1"``, ...).
        coef: least-squares regression coefficients in coded ``+/-1`` units.
        effects: the classical *effect* per term -- the change in mean response as a factor moves from
            its low to its high level, i.e. ``2 * coef`` (the intercept entry is just the grand mean).
        intercept: the grand mean of the response.
        residual_std: residual standard deviation when the design has spare runs (else ``None``).
    """

    terms: list[str]
    coef: np.ndarray
    effects: np.ndarray
    intercept: float
    residual_std: float | None

    def as_dict(self) -> dict[str, float]:
        """Map each non-intercept term to its effect."""
        return {t: float(e) for t, e in zip(self.terms, self.effects) if t != "intercept"}


def _code_two_level(x: np.ndarray) -> np.ndarray:
    """Map each column's two distinct levels to ``-1`` / ``+1`` (a one-level column maps to 0)."""
    x = np.asarray(x, dtype=np.float64)
    coded = np.empty_like(x)
    for j in range(x.shape[1]):
        u = np.unique(x[:, j])
        if u.size == 1:
            coded[:, j] = 0.0
        elif u.size == 2:
            coded[:, j] = np.where(x[:, j] == u[1], 1.0, -1.0)
        else:
            raise ValueError(f"factor {j} has {u.size} levels; factorial_effects needs two-level factors.")
    return coded


def factorial_effects(design, y, *, interactions: bool = True, coded: bool = False) -> FactorialEffects:
    """Estimate main effects and two-factor interactions from a two-level design.

    Fits the linear model ``y ~ 1 + x_i (+ x_i x_j)`` in coded ``+/-1`` units by least squares; the
    coefficients are half the classical effects. ``design`` is the ``(n, d)`` run matrix (the real
    factor levels, coded to ``+/-1`` automatically -- or pass ``coded=True`` if it is already ``+/-1``),
    ``y`` the measured responses. Set ``interactions=False`` for a main-effects-only (e.g. screening)
    fit.
    """
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("design must be (n, d) with one response per row.")
    xc = x if coded else _code_two_level(x)
    n, d = xc.shape
    cols = [np.ones(n)]
    names = ["intercept"]
    for j in range(d):
        cols.append(xc[:, j])
        names.append(f"x{j}")
    if interactions:
        for i, j in combinations(range(d), 2):
            cols.append(xc[:, i] * xc[:, j])
            names.append(f"x{i}:x{j}")
    f = np.column_stack(cols)
    coef, residual, *_ = np.linalg.lstsq(f, y, rcond=None)
    dof = n - f.shape[1]
    rstd = float(np.sqrt(residual[0] / dof)) if residual.size and dof > 0 else None
    effects = 2.0 * coef.copy()
    effects[0] = coef[0]  # the intercept is the grand mean, not an effect
    return FactorialEffects(names, coef, effects, float(coef[0]), rstd)


@dataclass
class ResponseSurface:
    """A fitted second-order response surface ``y = b0 + b'x + x'Bx`` and its canonical analysis.

    Attributes:
        coef: full coefficient vector (intercept, linears, then the upper-triangular second-order terms).
        terms: matching term names.
        b: linear coefficient vector ``(d,)``.
        B: symmetric ``(d, d)`` matrix of quadratic coefficients (cross terms split onto both halves).
        stationary_point: ``x*`` solving ``grad = b + 2 B x = 0`` (least-squares if ``B`` is singular).
        eigenvalues: eigenvalues of ``B`` -- all negative => the stationary point is a maximum, all
            positive => a minimum, mixed signs => a saddle (a *ridge* if some are ~0).
        kind: ``"maximum"`` / ``"minimum"`` / ``"saddle"``.
        residual_std: residual standard deviation when the design has spare runs (else ``None``).
    """

    coef: np.ndarray
    terms: list[str]
    b: np.ndarray
    B: np.ndarray
    stationary_point: np.ndarray
    eigenvalues: np.ndarray
    kind: str
    residual_std: float | None

    def predict(self, x) -> np.ndarray:
        """Predict the response at points ``x`` ``(m, d)`` from the fitted surface."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        quad = np.einsum("ni,ij,nj->n", x, self.B, x)
        return self.coef[0] + x @ self.b + quad

    def gradient(self, x) -> np.ndarray:
        """The response gradient ``b + 2 B x`` at ``x`` -- its direction is the path of steepest ascent."""
        x = np.asarray(x, dtype=np.float64)
        return self.b + 2.0 * self.B @ x


def response_surface(x, y) -> ResponseSurface:
    """Fit a full second-order (quadratic) response surface and analyse its stationary point.

    Least-squares-fits ``y = b0 + sum b_i x_i + sum_{i<=j} b_{ij} x_i x_j`` to the design runs ``x``
    ``(n, d)`` and responses ``y``, then solves for the stationary point ``x* = -1/2 B^{-1} b`` and
    classifies it from the eigenvalues of the quadratic matrix ``B``. Fit on the *coded* design for a
    well-conditioned model (the classic central-composite / Box-Behnken workflow).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x must be (n, d) with one response per row.")
    n, d = x.shape
    cols = [np.ones(n)]
    names = ["intercept"]
    for j in range(d):
        cols.append(x[:, j])
        names.append(f"x{j}")
    for i, j in combinations(range(d), 2):
        cols.append(x[:, i] * x[:, j])
        names.append(f"x{i}:x{j}")
    for j in range(d):
        cols.append(x[:, j] ** 2)
        names.append(f"x{j}^2")
    f = np.column_stack(cols)
    coef, residual, *_ = np.linalg.lstsq(f, y, rcond=None)
    dof = n - f.shape[1]
    rstd = float(np.sqrt(residual[0] / dof)) if residual.size and dof > 0 else None

    b = coef[1 : 1 + d].copy()
    bmat = np.zeros((d, d), dtype=np.float64)
    k = 1 + d
    for i, j in combinations(range(d), 2):
        bmat[i, j] = bmat[j, i] = 0.5 * coef[k]  # cross term split symmetrically
        k += 1
    for j in range(d):
        bmat[j, j] = coef[k]
        k += 1

    if abs(np.linalg.det(bmat)) > 1e-12:
        xs = np.linalg.solve(bmat, -0.5 * b)
    else:  # a ridge system: least-squares stationary point
        xs = np.linalg.lstsq(2.0 * bmat, -b, rcond=None)[0]
    eig = np.linalg.eigvalsh(bmat)
    tol = 1e-9 * max(1.0, float(np.max(np.abs(eig))))
    if np.all(eig < -tol):
        kind = "maximum"
    elif np.all(eig > tol):
        kind = "minimum"
    else:
        kind = "saddle"
    return ResponseSurface(coef, names, b, bmat, xs, eig, kind, rstd)


def design_diagnostics(design, model, *, ref=None) -> dict:
    """Quality diagnostics for a design under a model -- "is this a good design to run?".

    Builds the model matrix ``F = model(design)`` and reports, relative to a hypothetical perfectly
    orthogonal design (where each is ``1.0``):

      * ``d_efficiency`` -- ``det(M)**(1/p) / n``: overall coefficient-estimation precision;
      * ``a_efficiency`` -- ``p / (n * trace(M^-1))``: average coefficient variance;
      * ``g_efficiency`` -- ``p / (n * max prediction variance)`` over ``ref`` (or the design itself);
      * ``condition_number`` of ``M`` (large => near-collinear / fragile to fit);
      * ``max_correlation`` -- the largest absolute pairwise correlation among the non-intercept model
        columns (the aliasing check; ``0`` for an orthogonal design).

    ``model`` is a model-matrix function such as :func:`mixle.doe.optimal.polynomial_features`. Use it on
    the *coded* design for meaningful efficiencies.
    """
    f = np.asarray(model(design), dtype=np.float64)
    n, p = f.shape
    m = f.T @ f
    sign, logdet = np.linalg.slogdet(m)
    d_eff = float(np.exp(logdet / p) / n) if sign > 0 else 0.0
    try:
        inv = np.linalg.inv(m)
        a_eff = float(p / (n * np.trace(inv)))
        cond = float(np.linalg.cond(m))
        pts = np.asarray(ref, dtype=np.float64) if ref is not None else f
        pred_var = np.einsum("ij,jk,ik->i", pts, inv, pts)
        g_eff = float(p / (n * np.max(pred_var)))
    except np.linalg.LinAlgError:
        a_eff = g_eff = 0.0
        cond = float("inf")
    cols = f[:, 1:] if p > 1 and np.allclose(f[:, 0], 1.0) else f
    # A zero-variance column (a factor that never varies in this design, or a one-level dimension
    # passed through `model`) makes np.corrcoef produce 0/0 = NaN entries for that column, which then
    # silently propagates through max() into the returned diagnostic -- "is my design good" answered
    # with NaN instead of a meaningful score. Exclude degenerate columns from the correlation check;
    # a constant column has no correlation to report, not an undefined one.
    varying = cols[:, np.var(cols, axis=0) > 0] if cols.shape[1] else cols
    if varying.shape[1] >= 2:
        corr = np.corrcoef(varying, rowvar=False)
        max_corr = float(np.max(np.abs(corr - np.eye(corr.shape[0]))))
    else:
        max_corr = 0.0
    return {
        "d_efficiency": d_eff,
        "a_efficiency": a_eff,
        "g_efficiency": g_eff,
        "condition_number": cond,
        "max_correlation": max_corr,
        "n_runs": int(n),
        "n_params": int(p),
    }


__all__ = [
    "FactorialEffects",
    "factorial_effects",
    "ResponseSurface",
    "response_surface",
    "design_diagnostics",
]
