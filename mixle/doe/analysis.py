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

    Only *estimable* quantities are reported as effects. When the design cannot separate two or more
    terms -- classical aliasing/confounding, including a factor that never varies -- those individual
    terms have no unique estimate: their entries in ``coef`` / ``effects`` are ``nan``. ``aliases``
    says which terms are aliased with which; ``estimable_contrasts`` gives the one well-defined
    *combined* quantity the data actually supports for each such group.

    Attributes:
        terms: term names (``"intercept"``, ``"x0"``, ``"x0:x1"``, ...).
        coef: least-squares regression coefficients in coded ``+/-1`` units (``nan`` for a term that is
            aliased with another and so has no individual estimate).
        effects: the classical *effect* per term -- the change in mean response as a factor moves from
            its low to its high level, i.e. ``2 * coef`` (the intercept entry is just the grand mean).
        intercept: the grand mean of the response (``nan`` if the intercept itself is aliased, e.g. with
            a factor that never varies).
        residual_std: residual standard deviation when the design has spare runs (else ``None``).
        se: standard error of each entry of ``effects``. ``None`` throughout when the design has no
            spare degrees of freedom to estimate a residual variance (e.g. a saturated design);
            individual entries are ``nan`` wherever ``effects`` is (aliased terms).
        aliases: for each term, the *other* term names it is aliased with (empty list if the term is
            uniquely estimable).
        estimable_contrasts: for each alias group of two or more terms, keyed by its member term names
            joined with ``"+"`` (in term order), the ``(effect, se)`` of the one combined quantity that
            *is* estimable -- empty when the design has no aliasing.
    """

    terms: list[str]
    coef: np.ndarray
    effects: np.ndarray
    intercept: float
    residual_std: float | None
    se: np.ndarray | None
    aliases: dict[str, list[str]]
    estimable_contrasts: dict[str, tuple[float, float | None]]

    def as_dict(self) -> dict[str, float]:
        """Map each non-intercept term to its effect (``nan`` where aliased -- see ``estimable_contrasts``)."""
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


def _alias_groups(f: np.ndarray, tol: float = 1e-9) -> list[list[int]]:
    """Partition model-matrix column indices into alias sets.

    Every column of a two-level model matrix is built from ``+/-1`` entries, so two columns are either
    identical, exact opposites, or not perfectly correlated -- there is no in-between. "Column ``i`` is
    proportional to column ``j``" is therefore a genuine equivalence relation here: it is transitive, so
    it correctly follows a multi-term classical alias chain (e.g. the defining relation ``I = ABC``
    aliases ``A`` with ``BC``, ``B`` with ``AC``, *and* ``C`` with ``AB``). Columns in the same group
    carry exactly the same information and cannot be told apart by the data. A column of all zeros (a
    factor that never varies, from :func:`_code_two_level`'s single-level convention) is left in its own
    singleton group -- it is not proportional to anything, it is simply uninformative.
    """
    p = f.shape[1]
    norms = np.linalg.norm(f, axis=0)
    parent = list(range(p))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(p):
        if norms[i] < tol:
            continue
        for j in range(i + 1, p):
            if norms[j] < tol:
                continue
            cos = (f[:, i] @ f[:, j]) / (norms[i] * norms[j])
            if abs(abs(cos) - 1.0) < tol:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(p):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def factorial_effects(design, y, *, interactions: bool = True, coded: bool = False) -> FactorialEffects:
    """Estimate main effects and two-factor interactions from a two-level design.

    Fits the linear model ``y ~ 1 + x_i (+ x_i x_j)`` in coded ``+/-1`` units by least squares; the
    coefficients are half the classical effects. ``design`` is the ``(n, d)`` run matrix (the real
    factor levels, coded to ``+/-1`` automatically -- or pass ``coded=True`` if it is already coded, in
    which case every entry must be exactly ``-1.0`` or ``1.0``), ``y`` the measured responses. Set
    ``interactions=False`` for a main-effects-only (e.g. screening) fit.

    Rejects designs that cannot support the requested model outright: non-finite values, fewer runs
    than model parameters, and (for ``coded=True``) levels other than ``+/-1``. A design that has
    *enough* runs but still cannot separate every term -- aliasing, including a factor that never
    varies -- is not rejected: the aliased terms come back as ``nan`` in ``coef`` / ``effects``, with
    the one combined quantity the data does support reported in ``estimable_contrasts`` (see
    :class:`FactorialEffects`). That automatic resolution only covers the classical case where the
    aliasing is exact term-for-term confounding, as in any standard fractional-factorial /
    Plackett-Burman alias structure; a design that is rank-deficient for some other reason (e.g. too few
    runs and not a clean fractional design) raises instead of guessing which contrasts might be
    estimable.
    """
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("design must be (n, d) with one response per row.")
    if not np.all(np.isfinite(x)):
        raise ValueError("design contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values.")
    if coded:
        for j in range(x.shape[1]):
            col = x[:, j]
            bad = col[(col != -1.0) & (col != 1.0)]
            if bad.size:
                levels = sorted(set(bad.tolist()))
                raise ValueError(
                    f"factor {j} has non-coded level(s) {levels}; coded=True requires every factor to "
                    "already be at exactly -1.0/+1.0 (pass coded=False to auto-code raw factor levels)."
                )
        xc = x
    else:
        xc = _code_two_level(x)
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
    p = f.shape[1]
    # Deliberately no separate "n < p" gate: a classical fractional-factorial or Plackett-Burman
    # design routinely has fewer runs than the full requested model has parameters, and that is fine
    # as long as the aliasing is exactly resolvable (see the reduced-rank check below, which is what
    # actually distinguishes a legitimate fractional design from a genuinely undersized one).
    rank = int(np.linalg.matrix_rank(f))
    aliases: dict[str, list[str]] = {name: [] for name in names}
    estimable_contrasts: dict[str, tuple[float, float | None]] = {}
    if rank == p:
        groups = [[k] for k in range(p)]
        fit_cols = f
    else:
        groups = _alias_groups(f)
        fit_cols = np.column_stack([f[:, g[0]] for g in groups])
        if int(np.linalg.matrix_rank(fit_cols)) < fit_cols.shape[1]:
            zero_cols = [names[k] for k in range(p) if np.linalg.norm(f[:, k]) < 1e-9]
            hint = f" (factor column(s) that never vary: {zero_cols})" if zero_cols else ""
            raise ValueError(
                f"design is rank-deficient (rank {rank} of {p} parameters) beyond simple term "
                f"aliasing{hint}; cannot resolve a unique set of estimable contrasts. Reduce the "
                "requested model (e.g. interactions=False) or add runs."
            )
        for g in groups:
            if len(g) > 1:
                for k in g:
                    aliases[names[k]] = [names[m] for m in g if m != k]

    coef_fit, residual, *_ = np.linalg.lstsq(fit_cols, y, rcond=None)
    dof = n - fit_cols.shape[1]
    rstd = float(np.sqrt(residual[0] / dof)) if residual.size and dof > 0 else None
    se_fit = None
    if dof > 0 and residual.size:
        sigma2 = residual[0] / dof
        cov_fit = sigma2 * np.linalg.inv(fit_cols.T @ fit_cols)
        se_fit = np.sqrt(np.clip(np.diag(cov_fit), 0.0, None))

    coef = np.full(p, np.nan)
    se = np.full(p, np.nan) if se_fit is not None else None
    for gi, g in enumerate(groups):
        if len(g) == 1:
            coef[g[0]] = coef_fit[gi]
            if se is not None:
                se[g[0]] = se_fit[gi]
            continue
        base = g[0]
        value = float(coef_fit[gi])
        se_value = float(se_fit[gi]) if se_fit is not None else None
        eff_value = value if base == 0 else 2.0 * value
        eff_se = None if se_value is None else (se_value if base == 0 else 2.0 * se_value)
        estimable_contrasts["+".join(names[k] for k in g)] = (eff_value, eff_se)

    effects = 2.0 * coef.copy()
    effects[0] = coef[0]  # the intercept is the grand mean, not an effect
    se_effects = None
    if se is not None:
        se_effects = 2.0 * se.copy()
        se_effects[0] = se[0]

    return FactorialEffects(
        terms=names,
        coef=coef,
        effects=effects,
        intercept=float(coef[0]),
        residual_std=rstd,
        se=se_effects,
        aliases=aliases,
        estimable_contrasts=estimable_contrasts,
    )


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
