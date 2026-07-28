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
            joined by their signed defining relation (for example ``"x0-x1"`` for opposite columns),
            the ``(effect, se)`` of the exact combined quantity that *is* estimable -- empty when the
            design has no aliasing.
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


def _alias_orientations(f: np.ndarray, group: list[int]) -> list[int]:
    """Return each aliased column's exact +/- orientation relative to the group's first column."""
    base = f[:, group[0]]
    return [1 if float(base @ f[:, index]) >= 0.0 else -1 for index in group]


def _format_signed_contrast(names: list[str], group: list[int], orientations: list[int]) -> str:
    """Format the estimable relation in the public effects units."""
    output_scale = 2 if any(index != 0 for index in group) else 1
    pieces: list[str] = []
    for position, (index, orientation) in enumerate(zip(group, orientations)):
        effect_divisor = 1 if index == 0 else 2
        weight = output_scale * orientation / effect_divisor
        magnitude = abs(weight)
        label = names[index] if magnitude == 1.0 else f"{magnitude:g}*{names[index]}"
        if position == 0:
            pieces.append(label if weight > 0.0 else f"-{label}")
        else:
            pieces.append(("+" if weight > 0.0 else "-") + label)
    return "".join(pieces)


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
        value = float(coef_fit[gi])
        se_value = float(se_fit[gi]) if se_fit is not None else None
        orientations = _alias_orientations(f, g)
        output_scale = 2.0 if any(index != 0 for index in g) else 1.0
        eff_value = output_scale * value
        eff_se = None if se_value is None else output_scale * se_value
        contrast_name = _format_signed_contrast(names, g, orientations)
        estimable_contrasts[contrast_name] = (eff_value, eff_se)

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
        stationary_point: ``x*`` solving the stationarity equation ``B x* = -b/2``, or ``None`` when no
            such point exists -- i.e. the gradient ``b + 2 B x`` is never zero anywhere, which can only
            happen when ``B`` is singular (``kind == "no_stationary_point"``; see below).
        eigenvalues: eigenvalues of ``B`` -- all negative => a maximum, all positive => a minimum,
            mixed signs => a saddle; any eigenvalue ~0 (*with* a genuine stationary point) => a
            *ridge*, a whole line/subspace of equally-stationary points, of which
            ``stationary_point`` is one representative (the minimum-norm) point.
        kind: ``"maximum"`` / ``"minimum"`` / ``"saddle"`` / ``"ridge"`` / ``"no_stationary_point"``.
            The last case is distinct from a ridge: a ridge is a genuine (degenerate) stationary
            subspace, while ``"no_stationary_point"`` means the stationarity equation has no solution
            at all (e.g. the exact surface ``y = x0**2 + x1``, whose gradient's second component is
            always exactly ``1``).
        residual_std: residual standard deviation when the design has spare runs (else ``None``).
        model_rank: rank of the fitted quadratic model matrix.
        n_parameters: number of quadratic coefficients requested.
        degrees_of_freedom: residual degrees of freedom, ``n_runs - n_parameters``.
        estimable: whether every requested coefficient is identified. Construction refuses a
            non-estimable fit, so returned surfaces always report ``True``.
    """

    coef: np.ndarray
    terms: list[str]
    b: np.ndarray
    B: np.ndarray
    stationary_point: np.ndarray | None
    eigenvalues: np.ndarray
    kind: str
    residual_std: float | None
    model_rank: int
    n_parameters: int
    degrees_of_freedom: int
    estimable: bool

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
    ``(n, d)`` and responses ``y``, then solves the stationarity equation ``B x* = -b/2`` for the
    stationary point and classifies it from the eigenvalues of the quadratic matrix ``B``. When ``B``
    is singular a solution only exists if ``-b/2`` lies in its range, which is checked explicitly via
    the residual of the stationarity equation -- a least-squares "solution" that does not actually zero
    the gradient is reported as ``kind="no_stationary_point"`` rather than a fake stationary point (see
    :class:`ResponseSurface`). Fit on the *coded* design for a well-conditioned model (the classic
    central-composite / Box-Behnken workflow).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"x must be a non-empty two-dimensional (n, d) design; got shape {x.shape}.")
    if y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x must be (n, d) with one response per row.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite design evidence.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain one finite response per design row.")
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
    with np.errstate(over="ignore", invalid="ignore"):
        f = np.column_stack(cols)
    if not np.all(np.isfinite(f)):
        raise ValueError("the quadratic model matrix is not representable as finite float64.")
    model_rank = int(np.linalg.matrix_rank(f))
    n_parameters = f.shape[1]
    dof = n - n_parameters
    if model_rank != n_parameters:
        raise ValueError(
            "full quadratic response surface is not estimable: "
            f"model-matrix rank is {model_rank} of {n_parameters} parameters with {n} run(s)."
        )
    coef, residual, *_ = np.linalg.lstsq(f, y, rcond=None)
    if not np.all(np.isfinite(coef)):
        raise ValueError("quadratic response-surface coefficients are not finite.")
    rstd = float(np.sqrt(residual[0] / dof)) if residual.size and dof > 0 else None
    if rstd is not None and not np.isfinite(rstd):
        raise ValueError("quadratic response-surface residual standard deviation is not finite.")

    b = coef[1 : 1 + d].copy()
    bmat = np.zeros((d, d), dtype=np.float64)
    k = 1 + d
    for i, j in combinations(range(d), 2):
        bmat[i, j] = bmat[j, i] = 0.5 * coef[k]  # cross term split symmetrically
        k += 1
    for j in range(d):
        bmat[j, j] = coef[k]
        k += 1

    # Solve the stationarity equation B x* = -b/2 by least squares unconditionally -- this gives the
    # exact solution when B is full rank and the minimum-norm least-squares "candidate" when B is
    # singular. A candidate is only a genuine stationary point if it actually zeros the residual of
    # that equation; lstsq happily returns a best-fit point even when -b/2 is not in B's range (no
    # exact solution exists at all), so the residual must be checked explicitly rather than trusted.
    rhs = -0.5 * b
    xs_candidate, _, _, sv = np.linalg.lstsq(bmat, rhs, rcond=None)
    resid_norm = float(np.linalg.norm(bmat @ xs_candidate - rhs))
    scale = max(1.0, float(np.linalg.norm(rhs)), float(sv[0]) if sv.size else 0.0)
    has_stationary_point = resid_norm <= 1e-7 * scale

    eig = np.linalg.eigvalsh(bmat)
    tol = 1e-9 * max(1.0, float(np.max(np.abs(eig))))
    is_ridge = bool(np.any(np.abs(eig) <= tol))

    if not has_stationary_point:
        # e.g. the exact surface y = x0**2 + x1: B = [[2, 0], [0, 0]] is singular, but the gradient's
        # x1-component is the constant 1 and can never be cancelled -- there is no stationary point,
        # not even a ridge, and reporting one (with a nonzero gradient) would be silently wrong.
        kind = "no_stationary_point"
        xs = None
    elif is_ridge:
        kind = "ridge"
        xs = xs_candidate
    elif np.all(eig < -tol):
        kind = "maximum"
        xs = xs_candidate
    elif np.all(eig > tol):
        kind = "minimum"
        xs = xs_candidate
    else:
        kind = "saddle"
        xs = xs_candidate
    return ResponseSurface(
        coef=coef,
        terms=names,
        b=b,
        B=bmat,
        stationary_point=xs,
        eigenvalues=eig,
        kind=kind,
        residual_std=rstd,
        model_rank=model_rank,
        n_parameters=n_parameters,
        degrees_of_freedom=dof,
        estimable=True,
    )


def design_diagnostics(design, model, *, ref=None) -> dict:
    """Quality diagnostics for a design under a model -- "is this a good design to run?".

    Builds the model matrix ``F = model(design)`` and reports, relative to a hypothetical perfectly
    orthogonal design (where each is ``1.0``):

      * ``d_efficiency`` -- ``det(M)**(1/p) / n``: overall coefficient-estimation precision;
      * ``a_efficiency`` -- ``p / (n * trace(M^-1))``: average coefficient variance;
      * ``g_efficiency`` -- ``p / (n * max prediction variance)`` over ``ref`` (or the design itself);
      * ``condition_number`` of ``M`` (large => near-collinear / fragile to fit; ``None`` when the
        requested model is rank-deficient), plus a finite ``effective_condition_number`` within the
        estimable subspace and explicit ``rank`` / ``full_rank`` fields;
      * ``max_correlation`` -- the largest absolute pairwise correlation among the non-intercept model
        columns (the aliasing check; ``0`` for an orthogonal design).

    ``model`` is a model-matrix function such as :func:`mixle.doe.optimal.polynomial_features`. Use it on
    the *coded* design for meaningful efficiencies.

    ``ref`` -- when given -- is a *reference design*: raw candidate points in the same ``(m, d)``
    raw-factor-column form as ``design`` (e.g. a finer grid over the same region, or a wider region to
    probe extrapolation risk), not a pre-built model matrix. It is passed through the same ``model``
    used to build the design's own model matrix before its prediction variance is computed, so
    ``g_efficiency`` compares like with like; a ``ref`` whose raw column count does not match
    ``design``'s raises. ``g_efficiency`` is guaranteed to fall on the documented ``(0, 1]`` scale only
    for the self-referential case (``ref=None``, or ``ref`` equal to ``design`` itself) -- a reference
    set concentrated in a lower-variance region than the design's own points (e.g. just its centre) can
    legitimately score above ``1.0``.

    The raw design and every transformed model matrix must be nonempty, two-dimensional, finite, and
    preserve the run axis. A rank-zero model is rejected. A partially estimable model reports zero
    full-model efficiencies, ``condition_number=None``, and its finite effective condition number
    rather than publishing NaN/Inf quality evidence.
    """
    design_array = np.asarray(design, dtype=np.float64)
    if design_array.ndim != 2 or design_array.shape[0] == 0 or design_array.shape[1] == 0:
        raise ValueError(
            f"design must be a non-empty two-dimensional (n, d) array; got shape {design_array.shape}."
        )
    if not np.all(np.isfinite(design_array)):
        raise ValueError("design must contain only finite values.")
    f = np.asarray(model(design_array), dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != design_array.shape[0] or f.shape[1] == 0:
        raise ValueError(
            "model(design) must be a non-empty two-dimensional matrix with one row per design run; "
            f"got shape {f.shape} for {design_array.shape[0]} run(s)."
        )
    if not np.all(np.isfinite(f)):
        raise ValueError("model(design) must contain only finite features.")
    n, p = f.shape
    with np.errstate(over="ignore", invalid="ignore"):
        m = f.T @ f
    if not np.all(np.isfinite(m)):
        raise ValueError("design information matrix is not representable as finite float64.")
    singular_values = np.linalg.svd(f, compute_uv=False)
    rank = int(np.linalg.matrix_rank(f))
    if rank == 0:
        raise ValueError("model(design) has rank zero; no parameter contrast is estimable.")
    full_rank = rank == p
    sign, logdet = np.linalg.slogdet(m)
    with np.errstate(over="ignore", invalid="ignore"):
        d_eff = float(np.exp(logdet / p) / n) if full_rank and sign > 0 else 0.0

    if ref is None:
        pts = f
    else:
        # `ref` is documented as a reference *design* -- raw points, symmetric with `design` -- not an
        # already-expanded model matrix, so it must go through the same `model` transform `design`
        # did. Using it directly here previously fed e.g. a single raw feature column straight into a
        # two-column quadratic form below; numpy's einsum broadcasts a size-1 axis instead of raising a
        # shape error, so the result was silently wrong (and unboundedly so), not a crash.
        ref_array = np.asarray(ref, dtype=np.float64)
        if (
            ref_array.ndim != 2
            or ref_array.shape[0] == 0
            or ref_array.shape[1] != design_array.shape[1]
            or not np.all(np.isfinite(ref_array))
        ):
            raise ValueError(
                "ref must be a non-empty finite two-dimensional reference design with "
                f"{design_array.shape[1]} raw factor column(s); got shape {ref_array.shape}."
            )
        pts = np.asarray(model(ref_array), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] != ref_array.shape[0] or pts.shape[1] != p:
            width = pts.shape[1] if pts.ndim == 2 else f"{pts.ndim}-d"
            raise ValueError(
                f"ref, once passed through `model`, has {width} feature column(s); expected {p} to "
                "match the design's own model matrix. `ref` must be raw reference-design points (the "
                "same kind of input as `design`), not a pre-built model matrix."
            )
        if not np.all(np.isfinite(pts)):
            raise ValueError("model(ref) must contain only finite features.")

    if full_rank:
        inv = np.linalg.inv(m)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            a_eff = float(p / (n * np.trace(inv)))
            cond = float(np.linalg.cond(m))
            pred_var_raw = np.einsum("ij,jk,ik->i", pts, inv, pts)
        pred_scale = max(float(np.max(np.abs(pred_var_raw))), 1.0)
        if np.any(pred_var_raw < -1e-10 * pred_scale):
            raise ValueError("design diagnostics produced materially negative prediction variance.")
        pred_var = np.maximum(pred_var_raw, 0.0)
        max_var = float(np.max(pred_var))
        if max_var <= 0.0:
            raise ValueError("prediction variance is exactly zero at every reference point; g_efficiency is undefined.")
        g_eff = float(p / (n * max_var))
        if ref is None and g_eff > 1.0 + 1e-6:
            # Guaranteed by the hat-matrix identity trace(H) = p when the reference points ARE the
            # design's own points (their leverages sum to p, so the max is always >= the average p/n).
            # A violation here means the computation above regressed, not that this is a legitimate
            # diagnostic to report.
            raise RuntimeError(
                f"g_efficiency {g_eff!r} exceeds its theoretical <=1 ceiling for a design compared "
                "against its own points; this indicates a numerical bug, not a valid design."
            )
    else:
        # A rank-deficient design cannot estimate all requested parameters, so the full-model
        # efficiencies are exactly zero. Its ordinary condition number is infinite; expose that state
        # explicitly and report the finite condition number within the estimable subspace separately.
        a_eff = g_eff = 0.0
        cond = None
    effective_cond = float(singular_values[0] / singular_values[rank - 1])
    cols = f[:, 1:] if p > 1 and np.allclose(f[:, 0], 1.0) else f
    # A zero-variance column (a factor that never varies in this design, or a one-level dimension
    # passed through `model`) makes np.corrcoef produce 0/0 = NaN entries for that column, which then
    # silently propagates through max() into the returned diagnostic -- "is my design good" answered
    # with NaN instead of a meaningful score. Exclude degenerate columns from the correlation check;
    # a constant column has no correlation to report, not an undefined one.
    with np.errstate(over="ignore", invalid="ignore"):
        column_variance = np.var(cols, axis=0) if cols.shape[1] else np.array([])
    if not np.all(np.isfinite(column_variance)):
        raise ValueError("design diagnostic column variances are not finite.")
    varying = cols[:, column_variance > 0] if cols.shape[1] else cols
    if varying.shape[1] >= 2:
        corr = np.corrcoef(varying, rowvar=False)
        max_corr = float(np.max(np.abs(corr - np.eye(corr.shape[0]))))
    else:
        max_corr = 0.0
    numeric_diagnostics = np.asarray([d_eff, a_eff, g_eff, effective_cond, max_corr], dtype=np.float64)
    if not np.all(np.isfinite(numeric_diagnostics)):
        raise ValueError("design diagnostics produced non-finite quality evidence.")
    return {
        "d_efficiency": d_eff,
        "a_efficiency": a_eff,
        "g_efficiency": g_eff,
        "condition_number": cond,
        "effective_condition_number": effective_cond,
        "max_correlation": max_corr,
        "n_runs": int(n),
        "n_params": int(p),
        "rank": rank,
        "full_rank": full_rank,
    }


__all__ = [
    "FactorialEffects",
    "factorial_effects",
    "ResponseSurface",
    "response_surface",
    "design_diagnostics",
]
