"""Species-distribution / habitat-suitability modelling (workstream N, N1; IC-12).

A presence-only species-distribution model (SDM) fit as an inhomogeneous Poisson point process (IPP)
over a discretised study area -- the log-linear-intensity formulation that is mathematically equivalent
to MaxEnt (Renner & Warton, 2013). The study area is a grid of ``K`` cells with an environmental design
matrix ``X`` (``+ intercept``) and a per-cell area; presence detections are binned into per-cell counts
``n_c`` exactly the way :class:`~mixle.process.InhomogeneousPoissonProcessDistribution` bins event times
into per-bin counts. The log-intensity ``log lambda_c = X_c @ beta`` is fit by maximizing the *frozen*
IPP log-likelihood (never reimplemented here -- ``seq_log_density`` is called as the scorer) with a
ridge penalty and a ``log(area_c)`` offset; a Laplace approximation over ``beta`` is pushed forward
through the same log-link to give the fitted :class:`HabitatModel` a full IC-1 ``Posterior`` surface over
the suitability field, with its variance recalibrated on a held-out fold via
:func:`mixle.analysis.kriging.calibrate_variance` so credible intervals hit their nominal coverage.

Presence-only bias (uneven survey effort) is corrected by an offset-based analogue of the Berman--Turner
background/thinning device: optional ``background`` quadrature points are binned the same way as
presences and folded into the per-cell offset as extra effective area, so cells with more background
sampling opportunity require proportionally more detections to imply the same fitted intensity -- the fit
targets *relative* intensity rather than raw, effort-confounded detection counts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from mixle.analysis.kriging import calibrate_variance
from mixle.process import InhomogeneousPoissonProcessDistribution

__all__ = ["SpeciesObservation", "HabitatModel", "fit_sdm"]

# Internal train/calibration split for the held-out variance recalibration (every _HOLDOUT_STRIDE-th
# cell, by row order, is withheld from the calibration sub-fit and used to measure coverage).
_HOLDOUT_STRIDE = 5
_MIN_HOLDOUT_CELLS = 3
_MIN_TRAIN_CELLS = 8
_RATE_CLIP = 700.0  # exp() overflow guard on the log-intensity + offset


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """``exp`` with the same overflow guard used during fitting (MXR-080-0114).

    A beta fit under the training-time clip can still imply a linear predictor that overflows plain
    ``exp()`` when evaluated elsewhere -- a different design row (e.g. an extrapolated covariate), a wide
    posterior draw, or a directly-constructed :class:`HabitatModel` -- even though fitting itself never
    produced an overflow, because fitting always went through this same clip. Applying it at every public
    prediction site too keeps ``HabitatModel``'s outputs finite by construction rather than relying on
    callers to only ever ask for "reasonable" predictions.
    """
    return np.exp(np.clip(x, -_RATE_CLIP, _RATE_CLIP))


def _validate_level(level: float) -> float:
    """Validate a credible-interval level is finite and strictly in ``(0, 1)`` (MXR-080-0114).

    Mirrors :func:`mixle.analysis.kriging.calibrate_variance`'s own ``target`` validation: a level
    ``<= 0``, ``>= 1``, or NaN has no meaning as a central-interval mass and previously produced a
    silently nonsensical (even inverted-bounds, when negative) interval instead of raising.
    """
    lvl = float(level)
    if not (np.isfinite(lvl) and 0.0 < lvl < 1.0):
        raise ValueError(f"level must be finite and strictly in (0, 1), got {level!r}.")
    return lvl


def _validate_draw_count(n: int) -> int:
    """Validate a posterior draw count is a positive exact integer (MXR-080-0114).

    ``rng.multivariate_normal(..., size=int(n))`` previously truncated any non-integral ``n`` silently
    (e.g. ``2.7`` became ``2`` with no warning); a non-positive ``n`` is not a meaningful draw count.
    """
    n_int = int(n)
    if n != n_int or n_int <= 0:
        raise ValueError(f"n must be a positive exact integer, got {n!r}.")
    return n_int


def _validate_covariance(cov: np.ndarray, p: int, *, name: str) -> np.ndarray:
    """Validate ``cov`` is a finite, symmetric, ``(p, p)`` positive-semidefinite matrix (MXR-080-0114).

    Eigenvalue-based, consistent with the positive-(semi)definite check used elsewhere in this codebase
    (:func:`mixle.utils.vector.batched_pd_logdet`) rather than a determinant-sign test, which is not
    sufficient on its own (a matrix can have positive determinant while indefinite). Unlike that helper --
    written for distributions whose density requires strict positive-*definite*ness -- a Laplace
    covariance is a legitimate (if degenerate) covariance at exactly zero eigenvalue, so this allows the
    closed boundary (semidefinite) rather than rejecting it.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.shape != (p, p):
        raise ValueError(f"{name} must have shape ({p}, {p}) matching beta, got {c.shape}.")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"{name} must be finite.")
    if not np.allclose(c, c.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric.")
    eigvals = np.linalg.eigvalsh(c)
    tol = 1e-8 * max(float(np.max(np.abs(eigvals))), 1.0)
    if np.any(eigvals < -tol):
        raise ValueError(f"{name} must be positive-semidefinite; smallest eigenvalue is {float(eigvals.min())!r}.")
    return c


@dataclass
class SpeciesObservation:
    """One presence/absence record for a species (an IC-4 ``Observation`` specialisation).

    ``location`` is the (already discretised) study-area coordinate: for a presence record consumed by
    :func:`fit_sdm`, the first component of ``location`` is the fractional cell index (``[0, K)``) on the
    same grid as the ``covariates``/``cell_area`` passed to ``fit_sdm`` -- resolving a real-world
    ``crs``-referenced location onto that grid is covariate/CRS ingest (B-series), out of scope here.
    """

    species_id: str
    detection: bool
    location: np.ndarray
    crs: str | None = None
    covariates: dict[str, Any] = field(default_factory=dict)
    modality: str = "occurrence"
    provenance: dict[str, Any] = field(default_factory=dict)


class _PushforwardQuantity:
    """A minimal IC-1 ``DerivedQuantity``: pushforward draws + interval + the prior-dominated flag."""

    def __init__(self, samples: np.ndarray, prior_dominated: bool) -> None:
        self.samples = samples
        self.prior_dominated = prior_dominated

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Central ``level`` interval of the pushed-forward samples (empirical quantiles).

        Raises:
            ValueError: ``level`` is not finite and strictly in ``(0, 1)`` (MXR-080-0114).
        """
        lvl = _validate_level(level)
        alpha = (1.0 - lvl) / 2.0
        return np.quantile(self.samples, alpha, axis=0), np.quantile(self.samples, 1.0 - alpha, axis=0)


class HabitatModel:
    """Fitted IPP habitat-suitability field; satisfies IC-1 ``Posterior`` over the suitability field.

    ``beta``/``beta_cov`` are the Laplace-approximate posterior over the log-linear intensity
    coefficients; ``design`` is the ``(K, p)`` covariate design matrix (intercept + environmental
    covariates); ``cell_area`` is the per-cell area used as the Poisson offset during fitting. The
    suitability field is the fitted intensity ``lambda_c = exp(design_c @ beta)`` (:pyattr:`mean`);
    :meth:`samples`/:pyattr:`cov`/:meth:`credible_interval` push the beta-posterior forward through the
    same log-link (a delta-method / lognormal approximation), scaled by a held-out-calibrated variance
    multiplier, so every downstream consumer (N2's no-mine mask, N4's resistance raster) sees one
    calibrated field posterior.
    """

    def __init__(
        self,
        beta: np.ndarray,
        beta_cov: np.ndarray,
        design: np.ndarray,
        cell_area: np.ndarray,
        *,
        var_scale: float = 1.0,
        prior_dominated: bool = False,
    ) -> None:
        """Construct a fitted habitat-suitability posterior.

        Raises:
            ValueError: ``beta`` is not a non-empty finite 1-D array; ``design``/``cell_area`` are not
                finite with the shape ``beta`` implies (``design`` is ``(K, p)``, ``cell_area`` is
                ``(K,)``), or ``cell_area`` is not strictly positive; ``beta_cov`` is not a finite,
                symmetric, ``(p, p)`` positive-semidefinite matrix; or ``var_scale`` is not finite and
                strictly positive (MXR-080-0114: posterior construction must enforce a finite,
                shape-compatible, genuinely positive-semidefinite state rather than accept -- or silently
                clip -- an invalid one).
        """
        beta_arr = np.asarray(beta, dtype=np.float64).reshape(-1)
        if beta_arr.size == 0:
            raise ValueError("beta must be a non-empty 1-D array.")
        if not np.all(np.isfinite(beta_arr)):
            raise ValueError("beta must be finite.")
        p = beta_arr.shape[0]

        design_arr = np.asarray(design, dtype=np.float64)
        if design_arr.ndim != 2 or design_arr.shape[1] != p:
            raise ValueError(f"design must have shape (K, {p}) matching beta, got {design_arr.shape}.")
        if not np.all(np.isfinite(design_arr)):
            raise ValueError("design must be finite.")
        num_cells = design_arr.shape[0]

        cell_area_arr = np.asarray(cell_area, dtype=np.float64).reshape(-1)
        if cell_area_arr.shape != (num_cells,):
            raise ValueError(f"cell_area must have shape ({num_cells},) matching design, got {cell_area_arr.shape}.")
        if not np.all(np.isfinite(cell_area_arr)) or not np.all(cell_area_arr > 0.0):
            raise ValueError("cell_area must be finite and strictly positive.")

        beta_cov_arr = _validate_covariance(beta_cov, p, name="beta_cov")

        var_scale_f = float(var_scale)
        if not (np.isfinite(var_scale_f) and var_scale_f > 0.0):
            raise ValueError(f"var_scale must be finite and strictly positive, got {var_scale!r}.")

        self.beta = beta_arr
        self.beta_cov = beta_cov_arr
        self.design = design_arr
        self.cell_area = cell_area_arr
        self._var_scale = var_scale_f
        self._prior_dominated = bool(prior_dominated)

    def _log_lambda_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell ``(mean, calibrated variance)`` of ``log(lambda_c)`` under the beta-posterior."""
        mu = self.design @ self.beta
        raw_var = np.einsum("ci,ij,cj->c", self.design, self.beta_cov, self.design)
        return mu, self._var_scale * np.clip(raw_var, 0.0, None)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` intensity-field realisations by pushing ``beta`` draws through ``exp(X @ beta)``.

        Returns:
            ``(n, K)`` array of intensity-field draws.

        Raises:
            ValueError: ``n`` is not a positive exact integer (MXR-080-0114).
        """
        n_valid = _validate_draw_count(n)
        beta_draws = rng.multivariate_normal(self.beta, self._var_scale * self.beta_cov, size=n_valid)
        return _safe_exp(beta_draws @ self.design.T)

    @property
    def mean(self) -> np.ndarray:
        """Fitted intensity field ``lambda_c = exp(design_c @ beta)`` -- the suitability surface.

        Uses the same overflow-safe ``exp`` as fitting (MXR-080-0114): a linear predictor that never
        overflowed under the training-time clip can still overflow plain ``exp()`` at a different design
        row (e.g. an extrapolated covariate) or for a directly-constructed model.
        """
        return _safe_exp(self.design @ self.beta)

    @property
    def cov(self) -> np.ndarray:
        """Dense ``(K, K)`` delta-method covariance of the intensity field, recalibrated."""
        jac = self.mean[:, None] * self.design  # d(lambda_c)/d(beta) = lambda_c * X_c
        return self._var_scale * (jac @ self.beta_cov @ jac.T)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell central credible interval of the suitability field (lognormal delta-method).

        Raises:
            ValueError: ``level`` is not finite and strictly in ``(0, 1)`` (MXR-080-0114).
        """
        lvl = _validate_level(level)
        mu, var = self._log_lambda_moments()
        z = float(norm.ppf(0.5 + lvl / 2.0))
        sd = np.sqrt(var)
        return _safe_exp(mu - z * sd), _safe_exp(mu + z * sd)

    def derived_quantity(
        self, fn: Callable[[np.ndarray], np.ndarray], n: int, rng: np.random.Generator
    ) -> _PushforwardQuantity:
        """Pushforward ``fn`` over ``n`` intensity-field draws into a ``DerivedQuantity`` (IC-1)."""
        draws = self.samples(n, rng)
        return _PushforwardQuantity(np.asarray(fn(draws)), self._prior_dominated)

    def critical_habitat_mask(self, threshold: float) -> np.ndarray:
        """Boolean mask, True where fitted suitability ``lambda_c >= threshold``.

        The hard no-mine constraint N2 hands to H (same shape/role as a G9 seepage polygon).
        """
        return self.mean >= float(threshold)


def _validate_locations(raw: Sequence[float] | np.ndarray, num_cells: int, *, kind: str) -> np.ndarray:
    """Validate that every ``kind`` location is a finite cell index in the half-open domain ``[0, num_cells)``.

    ``_bin_cell_counts`` (like the ``np.histogram``-based binning it mirrors) silently ignores any value
    outside its bin edges rather than raising -- so an un-validated out-of-domain or NaN location does not
    error, it just vanishes from the fit as if the detection never happened (MXR-080-0111: a fit using only
    the locations ``-1`` and ``99`` completed normally, reporting zero detections everywhere). Locations
    reaching this point are expected to already be resolved onto the study-area grid (see
    :class:`SpeciesObservation`'s docstring), so a value outside ``[0, num_cells)`` reflects an upstream
    data/ingest bug, not a legitimate observation -- it is rejected outright rather than silently dropped,
    matching this module's existing convention of raising on invalid input.

    Raises:
        ValueError: any location is non-finite or outside ``[0, num_cells)``; the message reports how many
            failed each way (non-finite vs. out-of-domain) out of the total, plus a preview of the offending
            values, so the rejection is fully receipted rather than a bare "invalid input" refusal.
    """
    v = np.asarray(raw, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return v
    non_finite = ~np.isfinite(v)
    out_of_domain = ~non_finite & ((v < 0.0) | (v >= float(num_cells)))
    invalid = non_finite | out_of_domain
    if np.any(invalid):
        bad = v[invalid]
        preview = np.array2string(bad[: min(10, bad.size)], precision=3, separator=", ")
        raise ValueError(
            f"{kind} locations must be finite cell indices in [0, {num_cells}); rejected "
            f"{int(invalid.sum())} of {v.size} ({int(non_finite.sum())} non-finite, "
            f"{int(out_of_domain.sum())} outside [0, {num_cells})): {preview}"
            f"{' ...' if bad.size > 10 else ''}"
        )
    return v


def _bin_cell_counts(cell_indices: Sequence[float] | np.ndarray, num_cells: int) -> np.ndarray:
    """Per-cell counts via the exact IPP count-encoding: ``np.histogram`` over integer bin edges.

    Mirrors ``InhomogeneousPoissonProcessAccumulator``/``...DataEncoder``'s
    ``np.histogram(events, bins=edges)`` binning (inhomogeneous_poisson.py), treating each cell index
    ``[c, c+1)`` as one bin so the same frozen scorer can be reused unmodified. Callers are expected to
    have already rejected out-of-domain indices via :func:`_validate_locations`; this function stays
    correct in isolation too (MXR-080-0111): a phantom bin edge one past ``num_cells`` makes the true
    last cell ``[num_cells - 1, num_cells)`` left-closed/right-open like every other cell, instead of
    ``np.histogram``'s default closed-both-ends last bin, which would otherwise fold an index of exactly
    ``num_cells`` (nominally out-of-domain -- the domain is the half-open ``[0, num_cells)``) into cell
    ``num_cells - 1``.
    """
    idx = np.asarray(cell_indices, dtype=np.float64).reshape(-1)
    if idx.size == 0:
        return np.zeros(num_cells, dtype=np.float64)
    edges = np.arange(num_cells + 2, dtype=np.float64)  # one phantom edge beyond num_cells; see docstring
    counts, _ = np.histogram(idx, bins=edges)
    return counts[:num_cells].astype(np.float64)


def _clipped_rates_and_mask(
    beta: np.ndarray, design: np.ndarray, log_offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell fitted rate ``exp(clip(eta))`` and the clip's 0/1 derivative, at ``eta = design@beta + log_offset``.

    The clip is a standard optimizer-robustness guard: a wild L-BFGS-B line-search step can otherwise send
    ``eta`` far enough that ``exp(eta)`` overflows to ``inf``, which
    ``InhomogeneousPoissonProcessDistribution`` itself rejects outright (it requires finite rates) -- an
    unclipped evaluation would risk aborting the whole optimization on a step the line search would
    otherwise have corrected on its own.

    ``mask`` is ``d(clip(eta))/d(eta)``: ``1`` where ``eta`` is strictly inside ``(-_RATE_CLIP,
    _RATE_CLIP)`` (the clip is inactive, so ``rate`` responds to ``beta`` exactly like the unclipped
    ``exp``), and ``0`` where the clip has saturated ``rate`` at a constant (MXR-080-0113: outside the
    clip range, ``rate`` no longer changes with ``beta``, so by the chain rule neither the gradient nor
    the Hessian may treat it as if it still does). Every caller that differentiates through
    :func:`_clipped_rates_and_mask` must multiply by this mask to stay the exact derivative of the
    function actually being evaluated.
    """
    eta = design @ beta + log_offset
    mask = (eta > -_RATE_CLIP) & (eta < _RATE_CLIP)
    return _safe_exp(eta), mask.astype(np.float64)


def _nll_and_grad(
    beta: np.ndarray, design: np.ndarray, counts: np.ndarray, log_offset: np.ndarray, ridge: float, edges: np.ndarray
) -> tuple[float, np.ndarray]:
    """Penalized negative log-likelihood and its gradient at ``beta``.

    The frozen ``InhomogeneousPoissonProcessDistribution.seq_log_density`` is reused as-is, never
    reimplemented, to score the NLL value; only its (closed-form, standard Poisson-GLM) gradient is
    supplied locally, for speed. A module-level function (rather than a closure over ``_fit_beta``'s
    locals) so it can be exercised and finite-differenced directly by tests, independent of the optimizer.

    The gradient's data term is masked by the clip's derivative (MXR-080-0113): ``rate``'s dependence on
    ``beta`` vanishes wherever the clip has saturated it, so the unmasked ``design.T @ (rate - counts)``
    is only the derivative of the *unclipped* objective, not of the clipped ``nll`` value actually
    returned above -- silently wrong (and, per the audit, off by many orders of magnitude at a genuinely
    saturated cell) outside the clip range.
    """
    rates, mask = _clipped_rates_and_mask(beta, design, log_offset)
    dist = InhomogeneousPoissonProcessDistribution(rates, edges=edges)
    log_lik = float(dist.seq_log_density(counts[None, :])[0])
    nll = -log_lik + ridge * float(beta @ beta)
    grad = design.T @ (mask * (rates - counts)) + 2.0 * ridge * beta
    return nll, grad


def _penalized_hessian(beta: np.ndarray, design: np.ndarray, log_offset: np.ndarray, ridge: float) -> np.ndarray:
    """Exact Hessian of :func:`_nll_and_grad`'s objective at ``beta`` -- the Laplace posterior's precision.

    ``d^2/dbeta^2 [-loglik] = X^T diag(mask * rates) X`` is the standard Poisson-GLM Fisher information,
    restricted (via the same clip-derivative ``mask`` as the gradient, MXR-080-0113) to cells where the
    rate clip is inactive (MXR-080-0112 numerically verified this term, at ``mask == 1`` everywhere,
    against finite-differencing :func:`_nll_and_grad`'s gradient).
    ``d^2/dbeta^2 [ridge * beta @ beta] = 2 * ridge * I``: the gradient's ridge term is ``2 * ridge * beta``
    (correct), but differentiating a *first* derivative of ``2 * ridge * beta`` once more with respect to
    ``beta`` reproduces the same constant factor, ``2 * ridge * I`` -- not ``ridge * I``, which is what this
    Hessian previously added (MXR-080-0112), silently understating posterior uncertainty by folding in only
    half of the prior's actual curvature.
    """
    rates, mask = _clipped_rates_and_mask(beta, design, log_offset)
    p = design.shape[1]
    return design.T @ ((mask * rates)[:, None] * design) + 2.0 * ridge * np.eye(p)


def _fit_beta(
    design: np.ndarray, counts: np.ndarray, log_offset: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximize the frozen IPP log-likelihood (+ ridge penalty) for the log-linear coefficients.

    Builds an ``InhomogeneousPoissonProcessDistribution(rates, edges=arange(K+1))`` each evaluation and
    scores it with its own ``seq_log_density`` -- the frozen scorer is reused as-is, never reimplemented;
    only the (closed-form, standard Poisson-GLM) gradient of that same likelihood is supplied to the
    optimizer for speed.

    Returns:
        ``(beta, beta_cov, rates)`` -- the fitted coefficients, their Laplace covariance, and the fitted
        per-cell expected counts ``lambda_c * area_c`` (offset already folded in).

    Raises:
        ValueError: the optimizer does not report convergence, the fitted ``beta`` is not finite, or the
            penalized Hessian is singular (MXR-080-0113: an unconverged fit is not a valid posterior mode,
            so it must not be silently treated as one).
    """
    num_cells, p = design.shape
    edges = np.arange(num_cells + 1, dtype=np.float64)

    beta0 = np.zeros(p, dtype=np.float64)
    result = minimize(
        _nll_and_grad, beta0, args=(design, counts, log_offset, ridge, edges), jac=True, method="L-BFGS-B"
    )
    if not result.success:
        raise ValueError(f"SDM beta fit did not converge: {result.message}")
    beta_hat = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(beta_hat)):
        raise ValueError("SDM beta fit produced non-finite coefficients.")
    hessian = _penalized_hessian(beta_hat, design, log_offset, ridge)
    try:
        beta_cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "SDM penalized Hessian is singular; cannot form a Laplace covariance (try increasing ridge)."
        ) from exc
    rates_hat, _ = _clipped_rates_and_mask(beta_hat, design, log_offset)
    return beta_hat, beta_cov, rates_hat


def fit_sdm(
    occurrences: list[SpeciesObservation],
    covariates: np.ndarray,
    cell_area: np.ndarray,
    *,
    background: np.ndarray | None = None,
    ridge: float = 1e-3,
) -> HabitatModel:
    """Fit a presence-only habitat-suitability model as an inhomogeneous-Poisson point process.

    Discretizes the study area into ``K = covariates.shape[0]`` cells, bins ``occurrences`` (only
    ``detection=True`` records) into per-cell presence counts, and fits a log-linear intensity
    ``log lambda_c = [1, covariates_c] @ beta`` by maximizing the frozen IPP log-likelihood with a
    ``ridge * ||beta||^2`` penalty and a ``log(area_c)`` offset (Poisson GLM equivalent to MaxEnt). If
    ``background`` quadrature points are given, they are binned the same way and folded into the offset
    as extra effective area (an effort/thinning correction for presence-only sampling bias). The returned
    :class:`HabitatModel`'s field variance is recalibrated on an internal held-out cell fold via
    :func:`mixle.analysis.kriging.calibrate_variance` so its credible intervals hit their nominal (90%)
    coverage.

    Args:
        occurrences: presence records; each ``location``'s first component is the fractional cell index.
        covariates: ``(K, p - 1)`` environmental covariates per cell (an intercept column is prepended).
        cell_area: ``(K,)`` per-cell area (the Poisson offset).
        background: optional quadrature/background point locations (same cell-index convention as
            ``occurrences``), used to correct for uneven survey effort.
        ridge: L2 penalty strength on ``beta`` (also regularizes the Laplace covariance).

    Returns:
        A fitted :class:`HabitatModel`.

    Raises:
        ValueError: ``ridge`` is not finite and non-negative; ``covariates`` is not finite; ``cell_area``
            does not have exactly one finite, strictly positive entry per covariate row; any presence or
            background location is non-finite or outside the cell-index domain ``[0, K)`` (MXR-080-0111);
            or the internal beta fit does not converge to finite coefficients (MXR-080-0113).
    """
    if not (np.isfinite(ridge) and ridge >= 0.0):
        raise ValueError(f"ridge must be finite and non-negative, got {ridge!r}.")
    cov = np.atleast_2d(np.asarray(covariates, dtype=np.float64))
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariates must be finite.")
    num_cells = cov.shape[0]
    area = np.asarray(cell_area, dtype=np.float64).reshape(-1)
    if area.shape[0] != num_cells:
        raise ValueError("cell_area must have exactly one entry per covariate row (K cells).")
    if not np.all(np.isfinite(area)) or not np.all(area > 0.0):
        raise ValueError("cell_area must be finite and strictly positive.")
    design = np.column_stack([np.ones(num_cells), cov])
    p = design.shape[1]

    presence_idx = _validate_locations(
        [float(np.asarray(o.location).reshape(-1)[0]) for o in occurrences if o.detection],
        num_cells,
        kind="presence",
    )
    counts = _bin_cell_counts(presence_idx, num_cells)

    if background is not None:
        bg = _validate_locations(background, num_cells, kind="background")
        bg_counts = _bin_cell_counts(bg, num_cells)
        # Convert background/quadrature point density into an area-equivalent unit: each background
        # point stands in for `mean(area) / len(background)` of extra survey opportunity, so cells with
        # heavier background sampling need proportionally more detections to imply the same intensity.
        thinning_weight = float(area.mean()) / float(max(bg.size, 1))
        effective_area = area + thinning_weight * bg_counts
    else:
        effective_area = area
    # effective_area is always finite and strictly positive here: `area` is validated strictly positive
    # above, and adding a finite non-negative background-count term to it can only increase it -- no
    # pseudo-area clip is needed (MXR-080-0114: invalid area was previously silently turned into a tiny
    # positive pseudo-area, e.g. 1e-12, instead of being rejected).
    log_offset = np.log(effective_area)

    beta_hat, beta_cov, rates_hat = _fit_beta(design, counts, log_offset, ridge)

    data_curvature = float(np.trace(design.T @ (rates_hat[:, None] * design)))
    prior_curvature = float(ridge * p)
    prior_dominated = prior_curvature > data_curvature

    var_scale = 1.0
    idx = np.arange(num_cells)
    holdout_mask = (idx % _HOLDOUT_STRIDE) == 0
    train_mask = ~holdout_mask
    if holdout_mask.sum() >= _MIN_HOLDOUT_CELLS and train_mask.sum() >= max(_MIN_TRAIN_CELLS, p + 1):
        beta_train, beta_cov_train, _ = _fit_beta(design[train_mask], counts[train_mask], log_offset[train_mask], ridge)
        design_ho = design[holdout_mask]
        mu_ho = design_ho @ beta_train
        var_ho = np.einsum("ci,ij,cj->c", design_ho, beta_cov_train, design_ho)
        empirical_rate = (counts[holdout_mask] + 0.5) / np.exp(log_offset[holdout_mask])
        resid_ho = np.log(empirical_rate) - mu_ho
        var_scale = calibrate_variance(var_ho, resid_ho, target=0.9)

    return HabitatModel(
        beta=beta_hat,
        beta_cov=beta_cov,
        design=design,
        cell_area=area,
        var_scale=var_scale,
        prior_dominated=prior_dominated,
    )
