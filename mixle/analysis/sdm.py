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
        """Central ``level`` interval of the pushed-forward samples (empirical quantiles)."""
        alpha = (1.0 - level) / 2.0
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
        self.beta = np.asarray(beta, dtype=np.float64)
        self.beta_cov = np.asarray(beta_cov, dtype=np.float64)
        self.design = np.asarray(design, dtype=np.float64)
        self.cell_area = np.asarray(cell_area, dtype=np.float64)
        self._var_scale = float(var_scale)
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
        """
        beta_draws = rng.multivariate_normal(self.beta, self._var_scale * self.beta_cov, size=int(n))
        return np.exp(beta_draws @ self.design.T)

    @property
    def mean(self) -> np.ndarray:
        """Fitted intensity field ``lambda_c = exp(design_c @ beta)`` -- the suitability surface."""
        return np.exp(self.design @ self.beta)

    @property
    def cov(self) -> np.ndarray:
        """Dense ``(K, K)`` delta-method covariance of the intensity field, recalibrated."""
        jac = self.mean[:, None] * self.design  # d(lambda_c)/d(beta) = lambda_c * X_c
        return self._var_scale * (jac @ self.beta_cov @ jac.T)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell central credible interval of the suitability field (lognormal delta-method)."""
        mu, var = self._log_lambda_moments()
        z = float(norm.ppf(0.5 + level / 2.0))
        sd = np.sqrt(var)
        return np.exp(mu - z * sd), np.exp(mu + z * sd)

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


def _bin_cell_counts(cell_indices: Sequence[float] | np.ndarray, num_cells: int) -> np.ndarray:
    """Per-cell counts via the exact IPP count-encoding: ``np.histogram`` over integer bin edges.

    Mirrors ``InhomogeneousPoissonProcessAccumulator``/``...DataEncoder``'s
    ``np.histogram(events, bins=edges)`` binning (inhomogeneous_poisson.py), treating each cell index
    ``[c, c+1)`` as one bin so the same frozen scorer can be reused unmodified.
    """
    edges = np.arange(num_cells + 1, dtype=np.float64)
    idx = np.asarray(list(cell_indices), dtype=np.float64) if len(cell_indices) else np.empty(0)
    if idx.size == 0:
        return np.zeros(num_cells, dtype=np.float64)
    counts, _ = np.histogram(idx, bins=edges)
    return counts.astype(np.float64)


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
    """
    num_cells, p = design.shape
    edges = np.arange(num_cells + 1, dtype=np.float64)

    def _nll_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        log_lambda = design @ beta
        rates = np.exp(np.clip(log_lambda + log_offset, -_RATE_CLIP, _RATE_CLIP))
        dist = InhomogeneousPoissonProcessDistribution(rates, edges=edges)
        log_lik = float(dist.seq_log_density(counts[None, :])[0])
        nll = -log_lik + ridge * float(beta @ beta)
        grad = -(design.T @ (counts - rates)) + 2.0 * ridge * beta
        return nll, grad

    beta0 = np.zeros(p, dtype=np.float64)
    result = minimize(_nll_and_grad, beta0, jac=True, method="L-BFGS-B")
    beta_hat = result.x
    rates_hat = np.exp(np.clip(design @ beta_hat + log_offset, -_RATE_CLIP, _RATE_CLIP))
    # Laplace covariance: (X^T diag(lambda_c * area_c) X + ridge * I)^{-1}
    hessian = design.T @ (rates_hat[:, None] * design) + ridge * np.eye(p)
    beta_cov = np.linalg.inv(hessian)
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
    """
    cov = np.atleast_2d(np.asarray(covariates, dtype=np.float64))
    num_cells = cov.shape[0]
    area = np.asarray(cell_area, dtype=np.float64).reshape(-1)
    if area.shape[0] != num_cells:
        raise ValueError("cell_area must have exactly one entry per covariate row (K cells).")
    design = np.column_stack([np.ones(num_cells), cov])
    p = design.shape[1]

    presence_idx = [float(np.asarray(o.location).reshape(-1)[0]) for o in occurrences if o.detection]
    counts = _bin_cell_counts(presence_idx, num_cells)

    if background is not None:
        bg = np.asarray(background, dtype=np.float64).reshape(-1)
        bg_counts = _bin_cell_counts(bg.tolist(), num_cells)
        # Convert background/quadrature point density into an area-equivalent unit: each background
        # point stands in for `mean(area) / len(background)` of extra survey opportunity, so cells with
        # heavier background sampling need proportionally more detections to imply the same intensity.
        thinning_weight = float(area.mean()) / float(max(bg.size, 1))
        effective_area = area + thinning_weight * bg_counts
    else:
        effective_area = area
    log_offset = np.log(np.clip(effective_area, 1e-12, None))

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
