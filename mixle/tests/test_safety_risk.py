"""K4 DoD -- safety-risk / geotechnical hazard modeling (notes/exec/workstream-K.md).

A synthetic subsidence grid that is flat everywhere except a known steep column band (a linear ramp
of a fixed slope, big enough to trip a chosen ``gradient_limit``). ``safety_risk_surface`` should map
that field's per-cell tilt into a spatial exceedance-probability surface whose high-risk cells line up
with the steep band (IoU >= 0.7), whether the deformation arrives as a raw ``ndarray`` or as an IC-1
``Posterior`` with per-cell noise around the same ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.health_risk import incident_probability, safety_risk_surface
from mixle.reason.posterior_protocol import DerivedQuantity, Posterior


class _DeformationPosterior:
    """Minimal IC-1 `Posterior` over a `(rows, cols)` subsidence grid: Gaussian per-cell noise around
    a known-truth mean grid, exposed flat (`mean`/`samples`) the way `PosteriorField3D` would be."""

    def __init__(self, mean_grid: np.ndarray, noise_std: float = 0.05):
        self._mean_grid = np.asarray(mean_grid, dtype=float)
        self.grid_shape = self._mean_grid.shape
        self._noise_std = noise_std
        self._d = self._mean_grid.size

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        flat_mean = self._mean_grid.reshape(-1)
        return flat_mean[None, :] + rng.normal(0.0, self._noise_std, size=(n, self._d))

    @property
    def mean(self) -> np.ndarray:
        return self._mean_grid.reshape(-1)

    @property
    def cov(self) -> np.ndarray:
        return np.eye(self._d) * self._noise_std**2

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = 1.9599639845400545  # ~95% two-sided normal quantile; a synthetic stub, not a real fit
        halfwidth = z * self._noise_std
        m = self.mean
        return m - halfwidth, m + halfwidth

    def derived_quantity(self, fn, n: int, rng: np.random.Generator):
        out = fn(self.samples(n, rng))

        class _DQ:
            samples = out
            prior_dominated = False

            def credible_interval(self, level: float):
                a = (1.0 - level) / 2.0
                return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)

        return _DQ()


def _steep_band_grid(rows: int, cols: int, c0: int, c1: int, step: float) -> np.ndarray:
    """A `(rows, cols)` field: flat (0) before column `c0`, a linear ramp of slope `step` through
    `[c0, c1)`, then flat again -- identical across every row, so the true steep zone is exactly the
    column band `[c0, c1)` with no row-direction gradient contamination."""
    ramp_col = np.zeros(cols)
    for j in range(c0, c1):
        ramp_col[j] = (j - c0) * step
    ramp_col[c1:] = ramp_col[c1 - 1]
    return np.tile(ramp_col, (rows, 1))


def _iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    return float(intersection) / float(union) if union else 1.0


def test_subsidence_maps_to_risk():
    rows, cols, c0, c1, step, gradient_limit = 24, 24, 8, 16, 4.0, 1.0
    mean_grid = _steep_band_grid(rows, cols, c0, c1, step)
    posterior = _DeformationPosterior(mean_grid, noise_std=0.05)

    dq = safety_risk_surface(posterior, gradient_limit=gradient_limit)

    assert isinstance(dq, DerivedQuantity)
    risk = np.asarray(dq.samples).mean(axis=0).reshape(rows, cols)
    predicted_mask = risk > 0.5

    true_mask = np.zeros((rows, cols), dtype=bool)
    true_mask[:, c0:c1] = True

    assert _iou(predicted_mask, true_mask) >= 0.7
    assert dq.prior_dominated is False


def test_ndarray_input_is_deterministic_and_matches_zone_exactly():
    rows, cols, c0, c1, step, gradient_limit = 16, 16, 4, 10, 4.0, 1.0
    grid = _steep_band_grid(rows, cols, c0, c1, step)

    dq = safety_risk_surface(grid, gradient_limit=gradient_limit)

    assert isinstance(dq, DerivedQuantity)
    predicted_mask = np.asarray(dq.samples).reshape(rows, cols) > 0.5
    true_mask = np.zeros((rows, cols), dtype=bool)
    true_mask[:, c0:c1] = True
    assert _iou(predicted_mask, true_mask) == 1.0
    assert dq.prior_dominated is False


def test_slope_adds_to_deformation_gradient():
    rows, cols = 10, 10
    flat = np.zeros((rows, cols))

    dq_no_slope = safety_risk_surface(flat, gradient_limit=0.5)
    assert np.asarray(dq_no_slope.samples).sum() == 0.0

    steep_slope = np.full((rows, cols), 0.6)
    dq_with_slope = safety_risk_surface(flat, gradient_limit=0.5, slope=steep_slope)
    assert np.asarray(dq_with_slope.samples).sum() == float(rows * cols)


def test_posterior_stub_conforms_to_ic1():
    assert isinstance(_DeformationPosterior(np.zeros((3, 3))), Posterior)


def test_incident_probability_monotone_in_hazard_and_exposure():
    hazard = np.array([[0.1, 0.9], [0.1, 0.9]])
    p_no_exposure = incident_probability(hazard, np.zeros((2, 2)))
    p_high_exposure = incident_probability(hazard, np.full((2, 2), 5.0))

    assert np.all((p_no_exposure >= 0.0) & (p_no_exposure <= 1.0))
    assert np.all(p_high_exposure >= p_no_exposure)
    # a hazard-free, densely-occupied cell still carries essentially no incident risk
    assert p_high_exposure[0, 0] < p_high_exposure[0, 1]


def test_incident_probability_shape_mismatch_raises():
    with pytest.raises(ValueError):
        incident_probability(np.zeros((2, 2)), np.zeros((3, 3)))


def test_incident_probability_unknown_model_raises():
    with pytest.raises(ValueError):
        incident_probability(np.zeros((2, 2)), np.zeros((2, 2)), model="bogus")
