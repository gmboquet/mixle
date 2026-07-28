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

from mixle.analysis.health_risk import _DeterministicRisk, incident_probability, safety_risk_surface
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


def test_incident_probability_rejects_out_of_range_or_non_finite_hazard():
    """MXR-080-0098: an out-of-[0,1] or non-finite hazard is REJECTED, not silently clipped into a
    falsely-confident boundary probability (the pre-fix "logit" path clipped 5.0/-3.0 straight into
    [eps, 1-eps] without ever checking they were valid probabilities)."""
    exposure = np.zeros((2, 2))
    for bad_hazard in (
        np.array([[5.0, 0.5], [0.5, 0.5]]),
        np.array([[-3.0, 0.5], [0.5, 0.5]]),
        np.array([[float("nan"), 0.5], [0.5, 0.5]]),
        np.array([[float("inf"), 0.5], [0.5, 0.5]]),
    ):
        with pytest.raises(ValueError):
            incident_probability(bad_hazard, exposure)
        with pytest.raises(ValueError):
            incident_probability(bad_hazard, exposure, model="linear")


def test_incident_probability_rejects_non_finite_exposure_map():
    hazard = np.full((2, 2), 0.5)
    with pytest.raises(ValueError):
        incident_probability(hazard, np.array([[float("nan"), 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        incident_probability(hazard, np.array([[float("inf"), 0.0], [0.0, 0.0]]))


def test_incident_probability_valid_boundary_hazard_unchanged():
    """Negative control for MXR-080-0098: legitimate hazard values (including the exact 0/1 boundary,
    which the eps-clip still numerically stabilizes) keep working."""
    hazard = np.array([[0.0, 1.0], [0.25, 0.75]])
    p = incident_probability(hazard, np.zeros((2, 2)))
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_safety_risk_surface_rejects_invalid_gradient_limit_slope_and_deformation():
    """MXR-080-0098: gradient_limit must be finite and non-negative; slope and an ndarray deformation
    must be finite (a NaN either way would silently compare False against "> gradient_limit")."""
    grid = np.zeros((4, 4))
    for bad_limit in (-1.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            safety_risk_surface(grid, gradient_limit=bad_limit)

    bad_slope = np.full((4, 4), float("nan"))
    with pytest.raises(ValueError):
        safety_risk_surface(grid, gradient_limit=1.0, slope=bad_slope)

    bad_grid = np.zeros((4, 4))
    bad_grid[1, 1] = float("nan")
    with pytest.raises(ValueError):
        safety_risk_surface(bad_grid, gradient_limit=1.0)

    posterior = _DeformationPosterior(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        safety_risk_surface(posterior, gradient_limit=-1.0)
    with pytest.raises(ValueError):
        safety_risk_surface(posterior, gradient_limit=1.0, slope=np.full((3, 3), float("nan")))


@pytest.mark.parametrize("failure", ["nan", "wrong_shape"])
def test_safety_risk_surface_rejects_invalid_posterior_field_draws(failure):
    class InvalidPosterior(_DeformationPosterior):
        def samples(self, n, rng):
            if failure == "nan":
                return np.full((n, self._d), np.nan)
            return np.zeros((n, self._d - 1))

    with pytest.raises(ValueError, match="posterior deformation draws"):
        safety_risk_surface(InvalidPosterior(np.zeros((3, 3))), gradient_limit=1.0)


def test_safety_risk_surface_validates_returned_derived_quantity():
    class InvalidResultPosterior(_DeformationPosterior):
        def derived_quantity(self, fn, n, rng):
            fn(self.samples(n, rng))
            return type(
                "Quantity",
                (),
                {"samples": np.full((n, self._d), np.nan), "prior_dominated": False},
            )()

    with pytest.raises(ValueError, match="safety-risk samples"):
        safety_risk_surface(InvalidResultPosterior(np.zeros((3, 3))), gradient_limit=1.0)


def test_safety_risk_surface_valid_gradient_limit_unchanged():
    """Negative control for MXR-080-0098: a legitimate (including exactly zero) gradient_limit still
    produces the expected deterministic exceedance surface."""
    rows, cols = 6, 6
    flat = np.zeros((rows, cols))
    dq_zero_limit = safety_risk_surface(flat, gradient_limit=0.0)
    assert np.asarray(dq_zero_limit.samples).sum() == 0.0  # a flat field has zero gradient everywhere


def test_deterministic_risk_rejects_empty_or_non_finite_samples():
    """`_DeterministicRisk` (the ndarray-deformation carrier `safety_risk_surface` returns) had no
    construction-time validation: empty or NaN/Inf samples were silently accepted, unlike its sibling
    `_SampleDerivedQuantity` in the same module. Defense-in-depth so invalid state can never flow
    downstream even if some upstream pushforward fails to validate its own inputs."""
    with pytest.raises(ValueError):
        _DeterministicRisk(samples=np.zeros((1, 0)), grid_shape=(0,))
    with pytest.raises(ValueError):
        _DeterministicRisk(samples=np.array([[1.0, np.nan]]), grid_shape=(2,))
    with pytest.raises(ValueError):
        _DeterministicRisk(samples=np.array([[1.0, np.inf]]), grid_shape=(2,))


def test_deterministic_risk_accepts_valid_samples():
    """Negative control: a legitimate, non-empty, finite exceedance-indicator array still constructs
    cleanly and behaves as documented (credible interval collapses to a point)."""
    risk = _DeterministicRisk(samples=np.array([[0.0, 1.0, 1.0, 0.0]]), grid_shape=(2, 2))
    lo, hi = risk.credible_interval(0.9)
    assert np.array_equal(lo, hi)


# --------------------------------------------------------------------------------------------------
# MXR-080-1589: the "logit" model added log1p(exposure_map) to the hazard log-odds. log1p(0) is 0, so
# an unoccupied cell added no log-odds and the incident probability came back EQUAL to the hazard --
# contradicting this function's stated contract that an exceedance is only an incident when someone
# is exposed. It was also unbounded above, reporting more incidents than hazard exceedances.
# --------------------------------------------------------------------------------------------------
def test_zero_occupancy_means_no_incident_not_an_unchanged_hazard():
    """Audit repro: hazard 0.8 with exposure 0 used to return incident probability 0.8, not 0."""
    hazard = np.array([[0.8, 0.5], [0.2, 1.0]])
    p = incident_probability(hazard, np.zeros((2, 2)))
    np.testing.assert_allclose(p, np.zeros((2, 2)), atol=1e-8)
    # the "linear" model has always had this zero element; both scales must now agree on it
    np.testing.assert_allclose(
        incident_probability(hazard, np.zeros((2, 2)), model="linear"), np.zeros((2, 2)), atol=1e-8
    )


def test_incident_probability_never_exceeds_the_hazard_it_is_conditioned_on():
    """An exceedance becomes an incident only when someone is exposed, so incidents can never
    outnumber exceedances. The old unbounded log1p(exposure) odds boost let a dense cell report an
    incident probability well above its own hazard probability."""
    hazard = np.array([0.1, 0.5, 0.8])
    for exposure_level in (0.0, 0.5, 1.0, 5.0, 100.0, 1e6):
        p = incident_probability(hazard, np.full(hazard.shape, exposure_level))
        assert np.all(p <= hazard + 1e-9), f"exposure {exposure_level} produced {p} above hazard {hazard}"
        assert np.all((p >= 0.0) & (p <= 1.0))


def test_incident_probability_rises_monotonically_with_occupancy_and_saturates_at_the_hazard():
    hazard = np.full(6, 0.8)
    exposure = np.array([0.0, 0.25, 1.0, 4.0, 20.0, 1e6])
    p = incident_probability(hazard, exposure)
    assert p[0] == 0.0
    assert np.all(np.diff(p) > 0.0)  # strictly increasing in occupancy
    assert p[-1] == pytest.approx(0.8)  # saturates at the hazard probability, never past it


def test_partial_occupancy_scales_incident_risk_between_the_two_extremes():
    """Negative control: intermediate occupancy must land strictly between 'nobody there' and
    'certainly occupied' -- the fix must not collapse the model into a hard 0/hazard switch."""
    hazard = np.full(3, 0.6)
    p = incident_probability(hazard, np.array([0.0, 1.0, 1e6]))
    assert p[0] == 0.0
    assert 0.0 < p[1] < p[2]
    assert p[2] == pytest.approx(0.6)
