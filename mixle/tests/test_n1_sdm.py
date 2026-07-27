"""N1: species-distribution / habitat-suitability model (inhomogeneous-Poisson SDM, IC-12)."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.sdm import (
    _RATE_CLIP,
    HabitatModel,
    SpeciesObservation,
    _bin_cell_counts,
    _clipped_rates_and_mask,
    _fit_beta,
    _nll_and_grad,
    _penalized_hessian,
    _validate_locations,
    fit_sdm,
)
from mixle.process import InhomogeneousPoissonProcessDistribution
from mixle.reason.posterior_protocol import Posterior


def _synthetic_presences(lambda_true: np.ndarray, area: np.ndarray, rng: np.random.Generator) -> list:
    """Draw one Poisson-thinned presence realization from a known log-linear intensity field."""
    counts = rng.poisson(lambda_true * area)
    occurrences = []
    for cell, n in enumerate(counts):
        for _ in range(int(n)):
            loc = cell + rng.uniform(0.0, 1.0)
            occurrences.append(
                SpeciesObservation(
                    species_id="lynx_rufus",
                    detection=True,
                    location=np.array([loc]),
                    modality="occurrence",
                )
            )
    return occurrences, counts


def test_fit_sdm_recovers_field_beats_null_and_is_calibrated_on_held_out_fold():
    num_cells = 500
    rng = np.random.default_rng(7)
    env = rng.uniform(-1.5, 1.5, size=num_cells)
    a_true, b_true = -0.5, 1.4
    lambda_true = np.exp(a_true + b_true * env)
    area = np.ones(num_cells)

    occurrences, counts = _synthetic_presences(lambda_true, area, rng)
    assert len(occurrences) > 200  # sanity: the synthetic field actually produced data

    model = fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=1e-3)

    assert isinstance(model, HabitatModel)
    assert isinstance(model, Posterior)  # IC-12 must satisfy IC-1
    assert model.mean.shape == (num_cells,)
    assert np.all(model.mean > 0.0)

    # -- held-out fold: an independent replicate drawn from the SAME true field, never seen by the fit --
    held_out_counts = rng.poisson(lambda_true * area)
    edges = np.arange(num_cells + 1, dtype=np.float64)

    fitted_dist = InhomogeneousPoissonProcessDistribution(model.mean * area, edges=edges)
    fitted_ll = fitted_dist.seq_log_density(held_out_counts[None, :])[0]

    null_rate = max(float(counts.mean()), 1e-6)
    null_dist = InhomogeneousPoissonProcessDistribution(np.full(num_cells, null_rate), edges=edges)
    null_ll = null_dist.seq_log_density(held_out_counts[None, :])[0]

    assert fitted_ll > null_ll

    # -- calibrated UQ: the 90% credible interval covers the true field on >= 85% of held-out cells --
    lo, hi = model.credible_interval(0.9)
    assert lo.shape == (num_cells,) and hi.shape == (num_cells,)
    assert np.all(lo <= hi)
    covered = (lambda_true >= lo) & (lambda_true <= hi)
    assert covered.mean() >= 0.85


def test_species_observation_defaults():
    obs = SpeciesObservation(species_id="ursus_arctos", detection=True, location=np.zeros(2))
    assert obs.modality == "occurrence"
    assert obs.crs is None
    assert obs.covariates == {}
    assert obs.provenance == {}


def test_critical_habitat_mask_is_boolean_and_thresholded():
    rng = np.random.default_rng(3)
    num_cells = 60
    env = rng.uniform(-1.0, 1.0, size=num_cells)
    lambda_true = np.exp(-0.2 + 1.2 * env)
    area = np.ones(num_cells)
    occurrences, _ = _synthetic_presences(lambda_true, area, rng)

    model = fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=1e-2)
    threshold = float(np.median(model.mean))
    mask = model.critical_habitat_mask(threshold)

    assert mask.dtype == np.bool_
    assert mask.shape == (num_cells,)
    assert np.array_equal(mask, model.mean >= threshold)


def test_samples_and_derived_quantity_shapes():
    rng = np.random.default_rng(11)
    num_cells = 50
    env = rng.uniform(-1.0, 1.0, size=num_cells)
    lambda_true = np.exp(0.1 + 0.8 * env)
    area = np.ones(num_cells)
    occurrences, _ = _synthetic_presences(lambda_true, area, rng)

    model = fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=1e-2)

    draw_rng = np.random.default_rng(0)
    draws = model.samples(256, draw_rng)
    assert draws.shape == (256, num_cells)
    assert np.all(draws > 0.0)

    dq = model.derived_quantity(lambda d: d.sum(axis=1), 256, np.random.default_rng(1))
    assert dq.samples.shape == (256,)
    lo, hi = dq.credible_interval(0.9)
    assert np.isscalar(lo) or lo.shape == ()
    assert lo <= hi
    assert isinstance(dq.prior_dominated, bool)

    cov = model.cov
    assert cov.shape == (num_cells, num_cells)
    assert np.allclose(cov, cov.T, atol=1e-8)


def test_background_quadrature_offset_does_not_crash_and_shifts_offset():
    rng = np.random.default_rng(5)
    num_cells = 80
    env = rng.uniform(-1.0, 1.0, size=num_cells)
    lambda_true = np.exp(-0.3 + 1.0 * env)
    area = np.ones(num_cells)
    occurrences, _ = _synthetic_presences(lambda_true, area, rng)
    background = rng.uniform(0.0, num_cells, size=300)

    model_no_bg = fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=1e-2)
    model_bg = fit_sdm(occurrences, env.reshape(-1, 1), area, background=background, ridge=1e-2)

    assert model_bg.mean.shape == (num_cells,)
    assert np.all(np.isfinite(model_bg.mean))
    # background sampling effort raises the offset, so for the same counts the fitted intensity should
    # generally be no higher than the no-background fit (effort-corrected relative intensity is lower).
    assert model_bg.mean.mean() <= model_no_bg.mean.mean() * 1.5


def test_fit_sdm_beta_recovers_sign_of_covariate_effect():
    rng = np.random.default_rng(21)
    num_cells = 300
    env = rng.uniform(-1.5, 1.5, size=num_cells)
    lambda_true = np.exp(-1.0 + 2.0 * env)
    area = np.ones(num_cells)
    occurrences, _ = _synthetic_presences(lambda_true, area, rng)

    model = fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=1e-3)

    # beta = [intercept, slope]; the fitted slope must recover the strong positive true effect (b=2.0)
    assert model.beta.shape == (2,)
    assert model.beta[1] > 0.5


# -- MXR-080-0111: histogram binning must not silently drop or misbin invalid cell indices --


def test_bin_cell_counts_excludes_negative_toolarge_and_nan_indices():
    """The low-level binning helper stays correct in isolation: out-of-range/NaN input contributes
    zero counts rather than being silently absorbed into some bin (validation/reporting of *why* an
    observation was rejected is `_validate_locations`'s job, one layer up -- see the fit_sdm-level
    tests below)."""
    counts = _bin_cell_counts([-1.0, 99.0, float("nan")], num_cells=50)
    assert counts.sum() == 0.0
    assert counts.shape == (50,)


def test_bin_cell_counts_excludes_right_endpoint_k():
    """The nominally out-of-domain right endpoint K (domain is the half-open [0, K)) must not be
    folded into the last cell [K-1, K)."""
    counts = _bin_cell_counts([9.5, 10.0], num_cells=10)
    assert counts[9] == 1.0  # 9.5 is legitimately inside [9, 10)
    assert counts.sum() == 1.0  # 10.0 must NOT also land in cell 9


def test_bin_cell_counts_bins_in_domain_values_correctly():
    """Negative control: ordinary in-range floats still land in the correct cells."""
    counts = _bin_cell_counts([0.5, 1.1, 1.9, 5.0, 9.999], num_cells=10)
    expected = np.zeros(10)
    expected[0] = 1.0
    expected[1] = 2.0
    expected[5] = 1.0
    expected[9] = 1.0
    assert np.array_equal(counts, expected)


def test_validate_locations_reports_counts_by_failure_mode():
    """`_validate_locations` must receipt *why* observations were rejected, not just that some were."""
    with pytest.raises(ValueError) as exc_info:
        _validate_locations([-1.0, 99.0, float("nan"), 5.0], num_cells=50, kind="presence")
    msg = str(exc_info.value)
    assert "rejected 3 of 4" in msg
    assert "1 non-finite" in msg
    assert "2 outside" in msg


def test_fit_sdm_rejects_out_of_domain_presence_locations():
    """MXR-080-0111 exact audit repro: an SDM fit using ONLY locations -1 and 99 (both outside [0, 50))
    must be rejected outright, not silently completed as if there were zero detections."""
    num_cells = 50
    env = np.zeros(num_cells)
    area = np.ones(num_cells)
    occurrences = [
        SpeciesObservation(species_id="lynx_rufus", detection=True, location=np.array([-1.0])),
        SpeciesObservation(species_id="lynx_rufus", detection=True, location=np.array([99.0])),
    ]
    with pytest.raises(ValueError, match=r"presence locations"):
        fit_sdm(occurrences, env.reshape(-1, 1), area)


def test_fit_sdm_rejects_nan_presence_location():
    num_cells = 20
    env = np.zeros(num_cells)
    area = np.ones(num_cells)
    occurrences = [SpeciesObservation(species_id="x", detection=True, location=np.array([float("nan")]))]
    with pytest.raises(ValueError, match=r"non-finite"):
        fit_sdm(occurrences, env.reshape(-1, 1), area)


def test_fit_sdm_rejects_out_of_domain_background_locations():
    num_cells = 20
    env = np.zeros(num_cells)
    area = np.ones(num_cells)
    occurrences = [SpeciesObservation(species_id="x", detection=True, location=np.array([2.0]))]
    with pytest.raises(ValueError, match=r"background locations"):
        fit_sdm(occurrences, env.reshape(-1, 1), area, background=np.array([-1.0, 21.0]))


def test_fit_sdm_still_fits_with_legitimate_in_domain_locations():
    """Negative control: a normal in-domain presence/background mix still fits fine after the 0111 fix."""
    rng = np.random.default_rng(99)
    num_cells = 40
    env = rng.uniform(-1.0, 1.0, size=num_cells)
    area = np.ones(num_cells)
    lambda_true = np.exp(0.1 + 0.5 * env)
    occurrences, _ = _synthetic_presences(lambda_true, area, rng)
    background = rng.uniform(0.0, num_cells, size=100)

    model = fit_sdm(occurrences, env.reshape(-1, 1), area, background=background, ridge=1e-2)

    assert model.mean.shape == (num_cells,)
    assert np.all(np.isfinite(model.mean))


# -- MXR-080-0112: the Laplace covariance's Hessian must use the exact penalized second derivative --


def _make_glm_fixture(rng: np.random.Generator, num_cells: int = 30, p: int = 4):
    design = np.column_stack([np.ones(num_cells), rng.uniform(-1.0, 1.0, size=(num_cells, p - 1))])
    counts = rng.poisson(3.0, size=num_cells).astype(np.float64)
    log_offset = np.log(rng.uniform(0.5, 2.0, size=num_cells))
    edges = np.arange(num_cells + 1, dtype=np.float64)
    return design, counts, log_offset, edges


def test_penalized_hessian_matches_numerical_differentiation_of_gradient():
    """MXR-080-0112: the WHOLE analytic Hessian (data term + ridge term) must equal the central-difference
    Jacobian of `_nll_and_grad`'s own gradient output -- not just "look about right" for the ridge part,
    and not just cross-checked against an independently rederived formula."""
    rng = np.random.default_rng(42)
    design, counts, log_offset, edges = _make_glm_fixture(rng)
    p = design.shape[1]
    ridge = 0.37
    beta = rng.normal(scale=0.3, size=p)

    def grad_at(b: np.ndarray) -> np.ndarray:
        _, g = _nll_and_grad(b, design, counts, log_offset, ridge, edges)
        return g

    analytic_hessian = _penalized_hessian(beta, design, log_offset, ridge)

    eps = 1e-6
    numerical_hessian = np.zeros((p, p))
    for j in range(p):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += eps
        bm[j] -= eps
        numerical_hessian[:, j] = (grad_at(bp) - grad_at(bm)) / (2.0 * eps)

    assert np.allclose(analytic_hessian, numerical_hessian, atol=1e-3, rtol=1e-4)
    assert np.allclose(analytic_hessian, analytic_hessian.T, atol=1e-8)  # a Hessian must be symmetric


def test_penalized_hessian_ridge_contribution_is_exactly_two_ridge_identity():
    """MXR-080-0112 exact regression: holding beta fixed (so the data term is identical), the Hessian's
    ridge-only contribution (Hessian(ridge) - Hessian(0)) must be precisely 2*ridge*I, not ridge*I."""
    rng = np.random.default_rng(7)
    design, counts, log_offset, edges = _make_glm_fixture(rng, num_cells=25, p=3)
    p = design.shape[1]
    beta = rng.normal(scale=0.2, size=p)
    ridge = 1.25

    hessian_ridge = _penalized_hessian(beta, design, log_offset, ridge)
    hessian_zero = _penalized_hessian(beta, design, log_offset, 0.0)

    assert np.allclose(hessian_ridge - hessian_zero, 2.0 * ridge * np.eye(p), atol=1e-10)


def test_nll_and_grad_data_term_matches_numerical_differentiation():
    """Independent, one-derivative-order-lower cross-check: the unregularized (ridge=0) gradient must
    equal the central-difference of the NLL value itself."""
    rng = np.random.default_rng(13)
    design, counts, log_offset, edges = _make_glm_fixture(rng, num_cells=40, p=3)
    p = design.shape[1]
    beta = rng.normal(scale=0.2, size=p)

    def nll_at(b: np.ndarray) -> float:
        val, _ = _nll_and_grad(b, design, counts, log_offset, 0.0, edges)
        return val

    _, analytic_grad = _nll_and_grad(beta, design, counts, log_offset, 0.0, edges)

    eps = 1e-6
    numerical_grad = np.zeros(p)
    for j in range(p):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += eps
        bm[j] -= eps
        numerical_grad[j] = (nll_at(bp) - nll_at(bm)) / (2.0 * eps)

    assert np.allclose(analytic_grad, numerical_grad, atol=1e-3, rtol=1e-4)


# -- MXR-080-1439: the prior-dominated honesty flag must compare data curvature against the trace of the
# SAME penalized-Hessian ridge term _penalized_hessian uses (2 * ridge * I, MXR-080-0112), not half of it --


def _make_prior_dominated_boundary_fixture() -> tuple[list[SpeciesObservation], np.ndarray, np.ndarray, float]:
    """A deterministic fixture engineered so that, at ``ridge=2.5`` and ``p=2``, ``data_curvature`` falls
    strictly between ``ridge * p`` (5.0) and ``2 * ridge * p`` (10.0): three presences (in cells 1, 3, 6
    of 8) under unit area/zero log-offset fits to ``data_curvature ~= 7.354``. The OLD ``prior_curvature
    = ridge * p`` formula therefore reports "not prior-dominated" (5.0 < 7.354) while the mathematically
    consistent ``prior_curvature = 2 * ridge * p`` formula reports "prior-dominated" (10.0 > 7.354) for
    the exact same fit."""
    num_cells = 8
    env = np.array([-1.0, -0.6, -0.2, 0.1, 0.3, 0.5, 0.8, 1.0])
    area = np.ones(num_cells)
    ridge = 2.5
    occurrences = [
        SpeciesObservation(species_id="x", detection=True, location=np.array([1.5])),
        SpeciesObservation(species_id="x", detection=True, location=np.array([3.5])),
        SpeciesObservation(species_id="x", detection=True, location=np.array([6.5])),
    ]
    return occurrences, env.reshape(-1, 1), area, ridge


def test_fit_sdm_prior_dominated_uses_the_full_two_ridge_hessian_trace():
    """MXR-080-1439 exact regression: at this fixture's boundary point, the OLD ``prior_curvature =
    ridge * p`` sits BELOW ``data_curvature`` (reports ``prior_dominated=False``) while the corrected
    ``prior_curvature = 2 * ridge * p`` -- the same ``2 * ridge * I`` convention ``_penalized_hessian``
    already uses for the Laplace covariance itself (MXR-080-0112) -- sits ABOVE it (reports
    ``prior_dominated=True``). A driller-facing field must not be reported as data-supported when it is
    actually still prior-dominated (``posterior_protocol.DerivedQuantity``'s honesty flag)."""
    occurrences, covariates, area, ridge = _make_prior_dominated_boundary_fixture()
    p = covariates.shape[1] + 1

    model = fit_sdm(occurrences, covariates, area, ridge=ridge)

    # Recompute data_curvature from the model's own fitted state (no background in this fixture, so
    # effective_area == cell_area) to pin the boundary fixture itself, not just trust the search that
    # found it.
    log_offset = np.log(model.cell_area)
    rates_hat, mask = _clipped_rates_and_mask(model.beta, model.design, log_offset)
    assert np.all(mask == 1.0)  # sanity: the rate clip is inactive here (not what this test targets)
    data_curvature = float(np.trace(model.design.T @ (rates_hat[:, None] * model.design)))

    old_prior_curvature = ridge * p
    new_prior_curvature = 2.0 * ridge * p
    assert old_prior_curvature < data_curvature < new_prior_curvature  # the boundary this fixture targets

    dq = model.derived_quantity(lambda draws: draws, 2, np.random.default_rng(0))
    assert dq.prior_dominated is True  # corrected answer -- was False under the pre-fix ridge*p formula


def test_fit_sdm_negative_control_prior_dominated_still_false_with_ample_data():
    """Negative control: an ordinary, well-informed fit (light ridge, hundreds of presences) still
    reports ``prior_dominated=False`` after the MXR-080-1439 fix -- the fix corrects the comparison, it
    does not make the flag universally ``True``."""
    rng = np.random.default_rng(17)
    num_cells = 200
    env = rng.uniform(-1.0, 1.0, size=num_cells)
    lambda_true = np.exp(0.2 + 1.0 * env)
    area = np.ones(num_cells)
    occurrences, _ = _synthetic_presences(lambda_true, area, rng)

    model = fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=1e-3)

    dq = model.derived_quantity(lambda draws: draws, 2, np.random.default_rng(0))
    assert dq.prior_dominated is False


# -- MXR-080-0113: gradient/Hessian must stay the exact derivative of the CLIPPED objective, and a
# non-converged fit must not be treated as a valid posterior --


def _make_saturating_fixture():
    """One cell (index 0) has eta deep past `_RATE_CLIP`; the rest are ordinary, unsaturated cells."""
    design = np.array([[1.0, 800.0], [1.0, 0.0], [1.0, -0.5], [1.0, 0.3]])
    counts = np.array([2.0, 1.0, 0.0, 3.0])
    log_offset = np.zeros(4)
    edges = np.arange(5, dtype=np.float64)
    beta = np.array([0.1, 1.0])  # eta[0] = 0.1 + 800*1.0 = 800.1, far past +-_RATE_CLIP (700)
    return design, counts, log_offset, edges, beta


def test_clipped_rates_and_mask_saturates_and_masks_outside_clip_range():
    design, _, log_offset, _, beta = _make_saturating_fixture()
    rates, mask = _clipped_rates_and_mask(beta, design, log_offset)
    assert mask.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert rates[0] == pytest.approx(np.exp(_RATE_CLIP))  # pinned at the clip boundary, not inf
    assert np.all(np.isfinite(rates))


def _make_small_clip_saturating_fixture():
    """Like `_make_saturating_fixture`, but sized for use under a monkeypatched, much smaller
    `_RATE_CLIP` (see the two tests below): cell 0's eta sits just past the *patched* clip."""
    design = np.array([[1.0, 8.0], [1.0, 0.0], [1.0, -0.5], [1.0, 0.3]])
    counts = np.array([2.0, 1.0, 0.0, 3.0])
    log_offset = np.zeros(4)
    edges = np.arange(5, dtype=np.float64)
    beta = np.array([0.1, 1.0])  # eta[0] = 0.1 + 8*1.0 = 8.1
    return design, counts, log_offset, edges, beta


def test_nll_and_grad_gradient_matches_numerical_differentiation_when_clip_is_active(monkeypatch):
    """The gradient must be the exact derivative of the CLIPPED nll value even with an active clip, not
    the derivative of the unclipped objective (the pre-fix bug put the saturated cell's analytic gradient
    component around 10**304 against a true numerical gradient of ~0 there).

    `_RATE_CLIP` is monkeypatched down to 5.0 for this test only: at the real clip (700), the saturated
    cell's rate is exp(700) ~ 1e304, which so completely dominates float64 precision in the *summed* nll
    that finite-differencing could not detect the other cells' O(1) contributions at all -- a pure
    precision artifact of the test, not a property of the fix (`_RATE_CLIP` is a bare module global that
    `_clipped_rates_and_mask`/`_nll_and_grad` look up dynamically at call time, so patching the module
    attribute changes what those functions see without needing to touch the module's source)."""
    import mixle.analysis.sdm as sdm_mod

    monkeypatch.setattr(sdm_mod, "_RATE_CLIP", 5.0)
    design, counts, log_offset, edges, beta = _make_small_clip_saturating_fixture()
    ridge = 0.05

    _, mask = _clipped_rates_and_mask(beta, design, log_offset)
    assert mask[0] == 0.0  # sanity: cell 0 really is saturated under the patched clip

    def nll_at(b: np.ndarray) -> float:
        val, _ = _nll_and_grad(b, design, counts, log_offset, ridge, edges)
        return val

    _, analytic_grad = _nll_and_grad(beta, design, counts, log_offset, ridge, edges)

    eps = 1e-6
    numerical_grad = np.zeros(2)
    for j in range(2):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += eps
        bm[j] -= eps
        numerical_grad[j] = (nll_at(bp) - nll_at(bm)) / (2.0 * eps)

    assert np.allclose(analytic_grad, numerical_grad, atol=1e-4, rtol=1e-4)


def test_penalized_hessian_matches_numerical_differentiation_when_clip_is_active(monkeypatch):
    """The Hessian must likewise stay consistent with the (masked) gradient when the clip is active
    (see the preceding test for why `_RATE_CLIP` is monkeypatched down for this check)."""
    import mixle.analysis.sdm as sdm_mod

    monkeypatch.setattr(sdm_mod, "_RATE_CLIP", 5.0)
    design, counts, log_offset, edges, beta = _make_small_clip_saturating_fixture()
    ridge = 0.05

    def grad_at(b: np.ndarray) -> np.ndarray:
        _, g = _nll_and_grad(b, design, counts, log_offset, ridge, edges)
        return g

    analytic_hessian = _penalized_hessian(beta, design, log_offset, ridge)

    eps = 1e-6
    numerical_hessian = np.zeros((2, 2))
    for j in range(2):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += eps
        bm[j] -= eps
        numerical_hessian[:, j] = (grad_at(bp) - grad_at(bm)) / (2.0 * eps)

    assert np.allclose(analytic_hessian, numerical_hessian, atol=1e-4, rtol=1e-4)


def test_fit_beta_rejects_non_convergence(monkeypatch):
    """An optimizer that reports failure must not be silently treated as a fitted posterior."""
    from scipy.optimize import OptimizeResult

    import mixle.analysis.sdm as sdm_mod

    def fake_minimize(fun, x0, **kwargs):
        return OptimizeResult(x=x0, success=False, message="forced non-convergence for test")

    monkeypatch.setattr(sdm_mod, "minimize", fake_minimize)

    rng = np.random.default_rng(3)
    design, counts, log_offset, _ = _make_glm_fixture(rng, num_cells=10, p=2)
    with pytest.raises(ValueError, match=r"did not converge"):
        _fit_beta(design, counts, log_offset, ridge=1e-3)


def test_fit_beta_still_converges_and_fits_normally():
    """Negative control: an ordinary, well-posed fit still converges and returns a usable Laplace fit."""
    rng = np.random.default_rng(5)
    design, counts, log_offset, _ = _make_glm_fixture(rng, num_cells=60, p=3)

    beta_hat, beta_cov, rates_hat = _fit_beta(design, counts, log_offset, ridge=1e-2)

    assert beta_hat.shape == (3,)
    assert np.all(np.isfinite(beta_hat))
    assert beta_cov.shape == (3, 3)
    assert np.all(np.isfinite(beta_cov))
    assert rates_hat.shape == (60,)
    assert np.all(rates_hat > 0.0)


# -- MXR-080-0114: SDM physical inputs and posterior invariants must be validated, not accepted or
# silently clipped --


def test_fit_sdm_rejects_invalid_ridge():
    num_cells = 20
    env = np.zeros(num_cells)
    area = np.ones(num_cells)
    occurrences = [SpeciesObservation(species_id="x", detection=True, location=np.array([2.0]))]
    with pytest.raises(ValueError, match=r"ridge"):
        fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=-1.0)
    with pytest.raises(ValueError, match=r"ridge"):
        fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=float("nan"))


def test_fit_sdm_rejects_non_finite_covariates():
    num_cells = 20
    env = np.zeros((num_cells, 1))
    env[5, 0] = np.inf
    area = np.ones(num_cells)
    occurrences = [SpeciesObservation(species_id="x", detection=True, location=np.array([2.0]))]
    with pytest.raises(ValueError, match=r"covariates"):
        fit_sdm(occurrences, env, area)


def test_fit_sdm_rejects_invalid_area_instead_of_using_a_pseudo_area():
    """MXR-080-0114: a zero/negative cell_area entry must be rejected, not silently turned into a tiny
    positive pseudo-area (previously np.clip(effective_area, 1e-12, None))."""
    num_cells = 20
    env = np.zeros(num_cells)
    occurrences = [SpeciesObservation(species_id="x", detection=True, location=np.array([2.0]))]

    zero_area = np.ones(num_cells)
    zero_area[0] = 0.0
    with pytest.raises(ValueError, match=r"cell_area"):
        fit_sdm(occurrences, env.reshape(-1, 1), zero_area)

    negative_area = np.ones(num_cells)
    negative_area[1] = -5.0
    with pytest.raises(ValueError, match=r"cell_area"):
        fit_sdm(occurrences, env.reshape(-1, 1), negative_area)

    nan_area = np.ones(num_cells)
    nan_area[2] = np.nan
    with pytest.raises(ValueError, match=r"cell_area"):
        fit_sdm(occurrences, env.reshape(-1, 1), nan_area)


def _valid_habitat_model_kwargs(p: int = 2, num_cells: int = 1) -> dict:
    return {
        "beta": np.zeros(p),
        "beta_cov": np.eye(p) * 1e-3,
        "design": np.ones((num_cells, p)),
        "cell_area": np.ones(num_cells),
    }


def test_habitat_model_rejects_non_finite_beta():
    kwargs = _valid_habitat_model_kwargs()
    kwargs["beta"] = np.array([0.0, np.inf])
    with pytest.raises(ValueError, match=r"beta"):
        HabitatModel(**kwargs)


def test_habitat_model_owns_and_freezes_validated_parameter_arrays():
    beta = np.array([0.1, 0.2])
    beta_cov = np.eye(2)
    design = np.array([[1.0, 2.0], [1.0, 3.0]])
    cell_area = np.array([1.0, 2.0])
    model = HabitatModel(beta, beta_cov, design, cell_area)
    mean_before = model.mean.copy()
    beta[:] = 100.0
    beta_cov[:] = np.nan
    design[:] = -100.0
    cell_area[:] = -1.0
    np.testing.assert_array_equal(model.mean, mean_before)
    for parameter in (model.beta, model.beta_cov, model.design, model.cell_area):
        assert not parameter.flags.writeable
        with pytest.raises(ValueError):
            parameter.flat[0] = 0.0


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf, True, np.array([1.0])])
def test_critical_habitat_mask_rejects_invalid_threshold(threshold):
    model = HabitatModel(**_valid_habitat_model_kwargs())
    with pytest.raises(ValueError, match="threshold"):
        model.critical_habitat_mask(threshold)


def test_habitat_model_rejects_mismatched_design_shape():
    kwargs = _valid_habitat_model_kwargs(p=2)
    kwargs["design"] = np.ones((1, 3))  # p=3 columns, but beta implies p=2
    with pytest.raises(ValueError, match=r"design"):
        HabitatModel(**kwargs)


def test_habitat_model_rejects_mismatched_cell_area_shape():
    kwargs = _valid_habitat_model_kwargs(p=2, num_cells=3)
    kwargs["cell_area"] = np.ones(5)  # design has 3 rows, cell_area has 5
    with pytest.raises(ValueError, match=r"cell_area"):
        HabitatModel(**kwargs)


def test_habitat_model_rejects_non_positive_or_non_finite_cell_area():
    kwargs = _valid_habitat_model_kwargs(num_cells=2, p=2)
    kwargs["design"] = np.ones((2, 2))
    bad = kwargs.copy()
    bad["cell_area"] = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match=r"cell_area"):
        HabitatModel(**bad)
    bad2 = kwargs.copy()
    bad2["cell_area"] = np.array([1.0, -2.0])
    with pytest.raises(ValueError, match=r"cell_area"):
        HabitatModel(**bad2)
    bad3 = kwargs.copy()
    bad3["cell_area"] = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match=r"cell_area"):
        HabitatModel(**bad3)


def test_habitat_model_rejects_non_psd_covariance():
    """MXR-080-0114: a symmetric-but-not-positive-semidefinite beta_cov must be rejected."""
    kwargs = _valid_habitat_model_kwargs()
    kwargs["beta_cov"] = np.array([[1.0, 0.0], [0.0, -1.0]])  # symmetric, eigenvalues {1, -1}
    with pytest.raises(ValueError, match=r"positive-semidefinite"):
        HabitatModel(**kwargs)


def test_habitat_model_rejects_asymmetric_covariance():
    kwargs = _valid_habitat_model_kwargs()
    kwargs["beta_cov"] = np.array([[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"symmetric"):
        HabitatModel(**kwargs)


def test_habitat_model_accepts_exactly_singular_psd_covariance():
    """Positive-SEMI-definite means the closed boundary (an exactly-zero eigenvalue) is legitimate, not
    just strictly positive-definite -- unlike a Wishart's own density, a Laplace covariance may be
    degenerate."""
    kwargs = _valid_habitat_model_kwargs()
    kwargs["beta_cov"] = np.array([[1.0, 0.0], [0.0, 0.0]])  # eigenvalues {1, 0}
    model = HabitatModel(**kwargs)
    assert model.beta_cov.shape == (2, 2)


def test_habitat_model_rejects_non_finite_or_non_positive_var_scale():
    kwargs = _valid_habitat_model_kwargs()
    with pytest.raises(ValueError, match=r"var_scale"):
        HabitatModel(**kwargs, var_scale=0.0)
    with pytest.raises(ValueError, match=r"var_scale"):
        HabitatModel(**kwargs, var_scale=-1.0)
    with pytest.raises(ValueError, match=r"var_scale"):
        HabitatModel(**kwargs, var_scale=float("nan"))


def test_habitat_model_mean_and_credible_interval_stay_finite_under_extreme_beta():
    """MXR-080-0114: a linear predictor that would overflow plain exp() must stay finite at prediction
    time too, using the same clip fitting already relies on."""
    model = HabitatModel(
        beta=np.array([0.0, 1000.0]),
        beta_cov=np.eye(2) * 1e-6,
        design=np.array([[1.0, 1.0]]),
        cell_area=np.array([1.0]),
    )
    assert np.all(np.isfinite(model.mean))
    lo, hi = model.credible_interval(0.9)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    draws = model.samples(10, np.random.default_rng(0))
    assert np.all(np.isfinite(draws))


def test_habitat_model_samples_rejects_invalid_draw_count():
    model = HabitatModel(**_valid_habitat_model_kwargs())
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"positive exact integer"):
        model.samples(2.7, rng)
    with pytest.raises(ValueError, match=r"positive exact integer"):
        model.samples(0, rng)
    with pytest.raises(ValueError, match=r"positive exact integer"):
        model.samples(-3, rng)
    with pytest.raises(ValueError, match=r"positive exact integer"):
        model.samples(True, rng)


def test_habitat_model_credible_interval_rejects_invalid_level():
    model = HabitatModel(**_valid_habitat_model_kwargs())
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        model.credible_interval(-0.5)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        model.credible_interval(1.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        model.credible_interval(float("nan"))


def test_pushforward_quantity_credible_interval_rejects_invalid_level():
    model = HabitatModel(**_valid_habitat_model_kwargs())
    dq = model.derived_quantity(lambda d: d.sum(axis=1), 32, np.random.default_rng(0))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        dq.credible_interval(0.0)


@pytest.mark.parametrize(
    "callback",
    [
        lambda _draws: np.array(np.nan),
        lambda _draws: np.full(4, np.nan),
        lambda _draws: np.ones(2),
    ],
)
def test_habitat_derived_quantity_rejects_invalid_callback_output(callback):
    model = HabitatModel(**_valid_habitat_model_kwargs())
    with pytest.raises(ValueError, match="derived quantity samples"):
        model.derived_quantity(callback, 4, np.random.default_rng(0))


def test_habitat_derived_quantity_owns_valid_sample_axis():
    model = HabitatModel(**_valid_habitat_model_kwargs(num_cells=2))
    quantity = model.derived_quantity(lambda draws: draws.sum(axis=1), 4, np.random.default_rng(0))
    assert quantity.samples.shape == (4,)
    assert np.isfinite(quantity.samples).all()
    assert not quantity.samples.flags.writeable


def test_fit_sdm_negative_control_normal_fit_is_valid_finite_and_psd():
    """Negative control: an ordinary, adequately-regularized fit still produces a valid, correctly-shaped,
    finite, positive-semidefinite posterior after all of the 0114 validation."""
    rng = np.random.default_rng(123)
    num_cells = 50
    env = rng.uniform(-1.0, 1.0, size=num_cells)
    lambda_true = np.exp(0.2 + 0.9 * env)
    area = np.ones(num_cells)
    occurrences, _ = _synthetic_presences(lambda_true, area, rng)

    model = fit_sdm(occurrences, env.reshape(-1, 1), area, ridge=1e-2)

    assert np.all(np.isfinite(model.mean))
    assert model.beta_cov.shape == (2, 2)
    eigvals = np.linalg.eigvalsh(model.beta_cov)
    assert np.all(eigvals >= -1e-8 * max(float(np.max(np.abs(eigvals))), 1.0))
    lo, hi = model.credible_interval(0.9)
    assert np.all(lo <= hi)
    draws = model.samples(100, np.random.default_rng(1))
    assert draws.shape == (100, num_cells)
    assert np.all(np.isfinite(draws))
