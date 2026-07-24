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
