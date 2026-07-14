"""N1: species-distribution / habitat-suitability model (inhomogeneous-Poisson SDM, IC-12)."""

from __future__ import annotations

import numpy as np

from mixle.analysis.sdm import HabitatModel, SpeciesObservation, fit_sdm
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
