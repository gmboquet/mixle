"""The calibration gate must actually FAIL a miscalibrated posterior -- a gate that passes everything
is worse than no gate (it launders false confidence). These tests pin down that it passes calibrated
posteriors, fails overconfident and underconfident ones, catches a broken inference via SBC, and fails
closed when handed nothing to check."""

import numpy as np
import pytest

from mixle.inference.calibration_gate import (
    CalibrationVerifier,
    posterior_predictive_calibration,
    simulation_based_calibration,
)


def _predictive_ensemble(truth_sd: float, ensemble_sd: float, *, k: int = 400, m: int = 500, seed: int = 0):
    """Held-out truths drawn N(0, truth_sd^2); a predictive ensemble centered at 0 with ensemble_sd.
    ensemble_sd == truth_sd is calibrated; smaller is overconfident; larger is underconfident."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, truth_sd, size=k)
    ensemble = rng.normal(0.0, ensemble_sd, size=(k, m))
    return ensemble, y


def test_calibrated_posterior_predictive_passes():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=1.0)  # k=400: ample power
    verdict = posterior_predictive_calibration(ensemble, y)
    assert verdict.passed, verdict.reasons
    assert not verdict.low_power
    assert verdict.pit_error <= verdict.null_threshold
    assert abs(verdict.coverage_at_reference - 0.90) < 0.1


def test_overconfident_posterior_predictive_fails_and_says_why():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3)  # far too narrow
    verdict = posterior_predictive_calibration(ensemble, y)
    assert not verdict.passed
    assert verdict.coverage_at_reference < 0.90  # the dangerous direction: reports false certainty
    assert any("overconfident" in r for r in verdict.reasons)


def test_underconfident_posterior_predictive_also_fails():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=3.0)  # far too wide
    verdict = posterior_predictive_calibration(ensemble, y)
    assert not verdict.passed
    assert verdict.coverage_at_reference > 0.90
    assert any("underconfident" in r for r in verdict.reasons)


def test_tiny_holdout_is_flagged_low_power_not_false_alarmed():
    """The honest small-sample behaviour: with only a handful of held-out points, an OVERCONFIDENT
    posterior cannot be distinguished from a calibrated one -- the gate must flag low power and not
    manufacture a confident PASS or FAIL it can't statistically support."""
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3, k=6)  # overconfident but only 6 points
    verdict = posterior_predictive_calibration(ensemble, y)
    assert verdict.low_power
    assert verdict.indeterminate
    assert not verdict.passed
    assert any("LOW POWER" in r for r in verdict.reasons)


def test_calibration_status_is_passed_for_a_well_powered_calibrated_posterior():
    """calibration_status carries the honest three-way state instead of a bool + bolted-on
    low_power flag (mirrors mixle.evolve.verify.Verdict.calibration_status)."""
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=1.0)  # k=400: ample power
    verdict = posterior_predictive_calibration(ensemble, y)
    assert verdict.calibration_status == "passed"
    assert verdict.passed
    assert not verdict.low_power


def test_calibration_status_is_failed_for_a_miscalibrated_posterior():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3)
    verdict = posterior_predictive_calibration(ensemble, y)
    assert verdict.calibration_status == "failed"
    assert not verdict.passed
    assert not verdict.low_power


def test_calibration_status_is_indeterminate_not_a_pass_for_a_tiny_holdout():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3, k=6)
    verdict = posterior_predictive_calibration(ensemble, y)
    assert verdict.calibration_status == "indeterminate"
    assert not verdict.passed
    assert verdict.indeterminate
    assert verdict.low_power


def test_a_failure_that_misses_even_a_loose_threshold_is_failed_not_low_power():
    """A posterior so badly miscalibrated it fails even a LOOSE (low-power) null threshold is real
    evidence of a problem: calibration_status must read 'failed', not 'indeterminate', even though
    the threshold itself was loose enough to admit gross miscalibration.
    """
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.02, k=6)
    verdict = posterior_predictive_calibration(ensemble, y)
    assert verdict.null_threshold >= 1.0  # confirms this really is the loose-bar (low-power) regime
    assert verdict.calibration_status == "failed"
    assert not verdict.passed
    assert not verdict.low_power


def test_null_threshold_shrinks_as_holdout_grows():
    """The threshold must be sample-size-aware -- more held-out data => a tighter bar (more power)."""
    small = posterior_predictive_calibration(*_predictive_ensemble(1.0, 1.0, k=40)).null_threshold
    large = posterior_predictive_calibration(*_predictive_ensemble(1.0, 1.0, k=2000)).null_threshold
    assert large < small


def test_calibration_score_orders_calibrated_above_miscalibrated():
    good, y_good = _predictive_ensemble(truth_sd=1.0, ensemble_sd=1.0, seed=1)
    bad, y_bad = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.25, seed=1)
    assert posterior_predictive_calibration(good, y_good).score > posterior_predictive_calibration(bad, y_bad).score


# --- simulation-based calibration: catches a broken inference with no held-out real data at all ---

_TAU = 2.0  # prior sd on theta
_SIGMA = 1.0  # obs noise sd
_NOBS = 5  # observations per simulated dataset
_POST_VAR = 1.0 / (1.0 / _TAU**2 + _NOBS / _SIGMA**2)
_POST_SD = np.sqrt(_POST_VAR)


def _prior(rng):
    return np.array([rng.normal(0.0, _TAU)])


def _simulate(theta, rng):
    return theta[0] + rng.normal(0.0, _SIGMA, size=_NOBS)


def _correct_fit(y, rng):
    post_mean = _POST_VAR * (np.sum(y) / _SIGMA**2)  # conjugate Gaussian posterior mean (prior mean 0)
    return rng.normal(post_mean, _POST_SD, size=600)


def _overconfident_fit(y, rng):
    post_mean = _POST_VAR * (np.sum(y) / _SIGMA**2)
    return rng.normal(post_mean, _POST_SD / 3.0, size=600)  # deliberately 3x too tight


def test_sbc_passes_a_correct_conjugate_inference():
    verdict = simulation_based_calibration(_prior, _simulate, _correct_fit, n_sims=300, seed=0)
    assert verdict.passed, verdict.reasons


def test_sbc_fails_a_deliberately_overconfident_inference():
    verdict = simulation_based_calibration(_prior, _simulate, _overconfident_fit, n_sims=300, seed=0)
    assert not verdict.passed
    assert "mis-dispersed" in verdict.reasons[0]


def test_sbc_randomizes_exact_ties_against_the_continuous_uniform_null():
    def point_prior(_rng):
        return np.array([0.0])

    def point_simulator(_theta, _rng):
        return np.array([0.0])

    def point_fit(_y, rng):
        del rng
        return np.zeros(40)

    verdict = simulation_based_calibration(point_prior, point_simulator, point_fit, n_sims=300, seed=9)

    assert verdict.passed, verdict.reasons
    assert verdict.calibration_status == "passed"


def test_sbc_requires_and_controls_the_fit_rng():
    with pytest.raises(ValueError, match="named 'rng'"):
        simulation_based_calibration(_prior, _simulate, lambda _y: np.zeros(10), n_sims=5)

    seen: list[int] = []

    def recording_fit(_y, rng):
        seen.append(int(rng.randint(0, 2**31 - 1)))
        return rng.normal(size=20)

    simulation_based_calibration(
        _prior,
        _simulate,
        recording_fit,
        n_sims=8,
        error_tol=100.0,
        bins=2,
        min_expected_count_per_bin=1.0,
        low_power_threshold=101.0,
        seed=17,
    )
    first = seen.copy()
    seen.clear()
    simulation_based_calibration(
        _prior,
        _simulate,
        recording_fit,
        n_sims=8,
        error_tol=100.0,
        bins=2,
        min_expected_count_per_bin=1.0,
        low_power_threshold=101.0,
        seed=17,
    )
    assert seen == first


def test_underpowered_sbc_is_indeterminate_and_cannot_promote():
    verdict = simulation_based_calibration(
        _prior,
        _simulate,
        _correct_fit,
        n_sims=6,
        bins=10,
        error_tol=100.0,
        low_power_threshold=101.0,
        seed=3,
    )

    assert verdict.calibration_status == "indeterminate"
    assert verdict.indeterminate
    assert verdict.low_power
    assert not verdict.passed


def test_unseeded_randomized_checks_are_indeterminate_not_promotable():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=1.0)
    predictive = posterior_predictive_calibration(ensemble, y, pit_seed=None)
    sbc = simulation_based_calibration(
        _prior,
        _simulate,
        _correct_fit,
        n_sims=60,
        bins=10,
        error_tol=100.0,
        low_power_threshold=101.0,
        seed=None,
    )

    assert predictive.calibration_status == "indeterminate"
    assert not predictive.randomness_controlled
    assert not predictive.passed
    assert sbc.calibration_status == "indeterminate"
    assert not sbc.randomness_controlled
    assert not sbc.passed


# --- the IC-6 verifier adapter (route_task drop-in) ---


def test_verifier_passes_a_calibrated_payload():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=1.0)
    verdict = CalibrationVerifier().verify(claim={"payload": {"ensemble": ensemble, "held_out_y": y}})
    assert verdict["passed"] is True
    assert verdict["calibration_status"] == "passed"
    assert verdict["kind"] == "calibration"


def test_verifier_fails_an_overconfident_payload():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3)
    verdict = CalibrationVerifier().verify(claim={"payload": {"ensemble": ensemble, "held_out_y": y}})
    assert verdict["passed"] is False
    assert verdict["calibration_status"] == "failed"


def test_verifier_refuses_to_promote_an_underpowered_non_rejection():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3, k=6)
    verdict = CalibrationVerifier().verify(claim={"payload": {"ensemble": ensemble, "held_out_y": y}})

    assert verdict["passed"] is False
    assert verdict["calibration_status"] == "indeterminate"
    assert verdict["indeterminate"] is True
    assert verdict["low_power"] is True


def test_verifier_fails_closed_when_given_nothing_to_check():
    verdict = CalibrationVerifier().verify(claim={"payload": {"some_number": 42}})
    assert verdict["passed"] is False
    assert verdict["calibration_status"] == "failed"
    assert any("unchecked" in r or "no ensemble" in r for r in verdict["reasons"])


@pytest.mark.parametrize(
    ("ensemble", "held_out_y", "message"),
    [
        (np.empty((0, 4)), np.empty(0), "at least one held-out"),
        (np.empty((2, 0)), np.zeros(2), "at least one posterior"),
        (np.ones((2, 3)), np.array(1.0), "one-dimensional"),
        (np.array([[1.0, np.nan]]), np.ones(1), "finite"),
    ],
)
def test_predictive_calibration_rejects_malformed_inputs(ensemble, held_out_y, message):
    with pytest.raises(ValueError, match=message):
        posterior_predictive_calibration(ensemble, held_out_y)


def test_verifier_fails_closed_on_malformed_calibration_data():
    verdict = CalibrationVerifier().verify(claim={"payload": {"ensemble": [[1.0, float("nan")]], "held_out_y": [1.0]}})
    assert verdict["passed"] is False
    assert verdict["score"] == 0.0
    assert any("rejected" in reason and "finite" in reason for reason in verdict["reasons"])


def test_sbc_rejects_empty_simulations_and_draws():
    with pytest.raises(ValueError, match="n_sims"):
        simulation_based_calibration(_prior, _simulate, _correct_fit, n_sims=0)
    with pytest.raises(ValueError, match="posterior draws"):
        simulation_based_calibration(_prior, _simulate, lambda _y, rng: np.array([]), n_sims=1)
