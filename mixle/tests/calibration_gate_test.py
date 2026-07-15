"""The calibration gate must actually FAIL a miscalibrated posterior -- a gate that passes everything
is worse than no gate (it launders false confidence). These tests pin down that it passes calibrated
posteriors, fails overconfident and underconfident ones, catches a broken inference via SBC, and fails
closed when handed nothing to check."""

import numpy as np

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
    assert any("LOW POWER" in r for r in verdict.reasons)


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


def _correct_fit(y):
    post_mean = _POST_VAR * (np.sum(y) / _SIGMA**2)  # conjugate Gaussian posterior mean (prior mean 0)
    rng = np.random.default_rng(int(abs(np.sum(y) * 1e6)) % (2**32))
    return rng.normal(post_mean, _POST_SD, size=600)


def _overconfident_fit(y):
    post_mean = _POST_VAR * (np.sum(y) / _SIGMA**2)
    rng = np.random.default_rng(int(abs(np.sum(y) * 1e6)) % (2**32))
    return rng.normal(post_mean, _POST_SD / 3.0, size=600)  # deliberately 3x too tight


def test_sbc_passes_a_correct_conjugate_inference():
    verdict = simulation_based_calibration(_prior, _simulate, _correct_fit, n_sims=300, seed=0)
    assert verdict.passed, verdict.reasons


def test_sbc_fails_a_deliberately_overconfident_inference():
    verdict = simulation_based_calibration(_prior, _simulate, _overconfident_fit, n_sims=300, seed=0)
    assert not verdict.passed
    assert "mis-dispersed" in verdict.reasons[0]


# --- the IC-6 verifier adapter (route_task drop-in) ---


def test_verifier_passes_a_calibrated_payload():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=1.0)
    verdict = CalibrationVerifier().verify(claim={"payload": {"ensemble": ensemble, "held_out_y": y}})
    assert verdict["passed"] is True
    assert verdict["kind"] == "calibration"


def test_verifier_fails_an_overconfident_payload():
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3)
    verdict = CalibrationVerifier().verify(claim={"payload": {"ensemble": ensemble, "held_out_y": y}})
    assert verdict["passed"] is False


def test_verifier_fails_closed_when_given_nothing_to_check():
    verdict = CalibrationVerifier().verify(claim={"payload": {"some_number": 42}})
    assert verdict["passed"] is False
    assert any("unchecked" in r or "no ensemble" in r for r in verdict["reasons"])
