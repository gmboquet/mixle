"""Calibration and scoring outputs require valid probabilities, intervals, and samples."""

import numpy as np
import pytest

from mixle.inference import (
    ProbabilityCalibrator,
    brier_score,
    coverage_curve,
    energy_score,
    expected_calibration_error,
    interval_coverage,
    interval_score,
    pit_calibration_error,
    pit_values,
    reliability_curve,
    skill_score,
    top_label_confidence,
)


@pytest.mark.parametrize(
    "call",
    [
        lambda: expected_calibration_error(np.array([]), np.array([])),
        lambda: reliability_curve(np.array([1.2]), np.array([1.0])),
        lambda: expected_calibration_error(np.array([0.5]), np.array([0.3])),
        lambda: pit_calibration_error(np.array([])),
        lambda: pit_values(np.array([1.0]), np.array([1.1])),
        lambda: pit_values(np.array([1.0]), np.array([np.nan])),
    ],
)
def test_calibration_diagnostics_reject_nonprobabilities_and_empty_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_multiclass_calibration_and_brier_require_simplex_rows():
    invalid = np.array([[0.8, 0.8]])
    with pytest.raises(ValueError, match="sum to 1"):
        top_label_confidence(invalid, np.array([0]))
    with pytest.raises(ValueError, match="sum to 1"):
        brier_score(invalid, np.array([0]))


def test_intervals_must_be_finite_aligned_and_ordered():
    with pytest.raises(ValueError, match="lower <= upper"):
        interval_coverage(np.array([2.0]), np.array([1.0]), np.array([1.5]))
    with pytest.raises(ValueError, match="lower <= upper"):
        interval_score(np.array([2.0]), np.array([1.0]), np.array([1.5]), 0.1)
    with pytest.raises(ValueError, match="matching"):
        interval_score(np.array([0.0]), np.array([1.0, 2.0]), np.array([0.5]), 0.1)


def test_weighted_pava_preserves_tied_score_multiplicity():
    # The low score has 100 observations at rate 0.4; the high score has one observation at rate
    # zero. Monotonicity pools them, and the correct weighted fit is 40/101, not (0.4 + 0)/2.
    scores = np.concatenate([np.zeros(100), np.ones(1)])
    outcomes = np.concatenate([np.r_[np.ones(40), np.zeros(60)], np.zeros(1)])
    calibrator = ProbabilityCalibrator("isotonic").fit(scores, outcomes)
    np.testing.assert_allclose(calibrator.predict([0.0, 1.0]), 40.0 / 101.0)


def test_ensemble_and_coverage_scores_reject_empty_or_malformed_samples():
    with pytest.raises(ValueError):
        energy_score(np.empty((0, 2)), np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        coverage_curve(np.empty((2, 0)), np.array([0.0, 1.0]))


def test_undefined_skill_score_raises_instead_of_returning_nan():
    with pytest.raises(ValueError, match="undefined"):
        skill_score(0.0, 0.0)
