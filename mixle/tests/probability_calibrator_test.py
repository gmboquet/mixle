"""Tests for the score->probability recalibrator (mixle.inference.calibration)."""

import unittest

import numpy as np

from mixle.inference import (
    ProbabilityCalibrator,
    calibrate_probabilities,
    expected_calibration_error,
    pit_ensemble,
)


class IsotonicTest(unittest.TestCase):
    def test_ties_pool_to_the_group_mean_not_the_first_occurrence(self):
        # scores [1,1,1,5], outcomes [0,1,0,1]: running raw PAVA on the tied sequence and then
        # keeping only each tied group's FIRST occurrence's fitted value returned 0.0 (the first
        # point's own raw outcome) instead of the group's true rate 1/3 (1 success of 3).
        cal = calibrate_probabilities([1.0, 1.0, 1.0, 5.0], [0, 1, 0, 1], method="isotonic")
        self.assertAlmostEqual(float(cal.predict([1.0])[0]), 1.0 / 3.0, places=10)
        self.assertAlmostEqual(float(cal.predict([5.0])[0]), 1.0, places=10)

    def test_monotone_and_recovers_true_probability(self):
        # scores are a monotone-but-distorted transform of the true P(correct); isotonic should undo it.
        rng = np.random.RandomState(0)
        n = 4000
        latent = rng.uniform(0, 1, n)
        p_true = latent  # the real probability of a correct outcome
        y = (rng.uniform(0, 1, n) < p_true).astype(int)
        score = latent**3  # miscalibrated (overconfident-low) but monotone in p_true
        cal = calibrate_probabilities(score, y, method="isotonic")
        pred = cal.predict(score)
        # calibrated probabilities are non-decreasing in the score
        order = np.argsort(score)
        self.assertTrue(np.all(np.diff(pred[order]) >= -1e-9))
        # and calibration error drops sharply versus using the raw score as a probability
        self.assertLess(expected_calibration_error(pred, y), expected_calibration_error(score, y))
        self.assertLess(expected_calibration_error(pred, y), 0.05)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            ProbabilityCalibrator("isotonic").predict([0.5])


class PlattTest(unittest.TestCase):
    def test_platt_calibrates_logistic_scores(self):
        rng = np.random.RandomState(1)
        n = 3000
        s = rng.normal(0, 2, n)
        p = 1.0 / (1.0 + np.exp(-(0.9 * s - 0.3)))
        y = (rng.uniform(0, 1, n) < p).astype(int)
        cal = calibrate_probabilities(s, y, method="platt")
        pred = cal.predict(s)
        self.assertLess(expected_calibration_error(pred, y), 0.04)
        # monotone increasing in the raw score
        order = np.argsort(s)
        self.assertTrue(np.all(np.diff(pred[order]) >= -1e-9))

    def test_platt_survives_perfectly_separable_data(self):
        # Audit CAL-5: raw 0/1 targets sent a -> inf on separable data, so the "calibrated"
        # outputs were exact 0/1 -- maximal overconfidence. Smoothed targets keep them interior.
        rng = np.random.RandomState(7)
        s = np.concatenate([rng.uniform(-3, -1, 40), rng.uniform(1, 3, 40)])
        y = np.concatenate([np.zeros(40), np.ones(40)])
        cal = calibrate_probabilities(s, y, method="platt")
        pred = cal.predict(np.array([-2.0, 2.0]))
        self.assertGreater(pred[0], 1e-6)
        self.assertLess(pred[1], 1.0 - 1e-6)
        self.assertLess(pred[0], 0.1)
        self.assertGreater(pred[1], 0.9)


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUC = P(score of a correct item > score of an incorrect item) via the Mann-Whitney statistic."""
    from scipy.stats import rankdata

    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    r = rankdata(scores)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


class UninformativeScoreTest(unittest.TestCase):
    def test_meaningless_score_has_no_predictive_value_after_calibration(self):
        # THE key case: a 'confidence' with NO relationship to correctness (like a raw LLM token
        # likelihood vs. whether the fact is true). Calibrate on one split, evaluate on a HELD-OUT
        # split: the calibrated probability neither discriminates correct from incorrect (AUC ~ 0.5)
        # nor is miscalibrated -- the honest verdict that the raw score's 'likelihood' was meaningless.
        rng = np.random.RandomState(2)
        n = 8000
        score = rng.uniform(0, 1, n)  # random confidence
        y = (rng.uniform(0, 1, n) < 0.3).astype(int)  # correctness independent of the score
        tr, te = slice(0, n // 2), slice(n // 2, n)
        cal = calibrate_probabilities(score[tr], y[tr], method="isotonic")
        pred = cal.predict(score[te])
        self.assertLess(abs(_auc(pred, y[te]) - 0.5), 0.05)  # no discrimination on held-out
        self.assertLess(expected_calibration_error(pred, y[te]), 0.05)  # still calibrated to base rate


class PitEnsembleEmptyGuardTest(unittest.TestCase):
    def test_empty_ensemble_raises_instead_of_fabricating_uniform_noise(self):
        # with randomize=True, an empty ensemble used to return `u = v` (the raw jitter, nothing
        # else) -- indistinguishable from a genuinely well-calibrated PIT.
        with self.assertRaises(ValueError):
            pit_ensemble(np.array([1.0, 2.0]), np.zeros((2, 0)), randomize=True)

    def test_empty_ensemble_raises_instead_of_silent_nan(self):
        with self.assertRaises(ValueError):
            pit_ensemble(np.array([1.0, 2.0]), np.zeros((2, 0)), randomize=False)

    def test_informative_score_keeps_its_discrimination(self):
        # a score that DOES track correctness keeps its AUC after calibration (calibration fixes the
        # probability scale, it does not invent or destroy signal).
        rng = np.random.RandomState(4)
        n = 8000
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        score = p**2  # monotone in the true probability -> informative
        tr, te = slice(0, n // 2), slice(n // 2, n)
        cal = calibrate_probabilities(score[tr], y[tr], method="isotonic")
        pred = cal.predict(score[te])
        self.assertGreater(_auc(pred, y[te]), 0.7)
        self.assertLess(expected_calibration_error(pred, y[te]), 0.05)


if __name__ == "__main__":
    unittest.main()
