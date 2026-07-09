"""Tests for the log-normal candidate in the auto-suggester (WS-F).

Strictly-positive numeric fields now consider a log-normal model alongside the Gaussian (BIC
prefilter + held-out predictive validation), and ``get_estimator`` selects it when it gives a
smaller BIC. Signed data is unaffected (log-normal is only a candidate for positive support).
"""

import unittest

import numpy as np

from mixle.stats import GammaEstimator, GaussianEstimator, LogGaussianEstimator
from mixle.utils.automatic import analyze_structure, get_estimator


class AutomaticLogNormalTest(unittest.TestCase):
    def test_lognormal_data_recommends_and_builds_lognormal(self):
        rng = np.random.RandomState(0)
        data = list(rng.lognormal(mean=0.7, sigma=0.6, size=600))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "lognormal")
        self.assertIn("lognormal", field.model_scores_bits)
        self.assertIn("gaussian", field.model_scores_bits)
        self.assertLess(field.model_scores_bits["lognormal"], field.model_scores_bits["gaussian"])
        self.assertIsInstance(get_estimator(data), LogGaussianEstimator)

    def test_lognormal_confirmed_by_validation(self):
        rng = np.random.RandomState(1)
        data = list(rng.lognormal(mean=1.0, sigma=0.8, size=800))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.validation_recommendation, "lognormal")
        self.assertIn("lognormal", field.validation_scores_bits)

    def test_signed_floats_stay_gaussian(self):
        rng = np.random.RandomState(2)
        data = list(rng.normal(0.0, 2.0, size=400))  # includes negatives -> not positive support
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "gaussian")
        self.assertNotIn("lognormal", field.model_scores_bits)
        self.assertIsInstance(get_estimator(data), GaussianEstimator)

    def test_exponential_data_recommends_and_builds_gamma(self):
        # Exponential data (monotone-decreasing density from 0) is Gamma(k~1); the log-normal has an
        # interior mode and cannot fit it, so gamma wins the BIC decisively.
        rng = np.random.RandomState(4)
        data = list(rng.exponential(2.0, size=800))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "gamma")
        self.assertIn("gamma", field.model_scores_bits)
        self.assertLess(field.model_scores_bits["gamma"], field.model_scores_bits["lognormal"])
        self.assertIsInstance(get_estimator(data), GammaEstimator)

    def test_symmetric_positive_data_prefers_gaussian(self):
        # Tight, symmetric, strictly-positive data: the Gaussian code length should win.
        rng = np.random.RandomState(3)
        data = list(rng.normal(100.0, 3.0, size=600))
        self.assertTrue(all(v > 0 for v in data))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "gaussian")
        self.assertIsInstance(get_estimator(data), GaussianEstimator)

    def test_constant_positive_data_does_not_win_spuriously_as_lognormal(self):
        # Regression: for exactly-constant data, raw-space variance is exactly 0.0 (correctly
        # excluding the Gaussian candidate), but log-space variance of a constant array is NOT
        # exactly zero -- np.log(c) computed independently per repeated element rounds slightly
        # differently, leaving a ~1e-32-scale floating-point artifact. With gaussian excluded and
        # nothing else to compare against, that spurious near-zero log-variance used to win
        # unconditionally, producing a LogGaussianDistribution that assigns catastrophically low
        # density to any value even 1% away from the training constant.
        from mixle.inference.estimation import optimize

        for value, n in ((7.0, 10), (123.456, 50), (100.0, 50)):
            data = [value] * n
            with self.subTest(value=value, n=n):
                model = optimize(data, out=None)
                # the correct fallback (a degenerate Gaussian, its min_covar floor already handling
                # constant data elsewhere in the codebase), not a spurious LogGaussianDistribution
                self.assertEqual(type(model).__name__, "GaussianDistribution")


if __name__ == "__main__":
    unittest.main()
