"""Tests for the 2-component Gaussian mixture candidate (auto-K ladder, WS-F)."""

import unittest

import numpy as np

from mixle.stats import GaussianEstimator, MixtureEstimator
from mixle.utils.automatic import analyze_structure, get_estimator


class AutomaticMixtureTest(unittest.TestCase):
    def test_bimodal_data_recommends_and_builds_mixture(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(-8.0, 0.5, size=500)) + list(rng.normal(8.0, 0.5, size=500))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "mixture")
        self.assertIn("mixture", field.model_scores_bits)
        self.assertLess(field.model_scores_bits["mixture"], field.model_scores_bits["gaussian"])
        self.assertIsInstance(get_estimator(data), MixtureEstimator)

    def test_bimodal_no_longer_flagged_poor_calibration(self):
        # The mixture now explains data the GoF gate previously could only flag as uncalibrated.
        rng = np.random.RandomState(2)
        data = list(rng.normal(-8.0, 0.5, size=400)) + list(rng.normal(8.0, 0.5, size=400))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "mixture")

    def test_unimodal_gaussian_stays_gaussian(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(0.0, 1.0, size=1000))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "gaussian")
        self.assertIsInstance(get_estimator(data), GaussianEstimator)


if __name__ == "__main__":
    unittest.main()
