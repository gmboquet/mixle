"""Tests for the Student-t (heavy-tailed) candidate in the auto-suggester (WS-F)."""

import unittest

import numpy as np

from pysp.stats import GaussianEstimator, StudentTEstimator
from pysp.utils.automatic import analyze_structure, get_estimator


class AutomaticStudentTTest(unittest.TestCase):
    def test_heavy_tailed_data_recommends_and_builds_student_t(self):
        rng = np.random.RandomState(0)
        data = list(rng.standard_t(df=3, size=1200))  # heavy-tailed, signed
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "student_t")
        self.assertIn("student_t", field.model_scores_bits)
        self.assertLess(field.model_scores_bits["student_t"], field.model_scores_bits["gaussian"])
        self.assertIsInstance(get_estimator(data), StudentTEstimator)

    def test_gaussian_data_stays_gaussian(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(0.0, 1.0, size=1200))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "gaussian")
        self.assertIsInstance(get_estimator(data), GaussianEstimator)

    def test_student_t_applies_to_signed_support(self):
        # Heavy-tailed data centered at zero with negatives: log-normal/gamma never apply; the only
        # heavy-tail candidate is Student-t.
        rng = np.random.RandomState(2)
        data = list(rng.standard_t(df=4, size=1000) * 3.0 - 1.0)
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertIn("student_t", field.model_scores_bits)
        self.assertNotIn("lognormal", field.model_scores_bits)


if __name__ == "__main__":
    unittest.main()
