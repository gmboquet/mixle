"""Tests for the PIT goodness-of-fit / abstain gate on numeric profiles (WS-F)."""

import unittest

import numpy as np

from pysp.utils.automatic import analyze_structure


class GoodnessOfFitGateTest(unittest.TestCase):
    def test_well_fit_gaussian_is_calibrated(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 2.0, size=800))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "gaussian")
        self.assertIsNotNone(field.gof_pvalue)
        self.assertGreater(field.gof_pvalue, 0.05)  # calibrated -> large p-value
        self.assertFalse(any("poor calibration" in n for n in field.notes))

    def test_well_fit_lognormal_is_calibrated(self):
        rng = np.random.RandomState(1)
        data = list(rng.lognormal(0.5, 0.7, size=800))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(field.recommendation, "lognormal")
        self.assertIsNotNone(field.gof_pvalue)
        self.assertGreater(field.gof_pvalue, 0.01)

    def test_misfit_bimodal_data_is_flagged(self):
        # Two well-separated modes: no single Gaussian/log-normal/gamma fits -> low PIT p-value.
        rng = np.random.RandomState(2)
        data = list(rng.normal(-8.0, 0.5, size=400)) + list(rng.normal(8.0, 0.5, size=400))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertIsNotNone(field.gof_pvalue)
        self.assertLess(field.gof_pvalue, 0.01)
        self.assertTrue(any("poor calibration" in n for n in field.notes))


if __name__ == "__main__":
    unittest.main()
