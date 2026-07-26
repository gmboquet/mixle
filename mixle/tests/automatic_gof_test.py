"""Tests for the PIT goodness-of-fit / abstain gate on numeric profiles (WS-F)."""

import unittest

import numpy as np

from mixle.utils.automatic import analyze_structure


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

    def test_misfit_gapped_data_is_flagged_on_held_out_rows(self):
        # Two separated uniform slabs have hard edges and a broad zero-density
        # gap; even the selected two-Gaussian approximation is visibly wrong.
        rng = np.random.RandomState(2)
        data = list(np.concatenate([rng.uniform(0.0, 1.0, size=400), rng.uniform(9.0, 10.0, size=400)]))
        field = analyze_structure(data, pairwise=False).fields[0]
        self.assertIsNotNone(
            field.recommendation
        )  # a least-bad unimodal family is chosen (candidate set is now richer)
        self.assertIsNotNone(field.gof_pvalue)
        self.assertLess(field.gof_pvalue, 0.01)
        self.assertTrue(any("poor calibration" in n for n in field.notes))


if __name__ == "__main__":
    unittest.main()
