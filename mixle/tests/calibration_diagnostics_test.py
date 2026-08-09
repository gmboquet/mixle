"""Calibration diagnostics (mixle.inference.calibration)."""

import unittest

import numpy as np

from mixle.inference import (
    calibration_null_expectation,
    coverage_curve,
    expected_calibration_error,
    interval_coverage,
    maximum_calibration_error,
    pit_calibration_error,
    pit_ensemble,
    pit_histogram,
    pit_values,
    reliability_curve,
    top_label_confidence,
)


class ReliabilityTest(unittest.TestCase):
    def test_calibrated_forecaster_on_diagonal(self):
        rng = np.random.RandomState(0)
        p = rng.rand(20000)
        y = (rng.rand(20000) < p).astype(float)  # outcomes drawn at the stated probability
        rc = reliability_curve(p, y, bins=10)
        # observed frequency tracks mean predicted probability
        np.testing.assert_allclose(rc["obs_freq"], rc["mean_pred"], atol=0.03)

    def test_ece_near_zero_when_calibrated(self):
        rng = np.random.RandomState(1)
        p = rng.rand(20000)
        y = (rng.rand(20000) < p).astype(float)
        self.assertLess(expected_calibration_error(p, y, bins=15), 0.02)

    def test_overconfident_forecaster_has_large_ece(self):
        rng = np.random.RandomState(2)
        # true prob 0.5 but always reports 0.95 -> badly miscalibrated
        y = (rng.rand(5000) < 0.5).astype(float)
        p = np.full(5000, 0.95)
        self.assertGreater(expected_calibration_error(p, y), 0.4)
        self.assertGreater(maximum_calibration_error(p, y), 0.4)

    def test_bootstrap_ci_brackets_point(self):
        rng = np.random.RandomState(3)
        p = rng.rand(2000)
        y = (rng.rand(2000) < p).astype(float)
        ece, lo, hi = expected_calibration_error(p, y, ci=True, n_boot=200, seed=0)
        self.assertLessEqual(lo, ece)
        self.assertLessEqual(ece, hi + 1e-12)

    def test_reliability_ci_band_present(self):
        rng = np.random.RandomState(4)
        p = rng.rand(3000)
        y = (rng.rand(3000) < p).astype(float)
        rc = reliability_curve(p, y, bins=8, ci=True, n_boot=200, seed=1)
        self.assertIn("obs_lo", rc)
        self.assertTrue(np.all(rc["obs_lo"] <= rc["obs_freq"] + 1e-9))
        self.assertTrue(np.all(rc["obs_freq"] <= rc["obs_hi"] + 1e-9))


class MulticlassTest(unittest.TestCase):
    def test_top_label_confidence(self):
        prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        labels = np.array([0, 1])
        conf, correct = top_label_confidence(prob, labels)
        np.testing.assert_allclose(conf, [0.7, 0.8])
        np.testing.assert_allclose(correct, [1.0, 0.0])


class PITTest(unittest.TestCase):
    def test_pit_uniform_when_calibrated(self):
        rng = np.random.RandomState(5)
        from scipy.stats import norm

        y = rng.normal(2.0, 3.0, size=20000)
        u = pit_values(y, norm(2.0, 3.0).cdf)
        # mean ~0.5, low calibration error, flat histogram
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.02)
        self.assertLess(pit_calibration_error(u, bins=20), 0.05)

    def test_pit_underdispersed_is_u_shaped(self):
        rng = np.random.RandomState(6)
        from scipy.stats import norm

        y = rng.normal(0.0, 3.0, size=20000)  # truth wider than the forecast sd=1
        u = pit_values(y, norm(0.0, 1.0).cdf)
        hist = pit_histogram(u, bins=10)
        # mass piles up in the extreme bins
        self.assertGreater(hist["density"][0] + hist["density"][-1], 2.0 * (hist["density"][4] + hist["density"][5]))
        self.assertGreater(pit_calibration_error(u), 0.1)

    def test_pit_ensemble_uniform(self):
        rng = np.random.RandomState(7)
        n, m = 5000, 200
        mu = rng.normal(0, 1, size=n)
        forecasts = mu[:, None] + rng.normal(0, 1, size=(n, m))
        y = mu + rng.normal(0, 1, size=n)
        u = pit_ensemble(y, forecasts, seed=0)
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.02)
        self.assertLess(pit_calibration_error(u, bins=20), 0.06)


class CoverageTest(unittest.TestCase):
    def test_interval_coverage(self):
        y = np.array([0.0, 1.0, 2.0, 5.0])
        lo = np.zeros(4)
        hi = np.full(4, 2.0)
        res = interval_coverage(lo, hi, y)
        self.assertAlmostEqual(res["coverage"], 0.75)
        self.assertAlmostEqual(res["mean_width"], 2.0)

    def test_coverage_curve_tracks_diagonal_when_calibrated(self):
        rng = np.random.RandomState(8)
        n, m = 4000, 500
        mu = rng.normal(0, 1, size=n)
        forecasts = mu[:, None] + rng.normal(0, 1, size=(n, m))
        y = mu + rng.normal(0, 1, size=n)
        cc = coverage_curve(forecasts, y, levels=np.array([0.5, 0.8, 0.9]))
        np.testing.assert_allclose(cc["empirical"], cc["nominal"], atol=0.04)

    def test_coverage_curve_underdispersed_below_diagonal(self):
        rng = np.random.RandomState(9)
        n, m = 4000, 500
        mu = rng.normal(0, 1, size=n)
        forecasts = mu[:, None] + rng.normal(0, 0.3, size=(n, m))  # too sharp
        y = mu + rng.normal(0, 1, size=n)
        cc = coverage_curve(forecasts, y, levels=np.array([0.9]))
        self.assertLess(cc["empirical"][0], 0.9)


class CalibrationNullFloorTest(unittest.TestCase):
    """Audit CAL-1/2/3/7/8: finite-sample null floors and pointwise-CI honesty."""

    def test_null_expectation_matches_a_perfectly_calibrated_forecaster(self):
        rng = np.random.RandomState(0)
        p = rng.uniform(0.05, 0.95, size=2000)
        y = (rng.uniform(size=2000) < p).astype(float)  # calibrated by construction
        null = calibration_null_expectation(p, bins=10, n_sim=400, seed=1)
        self.assertGreater(null["ece"], 0.0)  # zero is NOT the null value
        self.assertLess(expected_calibration_error(p, y, bins=10), 2.0 * null["ece_q95"])
        self.assertLess(maximum_calibration_error(p, y, bins=10), 2.0 * null["mce_q95"])

    def test_mce_null_grows_with_bins(self):
        rng = np.random.RandomState(2)
        p = rng.uniform(0.05, 0.95, size=2000)
        few = calibration_null_expectation(p, bins=10, n_sim=200, seed=3)
        many = calibration_null_expectation(p, bins=500, n_sim=200, seed=3)
        self.assertGreater(many["mce"], 3.0 * few["mce"])  # the max is a bins artifact under the null

    def test_coverage_curve_null_expectation_tracks_a_perfect_small_ensemble(self):
        rng = np.random.RandomState(4)
        n, m = 6000, 5
        forecasts = rng.standard_normal((n, m))
        y = rng.standard_normal(n)  # same law: the ensemble is perfect, m is just small
        cc = coverage_curve(forecasts, y, levels=np.array([0.95]))
        self.assertLess(cc["null_expectation"][0], 0.75)  # far below the diagonal at m=5
        self.assertLess(abs(cc["empirical"][0] - cc["null_expectation"][0]), 0.05)

    def test_reliability_ci_reports_effective_bootstrap_counts(self):
        rng = np.random.RandomState(5)
        p = rng.uniform(0.0, 1.0, size=60)
        y = (rng.uniform(size=60) < p).astype(float)
        out = reliability_curve(p, y, bins=10, ci=True, n_boot=100, seed=6)
        eff = out["obs_ci_effective_boot"]
        self.assertEqual(eff.shape, out["obs_freq"].shape)
        self.assertTrue(np.all(eff >= 1) and np.all(eff <= 100))


if __name__ == "__main__":
    unittest.main()
