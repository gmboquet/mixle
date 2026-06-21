"""Earthquake forecasting: Gutenberg-Richter magnitudes + ETAS self-exciting point process."""

import unittest
import warnings

import numpy as np

from pysp.stats.seismicity import ETAS, GutenbergRichter


class GutenbergRichterTest(unittest.TestCase):
    def test_fit_recovers_b_value(self):
        mags = GutenbergRichter(b=1.0, m0=2.0).sampler(seed=0).sample(5000)
        self.assertAlmostEqual(GutenbergRichter.fit(mags, m0=2.0).b, 1.0, delta=0.05)

    def test_log_density_below_threshold_is_minus_inf(self):
        gr = GutenbergRichter(b=1.0, m0=3.0)
        self.assertEqual(gr.log_density(2.0), -np.inf)
        self.assertTrue(np.isfinite(gr.log_density(4.0)))


class ETASTest(unittest.TestCase):
    def setUp(self):
        self.et = ETAS(mu=0.2, A=4.0, alpha=1.2, c=0.02, p=1.3, m0=2.0)

    def test_intensity_spikes_after_a_mainshock_then_decays(self):
        times, mags = np.array([10.0]), np.array([6.0])  # an M6 at t=10
        spike = self.et.intensity(10.01, times, mags)
        later = self.et.intensity(15.0, times, mags)
        self.assertGreater(spike, 50 * self.et.mu)  # Omori burst
        self.assertGreater(spike, later)  # decays
        self.assertGreater(later, self.et.mu - 1e-9)  # toward background

    def test_branching_ratio_subcritical(self):
        n = self.et.branching_ratio(mean_magnitude=2.0 + 1.0 / np.log(10))
        self.assertTrue(0 < n < 1)

    def test_simulation_is_clustered(self):
        ts, _ = self.et.simulate(2000.0, b=1.0, seed=1)
        counts = np.histogram(ts, bins=200)[0]
        self.assertGreater(counts.var() / counts.mean(), 1.5)  # overdispersed vs Poisson (aftershocks)

    def test_fit_recovers_parameters(self):
        ts, ms = self.et.simulate(2000.0, b=1.0, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ETAS.fit(ts, ms, 2000.0, m0=2.0)
        self.assertAlmostEqual(fit.mu, 0.2, delta=0.1)
        self.assertAlmostEqual(fit.p, 1.3, delta=0.15)
        self.assertAlmostEqual(fit.branching_ratio(ms.mean()), self.et.branching_ratio(ms.mean()), delta=0.15)

    def test_forecast_is_elevated_after_a_large_event(self):
        hist_t, hist_m = np.array([100.0]), np.array([6.5])
        busy = self.et.expected_count(100.0, 101.0, hist_t, hist_m)  # one day after an M6.5
        quiet = self.et.expected_count(500.0, 501.0, hist_t, hist_m)  # long after
        self.assertGreater(busy, 10 * quiet)
        self.assertAlmostEqual(quiet, self.et.mu, delta=0.05)  # relaxes to background


if __name__ == "__main__":
    unittest.main()
