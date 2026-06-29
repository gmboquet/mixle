"""Power-law (Omori) marked Hawkes process: intensity, clustering, MLE recovery, forecasting."""

import unittest
import warnings

import numpy as np

from mixle.stats import ExponentialDistribution
from mixle.stats.processes.power_law_hawkes import PowerLawHawkesDistribution as PLH


class PowerLawHawkesTest(unittest.TestCase):
    def setUp(self):
        self.d = PLH(
            mu=0.2, A=4.0, c=0.02, p=1.3, window=2000.0, alpha=1.2, mark_dist=ExponentialDistribution(1.0 / np.log(10))
        )

    def test_intensity_spikes_then_power_law_decays(self):
        spike = self.d.intensity(10.01, [10.0], [5.0])
        later = self.d.intensity(15.0, [10.0], [5.0])
        self.assertGreater(spike, 50 * self.d.mu)
        self.assertGreater(spike, later)
        self.assertGreater(later, self.d.mu - 1e-9)

    def test_branching_ratio_subcritical(self):
        self.assertTrue(0 < self.d.branching_ratio(0.5) < 1)

    def test_sampler_is_clustered(self):
        ts, _ = self.d.sampler(seed=1).sample()
        counts = np.histogram(ts, bins=200)[0]
        self.assertGreater(counts.var() / counts.mean(), 1.5)  # overdispersed vs Poisson

    def test_mle_recovers_parameters(self):
        ts, ms = self.d.sampler(seed=1).sample()
        est = self.d.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update([(ts, ms)], None, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = est.estimate(None, acc.value())
        self.assertAlmostEqual(fit.mu, 0.2, delta=0.1)
        self.assertAlmostEqual(fit.alpha, 1.2, delta=0.3)
        self.assertAlmostEqual(fit.branching_ratio(ms.mean()), self.d.branching_ratio(ms.mean()), delta=0.15)

    def test_forecast_elevated_after_a_large_mark(self):
        busy = self.d.expected_count(10, 11, [10.0], [5.0])
        quiet = self.d.expected_count(500, 501, [10.0], [5.0])
        self.assertGreater(busy, 10 * quiet)
        self.assertAlmostEqual(quiet, self.d.mu, delta=0.05)

    def test_unmarked_process(self):
        du = PLH(mu=0.5, A=2.0, c=0.05, p=1.4, window=1500.0)  # no mark_dist -> marks are 0
        ts, ms = du.sampler(seed=0).sample()
        self.assertTrue(np.all(ms == 0.0))
        self.assertTrue(np.isfinite(du.log_density((ts, ms))))

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            PLH(mu=0.2, A=1.0, c=0.1, p=0.5, window=100.0)  # p must exceed 1


if __name__ == "__main__":
    unittest.main()
