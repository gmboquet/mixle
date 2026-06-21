"""Wrapped Cauchy circular distribution: density vs scipy, exact sampling, mean-resultant estimation."""

import unittest

import numpy as np
from scipy.stats import kstest, wrapcauchy

from pysp.stats import WrappedCauchyDistribution, estimate


class WrappedCauchyTest(unittest.TestCase):
    def setUp(self):
        self.mu, self.rho = 0.7, 0.6
        self.d = WrappedCauchyDistribution(self.mu, self.rho)

    def test_log_density_matches_scipy(self):
        th = np.array([0.1, 0.7, 2.0, -1.5])
        mine = self.d.seq_log_density(self.d.dist_to_encoder().seq_encode(th))
        ref = wrapcauchy.logpdf((th - self.mu) % (2 * np.pi), self.rho)  # scipy lives on [0, 2pi)
        np.testing.assert_allclose(mine, ref, atol=1e-10)
        np.testing.assert_allclose(mine, [self.d.log_density(t) for t in th], atol=1e-12)

    def test_density_integrates_to_one(self):
        g = np.linspace(-np.pi, np.pi, 6000)
        self.assertAlmostEqual(np.trapezoid([self.d.density(t) for t in g], g), 1.0, places=3)

    def test_sampler_matches_distribution(self):
        s = self.d.sampler(seed=0).sample(40000)
        self.assertAlmostEqual(float(np.mean(np.cos(s - self.mu))), self.rho, delta=0.02)  # E[cos(theta-mu)]=rho
        self.assertGreater(kstest((s - self.mu) % (2 * np.pi), "wrapcauchy", args=(self.rho,)).pvalue, 0.01)

    def test_mean_resultant_estimator_recovers_params(self):
        est = estimate(list(self.d.sampler(seed=1).sample(40000)), self.d.estimator())
        self.assertAlmostEqual(est.mu, self.mu, delta=0.03)
        self.assertAlmostEqual(est.rho, self.rho, delta=0.03)

    def test_rho_zero_is_uniform(self):
        s = WrappedCauchyDistribution(0.0, 0.0).sampler(seed=2).sample(20000)
        self.assertLess(np.hypot(np.mean(np.cos(s)), np.mean(np.sin(s))), 0.03)  # no mean resultant

    def test_invalid_rho_raises(self):
        with self.assertRaises(ValueError):
            WrappedCauchyDistribution(0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
