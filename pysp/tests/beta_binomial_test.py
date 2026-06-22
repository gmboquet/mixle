"""Beta-binomial: pmf vs scipy, sampling, and method-of-moments recovery of (a, b)."""

import unittest

import numpy as np
from scipy.stats import betabinom

from pysp.inference import estimate
from pysp.stats import BetaBinomialDistribution


class BetaBinomialTest(unittest.TestCase):
    def setUp(self):
        self.n, self.a, self.b = 10, 2.0, 3.0
        self.d = BetaBinomialDistribution(self.n, self.a, self.b)

    def test_log_pmf_matches_scipy(self):
        ks = np.arange(0, self.n + 1)
        mine = self.d.seq_log_density(self.d.dist_to_encoder().seq_encode(ks))
        np.testing.assert_allclose(mine, betabinom.logpmf(ks, self.n, self.a, self.b), atol=1e-10)
        np.testing.assert_allclose(mine, [self.d.log_density(int(k)) for k in ks], atol=1e-12)

    def test_normalizes_and_support(self):
        ks = np.arange(0, self.n + 1)
        self.assertAlmostEqual(np.exp(self.d.seq_log_density(ks)).sum(), 1.0, places=9)
        self.assertEqual(self.d.log_density(self.n + 1), -np.inf)
        self.assertEqual(self.d.log_density(-1), -np.inf)

    def test_sampler_matches_pmf(self):
        s = np.array(self.d.sampler(seed=0).sample(50000))
        emp = np.bincount(s, minlength=self.n + 1) / len(s)
        ref = betabinom.pmf(np.arange(self.n + 1), self.n, self.a, self.b)
        self.assertLess(np.abs(emp - ref).max(), 0.01)

    def test_moment_estimator_recovers_params(self):
        est = estimate(list(self.d.sampler(seed=1).sample(50000)), self.d.estimator())
        self.assertAlmostEqual(est.a, self.a, delta=0.2)
        self.assertAlmostEqual(est.b, self.b, delta=0.3)
        self.assertAlmostEqual(est.a / (est.a + est.b), self.a / (self.a + self.b), delta=0.02)

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            BetaBinomialDistribution(10, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
