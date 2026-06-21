"""Watson axial distribution: Kummer normalizer, sampling, and ML recovery (bipolar + girdle)."""

import unittest

import numpy as np
from scipy.special import hyp1f1

from pysp.stats import WatsonDistribution, estimate
from pysp.stats.multivariate.watson import _kummer_ratio


class WatsonTest(unittest.TestCase):
    def setUp(self):
        self.p = 3
        self.mu = np.array([0.0, 0.0, 1.0])
        rng = np.random.RandomState(0)
        u = rng.randn(50000, self.p)
        self.uniform = u / np.linalg.norm(u, axis=1, keepdims=True)

    def test_normalizer_matches_kummer(self):
        # E_uniform[exp(kappa (mu.x)^2)] = M(1/2, p/2, kappa)
        for kappa in (5.0, -5.0):
            mc = float(np.mean(np.exp(kappa * (self.uniform @ self.mu) ** 2)))
            self.assertAlmostEqual(mc, float(hyp1f1(0.5, self.p / 2.0, kappa)), delta=0.02 * abs(mc) + 0.01)

    def test_seq_matches_scalar(self):
        d = WatsonDistribution(self.mu, 4.0)
        s = d.sampler(seed=1).sample(6)
        np.testing.assert_allclose(d.seq_log_density(s), [d.log_density(x) for x in s], atol=1e-12)

    def test_sampler_is_unit_norm_axial_and_concentrated(self):
        for kappa in (5.0, -5.0):
            d = WatsonDistribution(self.mu, kappa)
            s = d.sampler(seed=1).sample(40000)
            np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-10)
            self.assertAlmostEqual(float(np.mean((s @ self.mu) ** 2)), _kummer_ratio(kappa, self.p), delta=0.02)
            self.assertAlmostEqual(float(np.mean((s @ self.mu) > 0)), 0.5, delta=0.02)  # antipodal symmetry

    def test_mle_recovers_axis_and_kappa(self):
        for kappa in (6.0, -6.0):
            d = WatsonDistribution(self.mu, kappa)
            est = estimate(list(d.sampler(seed=2).sample(40000)), d.estimator())
            self.assertGreater(abs(float(est.mu @ self.mu)), 0.99)  # axis up to sign
            self.assertAlmostEqual(est.kappa, kappa, delta=0.6)


if __name__ == "__main__":
    unittest.main()
