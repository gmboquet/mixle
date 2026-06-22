"""Wishart distribution over SPD matrices: density vs scipy, Bartlett sampling, closed-form scale MLE."""

import unittest

import numpy as np
from scipy.stats import wishart as sw

from pysp.inference import estimate
from pysp.stats import WishartDistribution


class WishartTest(unittest.TestCase):
    def setUp(self):
        self.V = np.array([[2.0, 0.3], [0.3, 1.0]])
        self.df = 6
        self.d = WishartDistribution(self.df, self.V)

    def test_log_density_matches_scipy(self):
        xs = np.array([self.d.sampler(seed=k).sample() for k in range(4)])
        mine = self.d.seq_log_density(xs)
        ref = np.array([sw.logpdf(x, self.df, self.V) for x in xs])
        np.testing.assert_allclose(mine, ref, atol=1e-9)
        np.testing.assert_allclose(mine, [self.d.log_density(x) for x in xs], atol=1e-10)

    def test_non_pd_is_minus_inf(self):
        self.assertEqual(self.d.log_density(np.array([[1.0, 2.0], [2.0, 1.0]])), -np.inf)  # indefinite

    def test_sampler_is_spd_with_correct_mean(self):
        s = self.d.sampler(seed=0).sample(40000)
        self.assertTrue(np.all(np.linalg.eigvalsh(s[:300]) > 0))  # all SPD
        np.testing.assert_allclose(s.mean(axis=0), self.df * self.V, atol=0.12)  # E[X] = df V

    def test_scale_estimator_recovers_V(self):
        est = estimate(list(self.d.sampler(seed=1).sample(20000)), self.d.estimator())
        np.testing.assert_allclose(est.scale, self.V, atol=0.06)

    def test_df_below_dim_raises(self):
        with self.assertRaises(ValueError):
            WishartDistribution(1, np.eye(3))


if __name__ == "__main__":
    unittest.main()
