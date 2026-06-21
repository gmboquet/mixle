"""Inverse-Wishart over SPD matrices: density vs scipy, sampling (invert Wishart), scale MLE."""

import unittest

import numpy as np
from scipy.stats import invwishart as siw

from pysp.stats import InverseWishartDistribution, estimate


class InverseWishartTest(unittest.TestCase):
    def setUp(self):
        self.P = np.array([[2.0, 0.3], [0.3, 1.0]])
        self.df = 8
        self.p = 2
        self.d = InverseWishartDistribution(self.df, self.P)

    def test_log_density_matches_scipy(self):
        xs = np.array([self.d.sampler(seed=k).sample() for k in range(4)])
        mine = self.d.seq_log_density(xs)
        ref = np.array([siw.logpdf(x, self.df, self.P) for x in xs])
        np.testing.assert_allclose(mine, ref, atol=1e-9)
        np.testing.assert_allclose(mine, [self.d.log_density(x) for x in xs], atol=1e-10)

    def test_non_pd_is_minus_inf(self):
        self.assertEqual(self.d.log_density(np.array([[1.0, 2.0], [2.0, 1.0]])), -np.inf)

    def test_sampler_is_spd_with_correct_mean(self):
        s = self.d.sampler(seed=0).sample(40000)
        self.assertTrue(np.all(np.linalg.eigvalsh(s[:300]) > 0))
        np.testing.assert_allclose(s.mean(axis=0), self.P / (self.df - self.p - 1), atol=0.04)  # E[X]=Psi/(df-p-1)

    def test_scale_estimator_recovers_psi(self):
        est = estimate(list(self.d.sampler(seed=1).sample(30000)), self.d.estimator())
        np.testing.assert_allclose(est.scale, self.P, atol=0.12)

    def test_df_too_small_raises(self):
        with self.assertRaises(ValueError):
            InverseWishartDistribution(2, np.eye(3))  # df must be > p-1 = 2


if __name__ == "__main__":
    unittest.main()
