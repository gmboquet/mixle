"""Skew-normal distribution: density vs scipy, exact sampling, 3-moment estimation."""

import unittest

import numpy as np
from scipy.stats import kstest, skewnorm

from pysp.inference import estimate
from pysp.stats import SkewNormalDistribution


class SkewNormalTest(unittest.TestCase):
    def setUp(self):
        self.loc, self.scale, self.alpha = 0.5, 2.0, 4.0
        self.d = SkewNormalDistribution(self.loc, self.scale, self.alpha)

    def test_log_density_matches_scipy(self):
        xs = np.array([-1.0, 0.5, 2.0, 5.0])
        mine = self.d.seq_log_density(xs)
        np.testing.assert_allclose(mine, skewnorm.logpdf(xs, self.alpha, self.loc, self.scale), atol=1e-10)
        np.testing.assert_allclose(mine, [self.d.log_density(x) for x in xs], atol=1e-12)

    def test_zero_shape_is_normal(self):
        d0 = SkewNormalDistribution(1.0, 2.0, 0.0)
        xs = np.array([-1.0, 0.5, 2.0])
        np.testing.assert_allclose(d0.seq_log_density(xs), skewnorm.logpdf(xs, 0.0, 1.0, 2.0), atol=1e-12)

    def test_sampler_matches_distribution(self):
        s = self.d.sampler(seed=0).sample(40000)
        self.assertGreater(kstest(s, "skewnorm", args=(self.alpha, self.loc, self.scale)).pvalue, 0.01)

    def test_moment_estimator_recovers_both_skew_directions(self):
        for alpha in (4.0, -4.0):
            d = SkewNormalDistribution(0.5, 2.0, alpha)
            est = estimate(list(d.sampler(seed=1).sample(60000)), d.estimator())
            self.assertAlmostEqual(est.loc, 0.5, delta=0.1)
            self.assertAlmostEqual(est.scale, 2.0, delta=0.1)
            self.assertAlmostEqual(est.shape, alpha, delta=0.7)

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            SkewNormalDistribution(0.0, -1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
