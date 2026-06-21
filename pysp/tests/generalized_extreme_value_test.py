"""Generalized Extreme Value distribution: log-density vs scipy, sampling, method-of-moments fit."""

import unittest

import numpy as np
from scipy.stats import genextreme, kstest

from pysp.stats import GeneralizedExtremeValueDistribution, estimate


class GeneralizedExtremeValueTest(unittest.TestCase):
    def test_log_density_matches_scipy_all_types(self):
        # scipy's genextreme uses c = -xi
        for loc, scale, xi in [(0.0, 2.0, 0.2), (1.0, 1.5, -0.3), (0.0, 2.0, 1e-12)]:
            d = GeneralizedExtremeValueDistribution(loc, scale, xi)
            xs = np.array([loc - 0.5, loc + 1.0, loc + 4.0])
            mine = d.seq_log_density(d.dist_to_encoder().seq_encode(xs))
            np.testing.assert_allclose(mine, genextreme.logpdf(xs, -xi, loc=loc, scale=scale), atol=1e-9)
            np.testing.assert_allclose(mine, [d.log_density(float(x)) for x in xs], atol=1e-12)

    def test_frechet_support_lower_bound(self):
        d = GeneralizedExtremeValueDistribution(0.0, 2.0, 0.5)  # xi>0 -> support x >= loc - scale/xi
        self.assertEqual(d.log_density(0.0 - 2.0 / 0.5 - 0.5), -np.inf)

    def test_sampler_matches_distribution(self):
        d = GeneralizedExtremeValueDistribution(0.0, 2.0, 0.2)
        s = d.sampler(seed=0).sample(40000)
        self.assertGreater(kstest(s, "genextreme", args=(-0.2, 0, 2.0)).pvalue, 0.01)

    def test_method_of_moments_recovers_all_three_types(self):
        for loc, scale, xi, seed in [(0.0, 2.0, 0.2, 0), (1.0, 1.5, -0.3, 1), (0.0, 2.0, 0.0, 3)]:
            d = GeneralizedExtremeValueDistribution(loc, scale, xi)
            est = estimate(list(d.sampler(seed=seed).sample(40000)), d.estimator())
            self.assertAlmostEqual(est.loc, loc, delta=0.1)
            self.assertAlmostEqual(est.scale, scale, delta=0.12)
            self.assertAlmostEqual(est.shape, xi, delta=0.05)


if __name__ == "__main__":
    unittest.main()
