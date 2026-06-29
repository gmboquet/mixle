"""Generalized Pareto distribution: log-density vs scipy, inverse-CDF sampling, method-of-moments fit."""

import unittest

import numpy as np
from scipy.stats import genpareto, kstest

from mixle.inference import estimate
from mixle.stats import GeneralizedParetoDistribution


class GeneralizedParetoTest(unittest.TestCase):
    def test_log_density_matches_scipy_all_shape_regimes(self):
        for scale, xi, loc in [(2.0, 0.3, 0.0), (1.5, -0.25, 1.0), (2.0, 1e-12, 0.0)]:
            d = GeneralizedParetoDistribution(scale, xi, loc)
            xs = np.array([loc + 0.1, loc + 1.0, loc + 3.0])
            mine = d.seq_log_density(d.dist_to_encoder().seq_encode(xs))
            np.testing.assert_allclose(mine, genpareto.logpdf(xs, xi, loc=loc, scale=scale), atol=1e-10)
            np.testing.assert_allclose(mine, [d.log_density(float(x)) for x in xs], atol=1e-12)

    def test_support_boundaries(self):
        d = GeneralizedParetoDistribution(1.5, -0.25, 1.0)  # finite upper endpoint at loc - sigma/xi
        self.assertEqual(d.log_density(0.5), -np.inf)  # below threshold
        self.assertEqual(d.log_density(d._upper() + 0.5), -np.inf)  # above upper endpoint

    def test_sampler_matches_distribution(self):
        d = GeneralizedParetoDistribution(2.0, 0.3, 0.0)
        s = d.sampler(seed=0).sample(20000)
        self.assertGreater(kstest(s, "genpareto", args=(0.3, 0, 2.0)).pvalue, 0.01)

    def test_method_of_moments_recovers_params(self):
        d = GeneralizedParetoDistribution(2.0, 0.3, 0.0)
        est = estimate(list(d.sampler(seed=1).sample(20000)), d.estimator())
        self.assertAlmostEqual(est.scale, 2.0, delta=0.15)
        self.assertAlmostEqual(est.shape, 0.3, delta=0.06)

    def test_method_of_moments_recovers_negative_shape(self):
        d = GeneralizedParetoDistribution(1.5, -0.25, 0.0)
        est = estimate(list(d.sampler(seed=2).sample(20000)), d.estimator())
        self.assertAlmostEqual(est.scale, 1.5, delta=0.1)
        self.assertAlmostEqual(est.shape, -0.25, delta=0.05)


if __name__ == "__main__":
    unittest.main()
