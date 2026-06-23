"""WS-4/WS-13: cdf()/quantile() added to gumbel/half_normal/inverse_gamma/inverse_gaussian.

Validated independently of scipy via the sampler: the analytic CDF must match the empirical CDF of a
large draw, and cdf(quantile(q)) must round-trip to q.
"""

import unittest

import numpy as np

import pysp
from pysp.capability import HasCDF
from pysp.stats.univariate.continuous.gumbel import GumbelDistribution
from pysp.stats.univariate.continuous.half_normal import HalfNormalDistribution
from pysp.stats.univariate.continuous.inverse_gamma import InverseGammaDistribution
from pysp.stats.univariate.continuous.inverse_gaussian import InverseGaussianDistribution

CASES = [
    GumbelDistribution(1.0, 2.0),
    HalfNormalDistribution(1.5),
    InverseGammaDistribution(3.0, 2.0),
    InverseGaussianDistribution(1.5, 3.0),
]


class CDFCoverageTest(unittest.TestCase):
    def test_cdf_matches_empirical(self):
        for dist in CASES:
            samples = np.asarray(dist.sampler(seed=4).sample(200_000), dtype=np.float64).ravel()
            for q in (0.1, 0.3, 0.5, 0.7, 0.9):
                x = float(np.quantile(samples, q))
                with self.subTest(dist=type(dist).__name__, q=q):
                    self.assertAlmostEqual(dist.cdf(x), q, delta=0.01)  # analytic cdf ~ empirical quantile level

    def test_quantile_inverts_cdf(self):
        for dist in CASES:
            for q in (0.05, 0.25, 0.5, 0.75, 0.95):
                with self.subTest(dist=type(dist).__name__, q=q):
                    self.assertAlmostEqual(dist.cdf(dist.quantile(q)), q, places=6)

    def test_has_cdf_capability(self):
        for dist in CASES:
            self.assertTrue(pysp.supports(dist, HasCDF))


if __name__ == "__main__":
    unittest.main()
