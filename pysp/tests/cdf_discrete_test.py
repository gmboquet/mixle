"""WS-4/WS-13: cdf() for discrete base distributions, validated against the summed pmf."""

import math
import unittest

from pysp.stats.univariate.discrete.bernoulli import BernoulliDistribution
from pysp.stats.univariate.discrete.binomial import BinomialDistribution
from pysp.stats.univariate.discrete.geometric import GeometricDistribution
from pysp.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
from pysp.stats.univariate.discrete.poisson import PoissonDistribution

# (dist, support-lo, evaluation points)
CASES = [
    (PoissonDistribution(4.0), 0, [0, 2, 4, 8, 15]),
    (BinomialDistribution(0.3, 10), 0, [0, 3, 6, 10]),
    (GeometricDistribution(0.4), 1, [1, 2, 5, 12]),
    (BernoulliDistribution(0.3), 0, [0, 1]),
    (NegativeBinomialDistribution(5.0, 0.4), 0, [0, 3, 7, 20]),
]


class DiscreteCDFTest(unittest.TestCase):
    def test_cdf_equals_summed_pmf(self):
        for dist, lo, points in CASES:
            for k in points:
                with self.subTest(dist=type(dist).__name__, k=k):
                    summed = sum(math.exp(dist.log_density(j)) for j in range(lo, k + 1))
                    self.assertAlmostEqual(dist.cdf(k), summed, places=8)

    def test_cdf_bounds_and_monotone(self):
        for dist, lo, _ in CASES:
            self.assertEqual(dist.cdf(lo - 1), 0.0)
            prev = -1.0
            for k in range(lo, lo + 25):
                c = dist.cdf(k)
                self.assertGreaterEqual(c + 1e-12, prev)  # non-decreasing
                self.assertLessEqual(c, 1.0 + 1e-9)
                prev = c


if __name__ == "__main__":
    unittest.main()
