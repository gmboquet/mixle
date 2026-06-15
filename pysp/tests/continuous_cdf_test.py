"""CDF / quantile (inverse-CDF) coverage for univariate continuous leaf distributions.

For a continuous family the four enumeration-suite capabilities are realized through the CDF and its
inverse: ``cdf(x)`` is the cumulative probability (the continuous 'index of' a value), ``quantile(q)``
returns the value at cumulative-probability index ``q`` (the continuous 'arbitrary-index' / unranking),
and a quantile grid enumerates the support in order. Each family is checked for: range, monotonicity,
quantile/cdf round-trip, and -- the key correctness tie-in -- that d/dx CDF matches the family's own
``exp(log_density)``.
"""

import math
import unittest

from pysp.stats.leaf.beta import BetaDistribution
from pysp.stats.leaf.exponential import ExponentialDistribution
from pysp.stats.leaf.gamma import GammaDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution
from pysp.stats.leaf.laplace import LaplaceDistribution
from pysp.stats.leaf.log_gaussian import LogGaussianDistribution
from pysp.stats.leaf.logistic import LogisticDistribution
from pysp.stats.leaf.pareto import ParetoDistribution
from pysp.stats.leaf.rayleigh import RayleighDistribution
from pysp.stats.leaf.student_t import StudentTDistribution
from pysp.stats.leaf.uniform import UniformDistribution
from pysp.stats.leaf.weibull import WeibullDistribution

CASES = [
    ("gaussian", GaussianDistribution(0.7, 2.0), [-1.0, 0.5, 2.0]),
    ("gamma", GammaDistribution(2.5, 1.7), [0.5, 2.0, 5.0]),
    ("exponential", ExponentialDistribution(1.7), [0.2, 1.0, 3.0]),
    ("beta", BetaDistribution(2.0, 3.0), [0.1, 0.5, 0.9]),
    ("laplace", LaplaceDistribution(0.3, 1.2), [-1.0, 0.3, 2.0]),
    ("log_gaussian", LogGaussianDistribution(0.2, 0.5), [0.5, 1.0, 3.0]),
    ("logistic", LogisticDistribution(0.4, 1.3), [-1.0, 0.4, 2.0]),
    ("pareto", ParetoDistribution(1.5, 3.0), [1.6, 2.5, 5.0]),
    ("rayleigh", RayleighDistribution(1.4), [0.3, 1.4, 3.0]),
    ("student_t", StudentTDistribution(5.0, 0.3, 1.2), [-1.0, 0.3, 2.0]),
    ("uniform", UniformDistribution(-1.0, 2.0), [-0.5, 0.0, 1.5]),
    ("weibull", WeibullDistribution(1.8, 2.2), [0.5, 2.0, 4.0]),
]


class ContinuousCDFTestCase(unittest.TestCase):
    def test_cdf_in_unit_interval_and_monotone(self):
        for name, dist, xs in CASES:
            vals = [dist.cdf(x) for x in xs]
            for v in vals:
                self.assertTrue(0.0 <= v <= 1.0, "%s: cdf out of [0,1]" % name)
            for i in range(len(vals) - 1):
                self.assertLessEqual(vals[i], vals[i + 1] + 1e-12, "%s: cdf not monotone" % name)

    def test_quantile_inverts_cdf(self):
        for name, dist, xs in CASES:
            for x in xs:
                self.assertAlmostEqual(dist.quantile(dist.cdf(x)), x, delta=1e-5, msg="%s: round-trip" % name)
            for q in (0.05, 0.5, 0.95):
                self.assertAlmostEqual(dist.cdf(dist.quantile(q)), q, delta=1e-6, msg="%s: cdf(quantile)" % name)

    def test_cdf_derivative_matches_density(self):
        # The strongest check: the CDF's slope is the family's own density, tying cdf/quantile to
        # log_density (so the scipy parameterization provably matches each distribution).
        h = 1e-5
        for name, dist, xs in CASES:
            for x in xs:
                fd = (dist.cdf(x + h) - dist.cdf(x - h)) / (2.0 * h)
                self.assertAlmostEqual(fd, math.exp(dist.log_density(x)), delta=1e-3, msg="%s at %s" % (name, x))


if __name__ == "__main__":
    unittest.main()
