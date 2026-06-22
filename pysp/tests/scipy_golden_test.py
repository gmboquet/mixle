"""WS-13: cross-check pysp log-density (and CDF) against scipy.stats for the standard families.

A stronger correctness check than parameter recovery: the closed-form density/CDF must match the
reference implementation pointwise, across the param conventions pysp uses.
"""

import unittest

import scipy.stats as ss

from pysp.stats.base.bernoulli import BernoulliDistribution
from pysp.stats.base.beta import BetaDistribution
from pysp.stats.base.binomial import BinomialDistribution
from pysp.stats.base.exponential import ExponentialDistribution
from pysp.stats.base.gamma import GammaDistribution
from pysp.stats.base.gaussian import GaussianDistribution
from pysp.stats.base.geometric import GeometricDistribution
from pysp.stats.base.laplace import LaplaceDistribution
from pysp.stats.base.logistic import LogisticDistribution
from pysp.stats.base.pareto import ParetoDistribution
from pysp.stats.base.poisson import PoissonDistribution
from pysp.stats.base.rayleigh import RayleighDistribution
from pysp.stats.base.student_t import StudentTDistribution
from pysp.stats.base.uniform import UniformDistribution
from pysp.stats.base.weibull import WeibullDistribution

# (pysp instance, scipy frozen, "pdf"|"pmf", evaluation points)
CONTINUOUS = [
    (GaussianDistribution(1.0, 4.0), ss.norm(loc=1.0, scale=2.0), [-1.0, 0.5, 1.0, 3.2]),
    (GammaDistribution(3.0, 0.5), ss.gamma(a=3.0, scale=0.5), [0.2, 1.0, 2.5, 5.0]),
    (ExponentialDistribution(2.0), ss.expon(scale=2.0), [0.1, 1.0, 3.0, 6.0]),
    (BetaDistribution(2.0, 5.0), ss.beta(2.0, 5.0), [0.05, 0.3, 0.6, 0.9]),
    (UniformDistribution(-1.0, 4.0), ss.uniform(loc=-1.0, scale=5.0), [-0.5, 0.0, 2.0, 3.9]),
    (LaplaceDistribution(1.0, 2.0), ss.laplace(loc=1.0, scale=2.0), [-2.0, 0.0, 1.0, 4.0]),
    (LogisticDistribution(1.0, 2.0), ss.logistic(loc=1.0, scale=2.0), [-3.0, 0.0, 1.0, 5.0]),
    (RayleighDistribution(2.0), ss.rayleigh(scale=2.0), [0.3, 1.0, 2.5, 5.0]),
    (StudentTDistribution(4.0, 0.0, 1.0), ss.t(df=4.0, loc=0.0, scale=1.0), [-3.0, -0.5, 0.7, 2.5]),
    (WeibullDistribution(1.5, 2.0), ss.weibull_min(c=1.5, scale=2.0), [0.2, 1.0, 2.0, 4.0]),
    (ParetoDistribution(1.0, 2.5), ss.pareto(b=2.5, scale=1.0), [1.2, 1.5, 3.0, 6.0]),
]
DISCRETE = [
    (PoissonDistribution(4.0), ss.poisson(mu=4.0), [0, 2, 4, 8]),
    (BinomialDistribution(0.3, 10), ss.binom(10, 0.3), [0, 3, 6, 10]),
    (GeometricDistribution(0.4), ss.geom(0.4), [1, 2, 5, 10]),
    (BernoulliDistribution(0.3), ss.bernoulli(0.3), [0, 1]),
]


class ScipyGoldenTest(unittest.TestCase):
    def test_continuous_logpdf_and_cdf_match_scipy(self):
        for dist, frozen, xs in CONTINUOUS:
            for x in xs:
                with self.subTest(dist=type(dist).__name__, x=x):
                    self.assertAlmostEqual(dist.log_density(x), float(frozen.logpdf(x)), places=8)
                    if callable(getattr(dist, "cdf", None)):
                        self.assertAlmostEqual(dist.cdf(x), float(frozen.cdf(x)), places=8)

    def test_discrete_logpmf_and_cdf_match_scipy(self):
        for dist, frozen, xs in DISCRETE:
            for x in xs:
                with self.subTest(dist=type(dist).__name__, x=x):
                    self.assertAlmostEqual(dist.log_density(x), float(frozen.logpmf(x)), places=8)
                    if callable(getattr(dist, "cdf", None)):
                        self.assertAlmostEqual(dist.cdf(x), float(frozen.cdf(x)), places=8)


if __name__ == "__main__":
    unittest.main()
