"""WS-13: cross-check mixle log-density (and CDF) against scipy.stats for the standard families.

A stronger correctness check than parameter recovery: the closed-form density/CDF must match the
reference implementation pointwise, across the param conventions mixle uses.
"""

import unittest

import scipy.stats as ss

from mixle.stats.univariate.continuous.beta import BetaDistribution
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.gumbel import GumbelDistribution
from mixle.stats.univariate.continuous.half_normal import HalfNormalDistribution
from mixle.stats.univariate.continuous.inverse_gamma import InverseGammaDistribution
from mixle.stats.univariate.continuous.inverse_gaussian import InverseGaussianDistribution
from mixle.stats.univariate.continuous.laplace import LaplaceDistribution
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
from mixle.stats.univariate.continuous.logistic import LogisticDistribution
from mixle.stats.univariate.continuous.pareto import ParetoDistribution
from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution
from mixle.stats.univariate.continuous.student_t import StudentTDistribution
from mixle.stats.univariate.continuous.uniform import UniformDistribution
from mixle.stats.univariate.continuous.weibull import WeibullDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.binomial import BinomialDistribution
from mixle.stats.univariate.discrete.geometric import GeometricDistribution
from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

# (mixle instance, scipy frozen, "pdf"|"pmf", evaluation points)
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
    (GumbelDistribution(1.0, 2.0), ss.gumbel_r(loc=1.0, scale=2.0), [-1.0, 1.0, 3.0, 6.0]),
    (HalfNormalDistribution(1.5), ss.halfnorm(scale=1.5), [0.2, 1.0, 2.5, 5.0]),
    (InverseGammaDistribution(3.0, 2.0), ss.invgamma(a=3.0, scale=2.0), [0.3, 0.7, 1.5, 3.0]),
    (InverseGaussianDistribution(1.5, 3.0), ss.invgauss(mu=1.5 / 3.0, scale=3.0), [0.3, 1.0, 2.0, 4.0]),
    (LogGaussianDistribution(0.0, 0.25), ss.lognorm(s=0.5, scale=1.0), [0.3, 1.0, 2.0, 4.0]),
]
DISCRETE = [
    (PoissonDistribution(4.0), ss.poisson(mu=4.0), [0, 2, 4, 8]),
    (BinomialDistribution(0.3, 10), ss.binom(10, 0.3), [0, 3, 6, 10]),
    (GeometricDistribution(0.4), ss.geom(0.4), [1, 2, 5, 10]),
    (BernoulliDistribution(0.3), ss.bernoulli(0.3), [0, 1]),
    (NegativeBinomialDistribution(5.0, 0.4), ss.nbinom(5, 0.4), [0, 3, 7, 15]),
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
