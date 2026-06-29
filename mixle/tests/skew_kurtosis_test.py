"""WS-13: base distributions expose closed-form skewness()/kurtosis() (excess), checked vs scipy."""

import unittest

import scipy.stats as ss

from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.gumbel import GumbelDistribution
from mixle.stats.univariate.continuous.laplace import LaplaceDistribution
from mixle.stats.univariate.continuous.logistic import LogisticDistribution
from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution
from mixle.stats.univariate.continuous.uniform import UniformDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.binomial import BinomialDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

CASES = [
    (GaussianDistribution(1.0, 4.0), ss.norm(1.0, 2.0)),
    (ExponentialDistribution(2.0), ss.expon(scale=2.0)),
    (GammaDistribution(3.0, 0.5), ss.gamma(a=3.0, scale=0.5)),
    (LaplaceDistribution(1.0, 2.0), ss.laplace(loc=1.0, scale=2.0)),
    (LogisticDistribution(1.0, 2.0), ss.logistic(loc=1.0, scale=2.0)),
    (UniformDistribution(-1.0, 4.0), ss.uniform(loc=-1.0, scale=5.0)),
    (RayleighDistribution(2.0), ss.rayleigh(scale=2.0)),
    (GumbelDistribution(1.0, 2.0), ss.gumbel_r(loc=1.0, scale=2.0)),
    (PoissonDistribution(4.0), ss.poisson(mu=4.0)),
    (BernoulliDistribution(0.3), ss.bernoulli(0.3)),
    (BinomialDistribution(0.3, 10), ss.binom(10, 0.3)),
]


class SkewKurtosisTest(unittest.TestCase):
    def test_matches_scipy(self):
        for dist, frozen in CASES:
            sk, ku = frozen.stats(moments="sk")
            with self.subTest(dist=type(dist).__name__):
                self.assertAlmostEqual(dist.skewness(), float(sk), places=7)
                self.assertAlmostEqual(dist.kurtosis(), float(ku), places=7)  # excess kurtosis


if __name__ == "__main__":
    unittest.main()
