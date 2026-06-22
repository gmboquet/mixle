"""WS-13: base distributions expose closed-form skewness()/kurtosis() (excess), checked vs scipy."""

import unittest

import scipy.stats as ss

from pysp.stats.base.bernoulli import BernoulliDistribution
from pysp.stats.base.binomial import BinomialDistribution
from pysp.stats.base.exponential import ExponentialDistribution
from pysp.stats.base.gamma import GammaDistribution
from pysp.stats.base.gaussian import GaussianDistribution
from pysp.stats.base.gumbel import GumbelDistribution
from pysp.stats.base.laplace import LaplaceDistribution
from pysp.stats.base.logistic import LogisticDistribution
from pysp.stats.base.poisson import PoissonDistribution
from pysp.stats.base.rayleigh import RayleighDistribution
from pysp.stats.base.uniform import UniformDistribution

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
