"""WS-13/WS-4: base distributions expose closed-form entropy() (nats), cross-checked against scipy."""

import unittest

import scipy.stats as ss

import pysp
from pysp.capability import HasEntropy
from pysp.stats.base.beta import BetaDistribution
from pysp.stats.base.exponential import ExponentialDistribution
from pysp.stats.base.gamma import GammaDistribution
from pysp.stats.base.gaussian import GaussianDistribution
from pysp.stats.base.gumbel import GumbelDistribution
from pysp.stats.base.half_normal import HalfNormalDistribution
from pysp.stats.base.laplace import LaplaceDistribution
from pysp.stats.base.log_gaussian import LogGaussianDistribution
from pysp.stats.base.logistic import LogisticDistribution
from pysp.stats.base.pareto import ParetoDistribution
from pysp.stats.base.rayleigh import RayleighDistribution
from pysp.stats.base.uniform import UniformDistribution
from pysp.stats.base.weibull import WeibullDistribution

# (pysp instance, scipy frozen with the same parameters)
CASES = [
    (GaussianDistribution(1.0, 4.0), ss.norm(1.0, 2.0)),
    (ExponentialDistribution(2.0), ss.expon(scale=2.0)),
    (LaplaceDistribution(1.0, 2.0), ss.laplace(loc=1.0, scale=2.0)),
    (UniformDistribution(-1.0, 4.0), ss.uniform(loc=-1.0, scale=5.0)),
    (LogisticDistribution(1.0, 2.0), ss.logistic(loc=1.0, scale=2.0)),
    (RayleighDistribution(2.0), ss.rayleigh(scale=2.0)),
    (GumbelDistribution(1.0, 2.0), ss.gumbel_r(loc=1.0, scale=2.0)),
    (WeibullDistribution(1.5, 2.0), ss.weibull_min(c=1.5, scale=2.0)),
    (ParetoDistribution(1.0, 2.5), ss.pareto(b=2.5, scale=1.0)),
    (GammaDistribution(3.0, 0.5), ss.gamma(a=3.0, scale=0.5)),
    (BetaDistribution(2.0, 5.0), ss.beta(2.0, 5.0)),
    (HalfNormalDistribution(1.5), ss.halfnorm(scale=1.5)),
    (LogGaussianDistribution(0.0, 0.25), ss.lognorm(s=0.5, scale=1.0)),
]


class EntropyMethodsTest(unittest.TestCase):
    def test_entropy_matches_scipy(self):
        for dist, frozen in CASES:
            with self.subTest(dist=type(dist).__name__):
                self.assertAlmostEqual(dist.entropy(), float(frozen.entropy()), places=8)

    def test_has_entropy_capability(self):
        for dist, _ in CASES:
            self.assertTrue(pysp.supports(dist, HasEntropy))


if __name__ == "__main__":
    unittest.main()
