"""WS-13/WS-4: base distributions expose closed-form entropy() (nats), cross-checked against scipy."""

import math
import unittest

import scipy.stats as ss

import mixle
from mixle.capability import HasEntropy
from mixle.stats.univariate.continuous.beta import BetaDistribution
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.gumbel import GumbelDistribution
from mixle.stats.univariate.continuous.half_normal import HalfNormalDistribution
from mixle.stats.univariate.continuous.laplace import LaplaceDistribution
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
from mixle.stats.univariate.continuous.logistic import LogisticDistribution
from mixle.stats.univariate.continuous.nakagami import NakagamiDistribution
from mixle.stats.univariate.continuous.pareto import ParetoDistribution
from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution
from mixle.stats.univariate.continuous.rician import RicianDistribution
from mixle.stats.univariate.continuous.skew_normal import SkewNormalDistribution
from mixle.stats.univariate.continuous.uniform import UniformDistribution
from mixle.stats.univariate.continuous.weibull import WeibullDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.geometric import GeometricDistribution
from mixle.stats.univariate.discrete.logseries import LogSeriesDistribution
from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
from mixle.stats.univariate.discrete.skellam import SkellamDistribution

# (mixle instance, scipy frozen with the same parameters)
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
    (BernoulliDistribution(0.3), ss.bernoulli(0.3)),
    (GeometricDistribution(0.4), ss.geom(0.4)),
    # F-9 completion (PR #434 left these six unfinished; see entropy_completeness_test.py for the
    # from-scratch numerical-integration/Monte-Carlo verification these scipy cross-checks don't cover).
    (SkewNormalDistribution(0.5, 1.5, 2.0), ss.skewnorm(2.0, loc=0.5, scale=1.5)),
    (NegativeBinomialDistribution(5.0, 0.3), ss.nbinom(5.0, 0.3)),
    (RicianDistribution(3.0, 1.0), ss.rice(3.0, scale=1.0)),
    (NakagamiDistribution(2.5, 2.0), ss.nakagami(2.5, scale=math.sqrt(2.0))),
    (SkellamDistribution(5.0, 5.0), ss.skellam(5.0, 5.0)),
    (LogSeriesDistribution(0.7), ss.logser(0.7)),
]


class EntropyMethodsTest(unittest.TestCase):
    def test_entropy_matches_scipy(self):
        for dist, frozen in CASES:
            with self.subTest(dist=type(dist).__name__):
                self.assertAlmostEqual(dist.entropy(), float(frozen.entropy()), places=8)

    def test_has_entropy_capability(self):
        for dist, _ in CASES:
            self.assertTrue(mixle.supports(dist, HasEntropy))


if __name__ == "__main__":
    unittest.main()
