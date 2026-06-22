"""WS-4/WS-13: base distributions expose exact mean()/variance(); verified against empirical moments.

Adding these closed-form moments populates the ``HasMoments`` capability and gives WS-13 an accuracy
anchor: each formula is checked against the sample mean/variance of a large draw.
"""

import unittest

import numpy as np

import pysp
from pysp.capability import HasMoments
from pysp.stats.base.bernoulli import BernoulliDistribution
from pysp.stats.base.beta import BetaDistribution
from pysp.stats.base.binomial import BinomialDistribution
from pysp.stats.base.exponential import ExponentialDistribution
from pysp.stats.base.gamma import GammaDistribution
from pysp.stats.base.gaussian import GaussianDistribution
from pysp.stats.base.geometric import GeometricDistribution
from pysp.stats.base.laplace import LaplaceDistribution
from pysp.stats.base.logistic import LogisticDistribution
from pysp.stats.base.poisson import PoissonDistribution
from pysp.stats.base.rayleigh import RayleighDistribution
from pysp.stats.base.uniform import UniformDistribution

CASES = [
    GaussianDistribution(2.0, 3.0),
    GammaDistribution(3.0, 0.5),
    ExponentialDistribution(2.0),
    PoissonDistribution(4.0),
    BetaDistribution(2.0, 5.0),
    BinomialDistribution(0.3, 10),
    GeometricDistribution(0.4),
    BernoulliDistribution(0.3),
    UniformDistribution(-1.0, 4.0),
    LaplaceDistribution(1.0, 2.0),
    LogisticDistribution(1.0, 2.0),
    RayleighDistribution(2.0),
]


class MomentMethodsTest(unittest.TestCase):
    def test_mean_variance_match_empirical(self):
        for dist in CASES:
            samples = np.asarray(dist.sampler(seed=7).sample(300_000), dtype=np.float64).ravel()
            emp_mean, emp_var = float(samples.mean()), float(samples.var())
            with self.subTest(dist=type(dist).__name__):
                self.assertTrue(
                    np.isclose(dist.mean(), emp_mean, rtol=0.05, atol=0.05),
                    f"{type(dist).__name__}: mean()={dist.mean()} vs empirical {emp_mean}",
                )
                self.assertTrue(
                    np.isclose(dist.variance(), emp_var, rtol=0.08, atol=0.05),
                    f"{type(dist).__name__}: variance()={dist.variance()} vs empirical {emp_var}",
                )

    def test_extended_families_match_empirical(self):
        from pysp.stats.base.log_gaussian import LogGaussianDistribution
        from pysp.stats.base.negative_binomial import NegativeBinomialDistribution
        from pysp.stats.base.pareto import ParetoDistribution
        from pysp.stats.base.student_t import StudentTDistribution
        from pysp.stats.base.weibull import WeibullDistribution

        extended = [
            NegativeBinomialDistribution(5.0, 0.4),
            StudentTDistribution(8.0, 1.0, 2.0),  # df=8 -> finite 4th moment (stable empirical var)
            WeibullDistribution(1.5, 2.0),
            ParetoDistribution(1.0, 5.0),  # alpha=5 -> finite 4th moment
            LogGaussianDistribution(0.0, 0.25),
        ]
        for dist in extended:
            samples = np.asarray(dist.sampler(seed=11).sample(300_000), dtype=np.float64).ravel()
            emp_mean, emp_var = float(samples.mean()), float(samples.var())
            with self.subTest(dist=type(dist).__name__):
                self.assertTrue(
                    np.isclose(dist.mean(), emp_mean, rtol=0.05, atol=0.05),
                    f"{type(dist).__name__}: mean()={dist.mean()} vs {emp_mean}",
                )
                self.assertTrue(
                    np.isclose(dist.variance(), emp_var, rtol=0.12, atol=0.05),
                    f"{type(dist).__name__}: variance()={dist.variance()} vs {emp_var}",
                )
        # param-range guards: undefined moments report inf
        self.assertEqual(StudentTDistribution(1.0, 0.0, 1.0).variance(), float("inf"))  # df<=2
        self.assertEqual(ParetoDistribution(1.0, 0.5).mean(), float("inf"))  # alpha<=1

    def test_has_moments_capability(self):
        for dist in CASES:
            self.assertTrue(pysp.supports(dist, HasMoments))
        # a family without the methods does not report the capability
        from pysp.stats.combinator.null_dist import NullDistribution

        self.assertFalse(pysp.supports(NullDistribution(), HasMoments))


if __name__ == "__main__":
    unittest.main()
