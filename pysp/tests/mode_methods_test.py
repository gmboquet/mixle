"""WS-13: base distributions expose mode() (the density argmax), verified numerically."""

import unittest

import numpy as np

import pysp
from pysp.stats.univariate.continuous.beta import BetaDistribution
from pysp.stats.univariate.continuous.gamma import GammaDistribution
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution
from pysp.stats.univariate.continuous.gumbel import GumbelDistribution
from pysp.stats.univariate.continuous.laplace import LaplaceDistribution
from pysp.stats.univariate.continuous.rayleigh import RayleighDistribution
from pysp.stats.univariate.continuous.weibull import WeibullDistribution
from pysp.stats.univariate.discrete.bernoulli import BernoulliDistribution
from pysp.stats.univariate.discrete.binomial import BinomialDistribution
from pysp.stats.univariate.discrete.poisson import PoissonDistribution

CONTINUOUS = [
    (GaussianDistribution(2.0, 3.0), -8.0, 12.0),
    (GammaDistribution(3.0, 0.5), 1e-4, 10.0),
    (BetaDistribution(2.0, 5.0), 1e-4, 1 - 1e-4),
    (WeibullDistribution(1.5, 2.0), 1e-4, 10.0),
    (RayleighDistribution(2.0), 1e-4, 12.0),
    (GumbelDistribution(1.0, 2.0), -10.0, 15.0),
    (LaplaceDistribution(1.0, 2.0), -15.0, 17.0),
]


class ModeMethodsTest(unittest.TestCase):
    def test_continuous_mode_is_density_argmax(self):
        for dist, lo, hi in CONTINUOUS:
            xs = np.linspace(lo, hi, 40001)
            dens = np.array([dist.density(x) for x in xs])
            with self.subTest(dist=type(dist).__name__):
                self.assertAlmostEqual(dist.mode(), xs[int(np.argmax(dens))], delta=(hi - lo) / 4000)

    def test_discrete_mode_is_pmf_argmax(self):
        for dist, ks in [
            (PoissonDistribution(4.3), range(0, 20)),
            (BinomialDistribution(0.3, 10), range(0, 11)),
            (BernoulliDistribution(0.3), range(0, 2)),
        ]:
            pmf = np.array([np.exp(dist.log_density(k)) for k in ks])
            with self.subTest(dist=type(dist).__name__):
                self.assertEqual(int(dist.mode()), list(ks)[int(np.argmax(pmf))])

    def test_summarize_includes_mode(self):
        s = pysp.summarize(GaussianDistribution(2.0, 3.0))
        self.assertAlmostEqual(s["mode"], 2.0)
        self.assertAlmostEqual(s["mode"], s["mean"])  # symmetric -> mode == mean == median

    def test_range_guards(self):
        self.assertEqual(GammaDistribution(0.5, 1.0).mode(), 0.0)  # k<1 -> mode at 0
        self.assertEqual(WeibullDistribution(0.8, 2.0).mode(), 0.0)  # shape<1 -> mode at 0


if __name__ == "__main__":
    unittest.main()
