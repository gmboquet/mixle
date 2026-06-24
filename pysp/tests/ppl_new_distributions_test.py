"""New PPL distribution primitives: parameter recovery as likelihoods and use as priors.

These expose pysp leaf distributions (HalfNormal, InverseGamma, InverseGaussian, Gumbel, SkewNormal,
Skellam, LogSeries) as pysp.ppl random variables -- the scale/variance priors and skewed/heavy-tailed
likelihoods a Bayesian modeller reaches for. Each is checked by sampling from the known pysp law and
recovering its parameters through the PPL EM/MLE path.
"""

import unittest

import numpy as np

import pysp.ppl as P
from pysp.ppl import free
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution
from pysp.stats.univariate.continuous.gumbel import GumbelDistribution
from pysp.stats.univariate.continuous.half_normal import HalfNormalDistribution
from pysp.stats.univariate.continuous.inverse_gamma import InverseGammaDistribution
from pysp.stats.univariate.continuous.inverse_gaussian import InverseGaussianDistribution
from pysp.stats.univariate.continuous.skew_normal import SkewNormalDistribution
from pysp.stats.univariate.discrete.logseries import LogSeriesDistribution
from pysp.stats.univariate.discrete.skellam import SkellamDistribution


class NewPplDistributionsTest(unittest.TestCase):
    def _recovers(self, truth, model_ctor, expected, n=5000, rel=0.2):
        data = truth.sampler(seed=0).sample(n)
        fitted = model_ctor().fit(data, how="em")
        s = fitted.summary()
        for key, val in expected.items():
            self.assertIn(key, s)
            self.assertLess(abs(float(s[key]) - val), rel * abs(val) + 0.15, f"{key}: {s[key]} vs {val}")

    def test_half_normal(self):
        self._recovers(HalfNormalDistribution(2.0), lambda: P.HalfNormal(free), {"sigma": 2.0})

    def test_inverse_gamma(self):
        self._recovers(InverseGammaDistribution(4.0, 6.0), lambda: P.InverseGamma(free, free), {"alpha": 4.0, "beta": 6.0})

    def test_inverse_gaussian(self):
        self._recovers(InverseGaussianDistribution(2.0, 3.0), lambda: P.InverseGaussian(free, free), {"mu": 2.0, "lam": 3.0})

    def test_gumbel(self):
        self._recovers(GumbelDistribution(1.0, 2.0), lambda: P.Gumbel(free, free), {"loc": 1.0, "scale": 2.0})

    def test_skew_normal(self):
        self._recovers(SkewNormalDistribution(0.0, 1.5, 4.0), lambda: P.SkewNormal(free, free, free), {"scale": 1.5})

    def test_skellam(self):
        self._recovers(SkellamDistribution(4.0, 2.0), lambda: P.Skellam(free, free), {"mu1": 4.0, "mu2": 2.0})

    def test_logseries(self):
        self._recovers(LogSeriesDistribution(0.6), lambda: P.LogSeries(free), {"p": 0.6})

    def test_half_normal_as_scale_prior(self):
        # the canonical use: a weakly-informative scale prior recovered by MAP
        data = GaussianDistribution(0.0, 3.0**2).sampler(seed=0).sample(3000)
        fitted = P.Normal(0.0, P.HalfNormal(2.0)).fit(data, how="map")
        self.assertLess(abs(fitted.summary()["sd"] - 3.0), 0.6)

    def test_all_exported(self):
        for name in ("HalfNormal", "InverseGamma", "InverseGaussian", "Gumbel", "SkewNormal", "Skellam", "LogSeries"):
            self.assertIn(name, P.__all__)
            self.assertTrue(hasattr(P, name))


if __name__ == "__main__":
    unittest.main()
