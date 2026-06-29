"""Censored / truncated maximum-likelihood fitting and Kaplan-Meier (mixle.ppl.survival)."""

import unittest
import warnings

import numpy as np

import mixle.ppl as P
from mixle.ppl import free
from mixle.ppl.survival import fit_censored, kaplan_meier
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.weibull import WeibullDistribution


class CensoredFitTest(unittest.TestCase):
    def test_right_censored_weibull_recovers_params(self):
        # naive fit (treating censored as events) is biased; the censored fit recovers the truth
        x = np.asarray(WeibullDistribution(1.5, 10.0).sampler(seed=0).sample(2000))
        time = np.minimum(x, 12.0)
        event = x <= 12.0
        self.assertGreater(float((~event).mean()), 0.1)  # there is real censoring
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = fit_censored(P.Weibull(free, free), time, event=event).summary()
            naive = P.Weibull(free, free).fit(time, how="em").summary()
        self.assertLess(abs(s["scale"] - 10.0), 1.0)
        self.assertLess(abs(s["shape"] - 1.5), 0.3)
        self.assertLess(naive["scale"], s["scale"])  # naive underestimates the scale

    def test_truncated_exponential_recovers_rate(self):
        ex = np.asarray(ExponentialDistribution(3.0).sampler(seed=1).sample(20000))  # mean 3, rate 1/3
        trunc = ex[ex > 2.0][:4000]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = fit_censored(P.Exponential(free), trunc, lower=2.0).summary()
            naive = P.Exponential(free).fit(trunc, how="em").summary()
        self.assertLess(abs(s["rate"] - 1.0 / 3.0), 0.05)  # recovers the true rate
        self.assertLess(naive["rate"], s["rate"])  # ignoring truncation underestimates the rate

    def test_normal_censoring_reduces_bias(self):
        g = np.asarray(GaussianDistribution(0.5, 1.0).sampler(seed=2).sample(3000))
        observed = g <= 1.5
        tg = np.minimum(g, 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cens = fit_censored(P.Normal(free, free), tg, event=observed).summary()
            naive = P.Normal(free, free).fit(tg, how="em").summary()
        self.assertLess(abs(cens["mean"] - 0.5), abs(naive["mean"] - 0.5))


class KaplanMeierTest(unittest.TestCase):
    def test_monotone_in_unit_interval(self):
        x = np.asarray(WeibullDistribution(1.5, 10.0).sampler(seed=0).sample(500))
        time = np.minimum(x, 12.0)
        event = x <= 12.0
        km = kaplan_meier(time, event)
        self.assertTrue(np.all(np.diff(km["survival"]) <= 1e-12))  # non-increasing
        self.assertGreaterEqual(km["survival"].min(), 0.0)
        self.assertLessEqual(km["survival"].max(), 1.0)
        self.assertTrue(np.all(km["at_risk"] >= km["events"]))


if __name__ == "__main__":
    unittest.main()
