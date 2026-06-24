"""Posterior / prior predictive checks for the pysp PPL (pysp.ppl.predictive)."""

import unittest

import pysp.ppl as P
from pysp.ppl.predictive import posterior_predictive_check, prior_predictive, prior_predictive_check
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution
from pysp.stats.univariate.continuous.skew_normal import SkewNormalDistribution


def _skewness(y):
    return float(((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-12))


class PosteriorPredictiveCheckTest(unittest.TestCase):
    def test_good_fit_pvalues_are_central(self):
        # a correctly-specified model reproduces the data's location/spread -> p-values away from 0/1
        data = GaussianDistribution(1.0, 4.0).sampler(seed=0).sample(500)
        fit = P.Normal(P.free, P.free).fit(data, how="em")
        r = posterior_predictive_check(fit, data, n_rep=400, seed=1)
        for stat in ("mean", "std"):
            self.assertGreater(r["p_value"][stat], 0.05)
            self.assertLess(r["p_value"][stat], 0.95)
        self.assertEqual(r["replicated"]["mean"].shape, (400,))

    def test_misspecified_model_is_flagged(self):
        # a Normal fit to skewed data cannot reproduce the skew -> an extreme Bayesian p-value
        data = SkewNormalDistribution(0.0, 1.0, 8.0).sampler(seed=0).sample(500)
        fit = P.Normal(P.free, P.free).fit(data, how="em")
        r = posterior_predictive_check(fit, data, statistics={"skew": _skewness}, n_rep=400, seed=1)
        self.assertLess(r["p_value"]["skew"], 0.02)


class PriorPredictiveTest(unittest.TestCase):
    def test_prior_predictive_varies_with_prior(self):
        model = P.Normal(0.0, P.HalfNormal(2.0))  # sigma drawn from the prior each replicate
        pp = prior_predictive(model, 50, n_rep=300, seed=0)
        self.assertEqual(pp["samples"].shape, (300, 50))
        self.assertGreater(pp["replicated"]["std"].std(), 0.1)  # spread varies because sigma is random

    def test_prior_predictive_check_flags_location_mismatch(self):
        # data centred at 1, but the prior centres the mean at 0 -> the mean statistic is extreme
        data = GaussianDistribution(1.0, 0.25).sampler(seed=0).sample(400)
        model = P.Normal(0.0, P.HalfNormal(0.5))
        r = prior_predictive_check(model, data, n_rep=300, seed=0)
        self.assertLess(r["p_value"]["mean"], 0.05)


if __name__ == "__main__":
    unittest.main()
