"""Posterior / prior predictive checks for the mixle PPL (mixle.ppl.predictive)."""

import unittest
from types import SimpleNamespace

import numpy as np

import mixle.ppl as P
from mixle.ppl.predictive import posterior_predictive_check, prior_predictive, prior_predictive_check
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.skew_normal import SkewNormalDistribution


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

    def test_rejects_empty_data_and_non_positive_replications(self):
        fitted = SimpleNamespace(predict=lambda size, rng: np.zeros(size))
        with self.assertRaises(ValueError):
            posterior_predictive_check(fitted, [], n_rep=1)
        for value in (0, -1):
            with self.subTest(n_rep=value), self.assertRaises(ValueError):
                posterior_predictive_check(fitted, [0.0], n_rep=value)
        with self.assertRaises(TypeError):
            posterior_predictive_check(fitted, [0.0], n_rep=1.5)

    def test_rejects_malformed_replicates_and_named_statistics(self):
        short = SimpleNamespace(predict=lambda size, rng: np.zeros(size - 1))
        with self.assertRaisesRegex(ValueError, "exactly 3 observations"):
            posterior_predictive_check(short, [1.0, 2.0, 3.0], n_rep=1)

        wrong_shape = SimpleNamespace(predict=lambda size, rng: np.zeros((size, 1)))
        with self.assertRaisesRegex(ValueError, "match observed shape"):
            posterior_predictive_check(wrong_shape, [1.0, 2.0], n_rep=1)

        nonfinite = SimpleNamespace(predict=lambda size, rng: np.full(size, np.nan))
        with self.assertRaisesRegex(ValueError, "replicate 0.*finite"):
            posterior_predictive_check(nonfinite, [1.0, 2.0], n_rep=1)

        valid = SimpleNamespace(predict=lambda size, rng: np.zeros(size))
        with self.assertRaisesRegex(ValueError, "'vector'.*scalar"):
            posterior_predictive_check(
                valid,
                [1.0, 2.0],
                statistics={"vector": lambda values: values},
                n_rep=1,
            )
        with self.assertRaisesRegex(ValueError, "'bad'.*non-finite"):
            posterior_predictive_check(
                valid,
                [1.0, 2.0],
                statistics={"bad": lambda values: np.nan},
                n_rep=1,
            )


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

    def test_structured_hyperprior_draw_keeps_its_vector_shape(self):
        model = P.Categorical(P.Dirichlet([1.0, 1.0, 1.0], name="probabilities"))
        pp = prior_predictive(model, 8, n_rep=4, statistics={"mean": np.mean}, seed=0)
        self.assertEqual(pp["samples"].shape, (4, 8))
        self.assertTrue(set(np.unique(pp["samples"])).issubset({0.0, 1.0, 2.0}))

    def test_rejects_non_positive_sample_counts(self):
        model = P.Normal(0.0, 1.0)
        for kwargs, error in [
            ({"size": 0, "n_rep": 1}, ValueError),
            ({"size": 1, "n_rep": 0}, ValueError),
            ({"size": 1.5, "n_rep": 1}, TypeError),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                prior_predictive(model, **kwargs)


if __name__ == "__main__":
    unittest.main()
