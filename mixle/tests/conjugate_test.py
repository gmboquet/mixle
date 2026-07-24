"""Conjugate-posterior inference derived from the exponential-family map (mixle.stats.conjugate)."""

import math
import unittest

import numpy as np
from scipy.special import betaln, gammaln

from mixle.stats import (
    BernoulliDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    MultivariateGaussianDistribution,
    PoissonDistribution,
    RayleighDistribution,
    conjugate_posterior,
    mixture_conjugate_posterior,
)


class BetaConjugateTest(unittest.TestCase):
    def test_bernoulli_posterior_and_evidence(self):
        rng = np.random.RandomState(0)
        x = (rng.rand(4000) < 0.3).astype(int)
        post = conjugate_posterior(BernoulliDistribution(0.5), x, prior={"a": 1.0, "b": 1.0})
        s, n = float(x.sum()), float(len(x))
        self.assertAlmostEqual(post.a, 1.0 + s)
        self.assertAlmostEqual(post.b, 1.0 + n - s)
        self.assertAlmostEqual(post.mean()["p"], (1.0 + s) / (2.0 + n))
        self.assertAlmostEqual(post.point_estimate().p, post.mean()["p"])
        # closed-form evidence == B(a_n,b_n)/B(a0,b0) exactly (Bernoulli base measure is 1)
        self.assertAlmostEqual(
            post.log_marginal_likelihood(), betaln(1.0 + s, 1.0 + n - s) - betaln(1.0, 1.0), places=9
        )

    def test_geometric_and_binomial_kinds(self):
        rng = np.random.RandomState(1)
        xg = rng.geometric(0.25, size=3000)
        pg = conjugate_posterior(GeometricDistribution(0.5), xg)
        self.assertEqual(pg.kind, "geometric")
        self.assertAlmostEqual(pg.point_estimate().p, pg.mean()["p"], places=12)
        self.assertGreater(pg.mean()["p"], 0.2)
        self.assertLess(pg.mean()["p"], 0.3)
        xb = rng.binomial(10, 0.4, size=2000)
        pb = conjugate_posterior(BinomialDistribution(0.5, 10), xb)
        self.assertEqual(pb.kind, "binomial")
        self.assertAlmostEqual(pb.mean()["p"], 0.4, delta=0.02)

    def test_negative_binomial_rebuilds_negative_binomial_family(self):
        # Regression: the NegativeBinomial conjugate posterior must rebuild a
        # NegativeBinomial likelihood/predictive (not silently fall through to Bernoulli).
        from mixle.stats import NegativeBinomialDistribution

        rng = np.random.RandomState(7)
        r = 3.0
        x = rng.negative_binomial(r, 0.4, size=3000)
        post = conjugate_posterior(NegativeBinomialDistribution(r, 0.5), x)
        self.assertEqual(post.kind, "negative_binomial")
        # posterior over p is still correct
        self.assertAlmostEqual(post.mean()["p"], 0.4, delta=0.03)
        for built in (post.point_estimate(), post.posterior_predictive()):
            self.assertIsInstance(built, NegativeBinomialDistribution)
            self.assertNotIsInstance(built, BernoulliDistribution)
            self.assertEqual(built.r, r)
            self.assertAlmostEqual(built.p, post.mean()["p"], places=12)


class GammaConjugateTest(unittest.TestCase):
    def test_poisson_posterior_predictive_and_exact_evidence(self):
        rng = np.random.RandomState(0)
        x = rng.poisson(4.0, 3000)
        a0, b0 = 2.0, 1.5
        post = conjugate_posterior(PoissonDistribution(1.0), x, prior={"shape": a0, "rate": b0})
        self.assertAlmostEqual(post.mean()["rate"], (a0 + x.sum()) / (b0 + len(x)))
        self.assertAlmostEqual(post.mean()["rate"], x.mean(), delta=0.05)
        # predictive is Negative-Binomial
        from mixle.stats import NegativeBinomialDistribution

        self.assertIsInstance(post.posterior_predictive(), NegativeBinomialDistribution)
        # absolute marginal likelihood (includes -sum log x_i!) matches the analytic Gamma identity
        an, bn = a0 + x.sum(), b0 + len(x)
        expect = -float(np.sum(gammaln(x + 1.0))) + gammaln(an) - gammaln(a0) + a0 * math.log(b0) - an * math.log(bn)
        self.assertAlmostEqual(post.log_marginal_likelihood(), expect, places=6)

    def test_exponential_rate(self):
        rng = np.random.RandomState(2)
        x = rng.exponential(3.0, 3000)  # mean 3 -> rate 1/3
        post = conjugate_posterior(ExponentialDistribution(1.0), x)
        self.assertAlmostEqual(post.mean()["rate"], 1.0 / 3.0, delta=0.02)
        self.assertAlmostEqual(post.point_estimate().beta, 3.0, delta=0.2)  # beta is the mean


class GaussianConjugateTest(unittest.TestCase):
    def test_nig_posterior_recovers_moments(self):
        rng = np.random.RandomState(0)
        x = rng.normal(5.0, 2.0, 5000)
        post = conjugate_posterior(GaussianDistribution(0.0, 1.0), x)
        m = post.mean()
        self.assertAlmostEqual(m["mu"], x.mean(), delta=0.05)
        self.assertAlmostEqual(m["sigma2"], x.var(), delta=0.1)
        from mixle.stats import StudentTDistribution

        self.assertIsInstance(post.posterior_predictive(), StudentTDistribution)

    def test_nig_sampling_is_consistent(self):
        rng = np.random.RandomState(0)
        x = rng.normal(-1.0, 1.5, 4000)
        post = conjugate_posterior(GaussianDistribution(0.0, 1.0), x)
        s = post.sample(20000, np.random.RandomState(1))
        self.assertTrue(np.all(s["sigma2"] > 0.0))
        self.assertAlmostEqual(s["mu"].mean(), post.mean()["mu"], delta=0.02)
        self.assertAlmostEqual(s["sigma2"].mean(), post.mean()["sigma2"], delta=0.05)

    def test_exact_evidence_matches_sequential_product(self):
        # Marginal likelihood factorises: p(x1..xn) == p(x1) * p(x2|x1) * ... (predictive chain).
        rng = np.random.RandomState(3)
        x = rng.normal(2.0, 1.0, 6)
        prior = {"m": 0.0, "kappa": 1.0, "a": 2.0, "b": 1.0}
        full = conjugate_posterior(GaussianDistribution(0.0, 1.0), x, prior=prior).log_marginal_likelihood()
        chain = 0.0
        for i in range(len(x)):
            post_prev = conjugate_posterior(GaussianDistribution(0.0, 1.0), x[:i], prior=prior)
            pred = post_prev.posterior_predictive()
            chain += float(pred.log_density(float(x[i])))
        self.assertAlmostEqual(full, chain, places=8)


class DirichletConjugateTest(unittest.TestCase):
    def test_categorical_posterior(self):
        rng = np.random.RandomState(0)
        data = list(rng.choice(["a", "b", "c"], 3000, p=[0.2, 0.3, 0.5]))
        post = conjugate_posterior(CategoricalDistribution({"a": 0.3, "b": 0.3, "c": 0.4}), data)
        mp = post.mean()["map"]
        self.assertAlmostEqual(mp["a"], 0.2, delta=0.03)
        self.assertAlmostEqual(mp["c"], 0.5, delta=0.03)
        sm = post.sample(5000, np.random.RandomState(1))["probs"]
        self.assertTrue(np.allclose(sm.sum(axis=1), 1.0))

    def test_integer_categorical(self):
        rng = np.random.RandomState(0)
        data = list(rng.choice([0, 1, 2, 3], 3000, p=[0.1, 0.2, 0.3, 0.4]))
        post = conjugate_posterior(IntegerCategoricalDistribution(0, [0.25] * 4), data)
        probs = post.mean()["probs"]
        self.assertAlmostEqual(probs[3], 0.4, delta=0.03)


class MvnConjugateTest(unittest.TestCase):
    def test_niw_posterior_and_samples(self):
        rng = np.random.RandomState(0)
        mu = np.array([1.0, -2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        x = list(rng.multivariate_normal(mu, cov, 5000))
        post = conjugate_posterior(MultivariateGaussianDistribution(np.zeros(2), np.eye(2)), x)
        m = post.mean()
        self.assertTrue(np.allclose(m["mean"], mu, atol=0.1))
        self.assertTrue(np.allclose(m["cov"], cov, atol=0.15))
        sm = post.sample(200, np.random.RandomState(1))
        self.assertTrue(all(np.all(np.linalg.eigvalsh(c) > 0.0) for c in sm["cov"]))
        from mixle.stats import MultivariateStudentTDistribution

        self.assertIsInstance(post.posterior_predictive(), MultivariateStudentTDistribution)


class NewClosedFormFamiliesTest(unittest.TestCase):
    """Each newly-added family returns a closed-form full-Bayesian posterior (no generic formula)."""

    def _recovers(self, dist, key, truth, delta, transform=lambda v: v, n=8000):
        x = dist.sampler(seed=1).sample(n)
        post = conjugate_posterior(dist, list(x) if np.ndim(x[0]) else x)
        self.assertAlmostEqual(transform(post.mean()[key]), truth, delta=delta)
        # full-Bayesian surface is present and runs
        post.sample(50, np.random.RandomState(0))
        self.assertTrue(np.isfinite(post.log_marginal_likelihood()))
        post.posterior_predictive()
        return post

    def test_rayleigh(self):
        self._recovers(RayleighDistribution(2.0), "sigma2", 4.0, 0.4)  # E[sigma2]=4

    def test_half_normal(self):
        from mixle.stats import HalfNormalDistribution

        self._recovers(HalfNormalDistribution(1.5), "sigma2", 2.25, 0.3)

    def test_log_gaussian(self):
        from mixle.stats import LogGaussianDistribution

        self._recovers(LogGaussianDistribution(0.5, 0.4), "mu", 0.5, 0.05)

    def test_gamma_known_shape(self):
        from mixle.stats import GammaDistribution

        self._recovers(GammaDistribution(3.0, 2.0), "rate", 0.5, 0.05)  # rate = 1/theta

    def test_inverse_gamma_known_shape(self):
        from mixle.stats import InverseGammaDistribution

        self._recovers(InverseGammaDistribution(4.0, 3.0), "beta", 3.0, 0.3)

    def test_inverse_gaussian_known_mean(self):
        from mixle.stats import InverseGaussianDistribution

        self._recovers(InverseGaussianDistribution(1.5, 2.0), "lam", 2.0, 0.3)

    def test_pareto_known_scale(self):
        from mixle.stats import ParetoDistribution

        self._recovers(ParetoDistribution(1.0, 3.0), "alpha", 3.0, 0.2)

    def test_negative_binomial_known_r(self):
        from mixle.stats import NegativeBinomialDistribution

        self._recovers(NegativeBinomialDistribution(5.0, 0.4), "p", 0.4, 0.03)

    def test_von_mises_known_concentration(self):
        from mixle.stats import VonMisesDistribution

        self._recovers(VonMisesDistribution(0.7, 3.0), "mu", 0.7, 0.05)

    def test_diagonal_gaussian(self):
        from mixle.stats import DiagonalGaussianDistribution

        d = DiagonalGaussianDistribution([1.0, -2.0], [2.0, 0.5])
        post = conjugate_posterior(d, list(d.sampler(seed=1).sample(8000)))
        self.assertTrue(np.allclose(post.mean()["mu"], [1.0, -2.0], atol=0.1))
        self.assertTrue(np.allclose(post.mean()["sigma2"], [2.0, 0.5], atol=0.15))
        self.assertTrue(np.isfinite(post.log_marginal_likelihood()))

    def test_rayleigh_evidence_matches_numerical(self):
        from math import lgamma

        x = RayleighDistribution(2.0).sampler(seed=3).sample(6)
        a0, b0 = 2.0, 2.0
        post = conjugate_posterior(RayleighDistribution(1.0), x, prior={"a": a0, "b": b0})
        g = np.linspace(1e-3, 80, 400000)
        ig = (b0**a0 / np.exp(lgamma(a0))) * g ** (-a0 - 1) * np.exp(-b0 / g)
        lik = np.prod([xi / g * np.exp(-(xi**2) / (2 * g)) for xi in x], axis=0)
        num = np.log(np.trapezoid(lik * ig, g))
        self.assertAlmostEqual(post.log_marginal_likelihood(), num, places=3)


class UnsupportedFamiliesTest(unittest.TestCase):
    def test_no_closed_form_conjugate_raises(self):
        from mixle.stats import BetaDistribution, MixtureDistribution

        with self.assertRaises(TypeError):  # full Beta: no closed-form conjugate
            conjugate_posterior(BetaDistribution(2.0, 2.0), [0.3, 0.5, 0.7])
        with self.assertRaises(TypeError):  # structured: not conjugate at all
            conjugate_posterior(
                MixtureDistribution([GaussianDistribution(0, 1), GaussianDistribution(5, 1)], [0.5, 0.5]), [0.1, 5.2]
            )


class WeightedTest(unittest.TestCase):
    def test_weights_match_replication(self):
        # integer weights must equal physically replicating the observations
        x = np.array([0.0, 1.0, 2.0, 3.0])
        w = np.array([1.0, 2.0, 3.0, 1.0])
        rep = np.repeat(x, w.astype(int))
        pw = conjugate_posterior(GaussianDistribution(0.0, 1.0), x, weights=w)
        pr = conjugate_posterior(GaussianDistribution(0.0, 1.0), rep)
        self.assertAlmostEqual(pw.mean()["mu"], pr.mean()["mu"], places=9)
        self.assertAlmostEqual(pw.mean()["sigma2"], pr.mean()["sigma2"], places=9)


class MixtureOfConjugatesTest(unittest.TestCase):
    def test_matches_numerical_posterior_exactly(self):
        # bimodal Beta prior; the closed-form mixture-of-conjugates posterior must equal the
        # grid-integrated posterior (mean, log-evidence, and the whole density).
        from scipy.stats import beta as B

        rng = np.random.RandomState(0)
        priors = [{"a": 12.0, "b": 3.0}, {"a": 3.0, "b": 12.0}]
        x = (rng.rand(15) < 0.75).astype(int)
        post = mixture_conjugate_posterior(BernoulliDistribution(0.5), x, priors, prior_weights=[0.5, 0.5])

        grid = np.linspace(1e-6, 1 - 1e-6, 400001)
        prior_pdf = 0.5 * B.pdf(grid, 12, 3) + 0.5 * B.pdf(grid, 3, 12)
        loglik = x.sum() * np.log(grid) + (len(x) - x.sum()) * np.log(1 - grid)
        un = np.exp(loglik - loglik.max()) * prior_pdf
        z = np.trapezoid(un, grid)
        num_mean = np.trapezoid(grid * un, grid) / z
        self.assertAlmostEqual(post.mean()["p"], num_mean, places=5)

        ml = np.trapezoid(np.exp(loglik) * prior_pdf, grid)
        self.assertAlmostEqual(post.log_marginal_likelihood(), np.log(ml), places=4)

        mix_pdf = sum(w * B.pdf(grid, c.a, c.b) for w, c in zip(post.weights, post.components))
        self.assertLess(np.max(np.abs(mix_pdf - un / z)), 1e-9)

    def test_weights_reweighted_by_evidence(self):
        # data strongly favouring high p must upweight the high-p prior component
        rng = np.random.RandomState(1)
        x = (rng.rand(40) < 0.8).astype(int)
        post = mixture_conjugate_posterior(
            BernoulliDistribution(0.5), x, [{"a": 20.0, "b": 2.0}, {"a": 2.0, "b": 20.0}], prior_weights=[0.5, 0.5]
        )
        self.assertGreater(post.weights[0], 0.95)
        # the predictive is a proper mixture, and sampling reproduces the posterior mean
        from mixle.stats import MixtureDistribution

        self.assertIsInstance(post.posterior_predictive(), MixtureDistribution)
        s = post.sample(50000, np.random.RandomState(2))["p"]
        self.assertAlmostEqual(s.mean(), post.mean()["p"], delta=0.01)

    def test_requires_closed_form_family(self):
        # full Beta has no closed-form conjugate (no evidence) -> cannot form a mixture-of-conjugates
        with self.assertRaises(TypeError):
            from mixle.stats import BetaDistribution

            mixture_conjugate_posterior(BetaDistribution(2.0, 2.0), [0.2, 0.4, 0.6], [{"a": 1.0}, {"a": 2.0}])


class ConjugateSamplerApiTest(unittest.TestCase):
    """The posterior follows the standard obj.sampler(seed).sample(size) convention."""

    def test_sampler_single_and_batch(self):
        post = conjugate_posterior(BernoulliDistribution(0.5), [1, 0, 1, 1, 0, 1, 1])
        single = post.sampler(seed=0).sample()
        self.assertTrue(np.isscalar(single["p"]) or np.ndim(single["p"]) == 0)  # one parameter set
        batch = post.sampler(seed=0).sample(5)
        self.assertEqual(batch["p"].shape, (5,))

    def test_sampler_is_seed_repeatable(self):
        post = conjugate_posterior(BernoulliDistribution(0.5), [1, 0, 1, 1, 0])
        a = post.sampler(seed=3).sample(10)["p"]
        b = post.sampler(seed=3).sample(10)["p"]
        self.assertTrue(np.array_equal(a, b))

    def test_legacy_sample_n_rng_still_works(self):
        post = conjugate_posterior(GaussianDistribution(0, 1), list(np.random.RandomState(0).randn(50)))
        draws = post.sample(3, rng=np.random.RandomState(1))
        self.assertIn("mu", draws)
        self.assertEqual(np.asarray(draws["mu"]).shape[0], 3)


class ConjugateInputValidationTest(unittest.TestCase):
    """An external review found conjugate_posterior accepted impossible/invalid observations instead
    of rejecting them: Bernoulli data outside {0, 1}, Binomial successes outside [0, n], a fractional
    or out-of-range IntegerCategorical observation, NaN/Inf observations, negative/NaN weights, and a
    non-positive or non-finite Beta prior all silently built a garbage posterior. ``_as_weighted_array``
    (data/weight finiteness) and ``_beta_prior`` (Beta hyperparameter validity) are the shared choke
    points for every family routed through them; Bernoulli/Binomial/IntegerCategorical additionally
    need their own domain checks since valid-observation constraints differ per family.
    """

    def test_bernoulli_rejects_value_outside_zero_one(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 2])

    def test_binomial_rejects_successes_exceeding_n(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BinomialDistribution(0.5, 3), [4])

    def test_binomial_rejects_negative_successes(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BinomialDistribution(0.5, 3), [-1])

    def test_binomial_rejects_fractional_successes(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BinomialDistribution(0.5, 3), [1.5])

    def test_binomial_boundary_successes_still_construct(self):
        post = conjugate_posterior(BinomialDistribution(0.5, 3), [0, 3])
        self.assertTrue(math.isfinite(post.a) and math.isfinite(post.b))

    def test_integer_categorical_rejects_fractional_observation(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(IntegerCategoricalDistribution(0, [0.25] * 4), [1.9, 1.9, 1.9])

    def test_integer_categorical_rejects_out_of_range_category(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(IntegerCategoricalDistribution(0, [0.25] * 4), [0, 1, 99])

    def test_rejects_nan_observation(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0.0, 1.0, float("nan")])

    def test_rejects_infinite_observation(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(GaussianDistribution(0.0, 1.0), [0.0, 1.0, float("inf")])

    def test_rejects_negative_weights(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], weights=np.array([1.0, -5.0, 1.0]))

    def test_rejects_nan_weights(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], weights=np.array([1.0, float("nan"), 1.0]))

    def test_zero_weight_boundary_still_constructs(self):
        post = conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], weights=np.array([0.0, 1.0, 1.0]))
        self.assertTrue(math.isfinite(post.a) and math.isfinite(post.b))

    def test_rejects_non_positive_beta_prior_a(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], prior={"a": -1.0, "b": 1.0})

    def test_rejects_non_positive_beta_prior_b(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], prior={"a": 1.0, "b": 0.0})

    def test_rejects_zero_beta_prior_a(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], prior={"a": 0.0, "b": 1.0})

    def test_rejects_nan_beta_prior(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], prior={"a": float("nan"), "b": 1.0})

    def test_rejects_infinite_beta_prior(self):
        with self.assertRaises(ValueError):
            conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1], prior={"a": float("inf"), "b": 1.0})

    def test_invalid_beta_prior_rejected_for_geometric_and_negative_binomial_too(self):
        # _beta_prior is the shared choke point for every Beta-posterior family, not just Bernoulli.
        from mixle.stats import NegativeBinomialDistribution

        with self.assertRaises(ValueError):
            conjugate_posterior(GeometricDistribution(0.5), [1, 2, 3], prior={"a": -1.0, "b": 1.0})
        with self.assertRaises(ValueError):
            conjugate_posterior(NegativeBinomialDistribution(3.0, 0.5), [1, 2, 3], prior={"a": -1.0, "b": 1.0})

    def test_valid_bernoulli_data_still_constructs(self):
        post = conjugate_posterior(BernoulliDistribution(0.5), [0, 1, 1, 0, 1])
        self.assertAlmostEqual(post.a, 1.0 + 3.0)
        self.assertAlmostEqual(post.b, 1.0 + 2.0)

    def test_valid_bernoulli_bool_data_still_constructs(self):
        post = conjugate_posterior(BernoulliDistribution(0.5), [True, False, True])
        self.assertAlmostEqual(post.a, 1.0 + 2.0)
        self.assertAlmostEqual(post.b, 1.0 + 1.0)

    def test_valid_binomial_data_still_constructs(self):
        post = conjugate_posterior(BinomialDistribution(0.5, 3), [0, 1, 2, 3])
        self.assertAlmostEqual(post.a, 1.0 + 6.0)
        self.assertAlmostEqual(post.b, 1.0 + 6.0)

    def test_valid_integer_categorical_data_still_constructs(self):
        post = conjugate_posterior(IntegerCategoricalDistribution(0, [0.25] * 4), [0, 1, 2, 3])
        self.assertTrue(np.allclose(post.alpha, [2.0, 2.0, 2.0, 2.0]))


class BetaBinomialExactPredictiveTest(unittest.TestCase):
    """An external review found BetaPosterior.posterior_predictive() mislabels a plug-in-at-the-mean
    approximation as the true Bayesian predictive for a Binomial likelihood with n > 1: the Beta-
    Bernoulli identity (plug-in-at-mean == true predictive) only holds because the Bernoulli mass is
    linear in p, but the Binomial mass is nonlinear in p for n > 1, so the plug-in is biased. Fixed by
    returning the closed-form Beta-Binomial(n, a, b) predictive (mixle.stats.BetaBinomialDistribution,
    matching the existing Poisson-Gamma -> NegativeBinomial closed-form predictive precedent) whenever
    the posterior was built from a Binomial with n > 1, while leaving n == 1 (Bernoulli, and
    Binomial(n=1)) on the still-exact plug-in path.
    """

    def test_binomial_predictive_matches_closed_form_beta_binomial(self):
        # Uniform Beta(1,1) prior, no data -> posterior stays Beta(1,1); the true Beta-Binomial(n=2,
        # a=1, b=1) predictive is uniform over {0,1,2} (each 1/3) -- NOT the plug-in Binomial(2, 0.5)
        # value of 1/4 at k=0 that the old code returned.
        from mixle.stats import BetaBinomialDistribution

        post = conjugate_posterior(BinomialDistribution(0.5, 2), [], prior={"a": 1.0, "b": 1.0})
        pred = post.posterior_predictive()
        self.assertIsInstance(pred, BetaBinomialDistribution)
        for k in range(3):
            self.assertAlmostEqual(pred.density(k), 1.0 / 3.0, places=12)

    def test_binomial_predictive_matches_scipy_betabinom(self):
        from scipy.stats import betabinom

        rng = np.random.RandomState(0)
        x = rng.binomial(5, 0.3, size=200)
        post = conjugate_posterior(BinomialDistribution(0.5, 5), x, prior={"a": 2.0, "b": 3.0})
        pred = post.posterior_predictive()
        for k in range(6):
            self.assertAlmostEqual(pred.density(k), betabinom.pmf(k, 5, post.a, post.b), places=10)

    def test_binomial_n1_predictive_unaffected_still_exact_plugin(self):
        # n_trials == 1 is mathematically identical to Bernoulli: plug-in-at-mean IS the exact
        # predictive there, so this path must stay untouched by the n > 1 fix.
        post = conjugate_posterior(BinomialDistribution(0.5, 1), [1, 0, 1, 1])
        pred = post.posterior_predictive()
        self.assertIsInstance(pred, BinomialDistribution)
        self.assertAlmostEqual(pred.p, post.mean()["p"], places=12)

    def test_bernoulli_predictive_unaffected(self):
        post = conjugate_posterior(BernoulliDistribution(0.5), [1, 0, 1, 1])
        pred = post.posterior_predictive()
        self.assertIsInstance(pred, BernoulliDistribution)
        self.assertAlmostEqual(pred.p, post.mean()["p"], places=12)

    def test_negative_binomial_predictive_unaffected(self):
        # negative_binomial reuses BetaPosterior.n_trials to store the known r parameter (not a trial
        # count) -- it must not be misread as a Binomial trial count by the new n > 1 branch.
        from mixle.stats import NegativeBinomialDistribution

        post = conjugate_posterior(NegativeBinomialDistribution(3.0, 0.5), [1, 2, 3, 0])
        pred = post.posterior_predictive()
        self.assertIsInstance(pred, NegativeBinomialDistribution)


if __name__ == "__main__":
    unittest.main()
