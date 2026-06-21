"""Conjugate-posterior inference derived from the exponential-family map (pysp.stats.conjugate)."""

import math
import unittest

import numpy as np
from scipy.special import betaln, gammaln

from pysp.stats import (
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


class GammaConjugateTest(unittest.TestCase):
    def test_poisson_posterior_predictive_and_exact_evidence(self):
        rng = np.random.RandomState(0)
        x = rng.poisson(4.0, 3000)
        a0, b0 = 2.0, 1.5
        post = conjugate_posterior(PoissonDistribution(1.0), x, prior={"shape": a0, "rate": b0})
        self.assertAlmostEqual(post.mean()["rate"], (a0 + x.sum()) / (b0 + len(x)))
        self.assertAlmostEqual(post.mean()["rate"], x.mean(), delta=0.05)
        # predictive is Negative-Binomial
        from pysp.stats import NegativeBinomialDistribution

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
        from pysp.stats import StudentTDistribution

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
        from pysp.stats import MultivariateStudentTDistribution

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
        from pysp.stats import HalfNormalDistribution

        self._recovers(HalfNormalDistribution(1.5), "sigma2", 2.25, 0.3)

    def test_log_gaussian(self):
        from pysp.stats import LogGaussianDistribution

        self._recovers(LogGaussianDistribution(0.5, 0.4), "mu", 0.5, 0.05)

    def test_gamma_known_shape(self):
        from pysp.stats import GammaDistribution

        self._recovers(GammaDistribution(3.0, 2.0), "rate", 0.5, 0.05)  # rate = 1/theta

    def test_inverse_gamma_known_shape(self):
        from pysp.stats import InverseGammaDistribution

        self._recovers(InverseGammaDistribution(4.0, 3.0), "beta", 3.0, 0.3)

    def test_inverse_gaussian_known_mean(self):
        from pysp.stats import InverseGaussianDistribution

        self._recovers(InverseGaussianDistribution(1.5, 2.0), "lam", 2.0, 0.3)

    def test_pareto_known_scale(self):
        from pysp.stats import ParetoDistribution

        self._recovers(ParetoDistribution(1.0, 3.0), "alpha", 3.0, 0.2)

    def test_negative_binomial_known_r(self):
        from pysp.stats import NegativeBinomialDistribution

        self._recovers(NegativeBinomialDistribution(5.0, 0.4), "p", 0.4, 0.03)

    def test_von_mises_known_concentration(self):
        from pysp.stats import VonMisesDistribution

        self._recovers(VonMisesDistribution(0.7, 3.0), "mu", 0.7, 0.05)

    def test_diagonal_gaussian(self):
        from pysp.stats import DiagonalGaussianDistribution

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
        from pysp.stats import BetaDistribution, MixtureDistribution

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
        from pysp.stats import MixtureDistribution

        self.assertIsInstance(post.posterior_predictive(), MixtureDistribution)
        s = post.sample(50000, np.random.RandomState(2))["p"]
        self.assertAlmostEqual(s.mean(), post.mean()["p"], delta=0.01)

    def test_requires_closed_form_family(self):
        # full Beta has no closed-form conjugate (no evidence) -> cannot form a mixture-of-conjugates
        with self.assertRaises(TypeError):
            from pysp.stats import BetaDistribution

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


if __name__ == "__main__":
    unittest.main()
