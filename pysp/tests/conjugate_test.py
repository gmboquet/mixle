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


class GenericFallbackTest(unittest.TestCase):
    def test_generic_mean_parameters_equal_empirical_suffstat(self):
        from pysp.stats.exp_family import to_exponential_family

        rng = np.random.RandomState(0)
        x = np.abs(rng.normal(0.0, 3.0, 1000)) + 0.1
        post = conjugate_posterior(RayleighDistribution(1.0), x, prior={"nu": 0.0})
        form = to_exponential_family(RayleighDistribution(1.0))
        emp = np.asarray(form.engine.to_numpy(form.sufficient_statistics(list(x)))).mean(axis=0)
        self.assertTrue(np.allclose(post.mean_parameters(), emp))
        self.assertIsInstance(post.point_estimate(), RayleighDistribution)


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


if __name__ == "__main__":
    unittest.main()
