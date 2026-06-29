"""The Posterior algebra: the posterior() factory over latent / parameter / predictive variables."""

import unittest

import numpy as np

from mixle.inference import ParameterPosterior, PredictivePosterior, posterior
from mixle.stats import sample
from mixle.stats.compute.posterior import LatentPosterior, Posterior
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution


class PredictivePosteriorTest(unittest.TestCase):
    def test_plug_in_predictive_matches_sampler(self):
        g = GaussianDistribution(2.0, 1.0)
        post = posterior(g, over="predictive")
        self.assertIsInstance(post, PredictivePosterior)
        self.assertIsInstance(post, Posterior)
        draws = np.asarray(post.samples(5000, rng=np.random.RandomState(0)), dtype=float)
        self.assertAlmostEqual(draws.mean(), 2.0, delta=0.1)  # recovers the model mean
        self.assertTrue(np.isscalar(post.sample(rng=np.random.RandomState(1))) or np.ndim(post.sample(0)) == 0)


class ParameterPosteriorConjugateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = (rng.random_sample(400) < 0.7).astype(int).tolist()  # Bernoulli(0.7)
        self.post = posterior(BernoulliDistribution(0.5), self.data, over="params", prior={"a": 1.0, "b": 1.0})

    def test_factory_returns_parameter_posterior(self):
        self.assertIsInstance(self.post, ParameterPosterior)
        self.assertEqual(self.post.kind, "conjugate")

    def test_mean_recovers_parameter(self):
        m = self.post.mean()
        # Beta posterior mean of p concentrates near the data frequency (~0.7)
        p = float(np.ravel(list(m.values())[0])[0]) if isinstance(m, dict) else float(np.ravel(m)[0])
        self.assertAlmostEqual(p, 0.7, delta=0.06)

    def test_sample_and_interval(self):
        one = self.post.sample(rng=np.random.RandomState(1))
        self.assertIsInstance(one, dict)  # a parameter set
        ci = self.post.interval(0.9)
        self.assertIsInstance(ci, dict)
        for lo_hi in ci.values():
            self.assertEqual(np.asarray(lo_hi).shape[0], 2)  # [lo, hi]

    def test_auto_picks_conjugate(self):
        self.assertEqual(posterior(BernoulliDistribution(0.5), self.data, over="params").kind, "conjugate")


class ParameterPosteriorMCMCTest(unittest.TestCase):
    def test_mcmc_path_when_forced(self):
        rng = np.random.RandomState(0)
        data = (rng.normal(3.0, 1.0, size=200)).tolist()
        post = posterior(
            GaussianDistribution(0.0, 1.0), data, over="params", method="mcmc", steps=300, burn_in=100, seed=0
        )
        self.assertIsInstance(post, ParameterPosterior)
        self.assertEqual(post.kind, "mcmc")
        draws = post.samples(10, rng=np.random.RandomState(1))
        self.assertEqual(len(draws), 10)
        mean = np.asarray(post.mean(), dtype=float)
        self.assertAlmostEqual(float(mean[0]), 3.0, delta=0.5)  # posterior mean of mu near 3


class LatentPosteriorFactoryTest(unittest.TestCase):
    def test_over_latent_dispatches_to_model(self):
        m = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        x = [-3.0, -2.8, 3.1, 2.9]
        post = posterior(m, x, over="latent")
        self.assertIsInstance(post, LatentPosterior)
        self.assertIsInstance(post, Posterior)
        labels = post.mode()
        self.assertEqual(list(labels), [0, 0, 1, 1])  # clear assignment to the two components

    def test_unsupported_over_raises(self):
        with self.assertRaises(ValueError):
            posterior(GaussianDistribution(0.0, 1.0), over="nonsense")
        with self.assertRaises(TypeError):
            posterior(GaussianDistribution(0.0, 1.0), [1.0], over="latent")  # Gaussian has no latent_posterior


class FacadeStillWorksTest(unittest.TestCase):
    def test_sample_facade_unchanged(self):
        g = GaussianDistribution(0.0, 1.0)
        draws = sample(g, 100, seed=0)
        self.assertEqual(len(np.asarray(draws)), 100)


if __name__ == "__main__":
    unittest.main()
