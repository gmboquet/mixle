"""mixle.stats.sample: one entry point dispatching across every samplable mixle object (Phase C)."""

import unittest

import numpy as np

from mixle.relations import Assignment
from mixle.stats import BernoulliDistribution, GaussianDistribution, MixtureDistribution, sample
from mixle.stats.bayes.conjugate import conjugate_posterior


class SamplingApiTest(unittest.TestCase):
    def test_distribution_size_rules(self):
        g = GaussianDistribution(0.0, 1.0)
        arr = sample(g, 1000, seed=0)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (1000,))
        self.assertEqual(np.ndim(sample(g, seed=1)), 0)  # size=None -> scalar

    def test_shared_rng_is_reproducible_and_advances(self):
        g = GaussianDistribution(0.0, 1.0)
        r1, r2 = np.random.RandomState(42), np.random.RandomState(42)
        a = [sample(g, 3, rng=r1) for _ in range(3)]
        b = [sample(g, 3, rng=r2) for _ in range(3)]
        self.assertTrue(all(np.allclose(x, y) for x, y in zip(a, b)))  # same rng seed -> same streams
        self.assertFalse(np.allclose(a[0], a[1]))  # the shared rng advances between calls

    def test_seed_matches_direct_sampler(self):
        g = GaussianDistribution(1.0, 2.0)
        np.testing.assert_array_equal(sample(g, 10, seed=5), g.sampler(seed=5).sample(10))

    def test_conjugate_posterior_parameter_draws(self):
        post = conjugate_posterior(BernoulliDistribution(0.5), [1, 0, 1, 1, 0, 1], prior={"a": 1.0, "b": 1.0})
        one = sample(post, seed=0)
        self.assertIsInstance(one, dict)
        self.assertEqual(np.ndim(one["p"]), 0)  # single parameter set
        many = sample(post, 100, seed=0)
        self.assertEqual(np.shape(many["p"]), (100,))

    def test_relation_member_draws(self):
        rel = Assignment(np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]))
        draws = sample(rel, 5, seed=0, temperature=1.0)  # kwargs forwarded
        self.assertEqual(len(draws), 5)
        self.assertEqual(tuple(sample(rel, temperature=0.0)), tuple(rel.solve().value))

    def test_latent_posterior_draws(self):
        mix = MixtureDistribution([GaussianDistribution(-3, 1.0), GaussianDistribution(3, 1.0)], [0.5, 0.5])
        q = mix.latent_posterior([-3.0, 3.0, -3.0])
        self.assertEqual(np.shape(sample(q, seed=0)), (3,))  # single latent assignment over 3 points
        self.assertIsInstance(sample(q, 4, seed=0), list)  # a collection of draws

    def test_unknown_object_raises(self):
        with self.assertRaises(TypeError):
            sample(42)


if __name__ == "__main__":
    unittest.main()
