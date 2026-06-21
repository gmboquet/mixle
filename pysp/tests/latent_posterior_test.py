"""LatentPosterior: q(z|x) as a first-class object -- the mixture (exact categorical) realization."""

import unittest

import numpy as np

from pysp.stats import (
    CategoricalLatentPosterior,
    GaussianDistribution,
    LatentPosterior,
    MixtureDistribution,
)


class MixtureLatentPosteriorTest(unittest.TestCase):
    def setUp(self):
        self.m = MixtureDistribution([GaussianDistribution(-5.0, 1.0), GaussianDistribution(5.0, 1.0)], [0.5, 0.5])
        self.x = [-5.1, -4.8, 5.2, 4.9, -5.0, 5.1]
        self.true = [0, 0, 1, 1, 0, 1]

    def test_is_latent_posterior_with_marginals(self):
        q = self.m.latent_posterior(self.x)
        self.assertIsInstance(q, LatentPosterior)
        self.assertIsInstance(q, CategoricalLatentPosterior)
        r = q.marginals()
        self.assertEqual(r.shape, (6, 2))
        np.testing.assert_allclose(r.sum(axis=1), 1.0)

    def test_mode_recovers_well_separated_components(self):
        self.assertEqual(list(self.m.latent_posterior(self.x).mode()), self.true)

    def test_sampling_recovers_truth_and_is_repeatable(self):
        q = self.m.latent_posterior(self.x)
        self.assertTrue(np.array_equal(q.sample(rng=0), self.true))  # well-separated -> certain
        self.assertTrue(np.array_equal(q.sample(rng=7), q.sample(rng=7)))  # seed-repeatable

    def test_entropy_zero_when_confident_positive_when_ambiguous(self):
        confident = self.m.latent_posterior(self.x).entropy()
        np.testing.assert_allclose(confident, 0.0, atol=1e-6)
        ambiguous = self.m.latent_posterior([0.0]).entropy()  # equidistant from both means
        self.assertGreater(ambiguous[0], 0.6)  # near log(2) for a 50/50 split

    def test_categorical_posterior_direct(self):
        r = np.array([[0.7, 0.3], [0.1, 0.9]])
        q = CategoricalLatentPosterior(r, support=["a", "b"])
        self.assertEqual(list(q.mode()), ["a", "b"])
        draws = np.array([q.sample(rng=i)[0] for i in range(400)])
        self.assertAlmostEqual(np.mean(draws == "a"), 0.7, delta=0.06)  # row 0 ~ 70% 'a'


if __name__ == "__main__":
    unittest.main()
