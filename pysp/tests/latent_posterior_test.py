"""LatentPosterior: q(z|x) as a first-class object -- the mixture (exact categorical) realization."""

import itertools
import unittest

import numpy as np
from scipy.stats import dirichlet

from pysp.stats import (
    CategoricalDistribution,
    CategoricalLatentPosterior,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    LatentPosterior,
    LDADistribution,
    MarkovChainLatentPosterior,
    MeanFieldLDAPosterior,
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


class HmmChainLatentPosteriorTest(unittest.TestCase):
    def setUp(self):
        self.m = HiddenMarkovModelDistribution(
            [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0), GaussianDistribution(6.0, 1.0)],
            [0.5, 0.3, 0.2],
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
            len_dist=CategoricalDistribution({5: 1.0}),
        )
        self.x = [-1.8, 2.1, 2.3, 5.9, -2.0]
        self.q = self.m.latent_posterior(self.x)

    def _brute_force(self):
        t, k = len(self.x), 3
        logpi, log_a = self.m.log_w, self.m.log_transitions
        logb = np.array([[self.m.topics[j].log_density(xt) for j in range(k)] for xt in self.x])
        paths = list(itertools.product(range(k), repeat=t))
        logw = np.array(
            [
                logpi[p[0]] + logb[0, p[0]] + sum(log_a[p[i], p[i + 1]] + logb[i + 1, p[i + 1]] for i in range(t - 1))
                for p in paths
            ]
        )
        w = np.exp(logw - logw.max())
        w /= w.sum()
        gamma = np.zeros((t, k))
        for p, wp in zip(paths, w):
            for i in range(t):
                gamma[i, p[i]] += wp
        return gamma, paths[int(np.argmax(w))], float(-np.sum(w * np.log(w)))

    def test_chain_posterior_is_latent_posterior(self):
        self.assertIsInstance(self.q, MarkovChainLatentPosterior)
        self.assertIsInstance(self.q, LatentPosterior)

    def test_marginals_mode_entropy_match_brute_force(self):
        gamma, mode, entropy = self._brute_force()
        np.testing.assert_allclose(self.q.marginals(), gamma, atol=1e-9)
        self.assertEqual(tuple(self.q.mode()), mode)
        self.assertTrue(np.array_equal(self.q.mode(), self.m.viterbi(self.x)))
        self.assertAlmostEqual(self.q.entropy(), entropy, places=9)

    def test_ffbs_sample_average_matches_marginals(self):
        s = np.array([self.q.sample(rng=i) for i in range(4000)])
        emp = np.array([[np.mean(s[:, t] == k) for k in range(3)] for t in range(len(self.x))])
        np.testing.assert_allclose(emp, self.q.marginals(), atol=0.03)
        self.assertTrue(np.array_equal(self.q.sample(rng=3), self.q.sample(rng=3)))  # repeatable


class LDAMeanFieldPosteriorTest(unittest.TestCase):
    def setUp(self):
        topics = [
            CategoricalDistribution({0: 0.45, 1: 0.45, 2: 0.05, 3: 0.05}),
            CategoricalDistribution({0: 0.05, 1: 0.05, 2: 0.45, 3: 0.45}),
        ]
        self.m = LDADistribution(topics, [0.1, 0.1])
        self.doc = [(0, 5), (1, 4), (2, 1)]  # mostly topic-0 words
        self.q = self.m.latent_posterior(self.doc)

    def test_is_latent_posterior(self):
        self.assertIsInstance(self.q, MeanFieldLDAPosterior)
        self.assertIsInstance(self.q, LatentPosterior)

    def test_topic_proportions_match_model_seq_posterior(self):
        enc = self.m.dist_to_encoder().seq_encode([self.doc])
        np.testing.assert_allclose(self.q.topic_proportions(), self.m.seq_posterior(enc)[0], atol=1e-9)
        self.assertGreater(self.q.topic_proportions()[0], 0.9)  # topic-0-dominated document

    def test_phi_rows_normalized(self):
        phi = self.q.marginals()
        self.assertEqual(phi.shape, (3, 2))
        np.testing.assert_allclose(phi.sum(axis=1), 1.0)

    def test_entropy_decomposes_into_dirichlet_plus_categorical(self):
        phi = self.q.marginals()
        h_z = -float(np.sum(self.q.counts[:, None] * np.where(phi > 0, phi * np.log(phi), 0.0)))
        expected = dirichlet(self.q.gamma).entropy() + h_z  # differential H[q(theta)] may be negative
        self.assertAlmostEqual(self.q.entropy(), expected, places=9)

    def test_sample_returns_theta_and_per_token_topics(self):
        theta, z = self.q.sample(rng=0)
        self.assertAlmostEqual(float(theta.sum()), 1.0)
        self.assertEqual(len(z), 10)  # total token count 5+4+1
        self.assertTrue(set(z.tolist()) <= {0, 1})
        self.assertTrue(np.array_equal(self.q.sample(rng=1)[1], self.q.sample(rng=1)[1]))  # repeatable


if __name__ == "__main__":
    unittest.main()
