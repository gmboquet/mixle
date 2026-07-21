"""LatentPosterior: q(z|x) as a first-class object -- the mixture (exact categorical) realization."""

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import dirichlet

from mixle.stats import (
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
from mixle.utils.optional_deps import HAS_PANDAS
from mixle.utils.optional_deps import pandas as pd

_SKIP_NO_PANDAS = unittest.skipUnless(HAS_PANDAS, "pandas not installed; pip install mixle[pandas]")


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

    def test_posterior_predictive_conditions_on_observation(self):
        x = [-5.0] * 100 + [5.0] * 100
        pp = np.array(self.m.posterior_predictive(x, seed=0))
        self.assertEqual(len(pp), 200)
        self.assertLess(pp[:100].mean(), -3.0)  # points near component 0 predict near -5
        self.assertGreater(pp[100:].mean(), 3.0)  # points near component 1 predict near +5
        self.assertTrue(np.array_equal(pp, np.array(self.m.posterior_predictive(x, seed=0))))  # repeatable
        amb = np.array(self.m.posterior_predictive([0.0] * 2000, seed=1))  # ambiguous -> ~50/50
        self.assertAlmostEqual(np.mean(amb < 0), 0.5, delta=0.07)

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

    def test_posterior_predictive_follows_inferred_states(self):
        m = HiddenMarkovModelDistribution(
            [GaussianDistribution(-5.0, 0.5), GaussianDistribution(5.0, 0.5)],
            [0.5, 0.5],
            [[0.9, 0.1], [0.1, 0.9]],
            len_dist=CategoricalDistribution({6: 1.0}),
        )
        x = [-5.0, -5.0, 5.0, 5.0, -5.0, 5.0]
        pp = np.array(m.posterior_predictive(x, seed=0))
        self.assertEqual(len(pp), 6)
        self.assertTrue(np.array_equal(np.sign(pp), np.sign(x)))  # well-separated: pattern is recovered
        self.assertTrue(np.array_equal(pp, np.array(m.posterior_predictive(x, seed=0))))  # repeatable

    def test_ffbs_sample_average_matches_marginals(self):
        s = np.array([self.q.sample(rng=i) for i in range(4000)])
        emp = np.array([[np.mean(s[:, t] == k) for k in range(3)] for t in range(len(self.x))])
        np.testing.assert_allclose(emp, self.q.marginals(), atol=0.03)
        self.assertTrue(np.array_equal(self.q.sample(rng=3), self.q.sample(rng=3)))  # repeatable

    @_SKIP_NO_PANDAS
    def test_to_dataframe_matches_brute_force_marginals_and_mode(self):
        gamma, mode, _entropy = self._brute_force()
        df = self.q.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["t", "state", "state_0_prob", "state_1_prob", "state_2_prob"])
        self.assertEqual(df.shape, (len(self.x), 5))
        np.testing.assert_array_equal(df["t"].to_numpy(), np.arange(len(self.x)))
        np.testing.assert_array_equal(df["state"].to_numpy(), np.array(mode))
        np.testing.assert_array_equal(df["state"].to_numpy(), self.q.mode())  # matches the class's own mode()
        prob_cols = df[["state_0_prob", "state_1_prob", "state_2_prob"]].to_numpy()
        np.testing.assert_allclose(prob_cols, gamma, atol=1e-9)
        np.testing.assert_allclose(prob_cols.sum(axis=1), 1.0)  # each row is a proper distribution

    @_SKIP_NO_PANDAS
    def test_to_parquet_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "hmm_latent_posterior.parquet"
            self.q.to_parquet(path)
            roundtrip = pd.read_parquet(path)
            pd.testing.assert_frame_equal(roundtrip, self.q.to_dataframe())


@_SKIP_NO_PANDAS
class MarkovChainLatentPosteriorToDataFrameHandComputedTest(unittest.TestCase):
    """A from-scratch, hand-derived check independent of ``HiddenMarkovModelDistribution``/brute force.

    2 states, 2 steps, uniform prior and uniform-row transitions (``A[j, k] = 0.5`` for every ``j, k``):
    the chain carries no information across time (the prior over ``(z_0, z_1)`` is exactly uniform and
    independent), so the joint posterior factorizes into independent per-step terms and both the
    smoothing marginals and the Viterbi path collapse to each step's own row-normalized emission
    likelihood -- ``b[t] / sum(b[t])`` -- worked out below by hand, not read off the implementation.
    """

    def setUp(self):
        self.log_pi = np.log([0.5, 0.5])
        self.log_A = np.log([[0.5, 0.5], [0.5, 0.5]])
        self.b = np.array([[0.8, 0.2], [0.3, 0.7]])  # t=0 favors state 0, t=1 favors state 1
        self.q = MarkovChainLatentPosterior(self.log_pi, self.log_A, np.log(self.b))

    def test_to_dataframe_matches_hand_derivation(self):
        df = self.q.to_dataframe()
        self.assertEqual(list(df.columns), ["t", "state", "state_0_prob", "state_1_prob"])
        self.assertEqual(df.shape, (2, 4))
        np.testing.assert_array_equal(df["t"].to_numpy(), [0, 1])
        # forward alpha_0 = pi*b[0] = [0.4, 0.1]; alpha_1,k = b[1,k]*0.25 = [0.075, 0.175];
        # backward beta_0 = [0.5, 0.5]; p(x) = sum(alpha_1) = 0.25 -- worked out in the PR description.
        # gamma_0 = alpha_0*beta_0/p(x) = [0.8, 0.2] == b[0] itself (already row-normalized);
        # gamma_1 = alpha_1*beta_1/p(x) = [0.3, 0.7] == b[1] itself.
        np.testing.assert_allclose(df["state_0_prob"].to_numpy(), [0.8, 0.3], atol=1e-12)
        np.testing.assert_allclose(df["state_1_prob"].to_numpy(), [0.2, 0.7], atol=1e-12)
        # MAP path = per-step argmax of the (independent) posterior = argmax of each row of b
        np.testing.assert_array_equal(df["state"].to_numpy(), [0, 1])
        # cross-check the hand derivation against the class's own (independently implemented) methods
        np.testing.assert_allclose(self.q.marginals(), self.b, atol=1e-12)
        self.assertEqual(list(self.q.mode()), [0, 1])


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

    def test_posterior_predictive_generates_from_inferred_topics(self):
        w = np.array(self.m.posterior_predictive(self.doc, n_words=3000, seed=0))  # topic-0 doc
        self.assertEqual(len(w), 3000)
        self.assertGreater(np.mean(np.isin(w, [0, 1])), 0.8)  # words come from topic-0 vocab
        w1 = np.array(self.m.posterior_predictive([(2, 8), (3, 7)], n_words=3000, seed=1))  # topic-1 doc
        self.assertGreater(np.mean(np.isin(w1, [2, 3])), 0.8)
        self.assertTrue(np.array_equal(w, np.array(self.m.posterior_predictive(self.doc, 3000, seed=0))))

    def test_sample_returns_theta_and_per_token_topics(self):
        theta, z = self.q.sample(rng=0)
        self.assertAlmostEqual(float(theta.sum()), 1.0)
        self.assertEqual(len(z), 10)  # total token count 5+4+1
        self.assertTrue(set(z.tolist()) <= {0, 1})
        self.assertTrue(np.array_equal(self.q.sample(rng=1)[1], self.q.sample(rng=1)[1]))  # repeatable


if __name__ == "__main__":
    unittest.main()
