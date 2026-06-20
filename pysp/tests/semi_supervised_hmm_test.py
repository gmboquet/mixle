"""Semi-supervised HMM: each observation may carry a per-position state prior.

Verifies the forward evidence against brute-force path enumeration (with and without a prior), that a one-hot
prior clamps the hidden path, that supervised (one-hot) and unsupervised EM recover a known model, the encoder /
seq_log_density consistency, the sampler, and that the former IndPi* names still resolve as aliases.
"""

import itertools
import unittest

import numpy as np

from pysp.stats import GaussianDistribution as G
from pysp.stats.latent.semi_supervised_hidden_markov_model import (
    IndPiHiddenMarkovModelDistribution,
    SemiSupervisedHiddenMarkovModelDistribution,
)
from pysp.utils.estimation import optimize

_A = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
_TOPICS = [G(-2.0, 1.0), G(2.0, 1.0), G(5.0, 1.0)]


def _brute(emissions, prior, topics, A):
    s = len(topics)
    P = None if prior is None else np.asarray(prior, dtype=float)
    tot = 0.0
    for path in itertools.product(range(s), repeat=len(emissions)):
        v = 1.0
        for t, z in enumerate(path):
            v *= np.exp(topics[z].log_density(emissions[t]))
            if P is not None:
                v *= P[t, z]
            if t > 0:
                v *= A[path[t - 1], z]
        tot += v
    return np.log(tot) if tot > 0 else -np.inf


class SemiSupervisedHMMTestCase(unittest.TestCase):
    def setUp(self):
        self.d = SemiSupervisedHiddenMarkovModelDistribution(_TOPICS, _A)
        self.em = [float(x) for x in np.random.RandomState(0).randn(5) * 3]

    def test_forward_matches_brute_force(self):
        self.assertAlmostEqual(self.d.log_density((self.em, None)), _brute(self.em, None, _TOPICS, _A), places=9)
        prior = np.random.RandomState(1).rand(5, 3) + 0.1
        self.assertAlmostEqual(self.d.log_density((self.em, prior)), _brute(self.em, prior, _TOPICS, _A), places=9)

    def test_one_hot_prior_clamps_path(self):
        forced = [0, 1, 2, 1, 0]
        prior = np.zeros((5, 3))
        for t, z in enumerate(forced):
            prior[t, z] = 1.0
        single = sum(_TOPICS[forced[t]].log_density(self.em[t]) for t in range(5)) + sum(
            np.log(_A[forced[t - 1], forced[t]]) for t in range(1, 5)
        )
        self.assertAlmostEqual(self.d.log_density((self.em, prior)), single, places=9)

    def test_shared_prior_row_broadcasts(self):
        prior_row = np.array([[0.7, 0.2, 0.1]])
        full = np.repeat(prior_row, 5, axis=0)
        self.assertAlmostEqual(self.d.log_density((self.em, prior_row)), self.d.log_density((self.em, full)), places=9)

    def test_seq_log_density_matches_scalar(self):
        prior = np.random.RandomState(2).rand(5, 3) + 0.1
        enc = self.d.dist_to_encoder().seq_encode([(self.em, None), (self.em, prior)])
        sld = self.d.seq_log_density(enc)
        np.testing.assert_allclose(
            sld, [self.d.log_density((self.em, None)), self.d.log_density((self.em, prior))], atol=1e-9
        )

    def test_supervised_recovery_one_step(self):
        rng = np.random.RandomState(0)
        topics = [G(-3.0, 1.0), G(3.0, 1.0)]
        A = np.array([[0.85, 0.15], [0.2, 0.8]])

        def supervised(n):
            s = rng.choice(2)
            em = []
            prior = np.zeros((n, 2))
            for t in range(n):
                if t > 0:
                    s = rng.choice(2, p=A[s])
                em.append(float(topics[s].sampler(seed=rng.randint(1 << 30)).sample()))
                prior[t, s] = 1.0
            return (em, prior)

        data = [supervised(15) for _ in range(80)]
        est = SemiSupervisedHiddenMarkovModelDistribution([G(0.0, 1.0), G(0.0, 1.0)], np.full((2, 2), 0.5)).estimator()
        m = optimize(data, est, max_its=1, rng=np.random.RandomState(2), out=None)
        means = sorted(t.mu for t in m.topics)
        self.assertAlmostEqual(means[0], -3.0, delta=0.3)
        self.assertAlmostEqual(means[1], 3.0, delta=0.3)
        np.testing.assert_allclose(m.transitions, A, atol=0.1)

    def test_unsupervised_em_runs(self):
        truth = SemiSupervisedHiddenMarkovModelDistribution(
            [G(-3.0, 1.0), G(3.0, 1.0)], np.array([[0.85, 0.15], [0.2, 0.8]])
        )
        data = [(truth.sampler(seed=i).sample()[0], None) for i in range(1, 120)]
        est = SemiSupervisedHiddenMarkovModelDistribution([G(0.0, 1.0), G(0.0, 1.0)], np.full((2, 2), 0.5)).estimator()
        m = optimize(data, est, max_its=40, rng=np.random.RandomState(3), out=None)
        means = sorted(t.mu for t in m.topics)
        self.assertLess(means[0], -1.5)
        self.assertGreater(means[1], 1.5)

    def test_sampler_returns_seq_and_none_prior(self):
        from pysp.stats import IntegerCategoricalDistribution

        d = SemiSupervisedHiddenMarkovModelDistribution(
            _TOPICS, _A, len_dist=IntegerCategoricalDistribution(4, [0.0, 0.0, 0.0, 0.5, 0.5])
        )
        s = d.sampler(seed=7).sample()
        self.assertIsInstance(s, tuple)
        self.assertEqual(len(s), 2)
        self.assertIsNone(s[1])
        self.assertGreaterEqual(len(s[0]), 4)
        batch = d.sampler(seed=7).sample(5)
        self.assertEqual(len(batch), 5)

    def test_indpi_aliases_resolve(self):
        self.assertIs(IndPiHiddenMarkovModelDistribution, SemiSupervisedHiddenMarkovModelDistribution)


if __name__ == "__main__":
    unittest.main()
