"""HMM steady-state initial distribution: pi tied to the transition matrix's equilibrium."""

import unittest

import numpy as np

from mixle.inference import estimate
from mixle.stats import (
    CategoricalDistribution,
    GaussianDistribution,
    GaussianEstimator,
    HiddenMarkovModelDistribution,
)
from mixle.stats.latent.hidden_markov import HiddenMarkovEstimator
from mixle.stats.sequences.markov_chain import stationary_distribution


class StationaryDistributionTest(unittest.TestCase):
    def test_is_a_fixed_point(self):
        a = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        pi = stationary_distribution(a)
        np.testing.assert_allclose(pi @ a, pi, atol=1e-10)
        self.assertAlmostEqual(pi.sum(), 1.0)
        self.assertTrue(np.all(pi >= 0.0))

    def test_two_state_closed_form(self):
        a = np.array([[0.85, 0.15], [0.25, 0.75]])  # pi = [0.25, 0.15] / 0.4
        np.testing.assert_allclose(stationary_distribution(a), [0.625, 0.375], atol=1e-10)


class HmmSteadyStateInitTest(unittest.TestCase):
    def setUp(self):
        self.a_true = np.array([[0.85, 0.15], [0.25, 0.75]])
        self.ld = CategoricalDistribution({20: 1.0})
        true = HiddenMarkovModelDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)],
            list(stationary_distribution(self.a_true)),
            self.a_true.tolist(),
            len_dist=self.ld,
        )
        self.data = true.sampler(seed=0).sample(400)
        self.init = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 2.0), GaussianDistribution(1.0, 2.0)],
            [0.5, 0.5],
            [[0.6, 0.4], [0.4, 0.6]],
            len_dist=self.ld,
        )

    def _fit(self, steady_state_init):
        est = HiddenMarkovEstimator(
            [GaussianEstimator(), GaussianEstimator()],
            len_estimator=self.ld.estimator(),
            steady_state_init=steady_state_init,
        )
        m = self.init
        for _ in range(20):
            m = estimate(self.data, est, m)
        return m

    def test_steady_state_init_ties_w_to_transitions(self):
        m = self._fit(steady_state_init=True)
        np.testing.assert_allclose(m.w, stationary_distribution(m.transitions), atol=1e-8)
        # and it recovers the truth (true stationary ~ [0.625, 0.375])
        np.testing.assert_allclose(np.sort(m.w), [0.375, 0.625], atol=0.05)

    def test_free_init_is_not_forced_to_stationary(self):
        m = self._fit(steady_state_init=False)
        # the free initial-state fit is its own estimate; transitions still recover the truth
        self.assertTrue(np.allclose(np.sort(m.transitions, axis=None), np.sort(self.a_true, axis=None), atol=0.08))


if __name__ == "__main__":
    unittest.main()
