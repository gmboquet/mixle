"""Semi-supervised HMM terminal (absorbing) hidden states: scoring, sampling, and Baum-Welch EM."""

import itertools
import unittest

import numpy as np

from mixle.inference import estimate
from mixle.stats import GaussianDistribution
from mixle.stats.latent.semi_supervised_hidden_markov_model import SemiSupervisedHiddenMarkovModelDistribution as SS


class SemiSupervisedTerminalStatesTest(unittest.TestCase):
    def setUp(self):
        self.topics = [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)]
        self.a = np.array([[0.6, 0.4], [0.5, 0.5]])
        self.d = SS(self.topics, self.a, terminal_states={1})  # state 1 absorbing, uniform initial

    def _brute(self, x):
        log_a = np.log(self.a)
        total = -np.inf
        for path in itertools.product([0, 1], repeat=len(x)):
            if path[-1] != 1 or any(z == 1 for z in path[:-1]):
                continue
            lp = sum(self.topics[path[t]].log_density(x[t]) for t in range(len(x))) + sum(
                log_a[path[t], path[t + 1]] for t in range(len(x) - 1)
            )  # uniform initial weight 1 (log_w = 0)
            total = np.logaddexp(total, lp)
        return total

    def test_forward_matches_brute_force(self):
        for x in [[2.0], [-2.0, 2.0], [-2.0, -2.0, 2.0]]:
            self.assertAlmostEqual(self.d.log_density((x, None)), self._brute(x), places=9)

    def test_seq_matches_scalar(self):
        data = [([2.0], None), ([-2.0, 2.0], None), ([-2.0, -2.0, 2.0], None)]
        enc = self.d.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(x) for x in data], atol=1e-12)

    def test_sampler_ends_in_terminal_state(self):
        g = SS(
            [GaussianDistribution(-5, 0.3), GaussianDistribution(0, 0.3), GaussianDistribution(5, 0.3)],
            [[0.45, 0.45, 0.1], [0.45, 0.45, 0.1], [0.0, 0.0, 1.0]],
            terminal_states={2},
        )
        s = g.sampler(seed=0).sample(2000)
        self.assertGreater(np.mean([e[-1] for e, _ in s]), 3.0)  # last emission from the terminal state (mean ~5)

    def test_baum_welch_recovers_parameters(self):
        true = SS(
            [GaussianDistribution(-5, 1.0), GaussianDistribution(0, 1.0), GaussianDistribution(5, 1.0)],
            [[0.45, 0.45, 0.1], [0.45, 0.45, 0.1], [0.0, 0.0, 1.0]],
            terminal_states={2},
        )
        data = true.sampler(seed=0).sample(250)
        init = SS(
            [GaussianDistribution(-3, 2.0), GaussianDistribution(1, 2.0), GaussianDistribution(4, 2.0)],
            [[0.4, 0.4, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]],
            terminal_states={2},
        )
        m = init
        for _ in range(20):
            m = estimate(data, init.estimator(), m)
        self.assertEqual(m.terminal_states, {2})
        np.testing.assert_allclose(sorted(t.mu for t in m.topics), [-5.0, 0.0, 5.0], atol=0.2)
        np.testing.assert_allclose(m.transitions[:2], [[0.45, 0.45, 0.1], [0.45, 0.45, 0.1]], atol=0.08)


if __name__ == "__main__":
    unittest.main()
