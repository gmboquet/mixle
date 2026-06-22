"""Lookback HMM terminal (absorbing) hidden states: scoring, batch scoring, and Baum-Welch EM."""

import itertools
import unittest

import numpy as np

from pysp.inference import estimate
from pysp.stats import CategoricalDistribution
from pysp.stats.base.integer_categorical import IntegerCategoricalDistribution
from pysp.stats.combinator.sequence import SequenceDistribution
from pysp.stats.latent.lookback_hidden_markov_model import LookbackHiddenMarkovModelDistribution as LB


def _mk(p):
    return SequenceDistribution(IntegerCategoricalDistribution(0, p), len_dist=CategoricalDistribution({1: 1.0}))


def _brute(log_b, log_w, log_a, term):
    length, k = log_b.shape
    total = -np.inf
    for path in itertools.product(range(k), repeat=length):
        if not term[path[-1]] or any(term[z] for z in path[:-1]):
            continue
        lp = (
            log_w[path[0]]
            + log_b[0, path[0]]
            + sum(log_a[path[t], path[t + 1]] + log_b[t + 1, path[t + 1]] for t in range(length - 1))
        )
        total = np.logaddexp(total, lp)
    return total


class LookbackTerminalStatesTest(unittest.TestCase):
    def setUp(self):
        self.w = np.array([0.7, 0.3])
        self.a = np.array([[0.6, 0.4], [0.5, 0.5]])
        self.d = LB([_mk([0.85, 0.15]), _mk([0.2, 0.8])], self.w, self.a, lag=0, terminal_states={1})

    def test_scoring_matches_brute_force_lag0(self):
        for x in [[1], [0, 1], [0, 0, 1], [1, 0, 1]]:
            ref = _brute(self.d._windowed_log_b(x), self.d.log_w, self.d.log_transitions, self.d._terminal_mask)
            self.assertAlmostEqual(self.d.log_density(x), ref, places=9)

    def test_scoring_matches_brute_force_lag1(self):
        init = [_mk([0.5, 0.5]), _mk([0.4, 0.6])]
        win = [
            SequenceDistribution(
                IntegerCategoricalDistribution(0, [0.6, 0.4]), len_dist=CategoricalDistribution({2: 1.0})
            ),
            SequenceDistribution(
                IntegerCategoricalDistribution(0, [0.3, 0.7]), len_dist=CategoricalDistribution({2: 1.0})
            ),
        ]
        d1 = LB(win, self.w, self.a, lag=1, init_dist=init, terminal_states={1})
        for x in [[0, 1], [0, 0, 1], [1, 0, 1]]:
            ref = _brute(d1._windowed_log_b(x), d1.log_w, d1.log_transitions, d1._terminal_mask)
            self.assertAlmostEqual(d1.log_density(x), ref, places=9)

    def _manual_data(self, seed, n=400):
        rng = np.random.RandomState(seed)
        emit = [[0.85, 0.15], [0.2, 0.8]]
        out = []
        for _ in range(n):
            z = rng.choice(2, p=self.w)
            states = [z]
            while z != 1:
                z = rng.choice(2, p=self.a[z])
                states.append(z)
            out.append([int(rng.choice(2, p=emit[s])) for s in states])
        return out

    def test_seq_log_density_matches_scalar(self):
        data = self._manual_data(0, 6)
        enc = self.d.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(s) for s in data], atol=1e-12)

    def test_baum_welch_recovers_parameters(self):
        data = self._manual_data(1)
        init = LB(
            [_mk([0.6, 0.4]), _mk([0.45, 0.55])], [0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], lag=0, terminal_states={1}
        )
        m = init
        for _ in range(30):
            m = estimate(data, init.estimator(), m)
        self.assertEqual(m.terminal_states, {1})
        emissions = sorted(np.exp(t.dist.log_p_vec)[0] for t in m.topics)
        np.testing.assert_allclose(emissions, [0.2, 0.85], atol=0.06)  # the two states' P(symbol 0)
        np.testing.assert_allclose(m.transitions[0], [0.6, 0.4], atol=0.07)  # non-terminal row recovers


if __name__ == "__main__":
    unittest.main()
