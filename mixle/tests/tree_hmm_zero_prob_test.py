"""Regression: a zero-probability observation must give tree-HMM log-density -inf, not NaN.

An observation with zero emission probability under every state made the max-subtracted emission row
NaN, which propagated through the upward beta recursion (``betas /= betas_sum`` with ``betas_sum == 0``)
and turned ``log_density`` / ``seq_log_density`` into NaN instead of the correct -inf. Fixed by zeroing
the impossible emission row and clamping the beta-normalization divisor in both the numpy path and the
tree numba kernel, keeping the true ``betas_sum`` for the log so the likelihood is -inf.
"""

import unittest
import warnings

import numpy as np

import mixle.stats as stats
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovEstimator, TreeHiddenMarkovModelDistribution
from mixle.utils.vector import ImpossibleEvidenceError

# tree datum format: list of ((node_id, parent_id), observation); 'c' has zero emission prob in both states.
_IMPOSSIBLE = [((0, -1), "a"), ((1, 0), "c"), ((2, 0), "b")]
_NORMAL = [((0, -1), "a"), ((1, 0), "b")]


def _tree_hmm(use_numba, zero_symbol):
    emit = (
        ({"a": 0.6, "b": 0.4, "c": 0.0}, {"a": 0.3, "b": 0.7, "c": 0.0})
        if zero_symbol
        else ({"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.7})
    )
    return TreeHiddenMarkovModelDistribution(
        topics=[stats.CategoricalDistribution(emit[0]), stats.CategoricalDistribution(emit[1])],
        w=[0.5, 0.5],
        transitions=[[0.7, 0.3], [0.4, 0.6]],
        len_dist=stats.IntegerCategoricalDistribution(min_val=0, p_vec=np.array([0.3, 0.4, 0.3])),
        terminal_level=3,
        use_numba=use_numba,
    )


class TreeHmmZeroProbTest(unittest.TestCase):
    def test_impossible_observation_log_density_is_neg_inf(self):
        for use_numba in (True, False):
            m = _tree_hmm(use_numba, True)
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a NaN/divide warning fails the test
                ld = m.log_density(_IMPOSSIBLE)
                enc = m.dist_to_encoder().seq_encode([_IMPOSSIBLE, _NORMAL])
                sl = np.asarray(m.seq_log_density(enc), dtype=float)
            self.assertEqual(ld, -np.inf, f"use_numba={use_numba}")
            self.assertEqual(sl[0], -np.inf, f"use_numba={use_numba}")
            self.assertTrue(np.isfinite(sl[1]), f"use_numba={use_numba}")  # normal tree unaffected

    def test_seq_update_rejects_an_impossible_tree_transactionally(self):
        for use_numba in (True, False):
            m = _tree_hmm(use_numba, True)
            est = TreeHiddenMarkovEstimator(
                [stats.CategoricalEstimator(), stats.CategoricalEstimator()], use_numba=use_numba
            )
            enc = m.dist_to_encoder().seq_encode([_NORMAL, _IMPOSSIBLE, _NORMAL])
            acc = est.accumulator_factory().make()
            with warnings.catch_warnings(), self.assertRaisesRegex(ImpossibleEvidenceError, "zero-probability"):
                warnings.simplefilter("error")  # a NaN/divide warning fails the test
                acc.seq_update(enc, np.ones(3), m)
            _, init_counts, state_counts, trans_counts, _, _ = acc.value()
            np.testing.assert_array_equal(init_counts, np.zeros(2))
            np.testing.assert_array_equal(state_counts, np.zeros(2))
            np.testing.assert_array_equal(trans_counts, np.zeros((2, 2)))

    def test_seq_posterior_rejects_an_impossible_tree(self):
        for use_numba in (True, False):
            m = _tree_hmm(use_numba, True)
            enc = m.dist_to_encoder().seq_encode([_NORMAL, _IMPOSSIBLE])
            with warnings.catch_warnings(), self.assertRaisesRegex(ImpossibleEvidenceError, "zero-probability"):
                warnings.simplefilter("error")
                m.seq_posterior(enc)

    def test_numba_numpy_agree_on_normal_data(self):
        # the guards are no-ops on ordinary data, so the two backends must agree to machine precision
        data = _tree_hmm(False, False).sampler(seed=5).sample(8)
        mn, mp = _tree_hmm(True, False), _tree_hmm(False, False)
        sn = np.asarray(mn.seq_log_density(mn.dist_to_encoder().seq_encode(data)), dtype=float)
        sp = np.asarray(mp.seq_log_density(mp.dist_to_encoder().seq_encode(data)), dtype=float)
        self.assertTrue(np.allclose(sn, sp, rtol=0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
