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

    def test_seq_update_accumulator_is_finite_with_an_impossible_tree_in_the_batch(self):
        # The docstring's "clamping the beta-normalization divisor" fix above only ever reached
        # log_density/seq_log_density (scoring). seq_update -- the actual EM M-step accumulator, on
        # both the numpy and numba paths -- had the identical unguarded betas /= betas_sum, so one
        # impossible tree in a fit batch turned init_counts/state_counts/trans_counts entirely to NaN,
        # silently corrupting every other tree's contribution too. Untested until now.
        for use_numba in (True, False):
            m = _tree_hmm(use_numba, True)
            est = TreeHiddenMarkovEstimator(
                [stats.CategoricalEstimator(), stats.CategoricalEstimator()], use_numba=use_numba
            )
            enc = m.dist_to_encoder().seq_encode([_NORMAL, _IMPOSSIBLE, _NORMAL])
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a NaN/divide warning fails the test
                acc = est.accumulator_factory().make()
                acc.seq_update(enc, np.ones(3), m)
            _, init_counts, state_counts, trans_counts, _, _ = acc.value()
            self.assertTrue(np.all(np.isfinite(init_counts)), f"use_numba={use_numba}: {init_counts}")
            self.assertTrue(np.all(np.isfinite(state_counts)), f"use_numba={use_numba}: {state_counts}")
            self.assertTrue(np.all(np.isfinite(trans_counts)), f"use_numba={use_numba}: {trans_counts}")
            # the two valid trees must still contribute real (nonzero) counts, not be wiped out
            self.assertGreater(float(np.sum(state_counts)), 0.0, f"use_numba={use_numba}")

    def test_seq_posterior_is_finite_with_an_impossible_tree_in_the_batch(self):
        # seq_posterior shares the same unguarded betas /= betas_sum shape as seq_update, on both the
        # numpy and numba (numba_posteriors) paths.
        for use_numba in (True, False):
            m = _tree_hmm(use_numba, True)
            enc = m.dist_to_encoder().seq_encode([_NORMAL, _IMPOSSIBLE])
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                post = m.seq_posterior(enc)
            for p in post:
                self.assertTrue(np.all(np.isfinite(p)), f"use_numba={use_numba}: {p}")

    def test_numba_numpy_agree_on_normal_data(self):
        # the guards are no-ops on ordinary data, so the two backends must agree to machine precision
        data = _tree_hmm(False, False).sampler(seed=5).sample(8)
        mn, mp = _tree_hmm(True, False), _tree_hmm(False, False)
        sn = np.asarray(mn.seq_log_density(mn.dist_to_encoder().seq_encode(data)), dtype=float)
        sp = np.asarray(mp.seq_log_density(mp.dist_to_encoder().seq_encode(data)), dtype=float)
        self.assertTrue(np.allclose(sn, sp, rtol=0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
