"""Regression for T4-01: fitting a tree-HMM on a shallow-only corpus (e.g. every tree is a
childless root) left the never-observed transition row at exact zero instead of raising or
filling a well-defined default. The model looked superficially finite -- w/transitions/topics
all finite, no warning, no exception out of optimize() -- but the zero row poisoned p_level (the
per-level marginal state-occupancy vector, w times successive powers of the transition matrix)
to exact zero at every level beyond what the corpus covered. log_density() on a later,
deeper out-of-sample tree then silently returned NaN (numpy path) or -inf (numba path), because
code elsewhere legitimately divides by p_level to remove a prior state-occupancy factor. Fixed by
filling an evidence-free row uniform (mirroring MarkovChainEstimator's identical convention) and
disclosing it through numerical_repairs(), so p_level stays strictly positive at every level.
"""

import unittest
import warnings

import numpy as np

from mixle.inference import optimize
from mixle.stats import GaussianEstimator
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovEstimator

# A legitimate, valid single-node tree (a root with no children) -- the whole corpus below is
# built out of copies of this, so no tree in the training corpus ever has an edge.
_ROOT_ONLY = [((0, -1), 0.1)]
# An ordinary out-of-sample tree that reaches deeper than anything the corpus above covers.
_DEEPER = [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9)]


class TreeHmmShallowCorpusTransitionRepairTest(unittest.TestCase):
    def test_shallow_only_fit_scores_a_deeper_tree_finitely(self):
        for use_numba in (False, True):
            est = TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()], use_numba=use_numba)
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a NaN/divide warning fails the test
                model = optimize([_ROOT_ONLY for _ in range(8)], est, out=None)

            self.assertTrue(np.all(np.isfinite(model.w)), f"use_numba={use_numba}")
            self.assertTrue(np.all(np.isfinite(model.transitions)), f"use_numba={use_numba}")
            # every row of a fitted transition matrix must be a valid probability distribution,
            # never a hard-zero row (that is the actual bug: a zero row is finite but not
            # normalized, and poisons p_level at any level reached only through that state)
            np.testing.assert_allclose(model.transitions.sum(axis=1), 1.0, err_msg=f"use_numba={use_numba}")
            # the repair must be disclosed, not silent
            repairs = model.numerical_repairs()
            self.assertEqual(len(repairs), 1, f"use_numba={use_numba}: {repairs}")
            self.assertIn("tree-hmm-row-uniform", repairs[0])

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ld = model.log_density(_DEEPER)
            self.assertTrue(np.isfinite(ld), f"use_numba={use_numba}: log_density={ld}")

    def test_well_evidenced_rows_are_untouched_by_the_repair(self):
        # Direct M-step check: state 2 never has an outgoing-transition edge (a bug for it alone),
        # while states 0 and 1 have real, asymmetric transition evidence. The fix must leave the
        # well-evidenced rows numerically identical to the plain normalized-count formula and must
        # only touch/flag the state that actually lacks evidence -- a guard scoped too wide would
        # instead perturb (or flag) rows 0/1 too.
        gauss_est = GaussianEstimator()
        topic_ss = []
        for mu in (0.0, 10.0, 50.0):
            acc = gauss_est.accumulator_factory().make()
            acc.initialize(mu, 1.0, np.random.RandomState(0))
            acc.initialize(mu + 0.1, 1.0, np.random.RandomState(1))
            topic_ss.append(acc.value())

        init_counts = np.array([5.0, 5.0, 20.0])
        trans_counts = np.array(
            [
                [8.0, 2.0, 0.0],
                [3.0, 7.0, 0.0],
                [0.0, 0.0, 0.0],  # state 2: no outgoing transition evidence at all
            ]
        )
        state_counts = init_counts + trans_counts.sum(axis=0)
        suff_stat = (3, init_counts, state_counts, trans_counts, topic_ss, None)

        est = TreeHiddenMarkovEstimator(
            [GaussianEstimator(), GaussianEstimator(), GaussianEstimator()], use_numba=False
        )
        model = est.estimate(nobs=None, suff_stat=suff_stat)

        np.testing.assert_allclose(model.transitions[0], trans_counts[0] / trans_counts[0].sum())
        np.testing.assert_allclose(model.transitions[1], trans_counts[1] / trans_counts[1].sum())
        np.testing.assert_allclose(model.transitions[2], np.full(3, 1.0 / 3.0))

        repairs = model.numerical_repairs()
        self.assertEqual(len(repairs), 1)
        self.assertIn("state(s): 2", repairs[0])


if __name__ == "__main__":
    unittest.main()
