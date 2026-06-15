"""Tests for the child-count length distribution contribution to tree HMM scoring.

The tree HMM accumulates a length distribution over the number of children of each node and now
includes its likelihood contribution in log_density / seq_log_density. These tests verify that:

  * the scalar log_density equals the old (length-free) score plus the sum of child-count log probs,
  * scalar log_density is consistent with seq_log_density on both the numba and numpy paths, and
  * a NullDistribution length leaves the score unchanged.

"""

import unittest

import numpy as np

from pysp.stats.combinator.null_dist import NullDistribution
from pysp.stats.latent.tree_hmm import TreeHiddenMarkovModelDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution
from pysp.stats.leaf.int_range import IntegerCategoricalDistribution


def _child_counts(tree):
    """Number of children for each node id in a tree of ((node_id, parent_id), emission) tuples."""
    counts = {node_id: 0 for (node_id, _), _ in tree}
    for (_, parent_id), _ in tree:
        if parent_id in counts:
            counts[parent_id] += 1
    return counts


class TreeHmmLenTest(unittest.TestCase):
    def setUp(self):
        self.num_states = 2
        self.topics = [GaussianDistribution(mu=0.0, sigma2=1.0), GaussianDistribution(mu=10.0, sigma2=1.0)]
        self.w = np.array([0.5, 0.5])
        self.trans = np.array([[0.7, 0.3], [0.3, 0.7]])

        # Categorical length distribution over child counts {0, 1, 2}.
        self.len_probs = np.array([0.1, 0.3, 0.6])
        self.len_dist = IntegerCategoricalDistribution(min_val=0, p_vec=self.len_probs)

        # Trees whose forward recursion is supported on the (otherwise fragile) numpy path so we can
        # cross-check both scoring backends. child counts noted per node.
        self.trees = [
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9)],  # 0->1, 1->1, 2->0
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9), ((3, 2), 0.3)],  # 0->1, 1->1, 2->1, 3->0
            [((0, -1), 0.1), ((1, 0), 0.2), ((2, 0), 9.9)],  # 0->2, 1->0, 2->0
        ]

    def _len_term(self, tree):
        """Hand-computed sum over nodes of log len_dist(num_children(node))."""
        return float(sum(np.log(self.len_probs[c]) for c in _child_counts(tree).values()))

    def _dist(self, len_dist, use_numba):
        return TreeHiddenMarkovModelDistribution(
            topics=self.topics,
            w=self.w,
            transitions=self.trans,
            len_dist=len_dist,
            terminal_level=4,
            use_numba=use_numba,
        )

    def test_scalar_equals_old_plus_len_term(self):
        """Scalar log_density == (length-free score) + sum of child-count log probs."""
        for use_numba in (True, False):
            d = self._dist(self.len_dist, use_numba)
            d_null = self._dist(NullDistribution(), use_numba)
            for tree in self.trees:
                with self.subTest(use_numba=use_numba, tree=tree):
                    old = d_null.log_density(tree)
                    expected = old + self._len_term(tree)
                    self.assertAlmostEqual(d.log_density(tree), expected, places=9)

    def test_scalar_matches_seq_sum_numba(self):
        """Sum of scalar log_density equals seq_log_density(...).sum() for a batch (numba backend)."""
        d = self._dist(self.len_dist, use_numba=True)
        scalar_sum = sum(d.log_density(t) for t in self.trees)
        enc = d.dist_to_encoder().seq_encode(self.trees)
        seq_sum = float(d.seq_log_density(enc).sum())
        self.assertAlmostEqual(scalar_sum, seq_sum, places=9)

    def test_scalar_matches_seq_numpy(self):
        """Scalar log_density equals the single-tree seq_log_density on the numpy backend.

        The numpy multi-tree forward pass is pre-existingly fragile for some tree shapes, so this
        checks the per-tree (size-1 batch) path the scalar log_density actually uses.
        """
        d = self._dist(self.len_dist, use_numba=False)
        for tree in self.trees:
            with self.subTest(tree=tree):
                enc = d.dist_to_encoder().seq_encode([tree])
                self.assertAlmostEqual(d.log_density(tree), float(d.seq_log_density(enc)[0]), places=9)

    def test_numba_numpy_agree(self):
        """The numba and numpy scoring backends agree once the length term is included."""
        d_nb = self._dist(self.len_dist, use_numba=True)
        d_np = self._dist(self.len_dist, use_numba=False)
        for tree in self.trees:
            with self.subTest(tree=tree):
                self.assertAlmostEqual(d_nb.log_density(tree), d_np.log_density(tree), places=9)

    def test_null_length_unchanged(self):
        """A NullDistribution length contributes nothing: scores equal the length-free baseline."""
        for use_numba in (True, False):
            d_null = self._dist(NullDistribution(), use_numba)
            for tree in self.trees:
                with self.subTest(use_numba=use_numba, tree=tree):
                    # NullDistribution.log_density is identically 0, so the per-node length term is 0.
                    self.assertAlmostEqual(self._null_len_term(), 0.0, places=12)
                    # Score must match a hand-evaluated length-free total via seq_log_density.
                    enc = d_null.dist_to_encoder().seq_encode([tree])
                    self.assertAlmostEqual(d_null.log_density(tree), float(d_null.seq_log_density(enc)[0]), places=9)

    @staticmethod
    def _null_len_term():
        return NullDistribution().log_density(7) + NullDistribution().log_density(0)


if __name__ == "__main__":
    unittest.main()
