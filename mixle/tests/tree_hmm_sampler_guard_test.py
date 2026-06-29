"""Regression tests for the tree HMM sampler's exponential-growth guard.

``TreeHiddenMarkovSampler.sample_tree`` draws ``len_dist``-many children at every level up to
``terminal_level``. A ``len_dist`` whose mean exceeds one child is a super-critical branching
process, so the tree size grows like ``mean_children ** terminal_level`` (e.g. ``{4: 1.0}`` with
``terminal_level=10`` is ~4**10 nodes) -- this used to silently spin at ~100% CPU and look like a
hang. The sampler now estimates the expected per-tree node count up front and fails fast with an
actionable ``ValueError``. These tests pin that behavior and confirm valid (finite-tree) configs
are unaffected.
"""

import unittest

import numpy as np

from mixle.stats.combinator.null_dist import NullDistribution
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovModelDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution


class TreeHmmSamplerGuardTest(unittest.TestCase):
    def setUp(self):
        self.topics = [GaussianDistribution(mu=-2.0, sigma2=1.0), GaussianDistribution(mu=2.0, sigma2=1.0)]
        self.w = np.array([0.5, 0.5])
        self.trans = np.array([[0.9, 0.1], [0.2, 0.8]])

    def _dist(self, len_dist, terminal_level):
        return TreeHiddenMarkovModelDistribution(
            topics=self.topics,
            w=self.w,
            transitions=self.trans,
            len_dist=len_dist,
            terminal_level=terminal_level,
        )

    def test_explosive_len_dist_raises_fast(self):
        """mean children > 1 with a large terminal_level fails fast instead of hanging."""
        # Always draws 4 children -> ~4**10 nodes per tree at terminal_level=10.
        len_dist = IntegerCategoricalDistribution(min_val=4, p_vec=np.array([1.0]))
        d = self._dist(len_dist, terminal_level=10)
        with self.assertRaises(ValueError) as ctx:
            d.sampler(seed=7)
        self.assertIn("blow-up", str(ctx.exception))

    def test_subcritical_len_dist_samples(self):
        """mean children <= 1 (mass on 0) terminates: the sampler builds finite trees."""
        # P(0)=0.55, P(2)=0.45 -> mean 0.9 children.
        len_dist = IntegerCategoricalDistribution(min_val=0, p_vec=np.array([0.55, 0.0, 0.45]))
        d = self._dist(len_dist, terminal_level=4)
        trees = d.sampler(seed=7).sample(20)
        self.assertEqual(len(trees), 20)
        self.assertTrue(all(len(t) >= 1 for t in trees))

    def test_critical_mean_one_not_flagged(self):
        """mean children exactly 1 is not flagged as a blow-up (no false positive)."""
        # P(0)=0.5, P(2)=0.5 -> mean 1.0 children.
        len_dist = IntegerCategoricalDistribution(min_val=0, p_vec=np.array([0.5, 0.0, 0.5]))
        d = self._dist(len_dist, terminal_level=10)
        # Must not raise at sampler construction.
        d.sampler(seed=7)

    def test_null_length_rejected_by_existing_check(self):
        """A NullDistribution length is rejected by the pre-existing sampler check (not the guard)."""
        d = self._dist(NullDistribution(), terminal_level=10)
        # Unchanged behavior: sampling needs a length distribution on the non-negative integers.
        with self.assertRaisesRegex(Exception, "non-negative integers"):
            d.sampler(seed=7)


if __name__ == "__main__":
    unittest.main()
