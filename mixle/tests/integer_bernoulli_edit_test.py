"""Tests for IntegerBernoulliEditDistribution's edit-transition log-density, including the "forced
transition" case: a value with p_mat(present | missing) == 1.0 exactly (so p_mat(missing | missing) == 0,
log = -inf). log_nsum/log_dvec's baseline-plus-delta vectorization trick relies on ordinary finite-
arithmetic cancellation and breaks silently for such values unless they are tracked and special-cased --
previously producing a wrongly-finite log_density for an impossible observation (a forced value observed
to stay missing) and a wrongly-infinite one for the correct/expected observation (the forced value
actually becoming present), instead of -inf and a proper finite value respectively.
"""

import itertools
import math
import unittest

import numpy as np

from mixle.stats.sets.integer_bernoulli_edit import IntegerBernoulliEditDistribution


def _all_subsets(n):
    for r in range(n + 1):
        yield from (list(c) for c in itertools.combinations(range(n), r))


class IntegerBernoulliEditBruteForceTestCase(unittest.TestCase):
    def setUp(self):
        # No forced values -- every p_mat(present|missing)/p_mat(present|present) strictly inside (0, 1).
        self.num_vals = 3
        self.dist = IntegerBernoulliEditDistribution(
            [
                (math.log(0.3), math.log(0.6)),
                (math.log(0.4), math.log(0.8)),
                (math.log(0.1), math.log(0.5)),
            ]
        )

    def test_conditional_density_sums_to_one_over_all_next_sets(self):
        # default init_dist is NullDistribution (log_density == 0.0 always), so the joint log_density
        # this class returns is exactly the conditional log p(x1 | x0) -- summing over all possible next
        # sets x1 for a fixed x0 must total 1.
        for x0 in _all_subsets(self.num_vals):
            total = sum(self.dist.density((x0, x1)) for x1 in _all_subsets(self.num_vals))
            self.assertAlmostEqual(total, 1.0, places=10)

    def test_seq_log_density_matches_scalar_log_density(self):
        data = [(x0, x1) for x0 in _all_subsets(self.num_vals) for x1 in _all_subsets(self.num_vals)]
        enc = self.dist.dist_to_encoder().seq_encode(data)
        seq = self.dist.seq_log_density(enc)
        scalar = np.array([self.dist.log_density(d) for d in data])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)


class IntegerBernoulliEditForcedTransitionTestCase(unittest.TestCase):
    def setUp(self):
        # value 0 is forced: p_mat(present | missing) == 1.0 exactly -> p_mat(missing | missing) == 0
        # (log = -inf). value 1 is an ordinary, non-forced value.
        self.p_present_present_0 = 0.9
        self.p_present_missing_1 = 0.4
        self.dist = IntegerBernoulliEditDistribution(
            [(0.0, math.log(self.p_present_present_0)), (math.log(self.p_present_missing_1), math.log(0.8))]
        )

    def test_forced_value_left_missing_in_both_sets_is_impossible(self):
        self.assertEqual(self.dist.log_density(([], [])), -np.inf)

    def test_forced_value_present_in_next_set_is_finite_not_positive_infinity(self):
        # value 0: forced transition happened as required (missing -> present). Previously: +inf.
        ld = self.dist.log_density(([], [0]))
        self.assertTrue(np.isfinite(ld), msg=f"expected a finite log-density, got {ld}")
        expected = 0.0 + math.log(
            1.0 - self.p_present_missing_1
        )  # log p(present|missing)[0] + log p(missing|missing)[1]
        self.assertAlmostEqual(ld, expected, places=10)

    def test_forced_value_kept_present_is_finite(self):
        ld = self.dist.log_density(([0], [0]))
        self.assertTrue(np.isfinite(ld), msg=f"expected a finite log-density, got {ld}")
        expected = math.log(self.p_present_present_0) + math.log(1.0 - self.p_present_missing_1)
        self.assertAlmostEqual(ld, expected, places=10)

    def test_forced_value_removed_is_finite(self):
        ld = self.dist.log_density(([0], []))
        self.assertTrue(np.isfinite(ld), msg=f"expected a finite log-density, got {ld}")
        expected = math.log(1.0 - self.p_present_present_0) + math.log(1.0 - self.p_present_missing_1)
        self.assertAlmostEqual(ld, expected, places=10)

    def test_non_forced_value_alone_is_unaffected(self):
        # sanity: value 1's own ordinary (kept present->present) transition is untouched by value 0
        # being forced -- value 0 must also be touched here (added), or the whole observation would be
        # the impossible "forced value left missing" case from the test above.
        ld = self.dist.log_density(([1], [0, 1]))
        expected = 0.0 + math.log(0.8)  # log p(present|missing)[0] (forced, ==1.0) + log p(present|present)[1]
        self.assertAlmostEqual(ld, expected, places=10)

    def test_seq_log_density_matches_scalar_including_impossible_rows(self):
        data = [([], []), ([], [0]), ([0], [0]), ([0], []), ([1], [0, 1]), ([0, 1], [])]
        enc = self.dist.dist_to_encoder().seq_encode(data)
        seq = self.dist.seq_log_density(enc)
        scalar = np.array([self.dist.log_density(d) for d in data])
        np.testing.assert_array_equal(seq, scalar)  # exact match including -inf, not just close

    def test_seq_log_density_has_no_nan_across_a_mixed_batch(self):
        data = [([], []), ([], [0]), ([0], [0]), ([0], []), ([1], [0, 1])]
        enc = self.dist.dist_to_encoder().seq_encode(data)
        seq = self.dist.seq_log_density(enc)
        self.assertFalse(np.any(np.isnan(seq)), msg=f"seq_log_density produced NaN: {seq}")


if __name__ == "__main__":
    unittest.main()
