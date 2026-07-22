"""mixle.utils.pvalues: binomial_rank's log-density histogram over composite Bernoulli evidence.

binomial_rank has no internal callers (only its own __main__ demo), but stays reachable as public
API via mixle.utils's lazy submodule export (mixle.utils.pvalues.binomial_rank) -- so it still gets
a guard against an empty usable-term set rather than a confusing low-level numpy failure.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle.utils.pvalues import binomial_rank


class BinomialRankTest(unittest.TestCase):
    def test_basic_histogram_still_works(self):
        acc_ll, acc_prob, (ll0, dll, cnt) = binomial_rank(np.log([0.3, 0.2]), count_vec=[3, 2], max_len=10000)
        self.assertEqual(len(acc_ll), len(acc_prob))
        self.assertAlmostEqual(float(acc_prob.sum()), 1.0, places=6)
        self.assertEqual(cnt, 5)

    def test_all_zero_counts_raises_a_clear_error_instead_of_a_low_level_failure(self):
        # regression: every term skipped (n == 0 for every entry) used to leave `entries` empty and
        # crash inside np.concatenate with "need at least one array to concatenate" -- itself a
        # ValueError, so this must check the MESSAGE, not just the type, or it would pass against
        # both the pre-fix numpy failure and the fix's own deliberate guard.
        with self.assertRaisesRegex(ValueError, "no usable binomial terms"):
            binomial_rank(np.log([0.3, 0.2]), count_vec=[0, 0])

    def test_all_negative_infinite_log_prob_raises_the_same_clear_error(self):
        with self.assertRaisesRegex(ValueError, "no usable binomial terms"):
            binomial_rank(np.array([-np.inf, -np.inf]), count_vec=[3, 2])


if __name__ == "__main__":
    unittest.main()
