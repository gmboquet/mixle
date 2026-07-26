"""mixle.utils.pvalues: binomial_rank's log-density histogram over composite Bernoulli evidence.

binomial_rank has no internal callers (only its own __main__ demo), but stays reachable as public
API via mixle.utils's lazy submodule export (mixle.utils.pvalues.binomial_rank).
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

    def test_all_zero_counts_are_the_valid_empty_experiment(self):
        ll, probability, (_ll0, dll, count) = binomial_rank(
            np.log([0.3, 0.2]),
            count_vec=[0, 0],
        )
        np.testing.assert_allclose(ll, [0.0])
        np.testing.assert_allclose(probability, [1.0])
        self.assertGreater(dll, 0.0)
        self.assertEqual(count, 0)

    def test_deterministic_probabilities_are_retained_in_the_count(self):
        ll, probability, (_ll0, _dll, count) = binomial_rank(
            np.array([-np.inf, 0.0]),
            count_vec=[3, 2],
        )
        np.testing.assert_allclose(ll, [0.0])
        np.testing.assert_allclose(probability, [1.0])
        self.assertEqual(count, 5)

    def test_single_fair_bernoulli_has_one_equal_likelihood_bin(self):
        ll, probability, (ll0, dll, count) = binomial_rank(
            np.log([0.5]),
            count_vec=[1],
        )
        np.testing.assert_allclose(ll, [np.log(0.5)])
        np.testing.assert_allclose(probability, [1.0])
        self.assertEqual(ll0, np.log(0.5))
        self.assertGreater(dll, 0.0)
        self.assertEqual(count, 1)

    def test_parallel_vectors_and_controls_are_validated(self):
        with self.assertRaisesRegex(ValueError, "exactly one count"):
            binomial_rank(np.log([0.3, 0.2]), count_vec=[1])
        with self.assertRaisesRegex(ValueError, "complementary"):
            binomial_rank(np.log([0.3]), log_p1_vec=np.log([0.3]), count_vec=[1])
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            binomial_rank(np.log([0.3]), count_vec=[1.5])
        with self.assertRaisesRegex(ValueError, "ll_eps"):
            binomial_rank(np.log([0.3]), ll_eps=0.0)
        with self.assertRaisesRegex(ValueError, "max_len"):
            binomial_rank(np.log([0.3]), max_len=0)


if __name__ == "__main__":
    unittest.main()
