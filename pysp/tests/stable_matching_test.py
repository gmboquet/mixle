"""WS-1: Gale-Shapley stable matching (combinatorial optimization)."""

import unittest

import numpy as np

from pysp.relations import is_stable_matching, stable_matching


class StableMatchingTest(unittest.TestCase):
    def test_known_instance_is_stable(self):
        pp = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
        rp = [[1, 0, 2], [0, 1, 2], [0, 1, 2]]
        m = stable_matching(pp, rp)
        self.assertTrue(is_stable_matching(m, pp, rp))

    def test_random_full_preferences_complete_and_stable(self):
        rng = np.random.RandomState(0)
        for n in (5, 20, 60):
            pp = [list(rng.permutation(n)) for _ in range(n)]
            rp = [list(rng.permutation(n)) for _ in range(n)]
            m = stable_matching(pp, rp)
            with self.subTest(n=n):
                self.assertEqual(sorted(m), list(range(n)))  # a complete bijection
                self.assertTrue(is_stable_matching(m, pp, rp))

    def test_stable_on_many_small_random_instances(self):
        # is_stable_matching is the ground-truth check (no blocking pair); the output must always pass
        for seed in range(300):
            r = np.random.RandomState(seed)
            k = 4
            pp = [list(r.permutation(k)) for _ in range(k)]
            rp = [list(r.permutation(k)) for _ in range(k)]
            m = stable_matching(pp, rp)
            self.assertTrue(is_stable_matching(m, pp, rp), f"unstable at seed {seed}")

    def test_partial_preferences_and_unequal_sizes(self):
        pp = [[0], [0, 1]]      # proposer 0 only accepts receiver 0
        rp = [[1, 0], [1]]      # receiver 0 prefers proposer 1; receiver 1 only accepts proposer 1
        m = stable_matching(pp, rp)
        self.assertEqual(m, [-1, 0])  # proposer 1 wins receiver 0; proposer 0 left unmatched
        self.assertTrue(is_stable_matching(m, pp, rp))

    def test_blocking_pair_detected(self):
        pp = [[0, 1], [0, 1]]
        rp = [[0, 1], [0, 1]]
        self.assertTrue(is_stable_matching([0, 1], pp, rp))
        self.assertFalse(is_stable_matching([1, 0], pp, rp))  # (0,0) and (1,1) both prefer each other


if __name__ == "__main__":
    unittest.main()
