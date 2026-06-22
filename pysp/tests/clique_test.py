"""WS-1: exact maximum clique / maximum independent set, checked vs brute force."""

import itertools
import unittest

import numpy as np

from pysp.relations import max_clique, max_independent_set


def _brute_clique_size(a):
    n = a.shape[0]
    for size in range(n, 0, -1):
        for c in itertools.combinations(range(n), size):
            if all(a[i, j] for i, j in itertools.combinations(c, 2)):
                return size
    return 0


class CliqueTest(unittest.TestCase):
    def test_matches_brute_force(self):
        for seed in range(300):
            r = np.random.RandomState(seed)
            n = r.randint(1, 8)
            a = (r.rand(n, n) < 0.5).astype(int)
            a = np.triu(a, 1)
            a = a + a.T
            mc = max_clique(a)
            mis = max_independent_set(a)
            with self.subTest(seed=seed):
                self.assertTrue(all(a[i, j] for i, j in itertools.combinations(mc, 2)))       # is a clique
                self.assertEqual(len(mc), _brute_clique_size(a))                              # and maximum
                self.assertTrue(all(not a[i, j] for i, j in itertools.combinations(mis, 2)))  # is independent
                comp = 1 - a
                if n:
                    np.fill_diagonal(comp, 0)
                self.assertEqual(len(mis), _brute_clique_size(comp))                          # and maximum

    def test_known(self):
        self.assertEqual(len(max_clique(1 - np.eye(5, dtype=int))), 5)        # K5
        self.assertEqual(len(max_independent_set(1 - np.eye(5, dtype=int))), 1)
        self.assertEqual(len(max_clique(np.zeros((4, 4), dtype=int))), 1)     # no edges
        self.assertEqual(len(max_independent_set(np.zeros((4, 4), dtype=int))), 4)


if __name__ == "__main__":
    unittest.main()
