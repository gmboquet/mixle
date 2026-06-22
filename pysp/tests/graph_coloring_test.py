"""WS-1: exact graph coloring (chromatic number), checked against brute force + known graphs."""

import itertools
import unittest

import numpy as np

from pysp.relations import graph_coloring


def _proper(coloring, a):
    n = a.shape[0]
    return all(coloring[i] != coloring[j] for i in range(n) for j in range(i + 1, n) if a[i, j])


def _brute_chromatic(a):
    n = a.shape[0]
    if n == 0:
        return 0
    for k in range(1, n + 1):
        for c in itertools.product(range(k), repeat=n):
            if max(c) == k - 1 and all(c[i] != c[j] for i in range(n) for j in range(i + 1, n) if a[i, j]):
                return k
    return n


class GraphColoringTest(unittest.TestCase):
    def test_matches_brute_force(self):
        for seed in range(300):
            r = np.random.RandomState(seed)
            n = r.randint(1, 7)
            a = (r.rand(n, n) < 0.4).astype(int)
            a = np.triu(a, 1)
            a = a + a.T
            k, col = graph_coloring(a)
            with self.subTest(seed=seed):
                self.assertTrue(_proper(col, a))            # proper coloring
                self.assertEqual(max(col) + 1 if n else 0, k)  # uses exactly k colors
                self.assertEqual(k, _brute_chromatic(a))    # and k is minimal

    def test_known_graphs(self):
        self.assertEqual(graph_coloring(1 - np.eye(5, dtype=int))[0], 5)  # K5
        even_cycle = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        self.assertEqual(graph_coloring(even_cycle)[0], 2)  # bipartite
        odd_cycle = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]])
        self.assertEqual(graph_coloring(odd_cycle)[0], 3)
        self.assertEqual(graph_coloring(np.zeros((4, 4), dtype=int))[0], 1)  # no edges -> 1 color


if __name__ == "__main__":
    unittest.main()
