"""WS-1: exact TSP (Held-Karp), checked against brute-force enumeration."""

import itertools
import unittest

import numpy as np

from mixle.relations import tsp_held_karp


def _tour_cost(d, tour):
    n = len(tour)
    return sum(d[tour[i], tour[(i + 1) % n]] for i in range(n))


class TSPTest(unittest.TestCase):
    def test_matches_brute_force(self):
        for seed in range(150):
            r = np.random.RandomState(seed)
            n = r.randint(3, 8)
            d = r.randint(1, 20, size=(n, n)).astype(float)
            np.fill_diagonal(d, 0)
            cost, tour = tsp_held_karp(d)
            brute = min(_tour_cost(d, [0, *p]) for p in itertools.permutations(range(1, n)))
            with self.subTest(seed=seed):
                self.assertEqual(sorted(tour), list(range(n)))  # a valid Hamiltonian cycle
                self.assertEqual(tour[0], 0)
                self.assertAlmostEqual(_tour_cost(d, tour), cost, places=9)  # reported cost is the tour's
                self.assertAlmostEqual(cost, brute, places=9)  # and it is optimal

    def test_symmetric_known(self):
        # square 0-1-2-3 with unit edges; optimal cycle cost 4
        d = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]], dtype=float)
        cost, tour = tsp_held_karp(d)
        self.assertEqual(cost, 4.0)
        self.assertEqual(sorted(tour), [0, 1, 2, 3])

    def test_tiny(self):
        self.assertEqual(tsp_held_karp([[0]])[0], 0.0)
        self.assertEqual(tsp_held_karp([[0, 5], [3, 0]])[0], 8.0)


if __name__ == "__main__":
    unittest.main()
