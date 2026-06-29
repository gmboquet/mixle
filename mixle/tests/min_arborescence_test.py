"""WS-1: minimum spanning arborescence (Chu-Liu/Edmonds), checked against brute force."""

import itertools
import unittest

import numpy as np

from mixle.relations import min_arborescence


def _brute(w, root):
    n = w.shape[0]
    nonroot = [v for v in range(n) if v != root]
    best = None
    for parents in itertools.product(range(n), repeat=len(nonroot)):
        pa = {nonroot[i]: parents[i] for i in range(len(nonroot))}
        if any(pa[v] == v or not np.isfinite(w[pa[v], v]) for v in nonroot):
            continue
        ok = True
        for v in nonroot:  # every node must reach the root without a cycle
            seen, u = set(), v
            while u != root:
                if u in seen:
                    ok = False
                    break
                seen.add(u)
                u = pa[u]
            if not ok:
                break
        if ok:
            cost = sum(w[pa[v], v] for v in nonroot)
            best = cost if best is None else min(best, cost)
    return best


class MinArborescenceTest(unittest.TestCase):
    def test_matches_brute_force(self):
        for seed in range(400):
            r = np.random.RandomState(seed)
            n = r.randint(2, 6)
            w = r.randint(1, 9, size=(n, n)).astype(float)
            w[r.rand(n, n) < 0.2] = np.inf
            np.fill_diagonal(w, np.inf)
            res = min_arborescence(w, 0)
            bf = _brute(w, 0)
            with self.subTest(seed=seed):
                if bf is None:
                    self.assertIsNone(res)
                else:
                    self.assertIsNotNone(res)
                    total, parent = res
                    self.assertAlmostEqual(total, bf, places=9)  # optimal cost
                    self.assertEqual(parent[0], -1)
                    self.assertAlmostEqual(  # parent[] really sums to total
                        sum(w[parent[v], v] for v in range(n) if v != 0), total, places=9
                    )

    def test_simple_chain(self):
        inf = np.inf
        w = np.array([[inf, 2.0, 9.0], [inf, inf, 3.0], [inf, inf, inf]])
        total, parent = min_arborescence(w, 0)
        self.assertEqual(total, 5.0)  # 0->1 (2) + 1->2 (3) beats 0->2 (9)
        self.assertEqual(parent, [-1, 0, 1])


if __name__ == "__main__":
    unittest.main()
