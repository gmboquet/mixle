"""WS-1: cardinality-constrained MILP (indicator/sparsity), checked vs brute force."""

import itertools
import unittest

import numpy as np
from scipy.optimize import linprog

from pysp.relations import cardinality_constrained_milp


def _brute(c, a, b, k, bounds):
    c = np.asarray(c, float)
    n = c.size
    best = None
    for sz in range(0, k + 1):
        for sub in itertools.combinations(range(n), sz):
            lo = np.array([bounds[i][0] for i in range(n)], float)
            hi = np.array([bounds[i][1] for i in range(n)], float)
            for i in range(n):
                if i not in sub:
                    lo[i] = hi[i] = 0.0
            r = linprog(c, A_ub=a, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
            if r.success and (best is None or r.fun < best):
                best = r.fun
    return best


class CardinalityMILPTest(unittest.TestCase):
    def test_matches_brute_force(self):
        for seed in range(30):
            r = np.random.RandomState(seed)
            n, m = r.randint(3, 5), r.randint(1, 3)
            c = r.randint(-4, 2, n).astype(float)
            a = r.randint(0, 3, (m, n)).astype(float)
            b = r.randint(2, 8, m).astype(float)
            bounds = [(-2.0, 2.0)] * n
            k = int(r.randint(1, n))
            res = cardinality_constrained_milp(c, a, b, k, bounds)
            bf = _brute(c, a, b, k, bounds)
            with self.subTest(seed=seed):
                self.assertIsNotNone(res)
                value, x = res
                self.assertAlmostEqual(value, bf, places=5)                  # optimal objective
                self.assertLessEqual(int(np.sum(np.abs(x) > 1e-6)), k)       # cardinality respected

    def test_sparsity_enforced(self):
        # want all three at +2 (c<0) but only 1 allowed nonzero -> pick the most negative cost
        c = np.array([-1.0, -3.0, -2.0])
        value, x = cardinality_constrained_milp(c, None, None, 1, [(0.0, 2.0)] * 3)
        self.assertEqual(int(np.sum(np.abs(x) > 1e-6)), 1)
        self.assertAlmostEqual(value, -6.0)  # x = [0, 2, 0]
        self.assertAlmostEqual(x[1], 2.0)


if __name__ == "__main__":
    unittest.main()
