"""WS-1: branch-and-bound MILP, checked against scipy.optimize.milp (HiGHS)."""

import unittest

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from pysp.relations import branch_and_bound_milp


class MILPTest(unittest.TestCase):
    def test_matches_scipy_milp(self):
        for seed in range(80):
            r = np.random.RandomState(seed)
            n, m = r.randint(2, 4), r.randint(1, 4)
            c = r.randint(-5, 5, n).astype(float)
            a = r.randint(0, 4, (m, n)).astype(float)
            b = r.randint(3, 12, m).astype(float)
            bounds = [(0.0, 5.0)] * n
            res = branch_and_bound_milp(c, a, b, integer=range(n), bounds=bounds)
            sol = milp(
                c, constraints=LinearConstraint(a, -np.inf, b), integrality=np.ones(n), bounds=Bounds([0] * n, [5] * n)
            )
            sval = float(sol.fun) if sol.success else None
            with self.subTest(seed=seed):
                if sval is None:
                    self.assertIsNone(res)
                else:
                    self.assertIsNotNone(res)
                    value, x = res
                    self.assertAlmostEqual(value, sval, places=6)  # optimal objective
                    self.assertTrue(np.allclose(x, np.round(x), atol=1e-6))  # integer-feasible
                    self.assertTrue(np.all(a @ x <= b + 1e-6))  # constraints satisfied

    def test_known_knapsack_max(self):
        # maximize 3a + 5b s.t. a + 2b <= 4, a,b in {0..4} integer -> a=4? 3*4=12 vs b=2 ->10; mix a=0,b=2=10
        c = np.array([3.0, 5.0])
        value, x = branch_and_bound_milp(c, [[1.0, 2.0]], [4.0], integer=[0, 1], bounds=[(0, 4), (0, 4)], sense="max")
        self.assertEqual(value, 12.0)  # a=4, b=0
        self.assertTrue(np.allclose(x, [4.0, 0.0]))

    def test_infeasible(self):
        # a >= 3 (i.e. -a <= -3) but a <= 1
        self.assertIsNone(branch_and_bound_milp([1.0], [[-1.0]], [-3.0], integer=[0], bounds=[(0, 1)]))


if __name__ == "__main__":
    unittest.main()
