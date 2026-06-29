"""WS-1: infeasibility diagnostics -- irreducible infeasible subset of linear constraints."""

import unittest

import numpy as np
from scipy.optimize import linprog

from mixle.relations import irreducible_infeasible_subset


def _feasible(a, b, bounds):
    if len(b) == 0:
        return True
    return bool(linprog(np.zeros(a.shape[1]), A_ub=a, b_ub=b, bounds=bounds, method="highs").success)


class IISTest(unittest.TestCase):
    def test_feasible_system_returns_none(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([5.0, 5.0])
        self.assertIsNone(irreducible_infeasible_subset(a, b, [(0, 10), (0, 10)]))

    def test_result_is_irreducible_and_infeasible(self):
        for seed in range(400):
            r = np.random.RandomState(seed)
            n, m = r.randint(1, 3), r.randint(2, 6)
            a = r.randint(-3, 4, (m, n)).astype(float)
            b = r.randint(-4, 4, m).astype(float)
            bounds = [(-10.0, 10.0)] * n
            sub = irreducible_infeasible_subset(a, b, bounds)
            if sub is None:
                self.assertTrue(_feasible(a, b, bounds))  # None only when feasible
                continue
            with self.subTest(seed=seed):
                a2, b2 = a[sub], b[sub]
                self.assertFalse(_feasible(a2, b2, bounds))  # the subset is infeasible
                for k in range(len(sub)):  # ... and minimal
                    self.assertTrue(_feasible(np.delete(a2, k, 0), np.delete(b2, k, 0), bounds))

    def test_unique_conflict(self):
        # x <= 1 and -x <= -3 (x >= 3) are the only constraints and both are needed -> IIS is exactly {0, 1}
        a = np.array([[1.0], [-1.0]])
        b = np.array([1.0, -3.0])
        self.assertEqual(set(irreducible_infeasible_subset(a, b, [(-100, 100)])), {0, 1})

    def test_redundant_constraint_filtered(self):
        # add a redundant upper bound x <= 2; the result is still a valid (minimal, infeasible) IIS of size 2
        a = np.array([[1.0], [-1.0], [1.0]])
        b = np.array([1.0, -3.0, 2.0])
        sub = irreducible_infeasible_subset(a, b, [(-100, 100)])
        self.assertEqual(len(sub), 2)
        self.assertIn(1, sub)  # x >= 3 is in every IIS here (it conflicts with each upper bound)


if __name__ == "__main__":
    unittest.main()
