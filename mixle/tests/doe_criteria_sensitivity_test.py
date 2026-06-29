"""Extra optimality criteria (G/E/c) and sensitivity methods (RBD-FAST, DGSM)."""

import unittest

import numpy as np

from mixle.doe import (
    available_criteria,
    c_criterion,
    dgsm,
    e_criterion,
    fast_indices,
    g_criterion,
    optimal_design,
    polynomial_features,
)


def _ishigami(x, a=7.0, b=0.1):
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * (x[:, 2] ** 4) * np.sin(x[:, 0])


_ISHIGAMI_BOUNDS = [(-np.pi, np.pi)] * 3


class OptimalCriteriaTest(unittest.TestCase):
    def setUp(self):
        self.F = polynomial_features(2)
        self.cand = np.random.RandomState(0).uniform(-1, 1, (40, 2))
        self.M = self.F(self.cand).T @ self.F(self.cand)

    def test_registered(self):
        names = available_criteria()
        self.assertIn("g", names)
        self.assertIn("e", names)

    def test_e_is_min_eigenvalue(self):
        self.assertAlmostEqual(e_criterion(self.M), float(np.linalg.eigvalsh(self.M)[0]))

    def test_c_is_negative_quadratic_form(self):
        c = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(c_criterion(c)(self.M), float(-c @ np.linalg.solve(self.M, c)))

    def test_singular_information_is_minus_inf(self):
        singular = np.zeros((6, 6))
        self.assertEqual(g_criterion(singular), -np.inf)
        self.assertEqual(c_criterion(np.ones(6))(singular), -np.inf)

    def test_criteria_drive_optimal_design(self):
        for crit in ("e", "g", c_criterion([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])):
            pts = optimal_design(None, 8, candidates=self.cand, model=self.F, criterion=crit)
            self.assertEqual(pts.shape[0], 8)


class FastIndicesTest(unittest.TestCase):
    def test_matches_ishigami_first_order(self):
        # analytic first-order Sobol indices for Ishigami: ~[0.314, 0.442, 0.0]
        out = fast_indices(_ishigami, _ISHIGAMI_BOUNDS, n=2000, harmonics=6, seed=1)
        s1 = out["S1"]
        np.testing.assert_allclose(s1, [0.314, 0.442, 0.0], atol=0.06)
        self.assertLess(s1[2], 0.05)  # x3 has no first-order effect


class DgsmTest(unittest.TestCase):
    def test_flags_interaction_driven_input(self):
        out = dgsm(_ishigami, _ISHIGAMI_BOUNDS, n=4096, seed=1)
        nu = out["nu"]
        # DGSM (unlike first-order Sobol) sees x3, which only matters through its interaction with x1
        self.assertGreater(nu[2], 0.01)
        self.assertEqual(int(np.argmax(nu)), 1)  # x2 most influential
        self.assertAlmostEqual(float(out["importance"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
