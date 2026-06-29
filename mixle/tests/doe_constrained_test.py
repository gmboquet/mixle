"""Tests for constrained Bayesian optimization (WS-E).

``probability_of_feasibility`` is torch-free and tested directly; the GP-surrogate
``propose_next_constrained`` / ``constrained_minimize`` paths require torch and are skipped without it.
"""

import importlib.util
import unittest

import numpy as np

from mixle.doe import (
    constrained_minimize,
    probability_of_feasibility,
    propose_next_constrained,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ProbabilityOfFeasibilityTest(unittest.TestCase):
    def test_single_constraint_monotone_in_mean(self):
        # Lower predicted constraint value -> more likely feasible (c <= 0).
        pf = probability_of_feasibility(mean=np.array([[-2.0], [0.0], [2.0]]), std=np.ones((3, 1)))
        self.assertTrue(np.all((pf >= 0.0) & (pf <= 1.0)))
        self.assertGreater(pf[0], pf[1])
        self.assertGreater(pf[1], pf[2])
        np.testing.assert_allclose(pf[1], 0.5, atol=1e-9)  # mean exactly on the boundary

    def test_zero_std_is_deterministic(self):
        pf = probability_of_feasibility(mean=np.array([[-1.0], [1.0]]), std=np.zeros((2, 1)))
        np.testing.assert_array_equal(pf, np.array([1.0, 0.0]))

    def test_multiple_constraints_multiply(self):
        # Two independent constraints each at the boundary -> 0.5 * 0.5 = 0.25.
        pf = probability_of_feasibility(mean=np.zeros((1, 2)), std=np.ones((1, 2)))
        np.testing.assert_allclose(pf, [0.25], atol=1e-9)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ConstrainedLoopTest(unittest.TestCase):
    def test_propose_next_constrained_in_bounds(self):
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        rng = np.random.RandomState(0)
        x = rng.uniform(-2.0, 2.0, size=(8, 2))
        y = np.sum(x**2, axis=1)
        c = (x[:, 0] + x[:, 1] - 1.0).reshape(-1, 1)  # constraint: x0 + x1 <= 1
        nxt = np.asarray(
            propose_next_constrained(x, y, c, bounds, n_candidates=128, seed=1, fit_kwargs={"max_its": 60})
        )
        self.assertEqual(nxt.shape, (2,))
        self.assertTrue(np.all(nxt >= -2.0) and np.all(nxt <= 2.0))

    def test_mismatched_constraint_rows_raise(self):
        with self.assertRaises(ValueError):
            propose_next_constrained(
                np.zeros((4, 2)), np.zeros(4), np.zeros((3, 1)), [(0.0, 1.0), (0.0, 1.0)], n_candidates=8
            )

    def test_constrained_minimum_respects_an_active_constraint(self):
        # Minimize (x-2)^2 subject to x <= 0: unconstrained optimum is x=2 (infeasible);
        # the constrained optimum sits at the boundary x=0.
        bounds = [(-3.0, 3.0)]

        def objective(p):
            return float((p[0] - 2.0) ** 2)

        def constraint(p):
            return float(p[0])  # feasible when x <= 0

        result = constrained_minimize(
            objective,
            [constraint],
            bounds,
            n_init=6,
            n_iter=20,
            seed=0,
            n_candidates=256,
            fit_kwargs={"max_its": 60},
        )
        self.assertEqual(result.c.shape, (26, 1))
        self.assertEqual(result.feasible.shape, (26,))
        self.assertTrue(np.any(result.feasible))  # found feasible points
        self.assertLessEqual(result.best_x[0], 1e-6)  # best feasible point honors x <= 0
        self.assertLess(result.best_y, 4.5)  # and beats a far-from-boundary feasible value like x=-1 -> 9

    def test_requires_at_least_one_constraint(self):
        with self.assertRaises(ValueError):
            constrained_minimize(lambda p: float(p[0]), [], [(0.0, 1.0)], n_init=3, n_iter=1)


if __name__ == "__main__":
    unittest.main()
