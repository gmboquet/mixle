"""Tests for multi-objective Bayesian optimization (ParEGO) -- WS-E.

``pareto_mask`` is torch-free and tested directly; the GP-surrogate ``multi_minimize`` path requires
torch and is skipped without it.
"""

import importlib.util
import unittest

import numpy as np

from pysp.doe import multi_minimize, pareto_mask

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ParetoMaskTest(unittest.TestCase):
    def test_identifies_non_dominated_rows(self):
        # Minimization: (1,2) and (2,1) are non-dominated; (3,3) is dominated by both.
        y = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        np.testing.assert_array_equal(pareto_mask(y), [True, True, False])

    def test_strict_domination_only(self):
        # Duplicate optimal rows are both kept (neither strictly dominates the other).
        y = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_array_equal(pareto_mask(y), [True, True, False])

    def test_single_point_is_its_own_front(self):
        np.testing.assert_array_equal(pareto_mask([[5.0, 7.0]]), [True])

    def test_all_on_a_tradeoff_curve_are_kept(self):
        y = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        np.testing.assert_array_equal(pareto_mask(y), [True, True, True, True])


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class MultiMinimizeTest(unittest.TestCase):
    def test_requires_at_least_two_objectives(self):
        with self.assertRaises(ValueError):
            multi_minimize([lambda p: float(p[0])], [(0.0, 1.0)], n_init=4, n_iter=1)

    def test_recovers_a_spread_pareto_front_on_competing_objectives(self):
        # f1 minimized at x=0, f2 at x=1: the Pareto front is the whole interval [0, 1].
        bounds = [(0.0, 1.0)]

        def f1(p):
            return float(p[0] ** 2)

        def f2(p):
            return float((p[0] - 1.0) ** 2)

        result = multi_minimize(
            [f1, f2], bounds, n_init=8, n_iter=20, seed=0, n_candidates=128, fit_kwargs={"max_its": 50}
        )
        self.assertEqual(result.y.shape, (28, 2))
        self.assertEqual(result.pareto_mask.shape, (28,))
        # The mask must be self-consistent: re-deriving the front from y gives the same set.
        np.testing.assert_array_equal(result.pareto_mask, pareto_mask(result.y))
        # The front should be non-trivial and span the trade-off (some point near each objective's min).
        self.assertGreaterEqual(result.pareto_x.shape[0], 2)
        self.assertLess(float(np.min(result.pareto_x[:, 0])), 0.25)  # a point favoring f1
        self.assertGreater(float(np.max(result.pareto_x[:, 0])), 0.75)  # a point favoring f2


if __name__ == "__main__":
    unittest.main()
