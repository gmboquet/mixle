"""Tests for multi-objective Bayesian optimization (ParEGO) -- WS-E.

``pareto_mask`` is torch-free and tested directly; the GP-surrogate ``multi_minimize`` path requires
torch and is skipped without it.

Also covers the MXR-080-0184 regression: NaN comparisons are always false, so a NaN-objective row could
never be *shown* to be dominated and was vacuously kept as Pareto-optimal; the scalarization step of the
optimizer had the same non-finite blind spot. Both the dominance check (``pareto_mask``) and the
scalarization step (``_scalarize``) now reject non-finite, empty, or wrong-width objective matrices
outright instead of computing over them.
"""

import importlib.util
import unittest

import numpy as np

from mixle.doe import multi_minimize, pareto_mask
from mixle.doe.multiobjective import _scalarize  # white-box: MXR-080-0184

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

    def test_multiple_dominance_relations_are_resolved_correctly(self):
        # Negative control for the MXR-080-0184 fix: an ordinary, all-finite matrix with several
        # known relations. A=(1,2) and B=(2,1) trade off and are non-dominated; C=(3,3) is dominated
        # by both A and B; D=(2,3) is dominated by A alone (1<=2 and 2<=3, strict on the first).
        y = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [2.0, 3.0]])
        np.testing.assert_array_equal(pareto_mask(y), [True, True, False, False])

    def test_nan_row_is_rejected_not_silently_marked_pareto(self):
        # MXR-080-0184: NaN comparisons are always false, so a NaN row could never be shown to be
        # dominated. Before the fix, all three rows below -- including the NaN row, whose second
        # objective (9.0) is worse than both other rows -- were marked Pareto-optimal.
        y = np.array([[1.0, 5.0], [4.0, 2.0], [np.nan, 9.0]])
        with self.assertRaises(ValueError):
            pareto_mask(y)

    def test_inf_row_is_rejected(self):
        y = np.array([[1.0, 5.0], [4.0, 2.0], [np.inf, 9.0]])
        with self.assertRaises(ValueError):
            pareto_mask(y)

    def test_empty_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            pareto_mask(np.empty((0, 2)))
        with self.assertRaises(ValueError):
            pareto_mask(np.empty((3, 0)))

    def test_wrong_width_rows_are_rejected(self):
        # Inconsistent row widths -- can't form a rectangular (N, M) objective matrix at all.
        with self.assertRaises(ValueError):
            pareto_mask([[1.0, 2.0], [3.0, 4.0, 5.0]])


class ScalarizeTest(unittest.TestCase):
    """White-box tests for the ParEGO scalarization step (MXR-080-0184)."""

    def test_well_formed_input_scalarizes_to_finite_values(self):
        y = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        scalar = _scalarize(y, weights=np.array([0.5, 0.5]), rho=0.05)
        self.assertEqual(scalar.shape, (3,))
        self.assertTrue(np.all(np.isfinite(scalar)))

    def test_nan_objective_is_rejected_not_scalarized(self):
        # Before the fix, a NaN in one column poisoned that column's min/max normalization, so every
        # row's scalarized score -- not just the offending row's -- came out NaN.
        y = np.array([[1.0, 5.0], [4.0, 2.0], [np.nan, 9.0]])
        with self.assertRaises(ValueError):
            _scalarize(y, weights=np.array([0.5, 0.5]), rho=0.05)

    def test_inf_objective_is_rejected(self):
        y = np.array([[1.0, 5.0], [4.0, 2.0], [np.inf, 9.0]])
        with self.assertRaises(ValueError):
            _scalarize(y, weights=np.array([0.5, 0.5]), rho=0.05)

    def test_width_mismatched_against_weights_is_rejected(self):
        y = np.array([[1.0, 5.0], [4.0, 2.0]])  # 2 objectives
        weights = np.array([0.3, 0.3, 0.4])  # 3 weights
        with self.assertRaises(ValueError):
            _scalarize(y, weights, rho=0.05)

    def test_empty_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            _scalarize(np.empty((0, 2)), weights=np.array([0.5, 0.5]), rho=0.05)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class MultiMinimizeTest(unittest.TestCase):
    def test_requires_at_least_two_objectives(self):
        with self.assertRaises(ValueError):
            multi_minimize([lambda p: float(p[0])], [(0.0, 1.0)], n_init=4, n_iter=1)

    def test_rejects_non_finite_objective_values(self):
        # MXR-080-0184 end-to-end: a mis-behaving objective returning NaN must stop the run with a
        # clear ValueError (raised by _scalarize before any GP step sees the poisoned scalar), not
        # silently propagate a non-finite score into the optimizer.
        def f1(p):
            return float(p[0] ** 2)

        def f_nan(_p):
            return float("nan")

        with self.assertRaises(ValueError):
            multi_minimize([f1, f_nan], [(0.0, 1.0)], n_init=4, n_iter=1, seed=0, n_candidates=16)

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
