"""Tests for optimal experimental design (D/A/I criteria + Fedorov exchange) -- WS-E."""

import unittest

import numpy as np

from mixle.doe import (
    available_criteria,
    optimal_design,
    polynomial_features,
    register_criterion,
)
from mixle.doe.optimal import (
    InfeasibleDesignError,
    _get_criterion,
    a_criterion,
    c_criterion,
    d_criterion,
    g_criterion,
    i_criterion,
)


class PolynomialFeaturesTest(unittest.TestCase):
    def test_linear_has_intercept_and_one_col_per_dim(self):
        f = polynomial_features(1)
        out = f(np.array([[2.0, 3.0]]))
        np.testing.assert_array_equal(out, np.array([[1.0, 2.0, 3.0]]))

    def test_quadratic_includes_squares_and_interactions(self):
        # d=2, degree=2 -> [1, x1, x2, x1^2, x1 x2, x2^2]
        f = polynomial_features(2)
        out = f(np.array([[2.0, 3.0]]))
        np.testing.assert_allclose(out, np.array([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]))

    def test_bias_can_be_dropped(self):
        f = polynomial_features(1, bias=False)
        self.assertEqual(f(np.zeros((5, 3))).shape, (5, 3))

    def test_degree_must_be_positive(self):
        with self.assertRaises(ValueError):
            polynomial_features(0)


class CriterionRegistryTest(unittest.TestCase):
    def test_builtin_names_and_aliases(self):
        names = available_criteria()
        for expected in ("d", "a", "i", "d_optimal", "a-optimal"):
            self.assertIn(expected, names)
        self.assertIs(_get_criterion("D"), d_criterion)  # case-insensitive
        self.assertIs(_get_criterion("a"), a_criterion)
        self.assertIs(_get_criterion("i"), i_criterion)

    def test_singular_information_is_negative_infinity(self):
        singular = np.zeros((2, 2))
        self.assertEqual(d_criterion(singular), -np.inf)
        self.assertEqual(a_criterion(singular), -np.inf)

    def test_d_criterion_rewards_larger_determinant(self):
        small = np.diag([1.0, 1.0])
        large = np.diag([4.0, 4.0])
        self.assertGreater(d_criterion(large), d_criterion(small))

    def test_unknown_criterion_lists_registered(self):
        with self.assertRaises(ValueError) as ctx:
            _get_criterion("banana")
        self.assertIn("banana", str(ctx.exception))

    def test_non_callable_rejected(self):
        with self.assertRaises(TypeError):
            register_criterion("bad", object())


class InformationMatrixValidationTest(unittest.TestCase):
    """MXR-080-0185: criteria must reject an information matrix that could never be F.T @ F."""

    def test_negative_definite_information_rejected_by_d_and_a(self):
        # -I_2: previously d_criterion returned 0 and a_criterion returned positive merit 2 --
        # a mathematically impossible information matrix silently outscored valid ones.
        neg_identity = -np.eye(2)
        with self.assertRaises(ValueError):
            d_criterion(neg_identity)
        with self.assertRaises(ValueError):
            a_criterion(neg_identity)

    def test_asymmetric_information_rejected(self):
        asym = np.array([[1.0, 2.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            d_criterion(asym)
        with self.assertRaises(ValueError):
            a_criterion(asym)

    def test_non_finite_information_rejected(self):
        nan_info = np.array([[np.nan, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            d_criterion(nan_info)
        inf_info = np.array([[np.inf, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            a_criterion(inf_info)

    def test_non_square_information_rejected(self):
        bad = np.ones((2, 3))
        with self.assertRaises(ValueError):
            d_criterion(bad)

    def test_valid_psd_matrix_still_scores_correctly(self):
        # Negative control: a genuine, hand-computable PSD information matrix must not be
        # rejected and must score exactly as before -- diag([4, 4]) has det=16, trace(inv)=0.5.
        info = np.diag([4.0, 4.0])
        self.assertAlmostEqual(d_criterion(info), float(np.log(16.0)))
        self.assertAlmostEqual(a_criterion(info), -0.5)

    def test_singular_psd_matrix_is_still_minus_inf_not_rejected(self):
        # A singular-but-PSD matrix (e.g. all zeros) is a legitimate degenerate design, not an
        # invalid one -- it must still return -inf rather than raise.
        singular = np.zeros((3, 3))
        self.assertEqual(d_criterion(singular), -np.inf)
        self.assertEqual(a_criterion(singular), -np.inf)

    def test_reference_dimension_mismatch_rejected_by_i_and_g(self):
        info = np.diag([1.0, 1.0])
        bad_ref = np.ones((3, 5))  # 5 columns != info's 2x2 dimension
        with self.assertRaises(ValueError):
            i_criterion(info, ref=bad_ref)
        with self.assertRaises(ValueError):
            g_criterion(info, ref=bad_ref)

    def test_reference_matching_dimension_is_accepted(self):
        info = np.diag([1.0, 1.0])
        good_ref = np.ones((3, 2))  # 2 columns matches info's 2x2 dimension
        self.assertNotEqual(i_criterion(info, ref=good_ref), -np.inf)
        self.assertNotEqual(g_criterion(info, ref=good_ref), -np.inf)

    def test_contrast_dimension_mismatch_rejected(self):
        crit = c_criterion([1.0, 0.0, 0.0])  # length-3 contrast
        info = np.diag([1.0, 1.0])  # 2x2 info
        with self.assertRaises(ValueError):
            crit(info)

    def test_contrast_matching_dimension_is_accepted(self):
        crit = c_criterion([1.0, 0.0])
        info = np.diag([1.0, 1.0])
        self.assertAlmostEqual(crit(info), -1.0)  # -c' M^-1 c = -(1*1*1 + 0) = -1


class OptimalDesignTest(unittest.TestCase):
    def test_d_optimal_linear_1d_concentrates_at_extremes(self):
        # For a linear model on [-1, 1], the D-optimal design sits at the endpoints.
        design = optimal_design([(-1.0, 1.0)], n=4, criterion="D", n_candidates=128, seed=0)
        self.assertEqual(design.shape, (4, 1))
        # Every chosen point should be near a boundary, not the interior.
        self.assertTrue(np.all(np.abs(design[:, 0]) > 0.6))
        # And both ends are represented.
        self.assertTrue(np.any(design[:, 0] < -0.6) and np.any(design[:, 0] > 0.6))

    def test_d_optimal_beats_random_subset_on_logdet(self):
        rng = np.random.RandomState(0)
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        model = polynomial_features(1)
        # Shared candidate pool so the comparison is apples-to-apples.
        from mixle.doe import sobol_design

        pool = sobol_design(bounds, 256, seed=1)
        design = optimal_design(None, n=6, candidates=pool, criterion="D", seed=0)
        f_opt = model(design)
        opt_logdet = np.linalg.slogdet(f_opt.T @ f_opt)[1]
        rand_logdets = []
        for _ in range(20):
            sub = pool[rng.choice(pool.shape[0], size=6, replace=False)]
            fr = model(sub)
            rand_logdets.append(np.linalg.slogdet(fr.T @ fr)[1])
        self.assertGreaterEqual(opt_logdet, max(rand_logdets) - 1e-9)

    def test_a_and_i_criteria_run_and_stay_in_pool(self):
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        for crit in ("A", "I"):
            design = optimal_design(bounds, n=8, criterion=crit, n_candidates=96, n_restarts=3, seed=2)
            self.assertEqual(design.shape, (8, 2))
            self.assertTrue(np.all(design >= -1e-9) and np.all(design <= 1.0 + 1e-9))

    def test_explicit_candidates_returns_subset_of_pool(self):
        pool = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        design = optimal_design(None, n=2, candidates=pool, criterion="D", seed=0)
        self.assertEqual(design.shape, (2, 1))
        for row in design:
            self.assertTrue(np.any(np.all(np.isclose(pool, row), axis=1)))

    def test_underdetermined_n_raises(self):
        # Linear model in 2-D has 3 parameters; n=2 < 3 is singular.
        with self.assertRaises(ValueError):
            optimal_design([(0.0, 1.0), (0.0, 1.0)], n=2, criterion="D", seed=0)

    def test_requires_bounds_or_candidates(self):
        with self.assertRaises(ValueError):
            optimal_design(None, n=4, criterion="D")

    def test_custom_criterion_is_registered(self):
        def const_crit(info, *, ref=None):
            return 0.0

        register_criterion("const-test-crit", const_crit, aliases=("ctc",))
        try:
            self.assertIs(_get_criterion("CTC"), const_crit)
        finally:
            from mixle.doe.optimal import _CRITERIA

            _CRITERIA.pop("const-test-crit", None)
            _CRITERIA.pop("ctc", None)


class RankDeficientDesignTest(unittest.TestCase):
    """MXR-080-0186: n >= p (row count) is not sufficient; the model matrix must have rank >= p."""

    def test_constant_rank_one_pool_raises_infeasible_not_assertion(self):
        # Every candidate point identical -> the linear model matrix (bias + 2 zero columns) has
        # rank 1 regardless of pool size. n=3 satisfies n >= p=3 by row count, so only a real rank
        # check (not n >= p) catches this; previously every restart returned -inf, best_sel stayed
        # None, and the sole guard was a bare `assert` -> a raw, uninformative AssertionError.
        pool = np.zeros((10, 2))
        with self.assertRaises(InfeasibleDesignError):
            optimal_design(None, n=3, candidates=pool, criterion="D", seed=0)

    def test_infeasible_error_is_a_value_error(self):
        # Catchable as a plain ValueError for callers that don't know the specific subclass.
        pool = np.zeros((10, 2))
        with self.assertRaises(ValueError):
            optimal_design(None, n=3, candidates=pool, criterion="D", seed=0)

    def test_rank_deficient_pool_message_names_the_rank_and_parameter_count(self):
        pool = np.zeros((10, 2))
        with self.assertRaises(InfeasibleDesignError) as ctx:
            optimal_design(None, n=3, candidates=pool, criterion="D", seed=0)
        msg = str(ctx.exception)
        self.assertIn("rank 1", msg)
        self.assertIn("3 model parameters", msg)

    def test_duplicated_two_point_pool_is_also_rank_deficient(self):
        # Less trivially degenerate than an all-identical pool: 10 rows but only 2 distinct
        # points, so the 2-D linear model matrix (bias + 2 vars, p=3) has rank <= 2 < p no matter
        # how the exchange search subsets it.
        pool = np.tile(np.array([[0.0, 0.0], [1.0, 1.0]]), (5, 1))
        with self.assertRaises(InfeasibleDesignError):
            optimal_design(None, n=3, candidates=pool, criterion="D", seed=0)

    def test_well_posed_pool_still_succeeds(self):
        # Negative control: a normal, rank-sufficient candidate pool still runs the exchange
        # search successfully and returns a real, valid optimal design.
        design = optimal_design([(-1.0, 1.0), (-1.0, 1.0)], n=6, criterion="D", n_candidates=64, seed=3)
        self.assertEqual(design.shape, (6, 2))

    def test_well_posed_explicit_candidates_still_succeeds(self):
        pool = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        design = optimal_design(None, n=2, candidates=pool, criterion="D", seed=0)
        self.assertEqual(design.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
