"""Tests for optimal experimental design (D/A/I criteria + Fedorov exchange) -- WS-E."""

import unittest

import numpy as np

from mixle.doe import (
    available_criteria,
    optimal_design,
    polynomial_features,
    register_criterion,
)
from mixle.doe.optimal import _get_criterion, a_criterion, d_criterion, i_criterion


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


if __name__ == "__main__":
    unittest.main()
