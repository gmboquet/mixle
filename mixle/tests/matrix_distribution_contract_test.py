import unittest

import numpy as np

from mixle.stats.matrix.inverse_wishart import (
    InverseWishartDistribution,
    InverseWishartEstimator,
    InverseWishartMomentFitError,
)
from mixle.stats.matrix.matrix_normal import (
    MatrixNormalDistribution,
    MatrixNormalEstimator,
    MatrixNormalFitError,
)
from mixle.stats.matrix.wishart import (
    WishartDistribution,
    WishartEstimator,
    WishartFitError,
    _solve_wishart_df,
)


class MatrixParameterContractTestCase(unittest.TestCase):
    def test_matrix_normal_requires_finite_exact_symmetric_covariances(self):
        mean = np.zeros((2, 2))
        invalid = (
            (np.asarray([[1.0, 0.1], [0.2, 1.0]]), np.eye(2)),
            (np.asarray([[1.0, np.nan], [np.nan, 1.0]]), np.eye(2)),
            (np.eye(2), np.asarray([[1.0, 0.1], [0.2, 1.0]])),
        )
        for row_covar, col_covar in invalid:
            with self.subTest(), self.assertRaises(ValueError):
                MatrixNormalDistribution(mean, row_covar, col_covar)
        with self.assertRaises(ValueError):
            MatrixNormalDistribution(
                np.asarray([[0.0, np.inf], [0.0, 0.0]]),
                np.eye(2),
                np.eye(2),
            )

    def test_constructor_arrays_are_copied_and_exposed_read_only(self):
        matrix_mean = np.zeros((2, 2))
        row_covar = np.eye(2)
        col_covar = np.eye(2)
        matrix_normal = MatrixNormalDistribution(
            matrix_mean,
            row_covar,
            col_covar,
        )
        wishart_scale = np.eye(2)
        wishart = WishartDistribution(4.0, wishart_scale)
        inverse_scale = np.eye(2)
        inverse = InverseWishartDistribution(5.0, inverse_scale)

        matrix_mean[0, 0] = 100.0
        row_covar[0, 0] = 4.0
        col_covar[0, 0] = 4.0
        wishart_scale[0, 0] = 4.0
        inverse_scale[0, 0] = 4.0
        np.testing.assert_array_equal(matrix_normal.mean, np.zeros((2, 2)))
        np.testing.assert_array_equal(matrix_normal.row_covar, np.eye(2))
        np.testing.assert_array_equal(matrix_normal.col_covar, np.eye(2))
        np.testing.assert_array_equal(wishart.scale, np.eye(2))
        np.testing.assert_array_equal(inverse.scale, np.eye(2))
        for array in (
            matrix_normal.mean,
            matrix_normal.row_covar,
            matrix_normal.col_covar,
            wishart.scale,
            inverse.scale,
        ):
            with self.subTest(array=repr(array)), self.assertRaises(ValueError):
                array[0, 0] = 2.0


class MatrixEventGeometryContractTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_normal = MatrixNormalDistribution(
            np.zeros((2, 3)),
            np.eye(2),
            np.eye(3),
        )
        self.wishart = WishartDistribution(4.0, np.eye(2))
        self.inverse = InverseWishartDistribution(5.0, np.eye(2))

    def test_scalar_and_batch_scorers_reject_wrong_geometry(self):
        cases = (
            (self.matrix_normal, np.zeros((2, 2)), np.zeros((3, 2, 2))),
            (self.wishart, np.eye(3), np.zeros((3, 3, 3))),
            (self.inverse, np.eye(3), np.zeros((3, 3, 3))),
        )
        for distribution, scalar, batch in cases:
            with self.subTest(distribution=type(distribution).__name__):
                with self.assertRaises(ValueError):
                    distribution.log_density(scalar)
                with self.assertRaises(ValueError):
                    distribution.seq_log_density(batch)

    def test_encoders_bind_event_dimensions_and_finiteness(self):
        with self.assertRaises(ValueError):
            self.matrix_normal.dist_to_encoder().seq_encode([np.zeros((2, 3)), np.full((2, 3), np.nan)])
        with self.assertRaises(ValueError):
            self.wishart.dist_to_encoder().seq_encode([np.eye(3)])
        with self.assertRaises(ValueError):
            self.inverse.dist_to_encoder().seq_encode([np.asarray([[1.0, 0.2], [0.1, 1.0]])])
        self.assertEqual(
            self.matrix_normal.dist_to_encoder().seq_encode([]).shape,
            (0, 2, 3),
        )
        self.assertEqual(
            self.wishart.dist_to_encoder().seq_encode([]).shape,
            (0, 2, 2),
        )
        self.assertEqual(
            self.inverse.dist_to_encoder().seq_encode([]).shape,
            (0, 2, 2),
        )

    def test_matrix_accumulators_reject_events_and_weights_atomically(self):
        cases = (
            (
                self.matrix_normal.estimator().accumulator_factory().make(),
                np.zeros((2, 3)),
                np.full((2, 3), np.nan),
            ),
            (
                self.wishart.estimator().accumulator_factory().make(),
                np.eye(2),
                np.asarray([[1.0, 0.2], [0.1, 1.0]]),
            ),
            (
                self.inverse.estimator().accumulator_factory().make(),
                np.eye(2),
                np.asarray([[1.0, 0.2], [0.1, 1.0]]),
            ),
        )
        for accumulator, valid, invalid in cases:
            before = accumulator.value()
            with self.subTest(kind=type(accumulator).__name__, case="event"):
                with self.assertRaises(ValueError):
                    accumulator.update(invalid, 1.0, None)
                self._assert_statistics_equal(accumulator.value(), before)
            with self.subTest(kind=type(accumulator).__name__, case="weight"):
                with self.assertRaises(ValueError):
                    accumulator.update(valid, -1.0, None)
                self._assert_statistics_equal(accumulator.value(), before)
            with self.subTest(kind=type(accumulator).__name__, case="batch"):
                with self.assertRaises(ValueError):
                    accumulator.seq_update(
                        np.asarray([valid]),
                        np.asarray([np.nan]),
                        None,
                    )
                self._assert_statistics_equal(accumulator.value(), before)

    def _assert_statistics_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for left_value, right_value in zip(left, right):
            if isinstance(left_value, np.ndarray):
                np.testing.assert_array_equal(left_value, right_value)
            else:
                self.assertEqual(left_value, right_value)


class MatrixFitContractTestCase(unittest.TestCase):
    def test_matrix_normal_low_rank_and_invalid_solver_configuration_are_typed(self):
        with self.assertRaises((TypeError, ValueError)):
            MatrixNormalEstimator(2, 2, max_iter=0)
        with self.assertRaises((TypeError, ValueError)):
            MatrixNormalEstimator(2, 2, tol=np.nan)
        estimator = MatrixNormalEstimator(2, 2)
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(np.eye(2), 1.0, None)
        with self.assertRaisesRegex(MatrixNormalFitError, "non-identifiable"):
            estimator.estimate(None, accumulator.value())

    def test_matrix_normal_success_has_convergence_diagnostics(self):
        distribution = MatrixNormalDistribution(
            np.zeros((2, 2)),
            np.eye(2),
            np.eye(2),
        )
        data = distribution.sampler(seed=3).sample(100)
        estimator = distribution.estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(data, np.ones(len(data)), None)
        fitted = estimator.estimate(None, accumulator.value())
        self.assertTrue(fitted.fit_metadata["converged"])
        self.assertGreater(fitted.fit_metadata["iterations"], 0)
        self.assertEqual(fitted.fit_metadata["repairs"], ())

    def test_inverse_wishart_undefined_mean_fit_is_rejected(self):
        distribution = InverseWishartDistribution(2.0, np.eye(2))
        with self.assertRaisesRegex(
            InverseWishartMomentFitError,
            "df > p \\+ 1",
        ):
            distribution.estimator()
        with self.assertRaises(InverseWishartMomentFitError):
            InverseWishartEstimator(2, 3.0)
        lower_df_draw = InverseWishartDistribution(1.5, np.eye(2)).sampler(seed=7).sample()
        self.assertTrue(np.all(np.linalg.eigvalsh(lower_df_draw) > 0.0))

    def test_inverse_wishart_fit_is_labeled_as_a_moment_method(self):
        estimator = InverseWishartEstimator(2, 5.0)
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(np.eye(2), 1.0, None)
        fitted = estimator.estimate(None, accumulator.value())
        self.assertEqual(fitted.fit_metadata["method"], "mean-moment")
        np.testing.assert_allclose(fitted.scale, 2.0 * np.eye(2))

    def test_wishart_solver_rejects_invalid_state_and_certifies_fit(self):
        with self.assertRaises(WishartFitError):
            _solve_wishart_df(2.0, 1.0, 2)
        estimator = WishartEstimator(2, df=None)
        source = WishartDistribution(8.0, np.eye(2))
        data = source.sampler(seed=5).sample(500)
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(data, np.ones(len(data)), None)
        fitted = estimator.estimate(None, accumulator.value())
        self.assertTrue(fitted.fit_metadata["converged"])
        self.assertIn(
            fitted.fit_metadata["solver"],
            {"bracketed-bisection", "fixed"},
        )
        self.assertEqual(fitted.fit_metadata["repairs"], ())

    def test_estimators_reject_forged_or_empty_statistics(self):
        with self.assertRaises(WishartFitError):
            WishartEstimator(2).estimate(
                None,
                (np.zeros((2, 2)), 0.0, 0.0),
            )
        with self.assertRaises(ValueError):
            WishartEstimator(2).estimate(
                None,
                (np.asarray([[1.0, 0.2], [0.1, 1.0]]), 1.0, 0.0),
            )
        with self.assertRaises(InverseWishartMomentFitError):
            InverseWishartEstimator(2, 5.0).estimate(
                None,
                (np.zeros((2, 2)), 0.0),
            )


if __name__ == "__main__":
    unittest.main()
