"""Boundary-contract regressions for the shared NumPy stationary kernels."""

from __future__ import annotations

import unittest

import numpy as np

from mixle.models._kernels import (
    exponential_from_scaled_dist,
    matern32_from_scaled_dist,
    matern52_from_scaled_dist,
    rbf_from_scaled_sqdist,
    stationary_kernel,
)


class StationaryKernelContractTest(unittest.TestCase):
    def test_valid_matrix_inputs_and_exact_matern_diagonal(self):
        x = np.array([[0.0], [2.0]])
        expected = 4.0
        for name in ("rbf", "matern32", "matern52"):
            covariance = stationary_kernel(x, x, 1.5, 2.0, name)
            self.assertEqual(covariance.shape, (2, 2))
            self.assertTrue(np.array_equal(np.diag(covariance), np.array([expected, expected])))

    def test_rejects_invalid_point_geometry(self):
        good = np.ones((2, 2))
        invalid = (
            1.0,
            np.ones((1, 1, 1)),
            np.empty((0, 2)),
            np.empty((2, 0)),
            np.array([[0.0, np.nan]]),
        )
        for points in invalid:
            with self.subTest(shape=np.shape(points)), self.assertRaises(ValueError):
                stationary_kernel(points, good, 1.0, 1.0, "rbf")
        self.assertEqual(stationary_kernel(np.ones(3), np.ones(2), 1.0, 1.0, "rbf").shape, (3, 2))
        with self.assertRaisesRegex(ValueError, "same feature width"):
            stationary_kernel(np.ones((2, 1)), good, 1.0, 1.0, "rbf")

    def test_rejects_invalid_hyperparameters(self):
        x = np.ones((2, 1))
        for value in (0, -1, np.nan, np.inf, True, [1.0]):
            with self.subTest(lengthscale=value), self.assertRaises(ValueError):
                stationary_kernel(x, x, value, 1.0, "rbf")
            with self.subTest(amplitude=value), self.assertRaises(ValueError):
                stationary_kernel(x, x, 1.0, value, "rbf")
        with self.assertRaisesRegex(ValueError, "unknown kernel"):
            stationary_kernel(x, x, 1.0, 1.0, "unknown")

    def test_shape_helpers_validate_distances_and_amplitude(self):
        helpers = (
            rbf_from_scaled_sqdist,
            matern32_from_scaled_dist,
            matern52_from_scaled_dist,
            exponential_from_scaled_dist,
        )
        for helper in helpers:
            with self.subTest(helper=helper.__name__), self.assertRaises(ValueError):
                helper(np.array([0.0, np.nan]), 1.0)
            with self.subTest(helper=helper.__name__), self.assertRaises(ValueError):
                helper(np.array([-1.0, 0.0]), 1.0)
            with self.subTest(helper=helper.__name__), self.assertRaises(ValueError):
                helper(np.array([0.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
