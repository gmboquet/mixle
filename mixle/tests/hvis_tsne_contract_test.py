import unittest

import numpy as np
import pytest
import scipy.sparse

from mixle.utils.hvis.tsne import (
    _convergence_update,
    _validate_dense_joint_probability,
    _validate_sparse_joint_probability,
)

pytestmark = pytest.mark.fast


class JointProbabilityContractTest(unittest.TestCase):
    def setUp(self):
        self.valid = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])

    def test_dense_validation_normalizes_valid_input(self):
        p = _validate_dense_joint_probability(self.valid)
        self.assertAlmostEqual(float(p.sum()), 1.0)
        np.testing.assert_array_equal(np.diag(p), np.zeros(3))

    def test_dense_validation_rejects_invalid_probability_models(self):
        invalid = [
            self.valid[:, :2],
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            np.array([[1.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, -1.0], [-1.0, 0.0]]),
            np.array([[0.0, np.nan], [np.nan, 0.0]]),
            np.zeros((2, 2)),
        ]
        for p in invalid:
            with self.subTest(p=p), self.assertRaises(ValueError):
                _validate_dense_joint_probability(p)

    def test_sparse_validation_rejects_asymmetry_before_triangle_use(self):
        with self.assertRaises(ValueError):
            _validate_sparse_joint_probability(scipy.sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]))

    def test_sparse_validation_normalizes_valid_input(self):
        p = _validate_sparse_joint_probability(scipy.sparse.csr_matrix(self.valid))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        np.testing.assert_array_equal(p.diagonal(), np.zeros(3))


class ConvergenceContractTest(unittest.TestCase):
    def test_small_nonnegative_improvement_converges(self):
        converged, _, _ = _convergence_update(1.0, 1.0 - 1.0e-8, 1.0e-7, 0)
        self.assertTrue(converged)

    def test_worsening_never_counts_as_convergence_and_eventually_diverges(self):
        converged, previous, count = _convergence_update(1.0, 1.1, 1.0e-7, 0)
        self.assertFalse(converged)
        converged, previous, count = _convergence_update(previous, 1.2, 1.0e-7, count)
        self.assertFalse(converged)
        with self.assertRaises(RuntimeError):
            _convergence_update(previous, 1.3, 1.0e-7, count)

    def test_nonfinite_objective_fails(self):
        with self.assertRaises(FloatingPointError):
            _convergence_update(1.0, np.nan, 1.0e-7, 0)


if __name__ == "__main__":
    unittest.main()
