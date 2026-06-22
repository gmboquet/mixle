"""Matrix normal MN(M, U, V): vec-MVN equivalence, sampling, and the flip-flop MLE."""

import unittest

import numpy as np
from scipy.stats import multivariate_normal as mvn

from pysp.inference import estimate
from pysp.stats import MatrixNormalDistribution


class MatrixNormalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.M = rng.randn(3, 2)
        self.U = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
        self.V = np.array([[1.4, 0.4], [0.4, 0.9]])
        self.d = MatrixNormalDistribution(self.M, self.U, self.V)

    def test_log_density_matches_vec_mvn(self):
        # vec stacks columns (Fortran order); cov(vec X) = V kron U
        x = np.random.RandomState(1).randn(3, 2)
        ref = mvn.logpdf(x.flatten("F"), self.M.flatten("F"), np.kron(self.V, self.U))
        self.assertAlmostEqual(self.d.log_density(x), ref, places=9)

    def test_seq_matches_scalar(self):
        xs = np.random.RandomState(2).randn(5, 3, 2)
        np.testing.assert_allclose(self.d.seq_log_density(xs), [self.d.log_density(x) for x in xs], atol=1e-10)

    def test_sampler_recovers_mean_and_kron_covariance(self):
        s = self.d.sampler(seed=1).sample(60000)
        np.testing.assert_allclose(s.mean(axis=0), self.M, atol=0.03)
        vecs = np.array([m.flatten("F") for m in s])
        np.testing.assert_allclose(np.cov(vecs.T), np.kron(self.V, self.U), atol=0.07)

    def test_flip_flop_mle_recovers_kronecker_product(self):
        est = estimate(list(self.d.sampler(seed=2).sample(20000)), self.d.estimator())
        np.testing.assert_allclose(est.mean, self.M, atol=0.04)
        # only the Kronecker product U (x) V is identifiable; the split is anchored at V[0,0]=1
        np.testing.assert_allclose(np.kron(est.row_covar, est.col_covar), np.kron(self.U, self.V), rtol=0.06, atol=0.06)
        self.assertAlmostEqual(est.col_covar[0, 0], 1.0)

    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            MatrixNormalDistribution(np.zeros((3, 2)), np.eye(2), np.eye(2))  # U must be (3,3)


if __name__ == "__main__":
    unittest.main()
