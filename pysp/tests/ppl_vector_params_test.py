"""pysp.ppl axis 2: vector / matrix-valued distribution parameters as inference targets.

An MVN's mean *vector* and full *covariance matrix* are inferable. The covariance uses a
Cholesky parameterization (Sigma = L Lᵀ, diagonal positive, off-diagonal real, assembled in
rebuild) so it is symmetric positive-definite by construction — reusing the scalar slot system.
"""

import unittest

import numpy as np

from pysp.ppl import MVN, free


class MVNParameterInferenceTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.true_mu = np.array([1.0, -2.0])
        self.true_cov = np.array([[2.0, 0.8], [0.8, 1.0]])
        self.X = [list(x) for x in rng.multivariate_normal(self.true_mu, self.true_cov, 3000)]

    def test_em_default_still_works(self):
        m = MVN(2).fit(self.X)
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.2))

    def test_mean_and_covariance_map(self):
        m = MVN(2, mean=free, cov=free).fit(self.X, how="map")
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.2))
        cov = np.asarray(m.params["cov"])
        self.assertTrue(np.allclose(cov, self.true_cov, atol=0.3))
        self.assertTrue(np.allclose(cov, cov.T))  # symmetric
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))  # positive-definite (Cholesky)

    def test_mean_and_covariance_ensemble(self):
        m = MVN(2, mean=free, cov=free).fit(
            self.X, how="ensemble", draws=800, burn=300, walkers=24, rng=np.random.RandomState(1)
        )
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.25))
        cov = np.asarray(m.params["cov"])
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))

    def test_mean_only(self):
        m = MVN(2, mean=free).fit(self.X, how="map")  # covariance fixed at I
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.2))


if __name__ == "__main__":
    unittest.main()
