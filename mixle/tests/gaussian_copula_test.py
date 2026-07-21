"""Gaussian copula: density vs the closed form, uniform marginals, inversion-estimator recovery."""

import unittest

import numpy as np
from scipy.stats import multivariate_normal, norm

from mixle.inference import estimate
from mixle.stats import GaussianCopulaDistribution


class GaussianCopulaTest(unittest.TestCase):
    def setUp(self):
        self.R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
        self.d = GaussianCopulaDistribution(self.R)

    def test_log_density_matches_closed_form(self):
        u = np.array([0.3, 0.7, 0.5])
        z = norm.ppf(u)
        ref = multivariate_normal.logpdf(z, np.zeros(3), self.R) - np.sum(norm.logpdf(z))
        self.assertAlmostEqual(self.d.log_density(u), ref, places=9)

    def test_seq_matches_scalar(self):
        u = np.random.RandomState(0).uniform(size=(6, 3))
        enc = self.d.dist_to_encoder().seq_encode(u)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(uu) for uu in u], atol=1e-12)

    def test_sampler_uniform_marginals_and_correlation(self):
        s = self.d.sampler(seed=0).sample(40000)
        np.testing.assert_allclose(s.mean(axis=0), 0.5, atol=0.02)  # uniform marginals
        zc = np.corrcoef(norm.ppf(np.clip(s, 1e-9, 1 - 1e-9)).T)
        np.testing.assert_allclose(zc, self.R, atol=0.03)  # normal-score correlation is R

    def test_inversion_estimator_recovers_correlation(self):
        s = list(self.d.sampler(seed=1).sample(40000))
        est = estimate(s, self.d.estimator())
        np.testing.assert_allclose(est.corr, self.R, atol=0.03)
        np.testing.assert_allclose(np.diag(est.corr), 1.0)  # valid correlation matrix

    def test_rejects_non_pd_corr(self):
        with self.assertRaises(ValueError):
            GaussianCopulaDistribution(np.array([[1.0, 1.5], [1.5, 1.0]]))  # det < 0, indefinite

    def test_rejects_not_pd_with_positive_determinant(self):
        # symmetric, unit diagonal, determinant > 0 (a determinant-sign check would accept this),
        # but eigenvalues [-0.6, -0.2, 3.8] -- not positive definite.
        bad = np.array([[1.0, -1.5, -1.5], [-1.5, 1.0, 1.2], [-1.5, 1.2, 1.0]])
        self.assertGreater(np.linalg.det(bad), 0.0)
        with self.assertRaises(ValueError):
            GaussianCopulaDistribution(bad)

    def test_rejects_non_symmetric_corr(self):
        with self.assertRaises(ValueError):
            GaussianCopulaDistribution(np.array([[1.0, 0.5, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    def test_rejects_non_unit_diagonal(self):
        with self.assertRaises(ValueError):
            GaussianCopulaDistribution(np.array([[2.0, 0.5], [0.5, 2.0]]))  # a covariance, not a correlation


if __name__ == "__main__":
    unittest.main()
