"""Compositional data analysis: Aitchison logratio transforms + logratio-normal distribution (Phase 6)."""

import unittest

import numpy as np

from pysp.stats import estimate
from pysp.stats.composition import AitchisonNormalDistribution as AitchisonNormal
from pysp.stats.composition import closure, clr, clr_inv, ilr, ilr_basis, ilr_inv


class LogratioTransformTest(unittest.TestCase):
    def setUp(self):
        self.x = closure(np.random.RandomState(0).gamma(2.0, size=(50, 4)))

    def test_ilr_round_trip(self):
        np.testing.assert_allclose(ilr_inv(ilr(self.x)), self.x, atol=1e-12)

    def test_clr_round_trip(self):
        np.testing.assert_allclose(clr_inv(clr(self.x)), self.x, atol=1e-12)

    def test_ilr_basis_is_orthonormal(self):
        v = ilr_basis(5)
        np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-12)

    def test_ilr_is_isometric(self):
        a, b = self.x[0:1], self.x[1:2]
        aitchison = np.linalg.norm(clr(a) - clr(b))  # Aitchison distance on the simplex
        euclidean = np.linalg.norm(ilr(a) - ilr(b))  # Euclidean distance in ilr space
        self.assertAlmostEqual(aitchison, euclidean, places=10)

    def test_closure_projects_to_simplex(self):
        c = closure(np.array([[2.0, 3.0, 5.0]]))
        np.testing.assert_allclose(c.sum(axis=1), 1.0)


class AitchisonNormalTest(unittest.TestCase):
    def setUp(self):
        self.true = AitchisonNormal(mean=np.array([0.5, -1.0, 0.3]), cov=np.diag([0.4, 0.6, 0.5]))

    def test_samples_lie_on_the_simplex(self):
        s = self.true.sampler(seed=1).sample(5000)
        np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue((s > 0).all())
        self.assertEqual(s.shape[1], 4)  # D-1=3 ilr coords -> D=4 parts

    def test_estimate_recovers_parameters(self):
        s = self.true.sampler(seed=2).sample(20000)
        fit = estimate([row for row in s], self.true.estimator())  # the pysp estimator/accumulator contract
        self.assertIsInstance(fit, AitchisonNormal)
        np.testing.assert_allclose(fit.mean, self.true.mean, atol=0.04)
        np.testing.assert_allclose(fit.cov, self.true.cov, atol=0.06)

    def test_log_density_peaks_at_the_center(self):
        center = self.true.mean_composition()
        edge = closure(np.array([0.9, 0.05, 0.03, 0.02]))[0]
        self.assertGreater(self.true.log_density(center), self.true.log_density(edge))

    def test_mean_composition_is_on_the_simplex(self):
        self.assertAlmostEqual(self.true.mean_composition().sum(), 1.0, places=10)

    def test_seq_log_density_matches_scalar(self):
        s = self.true.sampler(seed=3).sample(4)
        enc = self.true.dist_to_encoder().seq_encode([s[i] for i in range(4)])
        batch = self.true.seq_log_density(enc)
        self.assertEqual(batch.shape, (4,))
        self.assertAlmostEqual(self.true.log_density(s[0]), batch[0], places=10)


if __name__ == "__main__":
    unittest.main()
