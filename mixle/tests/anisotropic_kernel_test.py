"""Anisotropic RBF field kernel: directional correlation for geological layering (Phase 6)."""

import unittest

import numpy as np

from mixle.ppl import RBF, AnisotropicRBF


class AnisotropicRBFTest(unittest.TestCase):
    def setUp(self):
        xs = np.linspace(0, 10, 11)
        self.grid = np.array([[a, b] for a in xs for b in xs])
        self.p0 = np.array([[5.0, 5.0]])

    def test_positive_definite_and_symmetric(self):
        c = AnisotropicRBF(ranges=(8.0, 1.0)).covariance(self.grid)
        self.assertTrue(np.allclose(c, c.T))
        self.assertGreater(np.linalg.eigvalsh(c).min(), -1e-8)

    def test_longer_correlation_along_major_axis(self):
        k = AnisotropicRBF(ranges=(8.0, 1.0), angle=0.0)
        cx = k.covariance(np.vstack([self.p0, [[8.0, 5.0]]]))[0, 1]  # lag 3 along x (major)
        cy = k.covariance(np.vstack([self.p0, [[5.0, 8.0]]]))[0, 1]  # lag 3 along y (minor)
        self.assertGreater(cx, cy)
        self.assertGreater(cx, 0.5)
        self.assertLess(cy, 0.1)

    def test_rotation_aligns_the_major_axis(self):
        k = AnisotropicRBF(ranges=(8.0, 1.0), angle=np.pi / 4)
        off = 3.0 / np.sqrt(2)
        c_diag = k.covariance(np.vstack([self.p0, [[5 + off, 5 + off]]]))[0, 1]  # along the 45-deg axis
        c_perp = k.covariance(np.vstack([self.p0, [[5 + off, 5 - off]]]))[0, 1]  # across it
        self.assertGreater(c_diag, c_perp)

    def test_isotropic_ranges_match_plain_rbf(self):
        aniso = AnisotropicRBF(ranges=(2.0, 2.0)).covariance(self.grid)
        np.testing.assert_allclose(aniso, RBF(lengthscale=2.0).covariance(self.grid), atol=1e-12)

    def test_full_metric_form(self):
        c = AnisotropicRBF(metric=np.array([[1.0, 0.3], [0.3, 2.0]])).covariance(self.grid)
        self.assertGreater(np.linalg.eigvalsh(c).min(), -1e-8)


if __name__ == "__main__":
    unittest.main()
