"""Spherical (great-circle) field kernels: geodesic distance + PD GP/Matern priors on the sphere."""

import unittest

import numpy as np

from pysp.ppl import RBF, GreatCircleMatern, GreatCircleRBF, great_circle_distance

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

EARTH_KM = 6371.0088


class GreatCircleDistanceTest(unittest.TestCase):
    def test_known_geodesic_distances(self):
        self.assertAlmostEqual(great_circle_distance([0, 0], [0, 90], radius=EARTH_KM), 10007.54, delta=1.0)
        self.assertAlmostEqual(great_circle_distance([0, 0], [90, 0], radius=EARTH_KM), 10007.54, delta=1.0)
        self.assertAlmostEqual(great_circle_distance([0, 0], [0, 180], radius=EARTH_KM), 20015.09, delta=1.0)
        # New York -> London, a standard reference (~5570 km)
        self.assertAlmostEqual(
            great_circle_distance([40.7128, -74.0060], [51.5074, -0.1278], radius=EARTH_KM), 5570.0, delta=5.0
        )

    def test_pairwise_matrix_shape_and_symmetry(self):
        idx = np.array([[0.0, 0.0], [10.0, 20.0], [-30.0, 100.0]])
        d = great_circle_distance(idx, radius=EARTH_KM)
        self.assertEqual(d.shape, (3, 3))
        np.testing.assert_allclose(d, d.T)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-6)

    def test_two_columns_required(self):
        with self.assertRaises(ValueError):
            GreatCircleRBF().covariance(np.arange(5.0))  # 1-D index is not a lat/lon


class SphericalKernelPSDTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 40)))  # area-uniform on the sphere
        lon = rng.uniform(-180, 180, 40)
        self.idx = np.column_stack([lat, lon])

    def _check_psd(self, kernel):
        cov = kernel.covariance(self.idx)
        np.testing.assert_allclose(cov, cov.T)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), -1e-8)  # positive-definite on the sphere
        np.testing.assert_allclose(kernel.precision(self.idx) @ cov, np.eye(40), atol=1e-4)

    def test_rbf_is_psd(self):
        self._check_psd(GreatCircleRBF(lengthscale=2000.0, amplitude=1.0, radius=EARTH_KM))

    def test_matern_is_psd_for_each_nu(self):
        for nu in (0.5, 1.5, 2.5):
            self._check_psd(GreatCircleMatern(lengthscale=2000.0, nu=nu, radius=EARTH_KM))

    def test_invalid_nu_raises(self):
        with self.assertRaises(ValueError):
            GreatCircleMatern(nu=1.0, radius=EARTH_KM).covariance(self.idx)

    def test_small_angle_matches_planar_rbf(self):
        pts = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1]])  # ~11 km patch at the equator
        gc = GreatCircleRBF(lengthscale=20.0, radius=EARTH_KM).covariance(pts)
        planar = RBF(lengthscale=20.0).covariance(pts * 111.195)  # 1 deg ~ 111.195 km at the equator
        np.testing.assert_allclose(gc, planar, atol=2e-3)


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class SphericalFieldFitTest(unittest.TestCase):
    def test_fit_field_recovers_a_zonal_field_on_the_globe(self):
        from pysp.ppl import GaussianField, GaussianProxy, fit_field

        rng = np.random.RandomState(0)
        n = 60
        lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        lon = rng.uniform(-180, 180, n)
        idx = np.column_stack([lat, lon])
        truth = -30 * np.cos(np.radians(lat)) + 10  # warm equator, cold poles
        y = truth + rng.randn(n) * 1.5
        field = GaussianField(idx, GreatCircleMatern(lengthscale=4000.0, amplitude=20.0, nu=1.5, radius=EARTH_KM), name="T")
        post = fit_field(field, [GaussianProxy(y, slope=1.0, intercept=0.0, scale=1.5)], how="laplace")
        m = post.mean("T")
        self.assertGreater(np.corrcoef(m, truth)[0, 1], 0.9)
        self.assertTrue(np.all(post.sd("T") > 0))


if __name__ == "__main__":
    unittest.main()
