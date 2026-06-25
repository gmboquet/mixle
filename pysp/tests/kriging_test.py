"""Geostatistics: variograms and kriging (pysp.stats.kriging)."""

import unittest

import numpy as np
from scipy.spatial.distance import cdist

from pysp.analysis import (
    Variogram,
    calibrate_variance,
    empirical_variogram,
    fit_variogram,
    ordinary_kriging,
    universal_kriging,
)


class VariogramTest(unittest.TestCase):
    def test_gamma_monotone_bounded(self):
        vg = Variogram("spherical", nugget=0.2, psill=1.0, rng=5.0)
        h = np.linspace(0.01, 20, 50)
        g = vg.gamma(h)
        self.assertTrue(np.all(np.diff(g) >= -1e-9))  # non-decreasing
        self.assertAlmostEqual(g[-1], 1.2, delta=1e-6)  # nugget + psill at large h

    def test_empirical_variogram_rises(self):
        rng = np.random.RandomState(0)
        X = rng.uniform(0, 20, (300, 2))
        D = cdist(X, X)
        C = np.exp(-D / 4.0)
        field = np.linalg.cholesky(C + 1e-8 * np.eye(300)) @ rng.normal(0, 1, 300)
        ev = empirical_variogram(X, field)
        self.assertLess(ev["semivariance"][0], ev["semivariance"][-1])

    def test_fit_variogram_structure(self):
        rng = np.random.RandomState(1)
        X = rng.uniform(0, 20, (400, 2))
        D = cdist(X, X)
        C = np.exp(-D / 4.0)
        field = np.linalg.cholesky(C + 1e-8 * np.eye(400)) @ rng.normal(0, 1, 400)
        vg = fit_variogram(X, field, model="exponential")
        # variogram range is weakly identified; assert structure, not a tight range value
        self.assertGreater(vg.psill, 0)
        self.assertGreater(vg.rng, 0)
        self.assertLess(vg.nugget, vg.psill)  # correlated structure dominates the nugget

    def test_squared_exponential_is_gaussian_with_rbf_covariance(self):
        # 'squared_exponential' / 'rbf' are aliases of the Gaussian model; covariance is exp(-(h/rng)^2)
        h = np.array([0.0, 1.0, 2.0, 4.0])
        for name in ("squared_exponential", "squared-exponential", "rbf"):
            vg = Variogram(name, nugget=0.1, psill=2.0, rng=1.5)
            np.testing.assert_allclose(vg.cov_field(h), 2.0 * np.exp(-((h / 1.5) ** 2)))
            np.testing.assert_allclose(vg.gamma(h), Variogram("gaussian", 0.1, 2.0, 1.5).gamma(h))

    def test_squared_exponential_fit_and_krige_match_gaussian(self):
        rng = np.random.RandomState(2)
        X = rng.uniform(0, 10, (50, 2))
        z = np.sin(X[:, 0]) + np.cos(X[:, 1])
        q = np.array([[5.0, 5.0], [1.0, 9.0]])
        a = fit_variogram(X, z, model="gaussian")
        b = fit_variogram(X, z, model="squared_exponential")
        np.testing.assert_allclose([a.nugget, a.psill, a.rng], [b.nugget, b.psill, b.rng])
        pa = ordinary_kriging(X, z, a, q)["prediction"]
        pb = ordinary_kriging(X, z, Variogram("rbf", a.nugget, a.psill, a.rng), q)["prediction"]
        np.testing.assert_allclose(pa, pb)


class KrigingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.coords = rng.uniform(0, 10, (40, 2))
        self.z = np.sin(self.coords[:, 0]) + np.cos(self.coords[:, 1])
        self.vg = Variogram("exponential", nugget=0.0, psill=1.0, rng=3.0)

    def test_exact_interpolation_without_nugget(self):
        ok = ordinary_kriging(self.coords, self.z, self.vg, self.coords[:5])
        np.testing.assert_allclose(ok["prediction"], self.z[:5], atol=1e-6)
        np.testing.assert_allclose(ok["variance"], 0.0, atol=1e-6)

    def test_variance_grows_with_distance(self):
        near = ordinary_kriging(self.coords, self.z, self.vg, self.coords[:1] + 0.1)["variance"][0]
        far = ordinary_kriging(self.coords, self.z, self.vg, np.array([[100.0, 100.0]]))["variance"][0]
        self.assertGreater(far, near)

    def test_nugget_smooths(self):
        # with a nugget the prediction at a data location no longer equals the value exactly
        vg = Variogram("exponential", nugget=0.3, psill=1.0, rng=3.0)
        ok = ordinary_kriging(self.coords, self.z, vg, self.coords[:1])
        self.assertGreater(ok["variance"][0], 0.0)

    def test_universal_kriging_recovers_linear_trend(self):
        zlin = 2.0 + 0.5 * self.coords[:, 0] - 0.3 * self.coords[:, 1]
        q = np.array([[5.0, 5.0], [2.0, 8.0]])
        uk = universal_kriging(self.coords, zlin, self.vg, q, degree=1)
        true = 2.0 + 0.5 * q[:, 0] - 0.3 * q[:, 1]
        np.testing.assert_allclose(uk["prediction"], true, atol=1e-4)

    def test_heteroscedastic_noise_runs(self):
        rng = np.random.RandomState(1)
        noise = rng.uniform(0.01, 0.5, 40)
        ok = ordinary_kriging(
            self.coords, self.z, Variogram("exponential", 0.1, 1.0, 3.0), self.coords[:3], noise=noise
        )
        self.assertTrue(np.all(np.isfinite(ok["prediction"])))
        self.assertTrue(np.all(ok["variance"] >= 0))

    def test_anisotropy_changes_prediction(self):
        iso = Variogram("exponential", 0.0, 1.0, 3.0)
        aniso = Variogram("exponential", 0.0, 1.0, 3.0, anisotropy=(0.0, 4.0))
        q = np.array([[5.0, 5.0]])
        p_iso = ordinary_kriging(self.coords, self.z, iso, q)["prediction"][0]
        p_an = ordinary_kriging(self.coords, self.z, aniso, q)["prediction"][0]
        self.assertNotAlmostEqual(p_iso, p_an, places=4)


class CalibrationTest(unittest.TestCase):
    def test_recovers_underdispersion_factor(self):
        rng = np.random.RandomState(0)
        pv = rng.uniform(0.5, 2.0, 1000)
        # residuals are 1.5x too large for the stated variance -> need a ~2.25 multiplier
        resid = rng.normal(0, 1, 1000) * np.sqrt(pv) * 1.5
        c = calibrate_variance(pv, resid, target=0.9)
        self.assertAlmostEqual(c, 2.25, delta=0.5)

    def test_calibrated_coverage_hits_target(self):
        rng = np.random.RandomState(1)
        pv = rng.uniform(0.5, 2.0, 2000)
        resid = rng.normal(0, 1, 2000) * np.sqrt(pv) * 1.5
        c = calibrate_variance(pv, resid, target=0.9)
        from scipy.stats import norm

        z = norm.ppf(0.95)
        cov = np.mean(np.abs(resid) <= z * np.sqrt(c * pv))
        self.assertAlmostEqual(cov, 0.9, delta=0.03)


if __name__ == "__main__":
    unittest.main()
