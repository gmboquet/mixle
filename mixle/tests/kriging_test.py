"""Geostatistics: variograms and kriging (mixle.stats.kriging)."""

import unittest

import numpy as np
from scipy.spatial.distance import cdist

from mixle.analysis import (
    Variogram,
    calibrate_variance,
    empirical_variogram,
    fit_variogram,
    ordinary_kriging,
    universal_kriging,
)
from mixle.analysis.kriging import _clip_variance, _krige_solve


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

    def test_empirical_variogram_retains_pairs_at_the_default_max_dist_boundary(self):
        # MXR-080-0101 exact repro: 3 collinear, equally spaced points have pairwise distances
        # 1, 1, 2. The default max_dist is half the largest distance (1.0), so both distance-1 pairs
        # sit exactly on the outer bin edge. np.digitize's half-open-right bins used to classify them
        # as "beyond the last bin" and silently discard every pair -- fit_variogram then crashed on
        # an empty lag array. Both distance-1 pairs must now be retained in the last bin.
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        values = np.array([1.0, 2.0, 1.5])
        ev = empirical_variogram(coords, values)
        self.assertFalse(bool(ev["insufficient_evidence"]))
        self.assertEqual(ev["reason"], "")
        self.assertEqual(int(ev["count"].sum()), 2)  # both distance-1 pairs retained, not discarded
        # mean(0.5*(1.0-2.0)**2, 0.5*(2.0-1.5)**2) = mean(0.5, 0.125) = 0.3125, hand-computed exactly
        np.testing.assert_allclose(ev["semivariance"], [0.3125])

    def test_empirical_variogram_single_point_is_insufficient_evidence(self):
        # A single point has no pairs at all -- a more extreme case of "every pair discarded" than
        # the boundary bug above (dist.max() on an empty array used to crash immediately). Must now
        # return a typed insufficient-evidence result instead of crashing or fabricating a bin.
        ev = empirical_variogram(np.array([[0.0, 0.0]]), np.array([1.0]))
        self.assertTrue(bool(ev["insufficient_evidence"]))
        self.assertNotEqual(ev["reason"], "")
        self.assertEqual(ev["lag"].size, 0)
        self.assertEqual(ev["semivariance"].size, 0)
        self.assertEqual(ev["count"].size, 0)

    def test_fit_variogram_raises_clearly_on_the_boundary_repro(self):
        # Same audit repro as above, through fit_variogram: after the binning fix there is exactly 1
        # populated bin, still too few to identify a 3-parameter model (nugget, partial sill, range).
        # Must fail with a clear, typed ValueError instead of the pre-fix bare crash on an empty lag
        # array (`ValueError: zero-size array to reduction operation maximum which has no identity`).
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        values = np.array([1.0, 2.0, 1.5])
        with self.assertRaisesRegex(ValueError, "populated lag bin"):
            fit_variogram(coords, values)

    def test_fit_variogram_raises_clearly_on_a_single_point(self):
        with self.assertRaisesRegex(ValueError, "cannot fit a variogram"):
            fit_variogram(np.array([[0.0, 0.0]]), np.array([1.0]))

    def test_fit_variogram_still_fits_a_well_populated_point_set(self):
        # Negative control: a normal, well-populated point set is unaffected by the binning fix and
        # the new minimum-populated-bins check.
        rng = np.random.RandomState(5)
        X = rng.uniform(0, 20, (200, 2))
        D = cdist(X, X)
        C = np.exp(-D / 4.0)
        field = np.linalg.cholesky(C + 1e-8 * np.eye(200)) @ rng.normal(0, 1, 200)
        vg = fit_variogram(X, field, model="exponential")
        self.assertGreater(vg.psill, 0)
        self.assertGreater(vg.rng, 0)

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


class VariogramValidationTest(unittest.TestCase):
    """MXR-080-0102: Variogram construction must validate model/nugget/psill/rng/nu/anisotropy."""

    def test_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "model must be one of"):
            Variogram("bogus", nugget=0.1, psill=1.0, rng=3.0)

    def test_rejects_negative_nugget(self):
        with self.assertRaisesRegex(ValueError, "nugget"):
            Variogram("exponential", nugget=-0.1, psill=1.0, rng=3.0)

    def test_rejects_non_finite_nugget(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "nugget"):
                Variogram("exponential", nugget=bad, psill=1.0, rng=3.0)

    def test_rejects_negative_psill(self):
        with self.assertRaisesRegex(ValueError, "psill"):
            Variogram("exponential", nugget=0.1, psill=-1.0, rng=3.0)

    def test_rejects_non_positive_range(self):
        for bad_rng in (0.0, -3.0):
            with self.assertRaisesRegex(ValueError, "rng"):
                Variogram("exponential", nugget=0.1, psill=1.0, rng=bad_rng)

    def test_rejects_non_finite_range(self):
        with self.assertRaisesRegex(ValueError, "rng"):
            Variogram("exponential", nugget=0.1, psill=1.0, rng=float("inf"))

    def test_rejects_non_positive_matern_smoothness(self):
        for bad_nu in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "nu"):
                Variogram("matern", nugget=0.1, psill=1.0, rng=3.0, nu=bad_nu)

    def test_rejects_zero_anisotropy_ratio(self):
        # MXR-080-0102 exact repro: a zero anisotropy ratio divides coordinates by zero inside
        # _transform, producing NaN coordinates that previously propagated all the way through to
        # NaN kriging predictions and variances instead of being rejected here at construction.
        with self.assertRaisesRegex(ValueError, "anisotropy ratio"):
            Variogram("exponential", nugget=0.0, psill=1.0, rng=3.0, anisotropy=(0.0, 0.0))

    def test_rejects_negative_anisotropy_ratio(self):
        with self.assertRaisesRegex(ValueError, "anisotropy ratio"):
            Variogram("exponential", nugget=0.0, psill=1.0, rng=3.0, anisotropy=(0.0, -2.0))

    def test_rejects_non_finite_anisotropy_angle(self):
        with self.assertRaisesRegex(ValueError, "anisotropy angle"):
            Variogram("exponential", nugget=0.0, psill=1.0, rng=3.0, anisotropy=(float("nan"), 2.0))

    def test_rejects_malformed_anisotropy_tuple(self):
        with self.assertRaisesRegex(ValueError, "anisotropy"):
            Variogram("exponential", nugget=0.0, psill=1.0, rng=3.0, anisotropy=(1.0, 2.0, 3.0))

    def test_valid_parameters_still_construct(self):
        # Negative control: legitimate parameters across every model name, including anisotropy,
        # still construct without error.
        for model in ("spherical", "exponential", "gaussian", "squared_exponential", "rbf", "matern"):
            vg = Variogram(model=model, nugget=0.1, psill=1.0, rng=3.0, nu=1.5, anisotropy=(0.3, 2.0))
            self.assertEqual(vg.model, model)


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


class KrigingValidationTest(unittest.TestCase):
    """MXR-080-0102: coords/values/query/drift/noise must be validated before the solve."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.coords = rng.uniform(0, 10, (40, 2))
        self.z = np.sin(self.coords[:, 0]) + np.cos(self.coords[:, 1])
        self.vg = Variogram("exponential", nugget=0.1, psill=1.0, rng=3.0)

    def test_rejects_empty_coords(self):
        with self.assertRaisesRegex(ValueError, "coords"):
            ordinary_kriging(np.empty((0, 2)), np.empty((0,)), self.vg, self.coords[:1])

    def test_rejects_mismatched_values_length(self):
        with self.assertRaisesRegex(ValueError, "values"):
            ordinary_kriging(self.coords, self.z[:5], self.vg, self.coords[:1])

    def test_rejects_non_finite_coords(self):
        bad_coords = self.coords.copy()
        bad_coords[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "coords"):
            ordinary_kriging(bad_coords, self.z, self.vg, self.coords[:1])

    def test_rejects_non_finite_values(self):
        bad_z = self.z.copy()
        bad_z[0] = np.inf
        with self.assertRaisesRegex(ValueError, "values"):
            ordinary_kriging(self.coords, bad_z, self.vg, self.coords[:1])

    def test_rejects_query_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "query"):
            ordinary_kriging(self.coords, self.z, self.vg, np.array([[1.0, 2.0, 3.0]]))

    def test_rejects_empty_query(self):
        with self.assertRaisesRegex(ValueError, "query"):
            ordinary_kriging(self.coords, self.z, self.vg, np.empty((0, 2)))

    def test_rejects_non_finite_query(self):
        with self.assertRaisesRegex(ValueError, "query"):
            ordinary_kriging(self.coords, self.z, self.vg, np.array([[np.nan, 1.0]]))

    def test_universal_kriging_rejects_query_dimension_mismatch(self):
        # A query with the wrong dimensionality would otherwise build a drift0 basis with a
        # different column count than drift and fail deep inside the linear solve with a confusing
        # shape error; caught explicitly, before _poly_basis is even called.
        with self.assertRaisesRegex(ValueError, "query"):
            universal_kriging(self.coords, self.z, self.vg, np.array([[1.0, 2.0, 3.0]]))

    def test_krige_solve_rejects_mismatched_drift0_shape(self):
        drift = np.ones((40, 3))
        drift0 = np.ones((1, 2))  # wrong column count relative to drift
        with self.assertRaisesRegex(ValueError, "drift0"):
            _krige_solve(self.coords, self.z, self.vg, np.array([[5.0, 5.0]]), drift=drift, drift0=drift0, noise=None)

    def test_rejects_negative_noise(self):
        with self.assertRaisesRegex(ValueError, "noise"):
            ordinary_kriging(self.coords, self.z, self.vg, self.coords[:1], noise=-np.ones(40))

    def test_rejects_mismatched_noise_shape(self):
        with self.assertRaisesRegex(ValueError, "noise"):
            ordinary_kriging(self.coords, self.z, self.vg, self.coords[:1], noise=np.ones(5))

    def test_rejects_non_finite_noise(self):
        with self.assertRaisesRegex(ValueError, "noise"):
            ordinary_kriging(self.coords, self.z, self.vg, self.coords[:1], noise=np.full(40, np.nan))

    def test_valid_geometry_still_krige_correctly(self):
        # Negative control: legitimate, well-posed inputs (including heteroscedastic noise and a
        # quadratic drift) are unaffected by the new validation.
        q = np.array([[5.0, 5.0], [2.0, 8.0]])
        ok = ordinary_kriging(self.coords, self.z, self.vg, q, noise=np.full(40, 0.05))
        self.assertTrue(np.all(np.isfinite(ok["prediction"])))
        self.assertTrue(np.all(np.isfinite(ok["variance"])))
        self.assertTrue(np.all(ok["variance"] >= 0))
        uk = universal_kriging(self.coords, self.z, self.vg, q, degree=2)
        self.assertTrue(np.all(np.isfinite(uk["prediction"])))
        self.assertTrue(np.all(uk["variance"] >= 0))


class VarianceClipTest(unittest.TestCase):
    """MXR-080-0102: distinguish small numerical roundoff from a materially negative variance."""

    def test_tiny_roundoff_clips_to_zero(self):
        # -1e-15 relative to scale=1.0 is many orders of magnitude below the tolerance -- ordinary
        # floating-point roundoff, not a sign of an invalid solve -- and should clip cleanly.
        out = _clip_variance(np.array([-1e-15, 0.0, 0.2, 1.0]), scale=1.0)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.2, 1.0])

    def test_materially_negative_variance_raises(self):
        with self.assertRaisesRegex(ValueError, "materially negative"):
            _clip_variance(np.array([-0.5, 0.2]), scale=1.0)

    def test_tolerance_scales_with_problem_magnitude_not_an_absolute_constant(self):
        # The same absolute -1e-4 is roundoff-scale noise against a covariance scale of 1e6 but a
        # material failure against a scale of 1.0 -- the tolerance must be relative to the problem's
        # own scale, not a fixed absolute constant.
        out = _clip_variance(np.array([-1e-4]), scale=1e6)
        np.testing.assert_allclose(out, [0.0])
        with self.assertRaisesRegex(ValueError, "materially negative"):
            _clip_variance(np.array([-1e-4]), scale=1.0)

    def test_realistic_roundoff_at_exact_data_points_clips_cleanly(self):
        # Integration-level check, not synthetic: predicting AT the data locations with no nugget is
        # where roundoff bites hardest (the true variance is exactly 0). The real ordinary_kriging
        # call must not raise, confirming the tolerance comfortably covers real floating-point
        # roundoff rather than just the hand-picked value above.
        rng = np.random.RandomState(3)
        coords = rng.uniform(0, 10, (300, 2))
        z = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
        vg = Variogram("exponential", nugget=0.0, psill=1.0, rng=3.0)
        out = ordinary_kriging(coords, z, vg, coords[:30])
        self.assertTrue(np.all(np.isfinite(out["variance"])))
        np.testing.assert_allclose(out["variance"], 0.0, atol=1e-9)


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


class CalibrationValidationTest(unittest.TestCase):
    """MXR-080-0103: calibrate_variance must reject impossible targets and invalid samples."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.pv = rng.uniform(0.5, 2.0, 1000)
        self.resid = rng.normal(0, 1, 1000) * np.sqrt(self.pv)

    def test_rejects_target_below_zero(self):
        # MXR-080-0103 exact repro: previously converged silently to the lower boundary scale
        # factor (1e-6) instead of raising.
        with self.assertRaisesRegex(ValueError, r"target must be finite and strictly in \(0, 1\)"):
            calibrate_variance(self.pv, self.resid, target=-1.0)

    def test_rejects_target_at_zero(self):
        with self.assertRaisesRegex(ValueError, r"target must be finite and strictly in \(0, 1\)"):
            calibrate_variance(self.pv, self.resid, target=0.0)

    def test_rejects_target_at_one(self):
        with self.assertRaisesRegex(ValueError, r"target must be finite and strictly in \(0, 1\)"):
            calibrate_variance(self.pv, self.resid, target=1.0)

    def test_rejects_target_above_one(self):
        # Previously converged silently to the upper boundary scale factor (1e6) instead of raising.
        with self.assertRaisesRegex(ValueError, r"target must be finite and strictly in \(0, 1\)"):
            calibrate_variance(self.pv, self.resid, target=2.0)

    def test_rejects_nan_target(self):
        # Previously converged silently to the lower boundary scale factor instead of raising: NaN
        # comparisons are always False, so `coverage(mid) < target` was always False too.
        with self.assertRaisesRegex(ValueError, r"target must be finite and strictly in \(0, 1\)"):
            calibrate_variance(self.pv, self.resid, target=float("nan"))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            calibrate_variance(self.pv[:5], self.resid, target=0.9)

    def test_rejects_empty_sample(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            calibrate_variance(np.array([]), np.array([]), target=0.9)

    def test_rejects_non_finite_predicted_var(self):
        pv = np.concatenate([self.pv, [np.nan]])
        resid = np.concatenate([self.resid, [0.0]])
        with self.assertRaisesRegex(ValueError, "predicted_var must contain only finite"):
            calibrate_variance(pv, resid, target=0.9)

    def test_rejects_non_finite_residuals(self):
        pv = np.concatenate([self.pv, [1.0]])
        resid = np.concatenate([self.resid, [np.inf]])
        with self.assertRaisesRegex(ValueError, "residuals must contain only finite"):
            calibrate_variance(pv, resid, target=0.9)

    def test_rejects_zero_predicted_var(self):
        pv = np.concatenate([self.pv, [0.0]])
        resid = np.concatenate([self.resid, [0.0]])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            calibrate_variance(pv, resid, target=0.9)

    def test_rejects_negative_predicted_var(self):
        pv = np.concatenate([self.pv, [-1.0]])
        resid = np.concatenate([self.resid, [0.0]])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            calibrate_variance(pv, resid, target=0.9)

    def test_valid_target_and_data_still_calibrates(self):
        # Negative control: a legitimate target in (0, 1) with valid paired held-out data is
        # unaffected by the new validation.
        c = calibrate_variance(self.pv, self.resid, target=0.9)
        self.assertTrue(np.isfinite(c))
        self.assertGreater(c, 0.0)


if __name__ == "__main__":
    unittest.main()
