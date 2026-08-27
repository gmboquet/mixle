"""The Gaussian variance safeguards must be scale-relative, and must say when they bound (T4-6, T1-04).

T4-6: the M-step variance floor was an ABSOLUTE ``1e-8``, so the same measurements expressed in
different units fitted different variances -- 10 uV of noise recorded in volts (empirical variance
8.8e-11) came back 114x too wide, while the identical data in microvolts came back exact. A
maximum-likelihood fit of a location-scale family has to satisfy ``fit(c*x) == c**2 * fit(x)``. The
same absolute term dominated the ``max(min_covar, ridge * scale)`` in both vector estimators, so
their relative ``ridge`` could not rescue small-scale data either, and ``DiagonalGaussianEstimator``
applied the clamp with no entry in ``numerical_repairs()`` at all.

T1-04: ``numerical_repairs()`` judged the covariance ridge by the DIAGONAL alone, so the ridge that
lifted exactly-zero eigenvalues -- the singular-component case the ridge exists for, and the
docstring's own named example -- was reported as no repair. At 2 to 5 points in ``d=5`` the smallest
eigenvalue is ~1e-7 while the smallest diagonal entry is ~1e-2, so the ratio never moved.

Both directions are pinned here: a safeguard that binds must be disclosed, and an ordinary
well-scaled fit must still record nothing.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    DiagonalGaussianEstimator,
    GaussianEstimator,
    MultivariateGaussianEstimator,
)

_MICRO = 1.0e6


def _volt_signal(n=500, seed=7):
    """A 2 mV signal with 10 uV of noise -- ordinary sensor data, entirely below the old 1e-8 floor."""
    return list(np.random.default_rng(seed).normal(0.002, 1.0e-5, n))


def _vector_signal(n=400, dim=3, seed=11):
    """Correlated columns at unit scale, to be re-expressed in other units by the caller."""
    rng = np.random.default_rng(seed)
    mixing = np.triu(np.ones((dim, dim))) * (0.5 + rng.random((dim, dim)))
    return rng.standard_normal((n, dim)) @ mixing


class UnitEquivarianceTest(unittest.TestCase):
    """The fitted variance must scale exactly as ``c**2`` when the data is rescaled by ``c``."""

    def test_scalar_fit_is_equivariant_under_a_change_of_units(self):
        volts = _volt_signal()
        fitted_volts = optimize(volts, GaussianEstimator(), max_its=5)
        fitted_micro = optimize([v * _MICRO for v in volts], GaussianEstimator(), max_its=5)
        np.testing.assert_allclose(fitted_micro.sigma2, fitted_volts.sigma2 * _MICRO**2, rtol=1e-9)

    def test_scalar_small_scale_fit_is_the_variance_the_data_implied(self):
        # The floor used to widen this to 1e-8 -- 114x -- so a genuine 6-sigma excursion read as 0.56.
        volts = _volt_signal()
        fitted = optimize(volts, GaussianEstimator(), max_its=5)
        np.testing.assert_allclose(fitted.sigma2, float(np.var(np.asarray(volts))), rtol=1e-9)
        self.assertEqual(fitted.numerical_repairs(), ())

    def test_diagonal_fit_is_equivariant_under_a_change_of_units(self):
        x = _vector_signal()
        base = optimize([tuple(v) for v in x], DiagonalGaussianEstimator(dim=3), max_its=5)
        scaled = optimize([tuple(v * 1e-6) for v in x], DiagonalGaussianEstimator(dim=3), max_its=5)
        np.testing.assert_allclose(np.asarray(scaled.covar), np.asarray(base.covar) * 1e-12, rtol=1e-9)

    def test_multivariate_fit_is_equivariant_under_a_change_of_units(self):
        x = _vector_signal()
        base = optimize([tuple(v) for v in x], MultivariateGaussianEstimator(dim=3), max_its=5)
        scaled = optimize([tuple(v * 1e-6) for v in x], MultivariateGaussianEstimator(dim=3), max_its=5)
        np.testing.assert_allclose(np.asarray(scaled.covar), np.asarray(base.covar) * 1e-12, rtol=1e-9)

    def test_a_small_scale_vector_fit_recovers_the_empirical_covariance(self):
        x = _vector_signal() * 1e-6
        rows = [tuple(v) for v in x]
        empirical = np.cov(x.T, bias=True)
        full = optimize(rows, MultivariateGaussianEstimator(dim=3), max_its=5)
        diagonal = optimize(rows, DiagonalGaussianEstimator(dim=3), max_its=5)
        np.testing.assert_allclose(np.asarray(full.covar), empirical, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(diagonal.covar), np.diag(empirical), rtol=1e-9)


class AbsoluteFloorStillAvailableTest(unittest.TestCase):
    """An explicitly configured ``min_covar`` is a fixed regularizer and must keep behaving like one."""

    def test_scalar_explicit_min_covar_is_absolute(self):
        fitted = optimize(_volt_signal(), GaussianEstimator(min_covar=1.0e-3), max_its=5)
        self.assertEqual(fitted.sigma2, 1.0e-3)
        self.assertTrue(any("variance-floored" in r for r in fitted.numerical_repairs()))

    def test_diagonal_explicit_min_covar_is_absolute(self):
        rows = [(v, v * 2.0) for v in _volt_signal(n=200)]
        fitted = optimize(rows, DiagonalGaussianEstimator(dim=2, min_covar=1.0e-3), max_its=5)
        np.testing.assert_allclose(np.asarray(fitted.covar), np.full(2, 1.0e-3))
        self.assertTrue(any("variance-floored" in r for r in fitted.numerical_repairs()))

    def test_multivariate_explicit_min_covar_is_absolute(self):
        rows = [(v, v * 2.0 + 1e-6) for v in _volt_signal(n=200)]
        fitted = optimize(rows, MultivariateGaussianEstimator(dim=2, min_covar=1.0e-3), max_its=5)
        self.assertGreaterEqual(float(np.diag(np.asarray(fitted.covar)).min()), 1.0e-3)
        self.assertTrue(any("covariance-ridged" in r for r in fitted.numerical_repairs()))

    def test_the_default_safeguard_still_rescues_a_degenerate_scalar_fit(self):
        # Constant data implies zero variance: there is no correct answer, so the floor must still
        # produce a positive one -- sized by the data's own magnitude -- and disclose that it did.
        fitted = optimize([0.002] * 60, GaussianEstimator(), max_its=5)
        self.assertGreater(fitted.sigma2, 0.0)
        self.assertLess(fitted.sigma2, 1.0e-8)  # the old absolute floor, now far too wide for 2 mV
        self.assertTrue(any("variance-floored" in r for r in fitted.numerical_repairs()))

    def test_a_degenerate_scalar_fit_with_no_scale_at_all_falls_back_to_the_absolute_floor(self):
        # All-zero observations carry neither spread nor magnitude; min_covar is the last resort.
        fitted = optimize([0.0] * 60, GaussianEstimator(), max_its=5)
        self.assertEqual(fitted.sigma2, 1.0e-8)
        self.assertTrue(any("variance-floored" in r for r in fitted.numerical_repairs()))


class DiagonalFloorDisclosureTest(unittest.TestCase):
    """The diagonal estimator floored silently; a clamp has to be visible where the others are."""

    def test_a_floored_coordinate_is_recorded_in_repairs_and_provenance(self):
        rng = np.random.default_rng(3)
        rows = [(float(v), 7.0) for v in rng.standard_normal(80)]  # second coordinate is constant
        fitted = optimize(rows, DiagonalGaussianEstimator(dim=2), max_its=5)
        self.assertTrue(
            any(r.startswith("variance-floored(") for r in fitted.numerical_repairs()),
            fitted.numerical_repairs(),
        )
        provenance = fitted.fit_provenance()
        self.assertIsNotNone(provenance)
        self.assertTrue(any("variance-floored" in r for r in provenance.repairs), provenance.repairs)
        # and the floor really did bind: the constant coordinate has a manufactured variance
        self.assertGreater(float(np.asarray(fitted.covar)[1]), 0.0)

    def test_a_heterogeneous_unit_fit_has_nothing_to_report(self):
        # UPDATED (T1-03). This test used to assert the opposite: that this fit reports an inflated
        # coordinate, because the floor was ``1e-6 * mean(var)`` ACROSS coordinates and so the
        # smallest column here (variance 2.25, against a mean of 3.4e6) was lifted to 3.4 -- 1.5x
        # too wide -- purely because another column was measured in bigger units. Disclosing that
        # was the best available answer while the floor still bound; it is not an answer at all now
        # that the floor is priced per coordinate. The floor no longer touches a coordinate that has
        # a positive variance of its own, so the fit is the exact per-column MLE and there is
        # nothing to disclose. The disclosure contract itself (T4-6) is still pinned by the sibling
        # test above, on a constant column -- the case where a clamp genuinely still happens.
        rng = np.random.default_rng(42)
        x = np.column_stack(
            [
                3000.0 + 3200.0 * rng.standard_normal(203),
                4.0 + 3.2 * rng.standard_normal(203),
                5.8 + 1.5 * rng.standard_normal(203),
            ]
        )
        fitted = optimize([tuple(v) for v in x], DiagonalGaussianEstimator(dim=3), max_its=30, delta=None)
        self.assertEqual(fitted.numerical_repairs(), ())
        np.testing.assert_allclose(np.asarray(fitted.covar), np.var(x, axis=0), rtol=1e-12)

    def test_a_scale_homogeneous_diagonal_fit_stays_silent(self):
        rng = np.random.default_rng(7)
        rows = [tuple(v) for v in rng.standard_normal((200, 3))]
        fitted = optimize(rows, DiagonalGaussianEstimator(dim=3), max_its=30, delta=None)
        self.assertEqual(fitted.numerical_repairs(), ())


class RankDeficientRidgeDisclosureTest(unittest.TestCase):
    """A ridge that lifted zero eigenvalues is maximally binding, whatever the diagonal says."""

    def test_fewer_points_than_dimensions_reports_the_ridge(self):
        y = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.5]])  # n=2 in d=3: rank-1 scatter
        fitted = optimize([row for row in y], MultivariateGaussianEstimator(dim=3), max_its=1)
        eigenvalues = np.linalg.eigvalsh(np.asarray(fitted.covar))
        self.assertGreater(float(eigenvalues.min()), 0.0)  # the ridge did manufacture them
        self.assertTrue(
            any("covariance-ridged" in r for r in fitted.numerical_repairs()),
            fitted.numerical_repairs(),
        )

    def test_the_whole_silent_band_up_to_d_points_reports_the_ridge(self):
        # 1 < n <= d was uniformly silent; only the n=1 endpoint fired, and only because the
        # diagonal itself went non-positive there.
        rng = np.random.default_rng(11)
        for n in range(1, 6):
            with self.subTest(n=n):
                rows = [tuple(v) for v in rng.standard_normal((n, 5))]
                fitted = optimize(rows, MultivariateGaussianEstimator(dim=5), max_its=1)
                self.assertTrue(
                    any("covariance-ridged" in r for r in fitted.numerical_repairs()),
                    (n, fitted.numerical_repairs()),
                )

    def test_exactly_collinear_columns_report_the_ridge(self):
        rng = np.random.default_rng(5)
        c = rng.standard_normal(500)
        z = np.column_stack([c, c * 2.0, rng.standard_normal(500)])
        fitted = optimize([tuple(v) for v in z], MultivariateGaussianEstimator(dim=3), max_its=1)
        self.assertTrue(
            any("covariance-ridged" in r for r in fitted.numerical_repairs()),
            fitted.numerical_repairs(),
        )

    def test_the_repair_reaches_the_fit_provenance(self):
        y = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.5]])
        fitted = optimize([row for row in y], MultivariateGaussianEstimator(dim=3), max_its=1)
        provenance = fitted.fit_provenance()
        self.assertIsNotNone(provenance)
        self.assertTrue(any("covariance-ridged" in r for r in provenance.repairs), provenance.repairs)
        self.assertTrue(provenance.is_approximate())

    def test_a_full_rank_fit_records_no_ridge(self):
        rng = np.random.default_rng(3)
        rows = [tuple(v) for v in rng.standard_normal((200, 8))]
        fitted = optimize(rows, MultivariateGaussianEstimator(dim=8), max_its=5)
        self.assertEqual(fitted.numerical_repairs(), ())

    def test_a_near_singular_but_full_rank_fit_stays_silent(self):
        # Strongly correlated coordinates (smallest eigenvalue ~5e-13) are ill-conditioned, not
        # rank-deficient: the raw covariance factors on its own, so the ridge is not what rescued it.
        rng = np.random.default_rng(2)
        a = rng.standard_normal(400)
        w = np.column_stack([a, a + 1.0e-6 * rng.standard_normal(400), rng.standard_normal(400)])
        fitted = optimize([tuple(v) for v in w], MultivariateGaussianEstimator(dim=3), max_its=1)
        self.assertEqual(fitted.numerical_repairs(), ())


class NoNewNoiseOnOrdinaryFitsTest(unittest.TestCase):
    """The reverse overreach: disclosure must not start firing on fits that were never repaired."""

    def test_ordinary_fits_across_many_scales_and_shapes_record_nothing(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            n, dim = int(rng.integers(40, 300)), int(rng.integers(2, 6))
            mixing = rng.standard_normal((dim, dim))
            scale = 10.0 ** rng.uniform(-6.0, 6.0)
            x = scale * (rng.standard_normal((n, dim)) @ mixing + rng.standard_normal(dim))
            rows = [tuple(v) for v in x]
            with self.subTest(seed=seed, n=n, dim=dim):
                for estimator in (
                    MultivariateGaussianEstimator(dim=dim),
                    DiagonalGaussianEstimator(dim=dim),
                ):
                    fitted = optimize(rows, estimator, max_its=10, delta=None)
                    self.assertEqual(fitted.numerical_repairs(), (), (seed, type(estimator).__name__))
                scalar = optimize(list(x[:, 0]), GaussianEstimator(), max_its=10, delta=None)
                self.assertEqual(scalar.numerical_repairs(), (), seed)


if __name__ == "__main__":
    unittest.main()
