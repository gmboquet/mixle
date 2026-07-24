"""Smith max-stable process for spatial extremes: extremal coefficient, margins, dependence (Phase 6).

Also covers MXR-080-0105 (missing domain/identifiability validation, including a crash on a
single-location fit).
"""

import unittest

import numpy as np
from scipy.stats import norm, spearmanr

from mixle.analysis.max_stable import SmithMaxStable, fit_smith_maxstable


class SmithMaxStableTest(unittest.TestCase):
    def setUp(self):
        self.ms = SmithMaxStable(sigma=2.0 * np.eye(2))

    def test_extremal_coefficient_bounds_and_formula(self):
        self.assertAlmostEqual(self.ms.extremal_coefficient([0, 0]), 1.0, places=6)  # full dependence at h=0
        self.assertAlmostEqual(self.ms.extremal_coefficient([100, 0]), 2.0, places=4)  # independence far away
        a = np.sqrt(np.array([3.0, 0.0]) @ np.linalg.inv(self.ms.sigma) @ np.array([3.0, 0.0]))
        self.assertAlmostEqual(self.ms.extremal_coefficient([3, 0]), 2 * norm.cdf(a / 2))

    def test_extremal_coefficient_is_monotone(self):
        thetas = [self.ms.extremal_coefficient([h, 0]) for h in (0, 1, 2, 4, 8)]
        self.assertTrue(all(thetas[i] <= thetas[i + 1] + 1e-9 for i in range(len(thetas) - 1)))

    def test_bivariate_cdf_is_a_valid_probability(self):
        self.assertTrue(0.0 < self.ms.bivariate_cdf(1.0, 1.0, [2, 0]) < 1.0)

    def test_sampler_has_unit_frechet_margins(self):
        # n=1500 (down from 4000) keeps a comfortable safety margin on the atol=0.2 check: across
        # 20 seeds the worst-case median deviation observed was ~0.135 (a ~1.5x margin), with mean
        # deviation ~0.064 -- 0/20 failures. n_storms (the Schlather-algorithm storm count, which
        # controls approximation fidelity rather than Monte Carlo replication count) is left
        # unchanged since it governs bias, not just variance.
        s = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=0).sample(1500, n_storms=150)
        np.testing.assert_allclose(np.median(s, axis=0), 1.0 / np.log(2), atol=0.2)  # unit-Frechet median

    def test_short_range_extremes_are_more_dependent(self):
        # n=1000 (down from 3000) still leaves a wide margin on the near>far comparison: across 10
        # seeds the smallest observed gap (near - far Spearman correlation) was ~0.90, far above 0.
        loc = np.array([[0, 0], [0.5, 0], [8, 0]])
        s = self.ms.sampler(loc, seed=0).sample(1000, n_storms=150)
        near = spearmanr(s[:, 0], s[:, 1]).correlation
        far = spearmanr(s[:, 0], s[:, 2]).correlation
        self.assertGreater(near, far)

    def test_fit_recovers_the_dependence_scale(self):
        # Negative control for MXR-080-0105: a legitimate, well-posed fit (well-separated locations,
        # plenty of replicates) still identifies the dependence scale correctly and reports status="ok".
        true = SmithMaxStable(2.0**2 * np.eye(2))
        locs = np.random.RandomState(1).uniform(0, 12, (10, 2))
        fields = true.sampler(locs, seed=2).sample(500, n_storms=120)
        fit = fit_smith_maxstable(locs, fields)
        self.assertEqual(fit.status, "ok")
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(np.sqrt(fit.model.sigma[0, 0]), 2.0, delta=0.6)


class SmithMaxStableDomainValidationTest(unittest.TestCase):
    """MXR-080-0105: sigma/threshold/dimension domain checks that were previously silently skipped."""

    def test_sigma_must_be_finite(self):
        with self.assertRaises(ValueError):
            SmithMaxStable(sigma=np.array([[np.nan, 0.0], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            SmithMaxStable(sigma=np.array([[np.inf, 0.0], [0.0, 1.0]]))

    def test_sigma_must_be_square(self):
        # non-square sigma already raised out of np.linalg.inv in the old code (LinAlgError, a
        # ValueError subclass) -- but only incidentally, and with a bare linear-algebra message
        # instead of a clear domain one; check for the latter specifically.
        with self.assertRaises(ValueError) as ctx:
            SmithMaxStable(sigma=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        self.assertIn("sigma must be a square matrix", str(ctx.exception))

    def test_sigma_must_be_symmetric(self):
        # non-singular (det=5.5): the old code accepted this silently (only np.linalg.inv could
        # ever have caught anything, and it has no opinion on symmetry) -- an accidentally-singular
        # asymmetric matrix would raise for the wrong reason and not actually exercise this check.
        with self.assertRaises(ValueError) as ctx:
            SmithMaxStable(sigma=np.array([[2.0, 1.0], [0.5, 3.0]]))
        self.assertIn("symmetric", str(ctx.exception))

    def test_sigma_must_be_positive_definite(self):
        with self.assertRaises(ValueError):  # negative eigenvalue
            SmithMaxStable(sigma=np.array([[1.0, 2.0], [2.0, 1.0]]))
        with self.assertRaises(ValueError):  # positive semi-definite only (singular)
            SmithMaxStable(sigma=np.array([[1.0, 1.0], [1.0, 1.0]]))

    def test_h_dimension_must_match_sigma(self):
        # a mismatched h already raised out of the bare matmul in the old code (a generic numpy
        # gufunc-signature ValueError) -- check for the clear domain message instead.
        ms = SmithMaxStable(sigma=2.0 * np.eye(2))
        with self.assertRaises(ValueError) as ctx:
            ms.extremal_coefficient([1.0, 2.0, 3.0])
        self.assertIn("h must have shape", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            ms.bivariate_cdf(1.0, 1.0, [1.0, 2.0, 3.0])
        self.assertIn("h must have shape", str(ctx.exception))

    def test_sampler_locations_dimension_must_match_sigma(self):
        ms = SmithMaxStable(sigma=2.0 * np.eye(2))
        with self.assertRaises(ValueError):
            ms.sampler(np.array([[0.0, 0.0, 0.0]]))

    def test_sampler_locations_must_be_finite(self):
        ms = SmithMaxStable(sigma=2.0 * np.eye(2))
        with self.assertRaises(ValueError):
            ms.sampler(np.array([[0.0, np.nan]]))

    def test_bivariate_cdf_thresholds_must_be_strictly_positive(self):
        # unit-Frechet support is (0, inf); the closed form divides by z1/z2 and logs their ratio.
        ms = SmithMaxStable(sigma=2.0 * np.eye(2))
        for z1, z2 in ((-1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, -2.0)):
            with self.assertRaises(ValueError):
                ms.bivariate_cdf(z1, z2, [1, 0])

    def test_bivariate_cdf_still_works_for_a_legitimate_threshold(self):
        # negative control
        ms = SmithMaxStable(sigma=2.0 * np.eye(2))
        self.assertTrue(0.0 < ms.bivariate_cdf(0.5, 2.0, [1, 0]) < 1.0)


class FitSmithMaxStableValidationTest(unittest.TestCase):
    """MXR-080-0105: fit_smith_maxstable's shape/finiteness/identifiability checks and fit diagnostic."""

    def test_single_location_is_rejected_not_a_bare_numpy_crash(self):
        # audit's own repro: len(loc)==1 -> zero pairs -> lags.max() on an empty array used to raise
        # "ValueError: zero-size array to reduction operation maximum which has no identity".
        loc = np.array([[0.0, 0.0]])
        fields = np.random.RandomState(0).uniform(0, 1, size=(20, 1))
        with self.assertRaises(ValueError) as ctx:
            fit_smith_maxstable(loc, fields)
        self.assertNotIn("zero-size array", str(ctx.exception))  # a clear domain message, not the old crash

    def test_coincident_locations_are_rejected(self):
        # 2 locations at the same coordinates: 1 pair, but a zero lag -- the search bracket's upper
        # bound would otherwise degenerate to ~0, and the old code let scipy raise "The lower bound
        # exceeds the upper bound" instead of a clear domain message about the locations themselves.
        loc = np.array([[1.0, 1.0], [1.0, 1.0]])
        fields = np.random.RandomState(0).uniform(0, 1, size=(20, 2))
        with self.assertRaises(ValueError) as ctx:
            fit_smith_maxstable(loc, fields)
        self.assertIn("coincide", str(ctx.exception))

    def test_field_shape_must_match_locations(self):
        loc = np.array([[0.0, 0.0], [1.0, 0.0]])
        fields = np.random.RandomState(0).uniform(0, 1, size=(20, 5))  # wrong n_locations
        with self.assertRaises(ValueError):
            fit_smith_maxstable(loc, fields)

    def test_fields_must_be_finite(self):
        loc = np.array([[0.0, 0.0], [1.0, 0.0]])
        for bad in (np.nan, np.inf, -np.inf):
            fields = np.array([[1.0, 2.0], [bad, 3.0], [2.0, 1.0]])
            with self.assertRaises(ValueError):
                fit_smith_maxstable(loc, fields)

    def test_single_replicate_is_rejected(self):
        # with 1 replicate the rank transform is identically 0 for every location (nothing to rank
        # against), so the empirical extremal coefficient is identically 1 (full dependence) for
        # every pair regardless of distance -- not a real fit, and previously silently accepted.
        loc = np.array([[0.0, 0.0], [1.0, 0.0]])
        fields = np.array([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            fit_smith_maxstable(loc, fields)

    def test_locations_must_be_finite(self):
        # the old code let a NaN location propagate into the search bounds and raised scipy's
        # "Optimization bounds must be finite scalars" instead of a domain message about locations.
        loc = np.array([[0.0, 0.0], [np.nan, 0.0]])
        fields = np.random.RandomState(0).uniform(0, 1, size=(20, 2))
        with self.assertRaises(ValueError) as ctx:
            fit_smith_maxstable(loc, fields)
        self.assertIn("locations must be finite", str(ctx.exception))

    def test_degenerate_replicates_land_on_the_search_boundary_not_a_trusted_estimate(self):
        # Deterministic construction: identical field values across every location, for every
        # replicate, collapses every pair's rank-based nu to exactly 0 (theta_emp=1, "full
        # dependence") regardless of distance. The least-squares match then has no interior optimum
        # and is pushed to the bracket's upper edge -- SmithMaxStableFit must flag this rather than
        # silently reporting a confident dependence-scale estimate (the audit's own repro of this,
        # via a 1-replicate fit, landed sigma[0,0] within 1e-4 of the search bound before it was
        # rejected outright by test_single_replicate_is_rejected above; this is the same failure
        # mode surfacing through a differently-degenerate, but not outright-rejected, input).
        loc = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        rep = np.random.RandomState(0).uniform(0, 1, size=20)
        fields = np.tile(rep[:, None], (1, loc.shape[0]))  # identical across locations
        fit = fit_smith_maxstable(loc, fields)
        self.assertEqual(fit.status, "boundary")
        self.assertFalse(fit.converged)

    def test_fit_result_carries_size_diagnostics(self):
        loc = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        fields = np.random.RandomState(3).uniform(0, 1, size=(30, 3))
        fit = fit_smith_maxstable(loc, fields)
        self.assertEqual(fit.n_locations, 3)
        self.assertEqual(fit.n_replicates, 30)
        self.assertEqual(fit.n_pairs, 3)  # 3 choose 2
        self.assertGreaterEqual(fit.residual, 0.0)
        self.assertIsInstance(fit.model, SmithMaxStable)


if __name__ == "__main__":
    unittest.main()
