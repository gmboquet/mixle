"""Smith max-stable process for spatial extremes: extremal coefficient, margins, dependence (Phase 6).

Also covers MXR-080-0104 (sampling truncation was silently invalid at n_storms=0 and otherwise
unbounded-approximate) and MXR-080-0105 (missing domain/identifiability validation, including a crash
on a single-location fit).
"""

import unittest
import warnings

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
        # unchanged since it governs bias, not just variance. n_storms=150 is well below what the
        # default tol=1e-3 needs for this box (~17k storms; see MXR-080-0104), so the storm-count
        # safety-cap warning is expected here and suppressed -- this test intentionally trades
        # tol's guarantee for speed, exactly as it traded exactness for speed before that guarantee
        # existed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            s = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=0).sample(1500, n_storms=150)
        np.testing.assert_allclose(np.median(s, axis=0), 1.0 / np.log(2), atol=0.2)  # unit-Frechet median

    def test_short_range_extremes_are_more_dependent(self):
        # n=1000 (down from 3000) still leaves a wide margin on the near>far comparison: across 10
        # seeds the smallest observed gap (near - far Spearman correlation) was ~0.90, far above 0.
        # See test_sampler_has_unit_frechet_margins for why the safety-cap warning is suppressed.
        loc = np.array([[0, 0], [0.5, 0], [8, 0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            s = self.ms.sampler(loc, seed=0).sample(1000, n_storms=150)
        near = spearmanr(s[:, 0], s[:, 1]).correlation
        far = spearmanr(s[:, 0], s[:, 2]).correlation
        self.assertGreater(near, far)

    def test_fit_recovers_the_dependence_scale(self):
        # Negative control for MXR-080-0105: a legitimate, well-posed fit (well-separated locations,
        # plenty of replicates) still identifies the dependence scale correctly and reports status="ok".
        true = SmithMaxStable(2.0**2 * np.eye(2))
        locs = np.random.RandomState(1).uniform(0, 12, (10, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # see test_sampler_has_unit_frechet_margins
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


class SmithMaxStableSamplingTruncationTest(unittest.TestCase):
    """MXR-080-0104: sampling truncation must be validated/controlled, not silently invalid or
    unboundedly approximate."""

    def setUp(self):
        self.ms = SmithMaxStable(sigma=2.0 * np.eye(2))

    def test_n_storms_zero_is_rejected(self):
        # audit's own repro: n_storms=0 previously returned an all-zero field -- 0 is outside
        # unit-Frechet's (0, inf) support, so that was never a valid draw.
        samp = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=0)
        with self.assertRaises(ValueError):
            samp.sample(n_storms=0)

    def test_n_storms_negative_or_non_integer_is_rejected(self):
        samp = self.ms.sampler(np.array([[0, 0]]), seed=0)
        for bad in (-1, -5, 2.5):
            with self.assertRaises(ValueError):
                samp.sample(n_storms=bad)

    def test_tol_domain(self):
        samp = self.ms.sampler(np.array([[0, 0]]), seed=0)
        for bad_tol in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                samp.sample(tol=bad_tol)

    def test_box_sigma_domain(self):
        samp = self.ms.sampler(np.array([[0, 0]]), seed=0)
        for bad_box in (0.0, -2.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                samp.sample(box_sigma=bad_box)

    def test_at_least_one_storm_always_contributes(self):
        # Even under an extremely loose tol -- which makes the stopping check satisfied by the very
        # first storm -- the field must never be exactly 0: 0 is just as far outside unit-Frechet's
        # (0, inf) support as the old n_storms=0 bug's all-zero field was, regardless of which knob
        # produced it.
        samp = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=0)
        z = samp.sample(500, n_storms=5, tol=1e6)
        self.assertTrue(np.all(z > 0.0))
        self.assertTrue(np.all(np.isfinite(z)))

    def test_default_sample_is_a_sensible_positive_finite_field(self):
        # negative control: normal positive n_storms/tol (here: the defaults) still produce a valid,
        # well-formed draw.
        samp = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=3)
        z = samp.sample(5)
        self.assertEqual(z.shape, (5, 2))
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertTrue(np.all(z > 0.0))

    def test_unmet_tolerance_warns(self):
        # honesty check: a storm budget too small to reach tol must say so, not silently understate
        # its own error.
        samp = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=0)
        with self.assertWarns(UserWarning):
            samp.sample(5, n_storms=10, tol=1e-6)

    def test_tightening_tol_converges_toward_unit_frechet(self):
        # QQ-style numerical convergence check (not just an API check): the empirical CDF of
        # simulated draws, evaluated at the true unit-Frechet quantile for p, should move toward p as
        # tol shrinks (box_sigma held fixed and generous, isolating the storm-count truncation that
        # tol rigorously controls per MXR-080-0104's stopping rule).
        ms1d = SmithMaxStable(sigma=1.0 * np.eye(1))
        samp = ms1d.sampler(np.array([[0.0]]), seed=42)
        loose = samp.approximation_diagnostic(n=1200, n_storms=5000, tol=2.0, box_sigma=6.0)
        tight = samp.approximation_diagnostic(n=1200, n_storms=5000, tol=0.02, box_sigma=6.0)
        self.assertLess(tight.mean_abs_error, loose.mean_abs_error)

    def test_approximation_diagnostic_shape_and_a_tighter_config_measures_better(self):
        samp = self.ms.sampler(np.array([[0, 0], [1, 0]]), seed=1)
        loose = samp.approximation_diagnostic(n=400, n_storms=30, tol=5.0, box_sigma=1.0)
        tight = samp.approximation_diagnostic(n=400, n_storms=2000, tol=0.1, box_sigma=3.0)
        self.assertEqual(loose.probability_levels, (0.25, 0.5, 0.75, 0.9))
        self.assertEqual(loose.n_replicates, 400)
        self.assertEqual(loose.per_site_max_abs_error.shape, (2,))
        self.assertLess(tight.max_abs_error, loose.max_abs_error)

    def test_replicates_alone_do_not_fix_a_truncation_bias(self):
        # A loose tol/box gives a real, persistent bias -- more replicates should shrink the noise in
        # *measuring* that bias, not the bias itself.
        ms1d = SmithMaxStable(sigma=1.0 * np.eye(1))
        samp = ms1d.sampler(np.array([[0.0]]), seed=7)
        large_n = samp.approximation_diagnostic(n=3000, n_storms=20, tol=5.0, box_sigma=1.0)
        self.assertGreater(large_n.max_abs_error, 0.05)  # still a real bias despite 3000 replicates


if __name__ == "__main__":
    unittest.main()
