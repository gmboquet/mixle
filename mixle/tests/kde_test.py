"""Kernel density / mode / intensity estimation (mixle.stats.kde)."""

import unittest

import numpy as np
from numpy import trapezoid

from mixle.analysis import (
    KDE,
    intensity,
    kde,
    kde_mode,
    scott_bandwidth,
    silverman_bandwidth,
)


class KDETest(unittest.TestCase):
    def test_integrates_to_one(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 3000)
        f = kde(x)
        grid = np.linspace(-7, 7, 3000)
        self.assertAlmostEqual(trapezoid(f(grid), grid), 1.0, delta=0.01)

    def test_recovers_normal_density(self):
        rng = np.random.RandomState(1)
        x = rng.normal(0, 1, 5000)
        f = kde(x)
        self.assertAlmostEqual(float(f(np.array([0.0]))[0]), 1.0 / np.sqrt(2 * np.pi), delta=0.03)

    def test_boundary_correction_reduces_edge_bias(self):
        rng = np.random.RandomState(2)
        x = rng.exponential(1.0, 5000)  # true density at 0+ is 1.0
        plain = kde(x)
        refl = kde(x, bounds=(0.0, None))
        at0 = np.array([0.02])
        # reflection is much closer to the true edge density of 1.0
        self.assertLess(float(plain(at0)[0]), 0.7)
        self.assertGreater(float(refl(at0)[0]), 0.8)
        gi = np.linspace(0, 10, 4000)
        self.assertAlmostEqual(trapezoid(refl(gi), gi), 1.0, delta=0.02)

    def test_finite_interval_boundary_kernel_preserves_mass_across_bandwidth_scales(self):
        data = np.array([0.25, 0.75])
        grid = np.linspace(0.0, 1.0, 4001)
        for bandwidth in (0.05, 0.2, 1.0, 10.0, 100.0):
            with self.subTest(bandwidth=repr(bandwidth)):
                fitted = kde(data, bandwidth=bandwidth, bounds=(0.0, 1.0))
                self.assertAlmostEqual(trapezoid(fitted(grid), grid), 1.0, delta=2e-4)

    def test_evaluate_rejects_empty_or_nonfinite_points_before_support_masking(self):
        plain = kde(np.array([0.0, 1.0, 2.0]), bandwidth=0.5)
        bounded = kde(np.array([0.0, 0.5, 1.0]), bandwidth=0.5, bounds=(0.0, 1.0))
        for fitted in (plain, bounded):
            with self.assertRaises(ValueError):
                fitted.evaluate(np.array([]))
            with self.assertRaises(ValueError):
                fitted.evaluate(np.array([np.nan]))
            with self.assertRaises(ValueError):
                fitted.evaluate(np.array([np.inf]))

    def test_fitted_state_owns_and_freezes_training_and_bandwidth_arrays(self):
        original = np.array([0.0, 1.0, 2.0])
        fitted = kde(original, bandwidth=0.5)
        before = fitted.evaluate(np.array([0.0]))
        original[:] = 100.0
        np.testing.assert_array_equal(fitted.evaluate(np.array([0.0])), before)
        np.testing.assert_array_equal(fitted.data[:, 0], np.array([0.0, 1.0, 2.0]))
        self.assertFalse(fitted.data.flags.writeable)
        with self.assertRaises(ValueError):
            fitted.data[0, 0] = 50.0

        multi = kde(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]), bandwidth=[0.5, 0.25])
        self.assertFalse(multi.bandwidth.flags.writeable)
        with self.assertRaises(ValueError):
            multi.bandwidth[0] = 10.0

    def test_adaptive_integrates_to_one(self):
        rng = np.random.RandomState(3)
        x = rng.standard_t(3, 3000)  # heavy tails benefit from adaptive bw
        f = kde(x, adaptive=True)
        grid = np.linspace(x.min() - 1, x.max() + 1, 4000)
        self.assertAlmostEqual(trapezoid(f(grid), grid), 1.0, delta=0.03)

    def test_bandwidth_selectors_positive(self):
        rng = np.random.RandomState(4)
        x = rng.normal(0, 2, 1000)
        self.assertGreater(silverman_bandwidth(x), 0)
        self.assertGreater(scott_bandwidth(x), 0)


class ModeTest(unittest.TestCase):
    def test_recovers_mode(self):
        rng = np.random.RandomState(0)
        x = rng.normal(5.0, 1.0, 5000)
        self.assertAlmostEqual(kde_mode(x), 5.0, delta=0.3)

    def test_bimodal_mode_at_higher_peak(self):
        rng = np.random.RandomState(1)
        x = np.concatenate([rng.normal(-3, 0.4, 2000), rng.normal(3, 0.4, 6000)])  # taller peak at +3
        self.assertAlmostEqual(kde_mode(x), 3.0, delta=0.4)

    def test_bootstrap_ci_brackets_mode(self):
        rng = np.random.RandomState(2)
        x = rng.normal(0.0, 1.0, 3000)
        out = kde_mode(x, ci=True, n_boot=40, seed=0)
        self.assertLessEqual(out["ci_low"], out["mode"])
        self.assertLessEqual(out["mode"], out["ci_high"])


class IntensityTest(unittest.TestCase):
    def test_integral_recovers_event_count(self):
        rng = np.random.RandomState(0)
        events = np.sort(rng.uniform(0, 10, 200))
        grid = np.linspace(0, 10, 1000)
        lam = intensity(events, grid, domain=(0, 10), bandwidth=0.5)
        # the intensity integrates to ~ the number of events
        self.assertAlmostEqual(trapezoid(lam, grid), 200.0, delta=20.0)

    def test_inhomogeneous_rate_tracks_density(self):
        rng = np.random.RandomState(1)
        # events concentrated near t=8
        events = np.sort(np.concatenate([rng.uniform(0, 10, 50), rng.normal(8, 0.5, 300)]))
        grid = np.array([2.0, 8.0])
        lam = intensity(events, grid, domain=(0, 10), bandwidth=0.5)
        self.assertGreater(lam[1], 3 * lam[0])  # much higher intensity at t=8


class DegenerateInputTest(unittest.TestCase):
    """MXR-080-0100: empty, singleton, and constant samples used to silently produce zero or NaN
    automatic bandwidths (a constant 3-point sample evaluated at its own value returned NaN); unknown
    bandwidth-method strings silently fell through to Scott's rule; and numeric bandwidths, bounds,
    bootstrap counts, interval levels, and intensity event windows had no domain validation at all.
    Each case here now raises a clear ``ValueError`` instead, with a negative control confirming
    legitimate, non-degenerate input still produces a sensible finite result."""

    # -- automatic-bandwidth selectors: empty / singleton / constant samples --------------------

    def test_silverman_bandwidth_rejects_empty(self):
        with self.assertRaises(ValueError):
            silverman_bandwidth(np.array([]))

    def test_silverman_bandwidth_rejects_singleton(self):
        with self.assertRaises(ValueError):
            silverman_bandwidth(np.array([5.0]))

    def test_silverman_bandwidth_rejects_constant(self):
        with self.assertRaises(ValueError):
            silverman_bandwidth(np.array([3.0, 3.0, 3.0]))

    def test_scott_bandwidth_rejects_empty(self):
        with self.assertRaises(ValueError):
            scott_bandwidth(np.array([]))

    def test_scott_bandwidth_rejects_singleton(self):
        with self.assertRaises(ValueError):
            scott_bandwidth(np.array([5.0]))

    def test_scott_bandwidth_rejects_constant(self):
        with self.assertRaises(ValueError):
            scott_bandwidth(np.array([3.0, 3.0, 3.0]))

    def test_bandwidth_selectors_negative_control_still_positive(self):
        # non-degenerate input is completely unaffected by the new guards.
        rng = np.random.RandomState(4)
        x = rng.normal(0, 2, 1000)
        self.assertGreater(silverman_bandwidth(x), 0)
        self.assertGreater(scott_bandwidth(x), 0)

    def test_bandwidth_selectors_require_exact_positive_joint_dimension(self):
        sample = np.array([0.0, 1.0, 2.0])
        for bad_dimension in (0, -4, 1.5, True):
            with self.subTest(d=repr(bad_dimension)):
                with self.assertRaises(ValueError):
                    silverman_bandwidth(sample, d=bad_dimension)
                with self.assertRaises(ValueError):
                    scott_bandwidth(sample, d=bad_dimension)

    # -- KDE construction: empty / singleton / constant samples with automatic bandwidth --------

    def test_kde_rejects_empty_sample(self):
        with self.assertRaises(ValueError):
            kde(np.array([]))

    def test_kde_rejects_singleton_with_automatic_bandwidth(self):
        with self.assertRaises(ValueError):
            kde(np.array([5.0]))

    def test_kde_rejects_constant_sample_with_automatic_bandwidth(self):
        with self.assertRaises(ValueError):
            kde(np.array([3.0, 3.0, 3.0]))

    def test_kde_rejects_non_finite_data(self):
        with self.assertRaises(ValueError):
            kde(np.array([1.0, 2.0, float("nan")]), bandwidth=0.5)

    def test_kde_degenerate_policy_explicit_bandwidth_still_works(self):
        # the declared degenerate-kernel policy: a caller-supplied positive numeric bandwidth bypasses
        # automatic selection entirely, so a constant or singleton sample is still usable.
        f_const = kde(np.array([3.0, 3.0, 3.0]), bandwidth=0.5)
        val = f_const.evaluate(np.array([3.0]))[0]
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0.0)

        f_single = kde(np.array([5.0]), bandwidth=0.3)
        val2 = f_single.evaluate(np.array([5.0]))[0]
        self.assertTrue(np.isfinite(val2))
        self.assertGreater(val2, 0.0)

    def test_kde_negative_control_legitimate_sample_still_finite(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 500)
        f = kde(x)
        val = f.evaluate(np.array([0.0]))[0]
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0.0)

    # -- bandwidth method / numeric validation ---------------------------------------------------

    def test_kde_rejects_unknown_bandwidth_method_string(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bandwidth="banana")

    def test_kde_rejects_zero_bandwidth(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bandwidth=0.0)

    def test_kde_rejects_negative_bandwidth(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bandwidth=-1.0)

    def test_kde_rejects_nan_bandwidth(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bandwidth=float("nan"))

    def test_kde_rejects_inf_bandwidth(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bandwidth=float("inf"))

    # -- bounds validation --------------------------------------------------------------------

    def test_kde_rejects_inverted_bounds(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bounds=(5.0, -5.0))

    def test_kde_rejects_degenerate_zero_width_bounds(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bounds=(1.0, 1.0))

    def test_kde_rejects_non_finite_bound(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 100)
        with self.assertRaises(ValueError):
            kde(x, bounds=(float("nan"), None))

    def test_kde_negative_control_legitimate_bounds_still_work(self):
        rng = np.random.RandomState(2)
        x = rng.exponential(1.0, 2000)
        f = kde(x, bounds=(0.0, None))
        val = f.evaluate(np.array([0.02]))[0]
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0.0)

    # -- kde_mode: bootstrap count / interval level validation -----------------------------------

    def test_kde_mode_rejects_empty_sample(self):
        with self.assertRaises(ValueError):
            kde_mode(np.array([]))

    def test_kde_mode_rejects_zero_n_boot(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 200)
        with self.assertRaises(ValueError):
            kde_mode(x, ci=True, n_boot=0)

    def test_kde_mode_rejects_negative_n_boot(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 200)
        with self.assertRaises(ValueError):
            kde_mode(x, ci=True, n_boot=-5)

    def test_kde_mode_rejects_non_integer_n_boot(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 200)
        with self.assertRaises(ValueError):
            kde_mode(x, ci=True, n_boot=40.5)

    def test_kde_mode_rejects_ci_level_at_or_outside_bounds(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 200)
        for bad_level in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                kde_mode(x, ci=True, n_boot=20, ci_level=bad_level, seed=0)

    def test_kde_mode_negative_control_legitimate_ci_still_works(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 500)
        out = kde_mode(x, ci=True, n_boot=30, ci_level=0.9, seed=0)
        self.assertTrue(np.isfinite(out["mode"]))
        self.assertLessEqual(out["ci_low"], out["ci_high"])

    # -- intensity: empty / non-finite events and event-window (domain) validation --------------

    def test_intensity_rejects_empty_events(self):
        with self.assertRaises(ValueError):
            intensity(np.array([]), np.linspace(0, 1, 5))

    def test_intensity_rejects_non_finite_events(self):
        with self.assertRaises(ValueError):
            intensity(np.array([1.0, 2.0, float("inf")]), np.linspace(0, 1, 5), bandwidth=0.5)

    def test_intensity_rejects_nonfinite_or_multidimensional_grid(self):
        events = np.array([0.0, 1.0, 2.0])
        for bad_grid in (
            np.array([0.0, np.nan]),
            np.array([0.0, np.inf]),
            np.ones((2, 2)),
            np.array([]),
        ):
            with self.subTest(shape=repr(bad_grid.shape)):
                with self.assertRaises(ValueError):
                    intensity(events, bad_grid, bandwidth=0.5)

    def test_intensity_rejects_constant_events_with_automatic_bandwidth(self):
        with self.assertRaises(ValueError):
            intensity(np.array([1.0, 1.0, 1.0]), np.linspace(0, 2, 5))

    def test_intensity_rejects_inverted_domain(self):
        rng = np.random.RandomState(0)
        events = rng.uniform(0, 10, 50)
        with self.assertRaises(ValueError):
            intensity(events, np.linspace(0, 10, 5), domain=(5.0, 2.0))

    def test_intensity_rejects_collapsed_default_domain(self):
        # constant events with an explicit numeric bandwidth skip the automatic-bandwidth guard, but
        # the *default* domain (the event range) then collapses to a single point (lo == hi); this
        # used to silently blow up (divide by a clipped near-zero kernel mass) instead of raising.
        with self.assertRaises(ValueError):
            intensity(np.array([1.0, 1.0, 1.0]), np.linspace(0, 2, 5), bandwidth=0.5)

    def test_intensity_negative_control_legitimate_domain_still_works(self):
        rng = np.random.RandomState(0)
        events = np.sort(rng.uniform(0, 10, 200))
        grid = np.linspace(0, 10, 100)
        lam = intensity(events, grid, domain=(0, 10), bandwidth=0.5)
        self.assertTrue(np.all(np.isfinite(lam)))
        self.assertTrue(np.all(lam >= 0.0))


class MultivariateTest(unittest.TestCase):
    """MXR-080-0099: KDE.__init__ used to call .ravel() on an (n, d) sample, turning n paired
    d-dimensional observations into n*d unrelated 1-D observations -- destroying both which axis was
    which variable and which values came from the same observation. KDE now implements a true
    (n, d) axis-aligned Gaussian product kernel with a per-dimension bandwidth (Silverman/Scott
    generalized to the joint-dimension n^(-1/(d+4)) rate), preserving row-pairing throughout. bounds
    (reflection boundary correction) remains 1-D only and is now explicitly rejected for d > 1, and
    kde_mode/intensity -- which never claimed multivariate support -- now explicitly reject (n, d)
    input instead of silently flattening it the same way."""

    def test_construction_preserves_shape_and_pairing(self):
        # a (3, 2) input must stay (3, 2) -- not flattened to (6,) -- and rows must be untouched.
        data = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        f = KDE(data)
        self.assertEqual(f.n, 3)
        self.assertEqual(f.d, 2)
        self.assertEqual(f.data.shape, (3, 2))
        np.testing.assert_array_equal(f.data, data)

    def test_flattening_would_have_lost_the_correlation_pre_fix(self):
        # the audit's own reproduction: 3 paired 2-D points with an obvious cross-dimensional
        # correlation (each row's two coordinates are equal). Flattened to n*d = 6 unrelated 1-D
        # scalars [0, 0, 10, 10, 20, 20], no density over those 6 numbers alone could possibly encode
        # "coordinate 0 and coordinate 1 always match within a row" -- that information only exists
        # in the row pairing, which flattening discards. Demonstrate the paired (2-D) construction
        # keeps n and d distinct from the flattened count, which is the crux of the bug.
        data = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        flattened_n = data.size  # what the old .ravel()-based constructor would have used as n
        f = KDE(data)
        self.assertEqual(f.n, 3)
        self.assertNotEqual(f.n, flattened_n)
        self.assertEqual(f.d, 2)

    def test_correlated_2d_sample_density_is_higher_on_the_diagonal(self):
        # a 2-D sample where the two dimensions are strongly correlated (x1 ~= x0): a genuine
        # product-kernel joint density must concentrate mass near the diagonal x0 == x1 and assign
        # much less density off it. This is only possible because sample pairing survived
        # construction -- a flattened version has no notion of "which x0 went with which x1".
        rng = np.random.RandomState(0)
        n = 400
        base = rng.normal(0, 1, n)
        x0 = base
        x1 = base + rng.normal(0, 0.05, n)
        data = np.column_stack([x0, x1])
        f = kde(data)
        self.assertEqual(f.d, 2)
        on_diag = f.evaluate(np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]))
        off_diag = f.evaluate(np.array([[0.0, 3.0], [1.0, -1.0], [-1.0, 2.0]]))
        self.assertTrue(np.all(on_diag > off_diag * 10))

    def test_correlated_2d_sample_recovers_approximate_correlation(self):
        # a more quantitative version of the above: integrate the fitted joint density over a grid to
        # get an (approximate) correlation coefficient, and check it roughly matches the true sample
        # correlation -- the multivariate KDE's estimated correlation structure should track the
        # true one, which is impossible if pairing were destroyed by flattening.
        rng = np.random.RandomState(0)
        n = 400
        base = rng.normal(0, 1, n)
        x0 = base
        x1 = base + rng.normal(0, 0.05, n)
        f = kde(np.column_stack([x0, x1]))
        g1 = np.linspace(-3, 3, 60)
        g2 = np.linspace(-3, 3, 60)
        gg1, gg2 = np.meshgrid(g1, g2)
        pts = np.column_stack([gg1.ravel(), gg2.ravel()])
        dens = f.evaluate(pts).reshape(gg1.shape)
        dens = dens / dens.sum()
        m1 = np.sum(dens * gg1)
        m2 = np.sum(dens * gg2)
        cov12 = np.sum(dens * (gg1 - m1) * (gg2 - m2))
        var1 = np.sum(dens * (gg1 - m1) ** 2)
        var2 = np.sum(dens * (gg2 - m2) ** 2)
        est_corr = cov12 / np.sqrt(var1 * var2)
        true_corr = np.corrcoef(x0, x1)[0, 1]
        self.assertAlmostEqual(est_corr, true_corr, delta=0.15)

    def test_multivariate_density_integrates_to_one(self):
        rng = np.random.RandomState(7)
        data = rng.normal(0, 1, (1500, 2))
        f = kde(data)
        g = np.linspace(-5, 5, 250)
        gg1, gg2 = np.meshgrid(g, g)
        pts = np.column_stack([gg1.ravel(), gg2.ravel()])
        dens = f.evaluate(pts).reshape(gg1.shape)
        dx = g[1] - g[0]
        integral = float(dens.sum() * dx * dx)
        self.assertAlmostEqual(integral, 1.0, delta=0.03)

    def test_per_dimension_bandwidth_uses_joint_dimension_exponent(self):
        # Scott/Silverman must scale each dimension's bandwidth by n^(-1/(d+4)) using the TRUE joint
        # dimension d, not the univariate n^(-1/5) rate -- get the exponent wrong and the per-dimension
        # bandwidths silently don't match what selecting each column's bandwidth for a d-D product
        # kernel actually calls for.
        rng = np.random.RandomState(1)
        col0 = rng.normal(0, 3, 500)
        col1 = rng.normal(0, 1, 500)
        f = KDE(np.column_stack([col0, col1]), bandwidth="scott")
        expected = np.array([scott_bandwidth(col0, d=2), scott_bandwidth(col1, d=2)])
        np.testing.assert_allclose(f.bandwidth, expected)
        # negative control: must NOT match the univariate (d=1) exponent for the same columns.
        univariate0 = scott_bandwidth(col0, d=1)
        self.assertFalse(np.isclose(f.bandwidth[0], univariate0))

    def test_explicit_per_dimension_bandwidth_vector(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (300, 2))
        f = KDE(data, bandwidth=[0.5, 0.25])
        np.testing.assert_allclose(f.bandwidth, [0.5, 0.25])

    def test_scalar_bandwidth_broadcasts_to_every_dimension(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (300, 3))
        f = KDE(data, bandwidth=0.4)
        np.testing.assert_allclose(f.bandwidth, [0.4, 0.4, 0.4])

    def test_wrong_length_bandwidth_vector_rejected(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (300, 2))
        with self.assertRaises(ValueError):
            KDE(data, bandwidth=[0.5, 0.25, 0.1])

    def test_bounds_rejected_for_multivariate(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (300, 2))
        with self.assertRaises(ValueError):
            KDE(data, bounds=(0.0, None))

    def test_adaptive_bandwidth_supported_for_multivariate(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (800, 2))
        f = KDE(data, adaptive=True)
        val = f.evaluate(np.array([[0.0, 0.0], [1.0, 1.0]]))
        self.assertTrue(np.all(np.isfinite(val)))
        self.assertTrue(np.all(val > 0.0))

    def test_evaluate_rejects_dimension_mismatch(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (300, 3))
        f = KDE(data)
        with self.assertRaises(ValueError):
            f.evaluate(np.array([[0.0, 0.0]]))  # 2-D point against a 3-D KDE

    def test_more_than_2d_input_rejected(self):
        with self.assertRaises(ValueError):
            KDE(np.zeros((5, 2, 2)))

    def test_kde_mode_rejects_multivariate_input(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (300, 2))
        with self.assertRaises(ValueError):
            kde_mode(data)

    def test_intensity_rejects_multivariate_input(self):
        rng = np.random.RandomState(1)
        data = rng.normal(0, 1, (300, 2))
        with self.assertRaises(ValueError):
            intensity(data, np.linspace(0, 1, 5))

    def test_negative_control_1d_kde_unaffected(self):
        # 1-D construction and evaluation must behave exactly as before the multivariate rewrite.
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 3000)
        f = kde(x)
        self.assertEqual(f.d, 1)
        self.assertIsInstance(f.bandwidth, float)
        grid = np.linspace(-7, 7, 3000)
        self.assertAlmostEqual(trapezoid(f(grid), grid), 1.0, delta=0.01)

    def test_negative_control_1d_kde_mode_unaffected(self):
        rng = np.random.RandomState(0)
        x = rng.normal(5.0, 1.0, 5000)
        self.assertAlmostEqual(kde_mode(x), 5.0, delta=0.3)

    def test_negative_control_1d_intensity_unaffected(self):
        rng = np.random.RandomState(0)
        events = np.sort(rng.uniform(0, 10, 200))
        grid = np.linspace(0, 10, 1000)
        lam = intensity(events, grid, domain=(0, 10), bandwidth=0.5)
        self.assertAlmostEqual(trapezoid(lam, grid), 200.0, delta=20.0)


if __name__ == "__main__":
    unittest.main()
