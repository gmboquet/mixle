"""Kernel density / mode / intensity estimation (mixle.stats.kde)."""

import unittest

import numpy as np
from numpy import trapezoid

from mixle.analysis import (
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


if __name__ == "__main__":
    unittest.main()
