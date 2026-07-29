"""Tests for the DoE space-filling / classical design generators (WS-E)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from mixle.doe import (
    full_factorial,
    halton_design,
    latin_hypercube,
    maximin_latin_hypercube,
    maxpro_design,
    random_design,
    sobol_design,
)
from mixle.doe.designs import _as_bounds, _as_rng, _maxpro_criterion, _maxpro_swap, _scale_unit


def _within_bounds(x, bounds):
    b = np.asarray(bounds, dtype=np.float64)
    return bool(np.all(x >= b[:, 0] - 1e-12) and np.all(x <= b[:, 1] + 1e-12))


def _lhs_one_per_stratum(x, bounds, n):
    """Each axis must place exactly one point in each of the n equal strata."""
    b = np.asarray(bounds, dtype=np.float64)
    unit = (x - b[:, 0]) / (b[:, 1] - b[:, 0])
    for j in range(x.shape[1]):
        strata = np.clip(np.floor(unit[:, j] * n).astype(int), 0, n - 1)
        if sorted(strata.tolist()) != list(range(n)):
            return False
    return True


class DoeDesignsTest(unittest.TestCase):
    bounds = [(0.0, 1.0), (-2.0, 2.0), (10.0, 20.0)]

    def test_latin_hypercube_shape_bounds_and_stratification(self):
        n = 12
        x = latin_hypercube(self.bounds, n, seed=0)
        self.assertEqual(x.shape, (n, len(self.bounds)))
        self.assertTrue(_within_bounds(x, self.bounds))
        self.assertTrue(_lhs_one_per_stratum(x, self.bounds, n))

    def test_latin_hypercube_reproducible_and_seed_varies(self):
        a = latin_hypercube(self.bounds, 10, seed=7)
        b = latin_hypercube(self.bounds, 10, seed=7)
        c = latin_hypercube(self.bounds, 10, seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_latin_hypercube_center_at_stratum_midpoints(self):
        n = 5
        x = latin_hypercube([(0.0, 1.0)], n, seed=1, center=True)
        mids = np.sort(x[:, 0])
        np.testing.assert_allclose(mids, (np.arange(n) + 0.5) / n, atol=1e-12)

    def test_random_design_shape_and_bounds(self):
        x = random_design(self.bounds, 50, seed=3)
        self.assertEqual(x.shape, (50, 3))
        self.assertTrue(_within_bounds(x, self.bounds))

    def test_maximin_is_valid_lhs_and_not_worse(self):
        n = 10
        mm = maximin_latin_hypercube(self.bounds, n, seed=2, trials=40)
        self.assertEqual(mm.shape, (n, 3))
        self.assertTrue(_lhs_one_per_stratum(mm, self.bounds, n))

        def min_dist(x):
            b = np.asarray(self.bounds, dtype=np.float64)
            s = (x - b[:, 0]) / (b[:, 1] - b[:, 0])
            diff = s[:, None, :] - s[None, :, :]
            sq = np.sum(diff * diff, axis=2)
            return np.min(sq[np.triu_indices(n, k=1)])

        plain = latin_hypercube(self.bounds, n, seed=2)
        self.assertGreaterEqual(min_dist(mm) + 1e-12, min_dist(plain))

    def test_maxpro_minimizes_projection_criterion(self):
        bounds = [(0.0, 1.0)] * 4
        mp = maxpro_design(bounds, 20, seed=0)
        self.assertEqual(mp.shape, (20, 4))
        self.assertTrue(_within_bounds(mp, bounds))
        # the continuous refinement drives the MaxPro criterion far below a plain LHS (orders of magnitude)
        lhs = latin_hypercube(bounds, 20, seed=0)
        self.assertLess(_maxpro_criterion(mp), _maxpro_criterion(lhs))
        # MaxPro is NOT LHS-constrained (points move off the grid) but the criterion keeps every 1-D
        # projection near-uniform: no large gaps along any axis.
        for k in range(4):
            coords = np.sort(np.concatenate([[0.0], mp[:, k], [1.0]]))
            self.assertLess(float(np.max(np.diff(coords))), 0.2)

    def test_full_factorial_grid_size_and_corners(self):
        x = full_factorial([(0.0, 1.0), (0.0, 10.0)], levels=3)
        self.assertEqual(x.shape, (9, 2))
        # Corners of the box must be present.
        for corner in [(0.0, 0.0), (1.0, 10.0), (0.0, 10.0), (1.0, 0.0)]:
            self.assertTrue(np.any(np.all(np.isclose(x, corner), axis=1)), corner)

    def test_full_factorial_per_dim_levels_and_single_level_midpoint(self):
        x = full_factorial([(0.0, 1.0), (-4.0, 4.0)], levels=[4, 1])
        self.assertEqual(x.shape, (4, 2))
        np.testing.assert_allclose(np.unique(x[:, 1]), [0.0], atol=1e-12)  # single level -> midpoint

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            latin_hypercube([(1.0, 0.0)], 5)  # low >= high
        with self.assertRaises(ValueError):
            latin_hypercube(self.bounds, 0)  # n must be positive
        with self.assertRaises(ValueError):
            full_factorial(self.bounds, levels=[2, 2])  # wrong length
        with self.assertRaises(ValueError):
            random_design([], 5)  # no dimensions


class QuasiRandomDesignsTest(unittest.TestCase):
    bounds = [(0.0, 1.0), (-2.0, 2.0), (10.0, 20.0)]

    def test_sobol_shape_bounds_and_reproducibility(self):
        x = sobol_design(self.bounds, 16, seed=0)
        self.assertEqual(x.shape, (16, 3))
        self.assertTrue(_within_bounds(x, self.bounds))
        np.testing.assert_array_equal(x, sobol_design(self.bounds, 16, seed=0))
        self.assertFalse(np.array_equal(x, sobol_design(self.bounds, 16, seed=1)))

    def test_halton_shape_bounds_and_reproducibility(self):
        x = halton_design(self.bounds, 13, seed=0)
        self.assertEqual(x.shape, (13, 3))
        self.assertTrue(_within_bounds(x, self.bounds))
        np.testing.assert_array_equal(x, halton_design(self.bounds, 13, seed=0))

    def test_quasi_random_fills_more_evenly_than_uniform(self):
        # Lower discrepancy == more even space-filling. Sobol' should beat iid uniform.
        unit_bounds = [(0.0, 1.0)] * 3
        sob = sobol_design(unit_bounds, 64, seed=0)
        hal = halton_design(unit_bounds, 64, seed=0)
        rnd = random_design(unit_bounds, 64, seed=0)
        self.assertLess(qmc.discrepancy(sob), qmc.discrepancy(rnd))
        self.assertLess(qmc.discrepancy(hal), qmc.discrepancy(rnd))

    def test_quasi_random_validation(self):
        with self.assertRaises(ValueError):
            sobol_design(self.bounds, 0)
        with self.assertRaises(ValueError):
            halton_design([(1.0, 0.0)], 8)


class BoundsAndCountValidationTest(unittest.TestCase):
    """MXR-080-0174: every generator here shares _as_bounds (finite, low<high) and
    _require_exact_positive_int (exact integer, no silent truncation) for its count controls."""

    def test_infinite_and_nan_bounds_rejected_by_every_generator(self):
        bad_bounds_cases = (
            [(0.0, np.inf)],
            [(-np.inf, 1.0)],
            [(np.nan, 1.0)],
            [(0.0, 1.0), (0.0, np.inf)],  # only one dimension unbounded
        )
        for bad_bounds in bad_bounds_cases:
            for fn in (latin_hypercube, random_design, sobol_design, halton_design, maxpro_design):
                with self.subTest(bounds=repr(bad_bounds), fn=repr(fn.__name__)):
                    with self.assertRaises(ValueError):
                        fn(bad_bounds, 4)
            with self.subTest(bounds=repr(bad_bounds), fn="maximin_latin_hypercube"):
                with self.assertRaises(ValueError):
                    maximin_latin_hypercube(bad_bounds, 4)
            with self.subTest(bounds=repr(bad_bounds), fn="full_factorial"):
                with self.assertRaises(ValueError):
                    full_factorial(bad_bounds, levels=2)

    def test_fractional_and_negative_counts_rejected_instead_of_truncated(self):
        # Before: `if n <= 0: raise` let a fractional n (e.g. 3.7) straight through to int(n) == 3,
        # silently truncated. Now every count control rejects a non-integer value up front.
        # (maxpro_design's own restart/swap/iteration/n controls are covered separately, alongside its
        # optimizer-failure fallback, in MaxProValidationAndFallbackTest.)
        bounds = [(0.0, 1.0), (-2.0, 2.0)]
        with self.assertRaises(ValueError):
            latin_hypercube(bounds, 3.5)
        with self.assertRaises(ValueError):
            random_design(bounds, 3.5)
        with self.assertRaises(ValueError):
            sobol_design(bounds, 3.5)
        with self.assertRaises(ValueError):
            halton_design(bounds, 3.5)
        with self.assertRaises(ValueError):
            maximin_latin_hypercube(bounds, 5, trials=2.5)
        with self.assertRaises(ValueError):
            full_factorial(bounds, levels=[2.5, 2])
        # Negative and non-finite counts are equally rejected, not just fractional ones.
        with self.assertRaises(ValueError):
            latin_hypercube(bounds, -3)
        with self.assertRaises(ValueError):
            maximin_latin_hypercube(bounds, 5, trials=-1)
        with self.assertRaises(ValueError):
            full_factorial(bounds, levels=[-1, 2])
        with self.assertRaises(ValueError):
            full_factorial(bounds, levels=[0, 2])  # docstring requires each level >= 1
        with self.assertRaises(ValueError):
            latin_hypercube(bounds, float("nan"))
        with self.assertRaises(ValueError):
            latin_hypercube(bounds, float("inf"))
        # A bare non-integer scalar for `levels` used to crash inside np.linspace with a confusing
        # TypeError instead of failing validation cleanly.
        with self.assertRaises(ValueError):
            full_factorial(bounds, levels=2.5)

    def test_valid_finite_bounds_and_integer_counts_still_generate_correct_designs(self):
        """Negative control: legitimate input is unaffected by the new validation."""
        bounds = [(0.0, 1.0), (-2.0, 2.0), (10.0, 20.0)]
        x = latin_hypercube(bounds, 12, seed=0)
        self.assertEqual(x.shape, (12, 3))
        self.assertTrue(_within_bounds(x, bounds))
        # An exact-integer float (5.0) is accepted like the int 5 -- only fractional values are rejected.
        y = latin_hypercube(bounds, 5.0, seed=0)
        self.assertEqual(y.shape, (5, 3))
        z = full_factorial([(0.0, 1.0), (0.0, 10.0)], levels=[3, 2])
        self.assertEqual(z.shape, (6, 2))
        mm = maximin_latin_hypercube(bounds, 6, seed=1, trials=5)
        self.assertEqual(mm.shape, (6, 3))

    def test_extreme_finite_bounds_never_produce_nonfinite_points(self):
        """MXR-080-1475: representable points must not be lost to overflowing span arithmetic."""
        bounds = [(-1e308, 1e308)]
        designs = (
            random_design(bounds, 4, seed=1),
            latin_hypercube(bounds, 4, seed=1),
            maximin_latin_hypercube(bounds, 4, seed=1, trials=2),
            sobol_design(bounds, 4, seed=1),
            halton_design(bounds, 4, seed=1),
            maxpro_design(bounds, 4, seed=1, restarts=1, swaps=0, maxiter=0),
        )
        for design in designs:
            with self.subTest(design=repr(design)):
                self.assertTrue(np.all(np.isfinite(design)))
                self.assertTrue(_within_bounds(design, bounds))

    def test_extreme_single_level_midpoint_is_computed_without_overflow(self):
        design = full_factorial([(9e307, 1e308), (-1e308, 1e308)], levels=[1, 1])
        self.assertTrue(np.all(np.isfinite(design)))
        np.testing.assert_allclose(design[0], [9.5e307, 0.0], rtol=1e-15)


class MaxProValidationAndFallbackTest(unittest.TestCase):
    """MXR-080-0175: maxpro_design's restart/swap/iteration/n controls and its optimizer-failure fallback."""

    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def test_rejects_invalid_n_restart_swap_iteration_controls(self):
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 0)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, -2)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 3.5)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 6, restarts=0)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 6, restarts=-1)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 6, restarts=1.5)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 6, swaps=-1)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 6, swaps=2.5)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 6, maxiter=-1)
        with self.assertRaises(ValueError):
            maxpro_design(self.bounds, 6, maxiter=3.5)

    def test_zero_swaps_or_maxiter_are_legitimate_not_errors(self):
        # swaps=0 skips the discrete coordinate-exchange phase; maxiter=0 skips continuous refinement.
        # Neither is "restarts" (which must be >= 1 -- zero restarts would produce no design at all).
        self.assertEqual(maxpro_design(self.bounds, 6, swaps=0).shape, (6, 2))
        self.assertEqual(maxpro_design(self.bounds, 6, maxiter=0).shape, (6, 2))

    def test_falls_back_to_last_verified_design_when_optimizer_reports_failure(self):
        """A failed (success=False) optimizer result must never be clipped/compared as if it were a
        legitimate refinement -- the restart should fall back to its swap-refined discrete design."""
        n, d = 6, 2
        failed = SimpleNamespace(success=False, fun=float("nan"), x=np.full(n * d, np.nan), jac=np.full(n * d, np.nan))
        with patch("scipy.optimize.minimize", return_value=failed):
            got = maxpro_design(self.bounds, n, seed=0, restarts=1, swaps=20, maxiter=50)

        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(_within_bounds(got, self.bounds))

        # Reconstruct exactly what the swap-only stage produces from the same seed, to confirm the
        # fallback is the genuine verified discrete design, not merely "some finite" placeholder.
        b = _as_bounds(self.bounds)
        rng = _as_rng(0)
        start = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            start[:, j] = (rng.permutation(n) + rng.random_sample(n)) / n
        start = _maxpro_swap(start, rng, 20)
        expected = _scale_unit(start, b)
        np.testing.assert_array_equal(got, expected)

    def test_falls_back_when_optimizer_reports_success_but_non_finite(self):
        """success=True alone is not enough -- a non-finite fun/x/jac must also fall back."""
        n, d = 6, 2
        ok_x, ok_jac = np.full(n * d, 0.5), np.zeros(n * d)
        nan_x = np.concatenate([np.full(n * d - 1, 0.5), [np.nan]])
        inf_jac = np.concatenate([np.zeros(n * d - 1), [np.inf]])
        bad_results = {
            "non_finite_fun": SimpleNamespace(success=True, fun=float("inf"), x=ok_x, jac=ok_jac),
            "non_finite_x": SimpleNamespace(success=True, fun=0.1, x=nan_x, jac=ok_jac),
            "non_finite_jac": SimpleNamespace(success=True, fun=0.1, x=ok_x, jac=inf_jac),
        }
        for name, result in bad_results.items():
            with self.subTest(name), patch("scipy.optimize.minimize", return_value=result):
                got = maxpro_design(self.bounds, n, seed=0, restarts=1, swaps=20, maxiter=50)
            self.assertTrue(np.all(np.isfinite(got)), name)
            self.assertTrue(_within_bounds(got, self.bounds), name)

    def test_successful_refinement_still_improves_on_swap_only_baseline(self):
        """Negative control: a normal (unmocked) run's continuous stage still genuinely refines --
        the docstring's claim that it "cuts the criterion by far more than the swap phase alone"."""
        bounds = [(0.0, 1.0)] * 4
        n = 20
        refined = maxpro_design(bounds, n, seed=0, restarts=1)
        swap_only = maxpro_design(bounds, n, seed=0, restarts=1, maxiter=0)
        self.assertLess(_maxpro_criterion(refined), _maxpro_criterion(swap_only))


if __name__ == "__main__":
    unittest.main()
