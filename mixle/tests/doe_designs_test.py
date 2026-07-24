"""Tests for the DoE space-filling / classical design generators (WS-E)."""

import unittest

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
from mixle.doe.designs import _maxpro_criterion


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
                with self.subTest(bounds=bad_bounds, fn=fn.__name__):
                    with self.assertRaises(ValueError):
                        fn(bad_bounds, 4)
            with self.subTest(bounds=bad_bounds, fn="maximin_latin_hypercube"):
                with self.assertRaises(ValueError):
                    maximin_latin_hypercube(bad_bounds, 4)
            with self.subTest(bounds=bad_bounds, fn="full_factorial"):
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


if __name__ == "__main__":
    unittest.main()
