"""Mixture (simplex) experiment designs (mixle.doe.mixture)."""

import unittest
from math import comb

import numpy as np

from mixle.doe import simplex_centroid, simplex_lattice, to_pseudocomponents


class SimplexLatticeTest(unittest.TestCase):
    def test_count_levels_and_sum(self):
        x = simplex_lattice(3, 2)
        self.assertEqual(x.shape, (comb(3 + 2 - 1, 2), 3))  # C(q+m-1, m)
        np.testing.assert_allclose(x.sum(axis=1), 1.0)
        self.assertEqual(sorted(set(x.flatten().tolist())), [0.0, 0.5, 1.0])  # levels 0, 1/m, 1

    def test_higher_degree_count(self):
        self.assertEqual(simplex_lattice(4, 3).shape[0], comb(4 + 3 - 1, 3))

    def test_rejects_degenerate(self):
        with self.assertRaises(ValueError):
            simplex_lattice(1, 2)
        with self.assertRaises(ValueError):
            simplex_lattice(3, 0)


class SimplexCentroidTest(unittest.TestCase):
    def test_count_and_blends(self):
        x = simplex_centroid(3)
        self.assertEqual(x.shape, (2**3 - 1, 3))  # every non-empty subset
        np.testing.assert_allclose(x.sum(axis=1), 1.0)
        # includes the q pure components ...
        for pure in np.eye(3):
            self.assertTrue(np.any(np.all(np.isclose(x, pure), axis=1)))
        np.testing.assert_allclose(x[-1], np.full(3, 1.0 / 3))  # ... and the overall centroid last

    def test_q4_count(self):
        self.assertEqual(simplex_centroid(4).shape[0], 2**4 - 1)


class PseudocomponentTest(unittest.TestCase):
    def test_respects_lower_bounds_and_simplex(self):
        # Golden path: must keep working exactly as before the MXR-080-0180 validation was added.
        base = simplex_lattice(3, 2)
        lower = [0.1, 0.2, 0.05]
        x = to_pseudocomponents(base, lower)
        np.testing.assert_allclose(x.sum(axis=1), 1.0)
        self.assertTrue(np.all(x >= np.array(lower) - 1e-12))

    def test_rejects_infeasible_lower(self):
        with self.assertRaises(ValueError):
            to_pseudocomponents(simplex_lattice(3, 2), [0.5, 0.4, 0.3])  # sum >= 1

    def test_rejects_non_simplex_blend_with_valid_lower(self):
        # Regression test for MXR-080-0180: blend [2, -1] together with a finite, non-negative lower
        # bound vector that itself sums to < 1 (so the pre-fix checks all passed) used to be silently
        # accepted and mapped to [1.7, -0.7] -- a negative "proportion" that still (deceptively) summed
        # to 1. This is not a valid canonical simplex point (component 2 is negative) and must now be
        # rejected outright rather than silently produce an out-of-simplex result.
        with self.assertRaises(ValueError):
            to_pseudocomponents([[2, -1]], [0.1, 0.1])

    def test_rejects_blend_not_summing_to_one(self):
        # A blend that is itself non-negative but off the simplex (rows must sum to 1) must also be
        # rejected -- not just blends with an outright negative component.
        with self.assertRaises(ValueError):
            to_pseudocomponents([[0.2, 0.2, 0.2]], [0.1, 0.1, 0.05])  # rows sum to 0.6, not 1

    def test_rejects_nan_blend(self):
        with self.assertRaises(ValueError):
            to_pseudocomponents([[np.nan, 0.5, 0.5]], [0.1, 0.1, 0.05])

    def test_rejects_infinite_lower(self):
        with self.assertRaises(ValueError):
            to_pseudocomponents(simplex_lattice(3, 2), [np.inf, 0.1, 0.1])

    def test_rejects_nan_lower(self):
        with self.assertRaises(ValueError):
            to_pseudocomponents(simplex_lattice(3, 2), [np.nan, 0.1, 0.1])

    def test_rejects_wrong_length_blend(self):
        # blend has 2 components but lower specifies 3.
        with self.assertRaises(ValueError):
            to_pseudocomponents([[0.5, 0.5]], [0.1, 0.1, 0.05])

    def test_rejects_wrong_rank_blend(self):
        # A single blend passed as a 1-D vector (not wrapped as a (1, q) row) must raise a clear
        # ValueError, not an uninformative IndexError from indexing a nonexistent second axis.
        with self.assertRaises(ValueError):
            to_pseudocomponents([0.6, 0.4], [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
