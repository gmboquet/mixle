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
        base = simplex_lattice(3, 2)
        lower = [0.1, 0.2, 0.05]
        x = to_pseudocomponents(base, lower)
        np.testing.assert_allclose(x.sum(axis=1), 1.0)
        self.assertTrue(np.all(x >= np.array(lower) - 1e-12))

    def test_rejects_infeasible_lower(self):
        with self.assertRaises(ValueError):
            to_pseudocomponents(simplex_lattice(3, 2), [0.5, 0.4, 0.3])  # sum >= 1


if __name__ == "__main__":
    unittest.main()
