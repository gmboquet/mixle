"""Classical factorial / screening / response-surface designs (mixle.doe.factorial)."""

import unittest

import numpy as np

from mixle.doe import box_behnken, central_composite, fractional_factorial, plackett_burman


class FractionalFactorialTest(unittest.TestCase):
    def test_aliasing_and_shape(self):
        # 2^(5-2): factors d,e aliased as d=ab, e=ac
        x = fractional_factorial([(-1, 1)] * 5, "a b c ab ac", coded=True)
        self.assertEqual(x.shape, (8, 5))
        self.assertTrue(np.all(np.abs(x) == 1.0))
        np.testing.assert_allclose(x[:, 3], x[:, 0] * x[:, 1])  # d = a*b
        np.testing.assert_allclose(x[:, 4], x[:, 0] * x[:, 2])  # e = a*c

    def test_main_effects_orthogonal_and_balanced(self):
        x = fractional_factorial([(-1, 1)] * 5, "a b c ab ac", coded=True)
        np.testing.assert_allclose(x.sum(axis=0), 0.0)  # balanced
        gram = x.T @ x
        np.testing.assert_allclose(gram, 8.0 * np.eye(5))  # main effects mutually orthogonal

    def test_maps_into_bounds(self):
        x = fractional_factorial([(0.0, 10.0), (100.0, 200.0)], "a b")
        # coded -1->low, +1->high
        self.assertEqual(set(map(tuple, x.tolist())), {(0.0, 100.0), (10.0, 100.0), (0.0, 200.0), (10.0, 200.0)})

    def test_rejects_mismatched_generators(self):
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1)] * 3, "a b")  # 2 tokens, 3 dims


class PlackettBurmanTest(unittest.TestCase):
    def test_run_counts_and_orthogonality(self):
        for d, n in [(3, 4), (6, 8), (7, 8), (11, 12)]:  # 11 -> N=12 cyclic generator
            x = plackett_burman([(-1, 1)] * d, coded=True)
            self.assertEqual(x.shape, (n, d))
            self.assertTrue(np.all(np.abs(x) == 1.0))
            np.testing.assert_allclose(x.T @ x, n * np.eye(d))  # columns orthogonal


class CentralCompositeTest(unittest.TestCase):
    def test_structure_and_rotatable_alpha(self):
        x = central_composite([(-1, 1)] * 3, center=4, alpha="rotatable", coded=True)
        self.assertEqual(x.shape, (8 + 6 + 4, 3))  # factorial + axial + center
        self.assertAlmostEqual(np.max(np.abs(x)), 8**0.25)  # axial distance = (2^k)^(1/4)
        self.assertEqual(int(np.sum(np.all(x == 0.0, axis=1))), 4)  # center replicates

    def test_face_centered_inside_cube(self):
        x = central_composite([(-1, 1)] * 3, alpha="face", coded=True)
        self.assertAlmostEqual(np.max(np.abs(x)), 1.0)

    def test_orthogonal_blocks_are_orthogonal(self):
        # orthogonal CCD: the linear columns are orthogonal to the centered pure-quadratic columns
        x = central_composite([(-1, 1)] * 2, center=4, alpha="orthogonal", coded=True)
        q = x * x
        q = q - q.mean(axis=0)  # center the quadratic terms
        cross = x.T @ q
        np.testing.assert_allclose(cross, 0.0, atol=1e-9)


class BoxBehnkenTest(unittest.TestCase):
    def test_structure(self):
        x = box_behnken([(-1, 1)] * 3, coded=True)
        self.assertEqual(x.shape, (4 * 3 + 3, 3))  # 4*C(3,2) + 3 center
        for j in range(3):
            self.assertEqual(sorted(set(x[:, j])), [-1.0, 0.0, 1.0])  # 3 levels
        self.assertFalse(np.any(np.all(np.abs(x) == 1.0, axis=1)))  # no cube corners

    def test_requires_three_factors(self):
        with self.assertRaises(ValueError):
            box_behnken([(-1, 1)] * 2)


if __name__ == "__main__":
    unittest.main()
