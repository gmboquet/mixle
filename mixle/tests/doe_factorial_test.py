"""Classical factorial / screening / response-surface designs (mixle.doe.factorial)."""

import unittest

import numpy as np

from mixle.doe import (
    box_behnken,
    central_composite,
    central_composite_point_kinds,
    fractional_factorial,
    generator_alias_structure,
    plackett_burman,
)


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

    def test_infinite_or_nan_bounds_rejected(self):
        # MXR-080-0174: shared with mixle.doe.designs._as_bounds.
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1), (0.0, np.inf)], "a b")
        with self.assertRaises(ValueError):
            fractional_factorial([(np.nan, 1.0), (-1, 1)], "a b")


class GeneratorTokenValidationTest(unittest.TestCase):
    """MXR-080-0179: generator tokens that repeat a factor into a constant or duplicate column must
    raise instead of silently producing a degenerate design."""

    def test_letters_cancelling_to_a_constant_column_rejected(self):
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1)] * 3, "a b aa")  # aa -> all-ones constant column
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1)] * 2, "aabb b")  # a,a,b,b all cancel -> constant

    def test_two_factors_aliased_with_each_other_rejected(self):
        # Distinct from the *intentional* aliasing a fractional design is built on (a compound token
        # like "ab" is fine) -- this is two of the *named* factors being perfectly redundant.
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1)] * 3, "a b a")  # factors 0 and 2 identical
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1)] * 3, "a b -a")  # factors 0 and 2 exact opposites

    def test_intentional_compound_generators_still_allowed(self):
        # Negative control: a compound token that is NOT a duplicate of another factor's own word is
        # exactly the mechanism fractional designs are built on and must not be rejected.
        x = fractional_factorial([(-1, 1)] * 3, "a b ab", coded=True)
        self.assertEqual(x.shape, (4, 3))
        np.testing.assert_allclose(x[:, 2], x[:, 0] * x[:, 1])

    def test_malformed_token_still_rejected(self):
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1)] * 2, "a b1")  # '1' is not a base-factor letter
        with self.assertRaises(ValueError):
            fractional_factorial([(-1, 1)] * 2, "a +")  # empty token after stripping its sign


class GeneratorAliasStructureTest(unittest.TestCase):
    """MXR-080-0179: the alias structure fractional_factorial builds is published, not discarded."""

    def test_matches_textbook_2_to_the_5_minus_2(self):
        # Montgomery-style 2^(5-2): D = AB, E = AC; A, B, C are the generating (base) factors.
        structure = generator_alias_structure("a b c ab ac")
        self.assertEqual(structure, {"x3": "ab", "x4": "ac"})

    def test_matches_textbook_2_to_the_3_minus_1(self):
        # Resolution-III 2^(3-1): C = AB.
        self.assertEqual(generator_alias_structure("a b ab"), {"x2": "ab"})

    def test_negated_generator_is_sign_prefixed(self):
        self.assertEqual(generator_alias_structure("a b -ab"), {"x2": "-ab"})

    def test_full_factorial_has_no_aliasing(self):
        self.assertEqual(generator_alias_structure("a b c"), {})

    def test_raises_the_same_way_as_fractional_factorial_for_degenerate_generators(self):
        with self.assertRaises(ValueError):
            generator_alias_structure("a b aa")
        with self.assertRaises(ValueError):
            generator_alias_structure("a b a")


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


class CentralCompositeValidationTest(unittest.TestCase):
    """MXR-080-0174/0179: center-count and alpha validation, and the factorial-cube-vs-axial distinction."""

    bounds = [(-1.0, 1.0)] * 3

    def test_center_rejects_fractional_and_negative(self):
        with self.assertRaises(ValueError):
            central_composite(self.bounds, center=2.5)
        with self.assertRaises(ValueError):
            central_composite(self.bounds, center=-1)

    def test_center_zero_is_legitimate(self):
        x = central_composite(self.bounds, center=0, coded=True)
        self.assertEqual(x.shape, (8 + 6 + 0, 3))

    def test_numeric_alpha_rejects_nan_inf_and_nonpositive(self):
        # NaN previously slipped through `a <= 0.0` (always False for NaN) and silently produced a
        # design with NaN axial points.
        for bad_alpha in (float("nan"), float("inf"), float("-inf"), -1.0, 0.0):
            with self.subTest(alpha=bad_alpha):
                with self.assertRaises(ValueError):
                    central_composite(self.bounds, alpha=bad_alpha)

    def test_rotatable_axial_points_exceed_bounds_but_factorial_and_center_do_not(self):
        bounds = [(-1.0, 1.0)] * 3
        design = central_composite(bounds, center=4, alpha="rotatable", coded=False)
        kinds = central_composite_point_kinds(bounds, center=4)
        self.assertEqual(kinds.shape, (design.shape[0],))
        self.assertEqual(list(kinds[:8]), ["factorial"] * 8)
        self.assertEqual(list(kinds[8:14]), ["axial"] * 6)
        self.assertEqual(list(kinds[14:]), ["center"] * 4)
        axial_rows = design[kinds == "axial"]
        self.assertGreater(np.max(np.abs(axial_rows)), 1.0)  # extends past the +/-1 factorial cube
        non_axial = design[kinds != "axial"]
        self.assertTrue(np.all(non_axial >= -1.0 - 1e-9) and np.all(non_axial <= 1.0 + 1e-9))

    def test_face_alpha_keeps_axial_points_within_bounds_too(self):
        bounds = [(-1.0, 1.0)] * 3
        design = central_composite(bounds, center=4, alpha="face", coded=False)
        kinds = central_composite_point_kinds(bounds, center=4)
        axial_rows = design[kinds == "axial"]
        self.assertTrue(np.all(axial_rows >= -1.0 - 1e-9) and np.all(axial_rows <= 1.0 + 1e-9))


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

    def test_center_rejects_fractional_and_negative(self):
        with self.assertRaises(ValueError):
            box_behnken([(-1, 1)] * 3, center=2.5)
        with self.assertRaises(ValueError):
            box_behnken([(-1, 1)] * 3, center=-1)

    def test_center_zero_is_legitimate(self):
        x = box_behnken([(-1, 1)] * 3, center=0, coded=True)
        self.assertEqual(x.shape, (4 * 3 + 0, 3))

    def test_explicit_center_overrides_default_table(self):
        # Negative control: a valid explicit integer center still works exactly as before.
        x = box_behnken([(-1, 1)] * 3, center=7, coded=True)
        self.assertEqual(x.shape, (4 * 3 + 7, 3))


class SharedBoundsValidationTest(unittest.TestCase):
    """MXR-080-0174: every generator in this module shares designs._as_bounds."""

    def test_infinite_or_nan_bounds_rejected_everywhere(self):
        bad_bounds_cases = ([(-1.0, np.inf)] * 3, [(np.nan, 1.0)] * 3, [(-np.inf, 1.0)] * 3)
        for bad_bounds in bad_bounds_cases:
            with self.subTest(bounds=bad_bounds, fn="central_composite"):
                with self.assertRaises(ValueError):
                    central_composite(bad_bounds)
            with self.subTest(bounds=bad_bounds, fn="box_behnken"):
                with self.assertRaises(ValueError):
                    box_behnken(bad_bounds)
            with self.subTest(bounds=bad_bounds, fn="plackett_burman"):
                with self.assertRaises(ValueError):
                    plackett_burman(bad_bounds)


if __name__ == "__main__":
    unittest.main()
