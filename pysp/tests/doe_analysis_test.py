"""Factorial-effects and response-surface analysis (pysp.doe.analysis)."""

import unittest

import numpy as np

from pysp.doe import (
    central_composite,
    design_diagnostics,
    factorial_effects,
    fractional_factorial,
    latin_hypercube,
    response_surface,
)
from pysp.doe.optimal import polynomial_features


class FactorialEffectsTest(unittest.TestCase):
    def test_recovers_known_effects(self):
        x = fractional_factorial([(-1, 1)] * 2, "a b", coded=True)
        y = 10 + 3 * x[:, 0] - 2 * x[:, 1] + 1.5 * x[:, 0] * x[:, 1]  # coded model
        fe = factorial_effects(x, y, coded=True)
        eff = fe.as_dict()
        self.assertAlmostEqual(fe.intercept, 10.0)  # grand mean
        self.assertAlmostEqual(eff["x0"], 6.0)  # effect = 2 * coefficient
        self.assertAlmostEqual(eff["x1"], -4.0)
        self.assertAlmostEqual(eff["x0:x1"], 3.0)

    def test_auto_codes_real_levels(self):
        # real factor levels (not +/-1) must be coded internally to give the same effects
        x = fractional_factorial([(0.0, 10.0), (100.0, 200.0)], "a b")
        coded = fractional_factorial([(-1, 1)] * 2, "a b", coded=True)
        y = 5 + 2 * coded[:, 0] - coded[:, 1]
        eff = factorial_effects(x, y).as_dict()
        self.assertAlmostEqual(eff["x0"], 4.0)
        self.assertAlmostEqual(eff["x1"], -2.0)

    def test_rejects_three_level_factor(self):
        with self.assertRaises(ValueError):
            factorial_effects(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 2.0, 3.0]))


class ResponseSurfaceTest(unittest.TestCase):
    def _ccd(self):
        return central_composite([(-2, 2)] * 2, center=5, alpha="rotatable", coded=True)

    def test_finds_maximum(self):
        x = self._ccd()
        y = 20 - 2 * (x[:, 0] - 0.5) ** 2 - 4 * (x[:, 1] + 0.25) ** 2  # concave, max at (0.5, -0.25)
        rs = response_surface(x, y)
        self.assertEqual(rs.kind, "maximum")
        np.testing.assert_allclose(rs.stationary_point, [0.5, -0.25], atol=1e-6)
        self.assertTrue(np.all(rs.eigenvalues < 0))
        self.assertAlmostEqual(rs.predict(rs.stationary_point)[0], 20.0, places=6)
        np.testing.assert_allclose(rs.gradient(rs.stationary_point), 0.0, atol=1e-8)

    def test_classifies_minimum_and_saddle(self):
        x = self._ccd()
        self.assertEqual(response_surface(x, 1 + x[:, 0] ** 2 + x[:, 1] ** 2).kind, "minimum")
        self.assertEqual(response_surface(x, 5 + x[:, 0] ** 2 - x[:, 1] ** 2).kind, "saddle")


class DesignDiagnosticsTest(unittest.TestCase):
    def test_orthogonal_factorial_is_perfectly_efficient(self):
        x = fractional_factorial([(-1, 1)] * 3, "a b c", coded=True)
        d = design_diagnostics(x, polynomial_features(1))
        self.assertAlmostEqual(d["d_efficiency"], 1.0)
        self.assertAlmostEqual(d["a_efficiency"], 1.0)
        self.assertAlmostEqual(d["g_efficiency"], 1.0)
        self.assertAlmostEqual(d["condition_number"], 1.0)
        self.assertAlmostEqual(d["max_correlation"], 0.0)

    def test_factorial_beats_random_lhs(self):
        ff = design_diagnostics(fractional_factorial([(-1, 1)] * 3, "a b c", coded=True), polynomial_features(1))
        lhs = design_diagnostics(latin_hypercube([(-1, 1)] * 3, 8, seed=1), polynomial_features(1))
        self.assertGreaterEqual(ff["d_efficiency"], lhs["d_efficiency"])
        self.assertLessEqual(ff["max_correlation"], lhs["max_correlation"])


if __name__ == "__main__":
    unittest.main()
