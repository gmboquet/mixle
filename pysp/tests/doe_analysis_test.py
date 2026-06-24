"""Factorial-effects and response-surface analysis (pysp.doe.analysis)."""

import unittest

import numpy as np

from pysp.doe import central_composite, factorial_effects, fractional_factorial, response_surface


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


if __name__ == "__main__":
    unittest.main()
