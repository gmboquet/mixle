"""Factorial-effects and response-surface analysis (mixle.doe.analysis)."""

import unittest

import numpy as np

from mixle.doe import (
    central_composite,
    design_diagnostics,
    factorial_effects,
    fractional_factorial,
    latin_hypercube,
    response_surface,
)
from mixle.doe.optimal import polynomial_features


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


class FactorialEffectsIdentifiabilityTest(unittest.TestCase):
    """MXR-080-0163: rank-deficient / aliased / undersized / non-finite / mis-coded designs must not
    be silently handed to least squares and reported as if every term had a unique estimate."""

    def test_coded_true_rejects_arbitrary_numeric_levels(self):
        # each entry is individually a plausible "level", but coded=True demands the actual coded
        # +/-1 convention, not just any two-level column.
        x = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 7.0], [5.0, 7.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            factorial_effects(x, y, coded=True)

    def test_rejects_non_finite_design(self):
        x = fractional_factorial([(-1, 1)] * 2, "a b", coded=True)
        y = np.array([1.0, np.nan, 3.0, 4.0])
        with self.assertRaises(ValueError):
            factorial_effects(x, y, coded=True)

    def test_rejects_non_finite_response(self):
        x = fractional_factorial([(-1, 1)] * 2, "a b", coded=True)
        y = np.array([1.0, np.inf, 3.0, 4.0])
        with self.assertRaises(ValueError):
            factorial_effects(x, y, coded=True)

    def test_rank_deficient_design_not_explained_by_aliasing_is_rejected(self):
        # 3 runs asked to estimate 4 parameters (intercept, x0, x1, x0:x1); rank-deficient, and not a
        # clean fractional design (no pair of columns is exactly proportional), so nothing is estimable.
        x = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
        y = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            factorial_effects(x, y, coded=True)

    def test_classical_resolution_iii_alias_chain_reports_contrasts_not_fake_individual_effects(self):
        # 2**(3-1) half-fraction with defining relation I = ABC: x2 = x0*x1 exactly, so the fitted
        # model's x2 term is indistinguishable from the x0:x1 interaction term (and, by the same
        # defining relation, x0 from x1:x2, and x1 from x0:x2) -- a textbook resolution-III alias
        # structure, not a bug in the design.
        x = fractional_factorial([(-1, 1)] * 3, "a b ab", coded=True)
        y = 5.0 + 2.0 * x[:, 0] - 1.0 * x[:, 1] + 3.0 * x[:, 2]  # no true interactions
        fe = factorial_effects(x, y, coded=True, interactions=True)

        # individual terms inside an alias group have no separate estimate
        self.assertEqual(fe.aliases["x0"], ["x1:x2"])
        self.assertEqual(fe.aliases["x1"], ["x0:x2"])
        self.assertEqual(fe.aliases["x2"], ["x0:x1"])
        self.assertEqual(fe.aliases["intercept"], [])
        d = fe.as_dict()
        for aliased_term in ("x0", "x1", "x2", "x0:x1", "x0:x2", "x1:x2"):
            self.assertTrue(np.isnan(d[aliased_term]), aliased_term)

        # the intercept is unaliased and unaffected
        self.assertAlmostEqual(fe.intercept, 5.0)

        # but the combined (aliased-with-zero-true-effect) contrasts ARE estimable and exactly recover
        # the true generating coefficients, in the same 2x "effect" units as `effects`
        self.assertAlmostEqual(fe.estimable_contrasts["x0+x1:x2"][0], 4.0)
        self.assertAlmostEqual(fe.estimable_contrasts["x1+x0:x2"][0], -2.0)
        self.assertAlmostEqual(fe.estimable_contrasts["x2+x0:x1"][0], 6.0)

    def test_full_rank_design_is_unaffected_and_reports_sensible_uncertainty(self):
        # negative control: a genuine, properly-coded, full-rank design must still fit cleanly, with
        # every term uniquely estimable and no aliasing reported.
        rng = np.random.RandomState(0)
        x = fractional_factorial([(-1, 1)] * 2, "a b", coded=True)
        x = np.vstack([x, x])  # replicate for spare degrees of freedom to estimate uncertainty
        true_y = 10 + 3 * x[:, 0] - 2 * x[:, 1] + 1.5 * x[:, 0] * x[:, 1]
        y = true_y + rng.normal(scale=0.05, size=x.shape[0])
        fe = factorial_effects(x, y, coded=True)

        self.assertEqual(fe.aliases, {"intercept": [], "x0": [], "x1": [], "x0:x1": []})
        self.assertEqual(fe.estimable_contrasts, {})
        eff = fe.as_dict()
        self.assertAlmostEqual(eff["x0"], 6.0, delta=0.5)
        self.assertAlmostEqual(eff["x1"], -4.0, delta=0.5)
        self.assertAlmostEqual(eff["x0:x1"], 3.0, delta=0.5)

        self.assertIsNotNone(fe.se)
        self.assertTrue(np.all(np.isfinite(fe.se)))
        self.assertTrue(np.all(np.asarray(fe.se) > 0))
        # the true effects should land comfortably within a handful of standard errors
        for term, true_val in (("x0", 6.0), ("x1", -4.0), ("x0:x1", 3.0)):
            idx = fe.terms.index(term)
            self.assertLess(abs(eff[term] - true_val), 5 * fe.se[idx])


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
