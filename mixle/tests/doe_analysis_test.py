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


class ResponseSurfaceStationarityTest(unittest.TestCase):
    """MXR-080-0164: a singular B must not silently produce a fake "stationary point" whose gradient
    is not actually zero, and a genuine degenerate (ridge) surface must classify as such."""

    def _ccd(self):
        return central_composite([(-2, 2)] * 2, center=5, alpha="rotatable", coded=True)

    def test_no_exact_solution_is_reported_explicitly_not_as_a_fake_stationary_point(self):
        # y = x0**2 + x1 has gradient [2*x0, 1] -- the x1 component is the constant 1 and can never be
        # cancelled, so there is NO stationary point anywhere, not even a ridge. B = [[2, 0], [0, 0]] is
        # singular, so an unguarded least-squares solve returns *some* point without checking that its
        # gradient is actually zero.
        x = self._ccd()
        y = x[:, 0] ** 2 + x[:, 1]
        rs = response_surface(x, y)
        self.assertEqual(rs.kind, "no_stationary_point")
        self.assertIsNone(rs.stationary_point)

    def test_genuine_ridge_classifies_as_ridge_not_saddle(self):
        # y = x0**2 is exactly flat along x1 (zero curvature, zero gradient in that direction
        # everywhere): a genuine, exactly-solvable ridge -- the whole line x0=0 is stationary. This
        # must be distinguished both from "no_stationary_point" (a real solution exists here) and from
        # the previous always-falls-through-to-"saddle" bug.
        x = self._ccd()
        y = x[:, 0] ** 2
        rs = response_surface(x, y)
        self.assertEqual(rs.kind, "ridge")
        self.assertIsNotNone(rs.stationary_point)
        np.testing.assert_allclose(rs.gradient(rs.stationary_point), 0.0, atol=1e-6)
        # a ridge has (at least) one near-zero eigenvalue alongside a clearly nonzero one
        self.assertLess(np.min(np.abs(rs.eigenvalues)), 1e-6)
        self.assertGreater(np.max(np.abs(rs.eigenvalues)), 0.5)

    def test_well_conditioned_surfaces_are_unaffected(self):
        # negative control: a real (non-singular) minimum/maximum/saddle must still classify and
        # locate its stationary point exactly as before.
        x = self._ccd()
        y_max = 20 - 2 * (x[:, 0] - 0.5) ** 2 - 4 * (x[:, 1] + 0.25) ** 2
        rs_max = response_surface(x, y_max)
        self.assertEqual(rs_max.kind, "maximum")
        self.assertIsNotNone(rs_max.stationary_point)
        np.testing.assert_allclose(rs_max.stationary_point, [0.5, -0.25], atol=1e-6)
        np.testing.assert_allclose(rs_max.gradient(rs_max.stationary_point), 0.0, atol=1e-8)
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


class DesignDiagnosticsReferenceDesignTest(unittest.TestCase):
    """MXR-080-0165: `ref` is documented as a reference *design* (raw points) and must be transformed
    through `model` like `design` itself is, not fed straight into the prediction-variance quadratic
    form as if it were already a model matrix."""

    def test_ref_is_transformed_through_model_not_used_as_raw_rows(self):
        # a 1D linear model with an intercept: design's own model matrix has 2 columns, so a raw
        # (single-column) ref fed in directly is the exact shape mismatch the finding describes.
        design = np.array([[-1.0], [1.0]])
        model = polynomial_features(1)
        ref = np.array([[-0.5], [0.0], [0.5]])
        diag = design_diagnostics(design, model, ref=ref)

        # ground truth computed independently, from first principles, not by calling the library again
        f = model(design)
        inv = np.linalg.inv(f.T @ f)
        f_ref = model(ref)
        pred_var = np.einsum("ij,jk,ik->i", f_ref, inv, f_ref)
        expected = f.shape[1] / (f.shape[0] * np.max(pred_var))

        self.assertAlmostEqual(diag["g_efficiency"], expected)
        self.assertAlmostEqual(diag["g_efficiency"], 1.6)
        self.assertNotAlmostEqual(diag["g_efficiency"], 4.0)

    def test_ref_equal_to_design_matches_self_referential_default(self):
        # passing the design's own raw points back in as `ref` is a legitimate, natural use of the
        # parameter and must exactly reproduce the ref=None ("or the design itself") default -- an
        # asymmetric design is used so this does not hold by any accidental symmetry cancellation.
        design = np.array([[-1.0], [0.3], [1.0]])
        model = polynomial_features(1)
        default = design_diagnostics(design, model)
        via_ref = design_diagnostics(design, model, ref=design)
        self.assertAlmostEqual(default["g_efficiency"], via_ref["g_efficiency"])
        self.assertAlmostEqual(default["a_efficiency"], via_ref["a_efficiency"])

    def test_ref_with_wrong_factor_width_is_rejected(self):
        design = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])  # 2 factors
        model = polynomial_features(1)
        bad_ref = np.array([[-1.0], [1.0]])  # 1 raw factor column, design has 2
        with self.assertRaises(ValueError):
            design_diagnostics(design, model, ref=bad_ref)

    def test_self_referential_g_efficiency_never_exceeds_one(self):
        # negative control: g_efficiency compared against the design's own points (the documented
        # default, and the only shape ref=design can legitimately take) is bounded by construction --
        # the hat-matrix leverages of a design against itself always sum to p, so their max can never
        # be below the average p/n. Check this holds broadly, not just for one orthogonal example.
        rng = np.random.RandomState(7)
        for d in (1, 2, 3):
            for _ in range(5):
                design = rng.uniform(-1, 1, size=(rng.randint(6, 25), d))
                diag = design_diagnostics(design, polynomial_features(rng.randint(1, 3)))
                self.assertLessEqual(diag["g_efficiency"], 1.0 + 1e-6)
                self.assertTrue(np.isfinite(diag["g_efficiency"]))

    def test_wider_reference_region_reports_worse_efficiency_than_self_check(self):
        # a sensible, real-world use of ref: checking prediction variance over a region that extends
        # past the design itself (extrapolation risk) should score worse than the design's own
        # self-check, not better -- this only holds once ref is correctly feature-expanded.
        design = np.array([[-1.0], [1.0]])
        model = polynomial_features(1)
        self_check = design_diagnostics(design, model)
        extrapolated = design_diagnostics(design, model, ref=np.array([[-3.0], [3.0]]))
        self.assertLess(extrapolated["g_efficiency"], self_check["g_efficiency"])


if __name__ == "__main__":
    unittest.main()
