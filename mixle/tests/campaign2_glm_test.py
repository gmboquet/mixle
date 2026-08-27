"""Campaign-2 regressions for :mod:`mixle.inference.glm`: saturation, HC variants, API precision.

Three findings from the second verification pass of the 0.8.0 external test campaign.

* T1-06 -- gaussian / gamma / inverse-Gaussian fits refused the ENTIRE fit whenever
  ``residual_df <= 0`` (``ValueError: dispersion is not identifiable...``), including with
  ``robust=True`` -- while the sibling degeneracy (rank deficiency with positive residual df)
  already returned the computable part and refused only the Wald branch. A saturated fit's
  coefficients, fitted values and deviance are perfectly well defined; only the dispersion, the
  standard errors (robust ones included: every residual is numerically zero, so a sandwich would
  claim exact knowledge) and the maximised likelihood (unbounded as ``phi -> 0``) are not. The fix
  routes saturation through the same disclose-and-refuse machinery: warn at fit time, return the
  fit with ``dispersion``/``se`` NaN and ``log_likelihood`` None, and refuse ``z_values`` /
  ``p_values`` / ``aic`` / ``bic`` with the real reason.
* T1-07 -- ``robust=`` offered HC0 only, and HC0 is biased LOW exactly where robust standard
  errors are reached for: on the leveraged 11-point design below its slope SE is 86% under HC3's
  (0.041 against 0.287). Worse, ``robust="HC3"`` was read as a truthy bool and SILENTLY computed
  HC0. The fix implements HC1/HC2/HC3 as per-row rescalings folded into the PSD-by-construction
  ``(S B)'(S B)`` sandwich, validated against statsmodels 0.14.6 (constants below): agreement to
  ~1e-15 relative for gaussian against OLS/WLS and the frequency-weighted table against its
  expanded data set, ~1e-9 for poisson against a hand-built sandwich at statsmodels' converged mu.
* T1-09 -- API-surface precision: unknown family/link errors now print the menu instead of just
  echoing the typo; the aic/bic refusal distinguishes "this family/response defines no
  likelihood" from "the fit is saturated"; and the result records ``cov_type`` so the meaning of
  ``se`` is readable off the result, exactly as ``weight_type`` already was.
"""

import unittest
import warnings

import numpy as np

from mixle.inference.glm import glm

# ---------------------------------------------------------------- shared literal designs
# 11 rows with one high-leverage point (x = 6.0) and error spread tied to |x|
_LEV_XCOL = np.array([-0.6, 0.2, 1.1, -1.3, 0.5, -0.2, 0.9, -0.8, 0.3, -0.1, 6.0])
_LEV_X = np.column_stack([np.ones(11), _LEV_XCOL])
_LEV_Y = np.array([0.83, 1.31, 1.87, 0.02, 1.61, 0.75, 1.20, 0.14, 1.49, 0.98, 7.40])
_LEV_W = np.array([2.0, 0.5, 1.0, 3.0, 1.5, 0.8, 1.2, 2.5, 0.7, 1.1, 0.9])
_LEV_COUNTS = np.array([1.0, 3.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0])

_POIS_XCOL = np.array([-1.1, 0.3, 0.8, -0.4, 1.2, 0.1, -0.7, 0.6, -0.2, 3.0])
_POIS_X = np.column_stack([np.ones(10), _POIS_XCOL])
_POIS_Y = np.array([0.0, 2.0, 3.0, 1.0, 4.0, 1.0, 0.0, 2.0, 1.0, 9.0])

# statsmodels 0.14.6 references on the exact arrays above (see the module docstring):
# sm.OLS(_LEV_Y, _LEV_X).fit().HC{k}_se
_OLS_SE = {
    "HC0": np.array([0.09142353179903073, 0.0410291820188476]),
    "HC1": np.array([0.10107251732883686, 0.04535946740390466]),
    "HC2": np.array([0.09755983158400229, 0.10287734905415093]),
    "HC3": np.array([0.10627837593597785, 0.28666548655448953]),
}
# sm.WLS(_LEV_Y, _LEV_X, weights=_LEV_W).fit().HC{k}_se
_WLS_SE = {
    "HC0": np.array([0.09537467249576076, 0.05701210520940449]),
    "HC1": np.array([0.10544066772382248, 0.06302925382928705]),
    "HC2": np.array([0.11149970807815524, 0.11258848457106241]),
    "HC3": np.array([0.14831643844430767, 0.24366419848327114]),
}
# sm.OLS on the _LEV_COUNTS-expanded (21-row) data set: what the frequency-weighted table claims
_FREQ_SE = {
    "HC0": np.array([0.06662551344407348, 0.02532029335605047]),
    "HC1": np.array([0.07004439940623178, 0.026619603350642812]),
    "HC2": np.array([0.06887144743997402, 0.032006202092229034]),
    "HC3": np.array([0.07126708411093434, 0.04133929380988892]),
}
# poisson: HC0 from sm.GLM(...).fit(cov_type="HC0"); HC1-3 from the canonical-link sandwich
# B (sum a_i^2 g g') B built with plain numpy at statsmodels' converged mu (statsmodels' own GLM
# silently returns HC0 for every HC request, so the leverage-corrected reference is hand-built)
_POIS_SE = {
    "HC0": np.array([0.1573911386045777, 0.067697634296179]),
    "HC1": np.array([0.17596864248796348, 0.07568825610108866]),
    "HC2": np.array([0.1706643884579253, 0.15487611095735251]),
    "HC3": np.array([0.21840464955257086, 0.5453644377515197]),
}


def _saturated_fit(family="gaussian", **kw):
    x = np.array([[1.0, -0.4, 0.9], [1.0, 0.7, -0.2], [1.0, 1.3, 0.5]])
    y = np.array([1.2, 2.7, 3.4])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = glm(x, y, family=family, **kw)
    return fit, caught, x, y


class SaturatedFitReturnsWhatItCanTest(unittest.TestCase):
    """T1-06: saturation joins the disclose-and-refuse machinery instead of a blanket raise."""

    def test_a_square_full_rank_gaussian_fit_returns_its_computable_part(self):
        # before the fix: ValueError("dispersion is not identifiable without positive residual
        # degrees of freedom") -- coefficients, fitted values and deviance discarded with it
        fit, caught, x, y = _saturated_fit()
        self.assertTrue(fit.converged)
        self.assertEqual(fit.rank, 3)
        self.assertEqual(fit.residual_df, 0)
        np.testing.assert_allclose(x @ fit.coef, y, atol=1e-8)  # the interpolating fit
        np.testing.assert_allclose(fit.fitted, y, atol=1e-8)
        self.assertAlmostEqual(fit.deviance, 0.0, places=8)

    def test_what_genuinely_cannot_be_computed_stays_loudly_refused(self):
        fit, caught, _, _ = _saturated_fit()
        self.assertTrue(np.isnan(fit.dispersion))
        self.assertTrue(np.all(np.isnan(fit.se)))
        self.assertIsNone(fit.log_likelihood)
        self.assertEqual(len(caught), 1)
        self.assertIn("saturated", str(caught[0].message))

    def test_every_dispersion_estimating_family_and_both_robust_settings_return(self):
        # the finding named robust=True explicitly: the old raise fired before the sandwich
        # branch was even reached
        for family in ("gaussian", "gamma", "inverse_gaussian"):
            for robust in (False, True):
                with self.subTest(family=family, robust=robust):
                    fit, caught, _, _ = _saturated_fit(family=family, robust=robust)
                    self.assertTrue(np.all(np.isfinite(fit.coef)))
                    self.assertTrue(np.all(np.isnan(fit.se)))
                    self.assertTrue(any("saturated" in str(c.message) for c in caught))

    def test_wald_inference_refuses_with_the_real_reason(self):
        fit, _, _, _ = _saturated_fit()
        with self.assertRaisesRegex(ValueError, "saturated"):
            fit.z_values()
        with self.assertRaisesRegex(ValueError, "saturated"):
            fit.p_values()

    def test_aic_and_bic_name_saturation_not_a_missing_likelihood(self):
        # the generic "no defined likelihood" wording would send the user hunting through the
        # family/response combination when the actual problem is too few observations
        fit, _, _, _ = _saturated_fit()
        for criterion in ("aic", "bic"):
            with self.subTest(criterion=criterion):
                with self.assertRaisesRegex(ValueError, "saturated") as ctx:
                    getattr(fit, criterion)
                self.assertNotIn("defines no likelihood", str(ctx.exception))

    def test_an_underdetermined_design_reaches_the_rank_deficiency_machinery(self):
        # n < p: before the fix the dispersion raise fired AFTER the rank-deficiency warning and
        # threw away the minimum-norm fit that machinery had just disclosed
        x = np.array([[1.0, -0.4, 0.9, 0.3, -1.1], [1.0, 0.7, -0.2, 1.4, 0.6], [1.0, 1.3, 0.5, -0.8, 0.2]])
        y = np.array([1.2, 2.7, 3.4])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(x, y, family="gaussian")
        messages = [str(c.message) for c in caught]
        self.assertTrue(any("rank-deficient" in m for m in messages))
        self.assertTrue(any("saturated" in m for m in messages))
        self.assertTrue(np.all(np.isfinite(fit.coef)))
        with self.assertRaises(ValueError):
            fit.p_values()

    def test_one_extra_row_restores_full_inference(self):
        # guard-overreach control: the boundary sits exactly at residual_df == 0, so n = p + 1
        # must fit silently with finite standard errors and p-values
        x = np.column_stack([np.ones(4), [-0.4, 0.7, 1.3, 0.1], [0.9, -0.2, 0.5, 1.2]])
        y = np.array([1.2, 2.7, 3.4, 1.9])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(x, y, family="gaussian")
        self.assertEqual([str(c.message) for c in caught], [])
        self.assertEqual(fit.residual_df, 1)
        self.assertTrue(np.isfinite(fit.dispersion))
        self.assertTrue(np.all(np.isfinite(fit.se)))
        self.assertTrue(np.all(np.isfinite(fit.p_values())))

    def test_the_positive_residual_df_rank_deficient_path_is_unchanged(self):
        # the sibling degeneracy the finding pointed at: duplicated columns with rows to spare
        # keep their finite dispersion and standard errors, exactly as before
        rng = np.random.RandomState(1)
        base = rng.standard_normal((6, 2))
        x = np.column_stack([base, base])
        y = rng.standard_normal(6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(x, y, family="gaussian")
        self.assertEqual(fit.rank, 2)
        self.assertEqual(fit.residual_df, 4)
        self.assertTrue(np.isfinite(fit.dispersion))
        self.assertTrue(any("rank-deficient" in str(c.message) for c in caught))
        self.assertFalse(any("saturated" in str(c.message) for c in caught))

    def test_frequency_weights_count_replicates_toward_saturation(self):
        x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]])
        y = np.array([1.0, 2.5, 3.5])
        # 5 replicates over rank 2: dispersion identifiable, ordinary fit
        fit = glm(x, y, weights=np.array([2.0, 1.0, 2.0]), weight_type="frequency")
        self.assertEqual(fit.residual_df, 3)
        self.assertTrue(np.isfinite(fit.dispersion))
        # 2 replicates over rank 2: saturated, disclosed
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(x[:2], y[:2], weights=np.array([1.0, 1.0]), weight_type="frequency")
        self.assertEqual(fit.residual_df, 0)
        self.assertTrue(np.isnan(fit.dispersion))
        self.assertTrue(any("saturated" in str(c.message) for c in caught))

    def test_fixed_dispersion_families_at_zero_residual_df_are_untouched(self):
        # phi is fixed at 1 for poisson, so a square design never lacked a dispersion estimate;
        # its standard errors exist and must keep existing (control against overreach)
        x = np.array([[1.0, 0.0], [1.0, 1.0]])
        y = np.array([1.0, 3.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(x, y, family="poisson")
        self.assertEqual([str(c.message) for c in caught], [])
        self.assertEqual(fit.residual_df, 0)
        self.assertEqual(fit.dispersion, 1.0)
        self.assertTrue(np.all(np.isfinite(fit.se)))
        self.assertTrue(np.all(np.isfinite(fit.p_values())))


class LeverageCorrectedSandwichTest(unittest.TestCase):
    """T1-07: HC1/HC2/HC3 as leverage rescalings of the same PSD-by-construction sandwich."""

    def test_gaussian_matches_statsmodels_ols_for_every_variant(self):
        # tolerance achieved on this design: <= 2e-15 relative (statsmodels 0.14.6)
        for hc, expected in _OLS_SE.items():
            with self.subTest(hc=hc):
                fit = glm(_LEV_X, _LEV_Y, family="gaussian", robust=hc)
                np.testing.assert_allclose(fit.se, expected, rtol=1e-10)

    def test_robust_true_still_means_hc0(self):
        # the compatibility contract: every pre-existing robust=True caller is untouched
        np.testing.assert_allclose(glm(_LEV_X, _LEV_Y, family="gaussian", robust=True).se, _OLS_SE["HC0"], rtol=1e-10)

    def test_analytic_weights_match_statsmodels_wls(self):
        # achieved <= 6e-16 relative: the leverage of an analytic-weighted row carries its full
        # working weight, exactly as WLS's whitened-design hat diagonal does
        for hc, expected in _WLS_SE.items():
            with self.subTest(hc=hc):
                fit = glm(_LEV_X, _LEV_Y, family="gaussian", weights=_LEV_W, robust=hc)
                np.testing.assert_allclose(fit.se, expected, rtol=1e-10)

    def test_frequency_weights_match_the_expanded_data_set(self):
        # a frequency-weighted table must give what fitting the expanded rows gives -- for the
        # leverage corrections too, which is why each replicate carries the PER-REPLICATE
        # leverage. Achieved <= 4e-15 relative against statsmodels on the 21 expanded rows.
        expanded_x = np.repeat(_LEV_X, _LEV_COUNTS.astype(int), axis=0)
        expanded_y = np.repeat(_LEV_Y, _LEV_COUNTS.astype(int))
        for hc, expected in _FREQ_SE.items():
            with self.subTest(hc=hc):
                counted = glm(
                    _LEV_X, _LEV_Y, family="gaussian", weights=_LEV_COUNTS, weight_type="frequency", robust=hc
                )
                np.testing.assert_allclose(counted.se, expected, rtol=1e-10)
                expanded = glm(expanded_x, expanded_y, family="gaussian", robust=hc)
                np.testing.assert_allclose(counted.se, expanded.se, rtol=1e-9)

    def test_poisson_matches_the_reference_sandwich(self):
        # achieved <= 9e-10 relative (the gap is the two implementations' IRLS stopping points,
        # not the estimator): HC0 against sm.GLM directly, HC1-3 against the hand-built
        # canonical-link sandwich at statsmodels' converged mu
        for hc, expected in _POIS_SE.items():
            with self.subTest(hc=hc):
                fit = glm(_POIS_X, _POIS_Y, family="poisson", robust=hc)
                np.testing.assert_allclose(fit.se, expected, rtol=1e-6)

    def test_hc1_is_exactly_the_df_rescaled_hc0(self):
        hc0 = glm(_LEV_X, _LEV_Y, family="gaussian", robust="HC0")
        hc1 = glm(_LEV_X, _LEV_Y, family="gaussian", robust="HC1")
        np.testing.assert_allclose(hc1.se, hc0.se * np.sqrt(11.0 / (11.0 - hc0.rank)), rtol=1e-12)

    def test_hc0_understates_the_leverage_corrected_estimators_where_it_matters(self):
        # the finding's magnitude, pinned as an inequality on this design's leveraged slope:
        # HC0 sits 60% under HC2 and 86% under HC3 -- the 25-56% band the campaign measured is
        # design-specific, the direction is not
        se = {hc: glm(_LEV_X, _LEV_Y, family="gaussian", robust=hc).se for hc in ("HC0", "HC1", "HC2", "HC3")}
        self.assertTrue(np.all(se["HC3"] >= se["HC2"] - 1e-12))
        self.assertTrue(np.all(se["HC2"] >= se["HC0"] - 1e-12))
        self.assertTrue(np.all(se["HC1"] >= se["HC0"]))
        self.assertGreater(1.0 - se["HC0"][1] / se["HC3"][1], 0.25)

    def test_a_string_no_longer_silently_means_hc0(self):
        # THE reproduce case: robust="HC3" was read as a truthy bool, so the user asked for the
        # most conservative estimator and silently received the least
        hc3 = glm(_LEV_X, _LEV_Y, family="gaussian", robust="HC3")
        self.assertFalse(np.allclose(hc3.se, _OLS_SE["HC0"], rtol=1e-3))
        np.testing.assert_allclose(hc3.se, _OLS_SE["HC3"], rtol=1e-10)

    def test_lowercase_names_are_accepted(self):
        np.testing.assert_allclose(glm(_LEV_X, _LEV_Y, family="gaussian", robust="hc2").se, _OLS_SE["HC2"], rtol=1e-10)

    def test_an_unknown_estimator_name_refuses_with_the_menu(self):
        for bad in ("HC4", "huber", "sandwich"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "HC0.*HC1.*HC2.*HC3"):
                    glm(_LEV_X, _LEV_Y, family="gaussian", robust=bad)

    def test_the_result_records_which_covariance_it_carries(self):
        self.assertEqual(glm(_LEV_X, _LEV_Y, family="gaussian").cov_type, "model")
        self.assertEqual(glm(_LEV_X, _LEV_Y, family="gaussian", robust=True).cov_type, "HC0")
        self.assertEqual(glm(_LEV_X, _LEV_Y, family="gaussian", robust="hc3").cov_type, "HC3")

    def test_unit_leverage_refuses_hc2_hc3_and_names_the_remedy(self):
        # a dummy level with a single observation is fitted by itself alone (h = 1), so the
        # 1/(1-h) correction divides by zero: refuse loudly, and leave HC0/HC1 available
        x = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0.9, 1.4, 1.1, 0.6, 3.2])
        for hc in ("HC2", "HC3"):
            with self.subTest(hc=hc):
                with self.assertRaisesRegex(ValueError, "leverage 1"):
                    glm(x, y, family="gaussian", robust=hc)
        self.assertTrue(np.all(np.isfinite(glm(x, y, family="gaussian", robust="HC1").se)))

    def test_corrected_estimators_refuse_on_a_saturated_fixed_dispersion_fit(self):
        # n/(n - rank) and 1/(1 - h) both divide by zero at residual_df == 0; HC0 still exists
        x = np.array([[1.0, 0.0], [1.0, 1.0]])
        y = np.array([1.0, 3.0])
        with self.assertRaisesRegex(ValueError, "saturated"):
            glm(x, y, family="poisson", robust="HC1")
        self.assertTrue(np.all(np.isfinite(glm(x, y, family="poisson", robust="HC0").se)))


class ApiSurfacePrecisionTest(unittest.TestCase):
    """T1-09: the cheap coherent fixes -- menus in errors, causes in refusals."""

    def test_an_unknown_family_error_prints_the_menu(self):
        with self.assertRaisesRegex(ValueError, "guassian") as ctx:
            glm(_LEV_X, _LEV_Y, family="guassian")
        message = str(ctx.exception)
        for name in ("gaussian", "binomial", "poisson", "gamma", "inverse_gaussian", "negativebinomial"):
            self.assertIn(name, message)

    def test_an_unknown_link_error_prints_the_menu(self):
        with self.assertRaisesRegex(ValueError, "logti") as ctx:
            glm(_LEV_X, _LEV_Y, family="gaussian", link="logti")
        message = str(ctx.exception)
        for name in ("identity", "log", "logit", "probit", "cloglog"):
            self.assertIn(name, message)

    def test_the_documented_call_forms_are_pinned(self):
        # T1-09 hit both directions of the property/method split in the first ten minutes.
        # Changing either direction now breaks every existing caller on a release branch, so the
        # fix is a truthful docstring plus this pin: aic/bic stay properties, z/p stay methods.
        from mixle.inference.glm import GLMResult

        self.assertIsInstance(GLMResult.aic, property)
        self.assertIsInstance(GLMResult.bic, property)
        fit = glm(_LEV_X, _LEV_Y, family="gaussian")
        self.assertTrue(callable(fit.z_values))
        self.assertTrue(callable(fit.p_values))
        self.assertIn("aic", GLMResult.__doc__)
        self.assertIn("z_values()", GLMResult.__doc__)

    def test_the_no_likelihood_refusal_names_the_family_response_cause(self):
        # a binomial fit to proportions has no likelihood BY DEFINITION -- a different fact from
        # saturation, and the message must say which one applies
        fit = glm(np.column_stack([np.ones(4), np.arange(4.0)]), np.array([0.1, 0.4, 0.6, 0.9]), family="binomial")
        self.assertIsNone(fit.log_likelihood)
        with self.assertRaisesRegex(ValueError, "defines no likelihood") as ctx:
            _ = fit.aic
        self.assertNotIn("saturated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
