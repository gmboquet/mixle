"""Campaign regressions for :mod:`mixle.inference.glm`: the weight convention, and separation.

Two confirmed-major findings from the 0.8.0 external test campaign.

* T1-02 -- ``glm(weights=)`` read the weights as FREQUENCIES in the log-likelihood and as ANALYTIC
  (precision) weights in the standard errors, with no convention stated anywhere in the shipped
  package. The two are mutually exclusive and provably so from one result object: on a 6-row table
  whose counts sum to 114 the standard errors were exactly invariant under ``w -> 1000 w`` (the
  definition of an analytic weight) while the log-likelihood scaled by exactly 1000 (the definition
  of a frequency weight). Whichever the user believed they asked for, one headline output was wrong
  -- standard errors off by ``sqrt((114-2)/(6-2)) = 5.29``, p-values 5e-04 against 6e-82 -- and the
  sanity check of comparing the log-likelihood against a fit on expanded data PASSED to 15 digits,
  certifying the standard errors it disagreed with. BIC was wrong under BOTH readings, in every
  family including the fixed-dispersion ones, because it paired a likelihood summed over ``sum(w)``
  units with a ``log(rows)`` penalty. The fix states the convention, completes it, and offers the
  other one explicitly as ``weight_type="frequency"``.
* T1-03 -- the ``PerfectSeparationError`` guard fired for continuously coded separation but not for
  the dummy-coded form of the SAME data, which is the shape separation actually arrives in (a
  treatment arm with zero events). The guard triggered on fitted probabilities reaching exactly 0
  or 1; a two-point design stalls IRLS at ``min(mu) = 2.1e-11``, just short, and escaped with
  ``converged=True``, zero warnings, and a Wald p-value of 0.996 on a 2x2 whose Fisher exact p is
  0.0202. Detection is now the coding-independent linear-feasibility test for separation, so it
  cannot be moved by rescaling a dummy. The boundary matters as much as the defect: a merely
  well-separated but identifiable logistic fit reaches the boundary just as closely (true slope 8
  gives ``min(mu) = 3.4e-11``, CLOSER than the separated design), so any threshold on ``mu`` or
  ``|eta|`` able to catch the defect would also reject legitimate fits. That control is
  ``IdentifiableButExtremeFitsTest`` below, and it is not optional.
"""

import unittest
import warnings

import numpy as np
from scipy import stats

from mixle.inference.glm import PerfectSeparationError, glm

# The aggregated frequency table the campaign reproduced on: 6 distinct rows, counts summing to 114.
_TABLE_X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])
_TABLE_Y = np.array([2.0, 3.5, 4.0, 6.5, 7.0, 9.5])
_TABLE_COUNTS = np.array([12.0, 30.0, 7.0, 41.0, 19.0, 5.0])
_TABLE_INTEGER_Y = np.array([1.0, 2.0, 0.0, 3.0, 4.0, 6.0])

_SEPARATED_X = np.column_stack([np.ones(40), np.r_[np.ones(20), np.zeros(20)]])
_SEPARATED_Y = np.r_[np.ones(20), np.zeros(20)]
# 6 events among 20 exposed, 0 among 20 unexposed -- the commonest real shape of separation
_ZERO_CELL_Y = np.r_[np.ones(6), np.zeros(34)]


def _expanded(x: np.ndarray, y: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The row-by-row data set a frequency-weighted fit claims to stand in for."""
    repeats = counts.astype(int)
    return np.repeat(x, repeats, axis=0), np.repeat(y, repeats)


class WeightConventionIsStatedTest(unittest.TestCase):
    """T1-02: a user must be able to find out which convention they are getting."""

    def test_the_docstring_names_both_conventions_and_the_switch(self):
        # the campaign's decisive point was not that a convention was wrong but that NONE was
        # stated: the only text in the shipped wheel was "prior weights", so the two answers
        # were indistinguishable from the documentation
        doc = glm.__doc__
        for token in ("weight_type", "analytic", "frequency", "precision"):
            self.assertIn(token, doc)
        self.assertIn("weight_type", glm.__code__.co_varnames)

    def test_the_convention_is_reported_on_the_result(self):
        analytic = glm(_TABLE_X, _TABLE_Y, weights=_TABLE_COUNTS)
        counted = glm(_TABLE_X, _TABLE_Y, weights=_TABLE_COUNTS, weight_type="frequency")
        self.assertEqual(analytic.weight_type, "analytic")
        self.assertEqual(counted.weight_type, "frequency")

    def test_an_unknown_convention_is_refused(self):
        with self.assertRaisesRegex(ValueError, "weight_type must be"):
            glm(_TABLE_X, _TABLE_Y, weights=_TABLE_COUNTS, weight_type="freq")


class AnalyticWeightsAreCoherentTest(unittest.TestCase):
    """T1-02: under the default convention a weight is a precision, everywhere or nowhere."""

    def test_rescaling_analytic_weights_changes_nothing_at_all(self):
        # THE defect, stated as an invariance: an analytic weight carries only RELATIVE precision,
        # so where the dispersion is estimated it is absorbed by phi and the whole fit is
        # scale-free. Before the fix `se` was invariant (analytic) while `log_likelihood` scaled by
        # exactly the rescaling factor (frequency) -- two conventions in one result object.
        for family in ("gaussian", "gamma", "inverse_gaussian"):
            base = glm(_TABLE_X, _TABLE_Y, family=family, weights=_TABLE_COUNTS)
            for factor in (2.0, 10.0, 1000.0):
                with self.subTest(family=family, factor=factor):
                    scaled = glm(_TABLE_X, _TABLE_Y, family=family, weights=factor * _TABLE_COUNTS)
                    np.testing.assert_allclose(scaled.coef, base.coef, rtol=1e-12)
                    np.testing.assert_allclose(scaled.se, base.se, rtol=1e-12)
                    np.testing.assert_allclose(scaled.p_values(), base.p_values(), rtol=1e-9)
                    self.assertAlmostEqual(scaled.log_likelihood, base.log_likelihood, places=7)
                    self.assertAlmostEqual(scaled.aic, base.aic, places=7)
                    self.assertAlmostEqual(scaled.bic, base.bic, places=7)
                    self.assertEqual(scaled.residual_df, base.residual_df)

    def test_the_log_likelihood_is_the_precision_weighted_density(self):
        # an analytic weight divides that observation's dispersion; it does not replicate its
        # log-density. Reconstructed independently from the reported fitted values.
        fit = glm(_TABLE_X, _TABLE_Y, family="gaussian", weights=_TABLE_COUNTS)
        phi_mle = np.sum(_TABLE_COUNTS * (_TABLE_Y - fit.fitted) ** 2) / _TABLE_Y.size
        expected = np.sum(stats.norm.logpdf(_TABLE_Y, fit.fitted, np.sqrt(phi_mle / _TABLE_COUNTS)))
        self.assertAlmostEqual(fit.log_likelihood, float(expected), places=10)
        # and NOT the frequency form, which is what it used to report
        frequency_form = np.sum(_TABLE_COUNTS * stats.norm.logpdf(_TABLE_Y, fit.fitted, np.sqrt(phi_mle)))
        self.assertNotAlmostEqual(fit.log_likelihood, float(frequency_form), places=3)

    def test_fractional_weights_are_precisions_and_are_accepted(self):
        # halving every precision is a rescaling, so it must leave the estimated-dispersion fit
        # identical to the unweighted one -- it used to halve the log-likelihood
        unweighted = glm(_TABLE_X, _TABLE_Y, family="gaussian")
        halved = glm(_TABLE_X, _TABLE_Y, family="gaussian", weights=np.full(6, 0.5))
        np.testing.assert_allclose(halved.se, unweighted.se, rtol=1e-12)
        self.assertAlmostEqual(halved.log_likelihood, unweighted.log_likelihood, places=9)

    def test_unit_weights_are_indistinguishable_from_no_weights(self):
        for family, response in (
            ("gaussian", _TABLE_Y),
            ("gamma", _TABLE_Y),
            ("inverse_gaussian", _TABLE_Y),
            ("poisson", _TABLE_INTEGER_Y),
            ("negativebinomial", _TABLE_INTEGER_Y),
        ):
            with self.subTest(family=family):
                bare = glm(_TABLE_X, response, family=family)
                for weight_type in ("analytic", "frequency"):
                    ones = glm(_TABLE_X, response, family=family, weights=np.ones(6), weight_type=weight_type)
                    np.testing.assert_allclose(ones.se, bare.se, rtol=1e-12)
                    self.assertAlmostEqual(ones.log_likelihood, bare.log_likelihood, places=10)
                    self.assertAlmostEqual(ones.bic, bare.bic, places=10)


class FrequencyWeightsMatchTheExpandedFitTest(unittest.TestCase):
    """T1-02: the other convention, offered explicitly, and pinned to its own definition.

    A frequency weight means "this row stands for w identical observations", so the ONLY correct
    reference is the expanded data set. Every reported quantity is checked against it, not against
    a remembered constant.
    """

    def _pair(self, family, response, **kw):
        x_expanded, y_expanded = _expanded(_TABLE_X, response, _TABLE_COUNTS)
        counted = glm(_TABLE_X, response, family=family, weights=_TABLE_COUNTS, weight_type="frequency", **kw)
        return counted, glm(x_expanded, y_expanded, family=family, **kw)

    def test_every_reported_quantity_matches_the_expanded_data_set(self):
        for family, response in (
            ("gaussian", _TABLE_Y),
            ("gamma", _TABLE_Y),
            ("inverse_gaussian", _TABLE_Y),
            ("poisson", _TABLE_INTEGER_Y),
            ("negativebinomial", _TABLE_INTEGER_Y),
        ):
            counted, expanded = self._pair(family, response)
            with self.subTest(family=family):
                np.testing.assert_allclose(counted.coef, expanded.coef, rtol=1e-9)
                np.testing.assert_allclose(counted.se, expanded.se, rtol=1e-9)
                np.testing.assert_allclose(counted.p_values(), expanded.p_values(), rtol=1e-8)
                self.assertEqual(counted.residual_df, expanded.residual_df)
                self.assertAlmostEqual(counted.dispersion, expanded.dispersion, places=12)
                self.assertAlmostEqual(counted.log_likelihood, expanded.log_likelihood, places=9)
                self.assertAlmostEqual(counted.aic, expanded.aic, places=9)
                self.assertAlmostEqual(counted.bic, expanded.bic, places=9)

    def test_the_robust_sandwich_replicates_rather_than_rescales(self):
        # a frequency weight replicates a score contribution (meat sum w g g'), it does not scale
        # one (sum w^2 g g'). The weighted robust SEs used to disagree with the expanded fit by a
        # non-constant factor, so no rescaling could have reconciled them.
        for family, response in (("gaussian", _TABLE_Y), ("poisson", _TABLE_INTEGER_Y)):
            counted, expanded = self._pair(family, response, robust=True)
            with self.subTest(family=family):
                np.testing.assert_allclose(counted.se, expanded.se, rtol=1e-9)

    def test_a_binomial_frequency_table_matches_its_expansion(self):
        design = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        outcome = np.array([1.0, 0.0, 1.0, 0.0])
        counts = np.array([7.0, 13.0, 15.0, 5.0])
        x_expanded, y_expanded = _expanded(design, outcome, counts)
        counted = glm(design, outcome, family="binomial", weights=counts, weight_type="frequency")
        expanded = glm(x_expanded, y_expanded, family="binomial")
        np.testing.assert_allclose(counted.coef, expanded.coef, rtol=1e-9)
        np.testing.assert_allclose(counted.se, expanded.se, rtol=1e-9)
        self.assertEqual(counted.residual_df, expanded.residual_df)
        self.assertAlmostEqual(counted.bic, expanded.bic, places=9)

    def test_counts_must_be_whole_numbers(self):
        # a fractional count has no expanded data set to stand in for, and the message has to send
        # the user to the reading that does accept fractions rather than just refusing
        with self.assertRaisesRegex(ValueError, "whole numbers") as ctx:
            glm(_TABLE_X, _TABLE_Y, weights=np.full(6, 0.5), weight_type="frequency")
        self.assertIn("analytic", str(ctx.exception))


class BicCountsTheObservationsTheLikelihoodSpansTest(unittest.TestCase):
    """T1-02: BIC was wrong under BOTH conventions, in EVERY family."""

    def test_fixed_dispersion_families_agree_with_their_expansion(self):
        # the sharpest form of the defect: for Poisson / binomial / negative-binomial the weighted
        # and expanded fits agree exactly on coefficients, standard errors, log-likelihood AND aic
        # -- and used to disagree on bic alone (342.826 against 348.715), because the penalty
        # counted rows while the likelihood counted replicates
        for family, response in (("poisson", _TABLE_INTEGER_Y), ("negativebinomial", _TABLE_INTEGER_Y)):
            x_expanded, y_expanded = _expanded(_TABLE_X, response, _TABLE_COUNTS)
            expanded = glm(x_expanded, y_expanded, family=family)
            for weight_type in ("analytic", "frequency"):
                with self.subTest(family=family, weight_type=weight_type):
                    weighted = glm(_TABLE_X, response, family=family, weights=_TABLE_COUNTS, weight_type=weight_type)
                    self.assertAlmostEqual(weighted.aic, expanded.aic, places=9)
                    self.assertAlmostEqual(weighted.bic, expanded.bic, places=9)

    def test_bic_uses_the_reported_likelihood_sample_size(self):
        for weight_type, expected in (("analytic", 6.0), ("frequency", 114.0)):
            with self.subTest(weight_type=weight_type):
                fit = glm(_TABLE_X, _TABLE_Y, weights=_TABLE_COUNTS, weight_type=weight_type)
                self.assertEqual(fit.ll_nobs, expected)
                penalty = np.log(fit.ll_nobs) * (fit.rank + 1)
                self.assertAlmostEqual(fit.bic, -2.0 * fit.log_likelihood + penalty, places=9)

    def test_a_fixed_dispersion_likelihood_always_spans_the_weight_total(self):
        # a discrete density has no analytic-weight form, so `sum w log f` counts sum(w) units
        # whichever convention the standard errors follow -- and BIC has to say so
        fit = glm(_TABLE_X, _TABLE_INTEGER_Y, family="poisson", weights=_TABLE_COUNTS)
        self.assertEqual(fit.ll_nobs, 114.0)
        self.assertEqual(fit.residual_df, 6 - fit.rank)


class SeparationIsDetectedWhateverTheCodingTest(unittest.TestCase):
    """T1-03: the guard must key on the separation, not on the predictor's coding."""

    def test_dummy_coded_separation_is_named(self):
        # the finding verbatim: this returned coef=[-24.566, 49.132], se=[1e5, 1.5e5],
        # p=[0.9998, 0.9997], converged=True and zero warnings, while the continuously coded form
        # of the SAME data raised
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(_SEPARATED_X, _SEPARATED_Y, family="binomial")
        self.assertTrue(fit.separated)
        self.assertEqual(len(caught), 1)
        self.assertIn("perfect separation", str(caught[0].message))
        with self.assertRaisesRegex(ValueError, "undefined under perfect separation"):
            fit.p_values()
        with self.assertRaises(ValueError):
            fit.z_values()

    def test_neither_the_dummy_scale_nor_the_sample_size_can_hide_it(self):
        # the campaign swept these: every coding and every arm size stalled IRLS at the identical
        # point and escaped the old guard, so all of them have to be caught now
        for low, high in ((0.0, 1.0), (0.0, 10.0), (0.0, 100.0), (-1.0, 1.0)):
            for per_arm in (5, 20, 500):
                with self.subTest(coding=(low, high), per_arm=per_arm):
                    design = np.column_stack(
                        [np.ones(2 * per_arm), np.r_[np.full(per_arm, high), np.full(per_arm, low)]]
                    )
                    response = np.r_[np.ones(per_arm), np.zeros(per_arm)]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = glm(design, response, family="binomial")
                    self.assertTrue(fit.separated)

    def test_a_zero_cell_no_longer_reports_no_effect(self):
        # 6/20 exposed against 0/20 unexposed: Fisher's exact p is 0.0202, and the Wald branch
        # reported p = 0.9962 for the exposure coefficient with nothing to contradict it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = glm(_SEPARATED_X, _ZERO_CELL_Y, family="binomial")
        self.assertTrue(fit.separated)
        with self.assertRaises(ValueError):
            fit.p_values()

    def test_robust_standard_errors_do_not_route_around_the_refusal(self):
        # robust=True on the same separated fit gave p = [0.0, 1.8e-266] -- "overwhelming effect"
        # from the machinery that model-based SEs called "no effect", on identical data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = glm(_SEPARATED_X, _ZERO_CELL_Y, family="binomial", robust=True)
        self.assertTrue(fit.separated)
        with self.assertRaises(ValueError):
            fit.p_values()

    def test_a_factor_level_that_perfectly_predicts_is_caught(self):
        design = np.column_stack(
            [np.ones(45), np.r_[np.ones(15), np.zeros(30)], np.r_[np.zeros(15), np.ones(15), np.zeros(15)]]
        )
        response = np.r_[np.ones(15), np.ones(7), np.zeros(8), np.zeros(15)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = glm(design, response, family="binomial")
        self.assertTrue(fit.separated)


class SeparationLeavesTheValidOutputsAloneTest(unittest.TestCase):
    """T1-03: name the undefined branch, do not discard the fit -- the rest of it is correct."""

    def _fit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return glm(_SEPARATED_X, _ZERO_CELL_Y, family="binomial")

    def test_the_deviance_path_still_gives_the_right_answer(self):
        fit = self._fit()
        null = glm(np.ones((40, 1)), _ZERO_CELL_Y, family="binomial")
        statistic = null.deviance - fit.deviance
        # an independent likelihood-ratio test on the 2x2, computed from the cell counts alone
        exposed, unexposed, pooled = 6.0 / 20.0, 0.0, 6.0 / 40.0
        expected = 2.0 * (
            20 * (exposed * np.log(exposed / pooled) + (1 - exposed) * np.log((1 - exposed) / (1 - pooled)))
            + 20 * np.log((1 - unexposed) / (1 - pooled))
        )
        self.assertAlmostEqual(statistic, float(expected), places=6)
        self.assertLess(stats.chi2.sf(statistic, 1), 0.01)

    def test_the_fitted_probabilities_are_the_observed_rates(self):
        fit = self._fit()
        self.assertAlmostEqual(float(fit.fitted[0]), 0.3, places=9)
        self.assertLess(float(fit.fitted[-1]), 1e-8)

    def test_the_result_is_still_returned_with_a_usable_likelihood(self):
        # callers that only want coefficients or a model score keep working; only the Wald branch
        # refuses, exactly as a rank-deficient fit does
        fit = self._fit()
        self.assertTrue(fit.converged)
        self.assertEqual(fit.rank, 2)
        self.assertTrue(np.all(np.isfinite(fit.coef)))
        self.assertTrue(np.isfinite(fit.log_likelihood))
        self.assertTrue(np.isfinite(fit.aic))


class IdentifiableButExtremeFitsTest(unittest.TestCase):
    """T1-03 GUARD-OVERREACH CONTROL: a well-separated but identifiable fit is ordinary work.

    These fits reach the boundary at least as closely as the separated designs above -- a true
    slope of 8 gives ``min(mu) = 3.4e-11`` against the separated design's 2.1e-11 -- so they are
    exactly what a threshold on the fitted probabilities or on ``|eta|`` would destroy. Every one
    of them must fit silently and answer p_values().
    """

    def test_strong_but_overlapping_logistic_fits_are_untouched(self):
        rng = np.random.default_rng(0)
        for slope in (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0):
            with self.subTest(slope=slope):
                predictor = rng.normal(size=200)
                probability = 1.0 / (1.0 + np.exp(-(-0.3 + slope * predictor)))
                response = (rng.uniform(size=200) < probability).astype(float)
                design = np.column_stack([np.ones(200), predictor])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = glm(design, response, family="binomial")
                self.assertFalse(fit.separated)
                self.assertEqual([str(w.message) for w in caught], [])
                self.assertTrue(np.all(np.isfinite(fit.p_values())))

    def test_one_overlapping_observation_is_enough_to_identify_the_fit(self):
        # the separated design with a single observation flipped in each arm: the MLE exists again,
        # so the detector must reverse, on data one point away from the case it does flag
        response = np.r_[np.ones(19), np.zeros(1), np.zeros(19), np.ones(1)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = glm(_SEPARATED_X, response, family="binomial")
        self.assertFalse(fit.separated)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertTrue(np.all(np.isfinite(fit.p_values())))

    def test_a_sparse_but_nonempty_cell_still_fits(self):
        # one event in each arm: small, lopsided, and perfectly legitimate
        response = np.r_[np.ones(1), np.zeros(19), np.ones(1), np.zeros(19)]
        fit = glm(_SEPARATED_X, response, family="binomial")
        self.assertFalse(fit.separated)
        self.assertTrue(np.all(np.isfinite(fit.p_values())))

    def test_a_proportion_response_carrying_both_classes_is_not_separation(self):
        design = np.column_stack([np.ones(4), np.arange(4.0)])
        fit = glm(design, np.array([0.1, 0.4, 0.6, 0.9]), family="binomial")
        self.assertFalse(fit.separated)


class SeparationRemedyIsHonestTest(unittest.TestCase):
    """T1-03 secondary: the raising path pointed at a remedy that does not exist for binomial."""

    def test_the_error_does_not_promise_penalized_logistic_regression(self):
        # ridge_regression / elastic_net take no `family` argument and return least-squares
        # coefficients on separated 0/1 data, so sending a user there gives a linear probability
        # model, not the finite logistic fit the old wording implied
        design = np.column_stack([np.ones(40), np.r_[np.linspace(1, 2, 20), np.linspace(-2, -1, 20)]])
        with self.assertRaises(PerfectSeparationError) as ctx:
            glm(design, _SEPARATED_Y, family="binomial")
        message = str(ctx.exception)
        self.assertIn("LINEAR PROBABILITY", message)
        self.assertIn("deviance", message)

    def test_the_exception_docstring_does_not_certify_the_absence_of_separation(self):
        self.assertIn("separated", PerfectSeparationError.__doc__)


if __name__ == "__main__":
    unittest.main()
