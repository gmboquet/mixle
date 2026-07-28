"""Model comparison utilities (mixle.inference.model_comparison)."""

import unittest

import numpy as np
from scipy import stats

from mixle.inference import (
    clarke_test,
    compare_elpd,
    paired_score_difference,
    vuong_test,
)


class PairedScoreTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.y = rng.normal(0, 1, 2000)
        # losses (negative log score): A is the correct model, B over-dispersed
        self.loss_a = -stats.norm.logpdf(self.y, 0, 1)
        self.loss_b = -stats.norm.logpdf(self.y, 0, 2)

    def test_favors_better_model(self):
        res = paired_score_difference(self.loss_a, self.loss_b, lower_is_better=True)
        self.assertEqual(res["favored"], "A")
        self.assertLess(res["mean_diff"], 0)  # A has lower loss
        self.assertLess(res["ci_high"], 0)  # CI excludes zero

    def test_tie_for_equivalent_models(self):
        res = paired_score_difference(self.loss_a, self.loss_a.copy(), lower_is_better=True)
        self.assertEqual(res["favored"], "tie")
        self.assertAlmostEqual(res["mean_diff"], 0.0)

    def test_higher_is_better_flips(self):
        # pass log-likelihoods (higher better): A should still win
        res = paired_score_difference(-self.loss_a, -self.loss_b, lower_is_better=False)
        self.assertEqual(res["favored"], "A")


class VuongClarkeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.y = rng.normal(0, 1, 3000)
        self.ll_a = stats.norm.logpdf(self.y, 0, 1)
        self.ll_b = stats.norm.logpdf(self.y, 0, 2)

    def test_vuong_favors_correct_model(self):
        res = vuong_test(self.ll_a, self.ll_b)
        self.assertEqual(res["favored"], "A")
        self.assertGreater(res["statistic"], 2)

    def test_vuong_tie_for_equivalent(self):
        ll_c = stats.norm.logpdf(self.y, 0.01, 1.0)
        self.assertEqual(vuong_test(self.ll_a, ll_c)["favored"], "tie")

    def test_clarke_favors_correct_model(self):
        res = clarke_test(self.ll_a, self.ll_b)
        self.assertEqual(res["favored"], "A")
        self.assertGreater(res["statistic"], res["n"] / 2)

    def test_bic_correction_penalizes_complexity(self):
        # identical fit but model A has more parameters -> BIC correction should not favor A
        res = vuong_test(self.ll_a, self.ll_a.copy(), k_a=10, k_b=2, correction="bic")
        self.assertIn(res["favored"], ("B", "tie"))


class PairedValidationTest(unittest.TestCase):
    """Regression: none of the four comparison functions validated their paired inputs, so a
    length mismatch silently broadcast (numpy) into a confidently-wrong verdict instead of an
    error, and n=1 starved the ddof=1 standard deviation into NaN, which then silently resolved
    to a specific favored side (NaN comparisons are always False) instead of raising or tying."""

    def test_paired_score_difference_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            paired_score_difference(np.zeros(5), np.zeros(1))

    def test_vuong_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            vuong_test(np.zeros(5), np.zeros(1))

    def test_clarke_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            clarke_test(np.zeros(5), np.zeros(1))

    def test_compare_elpd_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            compare_elpd(np.zeros(5), np.zeros(1))

    def test_vuong_rejects_single_observation_instead_of_a_confident_verdict(self):
        with self.assertRaises(ValueError):
            vuong_test(np.array([1.0]), np.array([2.0]))

    def test_all_four_reject_nonfinite_input(self):
        for fn in (paired_score_difference, vuong_test, clarke_test, compare_elpd):
            with self.assertRaises(ValueError):
                fn(np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_all_four_reject_higher_dimensional_input_instead_of_flattening_it(self):
        # MXR-080-1606: all four document (n,) per-observation arrays but called .ravel() BEFORE
        # validating shape, so a malformed (2, 3) pair became six independent observations -- the
        # paired t route reported p=0.00593, Vuong p=4.59e-06, Clarke p=0.03125 off it. Flattening
        # turns folds/chains/outcomes/repeated measures into pseudo-replicates and overstates
        # precision, and the returned dict keeps no trace of the original unit.
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a + 0.5
        for fn in (paired_score_difference, vuong_test, clarke_test, compare_elpd):
            with self.assertRaises(ValueError) as ctx:
                fn(a, b)
            self.assertIn("(2, 3)", str(ctx.exception))

    def test_column_vectors_are_rejected_not_silently_accepted(self):
        for fn in (paired_score_difference, vuong_test, clarke_test, compare_elpd):
            with self.assertRaises(ValueError):
                fn(np.zeros((4, 1)), np.ones((4, 1)))

    def test_explicit_ravel_is_still_available_for_a_deliberate_flatten(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a + 0.5
        res = paired_score_difference(a.ravel(), b.ravel())
        self.assertEqual(res["favored"], "A")

    def test_lists_of_scalars_are_still_accepted(self):
        res = paired_score_difference([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        self.assertEqual(res["favored"], "A")


class DegenerateStandardErrorTest(unittest.TestCase):
    """Regression: a paired (or pointwise elpd) difference that is EXACTLY constant across every
    observation has se == 0 by construction -- ddof=1 std of a constant array is 0. Both
    paired_score_difference and compare_elpd used to divide-by-zero-guard that into t/z = 0.0
    unconditionally, which silently read a maximally certain nonzero difference as "tie": internally
    contradictory, since the CI (mean_diff +/- t_crit * se) collapses to a single nonzero point --
    "certainly not zero" -- while p=1.0/"tie" says "no evidence of any difference". A zero SE with a
    nonzero constant difference is the STRONGEST evidence a paired test can produce, not the weakest.
    Mirrors geweke_z's identical denom==0-with-nonzero-diff fix in mixle/inference/diagnostics.py.
    """

    def test_paired_score_difference_deterministic_difference_is_decisive_not_a_tie(self):
        # every observation agrees exactly: a=[1,1,1] beats b=[0,0,0] by a constant margin of 1.
        res = paired_score_difference(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(res["se"], 0.0)
        self.assertEqual(res["ci_low"], 1.0)
        self.assertEqual(res["ci_high"], 1.0)
        # ci excludes 0 (certain nonzero difference) -- p_value/favored must agree, not contradict it.
        self.assertEqual(res["p_value"], 0.0)
        self.assertEqual(res["t"], float("inf"))
        self.assertEqual(res["favored"], "B")  # B has the lower (better, lower_is_better=True) scores

    def test_paired_score_difference_deterministic_difference_flips_with_direction(self):
        res = paired_score_difference(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(res["p_value"], 0.0)
        self.assertEqual(res["t"], float("-inf"))
        self.assertEqual(res["favored"], "A")

    def test_paired_score_difference_true_tie_stays_a_tie(self):
        # both samples identical (not just equal in distribution): se == 0 AND mean_diff == 0 -- a
        # real tie, not a nonzero constant. The fix must not over-correct into always declaring
        # significance whenever se happens to be (exactly) zero.
        res = paired_score_difference(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(res["se"], 0.0)
        self.assertEqual(res["t"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["favored"], "tie")

    def test_compare_elpd_deterministic_difference_is_decisive_not_a_tie(self):
        res = compare_elpd(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(res["se"], 0.0)
        self.assertEqual(res["elpd_diff"], 3.0)
        self.assertEqual(res["z"], float("inf"))
        self.assertEqual(res["favored"], "A")

    def test_compare_elpd_deterministic_difference_flips_with_direction(self):
        res = compare_elpd(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(res["z"], float("-inf"))
        self.assertEqual(res["favored"], "B")

    def test_compare_elpd_true_tie_stays_a_tie(self):
        res = compare_elpd(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(res["se"], 0.0)
        self.assertEqual(res["z"], 0.0)
        self.assertEqual(res["favored"], "tie")

    def test_vuong_and_clarke_deterministic_difference_intentionally_still_tie(self):
        """Unlike paired_score_difference/compare_elpd above, vuong_test's omega<=1e-12*scale branch
        is Vuong's own variance pretest: omega^2 == 0 in the population changes the test's asymptotic
        distribution entirely (the normal-approximation statistic isn't just numerically unstable, it
        is the wrong reference distribution once the models are observationally indistinguishable), and
        clarke_test's tie here is an exact sign test's genuine power floor (the minimum two-sided
        binomial p-value at n=3 is 0.25, however unanimous the sign pattern is). Neither result carries
        a CI/se field that contradicts its own "tie" the way the other two did -- both are already
        self-consistent (vuong_test even says so via `indistinguishable: True`) -- so this is documented
        existing behavior, locked in here so it is not "fixed" to match the other two by mistake."""
        vuong = vuong_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(vuong["favored"], "tie")
        self.assertTrue(vuong["indistinguishable"])

        clarke = clarke_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(clarke["favored"], "tie")
        self.assertEqual(clarke["p_value"], 0.25)


class CompareElpdTest(unittest.TestCase):
    def test_favors_higher_elpd(self):
        rng = np.random.RandomState(2)
        y = rng.normal(0, 1, 2000)
        pa = stats.norm.logpdf(y, 0, 1)
        pb = stats.norm.logpdf(y, 0, 2)
        res = compare_elpd(pa, pb)
        self.assertEqual(res["favored"], "A")
        self.assertGreater(res["elpd_diff"], 0)
        self.assertGreater(res["se"], 0)

    def test_tie_within_two_se(self):
        rng = np.random.RandomState(3)
        y = rng.normal(0, 1, 2000)
        pa = stats.norm.logpdf(y, 0, 1)
        pb = stats.norm.logpdf(y, 0.005, 1.0)  # essentially identical
        self.assertEqual(compare_elpd(pa, pb)["favored"], "tie")


class ComparisonControlValidationTest(unittest.TestCase):
    """MXR-080-1605: neither ``ci_level`` nor the corrected parameter counts were validated.

    ``stats.t.ppf(0.5 + ci_level / 2, ...)`` reports no error of its own, so a negative level
    returned a negative critical value and therefore an inverted ``[+inf, -inf]`` interval; a level
    above one or NaN produced NaN endpoints. Both then fed the ``p < 1 - ci_level`` decision that
    names a favored model. On the correction path ``_complexity_correction`` subtracted ``k_a - k_b``
    unchecked: NaN made the test NaN while still naming a winner, ``inf`` drove ``p_value`` to
    exactly 0.0, and a negative count manufactured a highly significant advantage.
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        self.a = rng.normal(size=50)
        self.b = self.a + rng.normal(scale=0.1, size=50)

    def test_paired_score_difference_rejects_levels_outside_the_unit_interval(self):
        for level in (-1.0, 1.5, np.nan, 0.0, 1.0, True):
            with self.assertRaisesRegex(ValueError, "ci_level"):
                paired_score_difference(self.a, self.b, ci_level=level)

    def test_corrected_tests_reject_invalid_parameter_counts(self):
        for test in (vuong_test, clarke_test):
            for correction in ("aic", "bic"):
                for k_a in (np.nan, np.inf, -3, 1.5, True):
                    with self.assertRaisesRegex(ValueError, "k_a"):
                        test(self.a, self.b, k_a=k_a, k_b=1, correction=correction)
                with self.assertRaisesRegex(ValueError, "k_b"):
                    test(self.a, self.b, k_a=1, k_b=-1, correction=correction)

    def test_unknown_correction_is_rejected_before_the_counts_are_read(self):
        with self.assertRaisesRegex(ValueError, "correction must be"):
            vuong_test(self.a, self.b, k_a=1, k_b=1, correction="waic")

    def test_valid_controls_are_unchanged(self):
        # negative control: an ordinary level still yields a finite, correctly ordered interval, and
        # a valid correction still shifts the statistic by exactly k_a - k_b under AIC.
        out = paired_score_difference(self.a, self.b, ci_level=0.9)
        self.assertTrue(np.isfinite(out["ci_low"]) and np.isfinite(out["ci_high"]))
        self.assertLess(out["ci_low"], out["ci_high"])
        uncorrected = vuong_test(self.a, self.b)
        corrected = vuong_test(self.a, self.b, k_a=3, k_b=1, correction="aic")
        self.assertLess(corrected["statistic"], uncorrected["statistic"])
        # k_a == k_b makes the AIC correction a no-op, so the statistic must match exactly
        self.assertAlmostEqual(
            vuong_test(self.a, self.b, k_a=2, k_b=2, correction="aic")["statistic"],
            uncorrected["statistic"],
        )


if __name__ == "__main__":
    unittest.main()
