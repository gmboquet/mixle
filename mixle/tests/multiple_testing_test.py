"""Multiple-testing correction and evidence combination (mixle.inference.multiple_testing)."""

import unittest
from decimal import Decimal, localcontext

import numpy as np

from mixle.inference import (
    adjust_pvalues,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    fisher_combine,
    hochberg,
    holm,
    stouffer_combine,
    tippett_combine,
)

# Reference adjusted p-values for this vector are the well-known statsmodels values.
P = np.array([0.001, 0.008, 0.039, 0.041, 0.9])


class AdjustedPValueTest(unittest.TestCase):
    def test_bonferroni(self):
        np.testing.assert_allclose(bonferroni(P)["pvals_adjusted"], [0.005, 0.04, 0.195, 0.205, 1.0])

    def test_holm(self):
        np.testing.assert_allclose(holm(P)["pvals_adjusted"], [0.005, 0.032, 0.117, 0.117, 0.9])

    def test_hochberg(self):
        np.testing.assert_allclose(hochberg(P)["pvals_adjusted"], [0.005, 0.032, 0.082, 0.082, 0.9])

    def test_benjamini_hochberg(self):
        np.testing.assert_allclose(benjamini_hochberg(P)["pvals_adjusted"], [0.005, 0.02, 0.05125, 0.05125, 0.9])

    def test_benjamini_yekutieli(self):
        c = np.sum(1.0 / np.arange(1, 6))
        np.testing.assert_allclose(
            benjamini_yekutieli(P)["pvals_adjusted"],
            np.minimum(1.0, c * np.array([0.005, 0.02, 0.05125, 0.05125, 0.9])),
        )

    def test_adjusted_preserve_input_order(self):
        shuffled = P[::-1]
        adj = benjamini_hochberg(shuffled)["pvals_adjusted"]
        np.testing.assert_allclose(adj, benjamini_hochberg(P)["pvals_adjusted"][::-1])

    def test_dispatcher_matches_named(self):
        for method, fn in [
            ("bonferroni", bonferroni),
            ("holm", holm),
            ("hochberg", hochberg),
            ("bh", benjamini_hochberg),
            ("by", benjamini_yekutieli),
        ]:
            np.testing.assert_allclose(adjust_pvalues(P, method=method)["pvals_adjusted"], fn(P)["pvals_adjusted"])


class OrderingTest(unittest.TestCase):
    def test_power_ordering(self):
        # at fixed alpha: Bonferroni <= Holm <= Hochberg <= BH in number of rejections
        rng = np.random.RandomState(0)
        p = np.concatenate([rng.uniform(0, 0.01, 20), rng.uniform(0, 1, 80)])
        nb = bonferroni(p, alpha=0.05)["n_reject"]
        nh = holm(p, alpha=0.05)["n_reject"]
        nho = hochberg(p, alpha=0.05)["n_reject"]
        nbh = benjamini_hochberg(p, alpha=0.05)["n_reject"]
        self.assertLessEqual(nb, nh)
        self.assertLessEqual(nh, nho)
        self.assertLessEqual(nho, nbh)

    def test_fdr_controls_false_discoveries(self):
        # 90% nulls (uniform p), 10% strong alternatives -> BH should keep FDR near alpha
        rng = np.random.RandomState(1)
        false_disc = []
        for _ in range(200):
            nulls = rng.uniform(0, 1, 900)
            alts = rng.uniform(0, 1e-4, 100)
            p = np.concatenate([nulls, alts])
            is_null = np.concatenate([np.ones(900, bool), np.zeros(100, bool)])
            rej = benjamini_hochberg(p, alpha=0.1)["reject"]
            if rej.sum() > 0:
                false_disc.append(is_null[rej].sum() / rej.sum())
        self.assertLess(np.mean(false_disc), 0.12)

    def test_all_null_bonferroni_controls_fwer(self):
        rng = np.random.RandomState(2)
        any_reject = 0
        for _ in range(500):
            p = rng.uniform(0, 1, 50)
            if bonferroni(p, alpha=0.05)["n_reject"] > 0:
                any_reject += 1
        self.assertLess(any_reject / 500, 0.05)


class AlphaValidationTest(unittest.TestCase):
    """``adjusted <= alpha`` is a bare boolean comparison, so an out-of-range or NaN alpha never raised
    on its own: alpha=nan silently rejected nothing (a NaN comparison is always False) and alpha=1.5
    silently rejected everything -- both a plausible-looking but meaningless reject/n_reject instead of
    an error."""

    CORRECTIONS = (bonferroni, holm, hochberg, benjamini_hochberg, benjamini_yekutieli)

    def test_negative_alpha_raises(self):
        for fn in self.CORRECTIONS:
            with self.assertRaises(ValueError):
                fn(P, alpha=-0.5)

    def test_alpha_above_one_raises(self):
        for fn in self.CORRECTIONS:
            with self.assertRaises(ValueError):
                fn(P, alpha=1.5)

    def test_nan_alpha_raises(self):
        # `alpha <= x` alone would miss this silently (a NaN comparison is always False, so it looked
        # like a legitimate "reject nothing" result); the explicit `0.0 < alpha < 1.0` guard catches it.
        for fn in self.CORRECTIONS:
            with self.assertRaises(ValueError):
                fn(P, alpha=float("nan"))

    def test_alpha_boundary_zero_and_one_raise(self):
        # (0, 1) is open here (matching risk.py/scoring.py/select.py's significance-level convention,
        # not conformal.py's inclusive [0.0, 1.0] miscoverage-level convention): 0 only ever rejects an
        # exactly-zero adjusted p-value and 1 trivially rejects everything, neither a meaningful target
        # error rate.
        for fn in self.CORRECTIONS:
            with self.assertRaises(ValueError):
                fn(P, alpha=0.0)
            with self.assertRaises(ValueError):
                fn(P, alpha=1.0)

    def test_dispatcher_inherits_alpha_validation(self):
        with self.assertRaises(ValueError):
            adjust_pvalues(P, method="bh", alpha=float("nan"))

    def test_valid_alpha_unaffected(self):
        for fn in self.CORRECTIONS:
            res = fn(P, alpha=0.05)
            self.assertIn("n_reject", res)


class CombineTest(unittest.TestCase):
    def test_fisher_known_value(self):
        res = fisher_combine(np.array([0.1, 0.2, 0.05]))
        self.assertAlmostEqual(res["statistic"], -2 * np.sum(np.log([0.1, 0.2, 0.05])), places=10)
        self.assertEqual(res["df"], 6)

    def test_stouffer_weights_equal_unweighted_when_uniform(self):
        p = np.array([0.1, 0.2, 0.05])
        a = stouffer_combine(p)
        b = stouffer_combine(p, weights=np.ones(3))
        self.assertAlmostEqual(a["pvalue"], b["pvalue"])

    def test_combiners_small_when_all_significant(self):
        p = np.array([0.02, 0.03, 0.04])
        self.assertLess(fisher_combine(p)["pvalue"], 0.05)
        self.assertLess(stouffer_combine(p)["pvalue"], 0.05)

    def test_tippett(self):
        res = tippett_combine(np.array([0.01, 0.5, 0.8]))
        self.assertAlmostEqual(res["min_p"], 0.01)
        self.assertAlmostEqual(res["pvalue"], 1 - (1 - 0.01) ** 3)


class StoufferWeightValidationTest(unittest.TestCase):
    """A zero/negative/NaN Stouffer weight used to silently produce a combined z/p-value instead of
    raising: a negative weight gave a plausible-looking but meaningless result (it subtracts that
    study's evidence rather than combining it); an all-zero weight vector divided sqrt(sum(w*w)) == 0
    into a bare 0/0 = NaN with an unraised RuntimeWarning instead of a clear error."""

    P3 = np.array([0.1, 0.2, 0.05])

    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            stouffer_combine(self.P3, weights=np.array([-1.0, 1.0, 1.0]))

    def test_nan_weight_raises(self):
        with self.assertRaises(ValueError):
            stouffer_combine(self.P3, weights=np.array([float("nan"), 1.0, 1.0]))

    def test_all_zero_weights_raise(self):
        with self.assertRaises(ValueError):
            stouffer_combine(self.P3, weights=np.array([0.0, 0.0, 0.0]))

    def test_single_zero_weight_among_positive_still_valid(self):
        # A legitimate "exclude this study" case -- mirrors weighted_conformal's zero-weight
        # convention (individual weights may be zero; only an all-zero/non-positive total is
        # rejected) -- so weighting one study to 0 while the others stay positive must equal simply
        # dropping that study from the input.
        weighted = stouffer_combine(self.P3, weights=np.array([0.0, 1.0, 1.0]))
        excluded = stouffer_combine(self.P3[1:])
        self.assertAlmostEqual(weighted["z"], excluded["z"])
        self.assertAlmostEqual(weighted["pvalue"], excluded["pvalue"])

    def test_positive_weights_still_work(self):
        res = stouffer_combine(self.P3, weights=np.array([2.0, 3.0, 1.5]))
        self.assertTrue(np.isfinite(res["z"]))
        self.assertTrue(0.0 <= res["pvalue"] <= 1.0)


class TippettSubEpsilonTest(unittest.TestCase):
    """MXR-080-1604: ``1 - (1 - min_p) ** k`` rounds the evidence away before exponentiating.

    Below the float64 epsilon the subtraction returns exactly 1.0, so a finite nonzero minimum
    p-value combined to exactly 0.0 -- impossible, and unboundedly overconfident. The stable form is
    ``-expm1(k * log1p(-min_p))``. Expected values are computed here in exact decimal arithmetic, not
    copied from the implementation.
    """

    @staticmethod
    def _exact(min_p: str, k: int) -> Decimal:
        # `1 - min_p` needs enough significant digits to hold min_p's exponent, and the k-th power
        # needs k times that -- exactly the headroom float64 does not have, which is the whole point.
        with localcontext() as ctx:
            ctx.prec = 2000
            return Decimal(1) - (Decimal(1) - Decimal(min_p)) ** k

    def test_sub_epsilon_minimum_does_not_collapse_to_zero(self):
        for min_p_text in ("1e-17", "1e-20", "1e-100", "1e-300"):
            with self.subTest(min_p=min_p_text):
                min_p = float(min_p_text)
                got = tippett_combine(np.array([min_p, 0.9]))["pvalue"]
                self.assertGreater(got, 0.0, "a finite nonzero p-value cannot combine to exactly 0")
                expected = float(self._exact(min_p_text, 2))
                self.assertAlmostEqual(got / expected, 1.0, places=12)

    def test_epsilon_scale_minimum_is_not_inflated_to_two_epsilons(self):
        # 1e-16 used to come back as 2.220446049250313e-16 (two machine epsilons) instead of ~2e-16
        got = tippett_combine(np.array([1e-16, 0.9]))["pvalue"]
        expected = float(self._exact("1e-16", 2))
        self.assertAlmostEqual(got / expected, 1.0, places=12)
        self.assertNotEqual(got, 2.220446049250313e-16)

    def test_smallest_subnormal_minimum_still_survives(self):
        got = tippett_combine(np.array([5e-324, 0.9]))["pvalue"]
        self.assertGreater(got, 0.0)

    def test_scales_with_the_number_of_tests(self):
        """For a tiny min_p the combined value is ~k * min_p, so k must actually enter the result."""
        for k in (2, 5, 20):
            with self.subTest(k=k):
                pvals = np.array([1e-30] + [0.9] * (k - 1))
                got = tippett_combine(pvals)["pvalue"]
                self.assertAlmostEqual(got / (k * 1e-30), 1.0, places=12)

    def test_endpoints_and_ordinary_values_are_exact(self):
        self.assertEqual(tippett_combine(np.array([0.0, 0.5]))["pvalue"], 0.0)
        self.assertEqual(tippett_combine(np.array([1.0, 1.0]))["pvalue"], 1.0)
        # negative control: the ordinary, well-conditioned case is unchanged
        res = tippett_combine(np.array([0.01, 0.5, 0.8]))
        self.assertAlmostEqual(res["min_p"], 0.01)
        self.assertAlmostEqual(res["pvalue"], 1 - (1 - 0.01) ** 3)
        self.assertTrue(0.0 <= res["pvalue"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
