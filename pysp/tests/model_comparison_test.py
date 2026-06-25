"""Model comparison utilities (pysp.inference.model_comparison)."""

import unittest

import numpy as np
from scipy import stats

from pysp.inference import (
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


if __name__ == "__main__":
    unittest.main()
