"""Multiple-testing correction and evidence combination (pysp.inference.multiple_testing)."""

import unittest

import numpy as np

from pysp.inference import (
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


if __name__ == "__main__":
    unittest.main()
