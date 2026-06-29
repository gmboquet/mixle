"""Sampling completeness / richness / diversity estimators (mixle.stats.coverage)."""

import unittest

import numpy as np

from mixle.analysis import (
    ace,
    chao1,
    chao2,
    good_turing,
    hill_numbers,
    ice,
    rarefaction_curve,
    turing_coverage,
)


class TuringTest(unittest.TestCase):
    def test_coverage_and_unseen_mass(self):
        c = np.array([100, 10, 1, 1, 1])
        r = turing_coverage(c)
        self.assertAlmostEqual(r["unseen_mass"], 3.0 / 113.0)
        self.assertAlmostEqual(r["coverage"], 1.0 - 3.0 / 113.0)

    def test_no_singletons_full_coverage(self):
        r = turing_coverage(np.array([5, 5, 4, 3]))
        self.assertEqual(r["unseen_mass"], 0.0)
        self.assertEqual(r["coverage"], 1.0)


def _zipf_counts(seed, n_species=8000, n_draws=5000, s=1.05):
    # sample individuals from a bounded Zipf pmf over a fixed species set (avoids the unbounded
    # np.bincount(np.random.zipf(...)) blow-up). The default (many species, few draws) is the sparse,
    # many-singletons regime where Good-Turing is meant to operate.
    rng = np.random.RandomState(seed)
    p = 1.0 / np.arange(1, n_species + 1) ** s
    p /= p.sum()
    draws = rng.choice(n_species, size=n_draws, p=p)
    counts = np.bincount(draws, minlength=n_species)
    return counts[counts > 0]


class GoodTuringTest(unittest.TestCase):
    def test_probabilities_and_p0_sum_to_one(self):
        gt = good_turing(_zipf_counts(0))
        self.assertAlmostEqual(gt["p0"] + gt["proba"].sum(), 1.0, places=6)
        self.assertGreater(gt["p0"], 0.0)

    def test_singletons_discounted_in_sparse_regime(self):
        # with many singletons, Good-Turing reallocates mass to the unseen: r*_1 < 1 and the total
        # probability on singletons drops below their naive MLE f1/n
        counts = _zipf_counts(2)
        n = float(counts.sum())
        gt = good_turing(counts)
        self.assertLess(gt["r_star"][0], 1.0)
        singleton_total = gt["proba"][counts == 1].sum()
        self.assertLess(singleton_total, float((counts == 1).sum()) / n)


class Chao1Test(unittest.TestCase):
    def test_bias_corrected_formula(self):
        # f1=3, f2=0 -> f0 = 3*2/(2*1) = 3 ; S_obs=5 -> 8
        c = np.array([100, 10, 1, 1, 1])
        r = chao1(c)
        self.assertEqual(r["observed"], 5.0)
        self.assertAlmostEqual(r["estimate"], 8.0)
        self.assertLessEqual(r["ci_low"], r["estimate"])
        self.assertLessEqual(r["estimate"], r["ci_high"])

    def test_complete_sample_estimate_equals_observed(self):
        # no singletons or doubletons -> no inferred unseen species
        r = chao1(np.array([20, 15, 10, 8]))
        self.assertEqual(r["estimate"], r["observed"])

    def test_estimate_at_least_observed(self):
        rng = np.random.RandomState(2)
        c = np.bincount(rng.poisson(3, 200) + 1)
        c = c[c > 0]
        r = chao1(c)
        self.assertGreaterEqual(r["estimate"], r["observed"])


class Chao2Test(unittest.TestCase):
    def test_incidence_formula(self):
        # species site-counts: [3,1,1,4] -> q1=2,q2=0, m=4, corr=3/4
        inc = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]])
        r = chao2(inc)
        self.assertEqual(r["observed"], 4.0)
        self.assertAlmostEqual(r["estimate"], 4.0 + 0.75 * 2 * 1 / (2 * 1))


class HillTest(unittest.TestCase):
    def test_equal_abundance_gives_richness(self):
        c = np.array([7, 7, 7, 7, 7])
        np.testing.assert_allclose(hill_numbers(c, [0.0, 1.0, 2.0]), [5.0, 5.0, 5.0])

    def test_monotone_nonincreasing_in_q(self):
        c = np.array([100, 30, 10, 5, 1])
        d = hill_numbers(c, [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(np.all(np.diff(d) <= 1e-9))

    def test_q1_is_exp_shannon(self):
        c = np.array([5, 3, 2])
        p = c / c.sum()
        shannon = -np.sum(p * np.log(p))
        self.assertAlmostEqual(hill_numbers(c, 1.0)[0], np.exp(shannon))

    def test_q2_is_inverse_simpson(self):
        c = np.array([5, 3, 2])
        p = c / c.sum()
        self.assertAlmostEqual(hill_numbers(c, 2.0)[0], 1.0 / np.sum(p**2))


class RarefactionTest(unittest.TestCase):
    def test_endpoints(self):
        c = np.array([10, 5, 3, 1, 1])
        rc = rarefaction_curve(c)
        self.assertAlmostEqual(rc["expected_richness"][0], 1.0)  # one individual -> one species
        self.assertAlmostEqual(rc["expected_richness"][-1], float(c.size))  # full sample -> all species

    def test_monotone_increasing(self):
        rng = np.random.RandomState(3)
        c = np.bincount(rng.poisson(2, 300) + 1)
        c = c[c > 0]
        rc = rarefaction_curve(c)
        self.assertTrue(np.all(np.diff(rc["expected_richness"]) >= -1e-9))


class ACEICETest(unittest.TestCase):
    def test_ace_matches_hand_computation(self):
        c = np.array([100, 10, 1, 1, 1])
        r = ace(c, rare_threshold=10)
        self.assertAlmostEqual(r["estimate"], 14.0, places=6)
        self.assertGreaterEqual(r["estimate"], r["observed"])

    def test_ice_runs_and_at_least_observed(self):
        rng = np.random.RandomState(4)
        inc = (rng.rand(40, 12) < 0.2).astype(int)
        r = ice(inc, rare_threshold=10)
        self.assertGreaterEqual(r["estimate"], r["observed"])


if __name__ == "__main__":
    unittest.main()
