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


class AbundanceValidationTest(unittest.TestCase):
    """MXR-080-0077: ``_abund`` used to accept fractional and non-finite counts. Different estimators
    then interpreted the same invalid input two incompatible ways -- ``turing_coverage`` summed
    ``[1.5, 2.5]`` as a continuous total (4), while ``good_turing`` (via ``_freq_of_freq``) truncated
    it to abundance *classes* 1 and 2 -- and NaN was dropped indirectly by comparisons that NaN always
    fails (``NaN > 0`` is ``False``), silently shrinking the sample with no error. Every
    abundance-consuming estimator must now reject fractional, negative, and non-finite counts."""

    def test_fractional_counts_rejected_by_every_abundance_estimator(self):
        bad = np.array([1.5, 2.5])
        for fn in (turing_coverage, good_turing, chao1, ace, hill_numbers, rarefaction_curve):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(bad)

    def test_nan_is_rejected_not_silently_dropped(self):
        # NaN used to fail both the `< 0` and `> 0` comparisons in _abund and vanish from the sample
        # (a 3-item sample silently became a 2-item sample) instead of raising.
        with self.assertRaises(ValueError):
            turing_coverage(np.array([5.0, float("nan"), 3.0]))

    def test_infinite_counts_rejected(self):
        with self.assertRaises(ValueError):
            turing_coverage(np.array([5.0, float("inf")]))

    def test_negative_counts_rejected(self):
        with self.assertRaises(ValueError):
            turing_coverage(np.array([-1.0, 2.0]))

    def test_valid_integer_abundances_still_work(self):
        # negative control: a real integer abundance vector produces a sensible estimate everywhere
        # (exact-valued floats, e.g. 100.0, are legitimate integers and must still be accepted).
        c = np.array([100.0, 10.0, 1.0, 1.0, 1.0])
        self.assertEqual(turing_coverage(c)["f1"], 3.0)
        self.assertAlmostEqual(chao1(c)["estimate"], 8.0)
        ace_r = ace(c)
        self.assertGreaterEqual(ace_r["estimate"], ace_r["observed"])
        np.testing.assert_allclose(hill_numbers(np.array([7, 7, 7, 7, 7]), [0.0]), [5.0])
        self.assertEqual(rarefaction_curve(c)["expected_richness"].shape, (int(c.sum()),))


class GoodTuringSmallSupportTest(unittest.TestCase):
    """MXR-080-0078: the log-linear frequency-of-frequencies fit needs at least two distinct
    abundance classes to determine a slope. An empty sample used to raise a bare ``TypeError`` out of
    ``np.polyfit``, and an all-singleton sample -- a completely ordinary Good-Turing input, not an
    exotic one -- used to raise ``numpy.linalg.LinAlgError``. Empty samples must now report a typed
    insufficient-evidence result, and small-support samples must fall back to an unsmoothed
    (raw-frequency) estimate instead of crashing."""

    def test_empty_sample_is_insufficient_evidence_not_a_crash(self):
        r = good_turing(np.array([]))
        self.assertTrue(r["insufficient_evidence"])
        self.assertTrue(r["reason"])
        self.assertEqual(r["proba"].size, 0)
        self.assertTrue(np.isnan(r["p0"]))

    def test_all_singleton_sample_falls_back_instead_of_crashing(self):
        # [1, 1, 1]: three species each observed exactly once -- the audit's own reproduction. Only
        # one distinct frequency-of-frequency class (r=1), so the log-linear regression is
        # underdetermined; this used to raise numpy.linalg.LinAlgError out of np.polyfit.
        r = good_turing(np.array([1, 1, 1]))
        self.assertFalse(r["insufficient_evidence"])
        self.assertAlmostEqual(r["p0"] + r["proba"].sum(), 1.0, places=9)

    def test_single_frequency_class_reallocates_seen_mass_by_raw_frequency(self):
        # [2, 2, 2]: three equally-abundant doubleton species, still only one frequency class (r=2).
        # No singletons at all, so p0 (=f1/n) is exactly 0, and the fallback (r* = r) must split the
        # entire seen mass equally across the three equally-abundant species.
        r = good_turing(np.array([2, 2, 2]))
        self.assertFalse(r["insufficient_evidence"])
        self.assertAlmostEqual(r["p0"], 0.0)
        np.testing.assert_allclose(r["proba"], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    def test_two_frequency_classes_still_uses_the_smoothed_fit(self):
        # negative control: a sample with >= 2 distinct frequency classes is unaffected by the
        # small-support fallback and still goes through the ordinary log-linear smoothing path.
        r = good_turing(np.array([1, 1, 2]))
        self.assertFalse(r["insufficient_evidence"])
        self.assertEqual(r["r"].size, 2)
        self.assertAlmostEqual(r["p0"] + r["proba"].sum(), 1.0, places=9)

    def test_rich_sample_still_fits_normally(self):
        # negative control: a genuinely diverse sparse sample (many frequency classes) fits
        # Good-Turing exactly as before, unaffected by the empty/small-support special cases.
        gt = good_turing(_zipf_counts(5))
        self.assertFalse(gt["insufficient_evidence"])
        self.assertAlmostEqual(gt["p0"] + gt["proba"].sum(), 1.0, places=6)


class IncidenceValidationTest(unittest.TestCase):
    """MXR-080-0079: chao2/ice used to threshold every positive value to presence (1) and every
    negative/NaN value to absence (0) via a bare ``> 0`` comparison, so a matrix of continuous or
    malformed measurements (``0.2``, ``-1``, ``3``, ``NaN``) was silently accepted as a valid
    presence/absence design. Incidence input must now be genuinely binary and finite, with at least
    one site."""

    def test_non_binary_matrix_rejected(self):
        bad = np.array([[0.2, -1.0, 3.0, float("nan")], [1.0, 0.0, 1.0, 0.0]])
        for fn in (chao2, ice):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(bad)

    def test_matrix_with_no_sites_rejected(self):
        empty = np.empty((3, 0))
        for fn in (chao2, ice):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(empty)

    def test_valid_binary_matrix_still_works(self):
        # negative control: a real 0/1 incidence matrix still returns a richness estimate.
        inc = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]])
        self.assertAlmostEqual(chao2(inc)["estimate"], 4.0 + 0.75 * 2 * 1 / (2 * 1))
        r = ice(inc, rare_threshold=10)
        self.assertGreaterEqual(r["estimate"], r["observed"])

    def test_boolean_dtype_matrix_accepted(self):
        # negative control: a bool ndarray (True/False presence/absence) is a legitimate binary matrix.
        inc = np.array([[True, False, True], [False, True, True]])
        r = chao2(inc)
        self.assertEqual(r["observed"], 2.0)


if __name__ == "__main__":
    unittest.main()
