"""Rank aggregation / consensus / permutation distances (mixle.stats.rank_aggregation)."""

import unittest

import numpy as np

from mixle.analysis import (
    borda_count,
    cayley_distance,
    copeland,
    kemeny_consensus,
    kendall_distance,
    mallows_fit,
    spearman_footrule,
)


class DistanceTest(unittest.TestCase):
    def test_single_adjacent_swap(self):
        a = np.array([0, 1, 2, 3])
        b = np.array([1, 0, 2, 3])
        self.assertEqual(kendall_distance(a, b), 1)
        self.assertEqual(spearman_footrule(a, b), 2)
        self.assertEqual(cayley_distance(a, b), 1)

    def test_identity_is_zero(self):
        a = np.array([2, 0, 3, 1])
        self.assertEqual(kendall_distance(a, a), 0)
        self.assertEqual(spearman_footrule(a, a), 0)
        self.assertEqual(cayley_distance(a, a), 0)

    def test_reversal_is_max_kendall(self):
        a = np.array([0, 1, 2, 3])
        b = np.array([3, 2, 1, 0])
        self.assertEqual(kendall_distance(a, b), 6)  # all C(4,2) pairs discordant


class AggregationTest(unittest.TestCase):
    def test_unanimous_consensus(self):
        R = np.array([[0, 1, 2, 3]] * 5)
        self.assertTrue(np.array_equal(borda_count(R)["consensus"], [0, 1, 2, 3]))
        km = kemeny_consensus(R)
        self.assertTrue(np.array_equal(km["consensus"], [0, 1, 2, 3]))
        self.assertEqual(km["distance"], 0)

    def test_borda_recovers_majority_order(self):
        # most voters prefer 0>1>2; a couple disagree slightly
        rng = np.random.RandomState(0)
        base = np.array([0, 1, 2, 3, 4])
        rows = []
        for _ in range(30):
            r = base.copy()
            i = rng.randint(4)
            r[i], r[i + 1] = r[i + 1], r[i]
            rows.append(r)
        R = np.array(rows)
        self.assertTrue(np.array_equal(borda_count(R)["consensus"], base))

    def test_kemeny_matches_borda_on_easy_case(self):
        rng = np.random.RandomState(1)
        base = np.array([0, 1, 2, 3, 4])
        rows = [base.copy() for _ in range(10)]
        rows[0][1], rows[0][2] = rows[0][2], rows[0][1]
        R = np.array(rows)
        km = kemeny_consensus(R)
        self.assertTrue(km["exact"])
        self.assertTrue(np.array_equal(km["consensus"], base))

    def test_kemeny_local_search_for_large_m(self):
        rng = np.random.RandomState(2)
        base = np.arange(12)
        rows = []
        for _ in range(15):
            r = base.copy()
            i = rng.randint(11)
            r[i], r[i + 1] = r[i + 1], r[i]
            rows.append(r)
        km = kemeny_consensus(np.array(rows), exact_max_items=8)
        self.assertFalse(km["exact"])
        self.assertTrue(np.array_equal(km["consensus"], base))

    def test_copeland_runs(self):
        R = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2]])
        out = copeland(R)
        self.assertEqual(out["consensus"][0], 0)  # item 0 is the Condorcet winner


class MallowsTest(unittest.TestCase):
    def test_unanimous_gives_infinite_dispersion(self):
        R = np.array([[0, 1, 2, 3]] * 8)
        mf = mallows_fit(R)
        self.assertEqual(mf["theta"], float("inf"))
        self.assertEqual(mf["mean_distance"], 0.0)

    def test_tight_voters_higher_theta_than_loose(self):
        rng = np.random.RandomState(3)
        base = np.arange(5)
        tight = []
        for _ in range(40):
            r = base.copy()
            if rng.rand() < 0.3:
                i = rng.randint(4)
                r[i], r[i + 1] = r[i + 1], r[i]
            tight.append(r)
        loose = np.array([rng.permutation(5) for _ in range(40)])
        theta_tight = mallows_fit(np.array(tight))["theta"]
        theta_loose = mallows_fit(loose)["theta"]
        self.assertGreater(theta_tight, theta_loose)
        self.assertTrue(np.array_equal(mallows_fit(np.array(tight))["center"], base))

    def test_invalid_ranking_raises(self):
        with self.assertRaises(ValueError):
            borda_count(np.array([[0, 0, 1]]))


class PermutationValidationTest(unittest.TestCase):
    """MXR-080-0107 -- every distance/aggregation entry point must validate its ranking(s) as exact
    permutations before computing anything, instead of casting-then-checking (fractional rankings) or
    not checking at all (the public distance functions)."""

    def test_fractional_ranking_rejected_before_truncation(self):
        # the audit's own repro: casting [0.9, 1.1] to int *first* silently yields the "valid"
        # permutation [0, 1]; validation must happen before any int cast.
        with self.assertRaises(ValueError):
            borda_count(np.array([[0.9, 1.1]]))

    def test_duplicate_entries_rejected_by_distance_functions(self):
        a_dup = np.array([0, 0, 1])
        b = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            kendall_distance(a_dup, b)
        with self.assertRaises(ValueError):
            spearman_footrule(a_dup, b)
        with self.assertRaises(ValueError):
            cayley_distance(a_dup, b)

    def test_out_of_range_ids_rejected_not_crashed(self):
        # previously an uncaught IndexError from fancy-index assignment; must now be a clean ValueError.
        a_oor = np.array([0, 1, 5])
        b = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            kendall_distance(a_oor, b)
        with self.assertRaises(ValueError):
            spearman_footrule(a_oor, b)
        with self.assertRaises(ValueError):
            cayley_distance(a_oor, b)

    def test_mismatched_length_rankings_rejected_not_zero(self):
        # previously kendall_distance silently truncated to the shorter ranking's length and could
        # return 0 for two rankings of genuinely different sizes.
        a = np.array([0, 1, 2])
        b = np.array([0, 1, 2, 3, 4])
        with self.assertRaises(ValueError):
            kendall_distance(a, b)
        with self.assertRaises(ValueError):
            spearman_footrule(a, b)
        with self.assertRaises(ValueError):
            cayley_distance(a, b)

    def test_aggregation_functions_reject_invalid_permutations_too(self):
        with self.assertRaises(ValueError):
            copeland(np.array([[0, 1, 5]]))
        with self.assertRaises(ValueError):
            kemeny_consensus(np.array([[0.9, 1.1], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            mallows_fit(np.array([[0, 0, 1]]))

    def test_empty_rankings_and_item_sets_are_rejected_everywhere(self):
        for rankings in (np.array([]), np.empty((0, 3)), np.empty((2, 0))):
            for aggregate in (borda_count, copeland, kemeny_consensus, mallows_fit):
                with self.subTest(shape=rankings.shape, aggregate=aggregate.__name__):
                    with self.assertRaises(ValueError):
                        aggregate(rankings)
        for distance in (kendall_distance, spearman_footrule, cayley_distance):
            with self.subTest(distance=distance.__name__):
                with self.assertRaises(ValueError):
                    distance(np.array([]), np.array([]))

    def test_boolean_item_identities_are_rejected(self):
        ranking = np.array([False, True])
        with self.assertRaisesRegex(ValueError, "Boolean"):
            borda_count(ranking)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            kendall_distance(ranking, ranking)

    def test_kemeny_exactness_boundary_must_be_exact_nonnegative(self):
        rankings = np.array([[0, 1, 2], [1, 0, 2]])
        for bad_boundary in (-1, 1.5, True, np.nan):
            with self.subTest(exact_max_items=bad_boundary):
                with self.assertRaisesRegex(ValueError, "exact_max_items"):
                    kemeny_consensus(rankings, exact_max_items=bad_boundary)
                with self.assertRaisesRegex(ValueError, "exact_max_items"):
                    mallows_fit(rankings, exact_max_items=bad_boundary)

    def test_mallows_preserves_kemeny_search_guarantee(self):
        rankings = np.array([[0, 1, 2], [1, 0, 2]])
        exact = mallows_fit(rankings, exact_max_items=3)
        self.assertTrue(exact["exact"])
        self.assertEqual(exact["search_mode"], "exact_enumeration")
        self.assertEqual(exact["exact_max_items"], 3)
        heuristic = mallows_fit(rankings, exact_max_items=0)
        self.assertFalse(heuristic["exact"])
        self.assertEqual(heuristic["search_mode"], "adjacent_swap_local_search")
        self.assertEqual(heuristic["exact_max_items"], 0)

    def test_negative_control_identical_permutations_give_zero(self):
        a = np.array([2, 0, 3, 1])
        self.assertEqual(kendall_distance(a, a), 0)
        self.assertEqual(spearman_footrule(a, a), 0)
        self.assertEqual(cayley_distance(a, a), 0)

    def test_negative_control_known_swap_gives_known_distance(self):
        a = np.array([0, 1, 2, 3])
        b = np.array([1, 0, 2, 3])  # a single adjacent transposition of items 0 and 1
        self.assertEqual(kendall_distance(a, b), 1)
        self.assertEqual(spearman_footrule(a, b), 2)
        self.assertEqual(cayley_distance(a, b), 1)

    def test_negative_control_legitimate_matching_size_rankings_still_aggregate(self):
        R = np.array([[0, 1, 2, 3]] * 4)
        out = borda_count(R)
        self.assertTrue(np.array_equal(out["consensus"], [0, 1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
