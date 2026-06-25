"""Rank aggregation / consensus / permutation distances (pysp.stats.rank_aggregation)."""

import unittest

import numpy as np

from pysp.stats import (
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


if __name__ == "__main__":
    unittest.main()
