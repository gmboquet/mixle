"""Tests for the multinomial smart enumerators.

Covers MultinomialDistribution (mixle/stats/cat_multinomial.py) and
IntegerMultinomialDistribution (mixle/stats/int_multinomial.py) against brute force on a
tiny base (3 categories, trial counts <= 3): non-increasing order, exact multiset
de-duplication, log_prob == log_density, and top-k agreement with the brute-force top-k.

Both distributions include the multinomial base-measure coefficient. A real trial-count
distribution is required for exact joint enumeration over lengths.
"""

import itertools
import unittest

import numpy as np

from mixle.stats import *
from mixle.stats.compute.pdist import EnumerationError

TOL = 1e-9


def multisets(categories, max_n):
    """All multisets over categories with total count 0..max_n, as sorted (value, count) pair lists."""
    out = []
    for n in range(max_n + 1):
        for combo in itertools.combinations_with_replacement(sorted(categories), n):
            pairs = []
            for v in combo:
                if pairs and pairs[-1][0] == v:
                    pairs[-1] = (v, pairs[-1][1] + 1)
                else:
                    pairs.append((v, 1))
            out.append(pairs)
    return out


def canon(value):
    """Order-invariant canonical key for a list of (value, count) pairs."""
    return tuple(sorted(value))


def tiers(pairs):
    """Map rounded log_prob tiers to the set of canonical values in each tier (tie-safe compare)."""
    out = {}
    for v, lp in pairs:
        out.setdefault(round(lp, 8), set()).add(canon(v))
    return out


class MultinomialEnumeratorTestCase(unittest.TestCase):
    """Generic categorical support fails closed until a finite support manifest exists."""

    def setUp(self):
        self.base = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        self.len_dist = IntegerCategoricalDistribution(0, [0.15, 0.25, 0.35, 0.25])
        self.dist = MultinomialDistribution(self.base, len_dist=self.len_dist)

    def test_generic_support_is_not_misrepresented_as_exact(self):
        with self.assertRaises(EnumerationError) as cm:
            self.dist.enumerator()
        self.assertIn("support manifest", str(cm.exception))

    def test_all_length_and_normalization_modes_fail_closed(self):
        for dist in (
            MultinomialDistribution(self.base),
            MultinomialDistribution(self.base, len_dist=GeometricDistribution(0.5)),
            MultinomialDistribution(self.base, len_dist=self.len_dist, len_normalized=True),
        ):
            with self.subTest(dist=dist), self.assertRaises(EnumerationError):
                dist.enumerator()


class IntegerMultinomialEnumeratorTestCase(unittest.TestCase):
    """IntegerMultinomialDistribution enumeration (infinite support) against a bounded brute force."""

    BRUTE_MAX_N = 10
    TOP_K = 30

    def setUp(self):
        self.lengths = IntegerCategoricalDistribution(
            0,
            np.ones(self.BRUTE_MAX_N + 1) / (self.BRUTE_MAX_N + 1),
        )
        self.dist = IntegerMultinomialDistribution(
            1,
            [0.5, 0.3, 0.2],
            len_dist=self.lengths,
        )
        brute = [(v, self.dist.log_density(v)) for v in multisets((1, 2, 3), self.BRUTE_MAX_N)]
        brute.sort(key=lambda u: -u[1])
        self.brute = brute
        self.items = self.dist.enumerator().top_k(self.TOP_K)

    def test_order_non_increasing(self):
        lps = [lp for _, lp in self.items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL, "order violated at %d" % i)

    def test_log_prob_matches_log_density(self):
        for v, lp in self.items:
            self.assertAlmostEqual(lp, self.dist.log_density(v), delta=TOL, msg="lp mismatch at %r" % (v,))

    def test_values_deduped_as_multisets(self):
        keys = [canon(v) for v, _ in self.items]
        self.assertEqual(len(keys), len(set(keys)), "duplicate multisets yielded")

    def test_top_k_matches_brute_force_top_k(self):
        cutoff = self.items[-1][1]
        np.testing.assert_allclose(
            [lp for _, lp in self.items],
            [lp for _, lp in self.brute[: self.TOP_K]],
            atol=TOL,
            err_msg="score sequence mismatch",
        )
        full_tiers = tiers(self.brute)
        for tier, values in tiers(self.items).items():
            if tier > round(cutoff, 8):
                self.assertEqual(values, full_tiers[tier], "tier %r mismatch" % (tier,))
            else:
                self.assertTrue(values <= full_tiers[tier], "tier %r not a subset" % (tier,))

    def test_mass_dominance(self):
        cutoff = self.items[-1][1]
        seen = set(canon(v) for v, _ in self.items)
        for v, lp in self.brute:
            if lp > cutoff + TOL:
                self.assertIn(canon(v), seen, "missing %r" % (v,))

    def test_len_dist_is_included_in_enumeration_scores(self):
        # log_density includes len_dist's log-density of the total trial count (like the sibling
        # MultinomialDistribution), so the enumeration must too: lp == log_density still holds, but
        # scores now genuinely differ from the Null-len_dist baseline (self.items) rather than matching
        # it, since a real len_dist changes which count vectors are most probable.
        with_len = IntegerMultinomialDistribution(
            1,
            [0.5, 0.3, 0.2],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.2, 0.3, 0.4]),
        )
        items = with_len.enumerator().top_k(10)
        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL, "order violated at %d" % i)
        for v, lp in items:
            self.assertAlmostEqual(lp, with_len.log_density(v), delta=TOL)
        # a different len_dist must change the scores.
        self.assertGreater(np.max(np.abs(np.asarray(lps) - np.asarray([lp for _, lp in self.items[:10]]))), TOL)

    def test_zero_probability_category_skipped(self):
        dist = IntegerMultinomialDistribution(0, [0.6, 0.0, 0.4], len_dist=self.lengths)
        for v, lp in dist.enumerator().top_k(15):
            self.assertTrue(all(cat in (0, 2) for cat, _ in v), "zero-probability category emitted in %r" % (v,))
            self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL)

    def test_missing_length_distribution_raises(self):
        with self.assertRaises(EnumerationError) as cm:
            IntegerMultinomialDistribution(0, [1.0, 0.0]).enumerator()
        self.assertIn("trial-count distribution", str(cm.exception))

    def test_empty_probability_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            IntegerMultinomialDistribution(0, [])


if __name__ == "__main__":
    unittest.main()
