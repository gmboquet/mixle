"""Tests for the multinomial smart enumerators.

Covers MultinomialDistribution (pysp/stats/cat_multinomial.py) and
IntegerMultinomialDistribution (pysp/stats/int_multinomial.py) against brute force on a
tiny base (3 categories, trial counts <= 3): non-increasing order, exact multiset
de-duplication, log_prob == log_density, and top-k agreement with the brute-force top-k.

Note the density forms actually implemented: neither distribution includes the
multinomial coefficient in log_density, and IntegerMultinomialDistribution.log_density
also omits the len_dist contribution (its support is therefore infinite and ordered by
sum_k n_k * log p_k alone).
"""

import itertools
import unittest

import numpy as np

from pysp.stats import *
from pysp.stats.pdist import EnumerationError

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
    """MultinomialDistribution (categorical base) enumeration against brute force."""

    def setUp(self):
        self.base = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        self.len_dist = IntegerCategoricalDistribution(0, [0.15, 0.25, 0.35, 0.25])
        self.dist = MultinomialDistribution(self.base, len_dist=self.len_dist)
        with np.errstate(divide="ignore"):
            brute = [(v, self.dist.log_density(v)) for v in multisets("abc", 3)]
        brute = [(v, lp) for v, lp in brute if lp > -np.inf]
        brute.sort(key=lambda u: -u[1])
        self.brute = brute

    def test_order_non_increasing(self):
        items = list(self.dist.enumerator())
        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL, "order violated at %d" % i)

    def test_log_prob_matches_log_density(self):
        for v, lp in self.dist.enumerator():
            self.assertAlmostEqual(lp, self.dist.log_density(v), delta=TOL, msg="lp mismatch at %r" % (v,))

    def test_values_deduped_as_multisets(self):
        items = list(self.dist.enumerator())
        keys = [canon(v) for v, _ in items]
        self.assertEqual(len(keys), len(set(keys)), "duplicate multisets yielded")

    def test_complete_and_matches_brute_force(self):
        # 1 + 3 + 6 + 10 multisets of sizes 0..3 over three categories.
        items = list(self.dist.enumerator())
        self.assertEqual(len(items), 20)
        self.assertEqual(len(self.brute), 20)
        np.testing.assert_allclose(
            [lp for _, lp in items], [lp for _, lp in self.brute], atol=TOL, err_msg="score sequence mismatch"
        )
        self.assertEqual(tiers(items), tiers(self.brute), "tier values mismatch")

    def test_top_k_matches_brute_force_top_k(self):
        for k in (1, 3, 7, 12):
            top = self.dist.enumerator().top_k(k)
            self.assertEqual(len(top), k)
            np.testing.assert_allclose([lp for _, lp in top], [lp for _, lp in self.brute[:k]], atol=TOL)
            # Tie-safe value comparison: drop the (possibly split) trailing tier of each side.
            full_tiers = tiers(self.brute)
            for tier, values in tiers(top).items():
                if tier > round(top[-1][1], 8):
                    self.assertEqual(values, full_tiers[tier], "tier %r mismatch at k=%d" % (tier, k))
                else:
                    self.assertTrue(values <= full_tiers[tier], "tier %r not a subset at k=%d" % (tier, k))

    def test_infinite_length_distribution(self):
        dist = MultinomialDistribution(self.base, len_dist=GeometricDistribution(0.5))
        items = dist.enumerator().top_k(25)
        self.assertEqual(len(items), 25)
        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL)
        for v, lp in items:
            self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL)
        keys = [canon(v) for v, _ in items]
        self.assertEqual(len(keys), len(set(keys)))
        # Mass dominance: anything strictly more probable than the cutoff must be present.
        # The empty multiset is skipped: the geometric support starts at 1, but its
        # log_density applies the pmf formula outside the support too (density 1 at n=0).
        cutoff = lps[-1]
        seen = set(keys)
        for v in multisets("abc", 8):
            if len(v) > 0 and dist.log_density(v) > cutoff + TOL:
                self.assertIn(canon(v), seen, "missing %r" % (v,))

    def test_null_len_dist_raises(self):
        with self.assertRaises(EnumerationError):
            MultinomialDistribution(self.base).enumerator()

    def test_len_normalized_raises(self):
        with self.assertRaises(EnumerationError):
            MultinomialDistribution(self.base, len_dist=self.len_dist, len_normalized=True).enumerator()

    def test_non_enumerable_base_names_child(self):
        with self.assertRaises(EnumerationError) as cm:
            MultinomialDistribution(GaussianDistribution(0.0, 1.0), len_dist=self.len_dist).enumerator()
        self.assertIn("MultinomialDistribution.dist", str(cm.exception))
        with self.assertRaises(EnumerationError) as cm:
            MultinomialDistribution(self.base, len_dist=GaussianDistribution(0.0, 1.0)).enumerator()
        self.assertIn("MultinomialDistribution.len_dist", str(cm.exception))


class IntegerMultinomialEnumeratorTestCase(unittest.TestCase):
    """IntegerMultinomialDistribution enumeration (infinite support) against a bounded brute force."""

    BRUTE_MAX_N = 10
    TOP_K = 30

    def setUp(self):
        self.dist = IntegerMultinomialDistribution(1, [0.5, 0.3, 0.2])
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
        # Any count vector outside the brute-force bound scores below (BRUTE_MAX_N+1)*log(p_max),
        # so once the cutoff sits above that the brute force covers everything enumerated.
        cutoff = self.items[-1][1]
        self.assertGreater(
            cutoff, (self.BRUTE_MAX_N + 1) * np.log(0.5) + TOL, "brute-force bound too small to certify the top-k"
        )
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

    def test_len_dist_does_not_affect_log_density_scores(self):
        # log_density ignores len_dist, so the enumeration must too (contract: lp == log_density).
        with_len = IntegerMultinomialDistribution(
            1, [0.5, 0.3, 0.2], len_dist=IntegerCategoricalDistribution(0, [0.25, 0.25, 0.25, 0.25])
        )
        items = with_len.enumerator().top_k(10)
        np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in self.items[:10]], atol=TOL)
        for v, lp in items:
            self.assertAlmostEqual(lp, with_len.log_density(v), delta=TOL)

    def test_zero_probability_category_skipped(self):
        dist = IntegerMultinomialDistribution(0, [0.6, 0.0, 0.4])
        for v, lp in dist.enumerator().top_k(15):
            self.assertTrue(all(cat in (0, 2) for cat, _ in v), "zero-probability category emitted in %r" % (v,))
            self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL)

    def test_degenerate_category_raises(self):
        with self.assertRaises(EnumerationError) as cm:
            IntegerMultinomialDistribution(0, [1.0, 0.0]).enumerator()
        self.assertIn("probability one", str(cm.exception))

    def test_no_positive_categories_yields_only_empty(self):
        items = list(IntegerMultinomialDistribution(0, []).enumerator())
        self.assertEqual(items, [([], 0.0)])


if __name__ == "__main__":
    unittest.main()
