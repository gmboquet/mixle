"""Tests for mixle.enumeration.density_rank: rank + cumulative probability of an observation.

Covers the exact head (cross-checked against brute force over a finite support), the sampling
fallback (forced via a tiny max_exact, checked within a few standard errors of the exact value),
and the HMM case where the head is exact and deep tails fall back to sampling.
"""

import itertools
import math
import unittest

import numpy as np

from mixle.enumeration.density_rank import count_dp_rank, count_dp_seek, density_rank
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.sequence import SequenceDistribution
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution


class DensityRankTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.support = 15
        comps = [IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(self.support)))) for _ in range(4)]
        self.mix = MixtureDistribution(comps, list(rng.dirichlet(np.ones(4))))
        self.lp = {y: self.mix.log_density(y) for y in range(self.support)}

    def _brute(self, x):
        t = self.lp[x]
        mass = sum(math.exp(self.lp[y]) for y in range(self.support) if self.lp[y] >= t - 1e-12)
        rank = sum(1 for y in range(self.support) if self.lp[y] > t + 1e-12)
        return mass, rank

    def test_exact_head_matches_brute_force(self):
        for x in range(self.support):
            r = density_rank(self.mix, x)
            mass, rank = self._brute(x)
            self.assertTrue(r.exact)
            self.assertIn(r.method, ("exact-head", "exact-exhausted"))  # least-probable value exhausts
            self.assertEqual(r.rank, rank, "rank mismatch at %d" % x)
            self.assertAlmostEqual(r.cumulative_probability, mass, places=9, msg="G mismatch at %d" % x)

    def test_sampling_fallback_matches_exact_within_error(self):
        # Force the sampling path with a tiny exact budget; estimate must land near the exact G.
        for x in (3, 7, 11):
            exact_mass, exact_rank = self._brute(x)
            r = density_rank(self.mix, x, max_exact=2, n_samples=40000, seed=1)
            self.assertFalse(r.exact)
            self.assertEqual(r.method, "sampling")
            self.assertLess(abs(r.cumulative_probability - exact_mass), 5.0 * r.stderr + 0.01)
            # rank is now the unbiased Monte-Carlo count estimate (not None): near the exact rank.
            self.assertIsNotNone(r.rank)
            self.assertLessEqual(abs(r.rank - exact_rank), 5.0 * r.rank_stderr + 1.0)

    def test_cumulative_probability_monotone_in_rank(self):
        # More probable observations have smaller cumulative probability G (they sit earlier).
        results = sorted((self.lp[x], density_rank(self.mix, x).cumulative_probability) for x in range(self.support))
        gs = [g for _, g in results]  # sorted by ascending log_prob (least probable first)
        self.assertTrue(all(gs[i] >= gs[i + 1] - 1e-9 for i in range(len(gs) - 1)))

    def test_hmm_head_exact_and_tail_sampled(self):
        rng = np.random.RandomState(1)
        topics = [IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(4)))) for _ in range(2)]
        hmm = HiddenMarkovModelDistribution(
            topics, [0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], len_dist=IntegerCategoricalDistribution(1, [0.5, 0.3, 0.2])
        )
        most_probable = next(iter(hmm.enumerator()))[0]
        head = density_rank(hmm, most_probable)
        self.assertTrue(head.exact)
        self.assertEqual(head.rank, 0)  # the mode has nothing strictly more probable
        self.assertGreater(head.cumulative_probability, 0.0)

        deep = density_rank(hmm, [3, 3, 3], max_exact=20, n_samples=20000, seed=2)
        self.assertEqual(deep.method, "sampling")
        self.assertGreaterEqual(deep.cumulative_probability, 0.0)
        self.assertLessEqual(deep.cumulative_probability, 1.0)


class CountDPRankTestCase(unittest.TestCase):
    def test_exact_rank_matches_brute_force_for_composite(self):
        rng = np.random.RandomState(0)
        sizes = (5, 4, 6)
        dist = CompositeDistribution(
            tuple(IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(s)))) for s in sizes)
        )
        support = [list(t) for t in itertools.product(*[range(s) for s in sizes])]
        lp = {tuple(x): dist.log_density(x) for x in support}

        def brute(x):
            t = lp[tuple(x)]
            return sum(1 for y in support if lp[tuple(y)] > t + 1e-12)

        for x in support:
            r = count_dp_rank(dist, x, oversample=32)
            self.assertLessEqual(r.window_lower, brute(x))
            self.assertLessEqual(brute(x), r.window_upper)
            self.assertEqual(r.rank, brute(x), "rank mismatch at %s" % x)

    def test_deep_rank_without_enumeration(self):
        # A long iid sequence: support ~ sum_L 4^L is far too large to enumerate, but the count DP
        # returns a finite rank for a deep low-probability observation, bracketed by its window.
        seq = SequenceDistribution(
            IntegerCategoricalDistribution(0, [0.4, 0.3, 0.2, 0.1]), len_dist=PoissonDistribution(8.0)
        )
        r = count_dp_rank(seq, [3] * 12, oversample=16)
        self.assertGreater(r.rank, 10**6)  # deep
        self.assertLessEqual(r.window_lower, r.rank)
        self.assertLessEqual(r.rank, r.window_upper)

    def test_gap_band_tie_not_collapsed(self):
        # Two genuinely-distinct outcomes whose log-densities differ by a gap in the band (1e-12, 1e-9)
        # must NOT be collapsed as a tie: the exact-resolution tie threshold is 1e-12, the convention the
        # true rank is defined by. With a looser 1e-9 the gap-band pair would merge and the rank be wrong.
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        gap = 2e-10  # strictly inside (1e-12, 1e-9)
        p0 = 0.2
        p1 = p0 * math.exp(gap)
        rest = (1 - p0 - p1) / 2
        # descending probability: c, d (~0.3) > b (0.2*e^gap) > a (0.2)
        cat = CategoricalDistribution({"a": p0, "b": p1, "c": rest, "d": rest})
        self.assertEqual(count_dp_rank(cat, "a", oversample=64).rank, 3)  # c, d, b all strictly above
        self.assertEqual(count_dp_rank(cat, "b", oversample=64).rank, 2)  # c, d strictly above
        # the seek inverse resolves the gap-band pair exactly too
        rb = count_dp_seek(cat, 2, oversample=64)
        ra = count_dp_seek(cat, 3, oversample=64)
        self.assertEqual((rb.value, rb.exact, rb.rank_lower, rb.rank_upper), ("b", True, 2, 2))
        self.assertEqual((ra.value, ra.exact, ra.rank_lower, ra.rank_upper), ("a", True, 3, 3))


class CountDPSeekTestCase(unittest.TestCase):
    """count_dp_seek: arbitrary-index unranking (inverse of count_dp_rank) with a true-rank bracket."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.sizes = (5, 4, 6)
        self.dist = CompositeDistribution(
            tuple(IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(s)))) for s in self.sizes)
        )
        support = [list(t) for t in itertools.product(*[range(s) for s in self.sizes])]
        self.order = sorted(support, key=lambda x: -self.dist.log_density(x))

    def test_seek_returns_ith_value_and_bracket_near_index(self):
        slack = 3  # quantization can reorder near-tied values by a few ranks
        for i in (0, 1, 20, 50, len(self.order) - 1):
            r = count_dp_seek(self.dist, i, oversample=64)
            # the value sits at the i-th descending-probability position (log-prob matches the true
            # i-th up to the small near-tie quantization slack)
            self.assertAlmostEqual(r.log_prob, self.dist.log_density(self.order[i]), delta=0.05)
            # the reported true-rank bracket agrees with the requested index within that slack
            self.assertLessEqual(r.rank_lower - slack, i)
            self.assertLessEqual(i, r.rank_upper + slack)

    def test_round_trip_rank_of_seek(self):
        # rank(seek(i)) returns to i (within small quantization slack from near-ties).
        for i in range(0, 120, 11):
            v = count_dp_seek(self.dist, i, oversample=64).value
            self.assertLessEqual(abs(count_dp_rank(self.dist, v, oversample=64).rank - i), 2)

    def test_deep_seek_without_enumerating_prefix(self):
        # Seek a deep index in an infinite-support sequence: no prefix enumeration, valid value + bracket.
        seq = SequenceDistribution(
            IntegerCategoricalDistribution(0, [0.4, 0.3, 0.2, 0.1]), len_dist=PoissonDistribution(8.0)
        )
        r = count_dp_seek(seq, 10_000_000, oversample=16)
        self.assertEqual(len(r.value), 12)  # a deep, long sequence
        self.assertLessEqual(r.rank_lower, r.rank_upper)

    def test_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            count_dp_seek(self.dist, 10_000)  # finite support has 120 values
        with self.assertRaises(IndexError):
            count_dp_seek(self.dist, -1)


class MixtureCrossRankTestCase(unittest.TestCase):
    """mixture_cross_rank gives the TRUE marginal rank (not the tropical/dominant-component rank)."""

    def test_composite_mixture_true_rank_beats_tropical(self):
        from mixle.enumeration.density_rank import count_dp_rank, mixture_cross_rank

        rng = np.random.RandomState(0)
        fields = (4, 3, 5)

        def comp(seed):
            r = np.random.RandomState(seed)
            return CompositeDistribution(
                tuple(IntegerCategoricalDistribution(0, list(r.dirichlet(np.ones(s)))) for s in fields)
            )

        mix = MixtureDistribution([comp(1), comp(2), comp(3)], list(rng.dirichlet(np.ones(3))))
        support = [list(t) for t in itertools.product(*[range(s) for s in fields])]
        lp = {tuple(x): mix.log_density(x) for x in support}

        def brute(x):
            return sum(1 for y in support if lp[tuple(y)] > lp[tuple(x)] + 1e-9)

        cross_err = max(abs(mixture_cross_rank(mix, x, oversample=128) - brute(x)) for x in support)
        self.assertLessEqual(cross_err, 6)  # true-marginal rank, only small quantization error
        # On the worst value the tropical count_dp_rank bracket is wide, while cross-rank stays close.
        worst = max(support, key=lambda x: count_dp_rank(mix, x, oversample=64).window_upper)
        cr = count_dp_rank(mix, worst, oversample=64)
        self.assertGreater(cr.window_upper - cr.window_lower, 6)  # tropical can't pin the rank
        self.assertLessEqual(abs(mixture_cross_rank(mix, worst, oversample=128) - brute(worst)), 6)

    def test_leaf_mixture_true_rank(self):
        from mixle.enumeration.density_rank import mixture_cross_rank

        rng = np.random.RandomState(1)
        mix = MixtureDistribution(
            [IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(12)))) for _ in range(2)], [0.6, 0.4]
        )
        lp = {y: mix.log_density(y) for y in range(12)}

        def brute(y):
            return sum(1 for z in range(12) if lp[z] > lp[y] + 1e-9)

        errs = [abs(mixture_cross_rank(mix, y, oversample=128) - brute(y)) for y in range(12)]
        self.assertLessEqual(max(errs), 2)


class CumulativeProbabilityTestCase(unittest.TestCase):
    """The unbiased mass-histogram DP gives EXACT cumulative probability for decomposable families."""

    def test_composite_exact(self):
        from mixle.enumeration.density_rank import cumulative_probability

        rng = np.random.RandomState(0)
        comp = CompositeDistribution(
            tuple(IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(2)))) for _ in range(8))
        )
        support = list(itertools.product((0, 1), repeat=8))
        lp = {x: comp.log_density(list(x)) for x in support}

        def brute(x):
            t = lp[x]
            return sum(math.exp(lp[y]) for y in support if lp[y] >= t - 1e-12)

        for x in support:
            self.assertAlmostEqual(cumulative_probability(comp, list(x), oversample=16), brute(x), places=10)

    def test_sequence_exact(self):
        from mixle.enumeration.density_rank import cumulative_probability

        rng = np.random.RandomState(0)
        seq = SequenceDistribution(
            IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(3)))),
            len_dist=IntegerCategoricalDistribution(1, list(rng.dirichlet(np.ones(3)))),  # lengths 1..3
        )
        support = [list(t) for length in (1, 2, 3) for t in itertools.product(range(3), repeat=length)]
        lp = {tuple(x): seq.log_density(x) for x in support}

        def brute(x):
            t = lp[tuple(x)]
            return sum(math.exp(lp[tuple(y)]) for y in support if lp[tuple(y)] >= t - 1e-12)

        for x in support:
            self.assertAlmostEqual(cumulative_probability(seq, x, oversample=16), brute(x), places=10)


class CountDPTopPTestCase(unittest.TestCase):
    """count_dp_top_p reports the nucleus SIZE (without enumerating it) as a provable bracket."""

    def _true_nucleus_size(self, dist, p):
        return len(dist.enumerator().top_p(p, max_items=200000))

    def test_bracket_contains_true_nucleus_size(self):
        from mixle.enumeration.density_rank import count_dp_top_p
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        intc = IntegerCategoricalDistribution(0, [0.6, 0.3, 0.1])
        comp = CompositeDistribution((cat, intc, cat))
        for p in (0.3, 0.5, 0.8, 0.95, 1.0):
            r = count_dp_top_p(comp, p, oversample=64)
            true = self._true_nucleus_size(comp, p)
            self.assertLessEqual(r.size_lower, true, "p=%s lower" % p)
            self.assertLessEqual(true, r.size_upper, "p=%s upper" % p)
            if not r.truncated:
                self.assertGreaterEqual(r.covered_mass, p - 1e-9, "p=%s mass" % p)

    def test_leaf_bracket_is_exact(self):
        # A leaf has no convolution smear, so the bracket collapses to the exact nucleus size.
        from mixle.enumeration.density_rank import count_dp_top_p
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        leaf = CategoricalDistribution({i: w for i, w in enumerate([0.3, 0.25, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02])})
        for p in (0.3, 0.55, 0.8, 0.95, 1.0):
            r = count_dp_top_p(leaf, p, oversample=64)
            true = self._true_nucleus_size(leaf, p)
            self.assertEqual((r.size_lower, r.size_upper), (true, true), "p=%s" % p)

    def test_huge_support_without_enumeration(self):
        # A support too large to enumerate (6**12) still returns a bracket quickly.
        from mixle.enumeration.density_rank import count_dp_top_p
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        rng = np.random.RandomState(0)
        big = CompositeDistribution(
            tuple(
                CategoricalDistribution({i: pp for i, pp in enumerate(rng.dirichlet(np.ones(6) * 0.5))})
                for _ in range(12)
            )
        )
        r = count_dp_top_p(big, 0.9, oversample=32)
        self.assertLessEqual(r.size_lower, r.size_upper)
        self.assertGreater(r.size_lower, 0)
        self.assertGreaterEqual(r.covered_mass, 0.9 - 1e-9)

    def test_edge_cases(self):
        from mixle.enumeration.density_rank import count_dp_top_p
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        leaf = CategoricalDistribution({"a": 0.6, "b": 0.4})
        self.assertEqual((count_dp_top_p(leaf, 0.0).size_lower, count_dp_top_p(leaf, 0.0).size_upper), (0, 0))
        with self.assertRaises(ValueError):
            count_dp_top_p(leaf, 1.5)


if __name__ == "__main__":
    unittest.main()
