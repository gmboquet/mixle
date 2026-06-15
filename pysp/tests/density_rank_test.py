"""Tests for pysp.utils.density_rank: rank + cumulative probability of an observation.

Covers the exact head (cross-checked against brute force over a finite support), the sampling
fallback (forced via a tiny max_exact, checked within a few standard errors of the exact value),
and the HMM case where the head is exact and deep tails fall back to sampling.
"""

import itertools
import math
import unittest

import numpy as np

from pysp.stats.combinator.composite import CompositeDistribution
from pysp.stats.combinator.sequence import SequenceDistribution
from pysp.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from pysp.stats.latent.mixture import MixtureDistribution
from pysp.stats.leaf.int_range import IntegerCategoricalDistribution
from pysp.stats.leaf.poisson import PoissonDistribution
from pysp.utils.density_rank import count_dp_rank, density_rank


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
            exact_mass, _ = self._brute(x)
            r = density_rank(self.mix, x, max_exact=2, n_samples=40000, seed=1)
            self.assertFalse(r.exact)
            self.assertEqual(r.method, "sampling")
            self.assertIsNone(r.rank)
            self.assertLess(abs(r.cumulative_probability - exact_mass), 5.0 * r.stderr + 0.01)

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


class MixtureCrossRankTestCase(unittest.TestCase):
    """mixture_cross_rank gives the TRUE marginal rank (not the tropical/dominant-component rank)."""

    def test_composite_mixture_true_rank_beats_tropical(self):
        from pysp.utils.density_rank import count_dp_rank, mixture_cross_rank

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
        from pysp.utils.density_rank import mixture_cross_rank

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
        from pysp.utils.density_rank import cumulative_probability

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
        from pysp.utils.density_rank import cumulative_probability

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


if __name__ == "__main__":
    unittest.main()
