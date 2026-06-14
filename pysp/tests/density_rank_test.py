"""Tests for pysp.utils.density_rank: rank + cumulative probability of an observation.

Covers the exact head (cross-checked against brute force over a finite support), the sampling
fallback (forced via a tiny max_exact, checked within a few standard errors of the exact value),
and the HMM case where the head is exact and deep tails fall back to sampling.
"""

import math
import unittest

import numpy as np

from pysp.stats.hidden_markov import HiddenMarkovModelDistribution
from pysp.stats.int_range import IntegerCategoricalDistribution
from pysp.stats.mixture import MixtureDistribution
from pysp.utils.density_rank import density_rank


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


if __name__ == "__main__":
    unittest.main()
