"""Tests for partial / top-m ranking scoring in the Plackett-Luce distribution (WS-M)."""

import itertools
import math
import unittest

import numpy as np

from mixle.stats.rankings.plackett_luce import PlackettLuceDistribution


def _dist():
    return PlackettLuceDistribution(np.log(np.array([0.4, 0.3, 0.2, 0.1])))  # K=4


class PlackettLucePartialTest(unittest.TestCase):
    def test_partial_equals_bruteforce_marginal(self):
        # The probability of a top-m ordering equals the total PL probability of all full orderings
        # that begin with that prefix.
        d = _dist()
        for prefix in [(2, 0), (1, 3), (0,), (3, 1, 2)]:
            with self.subTest(prefix=prefix):
                partial = math.exp(d.log_density(list(prefix)))
                brute = sum(
                    math.exp(d.log_density(list(perm)))
                    for perm in itertools.permutations(range(d.dim))
                    if perm[: len(prefix)] == prefix
                )
                self.assertAlmostEqual(partial, brute, places=12)

    def test_full_ranking_unchanged(self):
        # A full permutation reduces to the standard sequential-softmax product.
        d = _dist()
        order = [2, 0, 3, 1]
        w = np.exp(d.log_w)
        expected = 0.0
        remaining = list(order)
        for item in order:
            expected += math.log(w[item] / sum(w[r] for r in remaining))
            remaining.remove(item)
        self.assertAlmostEqual(d.log_density(order), expected, places=12)

    def test_topm_orderings_normalize(self):
        # The top-m orderings partition the sample space, so their probabilities sum to 1.
        d = _dist()
        for m in (1, 2, 3, 4):
            total = sum(math.exp(d.log_density(list(p))) for p in itertools.permutations(range(d.dim), m))
            self.assertAlmostEqual(total, 1.0, places=12)

    def test_invalid_orderings_raise(self):
        d = _dist()
        with self.assertRaises(ValueError):
            d.log_density([0, 0])  # repeated item
        with self.assertRaises(ValueError):
            d.log_density([0, 9])  # out of range


if __name__ == "__main__":
    unittest.main()
