"""Lazy best-first enumerators for the ranking families (Mallows, Plackett-Luce, Spearman).

Each replaces an O(n!) materialize-and-sort with a lazy stream: Mallows via a Lehmer-code product, Plackett-Luce
via A* over prefixes, Spearman via Murty k-best assignment. Verified exact against brute force and lazy on
supports too large to enumerate.
"""

import itertools
import unittest

import numpy as np

from pysp.stats.graph.mallows import MallowsDistribution
from pysp.stats.graph.plackett_luce import PlackettLuceDistribution
from pysp.stats.graph.spearman_rho import SpearmanRankingDistribution


def _brute(dist, n):
    return sorted(
        ((float(dist.log_density(list(p))), tuple(p)) for p in itertools.permutations(range(n))),
        key=lambda u: -u[0],
    )


def _assert_exact(testcase, dist, n):
    mine = [(lp, tuple(v)) for v, lp in dist.enumerator()]
    brute = _brute(dist, n)
    testcase.assertEqual(len(mine), len(brute))
    np.testing.assert_allclose([m[0] for m in mine], [b[0] for b in brute], atol=1e-9)
    testcase.assertAlmostEqual(sum(np.exp(lp) for lp, _ in mine), 1.0, places=9)
    for _, v in mine:
        testcase.assertEqual(sorted(v), list(range(n)))


class RankingLazyEnumeratorTestCase(unittest.TestCase):
    def test_mallows_exact(self):
        rng = np.random.RandomState(0)
        _assert_exact(self, MallowsDistribution(rng.permutation(5), theta=0.7), 5)
        _assert_exact(self, MallowsDistribution(rng.permutation(4), theta=0.0), 4)  # uniform

    def test_plackett_luce_exact(self):
        rng = np.random.RandomState(1)
        _assert_exact(self, PlackettLuceDistribution(rng.randn(5)), 5)

    def test_spearman_exact(self):
        rng = np.random.RandomState(2)
        _assert_exact(self, SpearmanRankingDistribution(rng.permutation(5), rho=0.3), 5)

    def test_lazy_top_k_large_support(self):
        rng = np.random.RandomState(3)
        for dist in (
            MallowsDistribution(rng.permutation(10), theta=0.5),
            PlackettLuceDistribution(rng.randn(10)),
            SpearmanRankingDistribution(rng.permutation(10), rho=0.4),
        ):
            top = list(itertools.islice(dist.enumerator(), 5))  # 10! support
            self.assertEqual(len(top), 5)
            lds = [lp for _, lp in top]
            self.assertTrue(all(lds[i] >= lds[i + 1] - 1e-12 for i in range(4)))
            self.assertAlmostEqual(lds[0], max(dist.log_density(list(v)) for v, _ in [top[0]]), places=9)

    def test_top_p_and_mode(self):
        rng = np.random.RandomState(4)
        d = PlackettLuceDistribution(np.array([3.0, 1.0, 0.0, -1.0]))
        mode, lp = next(iter(d.enumerator()))
        # PL mode ranks items by descending worth
        self.assertEqual(list(mode), [0, 1, 2, 3])
        nucleus = d.enumerator().top_p(0.9)
        self.assertTrue(sum(np.exp(l) for _, l in nucleus) >= 0.9 - 1e-9)


if __name__ == "__main__":
    unittest.main()
