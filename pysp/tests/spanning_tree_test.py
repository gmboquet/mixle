"""Tests for the weighted spanning-tree distribution (Matrix-Tree normalizer, Wilson sampling, MLE)."""

import heapq
import itertools
import unittest
from collections import Counter

import numpy as np

from pysp.stats import SpanningTreeDistribution
from pysp.stats.graph.spanning_tree import _edge_marginals, _smoothed_edge_target
from pysp.utils.estimation import fit


def _all_trees(n):
    """Yield every spanning tree of K_n as a sorted edge tuple (via the Prufer bijection)."""
    if n == 2:
        yield ((0, 1),)
        return
    for seq in itertools.product(range(n), repeat=n - 2):
        deg = [1] * n
        for s in seq:
            deg[s] += 1
        leaves = [i for i in range(n) if deg[i] == 1]
        heapq.heapify(leaves)
        edges = []
        for s in seq:
            leaf = heapq.heappop(leaves)
            edges.append((min(leaf, s), max(leaf, s)))
            deg[leaf] -= 1
            deg[s] -= 1
            if deg[s] == 1:
                heapq.heappush(leaves, s)
        u = [i for i in range(n) if deg[i] == 1]
        edges.append((min(u[0], u[1]), max(u[0], u[1])))
        yield tuple(sorted(edges))


_W = np.array([[0.0, 2.0, 1.0, 3.0], [2.0, 0.0, 4.0, 1.0], [1.0, 4.0, 0.0, 2.0], [3.0, 1.0, 2.0, 0.0]])


class SpanningTreeTestCase(unittest.TestCase):
    def test_density_normalizes_over_all_spanning_trees(self):
        dist = SpanningTreeDistribution(_W)
        trees = [list(t) for t in _all_trees(4)]
        enc = dist.dist_to_encoder().seq_encode(trees)
        self.assertEqual(len(trees), 4 ** (4 - 2))  # Cayley's formula
        self.assertAlmostEqual(float(np.sum(np.exp(dist.seq_log_density(enc)))), 1.0, places=10)

    def test_seq_matches_scalar(self):
        dist = SpanningTreeDistribution(_W)
        trees = [list(t) for t in _all_trees(4)]
        enc = dist.dist_to_encoder().seq_encode(trees)
        np.testing.assert_allclose(dist.seq_log_density(enc), [dist.log_density(t) for t in trees])

    def test_enumerator_matches_brute_force_order(self):
        dist = SpanningTreeDistribution(_W)
        brute = [(list(t), dist.log_density(t)) for t in _all_trees(4)]
        brute.sort(key=lambda u: -u[1])
        items = list(dist.enumerator())

        # densities match in descending order (the lazy enumerator may break exact ties differently)
        np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in brute], atol=1.0e-12)
        # and the enumerated trees are exactly the support (compared as a set, tie-order-independent)
        self.assertEqual(
            {frozenset(t) for t, _ in items}, {frozenset(t) for t, _ in brute}
        )
        self.assertAlmostEqual(float(np.logaddexp.reduce([lp for _, lp in items])), 0.0, places=10)

    def test_enumerator_respects_sparse_support(self):
        weights = np.array(
            [
                [0.0, 2.0, 1.0, 0.0],
                [2.0, 0.0, 3.0, 4.0],
                [1.0, 3.0, 0.0, 5.0],
                [0.0, 4.0, 5.0, 0.0],
            ]
        )
        dist = SpanningTreeDistribution(weights)
        support = [
            list(t)
            for t in _all_trees(4)
            if all(weights[i, j] > 0.0 for i, j in t)
        ]
        brute = [(t, dist.log_density(t)) for t in support]
        brute.sort(key=lambda u: -u[1])

        items = list(dist.enumerator())
        self.assertEqual([t for t, _ in items], [t for t, _ in brute])
        self.assertAlmostEqual(float(np.logaddexp.reduce([lp for _, lp in items])), 0.0, places=10)

    def test_enumerator_is_lazy_and_complete(self):
        # the enumerator now streams trees in increasing cost (Gabow k-best), so max_edge_subsets no longer
        # caps it -- both the legacy argument and the default yield the full Cayley count of n^(n-2) trees.
        dist = SpanningTreeDistribution(_W)
        self.assertEqual(len(list(dist.enumerator(max_edge_subsets=1))), 4 ** (4 - 2))
        self.assertEqual(len(list(dist.enumerator(max_edge_subsets=None))), 4 ** (4 - 2))
        # lazily taking only the top few does not enumerate everything
        top = list(itertools.islice(dist.enumerator(), 3))
        self.assertEqual(len(top), 3)
        self.assertTrue(all(top[i][1] >= top[i + 1][1] for i in range(2)))

    def test_edge_marginals_sum_to_n_minus_1(self):
        dist = SpanningTreeDistribution(_W)
        self.assertAlmostEqual(_edge_marginals(dist.weights).sum() / 2.0, 3.0, places=10)

    def test_string_round_trip(self):
        dist = SpanningTreeDistribution(_W, name="t", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_wilson_sampler_matches_density(self):
        dist = SpanningTreeDistribution(_W)
        n = 80000
        samples = dist.sampler(seed=0).sample(n)
        empirical = Counter(tuple(t) for t in samples)
        trees = list(_all_trees(4))
        expected = np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode([list(t) for t in trees])))
        for t, p in zip(trees, expected):
            self.assertAlmostEqual(empirical[t] / n, p, delta=0.01)

    def test_estimator_recovers_edge_marginals(self):
        true = SpanningTreeDistribution(_W)
        data = true.sampler(seed=1).sample(6000)
        fitted = fit(data, true.estimator(), max_its=1, rng=np.random.RandomState(0), print_iter=0)
        np.testing.assert_allclose(_edge_marginals(fitted.weights), _edge_marginals(true.weights), atol=0.05)

    def test_pseudo_count_target_stays_in_spanning_tree_polytope(self):
        trees = [
            [(0, 1), (1, 2), (2, 3)],
            [(0, 2), (1, 2), (1, 3)],
        ]
        acc = SpanningTreeDistribution(_W).estimator().accumulator_factory().make()
        for tree in trees:
            acc.update(tree, 1.0, None)
        count, edge_counts = acc.value()
        candidate = (edge_counts + edge_counts.T) > 0.0
        np.fill_diagonal(candidate, False)

        target = _smoothed_edge_target(edge_counts, count, candidate, pseudo_count=5.0)
        raw = edge_counts / count
        prior = _edge_marginals(np.where(candidate, 1.0, 0.0))
        expected = (count * raw + 5.0 * prior) / (count + 5.0)

        np.testing.assert_allclose(target, expected * candidate, atol=1.0e-12)
        self.assertAlmostEqual(float(target.sum() / 2.0), 3.0, places=12)
        np.testing.assert_allclose(target[~candidate], 0.0, atol=1.0e-12)

    def test_negative_pseudo_count_raises(self):
        with self.assertRaises(ValueError):
            SpanningTreeDistribution(_W).estimator(pseudo_count=-1.0)

    def test_encoder_rejects_non_trees(self):
        dist = SpanningTreeDistribution(_W)
        with self.assertRaises(ValueError):  # a cycle, not a tree
            dist.dist_to_encoder().seq_encode([[(0, 1), (1, 2), (2, 0)]])
        with self.assertRaises(ValueError):  # wrong edge count
            dist.dist_to_encoder().seq_encode([[(0, 1), (1, 2)]])

    def test_invalid_weights_raise(self):
        with self.assertRaises(ValueError):
            SpanningTreeDistribution([[0.0, -1.0], [-1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
