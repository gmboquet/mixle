"""Tests for the weighted spanning-tree distribution (Matrix-Tree normalizer, Wilson sampling, MLE)."""

import heapq
import itertools
import unittest
from collections import Counter

import numpy as np

from pysp.stats import SpanningTreeDistribution
from pysp.stats.graph.spanning_tree import _edge_marginals
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
