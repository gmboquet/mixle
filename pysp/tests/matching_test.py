"""Tests for the weighted bipartite matching distribution (permanent normalizer, sampling, MLE)."""

import itertools
import unittest
from collections import Counter

import numpy as np

from pysp.stats import MatchingDistribution
from pysp.stats.graph.matching import _edge_marginals, _permanent
from pysp.utils.estimation import fit

_W = np.array([[2.0, 1.0, 3.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]])


def _perms(n):
    return [list(p) for p in itertools.permutations(range(n))]


class MatchingTestCase(unittest.TestCase):
    def test_density_normalizes_over_all_matchings(self):
        dist = MatchingDistribution(_W)
        enc = dist.dist_to_encoder().seq_encode(_perms(3))
        self.assertAlmostEqual(float(np.sum(np.exp(dist.seq_log_density(enc)))), 1.0, places=10)

    def test_ryser_permanent_matches_brute_force(self):
        brute = sum(np.prod([_W[i, p[i]] for i in range(3)]) for p in itertools.permutations(range(3)))
        self.assertAlmostEqual(_permanent(_W), brute, places=10)

    def test_edge_marginals_are_doubly_stochastic(self):
        marg = _edge_marginals(_W)
        np.testing.assert_allclose(marg.sum(axis=1), 1.0, atol=1.0e-10)
        np.testing.assert_allclose(marg.sum(axis=0), 1.0, atol=1.0e-10)

    def test_seq_matches_scalar(self):
        dist = MatchingDistribution(_W)
        enc = dist.dist_to_encoder().seq_encode(_perms(3))
        np.testing.assert_allclose(dist.seq_log_density(enc), [dist.log_density(p) for p in _perms(3)])

    def test_string_round_trip(self):
        dist = MatchingDistribution(_W, name="m", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_sampler_matches_density(self):
        dist = MatchingDistribution(_W)
        n = 60000
        samples = dist.sampler(seed=0).sample(n)
        empirical = Counter(tuple(s) for s in samples)
        perms = _perms(3)
        expected = np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode(perms)))
        for p, q in zip(perms, expected):
            self.assertAlmostEqual(empirical[tuple(p)] / n, q, delta=0.01)

    def test_estimator_recovers_edge_marginals(self):
        true = MatchingDistribution(_W)
        data = true.sampler(seed=1).sample(5000)
        fitted = fit(data, true.estimator(), max_its=1, rng=np.random.RandomState(0), print_iter=0)
        np.testing.assert_allclose(_edge_marginals(fitted.weights), _edge_marginals(true.weights), atol=0.05)

    def test_node_cap_and_validation(self):
        with self.assertRaises(ValueError):  # exceeds max_nodes
            MatchingDistribution(np.ones((4, 4)), max_nodes=3)
        with self.assertRaises(ValueError):  # non-positive weight
            MatchingDistribution([[1.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):  # not a permutation
            MatchingDistribution(_W).dist_to_encoder().seq_encode([[0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
