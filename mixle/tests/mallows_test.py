"""Tests for the Mallows permutation distribution (normalization, distance, sampling, estimation)."""

import itertools
import unittest

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats import MallowsDistribution


def _orderings(n):
    return [list(p) for p in itertools.permutations(range(n))]


def _kendall_objective(precede, sigma0):
    rank = {item: r for r, item in enumerate(sigma0)}
    total = 0.0
    for a in range(len(sigma0)):
        for b in range(len(sigma0)):
            if rank[a] < rank[b]:
                total += precede[b, a]
    return total


class MallowsTestCase(unittest.TestCase):
    def test_density_normalizes_over_all_orderings(self):
        dist = MallowsDistribution([2, 0, 1, 3], theta=0.8)
        enc = dist.dist_to_encoder().seq_encode(_orderings(4))
        self.assertAlmostEqual(float(np.sum(np.exp(dist.seq_log_density(enc)))), 1.0, places=10)

    def test_mode_is_central_permutation(self):
        sigma0 = [2, 0, 1, 3]
        dist = MallowsDistribution(sigma0, theta=1.5)
        orders = _orderings(4)
        probs = np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode(orders)))
        self.assertEqual(orders[int(np.argmax(probs))], sigma0)
        self.assertEqual(dist.kendall_distance(sigma0), 0)

    def test_theta_zero_is_uniform(self):
        dist = MallowsDistribution([0, 1, 2, 3], theta=0.0)
        enc = dist.dist_to_encoder().seq_encode(_orderings(4))
        np.testing.assert_allclose(np.exp(dist.seq_log_density(enc)), 1.0 / 24.0)

    def test_kendall_distance_matches_inversions(self):
        dist = MallowsDistribution([0, 1, 2, 3], theta=1.0)
        # reversing the identity gives the maximum distance n(n-1)/2 = 6.
        self.assertEqual(dist.kendall_distance([3, 2, 1, 0]), 6)
        self.assertEqual(dist.kendall_distance([1, 0, 2, 3]), 1)

    def test_seq_matches_scalar(self):
        dist = MallowsDistribution([1, 2, 0], theta=1.2)
        orders = _orderings(3)
        enc = dist.dist_to_encoder().seq_encode(orders)
        np.testing.assert_allclose(dist.seq_log_density(enc), [dist.log_density(o) for o in orders])

    def test_string_round_trip(self):
        dist = MallowsDistribution([2, 0, 1, 3], theta=0.8, name="m", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_sampler_frequencies_match_density(self):
        dist = MallowsDistribution([1, 2, 0], theta=1.2)
        n = 40000
        samples = dist.sampler(seed=0).sample(n)
        orders = _orderings(3)
        index = {tuple(o): i for i, o in enumerate(orders)}
        counts = np.zeros(len(orders))
        for s in samples:
            counts[index[tuple(s)]] += 1
        expected = np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode(orders)))
        np.testing.assert_allclose(counts / n, expected, atol=0.01)

    def test_estimator_recovers_central_permutation_and_theta(self):
        true = MallowsDistribution([3, 1, 4, 0, 2], theta=1.0)
        data = true.sampler(seed=1).sample(8000)
        fitted = fit(data, true.estimator(), max_its=1, rng=np.random.RandomState(0), print_iter=0)
        self.assertEqual(list(fitted.sigma0), list(true.sigma0))
        self.assertAlmostEqual(fitted.theta, 1.0, delta=0.15)

    def test_copeland_estimator_is_not_claimed_as_exact_kemeny(self):
        data = [[1, 2, 0, 3], [1, 0, 3, 2], [1, 3, 2, 0]]
        est = MallowsDistribution([0, 1, 2, 3]).estimator()
        acc = est.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)

        fitted = est.estimate(None, acc.value())
        count, precede = acc.value()
        exact = min(_kendall_objective(precede, p) for p in _orderings(4))

        self.assertEqual(count, 3.0)
        self.assertEqual(list(fitted.sigma0), [1, 0, 2, 3])
        self.assertGreater(_kendall_objective(precede, fitted.sigma0), exact)

    def test_encoder_rejects_non_permutations(self):
        with self.assertRaises(ValueError):
            MallowsDistribution([0, 1, 2]).dist_to_encoder().seq_encode([[0, 1, 1]])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            MallowsDistribution([0, 1, 2], theta=-1.0)
        with self.assertRaises(ValueError):
            MallowsDistribution([0, 0, 1])


if __name__ == "__main__":
    unittest.main()
