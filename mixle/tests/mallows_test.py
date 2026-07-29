"""Tests for the Mallows permutation distribution (normalization, distance, sampling, estimation)."""

import itertools
import math
import unittest

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats import MallowsDistribution
from mixle.stats.rankings.mallows import MallowsDataEncoder, MallowsEstimator


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

    def test_estimator_solves_the_exact_kemeny_center_within_cap(self):
        data = [[1, 2, 0, 3], [1, 0, 3, 2], [1, 3, 2, 0]]
        est = MallowsDistribution([0, 1, 2, 3]).estimator()
        acc = est.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)

        fitted = est.estimate(None, acc.value())
        count, precede = acc.value()
        exact = min(_kendall_objective(precede, p) for p in _orderings(4))

        self.assertEqual(count, 3.0)
        self.assertEqual(_kendall_objective(precede, fitted.sigma0), exact)
        self.assertTrue(fitted.fit_diagnostics.center_exact)
        self.assertEqual(fitted.fit_diagnostics.center_algorithm, "exact_kemeny_enumeration")

    def test_encoder_rejects_non_permutations(self):
        with self.assertRaises(ValueError):
            MallowsDistribution([0, 1, 2]).dist_to_encoder().seq_encode([[0, 1, 1]])
        with self.assertRaises(ValueError):
            # A fractional entry must not be silently truncated to an in-range integer by the
            # encoder's int cast before the permutation check ever sees it.
            MallowsDistribution([0, 1, 2]).dist_to_encoder().seq_encode([[0.5, 1.0, 2.0]])

    def test_log_density_rejects_non_permutations(self):
        # A malformed x isn't just an unlikely ordering, it isn't an ordering at all: log_density
        # (and density/kendall_distance, which it's built on) must reject it rather than silently
        # returning a finite score, matching the encoder's validation above.
        dist = MallowsDistribution([0, 1, 2], theta=1.0)
        with self.assertRaises(ValueError):
            dist.log_density([0, 0, 1])  # repeated item
        with self.assertRaises(ValueError):
            dist.log_density([0, 1, 5])  # out of range
        with self.assertRaises(ValueError):
            dist.log_density([0, 1])  # wrong length
        with self.assertRaises(ValueError):
            dist.log_density([-1, 1, 2])  # negative index would otherwise alias a valid rank
        with self.assertRaises(ValueError):
            dist.log_density([0.5, 1.0, 2.0])  # fractional entry would otherwise truncate to 0

    def test_density_and_kendall_distance_reject_non_permutations(self):
        dist = MallowsDistribution([0, 1, 2], theta=1.0)
        with self.assertRaises(ValueError):
            dist.density([0, 0, 1])
        with self.assertRaises(ValueError):
            dist.kendall_distance([0, 0, 1])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            MallowsDistribution([0, 1, 2], theta=-1.0)
        with self.assertRaises(ValueError):
            MallowsDistribution([0, 0, 1])

    def test_vector_scoring_and_encoder_dimension_contracts(self):
        dist = MallowsDistribution([0, 1, 2])
        for value in ([0, 0, 1], [0, 1], [0, 1, 4], [0.5, 1.0, 2.0]):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.seq_log_density(np.asarray([value]))
        self.assertEqual(MallowsDataEncoder(3), MallowsDataEncoder(3))
        self.assertNotEqual(MallowsDataEncoder(3), MallowsDataEncoder(4))
        self.assertIn("dim=3", str(MallowsDataEncoder(3)))

    def test_distribution_owns_immutable_center(self):
        center = np.asarray([2, 0, 1])
        dist = MallowsDistribution(center)
        center[:] = [0, 1, 2]
        np.testing.assert_array_equal(dist.sigma0, [2, 0, 1])
        with self.assertRaises(ValueError):
            dist.sigma0[0] = 0

    def test_accumulator_validates_evidence_and_owns_state(self):
        acc = MallowsDistribution([0, 1, 2]).estimator().accumulator_factory().make()
        with self.assertRaises(ValueError):
            acc.update([0, 0, 1], 1.0, None)
        with self.assertRaises(ValueError):
            acc.seq_update(np.asarray([[0, 1, 2]]), np.asarray([-1.0]), None)
        with self.assertRaises(ValueError):
            acc.seq_update(np.asarray([[0, 1, 2]]), np.asarray([1.0, 2.0]), None)
        acc.update([0, 1, 2], 1.0, None)
        _, precede = acc.value()
        precede[:] = 0.0
        self.assertGreater(acc.value()[1].sum(), 0.0)
        with self.assertRaises(ValueError):
            acc.from_value((1.0, np.zeros((3, 3))))

    def test_pseudo_count_and_controls_are_enforced(self):
        dist = MallowsDistribution([2, 0, 1], theta=1.0)
        estimate = dist.estimator(pseudo_count=2.0).estimate(None, (0.0, np.zeros((3, 3))))
        np.testing.assert_array_equal(estimate.sigma0, dist.sigma0)
        self.assertGreater(estimate.theta, 0.0)
        with self.assertRaises(ValueError):
            dist.estimator(pseudo_count=-1.0)
        with self.assertRaises(ValueError):
            dist.sampler().sample(-1)
        with self.assertRaises(ValueError):
            dist.sampler().sample(1.5)

    def test_near_zero_theta_is_the_continuous_uniform_limit(self):
        dist = MallowsDistribution([0, 1, 2, 3], theta=1.0e-309)
        self.assertAlmostEqual(dist.log_z, math.log(24.0), places=12)
        self.assertAlmostEqual(dist.density([3, 2, 1, 0]), 1.0 / 24.0, places=12)

    def test_large_center_approximation_requires_explicit_opt_in(self):
        counts = np.triu(np.ones((4, 4)), 1)
        with self.assertRaisesRegex(ValueError, "exact Mallows center search"):
            MallowsEstimator(4, center_exact_cap=3).estimate(None, (1.0, counts))
        fitted = MallowsEstimator(
            4,
            center_exact_cap=3,
            allow_approximate_center=True,
        ).estimate(None, (1.0, counts))
        self.assertFalse(fitted.fit_diagnostics.center_exact)
        self.assertEqual(fitted.fit_diagnostics.center_algorithm, "copeland_approximation")
        with self.assertRaises(TypeError):
            MallowsEstimator(3, allow_approximate_center="false")


if __name__ == "__main__":
    unittest.main()
