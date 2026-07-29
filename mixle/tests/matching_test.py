"""Tests for the weighted bipartite matching distribution (permanent normalizer, sampling, MLE)."""

import itertools
import unittest
from collections import Counter

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats import MatchingDistribution
from mixle.stats.rankings.matching import MatchingDataEncoder, MatchingEstimator, _edge_marginals, _permanent

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

    def test_enumerator_matches_brute_force_order(self):
        dist = MatchingDistribution(_W)
        brute = [(p, dist.log_density(p)) for p in _perms(3)]
        brute.sort(key=lambda u: -u[1])
        items = list(dist.enumerator())

        self.assertEqual([p for p, _ in items], [p for p, _ in brute])
        np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in brute], atol=1.0e-12)
        self.assertAlmostEqual(float(np.logaddexp.reduce([lp for _, lp in items])), 0.0, places=10)

    def test_string_round_trip(self):
        dist = MatchingDistribution(_W, name="m", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_sampler_matches_density(self):
        dist = MatchingDistribution(_W)
        # n=20000 keeps a comfortable margin under the delta=0.01 tolerance (worst observed
        # |empirical - expected| ~0.006-0.007 across many seeds, vs. the 0.01 threshold) while
        # cutting sampling cost ~3x relative to the original n=60000.
        n = 20000
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
        self.assertTrue(fitted.fit_diagnostics.converged)
        self.assertLess(fitted.fit_diagnostics.max_marginal_error, 1.0e-7)

    def test_estimator_pseudo_count_argument_is_respected(self):
        dist = MatchingDistribution(_W)
        self.assertEqual(dist.estimator(pseudo_count=0.0).pseudo_count, 0.0)
        self.assertEqual(dist.estimator(pseudo_count=2.5).pseudo_count, 2.5)
        with self.assertRaises(ValueError):
            dist.estimator(pseudo_count=-1.0)

    def test_node_cap_and_validation(self):
        with self.assertRaises(ValueError):  # exceeds max_nodes
            MatchingDistribution(np.ones((4, 4)), max_nodes=3)
        with self.assertRaises(ValueError):  # non-positive weight
            MatchingDistribution([[1.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):  # not a permutation
            MatchingDistribution(_W).dist_to_encoder().seq_encode([[0, 0, 1]])

    def test_extreme_finite_weights_normalize_and_sample(self):
        for scale in (1.0e-300, 1.0e308):
            with self.subTest(scale=repr(scale)):
                dist = MatchingDistribution(np.full((3, 3), scale))
                values = np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode(_perms(3))))
                np.testing.assert_allclose(values, 1.0 / 6.0, atol=1.0e-12)
                self.assertEqual(len(dist.sampler(seed=2).sample(20)), 20)

    def test_distribution_owns_immutable_parameters(self):
        weights = _W.copy()
        dist = MatchingDistribution(weights)
        weights[0, 0] = 1.0e100
        self.assertEqual(dist.weights[0, 0], _W[0, 0])
        with self.assertRaises(ValueError):
            dist.weights[0, 0] = 99.0
        with self.assertRaises(ValueError):
            dist.log_weights[0, 0] = 99.0

    def test_all_boundaries_validate_matching_support(self):
        dist = MatchingDistribution(_W)
        malformed = ([0, 0, 1], [0, 1], [0, 1, 4], [0.5, 1.0, 2.0])
        for value in malformed:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.log_density(value)
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.seq_log_density(np.asarray([value]))
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.dist_to_encoder().seq_encode([value])
        self.assertEqual(MatchingDataEncoder(3), MatchingDataEncoder(3))
        self.assertNotEqual(MatchingDataEncoder(3), MatchingDataEncoder(4))
        self.assertIn("dim=3", str(MatchingDataEncoder(3)))

    def test_accumulator_validates_evidence_and_copies_statistics(self):
        acc = MatchingEstimator(3).accumulator_factory().make()
        with self.assertRaises(ValueError):
            acc.update([0, 0, 1], 1.0, None)
        with self.assertRaises(ValueError):
            acc.seq_update(np.asarray([[0, 1, 2]]), np.asarray([-1.0]), None)
        with self.assertRaises(ValueError):
            acc.seq_update(np.asarray([[0, 1, 2]]), np.asarray([1.0, 2.0]), None)
        acc.update([0, 1, 2], 1.0, None)
        receipt = acc.value()
        receipt[1][:] = 0.0
        self.assertEqual(acc.value()[1].sum(), 3.0)
        malformed = np.zeros((3, 3))
        malformed[0, 0] = 3.0
        with self.assertRaises(ValueError):
            acc.from_value((1.0, malformed))

    def test_estimator_controls_targets_and_convergence_are_enforced(self):
        invalid = (
            {"dim": 0},
            {"dim": 3.5},
            {"dim": 3, "max_nodes": 2},
            {"dim": 3, "pseudo_count": -1.0},
            {"dim": 3, "max_steps": 0},
            {"dim": 3, "learning_rate": 0.0},
            {"dim": 3, "tol": 0.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                MatchingEstimator(**kwargs)
        malformed = np.zeros((3, 3))
        malformed[0, 0] = 3.0
        with self.assertRaises(ValueError):
            MatchingEstimator(3).estimate(None, (1.0, malformed))
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            MatchingEstimator(3, max_steps=1, tol=1.0e-15).estimate(1.0, (1.0, np.eye(3)))

    def test_sample_size_contract(self):
        sampler = MatchingDistribution(_W).sampler()
        with self.assertRaises(ValueError):
            sampler.sample(-1)
        with self.assertRaises(ValueError):
            sampler.sample(1.5)


if __name__ == "__main__":
    unittest.main()
