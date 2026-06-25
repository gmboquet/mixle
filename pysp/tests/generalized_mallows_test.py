"""Generalized Mallows (Kendall/Cayley/Hamming): exact normalizer, density, and parameter recovery."""

import itertools
import math
import unittest

import numpy as np

from pysp.stats import GeneralizedMallowsDistribution
from pysp.stats.rankings._permutation_kernels import permutation_distance
from pysp.stats.rankings.generalized_mallows import expected_distance, log_normalizer

_METRICS = ("kendall", "cayley", "hamming")


def _brute_logz(metric, theta, n):
    ident = np.arange(n)
    return math.log(
        sum(math.exp(-theta * permutation_distance(np.array(p), ident, metric)) for p in itertools.permutations(range(n)))
    )


def _brute_expected(metric, theta, n):
    ident, num, den = np.arange(n), 0.0, 0.0
    for p in itertools.permutations(range(n)):
        d = permutation_distance(np.array(p), ident, metric)
        w = math.exp(-theta * d)
        num, den = num + d * w, den + w
    return num / den


class NormalizerTest(unittest.TestCase):
    def test_log_normalizer_matches_brute_force(self):
        for metric in _METRICS:
            for theta in (0.0, 0.4, 1.3, 3.0):
                self.assertAlmostEqual(log_normalizer(metric, theta, 6), _brute_logz(metric, theta, 6), places=9)

    def test_expected_distance_matches_brute_force(self):
        for metric in _METRICS:
            for theta in (1e-9, 0.4, 1.3, 3.0):
                self.assertAlmostEqual(
                    expected_distance(metric, theta, 6), _brute_expected(metric, theta, 6), places=7
                )

    def test_density_sums_to_one(self):
        for metric in _METRICS:
            d = GeneralizedMallowsDistribution([2, 0, 1, 4, 3], 1.1, metric)
            self.assertAlmostEqual(sum(d.density(list(p)) for p in itertools.permutations(range(5))), 1.0, places=9)

    def test_seq_log_density_matches_scalar(self):
        d = GeneralizedMallowsDistribution([3, 1, 0, 2], 0.8, "cayley")
        perms = np.array(list(itertools.permutations(range(4))))
        np.testing.assert_allclose(d.seq_log_density(perms), [d.log_density(p) for p in perms], atol=1e-12)


class RecoveryTest(unittest.TestCase):
    def test_estimator_recovers_center_and_theta(self):
        center = [3, 1, 4, 0, 2, 5]
        for metric in _METRICS:
            true = GeneralizedMallowsDistribution(center, 1.0, metric)
            samp = true.sampler(seed=1).sample(3000)
            acc = true.estimator().accumulator_factory().make()
            acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
            fit = true.estimator().estimate(len(samp), acc.value())
            self.assertEqual(list(fit.sigma0), center, msg=metric)  # exact consensus recovery
            self.assertAlmostEqual(fit.theta, 1.0, delta=0.35, msg=metric)  # theta within sampling error

    def test_combine_equals_single_shard(self):
        true = GeneralizedMallowsDistribution([0, 2, 1, 3], 1.2, "hamming")
        enc = true.dist_to_encoder().seq_encode(true.sampler(seed=3).sample(400))
        est = true.estimator()

        def shard(rows):
            a = est.accumulator_factory().make()
            a.seq_update(rows, np.ones(len(rows)), None)
            return a

        a = shard(enc[:250])
        a.combine(shard(enc[250:]).value())
        full = shard(enc)
        self.assertEqual(list(est.estimate(400, a.value()).sigma0), list(est.estimate(400, full.value()).sigma0))


class ValidationTest(unittest.TestCase):
    def test_rejects_bad_metric_and_params(self):
        with self.assertRaises(ValueError):
            GeneralizedMallowsDistribution([0, 1, 2], 1.0, "ulam")  # not a closed-form metric here
        with self.assertRaises(ValueError):
            GeneralizedMallowsDistribution([0, 1, 1], 1.0, "kendall")  # not a permutation
        with self.assertRaises(ValueError):
            GeneralizedMallowsDistribution([0, 1, 2], -1.0, "kendall")  # theta < 0


if __name__ == "__main__":
    unittest.main()
