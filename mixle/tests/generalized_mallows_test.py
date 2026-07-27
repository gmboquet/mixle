"""Generalized Mallows (Kendall/Cayley/Hamming): exact normalizer, density, and parameter recovery."""

import itertools
import math
import unittest

import numpy as np

from mixle.stats import GeneralizedMallowsDistribution
from mixle.stats.rankings._permutation_kernels import METRICS, permutation_distance
from mixle.stats.rankings.generalized_mallows import (
    GeneralizedMallowsDataEncoder,
    GeneralizedMallowsEstimator,
    expected_distance,
    metric_log_normalizer,
)

_CLOSED = ("kendall", "cayley", "hamming")


def _brute_logz(metric, theta, n):
    ident = np.arange(n)
    return math.log(
        sum(
            math.exp(-theta * permutation_distance(np.array(p), ident, metric))
            for p in itertools.permutations(range(n))
        )
    )


def _brute_expected(metric, theta, n):
    ident, num, den = np.arange(n), 0.0, 0.0
    for p in itertools.permutations(range(n)):
        d = permutation_distance(np.array(p), ident, metric)
        w = math.exp(-theta * d)
        num, den = num + d * w, den + w
    return num / den


class NormalizerTest(unittest.TestCase):
    def test_kendall_near_zero_theta_is_the_uniform_limit(self):
        dist = GeneralizedMallowsDistribution([0, 1, 2, 3], 1.0e-309, "kendall")
        self.assertAlmostEqual(dist.log_z, math.log(24.0), places=12)
        self.assertAlmostEqual(dist.density([3, 2, 1, 0]), 1.0 / 24.0, places=12)

    def test_log_normalizer_matches_brute_force_all_metrics(self):
        # closed form (kendall/cayley/hamming) and exact permanent / enumeration (footrule/spearman/ulam)
        for metric in METRICS:
            for theta in (0.0, 0.4, 1.3, 3.0):
                self.assertAlmostEqual(
                    metric_log_normalizer(metric, theta, 6), _brute_logz(metric, theta, 6), places=7, msg=metric
                )

    def test_expected_distance_matches_brute_force(self):
        for metric in _CLOSED:
            for theta in (1e-9, 0.4, 1.3, 3.0):
                self.assertAlmostEqual(expected_distance(metric, theta, 6), _brute_expected(metric, theta, 6), places=7)

    def test_density_sums_to_one_all_metrics(self):
        for metric in METRICS:
            d = GeneralizedMallowsDistribution([2, 0, 1, 4, 3], 1.1, metric)
            self.assertAlmostEqual(
                sum(d.density(list(p)) for p in itertools.permutations(range(5))), 1.0, places=9, msg=metric
            )

    def test_seq_log_density_matches_scalar(self):
        d = GeneralizedMallowsDistribution([3, 1, 0, 2], 0.8, "cayley")
        perms = np.array(list(itertools.permutations(range(4))))
        np.testing.assert_allclose(d.seq_log_density(perms), [d.log_density(p) for p in perms], atol=1e-12)


class RecoveryTest(unittest.TestCase):
    def test_estimator_recovers_center_and_theta(self):
        center = [3, 1, 4, 0, 2, 5]
        for metric in METRICS:
            true = GeneralizedMallowsDistribution(center, 1.0, metric)
            samp = true.sampler(seed=1).sample(3000)
            acc = true.estimator().accumulator_factory().make()
            acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
            fit = true.estimator().estimate(len(samp), acc.value())
            self.assertEqual(list(fit.sigma0), center, msg=metric)  # exact consensus recovery
            self.assertAlmostEqual(fit.theta, 1.0, delta=0.35, msg=metric)  # theta within sampling error
            self.assertTrue(fit.fit_diagnostics.center_exact)

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
            GeneralizedMallowsDistribution([0, 1, 2], 1.0, "manhattan")  # unknown metric
        with self.assertRaises(ValueError):
            GeneralizedMallowsDistribution([0, 1, 1], 1.0, "kendall")  # not a permutation
        with self.assertRaises(ValueError):
            GeneralizedMallowsDistribution([0, 1, 2], -1.0, "kendall")  # theta < 0

    def test_log_density_rejects_non_permutations(self):
        # A malformed x isn't just an unlikely ordering, it isn't an ordering at all: log_density
        # (and density/distance, which it's built on) must reject it rather than silently
        # returning a finite score, matching the encoder's validation above.
        dist = GeneralizedMallowsDistribution([0, 1, 2], 1.0, "kendall")
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

    def test_density_and_distance_reject_non_permutations(self):
        dist = GeneralizedMallowsDistribution([0, 1, 2], 1.0, "kendall")
        with self.assertRaises(ValueError):
            dist.density([0, 0, 1])
        with self.assertRaises(ValueError):
            dist.distance([0, 0, 1])

    def test_encoder_rejects_non_permutations(self):
        dist = GeneralizedMallowsDistribution([0, 1, 2], 1.0, "kendall")
        with self.assertRaises(ValueError):
            dist.dist_to_encoder().seq_encode([[0, 1, 1]])
        with self.assertRaises(ValueError):
            # A fractional entry must not be silently truncated to an in-range integer by the
            # encoder's int cast before the permutation check ever sees it.
            dist.dist_to_encoder().seq_encode([[0.5, 1.0, 2.0]])

    def test_uncontrolled_normalizer_fallbacks_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Monte Carlo normalization"):
            GeneralizedMallowsDistribution(list(range(6)), 100.0, "ulam", n_mc=1, max_enum=5)
        with self.assertRaisesRegex(ValueError, "Monte Carlo normalization"):
            GeneralizedMallowsDistribution(list(range(5)), 100.0, "footrule", n_mc=1, max_exact=4)

    def test_parameters_are_owned_and_encoder_identity_includes_dimension(self):
        center = np.asarray([2, 0, 1])
        dist = GeneralizedMallowsDistribution(center, 1.0, "cayley")
        center[:] = [0, 1, 2]
        np.testing.assert_array_equal(dist.sigma0, [2, 0, 1])
        with self.assertRaises(ValueError):
            dist.sigma0[0] = 0
        self.assertEqual(GeneralizedMallowsDataEncoder(3), GeneralizedMallowsDataEncoder(3))
        self.assertNotEqual(GeneralizedMallowsDataEncoder(3), GeneralizedMallowsDataEncoder(4))
        self.assertIn("dim=3", str(GeneralizedMallowsDataEncoder(3)))

    def test_all_metric_samplers_return_exact_support_values(self):
        for metric in METRICS:
            dist = GeneralizedMallowsDistribution([0, 1, 2], 1.0, metric)
            draws = dist.sampler(seed=2).sample(20)
            with self.subTest(metric=metric):
                self.assertTrue(all(sorted(draw) == [0, 1, 2] for draw in draws))

    def test_accumulator_never_silently_drops_or_aliases_evidence(self):
        estimator = GeneralizedMallowsEstimator(3, reservoir=1)
        accumulator = estimator.accumulator_factory().make()
        accumulator.update([0, 1, 2], 1.0, None)
        before = accumulator.value()
        with self.assertRaises(MemoryError):
            accumulator.update([2, 1, 0], 1.0, None)
        self.assertEqual(accumulator.value()[0], before[0])
        before[1][:] = 0.0
        before[2][:] = 0.0
        before[3][0][:] = 0
        self.assertGreater(accumulator.value()[1].sum(), 0.0)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.asarray([[0, 1, 2]]), np.asarray([-1.0]), None)

    def test_pseudo_count_and_controls_are_explicit(self):
        base = GeneralizedMallowsDistribution([2, 0, 1], 1.0, "kendall")
        fitted = base.estimator(pseudo_count=2.0).estimate(None, (0.0, np.zeros((3, 3)), np.zeros((3, 3)), [], []))
        self.assertTrue(fitted.fit_diagnostics.regularized)
        np.testing.assert_array_equal(fitted.sigma0, base.sigma0)
        invalid = (
            {"dim": 1},
            {"dim": 3, "reservoir": 0},
            {"dim": 3, "n_mc": 0},
            {"dim": 3, "max_exact": 23},
            {"dim": 3, "pseudo_count": -1.0},
            {"dim": 3, "allow_approximate_center": "false"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                GeneralizedMallowsEstimator(**kwargs)
        with self.assertRaises(ValueError):
            base.sampler().sample(-1)
        with self.assertRaises(ValueError):
            base.sampler().sample(1.5)


if __name__ == "__main__":
    unittest.main()
