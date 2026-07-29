"""Tests for the general birth-death-sampling process (fossilized birth-death is a special case)."""

import math
import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import BirthDeathSamplingDistribution, BirthDeathSamplingEstimator


class BirthDeathSamplingTest(unittest.TestCase):
    def test_log_density_closed_form(self):
        d = BirthDeathSamplingDistribution(0.5, 0.3, 0.2)
        # n0=2: birth at t=1 (n:2->3), sampling at t=2 (n=3), death at t=4 (n:3->2); horizon T=5.
        traj = (2, 5.0, [(1.0, 0), (2.0, 2), (4.0, 1)])
        # integral n dt = 2*1 + 3*(2-1) + 3*(4-2) + 2*(5-4) = 2 + 3 + 6 + 2 = 13
        # sum log n at events = log2 + log3 + log3
        expected = (
            (math.log(2) + math.log(3) + math.log(3))
            + 1 * math.log(0.5)  # one birth
            + 1 * math.log(0.3)  # one death
            + 1 * math.log(0.2)  # one sampling
            - (0.5 + 0.3 + 0.2) * 13.0
        )
        self.assertAlmostEqual(d.log_density(traj), expected, places=10)

    def test_seq_matches_scalar(self):
        d = BirthDeathSamplingDistribution(0.7, 0.4, 0.3, initial_population=3, horizon=4.0)
        trajs = d.sampler(0).sample(20)
        enc = d.dist_to_encoder().seq_encode(trajs)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(t) for t in trajs])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_recovers_rates(self):
        true = BirthDeathSamplingDistribution(0.8, 0.4, 0.3, initial_population=6, horizon=4.0)
        data = true.sampler(1).sample(500)
        fit = optimize(data, BirthDeathSamplingEstimator(), max_its=1, rng=np.random.RandomState(0), out=None)
        self.assertAlmostEqual(fit.birth_rate, 0.8, delta=0.12)
        self.assertAlmostEqual(fit.death_rate, 0.4, delta=0.12)
        self.assertAlmostEqual(fit.sampling_rate, 0.3, delta=0.12)

    def test_pure_birth_death_no_sampling(self):
        # sampling_rate = 0: no sampling events, recovered sampling rate stays 0, no NaN/inf.
        true = BirthDeathSamplingDistribution(0.7, 0.5, 0.0, initial_population=8, horizon=4.0)
        data = true.sampler(2).sample(400)
        for traj in data:
            self.assertFalse(any(etype == 2 for _, etype in traj[2]))
            self.assertTrue(math.isfinite(true.log_density(traj)))
        fit = optimize(data, BirthDeathSamplingEstimator(), max_its=1, rng=np.random.RandomState(0), out=None)
        self.assertEqual(fit.sampling_rate, 0.0)
        self.assertAlmostEqual(fit.birth_rate, 0.7, delta=0.12)


class BirthDeathSamplingValidationTestCase(unittest.TestCase):
    """Regression coverage for BirthDeathSamplingDistribution.__init__ parameter validation.

    An external review flagged this constructor's `horizon` (the `[0, horizon]` observation window
    used by the sampler and the upper integration limit in log_density()) as unvalidated, unlike its
    sibling rate parameters just above it in the same constructor. Before this fix, a NaN horizon
    constructed silently and made `t >= d.horizon` in BirthDeathSamplingSampler._sample_one() always
    False (`x >= nan` is always False), dropping the Gillespie loop's only time-based stop condition;
    a negative horizon constructed silently too and produced a negative population-time integral,
    silently corrupting log_density().
    """

    def test_nan_horizon_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            BirthDeathSamplingDistribution(0.5, 0.3, 0.2, horizon=float("nan"))

    def test_negative_horizon_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            BirthDeathSamplingDistribution(0.5, 0.3, 0.2, horizon=-5.0)

    def test_infinite_horizon_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            BirthDeathSamplingDistribution(0.5, 0.3, 0.2, horizon=float("inf"))

    def test_zero_horizon_still_constructs(self):
        # Matches the >= 0 boundary already used for birth_rate/death_rate/sampling_rate in the same
        # validation loop: a zero-length window is a degenerate but valid empty observation, not an
        # error.
        d = BirthDeathSamplingDistribution(0.5, 0.3, 0.2, horizon=0.0)
        self.assertEqual(d.horizon, 0.0)

    def test_valid_horizon_still_constructs(self):
        d = BirthDeathSamplingDistribution(0.5, 0.3, 0.2, horizon=4.0)
        self.assertEqual(d.horizon, 4.0)

    def test_initial_population_must_be_an_exact_nonnegative_integer(self):
        for value in (True, -1, 0.5):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                BirthDeathSamplingDistribution(
                    0.5,
                    0.3,
                    0.2,
                    initial_population=value,
                )
        dist = BirthDeathSamplingDistribution(
            0.5,
            0.3,
            0.2,
            initial_population=0,
            horizon=4.0,
        )
        self.assertEqual(dist.sampler(seed=1).sample(), (0, 4.0, []))
        self.assertEqual(dist.log_density((0, 4.0, [])), 0.0)

    def test_trajectory_is_validated_transactionally(self):
        dist = BirthDeathSamplingDistribution(0.5, 0.3, 0.2)
        malformed = (
            (0.5, 4.0, []),
            (1, np.nan, []),
            (1, 4.0, [(np.nan, 0)]),
            (1, 4.0, [(-1.0, 0)]),
            (1, 4.0, [(1.0, 0), (1.0, 1)]),
            (1, 4.0, [(5.0, 0)]),
            (1, 4.0, [(1.0, 0.5)]),
            (1, 4.0, [(1.0, 3)]),
            (1, 4.0, [(1.0,)]),
            (1, 4.0),
        )
        for trajectory in malformed:
            with self.subTest(trajectory=trajectory), self.assertRaises((TypeError, ValueError)):
                dist.log_density(trajectory)

    def test_encoded_schema_and_weights_fail_closed(self):
        from mixle.stats.processes.birth_death import BirthDeathSamplingAccumulator

        dist = BirthDeathSamplingDistribution(0.5, 0.3, 0.2)
        valid = np.array([[1.0, 0.0, 0.0, 2.0, 0.0, 4.0]])
        malformed = (
            np.zeros((1, 5)),
            np.zeros((1, 7)),
            np.array([[0.5, 0.0, 0.0, 2.0, 0.0, 4.0]]),
            np.array([[-1.0, 0.0, 0.0, 2.0, 0.0, 4.0]]),
            np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 4.0]]),
            np.array([[0.0, 0.0, 0.0, 2.0, np.nan, 4.0]]),
        )
        for rows in malformed:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                dist.seq_log_density(rows)

        for weights in ([], [-1.0], [np.nan]):
            accumulator = BirthDeathSamplingAccumulator()
            before = accumulator.value()
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                accumulator.seq_update(valid, weights, None)
            self.assertEqual(accumulator.value(), before)

    def test_serialized_statistics_are_versioned_and_validated(self):
        from mixle.stats.processes.birth_death import BirthDeathSamplingAccumulator

        accumulator = BirthDeathSamplingAccumulator()
        accumulator.update((2, 4.0, [(1.0, 0)]), 1.0, None)
        value = accumulator.value()
        self.assertEqual(value.schema_version, 1)
        malformed = (
            (1.0, 0.0, 0.0, 0.0, 1.0, 4.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 4.0),
            (0.0, 0.0, 0.0, -1.0, 1.0, 4.0),
            (0.0, 0.0, 0.0, np.nan, 1.0, 4.0),
        )
        for statistic in malformed:
            with self.subTest(statistic=statistic):
                with self.assertRaises(ValueError):
                    BirthDeathSamplingAccumulator().combine(statistic)
                with self.assertRaises(ValueError):
                    BirthDeathSamplingEstimator().estimate(None, statistic)

    def test_fit_preserves_sampling_configuration_and_regularization(self):
        from mixle.utils.serialization import from_json, to_json

        dist = BirthDeathSamplingDistribution(
            0.5,
            0.3,
            0.2,
            initial_population=7,
            horizon=9.0,
            name="process",
            keys="shared",
        )
        estimator = dist.estimator(pseudo_count=2.0)
        statistic = (2.0, 1.0, 0.0, 4.0, 1.0, 4.0)
        fit = estimator.estimate(None, statistic)
        self.assertAlmostEqual(fit.birth_rate, 4.0 / 6.0)
        self.assertAlmostEqual(fit.death_rate, 3.0 / 6.0)
        self.assertAlmostEqual(fit.sampling_rate, 2.0 / 6.0)
        self.assertEqual(fit.initial_population, 7)
        self.assertEqual(fit.horizon, 9.0)
        self.assertEqual(fit.name, "process")
        self.assertEqual(fit.keys, "shared")

        loaded = from_json(to_json(estimator))
        np.testing.assert_array_equal(loaded.prior_shape, estimator.prior_shape)
        np.testing.assert_array_equal(loaded.prior_rate, estimator.prior_rate)
        self.assertEqual(loaded.initial_population, 7)
        self.assertEqual(loaded.horizon, 9.0)

    def test_explicit_gamma_prior_is_applied_per_rate(self):
        estimator = BirthDeathSamplingEstimator(
            initial_population=3,
            horizon=8.0,
            prior_shape=[2.0, 3.0, 4.0],
            prior_rate=[5.0, 6.0, 7.0],
        )
        fit = estimator.estimate(
            None,
            (1.0, 2.0, 3.0, 4.0, 1.0, 4.0),
        )
        np.testing.assert_allclose(
            [fit.birth_rate, fit.death_rate, fit.sampling_rate],
            [2.0 / 9.0, 4.0 / 10.0, 6.0 / 11.0],
        )
        with self.assertRaises(ValueError):
            BirthDeathSamplingEstimator(
                pseudo_count=1.0,
                prior_shape=2.0,
            )


if __name__ == "__main__":
    unittest.main()
