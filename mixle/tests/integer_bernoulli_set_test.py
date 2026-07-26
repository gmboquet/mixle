"""Contract tests for finite-support integer Bernoulli-set distributions."""

import unittest

import numpy as np

from mixle.engines import NumpyEngine
from mixle.stats.sets.integer_bernoulli_set import (
    IntegerBernoulliSetAccumulator,
    IntegerBernoulliSetDataEncoder,
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEstimator,
    IntegerBernoulliSetFitError,
)


class IntegerBernoulliSetDistributionTestCase(unittest.TestCase):
    def test_constructor_requires_complementary_log_probabilities(self):
        invalid_primary = (
            [np.nan],
            [np.inf],
            [0.1],
            [[np.log(0.5)]],
        )
        for log_pvec in invalid_primary:
            with self.subTest(log_pvec=log_pvec), self.assertRaises(ValueError):
                IntegerBernoulliSetDistribution(log_pvec)
        with self.assertRaises(ValueError):
            IntegerBernoulliSetDistribution(
                np.log([0.2, 0.7]),
                np.log([0.8]),
            )
        with self.assertRaises(ValueError):
            IntegerBernoulliSetDistribution(
                np.log([0.8]),
                np.log([0.8]),
            )

    def test_constructor_copies_freezes_and_canonicalizes_probabilities(self):
        source = np.log([0.2, 0.7])
        dist = IntegerBernoulliSetDistribution(source)
        source[:] = np.log(0.5)
        np.testing.assert_allclose(np.exp(dist.log_pvec), [0.2, 0.7])
        np.testing.assert_allclose(
            np.exp(dist.log_pvec) + np.exp(dist.log_nvec),
            1.0,
        )
        with self.assertRaises(ValueError):
            dist.log_pvec[0] = np.log(0.4)

    def test_scoring_sampling_and_enumeration_use_one_normalized_law(self):
        dist = IntegerBernoulliSetDistribution(np.log([0.8, 0.3]))
        items = list(dist.enumerator())
        self.assertAlmostEqual(sum(np.exp(score) for _, score in items), 1.0)
        for observation, score in items:
            self.assertAlmostEqual(score, dist.log_density(observation))
        samples = dist.sampler(4).sample(size=20)
        self.assertTrue(all(set(sample).issubset({0, 1}) for sample in samples))

    def test_raw_encoded_and_backend_scoring_reject_invalid_sets(self):
        dist = IntegerBernoulliSetDistribution(np.log([0.2, 0.7]))
        encoder = dist.dist_to_encoder()
        for observation in ([0.9], [-1], [2], [0, 0]):
            with self.subTest(observation=observation):
                with self.assertRaises((TypeError, ValueError)):
                    dist.log_density(observation)
                with self.assertRaises((TypeError, ValueError)):
                    encoder.seq_encode([observation])
        invalid_encoded = (
            (1, np.asarray([0]), np.asarray([-1])),
            (1, np.asarray([0]), np.asarray([2])),
            (1, np.asarray([0, 0]), np.asarray([1, 1])),
            (1, np.asarray([1]), np.asarray([0])),
        )
        for encoded in invalid_encoded:
            with self.subTest(encoded=encoded):
                with self.assertRaises(ValueError):
                    dist.seq_log_density(encoded)
                with self.assertRaises(ValueError):
                    dist.backend_seq_log_density(encoded, NumpyEngine())

    def test_scalar_sequence_and_backend_scores_agree(self):
        dist = IntegerBernoulliSetDistribution(
            np.asarray([0.0, np.log(0.4), -np.inf])
        )
        observations = [[0], [0, 1], [1], [0, 2]]
        encoded = dist.dist_to_encoder().seq_encode(observations)
        expected = np.asarray([dist.log_density(row) for row in observations])
        np.testing.assert_allclose(
            dist.seq_log_density(encoded),
            expected,
        )
        np.testing.assert_allclose(
            NumpyEngine().to_numpy(
                dist.backend_seq_log_density(encoded, NumpyEngine())
            ),
            expected,
        )

    def test_estimator_round_trip_preserves_key_and_prior_probabilities(self):
        dist = IntegerBernoulliSetDistribution(
            np.log([0.2, 0.7]),
            name="sets",
            keys="shared",
        )
        estimator = dist.estimator(pseudo_count=3.0)
        self.assertEqual(estimator.keys, "shared")
        self.assertEqual(estimator.name, "sets")
        np.testing.assert_allclose(estimator.suff_stat, [0.2, 0.7])
        fitted = estimator.estimate(None, (np.asarray([0.0, 1.0]), 1.0))
        self.assertEqual(fitted.keys, "shared")
        self.assertEqual(fitted.name, "sets")


class IntegerBernoulliSetInferenceTestCase(unittest.TestCase):
    def test_accumulator_validates_atomically_and_copies_restored_state(self):
        acc = IntegerBernoulliSetAccumulator(2, keys="shared")
        acc.update([0], 2.0, None)
        before_counts, before_total = acc.value()
        for observation, weight in (
            ([0, 0], 1.0),
            ([-1], 1.0),
            ([2], 1.0),
            ([1], -1.0),
            ([1], np.nan),
        ):
            with self.subTest(observation=observation, weight=weight):
                with self.assertRaises((TypeError, ValueError)):
                    acc.update(observation, weight, None)
                np.testing.assert_array_equal(acc.value()[0], before_counts)
                self.assertEqual(acc.value()[1], before_total)
        source = np.asarray([1.0, 0.0])
        acc.from_value((source, 2.0))
        source[0] = 2.0
        np.testing.assert_array_equal(acc.value()[0], [1.0, 0.0])

    def test_sequence_accumulation_validates_rows_weights_and_counts(self):
        acc = IntegerBernoulliSetAccumulator(2)
        encoded = IntegerBernoulliSetDataEncoder(2).seq_encode([[0], [1]])
        for weights in (
            np.asarray([1.0]),
            np.asarray([1.0, -1.0]),
            np.asarray([1.0, np.nan]),
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                acc.seq_update(encoded, weights, None)
        with self.assertRaises(ValueError):
            acc.from_value((np.asarray([3.0, 0.0]), 2.0))

    def test_stacked_accumulation_validates_weight_geometry(self):
        engine = NumpyEngine()
        dists = (
            IntegerBernoulliSetDistribution(np.log([0.2, 0.7])),
            IntegerBernoulliSetDistribution(np.log([0.8, 0.3])),
        )
        params = IntegerBernoulliSetDistribution.backend_stacked_params(
            dists,
            engine,
        )
        encoded = dists[0].dist_to_encoder().seq_encode([[0], [1]])
        for weights in (
            np.ones((2, 1)),
            np.asarray([[1.0, -1.0], [1.0, 1.0]]),
            np.asarray([[1.0, np.nan], [1.0, 1.0]]),
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                IntegerBernoulliSetDistribution.backend_stacked_sufficient_statistics(
                    encoded,
                    weights,
                    params,
                    engine,
                )

    def test_estimator_rejects_invalid_configuration_and_statistics(self):
        for min_prob in (-0.1, 0.51, np.nan):
            with self.subTest(min_prob=min_prob), self.assertRaises(ValueError):
                IntegerBernoulliSetEstimator(2, min_prob=min_prob)
        with self.assertRaises(ValueError):
            IntegerBernoulliSetEstimator(2, pseudo_count=-1.0)
        with self.assertRaises(ValueError):
            IntegerBernoulliSetEstimator(
                2,
                pseudo_count=1.0,
                suff_stat=np.asarray([0.5, 1.2]),
            )
        estimator = IntegerBernoulliSetEstimator(2)
        with self.assertRaises(IntegerBernoulliSetFitError):
            estimator.estimate(None, (np.zeros(2), 0.0))
        with self.assertRaises(ValueError):
            estimator.estimate(None, (np.asarray([2.0, 0.0]), 1.0))

    def test_probability_floor_preserves_complements_and_normalization(self):
        estimator = IntegerBernoulliSetEstimator(1, min_prob=0.1)
        fitted = estimator.estimate(None, (np.asarray([0.0]), 1.0))
        self.assertAlmostEqual(np.exp(fitted.log_pvec[0]), 0.1)
        self.assertAlmostEqual(np.exp(fitted.log_nvec[0]), 0.9)
        self.assertAlmostEqual(
            sum(np.exp(score) for _, score in fitted.enumerator()),
            1.0,
        )

    def test_positive_pseudo_count_identifies_an_empty_fit(self):
        fitted = IntegerBernoulliSetEstimator(
            2,
            pseudo_count=2.0,
        ).estimate(None, (np.zeros(2), 0.0))
        np.testing.assert_allclose(np.exp(fitted.log_pvec), [0.5, 0.5])
        self.assertTrue(fitted.fit_metadata["regularized"])


if __name__ == "__main__":
    unittest.main()
