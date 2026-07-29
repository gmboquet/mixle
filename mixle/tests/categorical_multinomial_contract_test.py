"""Probability and encoded-data contracts for categorical multinomial models."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.enumeration.algorithms import BufferedStream
from mixle.stats import (
    CategoricalDistribution,
    MultinomialDistribution,
    PointMassDistribution,
    PoissonDistribution,
)
from mixle.stats.multivariate.categorical_multinomial import (
    MultinomialEncodedData,
    MultisetProductEnumerator,
)


class CategoricalMultinomialProbabilityContractTest(unittest.TestCase):
    def setUp(self):
        self.base = CategoricalDistribution({"a": 0.5, "b": 0.5})
        self.dist = MultinomialDistribution(self.base)

    def test_score_is_normalized_count_vector_mass(self):
        self.assertAlmostEqual(self.dist.density([("a", 1), ("b", 1)]), 0.5)
        self.assertAlmostEqual(self.dist.density([("a", 2)]), 0.25)
        self.assertAlmostEqual(self.dist.density([("b", 2)]), 0.25)

    def test_duplicate_values_are_canonicalized_and_counts_are_integral(self):
        self.assertEqual(
            self.dist.log_density([("a", 1), ("a", 1), ("b", 1)]),
            self.dist.log_density([("a", 2), ("b", 1)]),
        )
        for invalid in (
            [("a", 0.5)],
            [("a", -1)],
            [("a", np.nan)],
            [(["unhashable"], 1)],
            [("not-a-pair",)],
        ):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                self.dist.log_density(invalid)

    def test_zero_total_normalized_semantics_match_scalar_batch_and_backend(self):
        dist = MultinomialDistribution(self.base, len_normalized=True)
        data = [[], [("a", 0)], [("a", 1), ("b", 1)]]
        encoded = dist.dist_to_encoder().seq_encode(data)
        scalar = np.asarray([dist.log_density(value) for value in data])
        batch = dist.seq_log_density(encoded)
        backend = np.asarray(dist.backend_seq_log_density(encoded, NUMPY_ENGINE))
        np.testing.assert_allclose(batch, scalar)
        np.testing.assert_allclose(backend, scalar)
        self.assertEqual(scalar[0], 0.0)
        self.assertEqual(scalar[1], 0.0)

    def test_normalized_score_does_not_claim_a_sampler(self):
        dist = MultinomialDistribution(
            self.base,
            len_dist=CategoricalDistribution({2: 1.0}),
            len_normalized=True,
        )
        with self.assertRaises(ValueError):
            dist.sampler()

    def test_sampler_and_score_represent_the_same_ordinary_multinomial_law(self):
        dist = MultinomialDistribution(self.base, len_dist=CategoricalDistribution({2: 1.0}))
        sampler = dist.sampler(seed=3)
        samples = sampler.sample(size=4000)
        mixed = sum(dict(value) == {"a": 1, "b": 1} for value in samples) / len(samples)
        self.assertAlmostEqual(mixed, dist.density([("a", 1), ("b", 1)]), delta=0.04)

    def test_sampled_length_and_requested_size_must_be_nonnegative_integers(self):
        invalid_length = MultinomialDistribution(
            self.base,
            len_dist=PointMassDistribution(1.5),
        )
        with self.assertRaises((TypeError, ValueError)):
            invalid_length.sampler(seed=2).sample()
        sampler = MultinomialDistribution(
            self.base,
            len_dist=PointMassDistribution(1),
        ).sampler(seed=2)
        for invalid in (1.5, -1, True):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                sampler.sample(size=invalid)


class CategoricalMultinomialEncodingContractTest(unittest.TestCase):
    def setUp(self):
        self.first = MultinomialDistribution(CategoricalDistribution({"a": 0.25, "b": 0.75}))
        self.second = MultinomialDistribution(CategoricalDistribution({"a": 0.6, "b": 0.4}))
        self.data = [[], [("a", 1), ("b", 1)], [("a", 2), ("a", 1)]]

    def test_encoder_identity_includes_both_children_and_normalization(self):
        first = MultinomialDistribution(PointMassDistribution(1)).dist_to_encoder()
        second = MultinomialDistribution(PointMassDistribution(2)).dist_to_encoder()
        normalized = MultinomialDistribution(
            PointMassDistribution(1),
            len_normalized=True,
        ).dist_to_encoder()
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, normalized)

    def test_encoder_returns_immutable_validated_data(self):
        encoded = self.first.dist_to_encoder().seq_encode(self.data)
        self.assertIsInstance(encoded, MultinomialEncodedData)
        self.assertEqual(self.first.dist_to_encoder().row_count(encoded), len(self.data))
        for array in (
            encoded.indices,
            encoded.inverse_totals,
            encoded.nonzero_totals,
            encoded.counts,
            encoded.totals,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0:1] = 0

    def test_corrupt_outer_and_child_payloads_are_rejected(self):
        encoded = self.first.dist_to_encoder().seq_encode(self.data)
        with self.assertRaises(ValueError):
            MultinomialEncodedData(
                encoded.indices,
                encoded.inverse_totals,
                encoded.nonzero_totals,
                encoded.encoded_values,
                encoded.encoded_lengths,
                encoded.counts,
                np.asarray([0, 7, 3]),
                False,
            )
        forged = MultinomialEncodedData(
            encoded.indices,
            encoded.inverse_totals,
            encoded.nonzero_totals,
            (np.asarray([0]), np.asarray(["a"], dtype=object)),
            encoded.encoded_lengths,
            encoded.counts,
            encoded.totals,
            False,
        )
        with self.assertRaises(ValueError):
            self.first.seq_log_density(forged)
        with self.assertRaises(ValueError):
            self.first.seq_log_density(tuple(encoded))

    def test_scalar_numpy_generated_and_stacked_scores_match(self):
        encoded = self.first.dist_to_encoder().seq_encode(self.data)
        scalar = np.asarray([self.first.log_density(value) for value in self.data])
        np.testing.assert_allclose(self.first.seq_log_density(encoded), scalar)
        np.testing.assert_allclose(
            np.asarray(self.first.backend_seq_log_density(encoded, NUMPY_ENGINE)),
            scalar,
        )
        params = MultinomialDistribution.backend_stacked_params(
            [self.first, self.second],
            NUMPY_ENGINE,
        )
        stacked = np.asarray(
            MultinomialDistribution.backend_stacked_log_density(
                encoded,
                params,
                NUMPY_ENGINE,
            )
        )
        expected = np.column_stack(
            [
                [self.first.log_density(value) for value in self.data],
                [self.second.log_density(value) for value in self.data],
            ]
        )
        np.testing.assert_allclose(stacked, expected)

    def test_length_accumulation_uses_observation_weights_not_trial_weights(self):
        dist = MultinomialDistribution(
            CategoricalDistribution({"a": 0.5, "b": 0.5}),
            len_dist=PoissonDistribution(2.0),
        )
        data = [[("a", 2)], [("b", 4)]]
        weights = np.asarray([2.0, 3.0])
        estimator = dist.estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(dist.dist_to_encoder().seq_encode(data), weights, dist)
        fitted = estimator.estimate(float(weights.sum()), accumulator.value())
        self.assertAlmostEqual(fitted.len_dist.lam, (2.0 * 2.0 + 3.0 * 4.0) / 5.0)


class MultisetProductEnumeratorContractTest(unittest.TestCase):
    def test_child_values_must_be_unique_and_scores_descending(self):
        duplicate = MultisetProductEnumerator(
            BufferedStream(iter([("a", -0.1), ("a", -0.2)])),
            1,
        )
        with self.assertRaises(ValueError):
            list(duplicate)
        unsorted = MultisetProductEnumerator(
            BufferedStream(iter([("a", -0.2), ("b", -0.1)])),
            1,
        )
        with self.assertRaises(ValueError):
            list(unsorted)

    def test_size_and_child_scores_are_validated(self):
        for invalid in (1.5, -1, True):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                MultisetProductEnumerator(BufferedStream(iter([("a", -0.1)])), invalid)
        for invalid_score in (np.nan, np.inf, -np.inf, 0.1):
            with self.subTest(invalid_score=invalid_score), self.assertRaises(ValueError):
                MultisetProductEnumerator(
                    BufferedStream(iter([("a", invalid_score)])),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
