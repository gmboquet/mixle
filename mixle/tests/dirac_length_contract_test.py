"""Boundary, evidence, and weighted-initialization contracts for Dirac lengths."""

import unittest

import numpy as np

from mixle.stats.combinator.null_dist import NullAccumulator, NullDistribution, NullEstimator
from mixle.stats.latent.dirac_length import (
    DiracLengthMixtureAccumulator,
    DiracLengthMixtureDistribution,
    DiracLengthMixtureEstimator,
)
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution


class DiracLengthContractTestCase(unittest.TestCase):
    def test_closed_probability_boundaries_are_exact_degenerate_laws(self):
        length = IntegerCategoricalDistribution(1, [0.4, 0.6])
        pure_dirac = DiracLengthMixtureDistribution(length, p=0.0, v=0)
        np.testing.assert_allclose([pure_dirac.density(x) for x in (0, 1, 2)], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(pure_dirac.posterior(0), [0.0, 1.0])
        np.testing.assert_array_equal(pure_dirac.posterior(1), [0.0, 0.0])
        self.assertEqual(pure_dirac.sampler(3).sample(20), [0] * 20)
        self.assertEqual(list(pure_dirac.enumerator()), [(0, 0.0)])

        pure_length = DiracLengthMixtureDistribution(length, p=1.0, v=0)
        np.testing.assert_allclose([pure_length.density(x) for x in (0, 1, 2)], [0.0, 0.4, 0.6])
        np.testing.assert_array_equal(pure_length.posterior(0), [0.0, 0.0])
        self.assertEqual(sorted(value for value, _ in pure_length.enumerator()), [1, 2])

        for model in (pure_dirac, pure_length):
            encoded = model.dist_to_encoder().seq_encode([0, 1, 2])
            np.testing.assert_allclose(
                model.seq_log_density(encoded),
                [model.log_density(value) for value in (0, 1, 2)],
            )
            np.testing.assert_allclose(
                model.seq_posterior(encoded),
                [model.posterior(value) for value in (0, 1, 2)],
            )

    def test_dirac_support_and_observations_require_exact_integers(self):
        length = IntegerCategoricalDistribution(0, [1.0])
        for invalid in (0.5, True, np.nan, np.inf, "0"):
            with self.subTest(location=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                DiracLengthMixtureDistribution(length, p=0.5, v=invalid)

        model = DiracLengthMixtureDistribution(length, p=0.5, v=0)
        for invalid in (0.5, True, np.nan, np.inf, "0"):
            with self.subTest(observation=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                model.log_density(invalid)
            with self.subTest(encoded_observation=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                model.dist_to_encoder().seq_encode([invalid])

    def test_fixed_zero_probability_is_preserved(self):
        estimator = DiracLengthMixtureEstimator(NullEstimator(), v=2, fixed_p=0.0)
        model = estimator.estimate(None, (np.asarray([7.0, 3.0]), None))
        self.assertEqual(model.p, 0.0)
        self.assertEqual(model.v, 2)

    def test_scalar_initialization_applies_the_full_observation_weight(self):
        weight = 10.0
        scalar = DiracLengthMixtureAccumulator(NullAccumulator(), v=0)
        scalar.initialize(0, weight, np.random.RandomState(7))

        vectorized = DiracLengthMixtureAccumulator(NullAccumulator(), v=0)
        encoded = DiracLengthMixtureDistribution(NullDistribution(), p=0.5, v=0).dist_to_encoder().seq_encode([0])
        vectorized.seq_initialize(encoded, np.asarray([weight]), np.random.RandomState(7))

        np.testing.assert_allclose(scalar.comp_counts, vectorized.comp_counts)
        self.assertAlmostEqual(float(scalar.comp_counts.sum()), weight)

    def test_impossible_evidence_has_zero_posterior_and_fails_transactionally(self):
        model = DiracLengthMixtureDistribution(IntegerCategoricalDistribution(1, [1.0]), p=0.0, v=0)
        self.assertEqual(model.log_density(1), -np.inf)
        np.testing.assert_array_equal(model.posterior(1), [0.0, 0.0])

        accumulator = model.estimator().accumulator_factory().make()
        before = accumulator.comp_counts.copy()
        with self.assertRaises(ValueError):
            accumulator.update(1, 1.0, model)
        np.testing.assert_array_equal(accumulator.comp_counts, before)
        accumulator.update(1, 0.0, model)
        np.testing.assert_array_equal(accumulator.comp_counts, before)

        encoded = model.dist_to_encoder().seq_encode([1])
        with self.assertRaises(ValueError):
            accumulator.seq_update(encoded, np.ones(1), model)
        accumulator.seq_update(encoded, np.zeros(1), model)
        np.testing.assert_array_equal(accumulator.comp_counts, before)

    def test_probability_weights_and_estimator_statistics_are_validated(self):
        length = IntegerCategoricalDistribution(0, [1.0])
        for invalid in (-0.1, 1.1, np.nan, np.inf, True):
            with self.subTest(probability=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                DiracLengthMixtureDistribution(length, p=invalid)

        model = DiracLengthMixtureDistribution(length, p=0.5)
        encoded = model.dist_to_encoder().seq_encode([0])
        accumulator = model.estimator().accumulator_factory().make()
        for invalid in ([-1.0], [np.nan], [np.inf], [True]):
            with self.subTest(weight=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                accumulator.seq_update(encoded, invalid, model)

        estimator = DiracLengthMixtureEstimator(NullEstimator())
        for invalid in (
            (np.zeros(3), None),
            (np.asarray([-1.0, 1.0]), None),
            (np.asarray([np.nan, 1.0]), None),
        ):
            with self.subTest(statistics=repr(invalid[0])), self.assertRaises(ValueError):
                estimator.estimate(None, invalid)


if __name__ == "__main__":
    unittest.main()
