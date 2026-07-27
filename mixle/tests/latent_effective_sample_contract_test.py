"""Cross-family contract tests for latent effective-sample accounting."""

import unittest

import numpy as np

from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_observation_weights,
    validated_weighted_responsibilities,
)
from mixle.stats.latent.semi_supervised_hidden_markov_model import SemiSupervisedHiddenMarkovEstimator
from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureEstimatorAccumulator
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator


class _RecordingEstimator:
    def __init__(self, result):
        self.result = result
        self.nobs = []

    def estimate(self, nobs, suff_stat):
        self.nobs.append(nobs)
        return self.result


class LatentEffectiveSampleContractTest(unittest.TestCase):
    def test_observation_weights_are_exact_finite_nonnegative_and_owned(self):
        for value in (True, -1.0, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    validated_observation_weight(value)

        source = np.array([1.0, 2.0])
        result = validated_observation_weights(source, 2)
        source[0] = 99.0
        np.testing.assert_array_equal(result, [1.0, 2.0])
        with self.assertRaises(ValueError):
            validated_observation_weights([1.0], 2)

    def test_counts_and_responsibilities_cannot_create_mass(self):
        with self.assertRaises(ValueError):
            validated_count_array([1.0, -1.0], (2,))
        with self.assertRaises(ValueError):
            validated_weighted_responsibilities(
                [[0.75, 0.5]],
                [1.0],
                2,
                label="test responsibilities",
            )
        assigned = validated_weighted_responsibilities(
            [[0.25, 0.25]],
            [1.0],
            2,
            label="test responsibilities",
            allow_unassigned=True,
        )
        np.testing.assert_array_equal(assigned, [[0.25, 0.25]])

    def test_declared_and_assigned_mass_produce_an_auditable_receipt(self):
        receipt = validate_effective_sample_mass(
            3.0,
            2.0,
            label="test effective sample",
            allow_unassigned=True,
        )
        self.assertEqual(receipt.contract, "mixle.latent_effective_sample/v1")
        self.assertEqual(receipt.unassigned_mass, 1.0)
        with self.assertRaises(ValueError):
            validate_effective_sample_mass(1.0, 2.0, label="test effective sample")

    def test_semi_supervised_initialization_conserves_nonunit_outer_weight(self):
        accumulator = SemiSupervisedMixtureEstimatorAccumulator(
            [CategoricalEstimator().accumulator_factory().make() for _ in range(3)]
        )
        accumulator.initialize(("x", None), 7.0, np.random.RandomState(3))
        self.assertAlmostEqual(float(accumulator.comp_counts.sum()), 7.0)

    def test_hmm_emission_estimators_receive_posterior_occupancy(self):
        first = _RecordingEstimator(CategoricalDistribution({"x": 1.0}))
        second = _RecordingEstimator(CategoricalDistribution({"x": 1.0}))
        estimator = SemiSupervisedHiddenMarkovEstimator([first, second])
        estimator.estimate(
            1.0,
            (
                np.array([[0.5, 0.0], [0.0, 0.5]]),
                np.array([1.25, 0.75]),
                ("first", "second"),
                None,
            ),
        )
        self.assertEqual(first.nobs, [1.25])
        self.assertEqual(second.nobs, [0.75])

    def test_tree_length_estimator_receives_node_mass(self):
        topics = [
            _RecordingEstimator(CategoricalDistribution({"x": 1.0})),
            _RecordingEstimator(CategoricalDistribution({"x": 1.0})),
        ]
        length = _RecordingEstimator(CategoricalDistribution({0: 1.0}))
        estimator = TreeHiddenMarkovEstimator(topics, len_estimator=length, use_numba=False)
        estimator.estimate(
            1.0,
            (
                2,
                np.array([1.0, 0.0]),
                np.array([2.0, 1.0]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                ("first", "second"),
                "length",
            ),
        )
        self.assertEqual(length.nobs, [3.0])
        self.assertEqual(topics[0].nobs, [2.0])
        self.assertEqual(topics[1].nobs, [1.0])


if __name__ == "__main__":
    unittest.main()
