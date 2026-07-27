"""Fixed-support, length, sampling, and statistic contracts for integer Markov chains."""

import copy
import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    IntegerCategoricalDistribution,
    IntegerMarkovChainDistribution,
    IntegerMarkovChainEstimator,
    IntegerMarkovChainStatistics,
    NonGenerativeIntegerMarkovChainError,
    SequenceDistribution,
)
from mixle.stats.combinator.null_dist import NullAccumulatorFactory, NullDistribution
from mixle.stats.compute.pdist import EnumerationError, ParameterEstimator


def _proper_chain(*, lag=2):
    init = SequenceDistribution(
        IntegerCategoricalDistribution(0, [0.5, 0.5]),
        len_dist=CategoricalDistribution({lag: 1.0}),
    )
    return IntegerMarkovChainDistribution(
        2,
        np.full((2**lag, 2), 0.5),
        lag=lag,
        init_dist=init,
        len_dist=CategoricalDistribution({lag: 0.4, lag + 1: 0.6}),
    )


class _RecordingEstimator(ParameterEstimator):
    def __init__(self):
        self.calls = []

    def accumulator_factory(self):
        return NullAccumulatorFactory()

    def estimate(self, nobs, suff_stat):
        self.calls.append((nobs, suff_stat))
        return NullDistribution()


class IntegerMarkovChainContractsTest(unittest.TestCase):
    def test_constructor_owns_one_finite_row_simplex_matrix(self):
        matrix = np.full((4, 2), 0.5)
        dist = IntegerMarkovChainDistribution(2, matrix, lag=2)
        matrix[0, 0] = 1.0
        self.assertEqual(dist.cond_dist[0, 0], 0.5)
        self.assertFalse(dist.cond_dist.flags.writeable)
        copied = copy.deepcopy(dist)
        np.testing.assert_array_equal(copied.cond_dist, dist.cond_dist)

        invalid = (
            (0, np.ones((1, 1)), 1),
            (2, np.ones((2, 2)), 0),
            (2, np.ones((2, 2)), 2),
            (2, [[0.7, 0.2], [0.5, 0.5]], 1),
            (2, [[1.1, -0.1], [0.5, 0.5]], 1),
            (2, [[np.nan, 0.0], [0.5, 0.5]], 1),
        )
        for num_values, conditional, lag in invalid:
            with self.assertRaises((TypeError, ValueError)):
                IntegerMarkovChainDistribution(num_values, conditional, lag=lag)

    def test_short_lengths_are_rejected_at_model_and_observation_boundaries(self):
        init = SequenceDistribution(
            IntegerCategoricalDistribution(0, [0.5, 0.5]),
            len_dist=CategoricalDistribution({2: 1.0}),
        )
        with self.assertRaisesRegex(ValueError, "only lengths >= lag"):
            IntegerMarkovChainDistribution(
                2,
                np.full((4, 2), 0.5),
                lag=2,
                init_dist=init,
                len_dist=CategoricalDistribution({1: 1.0}),
            )
        dist = _proper_chain()
        with self.assertRaisesRegex(ValueError, "at least lag"):
            dist.log_density([0])
        with self.assertRaisesRegex(ValueError, "at least lag"):
            dist.dist_to_encoder().seq_encode([[0]])

    def test_scalar_batch_and_actual_length_contracts_match_without_transitions(self):
        dist = _proper_chain()
        data = [[0, 1], [1, 0], [0, 1, 1]]
        encoded = dist.dist_to_encoder().seq_encode(data)
        self.assertEqual(dist.dist_to_encoder().row_count(encoded), 3)
        np.testing.assert_allclose(
            dist.seq_log_density(encoded),
            [dist.log_density(value) for value in data],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        empty = dist.dist_to_encoder().seq_encode([])
        self.assertEqual(dist.dist_to_encoder().row_count(empty), 0)
        self.assertEqual(dist.seq_log_density(empty).shape, (0,))

    def test_sampler_returns_exact_declared_lengths_and_rejects_factors(self):
        dist = _proper_chain()
        samples = dist.sampler(seed=4).sample(size=50)
        self.assertTrue(all(len(value) in (2, 3) for value in samples))
        factor = IntegerMarkovChainDistribution(2, np.full((2, 2), 0.5))
        with self.assertRaises(NonGenerativeIntegerMarkovChainError):
            factor.sampler()
        with self.assertRaises(EnumerationError):
            factor.enumerator()

    def test_encoder_and_estimator_reject_support_expansion(self):
        dist = _proper_chain()
        with self.assertRaisesRegex(ValueError, "declared support"):
            dist.dist_to_encoder().seq_encode([[0, 2]])
        bad = IntegerMarkovChainStatistics(
            1,
            2,
            2,
            (((0, 2), 1, 1.0),),
            1.0,
            None,
            1.0,
            None,
        )
        with self.assertRaises((TypeError, ValueError)):
            IntegerMarkovChainEstimator(2, lag=2).estimate(1.0, bad)

    def test_statistics_are_immutable_canonical_and_carry_child_counts(self):
        dist = _proper_chain()
        estimator = dist.estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update([0, 1], 2.0, dist)
        accumulator.update([1, 0, 1], 3.0, dist)
        statistics = accumulator.value()
        self.assertIsInstance(statistics, IntegerMarkovChainStatistics)
        self.assertEqual(statistics.initial_nobs, 5.0)
        self.assertEqual(statistics.length_nobs, 5.0)
        restored = estimator.accumulator_factory().make().from_value(statistics)
        restored_value = restored.value()
        self.assertEqual(restored_value.transition_counts, statistics.transition_counts)
        self.assertEqual(restored_value.length, statistics.length)
        np.testing.assert_array_equal(
            restored_value.initial.elements[1],
            statistics.initial.elements[1],
        )
        fitted = estimator.estimate(5.0, statistics)
        self.assertAlmostEqual(fitted.len_dist.density(2), 0.4)
        self.assertAlmostEqual(fitted.len_dist.density(3), 0.6)
        restored.scale(0.5)
        self.assertEqual(restored.value().initial_nobs, 2.5)
        self.assertEqual(restored.value().length_nobs, 2.5)

    def test_child_estimators_receive_their_effective_counts(self):
        initial = _RecordingEstimator()
        length = _RecordingEstimator()
        estimator = IntegerMarkovChainEstimator(
            2,
            lag=1,
            init_estimator=initial,
            len_estimator=length,
            pseudo_count=0.5,
        )
        statistics = IntegerMarkovChainStatistics(
            1,
            2,
            1,
            (((0,), 1, 2.0),),
            4.0,
            None,
            4.0,
            None,
        )
        fitted = estimator.estimate(4.0, statistics)
        self.assertEqual(initial.calls, [(4.0, None)])
        self.assertEqual(length.calls, [(4.0, None)])
        self.assertEqual(fitted.num_values, 2)

    def test_initial_child_must_prove_exact_integer_prefix_support(self):
        invalid = SequenceDistribution(
            CategoricalDistribution({"bad": 1.0}),
            len_dist=CategoricalDistribution({2: 1.0}),
        )
        with self.assertRaisesRegex(TypeError, "bounded integer support"):
            IntegerMarkovChainDistribution(
                2,
                np.full((4, 2), 0.5),
                lag=2,
                init_dist=invalid,
                len_dist=CategoricalDistribution({2: 1.0}),
            )
        with self.assertRaises(EnumerationError):
            _proper_chain().enumerator()._valid_prefix([1.9, 0])


if __name__ == "__main__":
    unittest.main()
