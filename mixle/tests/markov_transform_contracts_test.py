"""Contract tests for the dense Markov-transform distribution."""

import math
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.sequences.markov_transform import (
    MarkovTransformAccumulator,
    MarkovTransformDistribution,
    MarkovTransformEstimator,
)
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _length_distribution():
    return CompositeDistribution(
        (
            CategoricalDistribution({1: 1.0}),
            CategoricalDistribution({1: 1.0}),
            CategoricalDistribution({2: 1.0}),
        )
    )


def _distribution(*, length=True):
    return MarkovTransformDistribution(
        np.asarray([0.5, 0.5]),
        np.full((4, 2), 0.5),
        len_dist=_length_distribution() if length else None,
    )


class MarkovTransformProbabilityContractTest(unittest.TestCase):
    def test_count_bag_probability_includes_all_multinomial_coefficients(self):
        dist = _distribution(length=False)
        observation = (
            [(0, 1), (1, 1)],
            [(0, 1)],
            [(0, 1), (1, 1)],
        )
        self.assertAlmostEqual(math.exp(dist.log_density(observation)), 0.125)

    def test_fixed_length_bag_probabilities_normalize(self):
        dist = _distribution(length=False)
        parent_bags = ([(0, 1)], [(1, 1)])
        target_bags = ([(0, 2)], [(0, 1), (1, 1)], [(1, 2)])
        total = sum(
            math.exp(dist.log_density((left, right, target)))
            for left in parent_bags
            for right in parent_bags
            for target in target_bags
        )
        self.assertAlmostEqual(total, 1.0)

    def test_scalar_encoded_and_backend_paths_share_bag_semantics(self):
        dist = _distribution()
        data = [
            ([(1, 1), (0, 1), (1, 0)], [(0, 1)], [(1, 1), (0, 1)]),
            ([(0, 1)], [(1, 1)], [(0, 2)]),
        ]
        encoded = dist.dist_to_encoder().seq_encode(data)
        expected = np.asarray([dist.log_density(value) for value in data])
        np.testing.assert_allclose(dist.seq_log_density(encoded), expected)

    def test_legacy_encoding_delegates_to_the_canonical_child_encoder(self):
        dist = _distribution(length=True)
        data = [
            (
                [(0, 1)],
                [(1, 1)],
                [(0, 1), (1, 1)],
            )
        ]
        legacy = dist.seq_encode(data)
        canonical = dist.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(dist.seq_log_density(legacy), dist.seq_log_density(canonical))

    def test_constructor_rejects_nonprobability_parameters(self):
        bad_cases = (
            ([0.2, 0.2], np.full((4, 2), 0.5), 0.0),
            ([0.5, 0.5], np.full((3, 2), 0.5), 0.0),
            ([0.5, 0.5], np.asarray([[0.7, 0.4]] * 4), 0.0),
            ([0.5, 0.5], np.asarray([[0.5, -0.5]] * 4), 0.0),
            ([0.5, 0.5], np.full((4, 2), 0.5), 1.1),
        )
        for init, conditional, alpha in bad_cases:
            with self.subTest(init=init, shape=np.shape(conditional), alpha=alpha):
                with self.assertRaises((TypeError, ValueError)):
                    MarkovTransformDistribution(init, conditional, alpha=alpha)

    def test_observations_require_three_valid_integral_bags(self):
        dist = _distribution(length=False)
        invalid = (
            ([(0, 1)], [(0, 1)]),
            ([(2, 1)], [(0, 1)], []),
            ([(0.5, 1)], [(0, 1)], []),
            ([(0, 1.5)], [(0, 1)], []),
            ([(0, -1)], [(0, 1)], []),
            ([], [(0, 1)], [(0, 1)]),
        )
        for observation in invalid:
            with self.subTest(observation=observation):
                with self.assertRaises((TypeError, ValueError)):
                    dist.log_density(observation)


class MarkovTransformSamplingContractTest(unittest.TestCase):
    class _Draw:
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value

    def test_sampling_requires_length_distribution(self):
        with self.assertRaises(ValueError):
            _distribution(length=False).sampler()

    def test_sampling_validates_length_draws_and_impossible_parents(self):
        sampler = _distribution().sampler(seed=1)
        for draw in ((1, 1), (1.5, 1, 2), (-1, 1, 2), (0, 1, 1)):
            sampler.size_sampler = self._Draw(draw)
            with self.subTest(draw=draw):
                with self.assertRaises((TypeError, ValueError)):
                    sampler.sample()

    def test_sampling_accepts_empty_output_with_empty_parent(self):
        sampler = _distribution().sampler(seed=1)
        sampler.size_sampler = self._Draw((0, 1, 0))
        value = sampler.sample()
        self.assertEqual(value[0], [])
        self.assertEqual(value[2], [])


class MarkovTransformEstimationContractTest(unittest.TestCase):
    def test_empty_fit_requires_prior_and_pseudocount_produces_simplexes(self):
        empty = (np.zeros(2), csc_matrix((4, 2)), None)
        with self.assertRaises(ValueError):
            MarkovTransformEstimator(2).estimate(0.0, empty)

        model = MarkovTransformEstimator(2, pseudo_count=1.0).estimate(0.0, empty)
        np.testing.assert_allclose(model.init_prob_vec, [0.5, 0.5])
        np.testing.assert_allclose(model.cond_prob_mat.toarray(), np.full((4, 2), 0.5))

    def test_unobserved_transition_rows_use_uniform_backoff(self):
        counts = csc_matrix(([3.0], ([0], [1])), shape=(4, 2))
        model = MarkovTransformEstimator(2).estimate(None, (np.asarray([2.0, 1.0]), counts, None))
        np.testing.assert_allclose(model.cond_prob_mat.toarray()[0], [0.0, 1.0])
        np.testing.assert_allclose(model.cond_prob_mat.toarray()[1:], np.full((3, 2), 0.5))

    def test_estimator_rejects_malformed_or_negative_statistics(self):
        estimator = MarkovTransformEstimator(2)
        invalid = (
            (np.zeros(3), csc_matrix((4, 2)), None),
            (np.asarray([1.0, -1.0]), csc_matrix((4, 2)), None),
            (np.ones(2), csc_matrix((3, 2)), None),
            (np.ones(2), csc_matrix(([-1.0], ([0], [0])), shape=(4, 2)), None),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    estimator.estimate(None, value)


class MarkovTransformAccumulatorContractTest(unittest.TestCase):
    def test_duplicate_labels_match_canonical_counts(self):
        dist = _distribution(length=False)
        duplicate = ([(0, 1), (0, 2)], [(1, 1), (1, 1)], [(0, 1), (0, 1)])
        canonical = ([(0, 3)], [(1, 2)], [(0, 2)])
        first = MarkovTransformAccumulator(2)
        second = MarkovTransformAccumulator(2)
        first.update(duplicate, 1.0, dist)
        second.update(canonical, 1.0, dist)
        np.testing.assert_allclose(first.init_count, second.init_count)
        np.testing.assert_allclose(first.trans_count.toarray(), second.trans_count.toarray())

    def test_sequence_updates_reject_weight_truncation(self):
        dist = _distribution(length=False)
        data = [
            ([(0, 1)], [(1, 1)], [(0, 1)]),
            ([(1, 1)], [(0, 1)], [(1, 1)]),
        ]
        encoded = dist.dist_to_encoder().seq_encode(data)
        with self.assertRaises(ValueError):
            MarkovTransformAccumulator(2).seq_update(encoded, np.ones(1), dist)

    def test_value_and_from_value_do_not_alias_arrays(self):
        accumulator = MarkovTransformAccumulator(2)
        accumulator.initialize(([(0, 1)], [(1, 1)], [(0, 1)]), 1.0, np.random.RandomState(1))
        value = accumulator.value()
        value[0][0] += 100.0
        value[1].data[:] += 100.0
        self.assertLess(accumulator.init_count[0], 100.0)
        self.assertLess(float(accumulator.trans_count.max()), 100.0)

        restored = MarkovTransformAccumulator(2).from_value(accumulator.value())
        original_init = accumulator.init_count.copy()
        original_trans = accumulator.trans_count.toarray().copy()
        restored.init_count += 10.0
        restored.trans_count.data[:] += 10.0
        np.testing.assert_allclose(accumulator.init_count, original_init)
        np.testing.assert_allclose(accumulator.trans_count.toarray(), original_trans)


if __name__ == "__main__":
    unittest.main()
