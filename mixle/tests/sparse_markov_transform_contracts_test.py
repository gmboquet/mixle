"""Contract tests for sparse Markov-association count-bag models."""

import math
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from mixle.engines import NUMPY_ENGINE
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.null_dist import NullDistribution
from mixle.stats.compute.backend import backend_seq_log_density
from mixle.stats.sequences.sparse_markov_transform import (
    SparseMarkovAssociationAccumulator,
    SparseMarkovAssociationDistribution,
    SparseMarkovAssociationEstimator,
)
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _length_distribution(source=1, output=2):
    return CompositeDistribution(
        (
            CategoricalDistribution({source: 1.0}),
            CategoricalDistribution({output: 1.0}),
        )
    )


def _distribution(*, alpha=0.0, length=False, low_memory=False, init=None):
    return SparseMarkovAssociationDistribution(
        np.asarray([0.5, 0.5] if init is None else init),
        np.full((2, 2), 0.5),
        alpha=alpha,
        len_dist=_length_distribution() if length else NullDistribution(),
        low_memory=low_memory,
    )


class SparseMarkovProbabilityContractTest(unittest.TestCase):
    def test_count_bag_probabilities_normalize_at_fixed_lengths(self):
        dist = _distribution()
        source_bags = ([(0, 1)], [(1, 1)])
        output_bags = ([(0, 2)], [(0, 1), (1, 1)], [(1, 2)])
        total = sum(
            math.exp(dist.log_density((source, output)))
            for source in source_bags
            for output in output_bags
        )
        self.assertAlmostEqual(total, 1.0)

    def test_alpha_smoothed_source_law_is_shared_by_scoring_and_sampling(self):
        dist = SparseMarkovAssociationDistribution(
            [1.0, 0.0],
            np.eye(2),
            alpha=0.5,
            len_dist=_length_distribution(source=1, output=0),
        )
        self.assertAlmostEqual(math.exp(dist.log_density(([(1, 1)], []))), 0.25)
        draws = dist.sampler(seed=7).sample(300)
        state_one = sum(value[0][0][0] == 1 for value in draws)
        self.assertGreater(state_one, 40)
        self.assertLess(state_one, 110)

    def test_empty_parent_nonempty_output_is_consistently_impossible(self):
        data = [([], [(0, 1)])]
        for low_memory in (False, True):
            dist = _distribution(low_memory=low_memory)
            encoded = dist.dist_to_encoder().seq_encode(data)
            with self.subTest(low_memory=low_memory):
                self.assertEqual(dist.log_density(data[0]), -np.inf)
                self.assertEqual(dist.seq_log_density(encoded)[0], -np.inf)
                backend = backend_seq_log_density(dist, encoded, NUMPY_ENGINE)
                self.assertEqual(float(NUMPY_ENGINE.to_numpy(backend)[0]), -np.inf)

    def test_scalar_and_both_encoded_layouts_share_bag_semantics(self):
        data = [
            ([(1, 1), (0, 1), (1, 0)], [(0, 1), (1, 1)]),
            ([(0, 2)], [(1, 2)]),
        ]
        for low_memory in (False, True):
            dist = _distribution(alpha=0.2, low_memory=low_memory)
            encoded = dist.dist_to_encoder().seq_encode(data)
            expected = np.asarray([dist.log_density(value) for value in data])
            with self.subTest(low_memory=low_memory):
                np.testing.assert_allclose(dist.seq_log_density(encoded), expected)

    def test_constructor_and_observation_boundaries_reject_invalid_values(self):
        invalid_parameters = (
            ([0.2, 0.2], np.full((2, 2), 0.5), 0.0),
            ([0.5, 0.5], np.full((3, 2), 0.5), 0.0),
            ([0.5, 0.5], np.asarray([[0.7, 0.4]] * 2), 0.0),
            ([0.5, 0.5], np.asarray([[0.5, -0.5]] * 2), 0.0),
            ([0.5, 0.5], np.full((2, 2), 0.5), 1.1),
        )
        for initial, conditional, alpha in invalid_parameters:
            with self.subTest(initial=initial, shape=np.shape(conditional), alpha=alpha):
                with self.assertRaises((TypeError, ValueError)):
                    SparseMarkovAssociationDistribution(initial, conditional, alpha=alpha)

        dist = _distribution()
        invalid_observations = (
            ([(0, 1)],),
            ([(2, 1)], []),
            ([(0.5, 1)], []),
            ([(0, 1.5)], []),
            ([(0, -1)], []),
        )
        for observation in invalid_observations:
            with self.subTest(observation=observation):
                with self.assertRaises((TypeError, ValueError)):
                    dist.log_density(observation)


class SparseMarkovSamplingContractTest(unittest.TestCase):
    class _Draw:
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value

    def test_conditional_form_rejects_sampling(self):
        with self.assertRaises(TypeError):
            _distribution().sampler()

    def test_sampled_lengths_are_validated_before_indexing(self):
        sampler = _distribution(length=True).sampler(seed=1)
        for draw in ((1,), (1.5, 1), (-1, 1), (0, 1)):
            sampler.size_sampler = self._Draw(draw)
            with self.subTest(draw=draw):
                with self.assertRaises((TypeError, ValueError)):
                    sampler.sample()

    def test_empty_parent_and_output_can_be_sampled(self):
        sampler = _distribution(length=True).sampler(seed=1)
        sampler.size_sampler = self._Draw((0, 0))
        self.assertEqual(sampler.sample(), ([], []))


class SparseMarkovEstimationContractTest(unittest.TestCase):
    def test_empty_fit_and_pseudocount_both_produce_simplexes(self):
        empty = (np.zeros(2), None, None)
        unregularized = SparseMarkovAssociationEstimator(2).estimate(0.0, empty)
        np.testing.assert_allclose(unregularized.init_prob_vec, [0.5, 0.5])
        np.testing.assert_allclose(unregularized.cond_prob_mat.toarray(), np.full((2, 2), 0.5))

        model = SparseMarkovAssociationEstimator(2, pseudo_count=1.0).estimate(0.0, empty)
        np.testing.assert_allclose(model.init_prob_vec, [0.5, 0.5])
        np.testing.assert_allclose(model.cond_prob_mat.toarray(), np.full((2, 2), 0.5))

    def test_unobserved_rows_use_uniform_backoff(self):
        counts = csr_matrix(([3.0], ([0], [1])), shape=(2, 2))
        model = SparseMarkovAssociationEstimator(2).estimate(
            None,
            (np.asarray([2.0, 1.0]), counts, None),
        )
        np.testing.assert_allclose(model.cond_prob_mat.toarray(), [[0.0, 1.0], [0.5, 0.5]])

    def test_fully_smoothed_model_refits_unidentified_parameters_with_backoff(self):
        dist = _distribution(alpha=1.0)
        accumulator = SparseMarkovAssociationAccumulator(2)
        accumulator.update(([(0, 2)], [(1, 1)]), 1.0, dist)
        model = SparseMarkovAssociationEstimator(2, alpha=1.0).estimate(1.0, accumulator.value())
        np.testing.assert_allclose(model.init_prob_vec, [0.5, 0.5])
        np.testing.assert_allclose(model.cond_prob_mat.toarray(), np.full((2, 2), 0.5))

    def test_estimator_rejects_malformed_statistics(self):
        estimator = SparseMarkovAssociationEstimator(2)
        invalid = (
            (np.zeros(3), None, None),
            (np.asarray([1.0, -1.0]), None, None),
            (np.ones(2), csr_matrix((3, 2)), None),
            (np.ones(2), csr_matrix(([-1.0], ([0], [0])), shape=(2, 2)), None),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    estimator.estimate(None, value)


class SparseMarkovAccumulatorContractTest(unittest.TestCase):
    def test_duplicate_labels_match_canonical_counts(self):
        dist = _distribution(alpha=0.2)
        duplicate = ([(0, 1), (0, 2)], [(1, 1), (1, 1)])
        canonical = ([(0, 3)], [(1, 2)])
        first = SparseMarkovAssociationAccumulator(2)
        second = SparseMarkovAssociationAccumulator(2)
        first.update(duplicate, 1.0, dist)
        second.update(canonical, 1.0, dist)
        np.testing.assert_allclose(first.init_count, second.init_count)
        np.testing.assert_allclose(first.trans_count.toarray(), second.trans_count.toarray())

    def test_sequence_updates_reject_weight_truncation_and_impossible_evidence(self):
        dist = _distribution(low_memory=False)
        data = [([(0, 1)], [(1, 1)]), ([(1, 1)], [(0, 1)])]
        encoded = dist.dist_to_encoder().seq_encode(data)
        with self.assertRaises(ValueError):
            SparseMarkovAssociationAccumulator(2).seq_update(encoded, np.ones(1), dist)
        with self.assertRaises(ValueError):
            SparseMarkovAssociationAccumulator(2).update(([], [(0, 1)]), 1.0, dist)

    def test_statistic_state_is_copied_on_ingress_and_egress(self):
        accumulator = SparseMarkovAssociationAccumulator(2)
        accumulator.initialize(([(0, 1)], [(1, 1)]), 1.0, np.random.RandomState(1))
        value = accumulator.value()
        value[0][0] += 100.0
        value[1].data[:] += 100.0
        self.assertLess(accumulator.init_count[0], 100.0)
        self.assertLess(float(accumulator.trans_count.max()), 100.0)

        restored = SparseMarkovAssociationAccumulator(2).from_value(accumulator.value())
        original_init = accumulator.init_count.copy()
        original_trans = accumulator.trans_count.toarray().copy()
        restored.init_count += 10.0
        restored.trans_count.data[:] += 10.0
        np.testing.assert_allclose(accumulator.init_count, original_init)
        np.testing.assert_allclose(accumulator.trans_count.toarray(), original_trans)


if __name__ == "__main__":
    unittest.main()
