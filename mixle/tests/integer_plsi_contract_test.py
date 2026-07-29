"""Probability-law and schema contracts for integer PLSI."""

import itertools
import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.latent.integer_probabilistic_latent_semantic_indexing import (
    IntegerProbabilisticLatentSemanticIndexingDistribution,
    IntegerProbabilisticLatentSemanticIndexingEstimator,
)
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution


class IntegerPLSIContractTestCase(unittest.TestCase):
    def _dist(self):
        return IntegerProbabilisticLatentSemanticIndexingDistribution(
            [[0.5], [0.5]],
            [[1.0]],
            [1.0],
            len_dist=IntegerCategoricalDistribution(2, [1.0]),
        )

    def test_grouped_bag_scores_are_the_sampler_probability_law(self):
        dist = self._dist()
        bags = [[(0, 2)], [(0, 1), (1, 1)], [(1, 2)]]
        np.testing.assert_allclose([dist.density((0, bag)) for bag in bags], [0.25, 0.5, 0.25])
        self.assertAlmostEqual(sum(dist.density((0, bag)) for bag in bags), 1.0)

        encoded = dist.dist_to_encoder().seq_encode([(0, bag) for bag in bags])
        np.testing.assert_allclose(dist.seq_log_density(encoded), [dist.log_density((0, bag)) for bag in bags])
        np.testing.assert_allclose(
            NUMPY_ENGINE.to_numpy(dist.backend_seq_log_density(encoded, NUMPY_ENGINE)),
            dist.seq_log_density(encoded),
        )

        enumerated = list(itertools.islice(dist.enumerator(), 3))
        np.testing.assert_allclose([score for _, score in enumerated], np.log([0.5, 0.25, 0.25]))
        for value, score in enumerated:
            self.assertAlmostEqual(score, dist.log_density(value))

    def test_duplicate_entries_canonicalize_and_accumulate_consistently(self):
        dist = IntegerProbabilisticLatentSemanticIndexingDistribution(
            [[1.0]],
            [[1.0]],
            [1.0],
            len_dist=IntegerCategoricalDistribution(3, [1.0]),
        )
        duplicate = (0, [(0, 1), (0, 2)])
        canonical = (0, [(0, 3)])
        self.assertEqual(dist.log_density(duplicate), dist.log_density(canonical))

        estimator = dist.estimator()
        scalar = estimator.accumulator_factory().make()
        scalar.update(duplicate, 1.0, dist)
        vectorized = estimator.accumulator_factory().make()
        encoded = dist.dist_to_encoder().seq_encode([duplicate])
        vectorized.seq_update(encoded, np.ones(1), dist)
        for scalar_value, vector_value in zip(scalar.value()[:3], vectorized.value()[:3]):
            np.testing.assert_allclose(scalar_value, vector_value)
        self.assertEqual(scalar.word_count[0, 0], 3.0)

    def test_constructor_requires_matching_probability_simplexes(self):
        valid = ([[0.6], [0.4]], [[1.0]], [1.0])
        invalid = (
            ([[0.6], [0.5]], valid[1], valid[2]),
            ([[0.6], [-0.4]], valid[1], valid[2]),
            ([[np.nan], [np.nan]], valid[1], valid[2]),
            (valid[0], [[0.8]], valid[2]),
            (valid[0], [[0.5, 0.5]], valid[2]),
            (valid[0], valid[1], [0.5]),
        )
        for state_word, doc_state, doc_vec in invalid:
            with self.subTest(parameters=repr((state_word, doc_state, doc_vec))), self.assertRaises(ValueError):
                IntegerProbabilisticLatentSemanticIndexingDistribution(state_word, doc_state, doc_vec)

    def test_observation_schema_rejects_lossy_ids_and_invalid_counts(self):
        dist = self._dist()
        invalid = (
            (0.5, [(0, 2)]),
            (1, [(0, 2)]),
            (0, [(0.5, 2)]),
            (0, [(2, 2)]),
            (0, [(0, -1)]),
            (0, [(0, 1.5)]),
            (0, [(0, np.nan)]),
            (0, [(0, np.inf)]),
        )
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                dist.log_density(value)
            with self.subTest(encoded=repr(value)), self.assertRaises((TypeError, ValueError)):
                dist.dist_to_encoder().seq_encode([value])

    def test_empty_documents_have_defined_scalar_and_vector_component_scores(self):
        dist = IntegerProbabilisticLatentSemanticIndexingDistribution(
            [[0.7, 0.2], [0.3, 0.8]],
            [[0.4, 0.6]],
            [1.0],
            len_dist=IntegerCategoricalDistribution(0, [1.0]),
        )
        np.testing.assert_array_equal(dist.component_log_density((0, [])), np.zeros(2))
        encoded = dist.dist_to_encoder().seq_encode([(0, []), (0, [])])
        self.assertEqual(dist.seq_component_log_density(encoded).shape, (2, 2))
        np.testing.assert_array_equal(dist.seq_component_log_density(encoded), np.zeros((2, 2)))
        np.testing.assert_allclose(dist.seq_log_density(encoded), np.zeros(2))

    def test_zero_weight_impossible_rows_are_ignored_without_nan(self):
        dist = IntegerProbabilisticLatentSemanticIndexingDistribution(
            [[1.0], [0.0]],
            [[1.0]],
            [1.0],
            len_dist=IntegerCategoricalDistribution(1, [1.0]),
        )
        impossible = (0, [(1, 1)])
        estimator = dist.estimator()
        scalar = estimator.accumulator_factory().make()
        scalar.update(impossible, 0.0, dist)
        vectorized = estimator.accumulator_factory().make()
        encoded = dist.dist_to_encoder().seq_encode([impossible])
        vectorized.seq_update(encoded, np.zeros(1), dist)
        for value in (*scalar.value()[:3], *vectorized.value()[:3]):
            self.assertTrue(np.all(np.isfinite(value)))
            self.assertEqual(float(np.sum(value)), 0.0)
        with self.assertRaises(ValueError):
            scalar.update(impossible, 1.0, dist)
        with self.assertRaises(ValueError):
            vectorized.seq_update(encoded, np.ones(1), dist)

    def test_dimensions_and_estimator_statistics_have_exact_geometry(self):
        for invalid in (0, -1, 1.5, True, np.nan):
            with self.subTest(dimension=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                IntegerProbabilisticLatentSemanticIndexingEstimator(invalid, 1, 1)

        estimator = IntegerProbabilisticLatentSemanticIndexingEstimator(2, 1, 1)
        good = (np.zeros((1, 2)), np.zeros((1, 1)), np.zeros(1), None)
        estimator.estimate(None, good)
        for invalid in (
            (np.zeros((2, 1)), good[1], good[2], None),
            (good[0], np.zeros((1, 2)), good[2], None),
            (good[0], good[1], np.zeros(2), None),
            (np.asarray([[np.nan, 0.0]]), good[1], good[2], None),
            (np.asarray([[-1.0, 0.0]]), good[1], good[2], None),
        ):
            with self.subTest(statistics=repr(invalid[:3])), self.assertRaises(ValueError):
                estimator.estimate(None, invalid)


if __name__ == "__main__":
    unittest.main()
