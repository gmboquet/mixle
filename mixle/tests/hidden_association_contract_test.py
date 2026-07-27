"""Grouped-output probability and schema contracts for hidden association models."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.combinator.conditional import ConditionalDistribution
from mixle.stats.latent.hidden_association import HiddenAssociationDistribution
from mixle.stats.latent.integer_hidden_association import IntegerHiddenAssociationDistribution
from mixle.stats.multivariate.categorical_multinomial import MultinomialDistribution
from mixle.stats.multivariate.integer_multinomial import IntegerMultinomialDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution


class HiddenAssociationContractTestCase(unittest.TestCase):
    def _generic(self):
        given = MultinomialDistribution(
            CategoricalDistribution({"a": 1.0}),
            len_dist=CategoricalDistribution({1: 1.0}),
        )
        return HiddenAssociationDistribution(
            ConditionalDistribution({"a": CategoricalDistribution({"x": 0.5, "y": 0.5})}),
            given_dist=given,
            len_dist=CategoricalDistribution({2: 1.0}),
        )

    def _integer(self, use_numba=False):
        given = IntegerMultinomialDistribution(
            0,
            [1.0],
            len_dist=IntegerCategoricalDistribution(1, [1.0]),
        )
        return IntegerHiddenAssociationDistribution(
            [[0.5, 0.5]],
            [[1.0]],
            prev_dist=given,
            len_dist=IntegerCategoricalDistribution(2, [1.0]),
            use_numba=use_numba,
        )

    def test_generic_grouped_outputs_form_the_sampler_law(self):
        dist = self._generic()
        observations = (
            ([("a", 1)], [("x", 2)]),
            ([("a", 1)], [("x", 1), ("y", 1)]),
            ([("a", 1)], [("y", 2)]),
        )
        probabilities = np.exp([dist.log_density(value) for value in observations])
        np.testing.assert_allclose(probabilities, [0.25, 0.5, 0.25])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        encoded = dist.dist_to_encoder().seq_encode(observations)
        np.testing.assert_allclose(dist.seq_log_density(encoded), np.log(probabilities))
        np.testing.assert_allclose(
            NUMPY_ENGINE.to_numpy(dist.backend_seq_log_density(encoded, NUMPY_ENGINE)),
            np.log(probabilities),
        )

    def test_integer_grouped_outputs_match_in_scalar_and_both_encoded_paths(self):
        observations = (
            ([(0, 1)], [(0, 2)]),
            ([(0, 1)], [(0, 1), (1, 1)]),
            ([(0, 1)], [(1, 2)]),
        )
        for use_numba in (False, True):
            dist = self._integer(use_numba)
            probabilities = np.exp([dist.log_density(value) for value in observations])
            np.testing.assert_allclose(probabilities, [0.25, 0.5, 0.25])
            encoded = dist.dist_to_encoder().seq_encode(observations)
            np.testing.assert_allclose(dist.seq_log_density(encoded), np.log(probabilities))
            if not use_numba:
                np.testing.assert_allclose(
                    NUMPY_ENGINE.to_numpy(dist.backend_seq_log_density(encoded, NUMPY_ENGINE)),
                    np.log(probabilities),
                )

    def test_duplicate_entries_use_one_canonical_count_vector(self):
        generic = self._generic()
        self.assertEqual(
            generic.log_density(([("a", 1)], [("x", 1), ("x", 1)])),
            generic.log_density(([("a", 1)], [("x", 2)])),
        )
        integer = self._integer()
        self.assertEqual(
            integer.log_density(([(0, 1)], [(0, 1), (0, 1)])),
            integer.log_density(([(0, 1)], [(0, 2)])),
        )

    def test_grouped_bag_schemas_reject_invalid_counts_ids_and_empty_conditioning(self):
        generic = self._generic()
        for invalid in (
            ([("a", 1)], [("x", -1)]),
            ([("a", 1)], [("x", 0.5)]),
            ([("a", 1)], [("x", np.nan)]),
            ([], [("x", 1)]),
        ):
            with self.subTest(generic=invalid), self.assertRaises((TypeError, ValueError)):
                generic.log_density(invalid)
            with self.subTest(generic_encoder=invalid), self.assertRaises((TypeError, ValueError)):
                generic.dist_to_encoder().seq_encode([invalid])

        integer = self._integer()
        for invalid in (
            ([(0.5, 1)], [(0, 1)]),
            ([(1, 1)], [(0, 1)]),
            ([(0, 1)], [(0.5, 1)]),
            ([(0, 1)], [(2, 1)]),
            ([(0, 1)], [(0, -1)]),
            ([], [(0, 1)]),
        ):
            with self.subTest(integer=invalid), self.assertRaises((TypeError, ValueError)):
                integer.log_density(invalid)
            with self.subTest(integer_encoder=invalid), self.assertRaises((TypeError, ValueError)):
                integer.dist_to_encoder().seq_encode([invalid])

    def test_integer_constructor_requires_matching_simplexes_and_alpha(self):
        invalid = (
            ([[0.5, 0.6]], [[1.0]], 0.0),
            ([[0.5, -0.5]], [[1.0]], 0.0),
            ([[0.5, 0.5]], [[0.5]], 0.0),
            ([[0.5, 0.5]], [[0.5, 0.5]], 0.0),
            ([[0.5, 0.5]], [[1.0]], -0.1),
            ([[0.5, 0.5]], [[1.0]], 1.1),
            ([[0.5, 0.5]], [[1.0]], np.nan),
        )
        for state, conditional, alpha in invalid:
            with self.subTest(parameters=(state, conditional, alpha)), self.assertRaises(ValueError):
                IntegerHiddenAssociationDistribution(state, conditional, alpha=alpha)

    def test_impossible_positive_weight_evidence_is_transactional(self):
        dist = IntegerHiddenAssociationDistribution(
            [[1.0, 0.0]],
            [[1.0]],
            alpha=0.0,
            use_numba=False,
        )
        impossible = ([(0, 1)], [(1, 1)])
        self.assertEqual(dist.log_density(impossible), -np.inf)
        accumulator = dist.estimator().accumulator_factory().make()
        before = tuple(value.copy() for value in accumulator.value()[:3])
        with self.assertRaises(ValueError):
            accumulator.update(impossible, 1.0, dist)
        for actual, expected in zip(accumulator.value()[:3], before):
            np.testing.assert_array_equal(actual, expected)
        accumulator.update(impossible, 0.0, dist)
        for actual, expected in zip(accumulator.value()[:3], before):
            np.testing.assert_array_equal(actual, expected)

        generic = HiddenAssociationDistribution(
            ConditionalDistribution({"a": CategoricalDistribution({"x": 1.0, "y": 0.0})}),
            len_dist=CategoricalDistribution({1: 1.0}),
        )
        generic_impossible = ([("a", 1)], [("y", 1)])
        self.assertEqual(generic.log_density(generic_impossible), -np.inf)
        generic_accumulator = generic.estimator().accumulator_factory().make()
        before_generic = repr(generic_accumulator.value())
        with self.assertRaises(ValueError):
            generic_accumulator.update(generic_impossible, 1.0, generic)
        self.assertEqual(repr(generic_accumulator.value()), before_generic)


if __name__ == "__main__":
    unittest.main()
