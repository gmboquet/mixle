import copy
import unittest

import numpy as np

from mixle.stats import (
    EnumerationError,
    IntegerChowLiuTreeDistribution,
    IntegerChowLiuTreeEstimator,
)


def _binary_tree(**kwargs):
    return IntegerChowLiuTreeDistribution(
        [None, 0],
        [
            np.log([0.6, 0.4]),
            np.log([[0.7, 0.3], [0.2, 0.8]]),
        ],
        **kwargs,
    )


class IntegerChowLiuTreeTestCase(unittest.TestCase):
    def test_reachable_zero_mass_conditional_row_is_rejected(self):
        root = np.log([0.5, 0.5])
        conditional = np.asarray(
            [
                np.log([0.5, 0.5]),
                [-np.inf, -np.inf],
            ]
        )
        with self.assertRaisesRegex(ValueError, "reachable parent state"):
            IntegerChowLiuTreeDistribution(
                [None, 0],
                [root, conditional],
            )

        unreachable_root = np.asarray([0.0, -np.inf])
        IntegerChowLiuTreeDistribution(
            [None, 0],
            [unreachable_root, conditional],
        )

    def test_raw_encoded_and_backend_event_spaces_agree(self):
        dist = _binary_tree()
        invalid = (
            [-1, 0],
            [0.5, 0],
            [np.nan, 0],
            [2, 0],
            [0],
            [0, 0, 0],
        )
        for value in invalid:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    dist.log_density(value)
                with self.assertRaises((TypeError, ValueError)):
                    dist.dist_to_encoder().seq_encode([value])
                with self.assertRaises((TypeError, ValueError)):
                    dist.seq_log_density([value])
        np.testing.assert_allclose(
            dist.seq_log_density(dist.dist_to_encoder().seq_encode([])),
            np.empty(0),
        )

    def test_constructor_owns_immutable_probability_tables(self):
        root = np.log(np.asarray([0.6, 0.4]))
        conditional = np.log(np.asarray([[0.7, 0.3], [0.2, 0.8]]))
        dist = IntegerChowLiuTreeDistribution(
            [None, 0],
            [root, conditional],
        )
        expected = dist.log_density([0, 0])
        root[:] = np.log([0.1, 0.9])
        conditional[:] = np.log([[0.1, 0.9], [0.9, 0.1]])
        self.assertEqual(dist.log_density([0, 0]), expected)
        with self.assertRaises(ValueError):
            dist.conditional_log_densities[0][0] = 0.0
        with self.assertRaises(TypeError):
            dist.conditional_log_densities[0] = np.log([0.1, 0.9])
        self.assertEqual(
            dist.sampler(seed=7).sample(),
            IntegerChowLiuTreeDistribution(
                [None, 0],
                [
                    np.log([0.6, 0.4]),
                    np.log([[0.7, 0.3], [0.2, 0.8]]),
                ],
            )
            .sampler(seed=7)
            .sample(),
        )

    def test_scalar_and_batch_accumulation_use_one_canonical_schema(self):
        rows = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
        weights = np.asarray([0.5, 1.0, 1.5, 2.0])
        estimator = IntegerChowLiuTreeEstimator()

        scalar = estimator.accumulator_factory().make()
        for row, weight in zip(rows, weights):
            scalar.update(row, weight, None)
        batch = estimator.accumulator_factory().make()
        batch.seq_update(rows, weights, None)

        scalar_value = scalar.value()
        batch_value = batch.value()
        self.assertEqual(scalar_value.schema_version, 1)
        np.testing.assert_allclose(scalar_value.counts, batch_value.counts)
        np.testing.assert_allclose(
            scalar_value.marginal_counts,
            batch_value.marginal_counts,
        )
        self.assertTrue(np.all(scalar_value.counts[1, 0] == 0.0))
        self.assertTrue(np.all(scalar_value.counts[0, 0] == 0.0))

    def test_accumulator_rejects_bad_batches_before_mutation(self):
        estimator = IntegerChowLiuTreeEstimator(
            num_features=2,
            num_states=2,
        )
        for rows, weights in (
            ([[0, 0], [1, 1]], [1.0]),
            ([[0, 0]], [-1.0]),
            ([[0, 0]], [np.nan]),
            ([[0.5, 0]], [1.0]),
            ([[2, 0]], [1.0]),
            ([[0]], [1.0]),
        ):
            with self.subTest(rows=repr(rows), weights=repr(weights)):
                accumulator = estimator.accumulator_factory().make()
                before = accumulator.value()
                with self.assertRaises((TypeError, ValueError)):
                    accumulator.seq_update(rows, weights, None)
                after = accumulator.value()
                np.testing.assert_array_equal(before.counts, after.counts)
                np.testing.assert_array_equal(
                    before.marginal_counts,
                    after.marginal_counts,
                )

    def test_statistics_are_validated_copied_and_keyed(self):
        estimator = IntegerChowLiuTreeEstimator(keys="shared")
        source = estimator.accumulator_factory().make()
        source.seq_update([[0, 0], [1, 1]], [1.0, 1.0], None)
        value = source.value()
        restored = estimator.accumulator_factory().make().from_value(value)
        value.counts[0, 1, 0, 0] = 99.0
        self.assertEqual(restored.counts[0, 1, 0, 0], 1.0)

        malformed = copy.deepcopy(source.value())
        malformed.counts[1, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "upper-triangle"):
            estimator.estimate(None, malformed)

        first = estimator.accumulator_factory().make()
        first.update([0, 0], 1.0, None)
        second = estimator.accumulator_factory().make()
        second.update([1, 1], 1.0, None)
        shared = {}
        first.key_merge(shared)
        second.key_merge(shared)
        first.key_replace(shared)
        self.assertEqual(first.marginal_counts.sum(axis=1).tolist(), [2.0, 2.0])

    def test_estimator_validates_configuration_and_preserves_identity(self):
        for kwargs in (
            {"num_features": 0},
            {"num_states": 0},
            {"pseudo_count": -1.0},
            {"pseudo_count": np.nan},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                IntegerChowLiuTreeEstimator(**kwargs)

        estimator = IntegerChowLiuTreeEstimator(
            num_features=2,
            num_states=2,
            pseudo_count=0.5,
            keys="shared",
            name="tree",
            enumeration_item_budget=17,
        )
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(
            [[0, 0], [1, 1]],
            [1.0, 1.0],
            None,
        )
        model = estimator.estimate(None, accumulator.value())
        self.assertEqual(model.name, "tree")
        self.assertEqual(model.keys, "shared")
        self.assertEqual(model.pseudo_count, 0.5)
        self.assertEqual(model.enumeration_item_budget, 17)
        round_trip = model.estimator()
        self.assertEqual(round_trip.keys, "shared")
        self.assertEqual(round_trip.pseudo_count, 0.5)
        self.assertEqual(round_trip.enumeration_item_budget, 17)

    def test_enumeration_is_deferred_and_budgeted(self):
        dist = IntegerChowLiuTreeDistribution(
            [None, 0, 1],
            [
                np.log([0.5, 0.5]),
                np.log([[0.5, 0.5], [0.5, 0.5]]),
                np.log([[0.5, 0.5], [0.5, 0.5]]),
            ],
        )
        enumerator = dist.enumerator(max_items=4)
        self.assertEqual(enumerator.receipt["items_yielded"], 0)
        self.assertIsNone(enumerator.receipt["termination_reason"])
        with self.assertRaises(EnumerationError):
            next(enumerator)
        self.assertTrue(enumerator.receipt["truncated"])
        self.assertEqual(
            enumerator.receipt["termination_reason"],
            "item_budget_exhausted",
        )


if __name__ == "__main__":
    unittest.main()
