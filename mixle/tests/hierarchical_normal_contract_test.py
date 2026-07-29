"""Weighted, bounded, and boundary contracts for the hierarchical normal model."""

import unittest

import numpy as np

from mixle.stats.latent.hierarchical import (
    HierarchicalNormalAccumulator,
    HierarchicalNormalDataEncoder,
    HierarchicalNormalDistribution,
    HierarchicalNormalEstimator,
)


def _fit(groups, weights):
    estimator = HierarchicalNormalEstimator(max_iter=200, tol=1.0e-10)
    accumulator = estimator.accumulator_factory().make()
    encoded = HierarchicalNormalDataEncoder().seq_encode(groups)
    accumulator.seq_update(encoded, np.asarray(weights, dtype=np.float64), None)
    return estimator.estimate(float(np.sum(weights)), accumulator.value())


class HierarchicalNormalWeightContractTest(unittest.TestCase):
    def test_zero_weight_group_is_exactly_excluded(self):
        used = [[-1.0, -0.5, 0.0], [1.0, 1.5, 2.0]]
        ignored = [100.0, 101.0, 102.0]
        weighted = _fit(used + [ignored], [1.0, 1.0, 0.0])
        reference = _fit(used, [1.0, 1.0])
        np.testing.assert_allclose(
            [weighted.mu, weighted.tau, weighted.sigma],
            [reference.mu, reference.tau, reference.sigma],
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_frequency_weight_matches_explicit_group_replication(self):
        first = [-1.0, -0.5, 0.0]
        second = [1.0, 1.5, 2.0]
        weighted = _fit([first, second], [4.0, 2.0])
        replicated = _fit([first] * 4 + [second] * 2, [1.0] * 6)
        np.testing.assert_allclose(
            [weighted.mu, weighted.tau, weighted.sigma],
            [replicated.mu, replicated.tau, replicated.sigma],
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertEqual(weighted.fit_diagnostics.total_group_weight, 6.0)
        self.assertEqual(weighted.fit_diagnostics.total_observation_weight, 18.0)

    def test_accumulator_is_size_bucketed_not_per_group(self):
        accumulator = HierarchicalNormalAccumulator()
        groups = [[0.0, 1.0]] * 500 + [[0.0, 1.0, 2.0]] * 500
        encoded = HierarchicalNormalDataEncoder().seq_encode(groups)
        accumulator.seq_update(encoded, np.ones(1000), None)
        sizes, group_weights, *_ = accumulator.value()
        np.testing.assert_array_equal(sizes, [2, 3])
        np.testing.assert_array_equal(group_weights, [500.0, 500.0])
        self.assertEqual(len(accumulator.buckets), 2)

    def test_distinct_size_budget_fails_before_partial_mutation(self):
        accumulator = HierarchicalNormalAccumulator(max_group_sizes=2)
        encoder = HierarchicalNormalDataEncoder()
        accumulator.seq_update(encoder.seq_encode([[0.0], [0.0, 1.0]]), [1.0, 1.0], None)
        before = accumulator.value()
        with self.assertRaisesRegex(ValueError, "limit"):
            accumulator.seq_update(encoder.seq_encode([[0.0], [0.0, 1.0, 2.0]]), [1.0, 1.0], None)
        after = accumulator.value()
        for actual, expected in zip(after, before):
            np.testing.assert_array_equal(actual, expected)

    def test_weight_domain_is_finite_nonnegative_and_aligned(self):
        encoded = HierarchicalNormalDataEncoder().seq_encode([[0.0, 1.0]])
        for weights in ([], [-1.0], [np.nan], [1.0, 2.0]):
            with self.subTest(weights=repr(weights)), self.assertRaises((TypeError, ValueError)):
                HierarchicalNormalAccumulator().seq_update(encoded, weights, None)


class HierarchicalNormalBoundaryContractTest(unittest.TestCase):
    def test_model_and_posterior_parameters_are_strict(self):
        invalid_models = (
            (np.nan, 1.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, np.inf),
        )
        for args in invalid_models:
            with self.subTest(args=repr(args)), self.assertRaises((TypeError, ValueError)):
                HierarchicalNormalDistribution(*args)

        model = HierarchicalNormalDistribution(0.0, 1.0, 1.0)
        for size in (0, -1, 1.5, True):
            with self.subTest(size=repr(size)), self.assertRaises((TypeError, ValueError)):
                model.group_posterior(0.0, size)
            with self.subTest(size=repr(size)), self.assertRaises((TypeError, ValueError)):
                model.shrinkage(size)
        with self.assertRaises(ValueError):
            model.group_posterior(np.nan, 1)

    def test_group_and_encoder_geometry_is_explicit(self):
        model = HierarchicalNormalDistribution(0.0, 1.0, 1.0)
        for group in ([], [[0.0, 1.0]], [0.0, np.nan]):
            with self.subTest(group=repr(group)), self.assertRaises((TypeError, ValueError)):
                model.log_density(group)
        encoder = model.dist_to_encoder()
        empty = encoder.seq_encode([])
        self.assertEqual(tuple(array.shape for array in empty), ((0,), (0,), (0,)))
        self.assertEqual(encoder.row_count(empty), 0)
        with self.assertRaises(ValueError):
            encoder.seq_encode([[]])

    def test_no_data_returns_an_explicit_unidentified_default(self):
        estimator = HierarchicalNormalEstimator()
        empty = estimator.accumulator_factory().make().value()
        fitted = estimator.estimate(0.0, empty)
        self.assertEqual((fitted.mu, fitted.tau, fitted.sigma), (0.0, 1.0, 1.0))
        self.assertFalse(fitted.fit_diagnostics.identifiable)
        self.assertTrue(fitted.fit_diagnostics.converged)
        self.assertEqual(fitted.fit_diagnostics.termination_reason, "no_data")
        self.assertEqual(fitted.fit_diagnostics.objective_trace, ())

    def test_estimator_controls_and_statistics_are_validated(self):
        for kwargs in (
            {"max_iter": 0},
            {"max_iter": 1.5},
            {"tol": -1.0},
            {"tol": np.nan},
            {"max_group_sizes": 0},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                HierarchicalNormalEstimator(**kwargs)

        estimator = HierarchicalNormalEstimator()
        malformed = (
            ([1.5], [1.0], [0.0], [0.0], [0.0]),
            ([1.0], [-1.0], [0.0], [0.0], [0.0]),
            ([1.0], [1.0], [2.0], [1.0], [0.0]),
            ([1.0], [1.0], [0.0], [0.0], [-1.0]),
            ([1.0], [1.0, 2.0], [0.0], [0.0], [0.0]),
        )
        for statistics in malformed:
            with self.subTest(statistics=repr(statistics)), self.assertRaises((TypeError, ValueError)):
                estimator.estimate(1.0, statistics)

    def test_fit_receipt_tracks_a_finite_monotone_objective(self):
        rng = np.random.RandomState(4)
        group_means = rng.normal(2.0, 1.5, 30)
        groups = [rng.normal(mean, 0.8, size=3 + index % 4) for index, mean in enumerate(group_means)]
        fitted = _fit(groups, np.ones(len(groups)))
        receipt = fitted.fit_diagnostics
        self.assertTrue(receipt.identifiable)
        self.assertEqual(receipt.iterations, len(receipt.objective_trace) - 1)
        self.assertTrue(np.all(np.isfinite(receipt.objective_trace)))
        self.assertTrue(np.all(np.diff(receipt.objective_trace) >= -1.0e-8))
        self.assertGreater(receipt.total_observation_weight, receipt.total_group_weight)


if __name__ == "__main__":
    unittest.main()
