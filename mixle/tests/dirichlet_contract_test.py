import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.bayes.dirichlet import (
    DirichletAccumulator,
    DirichletConvergenceError,
    DirichletDistribution,
    DirichletEstimator,
    dirichlet_param_solve,
)


class DirichletBoundaryScoringContractTestCase(unittest.TestCase):
    def assert_score_parity(self, alpha, row, expected):
        dist = DirichletDistribution(alpha)
        encoded = dist.dist_to_encoder().seq_encode([row])
        scalar = dist.log_density(row)
        sequence = dist.seq_log_density(encoded)[0]
        backend = float(np.asarray(dist.backend_seq_log_density(encoded, NUMPY_ENGINE))[0])
        if np.isposinf(expected):
            self.assertTrue(np.isposinf(scalar))
            self.assertTrue(np.isposinf(sequence))
            self.assertTrue(np.isposinf(backend))
        elif np.isneginf(expected):
            self.assertTrue(np.isneginf(scalar))
            self.assertTrue(np.isneginf(sequence))
            self.assertTrue(np.isneginf(backend))
        else:
            self.assertAlmostEqual(scalar, expected)
            self.assertAlmostEqual(sequence, expected)
            self.assertAlmostEqual(backend, expected)

    def test_exact_zero_boundary_matches_across_scoring_paths(self):
        self.assert_score_parity([0.5, 0.5], [0.0, 1.0], np.inf)
        self.assert_score_parity([2.0, 2.0], [0.0, 1.0], -np.inf)
        self.assert_score_parity([1.0, 1.0], [0.0, 1.0], 0.0)
        self.assert_score_parity([0.5, 2.0, 1.0], [0.0, 0.0, 1.0], np.inf)

    def test_encoder_preserves_exact_boundary_and_scorers_ignore_forged_companions(self):
        dist = DirichletDistribution([0.5, 0.5])
        encoded = dist.dist_to_encoder().seq_encode([[0.0, 1.0]])
        self.assertTrue(np.isneginf(encoded[0][0, 0]))
        forged = (np.zeros_like(encoded[0]), encoded[1], np.full_like(encoded[2], 99.0))
        self.assertTrue(np.isposinf(dist.seq_log_density(forged)[0]))
        backend = dist.backend_seq_log_density(forged, NUMPY_ENGINE)
        self.assertTrue(np.isposinf(float(np.asarray(backend)[0])))

    def test_invalid_simplex_rows_score_negative_infinity(self):
        dist = DirichletDistribution([0.5, 0.5])
        rows = [[-0.1, 1.1], [0.2, 0.9], [np.nan, 1.0]]
        encoded = dist.dist_to_encoder().seq_encode(rows)
        np.testing.assert_array_equal(dist.seq_log_density(encoded), [-np.inf] * 3)
        np.testing.assert_array_equal(
            np.asarray(dist.backend_seq_log_density(encoded, NUMPY_ENGINE)),
            [-np.inf] * 3,
        )


class DirichletFittingContractTestCase(unittest.TestCase):
    def setUp(self):
        self.rows = np.asarray([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
        self.encoded = DirichletDistribution([1.0, 1.0, 1.0]).dist_to_encoder().seq_encode(self.rows)

    def test_scalar_and_sequence_accumulation_match(self):
        scalar = DirichletAccumulator(dim=3)
        scalar.update(self.rows[0], 0.5, None)
        scalar.update(self.rows[1], 2.0, None)
        sequence = DirichletAccumulator(dim=3)
        sequence.seq_update(self.encoded, np.asarray([0.5, 2.0]), None)
        for actual, expected in zip(scalar.value(), sequence.value()):
            np.testing.assert_allclose(actual, expected)

    def test_accumulator_rejects_invalid_observations_and_weights(self):
        acc = DirichletAccumulator(dim=3)
        for row in (
            [0.0, 0.5, 0.5],
            [-0.1, 0.5, 0.6],
            [0.2, 0.3, 0.6],
            [np.nan, 0.5, 0.5],
            [0.5, 0.5],
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                acc.update(row, 1.0, None)
        for weight in (-1.0, np.nan, [1.0]):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                acc.update(self.rows[0], weight, None)
        for weights in ([1.0], [[1.0], [1.0]], [1.0, -1.0], [1.0, np.nan]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                acc.seq_update(self.encoded, weights, None)
        self.assertEqual(acc.counts, 0.0)

    def test_serialized_statistics_are_validated_atomically(self):
        acc = DirichletAccumulator(dim=3)
        acc.seq_update(self.encoded, np.ones(2), None)
        before = acc.value()
        malformed = [
            (-1.0, np.zeros(3), np.zeros(3), np.zeros(3)),
            (2.0, np.zeros(2), np.ones(3), np.ones(3)),
            (2.0, np.asarray([np.nan, 0.0, 0.0]), np.ones(3), np.ones(3)),
            (2.0, np.zeros(3), np.ones(3), np.full(3, 2.0)),
        ]
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                acc.combine(value)
            after = acc.value()
            for actual, expected in zip(after, before):
                np.testing.assert_array_equal(actual, expected)


class DirichletEstimatorContractTestCase(unittest.TestCase):
    def test_constructor_rejects_malformed_controls_and_prior_statistics(self):
        for kwargs in (
            {"dim": 0},
            {"dim": 2.5},
            {"pseudo_count": -1.0},
            {"pseudo_count": np.nan},
            {"delta": 0.0},
            {"delta": np.inf},
            {"use_mpe": 1},
            {"dim": 3, "pseudo_count": 1.0, "suff_stat": [0.0, 0.0]},
            {"pseudo_count": 1.0, "suff_stat": [0.0, np.nan]},
            {"suff_stat": [-1.0, -1.0]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                DirichletEstimator(**kwargs)

    def test_empty_statistics_require_or_preserve_dimension(self):
        with self.assertRaisesRegex(ValueError, "cannot infer"):
            DirichletEstimator().estimate(None, (0.0, None, None, None))
        result = DirichletEstimator(dim=3).estimate(None, (0.0, None, None, None))
        np.testing.assert_array_equal(result.alpha, np.ones(3))
        self.assertTrue(result.fit_metadata["converged"])
        self.assertEqual(result.fit_metadata["repairs"], ())

    def test_prior_only_fit_is_recovered_without_silent_repair(self):
        prior = DirichletDistribution([2.0, 3.0, 4.0])
        result = prior.estimator(pseudo_count=2.0).estimate(
            None,
            (0.0, np.zeros(3), np.zeros(3), np.zeros(3)),
        )
        np.testing.assert_allclose(result.alpha, prior.alpha, rtol=1.0e-7, atol=1.0e-9)
        self.assertEqual(result.fit_metadata["solver"], "newton")
        self.assertTrue(result.fit_metadata["converged"])
        self.assertEqual(result.fit_metadata["repairs"], ())

    def test_estimator_rejects_malformed_sufficient_statistics(self):
        estimator = DirichletEstimator(dim=3)
        malformed = [
            (-1.0, np.zeros(3), np.zeros(3), np.zeros(3)),
            (1.0, np.zeros(2), np.ones(2) / 2.0, np.ones(2) / 4.0),
            (1.0, np.asarray([np.nan, 0.0, 0.0]), np.ones(3) / 3.0, np.ones(3) / 9.0),
            (1.0, np.zeros(3), np.ones(3), np.ones(3)),
        ]
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimator.estimate(None, value)

    def test_solver_reports_invalid_input_and_nonconvergence(self):
        with self.assertRaises(ValueError):
            dirichlet_param_solve(np.ones(2), np.asarray([-1.0]), 1.0e-8)
        with self.assertRaises(ValueError):
            dirichlet_param_solve(np.ones(2), np.asarray([-1.0, -1.0]), 0.0)
        with self.assertRaises(DirichletConvergenceError):
            dirichlet_param_solve(
                np.ones(2),
                np.asarray([-10.0, -0.1]),
                1.0e-16,
                max_iter=1,
            )


if __name__ == "__main__":
    unittest.main()
