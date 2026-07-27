"""Parameter-cache and event-shape contracts for probabilistic PCA."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import ProbabilisticPCADistribution


class PPCAParameterOwnershipTest(unittest.TestCase):
    @staticmethod
    def _parameters():
        return (
            np.array([[1.0], [0.5]]),
            np.array([0.2, -0.3]),
            0.4,
        )

    def test_source_and_public_array_mutation_cannot_desynchronize_cached_state(self):
        loadings, mean, noise = self._parameters()
        model = ProbabilisticPCADistribution(loadings, mean, noise)
        event = np.array([0.1, -0.2])
        expected = model.log_density(event)

        loadings[:] = 0.0
        mean[:] = 100.0
        public_loadings = model.w
        public_mean = model.mu
        public_inverse = model.inv_covar
        public_loadings[:] = 0.0
        public_mean[:] = 100.0
        public_inverse[:] = 0.0

        self.assertEqual(model.log_density(event), expected)
        np.testing.assert_allclose(model.w, [[1.0], [0.5]])
        np.testing.assert_allclose(model.mu, [0.2, -0.3])
        with self.assertRaises(AttributeError):
            model.w = np.zeros((2, 1))
        with self.assertRaises(AttributeError):
            model.mu = np.zeros(2)
        with self.assertRaises(AttributeError):
            model.sigma2 = 1.0

    def test_invalid_or_nonfinite_parameters_are_rejected(self):
        invalid = (
            ([], [], 1.0),
            (np.empty((2, 0)), [0.0, 0.0], 1.0),
            ([[np.nan], [0.0]], [0.0, 0.0], 1.0),
            ([[1.0], [0.0]], [0.0, np.inf], 1.0),
            ([[1.0], [0.0]], [0.0, 0.0], True),
            ([[1.0], [0.0]], [0.0, 0.0], "1"),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                ProbabilisticPCADistribution(*args)


class PPCAEventShapeTest(unittest.TestCase):
    def setUp(self):
        self.model = ProbabilisticPCADistribution([[1.0], [0.5]], [0.2, -0.3], 0.4)

    def test_scalar_routes_require_one_exact_finite_event(self):
        for event in ([0.1], [0.1, 0.2, 0.3], [[0.1, 0.2]], [0.1, np.nan]):
            with self.subTest(event=event), self.assertRaises((TypeError, ValueError)):
                self.model.log_density(event)
            with self.subTest(event=event), self.assertRaises((TypeError, ValueError)):
                self.model.density(event)

    def test_transform_distinguishes_one_event_from_an_exact_event_batch(self):
        self.assertEqual(self.model.transform([0.1, 0.2]).shape, (1,))
        self.assertEqual(self.model.transform([[0.1, 0.2], [0.3, 0.4]]).shape, (1, 2))
        for events in ([0.1], [[0.1]], [[[0.1, 0.2]]], [[0.1, np.nan]]):
            with self.subTest(events=events), self.assertRaises((TypeError, ValueError)):
                self.model.transform(events)

    def test_vectorized_and_backend_routes_require_n_by_d(self):
        valid = np.array([[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_allclose(
            self.model.backend_seq_log_density(valid, NUMPY_ENGINE),
            self.model.seq_log_density(valid),
        )
        for events in ([0.1, 0.2], [[0.1]], [[0.1, 0.2, 0.3]], [[0.1, np.nan]]):
            with self.subTest(events=events), self.assertRaises((TypeError, ValueError)):
                self.model.seq_log_density(events)
        for events in (
            np.array([0.1, 0.2]),
            np.array([[0.1]]),
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0.1, np.nan]]),
        ):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.model.backend_seq_log_density(events, NUMPY_ENGINE)

    def test_encoder_preserves_event_geometry_and_dimension_identity(self):
        encoder = self.model.dist_to_encoder()
        encoded = encoder.seq_encode([[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(encoded.shape, (2, 2))
        self.assertEqual(encoder.seq_encode([]).shape, (0, 2))
        self.assertNotEqual(encoder, ProbabilisticPCADistribution([[1.0]], [0.0], 1.0).dist_to_encoder())
        for events in ([0.1, 0.2], [[0.1]], [[0.1, 0.2, 0.3]], [[0.1, np.nan]]):
            with self.subTest(events=events), self.assertRaises((TypeError, ValueError)):
                encoder.seq_encode(events)

    def test_accumulator_routes_enforce_the_same_shapes_and_weights(self):
        accumulator = self.model.estimator().accumulator_factory().make()
        with self.assertRaises(ValueError):
            accumulator.update([0.1], 1.0, self.model)
        with self.assertRaises(ValueError):
            accumulator.update([0.1, 0.2], -1.0, self.model)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.array([[0.1]]), np.array([1.0]), self.model)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.array([[0.1, 0.2]]), np.array([1.0, 2.0]), self.model)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.array([[0.1, 0.2]]), np.array([np.nan]), self.model)


if __name__ == "__main__":
    unittest.main()
