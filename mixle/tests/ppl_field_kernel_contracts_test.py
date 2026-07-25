"""Release contracts for latent-field kernels and prior assembly."""

import unittest

import numpy as np

from mixle.ppl import (
    RBF,
    AnisotropicRBF,
    FieldSystem,
    GaussianField,
    GreatCircleMatern,
    GreatCircleRBF,
    RandomWalk,
    great_circle_distance,
)
from mixle.ppl.field import FieldKernel


class KernelGeometryContractTest(unittest.TestCase):
    def test_random_walk_uses_physical_spacing(self):
        irregular = np.array([0.0, 1.0, 3.0, 6.0])
        uniform = np.arange(4.0)
        linear_on_irregular = irregular.copy()
        q_irregular = RandomWalk(scale=1.0, order=2, ridge=None).precision(irregular)
        q_uniform = RandomWalk(scale=1.0, order=2, ridge=None).precision(uniform)
        self.assertAlmostEqual(float(linear_on_irregular @ q_irregular @ linear_on_irregular), 0.0, places=10)
        self.assertGreater(float(linear_on_irregular @ q_uniform @ linear_on_irregular), 0.1)

    def test_kernel_hyperparameters_and_metrics_are_validated(self):
        index = np.linspace(0.0, 1.0, 4)
        invalid = [
            lambda: RandomWalk(scale=0.0, ridge=1.0).precision(index),
            lambda: RandomWalk(scale=1.0, order=3, ridge=1.0).precision(index),
            lambda: RandomWalk(scale=1.0, ridge=-1.0).precision(index),
            lambda: RandomWalk(scale=1.0, ridge=1.0).precision(index[::-1]),
            lambda: RBF(lengthscale=0.0).covariance(index),
            lambda: RBF(amplitude=np.inf).covariance(index),
            lambda: RBF(jitter=0.0).covariance(index),
            lambda: AnisotropicRBF(ranges=(1.0,)).covariance(np.ones((4, 2))),
            lambda: AnisotropicRBF(metric=np.array([[1.0, 2.0], [0.0, 1.0]])).covariance(np.ones((4, 2))),
            lambda: AnisotropicRBF(metric=np.array([[1.0, 0.0], [0.0, -1.0]])).covariance(np.ones((4, 2))),
        ]
        for call in invalid:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_geographic_coordinates_and_radius_are_not_normalized_silently(self):
        for point in ([91.0, 0.0], [0.0, 181.0], [np.nan, 0.0], [0.0], [[0.0, 0.0, 1.0]]):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    great_circle_distance(point, [0.0, 0.0])
        for radius in (0.0, -1.0, np.inf):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    great_circle_distance([0.0, 0.0], [0.0, 1.0], radius=radius)
        valid = np.array([[0.0, 0.0], [10.0, 20.0]])
        for kernel in (
            GreatCircleRBF(lengthscale=0.0),
            GreatCircleRBF(radius=-1.0),
            GreatCircleMatern(amplitude=0.0),
            GreatCircleMatern(jitter=np.nan),
        ):
            with self.subTest(kernel=kernel):
                with self.assertRaises(ValueError):
                    kernel.covariance(valid)


class FieldPriorContractTest(unittest.TestCase):
    def test_covariance_provider_is_evaluated_once(self):
        class StatefulKernel(FieldKernel):
            def __init__(self):
                self.calls = 0

            def covariance(self, index):
                self.calls += 1
                return np.eye(len(index))

            def precision(self, index):  # pragma: no cover - must not be called
                raise AssertionError("precision must be derived from the canonical covariance")

        kernel = StatefulKernel()
        field = GaussianField(np.arange(3.0), kernel)
        self.assertEqual(kernel.calls, 1)
        np.testing.assert_allclose(field.precision, np.eye(3))

    def test_invalid_or_improper_gaussian_measure_is_rejected(self):
        class BadPrecision(FieldKernel):
            def precision(self, index):
                return np.array([[1.0, 2.0], [2.0, 1.0]])

        with self.assertRaisesRegex(ValueError, "positive-definite"):
            GaussianField(np.arange(2.0), BadPrecision())
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            GaussianField(np.arange(3.0), RandomWalk(scale=1.0, ridge=None))

    def test_field_system_requires_mesh_alignment_and_preserves_component_priors(self):
        first = GaussianField(np.array([0.0, 1.0, 2.0]), RBF(lengthscale=1.0), name="first")
        shifted = GaussianField(np.array([0.0, 1.5, 2.0]), RBF(lengthscale=1.0), name="shifted")
        with self.assertRaisesRegex(ValueError, "same index"):
            FieldSystem([first, shifted])

        second = GaussianField(np.array([0.0, 1.0, 2.0]), RBF(lengthscale=0.5), name="second")
        independent = FieldSystem([first, second])
        self.assertFalse(np.allclose(independent.fields[0].precision, independent.fields[1].precision))
        with self.assertRaisesRegex(ValueError, "discard"):
            FieldSystem([first, second], coregion=np.eye(2))


if __name__ == "__main__":
    unittest.main()
