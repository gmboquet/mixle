import unittest

import numpy as np

from mixle.stats.bayes.multivariate_normal_gamma import (
    MultivariateNormalGammaDistribution,
)
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution


class NormalGammaContractTest(unittest.TestCase):
    def test_constructor_requires_a_proper_scalar_law(self):
        invalid = (
            (np.nan, 1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, -1.0, 1.0),
            (0.0, 1.0, 1.0, np.inf),
            ([0.0], 1.0, 1.0, 1.0),
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    NormalGammaDistribution(*params)

    def test_setter_validates_atomically(self):
        dist = NormalGammaDistribution(1.0, 2.0, 3.0, 4.0)
        before = dist.get_parameters()
        with self.assertRaises(ValueError):
            dist.set_parameters((9.0, 0.0, 3.0, 4.0))
        self.assertEqual(dist.get_parameters(), before)

    def test_scalar_and_batch_scoring_validate_observation_support(self):
        dist = NormalGammaDistribution(0.0, 1.0, 2.0, 3.0)
        for value in ((0.0,), (0.0, 0.0), (0.0, -1.0), (np.inf, 1.0)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dist.log_density(value)

        for values in (
            np.ones(2),
            np.ones((2, 3)),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [np.nan, 1.0]]),
        ):
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    dist.seq_log_density(values)

        result = dist.seq_log_density(np.array([[0.0, 1.0], [1.0, 2.0]]))
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_cross_entropy_and_sampling_revalidate_mutated_state(self):
        dist = NormalGammaDistribution(0.0, 1.0, 2.0, 3.0)
        other = NormalGammaDistribution(1.0, 2.0, 3.0, 4.0)
        other.lam = 0.0
        with self.assertRaises(ValueError):
            dist.cross_entropy(other)

        sampler = dist.sampler(seed=1)
        for size in (True, 1.5, -1):
            with self.subTest(size=size):
                with self.assertRaises((TypeError, ValueError)):
                    sampler.sample(size)
        dist.a = 0.0
        with self.assertRaises(ValueError):
            sampler.sample()


class MultivariateNormalGammaContractTest(unittest.TestCase):
    def test_constructor_requires_aligned_finite_positive_vectors(self):
        invalid = (
            (0.0, [1.0], [1.0], [1.0]),
            ([[0.0]], [1.0], [1.0], [1.0]),
            ([0.0, 1.0], [1.0], [1.0, 1.0], [1.0, 1.0]),
            ([0.0], [0.0], [1.0], [1.0]),
            ([0.0], [1.0], [np.nan], [1.0]),
            ([0.0], [1.0], [1.0], [-1.0]),
            ([], [], [], []),
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    MultivariateNormalGammaDistribution(*params)

    def test_constructor_copies_inputs_and_setter_is_atomic(self):
        mu = np.array([0.0, 1.0])
        lam = np.array([1.0, 2.0])
        a = np.array([2.0, 3.0])
        b = np.array([3.0, 4.0])
        dist = MultivariateNormalGammaDistribution(mu, lam, a, b)
        mu[0] = 99.0
        self.assertEqual(dist.mu[0], 0.0)

        before = dist.get_parameters()
        with self.assertRaises(ValueError):
            dist.set_parameters(([9.0, 9.0], [1.0], [2.0, 2.0], [3.0, 3.0]))
        for actual, expected in zip(dist.get_parameters(), before):
            np.testing.assert_array_equal(actual, expected)

        returned = dist.get_parameters()
        returned[0][0] = 42.0
        self.assertEqual(dist.mu[0], 0.0)

    def test_scoring_requires_exact_vector_geometry_and_positive_precision(self):
        dist = MultivariateNormalGammaDistribution(
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        )
        invalid = (
            ([0.0], [1.0]),
            ([0.0, 1.0], [1.0]),
            ([[0.0, 1.0]], [1.0, 1.0]),
            ([0.0, 1.0], [1.0, 0.0]),
            ([0.0, np.inf], [1.0, 1.0]),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dist.log_density(value)

        valid = ([0.0, 1.0], [1.0, 2.0])
        self.assertTrue(np.isfinite(dist.log_density(valid)))
        with self.assertRaises(ValueError):
            dist.seq_log_density([valid, ([0.0, 1.0], [1.0, -1.0])])
        encoded = dist.dist_to_encoder().seq_encode([valid])
        self.assertEqual(len(encoded), 1)

    def test_cross_entropy_and_sampling_revalidate_state_and_dimensions(self):
        dist = MultivariateNormalGammaDistribution(
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        )
        other = MultivariateNormalGammaDistribution([0.0], [1.0], [2.0], [3.0])
        with self.assertRaises(ValueError):
            dist.cross_entropy(other)

        sampler = dist.sampler(seed=1)
        for size in (True, 1.5, -1):
            with self.subTest(size=size):
                with self.assertRaises((TypeError, ValueError)):
                    sampler.sample(size)
        dist.lam[0] = 0.0
        with self.assertRaises(ValueError):
            sampler.sample()


if __name__ == "__main__":
    unittest.main()
