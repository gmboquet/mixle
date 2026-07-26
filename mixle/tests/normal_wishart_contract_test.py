import unittest

import numpy as np

from mixle.stats.bayes.normal_wishart import NormalWishartDistribution


class NormalWishartContractTest(unittest.TestCase):
    def test_constructor_requires_exact_finite_geometry(self):
        invalid = (
            (0.0, 1.0, [[1.0]], 2.0),
            ([0.0, 0.0], [1.0], np.eye(2), 4.0),
            ([0.0, 0.0], 1.0, [1.0, 1.0], 4.0),
            ([0.0, 0.0], 1.0, np.eye(3), 4.0),
            ([0.0, 0.0], 1.0, [[1.0, 0.5], [0.0, 1.0]], 4.0),
            ([0.0, 0.0], 1.0, [[1.0, 2.0], [2.0, 1.0]], 4.0),
            ([0.0, 0.0], 1.0, np.eye(2), [4.0]),
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    NormalWishartDistribution(*params)

    def test_constructor_copies_inputs_and_setter_is_atomic(self):
        mu = np.array([0.0, 1.0])
        scale = np.eye(2)
        dist = NormalWishartDistribution(mu, 1.0, scale, 4.0)
        mu[0] = 99.0
        scale[0, 0] = 99.0
        np.testing.assert_array_equal(dist.mu, [0.0, 1.0])
        np.testing.assert_array_equal(dist.w_mat, np.eye(2))

        before = dist.get_parameters()
        with self.assertRaises(ValueError):
            dist.set_parameters(([9.0, 9.0], 1.0, [[1.0, 2.0], [2.0, 1.0]], 4.0))
        after = dist.get_parameters()
        np.testing.assert_array_equal(after[0], before[0])
        np.testing.assert_array_equal(after[2], before[2])
        self.assertEqual(after[1], before[1])
        self.assertEqual(after[3], before[3])

        returned = dist.get_parameters()
        returned[0][0] = 42.0
        returned[2][0, 0] = 42.0
        np.testing.assert_array_equal(dist.mu, [0.0, 1.0])
        np.testing.assert_array_equal(dist.w_mat, np.eye(2))

    def test_scoring_rejects_nonsymmetric_or_malformed_precision(self):
        dist = NormalWishartDistribution([0.0, 0.0], 1.0, np.eye(2), 4.0)
        invalid = (
            ([0.0], np.eye(2)),
            ([0.0, 0.0], [1.0, 1.0]),
            ([0.0, np.inf], np.eye(2)),
            ([0.0, 0.0], [[1.0, 1.0], [0.0, 1.0]]),
            ([0.0, 0.0], [[1.0, 10.0], [0.0, 1.0]]),
            ([0.0, 0.0], [[1.0, 100.0], [0.0, 1.0]]),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dist.log_density(value)

        self.assertEqual(
            dist.log_density(([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])),
            -np.inf,
        )
        self.assertTrue(np.isfinite(dist.log_density(([0.0, 0.0], np.eye(2)))))

    def test_cross_entropy_encoding_and_sampling_revalidate_contracts(self):
        dist = NormalWishartDistribution([0.0, 0.0], 1.0, np.eye(2), 4.0)
        other = NormalWishartDistribution([0.0], 1.0, [[1.0]], 2.0)
        with self.assertRaises(ValueError):
            dist.cross_entropy(other)

        valid = ([0.0, 0.0], np.eye(2))
        encoded = dist.dist_to_encoder().seq_encode([valid])
        self.assertEqual(len(encoded), 1)
        with self.assertRaises(ValueError):
            dist.dist_to_encoder().seq_encode([([0.0, 0.0], [[1.0, 1.0], [0.0, 1.0]])])

        sampler = dist.sampler(seed=1)
        for size in (True, 1.5, -1):
            with self.subTest(size=size):
                with self.assertRaises((TypeError, ValueError)):
                    sampler.sample(size)
        dist.w_mat[1, 0] = 10.0
        with self.assertRaises(ValueError):
            sampler.sample()
        with self.assertRaises(ValueError):
            dist.expected_precision()


if __name__ == "__main__":
    unittest.main()
