import unittest

import numpy as np

from mixle.stats.bayes.dict_dirichlet import (
    DictDirichletDistribution,
    UnspecifiedDirichletDimensionError,
)
from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution


class SymmetricDirichletContractTest(unittest.TestCase):
    def test_dimension_must_be_a_positive_integer(self):
        for dimension in (0, -1, True, 1.5, np.nan):
            with self.subTest(dimension=repr(dimension)):
                with self.assertRaises((TypeError, ValueError)):
                    SymmetricDirichletDistribution(1.0, dim=dimension)
        self.assertEqual(SymmetricDirichletDistribution(1.0, dim=np.int64(3)).dim, 3)

    def test_fixed_dimension_and_batch_rank_are_enforced(self):
        dist = SymmetricDirichletDistribution(2.0, dim=3)
        self.assertEqual(dist.log_density([0.5, 0.5]), -np.inf)
        invalid = (
            np.array([0.2, 0.3, 0.5]),
            np.ones((1, 1, 3)),
            np.array([[0.5, 0.5]]),
            np.empty((1, 0)),
        )
        for values in invalid:
            with self.subTest(shape=repr(values.shape)):
                with self.assertRaises(ValueError):
                    dist.seq_log_density(values)
                with self.assertRaises(ValueError):
                    dist.dist_to_encoder().seq_encode(values)

        empty = np.empty((0, 3))
        self.assertEqual(dist.seq_log_density(empty).shape, (0,))
        self.assertEqual(dist.dist_to_encoder().seq_encode(empty).shape, (0, 3))

    def test_sampler_validates_size_and_mutated_dimension(self):
        dist = SymmetricDirichletDistribution(2.0, dim=3)
        sampler = dist.sampler(seed=1)
        for size in (True, 1.5, -1):
            with self.subTest(size=repr(size)):
                with self.assertRaises((TypeError, ValueError)):
                    sampler.sample(size)
        dist.dim = 0
        with self.assertRaises(ValueError):
            sampler.sample()


class DictDirichletContractTest(unittest.TestCase):
    def test_fixed_alpha_is_copied_and_parameter_updates_are_atomic(self):
        alpha = {"a": 2.0, "b": 3.0}
        dist = DictDirichletDistribution(alpha)
        alpha["a"] = 99.0
        self.assertEqual(dist.alpha["a"], 2.0)

        returned = dist.get_parameters()
        returned["a"] = 42.0
        self.assertEqual(dist.alpha["a"], 2.0)

        before = dist.get_parameters()
        with self.assertRaises(ValueError):
            dist.set_parameters({"a": 1.0, "b": 0.0})
        self.assertEqual(dist.get_parameters(), before)

    def test_fixed_support_must_match_observation_exactly(self):
        dist = DictDirichletDistribution({"a": 2.0, "b": 3.0})
        self.assertEqual(dist.log_density({"a": 1.0}), -np.inf)
        self.assertEqual(
            dist.log_density({"a": 0.5, "b": 0.25, "c": 0.25}),
            -np.inf,
        )
        self.assertTrue(np.isfinite(dist.log_density({"b": 0.6, "a": 0.4})))
        np.testing.assert_array_equal(
            dist.seq_log_density(
                [
                    {"a": 1.0},
                    {"a": 0.4, "b": 0.6},
                    {"a": 0.4, "b": 0.5, "c": 0.1},
                ]
            ),
            [-np.inf, dist.log_density({"a": 0.4, "b": 0.6}), -np.inf],
        )

    def test_cross_entropy_requires_matching_support(self):
        first = DictDirichletDistribution({"a": 2.0, "b": 3.0})
        second = DictDirichletDistribution({"a": 2.0, "c": 3.0})
        with self.assertRaises(ValueError):
            first.cross_entropy(second)

    def test_scalar_information_operations_raise_typed_dimension_error(self):
        first = DictDirichletDistribution(2.0)
        second = DictDirichletDistribution(np.float64(3.0))
        with self.assertRaises(UnspecifiedDirichletDimensionError):
            first.entropy()
        with self.assertRaises(UnspecifiedDirichletDimensionError):
            first.cross_entropy(second)

        fixed = DictDirichletDistribution({"a": 2.0, "b": 3.0})
        self.assertTrue(np.isfinite(first.cross_entropy(fixed)))
        self.assertTrue(np.isfinite(fixed.cross_entropy(first)))

    def test_sampler_revalidates_concentrations_and_cardinality(self):
        dist = DictDirichletDistribution({"a": 2.0, "b": 3.0})
        sampler = dist.sampler(seed=1)
        for size in (True, 1.5, -1):
            with self.subTest(size=repr(size)):
                with self.assertRaises((TypeError, ValueError)):
                    sampler.sample(size)
        dist.alpha["a"] = 0.0
        with self.assertRaises(ValueError):
            sampler.sample()


if __name__ == "__main__":
    unittest.main()
