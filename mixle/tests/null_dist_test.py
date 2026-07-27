"""Tests for the NullDistribution's sampler contract (null_dist.py) and IgnoredEstimator's None-dist
default (ignored.py)."""

import unittest

from mixle.stats.combinator.ignored import IgnoredEstimator
from mixle.stats.combinator.null_dist import NeutralFactorError, NullDistribution
from mixle.stats.univariate.discrete.point_mass import PointMassDistribution


class NullContractTestCase(unittest.TestCase):
    def test_neutral_factor_is_not_generative(self):
        with self.assertRaises(NeutralFactorError):
            NullDistribution().sampler(seed=0)
        with self.assertRaises(NotImplementedError):
            NullDistribution().enumerator()

    def test_none_point_mass_is_the_proper_singleton_law(self):
        distribution = PointMassDistribution(None)
        self.assertIsNone(distribution.sampler(seed=0).sample())
        self.assertEqual(distribution.sampler(seed=0).sample(size=5), [None] * 5)
        self.assertEqual(list(distribution.enumerator()), [(None, 0.0)])
        self.assertEqual(distribution.log_density(None), 0.0)
        self.assertEqual(distribution.log_density("anything"), float("-inf"))


class IgnoredEstimatorNoneDistTestCase(unittest.TestCase):
    def test_explicit_none_dist_defaults_to_a_null_distribution_instance(self):
        # dist if dist is not None else NullDistribution -- missing call parens -- assigned the
        # CLASS itself, not an instance, so accumulator_factory()'s self.dist.dist_to_encoder() call
        # crashed with TypeError (an unbound method missing its `self` argument) on first use.
        est = IgnoredEstimator(dist=None)
        self.assertIsInstance(est.dist, PointMassDistribution)
        self.assertIsNone(est.dist.value)
        acc = est.accumulator_factory().make()
        self.assertIsNotNone(acc)

    def test_estimate_round_trips_through_a_none_dist_estimator(self):
        est = IgnoredEstimator(dist=None)
        acc = est.accumulator_factory().make()
        acc.update("anything", 1.0, None)
        fit = est.estimate(None, acc.value())
        self.assertIsInstance(fit.dist, PointMassDistribution)
        self.assertIsNone(fit.dist.value)


if __name__ == "__main__":
    unittest.main()
