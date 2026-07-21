"""Tests for the NullDistribution's sampler contract (null_dist.py) and IgnoredEstimator's None-dist
default (ignored.py)."""

import unittest

from mixle.stats.combinator.ignored import IgnoredEstimator
from mixle.stats.combinator.null_dist import NullDistribution


class NullSamplerTestCase(unittest.TestCase):
    def test_single_draw_returns_none(self):
        self.assertIsNone(NullDistribution().sampler(seed=0).sample())

    def test_batched_draw_returns_a_list_of_nones(self):
        # sample(size=n) must return a length-n collection per the DistributionSampler contract
        # (sample(size=None) -> single observation, sample(size=n) -> length-n collection) --
        # NullSampler previously returned bare None regardless of size, breaking any composite
        # sampler (e.g. CompositeDistribution) that zips child samples expecting len() == size.
        samples = NullDistribution().sampler(seed=0).sample(size=5)
        self.assertEqual(samples, [None] * 5)
        self.assertEqual(len(samples), 5)

    def test_zero_size_draw_returns_an_empty_list(self):
        self.assertEqual(NullDistribution().sampler(seed=0).sample(size=0), [])


class IgnoredEstimatorNoneDistTestCase(unittest.TestCase):
    def test_explicit_none_dist_defaults_to_a_null_distribution_instance(self):
        # dist if dist is not None else NullDistribution -- missing call parens -- assigned the
        # CLASS itself, not an instance, so accumulator_factory()'s self.dist.dist_to_encoder() call
        # crashed with TypeError (an unbound method missing its `self` argument) on first use.
        est = IgnoredEstimator(dist=None)
        self.assertIsInstance(est.dist, NullDistribution)
        acc = est.accumulator_factory().make()
        self.assertIsNotNone(acc)

    def test_estimate_round_trips_through_a_none_dist_estimator(self):
        est = IgnoredEstimator(dist=None)
        acc = est.accumulator_factory().make()
        acc.update("anything", 1.0, None)
        fit = est.estimate(None, acc.value())
        self.assertIsInstance(fit.dist, NullDistribution)


if __name__ == "__main__":
    unittest.main()
