"""Contract regressions for conditional joint laws, factors, and statistics."""

import unittest

import numpy as np

from mixle.stats import CategoricalDistribution, GaussianDistribution
from mixle.stats.combinator.conditional import (
    ConditionalBranchStatistics,
    ConditionalDistribution,
    ConditionalDistributionEstimator,
    ConditionalStatistics,
    NonGenerativeConditionalError,
)
from mixle.stats.combinator.null_dist import NullEstimator
from mixle.stats.compute.pdist import ContractError, ParameterEstimator
from mixle.stats.univariate.continuous.gaussian import GaussianEstimator


class RecordingEstimator(ParameterEstimator):
    """Delegate fitting while recording the effective count passed to the M-step."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    def accumulator_factory(self):
        return self.delegate.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        self.calls.append(nobs)
        return self.delegate.estimate(nobs, suff_stat)


class ConditionalContractsTest(unittest.TestCase):
    def test_factor_is_non_generative_but_supports_given_sampling(self):
        dist = ConditionalDistribution({"a": GaussianDistribution(0.0, 1.0)})
        self.assertEqual(dist.density_semantics().value, "likelihood_factor")
        with self.assertRaises(NonGenerativeConditionalError):
            dist.sampler(seed=3)
        self.assertTrue(np.isfinite(dist.conditional_sampler(seed=3).sample_given("a")))

    def test_joint_requires_total_routing_coverage(self):
        given = CategoricalDistribution({"a": 0.5, "missing": 0.5})
        with self.assertRaisesRegex(ValueError, "no conditional branch or default"):
            ConditionalDistribution(
                {"a": GaussianDistribution(0.0, 1.0)},
                default_dist=None,
                given_dist=given,
            )
        dist = ConditionalDistribution(
            {"a": GaussianDistribution(0.0, 1.0)},
            default_dist=GaussianDistribution(2.0, 1.0),
            given_dist=given,
        )
        for pair in dist.sampler(seed=4).sample(size=20):
            self.assertTrue(np.isfinite(dist.log_density(pair)))

    def test_scalar_surfaces_require_an_exact_pair(self):
        dist = ConditionalDistribution({"a": GaussianDistribution(0.0, 1.0)})
        acc = dist.estimator().accumulator_factory().make()
        for call in (
            lambda: dist.log_density(("a", 1.0, "extra")),
            lambda: dist.expected_log_density(("a", 1.0, "extra")),
            lambda: acc.update(("a", 1.0, "extra"), 1.0, None),
            lambda: acc.initialize(("a", 1.0, "extra"), 1.0, np.random.RandomState(1)),
        ):
            with self.assertRaises(ContractError):
                call()

    def test_cold_batch_update_preserves_counts_and_exact_layout(self):
        branch_a = RecordingEstimator(GaussianEstimator())
        branch_b = RecordingEstimator(GaussianEstimator())
        default = RecordingEstimator(GaussianEstimator())
        estimator = ConditionalDistributionEstimator(
            {"a": branch_a, "b": branch_b},
            default_estimator=default,
            given_estimator=NullEstimator(),
        )
        data = [("a", 1.0), ("a", 3.0), ("b", 5.0), ("other", 9.0)]
        weights = np.asarray([2.0, 3.0, 5.0, 7.0])
        acc = estimator.accumulator_factory().make()
        encoded = acc.acc_to_encoder().seq_encode(data)
        acc.seq_update(encoded, weights, None)
        stats = acc.value()

        self.assertIsInstance(stats, ConditionalStatistics)
        self.assertEqual(tuple(branch.key for branch in stats.branches), ("a", "b"))
        self.assertEqual(tuple(branch.nobs for branch in stats.branches), (5.0, 5.0))
        self.assertEqual(stats.default_nobs, 7.0)
        fitted = estimator.estimate(None, stats)
        self.assertEqual(branch_a.calls, [5.0])
        self.assertEqual(branch_b.calls, [5.0])
        self.assertEqual(default.calls, [7.0])
        self.assertEqual(tuple(fitted.dmap), ("a", "b"))

    def test_statistics_reject_missing_extra_or_duplicate_branches(self):
        estimator = ConditionalDistributionEstimator(
            {"a": GaussianEstimator(), "b": GaussianEstimator()},
            default_estimator=NullEstimator(),
            given_estimator=NullEstimator(),
        )
        zero = GaussianEstimator().accumulator_factory().make().value()
        cases = (
            ConditionalStatistics(1, (ConditionalBranchStatistics("a", 0.0, zero),), 0.0, None, 0.0, None),
            ConditionalStatistics(
                1,
                (
                    ConditionalBranchStatistics("a", 0.0, zero),
                    ConditionalBranchStatistics("a", 0.0, zero),
                ),
                0.0,
                None,
                0.0,
                None,
            ),
            ConditionalStatistics(
                1,
                (
                    ConditionalBranchStatistics("a", 0.0, zero),
                    ConditionalBranchStatistics("b", 0.0, zero),
                    ConditionalBranchStatistics("c", 0.0, zero),
                ),
                0.0,
                None,
                0.0,
                None,
            ),
        )
        for statistics in cases:
            with self.assertRaises((ContractError, ValueError)):
                estimator.estimate(None, statistics)

    def test_rng_layout_is_initialized_only_once(self):
        acc = ConditionalDistribution({"a": GaussianDistribution(0.0, 1.0)}).estimator().accumulator_factory().make()
        rng = np.random.RandomState(9)
        acc.initialize(("a", 1.0), 1.0, rng)
        child_rng = acc._acc_rng["a"]
        acc.initialize(("a", 2.0), 1.0, rng)
        self.assertTrue(acc._init_rng)
        self.assertIs(acc._acc_rng["a"], child_rng)

    def test_outer_key_pooling_uses_the_complete_conditional_statistic(self):
        estimator = ConditionalDistributionEstimator(
            {"a": GaussianEstimator()},
            default_estimator=NullEstimator(),
            given_estimator=NullEstimator(),
            keys="shared-conditional",
        )
        first = estimator.accumulator_factory().make()
        second = estimator.accumulator_factory().make()
        replacement = estimator.accumulator_factory().make()
        first.update(("a", 1.0), 2.0, None)
        second.update(("a", 3.0), 5.0, None)
        pooled = {}
        first.key_merge(pooled)
        second.key_merge(pooled)
        replacement.key_replace(pooled)
        self.assertEqual(replacement.value().branches[0].nobs, 7.0)

    def test_empty_encoder_string_and_geometry_are_stable(self):
        encoder = ConditionalDistribution({}).dist_to_encoder()
        self.assertEqual(str(encoder), "ConditionalDataEncoder({},default=None,given=None)")
        encoded = encoder.seq_encode([])
        self.assertEqual(encoder.row_count(encoded), 0)
        malformed = (1, (), (), (), None)
        with self.assertRaisesRegex(ValueError, "partition"):
            encoder.row_count(malformed)


if __name__ == "__main__":
    unittest.main()
