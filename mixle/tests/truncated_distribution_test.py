"""Tests for TruncatedDistribution (restrict a base distribution to an allowed support)."""

import math
import unittest

import numpy as np

from mixle.stats.combinator.truncated import (
    TruncatedDistribution,
    TruncatedProjectionEstimator,
    TruncatedStatistics,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution, BernoulliEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

TOL = 1e-12


class TruncatedDistributionTestCase(unittest.TestCase):
    def setUp(self):
        self.cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})

    def test_allowed_form_renormalizes(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        z = 0.5 + 0.3 + 0.15
        self.assertAlmostEqual(math.exp(t.log_density("a")), 0.5 / z, delta=TOL)
        self.assertEqual(t.log_density("d"), -np.inf)
        self.assertAlmostEqual(sum(math.exp(t.log_density(v)) for v in "abc"), 1.0, delta=TOL)
        self.assertEqual(t.support_size(), 3)

    def test_forbidden_form_on_infinite_base(self):
        t = TruncatedDistribution(PoissonDistribution(2.0), forbidden=[0])
        z = 1.0 - math.exp(-2.0)
        self.assertAlmostEqual(math.exp(t.log_density(1)), math.exp(-2.0) * 2.0 / z, delta=1e-12)
        self.assertEqual(t.log_density(0), -np.inf)
        self.assertIsNone(t.support_size())  # infinite base minus a finite set is still infinite

    def test_enumerator_is_descending_and_normalized(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        items = list(t.enumerator())
        self.assertEqual([v for v, _ in items], ["a", "b", "c"])
        self.assertAlmostEqual(sum(math.exp(lp) for _, lp in items), 1.0, delta=TOL)
        lps = [lp for _, lp in items]
        self.assertTrue(all(lps[i] >= lps[i + 1] for i in range(len(lps) - 1)))
        for v, lp in items:
            self.assertAlmostEqual(lp, t.log_density(v), delta=TOL)

    def test_seq_log_density_matches(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        enc = t.dist_to_encoder().seq_encode(["a", "b", "c", "d"])
        sld = t.seq_log_density(enc)
        np.testing.assert_allclose(sld[:3], [t.log_density(v) for v in "abc"], atol=TOL)
        self.assertEqual(sld[3], -np.inf)

    def test_sampler_respects_truncation(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        s = t.sampler(0).sample(4000)
        self.assertNotIn("d", set(s))
        z = 0.95
        self.assertAlmostEqual(sum(1 for v in s if v == "a") / 4000, 0.5 / z, delta=0.03)

    def test_fixed_truncation_estimation_round_trip(self):
        truth = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        data = truth.sampler(1).sample(6000)
        est = truth.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(truth.dist_to_encoder().seq_encode(data), np.ones(len(data)), truth)
        fitted = est.estimate(len(data), acc.value())
        self.assertIsInstance(fitted, TruncatedDistribution)
        z = 0.95
        self.assertAlmostEqual(math.exp(fitted.log_density("a")), 0.5 / z, delta=0.04)
        self.assertEqual(fitted.log_density("d"), -np.inf)

    def test_validation(self):
        with self.assertRaises(ValueError):
            TruncatedDistribution(self.cat)  # neither allowed nor forbidden
        with self.assertRaises(ValueError):
            TruncatedDistribution(self.cat, allowed=["a"], forbidden=["b"])  # both

    def test_forbidding_single_point_on_continuous_base_is_a_no_op(self):
        # A continuous base has zero measure at any single point: forbidding one must not change the
        # normalizing constant (only exclude that exact point from the retained support).
        base = GaussianDistribution(0.0, 1.0)
        t = TruncatedDistribution(base, forbidden=[0.0])
        self.assertEqual(t.log_z, 0.0)
        for x in (-1.3, 0.37, 2.0):
            self.assertAlmostEqual(t.log_density(x), base.log_density(x), delta=TOL)
        self.assertEqual(t.log_density(0.0), -np.inf)  # the forbidden point itself is still excluded

    def test_allowing_only_individual_points_on_continuous_base_retains_no_mass(self):
        # The mirror image: a finite list of individual points on a continuous base retains zero
        # measure (each point has probability zero), so there is no positive-measure "allowed"
        # subset to renormalize onto -- this is the pre-existing "retains no probability mass" error,
        # not a special case.
        base = GaussianDistribution(0.0, 1.0)
        with self.assertRaises(ValueError):
            TruncatedDistribution(base, allowed=[0.0, 1.0])

    def test_allowed_duplicates_are_deduplicated(self):
        # A duplicate entry in `allowed` must not be double-counted into the normalizing constant.
        base = BernoulliDistribution(0.25)  # P(0) = 0.75, P(1) = 0.25
        t = TruncatedDistribution(base, allowed=[0, 0])
        self.assertEqual(t.support_size(), 1)
        self.assertAlmostEqual(math.exp(t.log_density(0)), 1.0, delta=TOL)
        self.assertEqual(t.log_density(1), -np.inf)

    def test_forbidden_duplicates_are_deduplicated(self):
        # Same double-counting bug, mirrored onto `forbidden`: a repeated entry must not subtract its
        # mass twice (which would otherwise push the retained probability above 1).
        base = BernoulliDistribution(0.25)
        t = TruncatedDistribution(base, forbidden=[1, 1])
        self.assertAlmostEqual(math.exp(t.log_density(0)), 1.0, delta=TOL)
        self.assertEqual(t.log_density(1), -np.inf)

    def test_scalar_and_batch_accumulation_share_the_support_contract(self):
        distribution = TruncatedDistribution(BernoulliDistribution(0.25), allowed=[0])
        scalar = distribution.estimator().accumulator_factory().make()
        scalar.update(1, 2.0, distribution)
        scalar.update(0, 3.0, distribution)
        batch = distribution.estimator().accumulator_factory().make()
        encoded = distribution.dist_to_encoder().seq_encode([1, 0])
        batch.seq_update(encoded, np.asarray([2.0, 3.0]), distribution)
        self.assertEqual(scalar.value(), batch.value())
        self.assertEqual(
            scalar.value(),
            TruncatedStatistics(1, (3.0, 0.0), 3.0, 2.0),
        )

    def test_cold_accumulator_encoder_retains_the_support_rule(self):
        distribution = TruncatedDistribution(BernoulliDistribution(0.25), allowed=[0])
        accumulator = distribution.estimator().accumulator_factory().make()
        encoder = accumulator.acc_to_encoder()
        self.assertEqual(encoder, distribution.dist_to_encoder())
        accumulator.seq_initialize(
            encoder.seq_encode([0, 1]),
            np.asarray([4.0, 5.0]),
            np.random.RandomState(0),
        )
        self.assertEqual(accumulator.value(), TruncatedStatistics(1, (4.0, 0.0), 4.0, 5.0))

    def test_projection_estimator_is_explicit_and_reports_excluded_evidence(self):
        class RecordingBernoulliEstimator(BernoulliEstimator):
            def estimate(inner_self, nobs, suff_stat):
                inner_self.received_nobs = nobs
                return super().estimate(nobs, suff_stat)

        estimator = TruncatedProjectionEstimator(
            RecordingBernoulliEstimator(),
            allowed=[0, 1],
        )
        fitted = estimator.estimate(
            999.0,
            TruncatedStatistics(1, (3.0, 1.0), 3.0, 2.0),
        )
        self.assertEqual(estimator.base_estimator.received_nobs, 3.0)
        self.assertEqual(fitted.fit_receipt.accepted_weight, 3.0)
        self.assertEqual(fitted.fit_receipt.rejected_weight, 2.0)
        self.assertFalse(fitted.fit_receipt.likelihood_aware)

    def test_support_size_intersects_allowed_values_with_base_support(self):
        distribution = TruncatedDistribution(BernoulliDistribution(0.25), allowed=[0, 2])
        self.assertEqual(distribution.support_size(), 1)
        self.assertEqual([value for value, _ in distribution.enumerator()], [0])


if __name__ == "__main__":
    unittest.main()
