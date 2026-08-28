"""T1-01/T1-02: two more gaps in the GeneralizedPareto prior/support-consistency mechanism.

T1-01: `GeneralizedParetoEstimator.loc` used to be a bare public float attribute. Re-targeting the
SAME estimator at a new threshold (a natural GPD "threshold stability" workflow) left the frozen
`pseudo_count` prior's absolute (mean, second_moment) pair anchored to the OLD threshold, silently
corrupting the blend into a degenerate fit with no diagnostic signal. `.loc` is now a property whose
setter re-anchors the prior to the new origin, matching what rebuilding the prior distribution at the
new loc and calling `.estimator(pseudo_count)` again would give.

T1-02: `GeneralizedParetoAccumulator.combine()` given a foreign/plain-tuple partial (real observations,
no `.max_x` payload) taints the tracked max and used to make `value()` fall back to a plain tuple,
silently discarding that a `has_prior=True` accumulator's max-tracking protection -- supposed to be
unconditional -- could not be applied. `value()` now still reports a `GeneralizedParetoSuffStat` with
`max_x=None` and `max_unverified=True` in that case, and `estimate()` discloses the ambiguity via
`numerical_repairs()` instead of silently reproducing the T1-01 self-inconsistent-fit crash class.
"""

import unittest

from mixle.inference import estimate
from mixle.stats import GeneralizedParetoDistribution, GeneralizedParetoEstimator


class GeneralizedParetoPriorRetargetTest(unittest.TestCase):
    def test_mutating_loc_after_estimator_reanchors_the_prior_to_match_a_rebuilt_one(self):
        # Exact repro from the filed finding.
        raw_dist = GeneralizedParetoDistribution(1.0, 0.5, loc=0.0)
        data0 = raw_dist.sampler(seed=3).sample(40)
        prior = GeneralizedParetoDistribution(1.0, -0.9, loc=0.0)
        est = prior.estimator(pseudo_count=2000.0)  # built once, at loc=0

        offset = 1_700_000_000.0
        data_shifted = (data0 + offset).tolist()
        est.loc = offset  # re-target the SAME estimator at a new threshold

        m = estimate(data_shifted, est)
        # Before the fix this was a degenerate fit: shape=0.0, scale=1e-12 (the min_scale floor).
        self.assertLess(m.shape, 0.0)
        self.assertGreater(m.scale, 1.0e-6)

        # Ground truth: rebuild the prior at the new threshold instead of mutating .loc.
        prior_correct = GeneralizedParetoDistribution(1.0, -0.9, loc=offset)
        est_correct = prior_correct.estimator(pseudo_count=2000.0)
        m_correct = estimate(data_shifted, est_correct)

        self.assertAlmostEqual(m.shape, m_correct.shape, places=9)
        self.assertAlmostEqual(m.scale, m_correct.scale, places=6)

    def test_loc_setter_round_trip_is_a_no_op(self):
        # Setting .loc back to the value it already was must reproduce the un-mutated fit exactly
        # (not just approximately) -- the re-anchoring math should not perturb the identity case.
        prior = GeneralizedParetoDistribution(1.0, -0.9, loc=0.0)
        est = prior.estimator(pseudo_count=500.0)
        mean0_before, second0_before = float(est.suff_stat[0]), float(est.suff_stat[1])

        est.loc = 0.0  # no-op re-target

        self.assertEqual(float(est.suff_stat[0]), mean0_before)
        self.assertEqual(float(est.suff_stat[1]), second0_before)

    def test_ordinary_no_prior_loc_mutation_still_works_as_a_plain_threshold_change(self):
        # No pseudo_count prior configured: mutating .loc must behave exactly like the historical
        # bare-attribute assignment, with no re-anchoring machinery involved.
        est = GeneralizedParetoEstimator(loc=0.0)
        est.loc = 5.0
        self.assertEqual(est.loc, 5.0)
        self.assertIsNone(est.suff_stat)

        true_dist = GeneralizedParetoDistribution(2.0, 0.3, loc=5.0)
        data = true_dist.sampler(seed=11).sample(5000).tolist()
        m = estimate(data, est)
        self.assertEqual(m.numerical_repairs(), ())
        self.assertAlmostEqual(m.scale, 2.0, delta=0.2)
        self.assertAlmostEqual(m.shape, 0.3, delta=0.08)


class GeneralizedParetoForeignCombineTest(unittest.TestCase):
    def test_foreign_combine_on_prior_carrying_accumulator_discloses_unverified_support(self):
        # Exact repro from the filed finding: a "foreign" partial statistic (a plain (sum, sum2,
        # count) tuple with real observations but no .max_x payload -- e.g. a hand-assembled stat
        # from an external integration) combined into a has_prior accumulator.
        raw_dist = GeneralizedParetoDistribution(1.0, 0.5, loc=0.0)
        data = raw_dist.sampler(seed=3).sample(40).tolist()
        prior = GeneralizedParetoDistribution(1.0, -0.9, loc=0.0)
        est = prior.estimator(pseudo_count=2000.0)

        acc = est.accumulator_factory().make()
        self.assertTrue(acc.has_prior)

        foreign_tuple = (sum(data), sum(x * x for x in data), float(len(data)))
        acc.combine(foreign_tuple)
        self.assertTrue(acc._max_tainted)

        suff_stat = acc.value()
        self.assertTrue(getattr(suff_stat, "max_unverified", False))
        self.assertIsNone(suff_stat.max_x)

        m = est.estimate(float(len(data)), suff_stat)
        self.assertLess(m.shape, 0.0)  # the prior still drives shape negative
        # Before the fix, numerical_repairs() was empty here even though the fit is self-inconsistent
        # (implied upper endpoint below the training data's own max) -- fully undisclosed.
        self.assertTrue(any("support-consistency-unverified" in note for note in m.numerical_repairs()))

    def test_ordinary_same_factory_combine_never_discloses_the_unverified_note(self):
        # A same-factory combine() always carries the max payload, so the accumulator's protection
        # is never actually degraded here -- must not spuriously disclose an ambiguity that doesn't
        # exist.
        raw_dist = GeneralizedParetoDistribution(1.0, 0.5, loc=0.0)
        data = raw_dist.sampler(seed=3).sample(40).tolist()
        prior = GeneralizedParetoDistribution(1.0, -0.9, loc=0.0)
        est = prior.estimator(pseudo_count=2000.0)

        acc_a = est.accumulator_factory().make()
        acc_b = est.accumulator_factory().make()
        for x in data[:20]:
            acc_a.update(x, 1.0, None)
        for x in data[20:]:
            acc_b.update(x, 1.0, None)
        acc_a.combine(acc_b.value())
        suff_stat = acc_a.value()
        self.assertFalse(getattr(suff_stat, "max_unverified", False))

        m = est.estimate(40.0, suff_stat)
        self.assertLess(m.shape, 0.0)
        self.assertFalse(any("support-consistency-unverified" in note for note in m.numerical_repairs()))

    def test_foreign_combine_that_never_needs_the_clamp_stays_quiet(self):
        # A weak, positive-shape prior blends to a final shape>=0 fit, which has infinite support and
        # never needed the max in the first place -- the ambiguity note must not fire when it would
        # not have mattered.
        raw_dist = GeneralizedParetoDistribution(1.0, 0.5, loc=0.0)
        data = raw_dist.sampler(seed=3).sample(40).tolist()
        prior = GeneralizedParetoDistribution(1.0, 0.5, loc=0.0)
        est = prior.estimator(pseudo_count=1.0)

        acc = est.accumulator_factory().make()
        foreign_tuple = (sum(data), sum(x * x for x in data), float(len(data)))
        acc.combine(foreign_tuple)
        suff_stat = acc.value()

        m = est.estimate(float(len(data)), suff_stat)
        self.assertGreaterEqual(m.shape, 0.0)
        self.assertEqual(m.numerical_repairs(), ())


if __name__ == "__main__":
    unittest.main()
