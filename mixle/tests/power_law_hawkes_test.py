"""Power-law (Omori) marked Hawkes process: intensity, clustering, MLE recovery, forecasting."""

import unittest
import warnings

import numpy as np

from mixle.stats import ExponentialDistribution, GaussianDistribution
from mixle.stats.processes.power_law_hawkes import PowerLawHawkesDistribution as PLH


class PowerLawHawkesTest(unittest.TestCase):
    def setUp(self):
        self.d = PLH(
            mu=0.2, A=4.0, c=0.02, p=1.3, window=2000.0, alpha=1.2, mark_dist=ExponentialDistribution(1.0 / np.log(10))
        )

    def test_intensity_spikes_then_power_law_decays(self):
        spike = self.d.intensity(10.01, [10.0], [5.0])
        later = self.d.intensity(15.0, [10.0], [5.0])
        self.assertGreater(spike, 50 * self.d.mu)
        self.assertGreater(spike, later)
        self.assertGreater(later, self.d.mu - 1e-9)

    def test_branching_ratio_subcritical(self):
        self.assertTrue(0 < self.d.branching_ratio(0.5) < 1)

    def test_sampler_is_clustered(self):
        ts, _ = self.d.sampler(seed=1).sample()
        counts = np.histogram(ts, bins=200)[0]
        self.assertGreater(counts.var() / counts.mean(), 1.5)  # overdispersed vs Poisson

    def test_mle_recovers_parameters(self):
        ts, ms = self.d.sampler(seed=1).sample()
        est = self.d.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update([(ts, ms)], None, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = est.estimate(None, acc.value())
        self.assertAlmostEqual(fit.mu, 0.2, delta=0.1)
        self.assertAlmostEqual(fit.alpha, 1.2, delta=0.3)
        self.assertAlmostEqual(fit.branching_ratio(ms.mean()), self.d.branching_ratio(ms.mean()), delta=0.15)

    def test_forecast_elevated_after_a_large_mark(self):
        busy = self.d.expected_count(10, 11, [10.0], [5.0])
        quiet = self.d.expected_count(500, 501, [10.0], [5.0])
        self.assertGreater(busy, 10 * quiet)
        self.assertAlmostEqual(quiet, self.d.mu, delta=0.05)

    def test_unmarked_process(self):
        du = PLH(mu=0.5, A=2.0, c=0.05, p=1.4, window=1500.0)  # no mark_dist -> marks are 0
        ts, ms = du.sampler(seed=0).sample()
        self.assertTrue(np.all(ms == 0.0))
        self.assertTrue(np.isfinite(du.log_density((ts, ms))))

    def test_marked_log_density_includes_mark_law(self):
        # log_density previously scored only the temporal (excitation-kernel) term and silently
        # dropped the mark law entirely, so two realizations differing only in one mark value scored
        # identically. Reviewer's exact repro: under a standard Gaussian mark model, changing one
        # mark from 0 to 100 must change the score by log N(0;0,1) - log N(100;0,1) = 100^2/2 = 5000.
        d = PLH(mu=0.5, A=1.0, c=0.1, p=1.5, window=100.0, alpha=0.0, mark_dist=GaussianDistribution(0.0, 1.0))
        times = [10.0, 20.0, 30.0]
        ll_low = d.log_density((times, [0.0, 0.0, 0.0]))
        ll_high = d.log_density((times, [0.0, 0.0, 100.0]))
        self.assertAlmostEqual(ll_low - ll_high, 5000.0, delta=1.0)

    def test_seq_log_density_includes_mark_law(self):
        # seq_log_density delegates to log_density per realization, so the fix must show up there too.
        d = PLH(mu=0.5, A=1.0, c=0.1, p=1.5, window=100.0, alpha=0.0, mark_dist=GaussianDistribution(0.0, 1.0))
        times = [10.0, 20.0, 30.0]
        lls = d.seq_log_density([(times, [0.0, 0.0, 0.0]), (times, [0.0, 0.0, 100.0])])
        self.assertAlmostEqual(lls[0] - lls[1], 5000.0, delta=1.0)

    def test_unmarked_process_scores_ignore_mark_values(self):
        # Negative control for the mark-log-density fix: with no mark_dist and alpha=0.0 there is no
        # mark law and no excitation sensitivity, so log_density must stay independent of the marks.
        du = PLH(mu=0.5, A=2.0, c=0.05, p=1.4, window=1500.0)  # no mark_dist -> marks are 0
        times = [10.0, 20.0, 30.0]
        ll_a = du.log_density((times, [0.0, 0.0, 0.0]))
        ll_b = du.log_density((times, [3.0, -7.0, 42.0]))
        self.assertAlmostEqual(ll_a, ll_b, places=9)

    def test_mismatched_mark_length_raises(self):
        # A mark array shorter than the times array previously either raised an opaque numpy
        # broadcasting error or -- when the mark count happened to be 1 and broadcast cleanly --
        # silently returned a finite score computed from misaligned data.
        with self.assertRaises(ValueError):
            self.d.log_density(([10.0, 20.0, 30.0], [0.0, 0.0]))  # 2 marks, 3 times
        with self.assertRaises(ValueError):
            self.d.log_density(([10.0, 20.0, 30.0], [0.0]))  # single mark silently broadcast before the fix
        with self.assertRaises(ValueError):
            self.d.log_density(([10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 0.0]))  # 4 marks, 3 times

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            PLH(mu=0.2, A=1.0, c=0.1, p=0.5, window=100.0)  # p must exceed 1

    def test_estimator_preserves_name_and_keys(self):
        d = PLH(mu=0.2, A=1.0, c=0.1, p=1.5, window=100.0, name="my-hawkes", keys="shared-key")
        est = d.estimator()
        self.assertEqual(est.name, "my-hawkes")
        self.assertEqual(est.keys, "shared-key")
        acc = est.accumulator_factory().make()
        self.assertEqual(acc.name, "my-hawkes")
        self.assertEqual(acc.keys, "shared-key")

    def test_accumulator_value_is_not_aliased_to_later_updates(self):
        # value() previously returned self.realizations directly, so a value() taken before a later
        # update()/seq_update()/combine() on the SAME accumulator would silently grow too (the exact
        # aliasing hazard from_value's own copy-on-restore comment guards against, on the other side).
        est = self.d.estimator()
        acc = est.accumulator_factory().make()
        ts, ms = self.d.sampler(seed=2).sample()
        acc.update((ts, ms), 1.0, None)
        snapshot = acc.value()
        self.assertEqual(len(snapshot[0]), 1)
        acc.update((ts, ms), 1.0, None)
        self.assertEqual(len(snapshot[0]), 1, "value() snapshot mutated by a later update()")
        self.assertEqual(len(acc.value()[0]), 2)

    def test_accumulator_update_uses_weight(self):
        # update() previously stored every realization identically regardless of its weight.
        est = self.d.estimator()
        ts, ms = self.d.sampler(seed=3).sample()
        acc_low = est.accumulator_factory().make()
        acc_low.update((ts, ms), 0.1, None)
        acc_high = est.accumulator_factory().make()
        acc_high.update((ts, ms), 100.0, None)
        self.assertEqual(acc_low.value()[0][0][1], 0.1)
        self.assertEqual(acc_high.value()[0][0][1], 100.0)

    def test_accumulator_seq_update_uses_weights(self):
        est = self.d.estimator()
        ts, ms = self.d.sampler(seed=3).sample()
        acc = est.accumulator_factory().make()
        acc.seq_update([(ts, ms), (ts, ms)], [0.25, 4.0], None)
        weights = [w for _, w in acc.value()[0]]
        self.assertEqual(weights, [0.25, 4.0])

    def test_accumulator_seq_update_weights_default_to_one_when_none(self):
        # test_mle_recovers_parameters calls seq_update(..., None, ...); confirm that path treats a
        # missing weight vector as uniform weight 1.0 rather than raising or silently dropping data.
        est = self.d.estimator()
        ts, ms = self.d.sampler(seed=3).sample()
        acc = est.accumulator_factory().make()
        acc.seq_update([(ts, ms), (ts, ms)], None, None)
        weights = [w for _, w in acc.value()[0]]
        self.assertEqual(weights, [1.0, 1.0])

    def test_scale_multiplies_stored_weights(self):
        # scale() previously was a documented no-op; it must now multiply every stored weight by c,
        # matching the sibling Hawkes accumulators' scale() contract (e.g. HawkesProcessAccumulator).
        est = self.d.estimator()
        ts, ms = self.d.sampler(seed=3).sample()
        acc = est.accumulator_factory().make()
        acc.update((ts, ms), 1.0, None)
        acc.update((ts, ms), 3.0, None)
        acc.scale(2.0)
        weights = [w for _, w in acc.value()[0]]
        self.assertEqual(weights, [2.0, 6.0])

    def test_scale_does_not_alter_realization_data(self):
        # scale() must only touch the linear weight, never the raw event catalogue, window, or the
        # shared alpha constraint -- those are data/config, not weighted sufficient statistics.
        est = self.d.estimator()
        ts, ms = self.d.sampler(seed=3).sample()
        acc = est.accumulator_factory().make()
        acc.update((ts, ms), 1.0, None)
        before = acc.value()
        before_times, before_marks = before[0][0][0]
        acc.scale(5.0)
        after = acc.value()
        after_times, after_marks = after[0][0][0]
        np.testing.assert_array_equal(before_times, after_times)
        np.testing.assert_array_equal(before_marks, after_marks)
        self.assertEqual(before[1], after[1])  # window
        self.assertEqual(before[2], after[2])  # alpha_fixed

    def test_scale_by_one_is_identity(self):
        # Negative control for the scale fix: scaling by 1.0 must leave the weight unchanged.
        est = self.d.estimator()
        ts, ms = self.d.sampler(seed=7).sample()
        acc = est.accumulator_factory().make()
        acc.update((ts, ms), 2.5, None)
        before = acc.value()[0][0][1]
        acc.scale(1.0)
        after = acc.value()[0][0][1]
        self.assertEqual(before, after)

    def test_zero_weight_realization_does_not_affect_estimate(self):
        # End-to-end: a weight-0 realization must be equivalent to not having observed it at all.
        # Both accumulators produce mathematically identical weighted-MLE objectives (0 * finite log
        # density contributes exactly 0.0), so the fitted parameters must match tightly.
        ts1, ms1 = self.d.sampler(seed=5).sample()
        ts2, ms2 = self.d.sampler(seed=6).sample()
        est = self.d.estimator()

        acc_with_noise = est.accumulator_factory().make()
        acc_with_noise.update((ts1, ms1), 1.0, None)
        acc_with_noise.update((ts2, ms2), 0.0, None)  # should contribute nothing

        acc_without_noise = est.accumulator_factory().make()
        acc_without_noise.update((ts1, ms1), 1.0, None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_with = est.estimate(None, acc_with_noise.value())
            fit_without = est.estimate(None, acc_without_noise.value())

        self.assertAlmostEqual(fit_with.mu, fit_without.mu, places=6)
        self.assertAlmostEqual(fit_with.A, fit_without.A, places=6)
        self.assertAlmostEqual(fit_with.c, fit_without.c, places=6)
        self.assertAlmostEqual(fit_with.p, fit_without.p, places=6)

    def test_weighted_estimate_matches_replicated_unweighted_estimate(self):
        # Resampling-equivalence guarantee: weighting one realization by 3.0 must fit the same as
        # observing it three separate times with weight 1.0 each.
        ts, ms = self.d.sampler(seed=4).sample()
        est = self.d.estimator()

        acc_weighted = est.accumulator_factory().make()
        acc_weighted.update((ts, ms), 3.0, None)

        acc_replicated = est.accumulator_factory().make()
        for _ in range(3):
            acc_replicated.update((ts, ms), 1.0, None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_weighted = est.estimate(None, acc_weighted.value())
            fit_replicated = est.estimate(None, acc_replicated.value())

        self.assertAlmostEqual(fit_weighted.mu, fit_replicated.mu, places=6)
        self.assertAlmostEqual(fit_weighted.A, fit_replicated.A, places=6)
        self.assertAlmostEqual(fit_weighted.c, fit_replicated.c, places=6)
        self.assertAlmostEqual(fit_weighted.p, fit_replicated.p, places=6)


if __name__ == "__main__":
    unittest.main()
