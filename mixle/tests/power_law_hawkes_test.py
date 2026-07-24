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


if __name__ == "__main__":
    unittest.main()
