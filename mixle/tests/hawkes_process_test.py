"""Univariate exponential-kernel Hawkes process: exact likelihood, sampler, and EM recovery."""

import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import (
    HawkesProcessDataEncoder,
    HawkesProcessDistribution,
    HawkesProcessEstimator,
)


def _brute_log_density(t, mu, alpha, beta, window):
    """Direct O(n^2) point-process log-likelihood, independent of the recursive implementation."""
    t = np.asarray(t, dtype=float)
    ll = 0.0
    for i in range(len(t)):
        lam = mu + alpha * np.sum(np.exp(-beta * (t[i] - t[:i])))
        ll += np.log(lam)
    compensator = mu * window + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (window - t)))
    return float(ll - compensator)


class HawkesLikelihoodTest(unittest.TestCase):
    def setUp(self):
        self.dist = HawkesProcessDistribution(mu=0.6, alpha=0.7, beta=1.3, window=50.0)
        self.data = [np.sort(self.dist.sampler(seed=s).sample()) for s in range(25)]

    def test_log_density_matches_brute_force(self):
        for s in self.data[:6]:
            self.assertAlmostEqual(self.dist.log_density(s), _brute_log_density(s, 0.6, 0.7, 1.3, 50.0), places=8)

    def test_seq_matches_scalar(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        seq = self.dist.seq_log_density(enc)
        scalar = np.array([self.dist.log_density(s) for s in self.data])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)

    def test_empty_realization(self):
        # no events: likelihood is exp(-mu*window)
        self.assertAlmostEqual(self.dist.log_density([]), -0.6 * 50.0, places=10)
        enc = self.dist.dist_to_encoder().seq_encode([[], []])
        np.testing.assert_allclose(self.dist.seq_log_density(enc), [-30.0, -30.0], atol=1e-9)

    def test_out_of_window_is_minus_inf(self):
        self.assertEqual(self.dist.log_density([10.0, 60.0]), -np.inf)  # 60 > window
        self.assertEqual(self.dist.log_density([5.0, 3.0]), -np.inf)  # not sorted

    def test_tied_timestamps_rejected(self):
        # The conditional intensity is defined over the strict history t_j < t (see class docstring), so
        # a tied timestamp pair is not a valid t_1 < ... < t_n realization. Before the fix, the O(n)
        # recursion silently treated a zero-width gap the same as any other gap, letting the second of two
        # simultaneous events be excited by the first (t_j <= t in effect, not t_j < t): log_density([10.0,
        # 10.0]) returned a finite value equal to log_density([10.0, 10.0 + 1e-9]) instead of being
        # rejected like any other malformed ordering.
        self.assertEqual(self.dist.log_density([10.0, 10.0]), -np.inf)
        # a genuinely tied pair should score strictly worse than an arbitrarily-close-but-valid ordering,
        # since the tied realization is now rejected outright rather than silently scored
        self.assertNotEqual(self.dist.log_density([10.0, 10.0]), self.dist.log_density([10.0, 10.0 + 1e-9]))
        with self.assertRaises(ValueError):
            self.dist.dist_to_encoder().seq_encode([[10.0, 10.0]])
        # three events with a tie among the last two
        self.assertEqual(self.dist.log_density([1.0, 10.0, 10.0]), -np.inf)

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            HawkesProcessDistribution(mu=-1.0, alpha=0.5, beta=1.0, window=10.0)
        with self.assertRaises(ValueError):
            HawkesProcessDistribution(mu=1.0, alpha=0.5, beta=-1.0, window=10.0)
        with self.assertRaises(ValueError):
            HawkesProcessDistribution(mu=1.0, alpha=0.5, beta=1.0, window=0.0)
        for field in ("mu", "alpha", "beta", "window"):
            values = dict(mu=1.0, alpha=0.5, beta=1.0, window=10.0)
            values[field] = np.inf
            with self.subTest(field=repr(field)), self.assertRaises(ValueError):
                HawkesProcessDistribution(**values)

    def test_encoded_schema_is_bound_to_model(self):
        encoded = self.dist.dist_to_encoder().seq_encode([[1.0, 2.0], []])
        with self.assertRaises(ValueError):
            self.dist.seq_log_density((encoded[0], encoded[1], 500.0))
        bad_padding = encoded[0].copy()
        bad_padding[1, 0] = 0.0
        with self.assertRaises(ValueError):
            self.dist.seq_log_density((bad_padding, encoded[1], encoded[2]))
        with self.assertRaises(ValueError):
            self.dist.seq_log_density((encoded[0], np.array([3, 0]), encoded[2]))

    def test_string_round_trip(self):
        d = HawkesProcessDistribution(0.6, 0.7, 1.3, 50.0, name="h", keys="k")
        self.assertEqual(str(eval(str(d))), str(d))

    def test_encoder_equality(self):
        e1 = HawkesProcessDataEncoder(50.0)
        e2 = self.dist.dist_to_encoder()
        self.assertEqual(e1, e2)
        self.assertNotEqual(e1, HawkesProcessDataEncoder(51.0))


class HawkesSamplerTest(unittest.TestCase):
    def test_empirical_rate_matches_stationary_rate(self):
        # stationary intensity of a sub-critical Hawkes is mu / (1 - alpha/beta)
        d = HawkesProcessDistribution(mu=0.5, alpha=0.8, beta=1.6, window=4000.0)
        samples = d.sampler(seed=3).sample(20)
        empirical = np.mean([s.size for s in samples]) / 4000.0
        theoretical = 0.5 / (1.0 - 0.8 / 1.6)  # = 1.0
        self.assertAlmostEqual(empirical, theoretical, delta=0.05)

    def test_samples_are_sorted_and_in_window(self):
        d = HawkesProcessDistribution(0.5, 0.6, 1.5, 100.0)
        for s in d.sampler(seed=0).sample(10):
            self.assertTrue(np.all(np.diff(s) >= 0.0))
            self.assertTrue(s.size == 0 or (s[0] >= 0.0 and s[-1] <= 100.0))

    def test_event_budget_fails_with_receipt(self):
        sampler = HawkesProcessDistribution(
            100.0,
            0.0,
            1.0,
            1.0,
            max_events=0,
        ).sampler(seed=0)
        with self.assertRaises(RuntimeError):
            sampler.sample()
        self.assertIsNotNone(sampler.last_receipt)
        self.assertFalse(sampler.last_receipt["complete"])
        self.assertEqual(
            sampler.last_receipt["termination_reason"],
            "event_budget_exhausted",
        )


class HawkesEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.estimator = HawkesProcessEstimator(window=10.0)
        self.accumulator = self.estimator.accumulator_factory().make()

    def test_statistics_are_validated_versioned_snapshots(self):
        events = np.array([1.0, 2.0])
        self.accumulator.update(events, 1.0, None)
        events[0] = 9.0
        value = self.accumulator.value()
        self.assertEqual(value.schema_version, 1)
        np.testing.assert_array_equal(value.realizations[0], [1.0, 2.0])
        self.assertFalse(value.realizations[0].flags.writeable)
        self.assertFalse(value.weights.flags.writeable)
        with self.assertRaises(ValueError):
            self.accumulator.combine((value.realizations, value.weights, 20.0))

    def test_invalid_evidence_and_absent_fit_fail_closed(self):
        with self.assertRaises(ValueError):
            self.accumulator.update([2.0, 1.0], 1.0, None)
        with self.assertRaises(ValueError):
            self.accumulator.update([1.0], -1.0, None)
        with self.assertRaises(ValueError):
            self.estimator.estimate(None, self.accumulator.value())
        self.accumulator.update([], 1.0, None)
        with self.assertRaises(ValueError):
            self.estimator.estimate(None, self.accumulator.value())

    def test_encoded_weight_alignment_is_exact(self):
        encoded = HawkesProcessDataEncoder(10.0).seq_encode([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            self.accumulator.seq_update(encoded, np.ones(1), None)


class HawkesEMTest(unittest.TestCase):
    def test_em_recovers_parameters(self):
        truth = HawkesProcessDistribution(mu=0.5, alpha=0.6, beta=1.2, window=80.0)
        data = [np.sort(truth.sampler(seed=s).sample()) for s in range(120)]
        fit = optimize(
            data, HawkesProcessEstimator(window=80.0), max_its=80, rng=np.random.RandomState(0), print_iter=0
        )
        self.assertAlmostEqual(fit.mu, 0.5, delta=0.08)
        self.assertAlmostEqual(fit.branching_ratio, 0.5, delta=0.08)
        self.assertAlmostEqual(fit.beta, 1.2, delta=0.2)

    def test_em_is_monotone(self):
        truth = HawkesProcessDistribution(mu=0.5, alpha=0.6, beta=1.2, window=80.0)
        data = [np.sort(truth.sampler(seed=s).sample()) for s in range(60)]
        enc = truth.dist_to_encoder().seq_encode(data)
        lls = []
        for k in range(1, 9):
            m = optimize(
                data, HawkesProcessEstimator(window=80.0), max_its=k, rng=np.random.RandomState(0), print_iter=0
            )
            lls.append(float(np.sum(m.seq_log_density(enc))))
        self.assertTrue(all(lls[i] <= lls[i + 1] + 1.0 for i in range(len(lls) - 1)))

    def test_single_long_sequence_with_full_init(self):
        truth = HawkesProcessDistribution(mu=0.5, alpha=0.8, beta=1.6, window=2000.0)
        big = [np.sort(truth.sampler(seed=7).sample())]
        # Log-likelihood is already flat to 4 decimal places by ~iteration 55-60, and the mu/branching-ratio
        # deltas below have converged well within their thresholds by iteration 45 (mu delta ~0.06 vs. the
        # 0.15 budget here, and checked across many alternate data seeds to stay <=0.09 with the branching
        # ratio staying >0.4), so 45 iterations gives the same recovery claim as 100 for a fraction of the cost.
        fit = optimize(
            big,
            HawkesProcessEstimator(window=2000.0),
            max_its=45,
            init_p=1.0,
            rng=np.random.RandomState(0),
            print_iter=0,
        )
        # the branching-floor lets EM escape the alpha=0 Poisson absorbing state even from a sparse init
        self.assertGreater(fit.branching_ratio, 0.3)
        self.assertAlmostEqual(fit.mu, 0.5, delta=0.15)


if __name__ == "__main__":
    unittest.main()
