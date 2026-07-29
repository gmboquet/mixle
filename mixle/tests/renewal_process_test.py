"""WS-2: renewal process -- i.i.d. inter-arrivals from a base distribution on a censored window."""

import unittest

import numpy as np

from mixle.stats.processes.renewal_process import RenewalProcessDistribution
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class _FixedGapSampler:
    def __init__(self, value):
        self.value = value

    def sample(self, size=None, *, batched=True):
        if size is None:
            return self.value
        return [self.value] * size


class _FixedGapExponential(ExponentialDistribution):
    def __init__(self, value):
        super().__init__(1.0)
        self.value = value

    def sampler(self, seed=None):
        return _FixedGapSampler(self.value)


class _BadCDFExponential(ExponentialDistribution):
    def cdf(self, x):
        if np.isclose(x, 5.0):
            return -1.0
        return super().cdf(x)


class RenewalProcessTest(unittest.TestCase):
    def _truth(self):
        return RenewalProcessDistribution(GammaDistribution(k=3.0, theta=0.5), window=200.0)

    def test_sample_and_score(self):
        truth = self._truth()
        data = truth.sampler(seed=1).sample(20)
        # realizations are sorted event-time arrays within the window
        for d in data:
            self.assertTrue(np.all(np.diff(d) > 0) if len(d) > 1 else True)
            self.assertTrue(len(d) == 0 or d[-1] <= truth.window)
        self.assertTrue(np.isfinite(truth.log_density(data[0])))

    def test_seq_log_density_matches_scalar(self):
        truth = self._truth()
        data = truth.sampler(seed=2).sample(15)
        enc = truth.dist_to_encoder().seq_encode(data)
        seq = truth.seq_log_density(enc)
        scalar = np.array([truth.log_density(d) for d in data])
        self.assertTrue(np.allclose(seq, scalar, atol=1e-8))

    def test_out_of_window_events_score_neg_inf(self):
        truth = self._truth()
        self.assertEqual(truth.log_density(np.array([10.0, 250.0])), -np.inf)  # 250 > window

    def test_recovers_interarrival_parameters(self):
        truth = self._truth()
        data = truth.sampler(seed=3).sample(60)
        # Recover via the estimator's direct M-step (the closed-form full-data MLE) rather than fit():
        # this is fully deterministic and independent of any global init/engine state a parallel test
        # runner might leave behind.
        est = truth.estimator()
        acc = est.accumulator_factory().make()
        enc = truth.dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data), dtype=np.float64), None)
        model = est.estimate(None, acc.value())
        # consistent: recover Gamma shape/scale from the ~8000 observed gaps
        self.assertAlmostEqual(model.interarrival.k, 3.0, delta=0.4)
        self.assertAlmostEqual(model.interarrival.theta, 0.5, delta=0.1)
        self.assertEqual(model.window, 200.0)

    def test_cdf_results_must_be_probabilities(self):
        dist = RenewalProcessDistribution(
            _BadCDFExponential(1.0),
            window=10.0,
        )
        with self.assertRaisesRegex(ValueError, "non-probability"):
            dist.log_density([5.0])
        with self.assertRaises(TypeError):
            RenewalProcessDistribution(object(), window=10.0)
        with self.assertRaisesRegex(TypeError, "positive"):
            RenewalProcessDistribution(
                GaussianDistribution(0.0, 1.0),
                window=10.0,
            )

    def test_exponential_fit_includes_the_right_censored_tail(self):
        dist = RenewalProcessDistribution(
            ExponentialDistribution(1.0),
            window=10.0,
        )
        estimator = dist.estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update([1.0], 1.0, None)
        value = accumulator.value()
        self.assertEqual(value.schema_version, 1)
        fit = estimator.estimate(None, value)
        self.assertAlmostEqual(fit.interarrival.beta, 10.0, places=12)

        empty = estimator.accumulator_factory().make()
        empty.update([], 1.0, None)
        with self.assertRaisesRegex(ValueError, "no finite MLE"):
            estimator.estimate(None, empty.value())

    def test_accumulation_uses_the_same_strict_event_contract(self):
        dist = RenewalProcessDistribution(
            ExponentialDistribution(1.0),
            window=10.0,
        )
        accumulator = dist.estimator().accumulator_factory().make()
        before = accumulator.value()
        malformed = (
            [-1.0, 2.0],
            [2.0, np.nan],
            [2.0, 2.0],
            [3.0, 2.0],
            [2.0, 11.0],
            [0.0],
        )
        for events in malformed:
            with self.subTest(events=repr(events)), self.assertRaises(ValueError):
                accumulator.update(events, 1.0, None)
            after = accumulator.value()
            np.testing.assert_array_equal(
                after.completed_gaps,
                before.completed_gaps,
            )
            np.testing.assert_array_equal(
                after.censored_times,
                before.censored_times,
            )
        self.assertEqual(accumulator.acc_to_encoder().window, 10.0)

    def test_encoded_payload_is_bound_to_the_fixed_window(self):
        dist = RenewalProcessDistribution(
            ExponentialDistribution(1.0),
            window=10.0,
        )
        encoder = dist.dist_to_encoder()
        payload = list(encoder.seq_encode([[2.0], []]))
        payload[4] = payload[4].copy()
        payload[4][0] = 9.0
        with self.assertRaisesRegex(ValueError, "contradict"):
            dist.seq_log_density(tuple(payload))

        invalid = encoder.seq_encode([[2.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "invalid"):
            dist.estimator().accumulator_factory().make().seq_update(
                invalid,
                [1.0],
                None,
            )

    def test_sampler_rejects_bad_gaps_and_reports_budget_exhaustion(self):
        for gap in (0.0, -1.0, np.nan, np.inf):
            dist = RenewalProcessDistribution(
                _FixedGapExponential(gap),
                window=1.0,
            )
            with self.subTest(gap=repr(gap)), self.assertRaises((TypeError, ValueError)):
                dist.sampler(seed=1).sample()

        dist = RenewalProcessDistribution(
            _FixedGapExponential(0.1),
            window=10.0,
            max_events=2,
        )
        sampler = dist.sampler(seed=1)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            sampler.sample()
        self.assertFalse(sampler.last_receipt["complete"])
        self.assertEqual(
            sampler.last_receipt["termination_reason"],
            "event_budget_exhausted",
        )

    def test_statistics_validate_alignment_before_combine(self):
        dist = RenewalProcessDistribution(
            ExponentialDistribution(1.0),
            window=10.0,
        )
        accumulator = dist.estimator().accumulator_factory().make()
        accumulator.update([1.0], 1.0, None)
        value = list(accumulator.value())
        value[2] = np.asarray([])
        with self.assertRaisesRegex(ValueError, "align"):
            dist.estimator().accumulator_factory().make().combine(tuple(value))


if __name__ == "__main__":
    unittest.main()
