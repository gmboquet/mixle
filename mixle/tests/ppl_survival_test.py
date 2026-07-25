"""Censored / truncated maximum-likelihood fitting and Kaplan-Meier (mixle.ppl.survival)."""

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import mixle.ppl as P
from mixle.ppl import free
from mixle.ppl.survival import censored_loglik, fit_censored, kaplan_meier
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.uniform import UniformDistribution
from mixle.stats.univariate.continuous.weibull import WeibullDistribution


class CensoredFitTest(unittest.TestCase):
    def test_truncated_censoring_likelihoods_are_analytic(self):
        dist = UniformDistribution(0.0, 1.0)
        self.assertAlmostEqual(censored_loglik(dist, [0.5], event=[1]), 0.0)
        self.assertAlmostEqual(censored_loglik(dist, [0.75], event=[0]), np.log(0.25))
        self.assertAlmostEqual(
            censored_loglik(dist, [0.5], event=[1], lower=0.2, upper=0.8),
            np.log(1.0 / 0.6),
        )
        self.assertAlmostEqual(
            censored_loglik(dist, [0.75], event=[0], upper=0.8),
            np.log(0.05 / 0.8),
        )
        self.assertAlmostEqual(
            censored_loglik(dist, [0.75], event=[0], lower=0.2),
            np.log(0.25 / 0.8),
        )
        self.assertEqual(censored_loglik(dist, [0.8], event=[0], upper=0.8), -np.inf)

    def test_right_censored_weibull_recovers_params(self):
        # naive fit (treating censored as events) is biased; the censored fit recovers the truth
        x = np.asarray(WeibullDistribution(1.5, 10.0).sampler(seed=0).sample(2000))
        time = np.minimum(x, 12.0)
        event = x <= 12.0
        self.assertGreater(float((~event).mean()), 0.1)  # there is real censoring
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = fit_censored(P.Weibull(free, free), time, event=event).summary()
            naive = P.Weibull(free, free).fit(time, how="em").summary()
        self.assertLess(abs(s["scale"] - 10.0), 1.0)
        self.assertLess(abs(s["shape"] - 1.5), 0.3)
        self.assertLess(naive["scale"], s["scale"])  # naive underestimates the scale

    def test_truncated_exponential_recovers_rate(self):
        ex = np.asarray(ExponentialDistribution(3.0).sampler(seed=1).sample(20000))  # mean 3, rate 1/3
        trunc = ex[ex > 2.0][:4000]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = fit_censored(P.Exponential(free), trunc, lower=2.0).summary()
            naive = P.Exponential(free).fit(trunc, how="em").summary()
        self.assertLess(abs(s["rate"] - 1.0 / 3.0), 0.05)  # recovers the true rate
        self.assertLess(naive["rate"], s["rate"])  # ignoring truncation underestimates the rate

    def test_normal_censoring_reduces_bias(self):
        g = np.asarray(GaussianDistribution(0.5, 1.0).sampler(seed=2).sample(3000))
        observed = g <= 1.5
        tg = np.minimum(g, 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cens = fit_censored(P.Normal(free, free), tg, event=observed).summary()
            naive = P.Normal(free, free).fit(tg, how="em").summary()
        self.assertLess(abs(cens["mean"] - 0.5), abs(naive["mean"] - 0.5))

    def test_optimizer_receipt_and_failure_are_explicit(self):
        fitted = fit_censored(
            P.Exponential(free),
            [0.5, 1.0, 2.0],
            event=[1, 1, 0],
            seed=7,
            n_starts=1,
        )
        self.assertTrue(fitted.result.converged)
        self.assertEqual(fitted.result.seed, 7)
        self.assertEqual(fitted.result.starts, 1)
        self.assertTrue(np.isfinite(fitted.result.objective))

        failed = SimpleNamespace(
            success=False,
            x=np.array([0.0]),
            message="forced failure",
            nit=2,
            nfev=3,
        )
        with patch("scipy.optimize.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "forced failure"):
                fit_censored(P.Exponential(free), [1.0, 2.0], seed=3, n_starts=1)

    def test_seed_controls_reproducible_multistart_initialization(self):
        captured = []

        def successful(objective, start, **kwargs):
            captured.append(np.asarray(start).copy())
            return SimpleNamespace(
                success=True,
                x=np.asarray(start),
                message="ok",
                nit=1,
                nfev=1,
            )

        with patch("scipy.optimize.minimize", side_effect=successful):
            fit_censored(P.Exponential(free), [1.0, 2.0, 3.0], seed=1, n_starts=2)
            seed_one_start = captured[1].copy()
            captured.clear()
            fit_censored(P.Exponential(free), [1.0, 2.0, 3.0], seed=2, n_starts=2)
            seed_two_start = captured[1].copy()
        self.assertFalse(np.array_equal(seed_one_start, seed_two_start))

    def test_survival_inputs_and_truncation_are_strict(self):
        dist = UniformDistribution(0.0, 1.0)
        invalid = [
            ([], None, None, None),
            ([0.1, 0.2], [1], None, None),
            ([0.1], [2], None, None),
            ([0.1], ["event"], None, None),
            ([np.nan], [1], None, None),
            ([0.1], [1], 0.8, 0.2),
            ([0.1], [1], 0.2, None),
            ([0.9], [0], None, 0.8),
        ]
        for times, events, lower, upper in invalid:
            with self.subTest(times=times, events=events, lower=lower, upper=upper), self.assertRaises(
                ValueError
            ):
                censored_loglik(dist, times, event=events, lower=lower, upper=upper)


class KaplanMeierTest(unittest.TestCase):
    def test_monotone_in_unit_interval(self):
        x = np.asarray(WeibullDistribution(1.5, 10.0).sampler(seed=0).sample(500))
        time = np.minimum(x, 12.0)
        event = x <= 12.0
        km = kaplan_meier(time, event)
        self.assertTrue(np.all(np.diff(km["survival"]) <= 1e-12))  # non-increasing
        self.assertGreaterEqual(km["survival"].min(), 0.0)
        self.assertLessEqual(km["survival"].max(), 1.0)
        self.assertTrue(np.all(km["at_risk"] >= km["events"]))

    def test_rejects_malformed_survival_data(self):
        for times, events in [
            ([], None),
            ([1.0, 2.0], [1]),
            ([1.0], [2]),
            ([1.0], ["yes"]),
            ([np.inf], [1]),
        ]:
            with self.subTest(times=times, events=events), self.assertRaises(ValueError):
                kaplan_meier(times, events)


if __name__ == "__main__":
    unittest.main()
