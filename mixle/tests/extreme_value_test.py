"""Extreme-value & boundary estimation (mixle.stats.extreme)."""

import unittest
from unittest import mock

import numpy as np

from mixle.analysis import (
    endpoint_estimator,
    gpd_fit,
    hill_estimator,
    mean_residual_life,
    moment_estimator,
    n_records,
    peaks_over_threshold,
    record_times,
    return_level,
)


def _rgpd(rng, n, xi, beta):
    u = rng.rand(n)
    if abs(xi) < 1e-9:
        return -beta * np.log(1 - u)
    return (beta / xi) * ((1 - u) ** (-xi) - 1)


class GPDTest(unittest.TestCase):
    def test_mle_recovers_parameters(self):
        rng = np.random.RandomState(0)
        z = _rgpd(rng, 6000, 0.3, 2.0)
        fit = gpd_fit(z, method="mle")
        self.assertAlmostEqual(fit.shape, 0.3, delta=0.08)
        self.assertAlmostEqual(fit.scale, 2.0, delta=0.3)

    def test_pwm_recovers_parameters(self):
        rng = np.random.RandomState(1)
        z = _rgpd(rng, 6000, 0.2, 1.5)
        fit = gpd_fit(z, method="pwm")
        self.assertAlmostEqual(fit.shape, 0.2, delta=0.1)
        self.assertAlmostEqual(fit.scale, 1.5, delta=0.3)

    def test_bounded_tail_finite_endpoint(self):
        rng = np.random.RandomState(2)
        z = _rgpd(rng, 8000, -0.25, 1.0)  # support endpoint at 4.0
        fit = gpd_fit(z, method="mle")
        self.assertLess(fit.shape, 0)
        self.assertTrue(np.isfinite(fit.endpoint))
        self.assertGreater(fit.endpoint, z.max())

    def test_pot_and_return_level(self):
        rng = np.random.RandomState(3)
        body = rng.normal(0, 1, 9000)
        tail = _rgpd(rng, 1000, 0.2, 1.5) + 3.0
        x = np.concatenate([body, tail])
        fit = peaks_over_threshold(x, 3.0)
        self.assertGreater(fit.n_exceedances, 500)
        rl = return_level(fit, 10000)
        self.assertGreater(rl, x.max() * 0.5)

    # -- MXR-080-0090: filtering must be receipted, not silent; fits must be validated. --

    def test_rejects_nan_exceedances(self):
        z = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with self.assertRaises(ValueError):
            gpd_fit(z, method="pwm")

    def test_receipts_dropped_nonpositive(self):
        z = np.array([1.0, 2.0, -3.0, 0.0, 4.0, 5.0])
        fit = gpd_fit(z, method="pwm")
        self.assertEqual(fit.n_exceedances, 4)
        self.assertEqual(fit.n_dropped_nonpositive, 2)

    def test_n_total_zero_not_silently_replaced(self):
        # Before the fix, `n_total or n` silently swapped an explicit 0 for n (0 is falsy). Now an
        # n_total inconsistent with the exceedance count is rejected outright instead of being
        # coerced into something else silently.
        with self.assertRaises(ValueError):
            gpd_fit(np.array([1.0, 2.0, 3.0, 4.0]), method="pwm", n_total=0)

    def test_n_total_explicit_value_preserved(self):
        fit = gpd_fit(np.array([1.0, 2.0, 3.0, 4.0]), method="pwm", n_total=10)
        self.assertEqual(fit.n_total, 10)

    def test_mle_nonconvergence_raises(self):
        bad_result = mock.Mock(success=False, message="did not converge", x=np.array([0.1, 1.0]))
        with mock.patch("mixle.analysis.extreme.optimize.minimize", return_value=bad_result):
            with self.assertRaises(ValueError):
                gpd_fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), method="mle")

    def test_pwm_rejects_support_violating_fit(self):
        # PWM is a closed-form moment match with no built-in domain constraint; on this sample it
        # returns a shape/scale whose implied endpoint is below the sample's own maximum -- a fit that
        # asserts its own fitting data was impossible.
        z = np.array([0.285569, 0.704949, 0.371616, 0.466864, 0.744378, 1.388677])
        with self.assertRaises(ValueError):
            gpd_fit(z, method="pwm")

    def test_well_behaved_sample_fits_cleanly(self):
        # Negative control for all of the above: ordinary data trips none of the new validation.
        rng = np.random.RandomState(4)
        z = _rgpd(rng, 4000, 0.25, 1.5)
        fit = gpd_fit(z, method="mle")
        self.assertEqual(fit.n_dropped_nonpositive, 0)
        self.assertTrue(np.isfinite(fit.shape))
        self.assertGreater(fit.scale, 0)
        fit_pwm = gpd_fit(z, method="pwm")
        self.assertEqual(fit_pwm.n_dropped_nonpositive, 0)
        self.assertGreater(fit_pwm.scale, 0)


class TailIndexTest(unittest.TestCase):
    def test_hill_recovers_pareto_index(self):
        rng = np.random.RandomState(0)
        # Pareto(alpha=3): xi = 1/3
        x = (1 - rng.rand(20000)) ** (-1 / 3.0)
        self.assertAlmostEqual(hill_estimator(x, 800), 1.0 / 3.0, delta=0.06)

    def test_moment_handles_negative_xi(self):
        rng = np.random.RandomState(1)
        z = _rgpd(rng, 8000, -0.25, 1.0)
        self.assertAlmostEqual(moment_estimator(np.sort(z), 800), -0.25, delta=0.1)

    def test_hill_invalid_k(self):
        with self.assertRaises(ValueError):
            hill_estimator(np.arange(1, 11.0), 0)


class EndpointTest(unittest.TestCase):
    def test_bounded_endpoint_exceeds_max(self):
        rng = np.random.RandomState(2)
        z = _rgpd(rng, 8000, -0.3, 1.0)  # endpoint ~ 3.33
        ep = endpoint_estimator(z, 800)
        self.assertTrue(np.isfinite(ep))
        self.assertGreater(ep, z.max())
        self.assertLess(ep, z.max() + 3.0)

    def test_heavy_tail_unbounded(self):
        rng = np.random.RandomState(3)
        x = (1 - rng.rand(8000)) ** (-1 / 2.0)  # heavy tail, xi>0
        self.assertEqual(endpoint_estimator(x, 800), float("inf"))


class MeanResidualLifeTest(unittest.TestCase):
    def test_increasing_for_heavy_tail(self):
        rng = np.random.RandomState(0)
        x = _rgpd(rng, 5000, 0.3, 2.0)
        mrl = mean_residual_life(x, np.array([0.0, 1.0, 2.0, 3.0]))
        # mean excess increases with threshold for a heavy (xi>0) tail
        self.assertTrue(np.all(np.diff(mrl["mean_excess"]) > 0))


class RecordsTest(unittest.TestCase):
    def test_record_times(self):
        x = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        np.testing.assert_array_equal(record_times(x), [0, 2, 4, 5])

    def test_expected_count_near_harmonic(self):
        rng = np.random.RandomState(0)
        counts = [n_records(rng.normal(0, 1, 500)) for _ in range(200)]
        h500 = np.sum(1.0 / np.arange(1, 501))
        self.assertAlmostEqual(np.mean(counts), h500, delta=0.5)


if __name__ == "__main__":
    unittest.main()
