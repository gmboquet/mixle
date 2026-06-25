"""Extreme-value & boundary estimation (pysp.stats.extreme)."""

import unittest

import numpy as np

from pysp.analysis import (
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
