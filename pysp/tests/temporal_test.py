"""Time/date modelling on raw timestamps: PeriodicTime + SeasonalTimeSeries.

All event times are built as pure POSIX seconds (``day * 86400`` is exactly midnight UTC) so the tests
are timezone-independent.
"""

import datetime
import unittest

import numpy as np

from pysp.inference import estimate
from pysp.stats.processes.temporal import PeriodicTimeDistribution, SeasonalTimeSeries, cyclic_phase, to_unix_seconds


def _fit_periodic(times, period="day"):
    """Fit a PeriodicTimeDistribution through the pysp estimator contract."""
    return estimate(list(times), PeriodicTimeDistribution(period).estimator())


class DatetimeParsingTest(unittest.TestCase):
    def test_accepts_datetime_numpy_iso_and_unix(self):
        ref = datetime.datetime(2020, 1, 1, 9, 0, 0)
        v_obj = to_unix_seconds([ref])[0]
        v_np = to_unix_seconds(np.array(["2020-01-01T09:00:00"], dtype="datetime64[s]"))[0]
        v_iso = to_unix_seconds(["2020-01-01T09:00:00"])[0]
        v_unix = to_unix_seconds([v_obj])[0]
        np.testing.assert_allclose([v_np, v_iso, v_unix], v_obj)

    def test_cyclic_phase_range(self):
        phi = cyclic_phase(np.arange(0, 86400, 3600.0), "day")
        self.assertTrue(np.all((phi >= 0) & (phi < 2 * np.pi)))


class PeriodicTimeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        hours = 9.0 + rng.randn(4000) * 1.5  # events at ~09:00 UTC each day
        self.times = np.arange(4000) * 86400.0 + hours * 3600.0

    def test_fit_recovers_peak_and_concentration(self):
        pt = _fit_periodic(self.times, "day")
        self.assertAlmostEqual(pt.peak_phase_fraction() * 24, 9.0, delta=0.3)
        self.assertGreater(pt.conc, 1.0)  # clearly peaked, not uniform

    def test_density_integrates_to_one_over_the_cycle(self):
        pt = _fit_periodic(self.times, "day")
        grid = np.linspace(0, 86400, 20000, endpoint=False)
        enc = pt.dist_to_encoder().seq_encode(grid)
        integral = np.trapezoid(np.exp(pt.seq_log_density(enc)), grid)
        self.assertAlmostEqual(integral, 1.0, places=2)

    def test_sampler_clusters_at_the_peak(self):
        pt = _fit_periodic(self.times, "day")
        s = pt.sampler(seed=1).sample(8000) / 3600.0  # hours into the day
        self.assertAlmostEqual(np.mean(s), 9.0, delta=0.3)

    def test_day_of_week(self):
        rng = np.random.RandomState(1)
        dow = 2.0 + rng.randn(4000) * 0.4  # epoch is a Thursday, so +2 days = Saturday
        wtimes = np.arange(4000) * 7 * 86400.0 + dow * 86400.0
        pw = _fit_periodic(wtimes, "week")
        self.assertAlmostEqual(pw.peak_phase_fraction() * 7, 2.0, delta=0.3)

    def test_uniform_when_no_pattern(self):
        rng = np.random.RandomState(2)
        pt = _fit_periodic(rng.uniform(0, 365 * 86400, 5000), "day")
        self.assertLess(pt.conc, 0.3)  # no time-of-day structure -> near-uniform


class SeasonalTimeSeriesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.days = np.arange(365 * 3)
        self.ts = np.array([datetime.datetime(2018, 1, 1) + datetime.timedelta(days=int(d)) for d in self.days])
        self.trend, self.amp = 0.02, 5.0
        season = self.amp * np.sin(2 * np.pi * self.days / 365.25 - 1.0)
        self.y = 10 + self.trend * self.days + season + rng.randn(len(self.days)) * 0.5

    def test_fit_quality(self):
        m = SeasonalTimeSeries(periods=["year"], harmonics=2).fit(self.ts, self.y)
        rmse = np.sqrt(np.mean((m.mean(self.ts) - self.y) ** 2))
        self.assertLess(rmse, 0.6)  # ~ the 0.5 noise level

    def test_conditional_returns_a_distribution(self):
        from pysp.stats import GaussianDistribution

        m = SeasonalTimeSeries(periods=["year"], harmonics=2).fit(self.ts, self.y)
        d = m.conditional(self.ts[100])  # the predictive distribution at one timestamp
        self.assertIsInstance(d, GaussianDistribution)
        self.assertAlmostEqual(d.mu, m.mean(self.ts[100])[0], places=8)
        self.assertGreater(d.sigma2, 0.0)
        self.assertIsInstance(m.conditional(self.ts[:5]), list)  # array of times -> list of distributions

    def test_forecast_is_accurate(self):
        m = SeasonalTimeSeries(periods=["year"], harmonics=2).fit(self.ts, self.y)
        future_d = 365 * 3 + np.arange(60)
        future = np.array([datetime.datetime(2018, 1, 1) + datetime.timedelta(days=int(d)) for d in future_d])
        truth = 10 + self.trend * future_d + self.amp * np.sin(2 * np.pi * future_d / 365.25 - 1.0)
        self.assertLess(np.sqrt(np.mean((m.mean(future) - truth) ** 2)), 0.3)
        self.assertTrue(all(d.sigma2 > 0 for d in m.conditional(future)))

    def test_log_density_and_sampler(self):
        m = SeasonalTimeSeries(periods=["year"], harmonics=2).fit(self.ts, self.y)
        ld = m.log_density(self.ts, self.y)
        self.assertEqual(ld.shape, (len(self.y),))
        s = m.sampler(seed=0).sample(self.ts)  # draws values from the conditional p(value|time)
        self.assertEqual(s.shape, (len(self.y),))
        self.assertLess(abs(np.std(s - m.mean(self.ts)) - m.sigma), 0.3)  # residual spread ~ sigma

    def test_decompose_recovers_trend_and_seasonal_amplitude(self):
        m = SeasonalTimeSeries(periods=["year"], harmonics=2).fit(self.ts, self.y)
        dec = m.decompose(self.ts)
        slope = (dec["trend"][-1] - dec["trend"][0]) / len(self.days)
        self.assertAlmostEqual(slope, self.trend, delta=0.005)
        self.assertAlmostEqual((dec["year"].max() - dec["year"].min()) / 2, self.amp, delta=0.5)

    def test_two_seasonalities_at_once(self):
        rng = np.random.RandomState(3)
        secs = np.arange(0, 60 * 86400, 3600.0)  # hourly for 60 days
        daily = 3 * np.sin(2 * np.pi * secs / 86400.0)
        weekly = 2 * np.cos(2 * np.pi * secs / 604800.0)
        y = 20 + daily + weekly + rng.randn(len(secs)) * 0.3
        m = SeasonalTimeSeries(periods=["day", "week"], harmonics=1).fit(secs, y)
        self.assertLess(np.sqrt(np.mean((m.mean(secs) - y) ** 2)), 0.4)


if __name__ == "__main__":
    unittest.main()
