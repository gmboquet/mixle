"""DoD test for K5 -- health monitoring + exceedance alerts (mixle.analysis.health_risk)."""

from __future__ import annotations

import numpy as np

from mixle.analysis.health_risk import ExceedanceReport, exposure_exceedance_monitor

LIMIT = 80.0
ALPHA = 0.05


def _below_limit_series(rng: np.random.Generator, n: int) -> np.ndarray:
    """Synthetic compliant monitoring data: comfortably below ``LIMIT``, never exceeding it."""
    return rng.normal(loc=50.0, scale=5.0, size=n)


def test_exceedance_holds_false_alarm_rate():
    master_rng = np.random.default_rng(2026)
    calib = _below_limit_series(master_rng, 800)

    # --- false-alarm-rate check: many independent below-limit trials, no true excursions -----------
    n_trials, trial_len = 300, 40
    total_points = 0
    total_alerts = 0
    for _ in range(n_trials):
        trial = _below_limit_series(master_rng, trial_len)
        report = exposure_exceedance_monitor(trial, LIMIT, alpha=ALPHA, calib=calib)
        assert isinstance(report, ExceedanceReport)
        assert report.alerts.shape == trial.shape
        assert report.prob_exceed.shape == trial.shape
        assert report.false_alarm_target == ALPHA
        total_points += trial.shape[0]
        total_alerts += int(report.alerts.sum())

    empirical_false_alarm_rate = total_alerts / total_points
    assert 0.0 <= empirical_false_alarm_rate <= ALPHA + 0.02

    # --- detection check: a seeded exceedance excursion is flagged -----------------------------------
    excursion = _below_limit_series(master_rng, 120)
    excursion[60:75] = LIMIT + 40.0  # a clear, sustained excursion well past the limit
    excursion_report = exposure_exceedance_monitor(excursion, LIMIT, alpha=ALPHA, calib=calib)
    assert excursion_report.alerts[60:75].any()
    assert excursion_report.prob_exceed[60:75].mean() > excursion_report.prob_exceed[:60].mean()
