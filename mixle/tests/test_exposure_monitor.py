"""DoD test for K5 -- health monitoring + exceedance alerts (mixle.analysis.health_risk)."""

from __future__ import annotations

import numpy as np
import pytest

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

    # an adequately-sized, explicitly-supplied calib set is reported as genuinely calibrated
    # (MXR-080-0097 negative control).
    assert excursion_report.calibrated is True


def test_observed_value_enters_its_own_score():
    """MXR-080-0096: a sudden, one-off extreme reading must move ITS OWN prob_exceed/alert, not only
    the readings that follow it once a lagging rolling mean catches up."""
    master_rng = np.random.default_rng(11)
    calib = _below_limit_series(master_rng, 800)
    series = _below_limit_series(master_rng, 80)
    spike_t = 40
    series[spike_t] = LIMIT + 920.0  # a single, massive, one-timestep-only spike

    report = exposure_exceedance_monitor(series, LIMIT, alpha=ALPHA, calib=calib)

    assert report.warmed_up[spike_t]  # t=40 is well past the warm-up region (window/history << 40)
    assert report.prob_exceed[spike_t] > 0.99
    assert report.alerts[spike_t]
    # the point immediately before the spike (fit from history that does not yet include it) must NOT
    # itself be flagged purely from a look-ahead into t's own value.
    assert not report.alerts[spike_t - 1]


def test_warmup_points_do_not_leak_future_values():
    """MXR-080-0096: an early (not-yet-warmed-up) timestep's score must not depend on later readings --
    changing values far in the future must not change the past."""
    master_rng = np.random.default_rng(12)
    calib = _below_limit_series(master_rng, 800)
    base = _below_limit_series(master_rng, 40)

    series_a = base.copy()
    series_b = base.copy()
    series_b[35:] = LIMIT + 4000.0  # a dramatic change confined to the tail, far past the warm-up head

    report_a = exposure_exceedance_monitor(series_a, LIMIT, alpha=ALPHA, calib=calib)
    report_b = exposure_exceedance_monitor(series_b, LIMIT, alpha=ALPHA, calib=calib)

    warmup = ~report_a.warmed_up
    assert warmup.any()  # sanity: the series does have an actual warm-up region
    assert np.array_equal(report_a.warmed_up, report_b.warmed_up)
    np.testing.assert_array_equal(report_a.prob_exceed[warmup], report_b.prob_exceed[warmup])
    assert not report_a.alerts[warmup].any()
    assert not report_b.alerts[warmup].any()


def test_warmup_points_never_alert():
    """Unwarmed timesteps report an explicit not-yet-evaluated state, never a confident alert."""
    master_rng = np.random.default_rng(13)
    calib = _below_limit_series(master_rng, 800)
    series = _below_limit_series(master_rng, 40)
    series[0:4] = LIMIT + 500.0  # even an extreme reading inside the warm-up head cannot be confirmed

    report = exposure_exceedance_monitor(series, LIMIT, alpha=ALPHA, calib=calib)
    assert not report.warmed_up[0:4].any()
    assert not report.alerts[0:4].any()
    assert (report.prob_exceed[0:4] == 0.0).all()


def test_self_calibration_is_reported_uncalibrated():
    """MXR-080-0097: omitting calib self-calibrates on the monitored series itself -- dependent rolling
    scores, not held-out exchangeable ones -- so the report must say so, not silently claim the bound."""
    master_rng = np.random.default_rng(14)
    series = _below_limit_series(master_rng, 200)

    report = exposure_exceedance_monitor(series, LIMIT, alpha=ALPHA)  # calib omitted
    assert report.calibrated is False
    # still returns a well-formed, best-effort report (graceful degradation), not a crash
    assert report.alerts.shape == series.shape
    assert report.prob_exceed.shape == series.shape


def test_inadequate_calibration_set_is_reported_uncalibrated():
    """MXR-080-0097: an explicit but too-small calib set cannot give ``alpha`` real resolution and must
    be flagged uncalibrated rather than silently accepted (a degenerate near-infinite threshold that
    simply never alerts is not the same thing as a verified false-alarm bound)."""
    master_rng = np.random.default_rng(15)
    series = _below_limit_series(master_rng, 60)
    tiny_calib = _below_limit_series(master_rng, 10)  # far below _min_adequate_calib_size(0.05)

    report = exposure_exceedance_monitor(series, LIMIT, alpha=ALPHA, calib=tiny_calib)
    assert report.calibrated is False


def test_empty_effective_calibration_does_not_crash():
    """A calib set entirely inside its own warm-up region (too short to ever warm up) degrades to an
    explicit uncalibrated, non-alerting report instead of propagating a raw exception."""
    master_rng = np.random.default_rng(16)
    series = _below_limit_series(master_rng, 60)
    degenerate_calib = _below_limit_series(master_rng, 2)  # shorter than _MIN_LOCAL_HISTORY

    report = exposure_exceedance_monitor(series, LIMIT, alpha=ALPHA, calib=degenerate_calib)
    assert report.calibrated is False
    assert not report.alerts[report.warmed_up].any()


@pytest.mark.parametrize(
    ("bad_series", "bad_limit", "bad_alpha", "bad_calib"),
    [
        (np.array([1.0, float("nan"), 3.0]), LIMIT, ALPHA, None),
        (np.array([1.0, 2.0, 3.0]), float("nan"), ALPHA, None),
        (np.array([1.0, 2.0, 3.0]), LIMIT, 0.0, None),
        (np.array([1.0, 2.0, 3.0]), LIMIT, 1.0, None),
        (np.array([1.0, 2.0, 3.0]), LIMIT, ALPHA, np.array([1.0, float("inf")])),
    ],
)
def test_rejects_non_finite_or_out_of_domain_inputs(bad_series, bad_limit, bad_alpha, bad_calib):
    with pytest.raises(ValueError):
        exposure_exceedance_monitor(bad_series, bad_limit, alpha=bad_alpha, calib=bad_calib)
