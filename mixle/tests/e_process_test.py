"""P9 (experimental) -- e-processes give anytime-valid receipts.

The graduation receipt for the e-process mechanism is empirical anytime type-I control: under
the null, continuously peeking and rejecting the first time ``E_t >= 1/alpha`` must fire with
probability at most ``alpha`` -- the guarantee Ville's inequality promises. These tests verify
that, contrast it with a naively-peeked fixed-sample test (which blows its error budget), and
confirm the detector has power against a real mean shift.
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.e_process import (
    EProcess,
    MeanShiftDetector,
    normal_mixture_eprocess,
)


def _stream(rng, *, n, mean, sigma):
    return rng.normal(mean, sigma, size=n)


def test_eprocess_null_behavior_is_a_valid_e_value() -> None:
    """Under H0, E_t is right-skewed with a fixed-time Ville bound -- the e-value hallmark.

    We do NOT test the sample mean: with tau=1 the mixture e-value has infinite variance, so its
    empirical mean is meaningless. The robust, defining signatures are: most trajectories sit below
    1 (right skew, since the mean is pinned at 1), and at any FIXED time P(E_t >= 1/alpha) <= alpha.
    """
    sigma, tau, mu0, t, alpha = 1.0, 1.0, 0.0, 50, 0.1
    reps = 800
    finals = np.empty(reps)
    for i in range(reps):
        rng = np.random.default_rng(1000 + i)
        e = normal_mixture_eprocess(_stream(rng, n=t, mean=mu0, sigma=sigma), mu0=mu0, sigma=sigma, tau=tau)
        finals[i] = e[-1]
    assert np.median(finals) < 1.0, "under H0 the e-value should be right-skewed (median < 1)"
    fixed_time_reject = np.mean(finals >= 1.0 / alpha)
    assert fixed_time_reject <= alpha + 0.03, (
        f"fixed-time P(E_t >= 1/alpha) = {fixed_time_reject:.3f} exceeds alpha={alpha}"
    )


def test_anytime_type_I_control_under_continuous_peeking() -> None:
    """Rejecting the first time E_t >= 1/alpha must fire <= alpha of the time under H0."""
    alpha, sigma, tau, mu0, T = 0.1, 1.0, 1.0, 0.0, 400
    reps = 500
    false_alarms = 0
    for i in range(reps):
        rng = np.random.default_rng(7000 + i)
        report = MeanShiftDetector(mu0=mu0, sigma=sigma, tau=tau, alpha=alpha).scan(
            _stream(rng, n=T, mean=mu0, sigma=sigma)
        )
        false_alarms += int(report.detected)  # detected == ever crossed 1/alpha over the whole run
    rate = false_alarms / reps
    # Ville bounds this by alpha for the ENTIRE trajectory; empirically it sits below alpha.
    assert rate <= alpha + 0.03, f"anytime false-alarm rate {rate:.3f} exceeds alpha={alpha}"


def test_naive_peeked_fixed_test_inflates_but_eprocess_does_not() -> None:
    """A fixed-sample z-test peeked every step blows past alpha; the e-process stays valid."""
    alpha, sigma, mu0, T = 0.1, 1.0, 0.0, 400
    reps = 400
    z_crit = 1.959963984540054  # two-sided per-look 0.05 critical value -- but we peek every step
    naive_false = 0
    eproc_false = 0
    for i in range(reps):
        rng = np.random.default_rng(4200 + i)
        xs = _stream(rng, n=T, mean=mu0, sigma=sigma)
        # naive: after each observation, z-test the running mean; reject if it EVER crosses.
        csum = np.cumsum(xs - mu0)
        ns = np.arange(1, T + 1)
        z = csum / (sigma * np.sqrt(ns))
        naive_false += int(np.any(np.abs(z) > z_crit))
        # e-process on the same stream at the same alpha.
        e = normal_mixture_eprocess(xs, mu0=mu0, sigma=sigma, tau=1.0)
        eproc_false += int(np.any(e >= 1.0 / alpha))
    naive_rate = naive_false / reps
    eproc_rate = eproc_false / reps
    # The continuously-peeked fixed test spends far more than its nominal error budget...
    assert naive_rate > 0.30, f"expected the peeked fixed test to inflate; got {naive_rate:.3f}"
    # ...while the e-process holds its anytime guarantee.
    assert eproc_rate <= alpha + 0.03, f"e-process inflated to {eproc_rate:.3f}"
    assert eproc_rate < naive_rate, "the whole point: the e-process is far tighter under peeking"


def test_detects_a_real_mean_shift_with_power() -> None:
    """Under a genuine shift, the e-process crosses the threshold with high probability."""
    alpha, sigma, tau, mu0, T = 0.1, 1.0, 1.0, 0.0, 200
    shift = 0.7  # a moderate, not-obvious shift
    reps = 300
    detections = 0
    delays = []
    for i in range(reps):
        rng = np.random.default_rng(9100 + i)
        report = MeanShiftDetector(mu0=mu0, sigma=sigma, tau=tau, alpha=alpha).scan(
            _stream(rng, n=T, mean=mu0 + shift, sigma=sigma)
        )
        if report.detected:
            detections += 1
            delays.append(report.detection_time)
    power = detections / reps
    assert power >= 0.8, f"power against a {shift}-sigma shift was only {power:.2f}"
    assert np.median(delays) < T, "median detection happened before the stream ended"


def test_generic_eprocess_reject_rule_and_receipt() -> None:
    """The generic EProcess multiplies ratios and rejects at 1/alpha, peeking-safe."""
    e = EProcess()
    # Feed constant log-ratio 0.2 per step; after k steps log_e = 0.2k, e = exp(0.2k).
    for _ in range(20):
        e.update(0.2)
    assert e.n == 20
    assert np.isclose(e.log_e_value, 4.0)
    assert e.rejects(alpha=0.05)  # exp(4) = 54.6 >= 1/0.05 = 20
    # A process that spikes then decays must still report the peak crossing (peeking-safe).
    e2 = EProcess()
    for lr in [3.5, -3.0, -3.0]:  # crosses 1/0.05 at step 1 (e=33), then decays
        e2.update(lr)
    assert not e2.rejects(alpha=0.05), "current e-value has decayed below threshold"
    assert e2.ever_rejected(alpha=0.05), "but the peak crossed -- an anytime reject already happened"
    rec = e2.receipt(alpha=0.05)
    assert rec["rejected"] is True and rec["threshold"] == 20.0


def test_determinism_given_seed() -> None:
    xs = np.random.default_rng(123).normal(0.3, 1.0, size=100)
    a = normal_mixture_eprocess(xs, mu0=0.0, sigma=1.0, tau=1.0)
    b = normal_mixture_eprocess(xs, mu0=0.0, sigma=1.0, tau=1.0)
    assert np.array_equal(a, b)
    r1 = MeanShiftDetector(mu0=0.0, sigma=1.0, tau=1.0, alpha=0.05).scan(xs)
    r2 = MeanShiftDetector(mu0=0.0, sigma=1.0, tau=1.0, alpha=0.05).scan(xs)
    assert r1 == r2
