"""Confirmed-exposure influence measurement: a hierarchical within-subject event study.

Many SUBJECTS are each observed BEFORE and AFTER a known event time -- an exposure with a *confirmed*
timestamp (on social data, a retweet: the act proves the content was seen, and dates it). We estimate
whether, and how much, the event shifts each subject's generative activity, then pool those shifts into a
population effect with calibrated uncertainty.

A treated / control split turns the pooled shift into a **difference-in-differences**: the effect is
``(treated shift) - (control shift)``, so anything that moves everyone at the event time (a concurrent
external shock) cancels, and only the differential -- the influence attributable to the treatment --
survives. The natural control is *exposed non-actors*: subjects the same content reached who did not act.

Two stages, exact/closed-form where the family permits:

  1. **per-subject effect** -- from the activity family's sufficient statistics on the pre and post
     windows: a Gaussian mean-shift (``gaussian_effect``) or a Poisson log-rate shift for event counts
     (``poisson_lograte_effect``), each with its sampling variance.
  2. **hierarchical pooling** -- a random-effects (DerSimonian-Laird) meta-analysis over the per-subject
     effects: a precision-weighted population mean plus between-subject heterogeneity ``tau^2``, computed
     per group, with the DiD contrast and its propagated variance and an empirical-Bayes shrinkage of each
     subject's effect toward its group.

**Identification, stated rather than assumed away.** Within-subject differencing removes every
TIME-INVARIANT subject trait -- the homophily / selection-into-ties confound (Shalizi & Thomas 2011) is
exactly such a trait, so unit differencing annihilates it. The treated-vs-control contrast removes shocks
common to both groups. The residual threat is time-VARYING selection into the event (whatever made the
subject act *then*); it is mitigated -- not eliminated -- by a matched exposed-non-actor control, and
:func:`tipping_drift` reports how large an unmeasured differential drift would have to be to explain the
effect away (a transparent sensitivity bound, not a guarantee).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gaussian_effect(pre: np.ndarray, post: np.ndarray) -> tuple[float, float]:
    """Per-subject mean shift and its (Welch) sampling variance from pre/post activity samples."""
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if len(pre) < 2 or len(post) < 2:
        raise ValueError("need >=2 observations in each window for a variance")
    effect = float(post.mean() - pre.mean())
    var = float(post.var(ddof=1) / len(post) + pre.var(ddof=1) / len(pre))
    return effect, var


def poisson_lograte_effect(k_pre: float, t_pre: float, k_post: float, t_post: float) -> tuple[float, float]:
    """Per-subject log activity-rate shift ``log(rate_post) - log(rate_pre)`` for event counts over windows.

    ``k_*`` are event counts, ``t_*`` the window durations (or exposures). Uses a Haldane 0.5 correction so
    zero-count windows are finite; variance is the delta-method log-rate variance ``1/k_post + 1/k_pre``.
    """
    kp, kq = float(k_pre) + 0.5, float(k_post) + 0.5
    effect = float(np.log(kq / t_post) - np.log(kp / t_pre))
    var = float(1.0 / kq + 1.0 / kp)
    return effect, var


def _random_effects(y: np.ndarray, v: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """DerSimonian-Laird random-effects pool. Returns (mean, var_of_mean, tau2, EB-shrunk effects)."""
    w = 1.0 / v
    fe = float((w * y).sum() / w.sum())
    q = float((w * (y - fe) ** 2).sum())
    df = len(y) - 1
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if c > 0 and df > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mean = float((ws * y).sum() / ws.sum())
    var_mean = float(1.0 / ws.sum())
    # empirical-Bayes shrinkage of each subject toward the pooled mean
    shrunk = (y / v + mean / tau2) / (1.0 / v + 1.0 / tau2) if tau2 > 0 else np.full_like(y, mean)
    return mean, var_mean, tau2, shrunk


@dataclass
class EventStudyResult:
    """Pooled influence estimate. ``effect`` is the DiD ATT (treated minus control) when a control exists."""

    effect: float
    se: float
    z: float
    p_value: float
    ci: tuple[float, float]
    treated_mean: float
    treated_se: float
    control_mean: float | None
    control_se: float | None
    tau2_treated: float
    n_treated: int
    n_control: int
    shrunk_treated: np.ndarray

    def __str__(self) -> str:
        c = "" if self.control_mean is None else f", control {self.control_mean:+.4f}"
        return (
            f"EventStudyResult(effect={self.effect:+.4f} ± {self.se:.4f}, "
            f"95% CI [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}], z={self.z:.2f}, p={self.p_value:.2e}, "
            f"treated {self.treated_mean:+.4f}{c}, tau^2={self.tau2_treated:.4f}, "
            f"n={self.n_treated}+{self.n_control})"
        )


def _norm_sf(z: float) -> float:
    from math import erfc, sqrt

    return 0.5 * erfc(abs(z) / sqrt(2.0))


def hierarchical_event_study(
    treated_effects: np.ndarray,
    treated_vars: np.ndarray,
    control_effects: np.ndarray | None = None,
    control_vars: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
) -> EventStudyResult:
    """Pool per-subject effects into a population influence estimate (DiD if a control group is given).

    ``*_effects`` / ``*_vars`` are the per-subject shifts and their variances from stage 1. With a control
    group the reported ``effect`` is ``treated_mean - control_mean`` -- the difference-in-differences ATT.
    """
    y_t, v_t = np.asarray(treated_effects, float), np.asarray(treated_vars, float)
    t_mean, t_var, tau2, shrunk = _random_effects(y_t, v_t)

    if control_effects is not None and len(control_effects) > 0:
        y_c, v_c = np.asarray(control_effects, float), np.asarray(control_vars, float)
        c_mean, c_var, _, _ = _random_effects(y_c, v_c)
        effect, var = t_mean - c_mean, t_var + c_var
        n_c = len(y_c)
    else:
        c_mean = c_var = None
        effect, var, n_c = t_mean, t_var, 0

    se = float(np.sqrt(var))
    z = effect / se if se > 0 else 0.0
    from math import sqrt

    # normal quantile for the CI half-width
    zq = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _inv_norm_sf(alpha / 2)
    half = zq * se
    return EventStudyResult(
        effect=effect,
        se=se,
        z=float(z),
        p_value=float(2 * _norm_sf(z)),
        ci=(effect - half, effect + half),
        treated_mean=t_mean,
        treated_se=float(sqrt(t_var)),
        control_mean=c_mean,
        control_se=(None if c_var is None else float(sqrt(c_var))),
        tau2_treated=tau2,
        n_treated=len(y_t),
        n_control=n_c,
        shrunk_treated=shrunk,
    )


def _inv_norm_sf(p: float) -> float:
    """Inverse normal survival function (quantile) via a rational approximation (Acklam)."""
    from math import log, sqrt

    p = 1.0 - p  # to CDF quantile
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = sqrt(-2 * log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def tipping_drift(result: EventStudyResult) -> dict:
    """Sensitivity bound: the unmeasured differential drift that would explain the effect away.

    Within-subject DiD is unbiased only if, absent treatment, treated and control would have drifted
    equally. This returns the differential drift ``delta`` (in effect units) that nullifies the estimate
    (``= effect``) and the value that pushes the 95% CI through zero -- so a reader can judge whether a
    confound that large is plausible. Larger = more robust.
    """
    return {
        "drift_to_nullify_point": float(result.effect),
        "drift_to_nullify_ci": float(result.effect - np.sign(result.effect) * 1.959963984540054 * result.se),
        "effect_in_se_units": float(result.z),
    }
