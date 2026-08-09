"""Hierarchical within-subject event-study associations and identified DiD estimates.

Many SUBJECTS are each observed BEFORE and AFTER a known event time -- an exposure with a *confirmed*
timestamp (on social data, a retweet: the act proves the content was seen, and dates it). We estimate
whether, and how much, the event shifts each subject's generative activity, then pool those shifts into a
population effect with calibrated uncertainty.

A treated/control split computes the usual difference in changes. It becomes an identified causal
difference-in-differences estimate only when the caller supplies an
:class:`EventStudyIdentification` receipt. Without that receipt the calculation is explicitly labeled
as a before/after or treated/control *association*.

Two stages, exact/closed-form where the family permits:

  1. **per-subject effect** -- from the activity family's sufficient statistics on the pre and post
     windows: a Gaussian mean-shift (``gaussian_effect``) or a Poisson log-rate shift for event counts
     (``poisson_lograte_effect``), each with its sampling variance.
  2. **hierarchical pooling** -- a random-effects (DerSimonian-Laird) meta-analysis over the per-subject
     effects: a precision-weighted population mean plus between-subject heterogeneity ``tau^2``, computed
     per group, with the DiD contrast and its propagated variance and an empirical-Bayes shrinkage of each
     subject's effect toward its group.

Within-subject differencing removes additive time-invariant subject effects and a treated/control
contrast removes additive shocks common to both groups. Neither fact alone establishes parallel trends,
exchangeability, consistency, positivity, or absence of interference. :func:`tipping_drift` provides a
simple sensitivity calculation; it is not a substitute for an identification argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mixle.utils.exact import require_exact_bool


@dataclass(frozen=True)
class EventStudyIdentification:
    """Auditable assumptions for interpreting a treated/control contrast as causal DiD."""

    design_evidence: tuple[str, ...]
    parallel_trends_evidence: tuple[str, ...]
    exchangeability: bool
    positivity: bool
    consistency: bool
    no_interference: bool
    no_anticipation: bool
    sensitivity_analysis: str | None = None

    def __post_init__(self) -> None:
        for name, values in (
            ("design_evidence", self.design_evidence),
            ("parallel_trends_evidence", self.parallel_trends_evidence),
        ):
            if not values or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain at least one non-empty evidence reference")
        if self.sensitivity_analysis is not None and (
            not isinstance(self.sensitivity_analysis, str) or not self.sensitivity_analysis.strip()
        ):
            raise ValueError("sensitivity_analysis must be None or a non-empty reference")
        # Exact Booleans, for the same reason as CausalIdentification (MXR-080-1899): `identified`
        # is a truthiness conjunction, so a receipt deserialized from configuration text with
        # `no_anticipation: "false"` declared the assumption FAILS and was still read as identified.
        for name in ("exchangeability", "positivity", "consistency", "no_interference", "no_anticipation"):
            object.__setattr__(self, name, require_exact_bool(getattr(self, name), f"EventStudyIdentification.{name}"))

    @property
    def identified(self) -> bool:
        return bool(
            self.exchangeability
            and self.positivity
            and self.consistency
            and self.no_interference
            and self.no_anticipation
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable copy of the receipt."""
        return asdict(self)


def gaussian_effect(pre: np.ndarray, post: np.ndarray) -> tuple[float, float]:
    """Per-subject mean shift and its (Welch) sampling variance from pre/post activity samples."""
    pre = _finite_1d("pre", pre)
    post = _finite_1d("post", post)
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
    # A Boolean is not an event count (MXR-080-1899). `float(True)` is 1.0 and `(1.0).is_integer()`
    # is True, so `poisson_lograte_effect(True, t, False, t)` used to be read as "one event in the
    # pre-window, zero in the post-window" and returned a confident -1.1 log-rate shift. That is
    # exactly the shape a mis-wired caller produces -- passing an `any(events)` indicator, or a
    # pandas Boolean column, where the count belongs -- and there is no count it could plausibly
    # have meant, so it is refused rather than interpreted. Exposures are checked the same way: a
    # Boolean window duration is equally meaningless, and `t=True` would silently mean one unit.
    for name, value in (("k_pre", k_pre), ("k_post", k_post), ("t_pre", t_pre), ("t_post", t_post)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a number, not a Boolean; got {value!r}")
    raw_counts = (float(k_pre), float(k_post))
    durations = (float(t_pre), float(t_post))
    if not all(np.isfinite(value) for value in (*raw_counts, *durations)):
        raise ValueError("counts and exposures must be finite")
    if any(value < 0 or not value.is_integer() for value in raw_counts):
        raise ValueError("event counts must be non-negative integers")
    if any(value <= 0 for value in durations):
        raise ValueError("exposures must be strictly positive")
    if raw_counts[0] == 0.0 or raw_counts[1] == 0.0:
        # STAT-RR17-09: with a zero count, log((k + 0.5)/t) is pure Haldane offset, not rate
        # information -- under an exact null with unequal exposures those offsets do not cancel,
        # and Gaussian pooling over many such subjects drove p to 1.37e-12 at n=1000 while the
        # true effect was zero. Zero-count windows are refused here; pool the counts instead
        # (poisson_pooled_rate_ratio, exact at any sparsity).
        raise ValueError(
            "poisson_lograte_effect: a zero-count window carries no per-subject rate-ratio "
            "information at this scale (the Haldane offset would masquerade as an effect); "
            "use poisson_pooled_rate_ratio over all subjects for the sparse regime"
        )
    kp, kq = raw_counts[0] + 0.5, raw_counts[1] + 0.5
    effect = float(np.log(kq / t_post) - np.log(kp / t_pre))
    var = float(1.0 / kq + 1.0 / kp)
    return effect, var


def poisson_lograte_effects(k_pre: Any, t_pre: float, k_post: Any, t_post: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-subject log rate-ratio effects for Gaussian pooling, GATED on a measured count floor.

    The per-subject Gaussianization ``log((k + 0.5)/t)`` is structurally biased at sparse counts:
    measured under an exact rate null with exposures 1:3, the per-subject bias/SE ratio is 0.48
    at expected count 0.1, 0.13 at 1, 0.03 at 2, and 0.004 at 4 -- and pooling n subjects
    multiplies the bias contribution to z by sqrt(n), which is how a true null reached
    p = 1.37e-12 at n = 1000 (STAT-RR17-09). Both ARM MEANS must therefore reach 4.0 expected
    counts (the measured level where the bias stays negligible out to ~1e5 subjects); below the
    floor this refuses and names poisson_pooled_rate_ratio, which is exact at any sparsity for
    the common-ratio estimand.
    """
    k_pre_arr = np.asarray(list(k_pre), dtype=float)
    k_post_arr = np.asarray(list(k_post), dtype=float)
    if k_pre_arr.shape != k_post_arr.shape or k_pre_arr.ndim != 1 or k_pre_arr.size == 0:
        raise ValueError("k_pre and k_post must be equal-length non-empty count vectors")
    mean_pre, mean_post = float(k_pre_arr.mean()), float(k_post_arr.mean())
    if mean_pre < 4.0 or mean_post < 4.0:
        raise ValueError(
            f"sparse-count regime (arm means {mean_pre:.2f} pre / {mean_post:.2f} post are below "
            "the measured floor of 4.0): the per-subject Gaussian approximation is biased here "
            "and pooling amplifies the bias with sqrt(n) -- use poisson_pooled_rate_ratio, which "
            "is exact at any sparsity for the common rate ratio"
        )
    # Above the floor, an isolated zero count is legitimate smallness, not the sparse-regime
    # failure: the measured floor (bias/SE 0.004 at arm mean 4) was computed INCLUDING the
    # zero-count draws that occur at these rates, so the batch applies the Haldane form directly
    # rather than routing through the single-call refusal (which guards uncontexted sparse use).
    if not (np.isfinite(t_pre) and np.isfinite(t_post) and t_pre > 0 and t_post > 0):
        raise ValueError("exposures must be finite and positive")
    for name, arr in (("k_pre", k_pre_arr), ("k_post", k_post_arr)):
        if np.any(arr < 0) or np.any(arr != np.floor(arr)) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain non-negative integer counts")
    kp = k_pre_arr + 0.5
    kq = k_post_arr + 0.5
    effects = np.log(kq / float(t_post)) - np.log(kp / float(t_pre))
    variances = 1.0 / kq + 1.0 / kp
    return effects, variances


def poisson_pooled_rate_ratio(
    k_pre: Any, t_pre: float, k_post: Any, t_post: float, *, alpha: float = 0.05
) -> dict[str, Any]:
    """EXACT conditional inference for a COMMON post/pre rate ratio -- valid at any sparsity.

    Sums the counts over subjects: with a common ratio ``theta`` and a constant per-subject
    exposure ratio ``r = t_post/t_pre``, ``K_post | K_post + K_pre = N`` is exactly
    ``Binomial(N, theta*r / (1 + theta*r))``, so the test of ``theta = 1`` is an exact binomial
    test and the CI is the Clopper-Pearson interval transformed back to the ratio scale
    (STAT-RR17-09's sparse-regime route). ESTIMAND: the common ratio -- a deliberate reduction
    from the per-subject random-effects mean, which sparse counts cannot support; heterogeneity
    across subjects is NOT estimated here.
    """
    from scipy.stats import beta as _beta
    from scipy.stats import binomtest as _binomtest

    k_pre_arr = np.asarray(list(k_pre), dtype=float)
    k_post_arr = np.asarray(list(k_post), dtype=float)
    if k_pre_arr.shape != k_post_arr.shape or k_pre_arr.ndim != 1 or k_pre_arr.size == 0:
        raise ValueError("k_pre and k_post must be equal-length non-empty count vectors")
    for name, arr in (("k_pre", k_pre_arr), ("k_post", k_post_arr)):
        if np.any(arr < 0) or np.any(arr != np.floor(arr)) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain non-negative integer counts")
    if not (np.isfinite(t_pre) and np.isfinite(t_post) and t_pre > 0 and t_post > 0):
        raise ValueError("exposures must be finite and positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    total_post = int(k_post_arr.sum())
    total = int(k_pre_arr.sum()) + total_post
    if total == 0:
        raise ValueError("no events in either window: the rate ratio is unidentified")
    r = float(t_post) / float(t_pre)
    p0 = r / (1.0 + r)
    p_value = float(_binomtest(total_post, total, p0).pvalue)
    tail = alpha / 2.0
    p_lo = 0.0 if total_post == 0 else float(_beta.ppf(tail, total_post, total - total_post + 1))
    p_hi = 1.0 if total_post == total else float(_beta.ppf(1.0 - tail, total_post + 1, total - total_post))
    ratio = (total_post / (total - total_post)) / r if total_post < total else float("inf")
    lo = (p_lo / (1.0 - p_lo)) / r if p_lo < 1.0 else float("inf")
    hi = (p_hi / (1.0 - p_hi)) / r if p_hi < 1.0 else float("inf")
    return {
        "estimand": "common post/pre rate ratio (conditional exact; per-subject heterogeneity not estimated)",
        "ratio": ratio,
        "ci": (lo, hi),
        "ci_level": 1.0 - alpha,
        "p_value_ratio_equals_1": p_value,
        "events_post": total_post,
        "events_total": total,
        "exposure_ratio": r,
        "method": "sum-Poisson -> conditional binomial; exact binomial test and Clopper-Pearson CI",
    }


def _random_effects(y: np.ndarray, v: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """DerSimonian-Laird random-effects pool. Returns (mean, var_of_mean, tau2, EB-shrunk effects)."""
    y = _finite_1d("effects", y)
    v = _finite_1d("variances", v)
    if len(y) == 0:
        raise ValueError("need at least 1 subject to pool a random-effects estimate")
    if len(v) != len(y):
        raise ValueError("effects and variances must have the same length")
    if np.any(v <= 0):
        # a per-subject sampling variance of 0 (or negative) is physically impossible -- it would
        # hand that subject infinite (or negative) precision and silently corrupt every downstream
        # weighted formula (fe/tau2/mean/shrunk) into a confidently-wrong, well-formed-looking result.
        raise ValueError("variances must be strictly positive")
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
    """Pooled change contrast with explicit causal-identification metadata."""

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
    estimand: str
    identified: bool
    interpretation: str
    identification: dict[str, Any] | None

    def __str__(self) -> str:
        c = "" if self.control_mean is None else f", control {self.control_mean:+.4f}"
        return (
            f"EventStudyResult(estimand={self.estimand!r}, identified={self.identified}, "
            f"effect={self.effect:+.4f} ± {self.se:.4f}, "
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
    identification: EventStudyIdentification | None = None,
) -> EventStudyResult:
    """Pool per-subject changes into an association or explicitly identified DiD estimate.

    ``*_effects`` / ``*_vars`` are per-subject changes and sampling variances. A control group makes the
    numeric contrast ``treated_mean - control_mean``. It is labeled an ATT only when a complete
    ``identification`` receipt is attached; otherwise it remains a treated/control change association.
    """
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or not np.isfinite(alpha):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if (control_effects is None) != (control_vars is None):
        raise ValueError("control_effects and control_vars must be provided together")

    y_t = _finite_1d("treated_effects", treated_effects)
    v_t = _finite_1d("treated_vars", treated_vars)
    t_mean, t_var, tau2, shrunk = _random_effects(y_t, v_t)

    has_control = control_effects is not None
    if has_control:
        y_c = _finite_1d("control_effects", control_effects)
        v_c = _finite_1d("control_vars", control_vars)
        if len(y_c) == 0:
            raise ValueError("control group must contain at least one subject when supplied")
        c_mean, c_var, _, _ = _random_effects(y_c, v_c)
        effect, var = t_mean - c_mean, t_var + c_var
        n_c = len(y_c)
    else:
        c_mean = c_var = None
        effect, var, n_c = t_mean, t_var, 0

    if identification is not None and not isinstance(identification, EventStudyIdentification):
        raise TypeError("identification must be an EventStudyIdentification receipt")
    if identification is not None and not has_control:
        raise ValueError("causal difference-in-differences identification requires a control group")
    identified = bool(identification is not None and identification.identified)
    if identification is not None and not identified:
        raise ValueError(
            "identification must affirm exchangeability, positivity, consistency, no interference, and no anticipation"
        )
    if identified:
        estimand = "difference-in-differences average treatment effect on the treated"
        interpretation = "causal contrast under the attached identification assumptions"
    elif has_control:
        estimand = "treated-minus-control change association"
        interpretation = "association only; parallel trends and causal assumptions were not established"
    else:
        estimand = "before-after association"
        interpretation = "association only; no control group or causal identification was supplied"

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
        estimand=estimand,
        identified=identified,
        interpretation=interpretation,
        identification=None if identification is None else identification.to_dict(),
    )


def _finite_1d(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional finite numeric array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


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
