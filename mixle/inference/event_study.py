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
  2. **pooling, weighted by the estimand** -- the ASSOCIATION paths report a random-effects
     (DerSimonian-Laird) precision-weighted summary with between-subject heterogeneity ``tau^2``
     and empirical-Bayes shrinkage; the IDENTIFIED DiD path instead uses equal-subject-weight
     (arithmetic) group means with Student-t/Welch inference, because the ATT is an arithmetic
     mean and precision weighting estimates a different quantity under effect-variance dependence
     (STAT-RR21-01/STAT-RR22-07). DL diagnostics ride along in both modes.

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

    ``k_*`` are event counts, ``t_*`` the window durations (or exposures). The estimate is the
    Haldane form ``log((k_post + 0.5)/t_post) - log((k_pre + 0.5)/t_pre)`` with the matching
    plug-in variance ``1/(k_post + 0.5) + 1/(k_pre + 0.5)`` -- STRICTLY POSITIVE counts only:
    a zero-count window is REFUSED with a pointer to :func:`poisson_pooled_rate_ratio`
    (STAT-RR17-09: at zero the Haldane offset is pure correction, not rate information, and
    pooling it drove a true null to p = 1.37e-12). An earlier version of this docstring claimed
    zero counts were made finite and quoted a ``1/k`` variance; neither matched the executable
    contract (STAT-P20-03).
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


def _haldane_logit_moments(n: int, p: float) -> tuple[float, float]:
    """Exact mean and variance of ``log((K+0.5)/(n-K+0.5))`` for ``K ~ Binomial(n, p)`` by pmf summation."""
    from scipy.stats import binom as _binom

    k = np.arange(n + 1)
    pmf = _binom.pmf(k, n, p)
    logit_k = np.log((k + 0.5) / (n - k + 0.5))
    m1 = float(np.sum(pmf * logit_k))
    m2 = float(np.sum(pmf * logit_k * logit_k))
    return m1, m2 - m1 * m1


def _batch_counts(name: str, raw: Any) -> np.ndarray:
    """Batch count vector with the SAME Boolean refusal as the scalar route (STAT-RR21-04).

    ``np.asarray([...], dtype=float)`` coerces ``True`` to 1.0 silently -- exactly the mis-wired
    ``any(events)``-indicator shape the scalar route refuses by contract; a batch of them is the
    same mistake at scale, and mixed lists (``[True, 3]``) coerce through int64 with no bool dtype
    left to detect, so the ORIGINAL items are checked.
    """
    items = list(raw)
    if any(isinstance(value, (bool, np.bool_)) for value in items):
        raise TypeError(f"{name} must contain numbers, not Booleans")
    arr = np.asarray(items, dtype=float)
    return arr


def _positive_exposure(name: str, value: Any) -> float:
    """Exposure scalar with the scalar route's Boolean refusal (STAT-RR21-04)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number, not a Boolean; got {value!r}")
    return float(value)


def poisson_lograte_effects(
    k_pre: Any, t_pre: float, k_post: Any, t_post: float
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Per-subject log rate-ratio effects for Gaussian pooling, via the CONDITIONAL (binomial) route.

    Conditioning is what makes the heterogeneous estimand reachable. Given a subject's total
    ``n_i = k_pre,i + k_post,i`` and the common exposure ratio ``r = t_post/t_pre``,
    ``k_post,i | n_i ~ Binomial(n_i, p_i)`` with ``logit(p_i) = log(theta_i) + log(r)`` -- the
    subject's BASELINE RATE lambda_i cancels exactly, so nothing here assumes rate homogeneity.
    Two earlier constructions failed measurably on that point (pass 19 and its pass-20 replay):

    1. Count-estimated per-subject variances (``1/(k+0.5)`` forms) let inverse-variance weights
       correlate with the effects (measured corr(1/v, y) = -0.72; weighted true-null mean -0.150;
       z = -7.8 at n = 1000, 100% false rejection, flat in n).
    2. Debiasing at the pooled ARM means fixed the homogeneous fixture but assumed every subject
       shares one baseline rate: with half the subjects at rate 0.1 and half at 9.1 (exposures
       1:3) and every true ratio exactly 1, it rejected 400/400 with mean z -20.75/-65.62 at
       n = 1e3/1e4 (STAT-RR19-03) -- the nonlinear Haldane bias must be removed at each subject's
       own law, which is unknowable at the rate scale and EXACT at the conditional scale.

    Here each effect is the Haldane conditional logit ``log((k_post+0.5)/(k_pre+0.5)) - log(r)``,
    debiased by the exact pmf-summation bias of that statistic under ``Binomial(n_i, p_bar)``
    (``p_bar`` = pooled working proportion; under the null every subject's ``p_i`` equals it
    exactly, whatever the baseline rates), with the exact conditional variance at ``(n_i, p_bar)``
    as the pooling variance -- a function of ``n_i`` alone given ``p_bar``, so weights cannot
    correlate with the per-subject noise. Zero-total subjects (``n_i = 0``) carry no information
    about a rate ratio and are excluded from the returned arrays.

    Measured on the pass-20 fixtures through the real DL pool: heterogeneous 0.1/9.1 (r = 3) true
    null rejects 0.055 at n = 1000 (400 reps) and 0.0417 at n = 10000 (1200 reps); the milder 2/8
    (r = 2) mixture 0.018; the homogeneous 4.6 (r = 3) fixture 0.048; power at a true common
    ratio of 1.3 on the 2/8 mixture is 1.000. The 4.0 arm-mean floor remains as the pooled-z
    normal-approximation gate; below it use poisson_pooled_rate_ratio, exact at any sparsity for
    the common-ratio estimand.

    Returns ``(effects, variances, selection_receipt)``. The receipt records the POPULATION this
    route can speak for (STAT-RR22-01): zero-total subjects are excluded because the conditional
    likelihood is empty for them, and that exclusion is OUTCOME-DEPENDENT -- under extreme
    baseline heterogeneity (rates 0.01 and 10, true effects +1/-1 with all-subject ATT exactly
    zero), event-free low-rate subjects vanished differentially and the retained-subject contrast
    read -0.94 with 100% rejection while silently wearing the all-treated label. Hand the receipt
    to :func:`hierarchical_event_study` so the estimand names the population actually estimated.
    """
    k_pre_arr = _batch_counts("k_pre", k_pre)
    k_post_arr = _batch_counts("k_post", k_post)
    t_pre = _positive_exposure("t_pre", t_pre)
    t_post = _positive_exposure("t_post", t_post)
    if k_pre_arr.shape != k_post_arr.shape or k_pre_arr.ndim != 1 or k_pre_arr.size == 0:
        raise ValueError("k_pre and k_post must be equal-length non-empty count vectors")
    mean_pre, mean_post = float(k_pre_arr.mean()), float(k_post_arr.mean())
    if mean_pre < 4.0 or mean_post < 4.0:
        raise ValueError(
            f"sparse-count regime (arm means {mean_pre:.2f} pre / {mean_post:.2f} post are below "
            "the measured floor of 4.0): the per-subject Gaussian approximation is unreliable "
            "here -- use poisson_pooled_rate_ratio, which is exact at any sparsity for the "
            "common rate ratio"
        )
    if not (np.isfinite(t_pre) and np.isfinite(t_post) and t_pre > 0 and t_post > 0):
        raise ValueError("exposures must be finite and positive")
    for name, arr in (("k_pre", k_pre_arr), ("k_post", k_post_arr)):
        if np.any(arr < 0) or np.any(arr != np.floor(arr)) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain non-negative integer counts")
    r = float(t_post) / float(t_pre)
    totals = (k_pre_arr + k_post_arr).astype(int)
    keep = totals > 0
    if not np.any(keep):
        raise ValueError("every subject has zero events in both windows: no rate-ratio information")
    k_post_kept = k_post_arr[keep]
    k_pre_kept = k_pre_arr[keep]
    totals_kept = totals[keep]
    p_bar = float(k_post_kept.sum() / totals_kept.sum())
    p_bar = min(max(p_bar, 1e-12), 1.0 - 1e-12)
    null_logit = float(np.log(p_bar / (1.0 - p_bar)))
    bias_by_total: dict[int, float] = {}
    var_by_total: dict[int, float] = {}
    for total in np.unique(totals_kept):
        m1, v = _haldane_logit_moments(int(total), p_bar)
        bias_by_total[int(total)] = m1 - null_logit
        var_by_total[int(total)] = v
    effects = (
        np.log((k_post_kept + 0.5) / (k_pre_kept + 0.5))
        - np.log(r)
        - np.asarray([bias_by_total[int(total)] for total in totals_kept])
    )
    variances = np.asarray([var_by_total[int(total)] for total in totals_kept])
    selection_receipt = {
        "n_subjects": int(totals.size),
        "n_kept": int(totals_kept.size),
        "n_dropped_zero_total": int(totals.size - totals_kept.size),
    }
    return effects, variances, selection_receipt


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

    k_pre_arr = _batch_counts("k_pre", k_pre)
    k_post_arr = _batch_counts("k_post", k_post)
    t_pre = _positive_exposure("t_pre", t_pre)
    t_post = _positive_exposure("t_post", t_post)
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
    # STAT-RR22-06: the level the interval was computed at -- __str__ used to hardcode "95% CI"
    # while alpha=0.1 endpoints rendered under that label
    ci_level: float = 0.95
    # STAT-RR22-01: how many subjects each arm STARTED with before any route-level exclusion
    # (zero-total subjects under the conditional Poisson route); equal to n_treated/n_control
    # when nothing was excluded
    n_treated_population: int | None = None
    n_control_population: int | None = None
    # STAT-RR23-03: the reference behind `z`, `p_value`, and `ci`. None means the standard
    # normal (the association paths); a number means Student-t with these Welch degrees of
    # freedom -- `z` then HOLDS the t statistic (the field name is kept for compatibility, the
    # reference is not), and consumers converting `z` to a p-value must use this df, not a
    # normal table (2.236 reads p=.025 normal but p=.038 at the actual Welch df).
    df: float | None = None

    def __str__(self) -> str:
        c = "" if self.control_mean is None else f", control {self.control_mean:+.4f}"
        return (
            f"EventStudyResult(estimand={self.estimand!r}, identified={self.identified}, "
            f"effect={self.effect:+.4f} ± {self.se:.4f}, "
            f"{self.ci_level:.0%} CI [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}], "
            f"{'z' if self.df is None else f't(df={self.df:.1f})'}={self.z:.2f}, "
            f"p={self.p_value:.2e}, "
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
    treated_selection: dict[str, int] | None = None,
    control_selection: dict[str, int] | None = None,
) -> EventStudyResult:
    """Pool per-subject changes into an association or explicitly identified DiD estimate.

    ``*_effects`` / ``*_vars`` are per-subject changes and sampling variances. A control group makes the
    numeric contrast ``treated_mean - control_mean``. It is labeled an ATT only when a complete
    ``identification`` receipt is attached; otherwise it remains a treated/control change association.

    THE WEIGHTING FOLLOWS THE ESTIMAND (STAT-RR21-01). The ATT is the treated-population
    ARITHMETIC mean of unit effects, so the identified path uses equal-subject-weight group means
    with their empirical standard errors (``sd/sqrt(n)`` per group -- per-subject sampling noise
    is inside the observed spread, so this SE covers noise and heterogeneity together).
    DerSimonian-Laird precision weighting estimates a DIFFERENT quantity whenever effect size and
    sampling variance are dependent -- with treated unit effects evenly split between +1 and -1
    (true ATT exactly 0), precision weights track the totals that track the effects, and the DL
    contrast reported +0.197 with 92-100% rejection at nominal 5%. DL pooling remains what the
    ASSOCIATION paths report (a precision-weighted summary, named as such), and its tau^2 /
    shrunk-effect diagnostics are attached in both modes.

    THE POPULATION FOLLOWS THE RECEIPT (STAT-RR22-01). The conditional Poisson route excludes
    zero-total subjects -- an OUTCOME-DEPENDENT exclusion, so its arithmetic contrast estimates
    the event-positive population, not all treated units (measured: with baseline rates 0.01/10
    and true all-subject ATT exactly zero, differential exclusion drove the retained-subject
    contrast to -0.94 with 100% rejection). Pass each arm's ``selection_receipt`` (the third
    return of :func:`poisson_lograte_effects`); when any subjects were excluded, the identified
    estimand names the event-positive population explicitly and the result carries the original
    denominators. Refusing to attach the all-treated label is the point: this route cannot
    identify it.

    IDENTIFIED-PATH UNCERTAINTY (STAT-RR22-02): the reference is Student-t with Welch-
    Satterthwaite degrees of freedom (a normal quantile rejected 18.96% at nominal 5% with 2
    subjects per arm), the empirical variance is FLOORED by the supplied per-subject sampling
    variances (``mean(vars)/n`` per arm -- a degenerate all-identical-estimates arm used to
    report SE 0 and the contradiction ``CI=[1,1]`` with ``p=1``), the p-value and CI come from
    the same t reference, and subjects are assumed INDEPENDENT units: with 10 clusters repeated
    10 times per arm, rejection measured 55.6% at nominal 5% -- aggregate to cluster-level
    effects first.
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

    def _selection_counts(name: str, receipt: dict[str, int] | None, n_used: int) -> tuple[int, int]:
        # Omitting a receipt is the caller's ASSERTION that these effect arrays were not
        # outcome-selected (the Poisson batch route always returns one -- pass it through).
        if receipt is None:
            return n_used, 0
        required = {"n_subjects", "n_kept", "n_dropped_zero_total"}
        if not required.issubset(receipt):
            raise ValueError(f"{name} must carry n_subjects/n_kept/n_dropped_zero_total")
        # STAT-RR23-01: `int()` coercion accepted booleans, fractional counts (silently
        # truncated), negative drops, and populations smaller than the kept count -- a receipt
        # is selection PROVENANCE and nonsense provenance must refuse, not normalize.
        values = {}
        for key in required:
            value = receipt[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name}[{key!r}] must be an exact integer, got {value!r}")
            values[key] = int(value)
        if min(values.values()) < 0:
            raise ValueError(f"{name} counts must be non-negative, got {values}")
        if values["n_kept"] != n_used:
            raise ValueError(
                f"{name} says {values['n_kept']} subjects were kept but {n_used} effects were "
                "supplied -- the receipt must describe exactly these arrays"
            )
        if values["n_subjects"] != values["n_kept"] + values["n_dropped_zero_total"]:
            raise ValueError(
                f"{name} is internally inconsistent: n_subjects ({values['n_subjects']}) != "
                f"n_kept ({values['n_kept']}) + n_dropped_zero_total "
                f"({values['n_dropped_zero_total']})"
            )
        return values["n_subjects"], values["n_dropped_zero_total"]

    n_t_population, n_t_dropped = _selection_counts("treated_selection", treated_selection, len(y_t))
    n_c_population, n_c_dropped = (
        _selection_counts("control_selection", control_selection, len(y_c)) if has_control else (0, 0)
    )
    student_df = None
    if identified:
        any_dropped = n_t_dropped > 0 or n_c_dropped > 0
        if any_dropped:
            # STAT-RR22-01 -> STAT-RR23-01: outcome-dependent exclusion does not merely rename
            # the population -- it breaks the estimator for EVERY causal mean. The conditional
            # debias is computed at one pooled working probability, which is exact under a
            # common conditional law but biased for heterogeneous effects even as an estimate of
            # the event-positive mean: measured bias -0.092 with 0/80 nominal-95% coverage at
            # 4,000 subjects per arm (the SE shrinks, the bias does not). No relabeled ATT
            # survives that; the result is a SELECTED-SAMPLE ASSOCIATION, stated as such.
            identified = False
            estimand = (
                "selected-sample (event-positive) treated-minus-control change association "
                f"({len(y_t)}/{n_t_population} treated, {len(y_c)}/{n_c_population} controls "
                "retained after outcome-dependent zero-total exclusion; neither the all-treated "
                "nor the event-positive ATT is identified by this estimator -- the pooled-"
                "working-probability debias is biased under heterogeneous effects, measured "
                "-0.09 with 0% coverage of the event-positive mean)"
            )
            interpretation = (
                "association only: the identification receipt was supplied, but the estimator "
                "cannot deliver an unbiased causal mean for any pre-specifiable population once "
                "subjects are excluded by their outcomes (STAT-RR23-01); use "
                "poisson_pooled_rate_ratio for the exact common-ratio estimand, which needs no "
                "per-subject exclusion"
            )
        else:
            estimand = "difference-in-differences average treatment effect on the treated"
            interpretation = (
                "causal contrast under the attached identification assumptions; equal-subject-"
                "weight (arithmetic) group means, Student-t reference with Welch df, empirical "
                "SEs floored by the supplied sampling variances (STAT-RR21-01/RR22-02); subjects "
                "are assumed INDEPENDENT units -- aggregate clustered designs to cluster level "
                "first -- and the finite-sample t reference additionally assumes near-NORMAL arm "
                "means: a legal skewed unit-effect law (0.95 at -1, 0.05 at +19, exact mean "
                "zero) measured 31.0% rejection at nominal 5% with five treated subjects "
                "(STAT-RR23-02) -- treat small-arm p-values as approximate and grow the arms "
                "when unit effects may be heavy-tailed"
            )
        # The ATT is an ARITHMETIC mean over treated units: estimate it with equal weights and the
        # groups' own empirical spread. The DL numbers computed above stay as RE diagnostics.
        if len(y_t) < 2 or len(y_c) < 2:
            raise ValueError(
                "the identified ATT path needs at least 2 treated and 2 control subjects: the "
                "arithmetic-mean contrast takes its SE from each group's empirical spread"
            )
        t_mean = float(y_t.mean())
        c_mean = float(y_c.mean())
        # empirical variance covers noise + heterogeneity in expectation, but a degenerate arm
        # (all estimates identical) reported SE 0 beside supplied positive sampling variances --
        # and then CI [1,1] with p = 1 (STAT-RR22-02). Floor each arm at the known-noise
        # contribution mean(vars)/n; Welch df from the arm variances actually used.
        t_var_emp = float(y_t.var(ddof=1) / len(y_t))
        c_var_emp = float(y_c.var(ddof=1) / len(y_c))
        t_var = max(t_var_emp, float(np.mean(v_t) / len(y_t)))
        c_var = max(c_var_emp, float(np.mean(v_c) / len(y_c)))
        effect, var = t_mean - c_mean, t_var + c_var
        denominator = (t_var**2) / (len(y_t) - 1) + (c_var**2) / (len(y_c) - 1)
        student_df = float((t_var + c_var) ** 2 / denominator) if denominator > 0 else float(len(y_t) + len(y_c) - 2)
    elif has_control:
        estimand = "treated-minus-control change association (precision-weighted)"
        interpretation = "association only; parallel trends and causal assumptions were not established"
    else:
        estimand = "before-after association (precision-weighted)"
        interpretation = "association only; no control group or causal identification was supplied"

    se = float(np.sqrt(var))
    z = effect / se if se > 0 else 0.0
    from math import sqrt

    if student_df is not None:
        # STAT-RR22-02: the identified path's small-sample reference is Student-t (a normal
        # quantile rejected 18.96% at nominal 5% with 2 subjects per arm), and the p-value and
        # CI use the SAME reference so they can never contradict each other.
        from scipy import stats as _scipy_stats

        p_value = float(2.0 * _scipy_stats.t.sf(abs(z), student_df))
        quantile = float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, student_df))
    else:
        p_value = float(2 * _norm_sf(z))
        quantile = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _inv_norm_sf(alpha / 2)
    half = quantile * se
    return EventStudyResult(
        effect=effect,
        se=se,
        z=float(z),
        p_value=p_value,
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
        ci_level=1.0 - alpha,
        n_treated_population=n_t_population,
        n_control_population=n_c_population if has_control else None,
        df=student_df,
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
    (``= effect``) and the value that pushes the result's OWN confidence interval through zero -- at
    the result's own level and reference (STAT-RR23-03: a hard-coded Normal 1.96 disagreed with the
    identified path's Welch-t interval edge, 0.1235 vs the actual 0.0604, and silently priced a 90%
    result at 95%). Larger = more robust.
    """
    edge = result.ci[0] if result.effect > 0 else result.ci[1]
    return {
        "drift_to_nullify_point": float(result.effect),
        # the drift that moves the interval to touch zero = the near edge itself
        "drift_to_nullify_ci": float(edge),
        "effect_in_se_units": float(result.z),
        "ci_level": float(result.ci_level),
        "reference": "normal" if result.df is None else f"student-t(df={result.df:.2f})",
    }
