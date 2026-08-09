"""Calibration gate -- a deployable IC-6 verifier that challenges whether a posterior's reported
uncertainty is actually *earned*, instead of checking only its shape.

Motivation (found empirically, across several end-to-end pipelines): the UQ in this codebase is
*locally honest* -- every module computes a real posterior -- but was *globally unaudited*. A
systematically overconfident or miscalibrated posterior sailed straight through to a confident-looking
final answer, because nothing in the flow ever challenged the confidence. The calibration *primitives*
already existed (:mod:`mixle.inference.calibration`: ``pit_ensemble`` / ``interval_coverage`` /
``coverage_curve`` / ``pit_calibration_error``); what was missing was a *gate* that composes them into
a pass/fail verifier a pipeline (e.g. :func:`mixle.task.knowledge_routing.route_task`) can actually
stop on. This module is that gate -- a thin composition, not new statistics.

**Honest boundary, stated up front rather than buried.** This gate catches:

  * *overconfidence / underconfidence* of a posterior relative to HELD-OUT DATA it can be checked
    against (:func:`posterior_predictive_calibration`), and
  * *inference-algorithm bugs* -- an inference that is not self-consistent under its own generative
    model -- via simulation-based calibration (:func:`simulation_based_calibration`).

It does **not** catch model misspecification under genuine non-uniqueness. If the data you can hold
out simply cannot distinguish the true state (the textbook example: surface gravity cannot resolve a
source's depth -- shallow and deep bodies produce nearly identical surface fields), then a biased
posterior fits the held-out data perfectly and this gate will *correctly* pass it. Catching that
requires either data that can see the biased dimension (e.g. one borehole) or a physics-based prior
-- not more calibration checking. This gate reports what the available data can support; it never
manufactures confidence the data cannot justify, and it never claims to.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.random import RandomState

from mixle.inference.calibration import (
    coverage_curve,
    interval_coverage,
    pit_calibration_error,
    pit_ensemble,
)
from mixle.utils.immutable import detach_receipt_container

__all__ = [
    "CalibrationStatus",
    "CalibrationVerdict",
    "posterior_predictive_calibration",
    "simulation_based_calibration",
    "CalibrationVerifier",
]

CalibrationStatus = Literal["passed", "failed", "indeterminate"]


@dataclass(frozen=True)
class CalibrationVerdict:
    """The outcome of a calibration check with an explicit three-way decision state.

    The DECISION is a Monte-Carlo p-value compared to ``1 - null_quantile`` (STAT-RR17-08): the
    predictive gate uses a swap-randomization null that is exact under per-row exchangeability
    whatever the cross-row dependence of the ensemble, and SBC uses an i.i.d.-uniform null,
    which its protocol genuinely satisfies; both use ``(1 + count)/(1 + B)``, level-valid at any
    replicate count. ``null_threshold`` remains as a descriptive scale and the ``score`` anchor,
    never the decision constant (deciding against that estimated quantile realized levels of
    0.0033-0.0181 at nominal 0.01 depending only on its seed). ``calibration_status`` is
    ``'passed'`` only when the p-value clears alpha AND measured power against the named
    canonical alternative is adequate AND randomness is controlled; ``'indeterminate'`` means
    the data cannot support either promotion or rejection; ``'failed'`` means the decision rule
    rejected, which is real evidence of a problem even when power was otherwise low. A gate may
    promote only the literal ``'passed'`` state.
    """

    calibration_status: CalibrationStatus
    pit_error: float  # deviation of the PIT/rank histogram from uniform (0 == perfectly calibrated)
    null_threshold: float  # the sample-size-aware threshold pit_error is judged against
    coverage_error: float  # DIAGNOSTIC: max |empirical - nominal| across the coverage curve
    reference_level: float
    coverage_at_reference: float  # DIAGNOSTIC: empirical coverage of the reference-level central interval
    mean_interval_width: float
    n_points: int
    power_sufficient: bool = True
    expected_count_per_bin: float = float("nan")
    randomness_controlled: bool = True
    reasons: list[str] = field(default_factory=list)
    kind: str = "calibration"
    # STAT-RR17-08: the DECISION is a Monte-Carlo p-value that is level-valid at any replicate
    # count -- (1 + #{T_null >= T_obs}) / (1 + B) -- never a comparison against the (noisy,
    # fixed-seed) null-quantile estimate, which realized levels of 0.0033-0.0181 at nominal 0.01
    # depending only on the threshold seed. null_threshold remains as the score anchor and a
    # descriptive scale. p_value is NaN when an explicit tolerance override decided instead.
    p_value: float = float("nan")
    n_null: int = 0
    # measured Monte-Carlo power of the level-(1 - null_quantile) decision against ONE named
    # canonical alternative (0.8-dispersion mis-calibration), continuous-PIT approximation; the
    # old expected-count and threshold-magnitude heuristics computed no power at all
    power_estimate: float = float("nan")
    power_alternative: str = ""
    power_estimate_independent: float = float("nan")
    power_estimate_shared: float = float("nan")
    # the caller's declared ensemble construction; promotion power is gated on THIS regime's
    # executed power, and a misdeclaration is the caller asserting a false premise on record
    declared_dependence: str = ""
    # finite-m null expectation of the reference-interval's empirical coverage under
    # exchangeability (rank Monte Carlo): the direction label compares against THIS, because at
    # small m a perfectly calibrated ensemble's plug-in interval covers well below nominal
    # (~0.78 at m=8 for 0.90), and comparing to nominal called calibrated posteriors
    # "overconfident" by construction
    coverage_at_reference_null_expectation: float = float("nan")

    def __post_init__(self) -> None:
        # A receipt is a record. Detaching severs the caller's alias, so a mutation after
        # construction cannot rewrite evidence that was already recorded; `frozen=True` above
        # stops the field being rebound through the receipt itself. Containers keep their
        # concrete types -- see detach_receipt_container for why (MXR-080-1876).
        object.__setattr__(self, "reasons", detach_receipt_container(self.reasons))
        if self.calibration_status not in ("passed", "failed", "indeterminate"):
            raise ValueError(
                "calibration_status must be exactly 'passed', 'failed', or 'indeterminate'; "
                f"got {self.calibration_status!r}."
            )
        if self.calibration_status == "passed" and (not self.power_sufficient or not self.randomness_controlled):
            raise ValueError("a passed calibration verdict requires adequate power and controlled randomness")

    @property
    def passed(self) -> bool:
        """Whether this result is affirmative evidence suitable for promotion."""
        return self.calibration_status == "passed"

    @property
    def indeterminate(self) -> bool:
        """Whether the check lacks enough evidence to pass or fail."""
        return self.calibration_status == "indeterminate"

    @property
    def low_power(self) -> bool:
        """Compatibility diagnostic for indeterminate results caused by inadequate power."""
        return self.indeterminate and not self.power_sufficient

    @property
    def score(self) -> float:
        """A 0..1 ranking score: 1 at zero PIT error, 0.5 exactly AT the calibrated-null threshold,
        0 at twice the threshold, linear in between. Normalized by the null threshold so it is
        comparable across sample sizes. It is NOT 1 for every passing posterior -- a genuinely
        calibrated posterior typically lands around 0.6-0.7, so rank with it, never gate on a
        fixed cutoff like 0.9 (use ``calibration_status`` to gate)."""
        if self.null_threshold <= 0:
            return 1.0 if self.pit_error == 0 else 0.0
        return float(max(0.0, 1.0 - self.pit_error / (2.0 * self.null_threshold)))


def _uniformity_null_threshold(
    n: int, *, bins: int = 10, quantile: float = 0.99, n_null: int = 500, seed: int = 12345
) -> float:
    """The value :func:`~mixle.inference.calibration.pit_calibration_error` would reach on genuinely
    uniform data of size ``n`` -- Monte-Carlo'd at the ``quantile`` upper tail. Comparing an observed
    PIT/rank error against THIS (rather than a fixed constant) is what makes the gate a proper
    finite-sample test: it asks "is this posterior worse-calibrated than 99% of genuinely calibrated
    posteriors of the same sample size would be?", which is scale-correct at every ``n``."""
    rs = RandomState(seed)
    errs = [pit_calibration_error(rs.uniform(0.0, 1.0, size=int(n)), bins=bins) for _ in range(int(n_null))]
    return float(np.quantile(errs, quantile))


def _uniform_mc_pvalue(t_obs: float, n: int, *, bins: int, n_null: int, seed: int) -> float:
    """Monte-Carlo p-value of the PIT-uniformity statistic under an i.i.d. Uniform(0,1) null.

    ``(1 + #{T_null >= T_obs}) / (1 + n_null)`` is level-valid at ANY replicate count (the +1s
    count the observed configuration among the replicates), which is what replaces the previous
    compare-to-estimated-quantile rule whose realized level depended on the threshold seed
    (STAT-RR17-08). Valid when the PIT/rank values are independent across entries -- true for
    SBC by construction (each simulation draws its own parameters and data).
    """
    rs = RandomState(seed)
    count = sum(
        1 for _ in range(int(n_null)) if pit_calibration_error(rs.uniform(0.0, 1.0, size=int(n)), bins=bins) >= t_obs
    )
    return float((1.0 + count) / (1.0 + int(n_null)))


def _column_swap_pvalue(y: np.ndarray, ens: np.ndarray, *, bins: int, pit_seed: Any) -> tuple[float, float]:
    """Exact randomization p-value for PIT uniformity under COLUMN exchangeability.

    The randomization group swaps the observation vector with one whole ensemble column: under
    calibration, ``(y, col_1, ..., col_m)`` are exchangeable AS VECTORS both when the columns
    share posterior draws (the documented construction) and when all entries are independent.
    Exactness additionally requires ONE fixed statistic applied to every relabeling, so a single
    jitter seed -- drawn once from ``pit_seed`` -- randomizes the PIT of the observed AND every
    swapped configuration identically (pass 18 measured the varying-jitter version rejecting
    83.1% of a fully i.i.d. null at a claimed 21% level; the same-jitter control measured 13.0%).
    Returns ``(t_obs_decision, p)`` with ``p = (1 + #{T_j >= T_obs})/(1 + m)``, level-valid and
    quantized at ``1/(m+1)`` -- with few draws per point the test CANNOT reject at small alpha,
    which the measured power reports honestly.
    """
    k, m = ens.shape
    base = int(pit_seed) if isinstance(pit_seed, (int, np.integer)) else 0
    jitter_seed = (base * 2654435761 + 97) % (2**31 - 1)
    t_obs = float(pit_calibration_error(pit_ensemble(y, ens, randomize=True, seed=jitter_seed), bins=bins))
    count = 0
    for j in range(m):
        y_j = ens[:, j].copy()
        ens_j = ens.copy()
        ens_j[:, j] = y
        pit_j = pit_ensemble(y_j, ens_j, randomize=True, seed=jitter_seed)
        if pit_calibration_error(pit_j, bins=bins) >= t_obs:
            count += 1
    return t_obs, float((1.0 + count) / (1.0 + m))


def _measured_power(
    n: int,
    *,
    bins: int,
    alpha: float,
    n_null: int,
    m: int | None = None,
    n_alternative: int = 200,
    seed: int = 20260808,
) -> tuple[float, str]:
    """Measured power of the SBC decision (i.i.d. MC p-value) at this sample size.

    Against ONE named canonical alternative -- predictive dispersion 0.8x the truth, PIT values
    ``Phi(z/0.8)``. This executes the implemented decision: reject when
    ``(1 + #{T_null >= T_alt})/(1 + B) <= alpha``. Used by SBC only; the predictive gate's power
    runs the column-swap decision itself under both supported dependence regimes
    (:func:`_measured_gate_power`).
    """
    from scipy.stats import norm as _norm

    alternative = "predictive dispersion 0.8x truth (PIT = Phi(z/0.8))"
    del m
    rs = RandomState(seed)
    null_stats = np.sort(
        [pit_calibration_error(rs.uniform(0.0, 1.0, size=int(n)), bins=bins) for _ in range(int(n_null))]
    )
    hits = 0
    for _ in range(int(n_alternative)):
        t_alt = pit_calibration_error(_norm.cdf(rs.standard_normal(int(n)) / 0.8), bins=bins)
        count = int(null_stats.size - np.searchsorted(null_stats, t_alt, side="left"))
        if (1.0 + count) / (1.0 + null_stats.size) <= alpha:
            hits += 1
    return float(hits / float(n_alternative)), alternative


def _measured_gate_power(
    k: int,
    m: int,
    *,
    bins: int,
    alpha: float,
    n_alternative: int = 60,
    budget_ops: float = 6.5e9,
    seed: int = 20260809,
) -> tuple[float, float, str]:
    """Power of the ACTUAL column-swap decision, executed under BOTH supported dependence regimes.

    Pass 18 measured the previous helper -- an independent-rows approximation that never executed
    the decision -- reporting 0.95 where the shared-column regime's true rejection rate was 1.8%
    and even the independent regime's was 0.866; a shared-column mis-calibrated posterior was then
    PROMOTED on that fictitious power. Here each alternative replicate runs
    :func:`_column_swap_pvalue` itself: 'independent' draws every entry independently, 'shared'
    gives each ensemble column a shared component (the documented same-posterior-draws
    construction). The promotion gate uses the MINIMUM of the two. Returns ``(nan, nan, ...)``
    when execution would exceed the operation budget -- an unmeasured power never promotes
    (claim reduction, per the pass-18 closure requirement).
    """
    alternative = (
        "predictive marginal SD 0.8x truth (executed decision; min over independent and shared-column regimes)"
    )
    if 1.0 / (m + 1.0) > alpha:
        return 0.0, 0.0, alternative  # p-values are quantized above alpha: the test cannot reject
    per_replicate = 2.0 * (m + 1.0) * float(k) * float(m)  # both regimes execute the full decision
    n_alternative = int(min(n_alternative, np.floor(float(budget_ops) / per_replicate)))
    if n_alternative < 24:
        # fewer than 24 executed replicates cannot resolve the 0.5 promotion floor; an
        # unmeasured power never promotes (the caller sees indeterminate with this reason)
        return float("nan"), float("nan"), alternative
    rs = RandomState(seed)
    hits_independent = 0
    hits_shared = 0
    for i in range(int(n_alternative)):
        y = rs.standard_normal(k)
        ens = 0.8 * rs.standard_normal((k, m))
        _, p_ind = _column_swap_pvalue(y, ens, bins=bins, pit_seed=seed + 3 * i)
        hits_independent += p_ind <= alpha
        shared_truth = rs.standard_normal()
        y2 = shared_truth + rs.standard_normal(k)
        cols = rs.standard_normal(m)
        ens2 = 0.8 * (cols[None, :] + rs.standard_normal((k, m)))
        _, p_sh = _column_swap_pvalue(y2, ens2, bins=bins, pit_seed=seed + 3 * i + 1)
        hits_shared += p_sh <= alpha
    return (
        float(hits_independent / float(n_alternative)),
        float(hits_shared / float(n_alternative)),
        alternative,
    )


def _reference_coverage_null_expectation(m: int, level: float, *, n_sims: int = 4000, seed: int = 5) -> float:
    """Finite-``m`` null expectation of the reference interval's empirical coverage.

    Uniform-reference simulation of the same quantile rule. NEARLY distribution-free: whether
    ``y`` falls inside the interval would depend only on ranks if the endpoints were order
    statistics, but linear quantile interpolation adds a small shape dependence at very small
    ``m`` (measured expected coverage 0.632 / 0.644 / 0.636 for uniform / Gaussian / exponential
    ensembles at m = 5, nominal 0.95; the spread vanishes by m ~ 50) -- an order of magnitude
    below the nominal-vs-finite-m gap this reference exists to correct, and well inside the
    direction label's tolerance. The direction label must compare against
    THIS -- at small ``m`` the plug-in interval of a PERFECTLY calibrated ensemble covers well
    below nominal, and comparing to the nominal level branded calibrated posteriors
    "overconfident" by construction (STAT-RR17-08 / audit GATE-3).
    """
    rs = RandomState(seed)
    draws = rs.uniform(0.0, 1.0, size=(int(n_sims), int(m)))
    y = rs.uniform(0.0, 1.0, size=int(n_sims))
    lo = np.quantile(draws, (1.0 - level) / 2.0, axis=1)
    hi = np.quantile(draws, (1.0 + level) / 2.0, axis=1)
    return float(np.mean((y >= lo) & (y <= hi)))


def _validate_gate_parameters(
    *,
    reference_level: float | None = None,
    null_quantile: float,
    tolerance: float | None,
    bins: int,
    low_power_threshold: float,
    min_expected_count_per_bin: float,
) -> None:
    if reference_level is not None and (not np.isfinite(reference_level) or not 0.0 < reference_level < 1.0):
        raise ValueError("reference_level must be finite and strictly between 0 and 1")
    if not np.isfinite(null_quantile) or not 0.0 < null_quantile < 1.0:
        raise ValueError("null_quantile must be finite and strictly between 0 and 1")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0.0):
        raise ValueError("calibration tolerance must be finite and nonnegative")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer greater than one")
    if not np.isfinite(low_power_threshold) or low_power_threshold < 0.0:
        raise ValueError("low_power_threshold must be finite and nonnegative")
    if not np.isfinite(min_expected_count_per_bin) or min_expected_count_per_bin <= 0.0:
        raise ValueError("min_expected_count_per_bin must be finite and positive")


def _power_is_sufficient(
    n: int,
    *,
    bins: int,
    null_threshold: float,
    low_power_threshold: float,
    min_expected_count_per_bin: float,
) -> tuple[bool, float]:
    expected = float(n) / float(bins)
    return (
        expected >= min_expected_count_per_bin and null_threshold < low_power_threshold,
        expected,
    )


def posterior_predictive_calibration(
    ensemble: np.ndarray,
    held_out_y: np.ndarray,
    *,
    reference_level: float = 0.90,
    null_quantile: float = 0.99,
    pit_tol: float | None = None,
    bins: int = 10,
    low_power_threshold: float = 1.0,
    min_expected_count_per_bin: float = 5.0,
    pit_seed: int | RandomState | None = 0,
    ensemble_dependence: Literal["shared-draws", "independent"] = "shared-draws",
) -> CalibrationVerdict:
    """Check a posterior-predictive ensemble against held-out observations it never saw.

    ``ensemble_dependence`` declares how the ensemble was BUILT, because the test's power (not
    its level -- the column-swap p-value is exact in both regimes) differs radically between
    them: ``"shared-draws"`` (the default, fail-safe) means the same posterior draws generated
    every row (the documented construction), where power is often low and promotion honestly
    refuses; ``"independent"`` asserts every row used fresh draws, unlocking that regime's
    (usually much higher) executed power. The declaration is the CALLER'S assertion about their
    own sampling code -- it is recorded in the verdict and the receipt, and misdeclaring it is
    how a low-power non-rejection would masquerade as affirmative evidence (pass 18 measured a
    shared-column mis-calibrated posterior promoted on an independent-rows power fiction of
    0.95 against an actual 0.018).

    ``ensemble`` is ``(k, m)``: for each of ``k`` held-out points, ``m`` posterior-predictive draws
    (push ``m`` draws from the posterior through the forward model to the observation of each held-out
    point). ``held_out_y`` is the ``(k,)`` array of real observed values at those points.

    The decision is an exact column-swap randomization test on the randomized PIT of the
    held-out data (STAT-RR17-08): the observation vector is swapped with each whole ensemble
    column in turn and the identical statistic recomputed, giving ``p = (1 + count)/(1 + m)``.
    Under calibration, ``(y, col_1..col_m)`` are exchangeable as vectors BOTH for the documented
    construction (the same ``m`` posterior draws pushed through the forward model for every
    point -- the i.i.d.-uniform reference this replaces wrongly failed a correctly calibrated
    shared-parameter posterior 298/300 times there) AND for fully independent entries, so the
    p-value is level-exact in either regime. It is quantized at ``1/(m+1)``: with few draws per
    point the test cannot reject at small alpha, and the measured power reports that honestly.
    The decision rejects when ``p <= 1 - null_quantile``. ``null_threshold`` is still computed
    as a descriptive scale and the ``score`` anchor, never the decision constant. Pass
    ``pit_tol`` to override with a fixed tolerance if you have a reason to (that branch reports
    no p-value).

    Coverage numbers (:func:`~mixle.inference.calibration.coverage_curve` /
    :func:`~mixle.inference.calibration.interval_coverage`) are computed and reported as human-readable
    diagnostics and to label the *direction* of any miscalibration (over- vs under-confident), but the
    pass/fail itself is the PIT test, which subsumes them and is scale-correct.

    A non-rejection is promotable only when MEASURED power is adequate: the Monte-Carlo power of
    this decision at this ``k`` against the named canonical alternative (predictive dispersion
    0.8x the truth) must reach 0.50, alongside the ``min_expected_count_per_bin`` data floor.
    The previous bin-count/threshold-magnitude heuristics computed no power at all and reported
    ``power_sufficient=True`` where the actual rejection rate against that alternative was ~4.5%
    (STAT-RR17-08). Power against ONE named alternative is a floor, not a universal power claim.
    Otherwise the result is ``indeterminate``, never a pass.
    """
    ens = np.asarray(ensemble, dtype=float)
    y = np.asarray(held_out_y, dtype=float)
    _validate_gate_parameters(
        reference_level=reference_level,
        null_quantile=null_quantile,
        tolerance=pit_tol,
        bins=bins,
        low_power_threshold=low_power_threshold,
        min_expected_count_per_bin=min_expected_count_per_bin,
    )
    if ens.ndim != 2:
        raise ValueError(f"ensemble must be (k, m); got shape {ens.shape}")
    if y.ndim != 1:
        raise ValueError(f"held_out_y must be a one-dimensional (k,) array; got shape {y.shape}")
    if ens.shape[0] == 0:
        raise ValueError("calibration requires at least one held-out point")
    if ens.shape[1] == 0:
        raise ValueError("calibration requires at least one posterior-predictive draw per point")
    if y.shape[0] != ens.shape[0]:
        raise ValueError(f"held_out_y has {y.shape[0]} points but ensemble has {ens.shape[0]}")
    if not np.all(np.isfinite(ens)) or not np.all(np.isfinite(y)):
        raise ValueError("ensemble and held_out_y must contain only finite values")

    if ensemble_dependence not in ("shared-draws", "independent"):
        raise ValueError("ensemble_dependence must be 'shared-draws' or 'independent'")
    k = int(y.shape[0])
    pit = pit_ensemble(y, ens, randomize=True, seed=pit_seed)
    pit_err = float(pit_calibration_error(pit, bins=bins))
    alpha = 1.0 - null_quantile
    n_null = 500
    threshold = _uniformity_null_threshold(k, bins=bins, quantile=null_quantile)
    if pit_tol is not None:
        # explicit caller override: a fixed tolerance decides, and no p-value is reported for it
        threshold = float(pit_tol)
        p_value = float("nan")
        passed = pit_err <= threshold
    else:
        # STAT-RR17-08: the decision is the exact swap-randomization Monte-Carlo p-value -- valid
        # at any replicate count and under ANY cross-row dependence of the ensemble (the i.i.d.
        # null failed a correctly calibrated shared-parameter posterior 298/300 times, and the
        # old compare-to-estimated-quantile rule's realized level was an arbitrary function of
        # the threshold seed: 0.0033-0.0181 measured at nominal 0.01)
        base = pit_seed if isinstance(pit_seed, (int, np.integer)) else 0
        _t_decision, p_value = _column_swap_pvalue(y, ens, bins=bins, pit_seed=pit_seed)
        passed = p_value > alpha
    power_independent, power_shared, power_alternative = _measured_gate_power(
        k, int(ens.shape[1]), bins=bins, alpha=alpha
    )
    power_estimate = float(power_independent if ensemble_dependence == "independent" else power_shared)
    legacy_power, expected_count = _power_is_sufficient(
        k,
        bins=bins,
        null_threshold=threshold,
        low_power_threshold=low_power_threshold,
        min_expected_count_per_bin=min_expected_count_per_bin,
    )
    # power is MEASURED against the named canonical alternative; the legacy bin-count floor is
    # kept as a data-sufficiency screen, but it never substitutes for measured power again
    # NaN (power unmeasurable within budget) intentionally fails this comparison: an unmeasured
    # power never promotes (pass-18 closure: measure under every supported regime or reduce)
    power_sufficient = bool(legacy_power and power_estimate >= 0.5)
    randomness_controlled = pit_seed is not None

    curve = coverage_curve(ens, y)
    coverage_error = float(np.max(np.abs(curve["empirical"] - curve["nominal"])))
    lo = np.quantile(ens, (1.0 - reference_level) / 2.0, axis=1)
    hi = np.quantile(ens, (1.0 + reference_level) / 2.0, axis=1)
    ref = interval_coverage(lo, hi, y)
    coverage_at_ref = float(ref["coverage"])
    coverage_null_expectation = _reference_coverage_null_expectation(int(ens.shape[1]), float(reference_level))

    reasons: list[str] = []
    if not passed:
        # direction judged against the FINITE-m null expectation, not the nominal level: a
        # perfectly calibrated ensemble's plug-in interval covers ~0.78 at m=8 for nominal 0.90,
        # so the nominal comparison branded calibrated posteriors overconfident by construction
        direction = (
            "overconfident (intervals too narrow -- reports false certainty)"
            if coverage_at_ref < coverage_null_expectation
            else "underconfident (intervals too wide)"
        )
        decided_by = (
            f"PIT error {pit_err:.3f} > tolerance {threshold:.3f}"
            if pit_tol is not None
            else f"column-swap exact p = {p_value:.4f} <= alpha = {alpha:.4f}"
        )
        reasons.append(
            f"miscalibrated: {decided_by} for k={k}; {reference_level:.0%} interval covers "
            f"{coverage_at_ref:.1%} of held-out points against a finite-m calibrated expectation "
            f"of {coverage_null_expectation:.1%} -- {direction}"
        )
    else:
        decided_by = (
            f"PIT error {pit_err:.3f} <= tolerance {threshold:.3f}"
            if pit_tol is not None
            else f"column-swap exact p = {p_value:.4f} > alpha = {alpha:.4f}"
        )
        reasons.append(
            f"not detectably miscalibrated: {decided_by} for k={k}; "
            f"{reference_level:.0%} interval covers {coverage_at_ref:.1%} of held-out points "
            f"(finite-m calibrated expectation {coverage_null_expectation:.1%})"
        )
    if not power_sufficient:
        reasons.append(
            "INDETERMINATE / LOW POWER: "
            + (
                f"power not measurable within budget at (k={k}, m={int(ens.shape[1])}) -- an unmeasured "
                "power never promotes; reduce m, or decide with an explicit pit_tol"
                if np.isnan(power_estimate)
                else f"executed-decision power in the DECLARED '{ensemble_dependence}' regime is "
                f"{power_estimate:.2f} (independent={power_independent:.2f}, "
                f"shared-column={power_shared:.2f}) against the canonical alternative "
                f"({power_alternative}) at k={k} with {expected_count:.2f} expected points per bin; "
                "promotion requires declared-regime power >= 0.50 and at least "
                f"{min_expected_count_per_bin:.2f} per bin. Hold out more independent points, or "
                "build the ensemble with fresh per-point draws and declare it."
            )
        )
    if not randomness_controlled:
        reasons.append(
            "INDETERMINATE / UNCONTROLLED RANDOMNESS: pit_seed=None makes randomized PIT tie handling "
            "non-replayable; supply an explicit seed before promotion."
        )

    # A rejection is real evidence of a problem even when the test had weak power. A non-rejection
    # is affirmative evidence only when the test had enough power to detect a material defect.
    calibration_status: CalibrationStatus = (
        "failed" if not passed else ("passed" if power_sufficient and randomness_controlled else "indeterminate")
    )

    return CalibrationVerdict(
        calibration_status=calibration_status,
        pit_error=pit_err,
        null_threshold=threshold,
        coverage_error=coverage_error,
        reference_level=float(reference_level),
        coverage_at_reference=coverage_at_ref,
        mean_interval_width=float(ref["mean_width"]),
        n_points=k,
        power_sufficient=power_sufficient,
        expected_count_per_bin=expected_count,
        randomness_controlled=randomness_controlled,
        reasons=reasons,
        p_value=p_value,
        n_null=0 if pit_tol is not None else int(ens.shape[1]),
        power_estimate=power_estimate,
        power_alternative=power_alternative,
        power_estimate_independent=power_independent,
        power_estimate_shared=power_shared,
        declared_dependence=ensemble_dependence,
        coverage_at_reference_null_expectation=coverage_null_expectation,
    )


def _rng_aware_fit_call(
    fit: Callable[..., np.ndarray],
) -> Callable[[np.ndarray, RandomState], np.ndarray]:
    """Resolve the explicit RNG call shape without exception-driven user-code retries."""
    try:
        signature = inspect.signature(fit)
    except (TypeError, ValueError) as exc:
        raise ValueError("fit must have an inspectable fit(y, rng) signature") from exc
    rng_parameter = signature.parameters.get("rng")
    if rng_parameter is None or rng_parameter.kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ):
        raise ValueError("fit must declare an explicit RandomState parameter named 'rng'")
    placeholder = object()
    if rng_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
        try:
            signature.bind(placeholder, rng=placeholder)
        except TypeError as exc:
            raise ValueError("fit must accept the calibration RandomState as fit(y, *, rng=...)") from exc
        return lambda y, rng: fit(y, rng=rng)
    try:
        signature.bind(placeholder, placeholder)
    except TypeError as exc:
        raise ValueError("fit must accept the calibration RandomState as fit(y, rng)") from exc
    return lambda y, rng: fit(y, rng)


def simulation_based_calibration(
    prior_sampler: Callable[[RandomState], np.ndarray],
    simulate: Callable[[np.ndarray, RandomState], np.ndarray],
    fit: Callable[[np.ndarray, RandomState], np.ndarray],
    *,
    n_sims: int = 200,
    param_index: int = 0,
    null_quantile: float = 0.99,
    error_tol: float | None = None,
    bins: int = 10,
    low_power_threshold: float = 1.0,
    min_expected_count_per_bin: float = 5.0,
    seed: int | RandomState | None = 0,
) -> CalibrationVerdict:
    """Simulation-based calibration (Talts et al. 2018): does the inference recover its own generative
    model? Draw ``theta ~ prior``, simulate ``y ~ p(y|theta)``, refit a posterior, and record the rank
    of the true ``theta`` among the posterior draws. Under a correct inference those ranks are Uniform;
    a systematically over/under-dispersed posterior makes them pile up at the middle or the edges.

    PRECONDITION (STAT-RR17-08 / audit GATE-6): rank uniformity requires the posterior draws each
    ``fit`` returns to be (near-)independent -- exchangeability of ``theta_true`` with the draws.
    An UNTHINNED autocorrelated MCMC chain from a CORRECT inference produces systematically
    non-uniform ranks and fails this gate without any defect in the inference; thin the chain to
    approximate independence first (Talts et al., section 4). The decision itself is the
    level-exact Monte-Carlo p-value against the i.i.d.-uniform null, which the SBC protocol
    genuinely satisfies across simulations (each draws its own parameters and data).

    ``prior_sampler(rng) -> theta`` (a ``(d,)`` parameter vector, or scalar-as-``(1,)``);
    ``simulate(theta, rng) -> y`` (any shape the fitter accepts);
    ``fit(y, rng) -> posterior_draws`` (``(n_draws, d)`` or ``(n_draws,)``). The fitter must derive
    all of its stochasticity from the supplied RNG; accepting an opaque ``fit(y)`` callable would
    make ``seed`` unable to control the inference being calibrated. ``param_index`` selects which
    parameter's rank to test.

    Unlike :func:`posterior_predictive_calibration`, this needs no held-out real data -- it tests the
    inference *machinery* against synthetic ground truth it generates itself. It therefore catches a
    different class: an inference bug / wrong likelihood / mis-scaled posterior, even on data that would
    look perfectly fit. (It does not test whether the model matches *reality* -- that is
    :func:`posterior_predictive_calibration`'s job, on real held-out data.)
    """
    _validate_gate_parameters(
        null_quantile=null_quantile,
        tolerance=error_tol,
        bins=bins,
        low_power_threshold=low_power_threshold,
        min_expected_count_per_bin=min_expected_count_per_bin,
    )
    if isinstance(n_sims, bool) or not isinstance(n_sims, (int, np.integer)) or n_sims <= 0:
        raise ValueError("n_sims must be a positive integer")
    if isinstance(param_index, bool) or not isinstance(param_index, (int, np.integer)) or param_index < 0:
        raise ValueError("param_index must be a nonnegative integer")

    fit_call = _rng_aware_fit_call(fit)
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    ranks = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        prior_rng = RandomState(int(rng.randint(0, 2**31 - 1)))
        simulate_rng = RandomState(int(rng.randint(0, 2**31 - 1)))
        fit_rng = RandomState(int(rng.randint(0, 2**31 - 1)))
        rank_rng = RandomState(int(rng.randint(0, 2**31 - 1)))
        theta = np.atleast_1d(np.asarray(prior_sampler(prior_rng), dtype=float))
        if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
            raise ValueError("prior_sampler must return a non-empty finite scalar or one-dimensional vector")
        if param_index >= theta.size:
            raise ValueError(f"param_index {param_index} is outside the sampled parameter vector")
        y = simulate(theta, simulate_rng)
        try:
            simulated = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("simulate must return numeric data") from exc
        if simulated.size == 0 or not np.all(np.isfinite(simulated)):
            raise ValueError("simulate must return non-empty finite data")
        draws = np.atleast_1d(np.asarray(fit_call(y, fit_rng), dtype=float))
        if draws.ndim not in (1, 2) or draws.size == 0 or not np.all(np.isfinite(draws)):
            raise ValueError("fit must return non-empty finite one- or two-dimensional posterior draws")
        if draws.ndim == 1:
            draws_param = draws
            theta_param = float(theta[0]) if theta.shape[0] == 1 else float(theta[param_index])
        else:
            if param_index >= draws.shape[1]:
                raise ValueError(f"param_index {param_index} is outside the fitted posterior draws")
            draws_param = draws[:, param_index]
            theta_param = float(theta[param_index])
        n_draws = draws_param.shape[0]
        less = int(np.sum(draws_param < theta_param))
        equal = int(np.sum(draws_param == theta_param))
        # Randomized SBC rank: insert truth uniformly among ties, then jitter within the selected
        # discrete rank cell. Under calibrated inference this is continuous Uniform(0, 1), matching
        # the continuous null used below instead of comparing a tie-biased lattice to that null.
        ranks[i] = (less + rank_rng.uniform(0.0, equal + 1.0)) / (n_draws + 1.0)

    # a uniform rank histogram means calibrated inference; reuse the same PIT-uniformity metric,
    # judged against the same sample-size-aware null threshold (n_sims here plays the role of k).
    sbc_error = float(pit_calibration_error(ranks, bins=bins))
    alpha = 1.0 - null_quantile
    n_null = 500
    threshold = _uniformity_null_threshold(int(n_sims), bins=bins, quantile=null_quantile)
    if error_tol is not None:
        threshold = float(error_tol)
        p_value = float("nan")
        within_threshold = sbc_error <= threshold
    else:
        # STAT-RR17-08: decide by the level-exact MC p-value, never the estimated quantile. The
        # i.i.d.-uniform null is CORRECT here (unlike the predictive gate): each SBC simulation
        # draws its own parameters and data, so ranks are independent across sims by protocol.
        p_value = _uniform_mc_pvalue(
            sbc_error, int(n_sims), bins=bins, n_null=n_null, seed=(0 if seed is None else int(seed)) + 404
        )
        within_threshold = p_value > alpha
    power_estimate, power_alternative = _measured_power(int(n_sims), bins=bins, alpha=alpha, n_null=n_null)
    legacy_power, expected_count = _power_is_sufficient(
        int(n_sims),
        bins=bins,
        null_threshold=threshold,
        low_power_threshold=low_power_threshold,
        min_expected_count_per_bin=min_expected_count_per_bin,
    )
    power_sufficient = bool(legacy_power and power_estimate >= 0.5)
    randomness_controlled = seed is not None
    calibration_status: CalibrationStatus = (
        "failed"
        if not within_threshold
        else ("passed" if power_sufficient and randomness_controlled else "indeterminate")
    )
    decided_by = (
        f"error {sbc_error:.3f} vs tolerance {threshold:.3f}"
        if error_tol is not None
        else f"MC p = {p_value:.4f} vs alpha = {alpha:.4f}"
    )
    if calibration_status == "failed":
        reasons = [
            f"SBC ranks NOT uniform ({decided_by}): inference is "
            "mis-dispersed (over/under-confident) under its own generative model"
        ]
    elif calibration_status == "passed":
        reasons = [
            f"SBC ranks consistent with uniform ({decided_by} for "
            f"{n_sims} sims): inference is self-consistent under its own generative model"
        ]
    elif not power_sufficient:
        reasons = [
            f"SBC is INDETERMINATE / LOW POWER: {decided_by}, but measured power is "
            f"{power_estimate:.2f} against the canonical alternative ({power_alternative}) with "
            f"{expected_count:.2f} expected ranks per bin; promotion requires measured power >= 0.50 "
            f"and at least {min_expected_count_per_bin:.2f} per bin."
        ]
    else:
        reasons = [
            f"SBC is INDETERMINATE / UNCONTROLLED RANDOMNESS: error {sbc_error:.3f} is within threshold "
            f"{threshold:.3f}, but seed=None does not define replayable prior, simulator, fitter, and tie-break streams."
        ]
    return CalibrationVerdict(
        calibration_status=calibration_status,
        pit_error=sbc_error,
        null_threshold=threshold,
        coverage_error=float("nan"),
        reference_level=float("nan"),
        coverage_at_reference=float("nan"),
        mean_interval_width=float("nan"),
        n_points=int(n_sims),
        power_sufficient=power_sufficient,
        expected_count_per_bin=expected_count,
        randomness_controlled=randomness_controlled,
        reasons=reasons,
        kind="calibration-sbc",
        p_value=p_value,
        n_null=0 if error_tol is not None else n_null,
        power_estimate=power_estimate,
        power_alternative=power_alternative,
    )


class CalibrationVerifier:
    """An IC-6-shaped verifier (``.verify(claim, context) -> dict``) wrapping
    :func:`posterior_predictive_calibration`, so a calibration gate drops straight into
    :func:`mixle.task.knowledge_routing.route_task` (or any IC-6 consumer) as a real verifier that
    can *fail* a miscalibrated result -- the thing the routing verifiers used in the experiments only
    ever checked structurally.

    The tool result being verified must carry, on ``claim["payload"]`` (or ``context``), an
    ``"ensemble"`` ``(k, m)`` posterior-predictive draw matrix and a ``"held_out_y"`` ``(k,)`` array.
    A payload without them fails closed with an explicit reason -- a calibration verifier that silently
    passes anything it can't actually check would defeat its own purpose.
    """

    def __init__(
        self,
        *,
        reference_level: float = 0.90,
        null_quantile: float = 0.99,
        pit_tol: float | None = None,
        ensemble_dependence: Literal["shared-draws", "independent"] = "shared-draws",
    ) -> None:
        self.reference_level = reference_level
        self.null_quantile = null_quantile
        self.pit_tol = pit_tol
        # fail-safe default: the documented construction shares posterior draws across points,
        # and promotion power is gated on the declared regime (STAT-RR17-08, pass 18)
        self.ensemble_dependence = ensemble_dependence

    def verify(self, claim: Any, context: Any = None) -> dict[str, Any]:
        payload = (claim or {}).get("payload", {}) if isinstance(claim, dict) else {}
        source = payload if ("ensemble" in payload and "held_out_y" in payload) else (context or {})
        ensemble = source.get("ensemble") if isinstance(source, dict) else None
        held_out_y = source.get("held_out_y") if isinstance(source, dict) else None
        if ensemble is None or held_out_y is None:
            # STAT-RR17-08 (GATE-7): by this module's own taxonomy, "failed" means the observed
            # error exceeded the calibrated decision rule -- evidence of miscalibration. A missing
            # payload supports NEITHER verdict, so it is labeled indeterminate; the decision still
            # fails closed (passed=False), it just never counts as miscalibration evidence.
            return {
                "passed": False,
                "calibration_status": "indeterminate",
                "indeterminate": True,
                "score": 0.0,
                "kind": "calibration",
                "reasons": [
                    "no ensemble/held_out_y to calibrate against -- indeterminate and failing closed "
                    "rather than passing an unchecked posterior (or counting it as miscalibrated)"
                ],
            }
        try:
            verdict = posterior_predictive_calibration(
                np.asarray(ensemble),
                np.asarray(held_out_y),
                reference_level=self.reference_level,
                null_quantile=self.null_quantile,
                pit_tol=self.pit_tol,
                ensemble_dependence=self.ensemble_dependence,
            )
        except (TypeError, ValueError, FloatingPointError) as exc:
            return {
                "passed": False,
                "calibration_status": "indeterminate",
                "indeterminate": True,
                "score": 0.0,
                "kind": "calibration",
                "reasons": [f"calibration input rejected (indeterminate, failing closed): {exc}"],
            }
        return {
            "passed": verdict.passed,
            "calibration_status": verdict.calibration_status,
            "score": verdict.score,
            "kind": verdict.kind,
            "reasons": verdict.reasons,
            "indeterminate": verdict.indeterminate,
            "low_power": verdict.low_power,
            "power_sufficient": verdict.power_sufficient,
            "p_value": verdict.p_value,
            "power_estimate": verdict.power_estimate,
        }
