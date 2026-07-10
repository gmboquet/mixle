"""Data profiling and model recommendation for automatically-typed data.

Profiles sequences of observations to recommend per-field leaf estimators,
measure unconditional pairwise dependency hints, and assemble the composite
estimator via DatumNode. Estimator builders are imported from .factories.
"""

import math
import numbers
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    ParameterEstimator,
)

from .factories import (
    AMBIGUOUS_SCORE_GAP_BITS,
    EMBEDDING_MIN_DIM,
    ID_DISTINCT_FRACTION,
    ID_MIN_COUNT,
    INT_ID_RANGE_MULTIPLIER,
    MAX_INT_CATEGORICAL_DISTINCT,
    MAX_INT_CATEGORICAL_FRACTION,
    POISSON_DISPERSION_MAX,
    POISSON_DISPERSION_MIN,
    VALIDATION_ALPHA,
    VALIDATION_VARIANCE_FLOOR,
    _dense_integer_support,
    _has_torch,
    _integer_range,
    get_categorical_estimator,
    get_composite_estimator,
    get_dict_record_estimator,
    get_gamma_estimator,
    get_gaussian_estimator,
    get_gaussian_mixture_estimator,
    get_hybrid_embedding_estimator,
    get_hybrid_image_estimator,
    get_ignored_estimator,
    get_integer_categorical_estimator,
    get_lognormal_estimator,
    get_multivariate_gaussian_estimator,
    get_optional_estimator,
    get_poisson_estimator,
    get_sequence_estimator,
    get_set_estimator,
    get_student_t_estimator,
)


@dataclass
class MarginalFieldProfile:
    """Marginal evidence for one detected scalar field or structural feature."""

    path: tuple[Any, ...]
    role: str
    count: int
    missing_count: int
    missing_fraction: float
    observed_count: int
    kind: str
    recommendation: str
    bits_per_obs: float | None = None
    entropy_bits: float | None = None
    cardinality: int | None = None
    unique_fraction: float | None = None
    effective_cardinality: float | None = None
    is_constant: bool = False
    top_mass: float | None = None
    numeric_mean: float | None = None
    numeric_var: float | None = None
    integer_min: int | None = None
    integer_max: int | None = None
    integer_density: float | None = None
    model_scores_bits: dict[str, float] = field(default_factory=dict)
    model_score_gap_bits: float | None = None
    validation_scores_bits: dict[str, float] = field(default_factory=dict)
    validation_recommendation: str | None = None
    validation_score_gap_bits: float | None = None
    validation_count: int = 0
    validation_notes: list[str] = field(default_factory=list)
    gof_ks: float | None = None
    gof_pvalue: float | None = None
    notes: list[str] = field(default_factory=list)

    def robust_recommendation(self) -> str:
        """The model choice to actually trust, combining the in-sample (BIC) pick with the held-out
        validation check rather than leaving their disagreement as a note for a human to notice.

        Held-out generalization is stronger evidence than an in-sample penalized-likelihood score:
        a flexible family (a 3-parameter shape family, a 2-component mixture) can win BIC by fitting
        noise/outliers in the training data that do not repeat in held-out data, which is exactly the
        failure mode BIC's asymptotic parameter-count penalty does not always catch at small-to-moderate
        n. So: agree with ``recommendation`` whenever there is no held-out evidence, or the two already
        agree; defer to ``validation_recommendation`` only when it disagrees by a DECISIVE margin (the
        same ``AMBIGUOUS_SCORE_GAP_BITS`` threshold already used to flag a close call elsewhere in this
        module) -- a marginal, ambiguous validation preference is not enough evidence to overturn BIC.
        """
        if self.validation_recommendation is None or self.validation_recommendation == self.recommendation:
            return self.recommendation
        if self.validation_score_gap_bits is None or self.validation_score_gap_bits < AMBIGUOUS_SCORE_GAP_BITS:
            return self.recommendation
        return self.validation_recommendation

    def model_weights(self) -> dict[str, float]:
        """Return Schwarz (BIC) model weights over the scored candidates, summing to 1.

        From per-observation code lengths ``L_i`` (``model_scores_bits``) and the observed count
        ``n``, ``BIC_i = 2 * n * ln2 * L_i`` so the Schwarz weight is
        ``w_i proportional to exp(-0.5 * (BIC_i - min BIC)) = exp(-n * ln2 * (L_i - min L))`` -- an
        approximate posterior probability over the candidate models. Empty when nothing was scored.
        """
        return _bic_weights(self.model_scores_bits, self.observed_count)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the marginal field evidence."""
        return {
            "path": format_path(self.path),
            "role": self.role,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_fraction": self.missing_fraction,
            "observed_count": self.observed_count,
            "kind": self.kind,
            "recommendation": self.recommendation,
            "robust_recommendation": self.robust_recommendation(),
            "bits_per_obs": self.bits_per_obs,
            "entropy_bits": self.entropy_bits,
            "cardinality": self.cardinality,
            "unique_fraction": self.unique_fraction,
            "effective_cardinality": self.effective_cardinality,
            "is_constant": self.is_constant,
            "top_mass": self.top_mass,
            "numeric_mean": self.numeric_mean,
            "numeric_var": self.numeric_var,
            "integer_min": self.integer_min,
            "integer_max": self.integer_max,
            "integer_density": self.integer_density,
            "model_scores_bits": dict(self.model_scores_bits),
            "model_weights": self.model_weights(),
            "model_score_gap_bits": self.model_score_gap_bits,
            "gof_ks": self.gof_ks,
            "gof_pvalue": self.gof_pvalue,
            "validation_scores_bits": dict(self.validation_scores_bits),
            "validation_recommendation": self.validation_recommendation,
            "validation_score_gap_bits": self.validation_score_gap_bits,
            "validation_count": self.validation_count,
            "validation_notes": list(self.validation_notes),
            "notes": list(self.notes),
        }


@dataclass
class PairwiseDependencyHint:
    """Unconditional pairwise dependency hint measured from encoded values."""

    left: tuple[Any, ...]
    right: tuple[Any, ...]
    mi_bits: float
    adjusted_mi_bits: float
    bic_gain_bits: float
    normalized_mi: float
    left_entropy_bits: float
    right_entropy_bits: float
    joint_count: int
    method: str
    p_value: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the dependency hint."""
        return {
            "left": format_path(self.left),
            "right": format_path(self.right),
            "mi_bits": self.mi_bits,
            "adjusted_mi_bits": self.adjusted_mi_bits,
            "bic_gain_bits": self.bic_gain_bits,
            "normalized_mi": self.normalized_mi,
            "left_entropy_bits": self.left_entropy_bits,
            "right_entropy_bits": self.right_entropy_bits,
            "joint_count": self.joint_count,
            "method": self.method,
            "p_value": self.p_value,
            "notes": list(self.notes),
        }


@dataclass
class StructureProfile:
    """Structure-analysis result returned by ``analyze_structure``."""

    estimator: ParameterEstimator
    fields: list[MarginalFieldProfile]
    pairwise_hints: list[PairwiseDependencyHint]
    warnings: list[str]
    sampled_rows: int
    total_rows: int
    dependency_tree_edges: list[PairwiseDependencyHint] = field(default_factory=list)
    dependency_residual_edges: list[PairwiseDependencyHint] = field(default_factory=list)
    dependency_redundancy_ratio: float = 0.0
    encoded_pairwise_fields: int = 0
    pairwise_fields_available: int = 0
    pairwise_pairs_available: int = 0
    pairwise_pairs_checked: int = 0
    pairwise_pair_strategy: str = "none"

    def recommend(self) -> ParameterEstimator:
        """Return the estimator selected by the structure-analysis pass."""
        return self.estimator

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the structure profile."""
        return {
            "total_rows": self.total_rows,
            "sampled_rows": self.sampled_rows,
            "estimator": type(self.estimator).__name__,
            "fields": [u.summary() for u in self.fields],
            "pairwise_hints": [u.summary() for u in self.pairwise_hints],
            "dependency_tree_edges": [u.summary() for u in self.dependency_tree_edges],
            "dependency_residual_edges": [u.summary() for u in self.dependency_residual_edges],
            "dependency_redundancy_ratio": self.dependency_redundancy_ratio,
            "encoded_pairwise_fields": self.encoded_pairwise_fields,
            "pairwise_fields_available": self.pairwise_fields_available,
            "pairwise_pairs_available": self.pairwise_pairs_available,
            "pairwise_pairs_checked": self.pairwise_pairs_checked,
            "pairwise_pair_strategy": self.pairwise_pair_strategy,
            "warnings": list(self.warnings),
        }

    def explain(self) -> list[str]:
        """Render human-readable explanation lines for field and dependency choices."""
        lines = []
        for field_profile in self.fields:
            bits = "" if field_profile.bits_per_obs is None else " (~%.3f bits/obs)" % field_profile.bits_per_obs
            robust = field_profile.robust_recommendation()
            overridden = (
                " (validation-overridden from %s)" % field_profile.recommendation
                if robust != field_profile.recommendation
                else ""
            )
            lines.append(
                "%s: %s -> %s%s%s" % (format_path(field_profile.path), field_profile.kind, robust, overridden, bits)
            )
            for note in field_profile.notes:
                lines.append("  - %s" % note)
        for hint in self.pairwise_hints:
            p_value = "" if hint.p_value is None else ", p=%.3f" % hint.p_value
            lines.append(
                "%s <-> %s: %.3f bits MI, %.3f adjusted, %.3f BIC gain/obs%s"
                % (
                    format_path(hint.left),
                    format_path(hint.right),
                    hint.mi_bits,
                    hint.adjusted_mi_bits,
                    hint.bic_gain_bits,
                    p_value,
                )
            )
        for field_profile in self.fields:
            if field_profile.validation_recommendation is not None:
                gap = (
                    ""
                    if field_profile.validation_score_gap_bits is None
                    else ", gap %.3f bits/obs" % field_profile.validation_score_gap_bits
                )
                lines.append(
                    "%s validation: %s over %d rows%s"
                    % (
                        format_path(field_profile.path),
                        field_profile.validation_recommendation,
                        field_profile.validation_count,
                        gap,
                    )
                )
                for note in field_profile.validation_notes:
                    lines.append("  - %s" % note)
        for edge in self.dependency_tree_edges:
            lines.append(
                "tree edge %s <-> %s: %.3f BIC gain/obs"
                % (format_path(edge.left), format_path(edge.right), edge.bic_gain_bits)
            )
        if self.dependency_residual_edges:
            lines.append(
                "dependency residuals: %d non-tree accepted edges (ratio %.3f)"
                % (len(self.dependency_residual_edges), self.dependency_redundancy_ratio)
            )
        for warning in self.warnings:
            lines.append("warning: %s" % warning)
        return lines


def format_path(path: tuple[Any, ...]) -> str:
    """Format a tuple path as a compact JSONPath-like field reference."""
    if len(path) == 0:
        return "$"
    rv = "$"
    for part in path:
        if isinstance(part, int):
            rv += "[%d]" % part
        else:
            rv += "[%r]" % part
    return rv


def _path_sort_key(path: tuple[Any, ...]) -> tuple[tuple[int, Any], ...]:
    return tuple((0, part) if isinstance(part, int) else (1, repr(part)) for part in path)


def _is_missing_value(x: Any) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and math.isnan(float(x)))


def _is_sequence_like(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes, dict, set, frozenset))


def _entropy_from_counts(counts: Sequence[float]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    rv = 0.0
    for count in counts:
        if count > 0:
            p = float(count) / total
            rv -= p * math.log(p, 2.0)
    return rv


def _gaussian_bits(var: float) -> float | None:
    if var <= 0.0 or not math.isfinite(var):
        return None
    return max(0.0, 0.5 * math.log(2.0 * math.pi * math.e * var, 2.0))


def _bic_penalty_bits(num_params: int, nobs: int) -> float:
    if num_params <= 0 or nobs <= 1:
        return 0.0
    return 0.5 * float(num_params) * math.log(float(nobs), 2.0) / float(nobs)


def _categorical_bic_bits(vdict: dict[Any, float], num_levels: int | None = None) -> float | None:
    n = int(sum(vdict.values()))
    if n <= 0:
        return None
    k = len(vdict) if num_levels is None else int(num_levels)
    return _entropy_from_counts(vdict.values()) + _bic_penalty_bits(max(0, k - 1), n)


def _poisson_bits(vdict: dict[int, float], mean: float) -> float | None:
    if mean <= 0.0:
        return None
    total = float(sum(vdict.values()))
    if total <= 0.0:
        return None
    ll = 0.0
    for k, v in vdict.items():
        kk = int(k)
        ll += float(v) * (kk * math.log(mean) - mean - math.lgamma(kk + 1.0))
    return -ll / (total * math.log(2.0))


def _poisson_bic_bits(vdict: dict[int, float], mean: float) -> float | None:
    bits = _poisson_bits(vdict, mean)
    if bits is None:
        return None
    return bits + _bic_penalty_bits(1, int(sum(vdict.values())))


def _normal_cdf(x: float, mean: float, sd: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def _integer_gaussian_bic_bits(vdict: dict[int, float], mean: float, var: float) -> float | None:
    if var <= 0.0 or not math.isfinite(var):
        return None
    sd = math.sqrt(var)
    total = float(sum(vdict.values()))
    if total <= 0.0:
        return None
    ll = 0.0
    for k, v in vdict.items():
        kk = int(k)
        p = _normal_cdf(kk + 0.5, mean, sd) - _normal_cdf(kk - 0.5, mean, sd)
        ll += float(v) * math.log(max(p, 1.0e-300))
    return -ll / (total * math.log(2.0)) + _bic_penalty_bits(2, int(total))


def _gaussian_bic_bits(var: float, nobs: int) -> float | None:
    bits = _gaussian_bits(var)
    if bits is None:
        return None
    return bits + _bic_penalty_bits(2, nobs)


def _lognormal_bits(values: Sequence[float]) -> float | None:
    """Return the per-observation code length (bits) of an MLE log-normal fit, or None.

    For ``X`` log-normal, ``ln X ~ N(mu, sigma^2)`` and the differential entropy is
    ``E[ln X] + 0.5*ln(2*pi*e*sigma^2)``; the ``E[ln X]`` term is the change-of-variables Jacobian.
    Defined only for strictly positive data.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return None
    if arr.var() <= 0.0:
        # Raw-space variance is exactly zero (numerically constant data) -- unlike the Gaussian/
        # Student-t/Gamma candidates, which all reject this directly, log-space variance of a
        # constant array is NOT exactly zero: np.log(c) computed independently per repeated element
        # rounds slightly differently each time, leaving a ~1e-32-scale floating-point artifact that
        # _gaussian_bits would otherwise treat as a real (absurdly tight, winning) signal. Reject here
        # too, on the same raw-space test the other candidates already use.
        return None
    logs = np.log(arr)
    gauss = _gaussian_bits(float(logs.var()))
    if gauss is None:
        return None
    return float(logs.mean()) / math.log(2.0) + gauss


def _lognormal_bic_bits(values: Sequence[float], nobs: int) -> float | None:
    bits = _lognormal_bits(values)
    if bits is None:
        return None
    return bits + _bic_penalty_bits(2, nobs)


def _gamma_moments(arr: np.ndarray) -> tuple[float, float] | None:
    """Return the method-of-moments ``(shape k, scale theta)`` for positive data, or None."""
    if arr.size == 0 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return None
    mean = float(arr.mean())
    var = float(arr.var())
    if mean <= 0.0 or var <= 0.0:
        return None
    theta = var / mean
    return mean / theta, theta


def _gamma_nll_bits(arr: np.ndarray, k: float, theta: float) -> float:
    """Per-observation Gamma code length (bits) of data ``arr`` under ``Gamma(k, theta)``."""
    logs = np.log(arr)
    nats = -((k - 1.0) * float(logs.mean()) - float(arr.mean()) / theta - math.lgamma(k) - k * math.log(theta))
    return nats / math.log(2.0)


def _gamma_bic_bits(values: Sequence[float], nobs: int) -> float | None:
    arr = np.asarray(values, dtype=float)
    moments = _gamma_moments(arr)
    if moments is None:
        return None
    k, theta = moments
    return _gamma_nll_bits(arr, k, theta) + _bic_penalty_bits(2, nobs)


def _student_t_params(arr: np.ndarray) -> tuple[float, float, float] | None:
    """Return Student-t ``(df, loc, scale)`` from moments, or None when not heavier-tailed.

    df is set from the excess kurtosis (``excess = 6/(df-4)`` for a t, so ``df = 4 + 6/excess``);
    light-tailed data (non-positive excess) returns None so the candidate is skipped.
    """
    if arr.size < 4 or not np.all(np.isfinite(arr)):
        return None
    mean = float(arr.mean())
    var = float(arr.var())
    if var <= 0.0:
        return None
    excess = float(np.mean((arr - mean) ** 4)) / (var * var) - 3.0
    if excess <= 0.0:
        return None
    df = min(max(4.0 + 6.0 / excess, 2.5), 100.0)
    scale = math.sqrt(var * (df - 2.0) / df) if df > 2.0 else math.sqrt(var)
    return df, mean, scale


def _student_t_bic_bits(values: Sequence[float], nobs: int) -> float | None:
    from scipy import stats

    arr = np.asarray(values, dtype=float)
    params = _student_t_params(arr)
    if params is None:
        return None
    df, loc, scale = params
    nll = -float(np.mean(stats.t.logpdf(arr, df, loc=loc, scale=scale)))
    return nll / math.log(2.0) + _bic_penalty_bits(3, nobs)


def _validation_student_t_bits(train: Sequence[float], validation: Sequence[float]) -> float | None:
    from scipy import stats

    if not train or not validation:
        return None
    params = _student_t_params(np.asarray(train, dtype=float))
    if params is None:
        return None
    df, loc, scale = params
    val = np.asarray(validation, dtype=float)
    if not np.all(np.isfinite(val)):
        return None
    ll = float(np.sum(stats.t.logpdf(val, df, loc=loc, scale=scale)))
    return -ll / (float(val.size) * math.log(2.0))


_MIXTURE_EM_CAP = 5000


def _mixture_log_likelihood(x: np.ndarray, w: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    logp = np.log(w)[None, :] - 0.5 * np.log(2.0 * np.pi * var)[None, :] - 0.5 * (x[:, None] - mu) ** 2 / var
    m = logp.max(axis=1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))).sum())


def _fit_gaussian_mixture2(arr: np.ndarray, iters: int = 200, tol: float = 1.0e-6):
    """Fit a 2-component 1-D Gaussian mixture by EM. Returns (total_loglik, (w, mu, var)) or None.

    Large inputs are strided down to ``_MIXTURE_EM_CAP`` points for the fit (BIC is approximate);
    the returned log-likelihood is rescaled to the full sample size.
    """
    if arr.size < 6 or not np.all(np.isfinite(arr)):
        return None
    full_n = arr.size
    if full_n > _MIXTURE_EM_CAP:
        arr = arr[:: int(np.ceil(full_n / _MIXTURE_EM_CAP))]
    n = arr.size
    var0 = float(arr.var())
    if var0 <= 0.0:
        return None
    floor = 1.0e-6 * var0 + 1.0e-300
    srt = np.sort(arr)
    half = max(1, n // 2)
    lo, hi = srt[:half], srt[half:]
    mu = np.array([float(lo.mean()), float(hi.mean()) if hi.size else float(arr.max())])
    if mu[0] == mu[1]:
        mu = np.array([float(arr.min()), float(arr.max())])
    var = np.array([max(float(lo.var()), floor), max(float(hi.var()) if hi.size else var0, floor)])
    w = np.array([0.5, 0.5])
    x = arr
    prev = -np.inf
    for _ in range(int(iters)):
        logp = np.log(w)[None, :] - 0.5 * np.log(2.0 * np.pi * var)[None, :] - 0.5 * (x[:, None] - mu) ** 2 / var
        m = logp.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
        ll = float(lse.sum())
        resp = np.exp(logp - lse[:, None])
        nk = resp.sum(axis=0) + 1.0e-300
        w = nk / n
        mu = (resp * x[:, None]).sum(axis=0) / nk
        var = np.maximum((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk, floor)
        if ll - prev < tol * (abs(prev) + 1.0):
            break
        prev = ll
    ll_per_obs = _mixture_log_likelihood(x, w, mu, var) / n
    return ll_per_obs * full_n, (w, mu, var)


def _mixture_bic_bits(values: Sequence[float], nobs: int) -> float | None:
    arr = np.asarray(values, dtype=float)
    fit = _fit_gaussian_mixture2(arr)
    if fit is None:
        return None
    total_ll, _ = fit
    # 2-component 1-D Gaussian mixture: 2 means + 2 vars + 1 weight = 5 free parameters.
    return -total_ll / (arr.size * math.log(2.0)) + _bic_penalty_bits(5, nobs)


def _validation_mixture_bits(train: Sequence[float], validation: Sequence[float]) -> float | None:
    tarr = np.asarray(train, dtype=float)
    val = np.asarray(validation, dtype=float)
    if val.size == 0 or not np.all(np.isfinite(val)):
        return None
    fit = _fit_gaussian_mixture2(tarr)
    if fit is None:
        return None
    _, (w, mu, var) = fit
    return -_mixture_log_likelihood(val, w, mu, var) / (val.size * math.log(2.0))


def _looks_multimodal(arr: np.ndarray) -> bool:
    """Heuristic multimodality gate via Sarle's bimodality coefficient (sample-corrected).

    BC = (skew^2 + 1) / (excess_kurtosis + 3(n-1)^2/((n-2)(n-3))). The uniform reference is ~0.555;
    clearly multimodal (platykurtic) data approaches 1, while a unimodal Gaussian sits near 0.33. A
    threshold above the uniform value keeps the (overfit-prone) 2-component mixture out of the
    candidate set for unimodal data.
    """
    n = arr.size
    if n < 8:
        return False
    sd = float(arr.std())
    if sd <= 0.0:
        return False
    z = (arr - float(arr.mean())) / sd
    skew = float(np.mean(z**3))
    excess_kurtosis = float(np.mean(z**4)) - 3.0
    denom = excess_kurtosis + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0.0:
        return False
    return (skew * skew + 1.0) / denom > 0.6


def _numeric_candidate_bics(arr: np.ndarray, nobs: int) -> dict[str, float]:
    """Return per-candidate BIC code lengths for numeric data (support-typed).

    Gaussian and Student-t apply to any real data; a 2-component Gaussian mixture is added only when
    the data looks multimodal (it overfits unimodal samples otherwise); log-normal and gamma are
    added only for strictly-positive support. Used by both the marginal profiler and
    ``get_estimator`` so the candidate set and selection stay consistent.
    """
    candidates: dict[str, float | None] = {
        "gaussian": _gaussian_bic_bits(float(arr.var()), nobs),
        "student_t": _student_t_bic_bits(arr, nobs),
    }
    # The 2-component mixture is added only for plausibly-multimodal data with enough distinct
    # values that its components cannot collapse onto a few points (which would overfit wildly).
    if arr.size and np.unique(arr).size >= 12 and _looks_multimodal(arr):
        candidates["mixture"] = _mixture_bic_bits(arr, nobs)
    if arr.size and np.all(arr > 0.0):
        candidates["lognormal"] = _lognormal_bic_bits(arr, nobs)
        candidates["gamma"] = _gamma_bic_bits(arr, nobs)
    # registered continuous detectors (additive -- a richer family only wins if its BIC beats the builtins).
    # A flexible (>=3-parameter) shape family needs enough distinct values to be estimated honestly;
    # without this, it overfits a handful of repeated points and steals the recommendation from a simple
    # family (the same reason the 2-component mixture is gated on >=12 distinct values above).
    if arr.size:
        from mixle.utils.automatic.detectors import continuous_detectors

        n_distinct = int(np.unique(arr).size)
        for d in continuous_detectors():
            if d.name in candidates or not d.applies(arr):
                continue
            # a non-builtin family must see real data variety to justify winning over a simple builtin;
            # otherwise it overfits a few repeated points (a 2-param tail family on 2-3 distinct values, a
            # 3-param shape family on a dozen). The builtins stay ungated as the safe defaults.
            if n_distinct < (15 if d.n_params >= 3 else 10):
                continue
            candidates[d.name] = d.score(arr, nobs)
    return _clean_scores(candidates)


def _value_array_from_vdict(vdict: dict[Any, float], cap: int = 200000) -> np.ndarray:
    """Expand a value->count map into a representative numeric value array (capped for memory)."""
    keys: list[float] = []
    counts: list[int] = []
    for k, v in vdict.items():
        if isinstance(k, (int, float, np.integer, np.floating)) and math.isfinite(float(k)):
            c = int(round(float(v)))
            if c > 0:
                keys.append(float(k))
                counts.append(c)
    if not keys:
        return np.empty(0, dtype=float)
    total = sum(counts)
    if total > cap:
        scale = cap / total
        counts = [max(1, int(round(c * scale))) for c in counts]
    return np.repeat(np.asarray(keys, dtype=float), counts)


def _clean_scores(scores: dict[str, float | None]) -> dict[str, float]:
    return {k: float(v) for k, v in scores.items() if v is not None and math.isfinite(float(v))}


_NUMERIC_MODEL_MARGIN_BITS = 0.02


def _numeric_model_recommendation(scores: dict[str, float], margin: float = _NUMERIC_MODEL_MARGIN_BITS) -> str:
    """Pick the numeric model, defaulting to Gaussian unless an alternative beats it by ``margin``.

    Gamma/log-normal converge to the Gaussian for near-symmetric data, so a bare argmin would flip
    the default on a numerical tie. Only switch when the best alternative's code length is lower by
    at least ``margin`` bits/obs; otherwise keep the Gaussian.
    """
    if not scores:
        return "gaussian"
    if "gaussian" not in scores:
        return min(scores, key=lambda k: (scores[k], k))
    alternatives = {k: v for k, v in scores.items() if k != "gaussian"}
    if not alternatives:
        return "gaussian"
    best = min(alternatives, key=lambda k: (alternatives[k], k))
    return best if alternatives[best] < scores["gaussian"] - margin else "gaussian"


def _bic_weights(scores: dict[str, float], nobs: int) -> dict[str, float]:
    """Return normalized Schwarz (BIC) weights from per-observation code lengths.

    ``BIC_i = 2 * n * ln2 * L_i`` for code length ``L_i`` (bits/obs), so the Schwarz weight is
    ``w_i proportional to exp(-0.5 * (BIC_i - min BIC)) = exp(-n * ln2 * (L_i - min L))`` (all
    exponents <= 0, so no overflow). Returns ``{}`` for empty input.
    """
    if not scores:
        return {}
    n = max(int(nobs), 1)
    best = min(scores.values())
    raw = {k: math.exp(-n * math.log(2.0) * (v - best)) for k, v in scores.items()}
    total = sum(raw.values()) or 1.0
    return {k: w / total for k, w in raw.items()}


GOF_ABSTAIN_PVALUE = 0.01


def _numeric_pit(arr: np.ndarray, recommendation: str, gmean: float, gvar: float):
    """Return the probability-integral-transform F(x) of the data under the recommended fit, or None."""
    from scipy import stats

    if recommendation == "gaussian" and gvar > 0.0:
        return stats.norm.cdf(arr, loc=gmean, scale=math.sqrt(gvar))
    if recommendation == "lognormal":
        logs = np.log(arr)
        lvar = float(logs.var())
        if lvar > 0.0:
            return stats.norm.cdf(logs, loc=float(logs.mean()), scale=math.sqrt(lvar))
        return None
    if recommendation == "gamma":
        moments = _gamma_moments(arr)
        if moments is not None:
            k, theta = moments
            return stats.gamma.cdf(arr, a=k, scale=theta)
    from mixle.utils.automatic.detectors import get_detector

    detector = get_detector(recommendation)
    if detector is not None and detector.cdf is not None:
        return detector.cdf(arr)
    return None


def _pit_goodness_of_fit(pit: np.ndarray) -> tuple[float, float] | None:
    """Return (KS statistic, p-value) of the PIT against Uniform(0,1); None if degenerate.

    Under a correctly-specified model the PIT values are Uniform(0,1); a small p-value indicates
    miscalibration (the chosen family does not fit), which the profile surfaces as an abstain note.
    """
    from scipy import stats

    pit = np.asarray(pit, dtype=float)
    if pit.size < 2 or not np.all(np.isfinite(pit)):
        return None
    result = stats.kstest(np.clip(pit, 0.0, 1.0), "uniform")
    return float(result.statistic), float(result.pvalue)


def _score_gap_bits(scores: dict[str, float], recommendation: str) -> float | None:
    if recommendation not in scores or len(scores) <= 1:
        return None
    chosen = scores[recommendation]
    alternatives = [v for k, v in scores.items() if k != recommendation]
    if not alternatives:
        return None
    return min(alternatives) - chosen


def _recommended_integer_model(vdict: dict[Any, float]) -> tuple[str, dict[str, float]]:
    n = int(sum(vdict.values()))
    min_val, max_val, width = _integer_range(vdict)
    distinct = len(vdict)
    mean = sum(int(k) * v for k, v in vdict.items()) / float(max(1, n))
    var = max(0.0, sum(((int(k) - mean) ** 2) * v for k, v in vdict.items()) / float(max(1, n)))

    if (
        n >= ID_MIN_COUNT
        and distinct >= ID_DISTINCT_FRACTION * n
        and width >= INT_ID_RANGE_MULTIPLIER * max(1, distinct)
    ):
        return "ignored", {}

    scores: dict[str, float | None] = {}
    dense = _dense_integer_support(vdict)
    if dense:
        scores["integer_categorical"] = _categorical_bic_bits(vdict, num_levels=width)
    elif distinct <= max(MAX_INT_CATEGORICAL_DISTINCT, MAX_INT_CATEGORICAL_FRACTION * n):
        scores["categorical"] = _categorical_bic_bits(vdict, num_levels=distinct)
    if min_val >= 0:
        scores["poisson"] = _poisson_bic_bits(vdict, mean)
    scores["gaussian"] = _integer_gaussian_bic_bits(vdict, mean, var)

    # registered discrete detectors (additive -- a count family only wins if its BIC beats Poisson/Gaussian;
    # it must also see real variety, so it cannot overfit a couple of repeated integers).
    if distinct >= 5:
        from mixle.utils.automatic.detectors import discrete_detectors

        arr = _value_array_from_vdict(vdict)
        if arr.size:
            for d in discrete_detectors():
                if d.name not in scores and d.applies(arr):
                    scores[d.name] = d.score(arr, n)

    clean = _clean_scores(scores)
    if not clean:
        return "ignored", {}
    return min(clean.items(), key=lambda u: (u[1], u[0]))[0], clean


def _validation_split(
    values: Sequence[Any], validation_fraction: float, max_validation_rows: int, min_validation_count: int, seed: int
) -> tuple[list[Any], list[Any]] | None:
    if validation_fraction <= 0.0 or max_validation_rows <= 0:
        return None
    n = len(values)
    if n < min_validation_count or n < 2:
        return None
    validation_count = int(round(validation_fraction * n))
    validation_count = max(1, min(validation_count, max_validation_rows, n - 1))
    rng = np.random.RandomState(seed)
    validation_idx = set(int(i) for i in rng.choice(n, size=validation_count, replace=False))
    train = []
    validation = []
    for i, value in enumerate(values):
        if i in validation_idx:
            validation.append(value)
        else:
            train.append(value)
    if not train or not validation:
        return None
    return train, validation


def _validation_categorical_bits(
    train: Sequence[Any],
    validation: Sequence[Any],
    support: Sequence[Any] | None = None,
    alpha: float = VALIDATION_ALPHA,
) -> float | None:
    if not train or not validation or alpha <= 0.0:
        return None
    counts = defaultdict(float)
    for value in train:
        counts[value] += 1.0
    levels = set(counts.keys()) if support is None else set(support)
    if not levels:
        return None
    unknown_bucket = 1
    denom = float(len(train)) + alpha * float(len(levels) + unknown_bucket)
    total_bits = 0.0
    for value in validation:
        count = counts.get(value, 0.0) if value in levels else 0.0
        total_bits -= math.log((count + alpha) / denom, 2.0)
    return total_bits / float(len(validation))


def _validation_integer_categorical_bits(
    train: Sequence[int], validation: Sequence[int], min_val: int, max_val: int, alpha: float = VALIDATION_ALPHA
) -> float | None:
    if not train or not validation or alpha <= 0.0 or max_val < min_val:
        return None
    counts = defaultdict(float)
    for value in train:
        counts[int(value)] += 1.0
    width = max_val - min_val + 1
    unknown_bucket = 1
    denom = float(len(train)) + alpha * float(width + unknown_bucket)
    total_bits = 0.0
    for value in validation:
        k = int(value)
        count = counts.get(k, 0.0) if min_val <= k <= max_val else 0.0
        total_bits -= math.log((count + alpha) / denom, 2.0)
    return total_bits / float(len(validation))


def _validation_poisson_bits(train: Sequence[int], validation: Sequence[int]) -> float | None:
    if not train or not validation:
        return None
    if any(int(value) < 0 for value in train) or any(int(value) < 0 for value in validation):
        return None
    mean = sum(int(value) for value in train) / float(len(train))
    if mean <= 0.0:
        return None
    ll = 0.0
    for value in validation:
        k = int(value)
        ll += k * math.log(mean) - mean - math.lgamma(k + 1.0)
    return -ll / (float(len(validation)) * math.log(2.0))


def _validation_gaussian_bits(train: Sequence[float], validation: Sequence[float]) -> float | None:
    if not train or not validation:
        return None
    arr = np.asarray(train, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    mean = float(arr.mean())
    var = max(float(arr.var()), VALIDATION_VARIANCE_FLOOR)
    log_norm = 0.5 * math.log(2.0 * math.pi * var)
    ll = 0.0
    for value in validation:
        xx = float(value)
        if not math.isfinite(xx):
            return None
        ll -= log_norm + ((xx - mean) ** 2) / (2.0 * var)
    return -ll / (float(len(validation)) * math.log(2.0))


def _validation_lognormal_bits(train: Sequence[float], validation: Sequence[float]) -> float | None:
    """Held-out predictive code length (bits/obs) of a log-normal fit; positive data only."""
    if not train or not validation:
        return None
    tarr = np.asarray(train, dtype=float)
    if not np.all(np.isfinite(tarr)) or not np.all(tarr > 0.0):
        return None
    logs = np.log(tarr)
    mean = float(logs.mean())
    var = max(float(logs.var()), VALIDATION_VARIANCE_FLOOR)
    log_norm = 0.5 * math.log(2.0 * math.pi * var)
    ll = 0.0
    for value in validation:
        xx = float(value)
        if not math.isfinite(xx) or xx <= 0.0:
            return None
        lx = math.log(xx)
        # log p(x) = -ln x - log_norm - (ln x - mu)^2 / (2 var)  (the -ln x is the Jacobian)
        ll += -lx - log_norm - ((lx - mean) ** 2) / (2.0 * var)
    return -ll / (float(len(validation)) * math.log(2.0))


def _validation_gamma_bits(train: Sequence[float], validation: Sequence[float]) -> float | None:
    """Held-out predictive code length (bits/obs) of a Gamma fit; positive data only."""
    if not train or not validation:
        return None
    moments = _gamma_moments(np.asarray(train, dtype=float))
    if moments is None:
        return None
    k, theta = moments
    const = math.lgamma(k) + k * math.log(theta)
    ll = 0.0
    for value in validation:
        xx = float(value)
        if not math.isfinite(xx) or xx <= 0.0:
            return None
        ll += (k - 1.0) * math.log(xx) - xx / theta - const
    return -ll / (float(len(validation)) * math.log(2.0))


def _validation_integer_gaussian_bits(train: Sequence[int], validation: Sequence[int]) -> float | None:
    if not train or not validation:
        return None
    arr = np.asarray([int(value) for value in train], dtype=float)
    mean = float(arr.mean())
    sd = math.sqrt(max(float(arr.var()), VALIDATION_VARIANCE_FLOOR))
    ll = 0.0
    for value in validation:
        k = int(value)
        p = _normal_cdf(k + 0.5, mean, sd) - _normal_cdf(k - 0.5, mean, sd)
        ll += math.log(max(p, 1.0e-300))
    return -ll / (float(len(validation)) * math.log(2.0))


def _validate_marginal_profile(
    profile: MarginalFieldProfile,
    values: Sequence[Any],
    validation_fraction: float,
    max_validation_rows: int,
    min_validation_count: int,
    seed: int,
) -> MarginalFieldProfile:
    observed = [value for value in values if not _is_missing_value(value)]
    split = _validation_split(observed, validation_fraction, max_validation_rows, min_validation_count, seed)
    if split is None:
        return profile
    if profile.recommendation == "ignored":
        profile.validation_notes.append("predictive validation skipped for ignored field")
        return profile

    train, validation = split
    scores: dict[str, float | None] = {}

    if profile.kind in ("string", "boolean"):
        scores["categorical"] = _validation_categorical_bits(train, validation)

    elif profile.kind == "integer":
        candidates = set(profile.model_scores_bits.keys())
        if "categorical" in candidates:
            scores["categorical"] = _validation_categorical_bits(train, validation)
        if "integer_categorical" in candidates:
            train_int = [int(value) for value in train]
            if train_int:
                lo = min(train_int)
                hi = max(train_int)
                scores["integer_categorical"] = _validation_integer_categorical_bits(
                    [int(value) for value in train], [int(value) for value in validation], lo, hi
                )
        if "poisson" in candidates:
            scores["poisson"] = _validation_poisson_bits(
                [int(value) for value in train], [int(value) for value in validation]
            )
        if "gaussian" in candidates:
            scores["gaussian"] = _validation_integer_gaussian_bits(
                [int(value) for value in train], [int(value) for value in validation]
            )

    elif profile.kind == "numeric":
        train_f = [float(value) for value in train]
        val_f = [float(value) for value in validation]
        scores["gaussian"] = _validation_gaussian_bits(train_f, val_f)
        if "student_t" in profile.model_scores_bits:
            scores["student_t"] = _validation_student_t_bits(train_f, val_f)
        if "mixture" in profile.model_scores_bits:
            scores["mixture"] = _validation_mixture_bits(train_f, val_f)
        if "lognormal" in profile.model_scores_bits:
            scores["lognormal"] = _validation_lognormal_bits(train_f, val_f)
        if "gamma" in profile.model_scores_bits:
            scores["gamma"] = _validation_gamma_bits(train_f, val_f)

    clean = _clean_scores(scores)
    if not clean:
        return profile
    recommendation = min(clean.items(), key=lambda u: (u[1], u[0]))[0]
    profile.validation_scores_bits = clean
    profile.validation_recommendation = recommendation
    profile.validation_score_gap_bits = _score_gap_bits(clean, recommendation)
    profile.validation_count = len(validation)
    if profile.validation_score_gap_bits is not None and profile.validation_score_gap_bits < AMBIGUOUS_SCORE_GAP_BITS:
        profile.validation_notes.append(
            "top validation models are close: %.3f bits/obs gap" % profile.validation_score_gap_bits
        )
    if recommendation != profile.recommendation:
        profile.validation_notes.append(
            "validation prefers %s over marginal recommendation %s" % (recommendation, profile.recommendation)
        )
        if profile.robust_recommendation() == recommendation:
            profile.validation_notes.append(
                "gap is decisive -- robust_recommendation() overrides to %s" % recommendation
            )
    return profile


def _extract_field_series(
    data: Sequence[Any], path: tuple[Any, ...] = (), role: str = "field"
) -> dict[tuple[Any, ...], tuple[str, list[Any]]]:
    if len(data) == 0:
        return {path: (role, [])}

    observed = [u for u in data if u is not None]
    if not observed:
        return {path: (role, list(data))}

    if all(isinstance(u, tuple) for u in observed):
        max_len = max(len(u) for u in observed)
        rv: dict[tuple[Any, ...], tuple[str, list[Any]]] = {}
        for i in range(max_len):
            child_values = [u[i] if isinstance(u, tuple) and i < len(u) else None for u in data]
            rv.update(_extract_field_series(child_values, path + (i,), role="field"))
        return rv

    if all(_is_sequence_like(u) for u in observed):
        lengths = [len(u) if _is_sequence_like(u) else None for u in data]
        fixed = len({v for v in lengths if v is not None}) == 1
        if fixed:
            dim = next(v for v in lengths if v is not None)
            rv: dict[tuple[Any, ...], tuple[str, list[Any]]] = {}
            for i in range(dim):
                child_values = [list(u)[i] if _is_sequence_like(u) and len(u) > i else None for u in data]
                rv.update(_extract_field_series(child_values, path + (i,), role="field"))
            return rv
        elems = []
        for u in observed:
            elems.extend(list(u))
        rv = {path + ("length",): ("length", lengths)}
        rv.update(_extract_field_series(elems, path + ("element",), role="sequence_element"))
        return rv

    if all(isinstance(u, (set, frozenset)) for u in observed):
        members = []
        for u in observed:
            members.extend(list(u))
        return {
            path + ("set_size",): ("length", [len(u) if isinstance(u, (set, frozenset)) else None for u in data]),
            path + ("set_member",): ("set_member", members),
        }

    if all(isinstance(u, dict) for u in observed):
        keys = sorted({k for u in observed for k in u.keys()}, key=repr)
        rv: dict[tuple[Any, ...], tuple[str, list[Any]]] = {}
        for k in keys:
            child_values = [u.get(k, None) if isinstance(u, dict) else None for u in data]
            rv.update(_extract_field_series(child_values, path + ("key", k), role="field"))
        return rv

    return {path: (role, list(data))}


def _profile_series(path: tuple[Any, ...], role: str, values: Sequence[Any]) -> MarginalFieldProfile:
    missing = sum(1 for u in values if _is_missing_value(u))
    observed = [u for u in values if not _is_missing_value(u)]
    count = len(values)
    observed_count = len(observed)
    missing_fraction = 0.0 if count == 0 else missing / float(count)
    notes: list[str] = []

    if observed_count == 0:
        return MarginalFieldProfile(
            path, role, count, missing, missing_fraction, observed_count, "empty", "ignored", notes=notes
        )

    vdict = defaultdict(int)
    unhashable = False
    for u in observed:
        try:
            vdict[u] += 1
        except TypeError:
            unhashable = True
            break
    if unhashable:
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            "object",
            "ignored",
            notes=["unhashable values are not modeled by automatic profiling"],
        )

    entropy = _entropy_from_counts(vdict.values())
    top_mass = max(vdict.values()) / float(observed_count)
    cardinality = len(vdict)
    unique_fraction = cardinality / float(observed_count)
    effective_cardinality = 2.0**entropy
    is_constant = cardinality == 1
    if is_constant:
        notes.append("observed values are constant")

    all_bool = all(isinstance(u, (bool, np.bool_)) for u in observed)
    all_int = all(isinstance(u, (int, np.integer)) and not isinstance(u, (bool, np.bool_)) for u in observed)
    all_num = all(
        isinstance(u, numbers.Real) and not isinstance(u, (bool, np.bool_)) and math.isfinite(float(u))
        for u in observed
    )
    all_str = all(isinstance(u, (str, bytes)) for u in observed)

    if all_str:
        recommendation = "categorical"
        kind = "string"
        model_scores = _clean_scores({"categorical": _categorical_bic_bits(vdict)})
        if observed_count >= ID_MIN_COUNT and cardinality >= ID_DISTINCT_FRACTION * observed_count:
            recommendation = "ignored"
            kind = "string_identifier"
            notes.append("nearly every string value is unique")
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            kind,
            recommendation,
            bits_per_obs=entropy,
            entropy_bits=entropy,
            cardinality=cardinality,
            unique_fraction=unique_fraction,
            effective_cardinality=effective_cardinality,
            is_constant=is_constant,
            top_mass=top_mass,
            model_scores_bits=model_scores,
            notes=notes,
        )

    if all_bool:
        model_scores = _clean_scores({"categorical": _categorical_bic_bits(vdict)})
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            "boolean",
            "categorical",
            bits_per_obs=entropy,
            entropy_bits=entropy,
            cardinality=cardinality,
            unique_fraction=unique_fraction,
            effective_cardinality=effective_cardinality,
            is_constant=is_constant,
            top_mass=top_mass,
            model_scores_bits=model_scores,
            notes=notes,
        )

    if all_int:
        min_val, max_val, width = _integer_range(vdict)
        density = cardinality / float(width)
        mean = sum(int(k) * v for k, v in vdict.items()) / float(observed_count)
        var = max(0.0, sum(((int(k) - mean) ** 2) * v for k, v in vdict.items()) / float(observed_count))
        recommendation, model_scores = _recommended_integer_model(vdict)
        score_gap = _score_gap_bits(model_scores, recommendation)
        bits = model_scores.get(recommendation)
        kind = "integer"
        if recommendation == "ignored":
            recommendation = "ignored"
            kind = "integer_identifier"
            notes.append("sparse high-cardinality integer support looks identifier-like")
        elif recommendation == "poisson":
            dispersion = var / mean if mean > 0.0 else np.inf
            notes.append("selected by BIC-style code length; variance/mean dispersion is %.3f" % dispersion)
        elif recommendation == "gaussian" and min_val >= 0:
            notes.append("nonnegative integers are better explained by a discretized Gaussian code")
        if score_gap is not None and score_gap < AMBIGUOUS_SCORE_GAP_BITS:
            notes.append("top marginal models are close: %.3f bits/obs gap" % score_gap)
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            kind,
            recommendation,
            bits_per_obs=bits,
            entropy_bits=entropy,
            cardinality=cardinality,
            top_mass=top_mass,
            unique_fraction=unique_fraction,
            effective_cardinality=effective_cardinality,
            is_constant=is_constant,
            numeric_mean=mean,
            numeric_var=var,
            integer_min=min_val,
            integer_max=max_val,
            integer_density=density,
            model_scores_bits=model_scores,
            model_score_gap_bits=score_gap,
            notes=notes,
        )

    if all_num:
        arr = np.asarray(observed, dtype=float)
        mean = float(arr.mean())
        var = float(arr.var())
        # Support-typed candidates: Gaussian/Student-t for any real data, plus log-normal and gamma
        # for strictly-positive support (see _numeric_candidate_bics).
        model_scores = _numeric_candidate_bics(arr, observed_count)
        recommendation = _numeric_model_recommendation(model_scores)
        if recommendation == "lognormal":
            bits_per_obs = _lognormal_bits(arr)
        elif recommendation == "gamma":
            moments = _gamma_moments(arr)
            bits_per_obs = None if moments is None else _gamma_nll_bits(arr, *moments)
        elif recommendation in ("student_t", "mixture"):
            bits_per_obs = model_scores.get(recommendation)
        else:
            bits_per_obs = _gaussian_bits(var)

        # Goodness-of-fit gate: the PIT of the data under the recommended fit should be Uniform(0,1);
        # a small KS p-value flags miscalibration (none of the candidate families fit) -> abstain note.
        gof_ks = None
        gof_pvalue = None
        pit = _numeric_pit(arr, recommendation, mean, var)
        if pit is not None:
            gof = _pit_goodness_of_fit(pit)
            if gof is not None:
                gof_ks, gof_pvalue = gof
                if gof_pvalue < GOF_ABSTAIN_PVALUE:
                    notes.append(
                        "poor calibration: PIT-vs-uniform KS p=%.3g for %s; consider another family"
                        % (gof_pvalue, recommendation)
                    )
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            "numeric",
            recommendation,
            bits_per_obs=bits_per_obs,
            entropy_bits=entropy,
            cardinality=cardinality,
            unique_fraction=unique_fraction,
            effective_cardinality=effective_cardinality,
            is_constant=is_constant,
            top_mass=top_mass,
            numeric_mean=mean,
            numeric_var=var,
            model_scores_bits=model_scores,
            model_score_gap_bits=_score_gap_bits(model_scores, recommendation),
            gof_ks=gof_ks,
            gof_pvalue=gof_pvalue,
            notes=notes,
        )

    return MarginalFieldProfile(
        path,
        role,
        count,
        missing,
        missing_fraction,
        observed_count,
        "mixed_object",
        "ignored",
        entropy_bits=entropy,
        cardinality=cardinality,
        unique_fraction=unique_fraction,
        effective_cardinality=effective_cardinality,
        is_constant=is_constant,
        top_mass=top_mass,
        notes=["mixed scalar types are left unmodeled"],
    )


def _encode_for_pairwise(
    profile: MarginalFieldProfile, values: Sequence[Any], max_cardinality: int, num_bins: int
) -> tuple[list[Any], str] | None:
    if profile.role not in ("field", "length") or profile.recommendation == "ignored":
        return None

    encoded: list[Any] = []
    if profile.kind in ("numeric", "integer") and profile.recommendation in ("gaussian", "poisson"):
        finite = [float(u) for u in values if not _is_missing_value(u) and math.isfinite(float(u))]
        if len(finite) < 2:
            return None
        quantiles = np.linspace(0.0, 1.0, min(num_bins, len(set(finite))) + 1)[1:-1]
        edges = np.unique(np.quantile(np.asarray(finite, dtype=float), quantiles))
        for u in values:
            if _is_missing_value(u):
                encoded.append("__missing__")
            else:
                encoded.append(int(np.searchsorted(edges, float(u), side="right")))
        return encoded, "quantile_bins"

    observed = [u for u in values if not _is_missing_value(u)]
    if len(set(observed)) > max_cardinality:
        return None
    for u in values:
        encoded.append("__missing__" if _is_missing_value(u) else u)
    return encoded, "empirical_discrete"


def _mi_from_encoded(x: Sequence[Any], y: Sequence[Any]) -> tuple[float, float, float, float, float, int]:
    n = min(len(x), len(y))
    cx = defaultdict(int)
    cy = defaultdict(int)
    cxy = defaultdict(int)
    for i in range(n):
        xx = x[i]
        yy = y[i]
        cx[xx] += 1
        cy[yy] += 1
        cxy[(xx, yy)] += 1
    hx = _entropy_from_counts(cx.values())
    hy = _entropy_from_counts(cy.values())
    hxy = _entropy_from_counts(cxy.values())
    mi = max(0.0, hx + hy - hxy)

    # Plug-in MI is upward biased, especially with wide contingency tables.
    # The first-order Miller-Madow bias for an independence table is the same
    # term as the per-observation BIC penalty for adding dependence parameters.
    params = max(0, (len(cx) - 1) * (len(cy) - 1))
    bias = 0.0 if n <= 0 else params / (2.0 * float(n) * math.log(2.0))
    adjusted_mi = max(0.0, mi - bias)
    bic_gain = mi - _bic_penalty_bits(params, n)
    return mi, adjusted_mi, bic_gain, hx, hy, n


def _pairwise_permutation_p_value(
    x: Sequence[Any], y: Sequence[Any], observed_adjusted_mi: float, permutations: int, rng: np.random.RandomState
) -> float:
    if permutations <= 0:
        return 1.0
    y_arr = np.asarray(list(y), dtype=object)
    exceed = 0
    for _ in range(permutations):
        shuffled = y_arr.copy()
        rng.shuffle(shuffled)
        _, perm_adjusted, _, _, _, _ = _mi_from_encoded(x, shuffled.tolist())
        if perm_adjusted >= observed_adjusted_mi:
            exceed += 1
    return float(exceed + 1) / float(permutations + 1)


def _maximum_dependency_forest(hints: Sequence[PairwiseDependencyHint]) -> list[PairwiseDependencyHint]:
    parent: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    rank: dict[tuple[Any, ...], int] = {}

    def find(x: tuple[Any, ...]) -> tuple[Any, ...]:
        if x not in parent:
            parent[x] = x
            rank[x] = 0
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: tuple[Any, ...], y: tuple[Any, ...]) -> bool:
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    edges = []
    ordered = sorted(
        hints, key=lambda u: (-u.bic_gain_bits, -u.adjusted_mi_bits, format_path(u.left), format_path(u.right))
    )
    for hint in ordered:
        if union(hint.left, hint.right):
            edges.append(hint)
    return edges


def _edge_key(hint: PairwiseDependencyHint) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    left_key = _path_sort_key(hint.left)
    right_key = _path_sort_key(hint.right)
    return (hint.left, hint.right) if left_key <= right_key else (hint.right, hint.left)


def _pair_from_ordinal(ordinal: int, num_items: int) -> tuple[int, int]:
    remaining = int(ordinal)
    for i in range(num_items - 1):
        row_count = num_items - i - 1
        if remaining < row_count:
            return i, i + 1 + remaining
        remaining -= row_count
    return num_items - 2, num_items - 1


def _pair_index_schedule(num_items: int, max_pairs: int) -> tuple[list[tuple[int, int]], str, int]:
    total = num_items * (num_items - 1) // 2
    if total == 0 or max_pairs <= 0:
        return [], "none", total
    if max_pairs >= total:
        return [(i, j) for i in range(num_items) for j in range(i + 1, num_items)], "exhaustive", total

    ordinals = np.linspace(0, total - 1, max_pairs, dtype=int)
    pairs = [_pair_from_ordinal(int(k), num_items) for k in np.unique(ordinals)]
    return pairs, "stratified", total


def analyze_structure(
    data,
    pairwise: bool = True,
    max_pairwise_fields: int = 32,
    max_pairwise_pairs: int = 512,
    max_cardinality: int = 128,
    num_bins: int = 8,
    sample_size: int | None = 5000,
    validate_marginals: bool = True,
    validation_fraction: float = 0.25,
    max_validation_rows: int = 1000,
    validation_min_count: int = 30,
    validation_seed: int = 17,
    mi_threshold_bits: float = 0.05,
    bic_gain_threshold_bits: float = 0.0,
    pairwise_permutations: int = 0,
    permutation_alpha: float = 0.05,
    dependency_tree: bool = True,
    rng: np.random.RandomState | None = None,
    pseudo_count: float | None = 1.0,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> StructureProfile:
    """Profile data and return marginal recommendations plus pairwise hints.

    Integer marginals are compared by BIC-style average code length. Pairwise
    hints report plug-in MI, finite-sample adjusted MI, and BIC edge gain.
    Pairwise hints are deliberately unconditional and encoded through low-overhead
    empirical/quantile codes. They are useful evidence, not proof of topology:
    latent classes or states can explain the same bit gains.
    Marginal validation is a bounded deterministic train/validation split over
    scalar fields, meant as a low-overhead predictive sanity check on the BIC choice.
    """
    rows = list(normalize_input(data))  # accept a DataFrame / RDD / DataSource, not only a bare list
    total_rows = len(rows)
    root = DatumNode(data=rows)  # built directly (not via get_estimator) so its modality checks are inspectable
    estimator = root.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats)
    field_series = _extract_field_series(rows)
    fields = [
        _profile_series(path, role, values)
        for path, (role, values) in sorted(field_series.items(), key=lambda u: _path_sort_key(u[0]))
    ]
    if validate_marginals:
        for field_profile in fields:
            values = field_series[field_profile.path][1]
            _validate_marginal_profile(
                field_profile, values, validation_fraction, max_validation_rows, validation_min_count, validation_seed
            )

    warnings = ["pairwise hints are unconditional; latent mixture/state/topic structure can explain or hide them"]
    vec_dim = root._fixed_numeric_vector_dim()
    mat_shape = root._fixed_numeric_matrix_shape()
    if vec_dim is not None and vec_dim >= EMBEDDING_MIN_DIM:
        if _has_torch():
            warnings.append(
                "modality fingerprint: embedding (dim=%d >= %d) -> routed to a hybrid neural density "
                "(an exact coupling flow) instead of a bare multivariate Gaussian" % (vec_dim, EMBEDDING_MIN_DIM)
            )
        else:
            warnings.append(
                "modality fingerprint: embedding (dim=%d >= %d) would route to a hybrid neural density, "
                "but torch is not installed -- fell back to a multivariate Gaussian" % (vec_dim, EMBEDDING_MIN_DIM)
            )
    elif mat_shape is not None:
        if _has_torch():
            warnings.append(
                "modality fingerprint: image (shape=%s) -> routed through a frozen image_features extractor "
                "into a hybrid neural density instead of a per-row sequence model" % (mat_shape,)
            )
        else:
            warnings.append(
                "modality fingerprint: image (shape=%s) would route to a hybrid neural density, but torch "
                "is not installed -- fell back to the per-row sequence model" % (mat_shape,)
            )
    observed_rows = [u for u in rows if u is not None]
    if use_bstats and observed_rows and all(isinstance(u, dict) for u in observed_rows):
        warnings.append(
            "dict records are profiled by key, but the Bayesian (conjugate-prior) automatic "
            "estimator construction currently leaves dict-valued observations ignored"
        )
    for field_profile in fields:
        if (
            field_profile.validation_recommendation is not None
            and field_profile.validation_recommendation != field_profile.recommendation
        ):
            warnings.append(
                "validation disagrees with marginal recommendation for %s: %s vs %s"
                % (
                    format_path(field_profile.path),
                    field_profile.validation_recommendation,
                    field_profile.recommendation,
                )
            )
    hints: list[PairwiseDependencyHint] = []
    dependency_edges: list[PairwiseDependencyHint] = []
    residual_edges: list[PairwiseDependencyHint] = []
    redundancy_ratio = 0.0
    encoded_pairwise_fields = 0
    pairwise_fields_available = 0
    pairwise_pairs_available = 0
    pairwise_pairs_checked = 0
    pairwise_pair_strategy = "none"
    sampled_rows = total_rows

    if pairwise and total_rows > 1:
        pair_rows = rows
        if sample_size is not None and total_rows > sample_size:
            rng = np.random.RandomState(1) if rng is None else rng
            idx = rng.choice(total_rows, size=sample_size, replace=False)
            pair_rows = [rows[int(i)] for i in idx]
            sampled_rows = len(pair_rows)
            warnings.append("pairwise analysis sampled %d of %d rows" % (sampled_rows, total_rows))
        pair_series = _extract_field_series(pair_rows)
        encoded = []
        profile_by_path = {u.path: u for u in fields}
        for path, (_, values) in sorted(pair_series.items(), key=lambda u: _path_sort_key(u[0])):
            profile = profile_by_path.get(path)
            if profile is None:
                continue
            enc = _encode_for_pairwise(profile, values, max_cardinality, num_bins)
            if enc is not None:
                encoded.append((path, enc[0], enc[1]))
        pairwise_fields_available = len(encoded)
        if pairwise_fields_available > max_pairwise_fields:
            warnings.append(
                "pairwise analysis encoded %d of %d eligible fields" % (max_pairwise_fields, pairwise_fields_available)
            )
        encoded = encoded[:max_pairwise_fields]
        encoded_pairwise_fields = len(encoded)
        pair_schedule, pairwise_pair_strategy, pairwise_pairs_available = _pair_index_schedule(
            len(encoded), max_pairwise_pairs
        )
        checked = 0
        for i, j in pair_schedule:
            left, x, method_x = encoded[i]
            right, y, method_y = encoded[j]
            mi, adjusted_mi, bic_gain, hx, hy, n = _mi_from_encoded(x, y)
            checked += 1
            denom = min(hx, hy)
            norm = 0.0 if denom <= 0.0 else adjusted_mi / denom
            p_value = None
            if pairwise_permutations > 0 and adjusted_mi >= mi_threshold_bits:
                rng = np.random.RandomState(1) if rng is None else rng
                p_value = _pairwise_permutation_p_value(x, y, adjusted_mi, pairwise_permutations, rng)
            if (
                adjusted_mi >= mi_threshold_bits
                and bic_gain > bic_gain_threshold_bits
                and (p_value is None or p_value <= permutation_alpha)
            ):
                notes = ["finite-sample MI adjusted by Miller-Madow/BIC contingency penalty"]
                if p_value is not None:
                    notes.append("permutation test used %d shuffles" % pairwise_permutations)
                hints.append(
                    PairwiseDependencyHint(
                        left,
                        right,
                        mi,
                        adjusted_mi,
                        bic_gain,
                        norm,
                        hx,
                        hy,
                        n,
                        method="%s/%s" % (method_x, method_y),
                        p_value=p_value,
                        notes=notes,
                    )
                )
        pairwise_pairs_checked = checked
        hints.sort(key=lambda u: (-u.bic_gain_bits, -u.adjusted_mi_bits, format_path(u.left), format_path(u.right)))
        if pairwise_pairs_checked < pairwise_pairs_available:
            warnings.append(
                "pairwise analysis checked %d of %d eligible field pairs using %s scheduling"
                % (pairwise_pairs_checked, pairwise_pairs_available, pairwise_pair_strategy)
            )
        if dependency_tree:
            dependency_edges = _maximum_dependency_forest(hints)
            tree_keys = {_edge_key(edge) for edge in dependency_edges}
            residual_edges = [hint for hint in hints if _edge_key(hint) not in tree_keys]
            redundancy_ratio = 0.0 if len(hints) == 0 else len(residual_edges) / float(len(hints))
            if residual_edges:
                warnings.append(
                    "accepted dependency graph has %d non-tree edges; this can indicate "
                    "transitive dependence or latent/common-cause structure" % len(residual_edges)
                )

    return StructureProfile(
        estimator,
        fields,
        hints,
        warnings,
        sampled_rows,
        total_rows,
        dependency_tree_edges=dependency_edges,
        dependency_residual_edges=residual_edges,
        dependency_redundancy_ratio=redundancy_ratio,
        encoded_pairwise_fields=encoded_pairwise_fields,
        pairwise_fields_available=pairwise_fields_available,
        pairwise_pairs_available=pairwise_pairs_available,
        pairwise_pairs_checked=pairwise_pairs_checked,
        pairwise_pair_strategy=pairwise_pair_strategy,
    )


class DatumNode:
    """Accumulates type/structure evidence for one slot of the data.

    Tuples are treated as fixed-arity records (positional children). Lists,
    arrays, and other sized iterables are positional only if every observation
    has the same length (vector semantics); otherwise they are variable-length
    sequences of a merged element type with a length model. Sets map to a
    Bernoulli set model. Dicts map to keyed independent records in stats mode.
    """

    def __init__(self, parent=None, data=None):
        self.children = []
        self.dict_children = {}
        self.parent = parent
        self.vdict = defaultdict(int)
        self.len_dict = defaultdict(int)
        self.set_member = defaultdict(int)
        self.count = 0
        self.none_count = 0
        self.nan_count = 0
        self.pos_inf_count = 0
        self.neg_inf_count = 0
        self.str_count = 0
        self.float_count = 0
        self.int_count = 0
        self.bool_count = 0
        self.obj_count = 0
        self.neg_count = 0
        self.zero_count = 0
        self.tuple_count = 0
        self.seq_count = 0
        self.set_count = 0
        self.dict_count = 0

        if data is not None:
            self.add_data(data)

    def add_data(self, x):
        """Add an iterable of observations to the node profile."""
        for xx in x:
            self.add_datum(xx)

    def add_datum(self, x):
        """Add one observation and update scalar/container child profiles."""
        self.count += 1

        if x is None:
            self.none_count += 1
        elif isinstance(x, (str, bytes)):
            self.vdict[x] += 1
            self._analyze_type(x)
        elif isinstance(x, tuple):
            self.tuple_count += 1
            self.len_dict[len(x)] += 1
            for i, xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        elif isinstance(x, (set, frozenset)):
            self.set_count += 1
            self.len_dict[len(x)] += 1
            for xx in x:
                self.set_member[xx] += 1
        elif isinstance(x, dict):
            self.dict_count += 1
            present = set(x.keys())
            existing = set(self.dict_children.keys())
            for key, value in x.items():
                self._get_dict_child_node(key).add_datum(value)
            for key in existing - present:
                self.dict_children[key].add_datum(None)
        elif isinstance(x, Iterable):
            x = list(x)
            self.seq_count += 1
            self.len_dict[len(x)] += 1
            for i, xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        else:
            self._analyze_type(x)
            if not (isinstance(x, (float, np.floating)) and not math.isfinite(x)):
                self.vdict[x] += 1

    _COUNTERS = (
        "count",
        "none_count",
        "nan_count",
        "pos_inf_count",
        "neg_inf_count",
        "str_count",
        "float_count",
        "int_count",
        "bool_count",
        "obj_count",
        "neg_count",
        "zero_count",
        "tuple_count",
        "seq_count",
        "set_count",
        "dict_count",
    )

    def copy(self):
        """Return a deep copy of this profiling node and its children."""
        rv = DatumNode(self.parent)
        rv.children = [u.copy() for u in self.children]
        rv.dict_children = {k: v.copy() for k, v in self.dict_children.items()}
        rv.vdict = self.vdict.copy()
        rv.len_dict = self.len_dict.copy()
        rv.set_member = self.set_member.copy()
        for c in self._COUNTERS:
            setattr(rv, c, getattr(self, c))
        return rv

    def merge(self, x):
        """Merge another compatible profiling node into this node."""
        old_dict_count = self.dict_count
        for c in self._COUNTERS:
            setattr(self, c, getattr(self, c) + getattr(x, c))

        for i in range(len(x.children)):
            self.children[i] = self._get_child_node(i).merge(x.children[i])
        missing_right_keys = set(self.dict_children.keys()) - set(x.dict_children.keys())
        for k in missing_right_keys:
            for _ in range(x.dict_count):
                self.dict_children[k].add_datum(None)
        for k, v in x.dict_children.items():
            if k not in self.dict_children:
                child = DatumNode(self)
                for _ in range(old_dict_count):
                    child.add_datum(None)
                self.dict_children[k] = child
            self.dict_children[k] = self.dict_children[k].merge(v)
        for k, v in x.vdict.items():
            self.vdict[k] += v
        for k, v in x.len_dict.items():
            self.len_dict[k] += v
        for k, v in x.set_member.items():
            self.set_member[k] += v

        return self

    def _analyze_type(self, x, v=1):

        if isinstance(x, (bool, np.bool_)):
            self.bool_count += v
        elif isinstance(x, (float, np.floating)):
            if math.isnan(x):
                self.nan_count += v
            elif math.isinf(x):
                if x > 0:
                    self.pos_inf_count += v
                else:
                    self.neg_inf_count += v
            else:
                # Python floats carry continuous-measurement semantics even
                # when a realized value happens to be integral, e.g. a
                # standardized constant column represented as 0.0.
                self.float_count += v
            if x == 0:
                self.zero_count += v
            if math.isfinite(x) and x < 0:
                self.neg_count += v
        elif isinstance(x, (int, np.integer)):
            self.int_count += v
            if x == 0:
                self.zero_count += v
            if x < 0:
                self.neg_count += v
        elif isinstance(x, (str, bytes)):
            self.str_count += v
        else:
            self.obj_count += v

    def _leaf_estimator(self, pseudo_count, emp_suff_stat, use_bstats):
        if self.obj_count > 0 or len(self.vdict) == 0:
            return get_ignored_estimator(use_bstats=use_bstats)

        if self.str_count > 0:
            # identifier-like fields (nearly all values distinct) carry no
            # density information; ignore them instead of fitting a
            # one-bucket-per-row categorical
            if self.count >= ID_MIN_COUNT and len(self.vdict) >= ID_DISTINCT_FRACTION * self.count:
                return get_ignored_estimator(use_bstats=use_bstats)
            return get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        if self.bool_count > 0 and self.float_count == 0 and self.int_count == 0:
            return get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        if self.float_count > 0:
            # Support-typed selection over Gaussian / Student-t (any data) + log-normal / gamma
            # (positive support), via the same _numeric_candidate_bics used by the profiler. The
            # Gaussian stays the default unless an alternative beats it by the margin.
            builders = {
                "gaussian": get_gaussian_estimator,
                "student_t": get_student_t_estimator,
                "mixture": get_gaussian_mixture_estimator,
                "lognormal": get_lognormal_estimator,
                "gamma": get_gamma_estimator,
            }
            from mixle.utils.automatic.detectors import continuous_detectors

            for _d in continuous_detectors():
                builders.setdefault(_d.name, _d.factory)
            arr = _value_array_from_vdict(self.vdict)
            if arr.size:
                bics = _numeric_candidate_bics(arr, arr.size)
                if bics:
                    best = _numeric_model_recommendation(bics)
                    return builders[best](self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            return get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        if self.int_count > 0:
            recommendation, _ = _recommended_integer_model(self.vdict)
            if recommendation == "ignored":
                return get_ignored_estimator(use_bstats=use_bstats)
            if recommendation == "integer_categorical":
                return get_integer_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            if recommendation == "categorical":
                return get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            if recommendation == "poisson":
                return get_poisson_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            from mixle.utils.automatic.detectors import get_detector

            det = get_detector(recommendation)
            if det is not None:
                return det.factory(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
            return get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        return get_ignored_estimator(use_bstats=use_bstats)

    def _integer_moments(self):
        ss_0 = 0.0
        ss_1 = 0.0
        ss_2 = 0.0
        min_val = None
        max_val = None
        for k, v in self.vdict.items():
            kk = int(k)
            vv = float(v)
            ss_0 += vv
            ss_1 += kk * vv
            ss_2 += kk * kk * vv
            min_val = kk if min_val is None else min(min_val, kk)
            max_val = kk if max_val is None else max(max_val, kk)
        if ss_0 <= 0.0:
            return 0.0, 0.0, 0.0, 0, 0, 0
        mean = ss_1 / ss_0
        var = max(0.0, ss_2 / ss_0 - mean * mean)
        width = int(max_val - min_val + 1)
        return ss_0, mean, var, min_val, max_val, width

    def _integer_values_look_identifier_like(self) -> bool:
        n, _, _, _, _, width = self._integer_moments()
        distinct = len(self.vdict)
        if n < ID_MIN_COUNT or distinct < ID_DISTINCT_FRACTION * n:
            return False
        return width >= INT_ID_RANGE_MULTIPLIER * max(1, distinct)

    def _integer_values_look_poisson_like(self) -> bool:
        _, mean, var, _, _, _ = self._integer_moments()
        if mean <= 0.0:
            return False
        dispersion = var / mean
        return POISSON_DISPERSION_MIN <= dispersion <= POISSON_DISPERSION_MAX

    def _merged_child(self):
        if not self.children:
            # every observed sequence was empty (e.g. data = [[], [], []]) -- no element was ever
            # added, so there is no element type to merge. An empty DatumNode's own get_estimator()
            # already resolves to "ignored" (typed == 0), the correct answer: the length model (built
            # separately from self.len_dict, e.g. {0: n}) still captures "always empty" correctly.
            return DatumNode()
        child = self.children[0].copy()
        for u in self.children[1:]:
            child = child.merge(u)
        return child

    def get_estimator(self, pseudo_count: float | None = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
        """Infer and return an estimator for the profiled observations."""
        structured = self.tuple_count + self.seq_count + self.set_count + self.dict_count
        typed = self.count - self.none_count
        container_kinds = sum(u > 0 for u in (self.tuple_count, self.seq_count, self.set_count, self.dict_count))

        if typed == 0:
            rv = get_ignored_estimator(use_bstats=use_bstats)

        elif structured > 0 and (len(self.vdict) > 0 or self.obj_count > 0 or container_kinds > 1):
            # mixed scalars/containers or mixed container kinds: not modelable
            rv = get_ignored_estimator(use_bstats=use_bstats)

        elif self.set_count > 0:
            rv = get_set_estimator(self.set_member, self.set_count, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        elif self.dict_count > 0:
            if use_bstats:
                rv = get_ignored_estimator(use_bstats=use_bstats)
            else:
                keys = sorted(self.dict_children.keys(), key=repr)
                rv = get_dict_record_estimator(
                    keys,
                    [
                        self.dict_children[k].get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats)
                        for k in keys
                    ],
                )

        elif structured > 0:
            fixed_arity = len(self.len_dict) == 1
            if self.tuple_count > 0 and self.seq_count == 0 and fixed_arity:
                # records: positional composite
                rv = get_composite_estimator(
                    [u.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats) for u in self.children],
                    use_bstats=use_bstats,
                )
            elif self._fixed_numeric_vector_dim() is not None:
                vec_dim = self._fixed_numeric_vector_dim()
                if vec_dim >= EMBEDDING_MIN_DIM and _has_torch():
                    # modality fingerprint: "embedding" -- a high-dim numeric vector is far more often a
                    # frozen encoder's output than a handful of jointly-Gaussian measurements, so a bare
                    # multivariate Gaussian is the wrong default here (see EMBEDDING_MIN_DIM in factories.py).
                    rv = get_hybrid_embedding_estimator(vec_dim)
                else:
                    rv = get_multivariate_gaussian_estimator(vec_dim, use_bstats=use_bstats)
            elif self._fixed_numeric_matrix_shape() is not None and _has_torch():
                # modality fingerprint: "image" -- a homogeneous 2-D numeric array field, routed through a
                # frozen deterministic feature extractor into the same hybrid neural density.
                rv = get_hybrid_image_estimator()
            elif fixed_arity and self.tuple_count == 0 and not self._children_homogeneous():
                # fixed-length lists/vectors with positionally distinct types
                rv = get_composite_estimator(
                    [u.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats) for u in self.children],
                    use_bstats=use_bstats,
                )
            else:
                # variable-length (or homogeneous fixed-length) sequences
                child = self._merged_child()
                rv = get_sequence_estimator(
                    child.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats),
                    len_dict=self.len_dict,
                    pseudo_count=pseudo_count,
                    emp_suff_stat=emp_suff_stat,
                    use_bstats=use_bstats,
                )

        else:
            rv = self._leaf_estimator(pseudo_count, emp_suff_stat, use_bstats)

        if self.none_count > 0:
            rv = get_optional_estimator(rv, None, use_bstats=use_bstats)

        if self.nan_count > 0:
            rv = get_optional_estimator(rv, math.nan, use_bstats=use_bstats)

        # +-inf is tracked (pos_inf_count/neg_inf_count) but, unlike None/NaN, was never wrapped here --
        # the base estimator built above (e.g. GaussianEstimator, whose distribution requires finite
        # support) never sees these values (add_datum excludes them from vdict), so get_estimator()
        # returned successfully while optimize() on the SAME unfiltered data crashed. Wrap each sign
        # present the same way None/NaN already are, so a field containing infinities gets a genuinely
        # fittable estimator instead of one that silently can't handle the data it was inferred from.
        if self.pos_inf_count > 0:
            rv = get_optional_estimator(rv, math.inf, use_bstats=use_bstats)

        if self.neg_inf_count > 0:
            rv = get_optional_estimator(rv, -math.inf, use_bstats=use_bstats)

        return rv

    def _fixed_numeric_vector_dim(self):
        if (
            self.tuple_count > 0
            or self.seq_count == 0
            or self.set_count > 0
            or self.dict_count > 0
            or len(self.len_dict) != 1
        ):
            return None
        dim = next(iter(self.len_dict))
        if dim <= 1 or len(self.children) != dim:
            return None
        for child in self.children:
            if child.count != self.seq_count:
                return None
            if child.none_count > 0 or child.nan_count > 0 or child.pos_inf_count > 0 or child.neg_inf_count > 0:
                return None
            if child.str_count > 0 or child.bool_count > 0 or child.obj_count > 0:
                return None
            if child.tuple_count > 0 or child.seq_count > 0 or child.set_count > 0 or child.dict_count > 0:
                return None
            if child.int_count + child.float_count == 0:
                return None
        return dim

    def _fixed_numeric_matrix_shape(self):
        """Detect a homogeneous 2-D numeric array field (an "image"-shaped field): a fixed-length outer
        sequence whose every row is itself a fixed-length numeric vector of the same width. A 2-D/3-D
        numpy array iterates row-by-row into nested Iterables, so this is what an image datum looks like
        by the time it reaches DatumNode."""
        if (
            self.tuple_count > 0
            or self.seq_count == 0
            or self.set_count > 0
            or self.dict_count > 0
            or len(self.len_dict) != 1
        ):
            return None
        rows = next(iter(self.len_dict))
        if rows <= 1 or len(self.children) != rows:
            return None
        width = None
        for child in self.children:
            if child.count != self.seq_count:
                return None
            w = child._fixed_numeric_vector_dim()
            if w is None:
                return None
            if width is None:
                width = w
            elif w != width:
                return None
        return (rows, width)

    def _children_homogeneous(self):
        """True when all positional children carry the same scalar type profile,
        so a fixed-length list is better modeled as an iid sequence than a
        composite of per-position estimators."""
        if len(self.children) <= 1:
            return True

        def profile(u):
            return (
                u.str_count > 0,
                u.bool_count > 0,
                u.float_count > 0,
                u.int_count > 0,
                u.obj_count > 0,
                u.tuple_count > 0,
                u.seq_count > 0,
                u.set_count > 0,
                u.dict_count > 0,
                len(u.children) > 0,
            )

        profiles = {profile(u) for u in self.children}
        if len(profiles) > 1:
            return False

        # numeric positions with disjoint supports look like distinct dimensions
        p = next(iter(profiles))
        if p[2] or p[3]:
            return False

        return True

    def _get_child_node(self, idx: int):
        while len(self.children) <= idx:
            self.children.append(DatumNode(self))
        return self.children[idx]

    def _get_dict_child_node(self, key: Any):
        if key not in self.dict_children:
            child = DatumNode(self)
            for _ in range(max(0, self.dict_count - 1)):
                child.add_datum(None)
            self.dict_children[key] = child
        return self.dict_children[key]


def normalize_input(data, *, rdd_cap: int = 200000):
    """Coerce a profiler input to a list of records, accepting more than a bare Python list.

    Recognized inputs (each yields the same record stream the profiler/encoder consume):
    * a mixle :class:`~mixle.data.core.DataSource` (typed/structured) -> its ``records()``;
    * a pandas ``DataFrame`` (duck-typed via ``columns``/``itertuples``; pandas is never imported) ->
      one record per row across its columns (scalar for a single column, tuple otherwise);
    * a Spark ``RDD`` -> the first ``rdd_cap`` rows (profiling works on a bounded sample);
    * anything else (a list / sequence) is returned unchanged.
    """
    if hasattr(data, "records") and hasattr(data, "structure"):  # a mixle DataSource
        try:
            from mixle.data.core import DataSource

            if isinstance(data, DataSource):
                return list(data.records())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(data, "columns") and hasattr(data, "itertuples"):  # a pandas DataFrame (duck-typed)
        from mixle.data.sources.pandas_source import dataframe_records

        return dataframe_records(data)
    try:
        from mixle.utils.optional_deps import RDD_TYPES

        if RDD_TYPES and isinstance(data, RDD_TYPES):  # a Spark RDD -> a bounded local sample
            return data.take(int(rdd_cap))
    except Exception:  # noqa: BLE001
        pass
    return data


def get_estimator(data, pseudo_count: float | None = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
    """Profile ``data`` and return the automatically selected estimator."""
    return DatumNode(data=normalize_input(data)).get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats)


def get_prototype(
    data,
    *,
    seed: int | None = None,
    p: float = 0.1,
    pseudo_count: float | None = 1.0,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
):
    """Infer a model structure and return an initialized prototype distribution.

    Where :func:`get_estimator` returns the estimator used for fitting, this
    returns a concrete unfitted distribution whose tree mirrors the detected
    families. Use it when the inferred model shape should be inspected,
    customized, or passed to ``optimize(data, prototype)`` as a prototype.

        proto = get_prototype(records)     # see the inferred composite structure
        model = optimize(records, proto)   # fit it (or pass proto to fit(...))

    ``seed`` makes the randomized initialization reproducible; ``p`` is the
    per-observation keep-probability of the vectorized initializer. Remaining
    arguments mirror :func:`get_estimator`.
    """
    import numpy as np

    from mixle.stats.compute.sequence import seq_encode, seq_initialize

    rows = normalize_input(data)
    est = get_estimator(rows, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
    enc = seq_encode(rows, estimator=est)
    return seq_initialize(enc_data=enc, estimator=est, rng=np.random.RandomState(seed), p=p)
