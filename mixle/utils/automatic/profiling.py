"""Data profiling and model recommendation for automatically-typed data.

Profiles sequences of observations to recommend per-field leaf estimators,
measure unconditional pairwise dependency hints, and assemble the composite
estimator via DatumNode. Estimator builders are imported from .factories.
"""

import math
import numbers
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    ContractError,
    ParameterEstimator,
)

from .factories import (
    AMBIGUOUS_SCORE_GAP_BITS,
    AMBIGUOUS_TABLE_MIN_SHARE,
    EMBEDDING_MIN_DIM,
    ID_DISTINCT_FRACTION,
    ID_MIN_COUNT,
    INT_ID_RANGE_MULTIPLIER,
    MALFORMED_TABLE_MIN_SHARE,
    MAX_INT_CATEGORICAL_DISTINCT,
    MAX_INT_CATEGORICAL_FRACTION,
    POISSON_DISPERSION_MAX,
    POISSON_DISPERSION_MIN,
    VALIDATION_ALPHA,
    VALIDATION_VARIANCE_FLOOR,
    _dense_integer_support,
    _get_identifier_estimator,
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
    get_typed_mixture_estimator,
    parse_numeric_text,
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


@dataclass(frozen=True)
class MarginalSelectionDecision:
    """Immutable evidence and final family choice for one marginal field."""

    path: tuple[Any, ...]
    training_recommendation: str
    validation_recommendation: str | None
    validation_score_gap_bits: float | None
    selected_recommendation: str
    validation_overrode: bool


@dataclass(frozen=True)
class AutomaticSelectionResult:
    """The estimator returned to callers together with every decision that produced it."""

    estimator: ParameterEstimator
    decisions: tuple[MarginalSelectionDecision, ...]


@dataclass(frozen=True)
class ValueArraySamplingReceipt:
    """Evidence describing a bounded expansion of weighted numeric values."""

    input_weight: float
    input_cardinality: int
    output_count: int
    output_cardinality: int
    cap: int
    approximated: bool
    method: str
    max_cdf_error_bound: float


@dataclass(frozen=True)
class _SequenceObservation(Sequence[Any]):
    """Reusable immutable form of a list/array/generator-valued observation."""

    values: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


@dataclass(frozen=True)
class _MappingObservation(Mapping[Any, Any]):
    """Reusable immutable form of a mapping-valued observation."""

    entries: tuple[tuple[Any, Any], ...]

    def __getitem__(self, key: Any) -> Any:
        for candidate, value in self.entries:
            if candidate == key and type(candidate) is type(key):
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[Any]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


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
    selection: AutomaticSelectionResult | None = None

    def recommend(self) -> ParameterEstimator:
        """Return the estimator selected by the structure-analysis pass."""
        return self.estimator if self.selection is None else self.selection.estimator

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
    return x is None or (isinstance(x, (float, np.floating)) and not math.isfinite(float(x)))


def _is_sequence_like(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes, Mapping, set, frozenset))


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
    # Differential code lengths may legitimately be negative when a density is
    # concentrated enough to exceed one. Preserve that scale just as every
    # competing continuous family does.
    return 0.5 * math.log(2.0 * math.pi * math.e * var, 2.0)


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


def _gamma_mle(arr: np.ndarray) -> tuple[float, float] | None:
    """Return the maximum-likelihood Gamma ``(shape, scale)`` with location fixed at zero."""
    if arr.size == 0 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return None
    if float(arr.var()) <= 0.0:
        return None
    from scipy import stats

    try:
        shape, location, scale = stats.gamma.fit(arr, floc=0.0)
    except (FloatingPointError, RuntimeError, ValueError):
        return None
    if (
        location != 0.0
        or not math.isfinite(float(shape))
        or not math.isfinite(float(scale))
        or shape <= 0.0
        or scale <= 0.0
    ):
        return None
    return float(shape), float(scale)


def _gamma_nll_bits(arr: np.ndarray, k: float, theta: float) -> float:
    """Per-observation Gamma code length (bits) of data ``arr`` under ``Gamma(k, theta)``."""
    logs = np.log(arr)
    nats = -((k - 1.0) * float(logs.mean()) - float(arr.mean()) / theta - math.lgamma(k) - k * math.log(theta))
    return nats / math.log(2.0)


def _gamma_bic_bits(values: Sequence[float], nobs: int) -> float | None:
    arr = np.asarray(values, dtype=float)
    params = _gamma_mle(arr)
    if params is None:
        return None
    k, theta = params
    return _gamma_nll_bits(arr, k, theta) + _bic_penalty_bits(2, nobs)


def _student_t_mle(arr: np.ndarray) -> tuple[float, float, float] | None:
    """Return maximum-likelihood Student-t ``(df, loc, scale)``.

    The candidate remains gated to empirically heavy-tailed data; once admitted,
    all three charged parameters are fitted by likelihood rather than moments.
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
    from scipy import stats

    try:
        df, loc, scale = stats.t.fit(arr)
    except (FloatingPointError, RuntimeError, ValueError):
        return None
    if (
        not math.isfinite(float(df))
        or not math.isfinite(float(loc))
        or not math.isfinite(float(scale))
        or df <= 0.0
        or scale <= 0.0
    ):
        return None
    return float(df), float(loc), float(scale)


def _student_t_nll_bits(arr: np.ndarray, params: tuple[float, float, float]) -> float:
    from scipy import stats

    df, loc, scale = params
    return -float(np.mean(stats.t.logpdf(arr, df, loc=loc, scale=scale))) / math.log(2.0)


def _student_t_bic_bits(values: Sequence[float], nobs: int) -> float | None:
    arr = np.asarray(values, dtype=float)
    params = _student_t_mle(arr)
    if params is None:
        return None
    return _student_t_nll_bits(arr, params) + _bic_penalty_bits(3, nobs)


def _validation_student_t_bits(train: Sequence[float], validation: Sequence[float]) -> float | None:
    from scipy import stats

    if not train or not validation:
        return None
    params = _student_t_mle(np.asarray(train, dtype=float))
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


_FLOAT64_EPS = 2.220446049250313e-16
# Largest ``shape * log-magnitude`` an origin-anchored family's log-density may carry before float64
# stops being able to tell its code length from the next candidate's. Set at the selection margin
# itself: a rounding error of that size can decide the recommendation, which is precisely the
# failure this bounds.
_ORIGIN_ANCHORED_MAX_CONDITION = (0.02 * math.log(2.0)) / _FLOAT64_EPS


def _origin_anchored_scores_unmeasurable(arr: np.ndarray) -> bool:
    """Whether a positive-support family's code length on ``arr`` would be float64 rounding noise.

    A family parameterized as a ratio to zero -- gamma, log-normal, inverse-gamma, Weibull,
    inverse-Gaussian, Rayleigh -- writes its log-density with terms of size ``shape * log|x|`` and
    ``lgamma(shape)`` that cancel down to an O(1) answer. Fitted to a sample sitting far from the
    origin relative to its own spread, the shape it needs is about ``(mean/sd)**2``, so those terms
    run to 1e13 nats and higher: what comes back is the rounding residue, and a residue larger than
    the 0.02 bits/obs selection margin decides the recommendation. Measured on this tree, plain
    ``N(1e7, 1)`` drew a gamma "win" of 0.03 bits over the Gaussian and a 200-row ramp at 1.7e9 drew
    an inverse-gamma one, each yielding a fit whose spread was 10x-30x the data's with nothing
    reported as repaired.

    What this also declines, said plainly: genuinely gamma / log-normal / Weibull data whose
    coefficient of variation is below about 1e-6. Such a sample is a Gaussian to every digit float64
    carries -- every one of these families converges to the Gaussian as its shape grows -- and the
    shape it would need (>1e12) is past the cap the library's own Gamma and InverseGamma
    distributions enforce, so the family could not represent the answer even if it were selected.
    The location-scale candidates (Gaussian, Student-t, mixture, and any detector that also accepts
    the centered data) subtract the location before they do anything else and are never affected, so
    the data is still typed -- by the families that can still be scored.
    """
    if arr.size == 0:
        return False
    mean = float(arr.mean())
    sd = float(arr.std())
    if not (sd > 0.0) or not (mean > 0.0) or not math.isfinite(mean) or not math.isfinite(sd):
        return False
    shape = (mean / sd) ** 2
    if not math.isfinite(shape):
        return True
    # Both ``shape * log|x|`` and ``lgamma(shape) ~ shape * log(shape)`` appear; charge the larger.
    magnitude = abs(math.log(mean)) + math.log1p(shape) + 1.0
    return shape * magnitude > _ORIGIN_ANCHORED_MAX_CONDITION


def _numeric_candidate_bics(arr: np.ndarray, nobs: int) -> dict[str, float]:
    """Return per-candidate BIC code lengths for numeric data (support-typed).

    Gaussian and Student-t apply to any real data; a 2-component Gaussian mixture is added only when
    the data looks multimodal (it overfits unimodal samples otherwise); log-normal and gamma are
    added only for strictly-positive support. Used by both the marginal profiler and
    ``get_estimator`` so the candidate set and selection stay consistent.

    Origin-anchored families are additionally dropped when their code length on this data would be
    float64 rounding noise -- see :func:`_origin_anchored_scores_unmeasurable`. A detector declares
    which kind it is through its own support gate: one that still applies to the data with its mean
    removed reads only the location and the spread, so its score is unaffected.
    """
    candidates: dict[str, float | None] = {
        "gaussian": _gaussian_bic_bits(float(arr.var()), nobs),
        "student_t": _student_t_bic_bits(arr, nobs),
    }
    unmeasurable = _origin_anchored_scores_unmeasurable(arr)
    centered = arr - float(arr.mean()) if unmeasurable else None
    # The 2-component mixture is added only for plausibly-multimodal data with enough distinct
    # values that its components cannot collapse onto a few points (which would overfit wildly).
    if arr.size and np.unique(arr).size >= 12 and _looks_multimodal(arr):
        candidates["mixture"] = _mixture_bic_bits(arr, nobs)
    if arr.size and np.all(arr > 0.0) and not unmeasurable:
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
            # Two ways to be exempt. The detector's own support gate says which kind it is: a
            # location-scale family accepts the centered copy too, because it reads only location
            # and spread. An origin-anchored family refuses the centered copy -- its values are no
            # longer positive -- and is dropped unless it declares that its score is computed in a
            # form that survives the offset (``offset_stable``, which the Pareto sets). Only at a
            # conditioning where the naive form cannot be computed at all.
            if unmeasurable and not getattr(d, "offset_stable", False) and not d.applies(centered):
                continue
            candidates[d.name] = d.score(arr, nobs)
    return _clean_scores(candidates)


def _value_array_from_vdict(
    vdict: dict[Any, float], cap: int = 200000, *, return_receipt: bool = False
) -> np.ndarray | tuple[np.ndarray, ValueArraySamplingReceipt]:
    """Expand weighted numeric values without exceeding ``cap``.

    Integral weights whose total fits within the cap are expanded exactly.
    Larger or fractional weighted samples use deterministic midpoint
    stratification over the empirical CDF. That preserves mass without
    forcing one copy of every support value and bounds the CDF error by
    ``1 / (2 * output_count)``.
    """
    if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)) or int(cap) <= 0:
        raise ValueError("cap must be a positive integer")
    cap = int(cap)
    pairs: list[tuple[float, float]] = []
    for k, v in vdict.items():
        if not isinstance(k, (int, float, np.integer, np.floating)) or not math.isfinite(float(k)):
            continue
        weight = float(v)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("numeric value counts must be finite and nonnegative")
        if weight > 0.0:
            pairs.append((float(k), weight))
    pairs.sort(key=lambda item: item[0])
    total = math.fsum(weight for _, weight in pairs)
    if not pairs:
        values = np.empty(0, dtype=float)
        receipt = ValueArraySamplingReceipt(0.0, 0, 0, 0, cap, False, "empty", 0.0)
    elif total <= cap and all(weight.is_integer() for _, weight in pairs):
        values = np.repeat(
            np.asarray([key for key, _ in pairs], dtype=float),
            np.asarray([int(weight) for _, weight in pairs], dtype=np.int64),
        )
        receipt = ValueArraySamplingReceipt(
            total,
            len(pairs),
            int(values.size),
            int(np.unique(values).size),
            cap,
            False,
            "exact",
            0.0,
        )
    else:
        output_count = min(cap, max(1, int(round(total))))
        cumulative = np.cumsum(np.asarray([weight for _, weight in pairs], dtype=np.float64))
        positions = (np.arange(output_count, dtype=np.float64) + 0.5) * (total / output_count)
        indices = np.searchsorted(cumulative, positions, side="right")
        keys = np.asarray([key for key, _ in pairs], dtype=float)
        values = keys[np.minimum(indices, keys.size - 1)]
        receipt = ValueArraySamplingReceipt(
            total,
            len(pairs),
            output_count,
            int(np.unique(values).size),
            cap,
            True,
            "deterministic_midpoint_cdf",
            0.5 / output_count,
        )
    if return_receipt:
        return values, receipt
    return values


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


def _validation_numeric_pit(
    train: Sequence[float], validation: Sequence[float], recommendation: str
) -> np.ndarray | None:
    """Fit on ``train`` and return PIT values only for untouched validation rows."""
    from scipy import stats

    arr = np.asarray(train, dtype=np.float64)
    val = np.asarray(validation, dtype=np.float64)
    if arr.size < 2 or val.size < 2 or not np.all(np.isfinite(arr)) or not np.all(np.isfinite(val)):
        return None
    mean = float(arr.mean())
    var = float(arr.var())
    if recommendation == "gaussian" and var > 0.0:
        return stats.norm.cdf(val, loc=mean, scale=math.sqrt(var))
    if recommendation == "student_t":
        params = _student_t_mle(arr)
        if params is not None:
            df, loc, scale = params
            return stats.t.cdf(val, df, loc=loc, scale=scale)
        return None
    if recommendation == "mixture":
        fit = _fit_gaussian_mixture2(arr)
        if fit is None:
            return None
        _, (weights, means, variances) = fit
        return sum(
            weight * stats.norm.cdf(val, loc=component_mean, scale=math.sqrt(component_var))
            for weight, component_mean, component_var in zip(weights, means, variances)
        )
    if recommendation == "lognormal":
        logs = np.log(arr)
        lvar = float(logs.var())
        if lvar > 0.0 and np.all(val > 0.0):
            return stats.norm.cdf(np.log(val), loc=float(logs.mean()), scale=math.sqrt(lvar))
        return None
    if recommendation == "gamma":
        params = _gamma_mle(arr)
        if params is not None:
            k, theta = params
            return stats.gamma.cdf(val, a=k, scale=theta)
    from mixle.utils.automatic.detectors import get_detector

    detector = get_detector(recommendation)
    if detector is not None:
        model = _fit_detector_model(detector, train)
        cdf = None if model is None else getattr(model, "cdf", None)
        if callable(cdf):
            try:
                values = np.asarray([cdf(value) for value in val], dtype=np.float64)
            except Exception:  # noqa: BLE001
                return None
            return values if np.all(np.isfinite(values)) else None
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


def _numeric_side_is_memorizing(numeric_vdict: dict[Any, float], has_float: bool) -> bool:
    """Would the NUMBERS in a mixed number/string column be modeled as a table of observed values?

    This is the gate on the typed dispatch mixture, and it is shared by the profile report and the
    estimator builder so the two can never disagree about one column.

    The harm the mixture repairs is specific: a merged categorical over every observed value scores
    ``-inf`` at any number the numeric side's own model would have scored finitely. If the numeric
    side is itself a categorical -- a handful of repeated small integers, an integer identifier
    column -- there is no such number, the merged categorical is an equivalent and simpler model, and
    splitting would only churn the shape of a column that is genuinely categorical. So: split only
    when the numbers want a family with support past what was observed.

    Floats always want one. mixle reads a Python float as a continuous measurement even when the
    realized value is integral (see ``DatumNode._analyze_type``), so ``[2.0, 3.0] * 80`` on its own
    fits a continuous family, and the mixed column must not disagree with the same column minus its
    strings. Integers are asked the ordinary integer-model question instead.
    """
    if has_float:
        return False
    if not numeric_vdict:
        return True
    recommendation, _ = _recommended_integer_model(numeric_vdict)
    return recommendation in ("categorical", "integer_categorical", "ignored")


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
    params = _gamma_mle(np.asarray(train, dtype=float))
    if params is None:
        return None
    k, theta = params
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


def _fit_detector_model(detector: Any, train: Sequence[Any]) -> Any | None:
    """Fit one registered detector family using only the supplied training rows."""
    if not train:
        return None
    vdict: dict[Any, float] = defaultdict(float)
    for value in train:
        vdict[value] += 1.0
    try:
        estimator = detector.factory(vdict, None, True, False)
        accumulator = estimator.accumulator_factory().make()
        encoder = accumulator.acc_to_encoder()
        encoded = encoder.seq_encode(train)
        accumulator.seq_update(encoded, np.ones(len(train), dtype=np.float64), None)
        return estimator.estimate(float(len(train)), accumulator.value())
    except Exception:  # noqa: BLE001
        return None


def _validation_detector_bits(detector: Any, train: Sequence[Any], validation: Sequence[Any]) -> float | None:
    """Fit one registered family on train only and score untouched validation data."""
    if not validation:
        return None
    try:
        model = _fit_detector_model(detector, train)
        if model is None:
            return None
        scores = np.asarray([model.log_density(value) for value in validation], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None
    if scores.shape != (len(validation),) or not np.all(np.isfinite(scores)):
        return None
    return -float(scores.mean()) / math.log(2.0)


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

    if profile.kind in ("string", "boolean", "mixed_categorical"):
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
        from mixle.utils.automatic.detectors import get_detector

        for name in sorted(candidates):
            detector = get_detector(name)
            if detector is not None and detector.kind == "discrete":
                scores[name] = _validation_detector_bits(detector, train, validation)

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
        from mixle.utils.automatic.detectors import get_detector

        for name in sorted(profile.model_scores_bits):
            detector = get_detector(name)
            if detector is not None and detector.kind == "continuous":
                scores[name] = _validation_detector_bits(detector, train_f, val_f)

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
    if profile.kind == "numeric":
        pit = _validation_numeric_pit(train, validation, profile.robust_recommendation())
        gof = None if pit is None else _pit_goodness_of_fit(pit)
        if gof is not None:
            profile.gof_ks, profile.gof_pvalue = gof
            if profile.gof_pvalue < GOF_ABSTAIN_PVALUE:
                profile.notes.append(
                    "poor calibration on held-out data: PIT-vs-uniform KS p=%.3g for %s; "
                    "consider another family" % (profile.gof_pvalue, profile.robust_recommendation())
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

    if all(isinstance(u, Mapping) for u in observed):
        keys = sorted({k for u in observed for k in u.keys()}, key=repr)
        rv: dict[tuple[Any, ...], tuple[str, list[Any]]] = {}
        for k in keys:
            child_values = [u.get(k, None) if isinstance(u, Mapping) else None for u in data]
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
    nonfinite_count = sum(
        1 for value in values if isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
    )
    if nonfinite_count:
        notes.append(
            "%d non-finite value(s) are modeled as explicit optional outcomes carrying a fitted rate, "
            "not fitted as numeric data" % nonfinite_count
        )

    if observed_count == 0:
        return MarginalFieldProfile(
            path, role, count, missing, missing_fraction, observed_count, "empty", "ignored", notes=notes
        )

    # A string that reads as a number -- "46.8037", the shape a numeric CSV column takes once one
    # dirty cell coerces the whole column to object/str dtype -- is retyped to the float it names
    # before any kind/recommendation decision below, exactly as DatumNode.add_datum retypes it
    # before deciding str_count vs float_count. Sharing the predicate (parse_numeric_text) is what
    # keeps this report from claiming a different model than optimize() actually builds; see
    # test_profile_reports_the_typed_mixture_instead_of_ignored (T4-01) and
    # PandasObjectDtypeColumnTest.test_profile_names_the_column_correctly (T1-03).
    retyped = []
    for value in observed:
        parsed = parse_numeric_text(value) if isinstance(value, str) else None
        retyped.append(value if parsed is None else parsed)
    observed = retyped

    has_bool = any(isinstance(value, (bool, np.bool_)) for value in observed)
    has_nonbool_number = any(
        isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) for value in observed
    )
    if has_bool and has_nonbool_number:
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            "ambiguous_numeric_categorical",
            "ignored",
            notes=notes + ["Boolean and numeric values cannot be distinct keys under Python equality (True == 1)"],
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
    mixed_categorical = (
        all(isinstance(u, (str, bytes, int, np.integer, bool, np.bool_)) for u in observed)
        and len({type(value) for value in observed}) > 1
    )

    # A number/string mix is two disjoint types, not one unmodelable or one categorical column: the
    # estimator builds a typed dispatch mixture (a family for the numbers, a categorical for the
    # strings, weights from the observed type proportions). The recommendation reported here is
    # load-bearing, not decorative -- 'ignored' is what makes _validate_field_profile skip predictive
    # validation and _encode_for_pairwise refuse the column -- so it has to name what really happens.
    # The type predicate is deliberately the one DatumNode._analyze_type uses to fill str_count /
    # float_count / int_count rather than the looser numbers.Real: anything else (a Fraction, a
    # Decimal, a datetime) lands in obj_count there and is frozen, and claiming a typed mixture for a
    # column the estimator freezes would be the same kind of lie this repair exists to remove.
    numeric_observed = [
        u
        for u in observed
        if isinstance(u, (int, float, np.integer, np.floating)) and not isinstance(u, (bool, np.bool_))
    ]
    string_observed = [u for u in observed if isinstance(u, (str, bytes))]
    if numeric_observed and string_observed and len(numeric_observed) + len(string_observed) == observed_count:
        numeric_vdict: dict[Any, float] = defaultdict(int)
        for u in numeric_observed:
            numeric_vdict[u] += 1
        has_float = any(isinstance(u, (float, np.floating)) for u in numeric_observed)
        if not _numeric_side_is_memorizing(numeric_vdict, has_float):
            return MarginalFieldProfile(
                path,
                role,
                count,
                missing,
                missing_fraction,
                observed_count,
                "mixed_scalar",
                "typed_mixture",
                entropy_bits=entropy,
                cardinality=cardinality,
                unique_fraction=unique_fraction,
                effective_cardinality=effective_cardinality,
                is_constant=is_constant,
                top_mass=top_mass,
                notes=notes
                + [
                    "numeric and string values are modeled as a typed dispatch mixture "
                    "(%d numeric, %d string); one stray non-numeric cell no longer retypes the column"
                    % (len(numeric_observed), len(string_observed))
                ],
            )

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

    if mixed_categorical:
        model_scores = _clean_scores({"categorical": _categorical_bic_bits(vdict)})
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            "mixed_categorical",
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
            params = _gamma_mle(arr)
            bits_per_obs = None if params is None else _gamma_nll_bits(arr, *params)
        elif recommendation == "student_t":
            params = _student_t_mle(arr)
            bits_per_obs = None if params is None else _student_t_nll_bits(arr, params)
        elif recommendation == "mixture":
            bits_per_obs = model_scores.get(recommendation)
        else:
            bits_per_obs = _gaussian_bits(var)

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
            notes=notes,
        )

    # Everything that reaches here mixes a scalar type this profiler does not model -- a datetime, a
    # Decimal, a Fraction -- into the column, which is what still makes it unmodelable. A plain
    # number/string mix returned above as a typed mixture.
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
        notes=notes + ["mixed scalar types are left unmodeled"],
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
    modality: str | None = None,
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
    if modality not in {None, "embedding", "image"}:
        raise ValueError("modality must be None, 'embedding', or 'image'")
    rows = list(normalize_input(data))  # accept a DataFrame / RDD / DataSource, not only a bare list
    total_rows = len(rows)
    root = DatumNode(data=rows)  # built directly (not via get_estimator) so its modality checks are inspectable
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
    deployed_recommendations = {profile.path: profile.robust_recommendation() for profile in fields}
    estimator = root.get_estimator(
        pseudo_count,
        emp_suff_stat,
        use_bstats=use_bstats,
        recommendations=deployed_recommendations,
        modality=modality,
    )
    selection = AutomaticSelectionResult(
        estimator=estimator,
        decisions=tuple(
            MarginalSelectionDecision(
                path=profile.path,
                training_recommendation=profile.recommendation,
                validation_recommendation=profile.validation_recommendation,
                validation_score_gap_bits=profile.validation_score_gap_bits,
                selected_recommendation=profile.robust_recommendation(),
                validation_overrode=profile.robust_recommendation() != profile.recommendation,
            )
            for profile in fields
        ),
    )

    warnings = ["pairwise hints are unconditional; latent mixture/state/topic structure can explain or hide them"]
    # analyze_structure is the diagnostic surface, so a malformed table is REPORTED here rather than
    # refused the way get_estimator refuses it -- the whole point of asking for a profile is to see
    # what the data looks like, including that one row does not fit the shape the rest establish.
    ragged_diagnosis = diagnose_ragged_rows(rows)
    if ragged_diagnosis is not None:
        warnings.append(ragged_diagnosis.note())
    vec_dim = root._fixed_numeric_vector_dim()
    mat_shape = root._fixed_numeric_matrix_shape()
    if modality == "embedding":
        if vec_dim is None:
            raise ValueError("modality='embedding' requires fixed-length, finite numeric vectors")
        if _has_torch():
            warnings.append(
                "explicit modality: embedding (dim=%d) -> routed to a hybrid neural density "
                "(an exact coupling flow) instead of a bare multivariate Gaussian" % vec_dim
            )
        else:
            warnings.append(
                "explicit modality: embedding (dim=%d) would route to a hybrid neural density, "
                "but torch is not installed -- fell back to a multivariate Gaussian" % vec_dim
            )
    elif modality == "image":
        if mat_shape is None:
            raise ValueError("modality='image' requires homogeneous finite numeric matrices")
        if _has_torch():
            warnings.append(
                "explicit modality: image (shape=%s) -> routed through a frozen image_features extractor "
                "into a hybrid neural density instead of a per-row sequence model" % (mat_shape,)
            )
        else:
            warnings.append(
                "explicit modality: image (shape=%s) would route to a hybrid neural density, but torch "
                "is not installed -- fell back to the per-row sequence model" % (mat_shape,)
            )
    elif vec_dim is not None and vec_dim >= EMBEDDING_MIN_DIM:
        warnings.append(
            "wide numeric vector (dim=%d) retained mathematical-vector semantics; pass modality='embedding' "
            "only when provenance establishes embedding semantics" % vec_dim
        )
    elif mat_shape is not None:
        warnings.append(
            "numeric matrix (shape=%s) retained array semantics; pass modality='image' only when provenance "
            "establishes image semantics" % (mat_shape,)
        )
    elif vec_dim is None:
        # A fixed-length all-numeric vector is otherwise the one shape the automatic pipeline builds a
        # dependency-capturing joint estimator for (MultivariateGaussianEstimator); missing/non-finite
        # entries disqualify it (the multivariate-Gaussian distribution requires a fully-observed
        # vector -- there is no per-dimension optional/missing-value wrapping for it, unlike scalar
        # fields). Without this note the frontier just quietly has one fewer candidate: this makes the
        # skip explicit instead of silent, the same convention used for the embedding/image routes above.
        missing_vec_dim = root._fixed_numeric_vector_dim(allow_missing=True)
        if missing_vec_dim is not None:
            affected = sum(
                1
                for child in root.children
                if child.none_count > 0 or child.nan_count > 0 or child.pos_inf_count > 0 or child.neg_inf_count > 0
            )
            warnings.append(
                "modality fingerprint: %d of %d numeric vector field(s) have missing/non-finite values -> "
                "the multivariate-Gaussian joint/dependency route needs a fully-observed vector, so an "
                "independent per-field composite was used instead (missing-data-aware joint/dependency "
                "modeling is not implemented yet)" % (affected, missing_vec_dim)
            )
    observed_rows = [u for u in rows if u is not None]
    if use_bstats and observed_rows and all(isinstance(u, Mapping) for u in observed_rows):
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
        selection=selection,
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
            # A string that reads as a number -- "46.8037", the shape a numeric CSV column takes
            # once one dirty cell coerces the whole column to object/str dtype -- is counted as the
            # float it names rather than as the text itself. Left as a raw string, 300 genuinely
            # numeric rows all typed str were indistinguishable from an identifier column (nearly
            # every value distinct) and fit a frozen memorization table scoring every unseen value
            # -inf, the exact T4-01 failure reopened one dtype layer up. See parse_numeric_text.
            numeric_x = parse_numeric_text(x) if isinstance(x, str) else None
            key = x if numeric_x is None else numeric_x
            self._analyze_type(key)
            # Mirrors the non-finite-float skip the generic scalar branch below applies to a NATIVE
            # inf/-inf: a numeric-text overflow ("1e400" -> float('inf')) is counted via
            # pos_inf_count/neg_inf_count above, not stored as an observed vdict value.
            if not (isinstance(key, (float, np.floating)) and not math.isfinite(key)):
                self.vdict[key] += 1
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
        elif isinstance(x, Mapping):
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

    def _scalar_type_view(self, *, numeric: bool) -> "DatumNode":
        """Return a copy of this scalar leaf restricted to one side of a number/string type mix.

        The copy keeps every structural field intact and only narrows the type evidence, so
        :meth:`_leaf_estimator` run against it takes exactly the branch it would have taken had the
        column contained only those values -- identifier detection, integer-model selection, the
        BIC-scored continuous family choice, all of it. That is the point: the split must not become
        a second, parallel set of typing rules that drifts from the real ones.

        The missing/non-finite counters are cleared on both views because those wrappers are applied
        once, outside, around the whole leaf (see :meth:`get_estimator`); leaving them set here would
        wrap each branch a second time.
        """
        view = self.copy()
        view.none_count = 0
        view.nan_count = 0
        view.pos_inf_count = 0
        view.neg_inf_count = 0
        if numeric:
            view.vdict = defaultdict(
                int,
                {
                    key: value
                    for key, value in self.vdict.items()
                    if isinstance(key, numbers.Real) and not isinstance(key, (bool, np.bool_))
                },
            )
            view.str_count = 0
            view.count = self.float_count + self.int_count
        else:
            view.vdict = defaultdict(
                int, {key: value for key, value in self.vdict.items() if isinstance(key, (str, bytes))}
            )
            view.float_count = 0
            view.int_count = 0
            view.neg_count = 0
            view.zero_count = 0
            view.count = self.str_count
        return view

    def _typed_mixture_estimator(self, pseudo_count, emp_suff_stat, use_bstats, *, numeric_view=None):
        """Build the number/string dispatch mixture for a mixed scalar leaf.

        Neither branch inherits the field-level ``recommendation``: that recommendation was derived
        from the mixed column as a whole (``'ignored'`` before this repair, ``'typed_mixture'`` now)
        and says nothing about which family the numbers alone want, so each side re-derives its own
        from its own values.
        """
        numeric_view = self._scalar_type_view(numeric=True) if numeric_view is None else numeric_view
        string_view = self._scalar_type_view(numeric=False)
        return get_typed_mixture_estimator(
            string_view._leaf_estimator(pseudo_count, emp_suff_stat, use_bstats),
            numeric_view._leaf_estimator(pseudo_count, emp_suff_stat, use_bstats),
            use_bstats=use_bstats,
            string_support=tuple(string_view.vdict),
        )

    def _leaf_estimator(self, pseudo_count, emp_suff_stat, use_bstats, recommendation: str | None = None):
        if self.obj_count > 0 or len(self.vdict) == 0:
            if len(self.vdict) > 0:
                # A scalar type the profiler does not recognize -- datetime64/Timestamp being the
                # everyday case, e.g. read_csv(parse_dates=...) -- is frozen, not modeled, but the
                # frozen stand-in must still score the actual rows finitely. The bare ignored
                # estimator's point mass at None gave every real value -inf, so a table carrying a
                # timestamp column died in EM/DPM with the same useless non-finite-objective error
                # that identifier string columns used to (campaign T2-09a).
                return _get_identifier_estimator(self.vdict, use_bstats=use_bstats)
            return get_ignored_estimator(use_bstats=use_bstats)
        if self.bool_count > 0 and (self.int_count > 0 or self.float_count > 0):
            # A bool/numeric mix is ambiguous between a flag and a measurement, so it stays frozen
            # rather than guessed at -- with the same finite-scoring stand-in as above, for the same
            # reason. (vdict has already merged True with 1/1.0 -- Python hashes them equal -- and
            # the frozen empirical categorical simply inherits that.)
            return _get_identifier_estimator(self.vdict, use_bstats=use_bstats)

        if self.str_count > 0 and self.bool_count == 0 and (self.float_count > 0 or self.int_count > 0):
            # A scalar column carrying BOTH numbers and strings is two disjoint types, and until
            # 0.8.0 the branch below retyped all of it as one categorical over the observed values.
            # One stray "N/A" in 300 finite floats therefore produced a 301-level memorization table
            # that scored the sample mean -- and every other value not literally in the training set
            # -- at -inf, silently, with converged=True and repairs=(). Route by type instead: each
            # side is fit by these same leaf rules on its own values and the branch weights are the
            # observed type proportions.
            #
            # Deliberately NOT gated on which side is the MAJORITY: a stray number in a string column
            # collapses just as wrongly as a stray string in a numeric one, and the string branch's
            # density is unchanged either way (log(n_str/n) + log(count/n_str) == log(count/n),
            # exactly in real arithmetic and to within a ULP in floating point).
            # It IS gated on whether the numbers want a family with support past the observed values
            # -- see _numeric_side_is_memorizing, shared with the profile report so the two readings
            # of one column cannot diverge. ["low", 2, "high", 3] * 40 is a genuinely categorical
            # column and stays one.
            numeric_view = self._scalar_type_view(numeric=True)
            if not _numeric_side_is_memorizing(numeric_view.vdict, self.float_count > 0):
                return self._typed_mixture_estimator(pseudo_count, emp_suff_stat, use_bstats, numeric_view=numeric_view)

        if self.str_count > 0:
            # identifier-like fields (nearly all values distinct) carry no density information;
            # freeze them instead of fitting a one-bucket-per-row categorical. The frozen factor
            # must still score real rows finitely -- the bare ignored estimator's point mass at
            # None scored every actual identifier at -inf, so any EM/DPM fit over data containing
            # an ID column died with a useless non-finite-objective error.
            if self.count >= ID_MIN_COUNT and len(self.vdict) >= ID_DISTINCT_FRACTION * self.count:
                return _get_identifier_estimator(self.vdict, use_bstats=use_bstats)
            if recommendation == "ignored":
                return _get_identifier_estimator(self.vdict, use_bstats=use_bstats)
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
                    best = recommendation if recommendation in bics and recommendation in builders else None
                    if best is None:
                        best = _numeric_model_recommendation(bics)
                    return builders[best](self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            return get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        if self.int_count > 0:
            inferred, scores = _recommended_integer_model(self.vdict)
            if recommendation not in scores and recommendation != "ignored":
                recommendation = inferred
            recommendation = inferred if recommendation is None else recommendation
            if recommendation == "ignored":
                # Same repair as the string-identifier branch: an integer ID column is frozen,
                # but the frozen factor still scores actual values finitely.
                return _get_identifier_estimator(self.vdict, use_bstats=use_bstats)
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
        """Weighted ``(count, mean, variance, min, max, width)`` over the integer support.

        The moments are taken about ``min_val`` rather than about zero. ``sum(w*k*k)/W - mean**2``
        needs ``k*k`` to be representable, and an epoch timestamp squares to ~3e18 -- past the
        2**53 where float64 stops counting -- so the spread of a perfectly ordinary column came out
        wrong by thousands at second resolution and collapsed to exactly 0.0 at millisecond
        resolution. Integer differences from ``min_val`` are exact in Python before they reach
        float, and their squares stay O(range**2), so the variance is the data's own and is
        invariant under a shift of the column. ``_integer_values_look_poisson_like`` reads this
        variance to type the field, so the arithmetic is a family-selection input, not a diagnostic.
        """
        min_val = None
        max_val = None
        ss_0 = 0.0
        for k, v in self.vdict.items():
            kk = int(k)
            ss_0 += float(v)
            min_val = kk if min_val is None else min(min_val, kk)
            max_val = kk if max_val is None else max(max_val, kk)
        if ss_0 <= 0.0:
            return 0.0, 0.0, 0.0, 0, 0, 0
        ss_1 = 0.0
        ss_2 = 0.0
        for k, v in self.vdict.items():
            d = int(k) - min_val
            vv = float(v)
            ss_1 += d * vv
            ss_2 += float(d * d) * vv
        offset = ss_1 / ss_0
        mean = min_val + offset
        var = max(0.0, ss_2 / ss_0 - offset * offset)
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

    def get_estimator(
        self,
        pseudo_count: float | None = 1.0,
        emp_suff_stat: bool = True,
        use_bstats: bool = False,
        *,
        recommendations: dict[tuple[Any, ...], str] | None = None,
        path: tuple[Any, ...] = (),
        modality: str | None = None,
    ):
        """Infer and return an estimator for the profiled observations.

        **What happens to values that are not ordinary numbers.** :func:`mixle.inference.optimize`,
        the entry point most callers reach this through, summarizes this policy in its own docstring
        (see its "Identifier-like columns" and "Missing values" sections) and points back here for
        the full detail, written out below.

        * ``None`` and ``nan`` are the two spellings of ABSENT and mean one model: the leaf is
          wrapped in an :class:`~mixle.stats.combinator.optional.OptionalDistribution` whose
          missingness rate is fit from the data, so the model stays a normalized law and still
          samples. (They must agree: a pandas float Series stores ``None`` as ``nan`` behind the
          caller's back.)
        * ``+inf`` and ``-inf`` are values rather than absences, and each sign present gets its own
          fitted-rate wrapper -- an atom at the sentinel beside the base family, total mass one.
          Through 0.7.x these used the transparent (``est_prob=False``) wrapper instead, which
          scored the sentinel at ``log_density == 0.0``, probability one, on top of an unscaled base:
          total mass 2.0, and every such row free in the objective.
        * A scalar column carrying BOTH numbers and strings is modeled as a typed dispatch mixture
          rather than retyped wholesale as a categorical over the observed values -- see
          :func:`~mixle.utils.automatic.factories.get_typed_mixture_estimator`.
        * A scalar type this profiler does not recognize (``datetime``, ``Decimal``, ...), an
          ambiguous bool/number mix, and an identifier-like column are frozen: the empirical
          categorical over the observed values, held fixed by
          :class:`~mixle.stats.combinator.ignored.IgnoredDistribution`. A value not seen at
          profiling time scores ``-inf`` there, the same finite-support behavior every automatically
          fitted categorical has.
        """
        if modality not in {None, "embedding", "image"}:
            raise ValueError("modality must be None, 'embedding', or 'image'")
        if modality == "embedding" and self._fixed_numeric_vector_dim() is None:
            raise ValueError("modality='embedding' requires fixed-length, finite numeric vectors")
        if modality == "image" and self._fixed_numeric_matrix_shape() is None:
            raise ValueError("modality='image' requires homogeneous finite numeric matrices")
        recommendations = {} if recommendations is None else recommendations
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
                        self.dict_children[k].get_estimator(
                            pseudo_count,
                            emp_suff_stat,
                            use_bstats=use_bstats,
                            recommendations=recommendations,
                            path=path + ("key", k),
                        )
                        for k in keys
                    ],
                )

        elif structured > 0:
            fixed_arity = len(self.len_dict) == 1
            if self.tuple_count > 0 and self.seq_count == 0 and fixed_arity:
                # records: positional composite
                rv = get_composite_estimator(
                    [
                        u.get_estimator(
                            pseudo_count,
                            emp_suff_stat,
                            use_bstats=use_bstats,
                            recommendations=recommendations,
                            path=path + (index,),
                        )
                        for index, u in enumerate(self.children)
                    ],
                    use_bstats=use_bstats,
                )
            elif self._fixed_numeric_vector_dim() is not None:
                vec_dim = self._fixed_numeric_vector_dim()
                if modality == "image":
                    raise ValueError("modality='image' requires homogeneous finite numeric matrices")
                if modality == "embedding" and _has_torch():
                    rv = get_hybrid_embedding_estimator(vec_dim)
                else:
                    rv = get_multivariate_gaussian_estimator(vec_dim, use_bstats=use_bstats)
            elif self._fixed_numeric_matrix_shape() is not None and modality == "image" and _has_torch():
                rv = get_hybrid_image_estimator()
            elif modality == "embedding":
                raise ValueError("modality='embedding' requires fixed-length, finite numeric vectors")
            elif fixed_arity and self.tuple_count == 0 and not self._children_homogeneous():
                # fixed-length lists/vectors with positionally distinct types
                rv = get_composite_estimator(
                    [
                        u.get_estimator(
                            pseudo_count,
                            emp_suff_stat,
                            use_bstats=use_bstats,
                            recommendations=recommendations,
                            path=path + (index,),
                        )
                        for index, u in enumerate(self.children)
                    ],
                    use_bstats=use_bstats,
                )
            else:
                # variable-length (or homogeneous fixed-length) sequences
                child = self._merged_child()
                rv = get_sequence_estimator(
                    child.get_estimator(
                        pseudo_count,
                        emp_suff_stat,
                        use_bstats=use_bstats,
                        recommendations=recommendations,
                        path=path + ("element",),
                    ),
                    len_dict=self.len_dict,
                    pseudo_count=pseudo_count,
                    emp_suff_stat=emp_suff_stat,
                    use_bstats=use_bstats,
                )

        else:
            rv = self._leaf_estimator(
                pseudo_count,
                emp_suff_stat,
                use_bstats,
                recommendation=recommendations.get(path),
            )

        if self.none_count > 0:
            # Genuine optionality: the field is really absent. Fit the missingness rate so the
            # auto-built model stays generative -- the README quickstart ships a row with a
            # missing value and then calls model.sampler().sample(5).
            rv = get_optional_estimator(rv, None, use_bstats=use_bstats, est_prob=True)

        if self.nan_count > 0:
            # nan is the OTHER spelling of the same missingness -- a pandas float Series stores
            # None as nan behind the caller's back -- so it takes the same fitted-rate, generative
            # wrapper as None. With est_prob=False here, the two spellings of one dataset produced
            # different models: the nan one read .p as 0.0 and refused to sample.
            rv = get_optional_estimator(rv, math.nan, use_bstats=use_bstats, est_prob=True)

        # +-inf gets the same fitted-rate wrapper, for a reason of its own rather than by analogy.
        # Through 0.7.x these two calls passed est_prob=False, which makes the wrapper transparent:
        # the fitted model then reported .p == 0.0 ("an infinity never occurs") while simultaneously
        # scoring log_density(inf) == 0.0 (probability one), for total mass 2.0. Those rows cost
        # exactly zero nats, so 300 finite draws plus one inf produced a final_objective
        # bit-identical to the clean 300-point fit at n_observations=301, and the inflation grows
        # without bound in the number of infinities -- a free likelihood gain against any proper
        # competitor, invisible to vuong_test/clarke_test/compare_elpd because those consume plain
        # arrays and never see density_semantics(). est_prob=True is the proper model of the same
        # reading: an atom of mass p at the sentinel beside the base family scaled by 1-p, i.e. "a
        # continuous measurement that comes out infinite p of the time", which integrates to one.
        if self.pos_inf_count > 0:
            rv = get_optional_estimator(rv, math.inf, use_bstats=use_bstats, est_prob=True)

        if self.neg_inf_count > 0:
            rv = get_optional_estimator(rv, -math.inf, use_bstats=use_bstats, est_prob=True)

        return rv

    def _fixed_numeric_vector_dim(self, *, allow_missing: bool = False):
        """Dimension of a fixed-length all-numeric vector field, or ``None`` if this field does not
        qualify for the multivariate-Gaussian joint/dependency route.

        ``allow_missing=True`` skips only the missing/non-finite check, leaving every other
        eligibility rule intact -- used by :func:`analyze_structure` to tell whether missing data is
        specifically what disqualified an otherwise-eligible vector (see the "modality fingerprint:
        missing values" warning), as opposed to the field never having qualified at all (a tuple, a
        mixed-type row, etc.).
        """
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
            if not allow_missing and (
                child.none_count > 0 or child.nan_count > 0 or child.pos_inf_count > 0 or child.neg_inf_count > 0
            ):
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
    * a one-shot iterator (generator, ``map``/``filter``/``zip``, a file object) -> materialized to a
      list, because the profiler iterates the records more than once (schema detection, then fitting)
      and a one-shot iterator would be exhausted after the first pass;
    * anything else (a list / sequence) is returned unchanged.
    """
    if hasattr(data, "records") and hasattr(data, "structure"):  # a mixle DataSource
        try:
            from mixle.data.core import DataSource

            if isinstance(data, DataSource):
                data = list(data.records())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(data, "columns") and hasattr(data, "itertuples"):  # a pandas DataFrame (duck-typed)
        from mixle.data.sources.pandas_source import dataframe_records

        data = dataframe_records(data)
    try:
        from mixle.utils.optional_deps import RDD_TYPES

        if RDD_TYPES and isinstance(data, RDD_TYPES):  # a Spark RDD -> a bounded local sample
            data = data.take(int(rdd_cap))
    except Exception:  # noqa: BLE001
        pass
    # A one-shot iterator returns itself from __iter__ and is consumed on the first pass; the profiler
    # reads the records more than once (schema detection, then fitting), so an un-materialized iterator
    # would silently fit a wrong/empty model on the second pass. Materialize it to a reusable list. A
    # reusable sequence (list/tuple/ndarray) yields a fresh iterator and is left unchanged.
    try:
        if iter(data) is data:
            data = list(data)
    except TypeError:
        pass  # not iterable -- leave it to the caller's own validation
    # Both the docstring above and the comment below promise a reusable sequence is returned
    # UNCHANGED, but freezing rebuilt one unconditionally, so a plain list came back as a copy and
    # `normalize_input(xs) is xs` was false. Freezing still has to happen -- nested observations must
    # become immutable value graphs -- so rebuild only when it actually changed something. Flat
    # numeric records (the common case) freeze to themselves, so the original object is handed back.
    if isinstance(data, np.ndarray):
        return data  # numeric/structured array: nothing to freeze, and iterating it would rebox
    # A bare Series has its own dtype-family missing-value convention (NaN for numeric dtypes,
    # None for everything else -- see column_records), which the general per-value fallback below
    # does NOT apply: it normalizes pandas' sentinels one at a time but does not re-derive the
    # column's own convention, so the SAME data fits with a different missing_value depending on
    # whether the Series came from a plain or nullable-extension dtype (campaign four, T2-02, the
    # Series half). Route it through the same column-level rule a DataFrame's columns already get.
    if type(data).__name__ == "Series" and type(data).__module__.startswith("pandas"):
        from mixle.data.sources.pandas_source import column_records

        data = column_records(data)
    # pandas has two spellings of "missing" and uses them interchangeably: NaN, and the pd.NA
    # singleton that its nullable extension dtypes produce. NaN reached the numeric path correctly
    # while pd.NA -- an opaque object to numpy -- was profiled as a CATEGORICAL VALUE, so a numeric
    # column with one pd.NA fit as a categorical and scored every unseen number -inf, silently
    # (campaign three, T2-1). This is the single choke point every auto-inference container shape
    # passes through; the encode side is normalized to match in _data_records_for_encoding.
    #
    # A bare list has already lost whatever dtype its Series/column source had (the branch above
    # only catches a Series ITSELF, not `list(series)`), so `normalize_pandas_missing`'s per-value
    # fallback cannot re-derive the numeric-vs-not convention column_records applies and defaults
    # every pd.NA to None -- even for an all-float list. That fit an OptionalDistribution whose
    # missing_value=None can never again recognize a raw pd.NA as missing (pd.NA is only sentinel-
    # equivalent to NaN, not to None -- see optional._sentinel_key), so the model could not score or
    # re-encode the very pd.NA-carrying data it was just fit from (T1-02). flat_gap_marker makes the
    # same numeric-vs-not call column_records makes for a Series, from the list's own present values.
    from mixle.data.sources.pandas_source import flat_gap_marker, normalize_pandas_missing

    marker = flat_gap_marker(data)
    frozen = [_freeze_observation(normalize_pandas_missing(value, marker)) for value in data]
    if isinstance(data, (list, tuple)) and len(frozen) == len(data):
        if all(new_value is old_value for new_value, old_value in zip(frozen, data, strict=True)):
            return data
    return frozen


def _freeze_observation(value: Any) -> Any:
    """Materialize a nested observation once into a typed immutable value graph."""
    if isinstance(value, _SequenceObservation | _MappingObservation):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_observation(item) for item in value)
    if isinstance(value, Mapping):
        return _MappingObservation(tuple((key, _freeze_observation(item)) for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_observation(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return _SequenceObservation(tuple(_freeze_observation(item) for item in value))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return _SequenceObservation(tuple(_freeze_observation(item) for item in value))
    return value


@dataclass(frozen=True)
class RaggedRowDiagnosis:
    """What the arity evidence says about rows that do not all carry the same number of fields.

    ``malformed`` is the verdict: True when one arity is so dominant that the odd rows are a defect
    in a table (a lost or duplicated delimiter, a blank line), False when the majority is real but
    not overwhelming, which is the genuinely ambiguous case that gets fitted AND disclosed.
    """

    modal_width: int
    modal_count: int
    row_count: int
    row_index: int
    row_width: int
    malformed: bool

    @property
    def modal_share(self) -> float:
        """Fraction of the rows carrying the dominant arity -- the evidence the verdict rests on."""
        return self.modal_count / self.row_count

    def _evidence(self) -> str:
        # Exact counts rather than a percentage: one bad row in 2001 rounds to "100.0%", which reads
        # as though nothing were wrong in the very message that says something is.
        return "%d of %d rows have %d field(s), but row %d has %d" % (
            self.modal_count,
            self.row_count,
            self.modal_width,
            self.row_index,
            self.row_width,
        )

    def _sequence_opt_out(self) -> str:
        return (
            "if these really are variable-length sequences, ask for that reading explicitly with "
            "mixle.utils.automatic.get_estimator(data, ragged='sequence') and pass the result as the "
            "estimator"
        )

    def contract_error(self) -> ContractError:
        """The refusal for a malformed table, naming the offending row and the arity it must have."""
        return ContractError(
            "automatic structure inference (row %d)" % self.row_index,
            "a row of %d field(s)" % self.modal_width,
            "a row of %d field(s)" % self.row_width,
            "%s, so this is a table with a malformed row rather than variable-length sequence data: "
            "check row %d for a missing or extra field. Otherwise %s."
            % (self._evidence(), self.row_index, self._sequence_opt_out()),
        )

    def note(self) -> str:
        """One line naming the arity evidence, the reading taken, and the remedy for the other one."""
        if self.malformed:
            return (
                "ragged rows: %s -- read as a table with a malformed row; check row %d for a missing "
                "or extra field. Otherwise %s." % (self._evidence(), self.row_index, self._sequence_opt_out())
            )
        return (
            "ragged rows: %s -- read as variable-length sequence data (an element model plus a length "
            "model), NOT as a table with a malformed row. If it is a table, fix row %d, which has a "
            "missing or extra field; if it is sequence data, %s to make the reading explicit."
            % (self._evidence(), self.row_index, self._sequence_opt_out())
        )


def _positional_row_widths(rows: Sequence[Any]) -> list[int] | None:
    """Per-row arity when every row is a positional container, else ``None``.

    ``None`` means the arity question does not arise at all: scalars, mappings, sets, or a mix of
    shapes, none of which the profiler reads as a table of positional fields. Only sized rows are
    measured, so nothing here consumes a one-shot iterator that ``normalize_input`` left alone.
    """
    widths: list[int] = []
    for row in rows:
        if isinstance(row, tuple):
            widths.append(len(row))
        elif isinstance(row, (str, bytes)) or isinstance(row, Mapping) or isinstance(row, (set, frozenset)):
            return None
        elif isinstance(row, Iterable) and isinstance(row, Sized):
            widths.append(len(row))
        else:
            return None
    return widths


def _reads_as_positional_table(node: "DatumNode") -> bool:
    """Whether a node whose rows all share one arity takes one of the table readings.

    Mirrors the structured branch of :meth:`DatumNode.get_estimator`: a positional composite, a
    multivariate-Gaussian vector, or a keyed record. The ragged diagnosis asks this of the
    majority-arity rows ALONE, because the only raggedness worth reporting is raggedness that changed
    the reading. Data the profiler already reads as a sequence at a single arity -- homogeneous lists
    such as all-string CSV rows -- gets the same sequence model either way and stays silent.
    """
    structured = node.tuple_count + node.seq_count + node.set_count + node.dict_count
    if structured == 0 or node.count - node.none_count == 0:
        return False
    container_kinds = sum(u > 0 for u in (node.tuple_count, node.seq_count, node.set_count, node.dict_count))
    if len(node.vdict) > 0 or node.obj_count > 0 or container_kinds > 1:
        return False  # mixed scalars/containers: not modelable as anything, ragged or not
    if node.set_count > 0:
        return False
    if node.dict_count > 0:
        return True
    if len(node.len_dict) != 1:
        return False
    if node.tuple_count > 0:
        return True  # tuples are fixed-arity records
    if node._fixed_numeric_vector_dim() is not None:
        return True
    return not node._children_homogeneous()


def diagnose_ragged_rows(rows: Sequence[Any]) -> RaggedRowDiagnosis | None:
    """Weigh the arity evidence for rows of differing width, or ``None`` when there is nothing to say.

    ``None`` covers three distinct cases, all of which must stay silent: rows that are not positional
    containers, rows that all share one arity, and rows whose arities are spread widely enough that
    no single one is a majority -- the fingerprint of real variable-length data.
    """
    widths = _positional_row_widths(rows)
    if widths is None or len(widths) < 2:
        return None
    counts: dict[int, int] = defaultdict(int)
    for width in widths:
        counts[width] += 1
    if len(counts) == 1:
        return None
    # A tie cannot be a dominant arity in any case; breaking it toward the smaller width keeps the
    # answer independent of the order the widths happened to be counted in.
    modal_width, modal_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    share = modal_count / len(widths)
    if share < AMBIGUOUS_TABLE_MIN_SHARE:
        return None
    modal_rows = [row for row, width in zip(rows, widths, strict=True) if width == modal_width]
    if not _reads_as_positional_table(DatumNode(data=modal_rows)):
        return None
    row_index = next(index for index, width in enumerate(widths) if width != modal_width)
    return RaggedRowDiagnosis(
        modal_width=modal_width,
        modal_count=modal_count,
        row_count=len(widths),
        row_index=row_index,
        row_width=widths[row_index],
        malformed=share >= MALFORMED_TABLE_MIN_SHARE,
    )


def _apply_ragged_policy(rows: Sequence[Any], ragged: str) -> None:
    """Refuse a malformed table, disclose an ambiguous one, and leave sequence data alone."""
    if ragged not in {"auto", "sequence"}:
        raise ValueError("ragged must be 'auto' or 'sequence'")
    if ragged == "sequence":
        return
    diagnosis = diagnose_ragged_rows(rows)
    if diagnosis is None:
        return
    if diagnosis.malformed:
        raise diagnosis.contract_error()
    import warnings

    warnings.warn(diagnosis.note(), UserWarning, stacklevel=3)


def _identifier_like_field(node: "DatumNode") -> bool:
    """Whether a leaf's observed values trip the same identifier heuristics the estimator used."""
    if node.str_count > 0:
        return node.count >= ID_MIN_COUNT and len(node.vdict) >= ID_DISTINCT_FRACTION * node.count
    if node.int_count > 0:
        return node._integer_values_look_identifier_like()
    if node.obj_count > 0:
        # An unrecognized scalar type (a Timestamp column is the everyday case) with nearly every
        # value distinct is an identifier in the same sense a distinct-string column is.
        return node.count >= ID_MIN_COUNT and len(node.vdict) >= ID_DISTINCT_FRACTION * node.count
    return False


def _field_line(node: "DatumNode", label: str) -> str:
    """One evidence line for a field nothing could be fitted to: kind, cardinality, and why."""
    if node.str_count > 0:
        kind = "string"
    elif node.int_count > 0:
        kind = "integer"
    elif node.float_count > 0:
        kind = "float"
    elif node.obj_count > 0 and node.vdict:
        # Name the concrete type, so a datetime column reads "Timestamp field" rather than the
        # unhelpfully generic "unmodelable field".
        kind = type(next(iter(node.vdict))).__name__
    else:
        kind = "unmodelable"
    why = (
        "identifier-like: nearly every value is distinct"
        if _identifier_like_field(node)
        else "not modelable by automatic profiling"
    )
    return "%s is a %s field with %d distinct value(s) in %d observation(s) (%s)" % (
        label,
        kind,
        len(node.vdict),
        node.count,
        why,
    )


def _unmodelable_fields_report(node: "DatumNode") -> list[str]:
    """Per-field evidence lines for data in which automatic typing found nothing to fit.

    Called by :func:`~mixle.utils.automatic.factories.get_dpm_mixture` when every field resolved to
    a frozen (Ignored) factor, so the refusal can name each column, its cardinality, and the reason
    instead of failing with an unactionable global error.
    """
    if node.children:
        return [_field_line(child, "field %d" % index) for index, child in enumerate(node.children)]
    if node.dict_children:
        return [
            _field_line(child, "field %r" % (key,))
            for key, child in sorted(node.dict_children.items(), key=lambda item: repr(item[0]))
        ]
    return [_field_line(node, "the data")]


def _nonfinite_leaf_paths(node: "DatumNode", path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], int, int]]:
    """Depth-first ``(path, pos_inf_count, neg_inf_count)`` for every leaf carrying a native +-inf.

    A structural node (tuple/sequence/set/dict) never accumulates its own ``pos_inf_count`` /
    ``neg_inf_count`` -- only the scalar leaves that went through ``_analyze_type`` do -- so walking
    every node and checking its own counters (rather than special-casing "is this a leaf") finds
    exactly the fields :meth:`DatumNode.get_estimator` is about to wrap in a missingness sentinel.
    """
    found = []
    if node.pos_inf_count > 0 or node.neg_inf_count > 0:
        found.append((path, node.pos_inf_count, node.neg_inf_count))
    for index, child in enumerate(node.children):
        found.extend(_nonfinite_leaf_paths(child, path + (index,)))
    for key, child in node.dict_children.items():
        found.extend(_nonfinite_leaf_paths(child, path + ("key", key)))
    return found


def _warn_nonfinite_reclassified_as_missing(root: "DatumNode") -> None:
    """Disclose that auto-inference is about to fit a field's native +-inf values as missingness.

    :meth:`DatumNode.get_estimator` wraps any leaf carrying ``+inf``/``-inf`` in an
    :class:`~mixle.stats.combinator.optional.OptionalDistribution` with a fitted rate -- the same
    generative-missingness treatment ``None``/``NaN`` get, for the numerical-stability reason
    explained on that method's own docstring. But that docstring is reached only by reading the
    profiler; :func:`mixle.inference.optimize`'s "Missing values" section -- the documented contract
    most callers actually read -- lists only ``None``/``NaN``/``pd.NA``/``pd.NaT`` as the sentinels
    auto-inference (``estimator=None``) treats as gaps, and :doc:`/stability-and-missing-data` states
    that default fitting routes reject non-finite observations rather than silently reclassifying
    them -- which an *explicit* estimator does (``GaussianDistribution requires support x in
    (-inf,inf)``). Auto-inference's own reclassification must therefore be disclosed rather than
    silent, matching the convention already used for an ambiguous ragged table
    (:func:`_apply_ragged_policy`) and for missing data disqualifying the joint/dependency vector
    route (the "modality fingerprint" warning in :func:`analyze_structure`).
    """
    found = _nonfinite_leaf_paths(root)
    if not found:
        return
    detail = "; ".join("%s (%d +inf, %d -inf)" % (format_path(path), pos, neg) for path, pos, neg in found)
    import warnings

    warnings.warn(
        "auto-inference (estimator=None) found +inf/-inf value(s) in %s and is fitting them as a "
        "missingness indicator (OptionalDistribution(..., missing_value=+/-inf) with a fitted rate) "
        "rather than rejecting them -- deliberate (see DatumNode.get_estimator's docstring), but not "
        "the None/NaN/pd.NA/pd.NaT missingness documented for optimize()/fit(). Pass an explicit "
        "estimator (e.g. GaussianEstimator()) to reject non-finite values instead, or build the "
        "wrapper yourself (OptionalEstimator(..., missing_value=math.inf)) to make the reclassification "
        "explicit in your own code." % detail,
        UserWarning,
        stacklevel=3,
    )


def get_estimator(
    data,
    pseudo_count: float | None = 1.0,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
    *,
    modality: str | None = None,
    ragged: str = "auto",
):
    """Profile ``data`` and return the automatically selected estimator.

    ``ragged`` governs rows that do not all carry the same number of fields, a shape that is
    genuinely ambiguous between a table with a malformed row and variable-length sequence data.
    ``'auto'`` (the default) decides on the arity evidence: an overwhelmingly dominant arity means a
    malformed table and raises :class:`~mixle.stats.compute.pdist.ContractError` naming the offending
    row, a bare majority is fitted as a sequence with a warning saying so, and widely spread arities
    are fitted as a sequence silently. ``'sequence'`` takes the sequence reading unconditionally and
    without comment -- the escape hatch for variable-length data whose lengths happen to be nearly
    constant.

    A field carrying a native ``+inf``/``-inf`` is fitted, not rejected -- wrapped as a missingness
    sentinel with its own fitted rate, same as ``None``/``NaN`` -- and a ``UserWarning`` discloses it
    (see :func:`_warn_nonfinite_reclassified_as_missing`); pass an explicit estimator instead to get
    the reject-on-non-finite behavior documented for the family-level default route.

    An unrecognized scalar type, an ambiguous bool/number mix, and an identifier-like/high-
    cardinality field are all frozen instead: the empirical categorical over the values observed
    here, held fixed by :class:`~mixle.stats.combinator.ignored.IgnoredDistribution`. This is
    silent (no warning) because it is the correct, unsurprising answer for a field of this shape --
    but a value not seen here later scores ``-inf`` under it (``CategoricalDistribution``'s
    documented ``default_value``), which is worth knowing before scoring held-out or production
    data through the result; see :func:`DatumNode.get_estimator`'s docstring for the full policy and
    :func:`mixle.inference.optimize`'s docstring for what this means for a fitted model's score.
    """
    rows = normalize_input(data)
    _apply_ragged_policy(rows, ragged)
    root = DatumNode(data=rows)
    _warn_nonfinite_reclassified_as_missing(root)
    return root.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats, modality=modality)


def get_prototype(
    data,
    *,
    seed: int | None = None,
    p: float = 0.1,
    pseudo_count: float | None = 1.0,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
    modality: str | None = None,
    ragged: str = "auto",
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
    est = get_estimator(rows, pseudo_count, emp_suff_stat, use_bstats=use_bstats, modality=modality, ragged=ragged)
    enc = seq_encode(rows, estimator=est)
    return seq_initialize(enc_data=enc, estimator=est, rng=np.random.RandomState(seed), p=p)
