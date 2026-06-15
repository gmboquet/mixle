"""Automatic detection of data type for estimators.

Builds estimators for pysp.stats. By default the plain maximum-likelihood
estimators are produced; pass use_bstats=True to build the Bayesian path, which
attaches the conjugate default prior for each family so estimation performs the
closed-form conjugate / MAP update. get_dpm_mixture fits a Dirichlet process
mixture over automatically-typed data with variational inference.
"""

import math
import numbers
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

from pysp.stats.pdist import (
    ParameterEstimator,
)

T = TypeVar("T")

# Leaf-typing heuristics: integers with at most this many distinct values (or
# at most this fraction of observations) are modeled as categorical rather
# than Poisson/Gaussian; string fields where nearly every value is unique are
# treated as identifiers and ignored.
MAX_INT_CATEGORICAL_DISTINCT = 20
MAX_INT_CATEGORICAL_FRACTION = 0.05
MAX_INT_CATEGORICAL_RANGE_MULTIPLIER = 4.0
MAX_LENGTH_CATEGORICAL_DISTINCT = 25
MAX_LENGTH_CATEGORICAL_FRACTION = 0.20
INT_ID_RANGE_MULTIPLIER = 20.0
POISSON_DISPERSION_MIN = 0.25
POISSON_DISPERSION_MAX = 4.0
ID_DISTINCT_FRACTION = 0.95
ID_MIN_COUNT = 100
AMBIGUOUS_SCORE_GAP_BITS = 0.05
VALIDATION_ALPHA = 0.5
VALIDATION_VARIANCE_FLOOR = 1.0e-12


def _estimator_provider(use_bstats: bool = False):
    # Both the plain (MLE) and Bayesian (conjugate-prior) paths build pysp.stats
    # estimators now; ``use_bstats`` only selects whether a conjugate default
    # prior is attached (see the get_* helpers below). The parameter name is kept
    # for backwards compatibility -- it now means "build the Bayesian path".
    import pysp.stats as provider

    return provider


# Conjugate default priors, one per family. Each is attached when use_bstats=True
# so the stats estimator runs its closed-form conjugate / MAP update during
# estimation. Hyperparameters:
#   gaussian:     NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)
#   categorical:  DictDirichletDistribution(1.0 + 1.0e-12)
#   int_range:    SymmetricDirichletDistribution(1.0 + 1.0e-12)  (scalar symmetric)
#   poisson:      GammaDistribution(1.0001, 1.0e6)
#   exponential:  GammaDistribution(1.0001, 1.0e6)
#   setdist:      BetaDistribution(1, 1)
#   mvn:          NormalWishart(zeros(d), 1e-8, eye(d)*0.5, d + 2e-6)
_BAYES_DIRICHLET_ALPHA = 1.0 + 1.0e-12


# The conjugate prior families are reached through the ``pysp.stats`` package
# namespace (an allowed high-level dependency) rather than importing concrete
# distribution submodules here, keeping this builder free of concrete-class
# imports (see compute_metadata_test's import-hygiene guard).
def _gaussian_default_prior():
    import pysp.stats as provider

    return provider.NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)


def _categorical_default_prior(vdict):
    import pysp.stats as provider

    return provider.DictDirichletDistribution(_BAYES_DIRICHLET_ALPHA)


def _integer_categorical_default_prior():
    # The stats DirichletDistribution requires an explicit alpha vector, so the
    # symmetric scalar prior is the SymmetricDirichletDistribution, which the
    # IntegerCategorical conjugate path accepts and treats identically to a
    # scalar Dirichlet.
    import pysp.stats as provider

    return provider.SymmetricDirichletDistribution(_BAYES_DIRICHLET_ALPHA)


def _poisson_default_prior():
    import pysp.stats as provider

    return provider.GammaDistribution(1.0001, 1.0e6)


def _exponential_default_prior():
    import pysp.stats as provider

    return provider.GammaDistribution(1.0001, 1.0e6)


def _set_default_prior():
    import pysp.stats as provider

    return provider.BetaDistribution(1.0, 1.0)


def _mvn_default_prior(dim: int):
    import pysp.stats as provider

    # d-dimensional analogue of NormalGamma(0, 1e-8, 0.500001, 1.0):
    # nu = 2a + (d-1), W = (2b)^-1 * I
    return provider.NormalWishartDistribution(np.zeros(dim), 1.0e-8, np.eye(dim) * 0.5, dim + 2.0e-6)


DictRecordDistribution = _estimator_provider(False).DictRecordDistribution
DictRecordEstimator = _estimator_provider(False).DictRecordEstimator


def get_optional_estimator(est: ParameterEstimator, missing_value: Any | None = None, use_bstats: bool = False):
    return _estimator_provider(use_bstats).OptionalEstimator(est, missing_value=missing_value)


def get_length_estimator(
    len_dict: dict[int, int], pseudo_count: float | None = None, emp_suff_stat: bool = True, use_bstats: bool = False
) -> "ParameterEstimator":
    """Length model for sequences.

    Observed lengths are often bounded protocol/domain facts, not Poisson counts,
    so use an integer categorical model while the support is small. Fall back to a
    Poisson only when length support is broad enough to look count-like.
    """
    n = sum(len_dict.values())
    cutoff = max(MAX_LENGTH_CATEGORICAL_DISTINCT, MAX_LENGTH_CATEGORICAL_FRACTION * n)
    if len(len_dict) <= cutoff and _dense_integer_support(len_dict):
        return get_integer_categorical_estimator(dict(len_dict), pseudo_count, emp_suff_stat, use_bstats=use_bstats)
    return get_poisson_estimator(dict(len_dict), pseudo_count, emp_suff_stat, use_bstats=use_bstats)


def get_sequence_estimator(
    est: ParameterEstimator,
    len_dict: dict[int, int] | None = None,
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":
    len_est = None
    if len_dict:
        len_est = get_length_estimator(len_dict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
    SequenceEstimator = _estimator_provider(use_bstats).SequenceEstimator
    return SequenceEstimator(est) if len_est is None else SequenceEstimator(est, len_estimator=len_est)


def get_set_estimator(
    member_dict: dict[Any, int],
    num_sets: int,
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":
    """Bernoulli set model with membership probabilities from observed sets."""
    BernoulliSetEstimator = _estimator_provider(use_bstats).BernoulliSetEstimator
    if use_bstats:
        return BernoulliSetEstimator(prior=_set_default_prior())
    suff_stat = None
    if emp_suff_stat and num_sets > 0:
        suff_stat = {k: v / num_sets for k, v in member_dict.items()}
    return BernoulliSetEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat)


def get_ignored_estimator(use_bstats: bool = False) -> "ParameterEstimator":
    return _estimator_provider(use_bstats).IgnoredEstimator()


def get_composite_estimator(ests: Sequence[ParameterEstimator], use_bstats: bool = False) -> "ParameterEstimator":
    return _estimator_provider(use_bstats).CompositeEstimator(ests)


def get_dict_record_estimator(keys: Sequence[Any], ests: Sequence[ParameterEstimator]) -> "ParameterEstimator":
    return DictRecordEstimator(keys, ests)


def get_categorical_estimator(
    vdict: dict[T, float], pseudo_count: float | None = None, emp_suff_stat: bool = True, use_bstats: bool = False
) -> "ParameterEstimator":
    provider = _estimator_provider(use_bstats)
    if use_bstats:
        return provider.CategoricalEstimator(prior=_categorical_default_prior(vdict))

    if emp_suff_stat:
        cnt = sum(vdict.values())
        suff_stat = {k: v / cnt for k, v in vdict.items()}
    else:
        suff_stat = None

    return provider.CategoricalEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat)


def _integer_range(vdict: dict[Any, float]):
    vals = [int(k) for k in vdict.keys()]
    min_val = min(vals)
    max_val = max(vals)
    return min_val, max_val, max_val - min_val + 1


def _dense_integer_support(vdict: dict[Any, float]) -> bool:
    if len(vdict) == 0:
        return False
    _, _, width = _integer_range(vdict)
    return width <= max(MAX_INT_CATEGORICAL_DISTINCT, int(math.ceil(MAX_INT_CATEGORICAL_RANGE_MULTIPLIER * len(vdict))))


def get_integer_categorical_estimator(
    vdict: dict[int, float], pseudo_count: float | None = None, emp_suff_stat: bool = True, use_bstats: bool = False
) -> "ParameterEstimator":
    min_val, max_val, width = _integer_range(vdict)

    if use_bstats:
        return _estimator_provider(True).IntegerCategoricalEstimator(
            min_val=min_val, max_val=max_val, prior=_integer_categorical_default_prior()
        )

    suff_stat = None
    if emp_suff_stat:
        cnt = float(sum(vdict.values()))
        p_vec = np.zeros(width, dtype=float)
        if cnt > 0.0:
            for k, v in vdict.items():
                p_vec[int(k) - min_val] = float(v) / cnt
        suff_stat = (min_val, p_vec)

    return _estimator_provider(False).IntegerCategoricalEstimator(
        min_val=min_val, max_val=max_val, pseudo_count=pseudo_count, suff_stat=suff_stat
    )


def get_poisson_estimator(
    vdict: dict[int, float], pseudo_count: float | None = None, emp_suff_stat: bool = True, use_bstats: bool = False
) -> "ParameterEstimator":

    if use_bstats:
        return _estimator_provider(True).PoissonEstimator(prior=_poisson_default_prior())

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0

        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k * v

        ss_1 = ss_1 / ss_0

    elif pseudo_count is not None:
        ss_1 = 1.0

    else:
        ss_1 = None

    return _estimator_provider(False).PoissonEstimator(pseudo_count=pseudo_count, suff_stat=ss_1)


def get_gaussian_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0
        ss_2 = 0.0
        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k * v
                ss_2 += k * k * v
        ss_1 = ss_1 / ss_0
        ss_2 = (ss_2 / ss_0) - ss_1 * ss_1

    elif pseudo_count is not None:
        ss_1 = 1.0e-6
        ss_2 = 1.0e-6
    else:
        ss_1 = None
        ss_2 = None

    if use_bstats:
        return _estimator_provider(True).GaussianEstimator(prior=_gaussian_default_prior())

    return _estimator_provider(False).GaussianEstimator(
        pseudo_count=(pseudo_count, pseudo_count), suff_stat=(ss_1, ss_2)
    )


def get_multivariate_gaussian_estimator(dim: int, use_bstats: bool = False) -> "ParameterEstimator":
    if use_bstats:
        return _estimator_provider(True).MultivariateGaussianEstimator(dim=dim, prior=_mvn_default_prior(dim))
    return _estimator_provider(False).MultivariateGaussianEstimator(dim=dim)


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
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "path": format_path(self.path),
            "role": self.role,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_fraction": self.missing_fraction,
            "observed_count": self.observed_count,
            "kind": self.kind,
            "recommendation": self.recommendation,
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
            "model_score_gap_bits": self.model_score_gap_bits,
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
        return self.estimator

    def summary(self) -> dict[str, Any]:
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
        lines = []
        for field_profile in self.fields:
            bits = "" if field_profile.bits_per_obs is None else " (~%.3f bits/obs)" % field_profile.bits_per_obs
            lines.append(
                "%s: %s -> %s%s"
                % (format_path(field_profile.path), field_profile.kind, field_profile.recommendation, bits)
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


def _clean_scores(scores: dict[str, float | None]) -> dict[str, float]:
    return {k: float(v) for k, v in scores.items() if v is not None and math.isfinite(float(v))}


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
        scores["gaussian"] = _validation_gaussian_bits(
            [float(value) for value in train], [float(value) for value in validation]
        )

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
        model_scores = _clean_scores({"gaussian": _gaussian_bic_bits(var, observed_count)})
        return MarginalFieldProfile(
            path,
            role,
            count,
            missing,
            missing_fraction,
            observed_count,
            "numeric",
            "gaussian",
            bits_per_obs=_gaussian_bits(var),
            entropy_bits=entropy,
            cardinality=cardinality,
            unique_fraction=unique_fraction,
            effective_cardinality=effective_cardinality,
            is_constant=is_constant,
            top_mass=top_mass,
            numeric_mean=mean,
            numeric_var=var,
            model_scores_bits=model_scores,
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
    Pairwise hints are deliberately unconditional and encoded through cheap
    empirical/quantile codes. They are useful evidence, not proof of topology:
    latent classes or states can explain the same bit gains.
    Marginal validation is a bounded deterministic train/validation split over
    scalar fields, meant as a cheap predictive sanity check on the BIC choice.
    """
    rows = list(data)
    total_rows = len(rows)
    estimator = get_estimator(rows, pseudo_count=pseudo_count, emp_suff_stat=emp_suff_stat, use_bstats=use_bstats)
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
        self.inf_count = 0
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
        for xx in x:
            self.add_datum(xx)

    def add_datum(self, x):
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
        "inf_count",
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
                self.inf_count += v
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
        child = self.children[0].copy()
        for u in self.children[1:]:
            child = child.merge(u)
        return child

    def get_estimator(self, pseudo_count: float | None = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
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
                rv = get_multivariate_gaussian_estimator(self._fixed_numeric_vector_dim(), use_bstats=use_bstats)
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

        return rv

    def _fixed_numeric_vector_dim(self):
        if self.tuple_count > 0 or self.seq_count == 0 or self.set_count > 0 or len(self.len_dict) != 1:
            return None
        dim = next(iter(self.len_dict))
        if dim <= 1 or len(self.children) != dim:
            return None
        for child in self.children:
            if child.count != self.seq_count:
                return None
            if child.none_count > 0 or child.nan_count > 0 or child.inf_count > 0:
                return None
            if child.str_count > 0 or child.bool_count > 0 or child.obj_count > 0:
                return None
            if child.tuple_count > 0 or child.seq_count > 0 or child.set_count > 0 or child.dict_count > 0:
                return None
            if child.int_count + child.float_count == 0:
                return None
        return dim

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


def get_estimator(data, pseudo_count: float | None = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
    return DatumNode(data=data).get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats)


def get_dpm_mixture(
    data,
    rng=None,
    max_components: int = 20,
    max_its: int = 100,
    delta: float = 1.0e-6,
    pseudo_count: float | None = 1.0,
    print_iter: int = 1,
    out=None,
):
    """Fit a Dirichlet process mixture to automatically-typed data.

    Component estimators are constructed with get_estimator(use_bstats=True)
    (one independent conjugate-prior instance per stick), and the truncated
    stick-breaking posterior is fit with variational inference via
    pysp.utils.estimation.fit.

    Args:
        data: Sequence of observations of any auto-detectable type.
        rng (Optional[RandomState]): Source of randomness for initialization.
        max_components (int): Truncation level of the stick-breaking representation.
        max_its (int): Maximum number of variational iterations.
        delta (float): Stop when the ELBO improves by less than delta.
        pseudo_count (Optional[float]): Prior strength for the component priors.
        print_iter (int): Progress print frequency.
        out: Output stream for iteration logging (defaults to sys.stdout).

    Returns:
        DirichletProcessMixtureDistribution fit to the data.
    """
    import sys

    import pysp.stats as provider
    from pysp.utils.estimation import fit

    if rng is None:
        rng = np.random.RandomState()
    if out is None:
        out = sys.stdout

    comp_ests = [get_estimator(data, pseudo_count=pseudo_count, use_bstats=True) for _ in range(max_components)]
    est = provider.DirichletProcessMixtureEstimator(comp_ests)

    return fit(data, est, max_its=max_its, delta=delta, rng=rng, print_iter=print_iter, out=out)
