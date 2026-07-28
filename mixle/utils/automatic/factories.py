"""Estimator builders for automatically-typed data.

Builds estimators for mixle.stats. By default the plain maximum-likelihood
estimators are produced; pass use_bstats=True to build the Bayesian path, which
attaches the conjugate default prior for each family so estimation performs the
closed-form conjugate / MAP update.
"""

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, TypeVar

import numpy as np

from mixle.stats.compute.pdist import (
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
    # Both the plain (MLE) and Bayesian (conjugate-prior) paths build mixle.stats
    # estimators now; ``use_bstats`` only selects whether a conjugate default
    # prior is attached (see the get_* helpers below). The parameter name is kept
    # for backwards compatibility -- it now means "build the Bayesian path".
    import mixle.stats as provider

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


def _validate_pseudo_count(pseudo_count: float | None) -> None:
    if pseudo_count is None:
        return
    if isinstance(pseudo_count, bool) or not isinstance(pseudo_count, Real):
        raise TypeError("pseudo_count must be a finite non-negative real number or None")
    if not math.isfinite(float(pseudo_count)) or float(pseudo_count) < 0.0:
        raise ValueError("pseudo_count must be a finite non-negative real number or None")


def _validate_mass_map(
    values: dict[Any, float],
    *,
    name: str,
    require_positive_total: bool = False,
) -> float:
    if not isinstance(values, dict):
        raise TypeError(f"{name} must be a dictionary of observations to masses")
    total = 0.0
    for mass in values.values():
        if isinstance(mass, bool) or not isinstance(mass, Real):
            raise TypeError(f"{name} masses must be finite non-negative real numbers")
        mass = float(mass)
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError(f"{name} masses must be finite non-negative real numbers")
        total += mass
    if not math.isfinite(total):
        raise ValueError(f"{name} total mass must be finite")
    if require_positive_total and total <= 0.0:
        raise ValueError(f"{name} must contain positive total mass")
    return total


def _validate_positive_int(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _expanded_weighted_values(values: dict[Any, float], cap: int = 200_000) -> np.ndarray:
    """Materialize count-like weighted observations for initial-fit helpers."""
    keys: list[float] = []
    counts: list[int] = []
    for key, mass in values.items():
        if isinstance(key, Real) and not isinstance(key, bool) and math.isfinite(float(key)):
            count = int(round(float(mass)))
            if count > 0:
                keys.append(float(key))
                counts.append(count)
    if not keys:
        return np.empty(0, dtype=float)
    total = sum(counts)
    if total > cap:
        scale = cap / total
        counts = [max(1, int(round(count * scale))) for count in counts]
    return np.repeat(np.asarray(keys, dtype=float), counts)


# The conjugate prior families are reached through the ``mixle.stats`` package
# namespace (an allowed high-level dependency) rather than importing concrete
# distribution submodules here, keeping this builder free of concrete-class
# imports (see compute_metadata_test's import-hygiene guard).
def _gaussian_default_prior():
    import mixle.stats as provider

    return provider.NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)


def _categorical_default_prior(vdict):
    import mixle.stats as provider

    return provider.DictDirichletDistribution(_BAYES_DIRICHLET_ALPHA)


def _integer_categorical_default_prior():
    # The stats DirichletDistribution requires an explicit alpha vector, so the
    # symmetric scalar prior is the SymmetricDirichletDistribution, which the
    # IntegerCategorical conjugate path accepts and treats identically to a
    # scalar Dirichlet.
    import mixle.stats as provider

    return provider.SymmetricDirichletDistribution(_BAYES_DIRICHLET_ALPHA)


def _poisson_default_prior():
    import mixle.stats as provider

    return provider.GammaDistribution(1.0001, 1.0e6)


def _exponential_default_prior():
    import mixle.stats as provider

    return provider.GammaDistribution(1.0001, 1.0e6)


def _set_default_prior():
    import mixle.stats as provider

    return provider.BetaDistribution(1.0, 1.0)


def _mvn_default_prior(dim: int):
    import mixle.stats as provider

    # d-dimensional analogue of NormalGamma(0, 1e-8, 0.500001, 1.0):
    # nu = 2a + (d-1), W = (2b)^-1 * I
    return provider.NormalWishartDistribution(np.zeros(dim), 1.0e-8, np.eye(dim) * 0.5, dim + 2.0e-6)


DictRecordDistribution = _estimator_provider(False).DictRecordDistribution
DictRecordEstimator = _estimator_provider(False).DictRecordEstimator


def get_optional_estimator(est: ParameterEstimator, missing_value: Any | None = None, use_bstats: bool = False):
    """Wrap an estimator with an optional/missing-value model.

    ``est_prob=True`` because this is the automatic path: the caller handed over data that
    happens to contain missing values and asked for a model of it, so how often a value is
    missing is one of the things being learned. Leaving it off yields ``p=None``, which is a
    marginalized likelihood factor -- correct when conditioning on a known missingness pattern,
    but it cannot generate, so the fitted model scores and then raises on sampler().
    """
    return _estimator_provider(use_bstats).OptionalEstimator(est, missing_value=missing_value, est_prob=True)


def get_length_estimator(
    len_dict: dict[int, int], pseudo_count: float | None = None, emp_suff_stat: bool = True, use_bstats: bool = False
) -> "ParameterEstimator":
    """Length model for sequences.

    Observed lengths are often bounded protocol/domain facts, not Poisson counts,
    so use an integer categorical model while the support is small. Fall back to a
    Poisson only when length support is broad enough to look count-like.
    """
    _validate_pseudo_count(pseudo_count)
    n = _validate_mass_map(len_dict, name="len_dict")
    if any(isinstance(k, bool) or not isinstance(k, Integral) or int(k) < 0 for k in len_dict):
        raise ValueError("sequence lengths must be non-negative integers")
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
    """Return a sequence estimator with an optional empirical length model."""
    _validate_pseudo_count(pseudo_count)
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
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(member_dict, name="member_dict")
    if isinstance(num_sets, bool) or not isinstance(num_sets, Integral):
        raise TypeError("num_sets must be a non-negative integer")
    num_sets = int(num_sets)
    if num_sets < 0:
        raise ValueError("num_sets must be a non-negative integer")
    if any(float(count) > num_sets for count in member_dict.values()):
        raise ValueError("set-member counts cannot exceed num_sets")
    if member_dict and num_sets == 0:
        raise ValueError("non-empty member_dict requires num_sets > 0")
    BernoulliSetEstimator = _estimator_provider(use_bstats).BernoulliSetEstimator
    if use_bstats:
        return BernoulliSetEstimator(prior=_set_default_prior())
    suff_stat = None
    if emp_suff_stat and num_sets > 0:
        suff_stat = {k: v / num_sets for k, v in member_dict.items()}
    return BernoulliSetEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat)


def get_ignored_estimator(use_bstats: bool = False) -> "ParameterEstimator":
    """Return the estimator used for ignored or non-modelable fields."""
    return _estimator_provider(use_bstats).IgnoredEstimator()


def get_composite_estimator(ests: Sequence[ParameterEstimator], use_bstats: bool = False) -> "ParameterEstimator":
    """Return a composite estimator over an ordered list of field estimators."""
    return _estimator_provider(use_bstats).CompositeEstimator(ests)


def get_dict_record_estimator(keys: Sequence[Any], ests: Sequence[ParameterEstimator]) -> "ParameterEstimator":
    """Return a record estimator keyed by dictionary field names."""
    return DictRecordEstimator(keys, ests)


def get_categorical_estimator(
    vdict: dict[T, float], pseudo_count: float | None = None, emp_suff_stat: bool = True, use_bstats: bool = False
) -> "ParameterEstimator":
    """Return a categorical estimator from observed value counts."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict", require_positive_total=True)
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
    """Return an integer-categorical estimator over the observed dense support."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict", require_positive_total=True)
    if any(isinstance(k, bool) or not isinstance(k, Integral) for k in vdict):
        raise ValueError("integer-categorical observations must be integers")
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
    """Return a Poisson count estimator from empirical integer counts."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict")
    if any(isinstance(k, bool) or not isinstance(k, Integral) or int(k) < 0 for k in vdict):
        raise ValueError("Poisson observations must be non-negative integers")

    if use_bstats:
        return _estimator_provider(True).PoissonEstimator(prior=_poisson_default_prior())

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0

        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k * v

        # ss_0 is 0 when vdict is empty or every key was filtered out (non-finite) -- no data to
        # estimate a mean from, so fall back the same way the emp_suff_stat=False branch does below,
        # rather than dividing by zero.
        ss_1 = ss_1 / ss_0 if ss_0 > 0.0 else (1.0 if pseudo_count is not None else None)

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
    """Return a univariate Gaussian estimator from weighted numeric values."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict")

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0
        ss_2 = 0.0
        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k * v
                ss_2 += k * k * v
        # ss_0 is 0 when vdict is empty or every key was non-finite -- no data to estimate mean/variance
        # from, so fall back the same way the emp_suff_stat=False branch does below.
        if ss_0 > 0.0:
            ss_1 = ss_1 / ss_0
            ss_2 = (ss_2 / ss_0) - ss_1 * ss_1
            # A constant field has exactly zero empirical spread, and the sum-of-squares form can
            # cancel to a small negative value even when it does not. Either way there is no scale to
            # seed a prior from, and the estimator is right to refuse a non-positive prior variance --
            # profiling a constant column crashed there rather than reporting the column as constant.
            # Fall back the same way the no-data branch below does, keeping the prior pair coherent
            # instead of handing over a degenerate one.
            if not math.isfinite(ss_2) or ss_2 <= 0.0:
                ss_1, ss_2 = (1.0e-6, 1.0e-6) if pseudo_count is not None else (None, None)
        elif pseudo_count is not None:
            ss_1, ss_2 = 1.0e-6, 1.0e-6
        else:
            ss_1, ss_2 = None, None

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


def get_lognormal_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":
    """Return a LogGaussian (log-normal) estimator fit to the log of strictly-positive values."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict")
    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0
        ss_2 = 0.0
        for k, v in vdict.items():
            if math.isfinite(k) and k > 0.0:
                lk = math.log(k)
                ss_0 += v
                ss_1 += lk * v
                ss_2 += lk * lk * v
        # ss_0 is 0 when vdict is empty or every key was non-positive/non-finite (log-normal needs
        # strictly positive values) -- no data to estimate mean/variance from, fall back like the
        # emp_suff_stat=False branch does below rather than dividing by zero.
        if ss_0 > 0.0:
            ss_1 = ss_1 / ss_0
            ss_2 = (ss_2 / ss_0) - ss_1 * ss_1
            # A constant field has exactly zero empirical spread, and the sum-of-squares form can
            # cancel to a small negative value even when it does not. Either way there is no scale to
            # seed a prior from, and the estimator is right to refuse a non-positive prior variance --
            # profiling a constant column crashed there rather than reporting the column as constant.
            # Fall back the same way the no-data branch below does, keeping the prior pair coherent
            # instead of handing over a degenerate one.
            if not math.isfinite(ss_2) or ss_2 <= 0.0:
                ss_1, ss_2 = (1.0e-6, 1.0e-6) if pseudo_count is not None else (None, None)
        elif pseudo_count is not None:
            ss_1, ss_2 = 1.0e-6, 1.0e-6
        else:
            ss_1, ss_2 = None, None
    elif pseudo_count is not None:
        ss_1 = 1.0e-6
        ss_2 = 1.0e-6
    else:
        ss_1 = None
        ss_2 = None

    if use_bstats:
        return _estimator_provider(True).LogGaussianEstimator(prior=_gaussian_default_prior())

    return _estimator_provider(False).LogGaussianEstimator(
        pseudo_count=(pseudo_count, pseudo_count), suff_stat=(ss_1, ss_2)
    )


def get_gamma_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":
    """Return a Gamma estimator seeded with empirical mean/log-mean statistics."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict")
    if use_bstats:
        raise NotImplementedError("the Gamma factory has no conjugate prior for jointly unknown shape and scale")
    mean = 1.0
    mean_log = 0.0
    if emp_suff_stat:
        total = 0.0
        value_sum = 0.0
        log_sum = 0.0
        for key, v in vdict.items():
            if math.isfinite(key) and key > 0.0:
                total += v
                value_sum += key * v
                log_sum += math.log(key) * v
        if total > 0.0:
            mean = value_sum / total
            mean_log = log_sum / total
    return _estimator_provider(False).GammaEstimator(
        pseudo_count=(pseudo_count, pseudo_count),
        suff_stat=(mean, mean_log),
    )


def get_student_t_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":
    """Return a fixed-df Student-t estimator initialized from a likelihood fit."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict")
    if use_bstats:
        raise NotImplementedError("the Student-t factory has no conjugate prior for its fitted parameters")
    df, loc, scale = 5.0, 0.0, 1.0
    if emp_suff_stat:
        values = _expanded_weighted_values(vdict)
        if values.size >= 4 and float(values.var()) > 0.0:
            from scipy import stats

            try:
                fit_df, fit_loc, fit_scale = stats.t.fit(values)
            except (FloatingPointError, RuntimeError, ValueError):
                pass
            else:
                if (
                    math.isfinite(float(fit_df))
                    and math.isfinite(float(fit_loc))
                    and math.isfinite(float(fit_scale))
                    and fit_df > 0.0
                    and fit_scale > 0.0
                ):
                    df, loc, scale = float(fit_df), float(fit_loc), float(fit_scale)
    return _estimator_provider(False).StudentTEstimator(
        df=df,
        pseudo_count=pseudo_count,
        suff_stat=(loc, scale) if pseudo_count is not None else None,
    )


def get_gaussian_mixture_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
    n_components: int = 2,
) -> "ParameterEstimator":
    """Return a K-component Gaussian mixture estimator (robust init) for multimodal numeric data."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict")
    n_components = _validate_positive_int(n_components, name="n_components", minimum=2)
    provider = _estimator_provider(use_bstats)
    if use_bstats:
        components = [provider.GaussianEstimator(prior=_gaussian_default_prior()) for _ in range(n_components)]
        return provider.MixtureEstimator(
            components,
            robust=True,
            prior=provider.SymmetricDirichletDistribution(_BAYES_DIRICHLET_ALPHA),
        )
    components = [provider.GaussianEstimator() for _ in range(n_components)]
    return provider.MixtureEstimator(components, robust=True)


def get_multivariate_gaussian_estimator(dim: int, use_bstats: bool = False) -> "ParameterEstimator":
    """Return a multivariate Gaussian estimator for vectors of dimension ``dim``."""
    dim = _validate_positive_int(dim, name="dim")
    if use_bstats:
        return _estimator_provider(True).MultivariateGaussianEstimator(dim=dim, prior=_mvn_default_prior(dim))
    return _estimator_provider(False).MultivariateGaussianEstimator(dim=dim)


# --- explicit modality routing ------------------------------------------------------------
#
# Shape is not provenance: ordinary wide vectors and matrices retain mathematical-array semantics.
# When a caller explicitly identifies embedding or image data, these builders provide the corresponding
# neural density. EMBEDDING_MIN_DIM is retained as the threshold for a diagnostic that suggests the
# explicit embedding option without selecting it automatically.
EMBEDDING_MIN_DIM = 16
IMAGE_FEATURE_DIM = 16


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def get_hybrid_embedding_estimator(dim: int) -> "ParameterEstimator":
    """An exact neural density (a coupling flow) over an embedding-shaped ``dim``-vector field."""
    dim = _validate_positive_int(dim, name="dim")
    from mixle.models.neural_families import Flow

    return Flow(dim=dim).estimator()


def get_hybrid_image_estimator(dim: int = IMAGE_FEATURE_DIM) -> "ParameterEstimator":
    """A frozen ``image_features`` extractor composed with an exact neural density over the induced features."""
    dim = _validate_positive_int(dim, name="dim")
    from mixle.models.feature_map import FeatureMapEstimator, register_feature_fn
    from mixle.models.neural_families import Flow
    from mixle.represent.modality import image_features

    name = f"image_features_{dim}"
    register_feature_fn(
        name,
        lambda img, _dim=dim: image_features(img, dim=_dim),
        version="image-features-v1",
        feature_dim=dim,
    )
    return FeatureMapEstimator(name, Flow(dim=dim).estimator())


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
    mixle.inference.estimation.fit.

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

    import mixle.stats as provider
    from mixle.inference.estimation import fit

    from .profiling import get_estimator, normalize_input

    max_components = _validate_positive_int(max_components, name="max_components")
    max_its = _validate_positive_int(max_its, name="max_its")
    print_iter = _validate_positive_int(print_iter, name="print_iter")
    _validate_pseudo_count(pseudo_count)
    if isinstance(delta, bool) or not isinstance(delta, Real):
        raise TypeError("delta must be a finite non-negative real number")
    delta = float(delta)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be a finite non-negative real number")

    rows = list(normalize_input(data))
    if not rows:
        raise ValueError("data must contain at least one observation")

    if rng is None:
        rng = np.random.RandomState(0)  # fixed default: an un-seeded fit is deterministic
    if not callable(getattr(rng, "choice", None)):
        raise TypeError("rng must provide the NumPy random-state interface")
    if out is None:
        out = sys.stdout
    if not callable(getattr(out, "write", None)):
        raise TypeError("out must be a writable stream or None")

    comp_ests = [get_estimator(rows, pseudo_count=pseudo_count, use_bstats=True) for _ in range(max_components)]
    est = provider.DirichletProcessMixtureEstimator(comp_ests)

    return fit(rows, est, max_its=max_its, delta=delta, rng=rng, print_iter=print_iter, out=out)
