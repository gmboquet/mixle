"""Estimator builders for automatically-typed data.

Builds estimators for mixle.stats. By default the plain maximum-likelihood
estimators are produced; pass use_bstats=True to build the Bayesian path, which
attaches the conjugate default prior for each family so estimation performs the
closed-form conjugate / MAP update.
"""

import math
from collections.abc import Sequence
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
    """Wrap an estimator with an optional/missing-value model."""
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
    """Return a sequence estimator with an optional empirical length model."""
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

    return _estimator_provider(False).LogGaussianEstimator(
        pseudo_count=(pseudo_count, pseudo_count), suff_stat=(ss_1, ss_2)
    )


def get_gamma_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":
    """Return a Gamma estimator initialized from the method-of-moments fit of positive values."""
    k = 1.0
    theta = 1.0
    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0
        ss_2 = 0.0
        for key, v in vdict.items():
            if math.isfinite(key) and key > 0.0:
                ss_0 += v
                ss_1 += key * v
                ss_2 += key * key * v
        if ss_0 > 0.0:
            mean = ss_1 / ss_0
            var = (ss_2 / ss_0) - mean * mean
            if mean > 0.0 and var > 0.0:
                theta = var / mean
                k = mean / theta
    return _estimator_provider(False).GammaEstimator(suff_stat=(k, theta))


def get_student_t_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
) -> "ParameterEstimator":
    """Return a fixed-df Student-t estimator; df is set from the excess kurtosis of the data."""
    df = 5.0
    ss_0 = 0.0
    ss_1 = 0.0
    ss_2 = 0.0
    ss_4 = 0.0
    for key, v in vdict.items():
        if math.isfinite(key):
            ss_0 += v
            ss_1 += key * v
            ss_2 += key * key * v
    if ss_0 > 0.0:
        mean = ss_1 / ss_0
        var = (ss_2 / ss_0) - mean * mean
        if var > 0.0:
            for key, v in vdict.items():
                if math.isfinite(key):
                    ss_4 += v * (key - mean) ** 4
            excess_kurtosis = (ss_4 / ss_0) / (var * var) - 3.0
            if excess_kurtosis > 0.0:
                df = min(max(4.0 + 6.0 / excess_kurtosis, 2.5), 100.0)
    return _estimator_provider(False).StudentTEstimator(df=df)


def get_gaussian_mixture_estimator(
    vdict: dict[np.floating | float, float],
    pseudo_count: float | None = None,
    emp_suff_stat: bool = True,
    use_bstats: bool = False,
    n_components: int = 2,
) -> "ParameterEstimator":
    """Return a K-component Gaussian mixture estimator (robust init) for multimodal numeric data."""
    provider = _estimator_provider(False)
    components = [provider.GaussianEstimator() for _ in range(max(2, int(n_components)))]
    return provider.MixtureEstimator(components, robust=True)


def get_multivariate_gaussian_estimator(dim: int, use_bstats: bool = False) -> "ParameterEstimator":
    """Return a multivariate Gaussian estimator for vectors of dimension ``dim``."""
    if use_bstats:
        return _estimator_provider(True).MultivariateGaussianEstimator(dim=dim, prior=_mvn_default_prior(dim))
    return _estimator_provider(False).MultivariateGaussianEstimator(dim=dim)


# --- modality-fingerprint routing ---------------------------------------------------------
#
# A fixed-length numeric vector is not always "low-dimensional tabular numeric": at moderate-to-high
# dimension it is much more often an embedding (a frozen encoder's output, a pooled feature vector) than
# a handful of jointly-Gaussian measurements, and a bare multivariate Gaussian is the wrong default there
# -- it can only capture a unimodal ellipsoid, not the manifold structure embeddings actually have. Above
# EMBEDDING_MIN_DIM, route to a hybrid neural density (an exact normalizing flow) instead. A 2-D/3-D
# numeric array (an image-shaped field) is routed through a frozen, deterministic feature extractor
# (mixle.represent.modality.image_features) into the same hybrid density -- the "frozen encoder +
# structured head" pattern. Below the threshold (plain low-dim tabular numeric) nothing changes: this is
# additive, not a replacement of the existing per-coordinate/MVN path.
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
    from mixle.models.neural_families import Flow

    return Flow(dim=dim).estimator()


def get_hybrid_image_estimator(dim: int = IMAGE_FEATURE_DIM) -> "ParameterEstimator":
    """A frozen ``image_features`` extractor composed with an exact neural density over the induced features."""
    from mixle.models.feature_map import FeatureMapEstimator, register_feature_fn
    from mixle.models.neural_families import Flow
    from mixle.represent.modality import image_features

    name = f"image_features_{dim}"
    register_feature_fn(name, lambda img, _dim=dim: image_features(img, dim=_dim))
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

    from .profiling import get_estimator

    if rng is None:
        rng = np.random.RandomState(0)  # fixed default: an un-seeded fit is deterministic
    if out is None:
        out = sys.stdout

    comp_ests = [get_estimator(data, pseudo_count=pseudo_count, use_bstats=True) for _ in range(max_components)]
    est = provider.DirichletProcessMixtureEstimator(comp_ests)

    return fit(data, est, max_its=max_its, delta=delta, rng=rng, print_iter=print_iter, out=out)
