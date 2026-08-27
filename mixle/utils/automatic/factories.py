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
# Row-arity heuristics. A rectangular table carrying one malformed row and genuine variable-length
# sequence data arrive at the profiler as the same shape -- a positional container whose observed
# arities differ -- so the reading has to be chosen from how those arities are distributed. One arity
# holding at least MALFORMED_TABLE_MIN_SHARE of the rows is a table whose remaining rows lost or
# gained a field; a majority short of that is genuinely ambiguous and is disclosed; below
# AMBIGUOUS_TABLE_MIN_SHARE no arity is even a majority, which is what variable-length data looks
# like, and it is read as a sequence exactly as it always has been.
MALFORMED_TABLE_MIN_SHARE = 0.95
AMBIGUOUS_TABLE_MIN_SHARE = 0.5
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


def _anchored_moments(items: Sequence[tuple[float, float]]) -> tuple[float, float | None, float | None]:
    """Weighted ``(total, mean, variance)`` from ``(value, weight)`` pairs, computed shift-invariantly.

    ``sum(w*k*k)/W - mean**2`` is the textbook cancelling form. On data whose magnitude dwarfs its
    spread -- epoch timestamps in s/ms/us, instrument baselines, prices quoted against a large
    offset -- it loses roughly ``2*log2(abs(mean)/sd)`` bits, so at 1e9 it returns noise and at 1e12 it
    returns a negative number. The scalar Gaussian accumulator was already fixed to expand its
    scatter about a data anchor (see ``_anchored_pooled_variance`` in
    ``mixle.stats.univariate.continuous.gaussian``), but the auto-inference factories kept the
    cancelling form for the prior they seed, so the repaired defect survived in the one-argument
    entry point that :func:`mixle.inference.optimize` documents.

    Anchoring at an observed value makes every term ``O(W * spread**2)`` and the result invariant
    under a shift of the data. A constant column needs no special case: every deviation from an
    anchor that is itself one of the values is exactly zero, so the variance comes out exactly 0.0
    and the caller recognizes the column as constant. The only clamp is relative -- a residue below
    1e-12 of the largest term is cross-term rounding, not spread. Deliberately NO absolute
    ``eps * abs(mean)`` floor: at 1e15 that threshold is 0.89, which reads a real ``N(1e15, 1)`` sample
    (scatter 436 against a threshold of 395) as constant on some seeds and not on others.

    Returns ``(0.0, None, None)`` when no pair carries weight; the variance is never negative.
    """
    total = 0.0
    anchor: float | None = None
    for key, weight in items:
        if anchor is None:
            anchor = key
        total += weight
    if anchor is None or total <= 0.0:
        return 0.0, None, None
    shifted_sum = 0.0
    shifted_sum2 = 0.0
    for key, weight in items:
        delta = key - anchor
        shifted_sum += delta * weight
        shifted_sum2 += delta * delta * weight
    offset = shifted_sum / total
    mean = anchor + offset
    scatter = shifted_sum2 - total * offset * offset
    noise_scale = max(abs(shifted_sum2), total * offset * offset, 1.0e-300)
    if not math.isfinite(scatter) or scatter < 1.0e-12 * noise_scale:
        scatter = 0.0
    return total, mean, scatter / total


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


def get_optional_estimator(
    est: ParameterEstimator,
    missing_value: Any | None = None,
    use_bstats: bool = False,
    est_prob: bool = False,
):
    """Wrap an estimator with an optional/missing-value model.

    ``est_prob`` decides what the wrapper MEANS, so callers pass it deliberately rather than taking a
    default. With it false the fitted OptionalDistribution carries no rate: it is transparent to the
    density (contributing 0) and is a marginalized likelihood factor that cannot generate
    observations. With it true the missingness rate is fit from the data the profiler has already
    seen, the wrapper contributes log(p)/log(1-p) to the density, and the model can be sampled.

    Genuine missingness wants the rate, and at the automatic surface that is BOTH of its spellings:
    ``None`` and ``nan``. They must mean the same model, because they are interchanged behind the
    caller's back -- a pandas float Series silently stores ``None`` as ``nan`` -- so giving ``nan``
    the transparent non-generative wrapper made the same data produce a different (and unsamplable)
    model depending on the container it arrived in.

    The infinity sentinels (``+/-inf``) take the fitted rate for the SAME reason, though by a
    different argument. They are indeed values a numeric field can carry rather than absences, but
    that is precisely why the fitted-rate wrapper is the correct model for them: it is an atom of
    mass ``p`` at the sentinel beside the base family scaled by ``1-p``, which is exactly "this
    column is a continuous measurement that comes out infinite ``p`` of the time" and integrates to
    one. The transparent default asserts the opposite -- ``.p`` reads 0.0, "missingness never
    happens", while ``log_density(inf)`` returns 0.0, probability one -- for total mass 2.0, an
    improper density whose sentinel rows cost exactly zero nats. Auto-inference used the transparent
    default here through 0.7.x, so a single overflow or a JSON ``1e999`` bought an unbounded free
    gain in log-likelihood against any proper competitor on the same data, invisible to the paired
    comparison tests in ``mixle.inference`` because those consume plain arrays and cannot see
    ``density_semantics()``. Callers that genuinely want the marginalized factor -- inference
    conditional on an externally modeled missingness mechanism -- still get it by passing
    ``est_prob=False`` deliberately, which is what this argument is for.
    """
    return _estimator_provider(use_bstats).OptionalEstimator(est, missing_value=missing_value, est_prob=est_prob)


def get_typed_mixture_estimator(
    string_estimator: ParameterEstimator,
    numeric_estimator: ParameterEstimator,
    use_bstats: bool = False,
    string_support: Sequence[Any] | None = None,
) -> "ParameterEstimator":
    """Dispatch mixture for a scalar column that carries both numbers and strings.

    One dirty cell in a CSV -- a single ``"N/A"``, ``"NULL"``, ``"?"`` or ``""`` in an otherwise
    continuous column -- is the most ordinary data-quality defect there is, and until 0.8.0 it
    silently retyped the WHOLE column: 300 finite floats plus one ``"N/A"`` resolved to a
    categorical over the 301 observed values (wrapped in ``IgnoredDistribution`` once the identifier
    thresholds were crossed), a memorization table that scored every value it had not literally seen
    -- the sample mean included -- at ``-inf``, with ``fit_provenance()`` reporting
    ``converged=True`` and ``repairs=()``.

    The honest reading of such a column is that it carries two disjoint types, so the honest model is
    the one the library already has for disjoint types: a
    :class:`~mixle.stats.combinator.select.SelectDistribution` dispatch mixture whose branch label is
    the Python type, routed by the serializable
    :class:`~mixle.stats.combinator.select.TypeDispatch`. The branch is OBSERVED, so nothing is
    guessed and nothing is iterated: the weights are the two type proportions in closed form, each
    child is fit on its own subset by the ordinary leaf rules, and the result is a normalized law
    that samples.

    Two properties make this a repair rather than a re-design. The string branch's density is
    unchanged -- ``log(n_str/n) + log(count/n_str)`` is ``log(count/n)``, what the merged categorical
    gave, exactly in real arithmetic and to within a ULP in floating point -- so nothing that scored
    finitely before scores differently now. And on the case above the mixture reproduces the model
    you get by spelling the dirty cell ``nan`` instead, digit for digit (mean 49.997748666666666,
    variance 93.32376314029874, rate 0.0033222591362126247), which is the answer the library already
    called correct.

    ``string_support`` exists for one specific consequence of routing. A merged categorical is shown
    every label in every mixture component -- ``CategoricalAccumulator`` records a label even at zero
    weight -- but a ROUTED branch is only shown the rows that route to it, and a component that is
    initialized with no string rows at all hands its string branch an empty count map. Under the
    Bayesian path the symmetric ``DictDirichletDistribution(alpha)`` cannot widen that into anything
    (its parameters are a scalar, so there is no support to fall back on) and estimation raises
    "empty categorical fitting requires a prior with an explicit finite support" -- which is how
    ``get_dpm_mixture`` met this repair. Naming the observed string labels in the prior gives the
    empty branch the uniform over exactly those labels instead. The plain MLE path needs no
    equivalent: ``get_categorical_estimator`` already carries the empirical map as ``suff_stat``, and
    the empty case widens to a zero-count support over it.

    Args:
        string_estimator: Leaf estimator for the ``str``/``bytes`` values.
        numeric_estimator: Leaf estimator for the (non-boolean) real values.
        use_bstats: Build the Bayesian path for the wrapper's provider lookup.
        string_support: Observed string labels, used only to pin the Bayesian branch's Dirichlet
            support as described above. Ignored on the plain path and for a frozen (Ignored) branch,
            which is never re-estimated and so never meets the empty-count case.

    Returns:
        A ``SelectEstimator`` that routes by type and estimates the branch weights.
    """
    provider = _estimator_provider(use_bstats)
    if use_bstats and string_support and getattr(string_estimator, "has_conj_prior", False):
        string_estimator.set_prior(
            provider.DictDirichletDistribution({key: _BAYES_DIRICHLET_ALPHA for key in string_support})
        )
    router = provider.TypeDispatch([("str", "bytes"), ("real",)])
    return provider.SelectEstimator([string_estimator, numeric_estimator], router, estimate_weights=True)


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


def _get_identifier_estimator(vdict: dict[Any, float], use_bstats: bool = False) -> "ParameterEstimator":
    """Frozen, finite-scoring stand-in for a field automatic typing declines to model.

    The callers cover an identifier-like column (nearly every value distinct), a scalar type the
    profiler does not recognize (a datetime column being the everyday case), and an ambiguous
    bool/numeric mix. Such a column carries no density information the profiler is willing to fit,
    but it still appears in every row, so whatever stands in for it has to be able to SCORE a row.
    The bare IgnoredEstimator
    default -- a point mass at ``None`` -- assigns log-density ``-inf`` to every actual identifier,
    which poisoned any downstream fit over data containing such a column (EM/DPM raised "EM did not
    produce a finite objective" without naming the column). The child here is instead the empirical
    categorical over the observed values, held fixed by the Ignored wrapper: finite on every row of
    the data the model was inferred from, identical across mixture components (as a per-row constant
    factor it cannot distort responsibilities), and samplable. An identifier NOT seen at profiling
    time scores ``-inf`` -- deliberately the same finite-support behavior every other automatically
    fitted categorical column has (``default_value=0``), rather than a new smoothed regime that only
    identifier columns would get.
    """
    total = float(sum(vdict.values()))
    provider = _estimator_provider(use_bstats)
    child = provider.CategoricalDistribution(pmap={key: value / total for key, value in vdict.items()})
    return provider.IgnoredEstimator(dist=child)


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


MAX_INFERRED_ESCAPE_WEIGHT = 0.05


def _backoff_over_unobserved(sharp, vdict, min_val, observed_count, *, use_bstats):
    """Wrap a fitted-support integer-categorical so held-out integers stay scorable, *properly*.

    An inferred support covers only what was observed, so the sharp estimator alone returns ``-inf``
    for any held-out integer outside it -- one such row drives a whole held-out mean log-density to
    ``-inf`` (observed on real heterogeneous records with mixed-type categorical fields).

    The earlier remedy set ``IntegerCategoricalDistribution.default_value``, which is wrong here and
    was reported as MXR-080-1838: the integer support is *unbounded*, so a constant out-of-support
    probability is paid over infinitely many integers and the total mass diverges. That is an improper
    distribution, not smoothing, and the factory was putting one in every inferred model.

    A backoff mixture is proper: the fallback is itself a normalized distribution, so the two
    components' mass sums to 1 no matter how wide the unobserved region is.

    Only applied when a *valid* fallback exists. Poisson requires non-negative integers, so a support
    reaching below zero is left sharp rather than wrapped in a fallback that cannot represent it --
    an honest ``-inf`` beats a fabricated finite score.
    """
    if min_val < 0 or observed_count <= 0.0:
        return sharp
    # Only a support that fails to tile its own range needs a fallback. A small dense code set --
    # every integer in [min_val, max_val] observed -- already covers its support, so wrapping it
    # would claim mass for values the family legitimately calls impossible and would change the
    # inferred family for columns that are genuinely categorical (automatic_test's
    # small_cardinality_ints_categorical and dense_integer say so).
    keys = {int(k) for k in vdict}
    if len(keys) >= (max(keys) - min(keys) + 1):
        return sharp
    escape = min(1.0 / observed_count, MAX_INFERRED_ESCAPE_WEIGHT)
    return _estimator_provider(use_bstats).BackoffEstimator(
        sharp,
        get_poisson_estimator(vdict, None, emp_suff_stat=True, use_bstats=use_bstats),
        escape_weight=escape,
        max_escape_weight=MAX_INFERRED_ESCAPE_WEIGHT,
    )


def get_integer_categorical_estimator(
    vdict: dict[int, float], pseudo_count: float | None = None, emp_suff_stat: bool = True, use_bstats: bool = False
) -> "ParameterEstimator":
    """Return an integer-categorical estimator over the observed dense support."""
    _validate_pseudo_count(pseudo_count)
    _validate_mass_map(vdict, name="vdict", require_positive_total=True)
    if any(isinstance(k, bool) or not isinstance(k, Integral) for k in vdict):
        raise ValueError("integer-categorical observations must be integers")
    min_val, max_val, width = _integer_range(vdict)
    observed_count = float(sum(vdict.values()))

    if use_bstats:
        return _backoff_over_unobserved(
            _estimator_provider(True).IntegerCategoricalEstimator(
                min_val=min_val, max_val=max_val, prior=_integer_categorical_default_prior()
            ),
            vdict,
            min_val,
            observed_count,
            use_bstats=True,
        )

    suff_stat = None
    if emp_suff_stat:
        cnt = float(sum(vdict.values()))
        p_vec = np.zeros(width, dtype=float)
        if cnt > 0.0:
            for k, v in vdict.items():
                p_vec[int(k) - min_val] = float(v) / cnt
        suff_stat = (min_val, p_vec)

    return _backoff_over_unobserved(
        _estimator_provider(False).IntegerCategoricalEstimator(
            min_val=min_val, max_val=max_val, pseudo_count=pseudo_count, suff_stat=suff_stat
        ),
        vdict,
        min_val,
        observed_count,
        use_bstats=False,
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
        # Shift-invariant moments: the cancelling ``E[x^2] - E[x]^2`` form used to seed this prior
        # returned noise at an offset of 1e9 and a negative number at 1e12, which then took the
        # degenerate branch below. See :func:`_anchored_moments`.
        ss_0, ss_1, ss_2 = _anchored_moments([(float(k), float(v)) for k, v in vdict.items() if math.isfinite(k)])
        # ss_0 is 0 when vdict is empty or every key was non-finite -- no data to estimate mean/variance
        # from, so fall back the same way the emp_suff_stat=False branch does below.
        if ss_0 > 0.0:
            # A constant field has exactly zero empirical spread, so there is no scale to seed a
            # prior from and the estimator is right to refuse a non-positive prior variance. The
            # former fallback -- one pseudo-observation at (1e-6, 1e-6) -- is not "no information":
            # it is an observation at the ORIGIN, so it dragged the fitted mean toward zero by
            # n/(n+1) and inflated the variance by the squared distance from the data to zero. On
            # [300.0]*10 that returned mean 272.73 / variance 7438 for a column whose MLE is
            # (300, 0), with provenance still reporting a converged MLE fit and no repairs.
            # ``pseudo_count`` paired with ``suff_stat=(None, None)`` is the estimator's documented
            # spelling for "no pseudo-observations": it takes the plain maximum-likelihood moment
            # and applies -- and DISCLOSES through numerical_repairs() -- its own variance floor,
            # which is exactly what the explicit-prototype path already did for this data.
            if not math.isfinite(ss_2) or ss_2 <= 0.0:
                ss_1, ss_2 = None, None
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
        # Same shift-invariant moments as the Gaussian factory, over log-values: log(1e12 + z) is
        # 27.6 +/- 1e-12, so the cancelling form loses the spread there exactly as it does on the
        # raw scale. See :func:`_anchored_moments`.
        ss_0, ss_1, ss_2 = _anchored_moments(
            [(math.log(float(k)), float(v)) for k, v in vdict.items() if math.isfinite(k) and k > 0.0]
        )
        # ss_0 is 0 when vdict is empty or every key was non-positive/non-finite (log-normal needs
        # strictly positive values) -- no data to estimate mean/variance from, fall back like the
        # emp_suff_stat=False branch does below rather than dividing by zero.
        if ss_0 > 0.0:
            # A constant field has exactly zero empirical spread, so there is no scale to seed a
            # prior from and the estimator is right to refuse a non-positive prior variance. Handing
            # over a pseudo-observation at (1e-6, 1e-6) instead is an observation at log-value ~0,
            # i.e. at x = 1, which biases the fit toward 1 by n/(n+1) with nothing disclosed;
            # ``suff_stat=(None, None)`` is the estimator's spelling for "no pseudo-observations",
            # under which it takes the plain MLE and reports its variance floor through
            # numerical_repairs(). Same repair as the Gaussian factory above.
            if not math.isfinite(ss_2) or ss_2 <= 0.0:
                ss_1, ss_2 = None, None
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
                    # StudentTEstimator does a MOMENT fit, which needs a finite variance and so
                    # requires df > 2 -- it rejects anything less in its constructor. Accepting a
                    # fitted df > 0 here handed it df in (0, 2] and crashed with "moment fit
                    # requires finite df > 2", which is exactly what scipy returns on genuinely
                    # heavy-tailed data. A df that low is a real fit, just not one this estimator
                    # can use, so it falls through to the df = 5.0 default the same way an
                    # unusable (non-finite, non-positive-scale) fit already does.
                    and fit_df > 2.0
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

    Raises:
        ValueError: If ``data`` is empty, or if automatic typing found no modelable field at all
            (e.g. every column is identifier-like) -- the mixture would then score every
            observation identically and "fit" nothing; the error names each field, its
            cardinality, and the remedy.
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
    if _estimates_nothing(comp_ests[0]):
        # Every field resolved to an Ignored (frozen) factor, so each stick would score every row
        # identically and EM would return its initialization unchanged -- a "fit" of nothing.
        # Refuse with the per-field evidence instead of letting that vacuous success stand.
        from .profiling import DatumNode, _unmodelable_fields_report

        raise ValueError(
            "automatic typing found no modelable field, so a Dirichlet process mixture would score "
            "every observation identically and fit nothing: %s. Drop identifier-like fields (or "
            "replace them with modelable features) before fitting; if the values really are "
            "categories, build the component estimator explicitly (e.g. with "
            "mixle.utils.automatic.get_categorical_estimator) and fit it with "
            "mixle.inference.estimation.fit." % "; ".join(_unmodelable_fields_report(DatumNode(data=rows)))
        )
    est = provider.DirichletProcessMixtureEstimator(comp_ests)

    return fit(rows, est, max_its=max_its, delta=delta, rng=rng, print_iter=print_iter, out=out)


def _estimates_nothing(est: "ParameterEstimator") -> bool:
    """Whether an automatically-typed estimator carries no estimable statistic anywhere.

    True exactly when every leaf is an Ignored (frozen) factor: the root itself, or a composite /
    record whose every field is one. Optional wrappers, sequence estimators, and every concrete
    family DO estimate something (a rate, a length model, parameters), so they stop the walk.
    """
    provider = _estimator_provider(False)
    if isinstance(est, provider.IgnoredEstimator):
        return True
    if isinstance(est, (provider.CompositeEstimator, DictRecordEstimator)):
        return all(_estimates_nothing(child) for child in est.estimators)
    return False
