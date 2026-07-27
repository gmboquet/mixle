"""Copula combinator: glue arbitrary marginal distributions to a dependence structure (Sklar's theorem).

``GaussianCopulaDistribution`` (``mixle.stats.multivariate``) models *only* dependence, on the unit cube
``(0,1)^d`` -- it assumes its inputs are already the uniform scores ``u_i``. This combinator is the piece
that makes copulas *composable*: give it your CONTINUOUS marginals (any mixle leaves exposing ``cdf`` --
a Gamma, a StudentT, a VonMises, any continuous family mixed freely) and a copula core, and it forms the
joint

    f(x_1, ..., x_d) = c(F_1(x_1), ..., F_d(x_d)) * prod_i f_i(x_i)                       (Sklar)

where each ``F_i = marginals[i].cdf`` is the probability-integral transform (PIT) and ``c`` is the copula
density. That is the whole point of copulas: pick any marginals you like and couple them through one
dependence object, instead of hand-writing a bespoke multivariate leaf for every marginal combination.

This continuous-Sklar formula is only valid when every ``f_i`` is a genuine density. A DISCRETE marginal's
``log_density`` is a probability MASS, not a density; plugging a mass into the continuous copula-*density*
formula over/under-counts and the resulting "probabilities" no longer sum to 1 (two Bernoulli(0.5) margins
coupled by a correlated Gaussian copula, e.g., report outcome probabilities summing to roughly 2.5e8, not
1). The mathematically correct joint for a discrete or mixed marginal instead needs copula-*CDF* rectangle
differences (inclusion-exclusion over ``C``, not ``c``, at the corners of each discrete cell) -- none of
this combinator's copula cores (Gaussian/Clayton/Frank/Student-t/vine) currently expose a CDF, so rather
than silently producing an invalid probability model, ``CopulaDistribution.__init__`` rejects any marginal
with enumerable (discrete/atomic) support. Discrete/mixed marginals are not yet supported.

Estimation is **IFM** (Inference Functions for Margins): fit each marginal on its own column, PIT the data
through the fitted marginals, then fit the copula on the uniform scores. This is exact in a single M-step
(the accumulator buffers the raw columns, the same pattern the neural leaves use), not an approximate
coupled iteration.

The copula core is pluggable: any distribution on ``(0,1)^d`` implementing the mixle five-piece contract
works, as long as it also exposes a ``dim`` attribute matching ``len(marginals)`` (checked at construction).
:class:`~mixle.stats.multivariate.gaussian_copula.GaussianCopulaDistribution` is the first supported
core; Clayton/Frank/t cores can be dropped in later with no change here.

Reference: Nelsen, *An Introduction to Copulas* (2nd ed., Springer, 2006); Joe, *Dependence Modeling with
Copulas* (CRC, 2014).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from mixle.capability import HasCDF, supports
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.multivariate._copula_common import reject_out_of_unit_cube

_CLIP = 1.0e-12  # keep PIT scores strictly inside (0,1) so the copula's Phi^{-1} stays finite


class CopulaIFMStatistics(NamedTuple):
    """Versioned IFM state with explicit stage-specific effective counts."""

    schema_version: int
    marginal_statistics: tuple[Any, ...]
    columns: np.ndarray
    weights: np.ndarray
    marginal_effective_count: float
    copula_effective_count: float


class CopulaQuantileError(RuntimeError):
    """Raised when a marginal quantile violates its declared CDF inverse contract."""


def _finite_weight(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a finite non-negative scalar." % name) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("%s must be a finite non-negative scalar." % name)
    return result


def _validate_row(value: Any, dim: int, *, label: str = "copula observation") -> np.ndarray:
    try:
        row = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must contain exactly %d finite numeric coordinates." % (label, dim)) from exc
    if row.shape != (dim,) or not np.all(np.isfinite(row)):
        raise ValueError("%s must have shape (%d,) with finite numeric coordinates." % (label, dim))
    return row


def _validate_columns(value: Any, dim: int, *, label: str = "copula columns") -> np.ndarray:
    try:
        columns = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a finite numeric matrix with %d columns." % (label, dim)) from exc
    if columns.shape == (0,):
        columns = np.zeros((0, dim), dtype=np.float64)
    if columns.ndim != 2 or columns.shape[1] != dim or not np.all(np.isfinite(columns)):
        raise ValueError("%s must have shape (n, %d) with finite numeric values." % (label, dim))
    return columns


def _validate_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("copula weights must be a finite non-negative vector.") from exc
    if (
        weights.shape != (rows,)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("copula weights must be a finite non-negative vector aligned with the buffered rows.")
    return weights


def _validate_ifm_statistics(value: Any, dim: int) -> CopulaIFMStatistics:
    if not isinstance(value, CopulaIFMStatistics) or value.schema_version != 1:
        raise ValueError("copula IFM statistics must use schema version 1.")
    if not isinstance(value.marginal_statistics, tuple) or len(value.marginal_statistics) != dim:
        raise ValueError("copula IFM marginal-statistic arity must equal %d." % dim)
    columns = _validate_columns(value.columns, dim)
    weights = _validate_weights(value.weights, len(columns))
    marginal_count = _finite_weight(value.marginal_effective_count, name="marginal_effective_count")
    copula_count = _finite_weight(value.copula_effective_count, name="copula_effective_count")
    actual = float(weights.sum())
    tolerance = 1.0e-12 * max(1.0, actual)
    if abs(marginal_count - actual) > tolerance or abs(copula_count - actual) > tolerance:
        raise ValueError("copula IFM effective counts must equal the buffered weight sum.")
    return CopulaIFMStatistics(
        1,
        value.marginal_statistics,
        columns,
        weights,
        marginal_count,
        copula_count,
    )


def _is_discrete_marginal(marginal: Any) -> bool:
    """Return whether ``marginal`` has enumerable (discrete/atomic) support.

    Probes :meth:`~mixle.stats.compute.pdist.SequenceEncodableProbabilityDistribution.enumerator` the same
    way :func:`mixle.stats.combinator.truncated._is_discrete_base` /
    :func:`mixle.stats.combinator.censored._is_discrete_base` already do: a distribution with atomic support
    overrides ``enumerator()``, while a continuous one inherits the base class's default and raises
    :class:`EnumerationError`. A marginal that does not even expose ``enumerator`` (e.g. a minimal
    duck-typed stub) is likewise treated as continuous, matching the base class's own default. This is how
    the constructor below tells apart a ``marginal.log_density(x)`` that is a probability MASS (discrete
    support) from one that is a genuine density (continuous support) -- see the module docstring for why
    that distinction is load-bearing for the continuous Sklar decomposition this class implements.
    """
    if not hasattr(marginal, "enumerator"):
        return False
    try:
        marginal.enumerator()
    except EnumerationError:
        return False
    return True


class CopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Joint over ``d`` scalar fields = ``d`` marginals coupled by a copula core (Sklar's theorem).

    ``marginals`` is a length-``d`` sequence of CONTINUOUS mixle leaves, each exposing ``log_density``,
    deterministic ``cdf``/``quantile`` methods, and the estimator/encoder contract -- a marginal with
    enumerable (discrete/atomic) support is rejected at construction (see module docstring: the continuous
    Sklar density formula this class implements is not a valid probability model for discrete marginals).
    ``copula`` is a distribution on ``(0,1)^d`` (e.g. :class:`GaussianCopulaDistribution`) whose ``dim``
    must equal ``len(marginals)``. An observation is a length-``d`` tuple/array of scalars
    ``(x_1, ..., x_d)``.
    """

    def __init__(
        self,
        marginals: Sequence[SequenceEncodableProbabilityDistribution],
        copula: SequenceEncodableProbabilityDistribution,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.marginals = tuple(marginals)
        self.dim = len(self.marginals)
        if self.dim < 2:
            raise ValueError("CopulaDistribution needs at least 2 marginals; got %d" % self.dim)
        discrete = [i for i, m in enumerate(self.marginals) if _is_discrete_marginal(m)]
        if discrete:
            raise ValueError(
                "CopulaDistribution only supports continuous marginals. Its log_density/seq_log_density "
                "apply the continuous Sklar decomposition f(x) = c(F_1(x_1),...,F_d(x_d)) * prod_i f_i(x_i), "
                "which is valid only when every f_i (marginals[i].log_density) is a genuine density; "
                "marginal(s) at index %s have enumerable (discrete/atomic) support, where log_density is a "
                "probability MASS instead -- plugging a mass into the continuous copula-density formula "
                "over/under-counts and the model no longer sums to 1. A correct discrete/mixed joint needs "
                "copula-CDF rectangle differences, which none of this combinator's copula cores currently "
                "expose; discrete/mixed marginals are not yet supported." % discrete
            )
        missing_cdf = [i for i, marginal in enumerate(self.marginals) if not supports(marginal, HasCDF)]
        if missing_cdf:
            raise TypeError(
                "CopulaDistribution marginals must satisfy the deterministic HasCDF contract "
                "(callable cdf and quantile); missing at indices %s." % missing_cdf
            )
        copula_dim = getattr(copula, "dim", None)
        if copula_dim is None:
            raise ValueError(
                "CopulaDistribution's copula core must expose a `dim` attribute; %s does not." % type(copula).__name__
            )
        if int(copula_dim) != self.dim:
            raise ValueError(
                "CopulaDistribution has %d marginals but its copula core (%s) is %d-dimensional; "
                "these must match." % (self.dim, type(copula).__name__, int(copula_dim))
            )
        self.copula = copula
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "CopulaDistribution([%s], %s)" % (", ".join(map(str, self.marginals)), str(self.copula))

    def _pit_row(self, x: Sequence[float]) -> np.ndarray:
        """Probability-integral transform of one observation: ``u_i = clip(F_i(x_i))`` in ``(0,1)``."""
        row = _validate_row(x, self.dim)
        u = np.array([float(self.marginals[i].cdf(row[i])) for i in range(self.dim)], dtype=np.float64)
        reject_out_of_unit_cube(u)
        return np.clip(u, _CLIP, 1.0 - _CLIP)

    def _pit_columns(self, cols: np.ndarray) -> np.ndarray:
        """PIT an ``(n, d)`` array of raw observations to an ``(n, d)`` array of uniform scores."""
        cols = _validate_columns(cols, self.dim)
        u = np.empty_like(cols)
        for i in range(self.dim):
            u[:, i] = [float(self.marginals[i].cdf(v)) for v in cols[:, i]]
        reject_out_of_unit_cube(u)
        return np.clip(u, _CLIP, 1.0 - _CLIP)

    def density(self, x: Sequence[float]) -> float:
        """Return the joint density at one raw observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[float]) -> float:
        """Sklar's decomposition: sum of marginal log-densities + the copula log-density at the PIT scores."""
        row = _validate_row(x, self.dim)
        marg = sum(float(self.marginals[i].log_density(row[i])) for i in range(self.dim))
        return marg + float(self.copula.log_density(self._pit_row(row)))

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Vectorized: per-column marginal log-densities (summed) plus the copula log-density at the PIT scores."""
        if not isinstance(enc, tuple) or len(enc) != 2:
            raise ValueError("encoded copula data must be a (marginal_encodings, raw_columns) pair.")
        marg_encs, raw_cols = enc
        if not isinstance(marg_encs, tuple) or len(marg_encs) != self.dim:
            raise ValueError("encoded copula marginal arity must equal %d." % self.dim)
        raw_cols = _validate_columns(raw_cols, self.dim)
        rv = np.asarray(self.marginals[0].seq_log_density(marg_encs[0]), dtype=np.float64)
        if rv.shape != (len(raw_cols),):
            raise ValueError("marginal 0 batch score must contain exactly one value per observation.")
        for i in range(1, self.dim):
            marginal_scores = np.asarray(self.marginals[i].seq_log_density(marg_encs[i]), dtype=np.float64)
            if marginal_scores.shape != (len(raw_cols),):
                raise ValueError("marginal %d batch score must contain exactly one value per observation." % i)
            rv = rv + marginal_scores
        u = self._pit_columns(raw_cols)
        cop_enc = self.copula.dist_to_encoder().seq_encode(u)
        copula_scores = np.asarray(self.copula.seq_log_density(cop_enc), dtype=np.float64)
        if copula_scores.shape != (len(raw_cols),):
            raise ValueError("copula batch score must contain exactly one value per observation.")
        return rv + copula_scores

    def sampler(self, seed: int | None = None) -> CopulaSampler:
        """Return a sampler that draws copula scores and inverts the marginals."""
        return CopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> CopulaEstimator:
        """Return an IFM estimator for marginals followed by the copula core."""
        return CopulaEstimator(
            [m.estimator(pseudo_count=pseudo_count) for m in self.marginals],
            self.copula.estimator(pseudo_count=pseudo_count),
            dim=self.dim,
            copula_prototype=self.copula,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> CopulaDataEncoder:
        """Return an encoder that preserves marginal encodings and raw columns."""
        return CopulaDataEncoder([m.dist_to_encoder() for m in self.marginals])


class CopulaSampler(DistributionSampler):
    """Sample by drawing uniform scores and applying each marginal's deterministic quantile."""

    def __init__(self, dist: CopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self._cop_sampler = dist.copula.sampler(seed if seed is None else seed + 1)
        self._quantiles = tuple(marginal.quantile for marginal in dist.marginals)

    def _invert(self, index: int, u: float) -> float:
        target = float(np.clip(u, _CLIP, 1.0 - _CLIP))
        value = float(self._quantiles[index](target))
        if not np.isfinite(value):
            raise CopulaQuantileError(
                "marginal %d quantile returned a non-finite value for probability %.17g." % (index, target)
            )
        achieved = float(self.dist.marginals[index].cdf(value))
        if not np.isfinite(achieved) or not 0.0 <= achieved <= 1.0:
            raise CopulaQuantileError("marginal %d quantile/CDF contract returned an invalid probability." % index)
        if abs(achieved - target) > 1.0e-7:
            raise CopulaQuantileError(
                "marginal %d quantile/CDF round trip did not converge: target %.17g, achieved %.17g."
                % (index, target, achieved)
            )
        return value

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one joint observation or ``size`` iid observations."""
        if isinstance(size, (bool, np.bool_)):
            raise TypeError("size must be a non-negative integer or None.")
        n = 1 if size is None else int(size)
        if size is not None and (n != size or n < 0):
            raise ValueError("size must be a non-negative integer or None.")
        u = np.atleast_2d(self._cop_sampler.sample(n)).reshape(n, self.dist.dim)
        out = [
            tuple(self._invert(i, float(u[r, i])) for i in range(self.dist.dim)) for r in range(n)
        ]
        return out[0] if size is None else out


class CopulaDataEncoder(DataSequenceEncoder):
    """Encode a batch as ``(per-marginal encodings, raw (n, d) column array)``.

    The raw columns ride along because the copula's uniform scores ``u = F(x)`` depend on the marginals'
    *current* parameters (they change during fitting), so they cannot be baked in at encode time -- they are
    recomputed by :meth:`CopulaDistribution._pit_columns` against whatever distribution is scoring/estimating.
    """

    def __init__(self, marginal_encoders: Sequence[DataSequenceEncoder]) -> None:
        self.marginal_encoders = tuple(marginal_encoders)
        self.dim = len(self.marginal_encoders)

    def __str__(self) -> str:
        return "CopulaDataEncoder([%s])" % ", ".join(map(str, self.marginal_encoders))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CopulaDataEncoder) and self.marginal_encoders == other.marginal_encoders

    def seq_encode(self, x: Sequence[Sequence[float]]) -> tuple[tuple[Any, ...], np.ndarray]:
        """Encode each marginal column while retaining raw columns for PIT recomputation."""
        rows = tuple(_validate_row(row, self.dim, label="copula observation at row %d" % i) for i, row in enumerate(x))
        cols = np.asarray(rows, dtype=np.float64).reshape((len(rows), self.dim))
        marg_encs = tuple(self.marginal_encoders[i].seq_encode(cols[:, i].tolist()) for i in range(self.dim))
        return marg_encs, cols


class CopulaAccumulator(SequenceEncodableStatisticAccumulator):
    """Delegate per-column sufficient stats to marginal sub-accumulators; buffer raw columns for the copula.

    The copula stage is fit AFTER the marginals (IFM): it needs the marginals' fitted CDFs to PIT the data,
    so the raw columns are buffered (weighted) here and PIT-ed inside :meth:`CopulaEstimator.estimate`. This
    is the same buffer-the-rows pattern the neural leaves use, and it makes the IFM fit exact in one M-step.
    """

    def __init__(self, marginal_accumulators: Sequence[Any], dim: int, keys: str | None = None) -> None:
        self.marginal_accumulators = list(marginal_accumulators)
        self.dim = dim
        if len(self.marginal_accumulators) != self.dim:
            raise ValueError("copula accumulator marginal arity must equal dim.")
        self.keys = keys
        self._cols: list[np.ndarray] = []
        self._w: list[np.ndarray] = []

    def update(self, x: Sequence[float], weight: float, estimate: CopulaDistribution | None) -> None:
        """Update marginal accumulators and buffer one raw observation for IFM copula fitting."""
        row = _validate_row(x, self.dim)
        checked_weight = _finite_weight(weight, name="weight")
        if estimate is not None and estimate.dim != self.dim:
            raise ValueError("copula estimate dimension does not match its accumulator.")
        marg_est = estimate.marginals if estimate is not None else [None] * self.dim
        for i in range(self.dim):
            self.marginal_accumulators[i].update(row[i], checked_weight, marg_est[i])
        self._cols.append(row.reshape(1, self.dim))
        self._w.append(np.asarray([checked_weight], dtype=np.float64))

    def initialize(self, x: Sequence[float], weight: float, rng: np.random.RandomState | None) -> None:
        """Initialize marginal accumulators and buffer one raw observation."""
        row = _validate_row(x, self.dim)
        checked_weight = _finite_weight(weight, name="weight")
        for i in range(self.dim):
            self.marginal_accumulators[i].initialize(row[i], checked_weight, rng)
        self._cols.append(row.reshape(1, self.dim))
        self._w.append(np.asarray([checked_weight], dtype=np.float64))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: CopulaDistribution | None) -> None:
        """Update marginal accumulators and buffer encoded raw columns for IFM."""
        if not isinstance(enc, tuple) or len(enc) != 2:
            raise ValueError("encoded copula data must be a (marginal_encodings, raw_columns) pair.")
        marg_encs, raw_cols = enc
        if not isinstance(marg_encs, tuple) or len(marg_encs) != self.dim:
            raise ValueError("encoded copula marginal arity must equal dim.")
        raw_cols = _validate_columns(raw_cols, self.dim)
        weights = _validate_weights(weights, len(raw_cols))
        if estimate is not None and estimate.dim != self.dim:
            raise ValueError("copula estimate dimension does not match its accumulator.")
        marg_est = estimate.marginals if estimate is not None else [None] * self.dim
        for i in range(self.dim):
            self.marginal_accumulators[i].seq_update(marg_encs[i], weights, marg_est[i])
        self._cols.append(raw_cols)
        self._w.append(weights)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: np.random.RandomState | None) -> None:
        """Initialize marginal accumulators and buffer encoded raw columns."""
        if not isinstance(enc, tuple) or len(enc) != 2:
            raise ValueError("encoded copula data must be a (marginal_encodings, raw_columns) pair.")
        marg_encs, raw_cols = enc
        if not isinstance(marg_encs, tuple) or len(marg_encs) != self.dim:
            raise ValueError("encoded copula marginal arity must equal dim.")
        raw_cols = _validate_columns(raw_cols, self.dim)
        weights = _validate_weights(weights, len(raw_cols))
        for i in range(self.dim):
            self.marginal_accumulators[i].seq_initialize(marg_encs[i], weights, rng)
        self._cols.append(raw_cols)
        self._w.append(weights)

    def combine(self, suff_stat: CopulaIFMStatistics) -> CopulaAccumulator:
        """Merge marginal sufficient statistics and buffered raw columns."""
        checked = _validate_ifm_statistics(suff_stat, self.dim)
        for i in range(self.dim):
            self.marginal_accumulators[i].combine(checked.marginal_statistics[i])
        if len(checked.columns):
            self._cols.append(checked.columns)
            self._w.append(checked.weights)
        return self

    def value(self) -> CopulaIFMStatistics:
        """Return marginal stats, buffered raw columns, and buffered weights."""
        marg_vals = tuple(acc.value() for acc in self.marginal_accumulators)
        cols = np.concatenate(self._cols, axis=0) if self._cols else np.zeros((0, self.dim))
        w = np.concatenate(self._w) if self._w else np.zeros((0,))
        effective_count = float(w.sum())
        return CopulaIFMStatistics(1, marg_vals, cols, w, effective_count, effective_count)

    def from_value(self, x: CopulaIFMStatistics) -> CopulaAccumulator:
        """Restore marginal stats and raw-column buffers from ``value`` output."""
        checked = _validate_ifm_statistics(x, self.dim)
        for i in range(self.dim):
            self.marginal_accumulators[i].from_value(checked.marginal_statistics[i])
        self._cols = [checked.columns] if len(checked.columns) else []
        self._w = [checked.weights] if len(checked.columns) else []
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed merges to marginal accumulators."""
        for acc in self.marginal_accumulators:
            if hasattr(acc, "key_merge"):
                acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed replacements to marginal accumulators."""
        for acc in self.marginal_accumulators:
            if hasattr(acc, "key_replace"):
                acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> CopulaDataEncoder:
        """Return an encoder composed from the marginal accumulator encoders."""
        return CopulaDataEncoder([acc.acc_to_encoder() for acc in self.marginal_accumulators])


class CopulaAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for IFM copula estimation."""

    def __init__(self, marginal_factories: Sequence[Any], dim: int, keys: str | None = None) -> None:
        self.marginal_factories = list(marginal_factories)
        self.dim = dim
        if len(self.marginal_factories) != self.dim:
            raise ValueError("copula accumulator factory marginal arity must equal dim.")
        self.keys = keys

    def make(self) -> CopulaAccumulator:
        """Create an empty copula accumulator."""
        return CopulaAccumulator([f.make() for f in self.marginal_factories], self.dim, keys=self.keys)


class CopulaEstimator(ParameterEstimator):
    """IFM estimator: fit each marginal from its sub-stats, PIT the buffered data, fit the copula on the scores."""

    def __init__(
        self,
        marginal_estimators: Sequence[ParameterEstimator],
        copula_estimator: ParameterEstimator,
        dim: int,
        copula_prototype: SequenceEncodableProbabilityDistribution,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.marginal_estimators = tuple(marginal_estimators)
        self.copula_estimator = copula_estimator
        self.dim = dim
        if len(self.marginal_estimators) != self.dim:
            raise ValueError("copula estimator marginal arity must equal dim.")
        if int(getattr(copula_prototype, "dim", -1)) != self.dim:
            raise ValueError("copula estimator prototype dimension must equal dim.")
        self.copula_prototype = copula_prototype
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> CopulaAccumulatorFactory:
        """Return a factory for IFM copula sufficient-statistic accumulators."""
        return CopulaAccumulatorFactory(
            [e.accumulator_factory() for e in self.marginal_estimators], self.dim, keys=self.keys
        )

    def estimate(
        self, nobs: float | None, suff_stat: CopulaIFMStatistics
    ) -> CopulaDistribution:
        """Estimate marginals, transform buffered data by PIT, and estimate the copula core."""
        checked = _validate_ifm_statistics(suff_stat, self.dim)
        marginals = [
            self.marginal_estimators[i].estimate(
                checked.marginal_effective_count,
                checked.marginal_statistics[i],
            )
            for i in range(self.dim)
        ]

        # IFM stage 2: PIT the buffered data through the freshly-fitted marginals, fit the copula on the scores.
        fitted = CopulaDistribution(marginals, self.copula_prototype, name=self.name, keys=self.keys)
        if len(checked.columns):
            u = fitted._pit_columns(checked.columns)
            cop_enc = fitted.copula.dist_to_encoder().seq_encode(u)
            cop_acc = self.copula_estimator.accumulator_factory().make()
            cop_acc.seq_update(cop_enc, checked.weights, None)
            copula = self.copula_estimator.estimate(checked.copula_effective_count, cop_acc.value())
        else:
            copula = fitted.copula
        return CopulaDistribution(marginals, copula, name=self.name, keys=self.keys)
