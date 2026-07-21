"""Copula combinator: glue arbitrary marginal distributions to a dependence structure (Sklar's theorem).

``GaussianCopulaDistribution`` (``mixle.stats.multivariate``) models *only* dependence, on the unit cube
``(0,1)^d`` -- it assumes its inputs are already the uniform scores ``u_i``. This combinator is the piece
that makes copulas *composable*: give it your marginals (any mixle leaves exposing ``cdf`` -- a Gamma, a
StudentT, a VonMises, mixed freely) and a copula core, and it forms the joint

    f(x_1, ..., x_d) = c(F_1(x_1), ..., F_d(x_d)) * prod_i f_i(x_i)                       (Sklar)

where each ``F_i = marginals[i].cdf`` is the probability-integral transform (PIT) and ``c`` is the copula
density. That is the whole point of copulas: pick any marginals you like and couple them through one
dependence object, instead of hand-writing a bespoke multivariate leaf for every marginal combination.

Estimation is **IFM** (Inference Functions for Margins): fit each marginal on its own column, PIT the data
through the fitted marginals, then fit the copula on the uniform scores. This is exact in a single M-step
(the accumulator buffers the raw columns, the same pattern the neural leaves use), not an approximate
coupled iteration.

The copula core is pluggable: any distribution on ``(0,1)^d`` implementing the mixle five-piece contract
works. :class:`~mixle.stats.multivariate.gaussian_copula.GaussianCopulaDistribution` is the first supported
core; Clayton/Frank/t cores can be dropped in later with no change here.

Reference: Nelsen, *An Introduction to Copulas* (2nd ed., Springer, 2006); Joe, *Dependence Modeling with
Copulas* (CRC, 2014).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_CLIP = 1.0e-12  # keep PIT scores strictly inside (0,1) so the copula's Phi^{-1} stays finite


class CopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Joint over ``d`` scalar fields = ``d`` marginals coupled by a copula core (Sklar's theorem).

    ``marginals`` is a length-``d`` sequence of mixle leaves, each exposing ``log_density``, ``cdf``, a
    ``sampler()`` with ``sample()``, and the estimator/encoder contract. ``copula`` is a distribution on
    ``(0,1)^d`` (e.g. :class:`GaussianCopulaDistribution`). An observation is a length-``d`` tuple/array of
    scalars ``(x_1, ..., x_d)``.
    """

    def __init__(
        self,
        marginals: Sequence[SequenceEncodableProbabilityDistribution],
        copula: SequenceEncodableProbabilityDistribution,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.marginals = list(marginals)
        self.dim = len(self.marginals)
        if self.dim < 2:
            raise ValueError("CopulaDistribution needs at least 2 marginals; got %d" % self.dim)
        self.copula = copula
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "CopulaDistribution([%s], %s)" % (", ".join(map(str, self.marginals)), str(self.copula))

    def _pit_row(self, x: Sequence[float]) -> np.ndarray:
        """Probability-integral transform of one observation: ``u_i = clip(F_i(x_i))`` in ``(0,1)``."""
        u = np.array([float(self.marginals[i].cdf(x[i])) for i in range(self.dim)], dtype=np.float64)
        return np.clip(u, _CLIP, 1.0 - _CLIP)

    def _pit_columns(self, cols: np.ndarray) -> np.ndarray:
        """PIT an ``(n, d)`` array of raw observations to an ``(n, d)`` array of uniform scores."""
        cols = np.asarray(cols, dtype=np.float64)
        u = np.empty_like(cols)
        for i in range(self.dim):
            u[:, i] = [float(self.marginals[i].cdf(v)) for v in cols[:, i]]
        return np.clip(u, _CLIP, 1.0 - _CLIP)

    def density(self, x: Sequence[float]) -> float:
        """Return the joint density at one raw observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[float]) -> float:
        """Sklar's decomposition: sum of marginal log-densities + the copula log-density at the PIT scores."""
        marg = sum(float(self.marginals[i].log_density(x[i])) for i in range(self.dim))
        return marg + float(self.copula.log_density(self._pit_row(x)))

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Vectorized: per-column marginal log-densities (summed) plus the copula log-density at the PIT scores."""
        marg_encs, raw_cols = enc
        rv = self.marginals[0].seq_log_density(marg_encs[0])
        for i in range(1, self.dim):
            rv = rv + self.marginals[i].seq_log_density(marg_encs[i])
        u = self._pit_columns(raw_cols)
        cop_enc = self.copula.dist_to_encoder().seq_encode(u)
        return rv + np.asarray(self.copula.seq_log_density(cop_enc), dtype=np.float64)

    def sampler(self, seed: int | None = None) -> CopulaSampler:
        """Return a sampler that draws copula scores and inverts the marginals."""
        return CopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> CopulaEstimator:
        """Return an IFM estimator for marginals followed by the copula core."""
        return CopulaEstimator(
            [m.estimator() for m in self.marginals],
            self.copula.estimator(),
            dim=self.dim,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> CopulaDataEncoder:
        """Return an encoder that preserves marginal encodings and raw columns."""
        return CopulaDataEncoder([m.dist_to_encoder() for m in self.marginals])


class CopulaSampler(DistributionSampler):
    """Sample by drawing uniform scores from the copula, then inverting each marginal by its quantile.

    Requires each marginal's sampler to expose ``quantile`` OR the marginal itself to expose ``ppf``/``quantile``;
    falls back to inverse-CDF root finding on ``cdf`` when neither is present, so any ``cdf``-bearing marginal
    is sampleable.
    """

    def __init__(self, dist: CopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self._cop_sampler = dist.copula.sampler(seed if seed is None else seed + 1)

    def _invert(self, marginal: Any, u: float) -> float:
        for owner in (marginal, marginal.sampler()):
            for attr in ("quantile", "ppf", "inverse_cdf"):
                fn = getattr(owner, attr, None)
                if callable(fn):
                    return float(fn(u))
        return self._bisect_cdf(marginal, u)

    def _bisect_cdf(self, marginal: Any, u: float, lo: float = -1e6, hi: float = 1e6, iters: int = 100) -> float:
        # monotone bisection on the CDF -- a last-resort inverse for a marginal exposing only cdf
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if float(marginal.cdf(mid)) < u:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one joint observation or ``size`` iid observations."""
        n = 1 if size is None else int(size)
        u = np.atleast_2d(self._cop_sampler.sample(n)).reshape(n, self.dist.dim)
        out = [
            tuple(self._invert(self.dist.marginals[i], float(u[r, i])) for i in range(self.dist.dim)) for r in range(n)
        ]
        return out[0] if size is None else out


class CopulaDataEncoder(DataSequenceEncoder):
    """Encode a batch as ``(per-marginal encodings, raw (n, d) column array)``.

    The raw columns ride along because the copula's uniform scores ``u = F(x)`` depend on the marginals'
    *current* parameters (they change during fitting), so they cannot be baked in at encode time -- they are
    recomputed by :meth:`CopulaDistribution._pit_columns` against whatever distribution is scoring/estimating.
    """

    def __init__(self, marginal_encoders: Sequence[DataSequenceEncoder]) -> None:
        self.marginal_encoders = list(marginal_encoders)
        self.dim = len(self.marginal_encoders)

    def __str__(self) -> str:
        return "CopulaDataEncoder([%s])" % ", ".join(map(str, self.marginal_encoders))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CopulaDataEncoder) and self.marginal_encoders == other.marginal_encoders

    def seq_encode(self, x: Sequence[Sequence[float]]) -> tuple[tuple[Any, ...], np.ndarray]:
        """Encode each marginal column while retaining raw columns for PIT recomputation."""
        cols = np.asarray([[float(row[i]) for i in range(self.dim)] for row in x], dtype=np.float64)
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
        self.keys = keys
        self._cols: list[np.ndarray] = []
        self._w: list[np.ndarray] = []

    def update(self, x: Sequence[float], weight: float, estimate: CopulaDistribution | None) -> None:
        """Update marginal accumulators and buffer one raw observation for IFM copula fitting."""
        marg_est = estimate.marginals if estimate is not None else [None] * self.dim
        for i in range(self.dim):
            self.marginal_accumulators[i].update(x[i], weight, marg_est[i])
        self._cols.append(np.asarray([float(x[i]) for i in range(self.dim)], dtype=np.float64).reshape(1, self.dim))
        self._w.append(np.asarray([float(weight)], dtype=np.float64))

    def initialize(self, x: Sequence[float], weight: float, rng: np.random.RandomState | None) -> None:
        """Initialize marginal accumulators and buffer one raw observation."""
        for i in range(self.dim):
            self.marginal_accumulators[i].initialize(x[i], weight, rng)
        self._cols.append(np.asarray([float(x[i]) for i in range(self.dim)], dtype=np.float64).reshape(1, self.dim))
        self._w.append(np.asarray([float(weight)], dtype=np.float64))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: CopulaDistribution | None) -> None:
        """Update marginal accumulators and buffer encoded raw columns for IFM."""
        marg_encs, raw_cols = enc
        marg_est = estimate.marginals if estimate is not None else [None] * self.dim
        for i in range(self.dim):
            self.marginal_accumulators[i].seq_update(marg_encs[i], weights, marg_est[i])
        self._cols.append(np.asarray(raw_cols, dtype=np.float64).reshape(-1, self.dim))
        self._w.append(np.asarray(weights, dtype=np.float64).ravel())

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: np.random.RandomState | None) -> None:
        """Initialize marginal accumulators and buffer encoded raw columns."""
        marg_encs, raw_cols = enc
        for i in range(self.dim):
            self.marginal_accumulators[i].seq_initialize(marg_encs[i], weights, rng)
        self._cols.append(np.asarray(raw_cols, dtype=np.float64).reshape(-1, self.dim))
        self._w.append(np.asarray(weights, dtype=np.float64).ravel())

    def combine(self, suff_stat: tuple[tuple[Any, ...], np.ndarray, np.ndarray]) -> CopulaAccumulator:
        """Merge marginal sufficient statistics and buffered raw columns."""
        marg_stats, cols, w = suff_stat
        for i in range(self.dim):
            self.marginal_accumulators[i].combine(marg_stats[i])
        if len(cols):
            self._cols.append(np.asarray(cols, dtype=np.float64).reshape(-1, self.dim))
            self._w.append(np.asarray(w, dtype=np.float64).ravel())
        return self

    def value(self) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
        """Return marginal stats, buffered raw columns, and buffered weights."""
        marg_vals = tuple(acc.value() for acc in self.marginal_accumulators)
        cols = np.concatenate(self._cols, axis=0) if self._cols else np.zeros((0, self.dim))
        w = np.concatenate(self._w) if self._w else np.zeros((0,))
        return marg_vals, cols, w

    def from_value(self, x: tuple[tuple[Any, ...], np.ndarray, np.ndarray]) -> CopulaAccumulator:
        """Restore marginal stats and raw-column buffers from ``value`` output."""
        marg_vals, cols, w = x
        for i in range(self.dim):
            self.marginal_accumulators[i].from_value(marg_vals[i])
        cols = np.asarray(cols, dtype=np.float64).reshape(-1, self.dim)
        self._cols = [cols] if len(cols) else []
        self._w = [np.asarray(w, dtype=np.float64).ravel()] if len(cols) else []
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
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.marginal_estimators = list(marginal_estimators)
        self.copula_estimator = copula_estimator
        self.dim = dim
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> CopulaAccumulatorFactory:
        """Return a factory for IFM copula sufficient-statistic accumulators."""
        return CopulaAccumulatorFactory(
            [e.accumulator_factory() for e in self.marginal_estimators], self.dim, keys=self.keys
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[tuple[Any, ...], np.ndarray, np.ndarray]
    ) -> CopulaDistribution:
        """Estimate marginals, transform buffered data by PIT, and estimate the copula core."""
        marg_stats, cols, w = suff_stat
        marginals = [self.marginal_estimators[i].estimate(nobs, marg_stats[i]) for i in range(self.dim)]

        # IFM stage 2: PIT the buffered data through the freshly-fitted marginals, fit the copula on the scores.
        fitted = CopulaDistribution(marginals, self.copula_estimator_prototype(), name=self.name, keys=self.keys)
        if len(cols):
            u = fitted._pit_columns(np.asarray(cols, dtype=np.float64))
            cop_enc = fitted.copula.dist_to_encoder().seq_encode(u)
            cop_acc = self.copula_estimator.accumulator_factory().make()
            cop_acc.seq_update(cop_enc, np.asarray(w, dtype=np.float64).ravel(), None)
            copula = self.copula_estimator.estimate(nobs, cop_acc.value())
        else:
            copula = fitted.copula
        return CopulaDistribution(marginals, copula, name=self.name, keys=self.keys)

    def copula_estimator_prototype(self) -> SequenceEncodableProbabilityDistribution:
        """A copula instance usable for PIT-encoding before the copula is refit -- the estimator's own default."""
        # estimate() with an empty accumulator returns the copula's default (e.g. identity-correlation Gaussian).
        empty = self.copula_estimator.accumulator_factory().make()
        return self.copula_estimator.estimate(0.0, empty.value())
