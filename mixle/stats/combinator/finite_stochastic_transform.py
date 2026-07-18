"""Finite stochastic-transform combinator: push a finite discrete source through a noisy channel.

``FiniteStochasticTransformDistribution`` models ``Y | X = x ~ Categorical(kernel[x, :])`` where the
source ``X ~ dist`` ranges over ``{0, ..., m-1}`` and only the output ``Y in {0, ..., n-1}`` is
observed.  It is the stochastic analogue of :class:`~mixle.stats.combinator.transform.TransformDistribution`:
the transform is a fixed ``m x n`` row-stochastic ``kernel`` (a confusion matrix / discrete channel)
rather than a deterministic invertible map, so the source marginal is

    P(Y = y) = sum_x P(X = x) * kernel[x, y].

Estimation recovers the source ``dist`` from observed ``Y`` *without inverting the channel*.  The
naive route -- solve ``kernel^T p_X = P_hat(Y)`` for ``p_X`` (an ``m x n`` (pseudo-)inverse, O(m^3),
ill-conditioned, and blind to the simplex / to a structured source) -- is short-circuited two ways:

1. The data's only sufficient statistic is the length-``n`` vector of output counts, so the work is
   collapsed from the ``N`` observations to the ``n`` distinct outputs up front (one ``bincount``).
2. The deconvolution is then a single E-step on those counts: the posterior responsibilities
   ``R[x, y] = P(X = x) kernel[x, y] / P(Y = y)`` give the expected source counts ``c = R @ n_y`` by a
   matrix-vector product (O(n*m), independent of ``N``), which are fed straight to the source's own
   estimator.  Iterating ``fit`` is EM / Richardson-Lucy deconvolution -- monotone, constraint-aware,
   and reusing whatever (possibly structured) estimator the source provides.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.stats.combinator._base import SingleChildAccumulator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _row_stochastic(kernel: Any) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2:
        raise ValueError("kernel must be a 2-D (num_source x num_output) stochastic matrix.")
    if np.any(k < -1.0e-12):
        raise ValueError("kernel entries must be non-negative.")
    k = np.clip(k, 0.0, None)
    row_sums = k.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("every kernel row must have positive mass.")
    return k / row_sums


class FiniteStochasticTransformDistribution(SequenceEncodableProbabilityDistribution):
    """Finite discrete source pushed through a fixed finite stochastic kernel (noisy channel)."""

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        kernel: Any,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a finite stochastic-transform distribution.

        Args:
            dist: Source distribution over the integer states ``{0, ..., m-1}`` (``m = kernel.shape[0]``).
            kernel: ``m x n`` row-stochastic matrix; ``kernel[x, y] = P(Y = y | X = x)``.
            name: Optional name for the instance.
            keys: Optional parameter key for shared estimation.
        """
        self.dist = dist
        self.kernel = _row_stochastic(kernel)
        self.num_source, self.num_output = self.kernel.shape
        self.name = name
        self.keys = keys
        with np.errstate(divide="ignore"):
            self.log_kernel = np.log(self.kernel)
            # Source log-pmf over the m states, and the resulting output marginal log P(Y=y).
            self._log_px = np.asarray([float(dist.log_density(x)) for x in range(self.num_source)], dtype=np.float64)
            self._log_py = logsumexp(self._log_px[:, None] + self.log_kernel, axis=0)

    def __str__(self) -> str:
        return "FiniteStochasticTransformDistribution(%s, %s, name=%s, keys=%s)" % (
            str(self.dist),
            repr([list(row) for row in self.kernel]),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: int) -> float:
        """Return ``P(Y = x)``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: int) -> float:
        """Return ``log P(Y = x)`` for an integer output ``x in {0, ..., n-1}``."""
        xi = int(x)
        if xi != x or xi < 0 or xi >= self.num_output:
            return -np.inf
        return float(self._log_py[xi])

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return per-observation ``log P(Y = y)`` for an encoded batch of outputs."""
        y, valid = x
        rv = np.full(y.shape[0], -np.inf, dtype=np.float64)
        rv[valid] = self._log_py[y[valid]]
        return rv

    def support_size(self) -> int:
        """Number of finite outputs ``{0, ..., n-1}``."""
        return int(self.num_output)

    def sampler(self, seed: int | None = None) -> "FiniteStochasticTransformSampler":
        """Return a sampler drawing ``X ~ dist`` then ``Y ~ Categorical(kernel[X])``."""
        return FiniteStochasticTransformSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "FiniteStochasticTransformEstimator":
        """Return an estimator that recovers the source via the channel-inversion-free E-step."""
        return FiniteStochasticTransformEstimator(
            self.dist.estimator(pseudo_count=pseudo_count), self.kernel, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "FiniteStochasticTransformDataEncoder":
        """Return the data encoder for integer outputs."""
        return FiniteStochasticTransformDataEncoder(self.num_output)

    def enumerator(self) -> "FiniteStochasticTransformEnumerator":
        """Enumerate the finite outputs ``y`` in descending ``P(Y = y)`` order."""
        return FiniteStochasticTransformEnumerator(self)


class FiniteStochasticTransformEnumerator(DistributionEnumerator):
    """Enumerate the finite output support in descending marginal probability."""

    def __init__(self, dist: FiniteStochasticTransformDistribution) -> None:
        super().__init__(dist)
        order = np.argsort(-dist._log_py, kind="stable")
        self._items = [(int(y), float(dist._log_py[y])) for y in order if dist._log_py[y] > -np.inf]
        self._i = 0

    def __next__(self) -> tuple[int, float]:
        if self._i >= len(self._items):
            raise StopIteration
        item = self._items[self._i]
        self._i += 1
        return item


class FiniteStochasticTransformSampler(DistributionSampler):
    """Sample ``X`` from the source, then ``Y`` from that source state's kernel row."""

    def __init__(self, dist: FiniteStochasticTransformDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.rng = RandomState(seed)
        self.source_sampler = dist.dist.sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one output (or a list of ``size`` outputs)."""
        if size is None:
            x = int(self.source_sampler.sample())
            return int(self.rng.choice(self.dist.num_output, p=self.dist.kernel[x]))
        return [self.sample() for _ in range(size)]


class FiniteStochasticTransformAccumulator(SingleChildAccumulator):
    """Accumulate expected source counts via one channel E-step over aggregated output counts."""

    _child_attr = "source_accumulator"

    def __init__(
        self,
        source_accumulator: SequenceEncodableStatisticAccumulator,
        kernel: np.ndarray,
        keys: str | None = None,
    ) -> None:
        self.source_accumulator = source_accumulator
        self.kernel = kernel
        self.log_kernel = np.log(np.clip(kernel, 1.0e-300, None))
        self.num_source, self.num_output = kernel.shape
        self.keys = keys

    def _source_log_px(self, estimate: FiniteStochasticTransformDistribution | None) -> np.ndarray:
        if estimate is None:
            # First pass / initialization: uniform source prior.
            return np.full(self.num_source, -np.log(self.num_source), dtype=np.float64)
        return estimate._log_px

    def _disperse(self, n_y: np.ndarray, source_estimate: Any, log_px: np.ndarray) -> None:
        """Posterior-distribute the output counts ``n_y`` to the source states and feed the child.

        ``R[x, y] = exp(log_px[x] + log_kernel[x, y]) / P(Y = y)``; the expected source counts are
        ``c = R @ n_y`` -- the channel deconvolution as one matrix-vector product, no inversion.
        """
        active = n_y > 0.0
        if not np.any(active):
            return
        joint = log_px[:, None] + self.log_kernel[:, active]  # (m, |active|)
        log_norm = logsumexp(joint, axis=0)
        resp = np.exp(joint - log_norm)  # posterior R[:, active], columns sum to 1
        counts = resp @ n_y[active]  # expected source count per state, length m
        for x in range(self.num_source):
            c = float(counts[x])
            if c > 0.0:
                self.source_accumulator.update(x, c, source_estimate)

    def update(self, x: Any, weight: float, estimate: FiniteStochasticTransformDistribution | None) -> None:
        """Accumulate one observed output by dispersing its weight to source states."""
        xi = int(x)
        if xi < 0 or xi >= self.num_output:
            return
        n_y = np.zeros(self.num_output, dtype=np.float64)
        n_y[xi] = float(weight)
        src = None if estimate is None else estimate.dist
        self._disperse(n_y, src, self._source_log_px(estimate))

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: FiniteStochasticTransformDistribution | None,
    ) -> None:
        """Accumulate encoded outputs through one aggregated channel E-step."""
        y, valid = x
        # Collapse the N observations to the length-n output-count vector (the sufficient statistic),
        # then run a single E-step on it -- the per-iteration cost is O(n*m), independent of N.
        n_y = np.bincount(y[valid], weights=np.asarray(weights, dtype=np.float64)[valid], minlength=self.num_output)
        src = None if estimate is None else estimate.dist
        self._disperse(n_y, src, self._source_log_px(estimate))

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize source sufficient statistics from one observed output."""
        self.update(x, weight, None)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize source sufficient statistics from encoded outputs."""
        self.seq_update(x, weights, None)

    def acc_to_encoder(self) -> "FiniteStochasticTransformDataEncoder":
        """Return an encoder for finite integer outputs."""
        return FiniteStochasticTransformDataEncoder(self.num_output)


class FiniteStochasticTransformAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`FiniteStochasticTransformAccumulator`."""

    def __init__(
        self, source_factory: StatisticAccumulatorFactory, kernel: np.ndarray, keys: str | None = None
    ) -> None:
        self.source_factory = source_factory
        self.kernel = kernel
        self.keys = keys

    def make(self) -> FiniteStochasticTransformAccumulator:
        """Create an empty finite-stochastic-transform accumulator."""
        return FiniteStochasticTransformAccumulator(self.source_factory.make(), self.kernel, keys=self.keys)


class FiniteStochasticTransformEstimator(ParameterEstimator):
    """Estimator for a fixed-kernel finite stochastic transform (recovers the source)."""

    def __init__(
        self,
        source_estimator: ParameterEstimator,
        kernel: Any,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.source_estimator = source_estimator
        self.kernel = _row_stochastic(kernel)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> FiniteStochasticTransformAccumulatorFactory:
        """Return a factory for finite-transform sufficient-statistic accumulators."""
        return FiniteStochasticTransformAccumulatorFactory(
            self.source_estimator.accumulator_factory(), self.kernel, keys=self.keys
        )

    def estimate(self, nobs: float | None, suff_stat: Any) -> FiniteStochasticTransformDistribution:
        """Estimate the source distribution and re-wrap it with the fixed kernel."""
        source = self.source_estimator.estimate(nobs, suff_stat)
        return FiniteStochasticTransformDistribution(source, self.kernel, name=self.name, keys=self.keys)


class FiniteStochasticTransformDataEncoder(DataSequenceEncoder):
    """Encode integer outputs as an index array plus an in-range validity mask."""

    def __init__(self, num_output: int) -> None:
        self.num_output = int(num_output)

    def __str__(self) -> str:
        return "FiniteStochasticTransformDataEncoder(num_output=%d)" % self.num_output

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FiniteStochasticTransformDataEncoder) and other.num_output == self.num_output

    def seq_encode(self, x: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Encode integer outputs with a validity mask for the finite output range."""
        y = np.asarray(x).astype(np.int64, copy=False)
        valid = (y >= 0) & (y < self.num_output)
        safe = np.where(valid, y, 0)
        return safe, valid
