"""Laplace distributions over real values.

Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    idx = np.argsort(values)
    xv = values[idx]
    wv = weights[idx]
    cutoff = 0.5 * np.sum(wv)
    return float(xv[np.searchsorted(np.cumsum(wv), cutoff, side="left")])


class LaplaceDistribution(SequenceEncodableProbabilityDistribution):
    """Laplace distribution with location mu and scale b > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Laplace kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Laplace distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="laplace",
            distribution_type=cls,
            parameters=(ParameterSpec("mu"), ParameterSpec("b", constraint="positive")),
            statistics=(
                StatisticSpec("values", kind="raw_observations", scales=False),
                StatisticSpec("weights", kind="weights"),
            ),
            support="real",
        )

    def __init__(self, mu: float, b: float, name: str | None = None, keys: str | None = None) -> None:
        if b <= 0.0 or not np.isfinite(b):
            raise ValueError("LaplaceDistribution requires b > 0.")
        self.mu = float(mu)
        self.b = float(b)
        self.log_const = -math.log(2.0 * self.b)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "LaplaceDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.b),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        return self.log_const - abs(x - self.mu) / self.b

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return self.log_const - np.abs(x - self.mu) / self.b

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, b: Any, engine: Any) -> Any:
        """Engine-neutral Laplace log-density from explicit parameters."""
        return -engine.log(engine.asarray(2.0) * b) - engine.abs(x - mu) / b

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.mu), engine.asarray(self.b), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["LaplaceDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Laplace parameters for a homogeneous mixture kernel."""
        return {
            "mu": engine.asarray([d.mu for d in dists]),
            "b": engine.asarray([d.b for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Laplace log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params["mu"][None, :], params["b"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[tuple[Any, Any], ...]:
        """Return per-component raw weighted observations using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        counts = engine.to_numpy(engine.sum(ww, axis=0))
        rv = []
        for i in range(len(np.asarray(counts))):
            w_loc = ww[:, i]
            mask = w_loc > 0.0
            rv.append((xx[mask], w_loc[mask]))
        return tuple(rv)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import laplace as _sp

        return float(_sp.cdf(x, loc=self.mu, scale=self.b))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import laplace as _sp

        return float(_sp.ppf(q, loc=self.mu, scale=self.b))

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.mu)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(2.0 * self.b * self.b)

    def entropy(self) -> float:
        """Differential entropy 1 + log(2b)."""
        import math

        return float(1.0 + math.log(2.0 * self.b))

    def skewness(self) -> float:
        """Skewness (0)."""
        return 0.0

    def kurtosis(self) -> float:
        """Excess kurtosis (3)."""
        return 3.0

    def mode(self) -> float:
        """Mode (= the location mu)."""
        return float(self.mu)

    def sampler(self, seed: int | None = None) -> "LaplaceSampler":
        """Return a sampler for drawing observations from this distribution."""
        return LaplaceSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LaplaceEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return LaplaceEstimator(name=self.name, keys=self.keys)
        return LaplaceEstimator(pseudo_count=pseudo_count, suff_stat=(self.mu, self.b), name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "LaplaceDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return LaplaceDataEncoder()


class LaplaceSampler(DistributionSampler):
    """Draw iid Laplace observations."""

    def __init__(self, dist: LaplaceDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.rng.laplace(loc=self.dist.mu, scale=self.dist.b, size=size)


class LaplaceAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted observations for exact weighted-median M-step."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.values = []
        self.weights = []
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: LaplaceDistribution | None) -> None:
        """Store one positively weighted observation for the weighted-median M-step."""
        if weight > 0.0:
            self.values.append(float(x))
            self.weights.append(float(weight))

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: LaplaceDistribution | None) -> None:
        """Store positively weighted encoded observations for estimation."""
        mask = weights > 0.0
        if np.any(mask):
            self.values.append(np.asarray(x[mask], dtype=np.float64))
            self.weights.append(np.asarray(weights[mask], dtype=np.float64))

    def seq_update_engine(self, x: np.ndarray, weights: Any, estimate: LaplaceDistribution | None, engine: Any) -> None:
        """Engine-aware accumulation. Laplace's MLE is a weighted median, so the sufficient
        statistic is the (positively weighted) data itself; this path accepts engine (e.g. torch)
        weights and stores host arrays. Matches seq_update.
        """
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        mask = weights_np > 0.0
        if np.any(mask):
            self.values.append(np.asarray(np.asarray(x)[mask], dtype=np.float64))
            self.weights.append(weights_np[mask])

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    @staticmethod
    def _flatten(items) -> np.ndarray:
        if len(items) == 0:
            return np.asarray([], dtype=np.float64)
        return np.concatenate([np.asarray(u, dtype=np.float64).reshape(-1) for u in items])

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray]) -> "LaplaceAccumulator":
        """Merge raw weighted observations from another accumulator."""
        if len(suff_stat[0]):
            self.values.append(suff_stat[0])
            self.weights.append(suff_stat[1])
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened observations and weights."""
        return self._flatten(self.values), self._flatten(self.weights)

    def from_value(self, x: tuple[np.ndarray, np.ndarray]) -> "LaplaceAccumulator":
        """Replace accumulator contents from raw observations and weights."""
        self.values = [np.asarray(x[0], dtype=np.float64)]
        self.weights = [np.asarray(x[1], dtype=np.float64)]
        return self

    def scale(self, c: float) -> "LaplaceAccumulator":
        """Scale weights while preserving the raw observation payload."""
        self.weights = [np.asarray(w, dtype=np.float64) * c for w in self.weights]
        return self

    def acc_to_encoder(self) -> "LaplaceDataEncoder":
        """Return the encoder used by this accumulator."""
        return LaplaceDataEncoder()


class LaplaceAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LaplaceAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> LaplaceAccumulator:
        """Create a fresh Laplace accumulator."""
        return LaplaceAccumulator(name=self.name, keys=self.keys)


class LaplaceEstimator(ParameterEstimator):
    """Exact weighted-MLE estimator for Laplace location and scale."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_scale: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LaplaceAccumulatorFactory:
        """Return an accumulator factory for Laplace raw-observation statistics."""
        return LaplaceAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> LaplaceDistribution:
        """Estimate location and scale by the exact weighted MLE."""
        values, weights = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            mu0, _ = self.suff_stat
            values = np.concatenate([values, np.asarray([mu0])])
            weights = np.concatenate([weights, np.asarray([self.pseudo_count])])
        if len(values) == 0 or weights.sum() <= 0.0:
            return LaplaceDistribution(0.0, 1.0, name=self.name, keys=self.keys)
        mu = _weighted_median(values, weights)
        b = np.dot(np.abs(values - mu), weights)
        if self.pseudo_count is not None and self.suff_stat is not None:
            b += self.pseudo_count * self.suff_stat[1]
        b /= weights.sum()
        return LaplaceDistribution(mu, max(float(b), self.min_scale), name=self.name, keys=self.keys)


class LaplaceDataEncoder(DataSequenceEncoder):
    """Encode Laplace observations as a float array."""

    def __str__(self) -> str:
        return "LaplaceDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LaplaceDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError("LaplaceDistribution requires finite or infinite real-valued observations.")
        return rv
