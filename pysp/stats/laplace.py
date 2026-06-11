"""Create, estimate, and sample from a Laplace distribution."""
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState

from pysp.stats.pdist import (
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
    return float(xv[np.searchsorted(np.cumsum(wv), cutoff, side='left')])


class LaplaceDistribution(SequenceEncodableProbabilityDistribution):
    """Laplace distribution with location mu and scale b > 0."""

    def __init__(self, mu: float, b: float, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        if b <= 0.0 or not np.isfinite(b):
            raise ValueError('LaplaceDistribution requires b > 0.')
        self.mu = float(mu)
        self.b = float(b)
        self.log_const = -math.log(2.0 * self.b)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'LaplaceDistribution(%s, %s, name=%s, keys=%s)' % (
            repr(self.mu), repr(self.b), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        return self.log_const - abs(x - self.mu) / self.b

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        return self.log_const - np.abs(x - self.mu) / self.b

    def sampler(self, seed: Optional[int] = None) -> 'LaplaceSampler':
        return LaplaceSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'LaplaceEstimator':
        if pseudo_count is None:
            return LaplaceEstimator(name=self.name, keys=self.keys)
        return LaplaceEstimator(pseudo_count=pseudo_count, suff_stat=(self.mu, self.b),
                                name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'LaplaceDataEncoder':
        return LaplaceDataEncoder()


class LaplaceSampler(DistributionSampler):
    """Draw iid Laplace observations."""

    def __init__(self, dist: LaplaceDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return self.rng.laplace(loc=self.dist.mu, scale=self.dist.b, size=size)


class LaplaceAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted observations for exact weighted-median M-step."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.values = []
        self.weights = []
        self.name = name
        self.key = keys

    def update(self, x: float, weight: float, estimate: Optional[LaplaceDistribution]) -> None:
        if weight > 0.0:
            self.values.append(float(x))
            self.weights.append(float(weight))

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[LaplaceDistribution]) -> None:
        mask = weights > 0.0
        if np.any(mask):
            self.values.append(np.asarray(x[mask], dtype=np.float64))
            self.weights.append(np.asarray(weights[mask], dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    @staticmethod
    def _flatten(items) -> np.ndarray:
        if len(items) == 0:
            return np.asarray([], dtype=np.float64)
        return np.concatenate([np.asarray(u, dtype=np.float64).reshape(-1) for u in items])

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray]) -> 'LaplaceAccumulator':
        if len(suff_stat[0]):
            self.values.append(suff_stat[0])
            self.weights.append(suff_stat[1])
        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._flatten(self.values), self._flatten(self.weights)

    def from_value(self, x: Tuple[np.ndarray, np.ndarray]) -> 'LaplaceAccumulator':
        self.values = [np.asarray(x[0], dtype=np.float64)]
        self.weights = [np.asarray(x[1], dtype=np.float64)]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'LaplaceDataEncoder':
        return LaplaceDataEncoder()


class LaplaceAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LaplaceAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> LaplaceAccumulator:
        return LaplaceAccumulator(name=self.name, keys=self.keys)


class LaplaceEstimator(ParameterEstimator):
    """Exact weighted-MLE estimator for Laplace location and scale."""

    def __init__(self, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[float, float]] = None,
                 min_scale: float = 1.0e-8, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LaplaceAccumulatorFactory:
        return LaplaceAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray]) -> LaplaceDistribution:
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
        return 'LaplaceDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LaplaceDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError('LaplaceDistribution requires finite or infinite real-valued observations.')
        return rv
