"""Create, estimate, and sample from a continuous uniform distribution."""
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


class UniformDistribution(SequenceEncodableProbabilityDistribution):
    """Continuous uniform distribution on [low, high]."""

    def __init__(self, low: float, high: float, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        if high <= low or not np.isfinite(low) or not np.isfinite(high):
            raise ValueError('UniformDistribution requires finite low < high.')
        self.low = float(low)
        self.high = float(high)
        self.log_density_value = -math.log(self.high - self.low)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'UniformDistribution(%s, %s, name=%s, keys=%s)' % (
            repr(self.low), repr(self.high), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        return self.log_density_value if self.low <= x <= self.high else -np.inf

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        return np.where((x >= self.low) & (x <= self.high), self.log_density_value, -np.inf)

    def sampler(self, seed: Optional[int] = None) -> 'UniformSampler':
        return UniformSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'UniformEstimator':
        if pseudo_count is None:
            return UniformEstimator(name=self.name, keys=self.keys)
        return UniformEstimator(pseudo_count=pseudo_count, suff_stat=(self.low, self.high),
                                name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'UniformDataEncoder':
        return UniformDataEncoder()


class UniformSampler(DistributionSampler):
    """Draw iid uniform observations."""

    def __init__(self, dist: UniformDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return self.rng.uniform(self.dist.low, self.dist.high, size=size)


class UniformAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted min/max support statistics."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.count = 0.0
        self.min_val = np.inf
        self.max_val = -np.inf
        self.name = name
        self.key = keys

    def update(self, x: float, weight: float, estimate: Optional[UniformDistribution]) -> None:
        if weight > 0.0:
            self.count += weight
            self.min_val = min(self.min_val, x)
            self.max_val = max(self.max_val, x)

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[UniformDistribution]) -> None:
        mask = weights > 0.0
        if np.any(mask):
            self.count += np.sum(weights[mask], dtype=np.float64)
            self.min_val = min(self.min_val, float(np.min(x[mask])))
            self.max_val = max(self.max_val, float(np.max(x[mask])))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'UniformAccumulator':
        self.count += suff_stat[0]
        self.min_val = min(self.min_val, suff_stat[1])
        self.max_val = max(self.max_val, suff_stat[2])
        return self

    def value(self) -> Tuple[float, float, float]:
        return self.count, self.min_val, self.max_val

    def from_value(self, x: Tuple[float, float, float]) -> 'UniformAccumulator':
        self.count = x[0]
        self.min_val = x[1]
        self.max_val = x[2]
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

    def acc_to_encoder(self) -> 'UniformDataEncoder':
        return UniformDataEncoder()


class UniformAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for UniformAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> UniformAccumulator:
        return UniformAccumulator(name=self.name, keys=self.keys)


class UniformEstimator(ParameterEstimator):
    """MLE estimator for uniform support endpoints."""

    def __init__(self, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[float, float]] = None,
                 min_width: float = 1.0e-8, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_width = min_width
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> UniformAccumulatorFactory:
        return UniformAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float]) -> UniformDistribution:
        count, low, high = suff_stat
        if count <= 0.0:
            low, high = self.suff_stat if self.suff_stat is not None else (0.0, 1.0)
        elif self.pseudo_count is not None and self.suff_stat is not None:
            low = min(low, self.suff_stat[0])
            high = max(high, self.suff_stat[1])
        if high <= low:
            mid = 0.5 * (low + high)
            low = mid - 0.5 * self.min_width
            high = mid + 0.5 * self.min_width
        return UniformDistribution(low, high, name=self.name, keys=self.keys)


class UniformDataEncoder(DataSequenceEncoder):
    """Encode uniform observations as a float array."""

    def __str__(self) -> str:
        return 'UniformDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UniformDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError('UniformDistribution requires finite or infinite real-valued observations.')
        return rv
