"""Create, estimate, and sample from a Rayleigh distribution."""
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


class RayleighDistribution(SequenceEncodableProbabilityDistribution):
    """Rayleigh distribution with scale sigma > 0."""

    def __init__(self, sigma: float, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError('RayleighDistribution requires sigma > 0.')
        self.sigma = float(sigma)
        self.sigma2 = self.sigma * self.sigma
        self.log_sigma2 = math.log(self.sigma2)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'RayleighDistribution(%s, name=%s, keys=%s)' % (
            repr(self.sigma), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        if x < 0.0:
            return -np.inf
        if x == 0.0:
            return -np.inf
        return math.log(x) - self.log_sigma2 - x * x / (2.0 * self.sigma2)

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        xx, xx2, lx = x
        rv = lx - self.log_sigma2 - xx2 / (2.0 * self.sigma2)
        return np.where(xx >= 0.0, rv, -np.inf)

    def sampler(self, seed: Optional[int] = None) -> 'RayleighSampler':
        return RayleighSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'RayleighEstimator':
        if pseudo_count is None:
            return RayleighEstimator(name=self.name, keys=self.keys)
        return RayleighEstimator(pseudo_count=pseudo_count, suff_stat=self.sigma,
                                 name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'RayleighDataEncoder':
        return RayleighDataEncoder()


class RayleighSampler(DistributionSampler):
    """Draw iid Rayleigh observations."""

    def __init__(self, dist: RayleighDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return self.rng.rayleigh(scale=self.dist.sigma, size=size)


class RayleighAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted squared observations."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.count = 0.0
        self.sum2 = 0.0
        self.name = name
        self.key = keys

    def update(self, x: float, weight: float, estimate: Optional[RayleighDistribution]) -> None:
        if x < 0.0:
            raise ValueError('RayleighDistribution requires observations x >= 0.')
        self.count += weight
        self.sum2 += x * x * weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional[RayleighDistribution]) -> None:
        self.count += np.sum(weights, dtype=np.float64)
        self.sum2 += np.dot(x[1], weights)

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray,
                       rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'RayleighAccumulator':
        self.count += suff_stat[0]
        self.sum2 += suff_stat[1]
        return self

    def value(self) -> Tuple[float, float]:
        return self.count, self.sum2

    def from_value(self, x: Tuple[float, float]) -> 'RayleighAccumulator':
        self.count = x[0]
        self.sum2 = x[1]
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

    def acc_to_encoder(self) -> 'RayleighDataEncoder':
        return RayleighDataEncoder()


class RayleighAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for RayleighAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> RayleighAccumulator:
        return RayleighAccumulator(name=self.name, keys=self.keys)


class RayleighEstimator(ParameterEstimator):
    """Closed-form MLE estimator for Rayleigh scale."""

    def __init__(self, pseudo_count: Optional[float] = None, suff_stat: Optional[float] = None,
                 min_sigma: float = 1.0e-8, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_sigma = min_sigma
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> RayleighAccumulatorFactory:
        return RayleighAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> RayleighDistribution:
        count, sum2 = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            sum2 += self.pseudo_count * 2.0 * self.suff_stat * self.suff_stat
            count += self.pseudo_count
        sigma = math.sqrt(sum2 / (2.0 * count)) if count > 0.0 else 1.0
        return RayleighDistribution(max(sigma, self.min_sigma), name=self.name, keys=self.keys)


class RayleighDataEncoder(DataSequenceEncoder):
    """Encode Rayleigh observations with x, x**2, and log(x)."""

    def __str__(self) -> str:
        return 'RayleighDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RayleighDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 0.0) or np.any(np.isnan(rv))):
            raise ValueError('RayleighDistribution requires observations x >= 0.')
        with np.errstate(divide='ignore'):
            lx = np.log(rv)
        return rv, rv * rv, lx
