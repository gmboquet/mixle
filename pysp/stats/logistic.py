"""Create, estimate, and sample from a location-scale logistic distribution."""
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


class LogisticDistribution(SequenceEncodableProbabilityDistribution):
    """Logistic distribution with location loc and scale > 0."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError('LogisticDistribution requires scale > 0.')
        self.loc = float(loc)
        self.scale = float(scale)
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'LogisticDistribution(loc=%s, scale=%s, name=%s, keys=%s)' % (
            repr(self.loc), repr(self.scale), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        z = (x - self.loc) / self.scale
        return -self.log_scale - z - 2.0 * float(np.logaddexp(0.0, -z))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        z = (x - self.loc) / self.scale
        return -self.log_scale - z - 2.0 * np.logaddexp(0.0, -z)

    def sampler(self, seed: Optional[int] = None) -> 'LogisticSampler':
        """Return a sampler for drawing observations from this distribution."""
        return LogisticSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'LogisticEstimator':
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return LogisticEstimator(name=self.name, keys=self.keys)
        return LogisticEstimator(pseudo_count=pseudo_count,
                                 suff_stat=(self.loc, self.scale),
                                 name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'LogisticDataEncoder':
        """Return the data encoder used by this distribution for vectorized methods."""
        return LogisticDataEncoder()


class LogisticSampler(DistributionSampler):
    """Draw iid logistic observations."""

    def __init__(self, dist: LogisticDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return self.rng.logistic(loc=self.dist.loc, scale=self.dist.scale, size=size)


class LogisticAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for logistic estimation."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.key = keys

    def update(self, x: float, weight: float, estimate: Optional[LogisticDistribution]) -> None:
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray,
                   estimate: Optional[LogisticDistribution]) -> None:
        self.sum += np.dot(x, weights)
        self.sum2 += np.dot(x * x, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'LogisticAccumulator':
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> Tuple[float, float, float]:
        return self.sum, self.sum2, self.count

    def from_value(self, x: Tuple[float, float, float]) -> 'LogisticAccumulator':
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
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

    def acc_to_encoder(self) -> 'LogisticDataEncoder':
        return LogisticDataEncoder()


class LogisticAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LogisticAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> LogisticAccumulator:
        return LogisticAccumulator(name=self.name, keys=self.keys)


class LogisticEstimator(ParameterEstimator):
    """Moment estimator for logistic location and scale.

    The likelihood MLE has no closed-form M-step. The EM estimator uses the
    identities mean=loc and var=pi^2 scale^2 / 3; torch gradient MLE can refine
    both parameters when exact likelihood optimization is desired.
    """

    def __init__(self, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[float, float]] = None,
                 min_scale: float = 1.0e-8, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LogisticAccumulatorFactory:
        return LogisticAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float],
                 suff_stat: Tuple[float, float, float]) -> LogisticDistribution:
        sum_x, sum_x2, count = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            loc0, scale0 = self.suff_stat
            var0 = (math.pi * math.pi / 3.0) * scale0 * scale0
            sum_x += self.pseudo_count * loc0
            sum_x2 += self.pseudo_count * (var0 + loc0 * loc0)
            count += self.pseudo_count

        if count <= 0.0:
            return LogisticDistribution(name=self.name, keys=self.keys)

        loc = sum_x / count
        var = max(sum_x2 / count - loc * loc, 0.0)
        scale = math.sqrt(max(3.0 * var / (math.pi * math.pi),
                              self.min_scale * self.min_scale))
        return LogisticDistribution(loc=loc, scale=scale, name=self.name, keys=self.keys)


class LogisticDataEncoder(DataSequenceEncoder):
    """Encode logistic observations as a float array."""

    def __str__(self) -> str:
        return 'LogisticDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogisticDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError('LogisticDistribution requires finite or infinite real-valued observations.')
        return rv
