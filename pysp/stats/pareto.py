"""Create, estimate, and sample from a Pareto type-I distribution."""
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


class ParetoDistribution(SequenceEncodableProbabilityDistribution):
    """Pareto type-I distribution with scale xm > 0 and shape alpha > 0."""

    def __init__(self, xm: float, alpha: float, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        if xm <= 0.0 or alpha <= 0.0 or not np.isfinite(xm) or not np.isfinite(alpha):
            raise ValueError('ParetoDistribution requires xm > 0 and alpha > 0.')
        self.xm = float(xm)
        self.alpha = float(alpha)
        self.log_xm = math.log(self.xm)
        self.log_alpha = math.log(self.alpha)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'ParetoDistribution(%s, %s, name=%s, keys=%s)' % (
            repr(self.xm), repr(self.alpha), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        try:
            xx = float(x)
        except Exception:
            return -np.inf
        if not np.isfinite(xx) or xx < self.xm:
            return -np.inf
        return self.log_alpha + self.alpha * self.log_xm - (self.alpha + 1.0) * math.log(xx)

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        xx, lx = x
        rv = self.log_alpha + self.alpha * self.log_xm - (self.alpha + 1.0) * lx
        return np.where(xx >= self.xm, rv, -np.inf)

    def sampler(self, seed: Optional[int] = None) -> 'ParetoSampler':
        """Return a sampler for drawing observations from this distribution."""
        return ParetoSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'ParetoEstimator':
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return ParetoEstimator(name=self.name, keys=self.keys)
        return ParetoEstimator(pseudo_count=pseudo_count, suff_stat=(self.xm, self.alpha),
                               name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'ParetoDataEncoder':
        """Return the data encoder used by this distribution for vectorized methods."""
        return ParetoDataEncoder()


class ParetoSampler(DistributionSampler):
    """Draw iid Pareto observations."""

    def __init__(self, dist: ParetoDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return self.dist.xm * (self.rng.pareto(self.dist.alpha, size=size) + 1.0)


class ParetoAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted support minimum and log-sum statistics."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.count = 0.0
        self.sum_of_logs = 0.0
        self.min_val = np.inf
        self.name = name
        self.key = keys

    def update(self, x: float, weight: float, estimate: Optional[ParetoDistribution]) -> None:
        if x <= 0.0:
            raise ValueError('ParetoDistribution requires observations x > 0.')
        if weight > 0.0:
            self.count += weight
            self.sum_of_logs += math.log(x) * weight
            self.min_val = min(self.min_val, x)

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional[ParetoDistribution]) -> None:
        xx, lx = x
        mask = weights > 0.0
        if np.any(mask):
            self.count += np.sum(weights[mask], dtype=np.float64)
            self.sum_of_logs += np.dot(lx, weights)
            self.min_val = min(self.min_val, float(np.min(xx[mask])))

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray,
                       rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'ParetoAccumulator':
        self.count += suff_stat[0]
        self.sum_of_logs += suff_stat[1]
        self.min_val = min(self.min_val, suff_stat[2])
        return self

    def value(self) -> Tuple[float, float, float]:
        return self.count, self.sum_of_logs, self.min_val

    def from_value(self, x: Tuple[float, float, float]) -> 'ParetoAccumulator':
        self.count = x[0]
        self.sum_of_logs = x[1]
        self.min_val = x[2]
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

    def acc_to_encoder(self) -> 'ParetoDataEncoder':
        return ParetoDataEncoder()


class ParetoAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ParetoAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> ParetoAccumulator:
        return ParetoAccumulator(name=self.name, keys=self.keys)


class ParetoEstimator(ParameterEstimator):
    """MLE estimator for Pareto scale and shape."""

    def __init__(self, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[float, float]] = None,
                 min_denom: float = 1.0e-12, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_denom = min_denom
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ParetoAccumulatorFactory:
        return ParetoAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float]) -> ParetoDistribution:
        count, sum_logs, xm = suff_stat
        if count <= 0.0:
            xm, alpha = self.suff_stat if self.suff_stat is not None else (1.0, 1.0)
            return ParetoDistribution(xm, alpha, name=self.name, keys=self.keys)

        if self.pseudo_count is not None and self.suff_stat is not None:
            xm0, alpha0 = self.suff_stat
            xm = min(xm, xm0)
            sum_logs += self.pseudo_count * (math.log(xm0) + 1.0 / alpha0)
            count += self.pseudo_count

        denom = max(sum_logs - count * math.log(xm), self.min_denom)
        alpha = count / denom
        return ParetoDistribution(xm, alpha, name=self.name, keys=self.keys)


class ParetoDataEncoder(DataSequenceEncoder):
    """Encode Pareto observations with x and log(x)."""

    def __str__(self) -> str:
        return 'ParetoDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ParetoDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv <= 0.0) or np.any(np.isnan(rv))):
            raise ValueError('ParetoDistribution requires observations x > 0.')
        return rv, np.log(rv)
