"""Create, estimate, and sample from a location-scale Student's t distribution."""
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
from pysp.utils.special import gammaln


class StudentTDistribution(SequenceEncodableProbabilityDistribution):
    """Student's t distribution with degrees of freedom df, location loc, and scale > 0."""

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        if df <= 0.0 or scale <= 0.0 or not np.isfinite(df) or not np.isfinite(scale):
            raise ValueError('StudentTDistribution requires df > 0 and scale > 0.')
        self.df = float(df)
        self.loc = float(loc)
        self.scale = float(scale)
        self.log_const = float(gammaln((self.df + 1.0) / 2.0) - gammaln(self.df / 2.0)
                               - 0.5 * math.log(self.df * math.pi) - math.log(self.scale))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'StudentTDistribution(%s, loc=%s, scale=%s, name=%s, keys=%s)' % (
            repr(self.df), repr(self.loc), repr(self.scale), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return self.log_const - 0.5 * (self.df + 1.0) * math.log1p((z * z) / self.df)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.loc) / self.scale
        return self.log_const - 0.5 * (self.df + 1.0) * np.log1p((z * z) / self.df)

    def sampler(self, seed: Optional[int] = None) -> 'StudentTSampler':
        return StudentTSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'StudentTEstimator':
        if pseudo_count is None:
            return StudentTEstimator(df=self.df, name=self.name, keys=self.keys)
        return StudentTEstimator(df=self.df, pseudo_count=pseudo_count,
                                 suff_stat=(self.loc, self.scale),
                                 name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'StudentTDataEncoder':
        return StudentTDataEncoder()


class StudentTSampler(DistributionSampler):
    """Draw iid Student's t observations."""

    def __init__(self, dist: StudentTDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return self.rng.standard_t(self.dist.df, size=size) * self.dist.scale + self.dist.loc


class StudentTAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for fixed-df t estimation."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.key = keys

    def update(self, x: float, weight: float, estimate: Optional[StudentTDistribution]) -> None:
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[StudentTDistribution]) -> None:
        self.sum += np.dot(x, weights)
        self.sum2 += np.dot(x * x, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'StudentTAccumulator':
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> Tuple[float, float, float]:
        return self.sum, self.sum2, self.count

    def from_value(self, x: Tuple[float, float, float]) -> 'StudentTAccumulator':
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

    def acc_to_encoder(self) -> 'StudentTDataEncoder':
        return StudentTDataEncoder()


class StudentTAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for StudentTAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> StudentTAccumulator:
        return StudentTAccumulator(name=self.name, keys=self.keys)


class StudentTEstimator(ParameterEstimator):
    """Moment-style fixed-df estimator for Student's t location and scale.

    The exact MLE has no simple closed-form update. This estimator keeps df fixed
    and uses weighted moments, while gradient optimizers can fit all three
    parameters through pysp.stats.torch_engine.
    """

    def __init__(self, df: float = 5.0, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[float, float]] = None,
                 min_scale: float = 1.0e-8, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        if df <= 0.0 or not np.isfinite(df):
            raise ValueError('StudentTEstimator requires df > 0.')
        self.df = float(df)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StudentTAccumulatorFactory:
        return StudentTAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float]) -> StudentTDistribution:
        sum_x, sum_x2, count = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            loc0, scale0 = self.suff_stat
            var0 = scale0 * scale0 * self.df / (self.df - 2.0) if self.df > 2.0 else scale0 * scale0
            sum_x += self.pseudo_count * loc0
            sum_x2 += self.pseudo_count * (var0 + loc0 * loc0)
            count += self.pseudo_count

        if count <= 0.0:
            return StudentTDistribution(self.df, name=self.name, keys=self.keys)

        loc = sum_x / count
        var = max(sum_x2 / count - loc * loc, self.min_scale * self.min_scale)
        scale2 = var * (self.df - 2.0) / self.df if self.df > 2.0 else var
        scale = math.sqrt(max(scale2, self.min_scale * self.min_scale))
        return StudentTDistribution(self.df, loc=loc, scale=scale, name=self.name, keys=self.keys)


class StudentTDataEncoder(DataSequenceEncoder):
    """Encode Student's t observations as a float array."""

    def __str__(self) -> str:
        return 'StudentTDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StudentTDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError('StudentTDistribution requires finite or infinite real-valued observations.')
        return rv
