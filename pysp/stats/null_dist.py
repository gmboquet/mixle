"""Create, estimate, and sample from a null distribution.

Defines the NullDistribution, NullSampler, NullAccumulatorFactory, NullAccumulator,
NullEstimator, and the NullDataEncoder classes for use with pysparkplug.

The NullDistribution object and its related classes are space filling objects meant for consistency in type hints.

Notes:
    The density evaluates to 1.0 for any value (Any data type).
    The sampler generates None for any size input.
    Sequence encodings return None for any input.

"""
from typing import Any, Optional, Dict

import numpy as np
from numpy.random import RandomState

from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder


class NullDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    def __str__(self) -> str:
        return 'NullDistribution(name=%s)' % repr(self.name)

    def density(self, x: Optional[Any]) -> float:
        return 1.0

    def log_density(self, x: Optional[Any]) -> float:
        return 0.0

    def seq_log_density(self, x: Optional[Any]) -> float:
        return 0.0

    def sampler(self, seed: Optional[int] = None) -> 'NullSampler':
        return NullSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'NullEstimator':
        if pseudo_count is None:
            return NullEstimator(name=self.name)

        else:
            return NullEstimator(pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'NullDataEncoder':
        return NullDataEncoder()


class NullSampler(DistributionSampler):

    def __init__(self, dist: 'NullDistribution', seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> None:
        return None


class NullAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys: Optional[str] = None) -> None:
        self.key = keys

    def update(self, x: Optional[Any], weight: float, estimate: Optional['NullDistribution']) -> None:
        pass

    def seq_update(self,
                   x: Optional[Any],
                   weights: np.ndarray,
                   estimate: Optional['NullDistribution']) -> None:
        pass

    def initialize(self, x: Optional[Any], weight: float, rng: Optional['np.random.RandomState']) -> None:
        self.update(x, weight, None)

    def seq_initialize(self,
                       x: Optional[Any],
                       weights: np.ndarray,
                       rng: np.random.RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Optional[Any]) -> 'NullAccumulator':
        return self

    def value(self) -> None:
        return None

    def from_value(self, x: Optional[Any]) -> 'NullAccumulator':
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                pass
            else:
                stats_dict[self.key] = None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> 'NullDataEncoder':
        return NullDataEncoder()


class NullAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, keys: Optional[str] = None) -> None:
        self.keys = keys

    def make(self) -> 'NullAccumulator':
        return NullAccumulator(keys=self.keys)


class NullEstimator(ParameterEstimator):

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Any] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'NullAccumulatorFactory':
        return NullAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Optional[Any] = None) -> 'NullDistribution':
        return NullDistribution(name=self.name)


class NullDataEncoder(DataSequenceEncoder):

    def __str__(self) -> str:
        return 'NullDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, NullDataEncoder)

    def seq_encode(self, x: Any) -> None:
        return None
