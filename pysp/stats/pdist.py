"""Defines abstract classes for SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator,
ProbabilityDistribution, StatisticAccumulator, StatisticAccumulatorFactory, DataSequenceEncoder, ParameterEstimator,
ConditionalSampler, and DistributionSampler for classes of the pysp.stats.

"""
import math
import numpy as np
from abc import abstractmethod
from pysp.arithmetic import *
from typing import TypeVar, Optional, Any, Generic, Dict

SS = TypeVar('SS')


class ProbabilityDistribution:

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def density(self, x: Any) -> float:
        return math.exp(self.log_density(x))

    @abstractmethod
    def log_density(self, x: Any) -> float: ...

    @abstractmethod
    def sampler(self, seed: Optional[int] = None) -> 'DistributionSampler':
        ...

    @abstractmethod
    def estimator(self, pseudo_count: Optional[float] = None) -> 'ParameterEstimator':
        ...


class SequenceEncodableProbabilityDistribution(ProbabilityDistribution):

    def seq_ld_lambda(self):
        pass

    def seq_log_density(self, x: Any) -> np.ndarray:
        return np.asarray([self.log_density(u) for u in x])

    def seq_log_density_lambda(self):
        return [self.seq_log_density]

    @abstractmethod
    def dist_to_encoder(self) -> 'DataSequenceEncoder': ...


class DistributionSampler(object):

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, seed: Optional[int] = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def new_seed(self) -> int:
        return self.rng.randint(0, maxrandint)

    @abstractmethod
    def sample(self, size: Optional[int] = None) -> Any: ...


class ConditionalSampler(object):
    @abstractmethod
    def sample_given(self, x): ...


class StatisticAccumulator(Generic[SS]):

    def update(self, x: Any, weight: float, estimate) -> None:
        ...

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        self.update(x, weight, estimate=None)

    @abstractmethod
    def combine(self, suff_stat: SS) -> 'StatisticAccumulator':
        ...

    @abstractmethod
    def value(self) -> SS:
        ...

    @abstractmethod
    def from_value(self, x: SS) -> 'SequenceEncodableStatisticAccumulator':
        ...

    @abstractmethod
    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        ...


class SequenceEncodableStatisticAccumulator(StatisticAccumulator[SS]):

    def get_seq_lambda(self):
        pass

    @abstractmethod
    def seq_update(self, x, weights: np.ndarray, estimate) -> None: ...

    @abstractmethod
    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None: ...

    @abstractmethod
    def acc_to_encoder(self) -> 'DataSequenceEncoder': ...

class StatisticAccumulatorFactory(object):

    @abstractmethod
    def make(self) -> 'SequenceEncodableStatisticAccumulator': ...


class ParameterEstimator(Generic[SS]):

    @abstractmethod
    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'SequenceEncodableProbabilityDistribution': ...

    @abstractmethod
    def accumulator_factory(self) -> 'StatisticAccumulatorFactory': ...


class DataSequenceEncoder:

    def __str__(self) -> str:
        return self.__str__()

    def seq_encode(self, x: Any) -> Any:
        return x

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...





