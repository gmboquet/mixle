""""WeightedDistribution class.

This Distribution simply allows from weights on observations. I.e. Data type D is observed and an associated
score/weight is assigned to the data. This simply passes the weights and data downstream in aggregatation.

Likelihood evals are equivalent to normal likelihood calls to the base distribution.

"""
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from numpy.random import RandomState
import numpy as np
from typing import Dict, Any, Optional, Tuple, Sequence, TypeVar

D = TypeVar('D')
E = TypeVar('E')
SS = TypeVar('SS')


class WeightedDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, name: Optional[str] = None):
        self.dist = dist
        self.name = name

    def __str__(self) -> str:
        return 'WeightedDistribution(dist=%s, name=%s)' % (repr(self.dist), repr(self.name))

    def density(self, x: D) -> float:
        return self.dist.density(x)

    def log_density(self, x: D) -> float:
        return self.dist.log_density(x)

    def seq_log_density(self, x: Tuple[E, np.ndarray]) -> np.ndarray:
        return self.dist.seq_log_density(x[0])

    def dist_to_encoder(self) -> 'WeightedDataEncoder':
        return WeightedDataEncoder(encoder=self.dist.dist_to_encoder())

    def estimator(self, pseudo_count: Optional[float] = None) -> 'WeightedEstimator':
        if pseudo_count is not None:
            return WeightedEstimator(estimator=self.dist.estimator(pseudo_count=pseudo_count), name=self.name)
        else:
            return WeightedEstimator(estimator=self.dist.estimator(), name=self.name)

    def sampler(self, seed: Optional[int] = None) -> 'DistributionSampler':
        return self.dist.sampler(seed)


class WeightedAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, name: Optional[str] = None):
        self.accumulator = accumulator
        self.name = name

    def initialize(self, x: Tuple[D, float], weight: float, rng: np.random.RandomState) -> None:
        self.accumulator.initialize(x[0], weight*x[1], rng)

    def update(self, x: Tuple[D, float], weight: float, estimate: WeightedDistribution) -> None:
        self.accumulator.update(x[0], weight*x[1], estimate.dist)

    def seq_update(self, x, weights: np.ndarray, estimate: WeightedDistribution) -> None:
        self.accumulator.seq_update(x[0], weights*x[1], estimate.dist)

    def seq_initialize(self, x: Tuple[E, np.ndarray], weights: np.ndarray, rng: np.random.RandomState) -> None:
        self.accumulator.seq_initialize(x[0], weights*x[1], rng)

    def combine(self, suff_stat: SS) -> 'WeightedAccumulator':
        self.accumulator.combine(SS)
        return self

    def from_value(self, x: SS) -> 'WeightedAccumulator':
        self.accumulator.from_value(x)

        return self

    def value(self) -> Any:
        return self.accumulator.value()

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'WeightedDataEncoder':
        return WeightedDataEncoder(encoder=self.accumulator.acc_to_encoder())

class WeightedAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, factory: StatisticAccumulatorFactory, name: Optional[str] = None):
        self.factory = factory
        self.name = name

    def make(self) -> 'WeightedAccumulator':
        return WeightedAccumulator(accumulator=self.factory.make(), name=self.name)


class WeightedEstimator(ParameterEstimator):

    def __init__(self, estimator: ParameterEstimator, name: Optional[str] = None):
        self.estimator = estimator
        self.name = name

    def accumulator_factory(self) -> 'WeightedAccumulatorFactory':
        return WeightedAccumulatorFactory(factory=self.estimator.accumulator_factory(), name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'WeightedDistribution':
        return WeightedDistribution(dist=self.estimator.estimate(nobs, suff_stat), name=self.name)


class WeightedDataEncoder(DataSequenceEncoder):

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        self.encoder = encoder

    def __str__(self) -> str:
        return 'WeightedDataEncoder(encoder=%s)' % (repr(self.encoder))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WeightedDataEncoder):
            return other.encoder == self.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Tuple[D, float]]) -> Tuple[Any, np.ndarray]:
        return self.encoder.seq_encode([xx[0] for xx in x]), np.asarray([xx[1] for xx in x], dtype=float)


