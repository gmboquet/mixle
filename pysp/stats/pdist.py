"""Defines abstract classes for SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator,
ProbabilityDistribution, StatisticAccumulator, StatisticAccumulatorFactory, DataSequenceEncoder, ParameterEstimator,
ConditionalSampler, and DistributionSampler for classes of the pysp.stats.

"""
import itertools
import math
import numpy as np
from abc import abstractmethod
from pysp.arithmetic import *
from typing import TypeVar, Optional, Any, Generic, Dict, List, Tuple

SS = TypeVar('SS')


class EnumerationError(NotImplementedError):
    """Raised when a distribution (or a child of a combinator) cannot enumerate its support.

    The path argument identifies the offending child within a combinator, e.g.
    'CompositeDistribution.dists[1]'.
    """

    def __init__(self, dist: Any, path: str = '', reason: str = '') -> None:
        self.leaf = dist
        self.path = path
        self.reason = reason
        msg = '%s does not support enumeration' % type(dist).__name__
        if path:
            msg = '%s -> %s' % (path, msg)
        if reason:
            msg += ': %s' % reason
        super().__init__(msg)


def child_enumerator(child: 'ProbabilityDistribution', path: str) -> 'DistributionEnumerator':
    """Construct child.enumerator(), annotating EnumerationError with the child's path.

    Combinator enumerators use this so a failure deep in a nested model reports the
    full path to the offending leaf, e.g.
    'CompositeDistribution.dists[1] -> MixtureDistribution.components[0] -> GaussianDistribution ...'.
    """
    try:
        return child.enumerator()
    except EnumerationError as e:
        new_path = path if not e.path else '%s -> %s' % (path, e.path)
        raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None


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

    def enumerator(self) -> 'DistributionEnumerator':
        """Return a DistributionEnumerator over this distribution's support.

        Distributions with an enumerable (discrete) support override this; the
        default raises EnumerationError.
        """
        raise EnumerationError(self)


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


class DistributionEnumerator(object):
    """Lazy iterator over the support of dist in non-increasing probability order.

    Yields (value, log_prob) pairs, possibly infinitely many. Contract:
      - Each support value is yielded exactly once (deduplication is the
        enumerator's responsibility).
      - log_prob equals dist.log_density(value) up to float round-off (~1e-10),
        and the sequence of log_probs is non-increasing up to the same tolerance.
      - Values with zero probability are skipped, never yielded.
      - Ties are broken deterministically by insertion order; no further guarantee.
    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        self.dist = dist

    def __iter__(self) -> 'DistributionEnumerator':
        return self

    @abstractmethod
    def __next__(self) -> Tuple[Any, float]: ...

    def top_k(self, k: int) -> List[Tuple[Any, float]]:
        """Return the k most probable (value, log_prob) pairs (fewer if the support is smaller)."""
        return list(itertools.islice(self, k))


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





