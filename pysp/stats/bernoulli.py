"""Create, estimate, enumerate, and sample from a Bernoulli distribution.

Data type: bool or values in {0, 1}. The distribution has success
probability p and log-density log(p) for True/1 and log(1-p) for False/0.
"""
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState

from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class BernoulliDistribution(SequenceEncodableProbabilityDistribution):
    """Bernoulli distribution over {False, True} with success probability p."""

    def __init__(self, p: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        if p <= 0.0 or p >= 1.0:
            raise ValueError('BernoulliDistribution requires p in (0, 1).')
        self.p = float(p)
        self.log_p = math.log(self.p)
        self.log_1p = math.log1p(-self.p)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'BernoulliDistribution(%s, name=%s, keys=%s)' % (
            repr(self.p), repr(self.name), repr(self.keys))

    @staticmethod
    def _as_bool(x: Any) -> Optional[bool]:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        try:
            if x == 1:
                return True
            if x == 0:
                return False
        except Exception:
            return None
        return None

    def density(self, x: Union[bool, int]) -> float:
        return math.exp(self.log_density(x))

    def log_density(self, x: Union[bool, int]) -> float:
        xx = self._as_bool(x)
        if xx is None:
            return -np.inf
        return self.log_p if xx else self.log_1p

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        return np.where(x, self.log_p, self.log_1p)

    def sampler(self, seed: Optional[int] = None) -> 'BernoulliSampler':
        return BernoulliSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'BernoulliEstimator':
        if pseudo_count is None:
            return BernoulliEstimator(name=self.name, keys=self.keys)
        return BernoulliEstimator(pseudo_count=pseudo_count, suff_stat=self.p,
                                  name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'BernoulliDataEncoder':
        return BernoulliDataEncoder()

    def enumerator(self) -> 'BernoulliEnumerator':
        return BernoulliEnumerator(self)


class BernoulliEnumerator(DistributionEnumerator):
    """Enumerate False/True in descending probability order."""

    def __init__(self, dist: BernoulliDistribution) -> None:
        super().__init__(dist)
        self._entries = [(True, dist.log_p), (False, dist.log_1p)]
        self._entries.sort(key=lambda u: -u[1])
        self._pos = 0

    def __next__(self) -> Tuple[bool, float]:
        if self._pos >= len(self._entries):
            raise StopIteration
        rv = self._entries[self._pos]
        self._pos += 1
        return rv


class BernoulliSampler(DistributionSampler):
    """Draw iid Bernoulli observations."""

    def __init__(self, dist: BernoulliDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[bool, Sequence[bool]]:
        rv = self.rng.rand() < self.dist.p if size is None else self.rng.rand(size) < self.dist.p
        return bool(rv) if size is None else rv.tolist()


class BernoulliAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted success and observation counts."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.sum = 0.0
        self.count = 0.0
        self.name = name
        self.key = keys

    def update(self, x: Union[bool, int], weight: float, estimate: Optional[BernoulliDistribution]) -> None:
        xx = BernoulliDistribution._as_bool(x)
        if xx is None:
            raise ValueError('BernoulliDistribution requires observations in {False, True} or {0, 1}.')
        self.sum += float(xx) * weight
        self.count += weight

    def initialize(self, x: Union[bool, int], weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[BernoulliDistribution]) -> None:
        self.sum += np.dot(x.astype(np.float64), weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'BernoulliAccumulator':
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> Tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'BernoulliAccumulator':
        self.count = x[0]
        self.sum = x[1]
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

    def acc_to_encoder(self) -> 'BernoulliDataEncoder':
        return BernoulliDataEncoder()


class BernoulliAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for BernoulliAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> BernoulliAccumulator:
        return BernoulliAccumulator(name=self.name, keys=self.keys)


class BernoulliEstimator(ParameterEstimator):
    """Estimate a Bernoulli distribution from weighted success counts."""

    def __init__(self, pseudo_count: Optional[float] = None, suff_stat: Optional[float] = None,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BernoulliAccumulatorFactory:
        return BernoulliAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> BernoulliDistribution:
        count, psum = suff_stat
        if self.pseudo_count is not None:
            prior_p = 0.5 if self.suff_stat is None else self.suff_stat
            psum += self.pseudo_count * prior_p
            count += self.pseudo_count
        p = psum / count if count > 0.0 else 0.5
        p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
        return BernoulliDistribution(p, name=self.name, keys=self.keys)


class BernoulliDataEncoder(DataSequenceEncoder):
    """Encode Bernoulli observations as a boolean numpy array."""

    def __str__(self) -> str:
        return 'BernoulliDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BernoulliDataEncoder)

    def seq_encode(self, x: Sequence[Union[bool, int]]) -> np.ndarray:
        rv = np.asarray(x)
        valid = (rv == 0) | (rv == 1)
        if not np.all(valid):
            raise ValueError('BernoulliDistribution requires observations in {False, True} or {0, 1}.')
        return rv.astype(bool)
