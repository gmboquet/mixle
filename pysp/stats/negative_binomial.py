"""Create, estimate, enumerate, and sample from a negative binomial distribution.

The parameterization is the number of failures X before r successes, with
success probability p:

    P(X=x) = Gamma(x+r) / (Gamma(r) Gamma(x+1)) * p**r * (1-p)**x,
    x = 0, 1, 2, ...

The shape r is treated as fixed by the estimator; p has a closed-form M-step.
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
from pysp.utils.vector import gammaln


class NegativeBinomialDistribution(SequenceEncodableProbabilityDistribution):
    """Negative binomial distribution over non-negative integer counts."""

    def __init__(self, r: float, p: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        if r <= 0.0 or not np.isfinite(r):
            raise ValueError('NegativeBinomialDistribution requires r > 0.')
        if p <= 0.0 or p >= 1.0:
            raise ValueError('NegativeBinomialDistribution requires p in (0, 1).')
        self.r = float(r)
        self.p = float(p)
        self.log_p = math.log(self.p)
        self.log_1p = math.log1p(-self.p)
        self.log_gamma_r = float(gammaln(self.r))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'NegativeBinomialDistribution(%s, %s, name=%s, keys=%s)' % (
            repr(self.r), repr(self.p), repr(self.name), repr(self.keys))

    @staticmethod
    def _valid_count(x: Any) -> bool:
        try:
            xx = float(x)
        except Exception:
            return False
        return np.isfinite(xx) and xx >= 0.0 and math.floor(xx) == xx

    def density(self, x: int) -> float:
        return math.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        if not self._valid_count(x):
            return -np.inf
        xx = float(x)
        return (float(gammaln(xx + self.r)) - self.log_gamma_r - float(gammaln(xx + 1.0))
                + self.r * self.log_p + xx * self.log_1p)

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        xx, lgx1 = x
        return (gammaln(xx + self.r) - self.log_gamma_r - lgx1
                + self.r * self.log_p + xx * self.log_1p)

    def sampler(self, seed: Optional[int] = None) -> 'NegativeBinomialSampler':
        return NegativeBinomialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'NegativeBinomialEstimator':
        if pseudo_count is None:
            return NegativeBinomialEstimator(r=self.r, name=self.name, keys=self.keys)
        return NegativeBinomialEstimator(r=self.r, pseudo_count=pseudo_count, suff_stat=self.p,
                                         name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'NegativeBinomialDataEncoder':
        return NegativeBinomialDataEncoder()

    def enumerator(self) -> 'NegativeBinomialEnumerator':
        return NegativeBinomialEnumerator(self)


class NegativeBinomialEnumerator(DistributionEnumerator):
    """Enumerate the infinite support in descending probability order."""

    def __init__(self, dist: NegativeBinomialDistribution) -> None:
        super().__init__(dist)
        mode = math.floor((dist.r - 1.0) * (1.0 - dist.p) / dist.p) if dist.r > 1.0 else 0
        self._mode = int(max(0, mode))
        self._left = self._mode - 1
        self._right = self._mode + 1
        self._started = False

    def __next__(self) -> Tuple[int, float]:
        if not self._started:
            self._started = True
            return self._mode, self.dist.log_density(self._mode)
        lp_l = self.dist.log_density(self._left) if self._left >= 0 else -np.inf
        lp_r = self.dist.log_density(self._right)
        if lp_l >= lp_r:
            x, lp = self._left, lp_l
            self._left -= 1
        else:
            x, lp = self._right, lp_r
            self._right += 1
        return x, lp


class NegativeBinomialSampler(DistributionSampler):
    """Draw iid negative binomial observations."""

    def __init__(self, dist: NegativeBinomialDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        scale = (1.0 - self.dist.p) / self.dist.p
        lam = self.rng.gamma(shape=self.dist.r, scale=scale, size=size)
        rv = self.rng.poisson(lam=lam)
        return int(rv) if size is None else rv


class NegativeBinomialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum statistics."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.name = name
        self.key = keys

    def update(self, x: int, weight: float, estimate: Optional[NegativeBinomialDistribution]) -> None:
        if not NegativeBinomialDistribution._valid_count(x):
            raise ValueError('NegativeBinomialDistribution requires non-negative integer observations.')
        self.count += weight
        self.sum += float(x) * weight

    def initialize(self, x: int, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional[NegativeBinomialDistribution]) -> None:
        self.count += np.sum(weights, dtype=np.float64)
        self.sum += np.dot(x[0], weights)

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray,
                       rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'NegativeBinomialAccumulator':
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> Tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'NegativeBinomialAccumulator':
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

    def acc_to_encoder(self) -> 'NegativeBinomialDataEncoder':
        return NegativeBinomialDataEncoder()


class NegativeBinomialAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for NegativeBinomialAccumulator."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> NegativeBinomialAccumulator:
        return NegativeBinomialAccumulator(name=self.name, keys=self.keys)


class NegativeBinomialEstimator(ParameterEstimator):
    """Estimate p for a negative binomial distribution with fixed r."""

    def __init__(self, r: float = 1.0, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[float] = None, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        if r <= 0.0 or not np.isfinite(r):
            raise ValueError('NegativeBinomialEstimator requires r > 0.')
        self.r = float(r)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> NegativeBinomialAccumulatorFactory:
        return NegativeBinomialAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> NegativeBinomialDistribution:
        count, xsum = suff_stat
        if self.pseudo_count is not None:
            prior_p = 0.5 if self.suff_stat is None else float(self.suff_stat)
            prior_p = float(np.clip(prior_p, 1.0e-12, 1.0 - 1.0e-12))
            xsum += self.pseudo_count * self.r * (1.0 - prior_p) / prior_p
            count += self.pseudo_count
        p = (self.r * count) / (self.r * count + xsum) if count > 0.0 else 0.5
        p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
        return NegativeBinomialDistribution(self.r, p, name=self.name, keys=self.keys)


class NegativeBinomialDataEncoder(DataSequenceEncoder):
    """Encode count observations with precomputed log-factorials."""

    def __str__(self) -> str:
        return 'NegativeBinomialDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NegativeBinomialDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 0) or np.any(np.isnan(rv)) or np.any(np.floor(rv) != rv)):
            raise ValueError('NegativeBinomialDistribution requires non-negative integer observations.')
        return rv, gammaln(rv + 1.0)
