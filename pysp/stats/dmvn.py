"""Create, estimate, and sample from a diagonal Gaussian distribution (independent-multivariate Gaussian).

Defines the DiagonalGaussianDistribution, DiagonalGaussianSampler, DiagonalGaussianAccumulatorFactory,
DiagonalGaussianAccumulator, DiagonalGaussianEstimator, and the DiagonalGaussianDataEncoder classes for use with
pysparkplug.

The log-density of an 'n' dimensional diagonal-gaussian observation x = (x_1,x_2,...,x_n) with mean mu=(m_1,m_2,..,m_n),
and diagonal covariance matrix given by covar = diag(s2_1, s2_2,...,s2_n).

    log(p_mat(x)) = -0.5*sum_{i=1}^{n} (x_i-m_i)^2 / s2_i - 0.5*log(s2_i) - (n/2)*log(pi).

Data type: x (List[float], np.ndarray).

"""
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory
from numpy.random import RandomState
import pysp.utils.vector as vec
import numpy as np
from numpy.random import RandomState

from typing import Sequence, Optional, Dict, Any, Tuple, List, Union


class DiagonalGaussianDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, mu: Union[Sequence[float], np.ndarray], covar: Union[Sequence[float], np.ndarray],
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Create a DiagonalGaussianDistribution object with mean mu and covariance covar.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean of Gaussian distribution.
            covar (Union[Sequence[float], np.ndarray]): Variance of each component.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for object isntance.

        Attributes:
             dim (int): Dimension of the multivariate Gaussian. Determined by mean length.
             mu (np.ndarray): Mean of the Gaussian.
             covar (np.ndarray): Variance for each component.
             name (Optional[str]): Name of object instance.
             log_c (float): Normalizing constant for diagonal Gausisan.
             ca (np.ndarray): Term for likelihood-calc.
             cb (np.ndarray): Term for likelihood-calc.
             cc (np.ndarray): Term for likelihood-calc.
             key (Optional[str]): Key for merging sufficient statistics.

        """
        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.name = name
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c
        self.key = keys

    def __str__(self) -> str:
        s1 = repr(list(self.mu.flatten()))
        s2 = repr(list(self.covar.flatten()))
        s3 = repr(self.name)
        return 'DiagonalGaussianDistribution(%s, %s, name=%s)' % (s1, s2, s3)

    def density(self, x: Union[Sequence[float], np.ndarray]):
        return exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[float], np.ndarray]):
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DiagonalGaussianSampler':
        return DiagonalGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DiagonalGaussianEstimator':
        if pseudo_count is None:
            return DiagonalGaussianEstimator(name=self.name, keys=self.key)
        else:
            return DiagonalGaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), name=self.name, keys=self.key)

    def dist_to_encoder(self) -> 'DiagonalGaussianDataEncoder':
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianSampler(DistributionSampler):

    def __init__(self, dist: DiagonalGaussianDistribution, seed: Optional[int] = None) -> None:
        """DiagonalGaussianSampler object for sampling from DiagonalGaussian instance.

        Args:
            dist (DiagonalGaussianDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            dist (DiagonalGaussianDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[Sequence[np.ndarray], np.ndarray]:
        if size is None:
            rv = self.rng.randn(self.dist.dim)
            rv *= np.sqrt(self.dist.covar)
            rv += self.dist.mu
            return rv
        else:
            return [self.sample() for i in range(size)]


class DiagonalGaussianAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, dim: Optional[int] = None, keys: Optional[str] = None) -> None:
        """DiagonalGaussianAccumulator object for aggregating sufficient statistics from iid observations.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
             dim (Optional[int]): Optional dimension of Gaussian.
             count (float): Used for tracking weighted observations counts.
             sum (np.ndarray): Sum of observation vectors.
             sum2 (np.ndarray): Sum of squared observation vectors.
             key (Optional[str]): If set, merge sufficient statistics with objects containing matching keys.

        """
        self.dim = dim
        self.count = 0.0
        self.sum = vec.zeros(dim) if dim is not None else None
        self.sum2 = vec.zeros(dim) if dim is not None else None
        self.key = keys

    def update(self, x: Union[Sequence[float], np.ndarray], weight: float,
               estimate: Optional[DiagonalGaussianDistribution]) -> None:

        if self.dim is None:
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = x * weight
        self.count += weight
        self.sum += x_weight
        x_weight *= x
        self.sum2 += x_weight

    def initialize(self, x: Union[Sequence[float], np.ndarray], weight: float, rng: RandomState) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[DiagonalGaussianDistribution]) -> None:
        if self.dim is None:
            self.dim = len(x[0])
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = np.multiply(x.T, weights)
        self.count += weights.sum()
        self.sum += x_weight.sum(axis=1)
        x_weight *= x.T
        self.sum2 += x_weight.sum(axis=1)

    def seq_initialize(self, x, weights: np.ndarray, rng: RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray, float]) -> 'DiagonalGaussianAccumulator':
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, float]:
        return self.sum, self.sum2, self.count

    def from_value(self, x: Tuple[np.ndarray, np.ndarray, float]) -> 'DiagonalGaussianAccumulator':
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key].value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> 'DiagonalGaussianDataEncoder':
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, dim: Optional[int] = None, keys: Optional[str] = None) -> None:
        """DiagonalGaussianAccumulatorFactory object for creating DiagonalGaussianAccumulator objects.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
             dim (Optional[int]): Optional dimension of Gaussian.
             key (Optional[str]): If set, merge sufficient statistics with objects containing matching keys.

        """
        self.dim = dim
        self.key = keys

    def make(self) -> 'DiagonalGaussianAccumulator':
        return DiagonalGaussianAccumulator(dim=self.dim, keys=self.key)


class DiagonalGaussianEstimator(ParameterEstimator):

    def __init__(self, dim: Optional[int] = None, pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
                 suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """DiagonalGaussianEstimator object for estimating diagonal Gaussian distributions from aggregated sufficient
            statistics.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the sum of observations and sum of
                squared observations in estimation.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray]]): Sum of observations and sum of squared
                observations both having same dimension.
            name (Optinal[str]): Set name for object instance.
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
            name (Optinal[str]): Name for object instance.
            dim (int): Dimension of Gaussian, either set of determined from suff_stat arg.
            prior_mu (Optional[np.ndarray]): Set from suff_stat[0].
            prior_covar ((Optional[np.ndarray]): Set from suff_stat[1].
            pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the sum of observations and sum of
                squared observations in estimation.
            keys (Optional[str]): Key for merging sufficient statistics.

        """
        dim_loc = dim if dim is not None else (
            (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1])))) if suff_stat[0] is None else len(
                suff_stat[0]))

        self.name = name
        self.dim = dim_loc
        self.pseudo_count = pseudo_count
        self.prior_mu = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.prior_covar = None if suff_stat[1] is None else np.reshape(suff_stat[1], dim_loc)
        self.key = keys

    def accumulator_factory(self) -> 'DiagonalGaussianAccumulatorFactory':
        return DiagonalGaussianAccumulatorFactory(dim=self.dim, keys=self.key)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray, float]) \
            -> 'DiagonalGaussianDistribution':
        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (suff_stat[1] + (pc2 * self.prior_covar) - (mu * mu * nobs)) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - (mu * mu)

        return DiagonalGaussianDistribution(mu, covar, name=self.name)


class DiagonalGaussianDataEncoder(DataSequenceEncoder):

    def __init__(self, dim: Optional[int] = None) -> None:
        """DiagonalGaussianDataEncoder object for encoding sequences of iid diagonal-Gaussian observations."""
        self.dim = dim

    def __str__(self) -> str:
        return 'DiagonalGaussianDataEncoder(dim=' + str(self.dim) + ')'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DiagonalGaussianDataEncoder):
            return self.dim == other.dim
        else:
            return False

    def seq_encode(self, x: Sequence[Union[List[float], np.ndarray]]) -> np.ndarray:
        if self.dim is None:
            self.dim = len(x[0])
        xv = np.reshape(x, (-1, self.dim))
        return xv

