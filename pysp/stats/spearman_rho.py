"""Create, estimate, and sample from a Spearman ranking distribution.

Defines the SpearmanRankingDistribution, SpearmanRankingSampler, SpearmanRankingAccumulatorFactory,
SpearmanRankingAccumulator, SpearmanRankingEstimator, and the SpearmanRankingDataEncoder
classes for use with pysparkplug.

Data type: List[int] (Component-wise rank of K dimensional observation vector)

The Spearman ranking distribution with dimension K, has probability function

    p_mat(x_k;rho, sigma) = exp(-rho * ||x_k-sigma||^2 ) / sum_{k=0}^{K-1} exp(-rho * ||x_k-sigma||^2 ), for k = 0,1,..,K-1

where x_k list of integers containing a permutation of the integers 0,1,2,...K-1. Note sigma is a list of floats with
dimension equal to K representing the mean of the rank variables, and rho is a correlation coefficient.

"""

import numpy as np
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory
import itertools

from typing import Optional, Sequence, Union, Any, Dict, List, Tuple


class SpearmanRankingDistribution(SequenceEncodableProbabilityDistribution):
    """Spearman ranking distribution over permutations of 0,...,K-1 with location sigma and decay rate rho.

    Data type: List[int] (a permutation of the integers 0,1,...,K-1).
    """

    def __init__(self, sigma: Union[Sequence[float], np.ndarray], rho: float = 1.0, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """SpearmanRankingDistribution object for defining a Spearman ranking distribution.

        Args:
            sigma (Union[Sequence[float], np.ndarray]): Numpy array of means for the rank variables.
            rho (float): Decay rate on variance of ranks.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for object instance.

        Attributes:
            sigma (np.ndarray]): Numpy array of means for the rank variables.
            rho (float): Decay rate on variance of ranks.
            name (Optional[str]): Name for object instance.
            dim (int): Dimension of the rank variable.
            keys (Optional[str]): Set keys for object instance.

        """
        self.sigma = np.asarray(sigma)
        self.rho = rho
        self.name = name
        self.dim = len(sigma)
        self.keys = keys

        perms = map(np.asarray, map(list, itertools.permutations(range(self.dim))))
        self.log_const = np.log(sum(map(lambda u: np.exp(-rho * np.dot(self.sigma - u, self.sigma - u)), perms)))

    def __str__(self) -> str:
        """Returns string representation of SpearmanRankingDistribution object."""
        return 'SpearmanRankingDistribution(sigma=%s, rho=%s, name=%s, keys=%s)' % (
            repr(self.sigma), repr(self.rho), repr(self.name), repr(self.keys))

    def density(self, x: List[int]) -> float:
        """Density of Spearman ranking distribution at observation x.

        See log_density() for details.

        Args:
            x (List[int]): Permutation of the integers 0,1,...,K-1.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: List[int]) -> float:
        """Log-density of Spearman ranking distribution at observation x.

        The log-density is given by

            log(p(x; rho, sigma)) = -rho * ||x - sigma||^2 - log_const,

        where log_const normalizes over all K! permutations.

        Args:
            x (List[int]): Permutation of the integers 0,1,...,K-1.

        Returns:
            Log-density at observation x.

        """
        temp = np.subtract(x, self.sigma)
        return -self.rho * np.dot(temp, temp) - self.log_const

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): 2-d numpy array of N permutations with K columns.

        Returns:
            Numpy array of log-density (float) of length N.

        """
        temp = x - self.sigma
        temp *= temp
        rv = np.sum(temp, axis=1) * -self.rho
        rv -= self.log_const
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'SpearmanRankingSampler':
        """Create a SpearmanRankingSampler object from parameters of SpearmanRankingDistribution instance.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            SpearmanRankingSampler object.

        """
        return SpearmanRankingSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SpearmanRankingEstimator':
        """Create a SpearmanRankingEstimator with matching dimension.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            SpearmanRankingEstimator object.

        """
        return SpearmanRankingEstimator(self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'SpearmanRankingDataEncoder':
        """Returns a SpearmanRankingDataEncoder object for encoding sequences of data."""
        return SpearmanRankingDataEncoder()


class SpearmanRankingSampler(DistributionSampler):
    """Sampler for the SpearmanRankingDistribution. Draws permutations of 0,...,K-1."""

    def __init__(self, dist: SpearmanRankingDistribution, seed: Optional[int] = None) -> None:
        """SpearmanRankingSampler object.

        Args:
            dist (SpearmanRankingDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (np.random.RandomState): Random number generator.
            dist (SpearmanRankingDistribution): Distribution to sample from.
            perms (List[List[int]]): All K! permutations of 0,...,K-1.
            probs (np.ndarray): Probability of each permutation under dist.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

        self.perms = list(map(list, itertools.permutations(range(dist.dim))))
        encoder = self.dist.dist_to_encoder()
        self.probs = np.exp(dist.seq_log_density(encoder.seq_encode(self.perms)))

    def sample(self, size: Optional[int] = None) -> Union[List[int], Sequence[List[int]]]:
        """Draw iid samples (permutations of 0,...,K-1) from the Spearman ranking distribution.

        Args:
            size (Optional[int]): Number of samples to draw. If None, a single permutation is returned.

        Returns:
            A single permutation (List[int]) if size is None, else a list of size permutations.

        """
        idx = self.rng.choice(len(self.perms), p=self.probs, replace=True, size=size)

        if size is None:
            return self.perms[idx]
        else:
            return [self.perms[u] for u in idx]


class SpearmanRankingAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for the SpearmanRankingDistribution. Tracks the weighted sum of ranks and total weight."""

    def __init__(self, dim: int, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """SpearmanRankingAccumulator object.

        Args:
            dim (int): Dimension K of the rank vectors.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            sum (np.ndarray): Weighted component-wise sum of observed rank vectors.
            count (float): Sum of observation weights.
            key (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional name for object instance.

        """
        self.sum = np.zeros(dim, dtype=np.float64)
        self.count = 0.0
        self.key = keys
        self.name = name

    def update(self, x: Union[List[int], np.ndarray], weight: float, estimate: Optional[SpearmanRankingDistribution])\
            -> None:
        """Update sufficient statistics with a weighted observation.

        Args:
            x (Union[List[int], np.ndarray]): Permutation of the integers 0,1,...,K-1.
            weight (float): Weight for observation.
            estimate (Optional[SpearmanRankingDistribution]): Previous estimate (unused).

        """
        self.sum += np.multiply(x, weight)
        self.count += weight

    def initialize(self, x: Union[List[int], np.ndarray], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with a weighted observation.

        Args:
            x (Union[List[int], np.ndarray]): Permutation of the integers 0,1,...,K-1.
            weight (float): Weight for observation.
            rng (RandomState): Random number generator (unused).

        """
        if weight != 0:
            self.sum += np.multiply(x, weight)
            self.count += weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[SpearmanRankingDistribution]) -> None:
        """Vectorized update of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N permutations with K columns.
            weights (np.ndarray): Weights for each of the N observations.
            estimate (Optional[SpearmanRankingDistribution]): Previous estimate (unused).

        """
        self.sum += np.dot(x.T, weights)
        self.count += weights.sum()

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N permutations with K columns.
            weights (np.ndarray): Weights for each of the N observations.
            rng (RandomState): Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, np.ndarray]) -> 'SpearmanRankingAccumulator':
        """Combine sufficient statistics from another accumulator into this one.

        Args:
            suff_stat (Tuple[float, np.ndarray]): Tuple of count and component-wise rank sums.

        Returns:
            Self, with aggregated sufficient statistics.

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self) -> Tuple[float, np.ndarray]:
        """Returns sufficient statistics as a Tuple of count and component-wise rank sums."""
        return self.count, self.sum

    def from_value(self, x: Tuple[float, np.ndarray]) -> 'SpearmanRankingAccumulator':
        """Set sufficient statistics of accumulator from value x.

        Args:
            x (Tuple[float, np.ndarray]): Tuple of count and component-wise rank sums.

        Returns:
            Self, with sufficient statistics set to x.

        """
        self.sum = x[1]
        self.count = x[0]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                stats_dict[self.key] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.key] = (self.count, self.sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set sufficient statistics of object instance to suff stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                self.count = vals[0]
                self.sum = vals[1]

    def acc_to_encoder(self) -> 'SpearmanRankingDataEncoder':
        """Returns a SpearmanRankingDataEncoder object for encoding sequences of data."""
        return SpearmanRankingDataEncoder()


class SpearmanRankingAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating SpearmanRankingAccumulator objects."""

    def __init__(self, dim: int, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """SpearmanRankingAccumulatorFactory object.

        Args:
            dim (int): Dimension K of the rank vectors.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.keys = keys
        self.name = name
        self.dim = dim

    def make(self) -> 'SpearmanRankingAccumulator':
        """Returns a new SpearmanRankingAccumulator object."""
        return SpearmanRankingAccumulator(dim=self.dim, name=self.name, keys=self.keys)


class SpearmanRankingEstimator(ParameterEstimator):
    """Estimator for the SpearmanRankingDistribution from aggregated sufficient statistics."""

    def __init__(self, dim: int, pseudo_count: Optional[float] = None, suff_stat: Optional[Tuple[float, np.ndarray]] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """SpearmanRankingEstimator object.

        Args:
            dim (int): Dimension K of the rank vectors.
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.
            suff_stat (Optional[Tuple[float, np.ndarray]]): Tuple of count and component-wise rank sums.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.dim = dim

    def accumulator_factory(self) -> 'SpearmanRankingAccumulatorFactory':
        """Returns a SpearmanRankingAccumulatorFactory for creating SpearmanRankingAccumulator objects."""
        return SpearmanRankingAccumulatorFactory(self.dim, self.name, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, np.ndarray]) -> 'SpearmanRankingDistribution':
        """Estimate a SpearmanRankingDistribution from sufficient statistics.

        The location sigma is set to the rank order (argsort) of the component-wise rank sums
        with rho fixed at 1.0. If no data was observed, rho is set to 0.0.

        Args:
            nobs (Optional[float]): Number of observations (unused).
            suff_stat (Tuple[float, np.ndarray]): Tuple of count and component-wise rank sums.

        Returns:
            SpearmanRankingDistribution object.

        """
        count, vsum = suff_stat

        if count > 0:
            sigma = np.argsort(vsum)
            rho = 1.0
        else:
            sigma = vsum
            rho = 0.0

        return SpearmanRankingDistribution(sigma, rho, name=self.name, keys=self.keys)


class SpearmanRankingDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of rank vector (permutation) observations."""

    def __str__(self) -> str:
        """Returns string representation of SpearmanRankingDataEncoder object."""
        return 'SpearmanRankingDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Checks if other object is an instance of a SpearmanRankingDataEncoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a SpearmanRankingDataEncoder instance, else False.

        """
        return isinstance(other, SpearmanRankingDataEncoder)

    def seq_encode(self, x: Sequence[List[int]]) -> np.ndarray:
        """Encode a sequence of N rank vectors for vectorized functions.

        Args:
            x (Sequence[List[int]]): Sequence of N permutations of 0,1,...,K-1.

        Returns:
            2-d numpy array with N rows and K columns.

        """
        rv = np.asarray(x)
        return rv

