"""Evaluate, estimate, and sample from a gaussian distribution with mean mu and variance sigma2.

Defines the GaussianDistribution, GaussianSampler, GaussianAccumulatorFactory, GaussianAccumulator,
GaussianEstimator, and the GaussianDataEncoder classes for use with pysparkplug.

Data type: (float): The GaussianDistribution with mean mu and variance sigma2 > 0.0, has log-density
    log(f(x;mu, sigma2)) = -log(2*pi*sigma2) - (x-mu)^2/sigma2, for real-valued x.

"""
import numpy as np
from numpy.random import RandomState
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from typing import Optional, Tuple, List, Callable, Dict, Union, Any


class GaussianDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None) -> None:
        """GaussianDistribution object defines Gaussian distribution with mean mu and variance sigma2.

        Args:
            mu (float): Real-valued number.
            sigma2 (float): Positive real-valued number.
            name (Optional[str]): String for name of object.

        Attributes:
            mu (float): Mean of gaussian distribution.
            sigma2 (float): Variance of Gaussian distribution.
            name (Optional[str]): String for name of object.
            cont (float): Normalizing constant of Gaussian (depends on sigma2).
            log_const (float): Log of above.

        """
        self.mu = mu
        self.sigma2 = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)
        self.name = name

    def __str__(self) -> str:
        """Returns string representation of GaussianDistribution object."""
        return 'GaussianDistribution(%s, %s, name=%s)' % (repr(self.mu), repr(self.sigma2), repr(self.name))

    def density(self, x: float) -> float:
        """Density of Gaussian distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            Density of Gaussian at x.

        """
        return self.const * exp(-0.5 * (x - self.mu) * (x - self.mu) / self.sigma2)

    def log_density(self, x: float) -> float:
        """Log-density of Gaussian distribution at observation x.

        Log-density of Gaussian with mean mu and variance sigma2 given by,
            log(f(x;mu, sigma2)) = -log(2*pi*sigma2) - (x-mu)^2/sigma2, for real-valued x.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            Log-density at observation x.

        """
        return self.log_const - 0.5 * (x - self.mu) * (x - self.mu) / self.sigma2

    def seq_ld_lambda(self) -> List[Callable]:
        return [self.seq_log_density]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): Numpy array of floats.

        Returns:
            Numpy array of log-density (float) of len(x).

        """
        rv = x - self.mu
        rv *= rv
        rv *= -0.5 / self.sigma2
        rv += self.log_const

        return rv

    def sampler(self, seed:Optional[int] = None) -> 'GaussianSampler':
        """Create an GaussianSampler object from parameters of GaussianDistribution instance.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            GaussianSampler object.

        """
        return GaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GaussianEstimator':
        """Create GaussianEstimator with mu and sigma2 passed if pseudo_count is not None.

        Arg variable pseudo_count is used to pass and re-weight mu and sigma2 of GaussianDistribution instance. Simply
        creates a GaussianEstimator with name passed if pseudo_count is None.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            GaussianEstimator object.

        """
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return GaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name)
        else:
            return GaussianEstimator(name=self.name)

    def dist_to_encoder(self) -> 'GaussianDataEncoder':
        """Returns a GaussianDataEncoder object for encoding sequences of data."""
        return GaussianDataEncoder()


class GaussianSampler(DistributionSampler):

    def __init__(self, dist: GaussianDistribution, seed: Optional[int] = None) -> None:
        """GaussianSampler for drawing samples from GaussianSampler instance.

        Args:
            dist (GaussianDistribution): GaussianDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (GaussianDistribution): GaussianDistribution instance to sample from.
            rng (RandomState): RandomState with seed set to seed if passed in args.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from GaussianSampler object.

        Numpy array of length 'size' from Gaussian distribution with mean mu and scale sigma2 if size not None.
        Else a single sample is returned as float.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            'size' iid samples from Gaussian distribution.

        """
        return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """GaussianAccumulator object used to accumulate sufficient statistics from observed data.

        Args:
            keys (Optional[str]): Set key for GaussianAccumulator object.
            name (Optional[str]): Set name for GaussianAccumulator object.

        Attributes:
            sum (float): Sum of weighted observations (sum_i w_i*X_i).
            sum2 (float): Sum of weighted squared observations (sum_i w_i*X_i^2)
            count (float): Sum of weights for observations (sum_i w_i).
            count2 (float): Sum of weights for squared observations (sum_i w_i).
            count (float): Tracks the sum of weighted observations used to form sum.
            key (Optional[str]): Key string used to aggregate all sufficient statistics with same keys values.
            name (Optional[str]): Name for GaussianAccumulator object.

        """
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional['GaussianDistribution']) -> None:
        """Update sufficient statistics for GaussianAccumulator with one weighted observation.

        Args:
            x (float): Observation from Gaussian distribution.
            weight (float): Weight for observation.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        x_weight = x * weight
        self.sum += x_weight
        self.sum2 += x * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize GaussianAccumulator object with weighted observation

        Note: Just calls update().

        Args:
            x (float): Observation from Gaussian distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        """Vectorized initialization of GaussianAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[GaussianDistribution]) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x, weights)
        self.sum2 += np.dot(x * x, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        """Aggregates sufficient statistics with GaussianAccumulator member sufficient statistics.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            suff_stat (Tuple[float, float, float, float]): See above for details.

        Returns:
            GaussianAccumulator object.

        """
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self) -> Tuple[float, float, float, float]:
        """Returns sufficient statistics of GaussianAccumulator object (Tuple[float, float, float, float])."""
        return self.sum, self.sum2, self.count, self.count2

    def from_value(self, x: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        """Assigns sufficient statistics of GaussianAccumulator instance to x.

        Arg passed x is tuple of four floats:
            x[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            x[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            x[2] (float): Sum of weighted observations (sum_i w_i),
            x[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            x: See above for deatils.

        Returns:
            GaussianAccumulator object.

        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2, x3 = stats_dict[self.keys]
                self.sum += x0
                self.sum2 += x1
                self.count += x2
                self.count2 += x3

            else:
                stats_dict[self.keys] = (self.sum, self.sum2, self.count, self.count2)

    def key_replace(self, stats_dict: Dict[str, 'GaussianAccumulator']) -> None:
        """Set sufficient statistics of object instance to suff_stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.sum, self.sum2, self.count, self.count2 = stats_dict[self.keys]

    def acc_to_encoder(self) -> 'GaussianDataEncoder':
        """Returns a GaussianDataEncoder object for encoding sequences of data."""
        return GaussianDataEncoder()


class GaussianAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, name: Optional[str] = None, keys:  Optional[str] = None) -> None:
        """GaussianAccumulatorFactory object for creating GaussianAccumulator.

        Args:
            name (Optional[str]): Assign a name to GaussianAccumulatorFactory object.
            keys (Optional[str]): Assign keys member for GaussianAccumulators.

        Attributes:
            name (Optional[str]): Name of the GaussianAccumulatorFactory obejct.
            keys (Optional[str]): String id for merging sufficient statistics of GaussianAccumulator.

        """
        self.keys = keys
        self.name = name

    def make(self) -> 'GaussianAccumulator':
        """Return a GaussianAccumulator object with name and keys passed."""
        return GaussianAccumulator(name=self.name, keys=self.keys)


class GaussianEstimator(ParameterEstimator):

    def __init__(self,
                 pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
                 suff_stat: Tuple[Optional[float], Optional[float]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[str] = None):
        """GaussianEstimator object used to estimate GaussianDistribution from aggregated sufficient statistics.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two positive floats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of float and positive float.
            name (Optional[str]): Assign a name to GaussianEstimator.
            keys (Optional[str]): Assign keys to GaussianEstimator for combining sufficient statistics.

        Attributes:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for suff_stat.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and variance (sigma2).
            name (Optional[str]): String name of GaussianEstimator instance.
            keys (Optional[str]): String keys of GaussianEstimator instance for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GaussianAccumulatorFactory':
        """Return GaussianAccumulatorFactory with name and keys passed."""
        return GaussianAccumulatorFactory(self.name, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float, float]) -> 'GaussianDistribution':
        """Estimate a GaussianDistribution object from sufficient statistics aggregated from data.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i),\

        obtained from aggregation of observations.

        If member variable pseudo_count is not None, suff_stat is combined with re-weighted member instance variables
        suff_stat. If pseudo_count is None, arg suff_stat is used to form maximum likelihood estimates for mu and
        sigma2 of GaussianDistribution object.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See above for details.

        Returns:
            GaussianDistribution object.

        """
        nobs_loc1 = suff_stat[2]
        nobs_loc2 = suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (suff_stat[0] + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1]) / (
                        nobs_loc2 + self.pseudo_count[1])
        else:
            sigma2 = suff_stat[1] / nobs_loc2 - mu * mu

        return GaussianDistribution(mu, sigma2, name=self.name)


class GaussianDataEncoder(DataSequenceEncoder):
    """GaussianDataEncoder object for encoding sequences of iid Gaussian observations with data type float."""

    def __str__(self) -> str:
        """Returns string representation of GaussianDataEncoder object."""
        return 'GaussianDataEncoder'

    def __eq__(self, other) -> bool:
        """Checks if other object is an instance of a GaussianDataEncoder.

        Args:
            other (object): Obejct to compare.

        Returns:
            True if other is an instance of a GaussianDataEncoder, else False.

        """
        return isinstance(other, GaussianDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Encode sequence of iid Gaussian observations.

        Data type must be List[float] or np.ndarray[float].

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid Gaussian observations.

        Returns:
            A numpy array of floats.

        """
        rv = np.asarray(x, dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception('GaussianDistribution requires support x in (-inf,inf).')
        return rv


