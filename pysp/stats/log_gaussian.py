"""Evaluate, estimate, and sample from a log-gaussian distribution with location mu and scale sigma2.

Defines the LogGaussianDistribution, LogGaussianSampler, LogGaussianAccumulatorFactory, LogGaussianAccumulator,
LogGaussianEstimator, and the LogGaussianDataEncoder classes for use with pysparkplug.

Data type: (float): The LogGaussianDistribution with mu and sigma2 > 0.0, has log-density
    log(f(x;mu, sigma2)) = -log(2*pi*sigma2) - log(x) - (log(x)-mu)^2/sigma2, for positive-valued x.

"""
import numpy as np
from numpy.random import RandomState
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from typing import Optional, Tuple, List, Callable, Dict, Union, Any


class LogGaussianDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None) -> None:
        """LogGaussianDistribution object defines Gaussian distribution with mean mu and variance sigma2.

        Args:
            mu (float): Real-valued number.
            sigma2 (float): Positive real-valued number.
            name (Optional[str]): String for name of object.

        Attributes:
            mu (float): Location parameter for log-Gaussian distribution.
            sigma2 (float): Scale for log-Gaussian distribution.
            name (Optional[str]): String for name of object.
            cont (float): Normalizing constant (depends on sigma2).
            log_const (float): Log of above.

        """
        self.mu = mu
        self.sigma2 = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)
        self.name = name

    def __str__(self) -> str:
        """Returns string representation of object instance."""
        return 'LogGaussianDistribution(%s, %s, name=%s)' % (repr(self.mu), repr(self.sigma2), repr(self.name))

    def density(self, x: float) -> float:
        """Density of Log-Gaussian distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            Density of Log-Gaussian at x.

        """
        return self.const * exp(-0.5 * (np.log(x) - self.mu) ** 2 / self.sigma2) / x

    def log_density(self, x: float) -> float:
        """Log-density of log-Gaussian distribution at observation x.

        Log-density of Gaussian with mean mu and variance sigma2 given by,
            log(f(x;mu, sigma2)) = -log(2*pi*sigma2) - x - (x-mu)^2/sigma2, for positive x.

        Args:
            x (float): Positive valued observation of log-Gaussian.

        Returns:
            Log-density at observation x.

        """
        return self.log_const - 0.5 * (np.log(x) - self.mu) ** 2 / self.sigma2 - np.log(x)

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
        rv -= x

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'LogGaussianSampler':
        """Create an LogGaussianSampler object from parameters of LogGaussianDistribution instance.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            LogGaussianSampler object.

        """
        return LogGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'LogGaussianEstimator':
        """Create LogGaussianEstimator from attribute variables.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            LogGaussianEstimator object.

        """
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return LogGaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name)
        else:
            return LogGaussianEstimator(name=self.name)

    def dist_to_encoder(self) -> 'LogGaussianDataEncoder':
        """Returns a LogGaussianDataEncoder object for encoding sequences of data."""
        return LogGaussianDataEncoder()


class LogGaussianSampler(DistributionSampler):

    def __init__(self, dist: LogGaussianDistribution, seed: Optional[int] = None) -> None:
        """LogGaussianSampler for drawing samples from LogGaussianSampler instance.

        Args:
            dist (LogGaussianDistribution): LogGaussianDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (LogGaussianDistribution): LogGaussianDistribution instance to sample from.
            rng (RandomState): RandomState with seed set to seed if passed in args.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from LogGaussianSampler object.

        Numpy array of length 'size' from log-Gaussian distribution with scale beta if size not None. Else a single
        sample is returned as float.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            'size' iid samples from Gaussian distribution.

        """
        return np.exp(self.rng.normal(loc=self.dist.mu, scale=np.sqrt(self.dist.sigma2), size=size))


class LogGaussianAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """LogGaussianAccumulator object used to accumulate sufficient statistics from observed data.

        Args:
            keys (Optional[str]): Set key for LogGaussianAccumulator object.
            name (Optional[str]): Set name for LogGaussianAccumulator object.

        Attributes:
            log_sum (float): Sum of weighted observations (sum_i w_i*X_i).
            log_sum2 (float): Sum of weighted squared observations (sum_i w_i*X_i^2)
            count (float): Sum of weights for observations (sum_i w_i).
            count2 (float): Sum of weights for squared observations (sum_i w_i).
            count (float): Tracks the sum of weighted observations used to form sum.
            key (Optional[str]): Key string used to aggregate all sufficient statistics with same keys values.
            name (Optional[str]): Name for GaussianAccumulator object.

        """
        self.log_sum = 0.0
        self.log_sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional['LogGaussianDistribution']) -> None:
        """Update sufficient statistics for LogGaussianAccumulator with one weighted observation.

        Args:
            x (float): Observation from log-Gaussian distribution.
            weight (float): Weight for observation.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        x_weight = np.log(x) * weight
        self.log_sum += x_weight
        self.log_sum2 += np.log(x) * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize LogGaussianAccumulator object with weighted observation

        Note: Just calls update().

        Args:
            x (float): Observation from log-Gaussian distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        """Vectorized initialization of LogGaussianAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[LogGaussianDistribution]) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.log_sum += np.dot(x, weights)
        self.log_sum2 += np.dot(x * x, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: Tuple[float, float, float, float]) -> 'LogGaussianAccumulator':
        """Aggregates sufficient statistics with LogGaussianAccumulator member sufficient statistics.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*log(X_i)),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*log(X_i)^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            suff_stat (Tuple[float, float, float, float]): See above for details.

        Returns:
            GaussianAccumulator object.

        """
        self.log_sum += suff_stat[0]
        self.log_sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self) -> Tuple[float, float, float, float]:
        """Returns sufficient statistics of LogGaussianAccumulator object (Tuple[float, float, float, float])."""
        return self.log_sum, self.log_sum2, self.count, self.count2

    def from_value(self, x: Tuple[float, float, float, float]) -> 'LogGaussianAccumulator':
        """Assigns sufficient statistics of LogGaussianAccumulator instance to x.

        Arg passed x is tuple of four floats:
            x[0] (float): Sum of weighted observations (sum_i w_i*log(X_i)),
            x[1] (float): Sum of weighted observations (sum_i w_i*log(X_i)^2),
            x[2] (float): Sum of weighted observations (sum_i w_i),
            x[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            x: See above for details

        Returns:
            LogGaussianAccumulator object.

        """
        self.log_sum = x[0]
        self.log_sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merges LogGaussianAccumulator sufficient statistics with sufficient statistics contained in suff_stat dict
        that share the same key.

        Args:
            stats_dict (Dict[str, Any]): Dict containing 'key' string for LogGaussianAccumulator
                objects to combine sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set the sufficient statistics of LogGaussianAccumulator to stats_key sufficient statistics if key is in
            stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys string ids to LogGaussianAccumulator
                objects.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> 'LogGaussianDataEncoder':
        """Returns a LogGaussianDataEncoder object for encoding sequences of data."""
        return LogGaussianDataEncoder()


class LogGaussianAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, name: Optional[str] = None, keys:  Optional[str] = None) -> None:
        """LogGaussianAccumulatorFactory object for creating LogGaussianAccumulator.

        Args:
            name (Optional[str]): Assign a name to LogGaussianAccumulatorFactory object.
            keys (Optional[str]): Assign keys member for LogGaussianAccumulators.

        Attributes:
            name (Optional[str]): Name of the LogGaussianAccumulatorFactory object.
            keys (Optional[str]): String id for merging sufficient statistics of LogGaussianAccumulator.

        """
        self.keys = keys
        self.name = name

    def make(self) -> 'LogGaussianAccumulator':
        """Return a LogGaussianAccumulator object with name and keys passed."""
        return LogGaussianAccumulator(name=self.name, keys=self.keys)


class LogGaussianEstimator(ParameterEstimator):

    def __init__(self,
                 pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
                 suff_stat: Tuple[Optional[float], Optional[float]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[str] = None):
        """LogGaussianEstimator object used to estimate LogGaussianDistribution from aggregated sufficient statistics.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two positive floats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of float and positive float.
            name (Optional[str]): Assign a name to LogGaussianEstimator.
            keys (Optional[str]): Assign keys to LogGaussianEstimator for combining sufficient statistics.

        Attributes:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for suff_stat.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and variance (sigma2).
            name (Optional[str]): String name of LogGaussianEstimator instance.
            keys (Optional[str]): String keys of LogGaussianEstimator instance for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'LogGaussianAccumulatorFactory':
        """Return GaussianAccumulatorFactory with name and keys passed."""
        return LogGaussianAccumulatorFactory(self.name, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float, float]) \
            -> 'LogGaussianDistribution':
        """Estimate a LogGaussianDistribution object from sufficient statistics aggregated from data.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*log(X_i)),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*log(X_i)^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i),\

        obtained from aggregation of observations.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See above for details.

        Returns:
            LogGaussianDistribution object.

        """
        log_x, log_x2 = suff_stat[0], suff_stat[1]
        nobs_loc1, nobs_loc2 = suff_stat[2], suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (log_x + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1]) / (
                        nobs_loc2 + self.pseudo_count[1])
        else:
            sigma2 = np.sum( log_x2 - np.sum(log_x)**2 / nobs_loc1 ) / nobs_loc2

        return LogGaussianDistribution(mu, sigma2, name=self.name)


class LogGaussianDataEncoder(DataSequenceEncoder):
    """LogGaussianDataEncoder object for encoding sequences of iid Gaussian observations with data type float."""

    def __str__(self) -> str:
        """Returns string representation of LogGaussianDataEncoder object."""
        return 'LogGaussianDataEncoder'

    def __eq__(self, other) -> bool:
        """Checks if other object is an instance of a LogGaussianDataEncoder.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is an instance of a LogGaussianDataEncoder, else False.

        """
        return isinstance(other, LogGaussianDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Encode sequence of iid Log-Gaussian observations.

        Data type must be List[float] or np.ndarray[float].

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid log-Gaussian observations.

        Returns:
            A numpy array of floats.

        """
        rv = np.asarray(np.log(x), dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception('LogGaussianDistribution requires support x in (0,inf).')
        return rv


