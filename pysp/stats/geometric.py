"""Create, estimate, and sample from a geometric distribution with probability of success p.

Defines the GeometricDistribution, GeometricSampler, GeometricAccumulatorFactory, GeometricAccumulator,
GeometricEstimator, and the GeometricDataEncoder classes for use with pysparkplug.

Data type (int): The geometric distribution with probability of success p, has density

    P(x=k) = (k-1)*log(1-p) + log(p), for k = 1,2,...

"""
import numpy as np
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from numpy.random import RandomState
from typing import Optional, Tuple, Sequence, Dict, Union, Any


class GeometricDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, p: float, name: Optional[str] = None) -> None:
        """GeometricDistribution object defining geometric distribution with probability of success p.

        Mean: 1/p, Variance: (1-p)/p^2.

        Args:
            p (float): Must between (0,1).
            name (Optional[str]): Assign name to GeometricDistribution object.

        Attributes:
            p (float): Probability of success, must between (0,1).
            log_p (float): Log of probability of success p.
            log_1p (float): Log of 1-p (prob of failure).
            name (Optional[str]): Assign name to GeometricDistribution object.

        """
        self.p = p
        self.log_p = np.log(p)
        self.log_1p = np.log1p(-p)
        self.name = name

    def __str__(self) -> str:
        """Return string representation of GeometricDistribution instance."""
        return 'GeometricDistribution(%s, name=%s)' % (repr(self.p), repr(self.name))

    def density(self, x: int) -> float:
        """Density of geometric distribution evaluated at x.

            P(x=k) = (k-1)*log(1-p) + log(p), for x = 1,2,..., else 0.0.

        Args:
            x (int): Observed geometric value (1,2,3,....).


        Returns:
            Density of geometric distribution evaluated at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of geometric distribution evaluated at x.

        See density() for details.

        Args:
            x (int): Must be natural number (1,2,3,....).

        Returns:
            Log-density of geometric distribution evaluated at x.

        """
        return (x - 1) * self.log_1p + self.log_p

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density evaluated on sequence encoded x.

        Args:
            x (int): Numpy array of non-negative integers.

        Returns:
            Numpy array of log-density evaluated at each encoded observation value x.

        """
        rv = x - 1
        rv *= self.log_1p
        rv += self.log_p

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GeometricSampler':
        """Creates GeometricSampler object from GeometricDistribution instance.

        Args:
            seed (Optional[int]): Used to set seed on random number generator.

        Returns:
            GeometricSampler object.

        """
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GeometricEstimator':
        """Creates GeometricEstimator object.

        Args:
            pseudo_count (Optional[float]): Regularize summary statistics from object instance.

        Returns:
            GeometricEstimator object.

        """
        if pseudo_count is None:
            return GeometricEstimator(name=self.name)
        else:
            return GeometricEstimator(pseudo_count=pseudo_count, suff_stat=self.p, name=self.name)

    def dist_to_encoder(self) -> 'GeometricDataEncoder':
        """Returns GeometricDataEncoder object for encoding sequence of GeometricDistribution observations."""
        return GeometricDataEncoder()


class GeometricSampler(DistributionSampler):

    def __init__(self, dist: GeometricDistribution, seed: Optional[int] = None) -> None:
        """GeometricSampler object used to draw samples from GeometricDistribution.

        Args:
            dist (GeometricDistribution): GeometricDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        Attributes:
            rng (RandomState): RandomState with seed set for sampling.
            dist (GeometricDistribution): GeometricDistribution to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from geometric distribution.

        Generates a single geometric sample (int) if size is None, else a numpy array of integers of length size,
        iid samples, from the geometric distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be 1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None):
        """GeometricAccumulator object used to accumulate sufficient statistics from observations.

        Args:
            name (Optional[str]): Assign a name to the object instance.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        Attributes:
            sum (float): Aggregate weighted sum of observations.
            count (float): Aggregate sum of weighted observation count.
            name (Optional[str]): Assigned from name arg.
            key (Optional[str]): Assigned from keys arg.

        """
        self.sum = 0.0
        self.count = 0.0
        self.key = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional['GeometricDistribution']) -> None:
        """Update sufficient statistics for GeometricAccumulator with one weighted observation.

        Args:
            x (int): Positive integer observation of geometric distribution.
            weight (float): Weight for observation.
            estimate (Optional[GeometricDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional['GeometricDistribution']) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        sum increased by sum of weighted observations.
        count increased by sum of weights.

        Args:
            x (ndarray): Numpy array of positive integers.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[GeometricDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x, weights)
        self.count += np.sum(weights)

    def initialize(self, x: int, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize sufficient statistics of GeometricAccumulator with weighted observation.

        Note: Just calls update.

        Args:
            x (int): Positive integer observation of geometric distribution.
            weight (float): Positive real-valued weight for observation x.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        """Vectorized initialization of GeometricAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

        Args:
            x (ndarray): Numpy array of positive integers.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'GeometricAccumulator':
        """Combine aggregated sufficient statistics with sufficient statistics of GeometricAccumulator instance.

        Input suff_stat is Tuple[float, float] with:
            suff_stat[0] (float): sum of observation weights,
            suff_stat[1] (float): weighted sum of observations.

        Args:
            suff_stat (Tuple[float, float]): See above for details.

        Returns:
            GeometricAccumulator object.

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        """Returns sufficient statistics Tuple[float, float] of GeometricAccumulator instance."""
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'GeometricAccumulator':
        """Sets GeometricAccumulator instance sufficient statistic member variables to x.

        Args:
            x (Tuple[float, float]): Sum of observations weights and sum of weighted observations.

        Returns:
            GeometricAccumulator object.

        """
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                x0, x1 = stats_dict[self.key]
                self.count += x0
                self.sum += x1

            else:
                stats_dict[self.key] = (self.count, self.sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set sufficient statistics of object instance to suff_stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.count, self.sum = stats_dict[self.key]

    def acc_to_encoder(self) -> 'GeometricDataEncoder':
        """Returns GeometricDataEncoder object for encoding sequence of GeometricDistribution observations."""
        return GeometricDataEncoder()


class GeometricAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GeometricAccumulatorFactory object used to create GeometricAccumulator objects.

        Args:
            name (Optional[str]): Assign a name to the object instance.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        Attributes:
            name (Optional[str]): Assigned from name arg.
            keys (Optional[str]): Assigned from keys arg.

        """
        self.name = name
        self.keys = keys

    def make(self) -> 'GeometricAccumulator':
        """Return GeometricAccumulator with name and keys passed."""
        return GeometricAccumulator(name=self.name, keys=self.keys)


class GeometricEstimator(ParameterEstimator):

    def __init__(self, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """GeometricEstimator object for estimating GeometricDistribution object from aggregated sufficient statistics.

        Args:
            pseudo_count (Optional[float]): Float value for re-weighting suff_stat member variable.
            suff_stat (Optional[float]): Probability of success (value between (0,1)).
            name (Optional[str]): Assign a name to the object instance.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        Attributes:
            pseudo_count (Optional[float]): Assigned from pseudo_count arg.
            suff_stat (Optional[float]): Assigned from suff_stat arg (corrected for [0,1] constraint).
            name (Optional[str]): Assigned from name arg.
            keys (Optional[str]): Assigned from keys arg.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = min(min(suff_stat, 1.0),0.0) if suff_stat is not None else None
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GeometricAccumulatorFactory':
        """Create GeometricAccumulatorFactory object with name and keys passed."""
        return GeometricAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'GeometricDistribution':
        """Estimate geometric distribution from aggregated sufficient statistics (suff_stat).

        Uses suff_stat (Tuple[float, float]):
            suff_stat[0] (float): sum of weights of the observations (count),
            suff_stat[1] (float): weighted sum of observations (sum).

        If member variable pseudo_count is not None, then suff_stat arg is combined with pseudo_count weighted member
        variable of sufficient statistics.

        If member variable pseudo_count is not None, and member variable sufficient statistic is None, suff_stat arg
        is reweighted by pseudo_count alone.

        If no pseudo_count is set, p = suff_stat[0]/suff_stat[1] is passed to GeometricDistribution.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat (Tuple[float, float]): See above.

        Returns:
            GeometricDistribution object.

        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.pseudo_count * self.suff_stat) / (
                    suff_stat[1] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        else:
            p = suff_stat[0] / suff_stat[1]

        return GeometricDistribution(p, name=self.name)


class GeometricDataEncoder(DataSequenceEncoder):
    """GeometricDataEncoder object for encoding sequences of iid geometric observations with data type int."""

    def __str__(self) -> str:
        """Returns string representation of GeometricDataEncoder object."""
        return 'GeometricDataEncoder'

    def __eq__(self, other) -> bool:
        """Checks if object is equivalent to GeometricDataEncoder instance.

        Args:
            other (object): Object to be compared to self.

        Returns:
            True if other is GeometricDataEncoder instance, else False.

        """
        return isinstance(other, GeometricDataEncoder)

    def seq_encode(self, x: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        """Encode iid sequence of geometric observations for vectorized "seq_" function calls.

        Note: x should be list of numpy array of positive integers.

        Args:
            x (Union[Sequence[int], np.ndarray]): Positive integer geometric observations.

        Returns:
            Numpy array of positive integers.

        """
        rv = np.asarray(x)
        if np.any(rv < 1) or np.any(np.isnan(rv)):
            raise Exception('GeometricDistribution requires integers greater than 0 for x.')
        else:
            return np.asarray(rv, dtype=float)

