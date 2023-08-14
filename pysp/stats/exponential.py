"""Create, estimate, and sample from an exponential distribution with scale beta.

Defines the ExponentialDistribution, ExponentialSampler, ExponentialAccumulatorFactory,ExponentialAccumulator,
ExponentialEstimator, and the ExponentialDataEncoder classes for use with pysparkplug.

Data type: (float): The ExponentialDistribution with scale beta > 0.0, has log-density
    log(f(x;beta)) = -log(beta) - x/beta, for x > 0, else -np.inf.

"""
from typing import Optional, Tuple
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from numpy.random import RandomState
import numpy as np
from typing import List, Union, Dict, Any


class ExponentialDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, beta: float, name: Optional[str] = None):
        """ExponentialDistribution object for shape beta, with mean (1/beta).

        Data type: int.

        Log-Density given by,
            log(f(x;beta)) = -log(beta) - x/beta, for x > 0.

        Args:
            beta (float): Positive valued real number defining scale of exponential distribution.
            name (Optional[str]): Assign a name to ExponentialDistribution object.

        Attributes:
            beta (float): Positive valued real number defining scale of exponential distribution.
            log_beta (float): log of beta parameter.
            name (Optional[str]): Assign a name to ExponentialDistribution object.

        """
        self.beta = beta
        self.log_beta = np.log(beta)
        self.name = name

    def __str__(self) -> str:
        """Returns string representation of ExponentialDistribution instance."""
        return 'ExponentialDistribution(%s, name=%s)' % (repr(self.beta), repr(self.name))

    def density(self, x: float) -> float:
        """Evaluate the density of exponential distribution with scale beta.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Evaluate the log-density of exponential distribution with scale beta.

        log(f(x;beta)) = -log(beta) - x/beta, for x > 0, else -np.inf.

        Args:
            x (float): Positive real-valued number.

        Returns:
            Log-density evaluated at x.

        """
        if x < 0:
            return -inf
        else:
            return -x / self.beta - self.log_beta

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Vectorized call to log-density on each observation value x.

        Args:
            x (np.ndarray): Numpy array of floats.

        Returns:
            Numpy array of log-density (float) of len(x).

        """
        rv = x * (-1.0 / self.beta)
        rv -= self.log_beta
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'ExponentialSampler':
        """Create an ExponentialSampler object with scale beta.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            ExponentialSampler object.

        """
        return ExponentialSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'ExponentialEstimator':
        """Create ExponentialEstimator with beta passed as suff_stat.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            ExponentialEstimator.

        """
        if pseudo_count is None:
            return ExponentialEstimator(name=self.name)
        else:
            return ExponentialEstimator(pseudo_count=pseudo_count, suff_stat=self.beta, name=self.name)

    def dist_to_encoder(self) -> 'ExponentialDataEncoder':
        """Returns an ExponentialDataEncoder object. """
        return ExponentialDataEncoder()


class ExponentialSampler(DistributionSampler):

    def __init__(self, dist: 'ExponentialDistribution', seed: Optional[int] = None) -> None:
        """ExponentialSampler for drawing samples from ExponentialSampler instance.

        Args:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            rng (RandomState): RandomState with seed set to seed if passed in args.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from ExponentialSampler object.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            Numpy array of length 'size' from exponential distribution with scale beta if size not None. Else a single
            sample is returned as float.


        """
        return self.rng.exponential(scale=self.dist.beta, size=size)


class ExponentialAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys: Optional[str] = None) -> None:
        """ExponentialAccumulator object used to accumulate sufficient statistics.

        Args:
            keys (Optional[str]): Aggregate all sufficient statistics with same keys values.

        Attributes:
            sum (float): Tracks the sum of observation values.
            count (float): Tracks the sum of weighted observations used to form sum.
            key (Optional[str]): Aggregate all sufficient statistics with same key.

        """
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def update(self, x: float, weight: float, estimate: Optional['ExponentialDistribution']) -> None:
        """Update sufficient statistics for ExponentialAccumulator with one weighted observation.

        Args:
            x (float): Observation from exponential distribution.
            weight (float): Weight for observation.
            estimate (Optional['ExponentialDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def seq_update(self,
                   x: np.ndarray,
                   weights: np.ndarray,
                   estimate: Optional['ExponentialDistribution']) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        sum increased by sum of weighted observations.
        count increased by sum of weights.

        Args:
            x (ndarray): Numpy array of positvie floats.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional['ExponentialDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def initialize(self, x: float, weight: float, rng: Optional['np.random.RandomState']) -> None:
        """Initialize sufficient statistics of ExponentialAccumulator with weighted observation.

        Note: Just calls update.

        Args:
            x (float): Positive real-valued observation of exponential.
            weight (float): Positive real-valued weight for observation x.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self,
                       x: np.ndarray,
                       weights: np.ndarray,
                       rng: np.random.RandomState) -> None:
        """Vectorized initialization of ExponentialAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

        Args:
            x (ndarray): Numpy array of positive floats.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'ExponentialAccumulator':
        """Aggregates sufficient statistics with ExponentialAccumulator member sufficient statistics.

        Args:
            suff_stat (Tuple[float, float]): Aggregated count and sum.

        Returns:
            ExponentialAccumulator

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        """Returns Tuple[float, float] containing sufficient statistics (count and sum) of ExponentialAccumulator."""
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'ExponentialAccumulator':
        """Sets sufficient statistics (count and sum) of ExponentialAccumulator to x.

        Args:
            x (Tuple[float, float]): Sufficient statistics tuple (count, sum).

        Returns:
            ExponentialAccumulator

        """
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merges ExponentialAccumulator sufficient statistics with sufficient statistics contained in suff_stat dict
        that share the same key.

        Args:
            stats_dict (Dict[str, Any]): Dict containing 'key' string for ExponentialAccumulator
                objects that represent the same distribution.

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
        """Set the sufficient statistics of ExponentialAccumulator to stats_key sufficient statistics if key is in
            stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Map key to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.count = stats_dict[self.key][0]
                self.sum = stats_dict[self.key][1]

    def acc_to_encoder(self) -> 'ExponentialDataEncoder':
        """Returns an ExponentialDataEncoder object. """
        return ExponentialDataEncoder()


class ExponentialAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, keys: Optional[str] = None) -> None:
        """ExponentialAccumulatorFactory object for creating ExponentialAccumulator.

        Args:
            keys (Optional[str]): Used for merging sufficient statistics of ExponentialAccumulator.

        Attributes:
            keys (Optional[str]): Used for merging sufficient statistics of ExponentialAccumulator.

        """
        self.keys = keys

    def make(self) -> 'ExponentialAccumulator':
        """Create ExponentialAccumulator with keys passed."""
        return ExponentialAccumulator(keys=self.keys)


class ExponentialEstimator(ParameterEstimator):

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """ExponentialEstimator object estimates ExponentialDistribution from aggregated sufficient statistics.

        Args:
            pseudo_count (Optional[float]): Used to weight sufficient statistics.
            suff_stat (Optional[float]): Positive float value for scale of exponential distribution.
            name (Optional[str]): Assign a name to ExponentialEstimator.
            keys (Optional[str]): Assign keys to ExponentialEstimator for combining sufficient statistics.

        Attributes:
            pseudo_count (Optional[float]): Used to weight sufficient statistics.
            suff_stat (Optional[float]): Positive float value for scale of exponential distribution.
            name (Optional[str]): Assign a name to ExponentialEstimator.
            keys (Optional[str]): Assign keys to ExponentialEstimator for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'ExponentialAccumulatorFactory':
        """Create ExponentialAccumulatorFactory object with keys passed."""
        return ExponentialAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'ExponentialDistribution':
        """Estimate ExponentialDistribution from suff_stat arg.

        Estimate ExponentialDistribution from sufficient statistic tuple suff_stat, counting a float value for
        count and sum. If pseudo_count is set, this is used to re-weight the member value "suff_stat", which is the
        scale of ExponentialEstimator object.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat (Tuple[float, float]): Tuple of count and sum. Both are positive real-valued floats.

        Returns:
            ExponentialDistribution object.

        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[1] + self.suff_stat * self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[1] + self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        else:
            if suff_stat[0] > 0:
                p = suff_stat[1] / suff_stat[0]
            else:
                p = 1.0

        return ExponentialDistribution(beta=p, name=self.name)


class ExponentialDataEncoder(DataSequenceEncoder):
    """ExponentialDataEncoder object for encoding sequences of iid exponential observations with data type float."""

    def __str__(self) -> str:
        """Returns string representation of ExponentialDataEncoder."""
        return 'ExponentialDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check if object is an instance of ExponentialDataEncoder.

        Args:
            other (object): Object to compare.

        Returns:
            True if object is an instance of ExponentialDataEncoder, else False.

        """
        return isinstance(other, ExponentialDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Encode sequence of iid exponential observations.

        Data type must be a float.
        Data must also be positive real-valued numbers.

        Args:
            x (Union[List[float], np.ndarray]): IID numpy array or list of positive real-valued floats.

        Returns:
            Numpy array of floats.

        """
        rv = np.asarray(x, dtype=float)

        if np.any(rv <= 0) or np.any(np.isnan(rv)):
            raise Exception('Exponential requires x > 0.')

        return rv
