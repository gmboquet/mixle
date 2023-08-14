"""Create, estimate, and sample from a gamma distribution with shape k and scale theta.

Defines the GammaDistribution, GammaSampler, GammaAccumulatorFactory, GammaAccumulator, GammaEstimator,
and the GammaDataEncoder classes for use with pysparkplug.

Data type: (float): The GammaDistribution with shape k > 0.0 and scale theta > 0.0, has log-density
    log(f(x;k,theta)) = -gammaln(k) - k*log(theta) + (k-1) * log(x) - x / theta, for x > 0.0, else -np.inf

"""
import numpy as np
from numpy.random import RandomState
from typing import Tuple, List, Optional, Union, Dict, Any
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from pysp.utils.special import gammaln, digamma, trigamma


class GammaDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, k: float, theta: float, name: Optional[str] = None) -> None:
        """GammaDistribution for shape k and scale theta.

        Args:
            k (float): Positive real-valued number.
            theta (float): Positive real-valued number.
            name (Optional[str]): Assign a name to GammaDistribution instance.

        Attributes:
            k (float): Positive real-valued number.
            theta (float): Positive real-valued number.
            name (Optional[str]): Assign a name to GammaDistribution instance.
            log_const (float): Normalizing constant of gamma distribution.

        """
        self.k = k
        self.theta = theta
        self.log_const = -(gammaln(k) + k * log(theta))
        self.name = name

    def __str__(self) -> str:
        """Return string representation of GammaDistribution object."""
        return 'GammaDistribution(%s, %s, name=%s)' % (repr(self.k), repr(self.theta), repr(self.name))

    def density(self, x: float) -> float:
        """Density of gamma distribution evaluated at x.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            Density of gamma distribution evaluated at x.

        """
        return exp(self.log_const + (self.k - one) * log(x) - x / self.theta)

    def log_density(self, x: float) -> float:
        """Log-density of gamma distribution evaluated at x.

        Log-density given by,
        If x > 0.0,
            log(f(x;k,theta)) = -gammaln(k) - k*log(theta) + (k-1) * log(x) - x / theta,
        else,
            -np.inf
        Args:
            x (float): Positive real-valued number.

        Returns:
            Log-density of gamma distribution evaluated at x.

        """
        return self.log_const + (self.k - one) * log(x) - x / self.theta

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of sequence encoded observations from gamma distribution.

        Input must be x (Tuple[ndarray, ndarray]):
            x[0]: Numpy array of floats containing observations from gamma distribution.
            x[1]: Numpy array of floats containing log of observation values.

        Args:
            x (Tuple[np.ndarray, np.ndarray]): See above for details.

        Returns:
            Numpy array containing log-density evaluated at all observations of encoded sequence x.

        """
        rv = x[0] * (-1.0 / self.theta)
        if self.k != 1.0:
            rv += x[1] * (self.k - 1.0)
        rv += self.log_const

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GammaSampler':
        """Create a GammaSampler object from GammaDistribution.

        Args:
            seed (Optional[int]): Set seed on random number generator.

        Returns:
            GammaSampler object.

        """
        return GammaSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GammaEstimator':
        """Creates GammaEstimator object from GammaDistribution instance.

        Args:
            pseudo_count (Optional[float]): Re-weight the sufficient statistics of GammaDistribution instance if not
                None.

        Returns:
            GammaEstimator object.

        """
        if pseudo_count is None:
            return GammaEstimator(name=self.name)
        else:
            suff_stat = (self.k * self.theta, exp(digamma(self.k) + log(self.theta)))
            return GammaEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name)

    def dist_to_encoder(self) -> 'GammaDataEncoder':
        """Returns GammaDataEncoder object for encoding sequence of GammaDistribution observations."""
        return GammaDataEncoder()


class GammaSampler(DistributionSampler):

    def __init__(self, dist: 'GammaDistribution', seed: Optional[int] = None) -> None:
        """GammaSampler object used to draw samples from GammaDistribution.

        Args:
            dist (GammaDistribution): GammaDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        Attributes:
            rng (RandomState): RandomState with seed set for sampling.
            dist (GammaDistribution): GammaDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.


        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size'-iid observations from GammaSampler.

        Args:
            size (Optional[int]): Number of iid samples to draw from GammaSampler.

        Returns:
            Single sample (float) if size is None, else a numpy array of floats containing iid samples from
            GammaDistribution.

        """
        return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size)


class GammaAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys: Optional[str] = None) -> None:
        """GammaAccumulator object used to accumulate sufficient statistics from observations.

        Args:
            keys (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.

        Attributes:
            nobs (float): Number of observations accumulated.
            sum (float): Weighted-sum of observations accumulated.
            sum_of_logs (float): log weighted sum of weighted log(observations).
            key (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.

        """
        self.nobs = zero
        self.sum = zero
        self.sum_of_logs = zero
        self.key = keys

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        """Initialize sufficient statistics of GammaAccumulator with weighted observation.

        Note: Just calls update.

        Args:
            x (float): Positive real-valued observation of gamma.
            weight (float): Positive real-valued weight for observation x.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: Optional[RandomState]) -> None:
        """Vectorized initialization of GammaAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

        Args:
            x (Tuple[ndarray, ndarray]): Tuple of Numpy array of observations and log(observations).
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def update(self, x: float, weight: float, estimate: Optional['GammaDistribution']) -> None:
        """Update sufficient statistics for GammaAccumulator with one weighted observation.

        Args:
            x (float): Observation from gamma distribution.
            weight (float): Weight for observation.
            estimate (Optional[GammaDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        self.nobs += weight
        self.sum += x * weight
        self.sum_of_logs += log(x) * weight

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional['GammaDistribution']) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        Args:
            x (Tuple[ndarray, ndarray]): Tuple of Numpy array of observations and log(observations).
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[GammaDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x[0], weights)
        self.sum_of_logs += np.dot(x[1], weights)
        self.nobs += np.sum(weights)

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'GammaAccumulator':
        """Aggregates sufficient statistics with GammaAccumulator member sufficient statistics.

        Args:
            suff_stat (Tuple[float, float, float]): Aggregated sum, sum_of_logs, nobs.

        Returns:
            ExponentialAccumulator

        """
        self.nobs += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]

        return self

    def value(self) -> Tuple[float, float, float]:
        """Returns Tuple[float, float, float] containing sufficient statistics of GammaAccumulator."""
        return self.nobs, self.sum, self.sum_of_logs

    def from_value(self, x: Tuple[float, float, float]) -> 'GammaAccumulator':
        """Sets sufficient statistics GammaAccumulator to x.

        Args:
            x (Tuple[float, float, float]): Sufficient statistics tuple of length three..

        Returns:
            ExponentialAccumulator

        """
        self.nobs = x[0]
        self.sum = x[1]
        self.sum_of_logs = x[2]

        return self

    def key_merge(self, stats_dict: Dict[str,  Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                x0, x1, x2 = stats_dict[self.key]
                self.nobs += x0
                self.sum += x1
                self.sum_of_logs += x2

            else:
                stats_dict[self.key] = (self.nobs, self.sum, self.sum_of_logs)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set sufficient statistics of object instance to suff_stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                x0, x1, x2 = stats_dict[self.key]
                self.nobs = x0
                self.sum = x1
                self.sum_of_logs = x2

    def acc_to_encoder(self) -> 'GammaDataEncoder':
        """Return GammaDataEncoder for encoding sequence of data. """
        return GammaDataEncoder()


class GammaAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, keys: Optional[str] = None) -> None:
        """GammaAccumulatorFactory object for creating GammaAccumulator objects.

        Args:
            keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.

        Attributes:
            keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.

        """
        self.keys = keys

    def make(self) -> 'GammaAccumulator':
        """Returns GammaAccumulator object with keys passed."""
        return GammaAccumulator(keys=self.keys)


class GammaEstimator(ParameterEstimator):

    def __init__(self, pseudo_count: Tuple[float, float] = (0.0, 0.0), suff_stat: Tuple[float, float] = (1.0, 0.0),
                 threshold: float = 1.0e-8, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GammaEstimator object used for estimating GammaDistribution from aggregated data.

        Args:
            pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
            suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
            threshold (float): Threshold used for estimating the shape of gamma.
            name (Optional[str]): Assign a name to GammaEstimator.
            keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

        Attributes:
            pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
            suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
            threshold (float): Threshold used for estimating the shape of gamma.
            name (Optional[str]): Assign a name to GammaEstimator.
            keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GammaAccumulatorFactory':
        """Create GammaAccumulatorFactory with keys passed."""
        return GammaAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float]) -> 'GammaDistribution':
        """Obtain GammaDistribution from aggregated sufficient statistics of observed data.

        Takes sufficient statistic aggregated from observed data:
            suff_stat[0]: weighted sum of observations
            suff_stat[1]: weighted sum of log-observations
            suff_stat[2]: weighted observation count.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See description above for details.

        Returns:
            GammaDistribution object.

        """
        pc1, pc2 = self.pseudo_count
        ss1, ss2 = self.suff_stat

        if suff_stat[0] == 0:
            return GammaDistribution(1.0, 1.0)

        adj_sum = suff_stat[1] + ss1 * pc1
        adj_cnt = suff_stat[0] + pc1
        adj_mean = adj_sum / adj_cnt

        adj_lsum = suff_stat[2] + ss2 * pc2
        adj_lcnt = suff_stat[0] + pc2
        adj_lmean = adj_lsum / adj_lcnt

        k = self.estimate_shape(adj_mean, adj_lmean, self.threshold)

        return GammaDistribution(k, adj_sum / (k * adj_lcnt), name=self.name)

    @staticmethod
    def estimate_shape(avg_sum: float, avg_sum_of_logs: float, threshold: float) -> float:
        """Estimates the shape parameter of GammaDistribution.

        Args:
            avg_sum (float): Weighted sum of gamma observations.
            avg_sum_of_logs (float): Weighted log sum of gamma observations.
            threshold (float): Threshold used for assessing convergence of shape estimation.

        Returns:
            Estimate of shape parameter 'k'.

        """
        s = log(avg_sum) - avg_sum_of_logs
        old_k = inf
        k = (3 - s + sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s)
        while abs(old_k - k) > threshold:
            old_k = k
            k -= (log(k) - digamma(k) - s) / (one / k - trigamma(k))
        return k


class GammaDataEncoder(DataSequenceEncoder):
    """GammaDataEncoder object for encoding sequences of iid Gamma observations with data type float."""

    def __str__(self) -> str:
        """Return string representation of GammaDataEncoder."""
        return 'GammaDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check if object is instance of GammaDataEncoder.

        Args:
            other (object): An object to check for equality.

        Returns:
            True if object is instance of GammaDataEncoder, else False.

        """
        return isinstance(other, GammaDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode iid sequence of gamma observations for vectorized "seq_" function calls.

        Note: Each entry of x must be positive float.

        Args:
            x (Union[List[float], np.ndarray]): IID sequence of gamma distributed observations.

        Returns:
            Tuple of x as numpy array and log(x).

        """
        rv1 = np.asarray(x, dtype=float)

        if np.any(rv1 <= 0) or np.any(np.isnan(rv1)):
            raise Exception('GammaDistribution has support x > 0.')
        else:
            rv2 = np.log(rv1)
            return rv1, rv2
