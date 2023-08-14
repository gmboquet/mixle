"""Create, estimate, and sample from a Poisson distribution with rate lam > 0.0.

Defines the PoissonDistribution, PoissonSampler, PoissonAccumulatorFactory, PoissonAccumulator,
PoissonEstimator, and the PoissonDataEncoder classes for use with pysparkplug.

Data type (int): The Poisson distribution with rate lam, has log-density

    log(p_mat(x_mat=x; lam) = -x*log(lam) - log(x!) - lam,

for x in {0,1,2,...}, and

    log(p_mat(x_mat=x)) = -np.inf,

else.

"""
import numpy as np
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from pysp.utils.vector import gammaln
from math import log
from typing import Tuple, List, Union, Optional, Any, Dict, Sequence


class PoissonDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, lam: float, name: Optional[str] = None) -> None:
        """PoissonDistribution object defining Poisson distribution with mean lam > 0.0.

        Args:
            lam (float): Positive real-valued number.
            name (Optional[str]): String name for object instance.

        Attributes:
            lam (float): Mean of Poisson distribution.
            name (Optional[str]): String name for object instance.
            log_lambda (float): Log of attribute lam.
        """
        self.lam = lam
        self.log_lambda = log(lam)
        self.name = name

    def __str__(self) -> str:
        """Returns string representation of PoissonDistribution object."""
        return 'PoissonDistribution(%s, name=%s)' % (repr(self.lam), repr(self.name))

    def density(self, x: int) -> float:
        """Evaluate the density of Poisson distribution at observation x.

        Calls np.exp(log_density(x)). See log_density() for details.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            Density of Poisson distribution evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of Poisson distribution evaluated at x.

        Log-density given by,
            log(p_mat(x_mat=x; lam) = -x*log(lam) - log(x!) - lam, for x in {0,1,2,...}
        and -np.inf else.

        Note: log(Gamma(x+1.0)) = log(x!), where Gamma is the gamma function.

        Args:
            x (int): Must be a non-negative integer value (0,1,2,....).

        Returns:
            Log-density of Poisson distribution evaluated at x.

        """
        if x < 0:
            return -np.inf
        else:
            return x * self.log_lambda - gammaln(x + 1.0) - self.lam

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized log-density evaluated on sequence encoded x.

        Arg value x (Tuple[np.ndarray[int], np.ndarray[float]]) is seq_encoded Poisson data from
        PoissonDataEncoder.seq_encode(), containing
            x[0] (np.ndarray[int]): Non-negative integer valued Poisson iid observations,
            x[1] (np.ndarray[float]): np.log(Gamma(x[0]+1.0)), Gamma is the gamma function.

        Args:
            x: See above for details.

        Returns:
            Numpy array of log-density evaluated at each encoded observation value x.

        """
        rv = x[0] * self.log_lambda
        rv -= x[1]
        rv -= self.lam
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'PoissonSampler':
        """Create PoissonSampler object with PoissonDistribution instance and seed (Optional[int]) passed.

        Args:
            seed (Optional[int]): Optional seed for random number generator used in sampling.

        Returns:
            PoissonSampler object.

        """
        return PoissonSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'PoissonEstimator':
        """Creates PoissonEstimator object.

        Args:
            pseudo_count (Optional[float]): If passed, used to re-weight summary statistic lam from
                PoissonDistribution instance.

        Returns:
            PoissonEstimator object.

        """
        if pseudo_count is None:
            return PoissonEstimator(name=self.name)
        else:
            return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=self.lam, name=self.name)

    def dist_to_encoder(self) -> 'PoissonDataEncoder':
        """Return PoissonDataEncoder object."""
        return PoissonDataEncoder()


class PoissonSampler(DistributionSampler):

    def __init__(self, dist: 'PoissonDistribution', seed: Optional[int] = None) -> None:
        """PoissonSampler object used to draw samples from PoissonDistribution.

        Args:
            dist (PoissonDistribution): Set PoissonDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        Attributes:
            rng (RandomState): RandomState with seed set for sampling.
            dist (GeometricDistribution): PoissonDistribution to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from Poisson distribution.

        Generates a single Poisson sample (int) if size is None, else a numpy array of integers of length size
        containing iid samples, from the Poisson distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be 1.

        Returns:
            If size is None, int, else size length numpy array of ints.

        """
        return self.rng.poisson(lam=self.dist.lam, size=size)


class PoissonAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys: Optional[str] = None) -> None:
        """PoissonAccumulator object used to accumulate sufficient statistics from observed data.

        Args:
            keys (Optional[str]): Assign a string valued to key to object instance.

        Attributes:
             sum (float): Aggregate sum of weighted observations.
             count (float): Aggregate sum of observation weights.
             key (Optional[str]): Key for combining sufficient statistics with object instance containing the same key.

        """
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def initialize(self, x: int, weight: float, rng: Optional[np.random.RandomState] = None) -> None:
        """Initialize PoissonAccumulator object with weighted observation.

        Note: Just calls update().

        Args:
            x (int): Observation from Poisson distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray,
                       rng: Optional[np.random.RandomState] = None) -> None:
        """Vectorized initialization of PoissonAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

        Arg value x (Tuple[np.ndarray[int], np.ndarray[float]]) is seq_encoded Poisson data from
        PoissonDataEncoder.seq_encode(), containing
            x[0] (np.ndarray[int]): Non-negative integer valued Poisson iid observations,
            x[1] (np.ndarray[float]): np.log(Gamma(x[0]+1.0)), Gamma is the gamma function.

        Args:
            x: See above for details.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def update(self, x: int, weight: float, estimate: Optional['PoissonDistribution'] = None) -> None:
        """Update sufficient statistics for PoissonAccumulator with one weighted observation.

        Args:
            x (int): Observation from Poisson distribution.
            weight (float): Weight for observation.
            estimate (Optional[PoissonDistribution]): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += x * weight
        self.count += weight

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional['PoissonDistribution'] = None) -> None:
        """Vectorized update of PoissonAccumulator sufficient statistics with weighted observations.

        Arg value x (Tuple[np.ndarray[int], np.ndarray[float]]) is seq_encoded Poisson data from
        PoissonDataEncoder.seq_encode(), containing
            x[0] (np.ndarray[int]): Non-negative integer valued Poisson iid observations,
            x[1] (np.ndarray[float]): np.log(Gamma(x[0]+1.0)), Gamma is the gamma function.

        Args:
            x: See above for details.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[PoissonDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat: Tuple[float, float]) -> 'PoissonAccumulator':
        """Combine aggregated sufficient statistics with sufficient statistics of PoissonAccumulator instance.

        Input suff_stat is Tuple[float, float] with:
            suff_stat[0] (float): sum of observation weights,
            suff_stat[1] (float): weighted sum of observations.

        Args:
            suff_stat (Tuple[float, float]): See above for details.

        Returns:
            PoissonAccumulator object.

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        """Returns sufficient statistics Tuple[float, float] of PoissonAccumulator instance."""
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'PoissonAccumulator':
        """Sets PoissonAccumulator instance sufficient statistic member variables to x.

        Args:
            x (Tuple[float, float]): Sum of observations weights and sum of weighted observations.

        Returns:
            PoissonAccumulator object.

        """
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merges PoissonAccumulator sufficient statistics with sufficient statistics contained in suff_stat dict
        that share the same key.

        Args:
            stats_dict (Dict[str, Any]): Dict containing 'key' string for PoissonAccumulator
                objects to combine sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set the sufficient statistics of PoissonAccumulator to stats_key sufficient statistics if key is in
            stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys string ids to sufficient statistics.
                objects.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'PoissonDataEncoder':
        """Return PoissonDataEncoder object."""
        return PoissonDataEncoder()


class PoissonAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, keys: Optional[str] = None) -> None:
        """PoissonAccumulatorFactory object used for constructing PoissonAccumulator objects.

        Args:
            keys (Optional[str]): Assign keys to PoissonAccumulatorFactory object.

        Attributes:
             keys (Optional[str]): Tag for combining sufficient statistics of PoissonAccumulator objects when
                constructed.

        """
        self.keys = keys

    def make(self) -> 'PoissonAccumulator':
        """Returns PoissonAccumulator object with keys passed."""
        return PoissonAccumulator(keys=self.keys)


class PoissonEstimator(ParameterEstimator):

    def __init__(self, pseudo_count: Optional[float] = None, suff_stat: Optional[float] = None,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """PoissonEstimator object for estimating PoissonDistribution object from aggregated sufficient statistics.

        Args:
            pseudo_count (Optional[float]): Optional non-negative float.
            suff_stat (Optional[float]): Optional non-negative float.
            name (Optional[str]): Assign a name to PoissonEstimator.
            keys (Optional[str]): Assign keys to PoissonEstimator for combining sufficient statistics.

        Attributes:
            pseudo_count (Optional[float]): Re-weight suff_stat.
            suff_stat (Optional[float]): Mean of Poisson if not None.
            name (Optional[str]): String name of PoissonEstimator instance.
            keys (Optional[str]): String keys of PoissonEstimator instance for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> 'PoissonAccumulatorFactory':
        """Return PoissonAccumulatorFactory object with name and keys passed."""
        return PoissonAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'PoissonDistribution':
        """Estimate lambda of PoissonDistribution from aggregated sufficient statistcs suff_stat.

        Arg passed suff_stat is a Tuple of two floats containing:
            suff_stat[0] (float): Aggregated sum of observation weights,
            suff_stat[1] (float): Aggregated sum of weighted observations.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See above for details.

        Returns:
            PoissonDistribution object.

        """
        nobs, psum = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            return PoissonDistribution((psum + self.suff_stat * self.pseudo_count) / (nobs + self.pseudo_count),
                                       name=self.name)
        else:
            return PoissonDistribution(psum / nobs, name=self.name)


class PoissonDataEncoder(DataSequenceEncoder):
    """GeometricDataEncoder object for encoding sequences of iid Poisson observations with data type int."""

    def __str__(self) -> str:
        """Returns string representation of PoissonDataEncoder object."""
        return 'PoissonDataEncoder'

    def __eq__(self, other) -> bool:
        """Checks if object is equivalent to PoissonDataEncoder instance.

        Args:
            other (object): Object to be compared to self.

        Returns:
            True if other is GeometricDataEncoder instance, else False.

        """
        return isinstance(other, PoissonDataEncoder)

    def seq_encode(self, x: Union[np.ndarray, Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode iid sequence of Poisson observations for vectorized "seq_" function calls.

        Data type must be int. Values must be non-negative integers. Non-integer values are converted to integer.
        Returns Tuple of np.ndarray[int] of x, and np.log(Gamma(x+1.0)), where Gamma is the Gamma function.

        Args:
            x (Union[np.ndarray, Sequence[int]]): Sequence of iid non-negative integers valued Poisson observations.

        Returns:
            Tuple[ndarray[int], ndarray[float]].

        """
        rv1 = np.asarray(x)

        if np.any(rv1 < 0) or np.any(np.isnan(rv1)):
            raise Exception('Poisson requires non-negative integer values of x.')
        else:
            rv2 = gammaln(rv1 + 1.0)
            return rv1, rv2
