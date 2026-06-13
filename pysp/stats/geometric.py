"""Create, estimate, and sample from a geometric distribution with probability of success p.

Defines the GeometricDistribution, GeometricSampler, GeometricAccumulatorFactory, GeometricAccumulator,
GeometricEstimator, and the GeometricDataEncoder classes for use with pysparkplug.

Data type (int): The geometric distribution with probability of success p, has density

    P(x=k) = (k-1)*log(1-p) + log(p), for k = 1,2,...

"""
import math
import numpy as np
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, \
    DistributionEnumerator
from pysp.utils.enumeration import QuantizedCrossIndex, QuantizedEnumerationIndex
from numpy.random import RandomState
from typing import Optional, Tuple, Sequence, Dict, Union, Any


class GeometricDistribution(SequenceEncodableProbabilityDistribution):
    """Geometric distribution on ``{1, 2, ...}`` with success probability ``p``."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.capabilities import DistributionCapabilities
        return DistributionCapabilities(engine_ready=('numpy', 'torch'), kernel_status='numba_adapter')

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec
        return DistributionDeclaration(
            name='geometric',
            distribution_type=cls,
            parameters=(ParameterSpec('p', constraint='unit_interval'),),
            statistics=(StatisticSpec('count'), StatisticSpec('sum')),
            support='positive_integer',
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: Dict[str, Any], engine: Any) -> Tuple[Any, ...]:
        """Return per-row Geometric sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        return xx * 0.0 + engine.asarray(1.0), xx

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
        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError('GeometricDistribution requires p in (0, 1].')
        self.p = float(p)
        self.log_p = np.log(self.p)
        self.log_1p = np.log1p(-self.p)
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
        try:
            xx = float(x)
        except Exception:
            return -np.inf
        if not np.isfinite(xx) or xx < 1 or np.floor(xx) != xx:
            return -np.inf
        if self.p == 1.0:
            return 0.0 if xx == 1 else -np.inf
        return (xx - 1.0) * self.log_1p + self.log_p

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density evaluated on sequence encoded x.

        Args:
            x (int): Numpy array of non-negative integers.

        Returns:
            Numpy array of log-density evaluated at each encoded observation value x.

        """
        xx = np.asarray(x, dtype=np.float64)
        good = np.isfinite(xx) & (xx >= 1) & (np.floor(xx) == xx)
        if self.p == 1.0:
            return np.where(good & (xx == 1), 0.0, -np.inf)
        rv = (xx - 1.0) * self.log_1p + self.log_p
        rv = np.where(good, rv, -np.inf)

        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, p: Any, engine: Any) -> Any:
        """Engine-neutral geometric log-density from explicit parameters."""
        one = engine.asarray(1.0)
        good = (x >= one) & (engine.floor(x) == x)
        log_p = engine.log(p)
        log_1p = engine.log(one - p)
        rv = (x - one) * log_1p + log_p
        at_one = engine.where((x == one) & good, engine.asarray(0.0), engine.asarray(-np.inf))
        rv = engine.where(p == one, at_one, rv)
        return engine.where(good, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        xx = engine.asarray(x)
        return self.backend_log_density_from_params(xx, engine.asarray(self.p), engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence['GeometricDistribution'], engine: Any) -> Dict[str, Any]:
        """Return stacked geometric parameters for a homogeneous mixture kernel."""
        return {'p': engine.asarray([d.p for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: Dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of geometric log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params['p'][None, :], engine)

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

    def enumerator(self) -> 'GeometricEnumerator':
        """Returns GeometricEnumerator iterating the support {1, 2, ...} in descending probability order."""
        return GeometricEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index directly from the geometric tail formula."""
        if max_bits < 0:
            raise ValueError('max_bits must be non-negative.')
        if bin_width_bits <= 0:
            raise ValueError('bin_width_bits must be positive.')

        if self.log_p == -np.inf:
            return QuantizedEnumerationIndex.from_items(
                [], max_bits=max_bits, bin_width_bits=bin_width_bits, truncated=False)

        if self.log_1p == -np.inf:
            return QuantizedEnumerationIndex.from_items(
                [(1, float(self.log_p))], max_bits=max_bits,
                bin_width_bits=bin_width_bits, sorted_items=True, truncated=False)

        limit_nats = float(max_bits) * math.log(2.0)
        max_offset = int(math.floor((limit_nats + float(self.log_p)) / (-float(self.log_1p)) + 1.0e-12))
        if max_offset < 0:
            items = []
        else:
            items = [(x, float((x - 1) * self.log_1p + self.log_p)) for x in range(1, max_offset + 2)]

        return QuantizedEnumerationIndex.from_items(
            items, max_bits=max_bits, bin_width_bits=bin_width_bits,
            sorted_items=True, truncated=True)

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over bounded geometric prefixes."""
        dists = [self] + list(others)
        if any(not isinstance(dist, GeometricDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = tuple([float(max_bits)] * len(dists))
        if len(max_bits_tuple) != len(dists):
            raise ValueError('max_bits length must match the number of distributions.')

        def max_value(dist: 'GeometricDistribution', bit_bound: float) -> int:
            if bit_bound < 0.0 or dist.log_p == -np.inf:
                return 0
            if dist.log_1p == -np.inf:
                return 1 if -float(dist.log_p) / math.log(2.0) <= bit_bound + 1.0e-12 else 0
            limit_nats = float(bit_bound) * math.log(2.0)
            max_offset = int(math.floor((limit_nats + float(dist.log_p)) / (-float(dist.log_1p)) + 1.0e-12))
            return max(0, max_offset) + 1 if max_offset >= 0 else 0

        hi = max(max_value(dist, bit_bound) for dist, bit_bound in zip(dists, max_bits_tuple))
        items = [(value, tuple(float(dist.log_density(value)) for dist in dists))
                 for value in range(1, hi + 1)]
        return QuantizedCrossIndex.from_items(
            items, max_bits=max_bits_tuple, bin_width_bits=bin_width_bits, truncated=True)

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over two bounded geometric prefixes."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class GeometricEnumerator(DistributionEnumerator):

    def __init__(self, dist: GeometricDistribution) -> None:
        """Enumerates the support {1, 2, 3, ...} of a GeometricDistribution.

        The geometric pmf is strictly decreasing in x, so the natural order is already
        the descending-probability order. The iterator is infinite for p < 1.

        Args:
            dist (GeometricDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        self._x = 1

    def __next__(self) -> Tuple[int, float]:
        x = self._x
        lp = (x - 1) * self.dist.log_1p + self.dist.log_p
        if lp == -np.inf:
            raise StopIteration
        self._x += 1
        return (x, lp)


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
        if x >= 1:
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
        self.suff_stat = max(min(suff_stat, 1.0), 0.0) if suff_stat is not None else None
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
        elif suff_stat[1] == 0.0:
            p = 0.5
        else:
            p = suff_stat[0] / suff_stat[1]

        p = float(np.clip(p, 1.0e-12, 1.0))
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
        rv = np.asarray(x, dtype=np.float64)
        if np.any(rv < 1) or np.any(np.isnan(rv)) or np.any(np.floor(rv) != rv):
            raise ValueError('GeometricDistribution requires positive integer values for x.')
        else:
            return rv
