"""Create, estimate, and sample from a Categorical distribution defined on a range of integers starting a user
defined minimum value.

Defines the IntegerCategoricalDistribution, IntegerCategoricalSampler, IntegerCategoricalAccumulatorFactory,
IntegerCategoricalAccumulator, IntegerCategoricalEstimator, and the IntegerCategoricalDataEncoder classes for use
with pysparkplug.

Data type (int): The integer categorical distribution is defined through summary statistics min_val (int)
and vector of probabilities p_vec (np.ndarray[float]) that sum to 1.0. The range of values is given by
[min_val, min_val + len(p_vec) - ). The density is then,

    P(x_mat=i) = p_vec[i]

for x in {min_val,min_val+1, ..., min_val + length(p_vec) - 1}, else 0.0.

"""
import numpy as np
from numpy.random import RandomState
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableStatisticAccumulator, SequenceEncodableProbabilityDistribution, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, \
    DistributionEnumerator, EnumerationError
from pysp.utils.aliasing import coalesce_alias, MISSING
from pysp.utils.enumeration import QuantizedCrossIndex, QuantizedEnumerationIndex
from typing import List, Union, Tuple, Optional, Dict, Any, Sequence


class IntegerCategoricalDistribution(SequenceEncodableProbabilityDistribution):

    """Categorical distribution over a bounded integer range."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.capabilities import DistributionCapabilities
        return DistributionCapabilities(engine_ready=('numpy', 'torch'), kernel_status='numba_adapter')

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec
        return DistributionDeclaration(
            name='integer_categorical',
            distribution_type=cls,
            parameters=(
                ParameterSpec('min_val', constraint='integer', differentiable=False),
                ParameterSpec('p_vec', constraint='simplex_vector'),
            ),
            statistics=(
                StatisticSpec('min_val', kind='support_bound', additive=False, scales=False),
                StatisticSpec('count_vec', kind='count_vector'),
            ),
            support='bounded_integer',
        )

    def __init__(self, min_val: int, p_vec: Union[List[float], np.ndarray] = MISSING,
                 name: Optional[str] = None, prob_vec: Union[List[float], np.ndarray] = MISSING) -> None:
        """IntegerCategoricalDistribution object defining an integer categorical distribution.

        Args:
            min_val (int): Minimum value of the integer categorical support.
            p_vec (Union[List[float], np.ndarray]): Probability vector containing probability of each integer in the
                support range.
            name (Optional[str]): Assign name to IntegerCategoricalDistribution object.

        Attributes:
            p_vec (np.ndarray[float]): Must sum to 1.0. First probability is probability for p_mat(x_mat=min_val).
            min_val (int): Minimum value in support of integer categorical
            max_val (int): Maximum value in support of integer categorical set to min_val + length(p_vec) - 1.
            log_p_vec (np.ndarray[float]): Log of p_vec.
            num_vals (int): Total number of values in support of IntegerCategoricalDistribution instance.

        """
        p_vec = coalesce_alias('p_vec', p_vec, 'prob_vec', prob_vec, default=MISSING)
        with np.errstate(divide='ignore'):
            self.p_vec = np.asarray(p_vec, dtype=np.float64)
            self.min_val = min_val
            self.max_val = min_val + self.p_vec.shape[0] - 1
            self.log_p_vec = np.log(self.p_vec)
            self.num_vals = self.p_vec.shape[0]
            self.name = name

    def __str__(self) -> str:
        """Return a string representation of IntegerCategoricalDistribution object."""
        s1 = str(self.min_val)
        s2 = repr(list(self.p_vec))
        s3 = repr(self.name)

        return 'IntegerCategoricalDistribution(%s, %s, name=%s)' % (s1, s2, s3)

    def density(self, x: int) -> float:
        """Evaluate the density of the integer categorical at observation x.

        p_mat(x_mat=x) = p_vec[x] if x in support [min_val, max_val], else 0.0.

        Args:
            x (int): Integer value.

        Returns:
            Density at x.

        """
        return zero if x < self.min_val or x > self.max_val else self.p_vec[x - self.min_val]

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of the integer categorical at observation x.

        log_p(x_mat=x) = log_p_vec[x] if x in support [min_val, max_val], else -np.inf.

        Args:
            x (int): Integer value.

        Returns:
            Log-density at x.

        """
        return -inf if (x < self.min_val or x > self.max_val) else self.log_p_vec[x - self.min_val]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of IntegerCategorical log_density() for sequence encoded iid observations x.

        Args:
            x (np.ndarray[int]): Sequence encoded iid observation of integer categorical distribution.

        Returns:
            Numpy array of floats containing log_density() evaluated at each observation in x.

        """
        v = x - self.min_val
        u = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(x))
        rv.fill(-np.inf)
        rv[u] = self.log_p_vec[v[u]]

        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, min_val: int, log_p_vec: Any, engine: Any) -> Any:
        """Engine-neutral integer-categorical log-density from explicit parameters."""
        v = x - engine.asarray(min_val)
        good = (v >= 0) & (v < len(log_p_vec))
        safe_v = engine.clip(v, 0, len(log_p_vec) - 1)
        return engine.where(good, log_p_vec[safe_v], engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        xx = engine.asarray(x)
        return self.backend_log_density_from_params(xx, self.min_val, engine.asarray(self.log_p_vec), engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence['IntegerCategoricalDistribution'],
                               engine: Any) -> Dict[str, Any]:
        """Return stacked integer-categorical parameters for a homogeneous mixture kernel."""
        min_val = dists[0].min_val
        num_vals = dists[0].num_vals
        if any(d.min_val != min_val or d.num_vals != num_vals for d in dists):
            raise ValueError('Stacked IntegerCategoricalDistribution components require shared support.')
        return {
            '__pysp_component_axis__': {'log_p': 1},
            'min_val': min_val,
            'num_vals': num_vals,
            'log_p': engine.asarray(np.stack([d.log_p_vec for d in dists], axis=1)),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: Dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of integer-categorical log densities."""
        xx = engine.asarray(x)
        v = xx - engine.asarray(params['min_val'])
        good = (v >= 0) & (v < params['num_vals'])
        safe_v = engine.clip(v, 0, params['num_vals'] - 1)
        rv = params['log_p'][safe_v, :]
        return engine.where(good[:, None], rv, engine.asarray(-np.inf))

    @classmethod
    def backend_stacked_sufficient_statistics(cls, x: Any, weights: Any,
                                              params: Dict[str, Any], engine: Any) -> Tuple[Any, Any]:
        """Return component-stacked legacy ``(min_val, count_vec)`` statistics."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        rel = xx - engine.asarray(params['min_val'])
        rows = []
        for i in range(int(params['num_vals'])):
            mask = rel == engine.asarray(i)
            rows.append(engine.sum(ww * mask[:, None], axis=0))
        if rows:
            count_mat = engine.stack(rows, axis=1)
        else:
            count_mat = engine.zeros((tuple(getattr(ww, 'shape', (0, 0)))[1], 0))
        min_vals = engine.asarray(np.full(int(tuple(getattr(ww, 'shape', (0, 0)))[1]), int(params['min_val'])))
        return min_vals, count_mat

    def sampler(self, seed: Optional[int] = None) -> 'IntegerCategoricalSampler':
        """IntegerCategoricalSampler object for sampling from IntegerCategoricalDistribution instance.

        Args:
            seed (Optional[int]): Set seed for drawing random samples.

        Returns:
            IntegerCategoricalSampler object.

        """
        return IntegerCategoricalSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerCategoricalEstimator':
        """IntegerCategoricalEstimator object from instance of IntegerCategoricalDistribution object.

        If pseudo_count is not None, pass min_val and p_vec as sufficient statistics for aggregated estimaton.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics of IntegerCategoricalDistribution
                instance in estimation.

        Returns:
            IntegerCategoricalEstimator object.

        """
        if pseudo_count is None:
            return IntegerCategoricalEstimator(name=self.name)

        else:
            return IntegerCategoricalEstimator(pseudo_count=pseudo_count, suff_stat=(self.min_val, self.p_vec),
                                               name=self.name)

    def dist_to_encoder(self) -> 'IntegerCategoricalDataEncoder':
        """Return IntegerCategoricalDataEncoder object for encoding sequences of iid integer categorical
            observations."""
        return IntegerCategoricalDataEncoder()

    def enumerator(self) -> 'IntegerCategoricalEnumerator':
        """Return IntegerCategoricalEnumerator iterating the support in descending probability order."""
        return IntegerCategoricalEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index directly from the finite integer support."""
        items = [(self.min_val + i, float(lp)) for i, lp in enumerate(self.log_p_vec)]
        return QuantizedEnumerationIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over integer categorical ranges."""
        dists = [self] + list(others)
        if any(not isinstance(dist, IntegerCategoricalDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)

        lo = min(dist.min_val for dist in dists)
        hi = max(dist.max_val for dist in dists)
        items = []
        for value in range(lo, hi + 1):
            items.append((value, tuple(float(dist.log_density(value)) for dist in dists)))
        return QuantizedCrossIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over two integer categorical ranges."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class IntegerCategoricalEnumerator(DistributionEnumerator):

    def __init__(self, dist: IntegerCategoricalDistribution) -> None:
        """Enumerates the support [min_val, max_val] in descending probability order.

        Zero-probability entries of p_vec are skipped.

        Args:
            dist (IntegerCategoricalDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        order = np.argsort(-dist.log_p_vec, kind='stable')
        self._order = [int(i) for i in order if dist.p_vec[i] > 0.0]
        self._pos = 0

    def __next__(self) -> Tuple[int, float]:
        if self._pos >= len(self._order):
            raise StopIteration
        i = self._order[self._pos]
        self._pos += 1
        return (self.dist.min_val + i, float(self.dist.log_p_vec[i]))


class IntegerCategoricalSampler(DistributionSampler):

    def __init__(self, dist: 'IntegerCategoricalDistribution', seed: Optional[int] = None) -> None:
        """IntegerCategoricalSampler object for sampling from IntegerCategoricalDistribution.

        Args:
            dist (IntegerCategoricalDistribution): Set IntegerCategoricalDistribution instance to sample from.
            seed (Optional[int]): Set the seed for random number generator used to sample.

        Attributes:
            dist (IntegerCategoricalDistribution): IntegerCategoricalDistribution instance to sample from.
            rng (RandomState): RandomState object with seed set if passed.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, List[int]]:
        """Draw iid samples from IntegerCategoricalSampler object.

        Note: If size is None, a single sample is returned as an integer. If size > 0, a List of integers with
        length equal to size is returned.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Integer or List[int] of iid samples from IntegerCategoricalSampler instance.

        """
        if size is None:
            return self.rng.choice(range(self.dist.min_val, self.dist.max_val + 1), p=self.dist.p_vec)

        else:
            return list(self.rng.choice(range(self.dist.min_val, self.dist.max_val + 1), p=self.dist.p_vec, size=size))


class IntegerCategoricalAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None, keys: Optional[str] = None) \
            -> None:
        """IntegerCategoricalAccumulator object for accumulating sufficient statistics from observed data.

        If min_val and max_val are not provided, they are obtained from the data in accumulation step.

        Args:
            min_val (Optional[int]): Sets the minimum value of integer categorical range.
            max_val (Optional[int]): Sets the maximum value of integer categorical range.
            keys (Optional[str]): Set key for merging sufficient statistics of integer IntegerCategoricalAccumulator
                objects.

        Attributes:
            min_val (Optional[int]): Minimum value of integer categorical range.
            max_val (Optional[int]): Maximum value of integer categorical range.
            count_vec (Optional[np.ndarray]): Numpy array of floats for tracking probability weights for each integer
                value in support. Set to None if min_val and max_val are both not None.
            keys (Optional[str]): Key for merging sufficient statistics of integer IntegerCategoricalAccumulator
                objects.

        """
        self.min_val = min_val
        self.max_val = max_val

        if min_val is not None and max_val is not None:
            self.count_vec = vec.zeros(max_val - min_val + 1)

        else:
            self.count_vec = None

        self.key = keys

    def update(self, x: int, weight: float, estimate: Optional['IntegerCategoricalDistribution']) -> None:
        """Update sufficient statistics for IntegerCategoricalAccumulator with one weighted observation.

        If min_val and max_val are not set, count_vec is created. If x is larger than max_val of x is less than min_val
        a new value for max_val/min_val is set, and count_vec is increased to account for the new support range.

        Args:
            x (int): Observation from integer categorical distribution.
            weight (float): Weight for observation.
            estimate (Optional[ntegerCategoricalDistribution]): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """

        if self.count_vec is None:
            self.min_val = x
            self.max_val = x
            self.count_vec = np.asarray([weight])

        elif self.max_val < x:
            temp_vec = self.count_vec
            self.max_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[:len(temp_vec)] = temp_vec
            self.count_vec[x - self.min_val] += weight

        elif self.min_val > x:
            temp_vec = self.count_vec
            temp_diff = self.min_val - x
            self.min_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[temp_diff:] = temp_vec
            self.count_vec[x - self.min_val] += weight

        else:
            self.count_vec[x - self.min_val] += weight

    def initialize(self, x: int, weight: float, rng: RandomState) -> None:
        """Initialize IntegerCategoricalAccumulator object with weighted observation

        Note: Just calls update().

        Args:
            x (int): Observation from integer categorical distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        return self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of IntegerCategoricalAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

        Args:
            x (np.ndarray[int]): Sequence encoded iid observations of integer categorical distribution.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        return self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional['IntegerCategoricalDistribution'])\
            -> None:
        """Vectorized update of IntegerCategoricalAccumulator sufficient statistics with sequence encoded iid
            observations x.

        Note: Determines the range (support) of integer categorical from the sequence encoded data.

        Args:
            x (np.ndarray[int]): Sequence encoded iid observations of integer categorical distribution.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[IntegerCategoricalDistribution]): Previous estimate of IntegerCategoricalDistribution.

        Returns:
            None.

        """
        min_x = x.min()
        max_x = x.max()

        loc_cnt = np.bincount(x - min_x, weights=weights)

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff:(prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff:(min_diff + len(loc_cnt))] += loc_cnt

    def combine(self, suff_stat: Tuple[Optional[int], Optional[np.ndarray]]) -> 'IntegerCategoricalAccumulator':
        """Combine aggregated sufficient statistics with sufficient statistics of IntegerCategoricalAccumulator
            instance.

        Arg passed suff_stat is sufficient statistics a Tuple of length two containing:
            suff_stat[0] (int): Minimum value of the integer categorical,
            suff_stat[1] (np.ndarray[float]): Numpy array containing probabilities for each integer value. This also
                sets the support of integer categorical to have a maximum value of suff_stat[0] + len(suff_stat[0]) - 1.

        Member variables min_val, max_val, and count_vec are set from suff_stat arg if count_vec is None. Else,
        suff_stat is combined with the values of min_val, max_val, and count_vec.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerCategoricalAccumulator object.

        """
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val = suff_stat[0]
            self.max_val = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:
            if self.min_val == suff_stat[0] and len(self.count_vec) == len(suff_stat[1]):
                self.count_vec += suff_stat[1]

            else:
                min_val = min(self.min_val, suff_stat[0])
                max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

                count_vec = vec.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        return self

    def value(self) -> Tuple[int, np.ndarray]:
        """Returns member sufficient statistics Tuple[int, np.ndarray[float]] of IntegerCategoricalAccumulator
            instance.

        Entry 0 of returned value is the minimum value, and entry 1 is the probability weights for each integer value
        in the support.

        """
        return self.min_val, self.count_vec

    def from_value(self, x: Tuple[int, np.ndarray]) -> 'IntegerCategoricalAccumulator':
        """Sets IntegerCategoricalAccumulator instance sufficient statistic member variables to x.

        Arg passed x is sufficient statistics a Tuple of length two containing:
            x[0] (int): Minimum value of the integer categorical,
            x[1] (np.ndarray[float]): Numpy array containing probabilities for each integer value. This also sets the
                support of integer categorical to have a maximum value of x[0] + len(x[0]) - 1.

        Args:
            x (Tuple[int, np.ndarray[float]]): See above for details.

        Returns:
            IntegerCategoricalAccumulator object.

        """
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def scale(self, c: float) -> 'IntegerCategoricalAccumulator':
        """Scale count vector while preserving integer support metadata."""
        if self.count_vec is not None:
            self.count_vec *= c
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Aggregate member sufficient statistics with sufficient statistics of objects with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to corresponding sufficient stats.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())

            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set member sufficient statistics to suff stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to corresponding sufficient stats.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'IntegerCategoricalDataEncoder':
        """Return IntegerCategoricalDataEncoder object for encoding sequences of iid integer categorical
            observations."""
        return IntegerCategoricalDataEncoder()


class IntegerCategoricalAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerCategoricalAccumulatorFactory object for creating IntegerCategoricalAccumulator object.

        Args:
            min_val (Optional[int]): Set minimum value of integer categorical.
            max_val (Optional[int]): Set maximum value of integer categorical.
            keys (Optional[str]): Set keys for accumulating merging statistics of IntegerCategoricalAccumulator objects.

        Attributes:
            min_val (Optional[int]): Minimum value of integer categorical, if None estimated from data.
            max_val (Optional[int]): Maximum value of integer categorical, if None estimated from data.
            keys (Optional[str]): Key used for accumulating merging statistics of IntegerCategoricalAccumulator objects.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys

    def make(self) -> 'IntegerCategoricalAccumulator':
        """Returns IntegerCategoricalAccumulator object with min_val, max_val, and keys passed."""
        return IntegerCategoricalAccumulator(self.min_val, self.max_val, self.keys)


class IntegerCategoricalEstimator(ParameterEstimator):

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[int, np.ndarray]] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerCategoricalEstimator object for estimating IntegerCategoricalDistribution from aggregated sufficient
            statistics.

            Note: Must set either min_val and max_val, or suff_stat must be passed as arg.

            Sufficient statistics stored in estimator 'suff_stat' is a Tuple of int and np.ndarray[float],
                suff_stat[0] (int): Minimum value of the integer categorical distribution,
                suff_stat[1] (ndarray[float]): Probabilities for each integer observation in range
                    [suff_stat[0], suff_stat[0] + len(suff_stat[1])-1).

        Args:
            min_val (Optional[int]): Set minimum value of integer categorical.
            max_val (Optional[int]): Set maximum value of integer categorical.
            pseudo_count (Optional[float]): Used to re-weight suff_stat member variables in merging of sufficient
                statistics
            suff_stat: Set sufficient statistics. See above for details.
            name (Optional[str]): Assign a name to IntegerCategoricalEstimator object.
            keys (Optional[str]): Set keys for accumulating merging statistics of IntegerCategoricalAccumulator objects.

        Attributes:
            min_val (Optional[int]): Minimum value of integer categorical.
            max_val (Optional[int]): Maximum value of integer categorical.
            pseudo_count (Optional[float]): Used to re-weight suff_stat when merged with new aggregated data.
            suff_stat: See above for details.
            name (Optional[str]): Name to IntegerCategoricalEstimator object.
            keys (Optional[str]): Keys for accumulating merging statistics of IntegerCategoricalAccumulator objects.

        """
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'IntegerCategoricalAccumulatorFactory':
        """Returns IntegerCategoricalAccumulatorFactory object from member sufficient statistics of
            IntegerCategoricalEstimator.

        Note: If min_val and max_val are BOTH not None, these values are passed to IntegerCategoricalAccumulatorFactory.
        Else, they are obtained from member variable suff_stat. One of these conditions must be satisfied.

        Returns:

        """
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        return IntegerCategoricalAccumulatorFactory(min_val, max_val, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Optional[Tuple[int, np.ndarray]])\
            -> 'IntegerCategoricalDistribution':
        """Estimate an IntegerCategoricalDistribution object from aggregating sufficient statistics.

        Arg 'suff_stat' is a Tuple of int and np.ndarray[float],
            suff_stat[0] (int): Minimum value of the integer categorical distribution,
            suff_stat[1] (ndarray[float]): Probabilities for each integer observation in range
            [suff_stat[0], suff_stat[0] + len(suff_stat[1])-1).

        Arg suff_stat is aggregated sufficient statistics obtained from observations of integer categorical data, that
        is used to estimate the integer categorical distribution. If pseudo_count is not None, the integer categorical
        is estimated by a combing arg suff_stat and a re-weighted member variable 'suff_stat'.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat:

        Returns:
            IntegerCategoricalDistribution object.

        """
        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(suff_stat[0], (suff_stat[1] + pseudo_count_per_level) / adjusted_nobs,
                                                  name=self.name)

        elif self.pseudo_count is not None and self.min_val is not None and self.max_val is not None:

            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerCategoricalDistribution(min_val, (count_vec + pseudo_count_per_level) / adjusted_nobs,
                                                  name=self.name)

        elif self.pseudo_count is not None and self.suff_stat is not None:

            s_max_val = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            s_min_val = self.suff_stat[0]

            min_val = min(s_min_val, suff_stat[0])
            max_val = max(s_max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = s_min_val - min_val
            i1 = s_max_val - min_val + 1
            count_vec[i0:i1] = self.suff_stat[1] * self.pseudo_count

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            return IntegerCategoricalDistribution(min_val, count_vec / (count_vec.sum()), name=self.name)

        else:
            return IntegerCategoricalDistribution(suff_stat[0], suff_stat[1] / (suff_stat[1].sum()), name=self.name)


class IntegerCategoricalDataEncoder(DataSequenceEncoder):
    """IntegerCategoricalDataEncoder object for encoding sequences of iid integer categorical observations."""

    def __str__(self) -> str:
        """Returns IntegerCategoricalDataEncoder object for encoding data sequences."""
        return 'IntegerCategoricalDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Return True if other is an IntegerCategoricalDataEncoder, False is else."""
        return isinstance(other, IntegerCategoricalDataEncoder)

    def seq_encode(self, x: Union[List[int], np.ndarray]) -> np.ndarray:
        """Sequence encode iid integer categorical observations for "seq_" functions.

        Args:
            x (Union[List[int], np.ndarray]): Assumed int observations of integer categorical.

        Returns:
            Numpy array of integers.

        """
        return np.asarray(x, dtype=int)
