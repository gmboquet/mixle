"""Create, estimate, and sample from a Categorical distribution.

Defines the CategoricalDistribution, CategoricalSampler, CategoricalAccumulatorFactory, CategoricalAccumulator,
CategoricalEstimator, and the CategoricalDataEncoder classes for use with pysparkplug.

Data type: Any. The data type is taken as the categorical object and a probability is estimated.

If Data type is int, consider using pysp.stats.intrange (IntegerCategoricalDistribution) instead.

"""
import numpy as np
import math
from typing import Dict, Optional, Tuple, Any, TypeVar, Union, List
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from numpy.random import RandomState

T = TypeVar('T')

class CategoricalDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, pmap: Dict[Any, float], default_value: float = 0.0, name: Optional[str] = None) -> None:
        """Defines a CategoricalDistribution object for data type T.

        Density: For n observations of any data type, with support {x_0,x_1,....,x_{n-1}} the probability of a
            categorical observation is given by,
                Prob(x_mat) = p_i, if x_mat = x_i,
                Prob(x_mat) = default_value, if x_mat != x_i for any i.
            Note: default_value is set to 0.0 by default.

        Args:
            pmap (Dict[Any, float]): Keys (x_i) are the support of the categorical, the value is the probability of
                the key (p_i).
            default_value float: Value for prob of observation outside support of CategorialDistribution.
            name (str): Assigns a name to the CategoricalDistribution object.

        Attributes:
            name (str): Assigns a name to the CategoricalDistribution object.
            pmap (Dict[Any, float]): Keys (x_i) are the support of the categorical, the value is the probability of
                the key (p_i).
            default_value (float): Value for prob of observation outside support of CategorialDistribution, default to
                0.0.
            no_default (bool): True if a non-zero default value is given.
            log_default_value (float): log(default_value).
            log1p_default_value (float): log(1+default_value).

        """
        self.name = name
        self.pmap = pmap
        self.no_default = default_value != 0.0
        self.default_value = max(0.0, min(default_value, 1.0))
        self.log_default_value = float(-np.inf if default_value == 0 else math.log(default_value))
        self.log1p_default_value = float(math.log1p(default_value))



    def __str__(self) -> str:
        """Object string with member variables for CategoricalDistribution.

        Returns:
            String with pmap, defualt_value, and name printed.

        """
        s1 = ', '.join(['%s: %s' % (repr(k), repr(v)) for k, v in sorted(self.pmap.items(), key=lambda u: u[0])])
        s2 = repr(self.default_value)
        s3 = repr(self.name)

        return 'CategoricalDistribution({%s}, default_value=%s, name=%s)' % (s1, s2, s3)

    def density(self, x: Any) -> float:
        """Density evaluation of CategoricalDistribution.

        p_mat(x) = p_i, if x in pmap.keys(), else p_mat(x) = default_value.

        Args:
            x (Any): Evaluate CategoricalDistribution density value at x.

        Returns:
            float density value at x

        """
        return self.pmap.get(x, self.default_value) / (1.0 + self.default_value)

    def log_density(self, x: Any) -> float:
        """Log-Density evaluation of CategoricalDistribution.

        log(p_mat(x)) = log(p_i), if x in pmap.keys(), else log(p_mat(x)) = log(default_value).

        Args:
            x (Any): Evaluate CategoricalDistribution density value at x.

        Returns:
            Log-density of Categorical distribution evaluated at x.

        """
        return np.log(self.pmap.get(x, self.default_value)) - self.log1p_default_value

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of log-density for sequence encoded data.

        Input value x must be obtained from a call to CategoricalDataEncoder.seq_encode(data). Returns numpy array
        of log-density evaluated at all observations contained in encoded data x.

        Args:
            x: (Tuple[np.ndarray,np.ndarray]): Tuple of numpy indices for unique categories, and numpy array unique
                objects that index xs maps to.

        Returns:
            Numpy array of log-density evaluated at all observations contained in encoded data x.

        """
        with np.errstate(divide='ignore'):
            xs, val_map_inv = x
            mapped_log_prob = np.asarray([self.pmap.get(u, self.default_value) for u in val_map_inv], dtype=np.float64)
            np.log(mapped_log_prob, out=mapped_log_prob)
            mapped_log_prob -= self.log1p_default_value
            rv = mapped_log_prob[xs]

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'CategoricalSampler':
        """Creates CategoricalSampler for sampling from CategoricalDistribution.

        Args:
            seed (Optional[int]): Seed for setting random number generator used to sample.

        Returns:
            CategoricalSampler object.

        """
        return CategoricalSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'CategoricalEstimator':
        """Creates a CategoricalEstimator for estimating parameters of CategoricalDistribution.

        Args:
            pseudo_count (Optional[float]): If set, inflates counts for currently set sufficient statistic (pmap).

        Returns:
            CategoricalEstimator object.
        """
        if pseudo_count is None:
            return CategoricalEstimator(name=self.name)

        else:
            return CategoricalEstimator(pseudo_count=pseudo_count, suff_stat=self.pmap, name=self.name)

    def dist_to_encoder(self) -> 'CategoricalDataEncoder':
        """Creates a CategoricalDataEncoder object for sequence encoding data.

        Returns:
            CategoricalDataEncoder object.

        """
        return CategoricalDataEncoder()

class CategoricalSampler(DistributionSampler):

    def __init__(self, dist: CategoricalDistribution, seed: Optional[int] = None) -> None:
        """CategoricalSampler object used to generate samples from CategoricalDistribution.

        Args:
            dist (CategoricalDistribution): CategoricalDistribution used to draw samples from.
            seed (Optional[int]): Seed for setting random number generator used to sample.

        Attributes:
             rng (RandomState): RandomState with seed set to seed if provided. Else just RandomState().
             levels (List[Any]): Category labels for the CategoricalDistribution.
             probs (List[float]): Probabilities for each category in CategoricalDistribution.
             num_levels (int): Total number of categories. I.e. len(levels).

        """
        self.rng = RandomState(seed)
        temp            = list(dist.pmap.items())
        self.levels     = [u[0] for u in temp]
        self.probs      = [u[1] for u in temp]
        self.num_levels = len(self.levels)

    def sample(self, size: Optional[int] = None) -> Union[Any, List[Any]]:
        """Draw size-number of samples from CategoricalSampler object.

        If size is not provided, size is assumed = 1. If size > 1, a list is returned.

        Args:
            size (Optional[int]): Number of samples to be draw. If size is None, size = 1.

        Returns:
            List of levels if size > 1, else a single sample from levels with prob probs.

        """
        if size is None:
            idx = self.rng.choice(self.num_levels, p=self.probs, size=size)
            return self.levels[idx]

        else:
            levels = self.levels
            rv = self.rng.choice(self.num_levels, p=self.probs, size=size)

            return [levels[i] for i in rv]

class CategoricalAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys: Optional[str] = None) -> None:
        """CategoricalAccumulator object used for aggregating sufficient statistics of CategoricalDistribution.

        Sufficient statistics: count_map: Dict[category, category_count]

        Args:
            keys (Optional[str]): All CategoricalAccumulators with same keys will have suff-stats merged.

        Attributes:
            count_map (Dict[Any,float]): Keys (x_i) are the support of the categorical, the value is the weighted count
                of category obersvations.

        """
        self.count_map = dict()
        self.key = keys

    def update(self, x: Any, weight: float, estimate: Optional['CategoricalDistribution']) -> None:
        """Adds weight to the category_count for category x.

        If x is new Category label, a new key in the dict count_map is created and the count is incremented by weight.

        Args:
            x (Any): Category label.
            weight (float): Weight for the observation x.
            estimate (Optional['CategoricalDistribution']): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None, updates sufficient_stat of Accumulator, count_map.

        """
        self.count_map[x] = self.count_map.get(x, 0.0) + weight

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        """Initializes the CategoricalAccumulator sufficient statistics one observation at a time.

        Note: this is just a call to update, since there is no randomness in initialization.

        Args:
            x (Any): Category label.
            weight (float): Weight incrementing suff stat count_map counts for the observation x.
            rng (Optional[RandomState]): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None, initializes sufficient_stat of Accumulator, count_map.

        """
        self.update(x, weight, None)

    def get_seq_lambda(self):
        return [self.seq_update]

    def seq_update(self, x: (Tuple[np.ndarray, np.ndarray]), weights: np.ndarray,
                   estimate: Optional['CategoricalDistribution']) -> None:
        """Vectorized accumulation of Categorical sufficient statistics from encoded sequence of data.

        Requires data as encoded sequence from CategoricalDataEncoder.seq_encode(data).

        Args:
            x (Tuple[np.ndarray,np.ndarray]): Tuple of numpy indices for unique categories, and numpy array unique
                objects that index xs maps to.
            weights (np.ndarray): weights for each observation in encoded data set.
            estimate (Optional['CategoricalDistribution']): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        inv_key_map = x[1]
        bcnt = np.bincount(x[0], weights=weights)

        if len(self.count_map) == 0:
            self.count_map = dict(zip(inv_key_map, bcnt))

        else:
            for i in range(0, len(bcnt)):
                self.count_map[inv_key_map[i]] += bcnt[i]

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        """Vectorized initialization of Categorical sufficient statistics from encoded sequence of data.

        Requires data as encoded sequence from CategoricalDataEncoder.seq_encode(data).
        Note: this is just a call to seq_update, since there is no randomness in initialization.

        Args:
            x (Tuple[np.ndarray,np.ndarray]): Tuple of numpy indices for unique categories, and numpy array unique
                objects that index xs maps to.
            weights (np.ndarray): weights for each observation in encoded data set.
            rng (Optional[RandomState]): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        return self.seq_update(x, weights, None)

    def combine(self, suff_stat: Dict[Any, float]) -> 'CategoricalAccumulator':
        """Combine the sufficient statistics of CategoricalAccumulator with suff_stat.

        Args:
            suff_stat (Dict[Any, float]): Prior data observations aggregated into dictionary with category levels
                as keys and counts as values.

        Returns:
            None, updates the count_map of CategoricalAccumulator.

        """
        for k, v in suff_stat.items():
            self.count_map[k] = self.count_map.get(k, 0.0) + v

        return self

    def value(self) -> Dict[Any, float]:
        """Returns sufficient statistic of CategoricalAccumulator.

        Sufficient statistic value is a dictionary with category as keys and counts of categories as values.

        Returns:
            Dict[Any, float] of sufficient statistic.

        """
        return self.count_map.copy()

    def from_value(self, x: Dict[Any, float]) -> 'CategoricalAccumulator':
        """Set CategoricalAccumulator sufficient statistics and member variables from suff_stat dict defined in value().

        Takes sufficient statistic value from dictionary with category as keys and counts of categories as values. Sets
        count_map to the passed value x.

        Args:
            x (Dict[Any, float]): Dictionary with category as keys and counts of categories as values

        Returns:
            CategoricalAccumulator with member variable sufficient statistics set to x.

        """
        self.count_map = x

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Combines the sufficient statistics of CategoricalAccumulators that have the same key value.

        Args:
            stats_dict (Dict[str, Any]): Dictionary for mapping keys to CategoricalAccumulators.

        Returns:
            None

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())

            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set CategoricalAccumulator sufficient statistic member variables to the value of stats_dict
            accumualator with same stats_dict key as member variable key.

        Args:
            stats_dict (Dict[str, Any]): Maps member variable key to CategoricalAccumulator with
                same key.

        Returns:
            None

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'CategoricalDataEncoder':
        """Creates a CategoricalDataEncoder object for sequence encoding data.

        Returns:
            CategoricalDataEncoder object.

        """
        return CategoricalDataEncoder()

class CategoricalAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, keys: Optional[str] = None) -> None:
        """CategoricalAccumulatorFactory object used for lightweight construction of Accumulators.

        Args:
            keys (Optional[str]): Declare keys for merging sufficient statistics of CategoricalAccumulators.

        """
        self.keys = keys

    def make(self) -> 'CategoricalAccumulator':
        """Return a CategoricalAccumulator with keys passed.

        Returns:
            CategoricalAccumulator
        """
        return CategoricalAccumulator(keys=self.keys)

class CategoricalEstimator(ParameterEstimator):

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Dict[Any, float]] = None,
                 default_value: bool = False,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """CategoricalEstimator used to estimate CategoricalDistribution from sufficient statistics and create
        AccumulatorFactory objects.

        Args:
            pseudo_count (Optional[float]): Inflate sufficient statistic counts by pseudo_count.
            suff_stat (Optional[Dict[Any, float]]): Dictionary with category labels and probabilities as values.
            default_value (bool): True is default value should be set.
            name (Optional[str]): Assign name to be passed to Distribution, Accumulator, ect.
            keys (Optional[str]): Assign key to Estimator designating all same key estimators to later be combined,
                in accumulation.

        Attributes:
            pseudo_count (Optional[float]): Inflate sufficient statistic counts by pseudo_count.
            suff_stat (Optional[Dict[Any, float]]): Dictionary with category labels and probabilities as values.
            default_value (bool): True is default value should be set.
            name (Optional[str]): Assign name to be passed to Distribution, Accumulator, ect.
            keys (Optional[str]): Assign key to Estimator designating all same key estimators to later be combined,
                in accumulation.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.default_value = default_value
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> 'CategoricalAccumulatorFactory':
        """Create CategoricalAccumulatorFactory with keys passed is set.

        Returns:
            CategoricalAccumulatorFactory

        """
        return CategoricalAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Dict[Any, float]) -> 'CategoricalDistribution':
        """Estimate a CategoricalDistribution from suff_stat value.

        If default_value is True, we estimate a default value from the suff_stat counts. Else, it is set to 0.0.

        pseudo_count is used to averaged over the number of levels and added to the corresponding counts.

        If suff_stat member value is None, estimate for CategoricalDistribution is formed from the suff_stat passed.
        Otherwise, the suff_stat member value is combined with the suff_stat values passed to estimate.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.estimate.
            suff_stat (Dict[Any, float]): Dict with categories as keys and counts as values from accumulated data.

        Returns:
            CategoricalDistribution estimated from passed in suff_stat value and sufficient statistic member variable
                (if it is not None).

        """
        stats_sum = sum(suff_stat.values())

        if self.default_value:
            if stats_sum > 0:
                default_value = 1.0/stats_sum
                default_value *= default_value

            else:
                default_value = 0.5
        else:
            default_value = 0.0

        if self.pseudo_count is None and self.suff_stat is None:
            nobs_loc = stats_sum

            if nobs_loc == 0.0:
                p_map = {k : 1.0/float(len(suff_stat)) for k in suff_stat.keys()}
            else:
                p_map = {k: v / nobs_loc for k, v in suff_stat.items()}

        elif self.pseudo_count is not None and self.suff_stat is None:
            nobs_loc = stats_sum
            pseudo_count_per_level = self.pseudo_count/len(suff_stat)
            adjusted_nobs = nobs_loc + self.pseudo_count

            for k, v in suff_stat.items():
                suff_stat[k] = (v + pseudo_count_per_level) / adjusted_nobs

            p_map = suff_stat

        else:
            suff_stat_sum = sum(self.suff_stat.values())

            levels = set(suff_stat.keys()).union(self.suff_stat.keys())
            adjusted_nobs = suff_stat_sum * self.pseudo_count + stats_sum

            p_map = {k: (suff_stat.get(k, 0) + self.suff_stat.get(k, 0) * self.pseudo_count) / adjusted_nobs for k in levels}

        return CategoricalDistribution(pmap=p_map, default_value=default_value, name=self.name)

class CategoricalDataEncoder(DataSequenceEncoder):
    """CategoricalDataEncoder for encoding Categorical data for use with vectorized "seq_" functions."""

    def __str__(self) -> str:
        """Print out name of DataSequenceEncoder.

        Returns:
            (str) CategoricalDataEncoder.

        """
        return 'CategoricalDataEncoder'

    def __eq__(self, other) -> bool:
        """Define equivilence for CategoricalDataEncoder.

        Args:
            other (object): Check if object is CategoricalDataEncoder.

        Returns:
            True if object is CategoricalDataEncoder, else False.

        """
        return isinstance(other, CategoricalDataEncoder)

    def seq_encode(self, x: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Sequence encode list of categories for use with vectorized "seq_" functions.

        Args:
            x (List[Any]): List of category labels.

        Returns:
            Tuple of numpy indicies for unique categories in x, and numpy array unique objects that index xs maps to.

        """
        val_map_inv, uidx, xs = np.unique(x, return_index=True, return_inverse=True)
        val_map_inv = np.asarray([x[i] for i in uidx], dtype=object)

        return xs, val_map_inv

