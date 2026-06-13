"""Create, estimate, and sample from a Bernoulli set distribution.

Defines the BernoulliSetDistribution, BernoulliSetSampler, BernoulliSetAccumulatorFactory,
BernoulliSetAccumulator, BernoulliSetEstimator, BernoulliSetDataEncoder, and the BernoulliSetEnumerator
classes for use with pysparkplug.

Data type: Sequence[Any]: An observation is a set (any iterable of distinct hashable values) drawn from a
finite support S = {s_1,s_2,....,s_N}. Let x be a random subset of S. Each element s_k is included in x
independently with probability

    p_k = P(s_k is in x) , k = 1,2,...,N,

so the density of an observed set x is

    p(x) = prod_{s_k in x} p_k * prod_{s_k not in x} (1-p_k).

A comment on estimation: Note that probability of an element s_k belonging to the set is 0 if we do not encounter any
elements an observation sequence. For this reason, we need not state the support of the state-space in estimation.

"""
import numpy as np
from numpy.random import RandomState
from collections import defaultdict, OrderedDict
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, StatisticAccumulatorFactory, DistributionSampler, \
    DistributionEnumerator, EnumerationError
from pysp.utils.enumeration import BufferedStream, ProductEnumerator

from typing import Optional, Dict, Tuple, Any, Dict, List, Sequence, TypeVar, Union


class BernoulliSetDistribution(SequenceEncodableProbabilityDistribution):
    """Bernoulli set distribution: each support element is included in an observed set independently."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.capabilities import DistributionCapabilities
        return DistributionCapabilities(engine_ready=('numpy', 'torch'), kernel_status='generic_table')

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec
        return DistributionDeclaration(
            name='bernoulli_set',
            distribution_type=cls,
            parameters=(
                ParameterSpec('pmap', constraint='simplex_map'),
                ParameterSpec('min_prob', constraint='unit_interval', differentiable=False),
            ),
            statistics=(
                StatisticSpec('inclusion_counts', kind='count_map'),
                StatisticSpec('total_weight'),
            ),
            support='finite_hashable_set',
            differentiable=False,
        )

    def __init__(self, pmap: Dict[Any, float], min_prob: float = 1.0e-128, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """BernoulliSetDistribution object for creating a Bernoulli set distribution.

        Args:
            pmap (Dict[Any, float]): Maps values to probabilities.
            min_prob (float): Minimum probability for numerical stability in log prob calculations.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys for object instance.

        Attributes:
            key (Optional[str]): Keys for object instance.
            name (Optional[str]): Name to object instance.
            pmap (Dict[Any, float]): Maps elements in support to probabilities.
            required (Set): An observation must contain this subset of elements. Else, return probability 0.0.
            nlog_sum (float): Normalizing term for computing numerically stable likelihood.
            log_dmap (Dict[Any, float]):Map from elements to their corrected log probability of inclusion in the set.
            min_prob (float): Minimum probability for elements. Corrects for prob = 0.
            num_required (int): Number of required elements in a subset. Corrected if min_prob was non-zero.

        """
        self.key = keys
        self.name = name
        self.pmap = pmap
        self.required = set()
        self.nlog_sum = 0.0
        self.log_dmap = dict()

        if min_prob == 0:
            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = 0.0
                    self.required.add(k)
                elif v == 0.0:
                    self.log_dmap[k] = -np.inf
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv
            self.min_prob = 0.0
            self.num_required = len(self.required)

        else:
            min_pv = np.log(min_prob)
            min_nv = np.log1p(-min_prob)

            for k, v in pmap.items():
                if v == 1.0:
                    self.log_dmap[k] = min_nv - min_pv
                    self.nlog_sum += min_pv
                elif v == 0.0:
                    self.log_dmap[k] = min_pv - min_nv
                    self.nlog_sum += min_nv
                else:
                    vv = np.log1p(-v)
                    self.log_dmap[k] = np.log(v) - vv
                    self.nlog_sum += vv

            self.min_prob = min_prob
            self.num_required = 0

    def __str__(self) -> str:
        """Returns string representation of BernoulliSetDistribution object."""
        s1 = repr(sorted(self.pmap.items(), key=lambda t: t[0]))
        s2 = repr(self.min_prob)
        s3 = repr(self.name)
        s4 = repr(self.key)
        return 'BernoulliSetDistribution(dict(%s), min_prob=%s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: Sequence[Any]) -> float:
        """Density of the Bernoulli set distribution at observed set x.

        See log_density() for details.

        Args:
            x (Sequence[Any]): Observed set of distinct elements from the support of pmap.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[Any]) -> float:
        """Log-density of the Bernoulli set distribution at observed set x.

        Sums log(p_k / (1-p_k)) over the elements present in x, plus the constant
        sum_k log(1-p_k). Returns -inf if x is missing a required element (an element
        with p_k = 1 when min_prob is 0).

        Args:
            x (Sequence[Any]): Observed set of distinct elements from the support of pmap.

        Returns:
            Log-density at observation x.

        """
        if not self.required.issubset(x):
            return -np.inf
        rv = 0.0
        for v in x:
            rv += self.log_dmap[v]
        return self.nlog_sum + rv

    def seq_log_density(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Sequence encoded set observations from
                BernoulliSetDataEncoder.seq_encode().

        Returns:
            Numpy array of log-density values, one per encoded observation.

        """
        sz, idx, val_map_inv, xs = x

        dlog_loc = np.asarray([self.log_dmap[u] for u in val_map_inv], dtype=np.float64)

        rv = np.bincount(idx, weights=dlog_loc[xs], minlength=sz)
        rv += self.nlog_sum

        if self.num_required != 0:
            required_loc = np.isin(val_map_inv, list(self.required))
            req_cnt = np.bincount(idx, weights=required_loc[xs], minlength=sz)
            rv[req_cnt != self.num_required] = -np.inf

        return rv

    def backend_seq_log_density(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded object-valued sets."""
        sz, idx, val_map_inv, xs = x
        rv = engine.zeros(sz) + float(self.nlog_sum)

        if len(xs) > 0:
            dlog_loc = np.asarray([self.log_dmap[u] for u in val_map_inv], dtype=np.float64)
            rv = engine.index_add(rv, engine.asarray(idx), engine.asarray(dlog_loc)[engine.asarray(xs)])

        if self.num_required != 0:
            req_cnt = engine.zeros(sz)
            if len(xs) > 0:
                required_loc = np.isin(val_map_inv, list(self.required))
                req_cnt = engine.index_add(req_cnt, engine.asarray(idx),
                                           engine.asarray(np.asarray(required_loc[xs], dtype=np.float64)))
            rv = engine.where(req_cnt != float(self.num_required), engine.asarray(np.full(sz, -np.inf)), rv)

        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence['BernoulliSetDistribution'], engine: Any) -> Dict[str, Any]:
        """Return stacked Bernoulli-set parameters for shared object support."""
        labels = tuple(dists[0].pmap.keys())
        min_prob = float(dists[0].min_prob)
        if any(tuple(dist.pmap.keys()) != labels or float(dist.min_prob) != min_prob for dist in dists):
            raise ValueError('Stacked BernoulliSetDistribution components require shared support/min_prob.')
        log_d = np.asarray([[dist.log_dmap[label] for dist in dists] for label in labels], dtype=np.float64)
        required = np.asarray([[label in dist.required for dist in dists] for label in labels], dtype=np.float64)
        num_required = np.asarray([dist.num_required for dist in dists], dtype=np.float64)
        return {
            '__pysp_component_axis__': {'log_d': 1, 'nlog_sum': 0, 'required': 1, 'num_required': 0},
            'labels': labels,
            'log_d': engine.asarray(log_d),
            'nlog_sum': engine.asarray(np.asarray([dist.nlog_sum for dist in dists], dtype=np.float64)),
            'required': engine.asarray(required),
            'num_required': engine.asarray(num_required),
            'num_components': len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray],
                                    params: Dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Bernoulli-set log densities."""
        sz, idx, val_map_inv, xs = x
        label_to_idx = {label: i for i, label in enumerate(params['labels'])}
        mapped = np.asarray([label_to_idx.get(label, -1) for label in val_map_inv], dtype=np.int64)
        good = mapped >= 0
        safe = np.clip(mapped, 0, max(0, len(params['labels']) - 1))
        rv = engine.zeros((sz, int(params['num_components']))) + params['nlog_sum'][None, :]

        if len(xs) > 0:
            log_dloc = params['log_d'][engine.asarray(safe), :]
            log_dloc = engine.where(engine.asarray(good)[:, None], log_dloc, engine.asarray(-np.inf))
            rv = engine.index_add(rv, engine.asarray(idx), log_dloc[engine.asarray(xs), :])

        if np.any(np.asarray(engine.to_numpy(params['num_required'])) != 0):
            req_cnt = engine.zeros((sz, int(params['num_components'])))
            if len(xs) > 0:
                required_loc = params['required'][engine.asarray(safe), :]
                required_loc = engine.where(engine.asarray(good)[:, None], required_loc, engine.asarray(0.0))
                req_cnt = engine.index_add(req_cnt, engine.asarray(idx), required_loc[engine.asarray(xs), :])
            rv = engine.where(req_cnt != params['num_required'][None, :], engine.asarray(-np.inf), rv)

        return rv

    @classmethod
    def backend_stacked_sufficient_statistics(cls, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray],
                                              weights: Any, params: Dict[str, Any], engine: Any) \
            -> Tuple[Tuple[Dict[Any, float], float], ...]:
        """Return per-component legacy ``(count_map, total_weight)`` statistics."""
        sz, idx, val_map_inv, xs = x
        xx = engine.asarray(xs)
        ww = engine.asarray(weights)
        count_rows = []
        if len(xs) > 0:
            row_weights = ww[engine.asarray(idx)]
            zero_rows = row_weights * engine.asarray(0.0)
            for value_index in range(len(val_map_inv)):
                mask = xx == engine.asarray(value_index)
                count_rows.append(engine.sum(engine.where(mask[:, None], row_weights, zero_rows), axis=0))
            counts = np.asarray(engine.to_numpy(engine.stack(count_rows, axis=0)), dtype=np.float64)
        else:
            counts = np.zeros((0, int(params['num_components'])), dtype=np.float64)
        totals = np.asarray(engine.to_numpy(engine.sum(ww, axis=0)), dtype=np.float64)
        return tuple(({
            val_map_inv[j]: float(counts[j, component])
            for j in range(len(val_map_inv))
        }, float(totals[component])) for component in range(int(params['num_components'])))

    def sampler(self, seed: Optional[int] = None) -> 'BernoulliSetSampler':
        """Create a BernoulliSetSampler object from parameters of BernoulliSetDistribution instance.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            BernoulliSetSampler object.

        """
        return BernoulliSetSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'BernoulliSetEstimator':
        """Create a BernoulliSetEstimator, passing pmap as suff_stat if pseudo_count is given.

        Args:
            pseudo_count (Optional[float]): Used to re-weight the distribution's pmap in estimation.

        Returns:
            BernoulliSetEstimator object.

        """
        if pseudo_count is None:
            return BernoulliSetEstimator(min_prob=self.min_prob, name=self.name)
        else:
            return BernoulliSetEstimator(min_prob=self.min_prob, pseudo_count=pseudo_count, suff_stat=self.pmap,
                                         name=self.name)

    def dist_to_encoder(self) -> 'BernoulliSetDataEncoder':
        """Returns a BernoulliSetDataEncoder object for encoding sequences of data."""
        return BernoulliSetDataEncoder()

    def enumerator(self) -> 'BernoulliSetEnumerator':
        """Returns BernoulliSetEnumerator iterating subsets of the support in descending probability order."""
        return BernoulliSetEnumerator(self)


class BernoulliSetEnumerator(DistributionEnumerator):
    """Enumerates subsets of the pmap support in descending probability order."""

    def __init__(self, dist: BernoulliSetDistribution) -> None:
        """Enumerates subsets of dist.pmap's keys in descending probability order.

        Membership is independent per element: including element k contributes log_dmap[k]
        to the log-density and excluding it contributes 0 (relative to the nlog_sum offset).
        Each element therefore yields a sorted two-choice stream, and subsets are enumerated
        with a best-first product search. Elements with p_k = 0 are exclude-only; required
        elements (p_k = 1 with min_prob = 0) are include-only. Each subset corresponds to a
        unique inclusion-flag tuple, so deduplication is exact. Raises EnumerationError when
        a membership probability lies outside [0, 1], which breaks the independent-inclusion
        form of the log-density.

        Args:
            dist (BernoulliSetDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        vals = list(dist.pmap.keys())
        log_d = np.asarray([dist.log_dmap[v] for v in vals], dtype=np.float64)
        if np.any(np.isnan(log_d)) or np.any(np.isposinf(log_d)) or not np.isfinite(dist.nlog_sum):
            raise EnumerationError(dist, reason='membership probabilities must lie in [0, 1] for the '
                                                'independent-inclusion log-density to be well-defined')
        streams = []
        for v, d in zip(vals, log_d):
            if v in dist.required:
                choices = [(True, 0.0)]
            elif d == -np.inf:
                choices = [(False, 0.0)]
            elif d > 0.0:
                choices = [(True, float(d)), (False, 0.0)]
            else:
                choices = [(False, 0.0), (True, float(d))]
            streams.append(BufferedStream(iter(choices)))

        def combine(flags: Tuple[bool, ...]) -> List[Any]:
            return [v for v, f in zip(vals, flags) if f]

        self._product = ProductEnumerator(streams, combine=combine, offset=float(dist.nlog_sum))

    def __next__(self) -> Tuple[List[Any], float]:
        return next(self._product)


class BernoulliSetSampler(DistributionSampler):
    """BernoulliSetSampler object for drawing random sets from a BernoulliSetDistribution instance."""

    def __init__(self, dist: BernoulliSetDistribution, seed: Optional[int] = None) -> None:
        """BernoulliSetSampler object for generating samples from BernoulliSetDistribution object instance.

        Args:
            dist (BernoulliSetDistribution): Object instance to sample from.
            seed (Optional[int]): Set seed for random number generator.

        Attributes:
            rng (RandomState): RandomState object with seed set if passed in args.
            dist (BernoulliSetDistribution): Object instance to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[Sequence[Any], List[Sequence[Any]]]:
        """Draw iid set observations from the BernoulliSetDistribution instance.

        Args:
            size (Optional[int]): Number of sets to draw. If None, a single set is returned.

        Returns:
            A list of included elements if size is None, else a list of such lists of length size.

        """
        if size is not None:
            retval = [[] for i in range(size)]
            for k, v in self.dist.pmap.items():
                for i in np.flatnonzero(self.rng.rand(size) <= v):
                    retval[i].append(k)
            return retval

        else:
            retval = []
            for k, v in self.dist.pmap.items():
                if self.rng.rand() <= v:
                    retval.append(k)
            return retval


class BernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):
    """BernoulliSetAccumulator object for aggregating per-element inclusion counts from observed sets."""

    def __init__(self, keys: Optional[str] = None) -> None:
        """BernoulliSetAccumulator object for aggreating sufficient statistics from observed data.

        Args:
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
            pmap (Dict[Any, float]): Dictionary mapping values to set-inclusion probabilities.
            tot_sum (float): Weighted observation count.
            key (Optional[str]): Key for merging sufficient statistics.
        """
        self.pmap = defaultdict(float)
        self.tot_sum = 0.0
        self.key = keys

    def update(self, x: Sequence[Any], weight: float, estimate: Optional[BernoulliSetDistribution]) -> None:
        """Add weight to the inclusion count of each element of the observed set x.

        Args:
            x (Sequence[Any]): Observed set of distinct elements.
            weight (float): Weight for the observation.
            estimate (Optional[BernoulliSetDistribution]): Unused (kept for protocol consistency).

        """
        for u in x:
            self.pmap[u] += weight
        self.tot_sum += weight

    def initialize(self, x: Sequence[Any], weight: float, rng: Optional[RandomState]) -> None:
        """Initialize the accumulator with a weighted observation. Calls update().

        Args:
            x (Sequence[Any]): Observed set of distinct elements.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Unused (kept for protocol consistency).

        """
        self.update(x, weight, None)

    def seq_update(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional[BernoulliSetDistribution]) -> None:
        """Vectorized update of sufficient statistics from sequence encoded observations.

        Args:
            x (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Sequence encoded set observations from
                BernoulliSetDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            estimate (Optional[BernoulliSetDistribution]): Unused (kept for protocol consistency).

        """
        sz, idx, val_map_inv, xs = x
        agg_cnt = np.bincount(xs, weights[idx])

        for i, v in enumerate(agg_cnt):
            self.pmap[val_map_inv[i]] += v

        self.tot_sum += weights.sum()

    def seq_initialize(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray,
                       rng: np.random.RandomState) -> None:
        """Vectorized initialization of sufficient statistics. Calls seq_update().

        Args:
            x (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Sequence encoded set observations from
                BernoulliSetDataEncoder.seq_encode().
            weights (np.ndarray): Weights, one per encoded observation.
            rng (np.random.RandomState): Unused (kept for protocol consistency).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[Dict[Any, float], float]) -> 'BernoulliSetAccumulator':
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[Dict[Any, float], float]): Inclusion counts by element and total weight.

        Returns:
            This BernoulliSetAccumulator.

        """
        for k, v in suff_stat[0].items():
            self.pmap[k] += v
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> Tuple[Dict[Any, float], float]:
        """Returns the sufficient statistics: (inclusion counts by element, total weight)."""
        return dict(self.pmap), self.tot_sum

    def from_value(self, x: Tuple[Dict[Any, float], float]) -> 'BernoulliSetAccumulator':
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[Dict[Any, float], float]): Inclusion counts by element and total weight.

        Returns:
            This BernoulliSetAccumulator.

        """
        self.pmap = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator's statistics into stats_dict under its key, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged accumulators or sufficient statistics.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's statistics with the keyed statistics in stats_dict, if keyed.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged accumulators or sufficient statistics.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'BernoulliSetDataEncoder':
        """Returns a BernoulliSetDataEncoder object for encoding sequences of data."""
        return BernoulliSetDataEncoder()


class BernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    """BernoulliSetAccumulatorFactory object for creating BernoulliSetAccumulator objects."""

    def __init__(self, keys: Optional[str] = None) -> None:
        """BernoulliSetAccumulatorFactory object for creating instances of BernoulliSetAccumulator objects.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics.

        Attributes:
            keys (Optional[str]): Keys for merging sufficient statistics.

        """
        self.keys = keys

    def make(self) -> 'BernoulliSetAccumulator':
        """Returns a new BernoulliSetAccumulator object."""
        return BernoulliSetAccumulator(self.keys)


class BernoulliSetEstimator(ParameterEstimator):
    """BernoulliSetEstimator object for estimating a BernoulliSetDistribution from aggregated sufficient statistics."""

    def __init__(self, min_prob: float = 1.0e-128, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Dict[Any, float]] = None,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """BernoulliSetEstimator object for estimating Bernoulli set distribution from aggregated sufficient statistics.

        Args:
            min_prob (float): Minimum probability for elements estimated with prob = 0.
            pseudo_count (Optional[float]): Used to re-weight suff_stats in estimation.
            suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value to probability mapping.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set key for merging sufficient statistics.

        Attributes:
            min_prob (float): Minimum probability for elements estimated with prob = 0.
            pseudo_count (Optional[float]): Used to re-weight suff_stats in estimation.
            suff_stat (Optional[Dict[Any, float]]): Optional dictionary containing value to probability mapping.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set key for merging sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.min_prob = min_prob

    def accumulator_factory(self) -> 'BernoulliSetAccumulatorFactory':
        """Returns a BernoulliSetAccumulatorFactory for creating BernoulliSetAccumulator objects."""
        return BernoulliSetAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[Dict[Any, float], float]) -> 'BernoulliSetDistribution':
        """Estimate a BernoulliSetDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Unused (kept for protocol consistency).
            suff_stat (Tuple[Dict[Any, float], float]): Inclusion counts by element and total weight.

        Returns:
            BernoulliSetDistribution object.

        """
        if self.pseudo_count is not None and self.suff_stat is not None:
            keys = set(suff_stat[0].keys())
            keys.update(self.suff_stat.keys())

            pmap = {k: (self.suff_stat.get(k, 0.0) * self.pseudo_count + suff_stat[0].get(k, 0.0)) / (
                        self.pseudo_count + suff_stat[1]) for k in keys}

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            cnt = float(p + suff_stat[1])
            pmap = {k: (v + (p / 2.0)) / cnt for k, v in suff_stat[0].items()}

        else:

            if suff_stat[1] != 0:
                pmap = {k: v / suff_stat[1] for k, v in suff_stat[0].items()}
            else:
                pmap = {k: 0.5 for k in suff_stat[0].keys()}

        return BernoulliSetDistribution(pmap, min_prob=self.min_prob, name=self.name)

class BernoulliSetDataEncoder(DataSequenceEncoder):
    """BernoulliSetDataEncoder for encoding sequences of iid observations."""

    def __str__(self) -> str:
        return 'BernoulliSetDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BernoulliSetDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[Any]]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Encode a sequence of iid observations for use with vectorized functions.

        Return value 'rv' is a Tuple of length 4 containing:
            rv[0] (int): Number of observed sets.
            rv[1] (np.ndarray): Numpy array of integer indices for flattened array of values.
            rv[2] (np.ndarray): Numpy array of unique values. (dtype is object).
            rv[3] (np.ndarray): Numpy array of val_map (rv[2]) integer indices for flattened array of values.

        Args:
            x (Sequence[Sequence[Any]]): A sequence of iid Bernoulli set observations.

        Returns:
            See 'rv' above.

        """
        idx = []
        xs = []

        for i in range(len(x)):
            idx.extend([i] * len(x[i]))
            xs.extend(x[i])

        val_map, xs = np.unique(xs, return_inverse=True)

        idx = np.asarray(idx, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)

        return len(x), idx, val_map, xs
