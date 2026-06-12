"""Defines abstract classes for SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator,
ProbabilityDistribution, StatisticAccumulator, StatisticAccumulatorFactory, DataSequenceEncoder, ParameterEstimator,
ConditionalSampler, and DistributionSampler for classes of the pysp.stats.

"""
import itertools
import math
import numpy as np
from abc import abstractmethod
from pysp.arithmetic import *
from typing import TypeVar, Optional, Any, Generic, Dict, List, Tuple

SS = TypeVar('SS')


class EnumerationError(NotImplementedError):
    """Raised when a distribution (or a child of a combinator) cannot enumerate its support.

    The path argument identifies the offending child within a combinator, e.g.
    'CompositeDistribution.dists[1]'.
    """

    def __init__(self, dist: Any, path: str = '', reason: str = '') -> None:
        self.leaf = dist
        self.path = path
        self.reason = reason
        msg = '%s does not support enumeration' % type(dist).__name__
        if path:
            msg = '%s -> %s' % (path, msg)
        if reason:
            msg += ': %s' % reason
        super().__init__(msg)


def child_enumerator(child: 'ProbabilityDistribution', path: str) -> 'DistributionEnumerator':
    """Construct child.enumerator(), annotating EnumerationError with the child's path.

    Combinator enumerators use this so a failure deep in a nested model reports the
    full path to the offending leaf, e.g.
    'CompositeDistribution.dists[1] -> MixtureDistribution.components[0] -> GaussianDistribution ...'.
    """
    try:
        return child.enumerator()
    except EnumerationError as e:
        new_path = path if not e.path else '%s -> %s' % (path, e.path)
        raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None


class ProbabilityDistribution:
    """Base class for all probability distributions in pysp.stats.

    A distribution evaluates the (log-)density of a single observation of its data
    type, creates a DistributionSampler for drawing observations, and creates a
    ParameterEstimator for re-estimating itself from data. Discrete distributions
    may additionally provide a DistributionEnumerator over their support.
    """

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """Return a safe JSON-compatible representation of this distribution."""
        from pysp.utils.serialization import to_serializable
        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'ProbabilityDistribution':
        """Reconstruct a distribution from ``to_dict`` output."""
        from pysp.utils.serialization import from_serializable
        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError('decoded object is %s, not %s' % (type(rv).__name__, cls.__name__))
        return rv

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this distribution as safe strict JSON."""
        from pysp.utils.serialization import to_json
        return to_json(self, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> 'ProbabilityDistribution':
        """Deserialize a distribution from ``to_json`` output."""
        from pysp.utils.serialization import from_json
        rv = from_json(text)
        if not isinstance(rv, cls):
            raise TypeError('decoded object is %s, not %s' % (type(rv).__name__, cls.__name__))
        return rv

    @abstractmethod
    def density(self, x: Any) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    @abstractmethod
    def log_density(self, x: Any) -> float:
        """Return the log-density or log-mass at a single observation."""
        ...

    @abstractmethod
    def sampler(self, seed: Optional[int] = None) -> 'DistributionSampler':
        """Return a sampler for drawing observations from this distribution."""
        ...

    @abstractmethod
    def estimator(self, pseudo_count: Optional[float] = None) -> 'ParameterEstimator':
        """Return an estimator for fitting this distribution from data."""
        ...

    def to_fisher(self):
        """Return a Fisher-geometry view of this distribution.

        The default view is accumulator-backed, so distributions inherit a
        generic sufficient-statistic/Fisher-vector interface.  Individual
        distributions may override this with faster or more canonical views.
        """
        from pysp.utils.fisher import to_fisher
        return to_fisher(self)

    def enumerator(self) -> 'DistributionEnumerator':
        """Return a DistributionEnumerator over this distribution's support.

        Distributions with an enumerable (discrete) support override this; the
        default raises EnumerationError.
        """
        raise EnumerationError(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0):
        """Build a bounded bit-quantized index over this distribution's support.

        This is a convenience wrapper around ``self.enumerator().quantized_index``.
        Non-enumerable distributions raise EnumerationError through enumerator().

        Args:
            max_bits (float): Maximum information content in bits to index.
            bin_width_bits (float): Width of each quantized probability bin in bits.

        Returns:
            pysp.utils.enumeration.QuantizedEnumerationIndex.

        """
        return self.enumerator().quantized_index(max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_multi_cross_index(self, others: List['ProbabilityDistribution'], max_bits,
                                    bin_width_bits: float = 1.0):
        """Build an aligned bounded cross-bin view against other distributions.

        The generic implementation is a bounded candidate join: it unions the bounded
        quantized indexes of all participating distributions, then evaluates every
        candidate under every distribution. Structured distributions can override this
        to build the same aligned rows from support algebra instead.
        """
        from pysp.utils.enumeration import QuantizedCrossIndex, freeze

        dists = [self] + list(others)
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = tuple([float(max_bits)] * len(dists))
        if len(max_bits_tuple) != len(dists):
            raise ValueError('max_bits length must match the number of distributions.')
        if bin_width_bits <= 0:
            raise ValueError('bin_width_bits must be positive.')

        seen = set()
        values = []
        truncated = False
        for dist, bit_bound in zip(dists, max_bits_tuple):
            if bit_bound < 0.0:
                truncated = True
                continue
            index = dist.quantized_index(max_bits=bit_bound, bin_width_bits=bin_width_bits)
            truncated = truncated or index.truncated
            for value, _ in index.iter_from():
                key = freeze(value)
                if key not in seen:
                    seen.add(key)
                    values.append(value)

        items = []
        for value in values:
            items.append((value, tuple(float(dist.log_density(value)) for dist in dists)))
        return QuantizedCrossIndex.from_items(
            items, max_bits=max_bits_tuple, bin_width_bits=bin_width_bits,
            truncated=truncated)

    def quantized_cross_index(self, other: 'ProbabilityDistribution', max_bits,
                              bin_width_bits: float = 1.0):
        """Build an aligned bounded cross-bin view against another distribution."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class SequenceEncodableProbabilityDistribution(ProbabilityDistribution):
    """ProbabilityDistribution with vectorized log-density evaluation on encoded data.

    dist_to_encoder() returns a DataSequenceEncoder whose seq_encode() output is
    consumed by seq_log_density() (and by the matching accumulator's seq_update /
    seq_initialize), enabling fast vectorized estimation over iid sequences.
    """

    def seq_ld_lambda(self):
        """Return vectorized log-density callables for encoded data."""
        pass

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return np.asarray([self.log_density(u) for u in x])

    def seq_log_density_lambda(self):
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    @abstractmethod
    def dist_to_encoder(self) -> 'DataSequenceEncoder':
        """Return the data encoder used by this distribution for vectorized methods."""
        ...


class DistributionSampler(object):
    """Draws iid observations from a distribution using a seeded RandomState.

    sample(size=None) returns a single observation of the distribution's data type;
    sample(size=n) returns a length-n collection of observations.
    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, seed: Optional[int] = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def new_seed(self) -> int:
        """Return a fresh random seed drawn from this sampler's RandomState."""
        return self.rng.randint(0, maxrandint)

    @abstractmethod
    def sample(self, size: Optional[int] = None) -> Any: ...


class DistributionEnumerator(object):
    """Lazy iterator over the support of dist in non-increasing probability order.

    Yields (value, log_prob) pairs, possibly infinitely many. Contract:
      - Each support value is yielded exactly once (deduplication is the
        enumerator's responsibility).
      - log_prob equals dist.log_density(value) up to float round-off (~1e-10),
        and the sequence of log_probs is non-increasing up to the same tolerance.
      - Values with zero probability are skipped, never yielded.
      - Ties are broken deterministically by insertion order; no further guarantee.
    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        self.dist = dist

    def __iter__(self) -> 'DistributionEnumerator':
        return self

    @abstractmethod
    def __next__(self) -> Tuple[Any, float]: ...

    def top_k(self, k: int) -> List[Tuple[Any, float]]:
        """Return the k most probable (value, log_prob) pairs (fewer if the support is smaller)."""
        return list(itertools.islice(self, k))

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0):
        """Precompute a bounded bit-quantized index over this enumeration.

        The index groups values by floor((-log2 p(x)) / bin_width_bits), includes only
        values with -log2 p(x) <= max_bits, and returns exact log probabilities for
        indexed values. Building the index consumes this enumerator.

        Args:
            max_bits (float): Maximum information content in bits to index.
            bin_width_bits (float): Width of each quantized probability bin in bits.

        Returns:
            pysp.utils.enumeration.QuantizedEnumerationIndex.

        """
        from pysp.utils.enumeration import QuantizedEnumerationIndex
        return QuantizedEnumerationIndex.from_enumerator(self, max_bits=max_bits,
                                                         bin_width_bits=bin_width_bits)


class ConditionalSampler(object):
    """Sampler mixin for conditional draws: sample_given(x) draws from P(. | x)."""

    @abstractmethod
    def sample_given(self, x): ...


class StatisticAccumulator(Generic[SS]):
    """Accumulates weighted sufficient statistics of type SS from observations.

    update(x, weight, estimate) adds one observation (estimate is the previous model,
    used for E-step posteriors; it may be None during initialization). Accumulators
    merge across partitions via combine(suff_stat) / value() / from_value(), and
    key_merge / key_replace pool statistics shared across model components through
    a stats_dict keyed by the accumulator's key.
    """

    def update(self, x: Any, weight: float, estimate) -> None:
        ...

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        self.update(x, weight, estimate=None)

    @abstractmethod
    def combine(self, suff_stat: SS) -> 'StatisticAccumulator':
        ...

    @abstractmethod
    def value(self) -> SS:
        ...

    @abstractmethod
    def from_value(self, x: SS) -> 'SequenceEncodableStatisticAccumulator':
        ...

    @abstractmethod
    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        ...


class SequenceEncodableStatisticAccumulator(StatisticAccumulator[SS]):
    """StatisticAccumulator with vectorized updates on encoded data sequences.

    seq_update / seq_initialize consume the output of the matching
    DataSequenceEncoder's seq_encode() (obtained via acc_to_encoder()) together with
    a per-observation weight vector.
    """

    def get_seq_lambda(self):
        pass

    @abstractmethod
    def seq_update(self, x, weights: np.ndarray, estimate) -> None: ...

    @abstractmethod
    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None: ...

    @abstractmethod
    def acc_to_encoder(self) -> 'DataSequenceEncoder': ...

class StatisticAccumulatorFactory(object):
    """Factory whose make() returns a fresh, zeroed accumulator for one estimator."""

    @abstractmethod
    def make(self) -> 'SequenceEncodableStatisticAccumulator': ...


class ParameterEstimator(Generic[SS]):
    """Estimates a distribution from accumulated sufficient statistics.

    accumulator_factory() supplies accumulators that gather sufficient statistics of
    type SS, and estimate(nobs, suff_stat) maps those statistics (plus optional
    regularization configured on the estimator) to a new distribution.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Return a safe JSON-compatible representation of this estimator."""
        from pysp.utils.serialization import to_serializable
        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'ParameterEstimator':
        """Reconstruct an estimator from ``to_dict`` output."""
        from pysp.utils.serialization import from_serializable
        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError('decoded object is %s, not %s' % (type(rv).__name__, cls.__name__))
        return rv

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this estimator as safe strict JSON."""
        from pysp.utils.serialization import to_json
        return to_json(self, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> 'ParameterEstimator':
        """Deserialize an estimator from ``to_json`` output."""
        from pysp.utils.serialization import from_json
        rv = from_json(text)
        if not isinstance(rv, cls):
            raise TypeError('decoded object is %s, not %s' % (type(rv).__name__, cls.__name__))
        return rv

    @abstractmethod
    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'SequenceEncodableProbabilityDistribution': ...

    @abstractmethod
    def accumulator_factory(self) -> 'StatisticAccumulatorFactory': ...


class DataSequenceEncoder:
    """Encodes an iid data sequence into the vectorized form used by seq_* methods.

    seq_encode(x) transforms a sequence of observations into the encoding consumed
    by seq_log_density / seq_update / seq_initialize. Encoders must define __eq__
    (so equivalent encoders are interchangeable when batching) and a readable
    __str__.
    """

    def __str__(self) -> str:
        return type(self).__name__

    def seq_encode(self, x: Any) -> Any:
        """Encode the iid observation sequence x for vectorized evaluation."""
        return x

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...
