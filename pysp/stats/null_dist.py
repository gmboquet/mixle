"""Create, estimate, and sample from a null distribution.

Defines the NullDistribution, NullSampler, NullAccumulatorFactory, NullAccumulator,
NullEstimator, and the NullDataEncoder classes for use with pysparkplug.

The NullDistribution object and its related classes are space filling objects meant for consistency in type hints.

Notes:
    The density evaluates to 1.0 for any value (Any data type).
    The sampler generates None for any size input.
    Sequence encodings return None for any input.

"""
from typing import Any, Optional, Dict, Tuple

import numpy as np
from numpy.random import RandomState

from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, \
    DistributionEnumerator
from pysp.utils.enumeration import QuantizedCrossIndex, QuantizedEnumerationIndex


class NullDistribution(SequenceEncodableProbabilityDistribution):
    """Place-holder distribution assigning density 1.0 (log-density 0.0) to any observation (Any data type)."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.capabilities import DistributionCapabilities
        return DistributionCapabilities(engine_ready=('numpy', 'torch'), kernel_status='generic')

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.declarations import DistributionDeclaration
        return DistributionDeclaration(
            name='null',
            distribution_type=cls,
            parameters=(),
            statistics=(),
            support='any',
            differentiable=False,
        )

    def __init__(self, name: Optional[str] = None) -> None:
        """NullDistribution object.

        Args:
            name (Optional[str]): Optional name for object instance.

        Attributes:
            name (Optional[str]): Optional name for object instance.

        """
        self.name = name

    def __str__(self) -> str:
        """Returns string representation of NullDistribution object."""
        return 'NullDistribution(name=%s)' % repr(self.name)

    def density(self, x: Optional[Any]) -> float:
        """Density of NullDistribution. Always 1.0.

        Args:
            x (Optional[Any]): Observation of any type (ignored).

        Returns:
            1.0 for any input.

        """
        return 1.0

    def log_density(self, x: Optional[Any]) -> float:
        """Log-density of NullDistribution. Always 0.0.

        Args:
            x (Optional[Any]): Observation of any type (ignored).

        Returns:
            0.0 for any input.

        """
        return 0.0

    def seq_log_density(self, x: Optional[Any]) -> np.ndarray:
        """Vectorized log-density evaluated at sequence encoded input x. Always 0.0.

        Args:
            x (Optional[Any]): Sequence encoded data; NullDataEncoder returns the sequence length.

        Returns:
            A zero vector with one entry per encoded observation.

        """
        if isinstance(x, (int, np.integer)):
            return np.zeros(int(x), dtype=float)
        return np.zeros(0, dtype=float)

    def backend_seq_log_density(self, x: Optional[Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density: zero for every encoded row."""
        if isinstance(x, (int, np.integer)):
            return engine.zeros(int(x))
        return engine.zeros(0)

    @classmethod
    def backend_stacked_params(cls, dists: Tuple['NullDistribution', ...], engine: Any) -> Dict[str, Any]:
        """Return stacked parameters for homogeneous null mixtures."""
        return {'num_components': len(dists)}

    @classmethod
    def backend_stacked_log_density(cls, x: Optional[Any], params: Dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` zero matrix for null-component log densities."""
        n = int(x) if isinstance(x, (int, np.integer)) else 0
        return engine.zeros((n, int(params['num_components'])))

    @classmethod
    def backend_stacked_sufficient_statistics(cls, x: Optional[Any], weights: Any,
                                             params: Dict[str, Any], engine: Any) -> Tuple[None, ...]:
        """Return empty legacy statistics for each null component."""
        return tuple(None for _ in range(int(params['num_components'])))

    def sampler(self, seed: Optional[int] = None) -> 'NullSampler':
        """Create a NullSampler object.

        Args:
            seed (Optional[int]): Seed for random number generator (unused).

        Returns:
            NullSampler object.

        """
        return NullSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'NullEstimator':
        """Create a NullEstimator object.

        Args:
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).

        Returns:
            NullEstimator object.

        """
        if pseudo_count is None:
            return NullEstimator(name=self.name)

        else:
            return NullEstimator(pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'NullDataEncoder':
        """Returns a NullDataEncoder object for encoding sequences of data."""
        return NullDataEncoder()

    def enumerator(self) -> 'NullEnumerator':
        """Returns a NullEnumerator object enumerating the support of the NullDistribution."""
        return NullEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build the single-item bounded bit-quantized index for NullDistribution."""
        return QuantizedEnumerationIndex.from_items(
            [(None, 0.0)], max_bits=max_bits, bin_width_bits=bin_width_bits,
            sorted_items=True, truncated=False)

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view for null distributions."""
        dists = [self] + list(others)
        if any(not isinstance(dist, NullDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)
        return QuantizedCrossIndex.from_items(
            [(None, tuple([0.0] * len(dists)))], max_bits=max_bits,
            bin_width_bits=bin_width_bits, truncated=False)

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view for two null distributions."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class NullEnumerator(DistributionEnumerator):
    """Yields the single value None with probability one, matching NullSampler.sample()."""

    def __init__(self, dist: 'NullDistribution') -> None:
        """NullEnumerator object.

        Args:
            dist (NullDistribution): NullDistribution instance to enumerate.

        """
        super().__init__(dist)
        self._done = False

    def __next__(self) -> Tuple[None, float]:
        """Returns the single (None, 0.0) pair, then raises StopIteration."""
        if self._done:
            raise StopIteration
        self._done = True
        return (None, 0.0)


class NullSampler(DistributionSampler):
    """Sampler for the NullDistribution. Always returns None."""

    def __init__(self, dist: 'NullDistribution', seed: Optional[int] = None) -> None:
        """NullSampler object.

        Args:
            dist (NullDistribution): NullDistribution instance to sample from.
            seed (Optional[int]): Seed for random number generator (unused).

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> None:
        """Returns None for any requested size.

        Args:
            size (Optional[int]): Number of samples requested (ignored).

        Returns:
            None.

        """
        return None


class NullAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for NullDistribution. Accumulates no sufficient statistics."""

    def __init__(self, keys: Optional[str] = None) -> None:
        """NullAccumulator object.

        Args:
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            key (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.key = keys

    def update(self, x: Optional[Any], weight: float, estimate: Optional['NullDistribution']) -> None:
        """No-op update. Nothing is accumulated for the NullDistribution.

        Args:
            x (Optional[Any]): Observation of any type (ignored).
            weight (float): Weight for observation (ignored).
            estimate (Optional[NullDistribution]): Previous estimate (ignored).

        """
        pass

    def seq_update(self,
                   x: Optional[Any],
                   weights: np.ndarray,
                   estimate: Optional['NullDistribution']) -> None:
        """No-op vectorized update. Nothing is accumulated for the NullDistribution.

        Args:
            x (Optional[Any]): Sequence encoded data (ignored).
            weights (np.ndarray): Weights for observations (ignored).
            estimate (Optional[NullDistribution]): Previous estimate (ignored).

        """
        pass

    def seq_update_engine(self, x, weights, estimate, engine) -> None:
        # NullDistribution has no parameters: accumulation is a no-op on every engine.
        pass

    def initialize(self, x: Optional[Any], weight: float, rng: Optional['np.random.RandomState']) -> None:
        """No-op initialization for a single observation.

        Args:
            x (Optional[Any]): Observation of any type (ignored).
            weight (float): Weight for observation (ignored).
            rng (Optional[np.random.RandomState]): Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_initialize(self,
                       x: Optional[Any],
                       weights: np.ndarray,
                       rng: np.random.RandomState) -> None:
        """No-op vectorized initialization.

        Args:
            x (Optional[Any]): Sequence encoded data (ignored).
            weights (np.ndarray): Weights for observations (ignored).
            rng (np.random.RandomState): Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Optional[Any]) -> 'NullAccumulator':
        """Combine sufficient statistics (no-op).

        Args:
            suff_stat (Optional[Any]): Sufficient statistics (ignored).

        Returns:
            Self, unchanged.

        """
        return self

    def value(self) -> None:
        """Returns None (the NullAccumulator has no sufficient statistics)."""
        return None

    def from_value(self, x: Optional[Any]) -> 'NullAccumulator':
        """Set accumulator from sufficient statistics (no-op).

        Args:
            x (Optional[Any]): Sufficient statistics (ignored).

        Returns:
            Self, unchanged.

        """
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Register the key in stats_dict (the NullAccumulator stores no sufficient statistics).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                pass
            else:
                stats_dict[self.key] = None

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """No-op kept for interface consistency (the NullAccumulator stores no sufficient statistics).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics (ignored).

        Returns:
            None.

        """
        pass

    def acc_to_encoder(self) -> 'NullDataEncoder':
        """Returns a NullDataEncoder object for encoding sequences of data."""
        return NullDataEncoder()


class NullAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating NullAccumulator objects."""

    def __init__(self, keys: Optional[str] = None) -> None:
        """NullAccumulatorFactory object.

        Args:
            keys (Optional[str]): Optional key passed to created accumulators.

        Attributes:
            keys (Optional[str]): Optional key passed to created accumulators.

        """
        self.keys = keys

    def make(self) -> 'NullAccumulator':
        """Returns a new NullAccumulator object."""
        return NullAccumulator(keys=self.keys)


class NullEstimator(ParameterEstimator):
    """Estimator that always produces a NullDistribution regardless of the data."""

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Any] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """NullEstimator object.

        Args:
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).
            suff_stat (Optional[Any]): Kept for interface consistency (has no effect on estimation).
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            pseudo_count (Optional[float]): Kept for interface consistency.
            suff_stat (Optional[Any]): Kept for interface consistency.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'NullAccumulatorFactory':
        """Returns a NullAccumulatorFactory for creating NullAccumulator objects."""
        return NullAccumulatorFactory(self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Optional[Any] = None) -> 'NullDistribution':
        """Returns a NullDistribution; arguments are ignored.

        Args:
            nobs (Optional[float]): Number of observations (ignored).
            suff_stat (Optional[Any]): Sufficient statistics (ignored).

        Returns:
            NullDistribution object.

        """
        return NullDistribution(name=self.name)


class NullDataEncoder(DataSequenceEncoder):
    """Data encoder for the NullDistribution. Encodes any sequence as its length."""

    def __str__(self) -> str:
        """Returns string representation of NullDataEncoder object."""
        return 'NullDataEncoder'

    def __eq__(self, other) -> bool:
        """Checks if other object is an instance of a NullDataEncoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a NullDataEncoder instance, else False.

        """
        return isinstance(other, NullDataEncoder)

    def seq_encode(self, x: Any) -> int:
        """Encode a sequence of observations as its length.

        Args:
            x (Any): Sequence of observations of any type (ignored).

        Returns:
            Number of observations in the sequence.

        """
        return len(x)
