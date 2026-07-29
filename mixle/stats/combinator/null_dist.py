"""Neutral likelihood factor used for absent structural model slots.

``NullDistribution`` is the multiplicative identity for likelihood composition:
it contributes log-score zero for every input and carries no parameters. It is
therefore **not** a normalized generative probability distribution and cannot be
sampled or enumerated. Use ``PointMassDistribution(None)`` for a proper singleton
law whose only possible value is ``None``.

Notes:
    The density evaluates to 1.0 for any value (Any data type).
    Sequence encodings retain only the observation count.
"""

from typing import Any, Optional

import numpy as np

from mixle.enumeration.algorithms import QuantizedCrossIndex, QuantizedEnumerationIndex
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class NeutralFactorError(TypeError):
    """Raised when a neutral likelihood identity is used as a generative law."""


class NullDistribution(SequenceEncodableProbabilityDistribution):
    """Non-generative likelihood identity assigning log-score zero to every observation."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for the zero-cost null distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the null distribution."""
        from mixle.stats.compute.declarations import DistributionDeclaration

        return DistributionDeclaration(
            name="null",
            distribution_type=cls,
            parameters=(),
            statistics=(),
            support="neutral_factor",
            differentiable=False,
        )

    def __init__(self, name: str | None = None) -> None:
        """Create a non-generative likelihood factor that assigns unit score to every value.

        Args:
            name (Optional[str]): Optional distribution name.

        Attributes:
            name (Optional[str]): Optional distribution name.

        """
        self.name = name

    def __str__(self) -> str:
        """Return a constructor-style representation of the null distribution."""
        return "NullDistribution(name=%s)" % repr(self.name)

    def density(self, x: Any | None) -> float:
        """Density of NullDistribution. Always 1.0.

        Args:
            x (Optional[Any]): Observation of any type (ignored).

        Returns:
            1.0 for any input.

        """
        return 1.0

    def density_semantics(self):
        """Identify this exact score as a non-generative likelihood factor."""
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.LIKELIHOOD_FACTOR

    def log_density(self, x: Any | None) -> float:
        """Log-density of NullDistribution. Always 0.0.

        Args:
            x (Optional[Any]): Observation of any type (ignored).

        Returns:
            0.0 for any input.

        """
        return 0.0

    def seq_log_density(self, x: Any | None) -> np.ndarray:
        """Vectorized log-density evaluated at sequence encoded input x. Always 0.0.

        Args:
            x (Optional[Any]): Sequence encoded data; NullDataEncoder returns the sequence length.

        Returns:
            A zero vector with one entry per encoded observation.

        """
        if isinstance(x, (int, np.integer)):
            return np.zeros(int(x), dtype=float)
        return np.zeros(0, dtype=float)

    def backend_seq_log_density(self, x: Any | None, engine: Any) -> Any:
        """Engine-neutral vectorized log-density: zero for every encoded row."""
        if isinstance(x, (int, np.integer)):
            return engine.zeros(int(x))
        return engine.zeros(0)

    @classmethod
    def backend_stacked_params(cls, dists: tuple["NullDistribution", ...], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for homogeneous null mixtures."""
        return {"num_components": len(dists)}

    @classmethod
    def backend_stacked_log_density(cls, x: Any | None, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` zero matrix for null-component log densities."""
        n = int(x) if isinstance(x, (int, np.integer)) else 0
        return engine.zeros((n, int(params["num_components"])))

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any | None, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[None, ...]:
        """Return empty legacy statistics for each null component."""
        return tuple(None for _ in range(int(params["num_components"])))

    def sampler(self, seed: int | None = None) -> "NullSampler":
        """Reject generation from a likelihood identity with no normalized sample space."""
        raise NeutralFactorError(
            "NullDistribution is a neutral likelihood factor and cannot be sampled; "
            "use PointMassDistribution(None) for a proper singleton law."
        )

    def estimator(self, pseudo_count: float | None = None) -> "NullEstimator":
        """Create an estimator for the null distribution.

        Args:
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).

        Returns:
            NullEstimator configured with this distribution's name.

        """
        if pseudo_count is None:
            return NullEstimator(name=self.name)

        else:
            return NullEstimator(pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> "NullDataEncoder":
        """Return the encoder used for null-distribution observations."""
        return NullDataEncoder()

    def enumerator(self) -> "NullEnumerator":
        """Reject enumeration because a neutral factor has no probability support."""
        from mixle.stats.compute.pdist import EnumerationError

        raise EnumerationError(
            self,
            reason=(
                "NullDistribution is a neutral likelihood factor; use PointMassDistribution(None) for singleton support"
            ),
        )

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Reject indexing because a neutral factor has no probability support."""
        return super().quantized_index(max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Reject cross-indexing because neutral factors have no probability support."""
        return super().quantized_multi_cross_index(
            others,
            max_bits=max_bits,
            bin_width_bits=bin_width_bits,
        )

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Reject pairwise cross-indexing because neutral factors have no probability support."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class NullEnumerator(DistributionEnumerator):
    """Deprecated guard that rejects enumeration of a neutral factor."""

    def __init__(self, dist: "NullDistribution") -> None:
        """Create an enumerator for the null distribution's empty support.

        Args:
            dist (NullDistribution): NullDistribution instance to enumerate.

        """
        super().__init__(dist)
        raise NeutralFactorError("NullDistribution is not enumerable; use PointMassDistribution(None).")

    def __next__(self) -> tuple[None, float]:
        """Stop immediately; construction already rejects this non-enumerable factor."""
        raise StopIteration


class NullSampler(DistributionSampler):
    """Deprecated guard that rejects sampling from a neutral factor."""

    def __init__(self, dist: "NullDistribution", seed: int | None = None) -> None:
        """Create a sampler for the null distribution.

        Args:
            dist (NullDistribution): NullDistribution instance to sample from.
            seed (Optional[int]): Seed for random number generator (unused).

        """
        raise NeutralFactorError("NullDistribution is not samplable; use PointMassDistribution(None).")

    def sample(self, size: int | None = None, *, batched: bool = True) -> None | list[None]:
        """Reject every draw from this non-generative compatibility guard.

        Args:
            size (Optional[int]): Number of samples requested (unused).
            batched (bool): Compatibility flag (unused).

        """
        raise NeutralFactorError("NullDistribution is not samplable; use PointMassDistribution(None).")


class NullAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for NullDistribution. Accumulates no sufficient statistics."""

    def __init__(self, keys: str | None = None) -> None:
        """Create a null accumulator.

        Args:
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            key (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.keys = keys

    def update(self, x: Any | None, weight: float, estimate: Optional["NullDistribution"]) -> None:
        """No-op update. Nothing is accumulated for the NullDistribution.

        Args:
            x (Optional[Any]): Observation of any type (ignored).
            weight (float): Weight for observation (ignored).
            estimate (Optional[NullDistribution]): Previous estimate (ignored).

        """
        pass

    def seq_update(self, x: Any | None, weights: np.ndarray, estimate: Optional["NullDistribution"]) -> None:
        """No-op vectorized update. Nothing is accumulated for the NullDistribution.

        Args:
            x (Optional[Any]): Sequence encoded data (ignored).
            weights (np.ndarray): Weights for observations (ignored).
            estimate (Optional[NullDistribution]): Previous estimate (ignored).

        """
        pass

    def seq_update_engine(self, x, weights, estimate, engine) -> None:
        """Engine-neutral no-op accumulation for null statistics."""
        # NullDistribution has no parameters: accumulation is a no-op on every engine.
        pass

    def initialize(self, x: Any | None, weight: float, rng: Optional["np.random.RandomState"]) -> None:
        """No-op initialization for a single observation.

        Args:
            x (Optional[Any]): Observation of any type (ignored).
            weight (float): Weight for observation (ignored).
            rng (Optional[np.random.RandomState]): Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: Any | None, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """No-op vectorized initialization.

        Args:
            x (Optional[Any]): Sequence encoded data (ignored).
            weights (np.ndarray): Weights for observations (ignored).
            rng (np.random.RandomState): Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Any | None) -> "NullAccumulator":
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

    def from_value(self, x: Any | None) -> "NullAccumulator":
        """Set accumulator from sufficient statistics (no-op).

        Args:
            x (Optional[Any]): Sufficient statistics (ignored).

        Returns:
            Self, unchanged.

        """
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Register the key in stats_dict (the NullAccumulator stores no sufficient statistics).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                pass
            else:
                stats_dict[self.keys] = None

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """No-op kept for interface consistency (the NullAccumulator stores no sufficient statistics).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics (ignored).

        Returns:
            None.

        """
        pass

    def acc_to_encoder(self) -> "NullDataEncoder":
        """Return the encoder associated with this accumulator."""
        return NullDataEncoder()


class NullAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for null accumulators."""

    def __init__(self, keys: str | None = None) -> None:
        """Create a factory for null accumulators.

        Args:
            keys (Optional[str]): Optional key passed to created accumulators.

        Attributes:
            keys (Optional[str]): Optional key passed to created accumulators.

        """
        self.keys = keys

    def make(self) -> "NullAccumulator":
        """Return a fresh null accumulator."""
        return NullAccumulator(keys=self.keys)


class NullEstimator(ParameterEstimator):
    """Estimator that always produces a NullDistribution regardless of the data."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: Any | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an estimator for the null distribution.

        Args:
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).
            suff_stat (Optional[Any]): Kept for interface consistency (has no effect on estimation).
            name (Optional[str]): Optional estimator name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            pseudo_count (Optional[float]): Kept for interface consistency.
            suff_stat (Optional[Any]): Kept for interface consistency.
            name (Optional[str]): Optional estimator name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "NullAccumulatorFactory":
        """Return a factory for null accumulators."""
        return NullAccumulatorFactory(self.keys)

    def estimate(self, nobs: float | None, suff_stat: Any | None = None) -> "NullDistribution":
        """Returns a NullDistribution; arguments are ignored.

        Args:
            nobs (Optional[float]): Number of observations (ignored).
            suff_stat (Optional[Any]): Sufficient statistics (ignored).

        Returns:
            NullDistribution carrying this estimator's name.

        """
        return NullDistribution(name=self.name)


class NullDataEncoder(DataSequenceEncoder):
    """Data encoder for the NullDistribution. Encodes any sequence as its length."""

    def __str__(self) -> str:
        """Return the null encoder's display name."""
        return "NullDataEncoder"

    def __eq__(self, other) -> bool:
        """Return true when ``other`` is a null data encoder.

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

    def row_count(self, x: Any) -> int:
        """Return the exact number of rows stored by the null encoding."""
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            raise ValueError("null encoded data must be a non-negative integer row count")
        if x < 0:
            raise ValueError("null encoded row count must be non-negative")
        return int(x)
