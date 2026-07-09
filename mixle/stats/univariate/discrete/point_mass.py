"""Fixed point-mass distribution and estimator support.

This module implements a sequence-encodable Dirac distribution, including
sampling, encoding, enumeration, backend scoring, and the estimator hooks needed
for mixture and combinator use.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.algorithms import QuantizedEnumerationIndex, freeze
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(a, b))
        except Exception:
            return False
    try:
        return freeze(a) == freeze(b)
    except Exception:
        return a == b


class PointMassDistribution(SequenceEncodableProbabilityDistribution):
    """Fixed Dirac/point-mass distribution assigning all mass to one value."""

    @classmethod
    def compute_capabilities(cls):
        """Declare generic NumPy/Torch scoring support for fixed atoms."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the declaration for a fixed non-estimated atom."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec

        return DistributionDeclaration(
            name="point_mass",
            distribution_type=cls,
            parameters=(ParameterSpec("value", constraint="fixed", differentiable=False),),
            statistics=(),
            support="fixed_atom",
            differentiable=False,
        )

    def __init__(self, value: Any, name: str | None = None, keys: str | None = None) -> None:
        self.value = value
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "PointMassDistribution(%s, name=%s, keys=%s)" % (repr(self.value), repr(self.name), repr(self.keys))

    def density(self, x: Any) -> float:
        """Return the probability density or mass at a single observation."""
        return 1.0 if _same_value(x, self.value) else 0.0

    def log_density(self, x: Any) -> float:
        """Return the log-density or log-mass at a single observation."""
        return 0.0 if _same_value(x, self.value) else -np.inf

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return np.where(x, 0.0, -np.inf)

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density from encoded equality flags."""
        xx = engine.asarray(x)
        return engine.where(xx, engine.zeros(xx.shape), engine.asarray(-np.inf))

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["PointMassDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked point-mass parameters for identical fixed atoms."""
        value = dists[0].value
        if any(not _same_value(dist.value, value) for dist in dists):
            raise ValueError("Stacked PointMassDistribution components require the same fixed atom.")
        return {"num_components": len(dists)}

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of point-mass log densities."""
        xx = engine.asarray(x)
        nobs = int(tuple(getattr(xx, "shape", (len(x),)))[0])
        num_components = int(params["num_components"])
        base = engine.where(xx, engine.zeros(tuple(getattr(xx, "shape", (nobs,)))), engine.asarray(-np.inf))
        return base[:, None] + engine.zeros((nobs, num_components))

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[None, ...]:
        """Return per-component empty statistics for fixed point masses."""
        return tuple(None for _ in range(int(params["num_components"])))

    def support_size(self) -> int:
        """A single atom."""
        return 1

    def sampler(self, seed: int | None = None) -> "PointMassSampler":
        """Return a sampler for drawing observations from this distribution."""
        return PointMassSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "PointMassEstimator":
        """Return an estimator for fitting this distribution from data."""
        return PointMassEstimator(self.value, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "PointMassDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return PointMassDataEncoder(self.value)

    def enumerator(self) -> "PointMassEnumerator":
        """Return an enumerator over the distribution support when available."""
        return PointMassEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Return a bounded quantized support index for this observation."""
        return QuantizedEnumerationIndex.from_items(
            [(self.value, 0.0)], max_bits=max_bits, bin_width_bits=bin_width_bits, sorted_items=True, truncated=False
        )


class PointMassEnumerator(DistributionEnumerator):
    """Enumerate the single atom of a PointMassDistribution."""

    def __init__(self, dist: PointMassDistribution) -> None:
        super().__init__(dist)
        self._done = False

    def __next__(self) -> tuple[Any, float]:
        if self._done:
            raise StopIteration
        self._done = True
        return self.dist.value, 0.0


class PointMassSampler(DistributionSampler):
    """Sampler returning the fixed atom."""

    def __init__(self, dist: PointMassDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> Any | Sequence[Any]:
        """Return the fixed atom once or repeated ``size`` times."""
        if size is None:
            return self.dist.value
        return [self.dist.value for _ in range(int(size))]


class PointMassAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for a fixed point mass; no parameters are learned."""

    def __init__(self, value: Any, keys: str | None = None) -> None:
        self.atom = value
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: PointMassDistribution | None) -> None:
        """Ignore one observation because the atom is fixed."""
        pass

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: PointMassDistribution | None) -> None:
        """Ignore encoded observations because the atom is fixed."""
        pass

    def seq_update_engine(
        self, x: np.ndarray, weights: Any, estimate: PointMassDistribution | None, engine: Any
    ) -> None:
        """Ignore engine-resident observations because the atom is fixed."""
        # PointMass carries no free parameters: accumulation is a no-op on every engine.
        pass

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Ignore initialization observations because the atom is fixed."""
        pass

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Ignore encoded initialization observations because the atom is fixed."""
        pass

    def combine(self, suff_stat: Any) -> "PointMassAccumulator":
        """Return this accumulator because fixed atoms have no sufficient statistics."""
        return self

    def value(self) -> None:
        """Return ``None`` because fixed atoms have no sufficient statistics."""
        return None

    def from_value(self, x: Any) -> "PointMassAccumulator":
        """Return this accumulator because there is no state to restore."""
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Register the key with empty statistics when configured."""
        if self.keys is not None and self.keys not in stats_dict:
            stats_dict[self.keys] = None

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Ignore keyed replacements because there is no accumulator state."""
        pass

    def acc_to_encoder(self) -> "PointMassDataEncoder":
        """Return the equality-mask encoder for this atom."""
        return PointMassDataEncoder(self.atom)


class PointMassAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for PointMassAccumulator."""

    def __init__(self, value: Any, keys: str | None = None) -> None:
        self.value = value
        self.keys = keys

    def make(self) -> PointMassAccumulator:
        """Create an empty point-mass accumulator."""
        return PointMassAccumulator(self.value, keys=self.keys)


class PointMassEstimator(ParameterEstimator):
    """Estimator that always returns the configured point mass."""

    def __init__(
        self,
        value: Any,
        pseudo_count: float | None = None,
        suff_stat: Any | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.value = value
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> PointMassAccumulatorFactory:
        """Return a factory for point-mass no-op accumulators."""
        return PointMassAccumulatorFactory(self.value, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: Any | None = None) -> PointMassDistribution:
        """Return the configured point mass unchanged."""
        return PointMassDistribution(self.value, name=self.name, keys=self.keys)


class PointMassDataEncoder(DataSequenceEncoder):
    """Encode observations as a boolean equality mask against the fixed atom."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __str__(self) -> str:
        return "PointMassDataEncoder(value=%s)" % repr(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PointMassDataEncoder) and _same_value(other.value, self.value)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Encode observations as booleans indicating equality with the fixed atom."""
        return np.asarray([_same_value(v, self.value) for v in x], dtype=bool)
