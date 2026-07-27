"""Weighted observation wrapper around a base distribution.

Data type: ``WeightedObservation[D]`` (tuple-compatible): an observation is a pair ``(value, weight)``
where ``value`` has the base data type and ``weight`` is finite, non-negative evidence metadata. The
weight is explicitly **not a random coordinate** and does not enter the likelihood; it scales the
observation's sufficient-statistic contribution. Sampling and enumeration use the canonical neutral
metadata value ``1.0``. Likelihood evaluations validate the metadata and delegate on the value:

    P((x, w)) = P_base(x).

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple, TypeVar

import numpy as np

from mixle.engines.arithmetic import *
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)

D = TypeVar("D")
E = TypeVar("E")
SS = TypeVar("SS")


from mixle.inference.fisher import FixedFisherView, to_fisher


class WeightedObservation(NamedTuple):
    """A child value plus non-random, non-negative estimation-weight metadata."""

    value: Any
    weight: float


class WeightedStatistics(NamedTuple):
    """Child sufficient statistics plus their effective attached-times-external weight."""

    child: Any
    effective_weight: float


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a finite non-negative real number." % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a finite non-negative real number." % name) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("%s must be a finite non-negative real number." % name)
    return result


def _observation(value: Any) -> WeightedObservation:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("WeightedDistribution observations must be (value, weight) pairs.")
    return WeightedObservation(value[0], _finite_nonnegative(value[1], name="attached weight"))


def _effective_weight(external: Any, attached: Any) -> float:
    result = _finite_nonnegative(external, name="external weight") * _finite_nonnegative(
        attached,
        name="attached weight",
    )
    if not np.isfinite(result):
        raise ValueError("attached-times-external weight must be finite.")
    return result


def _validated_statistics(value: Any) -> WeightedStatistics:
    if not isinstance(value, WeightedStatistics):
        raise TypeError("weighted sufficient statistics must be WeightedStatistics.")
    return WeightedStatistics(
        value.child,
        _finite_nonnegative(value.effective_weight, name="effective weight"),
    )


def _validated_weight_vector(value: Any, size: int, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must be a length-%d vector of finite non-negative values." % (name, size))
    return result


class WeightedDistribution(SequenceEncodableProbabilityDistribution):
    """Distribution wrapper that attaches observation weights to a base distribution.

    Args:
        dist (SequenceEncodableProbabilityDistribution): Base distribution for the observed values.
        name (Optional[str]): Optional distribution name.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Base distribution for the observed values.
        name (Optional[str]): Optional distribution name.

    """

    def compute_capabilities(self):
        """Delegate generated-compute support to the wrapped value distribution."""
        from dataclasses import replace

        from mixle.stats.compute.capabilities import capabilities_for, delegated_engine_ready

        child = capabilities_for(self.dist)
        # delegate scoring to the child, but cap to composition-safe engines: this wrapper's kernel is
        # only verified on numpy/torch, so it must not inherit a leaf-only engine (e.g. jax) from the child
        return replace(child, engine_ready=delegated_engine_ready(child.engine_ready))

    def compute_declaration(self):
        """Return the generated-compute declaration for the weighted wrapper."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        child = declaration_for(self.dist)
        children = () if child is None else (child,)
        return DistributionDeclaration(
            name="weighted",
            distribution_type=type(self),
            parameters=(),
            statistics=(
                StatisticSpec("weighted_child", kind="child_stat"),
                StatisticSpec("effective_weight"),
            ),
            support="weighted_observation",
            children=children,
            child_roles=("value",) if child is not None else (),
            differentiable=False,
        )

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        name: str | None = None,
        keys: str | None = None,
    ):
        self.dist = dist
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the weighted distribution."""
        return "WeightedDistribution(dist=%s, name=%s, keys=%s)" % (
            repr(self.dist),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: WeightedObservation | tuple[D, float]) -> float:
        """Density of the base distribution at a weighted observation's value.

        Args:
            x (D): Observation value (weight excluded).

        Returns:
            Density of the base distribution at x.

        """
        return self.dist.density(_observation(x).value)

    def log_density(self, x: WeightedObservation | tuple[D, float]) -> float:
        """Log-density of the base distribution at a weighted observation's value.

        The observation weight does not enter the likelihood, so this is simply the
        base distribution's log-density evaluated on the value.

        Args:
            x (D): Observation value (weight excluded).

        Returns:
            Log-density of the base distribution at x.

        """
        return self.dist.log_density(_observation(x).value)

    def seq_log_density(self, x: tuple[E, np.ndarray]) -> np.ndarray:
        """Vectorized log-density of the base distribution on encoded values.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.

        Returns:
            Numpy array of base-distribution log-densities.

        """
        _validated_weight_vector(x[1], len(x[1]), name="attached weights")
        return self.dist.seq_log_density(x[0])

    def backend_seq_log_density(self, x: tuple[E, np.ndarray], engine: Any) -> Any:
        """Engine-neutral vectorized log-density delegated to the value distribution."""
        from mixle.stats.compute.backend import backend_seq_log_density

        _validated_weight_vector(x[1], len(x[1]), name="attached weights")
        return backend_seq_log_density(self.dist, x[0], engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence[WeightedDistribution], engine: Any) -> dict[str, Any]:
        """Return stacked child parameters for homogeneous weighted-wrapper mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        child_dists = [dist.dist for dist in dists]
        try:
            child_route = stacked_component_params(child_dists, engine)
        except ValueError as exc:
            raise ValueError("Weighted child %s is not stackable: %s" % (type(child_dists[0]).__name__, exc))
        return {"child_route": child_route, "num_components": len(dists)}

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[E, np.ndarray], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of child log densities, ignoring attached weights."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        _validated_weight_vector(x[1], len(x[1]), name="attached weights")
        return stacked_component_log_density(x[0], params["child_route"], engine)

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: tuple[E, np.ndarray], weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> Any:
        """Decline resident accumulation until its nested effective-count layout is representable."""
        raise NotImplementedError(
            "WeightedDistribution uses its host accumulator to preserve effective-weight metadata."
        )

    def dist_to_encoder(self) -> WeightedDataEncoder:
        """Returns a WeightedDataEncoder for encoding sequences of (value, weight) observations."""
        return WeightedDataEncoder(encoder=self.dist.dist_to_encoder())

    def to_fisher(self, **kwargs):
        """Fisher view for the weighted wrapper."""
        if hasattr(self, "dist"):
            return WeightedFisherView(self)
        return super().to_fisher(**kwargs)

    def estimator(self, pseudo_count: float | None = None) -> WeightedEstimator:
        """Create a WeightedEstimator wrapping the base distribution's estimator.

        Args:
            pseudo_count (Optional[float]): Passed through to the base distribution's estimator.

        Returns:
            WeightedEstimator object.

        """
        if pseudo_count is not None:
            return WeightedEstimator(
                estimator=self.dist.estimator(pseudo_count=pseudo_count),
                name=self.name,
                keys=self.keys,
            )
        else:
            return WeightedEstimator(
                estimator=self.dist.estimator(),
                name=self.name,
                keys=self.keys,
            )

    def sampler(self, seed: int | None = None) -> WeightedSampler:
        """Create a WeightedSampler producing (value, weight) pairs.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            WeightedSampler object.

        """
        return WeightedSampler(self, seed)

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate canonical ``(value, 1.0)`` observations in child-probability order."""
        return WeightedEnumerator(self)

    def support_size(self) -> int | None:
        """Return the child support size; attached weight is metadata, not a support coordinate."""
        return self.dist.support_size()


class WeightedSampler(DistributionSampler):
    """Sampler for ``(value, weight)`` observations from a weighted distribution.

    The likelihood does not model the weight, so samples carry the neutral weight 1.0: accumulating
    (value, 1.0) is equivalent to accumulating the bare value with the base distribution. Values are
    drawn from the base distribution's sampler.

    Args:
        dist (WeightedDistribution): WeightedDistribution to draw samples from.
        seed (Optional[int]): Seed to set for sampling with RandomState.

    Attributes:
        dist (WeightedDistribution): WeightedDistribution to draw samples from.
        rng (RandomState): Seeded RandomState for sampling.
        dist_sampler (DistributionSampler): Sampler for the base distribution.

    """

    def __init__(self, dist: WeightedDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist_sampler = dist.dist.sampler(seed=self.new_seed())

    def sample(
        self, size: int | None = None, *, batched: bool = True
    ) -> tuple[Any, float] | Sequence[tuple[Any, float]]:
        """Draw iid (value, weight) samples, each with weight 1.0.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            A single (value, 1.0) tuple if size is None, else a list of size such tuples.

        """
        if size is None:
            return WeightedObservation(self.dist_sampler.sample(), 1.0)
        else:
            return [WeightedObservation(v, 1.0) for v in self.dist_sampler.sample(size=size)]


class WeightedEnumerator(DistributionEnumerator):
    """Enumerate one canonical metadata representation for every child support value."""

    def __init__(self, dist: WeightedDistribution) -> None:
        super().__init__(dist)
        self._child = child_enumerator(dist.dist, "WeightedDistribution.dist")

    def __next__(self) -> tuple[WeightedObservation, float]:
        value, log_probability = next(self._child)
        return WeightedObservation(value, 1.0), float(log_probability)


class WeightedAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator that scales each observation's weight by its attached score.

    Args:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
        name (Optional[str]): Optional accumulator name.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
        name (Optional[str]): Optional accumulator name.

    """

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        name: str | None = None,
        keys: str | None = None,
    ):
        self.accumulator = accumulator
        self.name = name
        self.keys = keys if keys is not None else getattr(accumulator, "keys", None)
        self.effective_weight = 0.0

    def initialize(self, x: tuple[D, float], weight: float, rng: np.random.RandomState) -> None:
        """Initialize the base accumulator with observation x[0] weighted by weight*x[1].

        Args:
            x (Tuple[D, float]): Observation (value, weight) pair.
            weight (float): External weight on the observation.
            rng (RandomState): Random number generator for initialization.

        """
        observation = _observation(x)
        effective = _effective_weight(weight, observation.weight)
        self.accumulator.initialize(observation.value, effective, rng)
        self.effective_weight += effective

    def update(self, x: tuple[D, float], weight: float, estimate: WeightedDistribution) -> None:
        """Update the base accumulator with observation x[0] weighted by weight*x[1].

        Args:
            x (Tuple[D, float]): Observation (value, weight) pair.
            weight (float): External weight on the observation.
            estimate (WeightedDistribution): Previous estimate of the weighted distribution.

        """
        observation = _observation(x)
        effective = _effective_weight(weight, observation.weight)
        self.accumulator.update(
            observation.value,
            effective,
            None if estimate is None else estimate.dist,
        )
        self.effective_weight += effective

    def seq_update(self, x, weights: np.ndarray, estimate: WeightedDistribution) -> None:
        """Vectorized update of the base accumulator with weights scaled by the observation weights.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.
            weights (np.ndarray): External weights on the observations.
            estimate (WeightedDistribution): Previous estimate of the weighted distribution.

        """
        attached = _validated_weight_vector(x[1], len(x[1]), name="attached weights")
        external = _validated_weight_vector(weights, len(attached), name="external weights")
        effective = external * attached
        if np.any(~np.isfinite(effective)):
            raise ValueError("attached-times-external weights must be finite.")
        self.accumulator.seq_update(
            x[0],
            effective,
            None if estimate is None else estimate.dist,
        )
        self.effective_weight += float(effective.sum())

    def seq_update_engine(self, x, weights: Any, estimate: WeightedDistribution, engine: Any) -> None:
        """Engine-resident E-step: per-observation weights are scaled on the active engine and the
        base accumulator is routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        attached = _validated_weight_vector(x[1], len(x[1]), name="attached weights")
        external = np.asarray(engine.to_numpy(weights), dtype=np.float64)
        external = _validated_weight_vector(external, len(attached), name="external weights")
        effective = external * attached
        if np.any(~np.isfinite(effective)):
            raise ValueError("attached-times-external weights must be finite.")
        w = engine.asarray(effective)
        child_seq_update(self.accumulator, x[0], w, estimate.dist if estimate is not None else None, engine)
        self.effective_weight += float(effective.sum())

    def seq_initialize(self, x: tuple[E, np.ndarray], weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of the base accumulator with scaled weights.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.
            weights (np.ndarray): External weights on the observations.
            rng (RandomState): Random number generator for initialization.

        """
        attached = _validated_weight_vector(x[1], len(x[1]), name="attached weights")
        external = _validated_weight_vector(weights, len(attached), name="external weights")
        effective = external * attached
        if np.any(~np.isfinite(effective)):
            raise ValueError("attached-times-external weights must be finite.")
        self.accumulator.seq_initialize(x[0], effective, rng)
        self.effective_weight += float(effective.sum())

    def combine(self, suff_stat: WeightedStatistics) -> WeightedAccumulator:
        """Combine the base accumulator's sufficient statistics with suff_stat.

        Args:
            suff_stat (SS): Sufficient statistics of the base accumulator.

        Returns:
            This WeightedAccumulator.

        """
        checked = _validated_statistics(suff_stat)
        self.accumulator.combine(checked.child)
        self.effective_weight += checked.effective_weight
        return self

    def from_value(self, x: WeightedStatistics) -> WeightedAccumulator:
        """Set the base accumulator's sufficient statistics from x.

        Args:
            x (SS): Sufficient statistics of the base accumulator.

        Returns:
            This WeightedAccumulator.

        """
        checked = _validated_statistics(x)
        self.accumulator.from_value(checked.child)
        self.effective_weight = checked.effective_weight

        return self

    def value(self) -> WeightedStatistics:
        """Return child statistics and their effective attached-times-external weight."""
        return WeightedStatistics(self.accumulator.value(), self.effective_weight)

    def scale(self, c: float) -> WeightedAccumulator:
        """Scale the child accumulator through its family-specific protocol."""
        checked = _finite_nonnegative(c, name="scale")
        self.accumulator.scale(checked)
        self.effective_weight *= checked
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Pool child statistics and their effective weight under one parameter key."""
        if self.keys is None:
            self.accumulator.key_merge(stats_dict)
            return
        if self.keys in stats_dict:
            self.combine(stats_dict[self.keys])
        stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace child statistics and effective weight from their shared parameter pool."""
        if self.keys is None:
            self.accumulator.key_replace(stats_dict)
        elif self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> WeightedDataEncoder:
        """Returns a WeightedDataEncoder for encoding sequences of (value, weight) observations."""
        return WeightedDataEncoder(encoder=self.accumulator.acc_to_encoder())


class WeightedAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for weighted accumulators.

    Args:
        factory (StatisticAccumulatorFactory): Accumulator factory for the base distribution.
        name (Optional[str]): Optional name assigned to created accumulators.

    Attributes:
        factory (StatisticAccumulatorFactory): Accumulator factory for the base distribution.
        name (Optional[str]): Optional name assigned to created accumulators.

    """

    def __init__(
        self,
        factory: StatisticAccumulatorFactory,
        name: str | None = None,
        keys: str | None = None,
    ):
        self.factory = factory
        self.name = name
        self.keys = keys

    def make(self) -> WeightedAccumulator:
        """Returns a new WeightedAccumulator wrapping a fresh base accumulator."""
        return WeightedAccumulator(
            accumulator=self.factory.make(),
            name=self.name,
            keys=self.keys,
        )


class WeightedEstimator(ParameterEstimator):
    """Estimator for a weighted distribution from weighted observations.

    Args:
        estimator (ParameterEstimator): Estimator for the base distribution.
        name (Optional[str]): Optional name assigned to the estimated distribution.

    Attributes:
        estimator (ParameterEstimator): Estimator for the base distribution.
        name (Optional[str]): Optional name assigned to the estimated distribution.

    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        name: str | None = None,
        keys: str | None = None,
    ):
        self.estimator = estimator
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WeightedAccumulatorFactory:
        """Returns a WeightedAccumulatorFactory wrapping the base estimator's factory."""
        return WeightedAccumulatorFactory(
            factory=self.estimator.accumulator_factory(),
            name=self.name,
            keys=self.keys,
        )

    def estimate(self, nobs: float | None, suff_stat: WeightedStatistics) -> WeightedDistribution:
        """Estimate a WeightedDistribution from the base distribution's sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat (SS): Sufficient statistics of the base accumulator.

        Returns:
            WeightedDistribution wrapping the estimated base distribution.

        """
        checked = _validated_statistics(suff_stat)
        return WeightedDistribution(
            dist=self.estimator.estimate(checked.effective_weight, checked.child),
            name=self.name,
            keys=self.keys,
        )

    def resident_accumulation_supported(self) -> bool:
        """Use the host wrapper accumulator so effective-weight metadata cannot be dropped."""
        return False


class WeightedDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of iid ``(value, weight)`` observations.

    Args:
        encoder (DataSequenceEncoder): Encoder for the base distribution's values.

    Attributes:
        encoder (DataSequenceEncoder): Encoder for the base distribution's values.

    """

    def __init__(self, encoder: DataSequenceEncoder) -> None:
        self.encoder = encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the weighted encoder."""
        return "WeightedDataEncoder(encoder=%s)" % (repr(self.encoder))

    def __eq__(self, other: object) -> bool:
        """Return True if other is a WeightedDataEncoder with an equal base encoder."""
        if isinstance(other, WeightedDataEncoder):
            return other.encoder == self.encoder
        else:
            return False

    def seq_encode(self, x: Sequence[tuple[D, float]]) -> tuple[Any, np.ndarray]:
        """Encode a sequence of (value, weight) observations for vectorized use.

        Args:
            x (Sequence[Tuple[D, float]]): Sequence of iid (value, weight) observations.

        Returns:
            Tuple of base-encoded values and a numpy array of weights.

        """
        observations = [_observation(value) for value in x]
        return (
            self.encoder.seq_encode([observation.value for observation in observations]),
            np.asarray([observation.weight for observation in observations], dtype=np.float64),
        )


# --- Fisher view(s) co-located with this family ---
class WeightedFisherView(FixedFisherView):
    """Fisher view that scales child sufficient statistics by observation weights."""

    def __init__(self, dist: Any) -> None:
        self.child_view = to_fisher(dist.dist)
        super().__init__(dist, list(self.child_view.vectorizer.labels))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        observations = [_observation(value) for value in data]
        values = [observation.value for observation in observations]
        weights = np.asarray([observation.weight for observation in observations], dtype=np.float64)
        return self.child_view.expected_statistics_matrix(data=values) * weights[:, None]

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        enc_child, weights = enc_data
        checked = _validated_weight_vector(weights, len(weights), name="attached weights")
        return self.child_view.seq_expected_statistics(enc_child) * checked[:, None]

    def _model_mean(self) -> np.ndarray:
        return self.child_view.mean_statistics()

    def _model_fisher(self) -> np.ndarray:
        return np.asarray(self.child_view.fisher_information(ridge=0.0), dtype=np.float64)

    def score_center(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Return the empirical center used when centering weighted Fisher scores."""
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)
