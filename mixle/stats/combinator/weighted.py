"""Weighted observation wrapper around a base distribution.

Data type: Tuple[D, float]: An observation is a pair (value, weight) where value has the data type D of
the base distribution and weight is a non-negative score attached to the observation. The weight does not
enter the likelihood; it only scales the observation's contribution to the sufficient statistics during
estimation. Likelihood evaluations delegate to the base distribution on the value alone, i.e.

    P((x, w)) = P_base(x).

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

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
            statistics=(StatisticSpec("weighted_child", kind="child_stat"),),
            support="weighted_observation",
            children=children,
            child_roles=("value",) if child is not None else (),
            differentiable=False,
        )

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, name: str | None = None):
        self.dist = dist
        self.name = name

    def __str__(self) -> str:
        """Return a constructor-style representation of the weighted distribution."""
        return "WeightedDistribution(dist=%s, name=%s)" % (repr(self.dist), repr(self.name))

    def density(self, x: D) -> float:
        """Density of the base distribution at observation value x.

        Args:
            x (D): Observation value (weight excluded).

        Returns:
            Density of the base distribution at x.

        """
        return self.dist.density(x)

    def log_density(self, x: D) -> float:
        """Log-density of the base distribution at observation value x.

        The observation weight does not enter the likelihood, so this is simply the
        base distribution's log-density evaluated on the value.

        Args:
            x (D): Observation value (weight excluded).

        Returns:
            Log-density of the base distribution at x.

        """
        return self.dist.log_density(x)

    def seq_log_density(self, x: tuple[E, np.ndarray]) -> np.ndarray:
        """Vectorized log-density of the base distribution on encoded values.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.

        Returns:
            Numpy array of base-distribution log-densities.

        """
        return self.dist.seq_log_density(x[0])

    def backend_seq_log_density(self, x: tuple[E, np.ndarray], engine: Any) -> Any:
        """Engine-neutral vectorized log-density delegated to the value distribution."""
        from mixle.stats.compute.backend import backend_seq_log_density

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

        return stacked_component_log_density(x[0], params["child_route"], engine)

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: tuple[E, np.ndarray], weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> Any:
        """Return child legacy statistics with posterior weights scaled by observation weights."""
        from mixle.stats.compute.stacked import StackedEstimatorView, stacked_component_sufficient_statistics

        ww = engine.asarray(weights) * engine.asarray(x[1])[:, None]
        num_components = int(params["num_components"])
        component_estimators = tuple(getattr(est, "estimator", None) for est in getattr(estimator, "estimators", ()))
        child_estimator = (
            StackedEstimatorView(component_estimators) if len(component_estimators) == num_components else None
        )
        return stacked_component_sufficient_statistics(x[0], ww, params["child_route"], engine, child_estimator)

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
            return WeightedEstimator(estimator=self.dist.estimator(pseudo_count=pseudo_count), name=self.name)
        else:
            return WeightedEstimator(estimator=self.dist.estimator(), name=self.name)

    def sampler(self, seed: int | None = None) -> WeightedSampler:
        """Create a WeightedSampler producing (value, weight) pairs.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            WeightedSampler object.

        """
        return WeightedSampler(self, seed)

    def enumerator(self) -> DistributionEnumerator:
        """Delegates to the base distribution's enumerator (log_density is pure delegation)."""
        return child_enumerator(self.dist, "WeightedDistribution.dist")


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
            return self.dist_sampler.sample(), 1.0
        else:
            return [(v, 1.0) for v in self.dist_sampler.sample(size=size)]


class WeightedAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator that scales each observation's weight by its attached score.

    Args:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
        name (Optional[str]): Optional accumulator name.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
        name (Optional[str]): Optional accumulator name.

    """

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, name: str | None = None):
        self.accumulator = accumulator
        self.name = name

    def initialize(self, x: tuple[D, float], weight: float, rng: np.random.RandomState) -> None:
        """Initialize the base accumulator with observation x[0] weighted by weight*x[1].

        Args:
            x (Tuple[D, float]): Observation (value, weight) pair.
            weight (float): External weight on the observation.
            rng (RandomState): Random number generator for initialization.

        """
        self.accumulator.initialize(x[0], weight * x[1], rng)

    def update(self, x: tuple[D, float], weight: float, estimate: WeightedDistribution) -> None:
        """Update the base accumulator with observation x[0] weighted by weight*x[1].

        Args:
            x (Tuple[D, float]): Observation (value, weight) pair.
            weight (float): External weight on the observation.
            estimate (WeightedDistribution): Previous estimate of the weighted distribution.

        """
        self.accumulator.update(x[0], weight * x[1], estimate.dist)

    def seq_update(self, x, weights: np.ndarray, estimate: WeightedDistribution) -> None:
        """Vectorized update of the base accumulator with weights scaled by the observation weights.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.
            weights (np.ndarray): External weights on the observations.
            estimate (WeightedDistribution): Previous estimate of the weighted distribution.

        """
        self.accumulator.seq_update(x[0], weights * x[1], estimate.dist)

    def seq_update_engine(self, x, weights: Any, estimate: WeightedDistribution, engine: Any) -> None:
        """Engine-resident E-step: per-observation weights are scaled on the active engine and the
        base accumulator is routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        w = engine.asarray(weights) * engine.asarray(np.asarray(x[1], dtype=np.float64))
        child_seq_update(self.accumulator, x[0], w, estimate.dist if estimate is not None else None, engine)

    def seq_initialize(self, x: tuple[E, np.ndarray], weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of the base accumulator with scaled weights.

        Args:
            x (Tuple[E, np.ndarray]): Sequence encoded values and weights from WeightedDataEncoder.
            weights (np.ndarray): External weights on the observations.
            rng (RandomState): Random number generator for initialization.

        """
        self.accumulator.seq_initialize(x[0], weights * x[1], rng)

    def combine(self, suff_stat: SS) -> WeightedAccumulator:
        """Combine the base accumulator's sufficient statistics with suff_stat.

        Args:
            suff_stat (SS): Sufficient statistics of the base accumulator.

        Returns:
            This WeightedAccumulator.

        """
        self.accumulator.combine(suff_stat)
        return self

    def from_value(self, x: SS) -> WeightedAccumulator:
        """Set the base accumulator's sufficient statistics from x.

        Args:
            x (SS): Sufficient statistics of the base accumulator.

        Returns:
            This WeightedAccumulator.

        """
        self.accumulator.from_value(x)

        return self

    def value(self) -> Any:
        """Returns the base accumulator's sufficient statistics."""
        return self.accumulator.value()

    def scale(self, c: float) -> WeightedAccumulator:
        """Scale the child accumulator through its family-specific protocol."""
        self.accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed sufficient statistics of the base accumulator into stats_dict."""
        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace keyed sufficient statistics of the base accumulator from stats_dict."""
        self.accumulator.key_replace(stats_dict)

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

    def __init__(self, factory: StatisticAccumulatorFactory, name: str | None = None):
        self.factory = factory
        self.name = name

    def make(self) -> WeightedAccumulator:
        """Returns a new WeightedAccumulator wrapping a fresh base accumulator."""
        return WeightedAccumulator(accumulator=self.factory.make(), name=self.name)


class WeightedEstimator(ParameterEstimator):
    """Estimator for a weighted distribution from weighted observations.

    Args:
        estimator (ParameterEstimator): Estimator for the base distribution.
        name (Optional[str]): Optional name assigned to the estimated distribution.

    Attributes:
        estimator (ParameterEstimator): Estimator for the base distribution.
        name (Optional[str]): Optional name assigned to the estimated distribution.

    """

    def __init__(self, estimator: ParameterEstimator, name: str | None = None):
        self.estimator = estimator
        self.name = name

    def accumulator_factory(self) -> WeightedAccumulatorFactory:
        """Returns a WeightedAccumulatorFactory wrapping the base estimator's factory."""
        return WeightedAccumulatorFactory(factory=self.estimator.accumulator_factory(), name=self.name)

    def estimate(self, nobs: float | None, suff_stat: SS) -> WeightedDistribution:
        """Estimate a WeightedDistribution from the base distribution's sufficient statistics.

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat (SS): Sufficient statistics of the base accumulator.

        Returns:
            WeightedDistribution wrapping the estimated base distribution.

        """
        return WeightedDistribution(dist=self.estimator.estimate(nobs, suff_stat), name=self.name)


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
        return self.encoder.seq_encode([xx[0] for xx in x]), np.asarray([xx[1] for xx in x], dtype=float)


# --- Fisher view(s) co-located with this family ---
class WeightedFisherView(FixedFisherView):
    """Fisher view that scales child sufficient statistics by observation weights."""

    def __init__(self, dist: Any) -> None:
        self.child_view = to_fisher(dist.dist)
        super().__init__(dist, list(self.child_view.vectorizer.labels))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        values = [x[0] for x in data]
        weights = np.asarray([x[1] for x in data], dtype=np.float64)
        return self.child_view.expected_statistics_matrix(data=values) * weights[:, None]

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        enc_child, weights = enc_data
        return self.child_view.seq_expected_statistics(enc_child) * np.asarray(weights, dtype=np.float64)[:, None]

    def _model_mean(self) -> np.ndarray:
        return self.child_view.mean_statistics()

    def _model_fisher(self) -> np.ndarray:
        return np.asarray(self.child_view.fisher_information(ridge=0.0), dtype=np.float64)

    def score_center(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Return the empirical center used when centering weighted Fisher scores."""
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)
