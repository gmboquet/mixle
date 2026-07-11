"""Frozen deterministic feature maps composed with inner densities.

A deterministic, non-invertible feature function (for example
:func:`mixle.represent.modality.image_features`) reduces a raw item -- an image array, a signal, any
shape a plain scalar/vector family cannot represent -- to a fixed-length vector, and an inner
distribution/estimator (any five-piece mixle family, including a neural density) is fit on the
*induced feature-space distribution*.

Stated plainly: this is a genuine, well-defined density over the feature representation, not a claim
about the density of the raw item -- there is no Jacobian correction because the map is not invertible.
Anywhere this leaf is chosen, the reasoning is recorded so that distinction stays visible (see
``mixle.utils.automatic``'s modality routing).

Feature functions are looked up by a registered name, not passed as a raw callable, so the leaf
serializes: the same closed-registry pattern :mod:`mixle.models.neural_density` uses for its hoisted
``nn.Module`` classes. Register a feature function once (``register_feature_fn``); the leaf then
carries only the name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_FEATURE_FNS: dict[str, Callable[[Any], np.ndarray]] = {}


def register_feature_fn(name: str, fn: Callable[[Any], np.ndarray]) -> None:
    """Register ``fn`` (raw item -> fixed-length vector) under ``name`` so a leaf can carry just the name."""
    _FEATURE_FNS[name] = fn


def feature_fn(name: str) -> Callable[[Any], np.ndarray]:
    """Look up a registered feature function by name; raises if it was never registered."""
    try:
        return _FEATURE_FNS[name]
    except KeyError:
        raise KeyError(
            f"no feature function registered under {name!r}; call register_feature_fn(name, fn) first"
        ) from None


class FeatureMapDensity(SequenceEncodableProbabilityDistribution):
    """``p(feature_fn(x))`` for a registered, deterministic ``feature_fn`` and inner distribution."""

    __pysp_serializable__ = True

    def __init__(self, feature_name: str, inner: Any, name: str | None = None) -> None:
        self.feature_name = feature_name
        self.inner = inner
        self.name = name

    def __str__(self) -> str:
        return f"FeatureMapDensity({self.feature_name!r}, {self.inner})"

    def density(self, x: Any) -> float:
        """Return the induced feature-space density at raw item ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return ``log p(feature_fn(x))`` under the inner distribution."""
        return float(self.inner.log_density(feature_fn(self.feature_name)(x)))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return inner log densities for an already-featurized batch."""
        # x is the already-featurized (n, d) batch produced by FeatureMapEncoder.seq_encode.
        return self.inner.seq_log_density(x)

    def sampler(self, seed: int | None = None) -> FeatureMapSampler:
        """Return a sampler for the inner feature-space distribution."""
        return FeatureMapSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> FeatureMapEstimator:
        """Return an estimator that fits the inner estimator on registered features."""
        return FeatureMapEstimator(self.feature_name, self.inner.estimator(pseudo_count), name=self.name)

    def dist_to_encoder(self) -> FeatureMapEncoder:
        """Return the encoder that maps raw items to feature vectors."""
        return FeatureMapEncoder(self.feature_name)

    # to_dict/from_dict are inherited from ProbabilityDistribution: they delegate to the generic
    # to_serializable/from_serializable registry path, which recurses into `self.inner` (itself a
    # registered mixle distribution) via __pysp_getstate__/__pysp_setstate__ below -- no custom
    # encoding needed here, unlike NeuralDensity's module bytes.

    def __pysp_getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class FeatureMapSampler(DistributionSampler):
    """Sampler for the feature-space distribution induced by a feature-map leaf."""

    def __init__(self, dist: FeatureMapDensity, seed: int | None = None) -> None:
        self.dist = dist
        self.inner_sampler = dist.inner.sampler(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Samples from the INNER (feature-space) distribution -- there is no inverse feature map back to raw items."""
        return self.inner_sampler.sample(size)


class FeatureMapEncoder(DataSequenceEncoder):
    """Encode raw items by applying a registered deterministic feature function."""

    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name

    def __str__(self) -> str:
        return f"FeatureMapEncoder({self.feature_name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FeatureMapEncoder) and other.feature_name == self.feature_name

    def seq_encode(self, data: list) -> np.ndarray:
        """Convert raw items into a stacked feature matrix."""
        fn = feature_fn(self.feature_name)
        return np.stack([np.asarray(fn(x), dtype=float) for x in data])


class FeatureMapAccumulator(SequenceEncodableStatisticAccumulator):
    """Delegate accumulation to the inner estimator after feature extraction."""

    def __init__(self, feature_name: str, inner_acc: Any) -> None:
        self.feature_name = feature_name
        self.inner_acc = inner_acc

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        """Feature-map one raw item and add it to the inner accumulator."""
        inner_estimate = estimate.inner if estimate is not None else None
        self.inner_acc.update(feature_fn(self.feature_name)(x), weight, inner_estimate)

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Pass an already-featurized batch through to the inner accumulator."""
        inner_estimate = estimate.inner if estimate is not None else None
        self.inner_acc.seq_update(enc, weights, inner_estimate)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """Initialize the inner accumulator from one feature-mapped item."""
        self.inner_acc.initialize(feature_fn(self.feature_name)(x), weight, rng)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize the inner accumulator from an encoded feature batch."""
        self.inner_acc.seq_initialize(enc, weights, rng)

    def combine(self, other: Any) -> FeatureMapAccumulator:
        """Merge sufficient statistics into the inner accumulator."""
        self.inner_acc.combine(other)
        return self

    def value(self) -> Any:
        """Return the inner accumulator's sufficient-statistic value."""
        return self.inner_acc.value()

    def from_value(self, v: Any) -> FeatureMapAccumulator:
        """Restore the inner accumulator from its value representation."""
        self.inner_acc.from_value(v)
        return self

    def acc_to_encoder(self) -> FeatureMapEncoder:
        """Return the feature-map encoder expected by this accumulator."""
        return FeatureMapEncoder(self.feature_name)


class FeatureMapAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for feature-map accumulators wrapping an inner accumulator factory."""

    def __init__(self, feature_name: str, inner_factory: Any) -> None:
        self.feature_name = feature_name
        self.inner_factory = inner_factory

    def make(self) -> FeatureMapAccumulator:
        """Create a fresh feature-map accumulator."""
        return FeatureMapAccumulator(self.feature_name, self.inner_factory.make())


class FeatureMapEstimator(ParameterEstimator):
    """Fits ``inner`` on ``feature_fn(x)`` for raw items ``x`` -- the estimator side of :class:`FeatureMapDensity`."""

    def __init__(self, feature_name: str, inner: ParameterEstimator, name: str | None = None) -> None:
        self.feature_name = feature_name
        self.inner = inner
        self.name = name

    def accumulator_factory(self) -> FeatureMapAccumulatorFactory:
        """Return an accumulator factory that feature-maps raw inputs before inner accumulation."""
        return FeatureMapAccumulatorFactory(self.feature_name, self.inner.accumulator_factory())

    def estimate(self, nobs: float | None, suff_stat: Any) -> FeatureMapDensity:
        """Estimate the inner distribution and wrap it as a feature-map density."""
        return FeatureMapDensity(self.feature_name, self.inner.estimate(nobs, suff_stat), name=self.name)


def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover - serialization support is optional at import  # noqa: BLE001
        return
    register_serializable_class(FeatureMapDensity)


_register_serializable()
