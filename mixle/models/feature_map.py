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

Feature functions are looked up by an immutable registered identity rather than passed as a raw callable.
Every identity pins a name, version, output dimension, and code/schema digest in the serialized leaf.
Duplicate registration cannot silently replace an artifact's semantics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FeatureIdentity:
    """Immutable identity of a deterministic raw-item feature transform."""

    name: str
    version: str
    feature_dim: int
    digest: str


@dataclass(frozen=True)
class _FeatureRegistration:
    identity: FeatureIdentity
    fn: Callable[[Any], np.ndarray]


_FEATURE_FNS: dict[tuple[str, str], _FeatureRegistration] = {}


def register_feature_fn(
    name: str,
    fn: Callable[[Any], np.ndarray],
    *,
    version: str,
    feature_dim: int,
) -> FeatureIdentity:
    """Register one immutable, versioned raw-item feature transform.

    Repeating the exact registration is idempotent. Reusing ``(name, version)`` for different code or
    output schema raises; callers must publish a new version instead.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("feature name must be a non-empty string")
    if not isinstance(version, str) or not version:
        raise ValueError("feature version must be a non-empty string")
    if not callable(fn):
        raise TypeError("feature function must be callable")
    feature_dim = _positive_int(feature_dim, "feature_dim")
    identity = FeatureIdentity(name, version, feature_dim, _feature_digest(fn, feature_dim))
    key = (name, version)
    current = _FEATURE_FNS.get(key)
    if current is not None:
        if current.identity != identity:
            raise ValueError(
                f"feature identity {name!r} version {version!r} is already registered with "
                "different code or schema; publish a new version"
            )
        return current.identity
    _FEATURE_FNS[key] = _FeatureRegistration(identity, fn)
    return identity


def feature_fn(
    name: str,
    *,
    version: str | None = None,
    digest: str | None = None,
    feature_dim: int | None = None,
) -> Callable[[Any], np.ndarray]:
    """Resolve a feature function and optionally verify its complete pinned identity."""
    registration = _resolve_feature(name, version=version, digest=digest, feature_dim=feature_dim)
    return registration.fn


def _resolve_feature(
    name: str,
    *,
    version: str | None,
    digest: str | None,
    feature_dim: int | None,
) -> _FeatureRegistration:
    if version is None:
        matches = [registration for (candidate, _), registration in _FEATURE_FNS.items() if candidate == name]
        if len(matches) != 1:
            raise KeyError(
                f"feature {name!r} has {len(matches)} registered versions; specify an exact version"
            )
        registration = matches[0]
    else:
        try:
            registration = _FEATURE_FNS[(name, version)]
        except KeyError:
            raise KeyError(
                f"no feature function registered under name={name!r}, version={version!r}"
            ) from None
    identity = registration.identity
    if digest is not None and digest != identity.digest:
        raise ValueError(
            f"feature digest mismatch for {name!r} version {identity.version!r}: "
            f"artifact={digest!r}, registry={identity.digest!r}"
        )
    if feature_dim is not None and feature_dim != identity.feature_dim:
        raise ValueError(
            f"feature dimension mismatch for {name!r} version {identity.version!r}: "
            f"artifact={feature_dim}, registry={identity.feature_dim}"
        )
    return registration


def _identity_args(identity: FeatureIdentity) -> dict[str, Any]:
    return {
        "version": identity.version,
        "digest": identity.digest,
        "feature_dim": identity.feature_dim,
    }


def _feature_vector(identity: FeatureIdentity, raw_item: Any) -> np.ndarray:
    registration = _resolve_feature(identity.name, **_identity_args(identity))
    try:
        vector = np.asarray(registration.fn(raw_item), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"feature {identity.name!r} version {identity.version!r} could not encode the raw item"
        ) from exc
    if vector.shape != (identity.feature_dim,):
        raise ValueError(
            f"feature {identity.name!r} version {identity.version!r} must return shape "
            f"({identity.feature_dim},), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"feature {identity.name!r} version {identity.version!r} returned non-finite values"
        )
    return vector


class FeatureMapDensity(SequenceEncodableProbabilityDistribution):
    """``p(feature_fn(x))`` for a registered, deterministic ``feature_fn`` and inner distribution."""

    __pysp_serializable__ = True

    def __init__(
        self,
        feature_name: str,
        inner: Any,
        name: str | None = None,
        *,
        feature_version: str | None = None,
        feature_digest: str | None = None,
        feature_dim: int | None = None,
    ) -> None:
        registration = _resolve_feature(
            feature_name,
            version=feature_version,
            digest=feature_digest,
            feature_dim=feature_dim,
        )
        self.feature_name = registration.identity.name
        self.feature_version = registration.identity.version
        self.feature_digest = registration.identity.digest
        self.feature_dim = registration.identity.feature_dim
        self.inner = inner
        self.name = name

    @property
    def feature_identity(self) -> FeatureIdentity:
        return FeatureIdentity(
            self.feature_name,
            self.feature_version,
            self.feature_dim,
            self.feature_digest,
        )

    def __str__(self) -> str:
        return f"FeatureMapDensity({self.feature_name!r}, {self.inner})"

    def density(self, x: Any) -> float:
        """Return the induced feature-space density at raw item ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return ``log p(feature_fn(x))`` under the inner distribution."""
        return float(self.inner.log_density(_feature_vector(self.feature_identity, x)))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return inner log densities for an already-featurized batch."""
        features = _feature_matrix(x, self.feature_identity)
        return self.inner.seq_log_density(features)

    def sampler(self, seed: int | None = None) -> DistributionSampler:
        """Raw-item sampling is undefined because the feature map has no inverse."""
        raise NotImplementedError(
            "FeatureMapDensity scores raw items through a non-invertible feature map and cannot sample "
            "raw items. Use feature_sampler() to sample the explicitly separate inner feature distribution."
        )

    def feature_sampler(self, seed: int | None = None) -> DistributionSampler:
        """Return the sampler of the explicitly separate inner feature-space distribution."""
        return self.inner.sampler(seed)

    def estimator(self, pseudo_count: float | None = None) -> FeatureMapEstimator:
        """Return an estimator that fits the inner estimator on registered features."""
        return FeatureMapEstimator(
            self.feature_name,
            self.inner.estimator(pseudo_count),
            name=self.name,
            feature_version=self.feature_version,
            feature_digest=self.feature_digest,
            feature_dim=self.feature_dim,
        )

    def dist_to_encoder(self) -> FeatureMapEncoder:
        """Return the encoder that maps raw items to feature vectors."""
        return FeatureMapEncoder(
            self.feature_name,
            feature_version=self.feature_version,
            feature_digest=self.feature_digest,
            feature_dim=self.feature_dim,
        )

    # to_dict/from_dict are inherited from ProbabilityDistribution: they delegate to the generic
    # to_serializable/from_serializable registry path, which recurses into `self.inner` (itself a
    # registered mixle distribution) via __pysp_getstate__/__pysp_setstate__ below -- no custom
    # encoding needed here, unlike NeuralDensity's module bytes.

    def __pysp_getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        required = {"feature_name", "feature_version", "feature_digest", "feature_dim", "inner", "name"}
        if set(state) != required:
            raise ValueError(
                f"invalid FeatureMapDensity state fields: missing={sorted(required - set(state))}, "
                f"extra={sorted(set(state) - required)}"
            )
        _resolve_feature(
            state["feature_name"],
            version=state["feature_version"],
            digest=state["feature_digest"],
            feature_dim=state["feature_dim"],
        )
        self.__dict__.update(state)


class FeatureMapEncoder(DataSequenceEncoder):
    """Encode raw items by applying a registered deterministic feature function."""

    def __init__(
        self,
        feature_name: str,
        *,
        feature_version: str | None = None,
        feature_digest: str | None = None,
        feature_dim: int | None = None,
    ) -> None:
        registration = _resolve_feature(
            feature_name,
            version=feature_version,
            digest=feature_digest,
            feature_dim=feature_dim,
        )
        self.identity = registration.identity
        self.feature_name = self.identity.name

    def __str__(self) -> str:
        return (
            f"FeatureMapEncoder(name={self.identity.name!r}, version={self.identity.version!r}, "
            f"dim={self.identity.feature_dim}, digest={self.identity.digest[:12]!r})"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FeatureMapEncoder) and other.identity == self.identity

    def seq_encode(self, data: list) -> np.ndarray:
        """Convert raw items into a stacked feature matrix."""
        if not isinstance(data, list) or not data:
            raise ValueError("feature-map encoder data must be a non-empty list of raw items")
        return np.stack([_feature_vector(self.identity, raw_item) for raw_item in data])


class FeatureMapAccumulator(SequenceEncodableStatisticAccumulator):
    """Delegate accumulation to the inner estimator after feature extraction."""

    def __init__(self, identity: FeatureIdentity, inner_acc: Any) -> None:
        self.identity = identity
        self.feature_name = identity.name
        self.inner_acc = inner_acc

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        """Feature-map one raw item and add it to the inner accumulator."""
        inner_estimate = estimate.inner if estimate is not None else None
        self.inner_acc.update(_feature_vector(self.identity, x), weight, inner_estimate)

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Pass an already-featurized batch through to the inner accumulator."""
        inner_estimate = estimate.inner if estimate is not None else None
        self.inner_acc.seq_update(_feature_matrix(enc, self.identity), weights, inner_estimate)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """Initialize the inner accumulator from one feature-mapped item."""
        self.inner_acc.initialize(_feature_vector(self.identity, x), weight, rng)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize the inner accumulator from an encoded feature batch."""
        self.inner_acc.seq_initialize(_feature_matrix(enc, self.identity), weights, rng)

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
        return FeatureMapEncoder(
            self.identity.name,
            feature_version=self.identity.version,
            feature_digest=self.identity.digest,
            feature_dim=self.identity.feature_dim,
        )


class FeatureMapAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for feature-map accumulators wrapping an inner accumulator factory."""

    def __init__(self, identity: FeatureIdentity, inner_factory: Any) -> None:
        self.identity = identity
        self.inner_factory = inner_factory

    def make(self) -> FeatureMapAccumulator:
        """Create a fresh feature-map accumulator."""
        return FeatureMapAccumulator(self.identity, self.inner_factory.make())


class FeatureMapEstimator(ParameterEstimator):
    """Fits ``inner`` on ``feature_fn(x)`` for raw items ``x`` -- the estimator side of :class:`FeatureMapDensity`."""

    def __init__(
        self,
        feature_name: str,
        inner: ParameterEstimator,
        name: str | None = None,
        *,
        feature_version: str | None = None,
        feature_digest: str | None = None,
        feature_dim: int | None = None,
    ) -> None:
        registration = _resolve_feature(
            feature_name,
            version=feature_version,
            digest=feature_digest,
            feature_dim=feature_dim,
        )
        self.identity = registration.identity
        self.feature_name = self.identity.name
        if not isinstance(inner, ParameterEstimator):
            raise TypeError("inner must be a ParameterEstimator")
        self.inner = inner
        self.name = name

    def accumulator_factory(self) -> FeatureMapAccumulatorFactory:
        """Return an accumulator factory that feature-maps raw inputs before inner accumulation."""
        return FeatureMapAccumulatorFactory(self.identity, self.inner.accumulator_factory())

    def estimate(self, nobs: float | None, suff_stat: Any) -> FeatureMapDensity:
        """Estimate the inner distribution and wrap it as a feature-map density."""
        return FeatureMapDensity(
            self.feature_name,
            self.inner.estimate(nobs, suff_stat),
            name=self.name,
            feature_version=self.identity.version,
            feature_digest=self.identity.digest,
            feature_dim=self.identity.feature_dim,
        )


def _feature_matrix(value: Any, identity: FeatureIdentity) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("encoded feature batch must be a finite two-dimensional matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != identity.feature_dim:
        raise ValueError(
            f"encoded feature batch must have non-empty shape (n, {identity.feature_dim}), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("encoded feature batch must contain only finite values")
    return matrix


def _feature_digest(fn: Callable[[Any], np.ndarray], feature_dim: int) -> str:
    code = getattr(fn, "__code__", None)
    if code is None:
        raise TypeError("feature function must be a Python callable with inspectable code")
    closure = tuple(repr(cell.cell_contents) for cell in (fn.__closure__ or ()))
    payload = repr(
        (
            getattr(fn, "__module__", None),
            getattr(fn, "__qualname__", None),
            code.co_code.hex(),
            code.co_consts,
            code.co_names,
            code.co_varnames,
            fn.__defaults__,
            fn.__kwdefaults__,
            closure,
            feature_dim,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover - serialization support is optional at import  # noqa: BLE001
        return
    register_serializable_class(FeatureMapDensity)


_register_serializable()
