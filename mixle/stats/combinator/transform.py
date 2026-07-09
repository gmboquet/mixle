"""Invertible-transform wrappers for sequence-encodable distributions.

The module implements identity, affine, exponential, logit, and custom
transforms plus the distribution, sampler, accumulator, and encoder plumbing
needed to apply them inside Mixle combinators.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)


def _uses_density_correction(dist: SequenceEncodableProbabilityDistribution, density_correction: bool | None) -> bool:
    if density_correction is not None:
        return bool(density_correction)
    try:
        dist.enumerator()
        return False
    except EnumerationError:
        return True


class IdentityTransform:
    """Identity transform y = x."""

    def __str__(self) -> str:
        return "IdentityTransform()"

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdentityTransform)

    def forward(self, x: Any) -> Any:
        """Return ``x`` unchanged."""
        return x

    def inverse(self, y: Any) -> Any:
        """Return ``y`` unchanged."""
        return y

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return the log absolute inverse-Jacobian determinant."""
        return 0.0

    def invalid_inverse_value(self) -> float:
        """Return a safe child-space fill value for invalid inverses."""
        return 0.0


class AffineTransform:
    """Affine transform y = loc + scale * x."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        if scale == 0.0 or not np.isfinite(scale):
            raise ValueError("AffineTransform requires finite non-zero scale.")
        self.loc = float(loc)
        self.scale = float(scale)
        self._log_abs_inv = -math.log(abs(self.scale))

    def __str__(self) -> str:
        return "AffineTransform(loc=%s, scale=%s)" % (repr(self.loc), repr(self.scale))

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AffineTransform) and self.loc == other.loc and self.scale == other.scale

    def forward(self, x: Any) -> Any:
        """Apply the affine map to a child-space value."""
        return self.loc + self.scale * x

    def inverse(self, y: Any) -> Any:
        """Map a transformed-space value back to child space."""
        return (y - self.loc) / self.scale

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return the constant affine inverse-Jacobian correction."""
        return self._log_abs_inv

    def invalid_inverse_value(self) -> float:
        """Return a safe child-space fill value for invalid inverses."""
        return 0.0


class ExpTransform:
    """Exponential transform y = exp(x), mapping real x to positive y."""

    def __str__(self) -> str:
        return "ExpTransform()"

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExpTransform)

    def forward(self, x: Any) -> Any:
        """Map a real value to the positive scale."""
        return np.exp(x)

    def inverse(self, y: Any) -> Any:
        """Map a positive transformed value back to the real line."""
        if y <= 0.0:
            raise ValueError("ExpTransform inverse requires y > 0.")
        return math.log(y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return the log inverse-Jacobian correction for ``log(y)``."""
        if y <= 0.0:
            raise ValueError("ExpTransform inverse requires y > 0.")
        return -math.log(y)

    def invalid_inverse_value(self) -> float:
        """Return a safe child-space fill value for invalid inverses."""
        return 0.0


class LogTransform:
    """Log transform y = log(x), mapping positive x to real y."""

    def __str__(self) -> str:
        return "LogTransform()"

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogTransform)

    def forward(self, x: Any) -> Any:
        """Map a positive child value to the real line."""
        if x <= 0.0:
            raise ValueError("LogTransform forward requires x > 0.")
        return math.log(x)

    def inverse(self, y: Any) -> Any:
        """Map a real transformed value back to the positive scale."""
        return math.exp(y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return the log inverse-Jacobian correction for ``exp(y)``."""
        return float(y)

    def invalid_inverse_value(self) -> float:
        """Return a safe positive child-space fill value for invalid inverses."""
        return 1.0


class LogitTransform:
    """Logistic transform y = 1 / (1 + exp(-x)), mapping real x to (0, 1)."""

    def __str__(self) -> str:
        return "LogitTransform()"

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogitTransform)

    def forward(self, x: Any) -> Any:
        """Map a real value into the open unit interval."""
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def inverse(self, y: Any) -> Any:
        """Map a unit-interval value back to the real line."""
        if y <= 0.0 or y >= 1.0:
            raise ValueError("LogitTransform inverse requires 0 < y < 1.")
        return math.log(y) - math.log1p(-y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return the log inverse-Jacobian correction for the logit map."""
        if y <= 0.0 or y >= 1.0:
            raise ValueError("LogitTransform inverse requires 0 < y < 1.")
        return -math.log(y) - math.log1p(-y)

    def invalid_inverse_value(self) -> float:
        """Return a safe child-space fill value for invalid inverses."""
        return 0.0


class TransformDistribution(SequenceEncodableProbabilityDistribution):
    """Push a child distribution through a fixed invertible transform.

    Observations live in transformed space. For fixed continuous transforms,
    log-density uses the inverse transform and adds the inverse-Jacobian term.
    The transform is not learned; estimation inverse-transforms observations
    and delegates sufficient statistics to the child estimator.
    """

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        transform: Any | None = None,
        density_correction: bool | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dist = dist
        self.transform = transform if transform is not None else IdentityTransform()
        self.density_correction = _uses_density_correction(dist, density_correction)
        self.name = name
        self.keys = keys

    def compute_capabilities(self):
        """Return capabilities delegated from the child distribution where safe."""
        from mixle.stats.compute.capabilities import (
            DistributionCapabilities,
            capabilities_for,
            delegated_engine_ready,
        )

        child = capabilities_for(self.dist)
        # cap delegated caps to composition-safe engines: a leaf-only engine (e.g. jax) does not
        # propagate through the transform kernel until it is verified there
        return DistributionCapabilities(
            engine_ready=delegated_engine_ready(child.engine_ready),
            kernel_status=child.kernel_status,
            numpy_only_reason=child.numpy_only_reason,
        )

    def compute_declaration(self):
        """Return a declaration describing this distribution as a transformed child."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        child = declaration_for(self.dist)
        children = () if child is None else (child,)
        return DistributionDeclaration(
            name="transform",
            distribution_type=type(self),
            parameters=(),
            statistics=(StatisticSpec("base", kind="child_stat"),),
            support="transformed",
            children=children,
            child_roles=("base",) if child is not None else (),
            differentiable=all(c.differentiable for c in children),
        )

    def __str__(self) -> str:
        return "TransformDistribution(%s, transform=%s, density_correction=%s, name=%s, keys=%s)" % (
            str(self.dist),
            repr(self.transform),
            repr(self.density_correction),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the log-density or log-mass at a single observation."""
        try:
            inv = self.transform.inverse(x)
            rv = self.dist.log_density(inv)
            if self.density_correction:
                rv += self.transform.log_abs_det_inverse_jacobian(x)
            return rv
        except Exception:
            return -np.inf

    def seq_log_density(self, x: tuple[Any, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        child_enc, log_jac, valid = x
        rv = self.dist.seq_log_density(child_enc)
        if self.density_correction:
            rv = rv + log_jac
        return np.where(valid, rv, -np.inf)

    def backend_seq_log_density(self, x: tuple[Any, np.ndarray, np.ndarray], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for inverse-encoded observations."""
        from mixle.stats.compute.backend import backend_seq_log_density

        child_enc, log_jac, valid = x
        rv = backend_seq_log_density(self.dist, child_enc, engine)
        if self.density_correction:
            rv = rv + engine.asarray(log_jac)
        invalid = engine.zeros(rv.shape) + float("-inf")
        return engine.where(engine.asarray(valid), rv, invalid)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["TransformDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked child parameters for homogeneous fixed-transform mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        first = dists[0]
        if any(
            dist.transform != first.transform or dist.density_correction != first.density_correction
            for dist in dists[1:]
        ):
            raise ValueError("Stacked TransformDistribution components require a shared transform policy.")
        child_dists = [dist.dist for dist in dists]
        try:
            child_route = stacked_component_params(child_dists, engine)
        except ValueError as exc:
            raise ValueError("Transform child %s is not stackable: %s" % (type(child_dists[0]).__name__, exc))
        return {
            "child_route": child_route,
            "density_correction": bool(first.density_correction),
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(
        cls, x: tuple[Any, np.ndarray, np.ndarray], params: dict[str, Any], engine: Any
    ) -> Any:
        """Return an ``(n, k)`` matrix of transformed child log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        child_enc, log_jac, valid = x
        scores = stacked_component_log_density(child_enc, params["child_route"], engine)
        if params["density_correction"]:
            scores = scores + engine.asarray(log_jac)[:, None]
        invalid = engine.zeros(tuple(getattr(scores, "shape", (0, 0)))) + float("-inf")
        return engine.where(engine.asarray(valid)[:, None], scores, invalid)

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: tuple[Any, np.ndarray, np.ndarray], weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> Any:
        """Return child legacy statistics for valid inverse-transformed observations."""
        from mixle.stats.compute.stacked import StackedEstimatorView, stacked_component_sufficient_statistics

        child_enc, _, valid = x
        ww = engine.asarray(weights) * engine.asarray(valid)[:, None]
        num_components = int(params["num_components"])
        component_estimators = tuple(getattr(est, "estimator", None) for est in getattr(estimator, "estimators", ()))
        child_estimator = (
            StackedEstimatorView(component_estimators) if len(component_estimators) == num_components else None
        )
        return stacked_component_sufficient_statistics(child_enc, ww, params["child_route"], engine, child_estimator)

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: Any, recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import TransformGradientFitState

        return TransformGradientFitState(self, recurse(self.dist, engine, torch, leaves))

    def sampler(self, seed: int | None = None) -> "TransformSampler":
        """Return a sampler for drawing observations from this distribution."""
        return TransformSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "TransformEstimator":
        """Return an estimator for fitting this distribution from data."""
        return TransformEstimator(
            self.dist.estimator(pseudo_count=pseudo_count),
            self.transform,
            density_correction=self.density_correction,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "TransformDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return TransformDataEncoder(
            self.dist.dist_to_encoder(), self.transform, density_correction=self.density_correction
        )

    def enumerator(self) -> "TransformEnumerator":
        """Return an enumerator over the distribution support when available."""
        return TransformEnumerator(self)


class TransformEnumerator(DistributionEnumerator):
    """Enumerate transformed child support for discrete child distributions."""

    def __init__(self, dist: TransformDistribution) -> None:
        super().__init__(dist)
        self.child_iter = child_enumerator(dist.dist, "TransformDistribution.dist")

    def __next__(self) -> tuple[Any, float]:
        v, lp = next(self.child_iter)
        return self.dist.transform.forward(v), lp


class TransformSampler(DistributionSampler):
    """Sampler that transforms draws from the child distribution."""

    def __init__(self, dist: TransformDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.child_sampler = dist.dist.sampler(seed=self.new_seed())

    def sample(self, size: int | None = None):
        """Draw child samples and map them through the transform."""
        x = self.child_sampler.sample(size=size)
        if size is None:
            return self.dist.transform.forward(x)
        return [self.dist.transform.forward(v) for v in x]


class TransformAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator that delegates inverse-transformed observations to the child."""

    def __init__(
        self,
        accumulator: SequenceEncodableStatisticAccumulator,
        transform: Any,
        density_correction: bool | None = None,
        name: str | None = None,
    ) -> None:
        self.accumulator = accumulator
        self.transform = transform
        self.density_correction = density_correction
        self.name = name

    def update(self, x: Any, weight: float, estimate: TransformDistribution | None) -> None:
        """Accumulate one inverse-transformed observation when it is valid."""
        try:
            inv = self.transform.inverse(x)
        except Exception:
            return
        self.accumulator.update(inv, weight, None if estimate is None else estimate.dist)

    def seq_update(
        self, x: tuple[Any, np.ndarray, np.ndarray], weights: np.ndarray, estimate: TransformDistribution | None
    ) -> None:
        """Accumulate a batch using validity-masked child weights."""
        child_enc, _, valid = x
        self.accumulator.seq_update(
            child_enc, weights * valid.astype(float), None if estimate is None else estimate.dist
        )

    def seq_update_engine(
        self,
        x: tuple[Any, np.ndarray, np.ndarray],
        weights: Any,
        estimate: TransformDistribution | None,
        engine: Any,
    ) -> None:
        """Engine-resident E-step: the validity-masked weights are formed on the active engine and
        the child accumulator is routed through the engine. Matches seq_update.
        """
        from mixle.stats.compute.backend import child_seq_update

        child_enc, _, valid = x
        w = engine.asarray(weights) * engine.asarray(np.asarray(valid, dtype=np.float64))
        child_seq_update(self.accumulator, child_enc, w, None if estimate is None else estimate.dist, engine)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize from one inverse-transformed observation when it is valid."""
        try:
            inv = self.transform.inverse(x)
        except Exception:
            return
        self.accumulator.initialize(inv, weight, rng)

    def seq_initialize(
        self, x: tuple[Any, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize from a validity-masked encoded batch."""
        child_enc, _, valid = x
        self.accumulator.seq_initialize(child_enc, weights * valid.astype(float), rng)

    def combine(self, suff_stat: Any) -> "TransformAccumulator":
        """Merge child sufficient statistics."""
        self.accumulator.combine(suff_stat)
        return self

    def value(self) -> Any:
        """Return the child accumulator's serialized sufficient statistics."""
        return self.accumulator.value()

    def from_value(self, x: Any) -> "TransformAccumulator":
        """Restore the child accumulator from serialized sufficient statistics."""
        self.accumulator.from_value(x)
        return self

    def scale(self, c: float) -> "TransformAccumulator":
        """Scale delegated sufficient statistics by ``c``."""
        self.accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic merging to the child accumulator."""
        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic replacement to the child accumulator."""
        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "TransformDataEncoder":
        """Return the encoder associated with this accumulator."""
        return TransformDataEncoder(
            self.accumulator.acc_to_encoder(), self.transform, density_correction=self.density_correction
        )


class TransformAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for TransformAccumulator."""

    def __init__(
        self,
        factory: StatisticAccumulatorFactory,
        transform: Any,
        density_correction: bool | None = None,
        name: str | None = None,
    ) -> None:
        self.factory = factory
        self.transform = transform
        self.density_correction = density_correction
        self.name = name

    def make(self) -> TransformAccumulator:
        """Create a fresh transform accumulator."""
        return TransformAccumulator(
            self.factory.make(), self.transform, density_correction=self.density_correction, name=self.name
        )


class TransformEstimator(ParameterEstimator):
    """Estimator for fixed-transform distributions."""

    def __init__(
        self,
        estimator: ParameterEstimator,
        transform: Any | None = None,
        density_correction: bool | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.transform = transform if transform is not None else IdentityTransform()
        self.density_correction = density_correction
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> TransformAccumulatorFactory:
        """Return the accumulator factory for inverse-transformed observations."""
        return TransformAccumulatorFactory(
            self.estimator.accumulator_factory(),
            self.transform,
            density_correction=self.density_correction,
            name=self.name,
        )

    def estimate(self, nobs: float | None, suff_stat: Any) -> TransformDistribution:
        """Estimate the child distribution and wrap it with the fixed transform."""
        return TransformDistribution(
            self.estimator.estimate(nobs, suff_stat),
            transform=self.transform,
            density_correction=self.density_correction,
            name=self.name,
            keys=self.keys,
        )


class TransformDataEncoder(DataSequenceEncoder):
    """Encode transformed observations as inverse child data plus Jacobian terms."""

    def __init__(self, encoder: DataSequenceEncoder, transform: Any, density_correction: bool | None = True) -> None:
        self.encoder = encoder
        self.transform = transform
        self.density_correction = density_correction is not False

    def __str__(self) -> str:
        return "TransformDataEncoder(encoder=%s, transform=%s, density_correction=%s)" % (
            repr(self.encoder),
            repr(self.transform),
            repr(self.density_correction),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TransformDataEncoder)
            and other.encoder == self.encoder
            and other.transform == self.transform
            and other.density_correction == self.density_correction
        )

    def seq_encode(self, x: Sequence[Any]) -> tuple[Any, np.ndarray, np.ndarray]:
        """Encode observations as inverse child values, Jacobians, and validity flags."""
        inv_values = []
        valid = np.ones(len(x), dtype=bool)
        log_jac = np.zeros(len(x), dtype=np.float64)
        fill = self.transform.invalid_inverse_value()

        for i, y in enumerate(x):
            try:
                inv_values.append(self.transform.inverse(y))
                if self.density_correction:
                    log_jac[i] = self.transform.log_abs_det_inverse_jacobian(y)
                    if not np.isfinite(log_jac[i]):
                        valid[i] = False
            except Exception:
                inv_values.append(fill)
                log_jac[i] = -np.inf
                valid[i] = False

        return self.encoder.seq_encode(inv_values), log_jac, valid
