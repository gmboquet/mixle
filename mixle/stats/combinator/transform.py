"""Invertible-transform wrappers for sequence-encodable distributions.

The module implements identity, affine, exponential, logit, and custom
transforms plus the distribution, sampler, accumulator, and encoder plumbing
needed to apply them inside Mixle combinators.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.algorithms import freeze
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
from mixle.utils.exact import require_exact_bool


class TransformDomainError(ValueError):
    """Raised when an observation lies outside a transform's declared domain."""


class TransformCompatibilityError(TypeError):
    """Raised when a transform contract is incompatible with its child distribution."""


@dataclass(frozen=True)
class TransformContract:
    """Machine-checkable declaration for a deterministic bijective transform.

    ``input_domain`` and ``output_domain`` use the same coarse vocabulary as
    compute-declaration support tags. ``measure`` is ``"adaptive"`` when the
    transform supports both counting measure and continuous change-of-variables,
    or ``"discrete"`` for a counting-measure-only mapping.
    """

    input_domain: str
    output_domain: str
    measure: str = "adaptive"
    bijective: bool = True

    def __post_init__(self) -> None:
        if not self.input_domain or not self.output_domain:
            raise ValueError("transform contract domains must be non-empty strings.")
        if self.measure not in {"adaptive", "discrete"}:
            raise ValueError("transform contract measure must be 'adaptive' or 'discrete'.")
        if self.bijective is not True:
            raise ValueError("TransformDistribution requires a contract declaring a bijection.")


class TransformFitReceipt(NamedTuple):
    """Accepted/rejected evidence accounting attached to a fitted transform wrapper."""

    accepted_weight: float
    rejected_weight: float
    rejected_fraction: float


class TransformStatistics(NamedTuple):
    """Versioned child statistics plus transform-domain evidence accounting."""

    schema_version: int
    child: Any
    accepted_weight: float
    rejected_weight: float


#: Declared supports that are subsets of the reals, and so are covered by a transform whose input
#: domain is ``"real"``. Membership is a containment claim about the support, not a list of families:
#: ``unit_interval`` belongs here for exactly the reason ``unit_interval_open`` does -- [0, 1] is no
#: less a subset of R than (0, 1) -- but was omitted, so every Beta-supported child was refused by any
#: real-domain transform with "child support 'unit_interval' is not covered by transform input domain
#: 'real'", a composition that is mathematically fine. ``nonnegative_real`` is the unpunctuated
#: spelling some declarations use for ``non_negative_real``; both name the same set.
_REAL_SUPPORTS = frozenset(
    {
        "real",
        "bounded_real",
        "positive",
        "positive_real",
        "non_negative_real",
        "nonnegative_real",
        "positive_tail",
        "unit_interval",
        "unit_interval_open",
        "boolean",
        "bounded_integer",
        "non_negative_integer",
        "positive_integer",
    }
)
_POSITIVE_SUPPORTS = frozenset({"positive", "positive_real", "positive_tail", "positive_integer"})
_FINITE_SUPPORTS = frozenset(
    {
        "boolean",
        "bounded_integer",
        "finite_hashable_set",
        "finite_integer_sequence",
        "finite_integer_set",
        "finite_or_default_hashable",
        "fixed_atom",
        "permutation",
    }
)


def _finite_weight(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a finite non-negative real number." % name) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be a finite non-negative real number." % name)
    return result


def _validate_statistics(value: Any) -> TransformStatistics:
    if not isinstance(value, TransformStatistics) or value.schema_version != 1:
        raise ValueError("transform statistics must use schema version 1.")
    return TransformStatistics(
        1,
        value.child,
        _finite_weight(value.accepted_weight, name="accepted_weight"),
        _finite_weight(value.rejected_weight, name="rejected_weight"),
    )


def _enumerator_or_none(dist: SequenceEncodableProbabilityDistribution) -> DistributionEnumerator | None:
    try:
        return dist.enumerator()
    except EnumerationError:
        return None


def _uses_density_correction(
    dist: SequenceEncodableProbabilityDistribution,
    density_correction: bool | None,
) -> bool:
    if density_correction is not None:
        return require_exact_bool(density_correction, "density_correction")
    return _enumerator_or_none(dist) is None


def _same_value(left: Any, right: Any) -> bool:
    try:
        if freeze(left) == freeze(right):
            return True
    except TypeError:
        pass
    try:
        equal = left == right
        return bool(np.all(equal)) if isinstance(equal, np.ndarray) else bool(equal)
    except (TypeError, ValueError):
        return False


def _support_matches(domain: str, support: str) -> bool:
    if domain == "any":
        return True
    if domain == "real":
        return support in _REAL_SUPPORTS
    if domain == "positive_real":
        return support in _POSITIVE_SUPPORTS
    return domain == support


def _validate_transform_object(transform: Any) -> TransformContract:
    contract = getattr(transform, "contract", None)
    if not isinstance(contract, TransformContract):
        raise TransformCompatibilityError(
            "transform must expose a TransformContract as .contract; arbitrary duck-typed transforms are not accepted."
        )
    required = ("forward", "inverse", "log_abs_det_inverse_jacobian", "invalid_inverse_value")
    missing = [name for name in required if not callable(getattr(transform, name, None))]
    if missing:
        raise TransformCompatibilityError("transform contract is missing callable methods: %s." % ", ".join(missing))
    return contract


def _validate_finite_support(transform: Any, child_iter: DistributionEnumerator) -> None:
    seen: set[Any] = set()
    for count, (value, _) in enumerate(child_iter, start=1):
        if count > 100_000:
            raise TransformCompatibilityError(
                "finite transform validation exceeded 100000 support values; use a declared support-domain contract."
            )
        try:
            output = transform.forward(value)
            inverse = transform.inverse(output)
        except Exception as exc:
            raise TransformCompatibilityError(
                "transform is not defined over the child's enumerable support value %r." % (value,)
            ) from exc
        if not _same_value(value, inverse):
            raise TransformCompatibilityError(
                "transform inverse does not recover child support value %r from output %r." % (value, output)
            )
        try:
            key = freeze(output)
        except TypeError as exc:
            raise TransformCompatibilityError("transformed support values must have stable enumeration keys.") from exc
        if key in seen:
            raise TransformCompatibilityError("transform is not injective over the child's enumerable support.")
        seen.add(key)


def _validate_transform_child(
    dist: SequenceEncodableProbabilityDistribution,
    transform: Any,
    density_correction: bool,
) -> TransformContract:
    from mixle.stats.compute.declarations import declaration_for

    contract = _validate_transform_object(transform)
    declaration = declaration_for(dist)
    support = None if declaration is None else declaration.support
    child_iter = _enumerator_or_none(dist)
    if child_iter is not None:
        if density_correction:
            raise TransformCompatibilityError(
                "enumerable children use counting measure and cannot request a Jacobian density correction."
            )
        if support in _FINITE_SUPPORTS or (
            contract.input_domain != "any"
            and not (support is not None and _support_matches(contract.input_domain, support))
        ):
            _validate_finite_support(transform, child_iter)
        return contract
    if contract.measure == "discrete":
        raise TransformCompatibilityError("a discrete transform contract requires an enumerable child distribution.")
    if not density_correction:
        raise TransformCompatibilityError(
            "non-enumerable children require Jacobian density correction under an adaptive transform contract."
        )
    if contract.input_domain != "any" and (support is None or not _support_matches(contract.input_domain, support)):
        raise TransformCompatibilityError(
            "child support %r is not covered by transform input domain %r." % (support, contract.input_domain)
        )
    return contract


def _inverse_with_jacobian(transform: Any, value: Any, density_correction: bool) -> tuple[Any, float]:
    inverse = transform.inverse(value)
    log_jac = transform.log_abs_det_inverse_jacobian(value) if density_correction else 0.0
    if density_correction and not np.isfinite(log_jac):
        raise TransformDomainError("transform inverse Jacobian must be finite.")
    return inverse, float(log_jac)


class IdentityTransform:
    """Identity transform y = x."""

    contract = TransformContract("any", "same")

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

    contract = TransformContract("real", "real")

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        if not np.isfinite(loc):
            raise ValueError("AffineTransform requires finite loc.")
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

    contract = TransformContract("real", "positive_real")

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
            raise TransformDomainError("ExpTransform inverse requires y > 0.")
        return math.log(y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return the log inverse-Jacobian correction for ``log(y)``."""
        if y <= 0.0:
            raise TransformDomainError("ExpTransform inverse requires y > 0.")
        return -math.log(y)

    def invalid_inverse_value(self) -> float:
        """Return a safe child-space fill value for invalid inverses."""
        return 0.0


class LogTransform:
    """Log transform y = log(x), mapping positive x to real y."""

    contract = TransformContract("positive_real", "real")

    def __str__(self) -> str:
        return "LogTransform()"

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogTransform)

    def forward(self, x: Any) -> Any:
        """Map a positive child value to the real line."""
        if x <= 0.0:
            raise TransformDomainError("LogTransform forward requires x > 0.")
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

    contract = TransformContract("real", "unit_interval_open")

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
            raise TransformDomainError("LogitTransform inverse requires 0 < y < 1.")
        return math.log(y) - math.log1p(-y)

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return the log inverse-Jacobian correction for the logit map."""
        if y <= 0.0 or y >= 1.0:
            raise TransformDomainError("LogitTransform inverse requires 0 < y < 1.")
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
        fit_receipt: TransformFitReceipt | None = None,
    ) -> None:
        self.dist = dist
        self.transform = transform if transform is not None else IdentityTransform()
        self.density_correction = _uses_density_correction(dist, density_correction)
        _validate_transform_child(
            self.dist,
            self.transform,
            self.density_correction,
        )
        self.name = name
        self.keys = keys
        self.fit_receipt = fit_receipt

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
        output_support = self.transform.contract.output_domain
        if output_support == "same":
            output_support = "transformed" if child is None else child.support
        return DistributionDeclaration(
            name="transform",
            distribution_type=type(self),
            parameters=(),
            statistics=(
                StatisticSpec("schema_version", kind="scalar", additive=False, scales=False),
                StatisticSpec("base", kind="child_stat"),
                StatisticSpec("accepted_weight", kind="scalar"),
                StatisticSpec("rejected_weight", kind="scalar"),
            ),
            support=output_support,
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
            inv, log_jac = _inverse_with_jacobian(self.transform, x, self.density_correction)
        except TransformDomainError:
            return -np.inf
        return self.dist.log_density(inv) + log_jac

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
        self.seen: set[Any] = set()

    def __next__(self) -> tuple[Any, float]:
        v, lp = next(self.child_iter)
        output = self.dist.transform.forward(v)
        inverse = self.dist.transform.inverse(output)
        if not _same_value(v, inverse):
            raise TransformCompatibilityError("transform inverse failed during support enumeration.")
        key = freeze(output)
        if key in self.seen:
            raise TransformCompatibilityError("transform produced a duplicate enumerable outcome.")
        self.seen.add(key)
        return output, lp


class TransformSampler(DistributionSampler):
    """Sampler that transforms draws from the child distribution."""

    def __init__(self, dist: TransformDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.child_sampler = dist.dist.sampler(seed=self.new_seed())

    def sample(self, size: int | None = None, *, batched: bool = True):
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
        keys: str | None = None,
    ) -> None:
        self.accumulator = accumulator
        self.transform = transform
        self.density_correction = density_correction
        self.name = name
        self.keys = keys
        self.accepted_weight = 0.0
        self.rejected_weight = 0.0

    def update(self, x: Any, weight: float, estimate: TransformDistribution | None) -> None:
        """Accumulate one inverse-transformed observation when it is valid."""
        checked = _finite_weight(weight, name="weight")
        try:
            inv, _ = _inverse_with_jacobian(self.transform, x, self.density_correction is not False)
        except TransformDomainError:
            self.rejected_weight += checked
            return
        self.accumulator.update(inv, checked, None if estimate is None else estimate.dist)
        self.accepted_weight += checked

    def seq_update(
        self, x: tuple[Any, np.ndarray, np.ndarray], weights: np.ndarray, estimate: TransformDistribution | None
    ) -> None:
        """Accumulate a batch using validity-masked child weights."""
        child_enc, _, valid = x
        valid = np.asarray(valid)
        checked = np.asarray(weights, dtype=np.float64)
        if (
            valid.dtype != np.bool_
            or valid.ndim != 1
            or checked.shape != valid.shape
            or np.any(~np.isfinite(checked))
            or np.any(checked < 0.0)
        ):
            raise ValueError("transform validity mask and weights must be aligned, finite, and non-negative.")
        accepted = checked * valid
        self.accumulator.seq_update(child_enc, accepted, None if estimate is None else estimate.dist)
        self.accepted_weight += float(accepted.sum())
        self.rejected_weight += float(checked[~valid].sum())

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
        valid = np.asarray(valid)
        checked = np.asarray(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            dtype=np.float64,
        )
        if (
            valid.dtype != np.bool_
            or valid.ndim != 1
            or checked.shape != valid.shape
            or np.any(~np.isfinite(checked))
            or np.any(checked < 0.0)
        ):
            raise ValueError("transform validity mask and weights must be aligned, finite, and non-negative.")
        accepted = checked * valid
        w = engine.asarray(accepted)
        child_seq_update(self.accumulator, child_enc, w, None if estimate is None else estimate.dist, engine)
        self.accepted_weight += float(accepted.sum())
        self.rejected_weight += float(checked[~valid].sum())

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize from one inverse-transformed observation when it is valid."""
        checked = _finite_weight(weight, name="weight")
        try:
            inv, _ = _inverse_with_jacobian(self.transform, x, self.density_correction is not False)
        except TransformDomainError:
            self.rejected_weight += checked
            return
        self.accumulator.initialize(inv, checked, rng)
        self.accepted_weight += checked

    def seq_initialize(
        self, x: tuple[Any, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize from a validity-masked encoded batch."""
        child_enc, _, valid = x
        valid = np.asarray(valid)
        checked = np.asarray(weights, dtype=np.float64)
        if (
            valid.dtype != np.bool_
            or valid.ndim != 1
            or checked.shape != valid.shape
            or np.any(~np.isfinite(checked))
            or np.any(checked < 0.0)
        ):
            raise ValueError("transform validity mask and weights must be aligned, finite, and non-negative.")
        accepted = checked * valid
        self.accumulator.seq_initialize(child_enc, accepted, rng)
        self.accepted_weight += float(accepted.sum())
        self.rejected_weight += float(checked[~valid].sum())

    def combine(self, suff_stat: TransformStatistics) -> "TransformAccumulator":
        """Merge child sufficient statistics and evidence accounting."""
        checked = _validate_statistics(suff_stat)
        self.accumulator.combine(checked.child)
        self.accepted_weight += checked.accepted_weight
        self.rejected_weight += checked.rejected_weight
        return self

    def value(self) -> TransformStatistics:
        """Return versioned child statistics and transform-domain evidence accounting."""
        return TransformStatistics(
            1,
            self.accumulator.value(),
            self.accepted_weight,
            self.rejected_weight,
        )

    def from_value(self, x: TransformStatistics) -> "TransformAccumulator":
        """Restore child statistics and evidence accounting."""
        checked = _validate_statistics(x)
        self.accumulator.from_value(checked.child)
        self.accepted_weight = checked.accepted_weight
        self.rejected_weight = checked.rejected_weight
        return self

    def scale(self, c: float) -> "TransformAccumulator":
        """Scale delegated sufficient statistics and evidence weights by ``c``."""
        checked = _finite_weight(c, name="scale")
        self.accumulator.scale(checked)
        self.accepted_weight *= checked
        self.rejected_weight *= checked
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Pool the wrapper's full child statistics under its own key, or delegate when unkeyed."""
        if self.keys is None:
            self.accumulator.key_merge(stats_dict)
            return
        if self.keys in stats_dict:
            self.combine(stats_dict[self.keys])
        stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace from the wrapper pool, or delegate when this wrapper is unkeyed."""
        if self.keys is None:
            self.accumulator.key_replace(stats_dict)
        elif self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

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
        keys: str | None = None,
    ) -> None:
        self.factory = factory
        self.transform = transform
        self.density_correction = density_correction
        self.name = name
        self.keys = keys

    def make(self) -> TransformAccumulator:
        """Create a fresh transform accumulator."""
        return TransformAccumulator(
            self.factory.make(),
            self.transform,
            density_correction=self.density_correction,
            name=self.name,
            keys=self.keys,
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
        _validate_transform_object(self.transform)
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
            keys=self.keys,
        )

    def estimate(self, nobs: float | None, suff_stat: Any) -> TransformDistribution:
        """Estimate from accepted evidence and attach the rejected-weight receipt."""
        checked = _validate_statistics(suff_stat)
        if checked.accepted_weight <= 0.0:
            raise ValueError("transform estimation requires positive accepted weight.")
        total = checked.accepted_weight + checked.rejected_weight
        return TransformDistribution(
            self.estimator.estimate(checked.accepted_weight, checked.child),
            transform=self.transform,
            density_correction=self.density_correction,
            name=self.name,
            keys=self.keys,
            fit_receipt=TransformFitReceipt(
                checked.accepted_weight,
                checked.rejected_weight,
                checked.rejected_weight / total if total else 0.0,
            ),
        )


class TransformDataEncoder(DataSequenceEncoder):
    """Encode transformed observations as inverse child data plus Jacobian terms."""

    def __init__(self, encoder: DataSequenceEncoder, transform: Any, density_correction: bool | None = True) -> None:
        self.encoder = encoder
        self.transform = transform
        _validate_transform_object(self.transform)
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
                inverse, log_jac[i] = _inverse_with_jacobian(
                    self.transform,
                    y,
                    self.density_correction,
                )
                inv_values.append(inverse)
            except TransformDomainError:
                inv_values.append(fill)
                log_jac[i] = -np.inf
                valid[i] = False

        return self.encoder.seq_encode(inv_values), log_jac, valid
