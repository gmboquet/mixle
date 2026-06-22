"""Capability-based feature detection for pysp objects.

Determine what an object *supports* rather than what class it *is*, so callers can dispatch on
behaviour instead of concrete class names. This generalises the pattern the engine layer already
uses (``DistributionCapabilities.supports_engine`` / ``intersect_engine_ready``) to every capability
facet in the type hierarchy (see ``docs/ABSTRACTIONS.md``).

Two flavours of capability:

* **Method-presence** capabilities are ``runtime_checkable`` ``Protocol``s, so they work with
  ``isinstance`` and with static typing/autocomplete: ``Conditionable``, ``Marginalizable``,
  ``LatentStructured``, ``PosteriorPredictive``, ``EngineResidentEStep``.
* **Declaration / derived** capabilities cannot be expressed by a method name (they inspect the
  declaration or compose other capabilities); they subclass ``PredicateCapability`` and carry a
  ``check(obj)`` classmethod: ``Enumerable``, ``FiniteSupport``, ``RankableByIndex``,
  ``ExponentialFamily``, ``Neutral``.

``supports(obj, cap)`` / ``capabilities(obj)`` / ``require(obj, cap)`` are the uniform query surface
for both flavours. All pysp imports are deferred into the ``check`` methods so this module stays a
dependency-free leaf that any layer can import without cycles.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CapabilityError",
    "Conditionable",
    "Marginalizable",
    "LatentStructured",
    "PosteriorPredictive",
    "EngineResidentEStep",
    "Transform",
    "SupportsBackendScoring",
    "SupportsBackendComponentScoring",
    "SupportsStackedBackend",
    "PredicateCapability",
    "Enumerable",
    "FiniteSupport",
    "RankableByIndex",
    "ExponentialFamily",
    "SetValued",
    "Neutral",
    "ALL_CAPABILITIES",
    "FACET_PRESERVING",
    "supports",
    "capabilities",
    "require",
    "intersect_capabilities",
    "top_k",
]


class CapabilityError(TypeError):
    """Raised by :func:`require` when an object lacks a needed capability."""


# ---------------------------------------------------------------------------
# Method-presence capabilities (structural protocols)
# ---------------------------------------------------------------------------
@runtime_checkable
class Conditionable(Protocol):
    """Supports conditioning on a subset of coordinates: ``condition(observed) -> distribution``."""

    def condition(self, observed: dict[int, float]) -> Any: ...


@runtime_checkable
class Marginalizable(Protocol):
    """Supports marginalising to a subset of coordinates: ``marginal(keep) -> distribution``."""

    def marginal(self, keep: Any) -> Any: ...


@runtime_checkable
class LatentStructured(Protocol):
    """Exposes an explicit latent posterior q(z|x): ``latent_posterior(x) -> LatentPosterior``."""

    def latent_posterior(self, x: Any) -> Any: ...


@runtime_checkable
class PosteriorPredictive(Protocol):
    """Can draw/score new data conditioned on an observation's inferred latent state."""

    def posterior_predictive(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class EngineResidentEStep(Protocol):
    """An accumulator that can run its E-step on the active compute engine without leaving it."""

    def seq_update_engine(self, enc: Any, weights: Any, estimate: Any, engine: Any) -> None: ...


@runtime_checkable
class Transform(Protocol):
    """An invertible change of variables with a tractable Jacobian (combinator/transform.py)."""

    def forward(self, x: Any) -> Any: ...
    def inverse(self, y: Any) -> Any: ...
    def log_abs_det_inverse_jacobian(self, y: Any) -> float: ...


@runtime_checkable
class SupportsBackendScoring(Protocol):
    """A distribution that can score a batch directly on the active engine (``backend_seq_log_density``)."""

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any: ...


@runtime_checkable
class SupportsBackendComponentScoring(Protocol):
    """A distribution that exposes per-component engine scoring (``backend_seq_component_log_density``)."""

    def backend_seq_component_log_density(self, x: Any, engine: Any) -> Any: ...


@runtime_checkable
class SupportsStackedBackend(Protocol):
    """A component type that can score/parameterise a homogeneous stack on the engine."""

    def backend_stacked_params(self, dists: Any, engine: Any) -> Any: ...
    def backend_stacked_log_density(self, x: Any, params: Any, engine: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Declaration / derived capabilities (predicate via classmethod ``check``)
# ---------------------------------------------------------------------------
class PredicateCapability:
    """Base for capabilities determined by inspecting the object rather than a method name."""

    @classmethod
    def check(cls, obj: Any) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError


class Enumerable(PredicateCapability):
    """Its support can be iterated in descending-probability order (``enumerator()`` works)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        from pysp.utils.enumeration import supports_enumeration

        try:
            return bool(supports_enumeration(obj))
        except Exception:
            return False


class FiniteSupport(PredicateCapability):
    """Has a finite number of distinct support points (``support_size()`` is an int)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        fn = getattr(obj, "support_size", None)
        if not callable(fn):
            return False
        try:
            n = fn()
        except Exception:
            return False
        return isinstance(n, int) and n >= 0


class RankableByIndex(PredicateCapability):
    """Supports random access by integer rank (unranking) via the count-budget seek index.

    Finite-support enumerables are rankable by index through the count-DP semiring; the implication
    edge ``finite ==> enumerable ==> rankable`` is exactly this composition. (Decomposable combinators
    over infinite-but-countable children are also rankable structurally; this conservative check
    covers the finite-leaf case used by the dispatch layers.)
    """

    @classmethod
    def check(cls, obj: Any) -> bool:
        return Enumerable.check(obj) and FiniteSupport.check(obj)


class ExponentialFamily(PredicateCapability):
    """Declares an exponential-family form (generated stacked kernels + conjugate estimation)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        from pysp.stats.compute.declarations import declaration_for

        decl = declaration_for(obj)
        return decl is not None and decl.exponential_family is not None


class Neutral(PredicateCapability):
    """The identity / no-op element of a combinator (a Null distribution, accumulator, or encoder)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        from pysp.stats.combinator.null_dist import (
            NullAccumulator,
            NullDataEncoder,
            NullDistribution,
        )

        return isinstance(obj, (NullDistribution, NullAccumulator, NullDataEncoder))


class SetValued(PredicateCapability):
    """A distribution over sets with forced/required membership (``required`` / ``num_required``)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        return hasattr(obj, "num_required") and hasattr(obj, "required")


# Deferred capabilities — named in docs/ABSTRACTIONS.md but NOT yet reliably detectable, because the
# underlying families do not share a method surface. Formalising these is a family-surface unification
# (a refactor), not just adding a Protocol here:
#   * TemporalPointProcess — the point-process leaves (hawkes/inhomogeneous_poisson/birth_death) expose
#     no common intensity()/compensator() method; give them one first.
#   * ConjugateUpdatable — the conjugate-prior surface (set_prior / has_conj_prior / pseudo_count) is
#     inconsistent across leaves; unify it before adding the capability.
# (The PDE forward operator is free functions in ppl/pde_solve.py; ppl/dynamics.DynamicsOperator is
# already a formal ABC.)


# Capabilities that apply to distributions (iterated by ``capabilities(dist)``). EngineResidentEStep,
# EncodedDataHandle, EMStrategy and the Transform/backend protocols apply to other object kinds and are
# queried directly with ``supports(obj, Cap)`` rather than listed here.
ALL_CAPABILITIES: tuple[type, ...] = (
    Conditionable,
    Marginalizable,
    LatentStructured,
    PosteriorPredictive,
    Enumerable,
    FiniteSupport,
    RankableByIndex,
    ExponentialFamily,
    SetValued,
    SupportsBackendScoring,
    SupportsBackendComponentScoring,
    Neutral,
)

# Capabilities a decomposable combinator inherits from ALL of its children (the capability algebra,
# generalising ``intersect_engine_ready``). Exp-family / conditionable are intentionally absent: a
# combinator over exponential-family children is generally not itself exponential-family.
FACET_PRESERVING: tuple[type, ...] = (Enumerable, FiniteSupport, RankableByIndex)


# ---------------------------------------------------------------------------
# Query surface
# ---------------------------------------------------------------------------
def supports(obj: Any, capability: type) -> bool:
    """Return whether ``obj`` provides ``capability`` (a Protocol or a PredicateCapability)."""
    if isinstance(capability, type) and issubclass(capability, PredicateCapability):
        return bool(capability.check(obj))
    return isinstance(obj, capability)


def capabilities(obj: Any) -> frozenset[str]:
    """Return the set of capability names ``obj`` provides — "what can I do with this?"."""
    return frozenset(cap.__name__ for cap in ALL_CAPABILITIES if supports(obj, cap))


def require(obj: Any, capability: type, op: str | None = None) -> None:
    """Raise :class:`CapabilityError` (early, with a clear message) if ``obj`` lacks ``capability``."""
    if not supports(obj, capability):
        msg = "%s does not support %s" % (type(obj).__name__, capability.__name__)
        if op:
            msg += " (needed for %s)" % op
        raise CapabilityError(msg)


def intersect_capabilities(children: Any, caps: tuple[type, ...] = FACET_PRESERVING) -> frozenset[str]:
    """Capabilities shared by EVERY child — what a decomposable combinator over them preserves."""
    children = tuple(children)
    if not children:
        return frozenset(cap.__name__ for cap in caps)
    return frozenset(cap.__name__ for cap in caps if all(supports(child, cap) for child in children))


# ---------------------------------------------------------------------------
# A generic, capability-tiered algorithm — written once, works on any implementer
# ---------------------------------------------------------------------------
def top_k(dist: Any, k: int) -> list[tuple[Any, float]]:
    """Return the ``k`` highest-probability ``(value, log_prob)`` pairs.

    Dispatches on capability, not class: a :class:`RankableByIndex` distribution could random-access
    by rank, but every :class:`Enumerable` distribution can stream its support in descending
    probability — so one implementation covers all of them, including user-defined ones. Raises a
    clear :class:`CapabilityError` for distributions that support neither.
    """
    from itertools import islice

    require(dist, Enumerable, "top_k")
    return list(islice(dist.enumerator(), k))
