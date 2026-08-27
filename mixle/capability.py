"""Capability-based feature detection for mixle objects.

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
for both flavours. All mixle imports are deferred into the ``check`` methods so this module stays a
dependency-free leaf that any layer can import without cycles.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CapabilityError",
    "CapabilityUnavailable",
    "Conditionable",
    "Marginalizable",
    "LatentStructured",
    "PosteriorPredictive",
    "EngineResidentEStep",
    "Transform",
    "SupportsBackendScoring",
    "SupportsBackendComponentScoring",
    "SupportsStackedBackend",
    "TemporalPointProcess",
    "PredicateCapability",
    "Enumerable",
    "FiniteSupport",
    "RankableByIndex",
    "Shardable",
    "ExponentialFamily",
    "ConjugateUpdatable",
    "ExactDensity",
    "SetValued",
    "HasCDF",
    "HasMoments",
    "HasEntropy",
    "Discrete",
    "Continuous",
    "Fittable",
    "Optimizable",
    "Neutral",
    "ALL_CAPABILITIES",
    "FACET_PRESERVING",
    "supports",
    "capabilities",
    "require",
    "intersect_capabilities",
    "top_k",
    "CapabilitySpec",
    "CAPABILITY_CATALOG",
    "catalog",
    "describe",
    "summarize",
    "what_supports",
    "render_catalog_markdown",
]


class CapabilityError(TypeError):
    """Raised by :func:`require` when an object lacks a needed capability."""


class CapabilityUnavailable(LookupError):
    """Typed, intentional decline from a capability implementation.

    Predicate implementations may raise this when the capability is genuinely unavailable. Other
    exceptions are implementation defects and deliberately propagate instead of being converted into
    a false "unsupported" answer.
    """


# ---------------------------------------------------------------------------
# Method-presence capabilities (structural protocols)
# ---------------------------------------------------------------------------
@runtime_checkable
class Conditionable(Protocol):
    """Supports conditioning on a subset of coordinates: ``condition(observed) -> distribution``."""

    def condition(self, observed: dict[int, float]) -> Any:
        """Return the conditional distribution after fixing the indexed coordinates in ``observed``."""
        ...


@runtime_checkable
class Marginalizable(Protocol):
    """Supports marginalising to a subset of coordinates: ``marginal(keep) -> distribution``."""

    def marginal(self, keep: Any) -> Any:
        """Return the distribution over the coordinates or fields identified by ``keep``."""
        ...


@runtime_checkable
class LatentStructured(Protocol):
    """Exposes an explicit latent posterior q(z|x): ``latent_posterior(x) -> LatentPosterior``."""

    def latent_posterior(self, x: Any) -> Any:
        """Return the inferred latent-state posterior for observation ``x``."""
        ...


@runtime_checkable
class PosteriorPredictive(Protocol):
    """Can draw/score new data conditioned on an observation's inferred latent state."""

    def posterior_predictive(self, *args: Any, **kwargs: Any) -> Any:
        """Return the posterior-predictive distribution or sample representation for the supplied context."""
        ...


@runtime_checkable
class EngineResidentEStep(Protocol):
    """An accumulator that can run its E-step on the active compute engine without leaving it."""

    def seq_update_engine(self, enc: Any, weights: Any, estimate: Any, engine: Any) -> None:
        """Update sufficient statistics from encoded data resident on ``engine``."""
        ...


@runtime_checkable
class Transform(Protocol):
    """A contract-declared invertible change of variables with a tractable Jacobian."""

    contract: Any

    def forward(self, x: Any) -> Any:
        """Map values from the base space into the transformed space."""
        ...

    def inverse(self, y: Any) -> Any:
        """Map values from the transformed space back into the base space."""
        ...

    def log_abs_det_inverse_jacobian(self, y: Any) -> float:
        """Return ``log |det d inverse(y) / dy|`` for density correction."""
        ...

    def invalid_inverse_value(self) -> Any:
        """Return an in-domain fill value used to encode rejected observations safely."""
        ...


@runtime_checkable
class SupportsBackendScoring(Protocol):
    """A distribution that can score a batch directly on the active engine (``backend_seq_log_density``)."""

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Return per-row log densities for encoded data already resident on ``engine``."""
        ...


@runtime_checkable
class SupportsBackendComponentScoring(Protocol):
    """A distribution that exposes per-component engine scoring (``backend_seq_component_log_density``)."""

    def backend_seq_component_log_density(self, x: Any, engine: Any) -> Any:
        """Return per-row, per-component log densities on ``engine``."""
        ...


@runtime_checkable
class SupportsStackedBackend(Protocol):
    """A component type that can score/parameterise a homogeneous stack on the engine."""

    def backend_stacked_params(self, dists: Any, engine: Any) -> Any:
        """Pack homogeneous component parameters into an engine-native stacked representation."""
        ...

    def backend_stacked_log_density(self, x: Any, params: Any, engine: Any) -> Any:
        """Score encoded data against a stacked parameter representation on ``engine``."""
        ...


@runtime_checkable
class TemporalPointProcess(Protocol):
    """An event-time point process exposing its conditional intensity and compensator.

    ``intensity(t, times, marks=None)`` is the conditional rate lambda(t) given the history; the
    univariate Hawkes / inhomogeneous-Poisson leaves return a float, the multivariate variant a
    per-mark vector. ``expected_count(t_start, t_end, times, marks=None)`` is the integrated intensity
    (compensator) over the window. (Birth-death is a population process, not an intensity-based point
    process, and is intentionally excluded.)
    """

    def intensity(self, t: float, times: Any, marks: Any = None) -> Any:
        """Return the conditional event intensity at time ``t`` given the observed history."""
        ...

    def expected_count(self, t_start: float, t_end: float, times: Any, marks: Any = None) -> Any:
        """Return the integrated intensity over ``[t_start, t_end]`` given the observed history."""
        ...


# ---------------------------------------------------------------------------
# Declaration / derived capabilities (predicate via classmethod ``check``)
# ---------------------------------------------------------------------------
class PredicateCapability:
    """Base for capabilities determined by inspecting the object rather than a method name."""

    @classmethod
    def check(cls, obj: Any) -> bool:  # pragma: no cover - overridden
        """Return whether ``obj`` satisfies this capability predicate."""
        raise NotImplementedError


class Enumerable(PredicateCapability):
    """Its support can be iterated in descending-probability order (``enumerator()`` works)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` exposes a supported enumerator."""
        from mixle.enumeration.algorithms import supports_enumeration

        if not callable(getattr(obj, "enumerator", None)):
            return False
        return bool(supports_enumeration(obj))


class FiniteSupport(PredicateCapability):
    """Has a finite number of distinct support points (``support_size()`` is an int)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` reports a finite, non-negative support size.

        A Boolean is not a support count even though ``bool`` subclasses ``int``: ``support_size()``
        returning ``True`` used to certify a one-point support, so an object answering a yes/no
        question was routed as an enumerable finite distribution.
        """
        fn = getattr(obj, "support_size", None)
        if not callable(fn):
            return False
        n = fn()
        return isinstance(n, int) and not isinstance(n, bool) and n >= 0


class RankableByIndex(PredicateCapability):
    """Supports random access by integer rank (unranking) via the count-budget seek index.

    Finite-support enumerables are rankable by index through the count-DP semiring; the implication
    edge ``finite ==> enumerable ==> rankable`` is exactly this composition. (Decomposable combinators
    over infinite-but-countable children are also rankable structurally; this conservative check
    covers the finite-leaf case used by the dispatch layers.)
    """

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` can be unranked by finite support index."""
        return Enumerable.check(obj) and FiniteSupport.check(obj)


class Shardable(PredicateCapability):
    """Declares a non-atomic model-parallel decomposition axis (``decomposition().is_shardable``).

    The low-cost string predicate gates planning; the rich :class:`~mixle.stats.compute.decomposition.Decomposition`
    is built only when a node is actually sharded.
    """

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` declares a shardable decomposition axis."""
        from mixle.stats.compute.decomposition import decomposition_for

        return bool(decomposition_for(obj).is_shardable)


class ExponentialFamily(PredicateCapability):
    """Declares an exponential-family form (generated stacked kernels + conjugate estimation)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` declares an exponential-family form."""
        from mixle.stats.compute.declarations import declaration_for

        decl = declaration_for(obj)
        return decl is not None and decl.exponential_family is not None


class ConjugateUpdatable(PredicateCapability):
    """Has a closed-form conjugate Bayesian update (the top tier of the inference ladder).

    ``supports(x, ConjugateUpdatable)`` is True iff ``conjugate_posterior(x, data)`` returns an exact
    closed-form posterior; otherwise Bayesian inference falls back to the numerical fitters
    (MAP / Laplace / MCMC / VI). Detection is the single ``conjugate_posterior`` registry.
    """

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` has a registered closed-form conjugate update."""
        from mixle.stats.bayes.conjugate import is_conjugate_family

        return bool(is_conjugate_family(obj))


class ExactDensity(PredicateCapability):
    """``log_density`` returns the exact ``log p(x)``, not a bound or an approximation.

    False for models whose ``log_density`` is a variational lower bound (e.g. LDA / LLDA return a
    per-document ELBO) or a plug-in / Monte-Carlo estimate (e.g. HDPM, IBP). ``supports(x, ExactDensity)``
    lets code that needs an exact likelihood ``require`` it instead of silently trusting a bound.
    Detection: ``x.density_semantics()`` (default :attr:`DensitySemantics.EXACT`).
    """

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` reports exact density semantics."""
        from mixle.stats.compute.pdist import DensitySemantics

        fn = getattr(obj, "density_semantics", None)
        if not callable(fn):
            return False
        return fn() is DensitySemantics.EXACT


class Neutral(PredicateCapability):
    """The identity / no-op element of a combinator (a Null distribution, accumulator, or encoder)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` is a recognized null or identity value."""
        from mixle.stats.combinator.null_dist import (
            NullAccumulator,
            NullDataEncoder,
            NullDistribution,
        )

        return isinstance(obj, (NullDistribution, NullAccumulator, NullDataEncoder))


class SetValued(PredicateCapability):
    """A distribution over sets with forced/required membership (``required`` / ``num_required``)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` exposes set-valued membership metadata."""
        return hasattr(obj, "num_required") and hasattr(obj, "required")


class HasCDF(PredicateCapability):
    """Exposes an exact cumulative distribution function and its inverse (``cdf`` and ``quantile``).

    The CDF/quantile pair is what lets a continuous family answer probability-ordered / rank queries
    through the inverse transform (the continuous analogue of enumeration).
    """

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` provides both CDF and quantile operations."""
        return callable(getattr(obj, "cdf", None)) and callable(getattr(obj, "quantile", None))


class HasMoments(PredicateCapability):
    """Exposes closed-form moments (``mean`` and ``variance`` at least; optionally skewness/kurtosis)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` provides at least mean and variance methods."""
        return callable(getattr(obj, "mean", None)) and callable(getattr(obj, "variance", None))


class HasEntropy(PredicateCapability):
    """Exposes a closed-form (differential or Shannon) ``entropy()`` in nats."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` exposes an entropy method."""
        return callable(getattr(obj, "entropy", None))


class Discrete(PredicateCapability):
    """Countable support: the distribution is enumerable or has explicit finite support."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` has countable or explicitly finite support."""
        return FiniteSupport.check(obj) or Enumerable.check(obj)


class Continuous(PredicateCapability):
    """Continuous support: has a CDF/quantile but no countable (enumerable) support."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` looks continuous by capability predicates."""
        return HasCDF.check(obj) and not Enumerable.check(obj)


class Fittable(PredicateCapability):
    """Can be fit from data: exposes an ``estimator()`` (the M-step / MLE entry point)."""

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` can provide an estimator."""
        return callable(getattr(obj, "estimator", None))


class Optimizable(PredicateCapability):
    """Exposes a ranked objective over a structured space -- the ``Relation`` surface (``solve`` / ``top``).

    Optimization-as-distribution: the object can return its optimum (``solve``) and the k-best members
    (``top`` / ``enumerator``), so optimization is a first-class, capability-gated operation rather than
    a property of a specific class.
    """

    @classmethod
    def check(cls, obj: Any) -> bool:
        """Return whether ``obj`` exposes solve and top-k optimization methods."""
        return callable(getattr(obj, "solve", None)) and callable(getattr(obj, "top", None))


# All capability facets named in docs/ABSTRACTIONS.md are now formalised. ConjugateUpdatable and
# TemporalPointProcess (which previously needed a family-surface unification) are detectable above;
# the DOE Surrogate/Acquisition contracts live in doe/_contracts.py. (The PDE forward-operator and
# dynamics-operator contracts moved out with the mixle-pde plugin.)


# Capabilities that apply to distributions (iterated by ``capabilities(dist)``). EngineResidentEStep,
# EncodedDataHandle, EMStrategy and the Transform/backend protocols apply to other object kinds and are
# queried directly with ``supports(obj, Cap)`` rather than listed here.
ALL_CAPABILITIES: tuple[type, ...]

# Capabilities a decomposable combinator inherits from ALL of its children (the capability algebra,
# generalising ``intersect_engine_ready``). Exp-family / conditionable are intentionally absent: a
# combinator over exponential-family children is generally not itself exponential-family.
FACET_PRESERVING: tuple[type, ...] = (Enumerable, FiniteSupport, RankableByIndex)


# ---------------------------------------------------------------------------
# Query surface
# ---------------------------------------------------------------------------
_PROTOCOL_METHODS: dict[type, tuple[str, ...]] = {}


def _protocol_methods(capability: type) -> tuple[str, ...]:
    """Names a ``Protocol`` declares as *methods* -- what an implementer must actually be able to call."""
    cached = _PROTOCOL_METHODS.get(capability)
    if cached is None:
        names: dict[str, None] = {}
        for klass in getattr(capability, "__mro__", ()):
            if klass is object or klass is Protocol:
                continue
            for name, member in vars(klass).items():
                if not name.startswith("_") and inspect.isfunction(member):
                    names[name] = None
        cached = tuple(names)
        _PROTOCOL_METHODS[capability] = cached
    return cached


def _resolve_capability(capability: Any, caller: str) -> type:
    """Resolve ``capability`` -- the capability type itself, or its catalogued name string.

    :func:`capabilities` and :func:`describe` report capabilities as NAME STRINGS, so the query
    surface must accept its own vocabulary back: ``supports(x, "HasMoments")`` used to leak a raw
    ``isinstance() arg 2 must be a type`` TypeError for exactly the strings the library itself emits
    (t2/MXR wave-3). A name string resolves to the class of the same name in this module; a
    catalogued name with no queryable class here (a core contract / subsystem role such as
    ``"Relation"``) and an unknown name each get an error naming the string and what to pass instead.
    """
    if isinstance(capability, type):
        return capability
    if isinstance(capability, str):
        resolved = globals().get(capability)
        if isinstance(resolved, type):
            return resolved
        spec = _CATALOG_BY_NAME.get(capability)
        if spec is not None:
            raise TypeError(
                "%s(): %r is catalogued as a %s backed by %s and has no queryable capability class in "
                "mixle.capability -- import the contract from %s and pass the type itself."
                % (caller, capability, spec.kind, spec.backed_by, spec.home)
            )
        known = ", ".join(sorted(s.name for s in CAPABILITY_CATALOG if isinstance(globals().get(s.name), type)))
        raise TypeError(
            "%s(): unknown capability name %r -- pass a capability type (e.g. mixle.capability.HasMoments) "
            "or one of the catalogued names mixle.capabilities()/mixle.describe() report: %s."
            % (caller, capability, known)
        )
    raise TypeError(
        "%s(): capability must be a capability type or its catalogued name string, got %s."
        % (caller, type(capability).__name__)
    )


def supports(obj: Any, capability: type | str) -> bool:
    """Return whether ``obj`` provides ``capability`` (a Protocol, a PredicateCapability, or its name).

    ``capability`` may be the capability class itself or the catalogued NAME STRING that
    :func:`capabilities` / :func:`describe` report (``supports(x, "HasMoments")``), so the query
    surface round-trips its own vocabulary.

    For a ``runtime_checkable`` ``Protocol``, ``isinstance`` establishes attribute *presence* only,
    not callable shape -- an object with ``condition = 3`` cleared ``supports(obj, Conditionable)``
    and therefore ``require()``, and the routing failure surfaced later as a ``TypeError`` inside the
    operation. These are authoritative dispatch declarations, so every declared protocol method must
    at least be callable on ``obj``.
    """
    capability = _resolve_capability(capability, "supports")
    try:
        if isinstance(capability, type) and issubclass(capability, PredicateCapability):
            if isinstance(obj, type):
                return False
            return bool(capability.check(obj))
    except CapabilityUnavailable:
        return False
    if not isinstance(obj, capability):
        return False
    if not getattr(capability, "_is_protocol", False):
        return True
    return all(callable(getattr(obj, name, None)) for name in _protocol_methods(capability))


def capabilities(obj: Any) -> frozenset[str]:
    """Return the set of capability names ``obj`` provides — "what can I do with this?"."""
    return frozenset(cap.__name__ for cap in ALL_CAPABILITIES if supports(obj, cap))


def require(obj: Any, capability: type | str, op: str | None = None) -> None:
    """Raise :class:`CapabilityError` (early, with a clear message) if ``obj`` lacks ``capability``.

    Accepts the capability class or its catalogued name string, exactly like :func:`supports`.
    """
    capability = _resolve_capability(capability, "require")
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

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}")
    require(dist, Enumerable, "top_k")
    return list(islice(dist.enumerator(), k))


# ---------------------------------------------------------------------------
# The capability catalog — one legible place that answers "what can mixle do?"
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapabilitySpec:
    """One row of the capability vocabulary: what it means, what backs it, where it lives."""

    name: str
    summary: str  # one line, plain English
    kind: str  # "distribution facet" | "object contract" | "core contract" | "subsystem role"
    backed_by: str  # the method / registry / ABC that backs detection
    home: str  # the module that defines the protocol/ABC


# The single source of truth for the whole capability vocabulary, across every subsystem. Pure data
# (no imports), so it never drifts and can be rendered to docs or introspected at runtime.
CAPABILITY_CATALOG: tuple[CapabilitySpec, ...] = (
    # --- core contracts (the cast every distribution implements) ---
    CapabilitySpec(
        "Distribution",
        "score · sample · estimate",
        "core contract",
        "log_density/sampler/estimator (ABC)",
        "mixle.stats.compute.pdist",
    ),
    CapabilitySpec(
        "Sampler", "draw observations", "core contract", "DistributionSampler.sample (ABC)", "mixle.stats.compute.pdist"
    ),
    CapabilitySpec(
        "Estimator",
        "fit parameters from data (M-step)",
        "core contract",
        "ParameterEstimator.estimate (ABC)",
        "mixle.stats.compute.pdist",
    ),
    CapabilitySpec(
        "Enumerator",
        "k-best descending-probability iteration",
        "core contract",
        "DistributionEnumerator (ABC)",
        "mixle.stats.compute.pdist",
    ),
    # --- distribution facets (detect with capabilities(dist) / supports(dist, X)) ---
    CapabilitySpec(
        "Enumerable",
        "iterate the support in descending probability",
        "distribution facet",
        "enumerator()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "FiniteSupport", "a finite number of support points", "distribution facet", "support_size()", "mixle.capability"
    ),
    CapabilitySpec(
        "RankableByIndex",
        "random access / unrank the support by integer rank",
        "distribution facet",
        "count_budget_index()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Shardable",
        "declares a model-parallel decomposition axis (split across devices)",
        "distribution facet",
        "decomposition()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "ExponentialFamily",
        "canonical exp-family form; generated numpy/torch kernels",
        "distribution facet",
        "compute_declaration().exponential_family",
        "mixle.capability",
    ),
    CapabilitySpec(
        "ConjugateUpdatable",
        "closed-form conjugate Bayesian posterior",
        "distribution facet",
        "conjugate_posterior registry",
        "mixle.capability",
    ),
    CapabilitySpec(
        "ExactDensity",
        "log_density is the exact log p(x), not an ELBO / approximation",
        "distribution facet",
        "density_semantics()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Conditionable", "condition on a subset of coordinates", "distribution facet", "condition()", "mixle.capability"
    ),
    CapabilitySpec(
        "Marginalizable",
        "marginalise to a subset of coordinates",
        "distribution facet",
        "marginal()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "LatentStructured",
        "expose q(z|x), the latent posterior + posterior-predictive",
        "distribution facet",
        "latent_posterior()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "PosteriorPredictive",
        "sample/score new data from inferred latent state",
        "distribution facet",
        "posterior_predictive()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "TemporalPointProcess",
        "conditional intensity λ(t) + compensator",
        "distribution facet",
        "intensity()/expected_count()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "SetValued",
        "distribution over sets with forced membership",
        "distribution facet",
        "required/num_required",
        "mixle.capability",
    ),
    CapabilitySpec(
        "HasCDF",
        "exact cumulative distribution + inverse (rank queries via inverse transform)",
        "distribution facet",
        "cdf() and quantile()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "HasMoments",
        "closed-form moments (mean / variance / …)",
        "distribution facet",
        "mean() and variance()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "HasEntropy",
        "closed-form entropy in nats",
        "distribution facet",
        "entropy()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Discrete",
        "countable support (enumerable or finite)",
        "distribution facet",
        "FiniteSupport or Enumerable",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Continuous",
        "continuous support (has a CDF but no countable support)",
        "distribution facet",
        "HasCDF and not Enumerable",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Fittable",
        "can be fit from data (exposes an estimator / M-step)",
        "distribution facet",
        "estimator()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Optimizable",
        "ranked objective over a structured space (optimization-as-distribution)",
        "distribution facet",
        "solve() and top()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Neutral",
        "the identity / no-op element of a combinator",
        "distribution facet",
        "isinstance Null*",
        "mixle.capability",
    ),
    CapabilitySpec(
        "SupportsBackendScoring",
        "score a batch directly on the active engine",
        "distribution facet",
        "backend_seq_log_density()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "SupportsBackendComponentScoring",
        "score every latent component directly on the active engine",
        "distribution facet",
        "backend_seq_component_log_density()",
        "mixle.capability",
    ),
    # --- object contracts (non-distribution roles) ---
    CapabilitySpec(
        "EngineResidentEStep",
        "run the E-step on the engine without leaving it",
        "object contract",
        "seq_update_engine()",
        "mixle.capability",
    ),
    CapabilitySpec(
        "Transform",
        "invertible change of variables with a Jacobian",
        "object contract",
        "forward/inverse/log_abs_det",
        "mixle.capability",
    ),
    CapabilitySpec(
        "SupportsStackedBackend",
        "score a homogeneous component stack on the engine",
        "object contract",
        "backend_stacked_*",
        "mixle.capability",
    ),
    CapabilitySpec(
        "EncodedFold",
        "fold the E-step over distributed/streaming data",
        "object contract",
        "pysp_seq_* methods",
        "mixle.utils.parallel.planner",
    ),
    CapabilitySpec(
        "EMStrategy", "an EM-step strategy", "object contract", "step() -> EMStepResult", "mixle.inference.em"
    ),
    CapabilitySpec(
        "Relation",
        "optimisation-as-distribution over a constrained space",
        "subsystem role",
        "Relation (ABC): enumerate/solve/top/sample",
        "mixle.relations",
    ),
    CapabilitySpec(
        "ComputeEngine",
        "a numpy/torch/symbolic backend (REQUIRED_OPS)",
        "subsystem role",
        "ComputeEngine (ABC)",
        "mixle.engines.base",
    ),
    CapabilitySpec(
        "DecomposableSemiring",
        "a semiring for structural count/enumeration DP",
        "subsystem role",
        "DecomposableSemiring (ABC)",
        "mixle.enumeration.quantization",
    ),
    CapabilitySpec(
        "Surrogate",
        "a fit/predict surrogate for Bayesian optimisation",
        "subsystem role",
        "Surrogate protocol",
        "mixle.doe._contracts",
    ),
    CapabilitySpec(
        "Acquisition",
        "a BO acquisition function (EI/PI/UCB)",
        "subsystem role",
        "Acquisition protocol + register_acquisition",
        "mixle.doe._contracts",
    ),
    CapabilitySpec(
        "BeliefTrackable",
        "a weighted hypothesis portfolio with open-world mass, reweighted by evidence",
        "subsystem role",
        "HypothesisPortfolio.reweight()",
        "mixle.epistemic",
    ),
)

_catalog_names = [spec.name for spec in CAPABILITY_CATALOG]
if len(set(_catalog_names)) != len(_catalog_names):
    raise RuntimeError("CAPABILITY_CATALOG contains duplicate names")
_CATALOG_BY_NAME = {spec.name: spec for spec in CAPABILITY_CATALOG}
ALL_CAPABILITIES = tuple(
    globals()[spec.name]
    for spec in CAPABILITY_CATALOG
    if spec.kind == "distribution facet" and isinstance(globals().get(spec.name), type)
)
_unresolved_distribution_facets = [
    spec.name
    for spec in CAPABILITY_CATALOG
    if spec.kind == "distribution facet" and not isinstance(globals().get(spec.name), type)
]
if _unresolved_distribution_facets:
    raise RuntimeError("CAPABILITY_CATALOG has unresolved distribution facets: %r" % _unresolved_distribution_facets)
# the "interesting" facets to report as present/absent in describe()
_HIGHLIGHT = (
    "Enumerable",
    "RankableByIndex",
    "Conditionable",
    "Marginalizable",
    "LatentStructured",
    "ExponentialFamily",
    "ConjugateUpdatable",
    "TemporalPointProcess",
    "SetValued",
)


def catalog() -> tuple[CapabilitySpec, ...]:
    """Return the full capability vocabulary (every facet/contract/role), as data."""
    return CAPABILITY_CATALOG


def _category(have: frozenset[str], obj: Any = None) -> str:
    if "SetValued" in have:
        return "set-valued distribution"
    if "TemporalPointProcess" in have:
        return "temporal point process"
    if "FiniteSupport" in have:
        return "discrete distribution (finite support)"
    if "Enumerable" in have:
        return "discrete distribution (countable support)"
    if "LatentStructured" in have:
        return "latent-variable model"
    if obj is not None and callable(getattr(obj, "edges", None)):
        return "graphical model (factored joint over record fields)"
    return "distribution"


def describe(obj: Any) -> str:
    """Return a plain-English summary of what ``obj`` is and what you can do with it.

    The one-call answer to "what can this do?": its category, the capabilities it has and notably
    lacks, the engines it runs on, and how to fit it. Works on any object; richest for distributions.
    """
    # The Model facade intentionally hides log_density/sampler/estimator behind its own verbs
    # (m(x), m.sample), so duck-typing below cannot see it -- and describe() is the entry point the
    # top-level docstring routes newcomers through, so the FIRST object they hold (propose()'s
    # return) must not come back "no catalogued capability detected". Model already knows how to
    # explain itself. Function-scope import, mirroring lifecycle's own lazy import of this module.
    from mixle.lifecycle import Model as _LifecycleModel

    if isinstance(obj, _LifecycleModel):
        return obj.explain()
    # An estimator instance has the same dispatch gap: describe the family it estimates.
    if (
        not isinstance(obj, type)
        and callable(getattr(obj, "estimate", None))
        and callable(getattr(obj, "accumulator_factory", None))
    ):
        return (
            "%s — an estimator: fit it with mixle.inference.fit(data, %s(...)) or optimize(data, ...); the fitted result is the distribution it estimates."
            % (
                type(obj).__name__,
                type(obj).__name__,
            )
        )
    # A class (e.g. a model class) or any non-distribution object is described by its catalogued
    # capabilities only — the rich distribution view needs a live instance.
    is_instance = not isinstance(obj, type)
    name = obj.__name__ if isinstance(obj, type) else type(obj).__name__
    # The rich distribution view covers anything that can SCORE plus either sample or estimate.
    # Requiring an estimator() too blanked describe() on the flagship optimize() default -- the
    # structure-learned HeterogeneousBayesianNetwork scores and samples but carries no estimator,
    # and reported "no catalogued capability detected" (t5 wave-3). ``can`` below states only the
    # operations the object actually exposes.
    can_score = is_instance and hasattr(obj, "log_density")
    can_estimate = is_instance and callable(getattr(obj, "estimator", None))
    can_sample = is_instance and callable(getattr(obj, "sampler", None))
    is_dist = can_score and (can_estimate or can_sample)
    if not is_dist:
        have = sorted(s.name for s in CAPABILITY_CATALOG if _safe_supports(obj, s.name))
        base = "%s — %s" % (name, ("supports: " + " · ".join(have)) if have else "no catalogued capability detected")
        # A mixle.ppl model knows how it will be fit; surface the auto-selected inference route and caveats
        # without introducing a dependency on mixle.ppl.
        explain = getattr(obj, "explain_fit", None)
        if callable(explain):
            try:
                plan = explain()
                lines = [base, "  fit route: %s — %s" % (plan["route"], plan["reason"])]
                lines += ["             · %s" % c for c in plan.get("caveats", [])]
                return "\n".join(lines)
            except Exception:  # noqa: BLE001
                pass
        return base

    have = capabilities(obj)
    from mixle.stats.compute.pdist import DensitySemantics

    semantics = obj.density_semantics() if hasattr(obj, "density_semantics") else None
    can = (
        ["score"]
        + (["estimate"] if can_estimate else [])
        + ([] if semantics is DensitySemantics.LIKELIHOOD_FACTOR or not can_sample else ["sample"])
        + sorted(have & set(_HIGHLIGHT))
        + sorted(c for c in ("SupportsBackendScoring", "PosteriorPredictive") if c in have)
    )
    lines = ["%s — %s." % (name, _category(have, obj))]
    lines.append("  can:       " + " · ".join(can))
    if "ExactDensity" not in have and hasattr(obj, "density_semantics"):  # flag non-exact densities
        label = {
            DensitySemantics.LOWER_BOUND: "variational lower bound (ELBO)",
            DensitySemantics.UPPER_BOUND: "upper bound",
            DensitySemantics.ESTIMATE: "plug-in / stochastic estimate",
            DensitySemantics.LIKELIHOOD_FACTOR: "non-generative likelihood factor",
        }.get(obj.density_semantics(), "approximation")
        lines.append("  density:   log_density is a %s, NOT exact log p(x)" % label)
    engines = obj.supported_engines() if hasattr(obj, "supported_engines") else None
    if engines:
        lines.append("  engines:   " + ", ".join(engines))
    if "ConjugateUpdatable" in have:
        lines.append("  inference: closed-form conjugate Bayes, or numerical (MAP/Laplace/MCMC/VI)")
    elif can_estimate:
        lines.append("  inference: numerical (MAP/Laplace/MCMC/VI) — no closed-form conjugate prior")
    else:
        # No estimator: this is a fitted artifact -- advertising MAP/MCMC refit routes it cannot run
        # would be wrong; say how it is actually (re)fit instead.
        lines.append("  inference: a fitted artifact (no estimator) — refit from data with mixle.inference.optimize")
    if callable(getattr(obj, "describe", None)):
        lines.append("  more:      .describe() returns this object's own structured fit report")
    missing = [c for c in _HIGHLIGHT if c not in have]
    if missing:
        lines.append("  cannot:    " + " · ".join(missing))
    return "\n".join(lines)


_FINITE_SUMMARY_CAP = 100_000  # exact finite-support summation cap (density_rank's max_exact convention)


def _finite_support_table(obj: Any) -> tuple[tuple[list, list] | None, str | None]:
    """Exact ``(values, probabilities)`` over a small finite enumerable support, else ``(None, why)``.

    A finite weighted sum over the whole support IS a closed form: it is exact, not sampled or
    truncated. Only genuinely finite-and-enumerable objects qualify, and only up to
    ``_FINITE_SUMMARY_CAP`` support points (past that the receipt says so instead of stalling).
    """
    import math
    from itertools import islice

    try:
        finite = FiniteSupport.check(obj) and Enumerable.check(obj)
    except Exception:  # noqa: BLE001 - a broken predicate is "not derivable", not a summarize crash
        finite = False
    if not finite:
        return None, "no closed-form methods and no finite enumerable support to sum exactly"
    n = obj.support_size()
    if n == 0:
        return None, "the support is empty"
    if n > _FINITE_SUMMARY_CAP:
        return None, "finite support is too large to sum exactly (support_size=%d > %d)" % (n, _FINITE_SUMMARY_CAP)
    try:
        items = list(islice(obj.enumerator(), n))
    except Exception as exc:  # noqa: BLE001 - enumeration declined; report why instead of raising
        return None, "enumerating the support failed: %s: %s" % (type(exc).__name__, exc)
    values = [v for v, _ in items]
    probs = [math.exp(float(lp)) for _, lp in items]
    return (values, probs), None


def _mixture_moment_values(obj: Any) -> tuple[tuple[Any, Any] | None, str | None]:
    """Closed-form (mean, variance) of a finite mixture from its component moments, else ``(None, why)``.

    ``mean = sum_k w_k m_k`` and the law of total variance ``var = sum_k w_k (v_k + m_k^2) - mean^2``
    are elementary closed forms for ANY finite mixture whose components expose closed-form moments --
    a fitted Gaussian mixture returning an empty summary contradicted the docstring's "not available
    in closed form" promise (t5/t4 wave-3). Returns ``(None, None)`` for a non-mixture.
    """
    import numpy as np

    from mixle.stats.latent.mixture import MixtureDistribution

    if not isinstance(obj, MixtureDistribution):
        return None, None
    bad = sorted({type(c).__name__ for c in obj.components if not supports(c, HasMoments)})
    if bad:
        return None, "mixture component(s) %s expose no closed-form mean()/variance()" % ", ".join(bad)
    try:
        w = np.asarray(obj.w, dtype=float)
        means = [np.asarray(c.mean(), dtype=float) for c in obj.components]
        variances = [np.asarray(c.variance(), dtype=float) for c in obj.components]
    except Exception as exc:  # noqa: BLE001 - a component moment failed; report why
        return None, "a mixture component moment failed: %s: %s" % (type(exc).__name__, exc)
    mean_ndim = means[0].ndim
    var_ndim = variances[0].ndim
    if any(m.ndim != mean_ndim for m in means) or any(v.ndim != var_ndim for v in variances):
        return None, "mixture components report moments of differing dimension"
    if mean_ndim == 0 and var_ndim == 0:
        mean = float(sum(wk * float(mk) for wk, mk in zip(w, means)))
        second = float(sum(wk * (float(vk) + float(mk) ** 2) for wk, mk, vk in zip(w, means, variances)))
        return (mean, second - mean**2), None
    if mean_ndim == 1 and var_ndim == 1:  # diagonal/per-coordinate variances
        mean = sum(wk * mk for wk, mk in zip(w, means))
        second = sum(wk * (vk + mk**2) for wk, mk, vk in zip(w, means, variances))
        return (mean, second - mean**2), None
    if mean_ndim == 1 and var_ndim == 2:  # full component covariances
        mean = sum(wk * mk for wk, mk in zip(w, means))
        cov = sum(wk * (vk + np.outer(mk, mk)) for wk, mk, vk in zip(w, means, variances)) - np.outer(mean, mean)
        return (mean, cov), None
    return None, "mixture component moments have an unrecognized shape combination"


def summarize(obj: Any) -> dict[str, Any]:
    """Return every closed-form summary statistic ``obj`` exposes, selected by its capabilities.

    The numeric companion to :func:`describe` (which says *what* an object can do): mean/variance/std
    (plus skewness/kurtosis when present) for a ``HasMoments`` distribution, ``entropy`` for
    ``HasEntropy``, and the median for ``HasCDF``. Two closed forms are DERIVED when the methods are
    absent: a finite mixture's mean/variance from its component moments (law of total variance), and
    exact mean/variance/std/entropy/median/mode for a small finite enumerable support by summing the
    whole support. Keys absent from the result are not available in closed form for ``obj`` -- and
    the ``_status`` receipt names each core statistic that could not be produced and why -- so
    ``summarize`` never raises on a partially-featured distribution and never returns a bare empty
    report.
    """
    import numpy as np

    out: dict[str, Any] = {}
    status: dict[str, dict[str, Any]] = {}

    def capture(name: str, operation: Any) -> Any:
        try:
            value = operation()
        except Exception as exc:  # noqa: BLE001 - the report records each broken advertised method
            status[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            return None
        try:
            array = np.asarray(value)
        except Exception as exc:  # noqa: BLE001 - a moment with no rectangular array form
            # `np.asarray` sat OUTSIDE the try, so a RAGGED moment -- per-component means of
            # differing dimension, which a mixture over unlike components legitimately produces --
            # made the whole report raise, contradicting the "never raises" contract two lines up
            # and losing every other statistic that had already been computed (MXR-080-1903).
            # The value is still reported: it is what the object's advertised method returned, and
            # a report that drops it is less honest than one that says it could not be measured.
            # Its shape/finiteness are deliberately NOT reported -- there is no array to measure.
            status[name] = {"status": "unmeasurable", "error": f"{type(exc).__name__}: {exc}"}
            out[name] = value
            return value
        if array.ndim == 0:
            value = array.item()
        elif isinstance(value, np.ndarray):
            value = value.copy()
        finite = True
        if np.issubdtype(array.dtype, np.number):
            finite = bool(np.isfinite(array).all())
        status[name] = {"status": "available" if finite else "invalid", "shape": list(array.shape)}
        out[name] = value
        return value

    import numbers

    finite_table = finite_reason = None
    need_finite = not (supports(obj, HasMoments) and supports(obj, HasEntropy) and supports(obj, HasCDF))
    if need_finite:
        finite_table, finite_reason = _finite_support_table(obj)
    numeric_values = probabilities = None
    if finite_table is not None:
        values, probs = finite_table
        probabilities = probs
        if all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in values):
            numeric_values = [float(v) for v in values]

    if supports(obj, HasMoments):
        capture("mean", obj.mean)
        variance = capture("variance", obj.variance)
        if variance is not None:
            capture("std", lambda: np.sqrt(variance))
        else:
            status["std"] = {"status": "unavailable", "reason": "variance failed"}
        if callable(getattr(obj, "skewness", None)):
            capture("skewness", obj.skewness)
        if callable(getattr(obj, "kurtosis", None)):
            capture("kurtosis", obj.kurtosis)
    else:
        moments, moment_reason = _mixture_moment_values(obj)
        if moments is not None:
            mixture_mean, mixture_variance = moments
            capture("mean", lambda: mixture_mean)
            variance = capture("variance", lambda: mixture_variance)
            if variance is not None:
                arr = np.asarray(variance, dtype=float)
                # std from a full covariance is the sqrt of its DIAGONAL, not an elementwise sqrt
                capture("std", lambda: np.sqrt(np.diag(arr)) if arr.ndim == 2 else np.sqrt(arr))
        elif numeric_values is not None:
            mean = float(np.dot(probabilities, numeric_values))
            second = float(np.dot(probabilities, np.square(numeric_values)))
            capture("mean", lambda: mean)
            capture("variance", lambda: second - mean**2)
            capture("std", lambda: np.sqrt(max(second - mean**2, 0.0)))
        else:
            if moment_reason is None and finite_table is not None:
                moment_reason = "support values are not numeric; mean/variance are undefined for labels"
            reason = moment_reason or finite_reason or "no closed form is available"
            for name in ("mean", "variance", "std"):
                status[name] = {"status": "unavailable", "reason": reason}

    if supports(obj, HasEntropy):
        capture("entropy", obj.entropy)
    elif probabilities is not None:
        entropy = -float(sum(q * np.log(q) for q in probabilities if q > 0.0))
        capture("entropy", lambda: entropy)
    else:
        status["entropy"] = {
            "status": "unavailable",
            "reason": finite_reason or "no entropy() and no finite enumerable support to sum exactly",
        }

    if supports(obj, HasCDF):
        capture("median", lambda: obj.quantile(0.5))
    elif numeric_values is not None:
        order = np.argsort(numeric_values)
        cumulative = np.cumsum(np.asarray(probabilities, dtype=float)[order])
        median = numeric_values[int(order[int(np.searchsorted(cumulative, 0.5))])]
        capture("median", lambda: median)
    else:
        reason = finite_reason or (
            "support values are not numeric; a median needs an ordered support"
            if finite_table is not None
            else "no cdf()/quantile() pair and no finite numeric support"
        )
        status["median"] = {"status": "unavailable", "reason": reason}

    if callable(getattr(obj, "mode", None)):
        capture("mode", obj.mode)
    elif finite_table is not None and finite_table[0]:
        capture("mode", lambda: finite_table[0][0])  # the enumerator streams in descending probability

    out["_status"] = status
    return out


def _safe_supports(obj: Any, cap_name: str) -> bool:
    cap = globals().get(cap_name)
    if cap is None or not isinstance(cap, type):
        return False
    return supports(obj, cap)


def what_supports(capability: type, among: Any) -> list[str]:
    """Return the names of entries in ``among`` that provide ``capability``.

    ``among`` is an iterable of instances (or classes for method-presence protocols) — e.g.
    ``what_supports(Conditionable, [mvn, gaussian, mixture])``.
    """
    out = []
    for obj in among:
        if supports(obj, capability):
            out.append(getattr(obj, "__name__", type(obj).__name__))
    return out


def render_catalog_markdown() -> str:
    """Render :data:`CAPABILITY_CATALOG` as a markdown table (the source for docs/CAPABILITIES.md)."""
    rows = ["| Capability | What it means | Kind | Backed by | Home |", "|---|---|---|---|---|"]
    for s in CAPABILITY_CATALOG:
        rows.append("| `%s` | %s | %s | `%s` | `%s` |" % (s.name, s.summary, s.kind, s.backed_by, s.home))
    return "\n".join(rows)
