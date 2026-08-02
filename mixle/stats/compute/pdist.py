"""Defines abstract classes for SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator,
ProbabilityDistribution, StatisticAccumulator, StatisticAccumulatorFactory, DataSequenceEncoder, ParameterEstimator,
ConditionalSampler, and DistributionSampler for classes of the mixle.stats.

"""

import hashlib
import itertools
import math
import operator
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

import numpy as np

from mixle.engines.arithmetic import *

SS = TypeVar("SS")


def _positive_budget(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a positive integer." % name)
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("%s must be a positive integer." % name) from exc
    if result <= 0:
        raise ValueError("%s must be a positive integer." % name)
    return result


def _density_index(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("q must be a finite real number in [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("q must be a finite real number in [0, 1].") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("q must be a finite real number in [0, 1].")
    return result


def _invalid_score_policy(value: Any) -> str:
    if value not in ("raise", "omit"):
        raise ValueError("invalid_score must be 'raise' or 'omit'.")
    return value


class DensitySemantics(Enum):
    """What a distribution's ``log_density`` returns relative to the true log-density.

    The default contract is :attr:`EXACT`; override ``density_semantics()`` on models whose
    ``log_density`` is a variational bound or an approximation, so callers can tell an exact likelihood
    from one (e.g. LDA's per-document ELBO). Surfaced via the ``ExactDensity`` capability and ``describe``.
    """

    EXACT = "exact"  # the true log p(x)
    LOWER_BOUND = "lower_bound"  # value <= true log p(x); e.g. a variational ELBO
    UPPER_BOUND = "upper_bound"  # value >= true log p(x)
    ESTIMATE = "estimate"  # an approximation with no guaranteed direction (plug-in / Monte Carlo)
    LIKELIHOOD_FACTOR = "likelihood_factor"  # exact score factor, but not a normalized generative law


def join_density_semantics(semantics) -> "DensitySemantics":
    """Combine child density semantics for a combinator whose log_density is monotone in its children.

    A combinator whose score rises with each child's log_density -- a mixture's ``logsumexp``, a
    composite's sum -- inherits: a lower bound if any child is a lower bound, an upper bound if any is
    an upper bound, exactness only if all children are exact, and an undirected ``ESTIMATE`` if bounds
    of both directions (or any estimate) are mixed.
    """
    kinds = set(semantics)
    if DensitySemantics.LIKELIHOOD_FACTOR in kinds:
        return DensitySemantics.LIKELIHOOD_FACTOR
    has_lower = DensitySemantics.LOWER_BOUND in kinds
    has_upper = DensitySemantics.UPPER_BOUND in kinds
    if DensitySemantics.ESTIMATE in kinds or (has_lower and has_upper):
        return DensitySemantics.ESTIMATE
    if has_lower:
        return DensitySemantics.LOWER_BOUND
    if has_upper:
        return DensitySemantics.UPPER_BOUND
    return DensitySemantics.EXACT


@dataclass(frozen=True)
class FitProvenance:
    """How a fitted distribution was produced -- the estimator-side half of MXR-080-1190/1202.

    ``density_semantics()`` already tells a caller whether a score is an exact density, a bound, or an
    unnormalized factor. It says nothing about *fitting*: a `GaussianDistribution` returned by
    :func:`~mixle.inference.estimation.optimize` used to be byte-identical to one written by hand, so a
    consumer could not tell a converged fit from one that hit the iteration cap, nor learn that a
    covariance was repaired on the way. This record carries that, and :meth:`is_approximate` folds it
    into the one question most callers actually ask.

    ``None`` from :meth:`ProbabilityDistribution.fit_provenance` means the object was constructed
    directly and no fit produced it -- which is different from a fit that produced no receipt.

    Attributes:
        algorithm: the fitting procedure, e.g. ``"em"``, ``"block-em"``, ``"fused-em"``.
        estimator: class name of the :class:`ParameterEstimator` that drove it.
        objective: which objective was maximized -- ``"mle"``, ``"map"``, ``"vb"``.
        iterations: iterations actually run (not the cap).
        max_iterations: the cap that was in force.
        converged: True only if the objective gain fell below ``delta`` before the cap. False means the
            run stopped at ``max_iterations`` with the gain still above it -- an unconverged fit.
        delta: the convergence threshold, or ``None`` when the caller asked for a fixed iteration count.
        final_objective: the objective value of the returned model.
        objective_gain: the last accepted improvement; compare against ``delta``.
        n_observations: rows the fit consumed.
        repairs: numerical repairs applied during the fit, e.g. ``("min_covar-clamped",)``. Empty means
            none were recorded -- by a fit that records them.
        seed: the seed governing any stochastic initialization, when one was set.
    """

    algorithm: str
    estimator: str
    objective: str
    iterations: int
    max_iterations: int
    converged: bool
    delta: float | None = None
    final_objective: float | None = None
    objective_gain: float | None = None
    n_observations: int | None = None
    repairs: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        """Reject a receipt that describes a fit that cannot have happened.

        A provenance record exists to be trusted about the run it describes, so it has to be
        internally consistent: 99 iterations under a cap of 5, a negative iteration count, or an
        anonymous algorithm are not descriptions of anything (MXR-080-1190/1202). Validating here
        means a malformed receipt fails where it is built rather than being believed downstream.
        """
        for field_name in ("algorithm", "estimator", "objective"):
            text = getattr(self, field_name)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"FitProvenance {field_name} must be a non-empty name, got {text!r}")
        for field_name in ("iterations", "max_iterations"):
            count = getattr(self, field_name)
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 0:
                raise ValueError(f"FitProvenance {field_name} must be a non-negative integer, got {count!r}")
            # Canonicalize to a builtin int. An np.int64 passed validation and then made the
            # advertised JSON-compatible as_dict() raise inside json.dumps, so the receipt claimed a
            # serializability it did not have (MXR-080-1190/1202).
            object.__setattr__(self, field_name, int(count))
        if int(self.iterations) > int(self.max_iterations):
            raise ValueError(
                f"FitProvenance iterations ({self.iterations}) cannot exceed max_iterations "
                f"({self.max_iterations}); a run cannot outlast its own cap"
            )
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError(f"FitProvenance converged must be a boolean verdict, got {self.converged!r}")
        object.__setattr__(self, "converged", bool(self.converged))
        if self.converged and int(self.iterations) == 0:
            raise ValueError("FitProvenance cannot report convergence after zero iterations")
        # Optional measurements were checked for VALUE but not for TYPE, so a receipt could carry a
        # string or a list where a number belongs and only fail much later, if at all (MXR-080-1190/1202).
        for field_name in ("delta", "final_objective", "objective_gain"):
            measured = getattr(self, field_name)
            if measured is None:
                continue
            if isinstance(measured, (bool, np.bool_)) or not isinstance(measured, (int, float, np.number)):
                raise TypeError(f"FitProvenance {field_name} must be a real number when set, got {measured!r}")
            object.__setattr__(self, field_name, float(measured))
        for field_name in ("n_observations", "seed"):
            count = getattr(self, field_name)
            if count is None:
                continue
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
                raise TypeError(f"FitProvenance {field_name} must be an exact integer when set, got {count!r}")
            object.__setattr__(self, field_name, int(count))
        if self.delta is not None and (not math.isfinite(self.delta) or self.delta < 0.0):
            raise ValueError(f"FitProvenance delta must be finite and non-negative when set, got {self.delta!r}")
        if self.n_observations is not None and self.n_observations < 0:
            raise ValueError(f"FitProvenance n_observations must be non-negative, got {self.n_observations!r}")
        if not isinstance(self.repairs, (tuple, list)) or isinstance(self.repairs, (str, bytes)):
            raise TypeError(f"FitProvenance repairs must be a sequence of names, got {self.repairs!r}")
        # The first EM iteration has no baseline, so its gain is legitimately infinite, and a run over
        # an impossible model can leave the objective at -inf. Those are real observations but they
        # are not strict JSON (the library emits `allow_nan=False`, MXR-080-1762), and a receipt that
        # cannot serialize is a receipt that gets dropped. `None` -- "not a finite measurement" -- is
        # both true and portable, and keeps the in-memory and round-tripped records identical.
        for field_name in ("final_objective", "objective_gain"):
            measured = getattr(self, field_name)
            if measured is not None and not math.isfinite(float(measured)):
                object.__setattr__(self, field_name, None)
        object.__setattr__(self, "repairs", tuple(str(entry) for entry in self.repairs))

    def is_approximate(self) -> bool:
        """Whether this fit is a stopped-early or repaired approximation rather than a clean optimum.

        True when the run hit its iteration cap without converging, or when a numerical repair changed
        the parameters. Both mean the returned law is not simply "the estimator's answer on this data".
        """
        return (not self.converged) or bool(self.repairs)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FitProvenance":
        """Rebuild a receipt from :meth:`as_dict` output, re-validating every field.

        Fields are passed through as they were recorded rather than coerced. The coercions here used
        to be lossy in exactly the direction that matters: ``bool("false")`` is ``True``, so a
        persisted record whose convergence verdict had been corrupted to a string was rebuilt as a
        CONVERGED fit, and ``int("3")``/``float("nan")`` accepted text where a measurement belongs
        (MXR-080-1190/1202). ``__post_init__`` now type-checks each field, so handing it the recorded
        value makes a malformed record fail loudly instead of being silently reinterpreted.
        """
        if not isinstance(value, Mapping):
            raise TypeError(f"FitProvenance.from_dict expects a mapping, got {type(value).__name__}")
        missing = [
            key
            for key in ("algorithm", "estimator", "objective", "iterations", "max_iterations", "converged")
            if key not in value
        ]
        if missing:
            raise ValueError(f"FitProvenance record is missing required field(s): {missing}")
        return cls(
            algorithm=value["algorithm"],
            estimator=value["estimator"],
            objective=value["objective"],
            iterations=value["iterations"],
            max_iterations=value["max_iterations"],
            converged=value["converged"],
            delta=value.get("delta"),
            final_objective=value.get("final_objective"),
            objective_gain=value.get("objective_gain"),
            n_observations=value.get("n_observations"),
            repairs=tuple(value.get("repairs", ())),
            seed=value.get("seed"),
        )

    def as_dict(self) -> dict[str, Any]:
        """A JSON-compatible record, for provenance ledgers and receipts."""
        return {
            "algorithm": self.algorithm,
            "estimator": self.estimator,
            "objective": self.objective,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "converged": self.converged,
            "delta": self.delta,
            "final_objective": self.final_objective,
            "objective_gain": self.objective_gain,
            "n_observations": self.n_observations,
            "repairs": list(self.repairs),
            "seed": self.seed,
            "approximate": self.is_approximate(),
        }


class EnumerationError(NotImplementedError):
    """Raised when a distribution (or a child of a combinator) cannot enumerate its support.

    The path argument identifies the offending child within a combinator, e.g.
    'CompositeDistribution.dists[1]'.
    """

    def __init__(self, dist: Any, path: str = "", reason: str = "") -> None:
        self.leaf = dist
        self.path = path
        self.reason = reason
        msg = "%s does not support enumeration" % type(dist).__name__
        if path:
            msg = "%s -> %s" % (path, msg)
        if reason:
            msg += ": %s" % reason
        super().__init__(msg)


class KeyValidationError(ValueError):
    """Raised when keyed sufficient-statistic sites are incompatible.

    A key denotes an equality constraint across model sites.  Sites sharing a
    key must therefore have the same accumulator family and compatible estimator
    settings before their sufficient statistics are pooled.
    """

    pass


class ContractError(ValueError):
    """Raised when malformed data trips a combinator's ``seq_encode``/``estimate`` contract boundary.

    Mirrors :class:`EnumerationError`'s path-composition convention: ``path`` names the full field
    path through nested combinators (composed with ``" -> "``, outermost first), e.g.
    ``"CompositeDistribution.dists[2] -> SequenceDistribution.entries"``. Every ``ContractError``
    also carries what was expected, what actually arrived, and (when there is one) a concrete,
    non-generic suggestion for the likely fix -- so a caller reading only the message, not the
    traceback, knows what to change.
    """

    def __init__(self, path: str, expected: str, actual: str, fix: str = "") -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        self.fix = fix
        msg = "%s: expected %s, got %s." % (path, expected, actual)
        if fix:
            msg += " Fix: %s" % fix
        super().__init__(msg)


def prefix_contract_error(prefix: str, err: "ContractError") -> "ContractError":
    """Return a new ContractError with ``prefix`` prepended to ``err``'s field path.

    Used by a combinator to annotate a ``ContractError`` raised deep inside a child's
    ``seq_encode``/``estimate`` with the outer field position, so the final message names the
    FULL path down to where the failure actually occurred (e.g. a mixture-of-composites-of-
    sequences error names every level, not just the outermost combinator).
    """
    new_path = "%s -> %s" % (prefix, err.path) if err.path else prefix
    return ContractError(new_path, err.expected, err.actual, err.fix)


def child_enumerator(child: "ProbabilityDistribution", path: str) -> "DistributionEnumerator":
    """Construct child.enumerator(), annotating EnumerationError with the child's path.

    Combinator enumerators use this so a failure deep in a nested model reports the
    full path to the offending leaf, e.g.
    'CompositeDistribution.dists[1] -> MixtureDistribution.components[0] -> GaussianDistribution ...'.
    """
    try:
        return child.enumerator()
    except EnumerationError as e:
        new_path = path if not e.path else "%s -> %s" % (path, e.path)
        raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None


class FitProvenanceCarrier:
    """Mixin giving a fitted object the three provenance accessors.

    Split out of :class:`ProbabilityDistribution` because not every fitted result inherits from it:
    ``learn_structure`` returns a ``DependencyTreeDistribution``, whose MRO is ``(cls, object)``, so
    a public fitting entry point produced a model that could not report how it was fitted at all
    (MXR-080-1190/1202). The methods are plain attribute lookups with no other base-class dependency,
    so any fitted type can carry them.
    """

    def fit_provenance(self) -> "FitProvenance | None":
        """How this object was fitted, or ``None`` if it was constructed directly.

        A distribution is a value: nothing about its parameters says whether they came from a converged
        EM run, a run that hit its iteration cap, or a keyboard. :func:`~mixle.inference.estimation.optimize`
        attaches a :class:`FitProvenance` so that difference is machine-readable (MXR-080-1190/1202).
        """
        return getattr(self, "_fit_provenance", None)

    def numerical_repairs(self) -> tuple[str, ...]:
        """Numerical repairs applied while building this object -- ``()`` when none were recorded."""
        return tuple(getattr(self, "_numerical_repairs", ()))

    def with_fit_provenance(self, provenance: "FitProvenance") -> "FitProvenanceCarrier":
        """Record ``provenance`` on this object and return it."""
        if not isinstance(provenance, FitProvenance):
            raise TypeError("provenance must be a FitProvenance, got %s" % type(provenance).__name__)
        object.__setattr__(self, "_fit_provenance", provenance)
        return self


class ProbabilityDistribution(FitProvenanceCarrier, ABC):
    """Base class for all probability distributions in mixle.stats.

    A distribution evaluates the (log-)density of a single observation of its data
    type, creates a DistributionSampler for drawing observations, and creates a
    ParameterEstimator for re-estimating itself from data. Discrete distributions
    may additionally provide a DistributionEnumerator over their support.
    """

    def __repr__(self) -> str:
        return self.__str__()

    def fit_provenance(self) -> "FitProvenance | None":
        """How this object was fitted, or ``None`` if it was constructed directly.

        A distribution is a value: nothing about its parameters says whether they came from a converged
        EM run, a run that hit its iteration cap, or a keyboard. :func:`~mixle.inference.estimation.optimize`
        attaches a :class:`FitProvenance` so that difference is machine-readable (MXR-080-1190/1202).
        """
        return getattr(self, "_fit_provenance", None)

    def numerical_repairs(self) -> tuple[str, ...]:
        """Numerical repairs applied while building this object -- ``()`` when none were recorded.

        A repair means the parameters are not exactly what the estimator computed: a covariance that
        was jitter-healed back to positive-definite, a variance clamped to a floor. Reported here so
        :meth:`fit_provenance` can carry it, since a repaired fit is an approximation whether or not
        the objective converged.

        An empty tuple means *no repair was recorded*, which for a family that records none is not the
        same as a proof that none occurred.
        """
        return tuple(getattr(self, "_numerical_repairs", ()))

    def with_fit_provenance(self, provenance: "FitProvenance") -> "ProbabilityDistribution":
        """Record ``provenance`` on this object and return it.

        Mutates in place and returns ``self`` rather than copying: fitted models are large, the caller
        is the fitting routine that just produced this object, and no one else holds a reference yet.
        """
        if not isinstance(provenance, FitProvenance):
            raise TypeError("provenance must be a FitProvenance, got %s" % type(provenance).__name__)
        object.__setattr__(self, "_fit_provenance", provenance)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this distribution."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbabilityDistribution":
        """Reconstruct a distribution from ``to_dict`` output."""
        from mixle.utils.serialization import from_serializable

        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this distribution as safe strict JSON."""
        from mixle.utils.serialization import to_json

        return to_json(self, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "ProbabilityDistribution":
        """Deserialize a distribution from ``to_json`` output."""
        from mixle.utils.serialization import from_json

        rv = from_json(text)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def density(self, x: Any) -> float:
        """Return the probability density or mass at a single observation.

        Concrete default: exponentiate ``log_density`` (the abstract method subclasses must
        provide). Leaves with a cheaper closed form may override this.
        """
        return math.exp(self.log_density(x))

    def capabilities(self) -> frozenset[str]:
        """Return the capability names this distribution supports (see :mod:`mixle.capability`).

        Feature detection by behaviour rather than class — e.g. ``"Enumerable"``,
        ``"Conditionable"``, ``"ExponentialFamily"``, ``"RankableByIndex"``. Equivalent to
        ``mixle.capabilities(self)``; combinators report the set their children jointly preserve.
        """
        from mixle.capability import capabilities

        return capabilities(self)

    def density_semantics(self) -> DensitySemantics:
        """What ``log_density`` returns relative to the true log-density (default: exact).

        Override to declare that this distribution's ``log_density`` is a variational lower bound
        (ELBO), an upper bound, or an approximation rather than the exact ``log p(x)``. This is surfaced
        as the ``ExactDensity`` capability and noted in :func:`mixle.describe`, so code that needs an
        exact likelihood can ``require(x, ExactDensity)`` instead of silently trusting a bound.
        """
        return DensitySemantics.EXACT

    def tropical_displacement_bits(self) -> float:
        """Worst-case gap (in bits) between this law's structural *count* cost and its true log-density.

        The structural count-DP (and the ``seek`` built on it) bins each value by a cost that is
        EXACT for decomposable families -- composites/sequences/markov chains whose ``log p(x)`` is a
        sum of independent per-factor terms -- so this returns ``0.0`` by default.

        For a marginal family the latent index is summed out and the count index bins by the
        *tropical* (dominant-component/path) cost ``M(x)`` instead of the true ``log p(x)``. Because
        ``M(x) <= log p(x) <= M(x) + log N`` for an ``N``-way logsumexp, the two costs differ by at
        most ``log2(N)`` bits. :func:`mixle.enumeration.density_rank.marginal_seek` widens its rank
        bracket by exactly this many bits so the bracket is a *guaranteed* bound on the true marginal
        rank (not merely the tropical rank); override to return ``log2(N)`` for such a family.
        """
        return 0.0

    @abstractmethod
    def log_density(self, x: Any) -> float:
        """Return the log-density or log-mass at a single observation."""
        ...

    @abstractmethod
    def sampler(self, seed: int | None = None) -> "DistributionSampler":
        """Return a sampler for drawing observations from this distribution."""
        ...

    @abstractmethod
    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """Return an estimator for fitting this distribution from data."""
        ...

    def to_fisher(self, **kwargs):
        """Return a Fisher-geometry view of this distribution.

        The default view is accumulator-backed, so distributions inherit a generic
        sufficient-statistic/Fisher-vector interface.  Each distribution owns its Fisher view by
        overriding this method in its own module; families not yet migrated to a per-file hook are
        resolved by the transitional type-name dispatch in :func:`mixle.inference.fisher._legacy_to_fisher`.
        """
        from mixle.inference.fisher import _legacy_to_fisher

        return _legacy_to_fisher(self, **kwargs)

    def to_exponential_family(self, engine: Any = None):
        """Return the canonical exponential-family view, or ``None``.

        The canonical form is ``p(x) = h(x) * exp(<eta, T(x)> - A(eta))``.  The default
        reads ``declaration_for(self).exponential_family`` (the per-family
        ``ExponentialFamilySpec``) and wraps it in an
        :class:`~mixle.stats.compute.exp_family.ExponentialFamilyForm`; it returns ``None`` when
        this family is not a (single) exponential family.  There is no type switch --
        adding a family is a matter of providing its spec.
        """
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.compute.declarations import declaration_for
        from mixle.stats.compute.exp_family import ExponentialFamilyForm

        declaration = declaration_for(self)
        if declaration is None or declaration.exponential_family is None:
            return None
        return ExponentialFamilyForm(
            distribution=self,
            spec=declaration.exponential_family,
            engine=NUMPY_ENGINE if engine is None else engine,
        )

    def get_prior(self) -> Optional["ProbabilityDistribution"]:
        """Return the conjugate/parameter prior carried by this distribution, if any.

        A distribution participates in the Bayesian (variational) protocol by
        carrying a prior over its parameters. The default returns whatever was
        stored on the ``prior`` attribute (``None`` for a plain point model),
        so frequentist distributions answer ``None`` and behave as MLE models.
        """
        return getattr(self, "prior", None)

    def set_prior(self, prior: Optional["ProbabilityDistribution"]) -> None:
        """Attach a parameter prior to this distribution.

        The default just records the prior; conjugate families override this to
        precompute the variational expected natural parameters used by
        ``expected_log_density``.
        """
        self.prior = prior

    def has_conjugate_prior(self) -> bool:
        """Return whether this family supports a closed-form conjugate Bayesian update.

        Uniform, family-level signal backed by the single ``conjugate_posterior`` registry: ``True``
        means ``mixle.stats.bayes.conjugate_posterior(self, data)`` returns an exact closed-form
        posterior; ``False`` means Bayesian inference must go through the numerical fitters
        (MAP / Laplace / MCMC / VI). This is the top tier of the inference-capability ladder
        (see :class:`mixle.capability.ConjugateUpdatable`). Distinct from the per-instance
        ``has_conj_prior`` flag, which records whether a prior is currently *attached*.
        """
        from mixle.stats.bayes.conjugate import is_conjugate_family

        return is_conjugate_family(self)

    def expected_log_density(self, x: Any) -> float:
        """Return the variational expectation ``E_q[log p(x | theta)]``.

        When the distribution carries a conjugate parameter posterior ``q`` this
        is the Bayesian E-step term; for a plain point model (no prior) it
        degenerates to the plug-in ``log_density(x)``. Conjugate families
        override this with their closed form.
        """
        return self.log_density(x)

    def enumerator(self) -> "DistributionEnumerator":
        """Return a DistributionEnumerator over this distribution's support.

        Distributions with an enumerable (discrete) support override this; the
        default raises EnumerationError.
        """
        raise EnumerationError(self)

    def support_size(self) -> int | None:
        """Return the number of distinct support points, or ``None`` if infinite/unknown.

        This is the cardinality primitive for bounding a truncated descending-probability sum: after
        enumerating the top ``k`` items (whose smallest probability is ``p_k``), every un-enumerated
        item has probability ``<= p_k``, so the remaining mass is ``<= (support_size - k) * p_k`` (see
        :func:`mixle.enumeration.density_rank.truncated_sum_bound`). Finite discrete leaves return their
        cardinality; decomposable combinators compose it structurally; an upper bound is acceptable
        (it only loosens the tail bound). Infinite or continuous supports return ``None``.
        """
        return None

    def support_is_finite(self) -> bool:
        """Return whether the support has a known finite cardinality."""
        return self.support_size() is not None

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0):
        """Build a bounded bit-quantized index over this distribution's support.

        This is a convenience wrapper around ``self.enumerator().quantized_index``.
        Non-enumerable distributions raise EnumerationError through enumerator().

        Args:
            max_bits (float): Maximum information content in bits to index.
            bin_width_bits (float): Width of each quantized probability bin in bits.

        Returns:
            mixle.enumeration.algorithms.QuantizedEnumerationIndex.

        """
        return self.enumerator().quantized_index(max_bits=max_bits, bin_width_bits=bin_width_bits)

    def density_quantile(
        self,
        q: float,
        n_samples: int = 20000,
        seed: int | None = None,
        *,
        invalid_score: str = "raise",
    ) -> Any:
        """Return a representative value at cumulative-density index ``q`` (descending-density order).

        The arbitrary-index / inverse of the probability-ordered cumulative that
        :func:`mixle.enumeration.density_rank.density_rank` computes (``G(x) = P(p(Y) >= p(x))``): ``q = 0``
        is the mode, ``q -> 1`` walks into the tail.  Families with a closed form override this
        (univariate continuous leaves expose the spatial ``quantile``; multivariate Gaussians and von
        Mises-Fisher override ``density_quantile`` exactly); the default here is the universal
        **Monte-Carlo representative** for any samplable family whose support is uncountable or coupled
        (parameter priors, continuous mixtures, ...): draw ``n_samples`` points, order them by
        descending density, and return the one at fractional rank ``q``.  Stochastic and approximate --
        use the exact methods (``enumerator``/``count_dp_seek`` for discrete) where available.

        Args:
            q (float): Cumulative-density index in ``[0, 1]``.
            n_samples (int): Monte-Carlo sample budget.
            seed (Optional[int]): Sampler seed (reproducible).
            invalid_score (str): ``"raise"`` for non-finite sampled scores, or
                ``"omit"`` to discard them explicitly.
        """
        index = _density_index(q)
        count = _positive_budget(n_samples, "n_samples")
        policy = _invalid_score_policy(invalid_score)
        samples = self.sampler(seed).sample(count)
        if not hasattr(samples, "__len__") or len(samples) != count:
            raise ValueError(
                "sampler returned %s values for n_samples=%d." % (getattr(samples, "__len__", lambda: "?")(), count)
            )
        with np.errstate(divide="ignore"):
            scored = [(float(self.log_density(y)), y) for y in samples]
        invalid = [lp for lp, _ in scored if not math.isfinite(lp)]
        if invalid and policy == "raise":
            raise ValueError("density representative sampling produced %d non-finite log densities." % len(invalid))
        if invalid:
            scored = [(lp, y) for lp, y in scored if math.isfinite(lp)]
        if not scored:
            raise ValueError("density representative sampling produced no finite log densities.")
        scored.sort(key=lambda item: -item[0])
        pick = int(round(index * (len(scored) - 1))) if len(scored) > 1 else 0
        return scored[pick][1]

    def density_enumeration(
        self,
        num_points: int,
        n_samples: int = 20000,
        seed: int | None = None,
        *,
        dedup_key: Any = None,
        invalid_score: str = "raise",
    ) -> list[tuple[Any, float]]:
        """Return ``num_points`` representative ``(value, log_density)`` pairs in descending density.

        The continuous analogue of :meth:`enumerator` (which enumerates a countable support exactly):
        for an uncountable or coupled support there is no exact element enumeration, so this returns a
        **Monte-Carlo representative** sweep -- ``n_samples`` draws ordered by descending density,
        keeping the ``num_points`` most probable (distinct) representatives, i.e. "the support,
        most-probable region first".  Stochastic and approximate; prefer :meth:`enumerator` where the
        support is countable.

        Args:
            num_points (int): Number of representatives to return.
            n_samples (int): Monte-Carlo sample budget.
            seed (Optional[int]): Sampler seed (reproducible).
            dedup_key (Optional[Callable]): Stable equality key for values not
                supported by the default canonical ``freeze`` operation.
            invalid_score (str): ``"raise"`` for non-finite sampled scores, or
                ``"omit"`` to discard them explicitly.
        """
        from mixle.enumeration.algorithms import freeze

        points = _positive_budget(num_points, "num_points")
        count = _positive_budget(n_samples, "n_samples")
        if points > count:
            raise ValueError("num_points cannot exceed n_samples.")
        if dedup_key is not None and not callable(dedup_key):
            raise TypeError("dedup_key must be callable.")
        policy = _invalid_score_policy(invalid_score)
        samples = self.sampler(seed).sample(count)
        if not hasattr(samples, "__len__") or len(samples) != count:
            raise ValueError(
                "sampler returned %s values for n_samples=%d." % (getattr(samples, "__len__", lambda: "?")(), count)
            )
        with np.errstate(divide="ignore"):
            scored = [(float(self.log_density(y)), y) for y in samples]
        invalid = [lp for lp, _ in scored if not math.isfinite(lp)]
        if invalid and policy == "raise":
            raise ValueError("density representative sampling produced %d non-finite log densities." % len(invalid))
        if invalid:
            scored = [(lp, y) for lp, y in scored if math.isfinite(lp)]
        scored.sort(key=lambda item: -item[0])
        out: list[tuple[Any, float]] = []
        seen: set = set()
        for lp, y in scored:
            try:
                key = freeze(y if dedup_key is None else dedup_key(y))
            except TypeError as exc:
                raise TypeError(
                    "density representative values require a stable deduplication key; "
                    "supply dedup_key for type %s." % type(y).__name__
                ) from exc
            if key in seen:
                continue
            seen.add(key)
            out.append((y, lp))
            if len(out) >= points:
                break
        return out

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """Build a structural CountIndex over this distribution's support, bounded by depth.

        This is the count-semiring counterpart of ``quantized_index``: it returns per-fine-bucket
        *counts* of the complete model probability together with a structural unranker, so the
        support can be indexed without being enumerated. The default builds a leaf index from the
        exact ``enumerator()`` truncated at ``max_fine_bucket`` (efficient for closed-form/small-support
        families); exponential-support composers (Composite/Sequence/MarkovChain) override this with
        a dynamic program over the model's likelihood recursion.

        Args:
            quantizer (mixle.enumeration.quantization.Quantizer): Fine/coarse bucketing.
            max_fine_bucket (int): Inclusive depth bound on indexed fine buckets.

        Returns:
            Tuple (CountIndex, truncated) -- truncated is True when in-support values were dropped
            because they fell beyond the depth bound.

        """
        from mixle.enumeration.quantization.core import leaf_count_index

        return leaf_count_index(self.enumerator(), quantizer, max_fine_bucket)

    def count_budget_index(
        self, budget_bits: float, bin_width_bits: float = 1.0, oversample: int = 8, num_workers: int | None = None
    ):
        """Build a budget-bounded quantized seek index covering the top ``2**budget_bits`` values.

        Computes per-bin counts structurally (never enumerating the domain) and accumulates coarse
        bins in descending-probability order until the cumulative count reaches the budget. The
        returned LazyQuantizedEnumerationIndex supports arbitrary-rank seek/unranking; each unranked
        value carries its exact ``log_density``.

        Args:
            budget_bits (float): Index into the top ``2**budget_bits`` most probable values.
            bin_width_bits (float): Coarse output bin width in bits.
            oversample (int): Fine buckets per coarse bin (accumulation resolution).

        Returns:
            mixle.enumeration.algorithms.LazyQuantizedEnumerationIndex.

        """
        from mixle.enumeration.quantization.core import count_budget_index

        return count_budget_index(
            self, budget_bits, bin_width_bits=bin_width_bits, oversample=oversample, num_workers=num_workers
        )

    def count_budget_distinct(
        self,
        budget_bits: float,
        bin_width_bits: float = 1.0,
        oversample: int = 8,
        dedup: str = "canonical",
        start: int = 0,
        stop: int | None = None,
        max_entries: int = 1 << 16,
        num_workers: int | None = None,
    ):
        """Iterate DISTINCT (value, exact_log_prob) over the count-budget index, approx descending.

        For exact-count families this equals the ordered index stream. For the over-counting
        MARGINAL families (Mixture/HMM) it removes the component/path duplicates by one of two modes:

          - ``dedup='canonical'`` (default): a STATELESS predicate (``is_canonical_copy``) keeps a
            value only at its dominant copy (best-weighted component / min-cost path), via the
            value's structural fine bucket (``structural_fine_bucket`` -- the SAME sum-of-floored
            sub-buckets the count index used, so nested composite/sequence values are binned
            consistently and never dropped). O(1) memory and random-accessible: ``start``/``stop``
            select an arbitrary STRUCTURAL rank range, so you can begin anywhere and the work
            partitions across workers with no shared state. GUARANTEE: every distinct in-budget value
            is emitted at least once (completeness). It is NOT strictly once -- a value is emitted
            once per component/path that ties within its minimal 1-bit coarse bin; that residue is
            inherent to a stateless ``(value, coarse_bin)`` rule (the tied copies are
            indistinguishable to it) and is bounded by the number of components/contributing paths
            (not reduced by ``oversample``, which only refines the intermediate fine bucket). For a
            strictly-once stream, use ``dedup='window'`` or de-duplicate downstream.
          - ``dedup='window'``: a bounded ``max_entries`` LRU over the stream (catches every duplicate
            within the window regardless of dominance, but is sequential -- ``start`` must be 0).

        Note: ``start``/``stop`` index the STRUCTURAL enumeration, not the distinct rank. Jumping to
        the k-th *distinct* value in O(1) is not possible -- it needs exact distinct per-bin counts,
        which require materializing the component/path overlap structure.
        """
        from mixle.enumeration.quantization.core import distinct_budget_stream

        return distinct_budget_stream(
            self,
            budget_bits,
            bin_width_bits=bin_width_bits,
            oversample=oversample,
            dedup=dedup,
            start=start,
            stop=stop,
            max_entries=max_entries,
            num_workers=num_workers,
        )

    def is_canonical_copy(self, value, coarse_bin: int, quantizer) -> bool:
        """Return True if ``coarse_bin`` is ``value``'s dominant (canonical) bin in the count index.

        Stateless deduplication hook for the over-counting MARGINAL families: a value that the
        structural index emits once per component / state-path is kept only at the copy whose bin is
        the minimal (most probable) one. The default returns True -- exact-count families
        (Composite/Sequence/MarkovChain) never duplicate, so every copy is canonical.
        """
        return True

    def structural_fine_bucket(self, value, quantizer) -> int:
        """Minimum fine bucket where ``value`` is placed by this distribution's count index.

        Mirrors ``quantized_count_index`` exactly so that stateless canonical-copy dedup can predict
        the bucket the index actually used. The count DP bins a composite/nested value by a SUM of
        *floored* per-factor buckets (a convolution), which differs from a single floor of the exact
        joint log-density by up to the number of factors -- so a canonical check that used
        ``fine_bucket(log_density(value))`` would mispredict and silently drop nested values. The
        leaf default is that single floor (correct for atomic families); combinators
        (Composite/Sequence/Mixture) override to recurse the same way their count index composes.
        """
        return quantizer.fine_bucket(float(self.log_density(value)))

    def quantized_multi_cross_index(
        self, others: list["ProbabilityDistribution"], max_bits, bin_width_bits: float = 1.0
    ):
        """Build an aligned bounded cross-bin view against other distributions.

        The generic implementation is a bounded candidate join: it unions the bounded
        quantized indexes of all participating distributions, then evaluates every
        candidate under every distribution. Structured distributions can override this
        to build the same aligned rows from support algebra instead.
        """
        from mixle.enumeration.algorithms import QuantizedCrossIndex, freeze

        dists = [self] + list(others)
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = tuple([float(max_bits)] * len(dists))
        if len(max_bits_tuple) != len(dists):
            raise ValueError("max_bits length must match the number of distributions.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        seen = set()
        values = []
        truncated = False
        for dist, bit_bound in zip(dists, max_bits_tuple):
            if bit_bound < 0.0:
                truncated = True
                continue
            index = dist.quantized_index(max_bits=bit_bound, bin_width_bits=bin_width_bits)
            truncated = truncated or index.truncated
            for value, _ in index.iter_from():
                key = freeze(value)
                if key not in seen:
                    seen.add(key)
                    values.append(value)

        items = []
        for value in values:
            items.append((value, tuple(float(dist.log_density(value)) for dist in dists)))
        return QuantizedCrossIndex.from_items(
            items, max_bits=max_bits_tuple, bin_width_bits=bin_width_bits, truncated=truncated
        )

    def quantized_cross_index(self, other: "ProbabilityDistribution", max_bits, bin_width_bits: float = 1.0):
        """Build an aligned bounded cross-bin view against another distribution."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class SequenceEncodableProbabilityDistribution(ProbabilityDistribution):
    """ProbabilityDistribution with vectorized log-density evaluation on encoded data.

    dist_to_encoder() returns a DataSequenceEncoder whose seq_encode() output is
    consumed by seq_log_density() (and by the matching accumulator's seq_update /
    seq_initialize), enabling fast vectorized estimation over iid sequences.
    """

    engine_ready = ("numpy",)

    def supported_engines(self) -> tuple[str, ...]:
        """Return engine names this distribution can evaluate on directly."""
        from mixle.stats.compute.capabilities import capabilities_for

        return capabilities_for(self).engine_ready

    def supports_engine(self, engine: Any) -> bool:
        """Return True when the distribution can safely use ``engine``."""
        from mixle.stats.compute.capabilities import capabilities_for

        return capabilities_for(self).supports_engine(engine)

    def decomposition(self) -> "Decomposition":
        """Return how this distribution may be split across devices (model parallelism).

        Defaults to :meth:`Decomposition.atomic` -- not split, replicated. Combinators / latent
        families override this to declare their component / factor / state axis. See
        :mod:`mixle.stats.compute.decomposition`.
        """
        from mixle.stats.compute.decomposition import Decomposition

        return Decomposition.atomic()

    def seq_ld_lambda(self):
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return np.asarray([self.log_density(u) for u in x])

    def seq_expected_log_density(self, x: Any) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations.

        Degenerates to ``seq_log_density`` for a plain point model; conjugate
        families override with a closed form. See ``expected_log_density``.
        """
        return self.seq_log_density(x)

    def seq_log_density_lambda(self):
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    def kernel(self, engine=None, estimator: Optional["ParameterEstimator"] = None):
        """Return an engine-aware evaluation kernel for this distribution."""
        from mixle.stats.compute.kernel import kernel_for

        return kernel_for(self, engine=engine, estimator=estimator)

    @abstractmethod
    def dist_to_encoder(self) -> "DataSequenceEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        ...


class DistributionSampler(ABC):
    """Draws iid observations from a distribution using a seeded RandomState.

    sample(size=None) returns a single observation of the distribution's data type;
    sample(size=n) returns a length-n collection of observations.
    """

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        seed: int | None = None,
        *,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self.dist = dist
        # ``rng`` (keyword-only) lets callers share one RandomState across samplers for composable,
        # reproducible streams; ``seed`` remains the default scalar-seed path when no rng is supplied.
        self.rng = rng if rng is not None else np.random.RandomState(seed)

    def new_seed(self) -> int:
        """Return a fresh random seed drawn from this sampler's RandomState."""
        return self.rng.randint(0, maxrandint)

    @abstractmethod
    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw observations.

        Combinator samplers (mixture/sequence/...) accept ``batched``. With
        ``batched=True`` (the default) each child stream is drawn in one vectorized
        call instead of a per-draw Python loop -- far faster. Because every child
        sampler owns an independent ``RandomState``, batching consumes each stream
        in the same order as the loop, so the draws are identical to the legacy
        path. ``batched=False`` forces that legacy per-draw loop as a guaranteed-
        stable reference. Leaf samplers are already vectorized and ignore the flag.
        """
        ...


class NucleusResult(list[tuple[Any, float]]):
    """List-compatible nucleus prefix with explicit termination provenance."""

    def __init__(
        self,
        items: list[tuple[Any, float]],
        *,
        target_probability: float,
        cumulative_probability: float,
        reached_target: bool,
        capped: bool,
        exhausted: bool,
    ) -> None:
        super().__init__(items)
        self.target_probability = target_probability
        self.cumulative_probability = cumulative_probability
        self.reached_target = reached_target
        self.capped = capped
        self.exhausted = exhausted


class DistributionEnumerator(ABC):
    """Lazy iterator over the support of dist in non-increasing probability order.

    Yields (value, log_prob) pairs, possibly infinitely many. Contract:
      - Each support value is yielded exactly once (deduplication is the
        enumerator's responsibility).
      - log_prob equals dist.log_density(value) up to float round-off (~1e-10),
        and the sequence of log_probs is non-increasing up to the same tolerance.
      - Values with zero probability are skipped, never yielded.
      - Ties are broken deterministically by insertion order; no further guarantee.
    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        self.dist = dist

    def __iter__(self) -> "DistributionEnumerator":
        return self

    @abstractmethod
    def __next__(self) -> tuple[Any, float]: ...

    def top_k(self, k: int) -> list[tuple[Any, float]]:
        """Return the k most probable (value, log_prob) pairs (fewer if the support is smaller)."""
        return list(itertools.islice(self, k))

    def top_p(self, p: float, max_items: int | None = None) -> NucleusResult:
        """Return the smallest descending-probability prefix whose total probability reaches ``p``.

        The nucleus / minimal high-probability set: because values are yielded in non-increasing
        probability order, the returned prefix is a minimum-size set of outcomes whose summed mass is
        ``>= p`` (e.g. ``p=0.95`` gives a 95%-coverage support set -- the discrete analogue of nucleus
        sampling). Accumulation stops as soon as the cumulative probability reaches ``p``.

        ``max_items`` caps how many values are pulled so an infinite or heavy-tailed support cannot
        run away. A cap is mandatory when the support has no known finite cardinality. The returned
        list-compatible :class:`NucleusResult` reports whether the target was reached, the accumulated
        mass, and whether a cap stopped the search.

        Args:
            p (float): Target cumulative probability in ``[0, 1]``.
            max_items (Optional[int]): Hard cap on the number of values pulled.

        Returns:
            A list-compatible nucleus result with termination provenance.

        """
        if isinstance(p, (bool, np.bool_)):
            raise TypeError("p must be a finite real number in [0, 1].")
        try:
            target = float(p)
        except (TypeError, ValueError) as exc:
            raise TypeError("p must be a finite real number in [0, 1].") from exc
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("p must be a finite real number in [0, 1].")
        cap = None if max_items is None else _positive_budget(max_items, "max_items")
        known_size = self.dist.support_size()
        if known_size is not None:
            if isinstance(known_size, (bool, np.bool_)) or not isinstance(known_size, (int, np.integer)):
                raise TypeError("support_size() must return a non-negative integer or None.")
            if known_size < 0:
                raise ValueError("support_size() must return a non-negative integer or None.")
        if target > 0.0 and cap is None and known_size is None:
            raise ValueError("top_p on an unknown or infinite support requires max_items.")
        if target == 0.0:
            return NucleusResult(
                [],
                target_probability=target,
                cumulative_probability=0.0,
                reached_target=True,
                capped=False,
                exhausted=known_size == 0,
            )

        limit = cap
        if known_size is not None:
            limit = known_size if limit is None else min(limit, known_size)
        out: list[tuple[Any, float]] = []
        total = 0.0
        exhausted = False
        iterator = iter(self)
        while limit is None or len(out) < limit:
            try:
                value, raw_log_prob = next(iterator)
            except StopIteration:
                exhausted = True
                break
            if isinstance(raw_log_prob, (bool, np.bool_)):
                raise TypeError("enumerated log probabilities must be finite non-positive real numbers.")
            try:
                log_prob = float(raw_log_prob)
            except (TypeError, ValueError) as exc:
                raise TypeError("enumerated log probabilities must be finite non-positive real numbers.") from exc
            if not math.isfinite(log_prob) or log_prob > 0.0:
                raise ValueError("enumerated log probabilities must be finite and non-positive.")
            out.append((value, log_prob))
            total = math.fsum((total, math.exp(log_prob)))
            if total > 1.0 + 1.0e-10:
                raise ValueError("enumerated cumulative probability exceeds one.")
            if total >= target:
                break
        reached = total >= target
        if known_size is not None and len(out) >= known_size:
            exhausted = True
        capped = not reached and cap is not None and (known_size is None or cap < known_size) and len(out) >= cap
        return NucleusResult(
            out,
            target_probability=target,
            cumulative_probability=min(total, 1.0),
            reached_target=reached,
            capped=capped,
            exhausted=exhausted,
        )

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0):
        """Precompute a bounded bit-quantized index over this enumeration.

        The index groups values by floor((-log2 p(x)) / bin_width_bits), includes only
        values with -log2 p(x) <= max_bits, and returns exact log probabilities for
        indexed values. Building the index consumes this enumerator.

        Args:
            max_bits (float): Maximum information content in bits to index.
            bin_width_bits (float): Width of each quantized probability bin in bits.

        Returns:
            mixle.enumeration.algorithms.QuantizedEnumerationIndex.

        """
        from mixle.enumeration.algorithms import QuantizedEnumerationIndex

        return QuantizedEnumerationIndex.from_enumerator(self, max_bits=max_bits, bin_width_bits=bin_width_bits)

    # -- where-does-a-value-sit / what-is-at-this-index, as methods on the enumerator ---------------
    # The enumerator is the one home for "rank a value", "seek the value at an index", and "iterate
    # from an index" over the weighted structure it enumerates. These delegate to the descending-
    # probability machinery in mixle.enumeration.density_rank (imported lazily to keep the pdist
    # contract layer free of an eager dependency on the enumeration package).

    def rank(self, value: Any):
        """Rank and cumulative probability of ``value`` in the descending-probability order.

        Returns a :class:`~mixle.enumeration.density_rank.DensityRankResult` with ``.rank`` (0-based
        count of strictly-more-probable outcomes; ``None`` if only the sampling estimate was used) and
        ``.cumulative_probability`` (``G(value) = P(p(Y) >= p(value))``). Exact head enumeration where
        the support is countable, with a Monte-Carlo fallback for the deep tail.
        """
        from mixle.enumeration.density_rank import density_rank

        return density_rank(self.dist, value)

    def seek(self, index: int):
        """The value at descending-probability ``index`` (0-based) -- the inverse of :meth:`rank`.

        Returns a :class:`~mixle.enumeration.density_rank.CountDPSeekResult` carrying the value and a
        provable ``[rank_lower, rank_upper]`` bracket. Uses the structural count-DP for decomposable
        families, so arbitrarily deep indices are reachable without enumerating the prefix.
        """
        from mixle.enumeration.density_rank import count_dp_seek

        return count_dp_seek(self.dist, index)

    def seek_certified(self, index: int):
        """The value at descending ``index`` with a GUARANTEED bracket on its TRUE marginal rank.

        Unlike :meth:`seek` -- whose bracket bounds only the *tropical* rank for a marginal family
        (mixture/HMM) -- this widens the rank window by the family's ``tropical_displacement_bits`` and
        divides out the component over-count, so the returned
        :class:`~mixle.enumeration.density_rank.MarginalSeekResult` ``[true_rank_lower, true_rank_upper]``
        provably contains ``#{u : log p(u) > log p(value)}``. It pins the rank exactly (``.exact``) for
        decomposable / provably-disjoint families and for shallow indices, and otherwise returns the
        certified provable envelope. For a decomposable family it agrees with :meth:`seek`.
        """
        from mixle.enumeration.density_rank import marginal_seek

        return marginal_seek(self.dist, index)

    def cumulative(self, value: Any):
        """``G(value) = P(p(Y) >= p(value))`` -- total mass of outcomes at least as probable as ``value``.

        Returns a :class:`~mixle.enumeration.density_rank.CumulativeProbabilityResult` with
        ``.probability`` plus explicit certification (``.exact``, ``.dropped_upper``) of what that
        figure rests on.
        """
        from mixle.enumeration.density_rank import cumulative_probability

        return cumulative_probability(self.dist, value)

    def nucleus_size(self, p: float):
        """Size of :meth:`top_p` (the minimal ``>= p``-mass set) WITHOUT materializing it.

        Returns a :class:`~mixle.enumeration.density_rank.CountDPTopPResult` with a provable size
        bracket, from the structural count-DP -- usable when the nucleus is far too large to list.
        """
        from mixle.enumeration.density_rank import count_dp_top_p

        return count_dp_top_p(self.dist, p)

    def from_index(self, start: int, stop: int | None = None):
        """Iterate ``(value, log_prob)`` in descending-probability order starting at structural ``start``.

        Yields the same stream as iterating a fresh enumerator but beginning at index ``start`` (and
        ending before ``stop`` if given). A fresh underlying enumeration is used, so this does not
        consume ``self``. (Decomposable families admit a direct structural jump via the count-budget
        index; the current implementation skips the best-first prefix -- the structural fast path is a
        WS-3 performance follow-up.)
        """
        return itertools.islice(self.dist.enumerator(), start, stop)


class ConditionalSampler(ABC):
    """Sampler mixin for conditional draws: sample_given(x) draws from P(. | x)."""

    @abstractmethod
    def sample_given(self, x):
        """Draw a sample from the conditional distribution given ``x``."""
        ...


class StatisticAccumulator(ABC, Generic[SS]):
    """Accumulates weighted sufficient statistics of type SS from observations.

    update(x, weight, estimate) adds one observation (estimate is the previous model,
    used for E-step posteriors; it may be None during initialization). Accumulators
    merge across partitions via combine(suff_stat) / value() / from_value(), and
    key_merge / key_replace pool statistics shared across model components through
    a stats_dict keyed by the accumulator's key.
    """

    def update(self, x: Any, weight: float, estimate) -> None:
        """Accumulate one weighted observation under an optional current estimate."""
        ...

    def initialize(self, x: Any, weight: float, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics from one weighted observation."""
        self.update(x, weight, estimate=None)

    @abstractmethod
    def combine(self, suff_stat: SS) -> "StatisticAccumulator":
        """Merge serialized sufficient statistics into this accumulator."""
        ...

    @abstractmethod
    def value(self) -> SS:
        """Return this accumulator's serialized sufficient statistics."""
        ...

    @abstractmethod
    def from_value(self, x: SS) -> "SequenceEncodableStatisticAccumulator":
        """Restore this accumulator from serialized sufficient statistics."""
        ...

    def scale(self, c: float) -> "StatisticAccumulator":
        """Scale linear sufficient statistics in-place by ``c``.

        The structural default is correct for ordinary weighted sums, nested
        tuples/lists/dicts, and numeric arrays. Families whose ``value()``
        payload includes non-linear metadata such as support bounds must
        override this method and leave that metadata unscaled.
        """
        return self.from_value(scale_suff_stat(self.value(), c))

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Pool this accumulator's statistics into ``stats_dict`` under its merge key.

        The structural default implements the common single-key pattern: store the accumulator
        under ``self.keys`` the first time the key is seen, else ``combine`` into the one already
        there. Accumulators whose key does not cover their whole state (several named keys, a
        wrapped child accumulator, a non-accumulator payload) override this -- and ONLY those:
        as of 2026-07-14 every whole-state family uses this default, after a sweep found sixteen
        hand-rolled copies that had drifted into three broken shapes. If you are tempted to
        reimplement this per family, don't; extend the base or override with ``super()`` calls.
        A ``keys`` of ``None`` (the default) is a no-op.

        Why this exact shape is correct -- and the drifted shapes are not: the POOL must live in
        the dict. Combining INTO the dict-held accumulator (below) mutates the pool in place, so
        after all sites merge, the dict entry holds everyone's statistics and ``key_replace``
        hands that pool to every site. The observed broken variants each violate one half:
        pulling the dict's stats into ``self`` without writing back (the dict keeps the FIRST
        site only -- later sites' data silently discarded); merging only when the key is already
        present (nobody ever inserts: both passes are no-ops and tying does nothing); and
        replacing by pushing ``self`` INTO the dict (the LAST site overwrites the pool). The
        keyed-protocol sweep (`keyed_protocol_sweep_test.py`) enforces this behaviorally for
        every catalog family.
        """
        keys = getattr(self, "keys", None)
        if keys is not None:
            if keys in stats_dict:
                stats_dict[keys].combine(self.value())
            else:
                stats_dict[keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics from the pooled ``stats_dict`` entry (see key_merge)."""
        keys = getattr(self, "keys", None)
        if keys is not None and keys in stats_dict:
            self.from_value(stats_dict[keys].value())


class SequenceEncodableStatisticAccumulator(StatisticAccumulator[SS]):
    """StatisticAccumulator with vectorized updates on encoded data sequences.

    seq_update / seq_initialize consume the output of the matching
    DataSequenceEncoder's seq_encode() (obtained via acc_to_encoder()) together with
    a per-observation weight vector.
    """

    def get_seq_lambda(self):
        """Return optional low-level sequence-update kernels used by generated code."""
        pass

    @abstractmethod
    def seq_update(self, x, weights: np.ndarray, estimate) -> None:
        """Accumulate weighted sufficient statistics from sequence-encoded observations."""
        ...

    @abstractmethod
    def seq_initialize(self, x, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics from sequence-encoded observations."""
        ...

    @abstractmethod
    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return a sequence encoder compatible with this accumulator."""
        ...


def scale_suff_stat(x: Any, c: float) -> Any:
    """Return ``x`` with numeric sufficient-statistic leaves multiplied by ``c``."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.number) and not np.issubdtype(x.dtype, np.bool_):
            return x * c
        return x.copy()
    if isinstance(x, dict):
        return {k: scale_suff_stat(v, c) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(scale_suff_stat(v, c) for v in x)
    if isinstance(x, list):
        return [scale_suff_stat(v, c) for v in x]
    if isinstance(x, np.generic):
        if np.issubdtype(type(x), np.number) and not np.issubdtype(type(x), np.bool_):
            return x * c
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return x * c
    return x


class StatisticAccumulatorFactory(ABC):
    """Factory whose make() returns a fresh, zeroed accumulator for one estimator."""

    @abstractmethod
    def make(self) -> "SequenceEncodableStatisticAccumulator":
        """Create a fresh accumulator instance."""
        ...


class ParameterEstimator(ABC, Generic[SS]):
    """Estimates a distribution from accumulated sufficient statistics.

    accumulator_factory() supplies accumulators that gather sufficient statistics of
    type SS, and estimate(nobs, suff_stat) maps those statistics (plus optional
    regularization configured on the estimator) to a new distribution.
    """

    supported_sample_structures = frozenset({"iid", "exchangeable"})

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this estimator."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParameterEstimator":
        """Reconstruct an estimator from ``to_dict`` output."""
        from mixle.utils.serialization import from_serializable

        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this estimator as safe strict JSON."""
        from mixle.utils.serialization import to_json

        return to_json(self, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "ParameterEstimator":
        """Deserialize an estimator from ``to_json`` output."""
        from mixle.utils.serialization import from_json

        rv = from_json(text)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    @abstractmethod
    def estimate(self, nobs: float | None, suff_stat: SS) -> "SequenceEncodableProbabilityDistribution":
        """Estimate a distribution from accumulated sufficient statistics."""
        ...

    @abstractmethod
    def accumulator_factory(self) -> "StatisticAccumulatorFactory":
        """Return the accumulator factory used to collect this estimator's sufficient statistics."""
        ...

    def resident_accumulation_supported(self) -> bool:
        """Return whether engine-resident (fixed-width) sufficient statistics suffice for ``estimate``.

        Most exponential-family M-steps consume only the resident sufficient statistics,
        so the default is ``True``. Estimators whose M-step needs more than that (e.g. a
        full count histogram for the negative-binomial dispersion solve) override this to
        ``False`` so stacked/generated kernels fall back to the host accumulator, keeping
        every backend's fixed point identical.
        """
        return True

    def get_prior(self) -> Optional["ProbabilityDistribution"]:
        """Return the parameter prior configured on this estimator, if any.

        The unified estimation contract treats the prior as the single
        regularization concept: ``None`` gives maximum likelihood, a conjugate
        prior gives the Bayesian posterior update inside ``estimate``. The
        default reads the ``prior`` attribute (``None`` when unset).
        """
        return getattr(self, "prior", None)

    def model_log_density(self, model: "ProbabilityDistribution") -> float:
        """Return the prior log-density of ``model``'s parameters (the ELBO global term).

        Used by the variational/MAP objective in ``fit``. The default is ``0.0``
        (no prior); conjugate estimators override this to evaluate their prior at
        the model's parameters, mapping to the prior's parameterization first.
        """
        return 0.0


class DataSequenceEncoder(ABC):
    """Encodes an iid data sequence into the vectorized form used by seq_* methods.

    seq_encode(x) transforms a sequence of observations into the encoding consumed
    by seq_log_density / seq_update / seq_initialize. Encoders must define __eq__
    (so equivalent encoders are interchangeable when batching) and a readable
    __str__.
    """

    def __str__(self) -> str:
        return type(self).__name__

    def seq_encode(self, x: Any) -> Any:
        """Encode the iid observation sequence x for vectorized evaluation."""
        return x

    def nbytes(self, x: Any) -> int:
        """Return the approximate in-memory byte size of an encoded payload."""
        return encoded_nbytes(x)

    def row_count(self, x: Any) -> int:
        """Return the number of observations represented by ``x``.

        The default covers simple array/list encodings and structurally
        aligned tuples/dicts. Encoders with flattened, ragged, or otherwise
        specialized layouts must override this method so metadata wrappers can
        verify counts instead of trusting a caller assertion.
        """
        count = _infer_encoded_row_count(x)
        if count is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement row_count() for its encoded payload layout"
            )
        return count

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...


def _infer_encoded_row_count(x: Any) -> int | None:
    """Infer a leading observation count only when a payload is unambiguous."""
    shape = getattr(x, "shape", None)
    if shape is not None:
        shape = tuple(shape)
        return int(shape[0]) if shape else None
    if isinstance(x, list) and all(
        not isinstance(value, (dict, list, tuple, np.ndarray)) and not hasattr(value, "shape") for value in x
    ):
        return len(x)
    if isinstance(x, dict):
        children = x.values()
    elif isinstance(x, (list, tuple)):
        children = x
    else:
        return None
    counts = {count for value in children if (count := _infer_encoded_row_count(value)) is not None}
    return counts.pop() if len(counts) == 1 else None


def encoded_nbytes(x: Any) -> int:
    """Return an approximate byte size for nested encoded array payloads.

    Encoders mostly return arrays or tuples/lists/dicts of arrays. The helper
    keeps accounting structural and deterministic; Python object overhead is
    included only for scalar leaves where no array-native byte count exists.
    """
    return _encoded_nbytes(x, set())


def _encoded_nbytes(x: Any, seen: set[int]) -> int:
    oid = id(x)
    if oid in seen:
        return 0

    if isinstance(x, np.ndarray):
        seen.add(oid)
        return int(x.nbytes)

    nbytes = getattr(x, "nbytes", None)
    if nbytes is not None and not isinstance(x, (bytes, bytearray, str)):
        seen.add(oid)
        return int(nbytes)

    if hasattr(x, "numel") and hasattr(x, "element_size"):
        seen.add(oid)
        return int(x.numel() * x.element_size())

    if isinstance(x, dict):
        seen.add(oid)
        return sum(_encoded_nbytes(k, seen) + _encoded_nbytes(v, seen) for k, v in x.items())

    if isinstance(x, (list, tuple)):
        seen.add(oid)
        return sum(_encoded_nbytes(v, seen) for v in x)

    if isinstance(x, (bytes, bytearray)):
        return len(x)

    if isinstance(x, str):
        return len(x.encode("utf-8"))

    return sys.getsizeof(x)


_KEY_ATTRS = ("key", "keys", "weight_key", "comp_key", "init_key", "trans_key", "state_key")


def _key_attr_value(owner: Any, attr: str) -> Any:
    """Read a key attribute, calling it when the object exposes it as a no-argument accessor.

    Estimators and accumulators may carry their keys either as a plain attribute (``self.keys = [...]``,
    which every shipped family uses) or as a method (``def keys(self): ...``). getattr returns a bound
    method for the second form, which is not a key value and not a sequence, so key collection fell
    through to _canonical_key and rejected the estimator outright with "got method" -- a custom
    estimator following the documented API could not be fitted at all.

    Only a zero-argument callable is invoked: anything requiring arguments is not a keys accessor, and
    is left alone so the normal validation reports it.
    """
    import inspect  # deferred, as elsewhere in this module, to keep the import graph acyclic

    value = getattr(owner, attr)
    if not callable(value):
        return value
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return value
    if any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in signature.parameters.values()
    ):
        return value
    return value()


def _canonical_key(x: Any) -> tuple[str, Any]:
    """Return a typed, equality-stable representation of one permitted scalar key."""
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, str):
        return "str", x
    if isinstance(x, bytes):
        return "bytes", x
    if isinstance(x, bool):
        return "bool", x
    if isinstance(x, int):
        return "int", x
    if isinstance(x, float):
        if math.isnan(x):
            return "float", "nan"
        if math.isinf(x):
            return "float", "+inf" if x > 0.0 else "-inf"
        if x == 0.0:
            return "float", "0"
        return "float", x.hex()
    raise KeyValidationError(
        "Key values must be scalar str, bytes, bool, int, float, or NumPy scalar; got %s." % type(x).__name__
    )


def _is_key_value(x: Any) -> bool:
    """Return True for permitted scalar key values used by key_merge methods."""
    if x is None:
        return False
    try:
        _canonical_key(x)
    except KeyValidationError:
        return False
    return True


def _freeze_for_signature(x: Any, seen: set[int] | None = None) -> Any:
    """Convert result-affecting state into a deterministic hashable signature."""
    if seen is None:
        seen = set()
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, np.ndarray):
        if x.dtype.hasobject:
            content = _freeze_for_signature(x.tolist(), seen)
        else:
            contiguous = np.ascontiguousarray(x)
            content = hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()
        return "ndarray", tuple(x.shape), x.dtype.str, content
    if isinstance(x, float):
        return "float", _canonical_key(x)[1]
    if isinstance(x, (str, bytes, bool, int, type(None))):
        return type(x).__name__, x

    obj_id = id(x)
    if obj_id in seen:
        return "cycle", type(x).__module__, type(x).__qualname__
    if isinstance(x, dict):
        seen.add(obj_id)
        items = [(_freeze_for_signature(k, seen), _freeze_for_signature(v, seen)) for k, v in x.items()]
        seen.remove(obj_id)
        return "dict", tuple(sorted(items, key=repr))
    if isinstance(x, (list, tuple)):
        seen.add(obj_id)
        values = tuple(_freeze_for_signature(v, seen) for v in x)
        seen.remove(obj_id)
        return type(x).__name__, values
    if isinstance(x, (set, frozenset)):
        seen.add(obj_id)
        values = tuple(sorted((_freeze_for_signature(v, seen) for v in x), key=repr))
        seen.remove(obj_id)
        return type(x).__name__, values
    if callable(x):
        return "callable", getattr(x, "__module__", None), getattr(x, "__qualname__", type(x).__qualname__)
    if hasattr(x, "__dict__"):
        return _object_signature(x, seen)
    return type(x).__module__, type(x).__qualname__, repr(x)


def _object_signature(x: Any, seen: set[int] | None = None) -> Any:
    """Structural signature containing all result-affecting object state."""
    if seen is None:
        seen = set()
    obj_id = id(x)
    if obj_id in seen:
        return "cycle", type(x).__module__, type(x).__qualname__
    seen.add(obj_id)
    values = []
    for name, value in sorted(vars(x).items()):
        if name in ("name", "key", "keys", "weight_key", "comp_key", "init_key", "trans_key", "state_key"):
            continue
        values.append((name, _freeze_for_signature(value, seen)))
    seen.remove(obj_id)
    return (type(x).__module__, type(x).__qualname__, tuple(values))


def _accumulator_signature(accumulator: StatisticAccumulator, role: str) -> Any:
    try:
        value_sig = _freeze_for_signature(accumulator.value())
    except Exception as err:  # noqa: BLE001
        value_sig = ("value-error", type(err).__name__, str(err))
    return (type(accumulator).__module__, type(accumulator).__qualname__, role, value_sig)


def _register_key(registry: dict[Any, tuple[Any, str, int, Any]], key: Any, signature: Any, path: str) -> None:
    canonical = _canonical_key(key)
    old = registry.get(canonical)
    if old is None:
        registry[canonical] = (signature, path, 1, key)
        return
    old_signature, old_path, count, original = old
    if old_signature != signature:
        raise KeyValidationError(
            "Incompatible keyed sufficient-statistic sites for key %r: %s has %r, "
            "but %s has %r." % (original, old_path, old_signature, path, signature)
        )
    registry[canonical] = (old_signature, old_path, count + 1, original)


def _iter_children(x: Any) -> list[Any]:
    if isinstance(x, dict):
        return list(x.values())
    if isinstance(x, (list, tuple)):
        return list(x)
    return []


def _collect_estimator_keys(
    estimator: ParameterEstimator, registry: dict[Any, tuple[Any, str, int, Any]], path: str, visited: set[int]
) -> None:
    obj_id = id(estimator)
    if obj_id in visited:
        return
    visited.add(obj_id)

    estimator_sig = _object_signature(estimator)
    for attr in _KEY_ATTRS:
        if not hasattr(estimator, attr):
            continue
        keys = _key_attr_value(estimator, attr)
        if keys is None:
            continue
        if _is_key_value(keys):
            _register_key(
                registry,
                keys,
                (type(estimator).__module__, type(estimator).__qualname__, attr, estimator_sig),
                "%s.%s" % (path, attr),
            )
        elif isinstance(keys, (list, tuple)):
            for i, key in enumerate(keys):
                if key is None or isinstance(key, (list, tuple)):
                    continue
                if _is_key_value(key):
                    _register_key(
                        registry,
                        key,
                        (type(estimator).__module__, type(estimator).__qualname__, "%s[%d]" % (attr, i), estimator_sig),
                        "%s.%s[%d]" % (path, attr, i),
                    )
                else:
                    _canonical_key(key)
        else:
            _canonical_key(keys)

    for name, value in sorted(vars(estimator).items()):
        for i, child in enumerate(_iter_children(value)):
            if isinstance(child, ParameterEstimator):
                _collect_estimator_keys(child, registry, "%s.%s[%d]" % (path, name, i), visited)
        if isinstance(value, ParameterEstimator):
            _collect_estimator_keys(value, registry, "%s.%s" % (path, name), visited)


def _collect_accumulator_keys(
    accumulator: StatisticAccumulator, registry: dict[Any, tuple[Any, str, int, Any]], path: str, visited: set[int]
) -> None:
    obj_id = id(accumulator)
    if obj_id in visited:
        return
    visited.add(obj_id)

    for attr in _KEY_ATTRS:
        if not hasattr(accumulator, attr):
            continue
        key = _key_attr_value(accumulator, attr)
        if key is None:
            continue
        if _is_key_value(key):
            _register_key(registry, key, _accumulator_signature(accumulator, attr), "%s.%s" % (path, attr))
        elif isinstance(key, (list, tuple)):
            for index, item in enumerate(key):
                if item is None or isinstance(item, (list, tuple)):
                    continue
                if _is_key_value(item):
                    _register_key(
                        registry,
                        item,
                        _accumulator_signature(accumulator, "%s[%d]" % (attr, index)),
                        "%s.%s[%d]" % (path, attr, index),
                    )
                else:
                    _canonical_key(item)
        else:
            _canonical_key(key)

    for name, value in sorted(vars(accumulator).items()):
        if isinstance(value, StatisticAccumulator):
            _collect_accumulator_keys(value, registry, "%s.%s" % (path, name), visited)
        else:
            for i, child in enumerate(_iter_children(value)):
                if isinstance(child, StatisticAccumulator):
                    _collect_accumulator_keys(child, registry, "%s.%s[%d]" % (path, name, i), visited)


def _flag_annotation_mismatched_keys(estimator: ParameterEstimator, path: str, visited: set[int]) -> None:
    """Raise when a ``keys`` value's shape contradicts the family's own ctor declaration.

    Scalar families declare ``keys: str | None`` and use the value WHOLE as the shared-statistics
    dict key, while combinator families declare ``keys: tuple[...]`` and split it across their
    sub-accumulators. A tuple handed to a scalar family therefore never means what the caller
    intended (it ties as one opaque composite key instead of per-slot keys) -- and it used to slip
    through silently. The family's ``__init__`` annotation is the source of truth, so no per-family
    churn; families without an annotation are skipped.
    """
    import inspect

    obj_id = id(estimator)
    if obj_id in visited:
        return
    visited.add(obj_id)
    keys = getattr(estimator, "keys", None)
    if isinstance(keys, (list, tuple)):
        try:
            ann = inspect.signature(type(estimator).__init__).parameters["keys"].annotation
        except (ValueError, KeyError, TypeError):
            ann = inspect.Parameter.empty
        ann_text = ann if isinstance(ann, str) else str(ann)
        if ann is not inspect.Parameter.empty and "tuple" not in ann_text.lower() and "any" not in ann_text.lower():
            raise ValueError(
                "%s (%s) declares keys: %s but got the %s %r -- pass one shared string per site; "
                "tuple keys are the combinator convention only"
                % (path, type(estimator).__name__, ann_text, type(keys).__name__, keys)
            )
    for name, value in sorted(vars(estimator).items()):
        for child in _iter_children(value):
            if isinstance(child, ParameterEstimator):
                _flag_annotation_mismatched_keys(child, "%s.%s" % (path, name), visited)
        if isinstance(value, ParameterEstimator):
            _flag_annotation_mismatched_keys(value, "%s.%s" % (path, name), visited)


def validate_estimator_keys(estimator: ParameterEstimator) -> None:
    """Validate keyed estimator and accumulator sites before EM folds stats.

    The validator catches the classic keying footgun: two different families, or
    two sites with incompatible estimator settings, accidentally sharing the same
    key string.  Validation is intentionally protocol-level and best-effort; a
    family can still perform stricter checks in its own factory if needed.
    """
    _flag_annotation_mismatched_keys(estimator, type(estimator).__name__, set())
    estimator_registry: dict[Any, tuple[Any, str, int, Any]] = {}
    _collect_estimator_keys(estimator, estimator_registry, type(estimator).__name__, set())

    accumulator_registry: dict[Any, tuple[Any, str, int, Any]] = {}
    accumulator = estimator.accumulator_factory().make()
    _collect_accumulator_keys(accumulator, accumulator_registry, type(accumulator).__name__, set())

    missing_from_accumulator = set(estimator_registry) - set(accumulator_registry)
    missing_from_estimator = set(accumulator_registry) - set(estimator_registry)
    if missing_from_accumulator or missing_from_estimator:

        def describe(registry: dict[Any, tuple[Any, str, int, Any]], keys: set[Any]) -> list[str]:
            return ["%r at %s" % (registry[key][3], registry[key][1]) for key in sorted(keys, key=repr)]

        raise KeyValidationError(
            "Estimator/accumulator keyed sites disagree: estimator-only=%s; accumulator-only=%s."
            % (
                describe(estimator_registry, missing_from_accumulator),
                describe(accumulator_registry, missing_from_estimator),
            )
        )

    def protocol_family(signature: Any) -> tuple[str, str]:
        module, qualname = signature[0], signature[1]
        for suffix in ("Estimator", "Accumulator", "AccumulatorFactory"):
            if qualname.endswith(suffix):
                qualname = qualname[: -len(suffix)]
                break
        return module, qualname

    for canonical in estimator_registry:
        estimator_signature, estimator_path, estimator_count, original = estimator_registry[canonical]
        accumulator_signature, accumulator_path, accumulator_count, _ = accumulator_registry[canonical]
        if estimator_count != accumulator_count:
            raise KeyValidationError(
                "Key %r appears at %d estimator sites but %d accumulator merge sites (%s versus %s)."
                % (original, estimator_count, accumulator_count, estimator_path, accumulator_path)
            )
        if protocol_family(estimator_signature) != protocol_family(accumulator_signature):
            raise KeyValidationError(
                "Key %r routes estimator family %r at %s to accumulator family %r at %s."
                % (
                    original,
                    protocol_family(estimator_signature),
                    estimator_path,
                    protocol_family(accumulator_signature),
                    accumulator_path,
                )
            )


def estimator_has_keys(estimator: ParameterEstimator) -> bool:
    """Whether any site in the estimator tree carries a tying key.

    Keyed sites require the ``merge_accumulator_keys`` pass after accumulation; an EM driver that
    cannot run it (or whose update decomposition conflicts with pooling, e.g. block-EM's sparse
    per-component M-steps) must refuse or reroute keyed estimators rather than silently untie them.
    """
    registry: dict[Any, tuple[Any, str, int, Any]] = {}
    _collect_estimator_keys(estimator, registry, type(estimator).__name__, set())
    return bool(registry)


def validate_accumulator_keys(accumulator: StatisticAccumulator) -> None:
    """Validate keyed sites in an already-created accumulator tree."""
    accumulator_registry: dict[Any, tuple[Any, str, int, Any]] = {}
    _collect_accumulator_keys(accumulator, accumulator_registry, type(accumulator).__name__, set())


def merge_accumulator_keys(accumulator: StatisticAccumulator) -> None:
    """Pool keyed statistics across ``accumulator``'s tree -- the parameter-tying pass.

    Runs the ``key_merge``/``key_replace`` pair every EM driver applies exactly once after
    accumulation (see :func:`mixle.stats.compute.sequence.seq_estimate`), so sites sharing a key
    estimate from the pooled statistics.  A no-op when no site in the tree carries a key.  Call it
    on the fully-combined accumulator, never per shard: pooling twice would double-count the
    shared statistics on the second ``combine``.
    """
    stats_dict: dict[Any, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
