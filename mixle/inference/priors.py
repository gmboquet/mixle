"""Prior specifications for declaration-backed MAP fitting.

The classes in this module are lightweight, serializable descriptions of
common priors.  They intentionally do not depend on NumPy, Torch, or a concrete
compute engine; fitting code turns them into backend tensors at objective time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def as_prior_dict(prior: Any) -> Any:
    """Return a plain-Python representation of a prior specification."""
    if prior is None:
        return None
    as_dict = getattr(prior, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    if isinstance(prior, Mapping):
        return {key: as_prior_dict(value) for key, value in prior.items()}
    if isinstance(prior, tuple):
        return tuple(as_prior_dict(value) for value in prior)
    if isinstance(prior, list):
        return [as_prior_dict(value) for value in prior]
    return prior


@dataclass(frozen=True)
class NormalGammaPrior:
    """Normal-Gamma prior for Gaussian ``mu`` and precision ``tau``.

    The density is proportional to
    ``tau ** (alpha - 1) exp(-beta tau) sqrt(tau)
    exp(-0.5 * kappa * tau * (mu - mu0) ** 2)``.  Normalizing constants are
    omitted because MAP fitting only needs objective differences.
    """

    mu0: float = 0.0
    kappa: float = 0.0
    alpha: float = 1.0
    beta: float = 0.0

    def as_dict(self) -> dict:
        """Return the normalized prior payload consumed by MAP fitters."""
        return {
            "family": "normalgamma",
            "mu0": float(self.mu0),
            "kappa": float(self.kappa),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            # Backward-compatible aliases for the legacy dict API.
            "a": float(self.alpha),
            "b": float(self.beta),
        }


@dataclass(frozen=True)
class DirichletPrior:
    """Dirichlet prior for simplex-valued parameters."""

    alpha: Any

    def as_dict(self) -> dict:
        """Return the normalized Dirichlet prior payload."""
        return {"family": "dirichlet", "alpha": self.alpha}


@dataclass(frozen=True)
class BetaPrior:
    """Beta prior for unit-interval parameters."""

    alpha: float
    beta: float
    parameter: str | None = None

    def as_dict(self) -> dict:
        """Return the normalized Beta prior payload."""
        rv = {"family": "beta", "alpha": float(self.alpha), "beta": float(self.beta)}
        if self.parameter is not None:
            rv["parameter"] = self.parameter
        return rv


@dataclass(frozen=True)
class GammaPrior:
    """Gamma prior for positive scalar parameters or ordered-bound deltas."""

    shape: float
    rate: float
    parameter: str | None = None

    def as_dict(self) -> dict:
        """Return the normalized Gamma prior payload."""
        rv = {"family": "gamma", "shape": float(self.shape), "rate": float(self.rate)}
        if self.parameter is not None:
            rv["parameter"] = self.parameter
        return rv


@dataclass(frozen=True)
class CompositePrior:
    """Child priors for a ``CompositeDistribution``."""

    children: Sequence[Any]

    def as_dict(self) -> dict:
        """Return child priors as a plain composite-prior payload."""
        return {"family": "composite", "children": tuple(as_prior_dict(p) for p in self.children)}


@dataclass(frozen=True)
class ConditionalPrior:
    """Per-key, default, and given priors for a ``ConditionalDistribution``."""

    conditions: Mapping[Any, Any]
    default: Any | None = None
    given: Any | None = None

    def as_dict(self) -> dict:
        """Return keyed/default/given priors as a plain payload."""
        return {
            "family": "conditional",
            "conditions": {key: as_prior_dict(value) for key, value in self.conditions.items()},
            "default": as_prior_dict(self.default),
            "given": as_prior_dict(self.given),
        }


@dataclass(frozen=True)
class MixturePrior:
    """Component and weight priors for a ``MixtureDistribution``."""

    components: Sequence[Any] = ()
    weights: Any | None = None

    def as_dict(self) -> dict:
        """Return component and weight priors as a plain payload."""
        return {
            "family": "mixture",
            "components": tuple(as_prior_dict(p) for p in self.components),
            "weights": as_prior_dict(self.weights),
        }


@dataclass(frozen=True)
class MarkovChainPrior:
    """Initial, transition-row, and length priors for ``MarkovChainDistribution``."""

    initial: Any | None = None
    transitions: Mapping[Any, Any] | None = None
    length: Any | None = None

    def as_dict(self) -> dict:
        """Return initial, transition, and length priors as a plain payload."""
        return {
            "family": "markov_chain",
            "initial": as_prior_dict(self.initial),
            "transitions": {}
            if self.transitions is None
            else {key: as_prior_dict(value) for key, value in self.transitions.items()},
            "length": as_prior_dict(self.length),
        }


@dataclass(frozen=True)
class OptionalPrior:
    """Observed-child and missing-probability priors for ``OptionalDistribution``."""

    observed: Any | None = None
    missing: Any | None = None

    def as_dict(self) -> dict:
        """Return observed-child and missingness priors as a plain payload."""
        return {
            "family": "optional",
            "observed": as_prior_dict(self.observed),
            "missing": as_prior_dict(self.missing),
        }


@dataclass(frozen=True)
class RecordPrior:
    """Field priors for a named ``RecordDistribution``."""

    fields: Mapping[Any, Any]

    def as_dict(self) -> dict:
        """Return field priors as a plain record-prior payload."""
        return {
            "family": "record",
            "fields": {key: as_prior_dict(value) for key, value in self.fields.items()},
        }


def normal_gamma(mu0: float = 0.0, kappa: float = 0.0, alpha: float = 1.0, beta: float = 0.0) -> NormalGammaPrior:
    """Create a Normal-Gamma prior for Gaussian mean/precision parameters."""
    return NormalGammaPrior(mu0=mu0, kappa=kappa, alpha=alpha, beta=beta)


def dirichlet(alpha: Any) -> DirichletPrior:
    """Create a Dirichlet prior for simplex-valued parameters."""
    return DirichletPrior(alpha=alpha)


def beta(alpha: float, beta: float, parameter: str | None = None) -> BetaPrior:
    """Create a Beta prior for a unit-interval parameter."""
    return BetaPrior(alpha=alpha, beta=beta, parameter=parameter)


def gamma(shape: float, rate: float, parameter: str | None = None) -> GammaPrior:
    """Create a Gamma prior for a positive parameter."""
    return GammaPrior(shape=shape, rate=rate, parameter=parameter)


def composite(children: Sequence[Any]) -> CompositePrior:
    """Create a Composite prior from child prior specifications."""
    return CompositePrior(children=children)


def conditional(
    conditions: Mapping[Any, Any], default: Any | None = None, given: Any | None = None
) -> ConditionalPrior:
    """Create a Conditional prior over keyed/default/given child priors."""
    return ConditionalPrior(conditions=conditions, default=default, given=given)


def mixture(components: Sequence[Any] = (), weights: Any | None = None) -> MixturePrior:
    """Create a Mixture prior over component and weight priors."""
    return MixturePrior(components=components, weights=weights)


def markov_chain(
    initial: Any | None = None, transitions: Mapping[Any, Any] | None = None, length: Any | None = None
) -> MarkovChainPrior:
    """Create a Markov-chain prior over initial, transition, and length terms."""
    return MarkovChainPrior(initial=initial, transitions=transitions, length=length)


def optional(observed: Any | None = None, missing: Any | None = None) -> OptionalPrior:
    """Create an Optional prior over observed child and missingness terms."""
    return OptionalPrior(observed=observed, missing=missing)


def record(fields: Mapping[Any, Any]) -> RecordPrior:
    """Create a Record prior from field-name prior specifications."""
    return RecordPrior(fields=fields)


__all__ = [
    "BetaPrior",
    "ConditionalPrior",
    "CompositePrior",
    "DirichletPrior",
    "GammaPrior",
    "MarkovChainPrior",
    "MixturePrior",
    "NormalGammaPrior",
    "OptionalPrior",
    "RecordPrior",
    "as_prior_dict",
    "beta",
    "conditional",
    "composite",
    "dirichlet",
    "gamma",
    "markov_chain",
    "mixture",
    "normal_gamma",
    "optional",
    "record",
]
