"""Prior specifications for declaration-backed MAP fitting.

The classes in this module are lightweight, serializable descriptions of
common priors.  They intentionally do not depend on NumPy, Torch, or a concrete
compute engine; fitting code turns them into backend tensors at objective time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImproperPriorReceipt:
    """Explicit acknowledgement required to construct an improper limiting prior.

    Improper priors are not probability distributions and can yield an improper posterior.
    The non-empty rationale makes that deliberate modeling choice durable in serialized prior
    payloads instead of letting a zero hyperparameter silently change the statistical contract.
    """

    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("an improper-prior receipt requires a non-empty rationale.")

    def as_dict(self) -> dict[str, Any]:
        """Return the durable improper-limit acknowledgement."""
        return {
            "status": "improper_limit_acknowledged",
            "rationale": self.rationale.strip(),
        }


def improper(rationale: str) -> ImproperPriorReceipt:
    """Create the explicit receipt required for a supported improper limiting prior."""
    return ImproperPriorReceipt(rationale)


def _validated_scalar(
    name: str,
    value: Any,
    *,
    positive: bool,
    improper_receipt: ImproperPriorReceipt | None = None,
) -> float:
    if improper_receipt is not None and not isinstance(improper_receipt, ImproperPriorReceipt):
        raise TypeError("improper_receipt must be an ImproperPriorReceipt.")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    if positive and converted <= 0.0:
        if converted < 0.0 or improper_receipt is None:
            raise ValueError(f"{name} must be positive for a proper prior.")
    return converted


def _validate_receipt_scope(values: Sequence[Any], receipt: ImproperPriorReceipt | None) -> None:
    has_improper_limit = any(float(value) == 0.0 for value in values)
    if receipt is not None and not has_improper_limit:
        raise ValueError("improper_receipt was supplied, but every hyperparameter defines a proper prior.")


def _concentrations(alpha: Any) -> list[Any]:
    if isinstance(alpha, Mapping):
        values = list(alpha.values())
    elif hasattr(alpha, "tolist"):
        values = alpha.tolist()
        if not isinstance(values, list):
            values = [values]
    elif isinstance(alpha, Sequence) and not isinstance(alpha, (str, bytes, bytearray)):
        values = list(alpha)
    else:
        raise TypeError("Dirichlet alpha must be a non-empty one-dimensional sequence or mapping.")
    if not values or any(
        isinstance(value, Mapping) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)))
        for value in values
    ):
        raise ValueError("Dirichlet alpha must be a non-empty one-dimensional concentration vector.")
    return values


def _prior_status(receipt: ImproperPriorReceipt | None) -> dict[str, Any]:
    return {
        "proper": receipt is None,
        "improper_receipt": None if receipt is None else receipt.as_dict(),
    }


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
class PriorAlignmentReceipt:
    """Durable acknowledgement that a structural prior is partial or broadcast."""

    mode: str
    rationale: str

    def __post_init__(self) -> None:
        if self.mode not in ("partial", "broadcast"):
            raise ValueError("prior alignment mode must be 'partial' or 'broadcast'.")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("non-exact prior alignment requires a non-empty rationale.")

    def as_dict(self) -> dict[str, str]:
        """Return the serialized alignment acknowledgement."""
        return {"mode": self.mode, "rationale": self.rationale.strip()}


def _alignment_payload(receipt: PriorAlignmentReceipt | None) -> dict[str, Any]:
    if receipt is None:
        return {}
    if not isinstance(receipt, PriorAlignmentReceipt):
        raise TypeError("alignment_receipt must be a PriorAlignmentReceipt.")
    return {"alignment_receipt": receipt.as_dict()}


@dataclass(frozen=True)
class AlignedPrior:
    """Apply one prior through an explicitly acknowledged alignment mode."""

    prior: Any
    alignment_receipt: PriorAlignmentReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.alignment_receipt, PriorAlignmentReceipt):
            raise TypeError("alignment_receipt must be a PriorAlignmentReceipt.")
        if self.alignment_receipt.mode != "broadcast":
            raise ValueError("AlignedPrior requires a broadcast alignment receipt.")

    def as_dict(self) -> dict[str, Any]:
        """Return the wrapped prior and its durable alignment receipt."""
        return {
            "family": "aligned",
            "prior": as_prior_dict(self.prior),
            **_alignment_payload(self.alignment_receipt),
        }


def partial_alignment(rationale: str) -> PriorAlignmentReceipt:
    """Acknowledge that unspecified structural positions intentionally have no prior."""
    return PriorAlignmentReceipt("partial", rationale)


def broadcast_alignment(rationale: str) -> PriorAlignmentReceipt:
    """Acknowledge that one prior intentionally applies to every structural position."""
    return PriorAlignmentReceipt("broadcast", rationale)


def broadcast(prior: Any, rationale: str) -> AlignedPrior:
    """Wrap a prior for intentional broadcast across a model structure."""
    return AlignedPrior(prior, broadcast_alignment(rationale))


@dataclass(frozen=True)
class NormalGammaPrior:
    """Normal-Gamma prior for Gaussian ``mu`` and precision ``tau``.

    The density is proportional to
    ``tau ** (alpha - 1) exp(-beta tau) sqrt(tau)
    exp(-0.5 * kappa * tau * (mu - mu0) ** 2)``.  Normalizing constants are
    omitted because MAP fitting only needs objective differences.
    """

    mu0: float = 0.0
    kappa: float = 1.0e-6
    alpha: float = 2.0
    beta: float = 1.0
    improper_receipt: ImproperPriorReceipt | None = None

    def __post_init__(self) -> None:
        _validated_scalar("NormalGammaPrior.mu0", self.mu0, positive=False)
        _validated_scalar(
            "NormalGammaPrior.kappa",
            self.kappa,
            positive=True,
            improper_receipt=self.improper_receipt,
        )
        _validated_scalar(
            "NormalGammaPrior.alpha",
            self.alpha,
            positive=True,
            improper_receipt=self.improper_receipt,
        )
        _validated_scalar(
            "NormalGammaPrior.beta",
            self.beta,
            positive=True,
            improper_receipt=self.improper_receipt,
        )
        _validate_receipt_scope((self.kappa, self.alpha, self.beta), self.improper_receipt)

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
            **_prior_status(self.improper_receipt),
        }

    def to_distribution(self) -> Any:
        """Convert a proper specification to the stats-layer conjugate distribution."""
        if self.improper_receipt is not None:
            raise ValueError("an improper NormalGammaPrior cannot use the analytic conjugate estimator.")
        from mixle.stats.bayes.normal_gamma import NormalGammaDistribution

        return NormalGammaDistribution(self.mu0, self.kappa, self.alpha, self.beta)


@dataclass(frozen=True)
class DirichletPrior:
    """Dirichlet prior for simplex-valued parameters."""

    alpha: Any
    improper_receipt: ImproperPriorReceipt | None = None

    def __post_init__(self) -> None:
        values = _concentrations(self.alpha)
        for index, value in enumerate(values):
            _validated_scalar(
                f"DirichletPrior.alpha[{index}]",
                value,
                positive=True,
                improper_receipt=self.improper_receipt,
            )
        _validate_receipt_scope(values, self.improper_receipt)

    def as_dict(self) -> dict:
        """Return the normalized Dirichlet prior payload."""
        return {
            "family": "dirichlet",
            "alpha": self.alpha,
            **_prior_status(self.improper_receipt),
        }


@dataclass(frozen=True)
class BetaPrior:
    """Beta prior for unit-interval parameters."""

    alpha: float
    beta: float
    parameter: str | None = None
    improper_receipt: ImproperPriorReceipt | None = None

    def __post_init__(self) -> None:
        _validated_scalar(
            "BetaPrior.alpha",
            self.alpha,
            positive=True,
            improper_receipt=self.improper_receipt,
        )
        _validated_scalar(
            "BetaPrior.beta",
            self.beta,
            positive=True,
            improper_receipt=self.improper_receipt,
        )
        _validate_receipt_scope((self.alpha, self.beta), self.improper_receipt)

    def as_dict(self) -> dict:
        """Return the normalized Beta prior payload."""
        rv = {
            "family": "beta",
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            **_prior_status(self.improper_receipt),
        }
        if self.parameter is not None:
            rv["parameter"] = self.parameter
        return rv


@dataclass(frozen=True)
class GammaPrior:
    """Gamma prior for positive scalar parameters or ordered-bound deltas."""

    shape: float
    rate: float
    parameter: str | None = None
    improper_receipt: ImproperPriorReceipt | None = None

    def __post_init__(self) -> None:
        _validated_scalar(
            "GammaPrior.shape",
            self.shape,
            positive=True,
            improper_receipt=self.improper_receipt,
        )
        _validated_scalar(
            "GammaPrior.rate",
            self.rate,
            positive=True,
            improper_receipt=self.improper_receipt,
        )
        _validate_receipt_scope((self.shape, self.rate), self.improper_receipt)

    def as_dict(self) -> dict:
        """Return the normalized Gamma prior payload."""
        rv = {
            "family": "gamma",
            "shape": float(self.shape),
            "rate": float(self.rate),
            **_prior_status(self.improper_receipt),
        }
        if self.parameter is not None:
            rv["parameter"] = self.parameter
        return rv


@dataclass(frozen=True)
class CompositePrior:
    """Child priors for a ``CompositeDistribution``."""

    children: Sequence[Any]
    alignment_receipt: PriorAlignmentReceipt | None = None

    def __post_init__(self) -> None:
        _alignment_payload(self.alignment_receipt)

    def as_dict(self) -> dict:
        """Return child priors as a plain composite-prior payload."""
        return {
            "family": "composite",
            "children": tuple(as_prior_dict(p) for p in self.children),
            **_alignment_payload(self.alignment_receipt),
        }


@dataclass(frozen=True)
class ConditionalPrior:
    """Per-key, default, and given priors for a ``ConditionalDistribution``."""

    conditions: Mapping[Any, Any]
    default: Any | None = None
    given: Any | None = None
    alignment_receipt: PriorAlignmentReceipt | None = None

    def __post_init__(self) -> None:
        _alignment_payload(self.alignment_receipt)

    def as_dict(self) -> dict:
        """Return keyed/default/given priors as a plain payload."""
        return {
            "family": "conditional",
            "conditions": {key: as_prior_dict(value) for key, value in self.conditions.items()},
            "default": as_prior_dict(self.default),
            "given": as_prior_dict(self.given),
            **_alignment_payload(self.alignment_receipt),
        }


@dataclass(frozen=True)
class MixturePrior:
    """Component and weight priors for a ``MixtureDistribution``."""

    components: Sequence[Any] = ()
    weights: Any | None = None
    alignment_receipt: PriorAlignmentReceipt | None = None

    def __post_init__(self) -> None:
        _alignment_payload(self.alignment_receipt)

    def as_dict(self) -> dict:
        """Return component and weight priors as a plain payload."""
        return {
            "family": "mixture",
            "components": tuple(as_prior_dict(p) for p in self.components),
            "weights": as_prior_dict(self.weights),
            **_alignment_payload(self.alignment_receipt),
        }


@dataclass(frozen=True)
class MarkovChainPrior:
    """Initial, transition-row, and length priors for ``MarkovChainDistribution``."""

    initial: Any | None = None
    transitions: Mapping[Any, Any] | None = None
    length: Any | None = None
    alignment_receipt: PriorAlignmentReceipt | None = None

    def __post_init__(self) -> None:
        _alignment_payload(self.alignment_receipt)

    def as_dict(self) -> dict:
        """Return initial, transition, and length priors as a plain payload."""
        return {
            "family": "markov_chain",
            "initial": as_prior_dict(self.initial),
            "transitions": {}
            if self.transitions is None
            else {key: as_prior_dict(value) for key, value in self.transitions.items()},
            "length": as_prior_dict(self.length),
            **_alignment_payload(self.alignment_receipt),
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
    alignment_receipt: PriorAlignmentReceipt | None = None

    def __post_init__(self) -> None:
        _alignment_payload(self.alignment_receipt)

    def as_dict(self) -> dict:
        """Return field priors as a plain record-prior payload."""
        return {
            "family": "record",
            "fields": {key: as_prior_dict(value) for key, value in self.fields.items()},
            **_alignment_payload(self.alignment_receipt),
        }


def normal_gamma(
    mu0: float = 0.0,
    kappa: float = 1.0e-6,
    alpha: float = 2.0,
    beta: float = 1.0,
    improper_receipt: ImproperPriorReceipt | None = None,
) -> NormalGammaPrior:
    """Create a Normal-Gamma prior for Gaussian mean/precision parameters."""
    return NormalGammaPrior(
        mu0=mu0,
        kappa=kappa,
        alpha=alpha,
        beta=beta,
        improper_receipt=improper_receipt,
    )


def dirichlet(alpha: Any, improper_receipt: ImproperPriorReceipt | None = None) -> DirichletPrior:
    """Create a Dirichlet prior for simplex-valued parameters."""
    return DirichletPrior(alpha=alpha, improper_receipt=improper_receipt)


def beta(
    alpha: float,
    beta: float,
    parameter: str | None = None,
    improper_receipt: ImproperPriorReceipt | None = None,
) -> BetaPrior:
    """Create a Beta prior for a unit-interval parameter."""
    return BetaPrior(
        alpha=alpha,
        beta=beta,
        parameter=parameter,
        improper_receipt=improper_receipt,
    )


def gamma(
    shape: float,
    rate: float,
    parameter: str | None = None,
    improper_receipt: ImproperPriorReceipt | None = None,
) -> GammaPrior:
    """Create a Gamma prior for a positive parameter."""
    return GammaPrior(
        shape=shape,
        rate=rate,
        parameter=parameter,
        improper_receipt=improper_receipt,
    )


def composite(children: Sequence[Any], alignment_receipt: PriorAlignmentReceipt | None = None) -> CompositePrior:
    """Create a Composite prior from child prior specifications."""
    return CompositePrior(children=children, alignment_receipt=alignment_receipt)


def conditional(
    conditions: Mapping[Any, Any],
    default: Any | None = None,
    given: Any | None = None,
    alignment_receipt: PriorAlignmentReceipt | None = None,
) -> ConditionalPrior:
    """Create a Conditional prior over keyed/default/given child priors."""
    return ConditionalPrior(
        conditions=conditions,
        default=default,
        given=given,
        alignment_receipt=alignment_receipt,
    )


def mixture(
    components: Sequence[Any] = (),
    weights: Any | None = None,
    alignment_receipt: PriorAlignmentReceipt | None = None,
) -> MixturePrior:
    """Create a Mixture prior over component and weight priors."""
    return MixturePrior(components=components, weights=weights, alignment_receipt=alignment_receipt)


def markov_chain(
    initial: Any | None = None,
    transitions: Mapping[Any, Any] | None = None,
    length: Any | None = None,
    alignment_receipt: PriorAlignmentReceipt | None = None,
) -> MarkovChainPrior:
    """Create a Markov-chain prior over initial, transition, and length terms."""
    return MarkovChainPrior(
        initial=initial,
        transitions=transitions,
        length=length,
        alignment_receipt=alignment_receipt,
    )


def optional(observed: Any | None = None, missing: Any | None = None) -> OptionalPrior:
    """Create an Optional prior over observed child and missingness terms."""
    return OptionalPrior(observed=observed, missing=missing)


def record(fields: Mapping[Any, Any], alignment_receipt: PriorAlignmentReceipt | None = None) -> RecordPrior:
    """Create a Record prior from field-name prior specifications."""
    return RecordPrior(fields=fields, alignment_receipt=alignment_receipt)


__all__ = [
    "AlignedPrior",
    "BetaPrior",
    "ConditionalPrior",
    "CompositePrior",
    "DirichletPrior",
    "GammaPrior",
    "ImproperPriorReceipt",
    "MarkovChainPrior",
    "MixturePrior",
    "NormalGammaPrior",
    "OptionalPrior",
    "PriorAlignmentReceipt",
    "RecordPrior",
    "as_prior_dict",
    "beta",
    "broadcast",
    "broadcast_alignment",
    "conditional",
    "composite",
    "dirichlet",
    "gamma",
    "improper",
    "markov_chain",
    "mixture",
    "normal_gamma",
    "optional",
    "partial_alignment",
    "record",
]
