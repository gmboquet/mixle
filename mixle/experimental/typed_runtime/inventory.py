"""Machine-readable boundaries for the experimental typed runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RuntimeCapabilityStatus(StrEnum):
    """Maturity of one typed-runtime capability."""

    IMPLEMENTED = "implemented"
    NARROW = "narrow"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RuntimeCapability:
    """One capability claim with its operative boundary and source evidence."""

    name: str
    status: RuntimeCapabilityStatus
    boundary: str
    evidence: str

    def __post_init__(self) -> None:
        for field_name in ("name", "boundary", "evidence"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.status, RuntimeCapabilityStatus):
            raise TypeError("status must be a RuntimeCapabilityStatus value.")


_CAPABILITIES = (
    RuntimeCapability(
        "semantic_update_compiler",
        RuntimeCapabilityStatus.IMPLEMENTED,
        "Compiles inert model and estimator structure; execution admission still requires audited or explicit "
        "contract evidence.",
        "compiler.py and validation.py",
    ),
    RuntimeCapability(
        "local_mixture_execution",
        RuntimeCapabilityStatus.NARROW,
        "Observed-data MLE for finite mixtures on one process; it is not a general estimator executor.",
        "local.py",
    ),
    RuntimeCapability(
        "structured_estimator_execution",
        RuntimeCapabilityStatus.NARROW,
        "Declared structured axes on local NumPy CPU workers only.",
        "structured_execution.py",
    ),
    RuntimeCapability(
        "transactional_proposals",
        RuntimeCapabilityStatus.IMPLEMENTED,
        "Coordinates caller-provided participants, snapshots, objective evidence, and canaries.",
        "transaction.py",
    ),
    RuntimeCapability(
        "multi_host_transport",
        RuntimeCapabilityStatus.UNAVAILABLE,
        "No network transport or remote worker lifecycle is implemented.",
        "README.md: Not implemented yet",
    ),
    RuntimeCapability(
        "general_estimator_executor",
        RuntimeCapabilityStatus.UNAVAILABLE,
        "No adapter executes every estimator represented by a compiled update graph.",
        "README.md: Not implemented yet",
    ),
)


def runtime_capabilities() -> tuple[RuntimeCapability, ...]:
    """Return the immutable, machine-readable typed-runtime capability inventory."""

    return _CAPABILITIES


__all__ = [
    "RuntimeCapability",
    "RuntimeCapabilityStatus",
    "runtime_capabilities",
]
