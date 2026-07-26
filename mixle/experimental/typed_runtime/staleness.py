"""Typed stale-proposal admission, correction, and shrinkage policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import ConsistencyRequirement, UpdateContract
from mixle.experimental.typed_runtime.proposal import ProposalPacket, payload_fingerprint
from mixle.experimental.typed_runtime.transaction import RuntimeVersions


class StalenessAction(StrEnum):
    """Coordinator action for a locally computed proposal."""

    ACCEPT = "accept"
    SHRINK = "shrink"
    CORRECT = "correct"
    REJECT = "reject"


class CorrectionSemantics(StrEnum):
    """Whether a corrected payload is exact at the target state or approximate."""

    EXACT_REBASE = "exact_rebase"
    STATISTICAL_APPROXIMATION = "statistical_approximation"


@dataclass(frozen=True)
class CorrectionResult:
    """Identity-bound payload recomputed or approximated for target versions."""

    source_proposal_id: str
    source_payload_hash: str
    run_id: str
    model_id: str
    node_id: str
    target_model_version: int
    target_dependency_versions: tuple[tuple[str, int], ...]
    semantics: CorrectionSemantics
    method: str
    evidence_id: str
    payload: Any
    payload_hash: str = ""

    def __post_init__(self) -> None:
        strings = (
            self.source_proposal_id,
            self.source_payload_hash,
            self.run_id,
            self.model_id,
            self.node_id,
            self.method,
            self.evidence_id,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("correction result identities and evidence must be non-empty.")
        if self.target_model_version < 0:
            raise ValueError("correction target model version must be non-negative.")
        if not isinstance(self.semantics, CorrectionSemantics):
            raise TypeError("correction semantics must be a CorrectionSemantics value.")
        versions = self.target_dependency_versions
        if isinstance(versions, Mapping):
            versions = tuple(sorted((str(key), int(value)) for key, value in versions.items()))
            object.__setattr__(self, "target_dependency_versions", versions)
        if (
            len({key for key, _ in versions}) != len(versions)
            or any(not key or version < 0 for key, version in versions)
        ):
            raise ValueError("correction target dependency versions must be unique and non-negative.")
        computed_hash = payload_fingerprint(self.payload)
        if self.payload_hash and self.payload_hash != computed_hash:
            raise ValueError("correction payload_hash does not match the correction payload.")
        object.__setattr__(self, "payload_hash", computed_hash)

    @property
    def fingerprint(self) -> str:
        """Bind the exact correction payload, target state, method, and evidence."""

        return payload_fingerprint(
            (
                self.source_proposal_id,
                self.source_payload_hash,
                self.run_id,
                self.model_id,
                self.node_id,
                self.target_model_version,
                self.target_dependency_versions,
                self.semantics.value,
                self.method,
                self.evidence_id,
                self.payload_hash,
            )
        )


@dataclass(frozen=True)
class StalenessPolicy:
    """Hard lag limits and conservative stale-delta decay."""

    max_model_lag: int = 0
    max_node_lag: int = 0
    shrink_decay: float = 0.5

    def __post_init__(self) -> None:
        if self.max_model_lag < 0 or self.max_node_lag < 0:
            raise ValueError("staleness lag limits must be non-negative.")
        if not math.isfinite(self.shrink_decay) or self.shrink_decay <= 0.0:
            raise ValueError("shrink_decay must be finite and positive.")


@dataclass(frozen=True)
class StalenessReceipt:
    """Measured lag, consistency contract, and admission decision."""

    proposal_id: str
    action: StalenessAction
    reason: str
    model_lag: int
    node_lag: int
    scale: float
    correction_fingerprint: str | None
    correction_semantics: CorrectionSemantics | None
    source_payload_hash: str
    target_model_version: int
    target_dependency_versions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.reason or not self.source_payload_hash:
            raise ValueError("staleness receipt identities must be non-empty.")
        if not isinstance(self.action, StalenessAction):
            raise TypeError("staleness receipt action must be a StalenessAction value.")
        if self.correction_semantics is not None and not isinstance(
            self.correction_semantics, CorrectionSemantics
        ):
            raise TypeError("staleness receipt correction semantics are invalid.")
        if self.target_model_version < 0 or not math.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("staleness receipt target version and scale must be valid.")
        versions = self.target_dependency_versions
        if isinstance(versions, Mapping):
            versions = tuple(sorted((str(key), int(value)) for key, value in versions.items()))
            object.__setattr__(self, "target_dependency_versions", versions)
        if (
            len({key for key, _ in versions}) != len(versions)
            or any(not key or version < 0 for key, version in versions)
        ):
            raise ValueError("staleness target dependency versions must be unique and non-negative.")
        if self.action is StalenessAction.CORRECT:
            if self.correction_fingerprint is None or self.correction_semantics is None:
                raise ValueError("corrected staleness receipts require bound correction evidence.")
        elif self.correction_fingerprint is not None or self.correction_semantics is not None:
            raise ValueError("non-corrected staleness receipts cannot carry correction evidence.")

    @property
    def accepted(self) -> bool:
        """Whether the proposal may proceed to versioned commit preflight."""

        return self.action is not StalenessAction.REJECT

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible staleness receipt."""

        return {
            "proposal_id": self.proposal_id,
            "action": self.action.value,
            "accepted": self.accepted,
            "reason": self.reason,
            "model_lag": self.model_lag,
            "node_lag": self.node_lag,
            "scale": self.scale,
            "correction_fingerprint": self.correction_fingerprint,
            "correction_semantics": (
                self.correction_semantics.value
                if self.correction_semantics is not None
                else None
            ),
            "source_payload_hash": self.source_payload_hash,
            "target_model_version": self.target_model_version,
            "target_dependency_versions": dict(self.target_dependency_versions),
        }


def assess_staleness(
    proposal: ProposalPacket,
    contract: UpdateContract,
    current: RuntimeVersions,
    policy: StalenessPolicy,
    *,
    correction: CorrectionResult | None = None,
) -> StalenessReceipt:
    """Apply exact/strict/bounded/corrected semantics before payload mutation."""

    if payload_fingerprint(proposal.payload) != proposal.payload_hash:
        raise ValueError("proposal payload changed after its evidence was created.")
    target_versions = tuple(sorted(current.node_versions.items()))

    def result(
        action: StalenessAction,
        reason: str,
        model_lag: int,
        node_lag: int,
        scale: float,
        *,
        bound_correction: CorrectionResult | None = None,
    ) -> StalenessReceipt:
        return StalenessReceipt(
            proposal.proposal_id,
            action,
            reason,
            model_lag,
            node_lag,
            scale,
            bound_correction.fingerprint if bound_correction is not None else None,
            bound_correction.semantics if bound_correction is not None else None,
            proposal.payload_hash,
            current.model_version,
            target_versions,
        )

    if proposal.node_id not in current.node_versions:
        return result(StalenessAction.REJECT, "unknown-node", 0, 0, 0.0)
    proposal_node_version = proposal.dependency_version_map.get(proposal.node_id)
    if proposal_node_version is None:
        return result(StalenessAction.REJECT, "missing-node-version", 0, 0, 0.0)
    model_lag = current.model_version - proposal.base_model_version
    node_lag = current.node_versions[proposal.node_id] - proposal_node_version
    if model_lag < 0 or node_lag < 0:
        return result(StalenessAction.REJECT, "future-version", model_lag, node_lag, 0.0)
    if model_lag == 0 and node_lag == 0:
        if correction is not None:
            raise ValueError("a current-version proposal cannot carry stale correction evidence.")
        return result(StalenessAction.ACCEPT, "current-version", 0, 0, 1.0)
    if contract.exact:
        return result(
            StalenessAction.REJECT,
            "exact-update-requires-current-version",
            model_lag,
            node_lag,
            0.0,
        )
    if contract.consistency in (ConsistencyRequirement.STRICT_SYNCHRONOUS, ConsistencyRequirement.LOCAL_ONLY):
        return result(
            StalenessAction.REJECT,
            "consistency-requires-current-version",
            model_lag,
            node_lag,
            0.0,
        )
    if model_lag > policy.max_model_lag or node_lag > policy.max_node_lag:
        return result(StalenessAction.REJECT, "lag-bound-exceeded", model_lag, node_lag, 0.0)
    total_lag = model_lag + node_lag
    scale = math.exp(-policy.shrink_decay * total_lag)
    if contract.consistency is ConsistencyRequirement.CORRECTED_EVENTUAL:
        if correction is None:
            return result(
                StalenessAction.REJECT,
                "missing-drift-correction",
                model_lag,
                node_lag,
                0.0,
            )
        source_identity = (
            correction.source_proposal_id,
            correction.source_payload_hash,
            correction.run_id,
            correction.model_id,
            correction.node_id,
        )
        expected_source = (
            proposal.proposal_id,
            proposal.payload_hash,
            proposal.run_id,
            proposal.model_id,
            proposal.node_id,
        )
        if source_identity != expected_source:
            raise ValueError("correction result does not match the source proposal identity.")
        if payload_fingerprint(correction.payload) != correction.payload_hash:
            raise ValueError("correction payload changed after its evidence was created.")
        if (
            correction.target_model_version != current.model_version
            or correction.target_dependency_versions != target_versions
        ):
            raise ValueError("correction result does not match the complete target version vector.")
        correction_scale = (
            1.0
            if correction.semantics is CorrectionSemantics.EXACT_REBASE
            else scale
        )
        return result(
            StalenessAction.CORRECT,
            (
                "exact-drift-rebase"
                if correction.semantics is CorrectionSemantics.EXACT_REBASE
                else "bounded-lag-with-statistical-drift-correction"
            ),
            model_lag,
            node_lag,
            correction_scale,
            bound_correction=correction,
        )
    if correction is not None:
        raise ValueError("bounded-stale proposals cannot carry corrected-eventual evidence.")
    return result(
        StalenessAction.SHRINK,
        "bounded-stale-statistical-approximation",
        model_lag,
        node_lag,
        scale,
    )


def _scale_payload(value: Any, scale: float) -> Any:
    if isinstance(value, np.ndarray):
        return value * scale
    if isinstance(value, (int, float, np.number)):
        return value * scale
    if isinstance(value, Mapping):
        return {key: _scale_payload(item, scale) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_scale_payload(item, scale) for item in value)
    if isinstance(value, list):
        return [_scale_payload(item, scale) for item in value]
    raise TypeError("cannot shrink proposal payload type %s." % type(value).__name__)


def shrink_proposal(
    proposal: ProposalPacket,
    receipt: StalenessReceipt,
    *,
    proposal_id: str,
    correction: CorrectionResult | None = None,
) -> ProposalPacket:
    """Create an identity-bound proposal after an admitted stale transformation."""

    if receipt.proposal_id != proposal.proposal_id:
        raise ValueError("staleness receipt does not belong to proposal.")
    if receipt.source_payload_hash != proposal.payload_hash:
        raise ValueError("staleness receipt does not match the source proposal payload.")
    if receipt.action not in (StalenessAction.SHRINK, StalenessAction.CORRECT):
        raise ValueError("only shrink/correct decisions can transform a proposal.")
    if not 0.0 < receipt.scale <= 1.0:
        raise ValueError("admitted stale proposal scale must be in (0, 1].")
    if receipt.action is StalenessAction.CORRECT:
        if correction is None:
            raise ValueError("a corrected-eventual proposal requires a bound correction result.")
        if payload_fingerprint(correction.payload) != correction.payload_hash:
            raise ValueError("correction payload changed after its evidence was created.")
        if receipt.correction_fingerprint != correction.fingerprint:
            raise ValueError("staleness receipt does not bind the supplied correction result.")
        if receipt.correction_semantics is not correction.semantics:
            raise ValueError("staleness receipt and correction semantics disagree.")
        if (
            correction.target_model_version != receipt.target_model_version
            or correction.target_dependency_versions != receipt.target_dependency_versions
        ):
            raise ValueError("staleness receipt and correction target versions disagree.")
        exact_rebase = correction.semantics is CorrectionSemantics.EXACT_REBASE
        if exact_rebase and receipt.scale != 1.0:
            raise ValueError("an exact correction cannot be statistically shrunk.")
        payload = correction.payload if exact_rebase else _scale_payload(correction.payload, receipt.scale)
        staleness_semantics = (
            "exact_rebase" if exact_rebase else "corrected_statistical_approximation"
        )
        predicted_gain = None
        correction_fingerprint = correction.fingerprint
    else:
        if correction is not None:
            raise ValueError("bounded-stale shrinkage cannot carry a correction result.")
        payload = _scale_payload(proposal.payload, receipt.scale)
        staleness_semantics = "bounded_stale_approximation"
        predicted_gain = (
            proposal.predicted_gain * receipt.scale
            if proposal.predicted_gain is not None
            else None
        )
        correction_fingerprint = None
    return replace(
        proposal,
        proposal_id=proposal_id,
        base_model_version=receipt.target_model_version,
        dependency_versions=receipt.target_dependency_versions,
        payload=payload,
        predicted_gain=predicted_gain,
        measured_gain=None,
        staleness_semantics=staleness_semantics,
        correction_fingerprint=correction_fingerprint,
        payload_hash="",
    )


__all__ = [
    "CorrectionResult",
    "CorrectionSemantics",
    "StalenessAction",
    "StalenessPolicy",
    "StalenessReceipt",
    "assess_staleness",
    "shrink_proposal",
]
