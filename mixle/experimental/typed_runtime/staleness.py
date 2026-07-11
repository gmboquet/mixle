"""Typed stale-proposal admission, correction, and shrinkage policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import ConsistencyRequirement, UpdateContract
from mixle.experimental.typed_runtime.proposal import ProposalPacket
from mixle.experimental.typed_runtime.transaction import RuntimeVersions


class StalenessAction(StrEnum):
    """Coordinator action for a locally computed proposal."""

    ACCEPT = "accept"
    SHRINK = "shrink"
    CORRECT = "correct"
    REJECT = "reject"


@dataclass(frozen=True)
class StalenessPolicy:
    """Hard lag limits and conservative stale-delta decay."""

    max_model_lag: int = 0
    max_node_lag: int = 0
    shrink_decay: float = 0.5

    def __post_init__(self) -> None:
        if self.max_model_lag < 0 or self.max_node_lag < 0:
            raise ValueError("staleness lag limits must be non-negative.")
        if not math.isfinite(self.shrink_decay) or self.shrink_decay < 0.0:
            raise ValueError("shrink_decay must be finite and non-negative.")


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
        }


def assess_staleness(
    proposal: ProposalPacket,
    contract: UpdateContract,
    current: RuntimeVersions,
    policy: StalenessPolicy,
    *,
    correction_fingerprint: str | None = None,
) -> StalenessReceipt:
    """Apply exact/strict/bounded/corrected semantics before payload mutation."""

    if proposal.node_id not in current.node_versions:
        return StalenessReceipt(proposal.proposal_id, StalenessAction.REJECT, "unknown-node", 0, 0, 0.0, None)
    proposal_node_version = proposal.dependency_version_map.get(proposal.node_id)
    if proposal_node_version is None:
        return StalenessReceipt(
            proposal.proposal_id,
            StalenessAction.REJECT,
            "missing-node-version",
            0,
            0,
            0.0,
            correction_fingerprint,
        )
    model_lag = current.model_version - proposal.base_model_version
    node_lag = current.node_versions[proposal.node_id] - proposal_node_version
    if model_lag < 0 or node_lag < 0:
        return StalenessReceipt(
            proposal.proposal_id,
            StalenessAction.REJECT,
            "future-version",
            model_lag,
            node_lag,
            0.0,
            correction_fingerprint,
        )
    if model_lag == 0 and node_lag == 0:
        return StalenessReceipt(
            proposal.proposal_id,
            StalenessAction.ACCEPT,
            "current-version",
            0,
            0,
            1.0,
            correction_fingerprint,
        )
    if contract.exact:
        return StalenessReceipt(
            proposal.proposal_id,
            StalenessAction.REJECT,
            "exact-update-requires-current-version",
            model_lag,
            node_lag,
            0.0,
            correction_fingerprint,
        )
    if contract.consistency in (ConsistencyRequirement.STRICT_SYNCHRONOUS, ConsistencyRequirement.LOCAL_ONLY):
        return StalenessReceipt(
            proposal.proposal_id,
            StalenessAction.REJECT,
            "consistency-requires-current-version",
            model_lag,
            node_lag,
            0.0,
            correction_fingerprint,
        )
    if model_lag > policy.max_model_lag or node_lag > policy.max_node_lag:
        return StalenessReceipt(
            proposal.proposal_id,
            StalenessAction.REJECT,
            "lag-bound-exceeded",
            model_lag,
            node_lag,
            0.0,
            correction_fingerprint,
        )
    total_lag = model_lag + node_lag
    scale = math.exp(-policy.shrink_decay * total_lag)
    if contract.consistency is ConsistencyRequirement.CORRECTED_EVENTUAL:
        if not correction_fingerprint:
            return StalenessReceipt(
                proposal.proposal_id,
                StalenessAction.REJECT,
                "missing-drift-correction",
                model_lag,
                node_lag,
                0.0,
                None,
            )
        return StalenessReceipt(
            proposal.proposal_id,
            StalenessAction.CORRECT,
            "bounded-lag-with-drift-correction",
            model_lag,
            node_lag,
            scale,
            correction_fingerprint,
        )
    return StalenessReceipt(
        proposal.proposal_id,
        StalenessAction.SHRINK,
        "bounded-stale-delta",
        model_lag,
        node_lag,
        scale,
        correction_fingerprint,
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
    corrected_payload: Any | None = None,
) -> ProposalPacket:
    """Create a newly fingerprinted payload after an admitted stale-delta shrink."""

    if receipt.proposal_id != proposal.proposal_id:
        raise ValueError("staleness receipt does not belong to proposal.")
    if receipt.action not in (StalenessAction.SHRINK, StalenessAction.CORRECT):
        raise ValueError("only shrink/correct decisions can transform a proposal.")
    if not 0.0 < receipt.scale <= 1.0:
        raise ValueError("admitted stale proposal scale must be in (0, 1].")
    if receipt.action is StalenessAction.CORRECT and corrected_payload is None:
        raise ValueError("a corrected-eventual proposal requires a corrected_payload.")
    payload = corrected_payload if corrected_payload is not None else proposal.payload
    dependency_versions = proposal.dependency_version_map
    dependency_versions[proposal.node_id] += receipt.node_lag
    return replace(
        proposal,
        proposal_id=proposal_id,
        base_model_version=proposal.base_model_version + receipt.model_lag,
        dependency_versions=dependency_versions,
        payload=_scale_payload(payload, receipt.scale),
        predicted_gain=(proposal.predicted_gain * receipt.scale if proposal.predicted_gain is not None else None),
        measured_gain=None,
        payload_hash="",
    )


__all__ = ["StalenessAction", "StalenessPolicy", "StalenessReceipt", "assess_staleness", "shrink_proposal"]
