"""Evidence gates for frontier-training and effective-context claims."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mixle.experimental.typed_runtime.frontier_pilot import GraphMemoryPilotReceipt


class ClaimKind(StrEnum):
    """Scale claims guarded by independently inspectable evidence."""

    FRONTIER_TRAINING = "frontier_training"
    EFFECTIVE_TRILLION_CONTEXT = "effective_trillion_context"


class GateStatus(StrEnum):
    """Outcome of one claim gate."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class ScaleRunReceipt:
    """Externally measured scale evidence consumed by the claim evaluator."""

    run_id: str
    accelerator_count: int
    host_count: int
    real_distributed_transport: bool
    model_parameters: int
    source_horizon_tokens: int
    maximum_active_context_tokens: int
    baseline_time_to_target_seconds: float | None
    candidate_time_to_target_seconds: float | None
    baseline_peak_memory_bytes: int | None
    candidate_peak_memory_bytes: int | None
    quality_target_achieved: bool
    worker_loss_recovered: bool
    replay_verified: bool
    provenance_complete: bool
    evidence_uri: str

    def __post_init__(self) -> None:
        counts = (
            self.accelerator_count,
            self.host_count,
            self.model_parameters,
            self.source_horizon_tokens,
            self.maximum_active_context_tokens,
        )
        if not self.run_id or not self.evidence_uri or any(value < 0 for value in counts):
            raise ValueError("scale receipts require identity, evidence, and non-negative counts.")
        optional_values = (
            self.baseline_time_to_target_seconds,
            self.candidate_time_to_target_seconds,
            self.baseline_peak_memory_bytes,
            self.candidate_peak_memory_bytes,
        )
        if any(value is not None and value <= 0 for value in optional_values):
            raise ValueError("measured times and memory footprints must be positive when present.")

    @property
    def resource_improvement_measured(self) -> bool:
        """Whether candidate time-to-target or peak memory beats its paired baseline."""

        faster = (
            self.baseline_time_to_target_seconds is not None
            and self.candidate_time_to_target_seconds is not None
            and self.candidate_time_to_target_seconds < self.baseline_time_to_target_seconds
        )
        smaller = (
            self.baseline_peak_memory_bytes is not None
            and self.candidate_peak_memory_bytes is not None
            and self.candidate_peak_memory_bytes < self.baseline_peak_memory_bytes
        )
        return faster or smaller

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "accelerator_count": self.accelerator_count,
            "host_count": self.host_count,
            "real_distributed_transport": self.real_distributed_transport,
            "model_parameters": self.model_parameters,
            "source_horizon_tokens": self.source_horizon_tokens,
            "maximum_active_context_tokens": self.maximum_active_context_tokens,
            "baseline_time_to_target_seconds": self.baseline_time_to_target_seconds,
            "candidate_time_to_target_seconds": self.candidate_time_to_target_seconds,
            "baseline_peak_memory_bytes": self.baseline_peak_memory_bytes,
            "candidate_peak_memory_bytes": self.candidate_peak_memory_bytes,
            "quality_target_achieved": self.quality_target_achieved,
            "worker_loss_recovered": self.worker_loss_recovered,
            "replay_verified": self.replay_verified,
            "provenance_complete": self.provenance_complete,
            "evidence_uri": self.evidence_uri,
            "resource_improvement_measured": self.resource_improvement_measured,
        }


@dataclass(frozen=True)
class AcceptanceGateReceipt:
    """One falsifiable gate attached to one or more public claims."""

    gate: str
    status: GateStatus
    claims: tuple[ClaimKind, ...]
    observed: str
    evidence_uri: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "claims": [claim.value for claim in self.claims],
            "observed": self.observed,
            "evidence_uri": self.evidence_uri,
        }


@dataclass(frozen=True)
class FrontierClaimAssessment:
    """Claim decision that remains false for failed or missing required evidence."""

    gates: tuple[AcceptanceGateReceipt, ...]

    def claim_allowed(self, claim: ClaimKind) -> bool:
        relevant = tuple(gate for gate in self.gates if claim in gate.claims)
        return bool(relevant) and all(gate.status is GateStatus.PASSED for gate in relevant)

    @property
    def frontier_training_allowed(self) -> bool:
        return self.claim_allowed(ClaimKind.FRONTIER_TRAINING)

    @property
    def effective_trillion_context_allowed(self) -> bool:
        return self.claim_allowed(ClaimKind.EFFECTIVE_TRILLION_CONTEXT)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gates": [gate.as_dict() for gate in self.gates],
            "claims": {
                ClaimKind.FRONTIER_TRAINING.value: self.frontier_training_allowed,
                ClaimKind.EFFECTIVE_TRILLION_CONTEXT.value: self.effective_trillion_context_allowed,
            },
        }


def _gate(
    name: str,
    passed: bool,
    claims: tuple[ClaimKind, ...],
    observed: str,
    evidence_uri: str | None = None,
) -> AcceptanceGateReceipt:
    return AcceptanceGateReceipt(name, GateStatus.PASSED if passed else GateStatus.FAILED, claims, observed, evidence_uri)


def assess_frontier_claims(
    pilot: GraphMemoryPilotReceipt,
    scale_run: ScaleRunReceipt | None = None,
) -> FrontierClaimAssessment:
    """Apply the work plan's conservative claim policy to local and scale receipts."""

    frontier = (ClaimKind.FRONTIER_TRAINING,)
    context = (ClaimKind.EFFECTIVE_TRILLION_CONTEXT,)
    both = frontier + context
    gates = [
        _gate(
            "local-integrated-pilot",
            pilot.graph_quality_gain > 0.0 and pilot.recovery.passed,
            both,
            "quality_gain=%.6f recovery=%s" % (pilot.graph_quality_gain, pilot.recovery.passed),
        ),
        _gate(
            "local-negative-controls",
            all(receipt.oracle_passed for receipt in pilot.failure_receipts),
            both,
            "passed=%d/%d"
            % (sum(receipt.oracle_passed for receipt in pilot.failure_receipts), len(pilot.failure_receipts)),
        ),
    ]
    if scale_run is None:
        gates.extend(
            AcceptanceGateReceipt(name, GateStatus.NOT_RUN, claims, "no external scale receipt supplied")
            for name, claims in (
                ("real-8-gpu-transport", frontier),
                ("multi-host-recovery-replay", frontier),
                ("one-billion-parameter-quality-and-efficiency", frontier),
                ("trillion-token-source-horizon", context),
                ("bounded-active-context-with-provenance", context),
            )
        )
        return FrontierClaimAssessment(tuple(gates))

    evidence = scale_run.evidence_uri
    gates.extend(
        (
            _gate(
                "real-8-gpu-transport",
                scale_run.accelerator_count >= 8 and scale_run.real_distributed_transport,
                frontier,
                "accelerators=%d real_transport=%s"
                % (scale_run.accelerator_count, scale_run.real_distributed_transport),
                evidence,
            ),
            _gate(
                "multi-host-recovery-replay",
                scale_run.host_count >= 2 and scale_run.worker_loss_recovered and scale_run.replay_verified,
                frontier,
                "hosts=%d recovered=%s replay=%s"
                % (scale_run.host_count, scale_run.worker_loss_recovered, scale_run.replay_verified),
                evidence,
            ),
            _gate(
                "one-billion-parameter-quality-and-efficiency",
                scale_run.model_parameters >= 1_000_000_000
                and scale_run.quality_target_achieved
                and scale_run.resource_improvement_measured,
                frontier,
                "parameters=%d quality=%s resource_improvement=%s"
                % (
                    scale_run.model_parameters,
                    scale_run.quality_target_achieved,
                    scale_run.resource_improvement_measured,
                ),
                evidence,
            ),
            _gate(
                "trillion-token-source-horizon",
                scale_run.source_horizon_tokens >= 1_000_000_000_000,
                context,
                "source_horizon_tokens=%d" % scale_run.source_horizon_tokens,
                evidence,
            ),
            _gate(
                "bounded-active-context-with-provenance",
                scale_run.maximum_active_context_tokens < scale_run.source_horizon_tokens
                and scale_run.quality_target_achieved
                and scale_run.provenance_complete,
                context,
                "active=%d source=%d quality=%s provenance=%s"
                % (
                    scale_run.maximum_active_context_tokens,
                    scale_run.source_horizon_tokens,
                    scale_run.quality_target_achieved,
                    scale_run.provenance_complete,
                ),
                evidence,
            ),
        )
    )
    return FrontierClaimAssessment(tuple(gates))


__all__ = [
    "AcceptanceGateReceipt",
    "ClaimKind",
    "FrontierClaimAssessment",
    "GateStatus",
    "ScaleRunReceipt",
    "assess_frontier_claims",
]
