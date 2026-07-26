"""Evidence gates for frontier-training and effective-context claims."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mixle.experimental.typed_runtime.benchmark import FailureKind
from mixle.experimental.typed_runtime.frontier_pilot import GraphMemoryPilotReceipt
from mixle.experimental.typed_runtime.proposal import payload_fingerprint


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
    required_evidence_count: int = 1
    observed_evidence_count: int = 0
    required_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.gate or not self.observed:
            raise ValueError("acceptance gates require a name and observation.")
        if not isinstance(self.status, GateStatus):
            raise TypeError("acceptance-gate status must be GateStatus.")
        if (
            not isinstance(self.claims, tuple)
            or not self.claims
            or any(not isinstance(claim, ClaimKind) for claim in self.claims)
            or len(set(self.claims)) != len(self.claims)
        ):
            raise ValueError("acceptance gates require unique affected claims.")
        counts = (self.required_evidence_count, self.observed_evidence_count)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("acceptance-gate evidence cardinalities must be integers.")
        if self.required_evidence_count < 1 or self.observed_evidence_count < 0:
            raise ValueError("acceptance-gate evidence cardinalities must be positive/non-negative.")
        if (
            not self.required_fields
            or any(not isinstance(name, str) or not name for name in self.required_fields)
            or len(set(self.required_fields)) != len(self.required_fields)
        ):
            raise ValueError("acceptance gates must declare unique non-empty required evidence fields.")
        if self.status is GateStatus.PASSED and (
            self.observed_evidence_count < self.required_evidence_count or not self.evidence_uri
        ):
            raise ValueError("a passed acceptance gate requires its declared evidence count and evidence URI.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "claims": [claim.value for claim in self.claims],
            "observed": self.observed,
            "evidence_uri": self.evidence_uri,
            "required_evidence_count": self.required_evidence_count,
            "observed_evidence_count": self.observed_evidence_count,
            "required_fields": list(self.required_fields),
        }


@dataclass(frozen=True)
class FrontierClaimAssessment:
    """Claim decision that remains false for failed or missing required evidence."""

    gates: tuple[AcceptanceGateReceipt, ...]

    def __post_init__(self) -> None:
        names = [gate.gate for gate in self.gates]
        if len(names) != len(set(names)):
            raise ValueError("frontier claim assessments cannot contain duplicate gates.")

    def claim_allowed(self, claim: ClaimKind) -> bool:
        relevant = tuple(gate for gate in self.gates if claim in gate.claims)
        expected = _REQUIRED_GATES_BY_CLAIM[claim]
        observed = {gate.gate for gate in relevant}
        return expected.issubset(observed) and all(gate.status is GateStatus.PASSED for gate in relevant)

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
    *,
    required_evidence_count: int = 1,
    observed_evidence_count: int = 1,
    required_fields: tuple[str, ...],
) -> AcceptanceGateReceipt:
    return AcceptanceGateReceipt(
        name,
        GateStatus.PASSED if passed else GateStatus.FAILED,
        claims,
        observed,
        evidence_uri,
        required_evidence_count,
        observed_evidence_count,
        required_fields,
    )


_LOCAL_INTEGRATED_GATE = "local-integrated-pilot"
_LOCAL_CONTROLS_GATE = "local-negative-controls"
_REQUIRED_FAILURE_CASES = {
    "local-window-negative-control": (
        FailureKind.QUALITY_REGRESSION,
        "held-out-accuracy-target",
        True,
    ),
    "graph-retrieval-quality-control": (
        FailureKind.QUALITY_REGRESSION,
        "held-out-accuracy-target",
        False,
    ),
    "checkpoint-restart-control": (
        FailureKind.REPLAY_MISMATCH,
        "model-optimizer-rng-fingerprints",
        False,
    ),
}
_REQUIRED_GATES_BY_CLAIM = {
    ClaimKind.FRONTIER_TRAINING: frozenset(
        {
            _LOCAL_INTEGRATED_GATE,
            _LOCAL_CONTROLS_GATE,
            "real-8-gpu-transport",
            "multi-host-recovery-replay",
            "one-billion-parameter-quality-and-efficiency",
        }
    ),
    ClaimKind.EFFECTIVE_TRILLION_CONTEXT: frozenset(
        {
            _LOCAL_INTEGRATED_GATE,
            _LOCAL_CONTROLS_GATE,
            "trillion-token-source-horizon",
            "bounded-active-context-with-provenance",
        }
    ),
}


def assess_frontier_claims(
    pilot: GraphMemoryPilotReceipt,
    scale_run: ScaleRunReceipt | None = None,
) -> FrontierClaimAssessment:
    """Apply the work plan's conservative claim policy to local and scale receipts."""

    frontier = (ClaimKind.FRONTIER_TRAINING,)
    context = (ClaimKind.EFFECTIVE_TRILLION_CONTEXT,)
    both = frontier + context
    pilot_evidence = "sha256:" + payload_fingerprint(pilot.as_dict())
    failure_cases = [receipt.case_id for receipt in pilot.failure_receipts]
    controls_by_case = {receipt.case_id: receipt for receipt in pilot.failure_receipts}
    controls_complete = (
        len(failure_cases) == len(_REQUIRED_FAILURE_CASES)
        and len(controls_by_case) == len(_REQUIRED_FAILURE_CASES)
        and set(controls_by_case) == set(_REQUIRED_FAILURE_CASES)
        and all(
            receipt.benchmark_id == "graph-memory-pilot"
            and (receipt.kind, receipt.oracle, receipt.expected_failure) == specification
            and receipt.oracle_passed
            for case_id, specification in _REQUIRED_FAILURE_CASES.items()
            for receipt in (controls_by_case[case_id],)
        )
    )
    gates = [
        _gate(
            _LOCAL_INTEGRATED_GATE,
            pilot.graph_quality_gain > 0.0 and pilot.recovery.passed,
            both,
            "quality_gain=%.6f recovery=%s" % (pilot.graph_quality_gain, pilot.recovery.passed),
            pilot_evidence,
            required_fields=("graph_quality_gain", "recovery.passed"),
        ),
        _gate(
            _LOCAL_CONTROLS_GATE,
            controls_complete,
            both,
            "passed=%d/%d"
            % (sum(receipt.oracle_passed for receipt in pilot.failure_receipts), len(pilot.failure_receipts)),
            pilot_evidence,
            required_evidence_count=len(_REQUIRED_FAILURE_CASES),
            observed_evidence_count=len(failure_cases),
            required_fields=("case_id", "oracle", "expected_failure", "detected", "oracle_passed"),
        ),
    ]
    if scale_run is None:
        gates.extend(
            AcceptanceGateReceipt(
                name,
                GateStatus.NOT_RUN,
                claims,
                "no external scale receipt supplied",
                required_evidence_count=1,
                observed_evidence_count=0,
                required_fields=required_fields,
            )
            for name, claims, required_fields in (
                ("real-8-gpu-transport", frontier, ("accelerator_count", "real_distributed_transport")),
                (
                    "multi-host-recovery-replay",
                    frontier,
                    ("host_count", "worker_loss_recovered", "replay_verified"),
                ),
                (
                    "one-billion-parameter-quality-and-efficiency",
                    frontier,
                    (
                        "model_parameters",
                        "quality_target_achieved",
                        "baseline_time_to_target_seconds|baseline_peak_memory_bytes",
                        "candidate_time_to_target_seconds|candidate_peak_memory_bytes",
                    ),
                ),
                ("trillion-token-source-horizon", context, ("source_horizon_tokens",)),
                (
                    "bounded-active-context-with-provenance",
                    context,
                    (
                        "maximum_active_context_tokens",
                        "source_horizon_tokens",
                        "quality_target_achieved",
                        "provenance_complete",
                    ),
                ),
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
                required_fields=("accelerator_count", "real_distributed_transport"),
            ),
            _gate(
                "multi-host-recovery-replay",
                scale_run.host_count >= 2 and scale_run.worker_loss_recovered and scale_run.replay_verified,
                frontier,
                "hosts=%d recovered=%s replay=%s"
                % (scale_run.host_count, scale_run.worker_loss_recovered, scale_run.replay_verified),
                evidence,
                required_fields=("host_count", "worker_loss_recovered", "replay_verified"),
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
                required_fields=(
                    "model_parameters",
                    "quality_target_achieved",
                    "baseline_time_to_target_seconds|baseline_peak_memory_bytes",
                    "candidate_time_to_target_seconds|candidate_peak_memory_bytes",
                ),
            ),
            _gate(
                "trillion-token-source-horizon",
                scale_run.source_horizon_tokens >= 1_000_000_000_000,
                context,
                "source_horizon_tokens=%d" % scale_run.source_horizon_tokens,
                evidence,
                required_fields=("source_horizon_tokens",),
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
                required_fields=(
                    "maximum_active_context_tokens",
                    "source_horizon_tokens",
                    "quality_target_achieved",
                    "provenance_complete",
                ),
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
