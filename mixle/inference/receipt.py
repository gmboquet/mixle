"""``Receipt`` -- bind an answer's ledger, trace, calibration state, and provenance into one artifact
that a third party can re-verify offline, without re-running the teacher or
touching the substrate.

A receipt makes four claims, each independently checkable from data the receipt itself carries:

  * **ledger** -- an :class:`~mixle.inference.explain.Explanation`; ``is_exact()`` re-checks the additive
    identity (``sum(parts) + correction == total``) that is the evidence, not a summary of it.
  * **trace** -- an :class:`~mixle.task.replay.ExecutionTrace`; replaying it against the same tool
    registry must reproduce every step bit-for-bit (:func:`mixle.task.replay.is_bit_identical_replay`).
  * **calibration** -- the ``alpha``/``qhat`` (or density-gate) state the answer was served under; a
    receipt with unknown calibration is flagged, never assumed calibrated.
  * **provenance** -- where the evidence came from (source ids / citations), following the same
    dict shape :class:`mixle.substrate.core.SubstrateItem.provenance` and
    :class:`mixle.substrate.context.ContextPacket` citations already use.

Any of the four may be absent (a thin-shell :class:`~mixle.system.core.System` answer has no ledger yet);
:func:`verify_receipt` marks missing claims ``"absent"`` but does not promote an evidence-free receipt.
A declared trace without its exact executable registry, a calibration claim without observations, or a
provenance source without matching content is ``"unobserved"`` and cannot pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mixle.inference.explain import Explanation
from mixle.inference.integrity import canonical_digest, implementation_digest

if TYPE_CHECKING:
    from mixle.task.replay import ExecutionTrace


@dataclass
class Receipt:
    """The bound artifact: an answer plus everything needed to re-verify it offline."""

    answer: Any
    produced_by: str = ""
    ledger: Explanation | None = None
    trace: ExecutionTrace | None = None
    calibration: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    executables: dict[str, str] = field(default_factory=dict)
    bindings: dict[str, str] = field(default_factory=dict)
    receipt_digest: str = ""
    schema_version: str = "mixle-answer-receipt-v1"

    def __post_init__(self) -> None:
        """Seal every declared claim without overwriting bindings supplied by a stored receipt."""
        if not self.bindings:
            self.bindings = {name: canonical_digest(value) for name, value in self._claims().items()}
        if not self.receipt_digest:
            self.receipt_digest = canonical_digest({"content": self._content(), "bindings": self.bindings})

    def _claims(self) -> dict[str, Any]:
        claims: dict[str, Any] = {"answer": self.answer}
        if self.produced_by:
            claims["produced_by"] = self.produced_by
        if self.ledger is not None:
            claims["ledger"] = {
                "total": self.ledger.total,
                "parts": self.ledger.parts,
                "correction": self.ledger.correction,
            }
        if self.trace is not None:
            claims["trace"] = self.trace.to_json()
        if self.calibration is not None:
            claims["calibration"] = self.calibration
        if self.provenance:
            claims["provenance"] = self.provenance
        if self.policy:
            claims["policy"] = self.policy
        if self.executables:
            claims["executables"] = self.executables
        return claims

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "answer": self.answer,
            "produced_by": self.produced_by,
            "ledger": self._claims().get("ledger"),
            "trace": self.trace.to_json() if self.trace is not None else None,
            "calibration": self.calibration,
            "provenance": self.provenance,
            "policy": self.policy,
            "executables": self.executables,
        }

    def to_json(self) -> dict[str, Any]:
        """Return the answer receipt as JSON-compatible data."""
        return {**self._content(), "bindings": dict(self.bindings), "receipt_digest": self.receipt_digest}

    def to_knowledge_dict(self, *, id: str, project_id: str, task: str) -> dict[str, Any]:  # noqa: A002
        """A plain dict shaped like ``mixle_knowledge.contracts.AnswerReceipt`` (id/project_id/task/
        produced_by/answer/ledger/trace/calibration/provenance), aligned
        with the mixle-knowledge receipt contracts. Distinct from
        ``mixle_knowledge.contracts.ArtifactReceipt``, which certifies a trained model/artifact, not
        one served answer -- this is the per-answer evidence trail an offline consumer re-verifies
        (recompute the ledger, replay the trace, resolve the citations).

        Stays a plain dict on purpose: mixle core carries no dependency on mixle-knowledge (platform
        contract packages depend on core, never the other way); constructing the validated pydantic
        object (``AnswerReceipt(**receipt.to_knowledge_dict(...))``) is the receiving side's job.
        """
        integrity = {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "executables": self.executables,
            "bindings": self.bindings,
            "receipt_digest": self.receipt_digest,
        }
        return {
            "id": id,
            "project_id": project_id,
            "task": task,
            "answer": self.answer,
            "produced_by": self.produced_by,
            "ledger": self._claims().get("ledger"),
            "trace": self.trace.to_json() if self.trace is not None else None,
            "calibration": self.calibration,
            "provenance": {**self.provenance, "_receipt_integrity": integrity},
        }


@dataclass
class VerificationReport:
    """Which claims were observed and which required observations determine the verdict."""

    checks: dict[str, str] = field(default_factory=dict)  # pass | fail | absent | unobserved
    required: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Whether at least one evidence claim was observed and every required check passed."""
        evidence_checks = [name for name in self.required if name != "receipt_integrity"]
        return bool(evidence_checks) and all(self.checks.get(name) == "pass" for name in self.required)

    def summary(self) -> str:
        """Return a compact comma-separated verification summary."""
        return ", ".join(f"{name}={status}" for name, status in self.checks.items())


def verify_receipt(
    receipt: Receipt,
    *,
    tools: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    tol: float = 1e-9,
) -> VerificationReport:
    """Re-check every claim the receipt actually makes, using only the receipt's own bound data (plus
    the ``tools`` registry needed to re-execute a trace -- the one piece that cannot be inlined into the
    receipt itself, since a tool is a function, not data)."""
    from mixle.task.replay import is_bit_identical_replay

    checks: dict[str, str] = {}
    required = ["receipt_integrity"]
    current_claims = receipt._claims()
    expected_bindings = {name: canonical_digest(value) for name, value in current_claims.items()}
    expected_receipt_digest = canonical_digest({"content": receipt._content(), "bindings": receipt.bindings})
    checks["receipt_integrity"] = (
        "pass"
        if receipt.schema_version == "mixle-answer-receipt-v1"
        and receipt.bindings == expected_bindings
        and receipt.receipt_digest == expected_receipt_digest
        else "fail"
    )

    if receipt.ledger is not None:
        required.append("ledger_exact")
        checks["ledger_exact"] = "pass" if receipt.ledger.is_exact(atol=tol) else "fail"
    else:
        checks["ledger_exact"] = "absent"

    if receipt.trace is not None:
        required.extend(("trace_replayable", "executables_match"))
        if tools is None:
            checks["trace_replayable"] = "unobserved"
            checks["executables_match"] = "unobserved"
        else:
            step_tools = {step.tool for step in receipt.trace.steps}
            if not step_tools:
                checks["trace_replayable"] = "fail"
                checks["executables_match"] = "fail"
            elif not step_tools.issubset(tools):
                checks["trace_replayable"] = "unobserved"
                checks["executables_match"] = "unobserved"
            else:
                actual_executables = {name: implementation_digest(tools[name]) for name in sorted(step_tools)}
                checks["executables_match"] = "pass" if actual_executables == receipt.executables else "fail"
                checks["trace_replayable"] = (
                    "pass"
                    if checks["executables_match"] == "pass" and is_bit_identical_replay(receipt.trace, tools)
                    else "fail"
                )
    else:
        checks["trace_replayable"] = "absent"
        checks["executables_match"] = "absent"

    if receipt.calibration is not None:
        required.extend(("calibration_named", "calibration_observed"))
        has_alpha_or_gate = "qhat" in receipt.calibration or "density_gate" in receipt.calibration
        checks["calibration_named"] = "pass" if has_alpha_or_gate else "fail"
        observations = receipt.calibration.get("observations")
        observed_digest = receipt.calibration.get("evidence_digest")
        if observations is None:
            checks["calibration_observed"] = "unobserved"
        else:
            checks["calibration_observed"] = "pass" if canonical_digest(observations) == observed_digest else "fail"
    else:
        checks["calibration_named"] = "absent"
        checks["calibration_observed"] = "absent"

    if receipt.provenance:
        required.extend(("provenance_present", "provenance_observed"))
        sources = receipt.provenance.get("sources")
        checks["provenance_present"] = "pass" if isinstance(sources, list) and bool(sources) else "fail"
        source_statuses = []
        for source in sources if isinstance(sources, list) else []:
            if not isinstance(source, dict) or not isinstance(source.get("id"), str):
                source_statuses.append("fail")
                continue
            content = source.get("content")
            if content is None and evidence is not None:
                content = evidence.get(source["id"])
            if content is None:
                source_statuses.append("unobserved")
            else:
                source_statuses.append("pass" if canonical_digest(content) == source.get("digest") else "fail")
        checks["provenance_observed"] = (
            "fail"
            if "fail" in source_statuses
            else "unobserved"
            if "unobserved" in source_statuses or not source_statuses
            else "pass"
        )
    else:
        checks["provenance_present"] = "absent"
        checks["provenance_observed"] = "absent"

    return VerificationReport(checks=checks, required=tuple(required))
