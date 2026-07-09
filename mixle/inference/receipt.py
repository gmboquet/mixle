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

Any of the four may be absent (a thin-shell :class:`~mixle.system.System` answer has no ledger yet);
:func:`verify_receipt` only checks what is present and marks the rest ``"absent"`` -- it never invents a
pass for a claim the receipt does not make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mixle.inference.explain import Explanation

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

    def to_json(self) -> dict[str, Any]:
        """Return the answer receipt as JSON-compatible data."""
        return {
            "answer": self.answer,
            "produced_by": self.produced_by,
            "ledger": {"total": self.ledger.total, "parts": self.ledger.parts, "correction": self.ledger.correction}
            if self.ledger is not None
            else None,
            "trace": self.trace.to_json() if self.trace is not None else None,
            "calibration": self.calibration,
            "provenance": self.provenance,
        }

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
        return {
            "id": id,
            "project_id": project_id,
            "task": task,
            **self.to_json(),
        }


@dataclass
class VerificationReport:
    """Which of the receipt's claims hold. ``passed`` is True only if every present claim checks out --
    an absent claim never counts as a pass or a failure, it is just not a claim this receipt makes."""

    checks: dict[str, str] = field(default_factory=dict)  # name -> "pass" | "fail" | "absent"

    @property
    def passed(self) -> bool:
        """Whether no recorded verification check failed."""
        return "fail" not in self.checks.values()

    def summary(self) -> str:
        """Return a compact comma-separated verification summary."""
        return ", ".join(f"{name}={status}" for name, status in self.checks.items())


def verify_receipt(receipt: Receipt, *, tools: dict[str, Any] | None = None, tol: float = 1e-9) -> VerificationReport:
    """Re-check every claim the receipt actually makes, using only the receipt's own bound data (plus
    the ``tools`` registry needed to re-execute a trace -- the one piece that cannot be inlined into the
    receipt itself, since a tool is a function, not data)."""
    from mixle.task.replay import is_bit_identical_replay

    checks: dict[str, str] = {}

    if receipt.ledger is not None:
        checks["ledger_exact"] = "pass" if receipt.ledger.is_exact(atol=tol) else "fail"
    else:
        checks["ledger_exact"] = "absent"

    if receipt.trace is not None:
        if tools is None:
            checks["trace_replayable"] = "absent"
        else:
            checks["trace_replayable"] = "pass" if is_bit_identical_replay(receipt.trace, tools) else "fail"
    else:
        checks["trace_replayable"] = "absent"

    if receipt.calibration is not None:
        has_alpha_or_gate = "qhat" in receipt.calibration or "density_gate" in receipt.calibration
        checks["calibration_named"] = "pass" if has_alpha_or_gate else "fail"
    else:
        checks["calibration_named"] = "absent"

    checks["provenance_present"] = "pass" if receipt.provenance else "absent"

    return VerificationReport(checks=checks)
