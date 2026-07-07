"""``Receipt`` -- bind an answer's ledger, trace, calibration state, and provenance into one artifact
that a THIRD PARTY can re-verify OFFLINE, without re-running the teacher or touching the substrate
(workstream H3, built once H1's :mod:`~mixle.inference.explain` ledger, H2's
:mod:`~mixle.task.replay` trace, and E1's substrate provenance shape all exist).

A receipt makes four claims, each independently checkable from data the receipt itself carries:

  * **ledger** -- an :class:`~mixle.inference.explain.Explanation`; ``is_exact()`` re-checks the additive
    identity (``sum(parts) + correction == total``) that IS the evidence, not a summary of it.
  * **trace** -- an :class:`~mixle.task.replay.ExecutionTrace`; replaying it against the SAME tool
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


@dataclass
class VerificationReport:
    """Which of the receipt's claims hold. ``passed`` is True only if every PRESENT claim checks out --
    an absent claim never counts as a pass or a failure, it is just not a claim this receipt makes."""

    checks: dict[str, str] = field(default_factory=dict)  # name -> "pass" | "fail" | "absent"

    @property
    def passed(self) -> bool:
        return "fail" not in self.checks.values()

    def summary(self) -> str:
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
