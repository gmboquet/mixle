"""Answer a question from substrate evidence with citations.

Given a question and a pluggable ``answerer`` callable, this module retrieves
evidence, assembles a budgeted context packet, and either returns an answer from
that evidence or abstains when the evidence is too thin. The result carries the
evidence chain and citations back to substrate items.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.context import ContextBudget, ContextPacket
from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class Answer:
    """A cited answer or abstention with the evidence it rests on and a confidence."""

    question: str
    answer: str | None  # None when abstained
    abstained: bool
    confidence: float  # in [0, 1]: retrieval strength backing the answer
    context: ContextPacket
    note: str = ""  # why it abstained, or how the answer was produced
    evidence: list[SubstrateItem] = field(default_factory=list)

    def citations(self) -> list[dict[str, Any]]:
        """Where the answer's evidence came from -- the provenance the answer must be checkable against."""
        return self.context.provenance()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable answer with citations and confidence."""
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "confidence": round(self.confidence, 4),
            "note": self.note,
            "citations": self.citations(),
        }


def answer_from_substrate(
    substrate: Substrate,
    question: str,
    answerer: Callable[[str, str], str],
    *,
    budget: ContextBudget | None = None,
    hops: int = 1,
    min_evidence: int = 1,
    min_confidence: float = 0.1,
    compress: bool = True,
    scope: str | None = None,
    telemetry: Any = None,
) -> Answer:
    """Answer ``question`` from ``substrate`` via ``answerer``, or abstain when evidence is too thin.

    Args:
        answerer: ``(question, context_text) -> answer_str`` -- any model/rule; called only when there
            is enough evidence above the confidence floor (so a weak retrieval never fabricates).
        budget: the context budget handed to the answerer (default 2000 chars).
        hops: 1 = single-shot :func:`retrieve`; >1 = :func:`multihop` chaining that many hops.
        min_evidence: minimum evidence items the answerer must actually receive to attempt an answer --
            checked against ``packet.items`` (what survives context budgeting), never the pre-budget
            retrieved/hop-chain count. A small budget can drop most retrieved items before the answerer
            ever sees them; gating on the pre-budget count would let a caller believe far more evidence
            was used than the answerer actually got (MXR-080-0257).
        min_confidence: retrieval-strength floor below which it abstains rather than guess.
        compress: compress the context to fit more sources under budget.
        scope: restrict to a team/access scope.
    """
    budget = budget or ContextBudget()

    if hops > 1:
        from mixle.substrate.multihop import multihop

        chain = multihop(substrate, question, max_hops=hops, scope=scope, telemetry=telemetry)
        packet = chain.to_context(question, budget=budget, compress=compress)
        top_score = max((s.score for s in chain.steps if s.depth == 0), default=0.0)
    else:
        from mixle.substrate.retrieve import retrieve

        r = retrieve(substrate, question, k=max(budget.max_items, 6), scope=scope, telemetry=telemetry)
        packet = r.to_context(question, budget=budget, compress=compress)
        top_score = r.scores[0] if r.scores else 0.0

    confidence = _confidence(top_score, packet)
    # The gate and the abstention report below both read ONLY `packet.items` -- the final, aligned,
    # actually-delivered evidence -- never the pre-budget retrieved/hop-chain set. `packet.n_candidates`
    # (how many were retrieved before budgeting) is surfaced in the note as context, not as what was
    # gated or what is reported as evidence (MXR-080-0257): a caller who sees "min_evidence satisfied"
    # must be able to trust the answerer really received that many items.
    n_delivered = len(packet)

    if n_delivered < min_evidence or confidence < min_confidence:
        ans = Answer(
            question=question,
            answer=None,
            abstained=True,
            confidence=confidence,
            context=packet,
            note=(
                f"abstained: {n_delivered} item(s) delivered (of {packet.n_candidates} retrieved) at "
                f"confidence {confidence:.2f} (needs >= {min_evidence} delivered items above "
                f"{min_confidence}) -- escalate rather than guess"
            ),
            evidence=list(packet.items),
        )
        _emit(telemetry, ans)
        return ans

    text = answerer(question, packet.render())
    ans = Answer(
        question=question,
        answer=text,
        abstained=False,
        confidence=confidence,
        context=packet,
        note=f"answered from {len(packet)} cited source(s)",
        evidence=list(packet.items),
    )
    _emit(telemetry, ans)
    return ans


def _confidence(top_score: float, packet: ContextPacket) -> float:
    """A calibrated-ish confidence in [0, 1] from the best retrieval score and how much survived."""
    base = max(0.0, min(1.0, float(top_score)))
    # a packet that had to drop nearly everything to fit is weaker evidence than one that kept it
    coverage = min(1.0, len(packet) / max(packet.n_candidates, 1)) if packet.n_candidates else 1.0
    return round(base * (0.5 + 0.5 * coverage), 4)


def _emit(telemetry: Any, ans: Answer) -> None:
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        rec(
            "reason",
            features={"action": "answer", "n_evidence": len(ans.evidence)},
            choice="abstain" if ans.abstained else "answer",
            outcome={"confidence": ans.confidence, "n_cited": len(ans.context)},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break answering
        pass
