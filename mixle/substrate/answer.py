"""Answer a question from the substrate -- retrieve, assemble, answer or ABSTAIN, always cite (S3 seed).

This composes the substrate stack into one honest front door: given a question and a pluggable
``answerer`` (any callable ``(question, context) -> str`` -- a local student, an LLM, a rule), it
retrieves the evidence (single-shot or multi-hop), assembles a budgeted context packet, and either
answers FROM that evidence or ABSTAINS when the evidence is too thin. The result carries the full
evidence chain, so an answer is never returned without a traceable route from the question to the
knowledge it rests on -- the no-answer-without-provenance rule made operational.

This is the seed of the reasoner's evidence loop (S3): RETRIEVE is wired here; COMPUTE / SIMULATE /
CREATE / DELEGATE as evidence-acquiring actions layer on top. The answerer is intentionally external
so this stays model-agnostic and 99%-local -- a tiny distilled student answers most questions on the
laptop; only genuinely hard ones need escalation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.context import ContextBudget, ContextPacket
from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class Answer:
    """A cited answer (or an honest abstention) with the evidence it rests on and a confidence."""

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
        answerer: ``(question, context_text) -> answer_str`` -- any model/rule; called ONLY when there
            is enough evidence above the confidence floor (so a weak retrieval never fabricates).
        budget: the context budget handed to the answerer (default 2000 chars).
        hops: 1 = single-shot :func:`retrieve`; >1 = :func:`multihop` chaining that many hops.
        min_evidence: minimum retrieved items required to attempt an answer.
        min_confidence: retrieval-strength floor below which it abstains rather than guess.
        compress: compress the context to fit more sources under budget (workstream O3).
        scope: restrict to a team/access scope.
    """
    budget = budget or ContextBudget()

    if hops > 1:
        from mixle.substrate.multihop import multihop

        chain = multihop(substrate, question, max_hops=hops, scope=scope, telemetry=telemetry)
        evidence = chain.items
        packet = chain.to_context(question, budget=budget, compress=compress)
        top_score = max((s.score for s in chain.steps if s.depth == 0), default=0.0)
    else:
        from mixle.substrate.retrieve import retrieve

        r = retrieve(substrate, question, k=max(budget.max_items, 6), scope=scope, telemetry=telemetry)
        evidence = r.items
        packet = r.to_context(question, budget=budget, compress=compress)
        top_score = r.scores[0] if r.scores else 0.0

    confidence = _confidence(top_score, packet)

    if len(evidence) < min_evidence or confidence < min_confidence:
        ans = Answer(
            question=question,
            answer=None,
            abstained=True,
            confidence=confidence,
            context=packet,
            note=(
                f"abstained: {len(evidence)} item(s) at confidence {confidence:.2f} "
                f"(needs >= {min_evidence} items above {min_confidence}) -- escalate rather than guess"
            ),
            evidence=list(evidence),
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
