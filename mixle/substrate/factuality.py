"""``check_factuality()`` -- turn an answer into a per-claim receipt grounded in the substrate (B3).

An LLM answer is a paragraph of assertions; some are supported by what the system actually knows and
some are not. :func:`check_factuality` makes that checkable: it splits the answer into claims
(:func:`mixle.reason.llm.sentence_claims`), retrieves evidence for each from the substrate, and marks
the claim SUPPORTED only when retrieved evidence both scores above a floor and overlaps its content --
attaching the citing item as provenance. The result is a :class:`FactualityReceipt`: every claim tagged
supported/unsupported with its evidence, plus the grounded fraction.

This is the knowledge-grounded twin of :meth:`mixle.reason.llm.LLMUncertainty.assess_claims` (which
corroborates against self-consistency samples): same claim-level discipline, but the corroborator is the
substrate, so "is this answer true?" becomes "which of its claims can I cite, and which can't I?" -- the
no-claim-without-provenance rule applied after the fact to any answer, whatever produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class ClaimVerdict:
    """One claim from an answer, marked supported or not, with the evidence that (dis)confirms it."""

    claim: str
    supported: bool
    score: float  # best retrieval score for this claim's evidence
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FactualityReceipt:
    """A per-claim grounding of an answer against the substrate -- the receipt behind 'is this true?'."""

    answer: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def grounded_fraction(self) -> float:
        """Fraction of extracted claims supported by substrate evidence."""
        if not self.verdicts:
            return 1.0
        return round(sum(v.supported for v in self.verdicts) / len(self.verdicts), 4)

    def unsupported(self) -> list[ClaimVerdict]:
        """The claims the substrate could not corroborate -- exactly what to flag or retract."""
        return [v for v in self.verdicts if not v.supported]

    def is_grounded(self, threshold: float = 1.0) -> bool:
        """True iff the grounded fraction meets ``threshold`` (default 1.0: every claim must be cited)."""
        return self.grounded_fraction >= threshold

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable factuality receipt."""
        return {
            "grounded_fraction": self.grounded_fraction,
            "n_claims": len(self.verdicts),
            "n_unsupported": len(self.unsupported()),
            "claims": [
                {"claim": v.claim, "supported": v.supported, "score": round(v.score, 4), "citations": v.citations}
                for v in self.verdicts
            ],
        }


def _default_corroborates(evidence: str, claim: str) -> bool:
    from mixle.reason.llm import content_overlap

    return content_overlap(evidence, claim, threshold=0.5)


def check_factuality(
    substrate: Substrate,
    answer: str,
    *,
    extract: Callable[[str], list[str]] | None = None,
    corroborates: Callable[[str, str], bool] | None = None,
    min_score: float = 0.2,
    k: int = 4,
    scope: str | None = None,
) -> FactualityReceipt:
    """Ground each claim of ``answer`` against ``substrate``, returning a :class:`FactualityReceipt`.

    Args:
        extract: ``answer -> [claim, ...]`` (default :func:`mixle.reason.llm.sentence_claims`).
        corroborates: ``(evidence_text, claim) -> bool`` deciding if retrieved evidence supports a claim
            (default content-overlap; pass an NLI/entailment check for stronger grounding).
        min_score: retrieval-score floor; evidence below it doesn't count (guards low-signal embedder noise).
        k: evidence items retrieved per claim.
        scope: restrict retrieval to a team/access scope.
    """
    from mixle.reason.llm import sentence_claims
    from mixle.substrate.retrieve import retrieve

    extract = extract or sentence_claims
    corr = corroborates or _default_corroborates

    verdicts: list[ClaimVerdict] = []
    for claim in extract(answer):
        r = retrieve(substrate, claim, k=k, scope=scope)
        best = r.scores[0] if r.scores else 0.0
        citations: list[dict[str, Any]] = []
        supported = False
        for item, sc in zip(r.items, r.scores):
            if sc < min_score:
                continue
            if corr(_text(item), claim):
                supported = True
                citations.append({"id": item.id, "kind": item.kind, "score": round(float(sc), 4)})
        verdicts.append(ClaimVerdict(claim=claim, supported=supported, score=float(best), citations=citations))

    return FactualityReceipt(answer=answer, verdicts=verdicts)


def _text(item: SubstrateItem) -> str:
    return item.text or ""
