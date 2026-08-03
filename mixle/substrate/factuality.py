"""``check_factuality()`` -- turn an answer into a per-claim receipt grounded in the substrate (B3).

An LLM answer is a paragraph of assertions; some are supported by what the system actually knows, some
are actively contradicted by it, and some it simply says nothing about. :func:`check_factuality` makes
that checkable: it splits the answer into claims (:func:`mixle.reason.llm.sentence_claims`), retrieves
evidence for each from the substrate, and classifies every (evidence, claim) pair as SUPPORTED,
CONTRADICTED, or UNVERIFIED (:class:`Corroboration`) -- attaching the citing item as provenance either
way. The result is a :class:`FactualityReceipt`: every claim tagged supported/contradicted/uncited with
its evidence, plus the grounded fraction.

Lexical overlap alone is never treated as support: two statements can share almost every content word
and still mean opposite things ("the drug cures cancer" / "the drug cures no cancer and does not work"
overlap almost completely and disagree). Overlap only establishes retrieval CANDIDACY -- that the
evidence is plausibly *about* the same thing as the claim. The default corroborator
(:func:`_default_corroborates`) additionally checks negation/polarity agreement on the content words the
claim and evidence share, and calls a pair SUPPORTED only when candidacy holds *and* polarity agrees; a
polarity mismatch on shared content is CONTRADICTED, and anything it cannot confidently place either way
-- including plain lack of overlap -- is UNVERIFIED. This is a negation-cue heuristic, not a real
entailment model: pass a genuine NLI/entailment check as ``corroborates`` for stronger grounding.

An answer with no extractable claims (empty, unparseable, or evasive) is UNKNOWN, not grounded:
``grounded_fraction`` is ``None`` and :meth:`FactualityReceipt.is_grounded` fails closed (``False``) --
there is nothing to have verified, so it is never reported as a perfect factuality result.

This is the knowledge-grounded twin of :meth:`mixle.reason.llm.LLMUncertainty.assess_claims` (which
corroborates against self-consistency samples): same claim-level discipline, but the corroborator is the
substrate, so "is this answer true?" becomes "which of its claims can I cite, which does it contradict,
and which can I do neither for?" -- the no-claim-without-provenance rule applied after the fact to any
answer, whatever produced it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.utils.immutable import detach_receipt_container


class Corroboration(StrEnum):
    """A corroborator's verdict on one (evidence, claim) pair -- three-way, never a bare bool.

    Lexical candidacy (matching content words) is necessary but not sufficient for SUPPORTED: a
    contradiction can share every content word with the claim it contradicts (see the module
    docstring's drug/cancer example). UNVERIFIED is a real, distinct outcome from SUPPORTED -- "found
    no reason to doubt it" is not "confirmed" -- so a caller can never mistake silence for verification.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ClaimVerdict:
    """One claim from an answer, with what the substrate said about it and the evidence either way.

    ``supported`` is True only when some qualifying evidence was judged :attr:`Corroboration.SUPPORTED`
    (entailment-aware -- see :func:`_default_corroborates` -- never bare lexical overlap). ``contradicted``
    is True when some qualifying evidence was judged :attr:`Corroboration.CONTRADICTED` -- a stronger and
    distinct signal from merely being uncited: the substrate does not just fail to back this claim,
    something in it disagrees. The two are not mutually exclusive: conflicting evidence for the same
    claim (one item backs it, another disputes it) sets both.
    """

    claim: str
    supported: bool
    score: float  # best retrieval score for this claim's evidence -- relevance/candidacy, not support
    citations: list[dict[str, Any]] = field(default_factory=list)
    contradicted: bool = False
    contradictions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # A receipt is a record. Detaching severs the caller's alias, so a mutation after
        # construction cannot rewrite evidence that was already recorded; `frozen=True` above
        # stops the field being rebound through the receipt itself. Containers keep their
        # concrete types -- see detach_receipt_container for why (MXR-080-1876).
        object.__setattr__(self, "citations", detach_receipt_container(self.citations))
        object.__setattr__(self, "contradictions", detach_receipt_container(self.contradictions))


@dataclass(frozen=True)
class FactualityReceipt:
    """A per-claim grounding of an answer against the substrate -- the receipt behind 'is this true?'."""

    answer: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # A receipt is a record. Detaching severs the caller's alias, so a mutation after
        # construction cannot rewrite evidence that was already recorded; `frozen=True` above
        # stops the field being rebound through the receipt itself. Containers keep their
        # concrete types -- see detach_receipt_container for why (MXR-080-1876).
        object.__setattr__(self, "verdicts", detach_receipt_container(self.verdicts))

    @property
    def grounded_fraction(self) -> float | None:
        """Fraction of extracted claims supported by substrate evidence.

        ``None`` when the answer had no extractable claims at all (empty, unparseable, or evasive) --
        there is nothing that was checked, so this is UNKNOWN rather than a vacuous 1.0.
        """
        if not self.verdicts:
            return None
        return round(sum(v.supported for v in self.verdicts) / len(self.verdicts), 4)

    def unsupported(self) -> list[ClaimVerdict]:
        """The claims the substrate could not corroborate -- exactly what to flag or retract."""
        return [v for v in self.verdicts if not v.supported]

    def contradicted(self) -> list[ClaimVerdict]:
        """Claims where retrieved evidence actively disagreed -- worse than merely uncited."""
        return [v for v in self.verdicts if v.contradicted]

    def is_grounded(self, threshold: float = 1.0) -> bool:
        """True iff the grounded fraction meets ``threshold`` (default 1.0: every claim must be cited).

        Fails closed when there is nothing to assess: an answer with no extractable claims is never
        reported as "grounded", regardless of ``threshold`` -- a full-grounding gate must not wave
        through silence just because a vacuous fraction happens to clear the bar.
        """
        gf = self.grounded_fraction
        return gf is not None and gf >= threshold

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable factuality receipt."""
        return {
            "grounded_fraction": self.grounded_fraction,  # None serializes to JSON null, not 0.0/1.0
            "n_claims": len(self.verdicts),
            "n_unsupported": len(self.unsupported()),
            "n_contradicted": len(self.contradicted()),
            "claims": [
                {
                    "claim": v.claim,
                    "supported": v.supported,
                    "contradicted": v.contradicted,
                    "score": round(v.score, 4),
                    "citations": v.citations,
                    "contradictions": v.contradictions,
                }
                for v in self.verdicts
            ],
        }


# -- default corroborator: lexical candidacy gated by a negation/polarity check ---------------------
#
# Plain lexical overlap is a CANDIDACY signal, not a support signal: two statements can share every
# content word and mean opposite things ("the drug cures cancer" / "the drug cures no cancer and does
# not work"). This is not a real entailment model -- it is a negation-cue heuristic, tractable without
# the heavyweight NLI dependency this codebase does not carry. It only ever returns SUPPORTED when
# candidacy holds AND no shared content word's polarity disagrees between claim and evidence; anything
# it cannot place with that confidence -- including plain lack of overlap -- is UNVERIFIED, never
# guessed as supported.

_STOPWORDS = frozenset(
    "a an the is are was were be been being of to in on at by for with and or but it its this that "
    "as from into over under near".split()
)

_NEGATORS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "neither",
        "nor",
        "cannot",
        "without",
        "lacks",
        "lacking",
        "fails",
        "unable",
    }
)

_NEG_WINDOW = 4  # how many tokens after a negator its scope reaches, e.g. "not [X Y Z]"


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", str(text).strip()) if s]


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"n't\b", " not", str(text).lower())  # doesn't -> does not, isn't -> is not
    return re.findall(r"[a-z0-9]+", normalized)


def _content_words(text: str) -> set[str]:
    return {w for w in _tokens(text) if w not in _STOPWORDS and w not in _NEGATORS}


def _negated_content_words(text: str) -> set[str]:
    """Content words that fall within a negation marker's forward scope, sentence by sentence.

    A negator's scope never crosses a sentence boundary (so an unrelated negation two sentences later
    in a long evidence passage can't falsely taint an earlier claim's words) and stops at the next
    negator (so two independent negations in one sentence -- "cures no cancer and does not work" --
    are not merged into one span).
    """
    negated: set[str] = set()
    for sent in _sentences(text) or [text]:
        toks = _tokens(sent)
        for i, tok in enumerate(toks):
            if tok not in _NEGATORS:
                continue
            for nxt in toks[i + 1 : i + 1 + _NEG_WINDOW]:
                if nxt in _NEGATORS:
                    break  # a second negator starts its own scope; do not merge the two
                if nxt not in _STOPWORDS:
                    negated.add(nxt)
    return negated


def _default_corroborates(evidence: str, claim: str) -> Corroboration:
    """Lexical candidacy (:func:`~mixle.reason.llm.content_overlap`) gated by a negation/polarity check.

    Overlap alone only establishes that ``evidence`` is plausibly about the same thing as ``claim`` --
    candidacy, not support. Among the content words the two texts share, if any disagrees in polarity
    (negated in one text, not the other -- e.g. "cures cancer" vs "cures no cancer") the pair is
    CONTRADICTED regardless of how much else overlaps. Otherwise, candidacy is enough for SUPPORTED. No
    overlap at all is UNVERIFIED, not contradicted -- unrelated evidence does not disagree with a claim,
    it simply says nothing about it.
    """
    from mixle.reason.llm import content_overlap

    if not content_overlap(evidence, claim, threshold=0.5):
        return Corroboration.UNVERIFIED

    shared = _content_words(claim) & _content_words(evidence)
    if not shared:
        return Corroboration.UNVERIFIED
    neg_claim = _negated_content_words(claim)
    neg_evidence = _negated_content_words(evidence)
    if any((w in neg_claim) != (w in neg_evidence) for w in shared):
        return Corroboration.CONTRADICTED
    return Corroboration.SUPPORTED


def _coerce_verdict(result: Corroboration | bool) -> Corroboration:
    """Accept a legacy ``bool``-returning corroborator too: ``True`` -> SUPPORTED, ``False`` ->
    UNVERIFIED. Never CONTRADICTED -- a bare bool has no way to express a contradiction, so mapping a
    falsy legacy result to "unverified" rather than guessing is the only reading that can't overclaim.
    """
    if isinstance(result, Corroboration):
        return result
    return Corroboration.SUPPORTED if result else Corroboration.UNVERIFIED


def check_factuality(
    substrate: Substrate,
    answer: str,
    *,
    extract: Callable[[str], list[str]] | None = None,
    corroborates: Callable[[str, str], Corroboration | bool] | None = None,
    min_score: float = 0.2,
    k: int = 4,
    scope: str | None = None,
) -> FactualityReceipt:
    """Ground each claim of ``answer`` against ``substrate``, returning a :class:`FactualityReceipt`.

    Args:
        extract: ``answer -> [claim, ...]`` (default :func:`mixle.reason.llm.sentence_claims`).
        corroborates: ``(evidence_text, claim) -> Corroboration`` classifying retrieved evidence against
            a claim as SUPPORTED / CONTRADICTED / UNVERIFIED (default: lexical candidacy gated by a
            negation/polarity check -- see :func:`_default_corroborates`; pass a real NLI/entailment
            model for stronger grounding). A plain ``bool`` is still accepted for compatibility: ``True``
            maps to SUPPORTED, ``False`` to UNVERIFIED.
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
        contradictions: list[dict[str, Any]] = []
        supported = False
        contradicted = False
        for item, sc in zip(r.items, r.scores):
            if sc < min_score:
                continue
            verdict = _coerce_verdict(corr(_text(item), claim))
            entry = {"id": item.id, "kind": item.kind, "score": round(float(sc), 4)}
            if verdict is Corroboration.SUPPORTED:
                supported = True
                citations.append(entry)
            elif verdict is Corroboration.CONTRADICTED:
                contradicted = True
                contradictions.append(entry)
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                supported=supported,
                score=float(best),
                citations=citations,
                contradicted=contradicted,
                contradictions=contradictions,
            )
        )

    return FactualityReceipt(answer=answer, verdicts=verdicts)


def _text(item: SubstrateItem) -> str:
    return item.text or ""
