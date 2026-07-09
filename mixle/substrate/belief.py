"""Calibrated belief store: harvest claims from model output and assimilate them as CREDENCES.

A model's output is a paragraph of assertions -- some grounded, some not. This module turns each
assertion into a tracked belief with a probability (not a binary "fact" flag) that moves with the
STRENGTH of its evidence: a verifiable source (a document, an executable check, a held-out truth, a
real measurement) moves it strongly; the model's own say-so moves it weakly and is capped low. Every
belief carries its full ``evidence_history`` so the current credence is always reproducible by
replaying it, and evidence can be revised or :func:`retract`-ed, cascading to whatever depended on it.

Anti-laundering is the load-bearing property: evidence that resolves back to the claim itself, or to
another belief that has no independent (non-model-assertion) support of its own, contributes ZERO --
a claim cannot bootstrap high credence by citing itself or an equally ungrounded peer.

This is the write side of the knowledge substrate: harvest, credence
assimilation, and traceable history. The retrieval side is
:mod:`mixle.substrate.retrieve`.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem

# The weakest evidence tier: a model asserting something about itself, with no
# external check. This is intentionally kept outside
# mixle.doe.oracle.VERIFIABILITY_TIERS; here it is allowed only as low-capped
# evidence, never as strong verification.
MODEL_ASSERTION = "model_assertion"

# Evidence-tier strength, weakest to strongest: a model's own assertion moves credence the least; a
# real measurement moves it the most. Reuses mixle.doe.oracle's verifiability vocabulary (a verified
# oracle result is strong evidence) plus MODEL_ASSERTION as the floor tier.
_TIER_STRENGTH: dict[str, float] = {
    MODEL_ASSERTION: 0.05,
    "executable": 0.3,
    "simulation": 0.5,
    "held_out_truth": 0.8,
    "real_measurement": 1.0,
}

# A belief supported only by model-assertion-tier evidence can never rise above this credence -- the
# cap that keeps a model from bootstrapping high confidence in something it merely asserted.
MODEL_ASSERTION_CAP = 0.5

# Log-odds scale per unit of tier strength. Not fit to data -- a qualitative calibration knob so that
# one real_measurement entry lands "high" and one model_assertion entry lands at the cap.
_K = 3.0

_BELIEF_TAG = "belief"


@dataclass
class Claim:
    """One atomic proposition (or typed quantity) pulled from a model's output."""

    text: str
    produced_by: dict[str, Any] = field(default_factory=dict)
    quantity: float | None = None


@dataclass
class EvidenceEntry:
    """One piece of evidence that moved a belief's credence, in the order it was applied."""

    source_id: str
    tier: str
    direction: str = "+"  # "+" supports the claim, "-" contradicts it
    weight: float = 1.0
    time: float = field(default_factory=time.time)


@dataclass
class BeliefItem:
    """A claim's current credence plus the full evidence trail that produced it."""

    id: str
    claim: Claim
    credence: float
    evidence_history: list[EvidenceEntry] = field(default_factory=list)
    scope: str = "local"


def harvest_knowledge(
    model_output: str,
    *,
    source: dict[str, Any],
    extract: Callable[[str], list[str]] | None = None,
) -> list[Claim]:
    """Split ``model_output`` into atomic claims, each stamped with ``source`` (which model produced it,
    its confidence, etc). Default extraction is sentence-level (:func:`mixle.reason.llm.sentence_claims`);
    pass ``extract`` for a different atomic-proposition splitter."""
    from mixle.reason.llm import sentence_claims

    extract = extract or sentence_claims
    return [Claim(text=c, produced_by=dict(source)) for c in extract(model_output)]


def credence_from_history(evidence_history: Sequence[EvidenceEntry]) -> float:
    """The credence implied by an evidence history alone -- a pure function, so replaying a belief's
    stored ``evidence_history`` through this always reproduces its current ``credence`` exactly."""
    if not evidence_history:
        return 0.5  # neutral: no evidence yet
    logit = 0.0
    has_real_support = False
    for e in evidence_history:
        if e.weight <= 0:
            continue
        strength = _tier_strength(e.tier) * e.weight
        sign = 1.0 if e.direction == "+" else -1.0
        logit += sign * strength * _K
        if e.tier != MODEL_ASSERTION:
            has_real_support = True
    credence = 1.0 / (1.0 + math.exp(-logit))
    if not has_real_support:
        credence = min(credence, MODEL_ASSERTION_CAP)
    return credence


def assimilate(
    sub: Substrate,
    claim: Claim,
    evidence: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    scope: str = "local",
) -> BeliefItem:
    """Bayesian-ish update of the belief in ``claim`` from ``evidence`` -- never a binary write.

    Finds or creates the belief item (keyed on normalized claim text) and appends each evidence entry
    (``{"source_id", "tier", "direction": "+"/"-", "weight"}``); ``tier`` must be one of
    :data:`_TIER_STRENGTH` (``"model_assertion"`` plus :data:`mixle.doe.oracle.VERIFIABILITY_TIERS`).

    Anti-laundering: an entry whose ``source_id`` resolves back to THIS belief (a cycle) or to another
    belief item with no independent (non-model-assertion) support of its own is stored with an
    effective weight of zero -- it cannot move the credence, though it stays in the trail for audit.
    """
    entries = [evidence] if isinstance(evidence, dict) else list(evidence)
    key = _claim_key(claim.text)
    existing = _find(sub, key, scope)
    belief = (
        _from_item(existing)
        if existing is not None
        else BeliefItem(id=uuid.uuid4().hex[:16], claim=claim, credence=0.5, evidence_history=[], scope=scope)
    )

    for raw in entries:
        tier = raw["tier"]
        _tier_strength(tier)  # validates; raises on an unrecognized tier rather than silently accepting one
        source_id = str(raw.get("source_id", ""))
        weight = float(raw.get("weight", 1.0))
        if _launders(sub, source_id, belief.id, scope):
            weight = 0.0
        belief.evidence_history.append(
            EvidenceEntry(
                source_id=source_id,
                tier=tier,
                direction=raw.get("direction", "+"),
                weight=weight,
                time=float(raw["time"]) if "time" in raw else time.time(),
            )
        )

    belief.credence = credence_from_history(belief.evidence_history)
    sub.put(_to_item(belief))
    return belief


def retract(sub: Substrate, source_id: str, *, scope: str | None = None) -> list[BeliefItem]:
    """Remove every evidence entry citing ``source_id``, recomputing credence, and CASCADE: if that
    removal causes a belief to lose its only independent support, also strip citations of THAT belief
    from whatever cited it, recursively. Returns every belief item touched by the cascade."""
    changed: list[BeliefItem] = []
    frontier = {source_id}
    visited: set[str] = set()
    while frontier:
        cur = frontier.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for item in sub.all(kind="record", scope=scope):
            if _BELIEF_TAG not in item.tags:
                continue
            belief = _from_item(item)
            new_history = [e for e in belief.evidence_history if e.source_id != cur]
            if len(new_history) == len(belief.evidence_history):
                continue
            before_grounded = _has_real_support(belief)
            belief.evidence_history = new_history
            belief.credence = credence_from_history(new_history)
            sub.put(_to_item(belief))
            changed.append(belief)
            if before_grounded and not _has_real_support(belief):
                frontier.add(belief.id)  # this belief lost its grounding -- re-check its own citers
    return changed


def retrieve_beliefs(
    sub: Substrate, query: str, *, k: int = 8, min_credence: float | None = None, scope: str | None = None
) -> list[BeliefItem]:
    """Beliefs relevant to ``query``, optionally thresholded on ``min_credence`` and re-ranked by
    ``relevance * credence`` -- so a caller can weight by, or hard-filter on, how much the store
    actually believes each item (never a "fact" vs "non-fact" partition, only credence)."""
    hits = sub.search(query, k=max(int(k) * 3, int(k)), kind="record", scope=scope)
    scored: list[tuple[BeliefItem, float]] = []
    for item, sc in hits:
        if _BELIEF_TAG not in item.tags:
            continue
        belief = _from_item(item)
        if min_credence is not None and belief.credence < min_credence:
            continue
        scored.append((belief, sc * belief.credence))
    scored.sort(key=lambda t: -t[1])
    return [b for b, _ in scored[: int(k)]]


# --- internals --------------------------------------------------------------------------------------


def _tier_strength(tier: str) -> float:
    if tier not in _TIER_STRENGTH:
        raise ValueError(f"unknown evidence tier {tier!r}; expected one of {sorted(_TIER_STRENGTH)}")
    return _TIER_STRENGTH[tier]


def _has_real_support(belief: BeliefItem) -> bool:
    return any(e.tier != MODEL_ASSERTION and e.weight > 0 for e in belief.evidence_history)


def _claim_key(text: str) -> str:
    return " ".join(text.lower().split())


def _find(sub: Substrate, key: str, scope: str) -> SubstrateItem | None:
    for item in sub.all(kind="record", scope=scope):
        if _BELIEF_TAG in item.tags and item.payload.get("key") == key:
            return item
    return None


def _launders(sub: Substrate, source_id: str, belief_id: str, scope: str, _seen: set[str] | None = None) -> bool:
    """True iff citing ``source_id`` as evidence for ``belief_id`` would launder unearned credence:
    a direct or indirect cycle back to ``belief_id``, or a reference to another belief item that has
    no independent support of its own -- walked ALL THE WAY DOWN, not just one hop: ``source_id`` is
    laundering unless it resolves to something with at least one non-model-assertion entry that is
    ITSELF not laundering (recursively). A ``source_id`` that is not itself a belief item in the
    substrate (a document, an oracle receipt, any genuine external reference) is never laundering --
    that is the recursion's base case."""
    if source_id == belief_id:
        return True
    seen = _seen or set()
    if source_id in seen:
        return True  # a cycle among referenced claims that never reaches belief_id directly
    ref_item = sub.get(source_id)
    if ref_item is None or _BELIEF_TAG not in ref_item.tags:
        return False
    ref = _from_item(ref_item)
    next_seen = seen | {source_id}
    for e in ref.evidence_history:
        if e.tier != MODEL_ASSERTION and e.weight > 0 and not _launders(sub, e.source_id, belief_id, scope, next_seen):
            return False  # ref has at least one genuinely independent, non-circular real support
    return True  # every one of ref's entries is model-assertion-only or itself laundering


def _to_item(belief: BeliefItem) -> SubstrateItem:
    return SubstrateItem(
        id=belief.id,
        kind="record",
        text=belief.claim.text,
        payload={
            "key": _claim_key(belief.claim.text),
            "claim": {
                "text": belief.claim.text,
                "produced_by": belief.claim.produced_by,
                "quantity": belief.claim.quantity,
            },
            "credence": belief.credence,
            "evidence_history": [asdict(e) for e in belief.evidence_history],
        },
        tags=[_BELIEF_TAG],
        scope=belief.scope,
        provenance={"kind": _BELIEF_TAG},
    )


def _from_item(item: SubstrateItem) -> BeliefItem:
    p = item.payload
    c = p["claim"]
    return BeliefItem(
        id=item.id,
        claim=Claim(text=c["text"], produced_by=dict(c.get("produced_by", {})), quantity=c.get("quantity")),
        credence=p["credence"],
        evidence_history=[EvidenceEntry(**e) for e in p.get("evidence_history", [])],
        scope=item.scope,
    )
