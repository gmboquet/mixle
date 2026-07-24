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

A claimed tier is not itself evidence (MXR-080-0243): a strong tier only earns its full claimed
strength when ``source_id`` actually resolves to something real -- a document, an artifact, another
belief, any genuine substrate item -- checked once, at assimilation time, and recorded on the entry
(:attr:`EvidenceEntry.verified`) so replay never needs the substrate again. An unresolved or fabricated
``source_id`` (missing, made up, or naming nothing in the store) is represented as unverified and earns
only :data:`MODEL_ASSERTION`-strength credence, no matter what tier it claims: a bare string is never,
by itself, a receipt.

Resolution is scope-respecting, all the way down (MXR-080-0244): ``source_id`` is looked up through an
:class:`~mixle.substrate.spaces.AccessPolicy`-authorized view of the caller's own ``scope``, never a raw
global lookup, and every transitive hop of a multi-belief proof chain is resolved through that SAME
authorized view -- a caller can neither borrow another scope's private grounding to inflate its own
credence nor learn, by probing, whether some other scope's belief happens to be grounded. An id that
exists but is not visible to the caller is indistinguishable from one that does not exist at all.

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

from scipy import special

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.spaces import AccessPolicy

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

# The closed set of valid evidence directions (MXR-080-0242): "+" supports the claim, "-" contradicts
# it. Before this was validated, ANY other string (a typo, "positive", "up") silently fell through to
# "-" in the credence math -- contradicting evidence nobody actually submitted.
_DIRECTIONS = frozenset({"+", "-"})

_BELIEF_TAG = "belief"


@dataclass
class Claim:
    """One atomic proposition (or typed quantity) pulled from a model's output."""

    text: str
    produced_by: dict[str, Any] = field(default_factory=dict)
    quantity: float | None = None


@dataclass
class EvidenceEntry:
    """One piece of evidence that moved a belief's credence, in the order it was applied.

    ``verified`` records whether ``source_id`` resolved to something real, through an authorized scoped
    view, at the moment this entry was assimilated (MXR-080-0243/0244) -- computed once, by
    :func:`assimilate`, and carried on the entry itself so :func:`credence_from_history` can replay it
    without ever touching the substrate again. Direct construction (tests, deserializing an
    already-decided entry) defaults to ``True``, matching ``direction``/``weight``/``time``'s own
    "already-legitimate input" defaults; :func:`assimilate` is the untrusted-input boundary and always
    computes and passes this explicitly rather than relying on the default.
    """

    source_id: str
    tier: str
    direction: str = "+"  # "+" supports the claim, "-" contradicts it
    weight: float = 1.0
    time: float = field(default_factory=time.time)
    verified: bool = True

    def __post_init__(self) -> None:
        # MXR-080-0242: a closed direction enum, and finite/non-negative weight and time -- validated
        # here so EVERY construction path (assimilate's append, a test building one directly, or
        # _from_item deserializing a stored entry) gets the same guarantee; nothing downstream (the
        # logistic update, a ranking sort) can ever see a NaN/inf value or a direction that silently
        # meant "contradicts" without anyone asking for that.
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"unknown evidence direction {self.direction!r}; expected '+' (supports) or '-' (contradicts)"
            )
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"evidence weight must be a finite, non-negative number, got {self.weight!r}")
        if not math.isfinite(self.time):
            raise ValueError(f"evidence time must be a finite number, got {self.time!r}")


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
    stored ``evidence_history`` through this always reproduces its current ``credence`` exactly.

    An entry's claimed ``tier`` only counts at its claimed strength when :attr:`EvidenceEntry.verified`
    is True (MXR-080-0243); an unverified claim of a strong tier contributes at
    :data:`MODEL_ASSERTION`-strength instead, and does not count toward ``has_real_support`` (so it
    cannot, by itself, lift a belief past :data:`MODEL_ASSERTION_CAP`) -- otherwise merely CLAIMING a
    strong tier would escape the cap even though its actual numeric contribution never did.
    """
    if not evidence_history:
        return 0.5  # neutral: no evidence yet
    logit = 0.0
    has_real_support = False
    for e in evidence_history:
        if e.weight <= 0:
            continue
        grounding = _is_grounding(e)
        strength = (_tier_strength(e.tier) if grounding else _TIER_STRENGTH[MODEL_ASSERTION]) * e.weight
        sign = 1.0 if e.direction == "+" else -1.0
        logit += sign * strength * _K
        if grounding:
            has_real_support = True
    # MXR-080-0242: scipy's expit is a numerically stable logistic -- unlike 1/(1+exp(-x)), it never
    # overflows math.exp for a large-magnitude logit (a big weight, or many entries), instead saturating
    # cleanly to 0.0 or 1.0 at the extremes.
    credence = float(special.expit(logit))
    if not has_real_support:
        credence = min(credence, MODEL_ASSERTION_CAP)
    return credence


def assimilate(
    sub: Substrate,
    claim: Claim,
    evidence: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    scope: str = "local",
    policy: AccessPolicy | None = None,
) -> BeliefItem:
    """Bayesian-ish update of the belief in ``claim`` from ``evidence`` -- never a binary write.

    Finds or creates the belief item (keyed on normalized claim text) and appends each evidence entry
    (``{"source_id", "tier", "direction": "+"/"-", "weight"}``); ``tier`` must be one of
    :data:`_TIER_STRENGTH` (``"model_assertion"`` plus :data:`mixle.doe.oracle.VERIFIABILITY_TIERS`),
    and ``direction``, when given, must be ``"+"`` or ``"-"`` (MXR-080-0242).

    Verification (MXR-080-0243): a tier above :data:`MODEL_ASSERTION` only earns its claimed strength
    when ``source_id`` resolves to a real substrate item through an authorized view of ``scope`` (see
    :func:`_resolve`); an unresolved, fabricated, or omitted ``source_id`` is recorded as unverified and
    contributes at :data:`MODEL_ASSERTION`-strength regardless of what it claims.

    Anti-laundering (MXR-080-0244): an entry whose ``source_id`` resolves back to THIS belief (a cycle)
    or to another belief item with no independent (non-model-assertion, verified) support of its own is
    stored with an effective weight of zero -- it cannot move the credence, though it stays in the trail
    for audit. Resolution -- both the verification check above and the laundering check -- goes through
    ``policy`` (a fresh, home-scope-plus-PUBLIC-only :class:`~mixle.substrate.spaces.AccessPolicy` when
    omitted), never a raw global lookup, so citing another scope's private belief is indistinguishable
    from citing something that does not exist at all: neither the credence it produces nor anything else
    observable here can be used to learn whether that other scope's belief exists or is grounded.
    """
    entries = [evidence] if isinstance(evidence, dict) else list(evidence)
    key = _claim_key(claim.text)
    existing = _find(sub, key, scope)
    belief = (
        _from_item(existing)
        if existing is not None
        else BeliefItem(id=uuid.uuid4().hex[:16], claim=claim, credence=0.5, evidence_history=[], scope=scope)
    )
    policy = policy if policy is not None else AccessPolicy()

    for raw in entries:
        tier = raw["tier"]
        _tier_strength(tier)  # validates; raises on an unrecognized tier rather than silently accepting one
        direction = raw.get("direction", "+")
        if direction not in _DIRECTIONS:
            raise ValueError(f"unknown evidence direction {direction!r}; expected '+' (supports) or '-' (contradicts)")
        weight = float(raw.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"evidence weight must be a finite, non-negative number, got {weight!r}")
        entry_time = float(raw["time"]) if "time" in raw else time.time()
        if not math.isfinite(entry_time):
            raise ValueError(f"evidence time must be a finite number, got {entry_time!r}")
        source_id = str(raw.get("source_id", ""))

        # MXR-080-0243: does source_id resolve to something real, through OUR authorized view? A
        # model_assertion entry needs no receipt -- it is always weak regardless, so short-circuit
        # before ever resolving it.
        verified = tier == MODEL_ASSERTION or _resolve(sub, source_id, scope, policy) is not None
        # MXR-080-0244: same authorized view, recursively, at every hop of the citation -- see _launders.
        if _launders(sub, source_id, belief.id, scope, policy):
            weight = 0.0

        belief.evidence_history.append(
            EvidenceEntry(
                source_id=source_id,
                tier=tier,
                direction=direction,
                weight=weight,
                time=entry_time,
                verified=verified,
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
    actually believes each item (never a "fact" vs "non-fact" partition, only credence).

    MXR-080-0245: the belief tag and ``min_credence`` are filtered over the FULL ``"record"``-kind
    candidate domain, not a fixed-size prefetch -- a store where ordinary (non-belief) records or
    low-credence beliefs rank higher by raw relevance can no longer crowd out qualifying beliefs that
    rank lower but still exist. ``sub.search`` still does the actual relevance scoring (this module
    never touches core.py's ranking); the fix is asking it for every ``"record"``-kind candidate in
    scope rather than an arbitrary ``3*k`` cut of them.
    """
    domain_size = len(sub.all(kind="record", scope=scope))
    hits = sub.search(query, k=max(domain_size, int(k)), kind="record", scope=scope)
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


def _is_grounding(e: EvidenceEntry) -> bool:
    """Whether one evidence entry counts as independent, non-model-assertion support (MXR-080-0243):
    a claimed tier above the model's-own-say-so floor that ALSO checked out as verified when it was
    assimilated, and still carries positive weight. A claimed strong tier that never resolved to
    anything real does not count, no matter what its ``tier`` string says -- otherwise merely claiming
    a strong tier would let a belief escape :data:`MODEL_ASSERTION_CAP` even though its actual numeric
    contribution was clamped to model-assertion strength. The single source of truth for "does this
    entry count as real support", shared by :func:`credence_from_history`, :func:`_has_real_support`,
    and :func:`_launders`."""
    return e.tier != MODEL_ASSERTION and e.verified and e.weight > 0


def _has_real_support(belief: BeliefItem) -> bool:
    return any(_is_grounding(e) for e in belief.evidence_history)


def _claim_key(text: str) -> str:
    return " ".join(text.lower().split())


def _find(sub: Substrate, key: str, scope: str) -> SubstrateItem | None:
    for item in sub.all(kind="record", scope=scope):
        if _BELIEF_TAG in item.tags and item.payload.get("key") == key:
            return item
    return None


def _resolve(sub: Substrate, source_id: str, scope: str, policy: AccessPolicy) -> SubstrateItem | None:
    """The item ``source_id`` names, iff it exists AND ``scope`` is authorized (by ``policy``) to read
    the scope it is actually stored in (MXR-080-0244) -- never a raw, scope-blind ``sub.get``.

    ``scope`` doubles as the authorization principal here, the same way
    :class:`~mixle.substrate.spaces.Space` uses its ``team`` as its own principal id: belief.py has no
    separate identity concept, so the operating scope IS the caller's identity for this check. An id
    that does not exist and one that exists but is private to a scope the caller cannot read both
    resolve to ``None`` -- indistinguishable to every caller of this function, so neither can be used to
    probe whether some other scope's item exists or what it contains.
    """
    if not source_id:
        return None
    item = sub.get(source_id)
    if item is None:
        return None
    if not policy.can_read(scope, item.scope):
        return None
    return item


def _launders(
    sub: Substrate,
    source_id: str,
    belief_id: str,
    scope: str,
    policy: AccessPolicy,
    _seen: set[str] | None = None,
) -> bool:
    """True iff citing ``source_id`` as evidence for ``belief_id`` would launder unearned credence:
    a direct or indirect cycle back to ``belief_id``, or a reference to another belief item that has
    no independent support of its own -- walked ALL THE WAY DOWN, not just one hop: ``source_id`` is
    laundering unless it resolves to something with at least one non-model-assertion, verified entry
    that is ITSELF not laundering (recursively). A ``source_id`` that is not itself a belief item in the
    substrate (a document, an oracle receipt, any genuine external reference) is never laundering --
    that is the recursion's base case.

    MXR-080-0244: every resolution -- this hop and every recursive one -- goes through :func:`_resolve`,
    i.e. through ``policy``'s view of ``scope``, never a raw global lookup. A belief in a scope the
    caller cannot read resolves to ``None`` here exactly like a nonexistent id would, so it can neither
    be used to ground the caller's belief nor, through the credence that results, disclose whether it
    exists or is itself grounded. The SAME ``scope``/``policy`` -- the ORIGINAL caller's, not whatever
    scope a cited belief happens to live in -- is threaded through every recursive call, so a private
    belief three hops down a proof chain is exactly as inaccessible as one cited directly.
    """
    if source_id == belief_id:
        return True
    seen = _seen or set()
    if source_id in seen:
        return True  # a cycle among referenced claims that never reaches belief_id directly
    ref_item = _resolve(sub, source_id, scope, policy)
    if ref_item is None or _BELIEF_TAG not in ref_item.tags:
        return False
    ref = _from_item(ref_item)
    next_seen = seen | {source_id}
    for e in ref.evidence_history:
        if _is_grounding(e) and not _launders(sub, e.source_id, belief_id, scope, policy, next_seen):
            return False  # ref has at least one genuinely independent, non-circular, verified support
    return True  # every one of ref's entries is model-assertion-only, unverified, or itself laundering


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
