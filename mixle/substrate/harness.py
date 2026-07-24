"""Application harnesses around a configured reasoner.

A :class:`Harness` wraps a :class:`~mixle.substrate.reasoner.Reasoner` with
input validation, an immutable per-harness action whitelist, secret-redaction
guardrails applied at every boundary (input, action evidence, escalation
payload, retained trace), and an optional escalation callback. Each request
returns a :class:`HarnessResult` whose status is ``refused``, ``answered``, or
``escalated``.

``support_triage_harness`` and ``monitoring_harness`` provide ready-made
templates. ``register_harness`` and ``find_harnesses`` store harnesses as
scoped substrate artifacts so teams can discover reusable shells.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from mixle.substrate.act import Action, Investigation
from mixle.substrate.core import Substrate
from mixle.substrate.reasoner import Reasoner


@dataclass
class HarnessResult:
    """One request's outcome: which gate decided (refused/answered/escalated), and the evidence."""

    status: str  # 'refused' | 'answered' | 'escalated'
    answer: str | None = None
    reason: str = ""
    investigation: Any = None  # the underlying Investigation when the reasoner ran
    redactions: int = 0  # how many secrets the guardrails masked

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable harness result."""
        return {
            "status": self.status,
            "answer": self.answer,
            "reason": self.reason,
            "redactions": self.redactions,
        }


def _redact_action(action: Action) -> Action:
    """Wrap ``action`` so every evidence fragment it returns is redacted at the source.

    This closes the evidence-before-redaction gap where it actually happens: by the time
    :func:`~mixle.substrate.act.investigate` accumulates fragments into an
    :class:`~mixle.substrate.act.Investigation`'s steps, or hands them to the answerer, they are
    already clean -- regardless of which action kind produced them (retrieve/compute/simulate/create/
    delegate). Substrate-backed retrieval is separately redacted at the store boundary
    (``Substrate.put``); this covers every action, including the ones that never touch a substrate
    at all (a live compute/simulate/delegate call can return a secret the store boundary never sees).
    """

    def _run(question: str) -> list[str]:
        from mixle.substrate.security import redact_secrets

        return [redact_secrets(f) for f in action.run(question)]

    return replace(action, run=_run)


def _redact_investigation(inv: Investigation) -> Investigation:
    """Return a copy of ``inv`` with its answer and every step's evidence fragments redacted.

    Evidence is already redacted at the source (see ``_redact_action``), but this is the harness's
    last line of defense: whatever ``Investigation`` a caller retains via ``HarnessResult.investigation``
    must never show more than the top-level, masked answer does -- inspecting the full retained trace
    must not recover what the top-level masking was supposed to hide."""
    from mixle.substrate.security import redact_secrets

    steps = [replace(s, fragments=[redact_secrets(f) for f in s.fragments]) for s in inv.steps]
    answer = redact_secrets(inv.answer) if inv.answer is not None else None
    return replace(inv, answer=answer, steps=steps)


class Harness:
    """Schema + whitelist + guardrails + escalation around a reasoner (see module docstring).

    Args:
        reasoner: the configured :class:`Reasoner` (answerer + substrate + skills + actions).
        name / description: identity, used by the registry.
        validate: ``(request) -> None | str`` -- return an error string to REFUSE the request before
            any model runs (the input schema, as a callable so any validator plugs in).
        allowed_kinds: action kinds this harness may fire (whitelist; None = all).
        escalate: ``(request, result) -> str`` -- called on abstention with the REDACTED request and a
            redacted :class:`~mixle.substrate.act.Investigation`; its return is the escalated answer
            handed back (e.g. a ticket id). None = abstentions surface as 'escalated' with no handler
            note.
        min_confidence: the answer bar (passed through to ``ask``).
        on_result: optional UI hook, called with every HarnessResult (fire-and-forget).

    Action-view immutability: ``reasoner._actions`` is a shared, mutable structure -- other ``Harness``
    instances built from the same reasoner, or a later ``reasoner.add_action()`` call, can change it at
    any time. So that one harness's whitelist can never be corrupted by another harness's construction,
    or bypassed by a later ``add_action()``, a ``Harness`` never mutates ``reasoner._actions`` and never
    re-reads ``reasoner.actions`` after ``__init__``: it snapshots its own immutable action view --
    the whitelist-filtered (or, with ``allowed_kinds=None``, full) action list, each action wrapped to
    redact its own evidence -- exactly once, at construction time, and every request the harness ever
    serves uses only that view. Attach every action a harness should be able to use to the reasoner
    BEFORE constructing that harness; ``reasoner.add_action()`` calls (or another harness's
    construction) after that point affect only the shared reasoner and any harnesses built afterward,
    never one already built. The frozen view is available for inspection as ``harness.actions``.
    """

    def __init__(
        self,
        reasoner: Reasoner,
        *,
        name: str,
        description: str = "",
        validate: Callable[[str], str | None] | None = None,
        allowed_kinds: tuple[str, ...] | None = None,
        escalate: Callable[[str, Any], str] | None = None,
        min_confidence: float = 0.15,
        on_result: Callable[[HarnessResult], None] | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.name = name
        self.description = description
        self.validate = validate
        self.allowed_kinds = allowed_kinds
        self.escalate = escalate
        self.min_confidence = min_confidence
        self.on_result = on_result
        # immutable per-harness action view (see class docstring): snapshot + redaction-wrap once, here,
        # from whatever the reasoner exposes right now. Never written back to `reasoner._actions`, and
        # never re-derived from `reasoner.actions` after this point -- see `handle()`.
        allowed = set(allowed_kinds) if allowed_kinds is not None else None
        kept = [a for a in reasoner.actions if allowed is None or a.kind in allowed]
        self._action_view: tuple[Action, ...] = tuple(_redact_action(a) for a in kept)

    @property
    def actions(self) -> tuple[Action, ...]:
        """This harness's own immutable, redaction-wrapped action view (frozen at construction)."""
        return self._action_view

    def handle(self, request: str) -> HarnessResult:
        """Run one request through every gate: schema -> guardrails -> reasoner -> escalation."""
        from mixle.substrate.security import detect_secrets, redact_secrets

        # 1. schema: refuse before any model runs
        if self.validate is not None:
            problem = self.validate(request)
            if problem:
                return self._emit(HarnessResult(status="refused", reason=f"schema: {problem}"))

        # 2. input guardrail: secrets never reach an action, an index, or (below) an escalation payload
        n_red = len(detect_secrets(request).findings)
        clean = redact_secrets(request) if n_red else request

        # 3. the reasoner, over THIS harness's own immutable action view -- never `reasoner.actions`,
        #    so no other harness and no later `reasoner.add_action()` can change what this call can fire
        inv = self.reasoner.ask(clean, min_confidence=self.min_confidence, actions=self._action_view)

        # 4. redact once, up front: the answer and every step's evidence fragments. Action evidence is
        #    already redacted at the source (the action view's wrapped actions), but this also covers
        #    the answerer's own synthesis, and it is what BOTH the escalation callback and the returned
        #    HarnessResult/Investigation are allowed to see -- one redacted view, used everywhere below.
        redacted_inv = _redact_investigation(inv)

        # 5. abstention -> escalation policy (never a silent drop, never a guess); the callback gets the
        #    redacted request and a redacted investigation -- an escalation ticket must never carry the
        #    very secret the guardrails just stripped
        if inv.abstained:
            note = self.escalate(clean, redacted_inv) if self.escalate is not None else ""
            return self._emit(
                HarnessResult(
                    status="escalated",
                    reason=inv.note,
                    answer=note or None,
                    investigation=redacted_inv,
                    redactions=n_red,
                )
            )

        # 6. output guardrail: the top-level answer and the retained investigation share one redacted view
        return self._emit(
            HarnessResult(
                status="answered", answer=redacted_inv.answer or "", investigation=redacted_inv, redactions=n_red
            )
        )

    def _emit(self, result: HarnessResult) -> HarnessResult:
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:  # noqa: BLE001 - a UI hook must never break the request path
                pass
        return result


# -- R2: domain templates -------------------------------------------------------------------------


def support_triage_harness(
    substrate: Substrate,
    answerer: Callable[[str, str], str],
    *,
    escalate: Callable[[str, Any], str] | None = None,
    max_chars: int = 2000,
) -> Harness:
    """Support triage: retrieve-only over the team's knowledge, refuse empty/oversized requests,
    escalate anything the knowledge base cannot support -- the canonical 'never guess at a customer'."""

    def validate(req: str) -> str | None:
        if not req.strip():
            return "empty request"
        if len(req) > max_chars:
            return f"request over {max_chars} chars"
        return None

    reasoner = Reasoner(answerer, substrate=substrate, retrieve_min_score=0.2)
    return Harness(
        reasoner,
        name="support-triage",
        description="answer support questions from the knowledge base or escalate to a human",
        validate=validate,
        allowed_kinds=("retrieve",),
        escalate=escalate,
        min_confidence=0.3,
    )


def monitoring_harness(
    reasoner: Reasoner,
    *,
    escalate: Callable[[str, Any], str] | None = None,
) -> Harness:
    """Monitoring/alerting: compute + simulate allowed (run checks, what-ifs), no delegation out."""
    return Harness(
        reasoner,
        name="monitoring",
        description="run drift checks and what-ifs over deployed models; alert on trips",
        allowed_kinds=("retrieve", "compute", "simulate"),
        escalate=escalate,
        min_confidence=0.2,
    )


# -- R3: the harness registry on the substrate -----------------------------------------------------


def register_harness(substrate: Substrate, harness: Harness, *, scope: str = "local") -> str:
    """Index a harness on the substrate as a scoped artifact -- discoverable and shareable (P-scoped)."""
    return substrate.add(
        kind="artifact",
        text=f"harness {harness.name}: {harness.description}",
        payload={
            "harness": harness.name,
            "allowed_kinds": list(harness.allowed_kinds or []),
            "min_confidence": harness.min_confidence,
        },
        provenance={"origin": "harness-registry"},
        scope=scope,
        tags=["harness"],
    )


def find_harnesses(substrate: Substrate, query: str = "", *, scope: str | None = None) -> list[dict[str, Any]]:
    """Discover registered harnesses (optionally by query / scope). Returns their manifests."""
    items = [i for i in substrate.all(kind="artifact", scope=scope) if "harness" in i.tags]
    if query:
        q = set(query.lower().split())
        items = [i for i in items if q & set(i.text.lower().split())]
    return [{"id": i.id, "text": i.text, **i.payload, "scope": i.scope} for i in items]
