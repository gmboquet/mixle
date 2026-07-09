"""Application harnesses around a configured reasoner.

A :class:`Harness` wraps a :class:`~mixle.substrate.reasoner.Reasoner` with
input validation, an action whitelist, secret-redaction guardrails, and an
optional escalation callback. Each request returns a :class:`HarnessResult`
whose status is ``refused``, ``answered``, or ``escalated``.

``support_triage_harness`` and ``monitoring_harness`` provide ready-made
templates. ``register_harness`` and ``find_harnesses`` store harnesses as
scoped substrate artifacts so teams can discover reusable shells.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


class Harness:
    """Schema + whitelist + guardrails + escalation around a reasoner (see module docstring).

    Args:
        reasoner: the configured :class:`Reasoner` (answerer + substrate + skills + actions).
        name / description: identity, used by the registry.
        validate: ``(request) -> None | str`` -- return an error string to REFUSE the request before
            any model runs (the input schema, as a callable so any validator plugs in).
        allowed_kinds: action kinds the reasoner may fire (whitelist; None = all).
        escalate: ``(request, result) -> str`` -- called on abstention; its return is the escalated
            answer handed back (e.g. a ticket id). None = abstentions surface as 'escalated' with no
            handler note.
        min_confidence: the answer bar (passed through to ``ask``).
        on_result: optional UI hook, called with every HarnessResult (fire-and-forget).
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
        if allowed_kinds is not None:
            # enforce the whitelist structurally: strip disallowed actions from the reasoner itself
            kept = [a for a in self.reasoner.actions if a.kind in set(allowed_kinds)]
            self.reasoner._actions = kept  # noqa: SLF001 - the harness owns its reasoner's surface

    def handle(self, request: str) -> HarnessResult:
        """Run one request through every gate: schema -> guardrails -> reasoner -> escalation."""
        from mixle.substrate.security import detect_secrets, redact_secrets

        # 1. schema: refuse before any model runs
        if self.validate is not None:
            problem = self.validate(request)
            if problem:
                return self._emit(HarnessResult(status="refused", reason=f"schema: {problem}"))

        # 2. input guardrail: secrets never reach an action or an index
        n_red = len(detect_secrets(request).findings)
        clean = redact_secrets(request) if n_red else request

        # 3. the reasoner, over the whitelisted action space
        inv = self.reasoner.ask(clean, min_confidence=self.min_confidence)

        # 4. abstention -> escalation policy (never a silent drop, never a guess)
        if inv.abstained:
            note = self.escalate(request, inv) if self.escalate is not None else ""
            return self._emit(
                HarnessResult(
                    status="escalated",
                    reason=inv.note,
                    answer=note or None,
                    investigation=inv,
                    redactions=n_red,
                )
            )

        # 5. output guardrail: the answer is redacted too (evidence may contain a stored secret)
        answer = redact_secrets(inv.answer or "")
        return self._emit(HarnessResult(status="answered", answer=answer, investigation=inv, redactions=n_red))

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
