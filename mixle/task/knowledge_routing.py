"""M3 -- problem decomposition & tool/model routing over the IC-10 catalog (M3-core wiring).

``route_task`` turns one compound question into ordered sub-tasks and explicit, typed prerequisite
gaps -- never hidden prose intermediates -- matches each gap against the IC-10 catalog by declared
domain/schema, and drives the existing :func:`~mixle.task.orchestrate.orchestrate` loop to resolve
what it can. Every tool result is normalized into an IC-13-shaped ``KnowledgeItem``/``KnowledgeDelta``
dict with model/tool attribution and a receipt; nothing free-form ever becomes a canonical delta. Core
never imports ``mixle_knowledge`` -- everything here is a plain, dependency-free dict envelope shaped
to the frozen IC-13 contract (``notes/exec/contracts.md``), so M1c/M2a can validate and persist it.

This module reuses, rather than reinvents, three pieces of existing machinery:

* :class:`~mixle.task.task_decomposition.DecompositionProposer` -- an outcome-trained proposer over
  ordered sub-task sequences; ``proposer.plan_model.sample(rng)`` gives the decomposition order.
* :func:`~mixle.task.orchestrate.orchestrate` -- the step/re-plan/stop controller loop; this module
  supplies a trivial sequential plan model plus a :class:`~mixle.task.orchestrate.World` that resolves
  one gap per step.
* :class:`~mixle.scientist.ResearchProposal` -- the existing "here is how to find out" abstention
  output; :func:`research_proposal_to_gap` folds it into the same IC-13 gap shape rather than
  maintaining a second missing-knowledge lifecycle.

Pre-M5, the decomposition itself is heuristic/seeded (``proposer`` is fit on a seed corpus, not
learned online) -- that is an explicit non-goal here, deferred to M5.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mixle.task.catalog_router import CatalogEntry
from mixle.task.orchestrate import orchestrate

if TYPE_CHECKING:
    from mixle.scientist import ResearchProposal
    from mixle.task.task_decomposition import DecompositionProposer

__all__ = ["KnowledgeOrchestrationResult", "research_proposal_to_gap", "route_task"]


@dataclass
class KnowledgeOrchestrationResult:
    """An IC-13-shaped, dependency-free dict envelope: the answer, the IC-5 trace, an aggregate
    IC-13 ``KnowledgeDelta``-shaped dict, and the gaps still open after this call. M1c/M2a validate
    and persist ``delta``/``remaining_gaps`` against the real ``mixle_knowledge`` contracts; core
    never imports that package itself."""

    answer: Any
    trace: dict[str, Any]
    delta: dict[str, Any]
    remaining_gaps: list[dict[str, Any]]


def _stable_seed(question: str) -> int:
    """A deterministic RNG seed derived from the question text, so the same compound query always
    proposes the same sub-task order (no hidden global-RNG dependence)."""
    return int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16)


def _dedupe_preserve_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _new_gap(question: str, domain: str, *, index: int) -> dict[str, Any]:
    """A typed prerequisite gap for one sub-task, IC-13 ``KnowledgeGap``-shaped: an explicit
    ``required_schema``/``acceptance_criteria`` pair the decomposer emits instead of a hidden prose
    intermediate."""
    return {
        "id": f"gap-{index}-{domain}",
        "question": f"{question} :: requires {domain} evidence",
        "required_schema": {"type": "object", "domain": domain},
        "acceptance_criteria": [f"a verified {domain} item resolves this gap"],
        "status": "open",
        "priority": 50,
        "owner": None,
        "attempts": [],
        "resolved_by_item_ids": [],
    }


def research_proposal_to_gap(proposal: ResearchProposal, *, gap_id: str) -> dict[str, Any]:
    """Map an existing :class:`~mixle.scientist.ResearchProposal` (an abstention's "here is how to
    find out") into the frozen IC-13 gap shape, rather than maintaining a second missing-knowledge
    lifecycle alongside it. ``ResearchProposal`` carries no structured schema for the missing evidence,
    so ``required_schema`` is a best-effort description; ``acceptance_criteria`` is the proposal's own
    ranked acquisition options (mirroring :meth:`~mixle.scientist.ResearchProposal.render`'s top-4
    slice). ``nearest_knowledge``/``note`` have no home in the frozen ``KnowledgeGap`` fields and are
    intentionally dropped rather than smuggled in as ad hoc extra keys."""
    criteria = [str(opt["how"]) for opt in (proposal.options or [])[:4] if opt.get("how")]
    return {
        "id": gap_id,
        "question": proposal.question,
        "required_schema": {"type": "object", "description": proposal.missing},
        "acceptance_criteria": criteria,
        "status": "open",
        "priority": 50,
        "owner": None,
        "attempts": [],
        "resolved_by_item_ids": [],
    }


def _output_properties(entry: CatalogEntry) -> set[str]:
    schema = entry.schema or {}
    out = schema.get("output", schema)
    return set((out or {}).get("properties") or {})


def _candidates_for_gap(gap: dict[str, Any], catalog: list[CatalogEntry]) -> list[CatalogEntry]:
    """Match a gap's ``required_schema`` against the catalog's declared domain/output schemas and
    return compatible entries cheapest-and-most-reliable first. Only entries carrying a verifier are
    eligible: an un-verifiable tool's output can never resolve a gap on its own (step 5 of the M3
    algorithm), so it is not a routing candidate here at all."""
    required = gap.get("required_schema") or {}
    domain = required.get("domain")
    candidates = [e for e in catalog if e.verifier is not None and e.reliability > 0]
    if domain is not None:
        candidates = [e for e in candidates if e.owner == domain]
    req_props = set(required.get("properties") or {})
    if req_props:
        candidates = [e for e in candidates if req_props <= _output_properties(e)]
    return sorted(candidates, key=lambda e: (e.cost, -e.reliability, e.id))


def _match_existing_item(gap: dict[str, Any], known_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A gap is already answerable if the input bundle carries an item whose declared domain matches
    -- no tool call needed; the bundle's own evidence resolves it directly."""
    domain = (gap.get("required_schema") or {}).get("domain")
    if domain is None:
        return None
    for item in known_items:
        if (item.get("metadata") or {}).get("domain") == domain:
            return item
    return None


def _canonical_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _to_knowledge_item(gap: dict[str, Any], entry: CatalogEntry, raw: Any) -> dict[str, Any]:
    """Normalize a raw tool result into an IC-13-shaped ``KnowledgeItem`` dict -- never the raw
    unstructured result itself."""
    payload = raw if isinstance(raw, dict) else {"value": raw}
    schema = entry.schema or {}
    return {
        "id": f"item-{gap['id']}-{entry.id}",
        "kind": "artifact",
        "modality": "structured",
        "schema_uri": schema.get("schema_uri", "mixle://schema/catalog-result/1"),
        "schema_version": "1.0.0",
        "content_hash": _canonical_hash(payload),
        "payload": payload,
        "provenance": [{"tool_or_model": entry.id, "kind": "tool"}],
        "metadata": {"domain": entry.owner, "gap_id": gap["id"]},
    }


def _verdict_passed(verdict: Any) -> bool:
    if verdict is None:
        return False
    if isinstance(verdict, dict):
        return bool(verdict.get("passed"))
    return bool(getattr(verdict, "passed", False))


def _attempt(*, tool_or_model: str | None, query: str, status: str, produced_item_ids: list[str]) -> dict[str, Any]:
    return {
        "actor": "mixle.task.knowledge_routing.route_task",
        "query": query,
        "tool_or_model": tool_or_model,
        "status": status,
        "produced_item_ids": produced_item_ids,
    }


def _accepts_known_items(invoke: Any) -> bool:
    """Whether ``invoke`` declares a parameter literally named ``known_items`` it wants the
    accumulated real knowledge items passed into (called by that keyword, so the name must match
    exactly -- a second positional parameter with some other name is a different, unrelated
    interface, not an opt-in, and must not be called with a mismatched keyword).

    A one-arg ``invoke(gap)`` (every existing catalog entry written before this) is unaffected --
    this only widens what a *new* tool may opt into, via ``inspect.signature`` rather than a required
    interface change. ``TypeError``/``ValueError`` (an unintrospectable builtin/C callable) means
    "assume the old one-arg shape," the safe default.
    """
    import inspect

    try:
        params = inspect.signature(invoke).parameters
    except (TypeError, ValueError):
        return False
    known = params.get("known_items")
    return known is not None and known.kind in (
        known.POSITIONAL_OR_KEYWORD,
        known.KEYWORD_ONLY,
    )


def _invoke_tool(invoke: Any, gap: dict[str, Any], known_items: list[dict[str, Any]]) -> Any:
    """Call a catalog entry's ``invoke``, passing ``known_items`` (every real item resolved so far
    in this ``route_task`` call, plus whatever the input ``bundle`` carried) only if it opted in.

    This is the fix for a real, documented limitation (found and worked around by hand across
    several ``experiments/`` demos): previously ``invoke(gap)`` had no way to see another gap's
    already-resolved result within the same ``route_task`` call at all, forcing every genuinely
    multi-stage pipeline to bridge through separate sequential calls via the knowledge store. A tool
    that wants that now declares a second parameter and gets it directly; one that doesn't is called
    exactly as before.
    """
    if _accepts_known_items(invoke):
        return invoke(gap, known_items=known_items)
    return invoke(gap)


def _resolve_gap(gap: dict[str, Any], known_items: list[dict[str, Any]], catalog: list[CatalogEntry]) -> dict[str, Any]:
    """Resolve one gap: prefer evidence already in the bundle, else route to the cheapest matching,
    reliable, verifiable catalog entry and verify its result before ever treating it as canonical."""
    existing = _match_existing_item(gap, known_items)
    if existing is not None:
        return {"resolved": True, "item_ids": [existing["id"]], "item": None, "attempt": None}

    candidates = _candidates_for_gap(gap, catalog)
    if not candidates:
        return {
            "resolved": False,
            "item_ids": [],
            "item": None,
            "attempt": _attempt(
                tool_or_model=None, query=gap["question"], status="no_matching_tool", produced_item_ids=[]
            ),
        }

    entry = candidates[0]
    invoke = (entry.schema or {}).get("invoke")
    if invoke is None:
        return {
            "resolved": False,
            "item_ids": [],
            "item": None,
            "attempt": _attempt(
                tool_or_model=entry.id, query=gap["question"], status="no_executor", produced_item_ids=[]
            ),
        }

    raw = _invoke_tool(invoke, gap, known_items)
    item = _to_knowledge_item(gap, entry, raw)
    verdict = entry.verifier.verify(claim={"payload": item["payload"]}, context={"gap": gap, "entry": entry.id})
    if _verdict_passed(verdict):
        return {
            "resolved": True,
            "item_ids": [item["id"]],
            "item": item,
            "attempt": _attempt(
                tool_or_model=entry.id, query=gap["question"], status="resolved", produced_item_ids=[item["id"]]
            ),
        }
    return {
        "resolved": False,
        "item_ids": [],
        "item": None,
        "attempt": _attempt(tool_or_model=entry.id, query=gap["question"], status="failed", produced_item_ids=[]),
    }


@dataclass
class _RoutingWorld:
    """The :class:`~mixle.task.orchestrate.World` this module drives ``orchestrate`` against: one
    step resolves one gap. Never mutates the caller's input bundle -- ``known_items`` is read-only;
    newly produced items accumulate separately in ``add_items``."""

    gaps: list[dict[str, Any]]
    known_items: list[dict[str, Any]]
    catalog: list[CatalogEntry]
    add_items: list[dict[str, Any]] = field(default_factory=list)
    gap_updates: list[dict[str, Any]] = field(default_factory=list)
    tried_catalog_ids: set[str] = field(default_factory=set)
    _cursor: int = 0

    @property
    def done(self) -> bool:
        return self._cursor >= len(self.gaps)

    def step(self, action: dict[str, Any]) -> Any:
        idx = int(action["args"]["index"])
        gap = self.gaps[idx]
        self._cursor += 1
        outcome = _resolve_gap(gap, self.known_items + self.add_items, self.catalog)
        attempt = outcome["attempt"]
        if attempt is not None and attempt.get("tool_or_model"):
            self.tried_catalog_ids.add(attempt["tool_or_model"])
        if outcome["resolved"]:
            gap["status"] = "resolved"
            gap["resolved_by_item_ids"] = outcome["item_ids"]
            if outcome["item"] is not None:
                self.add_items.append(outcome["item"])
            self.gap_updates.append(
                {"gap_id": gap["id"], "status": "resolved", "resolved_by_item_ids": outcome["item_ids"]}
            )
        elif attempt is not None:
            gap["attempts"].append(attempt)
        return outcome

    def score(self) -> dict[str, Any]:
        n_resolved = sum(1 for g in self.gaps if g["status"] == "resolved")
        return {"resolved": n_resolved, "total": len(self.gaps)}


def _sequential_plan(n_gaps: int):
    """A trivial plan model: resolve gap 0, 1, 2, ... in order, then STOP -- the decomposition order
    was already decided by the proposer; ``orchestrate`` just needs a step/stop signal per gap."""

    def plan(_question: str, history: list[dict[str, Any]]) -> dict[str, Any] | None:
        i = len(history)
        if i >= n_gaps:
            return None
        return {"tool": "resolve_gap", "args": {"index": i}}

    return plan


def route_task(
    question: str,
    catalog: list[CatalogEntry],
    *,
    proposer: DecompositionProposer,
    budget: int,
    bundle: dict[str, Any] | None = None,
) -> KnowledgeOrchestrationResult:
    """Decompose ``question`` into ordered sub-tasks and explicit prerequisite gaps, route each gap
    to the cheapest reliable/verifiable catalog entry (or an item already in ``bundle``), drive
    :func:`~mixle.task.orchestrate.orchestrate` to resolve what it can, and return an IC-13-shaped
    answer/trace/delta/remaining-gaps envelope. Core emits plain dicts and never imports
    ``mixle_knowledge``; M1c/M2a validate and persist this envelope against the real contracts."""
    bundle = dict(bundle or {})
    known_items = list(bundle.get("items") or [])
    seed_gaps = list(bundle.get("gaps") or [])

    rng = np.random.RandomState(_stable_seed(question))
    sampled = proposer.plan_model.sample(rng)
    domains = _dedupe_preserve_order(sampled) or sorted({e.owner for e in catalog})
    new_gaps = [_new_gap(question, domain, index=i) for i, domain in enumerate(domains)]
    all_gaps = seed_gaps + new_gaps

    world = _RoutingWorld(gaps=all_gaps, known_items=known_items, catalog=list(catalog))
    result = orchestrate(question, _sequential_plan(len(all_gaps)), world, budget=max(int(budget), len(all_gaps)))

    remaining = [g for g in world.gaps if g["status"] != "resolved"]
    delta = {
        "base_bundle_id": bundle.get("id", ""),
        "base_revision": int(bundle.get("revision", 1)),
        "produced_by": "mixle.task.knowledge_routing.route_task",
        "model_version": None,
        "add_items": world.add_items,
        "add_gaps": new_gaps,
        "supersede_item_ids": [],
        "gap_updates": world.gap_updates,
        "provenance": [{"tool_or_model": "route_task", "note": f"{len(all_gaps)} sub-task gaps"}],
    }
    answer = {
        "resolved_gap_ids": [g["id"] for g in world.gaps if g["status"] == "resolved"],
        "unresolved_gap_ids": [g["id"] for g in remaining],
        "catalog_ids_considered": sorted(world.tried_catalog_ids),
    }
    return KnowledgeOrchestrationResult(
        answer=answer,
        trace=result.trace.to_json(),
        delta=delta,
        remaining_gaps=remaining,
    )
