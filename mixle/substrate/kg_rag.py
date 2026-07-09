"""KG-RAG -- typed retrieval over a knowledge graph, with an entity-linking leaf (D3).

Text retrieval finds passages; a knowledge graph answers with *facts*. :func:`link_entities` is the
entity-linking leaf: it maps a question's tokens onto the KG's entity inventory (longest-name-first, so
"new york city" links before "york"). :func:`retrieve_triples` returns the facts about the linked
entities -- filtered through an :class:`~mixle.reason.ontology.Ontology` when one is given, so a
schema-violating triple in an unvalidated store is never served as evidence. :func:`kg_action` packages that as
a reasoner :class:`~mixle.substrate.act.Action`, so ``investigate()`` / the :class:`Reasoner` can buy
*typed* evidence: the fragment for ``(ada, lives_in, paris)`` reads ``ada lives_in paris``, citable and
checkable against the graph rather than parsed back out of prose.
"""

from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def link_entities(question: str, entities: Any) -> list[str]:
    """The entity-linking leaf: which KG entities does the question mention?

    Matches each entity's normalized name as a token subsequence of the question, longest name first so
    multi-word entities win over their substrings. Returns the linked entities in match order."""
    q = f" {_norm(question)} "
    ranked = sorted((str(e) for e in entities), key=lambda e: -len(_norm(e)))
    linked: list[str] = []
    claimed = q
    for ent in ranked:
        name = _norm(ent)
        if not name:
            continue
        token = f" {name} "
        if token in claimed:
            linked.append(ent)
            claimed = claimed.replace(token, " * ")  # a matched span can't also link a shorter entity
    return linked


def retrieve_triples(
    triples: Any,
    question: str,
    *,
    ontology: Any = None,
    types: dict[str, str] | None = None,
    k: int = 8,
) -> dict[str, Any]:
    """Typed KG retrieval: link the question's entities, return the (schema-valid) facts about them.

    Returns ``{entities, facts, rejected}`` -- ``facts`` are the triples touching a linked entity (head
    or tail), at most ``k``; when an ``ontology`` (+ entity ``types``) is supplied, schema-violating
    triples are excluded and reported under ``rejected`` with named reasons, so an unvalidated store cannot
    inject a type-invalid fact as evidence."""
    triple_list = [tuple(t) for t in triples]
    inventory = {t[0] for t in triple_list} | {t[2] for t in triple_list}
    linked = link_entities(question, inventory)
    linked_set = set(linked)
    hits = [t for t in triple_list if t[0] in linked_set or t[2] in linked_set]

    rejected: list[dict[str, Any]] = []
    if ontology is not None:
        kept, rejected = ontology.filter_triples(hits, types or {})
        hits = kept
    return {"entities": linked, "facts": hits[: int(k)], "rejected": rejected}


def kg_action(
    triples: Any,
    *,
    ontology: Any = None,
    types: dict[str, str] | None = None,
    name: str = "kg",
    cost: float = 1.0,
    description: str = "",
    k: int = 8,
) -> Any:
    """A reasoner RETRIEVE action over a knowledge graph (typed facts, not passages).

    Contributes one fragment per fact (``head relation tail``); nothing links -> no evidence, so the
    reasoner falls through honestly instead of forcing a match. Relevance comes from the action's
    ``description`` plus the KG's own entity inventory (queries naming a known entity score)."""
    from mixle.substrate.act import Action

    inventory = sorted({str(t[0]) for t in (tuple(x) for x in triples)} | {str(tuple(x)[2]) for x in triples})
    desc = description or ("knowledge graph facts about " + " ".join(inventory[:20]))

    def _run(question: str) -> list[str]:
        out = retrieve_triples(triples, question, ontology=ontology, types=types, k=k)
        return [f"{h} {r} {t}" for h, r, t in out["facts"]]

    return Action(name=name, kind="retrieve", run=_run, cost=cost, description=desc, base_score=0.0)
