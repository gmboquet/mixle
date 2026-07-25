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

from mixle.substrate.core import _require_count

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _materialize_triples(triples: Any) -> tuple[tuple[Any, Any, Any], ...]:
    """Consume ``triples`` exactly once into an immutable tuple of validated 3-tuples (MXR-080-0253).

    ``triples`` can be any iterable, including a one-shot generator. :func:`kg_action` used to walk
    it more than once -- once (twice, in fact: a head-entity pass and a separate tail-entity pass) to
    advertise an entity inventory at construction, again later inside ``_run`` to actually answer a
    question -- and :func:`retrieve_triples` re-walked whatever it was handed on every call too. A
    generator is exhausted by its first pass, so a reproduced action built from one advertised an
    entity (from whichever pass got there first) and then returned no fact for it, ever: not stale
    data, a structural guarantee that was never met. Materializing once, here, at the single point
    every consumer passes through, makes every later iteration -- however many, from however many call
    sites -- replay the same fixed data.

    Arity is validated in the same pass: before this check, a short entry (e.g. a 2-tuple) raised an
    opaque ``IndexError`` wherever its missing slot was first read -- often nowhere near construction,
    with no indication which entry was at fault -- and a long one (e.g. a 4-tuple) was silently
    accepted by the entities-only code (inventory building, hit filtering) and only failed later, and
    inconsistently, wherever something finally unpacked all of it (``for h, r, t in ...``, or
    :meth:`~mixle.reason.ontology.Ontology.filter_triples`'s ``check_triple(*tr, types)``). Both are
    rejected here, loudly and at the boundary, instead.
    """
    out: list[tuple[Any, Any, Any]] = []
    for i, t in enumerate(triples):
        tup = tuple(t)
        if len(tup) != 3:
            raise ValueError(
                f"triples[{i}] must be a 3-tuple (subject, relation, object); got {len(tup)} element(s): {tup!r}"
            )
        out.append(tup)
    return tuple(out)


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
    inject a type-invalid fact as evidence.

    ``k`` must be an exact, non-negative :class:`int` (MXR-080-0253, see
    :func:`~mixle.substrate.core._require_count`) -- a negative ``k`` used to fall through to ordinary
    Python slicing (``hits[:-1]`` silently drops just the last hit) instead of being rejected.
    ``triples`` is materialized once, validated, into an immutable structure before any lookup runs
    (MXR-080-0253, see :func:`_materialize_triples`) -- safe no matter how many times ``triples`` is a
    one-shot iterable and no matter how many times this function is called with it."""
    k = _require_count(k, "k")
    triple_list = _materialize_triples(triples)
    inventory = {t[0] for t in triple_list} | {t[2] for t in triple_list}
    linked = link_entities(question, inventory)
    linked_set = set(linked)
    hits = [t for t in triple_list if t[0] in linked_set or t[2] in linked_set]

    rejected: list[dict[str, Any]] = []
    if ontology is not None:
        kept, rejected = ontology.filter_triples(hits, types or {})
        hits = kept
    return {"entities": linked, "facts": hits[:k], "rejected": rejected}


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
    ``description`` plus the KG's own entity inventory (queries naming a known entity score).

    ``triples`` is materialized exactly once, here, into an immutable, arity-validated structure
    (MXR-080-0253, see :func:`_materialize_triples`) -- both the inventory built below and every later
    ``_run`` call read from that same materialized data, so a one-shot ``triples`` generator can no
    longer be exhausted by the inventory pass and come up empty when the action actually runs.
    ``k`` shares :func:`retrieve_triples`'s exact-non-negative-integer contract, checked up front so a
    bad ``k`` fails at construction rather than silently inside a fired action."""
    from mixle.substrate.act import Action

    k = _require_count(k, "k")
    triples = _materialize_triples(triples)
    inventory = sorted({str(t[0]) for t in triples} | {str(t[2]) for t in triples})
    desc = description or ("knowledge graph facts about " + " ".join(inventory[:20]))

    def _run(question: str) -> list[str]:
        out = retrieve_triples(triples, question, ontology=ontology, types=types, k=k)
        return [f"{h} {r} {t}" for h, r, t in out["facts"]]

    return Action(name=name, kind="retrieve", run=_run, cost=cost, description=desc, base_score=0.0)
