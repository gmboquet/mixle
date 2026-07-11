"""Flagship app (G): the ontology-constrained knowledge-graph agent, end to end.

Classification: illustrative -- runs on small synthetic / stand-in data. It shows the
end-to-end workflow shape, not measured results on a real frontier-scale dataset. See
docs/example-execution-manifest.rst for which examples run on real public data.

An LLM-backed agent that maintains and answers from a knowledge graph -- but every fact must pass the
ontology, every assertion must clear a confidence floor, and every answer is typed retrieval over the
constrained graph. The pipeline:

  1. ONTOLOGY    -- classes, relation signatures, axioms (D1): the typed contract on knowledge.
  2. DECODE      -- a stochastic extractor's outputs are ontology-masked and confidence-floored (D2):
                    schema-violating hallucinations are structurally rejected, under-confident facts
                    withheld, never silently dropped.
  3. KG          -- the surviving facts fit a DistMult embedding; completion is ontology-typed (D1):
                    the model cannot place mass on a forbidden tail.
  4. KG-RAG      -- questions answer by entity linking + typed triple retrieval (D3), through the
                    reasoner with honest abstention.

Everything measured in-process; seconds, no GPU, no network.
"""

from __future__ import annotations

import numpy as np

from mixle.reason.graph_llm import GraphLLM
from mixle.reason.ontology import Ontology, OntologyConstrainedKG, constrained_decode
from mixle.substrate.act import investigate
from mixle.substrate.kg_rag import kg_action


def line(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    # 1. ONTOLOGY ------------------------------------------------------------------------------------
    ont = (
        Ontology()
        .add_class("Person")
        .add_class("Organization")
        .add_class("City")
        .add_relation("works_at", "Person", "Organization")
        .add_relation("lives_in", "Person", "City", "functional")
        .add_relation("headquartered_in", "Organization", "City")
    )
    types = {
        "ada": "Person",
        "bob": "Person",
        "acme": "Organization",
        "globex": "Organization",
        "paris": "City",
        "lyon": "City",
    }
    line("ONTOLOGY: the typed contract on knowledge")
    print(f"classes: {sorted(ont.classes)} | relations: {sorted(ont.relations)}")

    # 2. CONSTRAINED DECODE --------------------------------------------------------------------------
    rng = np.random.RandomState(0)

    def extractor(prompt: str) -> str:  # a stochastic 'LLM' fact extractor
        out = ["ada|works_at|acme", "bob|works_at|globex", "acme|headquartered_in|paris"]
        if rng.rand() < 0.5:
            out.append("paris|works_at|ada")  # a type-violating hallucination
        if rng.rand() < 0.2:
            out.append("ada|lives_in|lyon")  # asserted rarely: under-confident
        return ";".join(out)

    llm = GraphLLM(extractor, lambda s: [tuple(t.split("|")) for t in s.split(";") if t], n=25)
    dec = constrained_decode(llm, "extract the org chart", ont, types, floor=0.5)
    line("DECODE: only schema-consistent, confident facts are asserted")
    for t, c in dec.facts:
        print(f"  asserted  {t}  (confidence {c:.2f})")
    for r in dec.rejected:
        print(f"  REJECTED  {tuple(r['triple'])}  ({r['problems'][0]})")
    for t, c in dec.below_floor:
        print(f"  withheld  {t}  (confidence {c:.2f} < 0.5)")

    # 3. ONTOLOGY-TYPED KG COMPLETION ------------------------------------------------------------------
    from mixle.inference import optimize
    from mixle.stats.graphs.knowledge_graph import KnowledgeGraphEstimator

    ents = sorted(types)
    rels = sorted(ont.relations)
    e = {x: i for i, x in enumerate(ents)}
    r = {x: i for i, x in enumerate(rels)}
    train = [(e[h], r[rel], e[t]) for (h, rel, t) in dec.asserted()] * 30
    kg = optimize(
        train,
        KnowledgeGraphEstimator(num_entities=len(ents), num_relations=len(rels), dim=8),
        out=None,
        max_its=40,
        rng=np.random.RandomState(0),
    )
    ckg = OntologyConstrainedKG(kg, ont, entities=ents, relations=rels, types=types)
    line("KG: completion is typed -- mass only on schema-consistent tails")
    tail, p = ckg.complete("ada", "works_at")
    print(f"  complete(ada, works_at) = {tail} ({p:.3f})  [only Organizations were eligible]")
    print(f"  eligible tails: {sorted(ckg.tail_posterior('ada', 'works_at'))}")

    # 4. KG-RAG THROUGH THE REASONER --------------------------------------------------------------------
    line("KG-RAG: typed answers with honest abstention")
    action = kg_action(
        dec.asserted(), ontology=ont, types=types, description="org chart: who works where, ada bob acme globex"
    )
    answerer = lambda q, ctx: ctx.splitlines()[0] if ctx else ""  # noqa: E731
    for q in ["where does ada work", "what do we know about globex", "boiling point of xenon"]:
        inv = investigate(q, [action], answerer, min_confidence=0.2)
        print(f"  {q!r:32} -> " + (f"{inv.answer}" if not inv.abstained else "ABSTAINED (no linked entity)"))

    print("\nno fact without a schema, no assertion without confidence, no answer without provenance.")


if __name__ == "__main__":
    main()
