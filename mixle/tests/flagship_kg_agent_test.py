"""G: the ontology-constrained KG agent flagship — decode, typed completion, KG-RAG, each checked."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from mixle.reason.graph_llm import GraphLLM
from mixle.reason.ontology import Ontology, constrained_decode
from mixle.substrate.act import investigate
from mixle.substrate.kg_rag import kg_action


def _ont():
    return (
        Ontology()
        .add_class("Person")
        .add_class("Organization")
        .add_class("City")
        .add_relation("works_at", "Person", "Organization")
        .add_relation("headquartered_in", "Organization", "City")
    )


TYPES = {"ada": "Person", "bob": "Person", "acme": "Organization", "globex": "Organization", "paris": "City"}


def _decode():
    rng = np.random.RandomState(0)

    def extractor(prompt):
        out = ["ada|works_at|acme", "bob|works_at|globex", "acme|headquartered_in|paris"]
        if rng.rand() < 0.5:
            out.append("paris|works_at|ada")
        return ";".join(out)

    llm = GraphLLM(extractor, lambda s: [tuple(t.split("|")) for t in s.split(";") if t], n=20)
    return constrained_decode(llm, "extract", _ont(), TYPES, floor=0.5)


class KgAgentFlagshipTest(unittest.TestCase):
    def test_reliable_facts_are_asserted_and_hallucination_rejected(self):
        dec = _decode()
        self.assertIn(("ada", "works_at", "acme"), dec.asserted())
        self.assertNotIn(("paris", "works_at", "ada"), dec.asserted())
        self.assertTrue(any(tuple(r["triple"]) == ("paris", "works_at", "ada") for r in dec.rejected))

    def test_kg_rag_answers_typed_and_abstains(self):
        dec = _decode()
        action = kg_action(dec.asserted(), ontology=_ont(), types=TYPES, description="org chart ada bob acme globex")
        answerer = lambda q, ctx: ctx.splitlines()[0] if ctx else ""  # noqa: E731
        ans = investigate("where does ada work", [action], answerer, min_confidence=0.2)
        self.assertEqual(ans.answer, "ada works_at acme")
        off = investigate("boiling point of xenon", [action], answerer, min_confidence=0.2)
        self.assertTrue(off.abstained)  # no linked entity -> no fabricated answer


if __name__ == "__main__":
    unittest.main()
