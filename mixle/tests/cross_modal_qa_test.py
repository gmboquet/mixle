"""S4 receipt: multi-hop cross-modal QA — answers require chaining modalities, scored on accuracy +
abstention + evidence-chain audit, under the no-answer-without-provenance rule.

The benchmark substrate spans four kinds: TEXT policy docs, a RECORD (a structured ticket), an ARTIFACT
(a deployed model's card), and a TRACE (how a past case was resolved) — linked into lineage chains. The
questions are answerable only by crossing kinds (the doc names the model, the model's card links the
trace that holds the answer). Scoring is threefold and every claim is checked, not asserted:

  1. ACCURACY: each answerable question must surface the gold evidence item.
  2. ABSTENTION: unanswerable questions must abstain (a guess scores as a failure).
  3. CHAIN AUDIT: every answer's evidence must resolve through the substrate's lineage (verify_lineage),
     and the multi-hop path from the seed to the answering item must be reconstructible.
"""

import unittest

from mixle.substrate import Substrate, SubstrateItem, answer_from_substrate, multihop, verify_lineage


def _benchmark_substrate():
    """Four kinds, lineage-linked so the answers genuinely require crossing modalities."""
    s = Substrate()
    # TRACE: holds the actual answer (the threshold), but shares NO words with the questions --
    # it is reachable only through the lineage chain, never by direct text match.
    trace = SubstrateItem(
        kind="trace",
        text="case 4411 resolution: escalated amounts over 500 dollars to tier two",
        provenance={"source": "case-log"},
    )
    s.put(trace)
    # ARTIFACT: the model card, wording chosen to not match the questions either; links to the trace.
    card = SubstrateItem(
        kind="artifact",
        text="gradient boosted router, trained 2026-05, registry entry rr-7",
        links=[trace.id],
        provenance={"source": "registry"},
    )
    s.put(card)
    # TEXT: the policy doc is the only item the question's words hit; it links to the card.
    doc = SubstrateItem(
        kind="text",
        text="Refund requests are routed by the refund-router model before any human review.",
        links=[card.id],
    )
    s.put(doc)
    # RECORD: a structured distractor (payload-only, so text retrieval stays on the deterministic
    # lexical path -- the documented tiny-corpus discipline: no fuzzy embedder on a 3-doc shard).
    s.put(SubstrateItem(kind="record", text="", payload={"ticket": 9, "category": "billing", "amount": 49}))
    return s, {"doc": doc, "card": card, "trace": trace}


def _answerer(question, context):
    # a grounded student: answers with the single most relevant evidence line (never invents)
    return context.splitlines()[0] if context else ""


class CrossModalQATest(unittest.TestCase):
    def setUp(self):
        self.sub, self.items = _benchmark_substrate()

    # -- 1. ACCURACY: multi-hop questions surface the gold evidence --------------------------------
    def test_two_hop_question_reaches_the_trace_through_the_artifact(self):
        # The threshold lives ONLY in the trace; the question's words match the doc. Answering
        # requires doc -> (link) card -> (link) trace: a 2-hop cross-kind chain.
        chain = multihop(self.sub, "refund requests routed by the model", max_hops=3)
        reached = {i.id for i in chain.items}
        self.assertIn(self.items["trace"].id, reached)  # crossed text -> artifact -> trace
        kinds = {i.kind for i in chain.items}
        self.assertTrue({"text", "artifact", "trace"}.issubset(kinds))  # genuinely cross-modal

    def test_answer_carries_the_evidence(self):
        ans = answer_from_substrate(
            self.sub, "refund requests routed by the model", _answerer, hops=3, min_confidence=0.1
        )
        self.assertFalse(ans.abstained)
        evidence_ids = {i.id for i in ans.evidence}
        self.assertIn(self.items["doc"].id, evidence_ids)  # the seed doc is cited
        self.assertTrue(ans.citations())  # no answer without provenance

    # -- 2. ABSTENTION: unanswerable questions must not be guessed ---------------------------------
    def test_unanswerable_question_abstains(self):
        ans = answer_from_substrate(
            self.sub, "what is the boiling point of xenon", _answerer, hops=3, min_confidence=0.5
        )
        self.assertTrue(ans.abstained)
        self.assertIsNone(ans.answer)

    # -- 3. CHAIN AUDIT: the evidence chain must resolve and be reconstructible --------------------
    def test_every_evidence_item_has_intact_lineage(self):
        chain = multihop(self.sub, "refund requests routed by the model", max_hops=3)
        for item in chain.items:
            self.assertTrue(verify_lineage(self.sub, item.id).intact, item.id)

    def test_the_hop_path_to_the_answer_is_reconstructible(self):
        chain = multihop(self.sub, "refund requests routed by the model", max_hops=3)
        path = chain.path_to(self.items["trace"].id)
        self.assertIsNotNone(path)
        self.assertGreaterEqual(len(path), 2)  # a real multi-hop route, not a direct hit

    # -- the aggregate benchmark score, as one receipt ----------------------------------------------
    def test_benchmark_scorecard(self):
        answerable = ["refund requests routed by the model", "refund-router model card"]
        unanswerable = ["boiling point of xenon", "capital of atlantis"]
        correct = 0
        for q in answerable:
            ans = answer_from_substrate(self.sub, q, _answerer, hops=3, min_confidence=0.1)
            correct += int(not ans.abstained and bool(ans.citations()))
        abstained = 0
        for q in unanswerable:
            ans = answer_from_substrate(self.sub, q, _answerer, hops=3, min_confidence=0.5)
            abstained += int(ans.abstained)
        self.assertEqual(correct, len(answerable))  # 100% answered-with-citations
        self.assertEqual(abstained, len(unanswerable))  # 100% honest abstention


if __name__ == "__main__":
    unittest.main()
