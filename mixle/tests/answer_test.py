"""Answer-from-substrate (S3 seed): retrieve -> assemble -> answer or ABSTAIN, always cited."""

import unittest

from mixle.substrate import Answer, ContextBudget, Substrate, SubstrateItem, answer_from_substrate
from mixle.telemetry import Telemetry


def _first_line(_question, context):
    """A stand-in answerer: echo the first context line (a real one is a student / LLM / rule)."""
    lines = [ln for ln in context.split("\n") if ln.strip() and not ln.startswith("#")]
    return lines[0] if lines else "(none)"


def _refund_shard():
    s = Substrate()
    s.add("text", "the refund policy allows returns within 30 days of purchase")
    s.add("text", "refunds for defective items are processed immediately no restocking fee")
    return s  # 2 items -> deterministic lexical retrieval


class AnswerTest(unittest.TestCase):
    def test_covered_question_is_answered_with_citations(self):
        s = _refund_shard()
        a = answer_from_substrate(s, "refund policy returns", _first_line, budget=ContextBudget(max_chars=300))
        self.assertIsInstance(a, Answer)
        self.assertFalse(a.abstained)
        self.assertIsNotNone(a.answer)
        self.assertGreaterEqual(len(a.citations()), 1)  # never answered without provenance
        self.assertGreater(a.confidence, 0.0)

    def test_confidence_in_unit_interval(self):
        s = _refund_shard()
        a = answer_from_substrate(s, "refund policy", _first_line)
        self.assertGreaterEqual(a.confidence, 0.0)
        self.assertLessEqual(a.confidence, 1.0)


class AbstentionTest(unittest.TestCase):
    def test_abstains_and_never_calls_the_answerer(self):
        s = _refund_shard()
        called = {"v": False}

        def fabricator(q, c):
            called["v"] = True
            return "made up"

        a = answer_from_substrate(s, "airspeed velocity unladen swallow", fabricator, min_confidence=0.9)
        self.assertTrue(a.abstained)
        self.assertIsNone(a.answer)
        self.assertFalse(called["v"])  # the key property: no fabrication on thin evidence
        self.assertIn("abstained", a.note)

    def test_abstains_when_below_min_evidence(self):
        s = Substrate()  # empty substrate -> nothing to cite
        a = answer_from_substrate(s, "anything", _first_line, min_evidence=1)
        self.assertTrue(a.abstained)
        self.assertEqual(len(a.evidence), 0)


def _five_item_shard():
    s = Substrate()
    for i in range(5):
        s.add("text", f"widget policy alpha beta gamma document number {i} " + ("filler " * 40))
    return s  # 5 items, well under the embedder's 8-item floor -> deterministic lexical retrieval


class MinEvidenceGatingHonestyTest(unittest.TestCase):
    """MXR-080-0257: min_evidence must gate -- and abstention must report -- the FINAL packet's
    aligned, actually-delivered evidence (packet.items), never the pre-budget retrieved count."""

    def test_audit_scenario_min_evidence_abstains_when_budget_drops_evidence(self):
        # The audit's own scenario: 5 items are retrieved (>= min_evidence=3), but a budget this tight
        # lets only 1 survive into the packet the answerer actually sees. Checking the gate against the
        # pre-budget count of 5 would wrongly let this through; it must abstain instead.
        s = _five_item_shard()

        def _fabricator(_q, _c):
            raise AssertionError("must not be called: min_evidence should abstain before answering")

        a = answer_from_substrate(
            s,
            "widget policy alpha beta gamma document",
            _fabricator,
            budget=ContextBudget(max_chars=400, max_items=20),
            min_evidence=3,
            min_confidence=0.0,
            compress=False,
        )
        self.assertEqual(a.context.n_candidates, 5)  # 5 were retrieved...
        self.assertEqual(len(a.context.items), 1)  # ...but only 1 survived budgeting
        self.assertTrue(a.abstained)  # the gate must fail on the delivered count, not pass on 5
        self.assertIsNone(a.answer)

    def test_min_evidence_passes_when_enough_evidence_survives_budgeting(self):
        # Control: same shard and the same min_evidence=3, but a budget generous enough that 3 items
        # genuinely survive into the packet -- this must answer, not abstain.
        s = _five_item_shard()
        a = answer_from_substrate(
            s,
            "widget policy alpha beta gamma document",
            _first_line,
            budget=ContextBudget(max_chars=1300, max_items=20),
            min_evidence=3,
            min_confidence=0.0,
            compress=False,
        )
        self.assertEqual(a.context.n_candidates, 5)
        self.assertEqual(len(a.context.items), 3)
        self.assertFalse(a.abstained)
        self.assertIsNotNone(a.answer)
        self.assertGreaterEqual(len(a.evidence), 3)

    def test_abstention_reports_only_delivered_evidence(self):
        # Even when abstention is triggered by the confidence floor rather than min_evidence, the
        # reported `evidence` must be exactly what the packet delivered -- never the broader
        # pre-budget retrieved set (previously this reported all 5 retrieved items, 4 of which were
        # never rendered into the context the answerer would have seen).
        s = _five_item_shard()

        def _fabricator(_q, _c):
            raise AssertionError("must not be called on abstention")

        a = answer_from_substrate(
            s,
            "widget policy alpha beta gamma document",
            _fabricator,
            budget=ContextBudget(max_chars=400, max_items=20),
            min_evidence=1,
            min_confidence=0.99,  # force abstention via the confidence floor, not min_evidence
            compress=False,
        )
        self.assertTrue(a.abstained)
        self.assertEqual(a.context.n_candidates, 5)
        self.assertEqual(len(a.context.items), 1)
        delivered_ids = {i.id for i in a.context.items}
        reported_ids = {i.id for i in a.evidence}
        self.assertEqual(reported_ids, delivered_ids)  # reported evidence == actually delivered
        self.assertEqual(len(a.evidence), 1)  # not the 5 that were merely retrieved


class MultiHopAnswerTest(unittest.TestCase):
    def test_answers_over_a_lineage_chain(self):
        s = Substrate()
        s.put(SubstrateItem(kind="trace", text="lineage zeta eta theta", id="t1"))
        s.put(SubstrateItem(kind="artifact", text="pricing model regressor", links=["t1"], id="a1"))
        s.put(SubstrateItem(kind="text", text="the checkout price is sometimes wrong", links=["a1"], id="d1"))
        a = answer_from_substrate(s, "checkout price wrong", _first_line, hops=2, budget=ContextBudget(max_chars=400))
        self.assertFalse(a.abstained)
        self.assertGreaterEqual(len({i.kind for i in a.evidence}), 2)  # cited evidence spans kinds


class SerializeTest(unittest.TestCase):
    def test_as_dict_carries_citations(self):
        s = _refund_shard()
        d = answer_from_substrate(s, "refund policy", _first_line).as_dict()
        self.assertIn("citations", d)
        self.assertIn("confidence", d)
        self.assertIn("answer", d)


class TelemetryTest(unittest.TestCase):
    def test_emits_answer_and_abstain_events(self):
        s = _refund_shard()
        tel = Telemetry()
        answer_from_substrate(s, "refund policy", _first_line, telemetry=tel)
        answer_from_substrate(s, "unrelated nonsense query zzz", _first_line, min_confidence=0.99, telemetry=tel)
        # the 'reason' kind is shared with retrieve (list choice); the answer step tags action='answer'
        choices = {e.choice for e in tel.events(kind="reason") if e.features.get("action") == "answer"}
        self.assertEqual(choices, {"answer", "abstain"})


if __name__ == "__main__":
    unittest.main()
