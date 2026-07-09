"""Receipt: bind ledger + trace + calibration + provenance into one offline-re-verifiable artifact
(workstream H3)."""

import copy
import unittest

import numpy as np

from mixle.inference.explain import explain
from mixle.inference.receipt import Receipt, verify_receipt
from mixle.stats import CategoricalDistribution, CompositeDistribution, GaussianDistribution
from mixle.task.replay import ExecutionTrace, record_step


def _draw_normal(n: int, seed: int) -> list[float]:
    return np.random.RandomState(seed).normal(size=n).tolist()


_TOOLS = {"draw_normal": _draw_normal}


def _build_ledger():
    comp = CompositeDistribution((CategoricalDistribution({"a": 0.9, "b": 0.1}), GaussianDistribution(0.0, 1.0)))
    return explain(comp, ("a", 0.5))


def _build_trace():
    step = record_step(_TOOLS, "draw_normal", {"n": 3}, seed=7)
    return ExecutionTrace(request="demo", steps=[step])


class ReceiptVerifyTest(unittest.TestCase):
    def test_a_fully_populated_receipt_verifies_offline(self):
        receipt = Receipt(
            answer="a",
            produced_by="student-v1",
            ledger=_build_ledger(),
            trace=_build_trace(),
            calibration={"alpha": 0.1, "qhat": 0.83},
            provenance={"source_id": "corpus-42"},
        )
        report = verify_receipt(receipt, tools=_TOOLS)
        self.assertTrue(report.passed)
        self.assertEqual(report.checks["ledger_exact"], "pass")
        self.assertEqual(report.checks["trace_replayable"], "pass")
        self.assertEqual(report.checks["calibration_named"], "pass")
        self.assertEqual(report.checks["provenance_present"], "pass")

    def test_a_thin_shell_receipt_with_no_claims_has_no_failures(self):
        receipt = Receipt(answer="hi", produced_by="teacher")
        report = verify_receipt(receipt)
        self.assertTrue(report.passed)  # absent claims are honest, not failures
        self.assertEqual(set(report.checks.values()), {"absent"})

    def test_a_tampered_ledger_fails_verification(self):
        ledger = _build_ledger()
        tampered = copy.deepcopy(ledger)
        tampered.parts[0] = (tampered.parts[0][0], tampered.parts[0][1] + 5.0)  # break the additive identity
        receipt = Receipt(answer="a", ledger=tampered, provenance={"source_id": "x"})
        report = verify_receipt(receipt)
        self.assertFalse(report.passed)
        self.assertEqual(report.checks["ledger_exact"], "fail")

    def test_a_tampered_trace_fails_replay(self):
        trace = _build_trace()
        tampered = copy.deepcopy(trace)
        tampered.steps[0].result = [999.0, 999.0, 999.0]  # does not match what the seed actually produces
        receipt = Receipt(answer="x", trace=tampered, provenance={"source_id": "x"})
        report = verify_receipt(receipt, tools=_TOOLS)
        self.assertFalse(report.passed)
        self.assertEqual(report.checks["trace_replayable"], "fail")

    def test_trace_without_tools_is_reported_absent_not_a_false_pass(self):
        receipt = Receipt(answer="x", trace=_build_trace(), provenance={"source_id": "x"})
        report = verify_receipt(receipt)  # no tools supplied
        self.assertEqual(report.checks["trace_replayable"], "absent")
        self.assertTrue(report.passed)

    def test_calibration_missing_qhat_or_gate_fails_named_check(self):
        receipt = Receipt(answer="x", calibration={"alpha": 0.1}, provenance={"source_id": "x"})
        report = verify_receipt(receipt)
        self.assertEqual(report.checks["calibration_named"], "fail")
        self.assertFalse(report.passed)

    def test_receipt_round_trips_through_json(self):
        receipt = Receipt(
            answer="a",
            produced_by="student-v1",
            ledger=_build_ledger(),
            trace=_build_trace(),
            calibration={"alpha": 0.1, "qhat": 0.83},
            provenance={"source_id": "corpus-42"},
        )
        blob = receipt.to_json()
        self.assertEqual(blob["answer"], "a")
        self.assertEqual(blob["provenance"], {"source_id": "corpus-42"})
        self.assertIn("parts", blob["ledger"])
        self.assertIn("steps", blob["trace"])


class KnowledgeReceiptTransferTest(unittest.TestCase):
    """workstream H3: a Receipt -> a mixle-knowledge-shaped AnswerReceipt dict.

    mixle core has no dependency on the mixle-knowledge package (platform contracts depend on core,
    not the other way), so ``to_knowledge_dict`` produces a plain dict field-for-field matching
    ``mixle_knowledge.contracts.AnswerReceipt`` rather than importing it; the exact field set is
    pinned here as a regression guard against silent contract drift.
    """

    # mirrors mixle_knowledge.contracts.AnswerReceipt's fields exactly (created_at is defaulted
    # there, so it is intentionally absent from this dict -- everything else must round-trip).
    _KNOWLEDGE_ANSWER_RECEIPT_FIELDS = {
        "id",
        "project_id",
        "task",
        "answer",
        "produced_by",
        "ledger",
        "trace",
        "calibration",
        "provenance",
    }

    def test_dict_shape_matches_the_knowledge_contract_field_for_field(self):
        receipt = Receipt(
            answer="a",
            produced_by="student-v1",
            ledger=_build_ledger(),
            trace=_build_trace(),
            calibration={"alpha": 0.1, "qhat": 0.83},
            provenance={"source_id": "corpus-42"},
        )
        d = receipt.to_knowledge_dict(id="rcpt1", project_id="proj1", task="classify")
        self.assertEqual(set(d.keys()), self._KNOWLEDGE_ANSWER_RECEIPT_FIELDS)
        self.assertEqual(d["task"], "classify")
        self.assertEqual(d["answer"], "a")
        self.assertIn("parts", d["ledger"])
        self.assertIn("steps", d["trace"])

    def test_thin_shell_receipt_still_produces_a_valid_shape(self):
        receipt = Receipt(answer="hi", produced_by="teacher")
        d = receipt.to_knowledge_dict(id="rcpt2", project_id="proj1", task="chat")
        self.assertEqual(set(d.keys()), self._KNOWLEDGE_ANSWER_RECEIPT_FIELDS)
        self.assertIsNone(d["ledger"])
        self.assertIsNone(d["trace"])
        self.assertIsNone(d["calibration"])


if __name__ == "__main__":
    unittest.main()
