"""System facade -- the thin shell three verbs (answer/ingest/improve) sit behind (workstream J1/J8)."""

import builtins
import unittest
from unittest.mock import patch

from mixle.substrate.context import ContextBudget, assemble_context
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.ingest import ingest_documents
from mixle.system import Query, System, SystemConfig


def _fake_teacher(prompt: str) -> str:
    return f"answer to: {prompt}"


class QueryKnowledgeAlignmentTest(unittest.TestCase):
    """Query.from_knowledge_dict is the OTHER half of the mixle-knowledge alignment claim in Query's
    own docstring: build a Query directly from a real assembled ContextPacket (workstream E1's own
    to_knowledge_dict output), not just a claim that the field names happen to match."""

    def test_query_built_from_a_real_assembled_context_packet(self):
        substrate = Substrate()
        ingest_documents(substrate, ["cats are mammals that purr"], source="animal facts")
        pkt = assemble_context(substrate, "mammals", budget=ContextBudget(max_chars=200))
        d = pkt.to_knowledge_dict(id="pkt1", project_id="proj1", target_kind="frontier_llm")

        query = Query.from_knowledge_dict(d, scope="project")
        self.assertEqual(query.task, "mammals")
        self.assertEqual(query.text, pkt.render())
        self.assertEqual(query.scope, "project")

    def test_expected_output_schema_maps_to_expected_output(self):
        d = {"task": "extract", "payload": {"rendered": "hi"}, "expected_output_schema": {"type": "object"}}
        query = Query.from_knowledge_dict(d)
        self.assertEqual(query.expected_output, {"type": "object"})

    def test_missing_expected_output_schema_stays_none(self):
        d = {"task": "chat", "payload": {"rendered": "hi"}}
        query = Query.from_knowledge_dict(d)
        self.assertIsNone(query.expected_output)


class SystemAnswerTest(unittest.TestCase):
    def test_answer_routes_to_teacher_and_returns_a_receipt(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        reply, receipt = system.answer(Query("what is 2+2?", task="qa"))

        self.assertEqual(reply, "answer to: what is 2+2?")
        self.assertEqual(receipt["produced_by"], "teacher")
        # SPEND-a: spend is now the full Spend ledger shape (frontier_calls/oracle_calls/wall_ms/dollars),
        # not the earlier ad hoc {"frontier_calls": 1} dict.
        self.assertEqual(receipt["spend"], {"frontier_calls": 1, "oracle_calls": 0, "wall_ms": 0.0, "dollars": 0.0})
        self.assertEqual(receipt["task"], "qa")
        self.assertFalse(receipt["captured"])

    def test_answer_accepts_a_class_based_llm(self):
        class _LLM:
            def complete(self, prompt, *, system=None, **kwargs):
                return f"llm:{prompt}"

        system = System(SystemConfig(teacher=_LLM()))
        reply, _ = system.answer(Query("hi"))
        self.assertEqual(reply, "llm:hi")

    def test_answer_respects_an_explicit_budget(self):
        system = System(SystemConfig(teacher=_fake_teacher, default_budget=1))
        _, receipt = system.answer(Query("x"), budget=5)
        self.assertEqual(receipt["budget"], 5)


class SystemSpendLedgerTest(unittest.TestCase):
    """CARD SPEND-a: budget is a hard ceiling; every call's cost accumulates into System.total_spend."""

    def test_total_spend_accumulates_across_calls(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        for _ in range(3):
            system.answer(Query("x"))
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 3)

    def test_over_budget_request_is_refused_not_silently_served(self):
        calls = {"n": 0}

        def counting_teacher(prompt):
            calls["n"] += 1
            return f"answer to: {prompt}"

        system = System(SystemConfig(teacher=counting_teacher))
        reply, receipt = system.answer(Query("x"), budget=0)

        self.assertIsNone(reply)
        self.assertEqual(calls["n"], 0)  # the teacher was never called -- no silent overspend
        self.assertEqual(receipt["status"], "refused")
        self.assertEqual(receipt["shortfall"], 1.0)
        self.assertEqual(receipt["spend"], {"frontier_calls": 0, "oracle_calls": 0, "wall_ms": 0.0, "dollars": 0.0})
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 0)

    def test_a_refusal_does_not_perturb_a_later_successful_calls_running_total(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        system.answer(Query("a"))
        system.answer(Query("b"), budget=0)  # refused; must not silently count against total_spend
        system.answer(Query("c"))
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 2)

    def test_receipt_carries_both_incremental_and_running_spend(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        system.answer(Query("a"))
        _, receipt = system.answer(Query("b"))
        self.assertEqual(receipt["spend"]["frontier_calls"], 1)
        self.assertEqual(receipt["total_spend"]["frontier_calls"], 2)


class SystemFaultModesTest(unittest.TestCase):
    """CARD FAULT-a: teacher_down / store_down degrade to a named, flagged mode -- never silently."""

    def _broken_teacher(self, prompt: str) -> str:
        raise ConnectionError("teacher endpoint unreachable")

    def test_teacher_down_falls_back_to_store_only_and_flags_it(self):
        store = Substrate()
        store.put(SubstrateItem(kind="text", text="the rollout finished on schedule"))
        system = System(SystemConfig(teacher=self._broken_teacher, store=store))

        reply, receipt = system.answer(Query("rollout status"))
        self.assertIn("degraded: store-only", reply)
        self.assertIn("rollout finished on schedule", reply)
        self.assertEqual(receipt["status"], "answered")
        self.assertEqual(receipt["degraded_mode"], "teacher_down")
        self.assertIn("teacher endpoint unreachable", receipt["degraded_reason"])
        self.assertEqual(receipt["produced_by"], "store")
        # a degraded, store-only answer didn't actually spend a frontier call
        self.assertEqual(receipt["spend"]["frontier_calls"], 0)

    def test_teacher_down_with_no_usable_store_fails_honestly(self):
        system = System(SystemConfig(teacher=self._broken_teacher))
        reply, receipt = system.answer(Query("rollout status"))
        self.assertIsNone(reply)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("teacher unavailable", receipt["reason"])

    def test_store_down_falls_back_to_no_accumulation_and_flags_it(self):
        class _BrokenStore:
            """Every method a store might be asked for raises the SAME unreachable error -- ingest's
            KNOW-a path calls .all() (via assimilate's lookup) before it ever reaches .put()."""

            def put(self, item):
                raise OSError("store unreachable")

            def all(self, *args, **kwargs):
                raise OSError("store unreachable")

            def get(self, *args, **kwargs):
                raise OSError("store unreachable")

        system = System(SystemConfig(teacher=_fake_teacher, store=_BrokenStore()))
        report = system.ingest("the sky is blue", source={"model": "teacher-v1"})

        self.assertEqual(report["status"], "degraded_no_accumulation")
        self.assertFalse(report["assimilated"])
        self.assertEqual(report["degraded_mode"], "store_down")
        self.assertIn("store unreachable", report["degraded_reason"])


class SystemIngestTest(unittest.TestCase):
    def test_ingest_with_no_store_is_an_honest_noop(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        report = system.ingest("the sky is blue", source={"model": "teacher-v1"})
        self.assertEqual(report["status"], "no_store")
        self.assertFalse(report["assimilated"])

    def test_ingest_assimilates_via_the_belief_store_when_it_is_importable(self):
        """The primary path now that KNOW-a is built: a real claim, assimilated with real credence."""
        store = Substrate()
        system = System(SystemConfig(teacher=_fake_teacher, store=store))
        report = system.ingest("the sky is blue", source={"model": "teacher-v1"})

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["n_claims"], 1)
        self.assertEqual(len(report["items"]), 1)

    def test_ingest_falls_back_to_a_retrievable_substrate_item_when_no_belief_store_exists(self):
        """The defensive path SYS-a documents ('never a hard import of a card that may not be built
        yet'): still real and worth covering even though mixle.substrate.belief is always importable
        in THIS repo now -- simulate the import genuinely failing, the way it would for a caller
        missing that optional piece."""
        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "mixle.substrate.belief" or name.startswith("mixle.substrate.belief."):
                raise ImportError(f"simulated: {name} not installed")
            return real_import(name, *args, **kwargs)

        store = Substrate()
        system = System(SystemConfig(teacher=_fake_teacher, store=store))
        with patch("builtins.__import__", side_effect=_blocked_import):
            report = system.ingest("the sky is blue", source={"model": "teacher-v1"})

        self.assertEqual(report["status"], "ok_fallback")
        item = store.get(report["item_id"])
        self.assertIsNotNone(item)
        self.assertEqual(item.text, "the sky is blue")
        self.assertEqual(item.provenance, {"model": "teacher-v1"})


class SystemImproveTest(unittest.TestCase):
    def test_improve_on_an_empty_system_is_honest(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        report = system.improve(10)
        self.assertEqual(report["status"], "nothing_to_improve")
        self.assertEqual(report["budget"], 10)


class SystemColdStartCaptureTest(unittest.TestCase):
    """CARD SEED-a: from an empty system, answer then improve, and the second identical answer is free."""

    def _counting_teacher(self):
        calls = {"n": 0}

        def teacher(prompt):
            calls["n"] += 1
            return f"answer to: {prompt}"

        return teacher, calls

    def test_second_identical_query_after_improve_costs_no_frontier_calls(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        query = Query("what is the capital of Freedonia?")

        reply1, receipt1 = system.answer(query)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(receipt1["captured"])

        report = system.improve(10)
        self.assertEqual(report["status"], "captured")
        self.assertEqual(report["n_captured"], 1)

        reply2, receipt2 = system.answer(query)
        self.assertEqual(calls["n"], 1)  # no new frontier call -- served from the captured cache
        self.assertEqual(reply2, reply1)
        self.assertTrue(receipt2["captured"])
        self.assertEqual(receipt2["produced_by"], "captured")
        self.assertEqual(receipt2["spend"], {"frontier_calls": 0, "oracle_calls": 0, "wall_ms": 0.0, "dollars": 0.0})

    def test_repeat_query_before_improve_still_pays_for_a_fresh_teacher_call(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        query = Query("what is the capital of Freedonia?")
        system.answer(query)
        system.answer(query)
        self.assertEqual(calls["n"], 2)

    def test_capture_is_specific_to_the_captured_query_text(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        system.answer(Query("query one"))
        system.improve(10)
        system.answer(Query("query two"))  # a different query -- still a real teacher call
        self.assertEqual(calls["n"], 2)

    def test_improve_with_nothing_harvested_yet_is_still_honest(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        report = system.improve(10)
        self.assertEqual(report["status"], "nothing_to_improve")

    def test_same_text_different_task_does_not_share_a_captured_cache_entry(self):
        # regression: two queries with identical text but a different task (or scope) are different
        # questions and must not silently answer one from the other's captured cache
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        system.answer(Query("classify this", task="sentiment"))
        system.improve(10)
        system.answer(Query("classify this", task="topic"))
        self.assertEqual(calls["n"], 2)

    def test_same_text_different_scope_does_not_share_a_captured_cache_entry(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        system.answer(Query("classify this", scope="team-a"))
        system.improve(10)
        system.answer(Query("classify this", scope="team-b"))
        self.assertEqual(calls["n"], 2)


class SystemConfigFromEnvTest(unittest.TestCase):
    def test_from_env_requires_base_url_and_model(self):
        with self.assertRaises(ValueError) as ctx:
            SystemConfig.from_env()
        self.assertIn("MIXLE_TEACHER_BASE_URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
