"""System facade -- the thin shell three verbs (answer/ingest/improve) sit behind (workstream J1/J8)."""

import unittest

from mixle.substrate.core import Substrate
from mixle.system import Query, System, SystemConfig


def _fake_teacher(prompt: str) -> str:
    return f"answer to: {prompt}"


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


class SystemIngestTest(unittest.TestCase):
    def test_ingest_with_no_store_is_an_honest_noop(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        report = system.ingest("the sky is blue", source={"model": "teacher-v1"})
        self.assertEqual(report["status"], "no_store")
        self.assertFalse(report["assimilated"])

    def test_ingest_falls_back_to_a_retrievable_substrate_item_when_no_belief_store_exists(self):
        store = Substrate()
        system = System(SystemConfig(teacher=_fake_teacher, store=store))
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


class SystemConfigFromEnvTest(unittest.TestCase):
    def test_from_env_requires_base_url_and_model(self):
        with self.assertRaises(ValueError) as ctx:
            SystemConfig.from_env()
        self.assertIn("MIXLE_TEACHER_BASE_URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
