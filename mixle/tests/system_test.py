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
        self.assertEqual(receipt["spend"], {"frontier_calls": 1})
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
