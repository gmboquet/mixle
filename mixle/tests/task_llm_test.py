"""LLM surface (mixle.task.llm): label-constrained teacher + OpenAI-compatible client, no network in tests.

CallableLLM drives the labeler deterministically; OpenAICompatLLM is exercised with a monkeypatched HTTP post,
so the request shape and response parsing are verified without a server.
"""

import unittest

from mixle.task import llm as L
from mixle.task.llm import CallableLLM, OpenAICompatLLM, llm_labeler, pick_label


class PickLabelTest(unittest.TestCase):
    def test_exact_then_substring_then_fallback(self):
        labels = ["spam", "ham"]
        self.assertEqual(pick_label("spam", labels), "spam")
        self.assertEqual(pick_label("This is clearly SPAM.", labels), "spam")
        self.assertEqual(pick_label("no idea", labels), "spam")  # falls back to the first label


class LabelerTest(unittest.TestCase):
    def test_callable_llm_labeler(self):
        # a stub LLM that "reads" the prompt and answers spam when a spam word is present
        def fake(prompt, system=None):
            return "spam" if any(w in prompt.lower() for w in ("free", "prize", "winner")) else "ham"

        teacher = llm_labeler(CallableLLM(fake), ["spam", "ham"], instruction="Classify the email.")
        out = teacher(["free prize today", "team meeting at noon"])
        self.assertEqual(out, ["spam", "ham"])

    def test_labeler_plugs_into_distill(self):
        import numpy as np
        import pytest

        pytest.importorskip("torch")
        from mixle.task.distill import agreement, distill

        def fake(prompt, system=None):
            return "spam" if any(w in prompt.lower() for w in ("free", "prize", "winner", "buy")) else "ham"

        teacher = llm_labeler(CallableLLM(fake), ["spam", "ham"])
        rng = np.random.RandomState(0)
        spam, ham, filler = ["free", "prize", "winner", "buy"], ["meeting", "report", "team"], ["the", "a", "today"]
        texts = []
        for words in (spam, ham):
            for _ in range(60):
                toks = list(rng.choice(words, size=2)) + list(rng.choice(filler, size=3))
                rng.shuffle(toks)
                texts.append(" ".join(toks))
        student = distill(teacher, texts, n=4, dim=256, hidden=[32], epochs=150, seed=0)
        self.assertGreaterEqual(agreement(student, teacher(texts), texts), 0.85)


class ExtractorTest(unittest.TestCase):
    def test_llm_extractor_parses_json_fields(self):
        from mixle.task.llm import llm_extractor

        def fake(prompt, system=None):
            # a stub extraction LLM that returns JSON (with surrounding prose to test tolerant parsing)
            return 'Sure! Here you go:\n```json\n{"id": "1234", "vendor": "Acme", "missing": "x"}\n```'

        teacher = llm_extractor(CallableLLM(fake), ["id", "vendor", "amount"])
        out = teacher(["INV-1234 Acme $5.00"])
        self.assertEqual(out, [{"id": "1234", "vendor": "Acme"}])  # off-schema 'missing' dropped, absent omitted


class CallableLLMTest(unittest.TestCase):
    def test_single_arg_fn_is_called_once(self):
        calls = []

        def fn(prompt):
            calls.append(prompt)
            return "reply"

        self.assertEqual(CallableLLM(fn).complete("hi", system="be terse"), "reply")
        self.assertEqual(calls, ["hi"])

    def test_two_arg_fn_is_called_once(self):
        calls = []

        def fn(prompt, system):
            calls.append((prompt, system))
            return "reply"

        self.assertEqual(CallableLLM(fn).complete("hi", system="be terse"), "reply")
        self.assertEqual(calls, [("hi", "be terse")])

    def test_unrelated_type_error_inside_two_arg_fn_is_not_swallowed_by_a_retry(self):
        # A TypeError raised *inside* fn(prompt, system) for a reason unrelated to arity used to be
        # misread as "fn only takes one argument" and silently retried as fn(prompt) -- invoking fn
        # a second time and masking the real error.
        calls = []

        def fn(prompt, system):
            calls.append(prompt)
            raise TypeError("boom: unrelated bug inside fn")

        with self.assertRaises(TypeError):
            CallableLLM(fn).complete("hi", system="be terse")
        self.assertEqual(calls, ["hi"])  # called exactly once, not retried


class OpenAICompatTest(unittest.TestCase):
    def test_request_shape_and_parse(self):
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "  ham  "}}]}

        orig = L._http_post_json
        L._http_post_json = fake_post
        try:
            client = OpenAICompatLLM("http://localhost:11434/v1", "qwen2.5", api_key="secret")
            reply = client.complete("hi", system="be terse")
        finally:
            L._http_post_json = orig

        self.assertEqual(reply, "  ham  ")
        self.assertEqual(captured["url"], "http://localhost:11434/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["payload"]["model"], "qwen2.5")
        self.assertEqual(captured["payload"]["messages"][0], {"role": "system", "content": "be terse"})
        self.assertEqual(captured["payload"]["messages"][1], {"role": "user", "content": "hi"})


if __name__ == "__main__":
    unittest.main()
