"""LLM surface (mixle.task.llm): label-constrained teacher + OpenAI-compatible client, no network in tests.

CallableLLM drives the labeler deterministically; OpenAICompatLLM is exercised with a monkeypatched HTTP post,
so the request shape and response parsing are verified without a server.
"""

import os
import tempfile
import unittest
from unittest import mock

from mixle.task import llm as L
from mixle.task.llm import (
    CallableLLM,
    LLMParseError,
    LLMResponseError,
    OpenAICompatLLM,
    extract_json_object,
    llm_labeler,
    pick_label,
    validated_endpoint_url,
)


class PickLabelTest(unittest.TestCase):
    def test_exact_then_unique_bounded_reference(self):
        labels = ["spam", "ham"]
        self.assertEqual(pick_label("spam", labels), "spam")
        self.assertEqual(pick_label("This is clearly SPAM.", labels), "spam")
        with self.assertRaises(LLMParseError):
            pick_label("no idea", labels)
        with self.assertRaises(LLMParseError):
            pick_label("spam or ham", labels)
        with self.assertRaises(ValueError):
            pick_label("anything", [])


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

    def test_json_decoder_handles_string_braces_and_recovers_after_malformed_candidate(self):
        parsed = extract_json_object('bad: {"x": } later: {"value": "a } brace", "ok": true}')
        self.assertEqual(parsed, {"value": "a } brace", "ok": True})

    def test_extracted_values_must_satisfy_the_verbatim_string_contract(self):
        from mixle.task.llm import llm_extractor

        numeric = llm_extractor(CallableLLM(lambda *_: '{"amount": 5}'), ["amount"])
        with self.assertRaises(LLMResponseError):
            numeric(["amount 5"])
        invented = llm_extractor(CallableLLM(lambda *_: '{"amount": "6"}'), ["amount"])
        with self.assertRaises(LLMResponseError):
            invented(["amount 5"])


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

    def test_response_schema_is_validated(self):
        orig = L._http_post_json
        L._http_post_json = lambda *_args: {"choices": []}
        try:
            with self.assertRaises(LLMResponseError):
                OpenAICompatLLM("http://localhost/v1", "model").complete("hi")
        finally:
            L._http_post_json = orig


class EndpointSchemeAllowlistTest(unittest.TestCase):
    """A non-HTTP endpoint is rejected before ``urlopen`` is ever reached.

    ``base_url`` reaches this module from a caller -- often by way of a config file, a CLI flag or an
    environment variable -- and ``urlopen`` honors ``file:``, ``ftp:`` and ``data:`` just as readily as
    ``http:``. Without a scheme check, ``file:///etc/passwd`` is a valid "LLM endpoint": urlopen reads
    the local file and the bytes come back to be parsed as the model's reply.
    """

    def test_http_and_https_pass_through_unchanged(self):
        for url in (
            "http://localhost:11434/v1/chat/completions",
            "https://api.example.com/v1/chat/completions",
            "HTTPS://api.example.com/v1/chat/completions",  # scheme comparison is case-insensitive
        ):
            self.assertEqual(validated_endpoint_url(url), url)

    def test_other_schemes_and_non_urls_are_rejected(self):
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/model",
            "data:application/json,%7B%7D",
            "gopher://example.com/1",
            "/v1/chat/completions",  # no scheme at all
            "",
        ):
            with self.assertRaises(ValueError):
                validated_endpoint_url(url)

    def test_post_rejects_a_local_file_endpoint_without_opening_it(self):
        """The rejection must happen before the request is opened, not after reading the response."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            # Shaped exactly like a valid chat-completions reply: were the scheme check missing, this
            # file's contents would be returned to the caller as though a model had produced them.
            fh.write('{"choices": [{"message": {"content": "attacker-controlled"}}]}')
            local_path = fh.name
        try:
            with mock.patch("urllib.request.urlopen") as urlopen:
                with self.assertRaises(ValueError):
                    L._http_post_json(f"file://{local_path}", {}, {"model": "m"}, 1.0)
            urlopen.assert_not_called()
        finally:
            os.unlink(local_path)

    def test_the_ordinary_http_path_still_reaches_urlopen(self):
        """Guard against the allowlist being tightened into rejecting everything."""
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = mock.Mock(status=200, read=lambda _n: b'{"choices": []}')
            self.assertEqual(L._http_post_json("http://localhost:11434/v1", {}, {"model": "m"}, 1.0), {"choices": []})
        urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
