"""VLM surface (mixle.task.vlm): image-conditioned next-token scoring wired directly into
mixle.enumeration's descending-probability search. CallableVLM drives the enumeration tests
deterministically (no network); OpenAICompatVLM is exercised with a monkeypatched HTTP post, so the
request shape (image content, logprobs params, prefix continuation) and response parsing are
verified without a server.
"""

import unittest

from mixle.enumeration import best_first_decode, top_k_scored
from mixle.task import vlm as V
from mixle.task.vlm import (
    CallableVLM,
    OpenAICompatVLM,
    exact_token_scorer_for,
    score_candidate,
    score_fn_for,
)


def _toy_next_logprobs(image, prefix):
    """A tiny deterministic 'model': for a 'cat' image, 'a cat' scores higher than 'a dog'."""
    import math

    is_cat = image == "https://example.com/cat.png"
    if prefix == ():
        return [("a", math.log(0.9)), ("the", math.log(0.1))]
    if prefix == ("a",):
        if is_cat:
            return [("cat", math.log(0.8)), ("dog", math.log(0.2))]
        return [("dog", math.log(0.8)), ("cat", math.log(0.2))]
    if prefix in (("a", "cat"), ("a", "dog")):
        return [("<eos>", 0.0)]  # log(1.0): certain once the animal is named
    return []


class CallableVLMEnumerationTest(unittest.TestCase):
    def test_next_logprobs_for_binds_the_image_into_the_enumeration_shape(self):
        vlm = CallableVLM(_toy_next_logprobs)
        decode = vlm.next_logprobs_for("https://example.com/cat.png")
        results = list(best_first_decode(decode, eos="<eos>", max_len=5, max_results=2))
        self.assertEqual(len(results), 2)
        (seq0, lp0), (seq1, lp1) = results
        self.assertEqual(seq0, ("a", "cat", "<eos>"))  # the cat image's best-first completion is "a cat"
        self.assertGreaterEqual(lp0, lp1)  # best-first: descending order

    def test_a_different_image_flips_the_ranking(self):
        vlm = CallableVLM(_toy_next_logprobs)
        decode = vlm.next_logprobs_for("dog.png")
        (seq0, _lp0), _ = list(best_first_decode(decode, eos="<eos>", max_len=5, max_results=2))
        self.assertEqual(seq0, ("a", "dog", "<eos>"))

    def test_score_fn_for_ranks_a_fixed_candidate_set(self):
        vlm = CallableVLM(_toy_next_logprobs)
        decode = vlm.next_logprobs_for("https://example.com/cat.png")
        score = score_fn_for(decode, eos="<eos>")
        ranked = top_k_scored([("a", "cat"), ("a", "dog")], score)
        self.assertEqual(ranked[0][0], ("a", "cat"))
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_score_candidate_returns_neg_inf_for_a_token_the_model_never_considered(self):
        vlm = CallableVLM(_toy_next_logprobs)
        decode = vlm.next_logprobs_for("https://example.com/cat.png")
        self.assertEqual(score_candidate(decode, ("a", "bird"), eos="<eos>"), float("-inf"))

    def test_empty_candidate_is_scored_by_eos_probability(self):
        def root_with_eos(_image, prefix):
            if not prefix:
                return [("<eos>", -2.0), ("word", -0.2)]
            return [("<eos>", 0.0)]

        decode = CallableVLM(root_with_eos).next_logprobs_for("image")
        self.assertEqual(score_candidate(decode, (), eos="<eos>"), -2.0)

    def test_exact_interface_requires_token_ids_and_full_normalized_support(self):
        import math

        class ExactModel:
            vocab_size = 3
            eos_token_id = 2

            def next_token_logprobs(self, image, prefix_token_ids):
                if not prefix_token_ids:
                    return [math.log(0.6), math.log(0.3), math.log(0.1)]
                return [float("-inf"), float("-inf"), 0.0]

        decode, eos = exact_token_scorer_for(ExactModel(), "image")
        self.assertEqual(eos, 2)
        results = list(best_first_decode(decode, eos=eos, max_len=3, max_results=2))
        self.assertEqual(results[0][0], (0, 2))


class OpenAICompatVLMTest(unittest.TestCase):
    def test_first_token_request_shape(self):
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = payload
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "a",
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "a", "logprob": -0.1},
                                        {"token": "the", "logprob": -2.3},
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }

        orig = V._http_post_json
        V._http_post_json = fake_post
        try:
            client = OpenAICompatVLM("http://localhost:8000/v1", "llava-onevision", api_key="secret", top_logprobs=5)
            result = client.next_logprobs("https://example.com/cat.png", (), prompt="What is in this image?")
        finally:
            V._http_post_json = orig

        self.assertEqual(result, [("a", -0.1), ("the", -2.3)])
        self.assertEqual(captured["url"], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["payload"]["model"], "llava-onevision")
        self.assertEqual(captured["payload"]["max_tokens"], 1)
        self.assertIs(captured["payload"]["logprobs"], True)
        self.assertEqual(captured["payload"]["top_logprobs"], 5)
        self.assertNotIn("continue_final_message", captured["payload"])  # no prefix yet -> no continuation
        messages = captured["payload"]["messages"]
        self.assertEqual(len(messages), 1)  # just the user turn on the first token
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "What is in this image?"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(
            content[1]["image_url"]["url"], "https://example.com/cat.png"
        )  # already a bare "URL"-shaped string

    def test_continuation_request_sets_the_continue_flag_and_appends_the_prefix(self):
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured["payload"] = payload
            return {"choices": [{"logprobs": {"content": [{"token": "cat", "logprob": -0.05, "top_logprobs": []}]}}]}

        orig = V._http_post_json
        V._http_post_json = fake_post
        try:
            client = OpenAICompatVLM("http://localhost:8000/v1", "llava-onevision")
            client.next_logprobs("https://example.com/cat.png", ("a",), prompt="Describe it.")
        finally:
            V._http_post_json = orig

        self.assertIs(captured["payload"]["continue_final_message"], True)
        messages = captured["payload"]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1], {"role": "assistant", "content": "a"})

    def test_a_custom_continue_convention_is_honored(self):
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured["payload"] = payload
            return {"choices": [{"logprobs": {"content": [{"token": "x", "logprob": 0.0, "top_logprobs": []}]}}]}

        orig = V._http_post_json
        V._http_post_json = fake_post
        try:
            client = OpenAICompatVLM(
                "http://localhost:8000/v1", "m", continue_key="add_generation_prompt", continue_value=False
            )
            client.next_logprobs("https://example.com/cat.png", ("a",), prompt="p")
        finally:
            V._http_post_json = orig

        self.assertIs(captured["payload"]["add_generation_prompt"], False)
        self.assertNotIn("continue_final_message", captured["payload"])

    def test_next_logprobs_for_binds_image_and_prompt(self):
        def fake_post(url, headers, payload, timeout):
            return {"choices": [{"logprobs": {"content": [{"token": "a", "logprob": -0.1, "top_logprobs": []}]}}]}

        orig = V._http_post_json
        V._http_post_json = fake_post
        try:
            client = OpenAICompatVLM("http://localhost:8000/v1", "m")
            with self.assertRaisesRegex(ValueError, "truncated"):
                client.next_logprobs_for("https://example.com/cat.png", "What is this?")
            decode = client.next_logprobs_for(
                "https://example.com/cat.png",
                "What is this?",
                allow_truncated=True,
            )
            self.assertEqual(decode(()), [("a", -0.1)])
        finally:
            V._http_post_json = orig


class ImageContentCoercionTest(unittest.TestCase):
    def test_remote_url_passthrough(self):
        content = V._image_content("https://example.com/cat.png")
        self.assertEqual(content["image_url"]["url"], "https://example.com/cat.png")

    def test_raw_bytes_are_base64_encoded(self):
        content = V._image_content(b"\x89PNG\r\n\x1a\n")
        self.assertTrue(content["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_an_unrecognized_string_raises(self):
        with self.assertRaises(ValueError):
            V._image_content("not-a-path-or-url")


if __name__ == "__main__":
    unittest.main()
