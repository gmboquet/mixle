"""A tiny provider-agnostic VLM surface, wired directly into mixle.enumeration's descending-probability search.

Sibling of :mod:`mixle.task.llm`, extended with an image. A :class:`VLM` is anything with
``next_logprobs(image, prefix) -> [(token, log_prob), ...]`` -- the SAME shape
:func:`mixle.enumeration.best_first_decode` / :func:`mixle.enumeration.quantized_best_first_decode` already
expect from any autoregressive scorer, so binding an image into that shape (:meth:`OpenAICompatVLM.next_logprobs_for`)
is all "VLM enumeration support" needs to be: nothing about ``best_first_decode`` itself is vision-specific.

Scope, deliberately: this targets an **open-weight** vision-language model served behind an OpenAI-compatible
``/v1/chat/completions`` endpoint that returns real per-token ``logprobs`` (vLLM, TGI, and similar self-hosted
stacks serving e.g. LLaVA / Qwen-VL / ...). Proprietary hosted vision APIs (GPT-4V, Claude vision, Gemini vision)
generally do not expose true per-token logprobs for image-conditioned generation, so mixle.enumeration's
descending-probability *guarantee* only holds against a genuine logprob-serving endpoint -- :class:`OpenAICompatVLM`
does not attempt to approximate that guarantee against a black-box API that cannot honor it.

Two things this file gives you, both built on the one real network primitive (:meth:`OpenAICompatVLM.next_logprobs`):

- **Free-form top-k decoding**, exact and lazy, via the existing engine::

      vlm = OpenAICompatVLM("http://localhost:8000/v1", "llava-onevision")
      decode = vlm.next_logprobs_for(image, prompt="Describe this image in one sentence.")
      for tokens, log_prob in best_first_decode(decode, eos="<|eot_id|>", max_len=40, max_results=5):
          print("".join(tokens), log_prob)   # the 5 highest-probability captions, best first

- **Ranking a fixed candidate set** by the model's own teacher-forced probability, via
  :func:`mixle.enumeration.top_k_scored`::

      score = score_fn_for(decode)
      top_k_scored([("cat",), ("dog",), ("bird",)], score, k=3)

Teacher-forced candidate scoring costs one ``next_logprobs`` call per token (no batched echo/teacher-forcing
primitive is assumed to exist on the server) -- this is stated up front, not hidden behind a fast-looking API.
Candidates must be pre-tokenized (``Sequence[str]`` of the SAME token pieces the server's own tokenizer would
produce): a naive whitespace/character split would silently misalign with a real BPE tokenizer and score the
wrong thing, so no such convenience split is provided here.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mixle.task.llm import _http_post_json


@runtime_checkable
class VLM(Protocol):
    """Anything that can score an image-conditioned next-token continuation."""

    def next_logprobs(self, image: Any, prefix: tuple[str, ...]) -> Iterable[tuple[str, float]]: ...


class CallableVLM:
    """Wrap a plain ``fn(image, prefix) -> [(token, log_prob), ...]`` as a :class:`VLM` -- local models and tests."""

    def __init__(self, fn: Callable[[Any, tuple[str, ...]], Iterable[tuple[str, float]]]) -> None:
        self.fn = fn

    def next_logprobs(self, image: Any, prefix: tuple[str, ...]) -> Iterable[tuple[str, float]]:
        return self.fn(image, prefix)

    def next_logprobs_for(self, image: Any) -> Callable[[tuple[str, ...]], Iterable[tuple[str, float]]]:
        """Bind ``image`` into the ``next_logprobs(prefix)`` shape ``mixle.enumeration`` expects directly."""
        return lambda prefix: self.next_logprobs(image, prefix)


def _image_content(image: Any) -> dict[str, Any]:
    """Coerce ``image`` (raw bytes, a local file path, or an already-built data/remote URL string) into an
    OpenAI chat ``image_url`` content part."""
    if isinstance(image, (bytes, bytearray)):
        b64 = base64.b64encode(bytes(image)).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    text = str(image)
    if text.startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": text}}
    path = Path(text)
    if path.is_file():
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    raise ValueError(f"cannot interpret {image!r} as an image (expected raw bytes, a file path, or a URL/data URI)")


class OpenAICompatVLM:
    """A :class:`VLM` backed by an OpenAI-compatible ``/v1/chat/completions`` endpoint that returns real
    per-token ``logprobs`` for an open-weight vision-language model (a vLLM- or TGI-served LLaVA / Qwen-VL /
    ... deployment). See the module docstring for why this deliberately does not target proprietary hosted
    vision APIs.

    Continuing a partial completion (every ``next_logprobs`` call after the first token of a decode) needs
    the server to *prefill* the given prefix rather than start generation fresh; this uses vLLM's
    ``continue_final_message`` extension by default (append the prefix as a partial assistant message, set
    that flag). Pass ``continue_key``/``continue_value`` to target a server with a different convention.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        top_logprobs: int = 20,
        timeout: float = 60.0,
        continue_key: str = "continue_final_message",
        continue_value: Any = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.top_logprobs = int(top_logprobs)
        self.timeout = timeout
        self.continue_key = continue_key
        self.continue_value = continue_value
        self.extra_body = dict(extra_body) if extra_body else {}

    def next_logprobs(
        self, image: Any, prefix: tuple[str, ...], *, prompt: str, system: str | None = None
    ) -> list[tuple[str, float]]:
        """One image-conditioned next-token distribution given the tokens generated so far (``prefix``)."""
        messages: list[dict[str, Any]] = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": [{"type": "text", "text": prompt}, _image_content(image)]}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            **self.extra_body,
        }
        if prefix:
            messages.append({"role": "assistant", "content": "".join(prefix)})
            payload[self.continue_key] = self.continue_value
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out = _http_post_json(f"{self.base_url}/chat/completions", headers, payload, self.timeout)
        content = out["choices"][0]["logprobs"]["content"]
        if not content:
            return []
        step = content[0]
        top = step.get("top_logprobs") or [{"token": step["token"], "logprob": step["logprob"]}]
        return [(t["token"], float(t["logprob"])) for t in top]

    def next_logprobs_for(
        self, image: Any, prompt: str, *, system: str | None = None
    ) -> Callable[[tuple[str, ...]], Iterable[tuple[str, float]]]:
        """Bind ``image``/``prompt`` into the ``next_logprobs(prefix) -> [(token, log_prob), ...]`` shape
        :func:`mixle.enumeration.best_first_decode` / :func:`mixle.enumeration.quantized_best_first_decode`
        expect directly -- the whole bridge from "an image and a question" to "enumerate the top-k answers"."""
        return lambda prefix: self.next_logprobs(image, prefix, prompt=prompt, system=system)


def score_candidate(
    next_logprobs_fn: Callable[[tuple[str, ...]], Iterable[tuple[str, float]]], candidate_tokens: Sequence[str]
) -> float:
    """Teacher-forced total log-probability of ``candidate_tokens`` under ``next_logprobs_fn``.

    Walks one token at a time, reading off the ACTUAL log-probability of the candidate's own next token at
    each step -- never approximated or guessed. If a step's returned continuations do not include the
    candidate's token (e.g. it fell outside ``top_logprobs``), returns ``-inf`` rather than silently
    dropping or padding the score with a made-up value: that is a real "this candidate wasn't even
    considered by the model at that step" outcome, not a bug to hide.
    """
    prefix: tuple[str, ...] = ()
    total = 0.0
    for token in candidate_tokens:
        step = dict(next_logprobs_fn(prefix))
        if token not in step:
            return float("-inf")
        total += step[token]
        prefix = (*prefix, token)
    return total


def score_fn_for(
    next_logprobs_fn: Callable[[tuple[str, ...]], Iterable[tuple[str, float]]],
) -> Callable[[Sequence[str]], float]:
    """Bind a ``next_logprobs`` function into the ``score(candidate) -> float`` shape
    :func:`mixle.enumeration.top_k_scored` expects directly, for ranking a fixed candidate set."""
    return lambda candidate_tokens: score_candidate(next_logprobs_fn, candidate_tokens)


__all__ = [
    "VLM",
    "CallableVLM",
    "OpenAICompatVLM",
    "score_candidate",
    "score_fn_for",
]
