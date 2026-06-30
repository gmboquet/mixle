"""A tiny provider-agnostic LLM surface -- so a regular program can reach a small LM, and an LM can be a teacher.

mixle has no LLM client of its own; this is the minimal seam. An :class:`LLM` is anything with ``complete(prompt)
-> str``. :class:`CallableLLM` wraps a local function (a llama.cpp/transformers call, or a stub in tests);
:class:`OpenAICompatLLM` posts to any OpenAI-compatible ``/v1/chat/completions`` endpoint (Ollama, vLLM, TGI,
llama.cpp server, a hosted API) using only the standard library -- no ``openai``/``requests`` dependency.

The point of having it here: :func:`llm_labeler` turns an LLM into the *teacher* the rest of ``mixle.task``
distills from. ``teacher = llm_labeler(OpenAICompatLLM(...), ["spam", "ham"])`` plugs a frontier model straight
into :func:`mixle.task.distill.distill` / :func:`mixle.task.active.active_distill` -- the expensive LM labels a
little, the tiny local student serves the rest. The same seam lets an LLM *design* a model (:mod:`mixle.task.design`).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Anything that can turn a prompt into text. The whole contract the rest of the package depends on."""

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str: ...


class CallableLLM:
    """Wrap a plain ``fn(prompt) -> str`` (or ``fn(prompt, system)``) as an :class:`LLM` -- local models and tests."""

    def __init__(self, fn: Callable[..., str]) -> None:
        self.fn = fn

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        try:
            return self.fn(prompt, system)  # type: ignore[call-arg]
        except TypeError:
            return self.fn(prompt)


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON and parse JSON back, with only the standard library (monkeypatched in tests)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - caller-provided trusted endpoint
        return json.loads(resp.read().decode())


class OpenAICompatLLM:
    """An :class:`LLM` backed by any OpenAI-compatible ``/v1/chat/completions`` endpoint (stdlib HTTP only)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out = _http_post_json(f"{self.base_url}/chat/completions", headers, payload, self.timeout)
        return out["choices"][0]["message"]["content"]


def pick_label(text: str, labels: Sequence[str]) -> str:
    """Map a free-text LLM reply to one of ``labels`` (exact, then substring, else the first label)."""
    low = text.strip().lower()
    for label in labels:
        if low == str(label).lower():
            return label
    for label in labels:  # the model often answers in a sentence: "This looks like spam."
        if str(label).lower() in low:
            return label
    return labels[0]


def llm_labeler(
    llm: LLM,
    labels: Sequence[str],
    *,
    instruction: str | None = None,
    system: str | None = None,
) -> Callable[[list[str]], list[str]]:
    """Turn an LLM into a label-constrained teacher ``texts -> [label]`` for distillation / active labeling.

    Each item is classified into ``labels`` by a constrained prompt; the reply is mapped back with
    :func:`pick_label`. The returned callable has the batched-teacher shape the rest of ``mixle.task`` expects.
    """
    labels = list(labels)
    options = ", ".join(str(label) for label in labels)
    instr = instruction or "Classify the following text."
    sys = system or f"You are a precise classifier. Answer with exactly one of: {options}. Output only the label."

    def teacher(texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            reply = llm.complete(f"{instr}\n\nText: {t}\n\nLabel ({options}):", system=sys)
            out.append(pick_label(reply, labels))
        return out

    return teacher
