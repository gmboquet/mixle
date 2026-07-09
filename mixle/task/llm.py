"""Provider-agnostic LLM adapters for task teachers and local model calls.

mixle has no LLM client of its own; this is the minimal adapter. An :class:`LLM` is anything with ``complete(prompt)
-> str``. :class:`CallableLLM` wraps a local function (a llama.cpp/transformers call, or a test double);
:class:`OpenAICompatLLM` posts to any OpenAI-compatible ``/v1/chat/completions`` endpoint (Ollama, vLLM, TGI,
llama.cpp server, a hosted API) using only the standard library -- no ``openai``/``requests`` dependency.

:func:`llm_labeler` turns an LLM into the teacher the rest of ``mixle.task``
distills from. ``teacher = llm_labeler(OpenAICompatLLM(...), ["spam", "ham"])`` plugs a hosted or local model
into :func:`mixle.task.distill.distill` / :func:`mixle.task.active.active_distill`: the LLM labels selected
examples and the calibrated local student serves requests that it can answer. The same adapter lets an LLM
propose a model specification (:mod:`mixle.task.design`).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Anything that can turn a prompt into text. The whole contract the rest of the package depends on."""

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Return model text for a prompt and optional system instruction."""
        ...


class CallableLLM:
    """Wrap a plain ``fn(prompt) -> str`` (or ``fn(prompt, system)``) as an :class:`LLM` -- local models and tests."""

    def __init__(self, fn: Callable[..., str]) -> None:
        self.fn = fn

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Call the wrapped Python function and return its text output."""
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
        """Call an OpenAI-compatible chat-completions endpoint and return message text."""
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


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (tolerates code fences / surrounding prose)."""
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except ValueError:
                    return {}
    return {}


def llm_extractor(
    llm: LLM,
    fields: Sequence[str],
    *,
    instruction: str | None = None,
    system: str | None = None,
) -> Callable[[list[str]], list[dict[str, str]]]:
    """Turn an LLM into a field-extraction teacher ``texts -> [{field: value}]`` for :func:`mixle.task.extract.distill_extractor`.

    Each text is extracted into a JSON object over ``fields`` (values must be verbatim substrings so they align to
    token spans during distillation). The returned callable has the batched-teacher shape the extractor expects.
    """
    fields = list(fields)
    field_list = ", ".join(fields)
    instr = instruction or "Extract the fields from the text."
    sys = system or (
        f"You extract fields from text. Return a single JSON object with keys {field_list}. Each value must be an "
        "exact substring of the text (copy it verbatim). Omit a key if the field is absent. Output only JSON."
    )

    def teacher(texts: list[str]) -> list[dict[str, str]]:
        out = []
        for t in texts:
            reply = llm.complete(f"{instr}\n\nFields: {field_list}\n\nText: {t}\n\nJSON:", system=sys)
            parsed = _extract_json_object(reply)
            out.append({k: str(v) for k, v in parsed.items() if k in fields and v not in (None, "")})
        return out

    return teacher
