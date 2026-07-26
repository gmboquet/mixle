"""Provider-agnostic image-conditioned scoring with explicit exactness boundaries.

Sibling of :mod:`mixle.task.llm`, extended with an image. A :class:`VLM` is anything with
``next_logprobs(image, prefix) -> [(token, log_prob), ...]``. A generic
string-token callback is useful for diagnostics and fixed candidate ranking,
but it is not automatically an exact enumeration oracle.

Scope, deliberately: this targets an **open-weight** vision-language model served behind an OpenAI-compatible
``/v1/chat/completions`` endpoint that returns real per-token ``logprobs`` (vLLM, TGI, and similar self-hosted
stacks serving e.g. LLaVA / Qwen-VL / ...). These endpoints generally expose
only a truncated top-logprob list and continue from concatenated token text.
That cannot certify complete probability order because omitted tokens and
server re-tokenization are unbounded. :class:`OpenAICompatVLM` therefore
requires explicit ``allow_truncated=True`` before returning such a callback.

Exact best-first enumeration is available through
:func:`exact_token_scorer_for`, which requires tokenizer IDs and a complete
log-probability vector for every vocabulary item at every prefix.

- **Diagnostic truncated decoding** from an OpenAI-compatible endpoint::

      vlm = OpenAICompatVLM("http://localhost:8000/v1", "llava-onevision")
      decode = vlm.next_logprobs_for(
          image,
          prompt="Describe this image in one sentence.",
          allow_truncated=True,
      )
      for tokens, log_prob in best_first_decode(decode, eos="<|eot_id|>", max_len=40, max_results=5):
          print("".join(tokens), log_prob)   # the 5 highest-probability captions, best first

- **Ranking a fixed candidate set** by the model's own teacher-forced probability, via
  :func:`mixle.enumeration.top_k_scored`::

      score = score_fn_for(decode, eos="<|eot_id|>")
      top_k_scored([("cat",), ("dog",), ("bird",)], score, k=3)

Teacher-forced candidate scoring includes the termination token and costs one
call per candidate token plus one for EOS. Against the OpenAI-compatible
bridge it remains a truncated diagnostic, not an exact probability, because
the endpoint does not bind caller-visible strings to stable tokenizer IDs.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.task.llm import _http_post_json


@runtime_checkable
class VLM(Protocol):
    """Anything that can score an image-conditioned next-token continuation."""

    def next_logprobs(self, image: Any, prefix: tuple[str, ...]) -> Iterable[tuple[str, float]]: ...


@runtime_checkable
class TokenizerBoundVLM(Protocol):
    """Exact VLM contract: stable token IDs plus complete vocabulary support."""

    vocab_size: int
    eos_token_id: int

    def next_token_logprobs(
        self,
        image: Any,
        prefix_token_ids: tuple[int, ...],
    ) -> Sequence[float]: ...


class CallableVLM:
    """Wrap a plain ``fn(image, prefix) -> [(token, log_prob), ...]`` as a :class:`VLM` -- local models and tests."""

    def __init__(
        self,
        fn: Callable[[Any, tuple[str, ...]], Iterable[tuple[str, float]]],
    ) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn

    def next_logprobs(self, image: Any, prefix: tuple[str, ...]) -> Iterable[tuple[str, float]]:
        return self.fn(image, prefix)

    def next_logprobs_for(self, image: Any) -> Callable[[tuple[str, ...]], Iterable[tuple[str, float]]]:
        """Bind an image without claiming complete support or stable tokenization."""
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
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, Integral) or top_logprobs < 1:
            raise ValueError("top_logprobs must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if not isinstance(continue_key, str) or not continue_key:
            raise ValueError("continue_key must be a non-empty string")
        if extra_body is not None and not isinstance(extra_body, dict):
            raise TypeError("extra_body must be a dictionary or None")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.top_logprobs = int(top_logprobs)
        self.timeout = float(timeout)
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
        result = [(t["token"], float(t["logprob"])) for t in top]
        _step_map(result)
        return result

    def next_logprobs_for(
        self,
        image: Any,
        prompt: str,
        *,
        system: str | None = None,
        allow_truncated: bool = False,
    ) -> Callable[[tuple[str, ...]], Iterable[tuple[str, float]]]:
        """Bind a truncated string-token diagnostic callback.

        ``allow_truncated=True`` is an explicit acknowledgement that this
        callback cannot certify exact enumeration or exact candidate scores.
        """
        if not isinstance(allow_truncated, bool):
            raise ValueError("allow_truncated must be a boolean")
        if not allow_truncated:
            raise ValueError(
                "OpenAI-compatible top_logprobs are truncated and re-tokenize prefixes; "
                "pass allow_truncated=True for diagnostic use"
            )
        return lambda prefix: self.next_logprobs(image, prefix, prompt=prompt, system=system)


def _step_map(entries: Iterable[tuple[Any, float]]) -> dict[Any, float]:
    result: dict[Any, float] = {}
    for token, raw_log_probability in entries:
        if token in result:
            raise ValueError(f"next-token distribution contains duplicate token {token!r}")
        if (
            isinstance(raw_log_probability, bool)
            or not isinstance(raw_log_probability, Real)
            or np.isnan(raw_log_probability)
            or raw_log_probability > 0.0
        ):
            raise ValueError("next-token log probabilities must be numbers in [-inf, 0]")
        result[token] = float(raw_log_probability)
    return result


def exact_token_scorer_for(
    model: TokenizerBoundVLM,
    image: Any,
) -> tuple[Callable[[tuple[int, ...]], list[tuple[int, float]]], int]:
    """Bind a tokenizer-ID, full-vocabulary VLM for exact enumeration.

    Every callback result is checked for complete support, normalized
    probability mass, valid token IDs, and stable vocabulary size. The
    returned EOS token ID must be passed to the enumeration engine.
    """
    if not isinstance(model, TokenizerBoundVLM):
        raise TypeError("model must implement TokenizerBoundVLM")
    vocab_size = model.vocab_size
    eos = model.eos_token_id
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral) or vocab_size < 2:
        raise ValueError("vocab_size must be an integer >= 2")
    if isinstance(eos, bool) or not isinstance(eos, Integral) or not 0 <= eos < vocab_size:
        raise ValueError("eos_token_id must index the declared vocabulary")
    vocab_size = int(vocab_size)
    eos = int(eos)

    def score(prefix: tuple[int, ...]) -> list[tuple[int, float]]:
        if any(
            isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= int(token) < vocab_size
            for token in prefix
        ):
            raise ValueError("prefix contains a token outside the declared vocabulary")
        values = np.asarray(
            model.next_token_logprobs(image, tuple(int(token) for token in prefix)),
            dtype=np.float64,
        )
        if values.shape != (vocab_size,):
            raise ValueError(f"full-support scorer must return {vocab_size} log probabilities, got {values.shape}")
        if np.any(np.isnan(values)) or np.any(values > 0.0):
            raise ValueError("full-support log probabilities must lie in [-inf, 0]")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("full-support distribution cannot assign zero mass to every token")
        maximum = float(np.max(finite))
        log_total = maximum + float(np.log(np.exp(values - maximum).sum()))
        if not np.isclose(log_total, 0.0, atol=1.0e-7, rtol=1.0e-7):
            raise ValueError("full-support log probabilities must normalize to one")
        return [(token, float(values[token])) for token in range(vocab_size)]

    return score, eos


def score_candidate(
    next_logprobs_fn: Callable[[tuple[Any, ...]], Iterable[tuple[Any, float]]],
    candidate_tokens: Sequence[Any],
    *,
    eos: Any,
) -> float:
    """Score a fixed candidate plus its required termination token.

    Walks one token at a time, reading off the ACTUAL log-probability of the candidate's own next token at
    each step, then adds ``P(EOS | candidate)``. If a step's returned continuations do not include the
    candidate's token (e.g. it fell outside ``top_logprobs``), returns ``-inf`` rather than silently
    dropping or padding the score. The empty candidate is therefore scored by
    its EOS probability, never assigned log probability zero by construction.
    """
    tokens = tuple(candidate_tokens)
    if eos in tokens:
        raise ValueError("candidate_tokens must exclude the termination token")
    prefix: tuple[Any, ...] = ()
    total = 0.0
    for token in (*tokens, eos):
        step = _step_map(next_logprobs_fn(prefix))
        if token not in step:
            return float("-inf")
        total += step[token]
        if token != eos:
            prefix = (*prefix, token)
    return float(total)


def score_fn_for(
    next_logprobs_fn: Callable[[tuple[Any, ...]], Iterable[tuple[Any, float]]],
    *,
    eos: Any,
) -> Callable[[Sequence[Any]], float]:
    """Bind a ``next_logprobs`` function into the ``score(candidate) -> float`` shape
    :func:`mixle.enumeration.top_k_scored` expects directly, for ranking a fixed candidate set."""
    return lambda candidate_tokens: score_candidate(next_logprobs_fn, candidate_tokens, eos=eos)


__all__ = [
    "VLM",
    "TokenizerBoundVLM",
    "CallableVLM",
    "OpenAICompatVLM",
    "exact_token_scorer_for",
    "score_candidate",
    "score_fn_for",
]
