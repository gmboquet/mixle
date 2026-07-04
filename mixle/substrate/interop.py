"""Interop: wrap an EXTERNAL model/agent as a UQ-carrying, self-doubting reasoner delegate (Q).

A reasoner can ``delegate`` to an external worker -- another agent, a hosted LLM, a remote tool. But an
external model is untrusted: it may confidently hallucinate. :class:`ExternalModel` wraps any external
``generate`` callable so every answer carries its OWN uncertainty -- semantic entropy over resampled
answers (via :func:`mixle.inference.uq`), the model's disagreement with itself. :func:`external_action`
turns that into a reasoner :class:`~mixle.substrate.act.Action`: it calls the external model, and when
the model is NOT confident (entropy above a calibrated cutoff) it contributes NO evidence, so the
reasoner never trusts a self-contradicting external answer -- it falls through to abstain instead.

This is the interop half of the 99%-local topology: external capability is reachable (A2A / remote tool
/ hosted LLM), but it enters the evidence loop only with a UQ receipt attached and only when it clears
the same honesty bar as everything local. The cost stays high (external calls are the escalation of last
resort), and an uncertain external answer is treated as no answer, not a guess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ExternalAnswer:
    """An external model's answer plus its self-measured uncertainty (semantic entropy)."""

    prompt: Any
    answer: Any
    entropy: float
    confident: bool


class ExternalModel:
    """An external ``generate`` callable wrapped so each answer carries semantic-entropy UQ.

    Args:
        generate: ``prompt -> answer`` (an external agent / LLM / remote tool). Called multiple times
            per query to measure how much its meaning varies (the uncertainty signal).
        calibration_prompts: optional example prompts; the (1-alpha) quantile of their semantic entropy
            becomes the "too uncertain" cutoff. Without them, ``max_entropy`` must be given (or every
            answer is treated as confident).
        equivalent: ``(a, b) -> bool`` meaning-equivalence for clustering samples (default: exact match).
        max_entropy: an explicit uncertainty cutoff, overriding the calibrated one.
        samples: how many resamples to draw when measuring entropy.
    """

    def __init__(
        self,
        generate: Callable[[Any], Any],
        *,
        calibration_prompts: Any = None,
        equivalent: Callable[[Any, Any], bool] | None = None,
        max_entropy: float | None = None,
        alpha: float = 0.1,
        samples: int = 8,
    ) -> None:
        from mixle.inference.uq import uq

        self.generate = generate
        self.samples = int(samples)
        self._uq = uq(generate, calibration_prompts, alpha=alpha, equivalent=equivalent)
        if max_entropy is not None:
            self._uq.payload["max_entropy"] = float(max_entropy)

    @property
    def max_entropy(self) -> float:
        return float(self._uq.payload.get("max_entropy", float("inf")))

    def answer(self, prompt: Any) -> ExternalAnswer:
        """Call the external model and attach its semantic-entropy UQ (confident iff below the cutoff)."""
        text = self.generate(prompt)
        entropy = self._uq.semantic_entropy(prompt, n=self.samples)
        return ExternalAnswer(
            prompt=prompt,
            answer=text,
            entropy=float(entropy),
            confident=entropy <= self.max_entropy,
        )

    def confident(self, prompt: Any) -> bool:
        return self._uq.confident(prompt, n=self.samples)


def external_action(
    model: ExternalModel,
    *,
    name: str = "external",
    cost: float = 8.0,
    description: str = "",
    trust_uncertain: bool = False,
) -> Any:
    """A reasoner DELEGATE action backed by a UQ-wrapped external model (see module docstring).

    By default (``trust_uncertain=False``) the action contributes evidence ONLY when the external model
    is confident about the query; an uncertain external answer yields no fragment, so the reasoner treats
    it as no answer rather than a guess. The fragment carries the model's entropy so the trace records how
    sure the external source was. Cost defaults high -- external calls are the escalation of last resort."""
    from mixle.substrate.act import Action

    def _run(question: str) -> list[str]:
        result = model.answer(question)
        if not result.confident and not trust_uncertain:
            return []  # self-contradicting external answer -> withhold, don't fabricate confidence
        tag = "confident" if result.confident else "uncertain"
        return [f"external[{tag}, entropy={result.entropy:.3f}] => {result.answer}"]

    return Action(name=name, kind="delegate", run=_run, cost=cost, description=description)
