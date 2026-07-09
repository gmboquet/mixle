"""External model interop for reasoner delegation.

:class:`ExternalModel` wraps a ``generate`` callable from an external model,
agent, hosted LLM, or remote tool. It can estimate semantic uncertainty by
sampling multiple answers and clustering them through an equivalence function.

:func:`external_action` adapts the wrapper into a reasoner
:class:`~mixle.substrate.act.Action`. When the external model is above its
uncertainty cutoff, the action contributes no evidence, allowing the reasoner
to continue or abstain.
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
        """Semantic-entropy cutoff used to decide whether answers are trusted."""
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
        """Return whether the external model is calibrated-confident on ``prompt``."""
        return self._uq.confident(prompt, n=self.samples)


def external_action(
    model: ExternalModel,
    *,
    name: str = "external",
    cost: float = 8.0,
    description: str = "",
    trust_uncertain: bool = False,
) -> Any:
    """A reasoner delegate action backed by a UQ-wrapped external model (see module docstring).

    By default (``trust_uncertain=False``) the action contributes evidence only when the external model
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
