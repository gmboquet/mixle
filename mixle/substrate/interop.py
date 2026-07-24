"""External model interop for reasoner delegation.

:class:`ExternalModel` wraps a ``generate`` callable from an external model,
agent, hosted LLM, or remote tool. It can estimate semantic uncertainty by
sampling multiple answers and clustering them through an equivalence function.
The returned answer is always one of the assessed samples, never a separate,
unassessed call (MXR-080-0270) -- the verdict genuinely describes the text the
caller actually receives.

:func:`external_action` adapts the wrapper into a reasoner
:class:`~mixle.substrate.act.Action`. When the external model is not
CONFIDENT -- genuinely uncertain, or never calibrated at all -- the action
contributes no evidence, allowing the reasoner to continue or abstain.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Confidence(StrEnum):
    """The three-state verdict on an external answer's semantic-entropy assessment (MXR-080-0270) --
    a closed vocabulary, never a bare boolean: a bare ``confident=True`` cannot distinguish "assessed
    against a calibrated cutoff and found consistent" from "no calibration policy was ever configured,
    so there was nothing to compare the entropy against" -- exactly the fail-open overclaim this
    finding fixes (mirrors :class:`~mixle.substrate.freshness.FreshnessState`).

    CONFIDENT: a calibration policy (``calibration_prompts`` or an explicit ``max_entropy``) is
        configured, and the returned answer's own measured semantic entropy is at or below the cutoff.
    UNCERTAIN: a calibration policy is configured, but the measured semantic entropy exceeds the
        cutoff -- the sampled batch (which the returned answer is itself drawn from) disagrees with
        itself more than the calibrated policy tolerates.
    UNCALIBRATED: no calibration policy is configured at all -- there is no cutoff to compare the
        entropy against, so the verdict fails closed rather than silently treating a missing cutoff
        as "anything finite is confident".
    """

    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    UNCALIBRATED = "uncalibrated"


@dataclass
class ExternalAnswer:
    """An external model's answer plus its self-measured uncertainty (semantic entropy).

    :attr:`answer` is itself one of the samples measured for :attr:`entropy` (MXR-080-0270), drawn
    from the same batch rather than a separate, unassessed call -- so :attr:`state` genuinely
    describes the returned text's own membership in the measured cluster.
    """

    prompt: Any
    answer: Any
    entropy: float
    state: Confidence

    @property
    def confident(self) -> bool:
        """Read-only convenience view: ``True`` iff :attr:`state` is :attr:`Confidence.CONFIDENT`.

        Never itself the source of truth -- :attr:`state` is (MXR-080-0270): both
        :attr:`Confidence.UNCERTAIN` and :attr:`Confidence.UNCALIBRATED` read ``False`` here, so an
        uncalibrated model can no longer read as confident just because its entropy happened to be
        finite.
        """
        return self.state is Confidence.CONFIDENT

    @property
    def calibrated(self) -> bool:
        """``True`` unless :attr:`state` is :attr:`Confidence.UNCALIBRATED`."""
        return self.state is not Confidence.UNCALIBRATED


class ExternalModel:
    """An external ``generate`` callable wrapped so each answer carries semantic-entropy UQ.

    Args:
        generate: ``prompt -> answer`` (an external agent / LLM / remote tool). Called multiple times
            per query to measure how much its meaning varies (the uncertainty signal).
        calibration_prompts: example prompts whose (1-alpha) semantic-entropy quantile becomes the
            "too uncertain" cutoff -- a calibration policy. Required unless ``max_entropy`` is given
            directly: with neither (or with calibration prompts too few/degenerate to yield a finite
            quantile), no calibration policy exists at all, and every answer's
            :attr:`~ExternalAnswer.state` is :attr:`Confidence.UNCALIBRATED` (MXR-080-0270) -- never
            silently treated as confident just because its entropy happened to be finite.
        equivalent: ``(a, b) -> bool`` meaning-equivalence for clustering samples (default: exact match).
        max_entropy: an explicit uncertainty cutoff, overriding the calibrated one. Must be finite and
            non-negative -- entropy is never negative, so a non-finite or negative cutoff is always a
            caller error, not a valid (if strict) policy.
        samples: how many draws to sample per :meth:`answer` call. The first draw becomes the returned
            answer; entropy is measured over the same batch it came from (MXR-080-0270).
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
            max_entropy = float(max_entropy)
            if not math.isfinite(max_entropy) or max_entropy < 0.0:
                raise ValueError(f"max_entropy must be a finite, non-negative cutoff, got {max_entropy!r}")
            self._uq.payload["max_entropy"] = max_entropy

    @property
    def max_entropy(self) -> float | None:
        """Semantic-entropy cutoff used to decide confidence, or ``None`` if no calibration policy
        is configured (MXR-080-0270).

        ``None`` whenever neither ``calibration_prompts`` nor an explicit ``max_entropy`` produced a
        finite cutoff -- callers must fail closed on ``None``, never treat a missing cutoff as
        "anything finite is confident" the way an unguarded ``entropy <= inf`` silently would.
        """
        raw = float(self._uq.payload.get("max_entropy", float("inf")))
        return raw if math.isfinite(raw) else None

    def answer(self, prompt: Any) -> ExternalAnswer:
        """Sample the external model; the first draw is the returned answer (MXR-080-0270).

        Every draw -- the one returned and every other sample used to measure entropy -- comes from
        the same batch, so the confidence verdict genuinely describes the returned text's own
        semantic position in the cluster, rather than a separately generated call that was never
        actually compared to anything. Fails closed to :attr:`Confidence.UNCALIBRATED` when no
        calibration policy (``calibration_prompts`` or an explicit ``max_entropy``) is configured,
        however low the measured entropy happens to be.
        """
        from mixle.inference.uncertainty import semantic_entropy as _semantic_entropy

        equivalent = self._uq.payload.get("equivalent")
        draws = [self.generate(prompt) for _ in range(self.samples)]
        entropy = float(_semantic_entropy(draws, equivalent))
        cutoff = self.max_entropy
        if cutoff is None:
            state = Confidence.UNCALIBRATED
        elif entropy <= cutoff:
            state = Confidence.CONFIDENT
        else:
            state = Confidence.UNCERTAIN
        return ExternalAnswer(prompt=prompt, answer=draws[0], entropy=entropy, state=state)

    def confident(self, prompt: Any) -> bool:
        """Whether the external model is calibrated-confident on ``prompt``.

        Fails closed (``False``) when no calibration policy is configured (MXR-080-0270) -- an
        uncalibrated model is never reported confident, regardless of measured entropy.
        """
        cutoff = self.max_entropy
        return cutoff is not None and self._uq.confident(prompt, n=self.samples, max_entropy=cutoff)


def external_action(
    model: ExternalModel,
    *,
    name: str = "external",
    cost: float = 8.0,
    description: str = "",
    trust_uncertain: bool = False,
) -> Any:
    """A reasoner delegate action backed by a UQ-wrapped external model (see module docstring).

    By default (``trust_uncertain=False``) the action contributes evidence only when the external
    model's answer is :attr:`Confidence.CONFIDENT`; an :attr:`~Confidence.UNCERTAIN` or
    :attr:`~Confidence.UNCALIBRATED` answer yields no fragment, so the reasoner treats it as no answer
    rather than a guess (MXR-080-0270: a never-calibrated model is withheld exactly like a genuinely
    self-contradicting one -- both are "we cannot vouch for this", not "confident"). The fragment
    carries the model's entropy and verdict so the trace records how sure the external source was, and
    why. Cost defaults high -- external calls are the escalation of last resort."""
    from mixle.substrate.act import Action

    def _run(question: str) -> list[str]:
        result = model.answer(question)
        if not result.confident and not trust_uncertain:
            return []  # uncalibrated or self-contradicting external answer -> withhold, don't fabricate confidence
        return [f"external[{result.state.value}, entropy={result.entropy:.3f}] => {result.answer}"]

    return Action(name=name, kind="delegate", run=_run, cost=cost, description=description)
