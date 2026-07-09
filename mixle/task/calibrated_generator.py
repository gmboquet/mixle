"""``CalibratedGenerator`` -- conformal generation with honest abstention.

The generation-side sibling of :class:`~mixle.task.calibrate.CalibratedTaskModel`. That class gates
*classification*: it turns an uncalibrated softmax into conformal label sets and escalates on an
ambiguous (empty or multi-label) set. This module gates *generation*: draw ``k`` candidates from any
generator (a :class:`~mixle.task.llm.CallableLLM`, a sampler, a beam), score each candidate with any
mixle-scoreable model, and calibrate a conformal threshold on a held-out set so that serving the
best-scored candidate carries the same finite-sample coverage guarantee -- instead of always emitting
whatever scored highest, regardless of whether the score means anything.

The trick is treating the ``k`` candidates the way :class:`CalibratedTaskModel` treats classes: raw
candidate scores are softmax-normalized per prompt into selection probabilities, and
:func:`mixle.inference.conformal.conformal_label_threshold` / :func:`~mixle.inference.conformal.conformal_label_sets`
calibrate + apply the exact LAC threshold the classification sibling uses. A singleton conformal set ->
serve that candidate (covered at ``1 - alpha``); an empty or multi-candidate set -> honest "I'm not
sure" (:data:`ABSTAIN`) rather than a silent guess. ``ABSTAIN`` is ``None``, the same sentinel value as
:data:`mixle.task.calibrate.ESCALATE`, so a :class:`~mixle.task.cascade.Cascade` built on a
``CalibratedGenerator`` escalates on abstention exactly the way it escalates on ``ESCALATE`` -- no
special-casing needed on the cascade side.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.inference.conformal import conformal_label_sets, conformal_label_threshold

ABSTAIN = None  # sentinel returned when no candidate clears the calibrated threshold; equals Cascade's ESCALATE


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _derive_seed(base_seed: int, prompt: Any) -> int:
    """A stable (cross-process) per-prompt seed derived from ``base_seed`` -- unlike builtin ``hash()``,
    which is salted per-process by default, so it cannot be used to reproduce a draw across runs."""
    digest = hashlib.sha256(f"{base_seed}:{prompt!r}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class CalibratedGenerator:
    """Draw ``k`` scored candidates and serve the best one under a conformal accept-or-abstain guarantee.

    Args:
        generate: ``generate(prompt, k) -> Sequence[candidate]`` (an ``rng`` keyword is passed if the
            callable accepts one; falls back to the two-argument form otherwise). Any generator that can
            draw ``k`` candidates for a prompt works: a wrapped :class:`~mixle.task.llm.CallableLLM`
            sampled ``k`` times, a beam, a stochastic sampler.
        score: ``score(candidate) -> float``, any mixle-scoreable model. Higher is better; the score
            need not be a calibrated probability -- that is exactly what conformal calibration fixes.
        alpha: target miscoverage rate.
        k: number of candidates to draw per prompt.
        seed: base seed for candidate draws; combined with the prompt (see :func:`_derive_seed`) so
            different prompts get different, but reproducible, draws.
    """

    def __init__(
        self,
        generate: Callable[..., Sequence[Any]],
        score: Callable[[Any], float],
        alpha: float = 0.1,
        *,
        k: int = 8,
        qhat: float | None = None,
        seed: int = 0,
    ) -> None:
        self.generate = generate
        self.score = score
        self.alpha = float(alpha)
        self.k = int(k)
        self.qhat = qhat
        self.seed = int(seed)

    def _draw(self, prompt: Any, *, seed: int) -> list[Any]:
        rng = np.random.default_rng(seed)
        try:
            cands = self.generate(prompt, self.k, rng=rng)
        except TypeError:
            cands = self.generate(prompt, self.k)
        cands = list(cands)
        if not cands:
            raise ValueError(f"generate(...) returned no candidates for prompt {prompt!r}")
        return cands

    def _scored(self, prompt: Any, *, seed: int) -> tuple[list[Any], np.ndarray]:
        cands = self._draw(prompt, seed=seed)
        scores = np.asarray([float(self.score(c)) for c in cands], dtype=float)
        return cands, scores

    def calibrate(
        self, prompts: Sequence[Any], is_correct: Callable[[Any, Any], bool], *, seed: int | None = None
    ) -> CalibratedGenerator:
        """Fit the conformal threshold on a held-out set of prompts, given a correctness oracle.

        For each held-out prompt, ``k`` candidates are drawn and scored; the scores are softmax-normalized
        into per-prompt selection probabilities. The calibration score for that prompt is the probability
        mass landing on whichever candidate(s) ``is_correct(prompt, candidate)`` accepts (``0`` if none of
        the ``k`` draws is correct -- an honest miss that the calibration prices in, the same way a
        held-out example whose true class never appears in a small candidate set prices into a wider
        conformal set). :func:`mixle.inference.conformal.conformal_label_threshold` on those scores gives
        the same LAC threshold :class:`~mixle.task.calibrate.CalibratedTaskModel` calibrates for label sets.
        """
        rng_seed = self.seed if seed is None else int(seed)
        cal_prob_true = []
        for i, prompt in enumerate(prompts):
            cands, scores = self._scored(prompt, seed=_derive_seed(rng_seed, (i, prompt)))
            probs = _softmax(scores)
            mass = sum(p for c, p in zip(cands, probs) if is_correct(prompt, c))
            cal_prob_true.append(mass)
        if not cal_prob_true:
            raise ValueError("calibrate(...) needs at least one held-out prompt")
        self.qhat = conformal_label_threshold(np.asarray(cal_prob_true, dtype=float), alpha=self.alpha)
        return self

    def candidate_set(self, prompt: Any, *, seed: int | None = None) -> list[Any]:
        """The conformal candidate set for ``prompt`` -- candidates whose selection probability clears
        the calibrated threshold. A singleton set is what :meth:`serve` accepts; empty/multi abstains."""
        if self.qhat is None:
            raise RuntimeError("call calibrate(...) (or set qhat) before candidate_set(...)")
        call_seed = _derive_seed(self.seed, prompt) if seed is None else int(seed)
        cands, scores = self._scored(prompt, seed=call_seed)
        probs = _softmax(scores)
        sets, _ = conformal_label_sets(np.empty(0), probs[None, :], alpha=self.alpha, qhat=self.qhat)
        return [cands[i] for i in np.flatnonzero(sets[0])]

    def serve(self, prompt: Any, *, seed: int | None = None) -> Any:
        """Draw ``k`` candidates and return the best one if it conformally clears the threshold, else
        :data:`ABSTAIN`. ``ABSTAIN`` is returned for both an empty set (nothing confident enough) and a
        multi-candidate set (genuinely ambiguous) -- the same honest-uncertainty split
        :class:`~mixle.task.calibrate.CalibratedTaskModel` uses for classification."""
        admitted = self.candidate_set(prompt, seed=seed)
        return admitted[0] if len(admitted) == 1 else ABSTAIN

    def decide(self, prompt: Any, *, seed: int | None = None) -> Any:
        """Alias for :meth:`serve` with the same name as :meth:`CalibratedTaskModel.decide`, so a
        ``CalibratedGenerator`` drops into :class:`~mixle.task.cascade.Cascade` unmodified."""
        return self.serve(prompt, seed=seed)

    def __call__(self, prompt: Any, *, seed: int | None = None) -> Any:
        return self.serve(prompt, seed=seed)

    def abstention_rate(self, prompts: Sequence[Any], *, seed: int | None = None) -> float:
        """Empirical fraction of ``prompts`` that would abstain -- the generation analogue of
        :meth:`CalibratedTaskModel.escalation_rate`."""
        prompts = list(prompts)
        if not prompts:
            return 0.0
        outcomes = [self.serve(p, seed=seed) for p in prompts]
        return float(np.mean([o is ABSTAIN for o in outcomes]))


__all__ = ["ABSTAIN", "CalibratedGenerator"]
