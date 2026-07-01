"""Uncertainty quantification for LLM output -- the thing LLMs famously lack, on the mixle UQ spine.

An LLM emits fluent text with no honest sense of whether it knows the answer. Wrapping *any*
``generate(prompt) -> str`` callable, :class:`LLMUncertainty` turns repeated stochastic samples into
calibrated uncertainty:

* **Semantic entropy** (Kuhn et al. 2023): sample the model ``n`` times, cluster the answers by
  *meaning* (not surface form), and take the entropy over meaning-clusters. High = the model
  disagrees with itself about *what* the answer is -- a hallucination signal -- while merely
  rephrasing one answer clusters to low entropy. (:func:`mixle.inference.semantic_entropy`.)
* **Epistemic vs aleatoric split**: draw samples under several *members* (paraphrased prompts, or
  higher temperature as a proxy ensemble); the disagreement *across* members is epistemic (the model
  is unsure), the spread *within* is aleatoric (the question is genuinely open).
  (:func:`mixle.inference.decompose_entropy`.)
* **Conformal answer-or-abstain**: calibrate a confidence threshold on labeled examples so that
  *when the model answers, it is correct with probability >= 1 - alpha* -- a finite-sample selective-
  risk guarantee. The model abstains on questions it does not know instead of confabulating.

Domain-neutral in its dependency: it takes a plain ``generate`` callable and an ``equivalent``
relation, so it works with a local ``mixle.task`` model, an OpenAI-compatible endpoint (mlops), or a
mock -- no hard LLM dependency here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.uncertainty import (
    UncertaintyDecomposition,
    cluster_samples,
    decompose_entropy,
    semantic_entropy,
)

_STOP = frozenset(
    "a an the is are was were be been being of to in on at by for with and or but it its this that "
    "as from into over under near".split()
)


def sentence_claims(text: str) -> list[str]:
    """Split a response into atomic claims (sentence-ish units) -- the default claim extractor."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


def _content_words(s: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if w not in _STOP}


def content_overlap(sample: str, claim: str, *, threshold: float = 0.6) -> bool:
    """Simple corroboration test: does ``sample`` cover >= ``threshold`` of ``claim``'s content words?

    Counts every content word equally, so boilerplate shared across responses ("the tower is located
    in ...") can mask that the *informative* word (the city) differs. :func:`information_corroborator`
    fixes that by weighting words by their information content; it is the default in
    :meth:`LLMUncertainty.assess_claims`.
    """
    cw = _content_words(claim)
    if not cw:
        return False
    return len(cw & _content_words(sample)) / len(cw) >= threshold


def information_corroborator(samples: Sequence[str], *, overlap: float = 0.5) -> Callable[[str, str], bool]:
    """Build a corroboration test that weights each word by its *information content* over ``samples``.

    A word appearing in nearly every sample is boilerplate (low information, low weight); a rare word
    carries the actual claim (high weight). A sample corroborates a claim when it covers at least
    ``overlap`` of the claim's *information-weighted* words -- so whether the distinctive fact (a city,
    a number, a name) matches drives the decision, not the shared filler. Inverse-document-frequency
    weighting: ``w(word) = log((N + 1) / (df + 0.5))``.
    """
    df: Counter[str] = Counter()
    for s in samples:
        df.update(_content_words(s))
    n = len(samples)

    def weight(w: str) -> float:
        return math.log((n + 1.0) / (df.get(w, 0) + 0.5))

    def corroborates(sample: str, claim: str) -> bool:
        cw = _content_words(claim)
        if not cw:
            return False
        sw = _content_words(sample)
        num = sum(weight(w) for w in cw if w in sw)
        den = sum(weight(w) for w in cw)
        return den > 0.0 and num / den >= overlap

    return corroborates


@dataclass(frozen=True)
class LLMAssessment:
    """One prompt's assessed answer with uncertainty.

    ``answer`` is the majority meaning-cluster's representative; ``confidence`` its cluster share in
    ``[0, 1]``; ``semantic_entropy`` the nats of meaning-uncertainty; ``clusters`` the
    ``[(representative, probability), ...]`` distribution over meanings; ``samples`` the raw draws.
    """

    answer: Any
    confidence: float
    semantic_entropy: float
    clusters: list[tuple[Any, float]]
    samples: list[Any]


@dataclass(frozen=True)
class ClaimAssessment:
    """Reliability of one claim inside a response, by cross-sample corroboration.

    ``support`` is the fraction of independent resamples that corroborate the claim (in ``[0, 1]``);
    ``reliable`` is ``support >= threshold``. A claim the model actually knows recurs across samples
    (high support); a fabricated one appears once and vanishes (low support).
    """

    claim: str
    support: float
    reliable: bool


@dataclass(frozen=True)
class InformationAssessment:
    """UQ over the *information content* of a response: every claim scored, plus a summary.

    ``claims`` is the per-claim reliability; ``reliability`` the mean support (how trustworthy the
    response's information is overall); ``fabricated`` the claims below threshold (likely hallucinated).
    """

    claims: list[ClaimAssessment]
    reliability: float

    @property
    def fabricated(self) -> list[ClaimAssessment]:
        return [c for c in self.claims if not c.reliable]


class LLMUncertainty:
    """Calibrated uncertainty and selective prediction for any ``generate(prompt) -> str`` LLM.

    Args:
        generate: ``callable(prompt) -> str`` -- one stochastic sample from the model.
        equivalent: ``callable(a, b) -> bool`` deciding whether two answers mean the same thing
            (default exact match; pass a normalizer / embedding / entailment check for real text).
        n: default number of samples per prompt.
    """

    def __init__(
        self,
        generate: Callable[[str], Any],
        *,
        equivalent: Callable[[Any, Any], bool] | None = None,
        n: int = 10,
    ) -> None:
        self.generate = generate
        self.equivalent = equivalent
        self.n = int(n)
        self._threshold: float | None = None
        self._alpha: float | None = None

    def sample(self, prompt: str, n: int | None = None) -> list[Any]:
        """Draw ``n`` stochastic responses to ``prompt``."""
        return [self.generate(prompt) for _ in range(int(n or self.n))]

    def assess(self, prompt: str, n: int | None = None) -> LLMAssessment:
        """Sample, cluster by meaning, and report the answer with its semantic uncertainty."""
        samples = self.sample(prompt, n)
        c = cluster_samples(samples, self.equivalent)
        top = int(np.argmax(c.probs))
        clusters = sorted(zip(c.representatives, c.probs.tolist()), key=lambda t: -t[1])
        return LLMAssessment(
            answer=c.representatives[top],
            confidence=float(c.probs[top]),
            semantic_entropy=semantic_entropy(samples, self.equivalent),
            clusters=clusters,
            samples=samples,
        )

    def decompose(self, prompts: Sequence[str], n: int | None = None) -> UncertaintyDecomposition:
        """Epistemic/aleatoric split across *member* prompts (paraphrases of one question).

        Each prompt is a member; all members' samples are pooled to define shared meaning-clusters,
        then each member's distribution over those clusters feeds :func:`decompose_entropy`. Epistemic
        = disagreement across paraphrasings (prompt-sensitivity / model uncertainty); aleatoric =
        within-member spread.
        """
        members = [self.sample(p, n) for p in prompts]
        pooled = [s for member in members for s in member]
        clusters = cluster_samples(pooled, self.equivalent)
        reps = clusters.representatives
        eq = self.equivalent if self.equivalent is not None else (lambda a, b: a == b)

        def dist(member: list[Any]) -> np.ndarray:
            counts = np.zeros(len(reps))
            for s in member:
                ci = next((i for i, r in enumerate(reps) if eq(s, r)), None)
                if ci is not None:
                    counts[ci] += 1
            return counts / counts.sum() if counts.sum() else counts

        return decompose_entropy(np.array([dist(m) for m in members]))

    # -- claim-level UQ: reliability of the information inside a response ----------------------
    def assess_claims(
        self,
        prompt: str,
        *,
        extract: Callable[[str], Sequence[str]] | None = None,
        corroborates: Callable[[str, str], bool] | None = None,
        n: int | None = None,
        threshold: float = 0.5,
    ) -> InformationAssessment:
        """Score the reliability of each *claim* in the response by cross-sample corroboration.

        Finer-grained than :meth:`assess`: a response can be internally consistent (low semantic
        entropy) yet contain one fabricated fact. This decomposes the response into claims and checks
        each *unit of information* separately -- a claim the model knows recurs across independent
        resamples; a hallucinated one appears once. This is UQ *on the information in what is said*,
        not just on the answer as a whole.

        Args:
            prompt: the query.
            extract: ``response -> [claim, ...]`` (default :func:`sentence_claims`).
            corroborates: ``(other_sample, claim) -> bool`` -- does a resample support the claim?
                (default :func:`content_overlap`; pass an entailment/NLI check for real text).
            n: number of samples (the first is the response scored; the rest corroborate).
            threshold: support below which a claim is flagged as unreliable/fabricated.
        """
        extract = extract or sentence_claims
        samples = self.sample(prompt, n)
        if len(samples) < 2:
            samples = samples + self.sample(prompt, 2 - len(samples))
        primary, others = samples[0], samples[1:]
        # default corroboration weights words by information content over the drawn samples, so the
        # distinctive fact drives the decision rather than shared boilerplate.
        corr = corroborates or information_corroborator(samples)
        claims = list(extract(primary))
        assessed: list[ClaimAssessment] = []
        for claim in claims:
            support = float(np.mean([corr(s, claim) for s in others])) if others else 1.0
            assessed.append(ClaimAssessment(claim, support, support >= threshold))
        reliability = float(np.mean([c.support for c in assessed])) if assessed else 1.0
        return InformationAssessment(assessed, reliability)

    # -- conformal answer-or-abstain ----------------------------------------------------------
    def calibrate(
        self,
        examples: Sequence[tuple[str, Any]],
        *,
        correct: Callable[[Any, Any], bool] | None = None,
        alpha: float = 0.1,
        n: int | None = None,
    ) -> LLMUncertainty:
        """Calibrate a confidence threshold for selective risk ``<= alpha`` on labeled ``(prompt, gold)``.

        For each example, the model's answer (majority meaning-cluster) and its confidence are
        computed; ``correct(answer, gold)`` (default the ``equivalent`` relation) marks it right or
        wrong. The threshold is the lowest confidence at which the selective error rate on the
        calibration set is ``<= alpha`` -- so :meth:`answer` abstains below it and, when it answers,
        is right with probability about ``1 - alpha``.
        """
        corr = correct or self.equivalent or (lambda a, b: a == b)
        confs, errs = [], []
        for prompt, gold in examples:
            a = self.assess(prompt, n)
            confs.append(a.confidence)
            errs.append(0.0 if corr(a.answer, gold) else 1.0)
        confs = np.asarray(confs)
        errs = np.asarray(errs)
        # among candidate thresholds (observed confidences), the smallest tau whose answered set has
        # empirical error <= alpha; if none qualifies, refuse everything (tau just above the max).
        best = float(confs.max()) + 1e-9
        for tau in np.unique(confs):
            answered = confs >= tau
            if answered.any() and errs[answered].mean() <= alpha:
                best = float(tau)
                break
        self._threshold = best
        self._alpha = float(alpha)
        return self

    def answer(self, prompt: str, n: int | None = None) -> LLMAssessment | None:
        """Answer ``prompt`` if confident enough, else ``None`` (abstain).

        Requires a prior :meth:`calibrate`. Returns the :class:`LLMAssessment` when
        ``confidence >= threshold`` (so the answer meets the selective-risk guarantee), else ``None``.
        """
        if self._threshold is None:
            raise RuntimeError("call calibrate(...) before answer()")
        a = self.assess(prompt, n)
        return a if a.confidence >= self._threshold else None
