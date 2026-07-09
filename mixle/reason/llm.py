"""Uncertainty quantification for LLM output on Mixle's UQ surface.

An LLM can emit fluent text without calibrated confidence. Wrapping any
``generate(prompt) -> str`` callable, :class:`LLMUncertainty` turns repeated
stochastic samples into calibrated uncertainty:

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

The dependency boundary is domain-neutral: it takes a plain ``generate``
callable and an ``equivalent`` relation, so it works with a local
``mixle.task`` model, an OpenAI-compatible endpoint, or a test double without a
hard LLM dependency in this module.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.calibration import ProbabilityCalibrator, calibrate_probabilities
from mixle.inference.uncertainty import (
    UncertaintyDecomposition,
    cluster_samples,
    decompose_entropy,
    marginalize_meaning,
    semantic_entropy,
)

_STOP = frozenset(
    "a an the is are was were be been being of to in on at by for with and or but it its this that "
    "as from into over under near".split()
)


def _auc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    """Rank-based AUC of ``scores`` vs binary ``outcomes`` -- how well the signal separates right/wrong."""
    pos = np.sum(outcomes == 1.0)
    neg = np.sum(outcomes == 0.0)
    if pos == 0 or neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(scores)) + 1.0  # average-rank ties are negligible for a diagnostic
    return float((ranks[outcomes == 1.0].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


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
        """Return claims assessed as unreliable."""
        return [c for c in self.claims if not c.reliable]


@dataclass(frozen=True)
class FactualityModel:
    """A fitted map from a per-prompt uncertainty signal to a *calibrated* ``P(answer is correct)``.

    The signal (self-consistency, a token likelihood, ...) is only a raw number; the calibrator turns
    it into a genuine probability of the *information* being correct, learned against labeled facts.
    ``discrimination`` (held-out AUC on the fit set) reports how much the signal actually knew about
    correctness -- ~0.5 means the signal was unrelated to truth, no matter how confident it looked.
    """

    calibrator: ProbabilityCalibrator
    signal: Callable[[str], float]
    discrimination: float

    def probability(self, prompt: str) -> float:
        """Calibrated probability that the answer's information is correct."""
        return float(self.calibrator.predict([float(self.signal(prompt))])[0])


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
        """Draw ``n`` stochastic responses to ``prompt``.

        ``generate`` may return a plain string, or a ``(text, logprob)`` pair -- the sequence
        log-probability ``log P(s)``. When logprobs are provided they are used to marginalize the
        string distribution over meaning classes exactly (:func:`mixle.inference.marginalize_meaning`)
        rather than by sample counting.
        """
        return [self.generate(prompt) for _ in range(int(n or self.n))]

    @staticmethod
    def _split(samples: list[Any]) -> tuple[list[Any], np.ndarray | None]:
        """Separate ``(text, logprob)`` pairs into texts + log-probs; pass strings through unchanged."""
        if samples and isinstance(samples[0], tuple) and len(samples[0]) == 2:
            return [s[0] for s in samples], np.array([float(s[1]) for s in samples])
        return samples, None

    def assess(self, prompt: str, n: int | None = None) -> LLMAssessment:
        """Sample, marginalize the string distribution over meaning classes, and report the answer.

        The reported ``confidence`` is the marginal probability of the top *meaning* (summed over its
        equivalence class of strings), and ``semantic_entropy`` the entropy of that meaning marginal
        -- not a per-string token probability.
        """
        texts, log_probs = self._split(self.sample(prompt, n))
        m = marginalize_meaning(texts, self.equivalent, log_probs=log_probs)
        top = int(np.argmax(m.probs))
        clusters = sorted(zip(m.representatives, m.probs.tolist()), key=lambda t: -t[1])
        return LLMAssessment(
            answer=m.representatives[top],
            confidence=float(m.probs[top]),
            semantic_entropy=semantic_entropy(texts, self.equivalent, log_probs=log_probs),
            clusters=clusters,
            samples=texts,
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

    # -- calibrated information likelihood ----------------------------------------------------
    def fit_factuality(
        self,
        examples: Sequence[tuple[str, Any]],
        *,
        signal: Callable[[str], float] | None = None,
        correct: Callable[[Any, Any], bool] | None = None,
        method: str = "isotonic",
        n: int | None = None,
    ) -> FactualityModel:
        """Learn a calibrated ``P(answer is correct)`` from a raw signal, on labeled ``(prompt, gold)``.

        The model's raw confidence (its self-consistency, or a token likelihood) is *not* a
        probability that the information is correct -- it can be systematically over/under-confident,
        or unrelated to truth. This fits a :class:`~mixle.inference.ProbabilityCalibrator` mapping the
        signal to the empirical correctness rate, so the output *is* a probability of the information
        being right. ``discrimination`` (AUC of signal vs correctness) reports how much the raw signal
        knew at all -- ~0.5 means it was unrelated to truth, calibration or not.

        Args:
            examples: labeled ``(prompt, gold_answer)`` pairs.
            signal: ``prompt -> float`` raw score (default: the self-consistency confidence from
                :meth:`assess`).
            correct: ``(answer, gold) -> bool`` (default the ``equivalent`` relation).
            method: calibration map -- ``"isotonic"`` or ``"platt"``.
            n: samples per prompt.
        """
        corr = correct or self.equivalent or (lambda a, b: a == b)
        scores, outcomes = [], []
        for prompt, gold in examples:
            a = self.assess(prompt, n)
            scores.append(float(signal(prompt)) if signal is not None else a.confidence)
            outcomes.append(1.0 if corr(a.answer, gold) else 0.0)
        scores_arr = np.asarray(scores, dtype=float)
        outcomes_arr = np.asarray(outcomes, dtype=float)
        calibrator = calibrate_probabilities(scores_arr, outcomes_arr, method=method)
        sig = signal if signal is not None else (lambda p: self.assess(p, n).confidence)
        return FactualityModel(calibrator, sig, _auc(scores_arr, outcomes_arr))
