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
* **Selective answer-or-abstain** (:meth:`LLMUncertainty.calibrate`): calibrate a confidence threshold
  on labeled examples so that *when the model answers, it is correct with probability >= 1 - alpha* --
  with probability >= ``1 - delta`` over the random calibration set itself (Geifman & El-Yaniv 2017;
  Angelopoulos et al. 2021 "Learn then Test"): a proper finite-sample ``(alpha, delta)``-PAC
  selective-risk guarantee via an exact Clopper-Pearson bound, Bonferroni-corrected across every
  candidate threshold tried -- not a same-sample point estimate. The guarantee is conditional on
  i.i.d. traffic AND the entire serving policy (generator behavior, equivalence relation, sample
  count) remaining exactly the calibrated one -- identity checks catch rebinding, not in-place
  state change; pin and compare the policy token (STAT-RR22-09/RR23-09). The model abstains on
  questions it does not know instead of confabulating.
* **Claim-level corroboration** (:meth:`LLMUncertainty.assess_claims`): lexical overlap only ever
  establishes *candidacy* -- that a resample is plausibly about the same thing as a claim -- never
  support by itself; a negation/polarity check on the shared content words is what separates genuine
  corroboration from a resample that shares the claim's vocabulary while disagreeing with it.

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
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import rankdata

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
    """Rank-based AUC of ``scores`` vs binary ``outcomes`` -- how well the signal separates right/wrong.

    Mann-Whitney-U-equivalent rank-sum formula, using AVERAGE ranks for ties
    (:func:`scipy.stats.rankdata` with ``method="average"``) (MXR-080-0294). Self-consistency-style
    scores are frequently discrete (``k / n`` agreement), so exact ties are the common case here, not
    an edge case: a plain ``argsort``-of-``argsort`` ranking breaks ties arbitrarily by array position,
    which makes the reported AUC depend on input ORDER within a tied group -- nonsensical, since AUC is
    supposed to depend only on the ``(score, outcome)`` multiset -- and is systematically biased, e.g.
    four identical scores split 2 correct / 2 incorrect can report 0.25 instead of the correct,
    order-invariant 0.5 for a signal with zero discrimination. Average ranks are the textbook fix: every
    tied value shares the mean of the ranks its group spans, which is what makes this formula compute
    ``P(random positive scores higher than random negative) + 0.5 * P(tie)``, the correct generalization
    of AUC to tied scores.
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if scores.shape != outcomes.shape:
        raise ValueError(f"scores and outcomes must have matching shape, got {scores.shape} and {outcomes.shape}.")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite for AUC.")
    if not np.isfinite(outcomes).all() or not np.all((outcomes == 0.0) | (outcomes == 1.0)):
        raise ValueError("outcomes must be finite and binary (0.0/1.0) for AUC.")
    pos = np.sum(outcomes == 1.0)
    neg = np.sum(outcomes == 0.0)
    if pos == 0 or neg == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
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
    fixes that by weighting words by their information content, and is the candidacy signal behind
    :meth:`LLMUncertainty.assess_claims`'s default corroborator. Like :func:`information_corroborator`,
    this measures topical overlap only, never polarity: a negated sentence sharing a claim's content
    words still counts as covering them. :mod:`mixle.substrate.factuality` layers a negation/polarity
    check on top of this exact function for that reason (see its ``Corroboration``), and
    :func:`_default_claim_corroborator` below does the analogous thing for this module.
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

    This is a CANDIDACY test, not a support test: it says nothing about polarity, so a sample that
    negates the claim while sharing its distinctive words still corroborates by this measure alone
    (:func:`content_overlap` has the identical limitation). :meth:`LLMUncertainty.assess_claims`'s
    default corroborator (:func:`_default_claim_corroborator`) uses this as its candidacy signal, gated
    by a negation/polarity check -- see :class:`Corroboration`.
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


# -- claim-level corroboration: lexical candidacy gated by a negation/polarity check ---------------
#
# Plain lexical overlap (content_overlap / information_corroborator) is a CANDIDACY signal, not a
# support signal: two statements can share almost every content word and mean opposite things ("the
# drug cures cancer" / "the drug cures no cancer and does not work"). This is not a real entailment
# model -- it is a negation-cue heuristic, tractable without an NLI dependency this codebase does not
# otherwise carry. It mirrors mixle.substrate.factuality's MXR-080-0258 fix and is duplicated here
# rather than imported: mixle.substrate.factuality already imports content_overlap from this module,
# and this module should not import back from mixle.substrate for a few lines of regex neither side
# needs the other's copy of.

_NEGATORS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "neither",
        "nor",
        "cannot",
        "without",
        "lacks",
        "lacking",
        "fails",
        "unable",
    }
)

_NEG_WINDOW = 4  # how many tokens after a negator its scope reaches, e.g. "not [X Y Z]"


def _tokens(text: str) -> list[str]:
    """Tokenize with n't-contraction expansion (doesn't -> does not) so a negation survives a
    contraction instead of splitting into two non-negator fragments ("doesn" + "t")."""
    normalized = re.sub(r"n't\b", " not", str(text).lower())
    return re.findall(r"[a-z0-9]+", normalized)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", str(text).strip()) if s]


def _negated_content_words(text: str) -> set[str]:
    """Content words inside a negation marker's forward scope, sentence-scoped.

    A negator's scope never crosses a sentence boundary (an unrelated negation two sentences later in
    a long response can't falsely taint an earlier claim's words) and stops at the next negator (so
    two independent negations in one sentence -- "cures no cancer and does not work" -- are not merged
    into one span).
    """
    negated: set[str] = set()
    for sent in _sentences(text) or [text]:
        toks = _tokens(sent)
        for i, tok in enumerate(toks):
            if tok not in _NEGATORS:
                continue
            for nxt in toks[i + 1 : i + 1 + _NEG_WINDOW]:
                if nxt in _NEGATORS:
                    break  # a second negator starts its own scope; do not merge the two
                if nxt not in _STOP:
                    negated.add(nxt)
    return negated


class Corroboration(StrEnum):
    """A corroborator's verdict on one (sample, claim) pair -- three-way, never a bare bool.

    Mirrors :class:`mixle.substrate.factuality.Corroboration` (MXR-080-0258): lexical candidacy
    (matching content words) is necessary but not sufficient for SUPPORTED -- a contradiction can
    share every content word with the claim it contradicts (see the module-level comment above this
    section for the exact example). UNVERIFIED is a real, distinct outcome from SUPPORTED -- "found no
    reason to doubt it" is not "confirmed" -- so a caller can never mistake silence for verification.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"


def _coerce_verdict(result: Corroboration | bool) -> Corroboration:
    """Accept a legacy ``bool``-returning corroborator too: ``True`` -> SUPPORTED, ``False`` ->
    UNVERIFIED. Never CONTRADICTED -- a bare bool has no way to express a contradiction, so mapping a
    falsy legacy result to "unverified" rather than guessing is the only reading that can't overclaim.
    """
    if isinstance(result, Corroboration):
        return result
    return Corroboration.SUPPORTED if result else Corroboration.UNVERIFIED


def _default_claim_corroborator(samples: Sequence[str]) -> Callable[[str, str], Corroboration]:
    """The default :meth:`LLMUncertainty.assess_claims` corroborator (MXR-080-0295):
    :func:`information_corroborator`'s information-weighted overlap as candidacy, gated by a
    negation/polarity check.

    Overlap alone only establishes that ``sample`` is plausibly about the same thing as ``claim`` --
    candidacy, not support. Among the content words the two texts share, if any disagrees in polarity
    (negated in one, not the other -- e.g. "the drug works" vs "the drug does not work") the pair is
    CONTRADICTED regardless of how much else overlaps. Otherwise, candidacy is enough for SUPPORTED. No
    overlap at all -- or candidacy with no shared word left to check polarity on -- is UNVERIFIED,
    never guessed as supported.
    """
    candidacy = information_corroborator(samples)

    def corroborates(sample: str, claim: str) -> Corroboration:
        if not candidacy(sample, claim):
            return Corroboration.UNVERIFIED
        shared = (_content_words(claim) - _NEGATORS) & (_content_words(sample) - _NEGATORS)
        if not shared:
            return Corroboration.UNVERIFIED
        neg_claim = _negated_content_words(claim)
        neg_sample = _negated_content_words(sample)
        if any((w in neg_claim) != (w in neg_sample) for w in shared):
            return Corroboration.CONTRADICTED
        return Corroboration.SUPPORTED

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
class Generation:
    """One stochastic sample from ``generate``, normalized: text immutably paired with its
    log-probability when the generator provides one (MXR-080-0296).

    Before this, only :meth:`LLMUncertainty.assess` split a ``(text, logprob)`` pair from a plain
    string -- :meth:`LLMUncertainty.decompose` clustered the raw tuples themselves (comparing
    ``(text, logprob)`` tuples for equality instead of the text, silently discarding every logprob),
    and :meth:`LLMUncertainty.assess_claims` handed a raw tuple straight to string-only claim
    extractors and corroborators. :meth:`LLMUncertainty.sample` now builds ``list[Generation]`` once,
    at the boundary, so every consumer sees the same normalized shape and a text/logprob split can no
    longer happen in only some call paths.
    """

    text: str
    logprob: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text))
        if self.logprob is None:
            return
        if isinstance(self.logprob, bool) or not isinstance(self.logprob, (int, float)):
            raise TypeError(
                f"Generation.logprob must be a real number or None, got {type(self.logprob).__name__}: {self.logprob!r}"
            )
        flp = float(self.logprob)
        if not math.isfinite(flp):
            raise ValueError(f"Generation.logprob must be finite, got {flp!r}")
        object.__setattr__(self, "logprob", flp)


def _coerce_generation(raw: Any) -> Generation:
    """Normalize one raw ``generate()`` return value (a plain string, or a ``(text, logprob)`` pair)
    to a :class:`Generation`."""
    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise ValueError(
                f"generate() returned a tuple of length {len(raw)}; expected a plain string or a (text, logprob) pair."
            )
        text, logprob = raw
        return Generation(text=text, logprob=logprob)
    return Generation(text=raw)


def _require_positive_n(value: Any) -> int:
    """Validate a strictly positive integer sample count (MXR-080-0296).

    ``n or self.n``-style fallbacks silently treat ``0`` as "use the default" (``0`` is falsy in
    Python) and let a negative count fall through to ``range(negative)`` -- an empty range, i.e. zero
    samples drawn with no error at all. Every entry point that accepts a sample count funnels through
    this instead: ``0``, a negative count, and a non-integer are all rejected up front rather than
    silently reinterpreted. ``bool`` is rejected too (a subclass of ``int`` in Python; ``True``/``False``
    as a sample count is almost certainly a caller mistake, not intent).
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"n must be a positive integer, got {type(value).__name__}: {value!r}")
    if value < 1:
        raise ValueError(f"n must be a positive integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class ClaimAssessment:
    """Reliability of one claim inside a response, by cross-sample corroboration.

    ``support`` is the fraction of independent resamples whose corroboration verdict was
    :attr:`Corroboration.SUPPORTED` (in ``[0, 1]``); ``contradicted`` is the fraction verdicted
    :attr:`Corroboration.CONTRADICTED` -- a distinct, stronger-than-merely-unsupported signal: some
    resample did not just fail to back the claim, it actively disagreed (MXR-080-0295; mirrors
    :class:`mixle.substrate.factuality.ClaimVerdict`). ``reliable`` requires both good support AND
    zero contradiction (``contradicted == 0.0 and support >= threshold``): lexical overlap with a
    resample is never enough by itself, since a negated resample can share most of a claim's
    vocabulary while disagreeing with it. A claim the model actually knows recurs, unnegated, across
    samples (high support, zero contradiction); a fabricated one appears once and vanishes (low
    support); a claim the model contradicts itself on is flagged regardless of raw overlap.
    """

    claim: str
    support: float
    contradicted: float
    reliable: bool


@dataclass(frozen=True)
class InformationAssessment:
    """UQ over the *information content* of a response: every claim scored, plus a summary.

    ``claims`` is the per-claim reliability; ``reliability`` the mean support across claims (how
    trustworthy the response's information is overall) -- ``None`` when the response had no
    extractable claims at all (empty, unparseable, or evasive) rather than a vacuous ``1.0``
    (MXR-080-0295; mirrors :attr:`mixle.substrate.factuality.FactualityReceipt.grounded_fraction`):
    there is nothing that was checked, so this is UNASSESSED, not "everything checked out."
    ``fabricated`` is the claims below threshold (likely hallucinated).
    """

    claims: list[ClaimAssessment]
    reliability: float | None

    @property
    def fabricated(self) -> list[ClaimAssessment]:
        """Return claims assessed as unreliable."""
        return [c for c in self.claims if not c.reliable]

    def is_reliable(self, threshold: float = 1.0) -> bool:
        """True iff overall reliability meets ``threshold`` (default 1.0: every claim must be reliable).

        Fails closed when there is nothing to assess: a response with no extractable claims is never
        reported as "reliable", regardless of ``threshold`` -- mirrors
        :meth:`mixle.substrate.factuality.FactualityReceipt.is_grounded`.
        """
        return self.reliability is not None and self.reliability >= threshold


@dataclass(frozen=True)
class FactualityModel:
    """A fitted map from a per-prompt uncertainty signal to a *calibrated* ``P(answer is correct)``.

    The signal (self-consistency, a token likelihood, ...) is only a raw number; the calibrator turns
    it into a genuine probability of the *information* being correct, learned against labeled facts.
    ``discrimination`` (RESUBSTITUTION AUC on the fit rows, tie-correct -- see :func:`_auc`;
    "held-out" was a false label, STAT-RR23-07) reports how much
    the signal actually knew about correctness -- ~0.5 means the signal was unrelated to truth, no
    matter how confident it looked.
    """

    calibrator: ProbabilityCalibrator
    signal: Callable[[str], float]
    discrimination: float

    def probability(self, prompt: str) -> float:
        """Calibrated probability that the answer's information is correct."""
        return float(self.calibrator.predict([float(self.signal(prompt))])[0])


# -- selective-risk calibration: a finite-sample (alpha, delta)-PAC threshold, not a same-sample
#    point estimate (MXR-080-0293) -----------------------------------------------------------------
#
# This solves a different statistical task from mixle.inference.conformal's split-conformal machinery
# (interval/set coverage: "does the emitted region contain the truth", an unconditional/marginal
# guarantee over every test point) -- this is SELECTIVE classification / risk control: "given that we
# chose to answer (confidence >= tau), is the answer correct", a guarantee CONDITIONAL on a
# data-dependent selection event. Marginal coverage says nothing about that conditional rate, so
# mixle.inference.conformal.conformal_label_threshold is not a drop-in fit here; this is instead the
# standard construction for exactly this problem (Geifman & El-Yaniv 2017 "Selective Classification for
# Deep Neural Networks"; Angelopoulos et al. 2021 "Learn then Test"), self-contained in this module the
# same way MXR-080-0295's corroboration heuristic above is.


def _clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided ``(1 - delta)``-confidence upper bound on a Binomial rate from ``k`` successes
    (here: selective errors) in ``n`` trials (Clopper & Pearson 1934): ``Beta^-1(1 - delta; k+1, n-k)``.

    An EXACT tail bound, not an asymptotic approximation (e.g. Hoeffding) -- which is what keeps
    :func:`_selective_risk_threshold` usable at the calibration-set sizes this module actually sees,
    especially in the near-zero-error regime a well-behaved confidence signal produces at high
    thresholds. ``k >= n`` (every answer in the bucket was wrong) returns 1.0: no finite sample rules
    out a true 100% error rate.
    """
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, k + 1, n - k))


# The candidate thresholds: 1001 evenly spaced values, fixed here in code before any data exists.
# Candidates must NOT come from the calibration sample itself (the earlier revision tested every
# unique observed confidence): a Bonferroni union bound is over a pre-specified hypothesis family,
# and deriving the family from the same sample that tests it re-opens the selection-effect hole the
# correction exists to close. The price is that the threshold is quantized to steps of 0.001.
_CANDIDATE_GRID = np.linspace(0.0, 1.0, 1001)


def _selective_risk_threshold(confs: np.ndarray, errs: np.ndarray, *, alpha: float, delta: float) -> float:
    """The smallest confidence threshold whose TRUE selective risk is ``<= alpha`` with probability
    ``>= 1 - delta`` (MXR-080-0293) -- see :meth:`LLMUncertainty.calibrate` for the full statement of
    the guarantee and its assumptions.

    For each candidate threshold ``tau`` in the pre-specified grid ``_CANDIDATE_GRID``, replaces the
    same-sample empirical error on ``{conf >= tau}`` with its exact Clopper-Pearson upper confidence
    bound (:func:`_clopper_pearson_upper`) at level ``1 - delta / 1001`` -- a Bonferroni correction
    across the whole grid, exactly the correction the original same-sample selection omitted, which
    is how it could approve a threshold after one lucky small sample. The correction makes the bound
    simultaneously valid for every candidate at once, so selecting the smallest ``tau`` whose bound
    clears ``alpha`` remains valid regardless of the selection rule. Candidates are scanned from
    smallest to largest (most to least inclusive answered set); the first one whose bound clears
    ``alpha`` is returned, maximizing how often the model answers subject to the risk guarantee.

    Returns ``+inf`` (refuse everything) if no threshold's bound clears ``alpha`` -- too little
    calibration data for the requested ``(alpha, delta)``, or the signal genuinely does not
    discriminate well enough, rather than deploying an uncertified threshold.
    """
    per_test_delta = delta / float(_CANDIDATE_GRID.size)
    for tau in _CANDIDATE_GRID:
        answered = confs >= tau
        n_tau = int(answered.sum())
        if n_tau == 0:
            continue
        k_tau = int(round(float(errs[answered].sum())))
        if _clopper_pearson_upper(k_tau, n_tau, per_test_delta) <= alpha:
            return float(tau)
    return float("inf")


class LLMUncertainty:
    """Calibrated uncertainty and selective prediction for any ``generate(prompt) -> str`` LLM.

    Args:
        generate: ``callable(prompt) -> str`` -- one stochastic sample from the model.
        equivalent: ``callable(a, b) -> bool`` deciding whether two answers mean the same thing
            (default exact match; pass a normalizer / embedding / entailment check for real text).
        n: default number of samples per prompt (must be a positive integer).
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
        self.n = _require_positive_n(n)
        self._threshold: float | None = None
        self._certified_policy: dict | None = None
        self._alpha: float | None = None
        self._delta: float | None = None

    def sample(self, prompt: str, n: int | None = None) -> list[Generation]:
        """Draw ``n`` stochastic responses to ``prompt``, normalized to :class:`Generation` records.

        ``generate`` may return a plain string, or a ``(text, logprob)`` pair -- the sequence
        log-probability ``log P(s)``. Normalized ONCE here, at the boundary (MXR-080-0296): every one
        of :meth:`assess` / :meth:`decompose` / :meth:`assess_claims` consumes the same
        ``list[Generation]`` shape rather than each re-deriving (or failing to re-derive) whether
        ``generate`` returned tuples. ``generate`` must return the same shape on every call (plain
        text throughout, or ``(text, logprob)`` throughout) -- a generator that mixes the two raises
        rather than silently letting the unrecognized shape through unflagged. ``n`` (explicit or the
        constructor default) must be a positive integer -- see :func:`_require_positive_n`.
        """
        count = _require_positive_n(n) if n is not None else self.n
        samples = [_coerce_generation(self.generate(prompt)) for _ in range(count)]
        has_logprob = {g.logprob is not None for g in samples}
        if len(has_logprob) > 1:
            raise ValueError("generate() must consistently return plain text or (text, logprob) pairs, not a mix.")
        return samples

    @staticmethod
    def _unzip(samples: list[Generation]) -> tuple[list[str], np.ndarray | None]:
        """Split normalized :class:`Generation` records into an aligned text list + logprob array
        (``None`` when the generator did not supply log-probabilities -- :meth:`sample` already
        guarantees every record in ``samples`` agrees on that)."""
        texts = [g.text for g in samples]
        if samples and samples[0].logprob is not None:
            return texts, np.array([g.logprob for g in samples], dtype=float)
        return texts, None

    def assess(self, prompt: str, n: int | None = None) -> LLMAssessment:
        """Sample, marginalize the string distribution over meaning classes, and report the answer.

        The reported ``confidence`` is the marginal probability of the top *meaning* (summed over its
        equivalence class of strings), and ``semantic_entropy`` the entropy of that meaning marginal
        -- not a per-string token probability.
        """
        texts, log_probs = self._unzip(self.sample(prompt, n))
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
        within-member spread. Clusters on each sample's TEXT (MXR-080-0296) -- previously this
        clustered whatever :meth:`sample` returned raw, so a ``(text, logprob)`` generator got its
        tuples compared for equality instead of their text, and every logprob was silently discarded.
        """
        members = [[g.text for g in self.sample(p, n)] for p in prompts]
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
        corroborates: Callable[[str, str], Corroboration | bool] | None = None,
        n: int | None = None,
        threshold: float = 0.5,
    ) -> InformationAssessment:
        """Score the reliability of each *claim* in the response by cross-sample corroboration.

        Finer-grained than :meth:`assess`: a response can be internally consistent (low semantic
        entropy) yet contain one fabricated fact. This decomposes the response into claims and checks
        each *unit of information* separately -- a claim the model knows recurs, unnegated, across
        independent resamples; a hallucinated one appears once; a claim a resample actively disagrees
        with is flagged regardless of how much vocabulary it shares with that resample. This is UQ *on
        the information in what is said*, not just on the answer as a whole.

        Args:
            prompt: the query.
            extract: ``response -> [claim, ...]`` (default :func:`sentence_claims`).
            corroborates: ``(other_sample, claim) -> Corroboration`` -- does a resample support the
                claim? (default: :func:`information_corroborator`'s overlap as candidacy, gated by a
                negation/polarity check -- see :func:`_default_claim_corroborator`; pass an
                entailment/NLI check for real text.) A plain ``bool``-returning callable is still
                accepted for compatibility: ``True`` -> SUPPORTED, ``False`` -> UNVERIFIED, never
                CONTRADICTED (a bare bool cannot express disagreement).
            n: number of samples (the first is the response scored; the rest corroborate). Must be a
                positive integer -- see :func:`_require_positive_n`.
            threshold: support below which a claim is flagged as unreliable/fabricated.
        """
        extract = extract or sentence_claims
        samples = [g.text for g in self.sample(prompt, n)]
        if len(samples) < 2:
            samples = samples + [g.text for g in self.sample(prompt, 2 - len(samples))]
        primary, others = samples[0], samples[1:]
        # default corroboration weights words by information content over the drawn samples (as
        # candidacy only), then gates on a negation/polarity check -- see _default_claim_corroborator.
        corr = corroborates or _default_claim_corroborator(samples)
        claims = list(extract(primary))
        assessed: list[ClaimAssessment] = []
        for claim in claims:
            if others:
                verdicts = [_coerce_verdict(corr(s, claim)) for s in others]
                support = sum(1 for v in verdicts if v is Corroboration.SUPPORTED) / len(verdicts)
                contradicted = sum(1 for v in verdicts if v is Corroboration.CONTRADICTED) / len(verdicts)
            else:
                # Unreachable in practice: the top-up above guarantees len(others) >= 1 whenever
                # sample() drew at least 1 record (now always true -- see _require_positive_n). Fails
                # closed rather than assuming "reliable" if it is ever reached some other way.
                support, contradicted = 0.0, 0.0
            reliable = contradicted == 0.0 and support >= threshold
            assessed.append(ClaimAssessment(claim, support, contradicted, reliable))
        reliability = float(np.mean([c.support for c in assessed])) if assessed else None
        return InformationAssessment(assessed, reliability)

    # -- selective answer-or-abstain -----------------------------------------------------------
    def calibrate(
        self,
        examples: Sequence[tuple[str, Any]],
        *,
        correct: Callable[[Any, Any], bool] | None = None,
        alpha: float = 0.1,
        delta: float = 0.05,
        n: int | None = None,
        policy_token: str | None = None,
    ) -> LLMUncertainty:
        """Calibrate a selective-risk threshold with a finite-sample ``(alpha, delta)``-PAC guarantee.

        For each labeled example, the model's answer (majority meaning-cluster) and its confidence are
        computed; ``correct(answer, gold)`` (default the ``equivalent`` relation) marks it right or
        wrong. The threshold is chosen by :func:`_selective_risk_threshold` (MXR-080-0293): the
        smallest confidence ``tau`` (the most inclusive answered set) from a pre-specified 1001-point
        candidate grid whose exact Clopper-Pearson upper confidence bound on the selective error rate
        -- Bonferroni-corrected across the whole grid -- is ``<= alpha``.

        Statistical guarantee and assumptions: assuming the calibration ``(prompt, gold)`` examples and
        future queries are i.i.d. draws from the same distribution, AND the entire stochastic
        serving policy -- the generator's conditional output law, the equivalence relation's
        behavior, and the per-prompt sample count -- remains BEHAVIORALLY IDENTICAL to the
        calibrated one, with probability ``>= 1 - delta`` over the randomness of the calibration
        set, the deployed threshold's TRUE selective risk (``P(wrong | confidence >= threshold)``
        on a fresh query) is ``<= alpha``. The behavioral condition is load-bearing and only
        PARTLY enforceable: :meth:`answer` refuses rebound callables and sample-count overrides
        (STAT-RR21-17), but IN-PLACE state mutation inside the same callable passes any identity
        check -- an exact-wheel probe that flipped ``generator.mode`` served selective risk 1.0
        under a retained 0.10 certificate (STAT-RR22-09). Pin the model identifier, prompt
        template, decoding parameters, and equivalence version, pass them as ``policy_token``,
        and recalibrate whenever any of them changes; the token is stored on the certificate and
        echoed so serving infrastructure can compare it against the live deployment. This is a proper
        finite-sample ``(alpha, delta)``-PAC guarantee (Geifman & El-Yaniv 2017 "Selective
        Classification for Deep Neural Networks"; Angelopoulos et al. 2021 "Learn then Test") -- NOT a
        same-sample point estimate. The previous implementation picked the smallest threshold whose
        SAME-SAMPLE empirical error happened to be ``<= alpha``, with no correction for either (a) the
        calibration set being finite, or (b) implicitly searching over every candidate threshold on
        that same data -- so a small calibration set could approve a threshold after one lucky correct
        example, with no relationship to the advertised ``1 - alpha`` guarantee. ``delta`` is the
        honest second parameter that same-sample selection omitted entirely: no finite calibration set
        can certify a threshold with zero failure probability.

        If no threshold's bound clears ``alpha`` (too little calibration data for the requested
        ``alpha``/``delta``), the threshold is set just above the maximum observed confidence, so
        :meth:`answer` abstains on everything rather than deploying an uncertified threshold.

        Args:
            examples: labeled ``(prompt, gold_answer)`` pairs to calibrate against; must be non-empty.
            correct: ``(answer, gold) -> bool`` (default the ``equivalent`` relation). The return
                value must be a real Boolean or an exact 0/1 integer; anything else raises rather
                than being truthiness-coerced (``bool("false")`` is ``True``).
            alpha: target selective risk (miscoverage), in the open interval ``(0.0, 1.0)`` -- the
                guarantee is on the TRUE risk, not this calibration set's empirical risk.
            delta: failure probability of the guarantee itself, in the open interval ``(0.0, 1.0)`` --
                the guarantee holds with probability ``>= 1 - delta`` over the random calibration set.
            n: samples per prompt (passed to :meth:`assess`).
        """
        if not 0.0 < alpha < 1.0:
            # D-0143: the advertised domain is the open interval for BOTH selective-risk
            # implementations -- risk control is unachievable at 0 and vacuous at 1.
            raise ValueError(f"alpha must be in the open interval (0.0, 1.0), got {alpha!r}.")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in the open interval (0.0, 1.0), got {delta!r}.")
        examples = list(examples)
        if not examples:
            raise ValueError("calibrate() requires at least one labeled example.")
        corr = correct or self.equivalent or (lambda a, b: a == b)
        confs, errs = [], []
        for prompt, gold in examples:
            a = self.assess(prompt, n)
            confs.append(a.confidence)
            verdict = corr(a.answer, gold)
            # STAT-NEW3: bool("false") is True, so a callback returning the STRING "false" counted
            # 200 wrong answers as correct and certified threshold 0.0. Correctness evidence is the
            # foundation the (alpha, delta) guarantee stands on -- only booleans and exact 0/1
            # integers are accepted, matching D-0143 and the task-route contract.
            if isinstance(verdict, (bool, np.bool_)):
                wrong = not bool(verdict)
            elif isinstance(verdict, (int, np.integer)) and int(verdict) in (0, 1):
                wrong = int(verdict) == 0
            else:
                raise ValueError(
                    f"correct(answer, gold) must return a bool or 0/1 integer, got {verdict!r} "
                    f"of type {type(verdict).__name__}"
                )
            errs.append(1.0 if wrong else 0.0)
        confs_arr = np.asarray(confs, dtype=float)
        errs_arr = np.asarray(errs, dtype=float)
        self._threshold = _selective_risk_threshold(confs_arr, errs_arr, alpha=alpha, delta=delta)
        self._alpha = float(alpha)
        self._delta = float(delta)
        # STAT-RR21-17: the PAC theorem certifies ONE fixed randomized prediction policy -- the
        # generator, the answer-equivalence relation, and the per-prompt sample count together.
        # The threshold used to survive any of them changing: answer(prompt, n=1) on an n=2
        # certificate answered everything at selective risk 0.6015 against the certified 0.10,
        # mutating `equivalent` reached 0.604, and swapping `generate` reached 1.0 -- all while
        # answer() still advertised the guarantee. The certificate now names its policy and
        # answer() refuses to serve under any other.
        self._certified_policy = {
            "n": _require_positive_n(n) if n is not None else self.n,
            "generate": self.generate,
            "equivalent": self.equivalent,
            # STAT-RR22-09: identity binds the OBJECT, not its behavior; the caller-supplied
            # token names the model/prompt/decoding/equivalence VERSION the certificate covers.
            "policy_token": None if policy_token is None else str(policy_token),
        }
        return self

    def certified_policy_token(self) -> str | None:
        """The caller-pinned policy version the certificate covers, or None if none was supplied.

        STAT-RR22-09: callable identity cannot bind behavior; this token is the caller's own
        record of the model/prompt/decoding/equivalence version, stored at calibration for
        infrastructure-level comparison against the live deployment.
        """
        certified = getattr(self, "_certified_policy", None)
        return None if certified is None else certified.get("policy_token")

    def answer(self, prompt: str, n: int | None = None) -> LLMAssessment | None:
        """Answer ``prompt`` if confident enough, else ``None`` (abstain).

        Requires a prior :meth:`calibrate`. Returns the :class:`LLMAssessment` when
        ``confidence >= threshold`` (so the answer meets the calibrated ``(alpha, delta)``
        selective-risk guarantee), else ``None``.

        The guarantee covers exactly the CALIBRATED policy -- generator, equivalence relation,
        and per-prompt sample count -- so serving under any other is refused rather than silently
        keeping the certificate (STAT-RR21-17: an ``n=1`` override on an ``n=2`` certificate
        answered 20,000/20,000 fresh queries at selective risk 0.6015 against the certified 0.10;
        a mutated equivalence reached 0.604 and a swapped generator 1.0). WHAT THE REFUSALS CAN
        AND CANNOT CATCH (STAT-RR22-09): rebinding and count drift are caught by identity checks;
        IN-PLACE mutation of state inside the same callable is not detectable from here -- a
        flipped ``generator.mode`` passed the guard and served risk 1.0 under the retained 0.10
        certificate -- so the certificate is conditional on the serving policy remaining
        behaviorally identical, and ``certified_policy_token()`` exposes the caller-pinned
        model/prompt/decoding/version token for infrastructure-level comparison. To serve a
        different policy, calibrate it: build the object with that generator/equivalence/n and
        call :meth:`calibrate` again. :meth:`assess` remains available at any ``n`` -- it claims
        no certificate.
        """
        if self._threshold is None:
            raise RuntimeError("call calibrate(...) before answer()")
        certified = getattr(self, "_certified_policy", None)
        if certified is not None:
            requested_n = _require_positive_n(n) if n is not None else self.n
            if requested_n != certified["n"]:
                raise ValueError(
                    f"answer(n={requested_n}) does not match the certified sample count "
                    f"n={certified['n']}: the (alpha, delta) selective-risk certificate covers "
                    "only the calibrated policy (STAT-RR21-17 -- an n=1 override served 60.15% "
                    "risk against a 10% certificate). Serve at the certified n, or recalibrate."
                )
            if self.generate is not certified["generate"] or self.equivalent is not certified["equivalent"]:
                raise ValueError(
                    "the generator or equivalence relation changed after calibration: the "
                    "(alpha, delta) certificate covers only the calibrated policy "
                    "(STAT-RR21-17 -- a swapped generator served 100% risk under a 10% "
                    "certificate). Recalibrate with the current policy before serving."
                )
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
        being right. ``discrimination`` is the RESUBSTITUTION (fit-sample) tie-corrected AUC of
        signal vs correctness -- the same rows that fit the calibrator, so it is optimistic and
        unstable at small n (STAT-RR23-07: a 20-row independent-null draw read 0.96 there while
        100,000 fresh rows measured 0.4997); measure discrimination you intend to report on rows
        this fit never saw. ``correct`` must return a REAL Boolean or exact 0/1 integer -- the
        correctness verdicts are the calibration TARGET, and truthiness coercion let a callback
        returning the string "false" turn an always-wrong generator (actual correctness 0.0) into
        ``probability(...) == 1.0``; the same refusal already guards :meth:`calibrate`.

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
            verdict = corr(a.answer, gold)
            # STAT-RR23-07 (the STAT-NEW3 class, third instance): bool("false") is True, so a
            # string-returning oracle fabricated the calibration target -- every wrong answer
            # counted correct and an always-wrong generator calibrated to probability 1.0.
            if isinstance(verdict, (bool, np.bool_)):
                is_correct = bool(verdict)
            elif isinstance(verdict, (int, np.integer)) and int(verdict) in (0, 1):
                is_correct = int(verdict) == 1
            else:
                raise ValueError(
                    f"correct(answer, gold) must return a bool or 0/1 integer, got {verdict!r} "
                    f"of type {type(verdict).__name__}"
                )
            outcomes.append(1.0 if is_correct else 0.0)
        scores_arr = np.asarray(scores, dtype=float)
        outcomes_arr = np.asarray(outcomes, dtype=float)
        calibrator = calibrate_probabilities(scores_arr, outcomes_arr, method=method)
        sig = signal if signal is not None else (lambda p: self.assess(p, n).confidence)
        return FactualityModel(calibrator, sig, _auc(scores_arr, outcomes_arr))
