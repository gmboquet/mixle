"""``CalibratedGenerator`` -- certified selective generation with abstention.

The generation-side sibling of :class:`~mixle.task.calibrate.CalibratedTaskModel`. That class gates
*classification*: it turns an uncalibrated softmax into conformal label sets and escalates on an
ambiguous label set. Open-ended generation has no fixed label space: candidates are newly sampled
and their identities change by prompt, so applying label-set conformal prediction to candidate slots
would not provide classification coverage.

This module instead certifies one fixed measurable outcome: whether the top-scored generated
candidate is correct when its score clears a threshold. Calibration is split into threshold
proposal and independent certification halves. Candidate thresholds are proposed on the first half;
the second half supplies simultaneous Bonferroni-corrected exact binomial upper bounds, allowing the
most permissive threshold whose accepted-error upper bound is at most ``alpha`` to be selected without
reusing its evidence. The guarantee is a confidence statement about selective risk under the usual
i.i.d./exchangeability and stable-generation assumptions, not conformal coverage of changing strings.

An uncertified prompt returns :data:`ABSTAIN` (``None``), the same sentinel as
:data:`mixle.task.calibrate.ESCALATE`, so cascades can escalate without special casing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from mixle.utils.callables import accepts_call

ABSTAIN = None  # sentinel returned when no candidate clears the calibrated threshold; equals Cascade's ESCALATE


def _derive_seed(base_seed: int, prompt: Any) -> int:
    """A stable (cross-process) per-prompt seed derived from ``base_seed`` -- unlike builtin ``hash()``,
    which is salted per-process by default, so it cannot be used to reproduce a draw across runs."""
    digest = hashlib.sha256(f"{base_seed}:{prompt!r}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _binomial_error_upper(errors: int, accepted: int, tail_probability: float) -> float:
    """One-sided Clopper-Pearson upper bound for an accepted-error rate."""
    if accepted <= 0 or errors < 0 or errors > accepted:
        raise ValueError("binomial counts must satisfy 0 <= errors <= accepted and accepted > 0")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must be in (0, 1)")
    if errors == accepted:
        return 1.0
    return float(beta_distribution.ppf(1.0 - tail_probability, errors + 1, accepted - errors))


class CalibratedGenerator:
    """Draw ``k`` candidates and serve the best under a certified selective-risk gate.

    Args:
        generate: ``generate(prompt, k) -> Sequence[candidate]`` (an ``rng`` keyword is passed if the
            callable accepts one; falls back to the two-argument form otherwise). Any generator that can
            draw ``k`` candidates for a prompt works: a wrapped :class:`~mixle.task.llm.CallableLLM`
            sampled ``k`` times, a beam, a stochastic sampler.
        score: ``score(candidate) -> float``, any mixle-scoreable model. Higher is better; the score
            need not be a calibrated probability; calibration thresholds its ordering statistic directly.
        alpha: maximum certified error rate among accepted candidates.
        k: number of candidates to draw per prompt.
        qhat: optional manually supplied acceptance threshold. This remains callable
            but has no risk certificate; only :meth:`calibrate` populates
            :attr:`risk_receipt`.
        confidence: simultaneous confidence level for the held-out risk certificate.
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
        confidence: float = 0.95,
    ) -> None:
        if not callable(generate) or not callable(score):
            raise TypeError("generate and score must be callable")
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha)
            or not 0.0 < float(alpha) < 1.0
        ):
            raise ValueError("alpha must be a finite number strictly between 0 and 1")
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be an exact positive integer")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an exact integer")
        if (
            isinstance(confidence, (bool, np.bool_))
            or not isinstance(confidence, (int, float, np.integer, np.floating))
            or not np.isfinite(confidence)
            or not 0.0 < float(confidence) < 1.0
        ):
            raise ValueError("confidence must be a finite number strictly between 0 and 1")
        if qhat is not None and (
            isinstance(qhat, (bool, np.bool_))
            or not isinstance(qhat, (int, float, np.integer, np.floating))
            or np.isnan(qhat)
        ):
            raise ValueError("qhat must be a real threshold, positive infinity, or None")
        self.generate = generate
        self.score = score
        self.alpha = float(alpha)
        self.k = int(k)
        self.qhat = None if qhat is None else float(qhat)
        self.seed = int(seed)
        self.confidence = float(confidence)
        self.risk_receipt: dict[str, Any] | None = None

    def _draw(self, prompt: Any, *, seed: int) -> list[Any]:
        rng = np.random.default_rng(seed)
        if accepts_call(self.generate, prompt, self.k, rng=rng):
            cands = self.generate(prompt, self.k, rng=rng)
        else:
            cands = self.generate(prompt, self.k)
        cands = list(cands)
        if len(cands) != self.k:
            raise ValueError(f"generate(...) must return exactly k={self.k} candidates, got {len(cands)}")
        return cands

    def _scored(self, prompt: Any, *, seed: int) -> tuple[list[Any], np.ndarray]:
        cands = self._draw(prompt, seed=seed)
        scores = np.asarray([float(self.score(c)) for c in cands], dtype=float)
        if scores.shape != (self.k,) or np.any(~np.isfinite(scores)):
            raise ValueError("candidate scores must be finite scalars with exactly one score per candidate")
        return cands, scores

    def _selection(self, prompt: Any, *, seed: int) -> tuple[Any, float]:
        cands, scores = self._scored(prompt, seed=seed)
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        # The top score is the fixed acceptance statistic. Calibration makes no
        # probabilistic claim about its scale; it only thresholds the stable ranking
        # policy and certifies the resulting accept/error event independently.
        statistic = float(scores[best])
        return cands[best], statistic

    def calibrate(
        self, prompts: Sequence[Any], is_correct: Callable[[Any, Any], bool], *, seed: int | None = None
    ) -> CalibratedGenerator:
        """Certify a held-out accepted-error threshold using an explicit correctness oracle.

        The first half proposes score-margin thresholds. The independent second
        half evaluates every proposal with simultaneous exact binomial bounds.
        If no nonempty accepted subset certifies risk ``<= alpha``, the threshold
        is ``+inf`` and serving abstains everywhere.
        """
        if not callable(is_correct):
            raise TypeError("is_correct must be callable")
        prompts = list(prompts)
        if len(prompts) < 2:
            raise ValueError("calibrate(...) needs at least two held-out prompts for proposal/certification splitting")
        if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))):
            raise ValueError("seed must be an exact integer or None")
        rng_seed = self.seed if seed is None else int(seed)
        statistics: list[float] = []
        errors: list[bool] = []
        for i, prompt in enumerate(prompts):
            candidate, statistic = self._selection(prompt, seed=_derive_seed(rng_seed, (i, prompt)))
            verdict = is_correct(prompt, candidate)
            if not isinstance(verdict, (bool, np.bool_)):
                raise TypeError("is_correct must return a boolean")
            statistics.append(statistic)
            errors.append(not bool(verdict))

        split = len(prompts) // 2
        proposal_stats = np.asarray(statistics[:split], dtype=float)
        certification_stats = np.asarray(statistics[split:], dtype=float)
        certification_errors = np.asarray(errors[split:], dtype=bool)
        thresholds = np.unique(np.concatenate((np.asarray([-np.inf]), proposal_stats)))
        per_threshold_tail = (1.0 - self.confidence) / len(thresholds)

        candidates: list[dict[str, Any]] = []
        for threshold in thresholds:
            accepted_mask = certification_stats >= threshold
            accepted = int(accepted_mask.sum())
            if accepted == 0:
                continue
            n_errors = int(certification_errors[accepted_mask].sum())
            upper = _binomial_error_upper(n_errors, accepted, per_threshold_tail)
            candidates.append(
                {
                    "threshold": float(threshold),
                    "accepted": accepted,
                    "errors": n_errors,
                    "error_upper": upper,
                }
            )
        eligible = [candidate for candidate in candidates if candidate["error_upper"] <= self.alpha]
        chosen = min(eligible, key=lambda candidate: candidate["threshold"]) if eligible else None
        self.qhat = float(chosen["threshold"]) if chosen is not None else float("inf")
        self.risk_receipt = {
            "method": "split-selective-risk/clopper-pearson-bonferroni/v1",
            "target_error": self.alpha,
            "confidence": self.confidence,
            "proposal_count": split,
            "certification_count": len(prompts) - split,
            "thresholds_tested": len(thresholds),
            "threshold": ("inf" if np.isposinf(self.qhat) else "-inf" if np.isneginf(self.qhat) else self.qhat),
            "statistic": "top_score",
            "candidate_count": self.k,
            "seed": rng_seed,
            "accepted": 0 if chosen is None else chosen["accepted"],
            "errors": 0 if chosen is None else chosen["errors"],
            "error_upper": None if chosen is None else chosen["error_upper"],
            "assumptions": [
                "calibration certification and serving cases are exchangeable",
                "candidate generation and scoring policies remain fixed after calibration",
            ],
        }
        return self

    def candidate_set(self, prompt: Any, *, seed: int | None = None) -> list[Any]:
        """Return ``[best_candidate]`` when its calibrated statistic clears the risk gate, else ``[]``."""
        if self.qhat is None:
            raise RuntimeError("call calibrate(...) (or set qhat) before candidate_set(...)")
        call_seed = _derive_seed(self.seed, prompt) if seed is None else int(seed)
        candidate, statistic = self._selection(prompt, seed=call_seed)
        return [candidate] if statistic >= self.qhat else []

    def serve(self, prompt: Any, *, seed: int | None = None) -> Any:
        """Return the best candidate only when the certified risk gate accepts, else :data:`ABSTAIN`."""
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
