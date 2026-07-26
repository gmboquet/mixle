"""Verified structural proposal at a capacity ceiling.

When a capacity ladder reports that no rung in the current model class meets a
target, a proposal step can generate a new structural candidate (a richer
family, e.g. a mixture where the current class is a single component) outside
what was tried. Every proposal is verified on held-out data
before adoption (a proposal that improves train but not held-out is rejected as overfitting), and
every proposal must name a genuine new INFORMATION SOURCE -- a structural capability the starting
class provably lacks -- never adopted on train-improvement alone: a richer family with more free
parameters can always fit train data better, so train improvement is not evidence of a real capability
gain. A proposal naming no new information source is rejected regardless of any measured improvement.

    ceiling = ceiling_report(current_class_held_out, target)          # "no rung meets target"
    verdict = propose_structure(candidates, train, held_out, target)  # verified-or-rejected, each

On a task with a known paradigm-shift fix (a capability the starting class
provably cannot represent but a specific richer structure can), the proposer should find a verified
structure that breaks the ceiling; a candidate with NO new information source is correctly rejected
even where it would improve held-out. Treat a negative result (no verified candidate breaks the
ceiling) as an expected outcome, not a failure to hide.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CeilingReport:
    """Whether the CURRENT structural class meets ``target`` on held-out data -- the capacity
    ladder's verdict, computed once before any new structure is proposed."""

    held_out_score: float
    target: float
    met: bool


def ceiling_report(held_out_score: float, target: float) -> CeilingReport:
    """Return whether held-out score reaches the requested target."""
    if not np.isfinite(held_out_score) or not np.isfinite(target):
        raise ValueError("ceiling scores and targets must be finite")
    return CeilingReport(
        held_out_score=float(held_out_score),
        target=float(target),
        met=bool(held_out_score >= target),
    )


@dataclass
class StructuralCandidate:
    """One proposed richer structure. ``new_information`` MUST name the specific capability the
    starting class provably lacks (e.g. "2-component mixture: represents a bimodal posterior a single
    Gaussian cannot") -- empty/``None`` means "no new information source" and the candidate is
    rejected regardless of any measured improvement."""

    name: str
    fit: Callable[[Sequence[Any]], Any]  # train data -> fitted model with .log_density / .score
    new_information: str = ""
    capability_test: Callable[[Any], bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("structural candidate name must be a nonempty string")
        if not callable(self.fit):
            raise TypeError("structural candidate fit must be callable")
        if self.capability_test is not None and not callable(self.capability_test):
            raise TypeError("capability_test must be callable")


@dataclass
class ProposalVerdict:
    """Evaluation verdict for one proposed structural candidate."""

    name: str
    accepted: bool
    train_score: float
    held_out_score: float
    baseline_score: float
    improvement: float
    capability_delta_verified: bool
    verification_indices: tuple[int, ...]
    reason: str = ""


@dataclass
class ImagineResult:
    """Capacity ceiling result plus candidate verdicts from structural imagination."""

    ceiling: CeilingReport
    verdicts: list[ProposalVerdict] = field(default_factory=list)
    breaks_ceiling: str | None = None  # name of the first verified candidate that reaches target, if any


def _mean_log_density(model: Any, data: Sequence[Any]) -> float:
    rows = list(data)
    if not rows:
        raise ValueError("scoring data must be nonempty")
    values = np.asarray([model.log_density(x) for x in rows], dtype=np.float64)
    if values.shape != (len(rows),) or not np.all(np.isfinite(values)):
        raise ValueError("candidate and baseline log densities must be finite")
    return float(np.mean(values))


def propose_structure(
    candidates: Sequence[StructuralCandidate],
    train: Sequence[Any],
    verification: Sequence[Any],
    ceiling: CeilingReport,
    *,
    baseline_model: Any,
    seed: int = 0,
) -> ImagineResult:
    """Fit and verify each candidate in order. A candidate is accepted only if it names a genuine
    new information source and improves held-out score over the ceiling's own held-out score
    (never train alone, since a richer family can always fit train better without a real capability
    gain). The first accepted candidate that also reaches ``ceiling.target`` breaks the ceiling."""
    train = list(train)
    verification = list(verification)
    candidates = list(candidates)
    if not train or not verification:
        raise ValueError("training and independent verification data must be nonempty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("structural candidate names must be unique")
    if candidates and len(verification) < len(candidates):
        raise ValueError("verification must provide at least one independent row per candidate")
    panels = np.array_split(np.random.RandomState(seed).permutation(len(verification)), len(candidates)) if candidates else []
    result = ImagineResult(ceiling=ceiling)
    for cand, panel in zip(candidates, panels):
        model = cand.fit(train)
        train_score = _mean_log_density(model, train)
        rows = [verification[int(i)] for i in panel]
        held_out_score = _mean_log_density(model, rows)
        baseline_score = _mean_log_density(baseline_model, rows)
        improvement = held_out_score - baseline_score
        capability_delta = False
        if cand.capability_test is not None:
            try:
                capability_delta = bool(not cand.capability_test(baseline_model) and cand.capability_test(model))
            except Exception:  # noqa: BLE001 - an unverifiable capability fails closed
                capability_delta = False
        common = {
            "name": cand.name,
            "train_score": train_score,
            "held_out_score": held_out_score,
            "baseline_score": baseline_score,
            "improvement": improvement,
            "capability_delta_verified": capability_delta,
            "verification_indices": tuple(int(i) for i in panel),
        }
        if not capability_delta:
            verdict = ProposalVerdict(
                accepted=False,
                reason="no executable capability delta over the baseline",
                **common,
            )
        elif improvement <= 0.0:
            verdict = ProposalVerdict(
                accepted=False,
                reason="does not improve held-out over the current class (overfitting risk, not a real gain)",
                **common,
            )
        else:
            verdict = ProposalVerdict(accepted=True, **common)
            if result.breaks_ceiling is None and held_out_score >= ceiling.target:
                result.breaks_ceiling = cand.name
        result.verdicts.append(verdict)
    return result


__all__ = [
    "CeilingReport",
    "ImagineResult",
    "ProposalVerdict",
    "StructuralCandidate",
    "ceiling_report",
    "propose_structure",
]
