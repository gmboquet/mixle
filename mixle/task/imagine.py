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


@dataclass
class CeilingReport:
    """Whether the CURRENT structural class meets ``target`` on held-out data -- the capacity
    ladder's verdict, computed once before any new structure is proposed."""

    held_out_score: float
    target: float
    met: bool


def ceiling_report(held_out_score: float, target: float) -> CeilingReport:
    """Return whether held-out score reaches the requested target."""
    return CeilingReport(held_out_score=held_out_score, target=target, met=held_out_score >= target)


@dataclass
class StructuralCandidate:
    """One proposed richer structure. ``new_information`` MUST name the specific capability the
    starting class provably lacks (e.g. "2-component mixture: represents a bimodal posterior a single
    Gaussian cannot") -- empty/``None`` means "no new information source" and the candidate is
    rejected regardless of any measured improvement."""

    name: str
    fit: Callable[[Sequence[Any]], Any]  # train data -> fitted model with .log_density / .score
    new_information: str = ""


@dataclass
class ProposalVerdict:
    """Evaluation verdict for one proposed structural candidate."""

    name: str
    accepted: bool
    train_score: float
    held_out_score: float
    reason: str = ""


@dataclass
class ImagineResult:
    """Capacity ceiling result plus candidate verdicts from structural imagination."""

    ceiling: CeilingReport
    verdicts: list[ProposalVerdict] = field(default_factory=list)
    breaks_ceiling: str | None = None  # name of the first verified candidate that reaches target, if any


def _mean_log_density(model: Any, data: Sequence[Any]) -> float:
    import numpy as np

    return float(np.mean([model.log_density(x) for x in data]))


def propose_structure(
    candidates: Sequence[StructuralCandidate],
    train: Sequence[Any],
    held_out: Sequence[Any],
    ceiling: CeilingReport,
) -> ImagineResult:
    """Fit and verify each candidate in order. A candidate is accepted only if it names a genuine
    new information source and improves held-out score over the ceiling's own held-out score
    (never train alone, since a richer family can always fit train better without a real capability
    gain). The first accepted candidate that also reaches ``ceiling.target`` breaks the ceiling."""
    result = ImagineResult(ceiling=ceiling)
    for cand in candidates:
        model = cand.fit(train)
        train_score = _mean_log_density(model, train)
        held_out_score = _mean_log_density(model, held_out)
        if not cand.new_information:
            verdict = ProposalVerdict(
                cand.name, False, train_score, held_out_score, reason="no named new information source"
            )
        elif held_out_score <= ceiling.held_out_score:
            verdict = ProposalVerdict(
                cand.name,
                False,
                train_score,
                held_out_score,
                reason="does not improve held-out over the current class (overfitting risk, not a real gain)",
            )
        else:
            verdict = ProposalVerdict(cand.name, True, train_score, held_out_score)
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
