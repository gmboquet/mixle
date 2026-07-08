"""F11: deployment of the family -- the trained checkpoint family served through the existing stack.

**Scope, read this first.** F11 is, per the roadmap, thin composition over machinery that already
exists and is already receipted:

* **J2** (:mod:`mixle.task.checkpoint_family_ladder`) builds the family itself: a headline causal LM
  plus a ladder of decreasing-size rungs, each with its own real eval report.
* **I1** (:mod:`mixle.models.unified_quantizer`) turns a torch model's real parameter tensors into
  "I-quantized artifacts" -- per-tensor auto-picked quantization with a measured bytes/error receipt.
* **J4** (:mod:`mixle.task.frontier_to_native`) builds the edge tier: a frontier-distilled, LNS-compressed,
  calibrated student served behind a :class:`~mixle.task.cascade.Cascade`, with its own cost/quality receipt.
* **Economics** (:mod:`mixle.task.economics`) supplies :class:`~mixle.task.economics.CostModel`, the unit
  costs a per-request dollar figure is built from.

This module does not re-derive compression, quantization, distillation, calibration, or cost
arithmetic; it takes J2's family, I1-quantizes every rung (plus the headline) into a real measured
artifact, and reports a cost/quality **frontier** across those artifacts -- the roadmap's "end-to-end
serve receipt (cost/quality frontier plot)" acceptance criterion -- next to J4's own served-cascade
receipt for the edge tier.

**A real constraint this discovered, not glossed over: two receipted "quality" axes don't share
units, so this module does not force them onto one line.** J2's family rungs are causal LMs scored by
F10's synthetic eval suite (:mod:`mixle.models.eval_harness`) -- perplexity plus three accuracy-style
tasks. J4's edge student is a distilled classifier scored by held-out label accuracy on whatever task
it was trained for. Both are real, receipted quality numbers, but they measure different capabilities
on different tasks; averaging them into one scalar would manufacture a false equivalence the
underlying receipts do not support. So :func:`deploy_family` reports two things side by side instead:
a same-axis cost/quality **frontier across the J2 family + headline** (comparable, because every point
is the SAME eval suite on the SAME task), and J4's own :class:`~mixle.task.frontier_to_native.CascadeReceipt`
as the edge tier's cost/quality trade on ITS task, unmerged. ``ServeReceipt.summary()`` prints both.

**Why this does not build a full :class:`~mixle.task.router.Router` across the whole family.**
``Router`` requires every non-final tier to expose ``decide(x)`` (a :class:`~mixle.task.calibrate.CalibratedTaskModel`
shape returning a label or ``ESCALATE``); J2's family rungs are plain causal LMs with no calibrated
decision boundary, and building one would mean inventing an eval-suite-specific classifier wrapper
this task does not ask for. J4's own 2-tier :class:`~mixle.task.cascade.Cascade` (student, calibrated;
frontier, a callable) already IS the calibrated-routing piece this module reuses unmodified via
:func:`~mixle.task.frontier_to_native.build_served_cascade` -- what's genuinely new here is turning
J2's *causal-LM* family into comparable priced artifacts, which is a quantization/costing question,
not a routing one.

**Cost model.** Every artifact's per-request cost is priced off ONE :class:`~mixle.task.economics.CostModel`,
scaled by real measured artifact bytes relative to the most expensive (headline/frontier) artifact:
``cost_per_request = cost.c_frontier * (artifact_bytes / headline_artifact_bytes)``. This is an honest,
declared proxy (inference cost tracks model size, not size-independent) rather than a literal cloud
price list -- the real number in the receipt is ``artifact_bytes`` (I1's measured quantized footprint);
the dollar figure is a transparent, reproducible scaling of it through the SAME :class:`CostModel` the
rest of the stack (J4's cascade) already uses for its own tier's cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.models.unified_quantizer import quantize_tensor
from mixle.task.checkpoint_family_ladder import FamilyLadderResult
from mixle.task.economics import CostModel
from mixle.task.frontier_to_native import CascadeReceipt

__all__ = [
    "ArtifactReceipt",
    "FrontierPoint",
    "ServeReceipt",
    "quantize_family_artifacts",
    "deploy_family",
]

# The three F10 eval tasks that are accuracy-style (higher_is_better=True, bounded roughly in [0, 1]);
# held_out_perplexity is excluded (lower_is_better, unbounded) so the frontier's quality axis is a
# single consistent direction ("higher = better") without needing to invert/rescale perplexity.
_QUALITY_TASKS = ("modular_arithmetic", "parity_reasoning", "in_context_induction")


def _quality_score(eval_report: Any) -> float:
    """Mean accuracy across :data:`_QUALITY_TASKS` -- F10's real per-task scores, not a re-derived metric."""
    scores = eval_report.scores()
    vals = [scores[t] for t in _QUALITY_TASKS if t in scores]
    if not vals:
        raise ValueError(f"eval_report for {eval_report.checkpoint_id!r} has none of {_QUALITY_TASKS}")
    return float(np.mean(vals))


@dataclass(frozen=True)
class ArtifactReceipt:
    """One model's I1-quantized deployment artifact: real measured bytes/error rolled up over every
    parameter tensor -- I1's own :class:`~mixle.models.unified_quantizer.QuantizationReceipt` per tensor,
    summed/averaged here, not re-derived."""

    name: str
    n_tensors: int
    dense_bytes: int
    quantized_bytes: int
    compression_ratio: float
    mean_reconstruction_error: float
    method_counts: dict[str, int]

    def summary(self) -> str:
        methods = ", ".join(f"{m}={n}" for m, n in sorted(self.method_counts.items()))
        return (
            f"{self.name}: {self.n_tensors} tensors, {self.dense_bytes}B dense -> {self.quantized_bytes}B "
            f"quantized ({self.compression_ratio:.2f}x), mean_recon_error={self.mean_reconstruction_error:.4g} "
            f"[{methods}]"
        )


def quantize_family_artifacts(model: Any, *, name: str, bits: int = 8, seed: int = 0) -> ArtifactReceipt:
    """I1: ``quantize_tensor(method="auto")`` over every real, non-empty parameter tensor of ``model``.

    Rolls I1's per-tensor receipts (chosen method, measured bytes, measured reconstruction error) into
    one per-model :class:`ArtifactReceipt` -- real totals over real per-tensor measurements, never a
    single assumed bits-per-parameter constant. ``seed`` is offset per tensor (``seed + i``) so the
    auto-pick bandit's tie-breaking is deterministic but not identical across tensors of the same shape.
    """
    dense_bytes = 0
    quantized_bytes = 0
    errors: list[float] = []
    method_counts: dict[str, int] = {}
    n_tensors = 0
    for i, p in enumerate(model.parameters()):
        arr = p.detach().cpu().numpy()
        if arr.size == 0:
            continue
        qt = quantize_tensor(arr, method="auto", bits=bits, seed=seed + i)
        n_tensors += 1
        dense_bytes += arr.size * 4
        quantized_bytes += qt.receipt.nbytes
        errors.append(qt.receipt.reconstruction_error)
        method_counts[qt.method] = method_counts.get(qt.method, 0) + 1
    if n_tensors == 0:
        raise ValueError(f"model {name!r} has no non-empty parameter tensors to quantize")
    return ArtifactReceipt(
        name=name,
        n_tensors=n_tensors,
        dense_bytes=dense_bytes,
        quantized_bytes=quantized_bytes,
        compression_ratio=(dense_bytes / quantized_bytes) if quantized_bytes else float("inf"),
        mean_reconstruction_error=float(np.mean(errors)),
        method_counts=method_counts,
    )


@dataclass(frozen=True)
class FrontierPoint:
    """One priced, quality-scored point on the family's cost/quality frontier."""

    name: str
    real_target: str
    artifact: ArtifactReceipt
    cost_per_request: float
    quality: float


@dataclass(frozen=True)
class ServeReceipt:
    """F11's end-to-end serve receipt: the J2-family cost/quality frontier, plus J4's own edge-tier
    served-cascade receipt reported alongside it (see the module docstring for why the two axes are
    kept separate rather than merged)."""

    points: list[FrontierPoint]
    edge_cascade: CascadeReceipt | None = field(default=None)

    def frontier_sorted_by_cost(self) -> list[FrontierPoint]:
        return sorted(self.points, key=lambda p: p.cost_per_request)

    def is_monotone_frontier(self, *, tol: float = 1e-9) -> bool:
        """Real, checkable claim: walking the family points cheapest-first, quality never DROPS below
        tolerance -- i.e. paying more for a bigger rung is never strictly worse on the eval suite."""
        ordered = self.frontier_sorted_by_cost()
        return all(b.quality >= a.quality - tol for a, b in zip(ordered, ordered[1:]))

    def frontier_plot(self, *, width: int = 40) -> str:
        """A deterministic, dependency-free ASCII cost/quality scatter (no matplotlib in this repo's
        dependency set) -- x = cost_per_request (log-scaled across the family's real span), y = quality.
        This is a real, reproducible rendering of the receipted numbers below, not decoration."""
        ordered = self.frontier_sorted_by_cost()
        costs = [p.cost_per_request for p in ordered]
        c_lo, c_hi = min(costs), max(costs)
        log_lo = np.log10(c_lo) if c_lo > 0 else 0.0
        log_hi = np.log10(c_hi) if c_hi > 0 else 0.0
        span = (log_hi - log_lo) or 1.0

        rows = []
        for p in ordered:
            log_c = np.log10(p.cost_per_request) if p.cost_per_request > 0 else log_lo
            col = int(round((log_c - log_lo) / span * (width - 1)))
            bar = [" "] * width
            bar[col] = "*"
            rows.append(f"  {''.join(bar)}  {p.name:12s} cost=${p.cost_per_request:.6f}  quality={p.quality:.3f}")
        header = f"  cost/quality frontier (x: log cost, ${c_lo:.6f}..${c_hi:.6f}; * = one family point)"
        return "\n".join([header, *rows])

    def summary(self) -> str:
        lines = ["F11 served family -- J2 family cost/quality frontier:"]
        for p in self.frontier_sorted_by_cost():
            lines.append(
                f"  {p.name:12s} ({p.real_target}): cost=${p.cost_per_request:.6f}/req quality={p.quality:.3f} "
                f"| {p.artifact.summary()}"
            )
        lines.append(f"  monotone (quality non-decreasing in cost): {self.is_monotone_frontier()}")
        lines.append("")
        lines.append(self.frontier_plot())
        if self.edge_cascade is not None:
            lines.append("")
            lines.append("J4 edge-tier served-cascade receipt (own task, own quality axis):")
            lines.append(self.edge_cascade.summary())
        return "\n".join(lines)


def deploy_family(
    family: FamilyLadderResult,
    headline_model: Any,
    *,
    edge_cascade_receipt: CascadeReceipt | None = None,
    cost: CostModel | None = None,
    bits: int = 8,
    seed: int = 0,
) -> ServeReceipt:
    """Build F11's end-to-end serve receipt from a J2 :class:`FamilyLadderResult` plus its own
    ``headline_model``.

    Every rung (and the headline) is I1-quantized into a real measured :class:`ArtifactReceipt`
    (:func:`quantize_family_artifacts`); each artifact's per-request cost is priced off ``cost``
    (a :class:`~mixle.task.economics.CostModel`, ``CostModel(c_frontier=1.0)`` by default) scaled by
    real measured bytes relative to the headline's own artifact (see module docstring); quality is
    J2's own real :class:`~mixle.models.eval_harness.EvalReport` for that rung/headline, reduced to
    :func:`_quality_score`. ``edge_cascade_receipt`` (optional) is J4's own
    :class:`~mixle.task.frontier_to_native.CascadeReceipt` for the edge tier, carried through
    unmodified and reported alongside the family frontier rather than merged into it.
    """
    cost = cost if cost is not None else CostModel(c_frontier=1.0)

    headline_artifact = quantize_family_artifacts(headline_model, name="headline", bits=bits, seed=seed)
    headline_bytes = headline_artifact.quantized_bytes
    if headline_bytes <= 0:
        raise ValueError("headline artifact has zero quantized bytes; cannot price the family relative to it")

    points = [
        FrontierPoint(
            name="headline",
            real_target="headline",
            artifact=headline_artifact,
            cost_per_request=cost.c_frontier,
            quality=_quality_score(family.headline_eval),
        )
    ]
    for i, rung in enumerate(family.rungs):
        artifact = quantize_family_artifacts(rung.model, name=rung.name, bits=bits, seed=seed + 1000 * (i + 1))
        rel_cost = cost.c_frontier * (artifact.quantized_bytes / headline_bytes)
        points.append(
            FrontierPoint(
                name=rung.name,
                real_target=rung.real_target,
                artifact=artifact,
                cost_per_request=rel_cost,
                quality=_quality_score(rung.eval_report),
            )
        )

    return ServeReceipt(points=points, edge_cascade=edge_cascade_receipt)
