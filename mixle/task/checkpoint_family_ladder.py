"""J2: the checkpoint -> family ladder -- iterate G3/J1 down a size ladder, receipted (roadmap J2).

**Scope, read this first.** J2 is explicitly named "thin orchestration" over machinery that is already
built: J1's unified :func:`~mixle.models.compress.compress` front door (itself wrapping G3's
:func:`~mixle.models.coarsening.coarsen` for the non-sampling/hybrid paths and the existing sampling-KD
stack for the full-data path) and F10's :func:`~mixle.models.eval_harness.evaluate_checkpoint` /
:func:`~mixle.models.eval_harness.track_regression`. This module builds neither compression nor
evaluation machinery; it drives J1 repeatedly down a sequence of decreasing target sizes (a small,
laptop-sized stand-in for the roadmap's real "70B -> 8B -> 1B -> edge" progression), collecting BOTH
J1/G3's own divergence receipts and a fresh F10 eval report at every rung, and gates each rung's eval
scores against the previous rung's (or the headline's) via F10's own :func:`track_regression` -- the same
"score, don't just threshold" GO/NO-GO shape :mod:`mixle.task.pilot_ladder` (F7) and
:mod:`mixle.task.capacity` use for their own rung ladders.

**Which of J2's named dependencies this module can actually reach, from this worktree's base**
(``origin/pilot-ladder``) **and which it cannot:**

* **J1** (unified ``compress()`` front door, PR #170) and **F10** (eval harness, PR #184) are NOT
  ancestors of ``origin/pilot-ladder`` -- both live on their own divergent branches
  (``origin/compress-front-door``, ``origin/eval-harness``) that were never merged into this worktree's
  base. Both are REQUIRED foundations for J2 (not optional opt-ins like F2/F5 were for F7), so unlike
  :mod:`mixle.task.pilot_ladder`'s treatment of its own unreachable pieces (document + raise
  ``NotImplementedError`` on opt-in), this module cannot merely skip them. Instead the exact files this
  module imports were pulled across via ``git show <branch>:<path>`` and committed alongside this module,
  byte-for-byte, from:

    - ``mixle/models/compress.py``           <- ``origin/compress-front-door`` (PR #170)
    - ``mixle/models/coarsening.py``         <- ``origin/compress-front-door`` (G3, PR #170's own dependency)
    - ``mixle/models/sigma_weighted_projection.py`` <- ``origin/compress-front-door`` (G2, transitive dependency)
    - ``mixle/models/eval_harness.py``       <- ``origin/eval-harness`` (PR #184)

  plus their own test files (``compress_test.py``, ``coarsening_test.py``,
  ``sigma_weighted_projection_test.py``, ``eval_harness_test.py``), run unmodified in this worktree to
  confirm the vendored copies are the exact, already-reviewed versions. Before vendoring, every OTHER
  transitive dependency ``compress.py`` needs (``mixle/models/moment_propagation.py``,
  ``mixle/task/acquire.py``, ``mixle/task/bandit.py``, ``mixle/task/distill_methods.py``,
  ``mixle/models/transformer.py``) was diffed against ``origin/compress-front-door``'s copies and found
  byte-identical to what already lives on ``origin/pilot-ladder`` -- so nothing else needed vendoring, and
  there is no version-skew risk between the vendored files and this branch's existing ones.
  ``mixle/models/eval_harness.py`` has no repo-internal imports beyond ``numpy`` and is entirely
  self-contained.

**A real constraint this discovered, not papered over**: :func:`~mixle.models.coarsening.coarsen`'s output
(:class:`~mixle.models.coarsening.CoarsenedLM`) may contain ``MergedBlock`` instances in place of plain
``Block``s. ``MergedBlock`` does not expose the ``.attn``/``.ln1``/``.ln2``/``.mlp`` attributes
``coarsen()``'s own internals (``_block_branch``) require -- calling ``coarsen()`` a second time on an
already-coarsened model raises ``AttributeError: 'MergedBlock' object has no attribute 'ln1'`` (confirmed
directly: see this module's own test file). So a literal CHAIN of ``compress()`` calls (rung *i*'s output
model becomes rung *i+1*'s input) only works when rung *i* used ``method="sampling_kd"`` (a fresh
architecture with plain ``Block``s) -- it is NOT generally safe for the default ``"non_sampling"``/
``"hybrid"`` methods this module defaults to. Rather than silently restricting the ladder to
``sampling_kd`` (which would abandon J2's explicit "data-free except where receipts demand
micro-calibration" requirement) or crash on the second rung, :func:`build_checkpoint_family` has every
rung ``compress()`` the ORIGINAL headline model directly, with an increasingly generous divergence
``budget``/``trust_region`` per rung. This is a real, honest limitation of a single (non-chained)
``coarsen()`` pass: from ``n_layer`` blocks it can merge at most every adjacent pair once, i.e. reach
``ceil(n_layer / 2)`` at best -- a real depth floor for one pass, not an unbounded cascade. The ladder
still walks a genuinely DECREASING (non-increasing) sequence of measured model sizes; it just cannot go
below that floor without a different method for that rung (a documented extension point, not built here).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.models.coarsening import ScaleReceipt
from mixle.models.compress import CompressionReceipt, compress
from mixle.models.eval_harness import EvalReport, RegressionFlag, evaluate_checkpoint, track_regression

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "FamilyLadderResult",
    "FamilyRung",
    "RungSpec",
    "build_checkpoint_family",
    "count_params",
]


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("build_checkpoint_family requires torch")


def count_params(model: Any) -> int:
    """Real parameter count of a torch module -- the scalar every rung's "target size" claim is measured
    against (never assumed from a nominal label)."""
    return int(sum(p.numel() for p in model.parameters()))


@dataclass
class RungSpec:
    """One size-ladder step's INPUT: a documentary label for the real rung this stands in for, plus the
    :func:`~mixle.models.compress.compress` and eval-budget knobs for that step.

    ``method`` defaults to ``"hybrid"`` -- J1's receipt-directed micro-calibration path -- per J2's
    "data-free except where receipts demand micro-calibration" requirement: every rung starts from G1/G3's
    free, data-free ``coarsen()`` result and spends a REAL but tiny calibration budget only on the stages
    G1's own closure-error receipts flag as poorly approximated (see
    :mod:`mixle.models.compress`'s ``_hybrid``). ``budget``/``trust_region`` are the lever that makes
    successive rungs SMALLER (a more generous divergence budget lets ``coarsen()`` accept more depth
    merges); callers walk them in an increasing sequence across a ``RungSpec`` list to build a decreasing
    size ladder -- this module does not choose that sequence for you (there is no honest way to invert
    "target size" to "divergence budget" without running ``coarsen()`` itself), matching G3's own
    data-free, budget-not-size-parameterized interface.
    """

    name: str
    real_target: str
    method: str = "hybrid"
    budget: float = 1.0
    trust_region: float = 1.0
    n_mc: int = 32
    sample_budget: int | None = None
    hybrid_sample_fraction: float = 0.01
    hybrid_max_stages: int = 3
    hybrid_epochs: int = 20
    hybrid_lr: float = 5e-4
    kd_epochs: int = 200
    kd_lr: float = 1e-2
    target_n_layer: int | None = None
    seed: int = 0

    # the GO/NO-GO gate for this rung: max relative regression (F10's track_regression threshold) any
    # eval task may show before this rung is flagged and the ladder halts.
    max_relative_eval_regression: float = 0.15
    regression_reference: str = "prior"  # "prior" (J2's stated criterion) or "best" (F10's stricter option)


@dataclass
class FamilyRung:
    """One size-ladder step's full receipted OUTPUT: the compressed model, J1/G3's own divergence
    receipts, F10's eval report, how many REAL calibration samples this rung spent, and whether it stayed
    within its stated eval budget."""

    name: str
    real_target: str
    model: Any
    n_params: int
    compression_ratio: float  # n_params / headline_n_params, measured (not assumed)
    compression_receipt: CompressionReceipt  # J1's own receipt (method used, quality, sample_count)
    non_sampling_receipts: dict[str, ScaleReceipt]  # G3's per-scale closure-error/KL receipts
    eval_report: EvalReport  # F10's per-checkpoint receipt
    calibration_samples_spent: int
    within_eval_budget: bool
    regression_flags: list[RegressionFlag] = field(default_factory=list)
    reason: str = ""


@dataclass
class FamilyLadderResult:
    """The whole ladder's receipted outcome: the headline's own eval report, every attempted rung, where
    (if anywhere) it halted, and the TOTAL real calibration-sample spend across the whole ladder --
    J2's "total calibration data measured and reported" acceptance criterion."""

    headline_eval: EvalReport
    headline_n_params: int
    rungs: list[FamilyRung]
    halted_at: str | None
    total_calibration_samples: int
    calibration_pool_size: int

    def passed_rungs(self) -> list[str]:
        return [r.name for r in self.rungs if r.within_eval_budget]

    def full_kd_equivalent_samples(self) -> int:
        """What full sampling-KD would have cost had EVERY rung used it: ``pool_size * n_rungs`` --
        the denominator J1's own <=1%-of-full-KD acceptance criterion is measured against, extended to a
        whole ladder rather than one call."""
        return self.calibration_pool_size * len(self.rungs)

    def total_calibration_fraction(self) -> float:
        """``total_calibration_samples / full_kd_equivalent_samples()`` -- the real fraction of a
        full-sampling-KD-every-rung ladder this run actually spent, ``0.0`` if there were no rungs."""
        denom = self.full_kd_equivalent_samples()
        return float(self.total_calibration_samples) / denom if denom > 0 else 0.0


def build_checkpoint_family(
    headline_model: Any,
    rung_specs: Sequence[RungSpec],
    *,
    calibration_data: Any,
    eval_data: Any = None,
    eval_seed: int = 0,
    eval_n_examples: int = 256,
) -> FamilyLadderResult:
    """Walk ``rung_specs`` in order, ``compress()``-ing ``headline_model`` at each rung's own divergence
    budget, collecting J1/G3's receipts plus a fresh F10 eval report, and gating progression on whether
    the rung's eval scores stayed within its stated budget of the reference report (F10's own
    :func:`~mixle.models.eval_harness.track_regression`, mirroring F7's GO/NO-GO gate one level up).

    Every rung ``compress()``-es the ORIGINAL ``headline_model`` (not the previous rung's output) -- see
    this module's docstring for why chaining is unsafe for the default ``non_sampling``/``hybrid``
    methods. The ladder still halts the first time a rung fails its own eval budget, exactly like
    :func:`mixle.task.pilot_ladder.run_pilot_ladder`.
    """
    _require_torch()
    if not rung_specs:
        raise ValueError("build_checkpoint_family requires at least one RungSpec")

    eval_data = eval_data if eval_data is not None else calibration_data
    headline_n_params = count_params(headline_model)
    headline_eval = evaluate_checkpoint(
        headline_model, checkpoint_id="headline", seed=eval_seed, n_examples=eval_n_examples
    )

    reports: list[EvalReport] = [headline_eval]
    rungs: list[FamilyRung] = []
    halted_at: str | None = None
    total_calibration_samples = 0

    for spec in rung_specs:
        compressed = compress(
            headline_model,
            method=spec.method,
            calibration_data=calibration_data,
            eval_data=eval_data,
            sample_budget=spec.sample_budget,
            budget=spec.budget,
            trust_region=spec.trust_region,
            n_mc=spec.n_mc,
            seed=spec.seed,
            kd_epochs=spec.kd_epochs,
            kd_lr=spec.kd_lr,
            hybrid_sample_fraction=spec.hybrid_sample_fraction,
            hybrid_max_stages=spec.hybrid_max_stages,
            hybrid_epochs=spec.hybrid_epochs,
            hybrid_lr=spec.hybrid_lr,
            target_n_layer=spec.target_n_layer,
        )
        n_samples = int(compressed.receipt.sample_count)
        total_calibration_samples += n_samples

        n_params = count_params(compressed.model)
        ratio = n_params / headline_n_params if headline_n_params > 0 else float("nan")

        eval_report = evaluate_checkpoint(
            compressed.model, checkpoint_id=spec.name, seed=eval_seed, n_examples=eval_n_examples
        )
        trial_reports = [*reports, eval_report]
        regression = track_regression(
            trial_reports, threshold=spec.max_relative_eval_regression, reference=spec.regression_reference
        )
        this_index = len(trial_reports) - 1
        rung_flags = [f for f in regression.flags if f.checkpoint_index == this_index]
        within_budget = not rung_flags

        if within_budget:
            reason = (
                f"all {len(eval_report.tasks)} eval tasks within "
                f"{spec.max_relative_eval_regression:.2%} of the {spec.regression_reference!r} reference"
            )
        else:
            reason = "; ".join(
                f"{f.task} regressed {-f.relative_delta:.2%} vs {f.reference_checkpoint_id} (budget {f.threshold:.2%})"
                for f in rung_flags
            )

        rungs.append(
            FamilyRung(
                name=spec.name,
                real_target=spec.real_target,
                model=compressed.model,
                n_params=n_params,
                compression_ratio=ratio,
                compression_receipt=compressed.receipt,
                non_sampling_receipts=compressed.non_sampling_receipts,
                eval_report=eval_report,
                calibration_samples_spent=n_samples,
                within_eval_budget=within_budget,
                regression_flags=rung_flags,
                reason=reason,
            )
        )
        reports.append(eval_report)

        if not within_budget:
            halted_at = spec.name
            break

    return FamilyLadderResult(
        headline_eval=headline_eval,
        headline_n_params=headline_n_params,
        rungs=rungs,
        halted_at=halted_at,
        total_calibration_samples=total_calibration_samples,
        calibration_pool_size=int(len(calibration_data)),
    )
