"""F7: the pilot ladder -- ``capacity_ladder``-style GO/NO-GO rung staging for the frontier trainer.

**Scope, read this first.** The roadmap card this module implements (F7, "(L, calendar-dominant)") names
REAL rungs: 1B/8k-context/8-GPU, then 8B/128k/256-GPU, then 8B/10M-context/1000-GPU, then a headline run
whose size/context/MoE-vs-dense choice is made by F5's scaling-law fit under a fixed compute box. None of
that hardware exists in this environment, and pretending otherwise would be dishonest. What this module
builds instead -- and what is actually real, tested, and runs today -- is the ORCHESTRATION MACHINERY: a
:class:`Rung` ladder walker that runs each rung's training, collects the roadmap's named artifacts (MFU,
loss curve, forgetting curve, decision-journal entry), and GATES progression to the next rung on a real
GO/NO-GO check against that rung's measured training-health receipts. It is exercised here at a tiny
simulated scale (a handful of rungs of increasing but still laptop-sized model/data size) standing in for
the real 1B -> 8B -> headline progression -- the gate/journal/artifact-collection *logic* is identical to
what a real run would drive, only the scale is fake.

**Which of the roadmap's staged sub-pieces this module can actually reach, from this worktree's base**
(``release/0.7.0``) **and which it cannot:**

* **F4** (training-health + MFU receipts, PR #147) -- merged into this worktree's base. Wired in for real
  via :class:`mixle.utils.parallel.training_health.TrainingHealthMonitor`; every rung's MFU/loss-curve/
  restart-continuity artifacts come from that exact machinery, not a re-implementation.
* **F1** (TP/PP/CP atop FSDP2, PR #171) -- merged into this worktree's base. Its own test suite already
  exercises the parallelism mechanics; this orchestrator does not re-simulate TP/PP/CP sharding (that
  would just be re-testing F1), it notes F1 as an assumed-healthy dependency of the rung-i shakeout.
* **H2** (MoE block + dense->MoE upcycling, PR #167) -- merged into this worktree's base. Wired in for
  real: rung ii calls :func:`mixle.models.moe.upcycle_dense_to_moe` on the rung's trained dense block and
  records the measured MoE-vs-dense output-gap receipt the roadmap calls "the H2 MoE-vs-dense decision".
* **F9** (muP width transfer, PR #155) -- merged into this worktree's base. Wired in for real: when a rung
  opts in, its learning rate is *transferred* (not re-tuned) from a stated base width via
  :func:`mixle.models.mup.transfer_lr`, and the transfer receipt is recorded.
* **F2** (fault-tolerant checkpointing, roadmap card F2) -- lives on ``origin/fault-tolerant-checkpointing``,
  which is NOT an ancestor of this worktree's base and diverges from it (see that branch's own diff stat).
  Pulling its module across would mean vendoring code this PR did not review line-by-line just to claim a
  checkbox; instead this module documents F2 as unreachable-here and raises :class:`NotImplementedError`
  if a caller explicitly opts a rung into fault-injection (``Rung.exercise_fault_tolerance=True``) rather
  than silently skipping it.
* **F5** (scaling-law fits, roadmap card F5) -- same story, lives on ``origin/scaling-law-fits``, not
  reachable from this base. Rungs iii/iv nominally depend on F5 to *choose* the next rung's size/context;
  here that choice is a documented manual stand-in, and ``Rung.exercise_scaling_law_fit=True`` raises
  :class:`NotImplementedError` rather than fabricating a fit.
* **E7** (referee evaluation suite) / **E8** (a later long-context item) -- neither exists yet anywhere in
  this repository (see ``mixle/experimental/README.md``: E7 is explicitly "later items on the same roadmap
  track and don't exist yet"). Same treatment: documented, ``NotImplementedError`` on explicit opt-in.

Every "not reachable" piece above is a named, spelled-out reason attached to the rung's artifacts
(:attr:`RungArtifacts.skipped_pieces`), never a silent no-op.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.epistemic.journal import DecisionRecord, EpistemicJournal
from mixle.epistemic.loop import EpistemicStep
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio
from mixle.utils.parallel.training_health import TrainingHealthMonitor, flop_config_from_causal_lm

try:
    import numpy as np
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "PILOT_LADDER_UNAVAILABLE_PIECES",
    "PILOT_LADDER_ASSUMED_HEALTHY_PIECES",
    "PilotLadderResult",
    "Rung",
    "RungArtifacts",
    "RungOutcome",
    "run_pilot_ladder",
]

#: roadmap sub-pieces this module cannot reach from this worktree's base, and exactly why -- see the
#: module docstring. Keyed by the name a :class:`Rung`'s ``decision_pieces`` may cite.
PILOT_LADDER_UNAVAILABLE_PIECES: dict[str, str] = {
    "F2": (
        "fault-tolerant checkpointing (roadmap F2) lives on origin/fault-tolerant-checkpointing, which is "
        "not an ancestor of this worktree's base (release/0.7.0) and diverges from "
        "it; this rung ran without fault injection."
    ),
    "F5": (
        "scaling-law fits (roadmap F5) live on origin/scaling-law-fits, not reachable from this worktree's "
        "base; this rung did not fit a scaling law and used a manually-chosen stand-in configuration "
        "instead of one F5 would have chosen."
    ),
    "E7": (
        "the E7 referee evaluation suite does not exist yet anywhere in this repository "
        "(see mixle/experimental/README.md's graduation rule); this rung ran without an E7 bake-off."
    ),
    "E8": (
        "E8 is a later long-context roadmap item that has not been built yet; this rung ran without it. "
        "(F1's TP/PP/CP, PR #171, already covers context parallelism as a separate roadmap item and is "
        "listed under PILOT_LADDER_ASSUMED_HEALTHY_PIECES below -- it is not the same thing as E8.)"
    ),
}

#: roadmap sub-pieces that ARE merged into this worktree's base but are treated as an assumed-healthy
#: dependency rather than re-simulated by this orchestrator -- their own test suites already exercise them.
PILOT_LADDER_ASSUMED_HEALTHY_PIECES: dict[str, str] = {
    "F1": (
        "TP/PP/CP atop FSDP2 (roadmap F1, PR #171) is merged into this worktree's base but is not "
        "re-simulated here; this rung assumes it healthy and relies on F1's own test suite for that."
    ),
}


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("run_pilot_ladder requires torch")


@dataclass
class Rung:
    """One pilot-ladder rung: a tiny simulated stand-in for a REAL roadmap rung's size/context/GPU count.

    ``real_target`` documents the real rung this stands in for (e.g. ``"1B params / 8k context / 8 GPUs"``)
    purely for the record -- nothing here can measure that scale, so the actual training below runs at
    ``vocab``/``d_model``/``n_layer``/``n_head``/``block`` sizes chosen to finish in seconds on a laptop.
    ``n_workers`` is a documented stand-in for the real rung's GPU count; this module does not spawn
    ``n_workers`` real processes (see the F1 note in the module docstring) but records it as part of the
    rung's identity for the decision journal.
    """

    name: str
    real_target: str
    decision_pieces: tuple[str, ...]

    # tiny simulated model shape (deliberately laptop-sized, NOT the real rung's size)
    vocab: int = 64
    d_model: int = 16
    n_layer: int = 2
    n_head: int = 2
    block: int = 8
    n_workers: int = 1

    # tiny simulated training run
    steps: int = 40
    switch_step: int | None = None  # step index where synthetic training data switches task A -> B
    batch_size: int = 8
    lr: float = 1e-2
    seed: int = 0

    # GO/NO-GO criteria
    max_final_loss: float = 3.0
    max_forgetting_gap: float | None = 1.5
    require_continuity: bool = True

    # F9 (muP transfer) opt-in
    exercise_mup_transfer: bool = False
    mup_base_width: int | None = None

    # H2 (MoE-vs-dense) opt-in
    exercise_moe_decision: bool = False
    moe_experts: int = 4
    moe_max_relative_diff: float = 1.0

    # unreachable-here pieces: explicit opt-in raises NotImplementedError rather than silently no-op-ing
    exercise_fault_tolerance: bool = False
    exercise_eval_suite: bool = False
    exercise_context_parallel: bool = False
    exercise_scaling_law_fit: bool = False


@dataclass
class RungArtifacts:
    """The roadmap's per-rung artifacts: MFU, loss curve, forgetting curve, plus this pilot's bookkeeping."""

    rung: str
    health_report: dict[str, Any]
    loss_curve: list[float]
    forgetting_curve: list[float]
    forgetting_gap: float | None
    final_loss: float
    mfu_mean: float | None
    skipped_pieces: dict[str, str] = field(default_factory=dict)
    exercised_receipts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RungOutcome:
    """One rung's full outcome: its artifacts, the GO/NO-GO verdict, why, and its journal entry."""

    artifacts: RungArtifacts
    passed: bool
    reason: str
    decision_record: DecisionRecord


@dataclass
class PilotLadderResult:
    """The whole ladder's outcome: every attempted rung, where (if anywhere) it halted, and the journal."""

    outcomes: list[RungOutcome]
    halted_at: str | None
    journal: EpistemicJournal

    def passed_rungs(self) -> list[str]:
        return [o.artifacts.rung for o in self.outcomes if o.passed]


def _check_unreachable_opt_ins(rung: Rung) -> None:
    opt_ins = {
        "exercise_fault_tolerance": "F2",
        "exercise_eval_suite": "E7",
        "exercise_context_parallel": "E8",
        "exercise_scaling_law_fit": "F5",
    }
    for attr, piece in opt_ins.items():
        if getattr(rung, attr):
            raise NotImplementedError(
                f"rung {rung.name!r} opted into {piece} ({attr}=True), which is unavailable in this "
                f"worktree: {PILOT_LADDER_UNAVAILABLE_PIECES[piece]}"
            )


def _make_batch(vocab: int, block: int, batch_size: int, task: str, gen: torch.Generator) -> tuple[Any, Any]:
    """A tiny synthetic supervised task: predict a fixed function of two token positions.

    Task ``"A"`` labels by the first two tokens, task ``"B"`` by the last two -- different enough that
    learning B can plausibly interfere with what was learned for A (the same shared attention/embedding
    weights have to serve both), which is exactly the mechanism a forgetting curve needs to have anything
    to measure.
    """
    x = torch.randint(0, vocab, (batch_size, block), generator=gen)
    if task == "A":
        y = (x[:, 0] + x[:, 1]) % vocab
    else:
        y = (x[:, -1] + x[:, -2]) % vocab
    return x, y


def _train_step(model: Any, opt: Any, x: Any, y: Any) -> tuple[float, float]:
    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)  # measure, don't clip signal
    opt.step()
    return float(loss.item()), float(grad_norm.item())


def _eval_loss(model: Any, vocab: int, block: int, batch_size: int, task: str, gen: torch.Generator) -> float:
    model.eval()
    with torch.no_grad():
        x, y = _make_batch(vocab, block, batch_size, task, gen)
        loss = torch.nn.functional.cross_entropy(model(x), y)
    model.train()
    return float(loss.item())


def _run_rung(rung: Rung, *, peak_flops_per_sec: float) -> RungArtifacts:
    """Train ``rung``'s tiny simulated model and collect its MFU / loss-curve / forgetting-curve artifacts."""
    _require_torch()
    _check_unreachable_opt_ins(rung)

    from mixle.models.transformer import build_causal_lm

    torch.manual_seed(rung.seed)
    model = build_causal_lm(
        vocab=rung.vocab, d_model=rung.d_model, n_layer=rung.n_layer, n_head=rung.n_head, block=rung.block
    )

    skipped: dict[str, str] = {}
    exercised: dict[str, Any] = {}
    lr = rung.lr

    # F9: muP width transfer -- a real transferred lr, not a re-tuned one.
    if rung.exercise_mup_transfer:
        if rung.mup_base_width is None:
            raise ValueError(f"rung {rung.name!r}: exercise_mup_transfer=True requires mup_base_width")
        from mixle.models.mup import transfer_lr

        lr = transfer_lr(rung.lr, rung.mup_base_width, rung.d_model)
        exercised["F9_mup_transfer"] = {
            "base_lr": rung.lr,
            "base_width": rung.mup_base_width,
            "target_width": rung.d_model,
            "transferred_lr": lr,
        }
    elif "F9" in rung.decision_pieces:
        skipped["F9"] = f"rung {rung.name!r} named F9 in decision_pieces but exercise_mup_transfer=False"

    # H2: MoE-vs-dense upcycling receipt -- measured, not assumed. Recorded as a decision receipt; the
    # trained model below stays dense (swapping architecture mid-optimizer-state is a separate integration
    # step once a rung is actually promoted, out of scope for this pilot's gating logic).
    if rung.exercise_moe_decision:
        from mixle.models.moe import upcycle_dense_to_moe

        _moe_block, moe_receipt = upcycle_dense_to_moe(model.blocks[0], rung.moe_experts, seed=rung.seed)
        moe_receipt["decision"] = (
            "moe" if moe_receipt["relative_output_diff"] <= rung.moe_max_relative_diff else "dense"
        )
        exercised["H2_moe_vs_dense"] = moe_receipt
    elif "H2" in rung.decision_pieces:
        skipped["H2"] = f"rung {rung.name!r} named H2 in decision_pieces but exercise_moe_decision=False"

    for piece in rung.decision_pieces:
        if piece in PILOT_LADDER_UNAVAILABLE_PIECES and piece not in skipped:
            skipped[piece] = PILOT_LADDER_UNAVAILABLE_PIECES[piece]
        elif piece in PILOT_LADDER_ASSUMED_HEALTHY_PIECES and piece not in skipped:
            skipped[piece] = PILOT_LADDER_ASSUMED_HEALTHY_PIECES[piece]

    flop_config = flop_config_from_causal_lm(model, seq_len=rung.block)
    monitor = TrainingHealthMonitor(flop_config=flop_config, peak_flops_per_sec=peak_flops_per_sec)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    switch_step = rung.switch_step if rung.switch_step is not None else rung.steps // 2
    gen_a = torch.Generator().manual_seed(rung.seed * 1000 + 1)
    gen_b = torch.Generator().manual_seed(rung.seed * 1000 + 2)
    eval_gen_a = torch.Generator().manual_seed(rung.seed * 1000 + 3)

    loss_curve: list[float] = []
    forgetting_curve: list[float] = []
    eval_every = max(1, rung.steps // 10)

    for step in range(rung.steps):
        task = "A" if step < switch_step else "B"
        gen = gen_a if task == "A" else gen_b
        x, y = _make_batch(rung.vocab, rung.block, rung.batch_size, task, gen)
        t0 = time.time()
        loss, grad_norm = _train_step(model, opt, x, y)
        dt = time.time() - t0
        monitor.observe_step(step, loss, grad_norm=grad_norm, step_time_s=dt, batch_size=rung.batch_size)
        loss_curve.append(loss)
        if step % eval_every == 0 or step == rung.steps - 1:
            forgetting_curve.append(_eval_loss(model, rung.vocab, rung.block, rung.batch_size, "A", eval_gen_a))

    final_loss = loss_curve[-1]
    forgetting_gap = (forgetting_curve[-1] - min(forgetting_curve)) if forgetting_curve else None
    health_report = monitor.report()
    mfu_mean = health_report["mfu"]["mean"]

    return RungArtifacts(
        rung=rung.name,
        health_report=health_report,
        loss_curve=loss_curve,
        forgetting_curve=forgetting_curve,
        forgetting_gap=forgetting_gap,
        final_loss=final_loss,
        mfu_mean=mfu_mean,
        skipped_pieces=skipped,
        exercised_receipts=exercised,
    )


def _gate(rung: Rung, artifacts: RungArtifacts) -> tuple[bool, str]:
    """The real GO/NO-GO check: measured training-health receipts against ``rung``'s stated criteria."""
    reasons: list[str] = []
    if artifacts.final_loss > rung.max_final_loss:
        reasons.append(f"final loss {artifacts.final_loss:.4f} > target {rung.max_final_loss}")
    if (
        rung.max_forgetting_gap is not None
        and artifacts.forgetting_gap is not None
        and artifacts.forgetting_gap > rung.max_forgetting_gap
    ):
        reasons.append(f"forgetting gap {artifacts.forgetting_gap:.4f} > allowed {rung.max_forgetting_gap}")
    if rung.require_continuity and not artifacts.health_report["restarts"]["continuity_ok"]:
        reasons.append("training-health flagged a restart discontinuity")
    passed = not reasons
    reason = "; ".join(reasons) if reasons else "all GO/NO-GO criteria met"
    return passed, reason


def _go_likelihood_fn(margin: float):
    """A logistic likelihood over ``{"go", "no_go"}`` from a real GO/NO-GO margin (positive = passing).

    This is the same "score, don't just threshold" idea :func:`mixle.epistemic.portfolio.HypothesisPortfolio.
    surprise_score` is built for: a rung that barely passes is less surprising to have passed than one that
    passes by a mile, and the journaled belief update should say so rather than collapsing straight to 0/1.
    """

    def likelihood(h: Hypothesis, _observation: Any) -> float:
        z = max(-50.0, min(50.0, 4.0 * margin))  # clamp: logistic saturates long before this, avoids overflow
        p_go = 1.0 / (1.0 + math.exp(-z))
        return p_go if h.id == "go" else (1.0 - p_go)

    return likelihood


def _journal_rung(
    journal: EpistemicJournal, rung: Rung, artifacts: RungArtifacts, passed: bool, reason: str
) -> DecisionRecord:
    """Append one real decision-journal entry: a Bayesian belief update plus the actual GO/NO-GO action."""
    hyps = (Hypothesis(id="go", payload={"rung": rung.name}), Hypothesis(id="no_go", payload={"rung": rung.name}))
    prior = HypothesisPortfolio(hyps, np.array([0.5, 0.5]), w_open=0.0)

    loss_margin = (rung.max_final_loss - artifacts.final_loss) / max(abs(rung.max_final_loss), 1e-6)
    if rung.max_forgetting_gap is not None and artifacts.forgetting_gap is not None:
        forgetting_margin = (rung.max_forgetting_gap - artifacts.forgetting_gap) / max(
            abs(rung.max_forgetting_gap), 1e-6
        )
        margin = min(loss_margin, forgetting_margin)
    else:
        margin = loss_margin

    likelihood = _go_likelihood_fn(margin)
    observation = {
        "rung": rung.name,
        "final_loss": artifacts.final_loss,
        "forgetting_gap": artifacts.forgetting_gap,
        "mfu_mean": artifacts.mfu_mean,
        "margin": margin,
    }
    surprise = prior.surprise_score(observation, likelihood)
    posterior = prior.reweight(observation, likelihood)

    step = EpistemicStep(
        observation=observation,
        portfolio_before=prior,
        portfolio_after=posterior,
        surprise=surprise,
        next_action="advance_to_next_rung" if passed else "halt_ladder",
        next_action_eig=None,
    )
    return journal.append(
        step,
        action_considered=["advance_to_next_rung", "halt_ladder"],
        rationale=reason,
        timestamp=time.time(),
    )


def run_pilot_ladder(rungs: Sequence[Rung], *, peak_flops_per_sec: float = 1.0e12) -> PilotLadderResult:
    """Run each of ``rungs`` in order, gating progression on a real GO/NO-GO check of its own artifacts.

    For every rung: train the rung's tiny simulated model, collect MFU / loss-curve / forgetting-curve
    artifacts (reusing :mod:`mixle.utils.parallel.training_health`'s exact machinery), exercise whichever
    of F9/H2 the rung opted into for real, append one Bayesian decision-journal entry
    (:class:`mixle.epistemic.journal.EpistemicJournal`) recording the belief update and the actual
    GO/NO-GO action taken, and -- this is the gate -- stop the ladder the first time a rung's measured
    receipts fail its own stated criteria. ``peak_flops_per_sec`` is an arbitrary stand-in "hardware peak"
    (no real hardware backs any MFU number this produces); it only needs to be a fixed positive constant
    for MFU to be comparable *across this ladder's own rungs*, which is all the GO/NO-GO gate uses it for.
    """
    _require_torch()
    journal = EpistemicJournal()
    outcomes: list[RungOutcome] = []
    halted_at: str | None = None

    for rung in rungs:
        artifacts = _run_rung(rung, peak_flops_per_sec=peak_flops_per_sec)
        passed, reason = _gate(rung, artifacts)
        record = _journal_rung(journal, rung, artifacts, passed, reason)
        outcomes.append(RungOutcome(artifacts=artifacts, passed=passed, reason=reason, decision_record=record))
        if not passed:
            halted_at = rung.name
            break

    return PilotLadderResult(outcomes=outcomes, halted_at=halted_at, journal=journal)
