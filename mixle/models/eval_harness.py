"""General-capability eval harness: per-checkpoint receipts + regression tracking across a checkpoint sequence.

Roadmap F10. Mirrors the ``.report()``-style receipt convention of
:class:`mixle.utils.parallel.training_health.TrainingHealthMonitor` (F4) and
:class:`mixle.evolve.population.OperatorBandit` / ``GenerationReport``: no I/O, pure in-process accounting,
JSON-serializable output.

**The eval suite is a small, honest SYNTHETIC proxy, not a real published benchmark.** It does not claim to
measure MMLU-style world knowledge or HellaSwag-style commonsense; it measures four narrow, well-defined
capability *axes* a decoder-only :class:`mixle.models.transformer.CausalLM` can plausibly discriminate on at
toy scale, each generated from a fixed, seedable procedure so scores are exactly reproducible and comparable
across checkpoints:

* **held-out perplexity** -- cross-entropy against sequences drawn from a fixed order-1 Markov chain over the
  model's vocabulary (a stand-in for "does the model fit *a* held-out next-token distribution", not any
  particular pretraining corpus).
* **modular-arithmetic reasoning** -- ``a + b = ?`` under a small modulus, a single next-token prediction; a
  minimal, unambiguous proxy for symbolic/algorithmic reasoning.
* **parity (counting) reasoning** -- the XOR parity of a random bitstring, a second algorithmic axis
  deliberately orthogonal to arithmetic (it requires tracking a running count, not a lookup table).
* **in-context induction** -- a synthetic induction-head probe (``... A B ... A -> ?``, correct answer
  ``B``): plant a bigram once in the context, then ask the model to complete it after seeing the first token
  again. This is the standard synthetic proxy for in-context learning (Olsson et al.'s induction heads).

**One command per checkpoint** (F10's acceptance bar): :func:`evaluate_checkpoint` takes a model and returns a
complete :class:`EvalReport` -- no multi-step manual orchestration.

**Regression tracking across rungs and across the J2 compression ladder**: :func:`track_regression` takes a
*sequence* of :class:`EvalReport` (successive training rungs, or successive steps of a not-yet-built J2
checkpoint-to-family compression ladder -- e.g. one report per rung of :mod:`mixle.models.qat` /
:func:`mixle.task.quantize.quantize_mlp` applied progressively) and flags any metric that moved measurably
worse than its best-so-far value, beyond a stated relative threshold.

Integration points for later roadmap items (neither is required to exist for F10 to be useful today):

* **E7 (long-context referee)** -- a not-yet-built long-context judge would slot in as one more task in
  ``_TASKS`` (or a second harness whose :class:`EvalReport` merges into this one via ``EvalReport.tasks``);
  nothing here assumes a fixed task count. ``EvalReport.metadata`` is free-form so a referee verdict can ride
  alongside these four scores without a schema change.
* **J2 (compression ladder)** -- :func:`track_regression` takes an arbitrary ordered sequence of reports, so a
  ladder of checkpoints (fp32 -> QAT int8 -> QAT int4, or successive distillation students) is scored by
  calling :func:`evaluate_checkpoint` once per rung and handing the list straight to
  :func:`track_regression`; the ``checkpoint_id`` on each report is exactly the label J2 would assign to a
  ladder rung.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "TaskResult",
    "EvalReport",
    "RegressionFlag",
    "RegressionReport",
    "evaluate_checkpoint",
    "track_regression",
    "markov_transition_matrix",
]

_MIN_VOCAB = 12  # digits 0..9 + PLUS + EQUALS for the arithmetic task
_MIN_BLOCK = 8
_PERPLEXITY_FAMILY_SEED = 20240101  # fixes *which* Markov chain is "the benchmark"; independent of the sample seed


def markov_transition_matrix(vocab: int) -> np.ndarray:
    """The fixed order-1 Markov chain the perplexity task scores against.

    Keyed only on ``vocab`` (via a module-level constant seed), not on the caller's sample ``seed`` -- the
    *benchmark distribution* is a fixed property of the eval suite (like a real held-out benchmark), while the
    per-call ``seed`` only controls which *samples* from it are drawn for a given run. This is what lets a
    training loop legitimately learn this chain (it is a fixed, nameable distribution) while individual eval
    runs still draw fresh, unseen sequences from it.
    """
    return np.random.default_rng(_PERPLEXITY_FAMILY_SEED).dirichlet(np.full(vocab, 0.3), size=vocab)


# ---------------------------------------------------------------------------
# Per-task result + the per-checkpoint receipt
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """One capability-axis score: a name, a scalar, and which direction is "better" for regression math."""

    name: str
    score: float
    higher_is_better: bool
    n_examples: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """The per-rung receipt: every task's score for one checkpoint, JSON-serializable via :meth:`report`."""

    checkpoint_id: str
    tasks: list[TaskResult]
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def scores(self) -> dict[str, float]:
        """``{task_name: score}`` -- the compact view :func:`track_regression` consumes."""
        return {t.name: t.score for t in self.tasks}

    def report(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "seed": self.seed,
            "tasks": [
                {
                    "name": t.name,
                    "score": t.score,
                    "higher_is_better": t.higher_is_better,
                    "n_examples": t.n_examples,
                    "details": t.details,
                }
                for t in self.tasks
            ],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# The synthetic task suite -- each fn takes (model, vocab, block, rng) and returns a TaskResult.
# ---------------------------------------------------------------------------


def _held_out_perplexity_task(model: Any, vocab: int, block: int, rng: Any, n_examples: int) -> TaskResult:
    """Cross-entropy / perplexity against a fixed order-1 Markov chain over the vocabulary.

    See :func:`markov_transition_matrix`: the chain itself is a fixed benchmark distribution; ``rng`` (seeded
    per call) only draws fresh sample sequences from it, so scores stay reproducible and comparable across
    checkpoints while the individual sequences scored are not literally memorized inputs.
    """
    import torch

    ctx_len = min(block, 16)
    trans = markov_transition_matrix(vocab)  # trans[i] = P(next | current=i), fixed benchmark distribution

    seqs = np.empty((n_examples, ctx_len), dtype=np.int64)
    cur = rng.integers(0, vocab, size=n_examples)
    seqs[:, 0] = cur
    for t in range(1, ctx_len):
        nxt = np.array([rng.choice(vocab, p=trans[c]) for c in cur])
        seqs[:, t] = nxt
        cur = nxt

    x = torch.as_tensor(seqs[:, :-1])
    y = torch.as_tensor(seqs[:, -1])
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
    loss_v = float(loss.item())
    ppl = float(np.exp(min(loss_v, 50.0)))  # clip before exp so a garbage model can't overflow to inf
    return TaskResult(
        name="held_out_perplexity",
        score=ppl,
        higher_is_better=False,
        n_examples=n_examples,
        details={"cross_entropy": loss_v, "context_len": ctx_len},
    )


def _arithmetic_task(model: Any, vocab: int, block: int, rng: Any, n_examples: int) -> TaskResult:
    """Modular addition ``a + b = ?`` (mod ``m``) as a single next-token prediction; accuracy is the score."""
    import torch

    m = min(vocab - 2, 10)
    plus_id, eq_id = vocab - 2, vocab - 1

    a = rng.integers(0, m, size=n_examples)
    b = rng.integers(0, m, size=n_examples)
    target = (a + b) % m

    seq = np.stack([a, np.full(n_examples, plus_id), b, np.full(n_examples, eq_id)], axis=1)
    x = torch.as_tensor(seq.astype(np.int64))
    y = torch.as_tensor(target.astype(np.int64))
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1)
    acc = float((pred == y).float().mean().item())
    return TaskResult(
        name="modular_arithmetic",
        score=acc,
        higher_is_better=True,
        n_examples=n_examples,
        details={"modulus": m, "chance_accuracy": 1.0 / m},
    )


def _parity_task(model: Any, vocab: int, block: int, rng: Any, n_examples: int) -> TaskResult:
    """XOR parity of a random bitstring as a next-token prediction; a counting axis orthogonal to arithmetic."""
    import torch

    # capped at 5 (not min(block, 10)): parity of >~6 independent bits is a well-known hard case for
    # gradient descent on standard attention architectures at this scale (a real property of the parity
    # problem, not a harness bug) -- 5 bits keeps the task genuinely non-linear (unlike 1-2 bits) while
    # staying learnable by a toy model within a reasonable training budget.
    bit_len = min(block, 5)
    bits = rng.integers(0, 2, size=(n_examples, bit_len))
    target = bits.sum(axis=1) % 2

    x = torch.as_tensor(bits.astype(np.int64))
    y = torch.as_tensor(target.astype(np.int64))
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1)
    acc = float((pred == y).float().mean().item())
    return TaskResult(
        name="parity_reasoning",
        score=acc,
        higher_is_better=True,
        n_examples=n_examples,
        details={"bit_len": bit_len, "chance_accuracy": 0.5},
    )


def _induction_task(model: Any, vocab: int, block: int, rng: Any, n_examples: int) -> TaskResult:
    """Synthetic induction-head probe: ``... A B ... A -> ?``, correct completion ``B``.

    The standard synthetic proxy for in-context learning: a bigram ``(A, B)`` is planted once early in the
    context; the context ends by repeating ``A``, and a model with induction-head-like behavior copies ``B``.
    """
    import torch

    ctx_len = min(block, 16)
    if ctx_len < 4:
        raise ValueError("induction task needs block >= 4")

    seqs = rng.integers(0, vocab, size=(n_examples, ctx_len))
    a = rng.integers(0, vocab, size=n_examples)
    b = rng.integers(0, vocab, size=n_examples)
    b = np.where(b == a, (b + 1) % vocab, b)  # A != B
    plant_pos = rng.integers(0, ctx_len - 3, size=n_examples)  # leave room for [.., A, B, .., A]
    for i in range(n_examples):
        p = int(plant_pos[i])
        seqs[i, p] = a[i]
        seqs[i, p + 1] = b[i]
        seqs[i, ctx_len - 1] = a[i]  # repeat A as the final context token

    x = torch.as_tensor(seqs.astype(np.int64))
    y = torch.as_tensor(b.astype(np.int64))
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1)
    acc = float((pred == y).float().mean().item())
    return TaskResult(
        name="in_context_induction",
        score=acc,
        higher_is_better=True,
        n_examples=n_examples,
        details={"context_len": ctx_len, "chance_accuracy": 1.0 / vocab},
    )


_TASKS = (_held_out_perplexity_task, _arithmetic_task, _parity_task, _induction_task)


# ---------------------------------------------------------------------------
# One command per checkpoint
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    model: Any,
    *,
    checkpoint_id: str = "checkpoint",
    seed: int = 0,
    n_examples: int = 256,
    metadata: dict[str, Any] | None = None,
) -> EvalReport:
    """Run the full synthetic capability suite on ``model`` and return one structured :class:`EvalReport`.

    ``model`` is a real (or toy) :class:`mixle.models.transformer.CausalLM` -- anything exposing ``.vocab``,
    ``.block``, and ``__call__(x) -> (batch, vocab)`` next-token logits works. This is the single entry point:
    no separate setup per task, no manual orchestration -- the "one command per checkpoint" F10 asks for.
    """
    if not hasattr(model, "vocab") or not hasattr(model, "block"):
        raise TypeError("evaluate_checkpoint expects a CausalLM-like model with .vocab and .block attributes")
    vocab, block = int(model.vocab), int(model.block)
    if vocab < _MIN_VOCAB:
        raise ValueError(f"eval suite needs vocab >= {_MIN_VOCAB} (got {vocab})")
    if block < _MIN_BLOCK:
        raise ValueError(f"eval suite needs block >= {_MIN_BLOCK} (got {block})")

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        rng = np.random.default_rng(seed)
        results = [task(model, vocab, block, rng, n_examples) for task in _TASKS]
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    return EvalReport(checkpoint_id=checkpoint_id, tasks=results, seed=seed, metadata=dict(metadata or {}))


# ---------------------------------------------------------------------------
# Regression tracking across a checkpoint sequence (training rungs or a J2 compression ladder)
# ---------------------------------------------------------------------------


@dataclass
class RegressionFlag:
    """One metric that regressed beyond ``threshold`` relative to its best-so-far value in the sequence."""

    task: str
    checkpoint_id: str
    checkpoint_index: int
    current_score: float
    reference_score: float
    reference_checkpoint_id: str
    reference_index: int
    relative_delta: float  # signed: negative always means "worse", regardless of task direction
    threshold: float


@dataclass
class RegressionReport:
    """The regression-tracking receipt over an ordered sequence of :class:`EvalReport`."""

    flags: list[RegressionFlag]
    n_checkpoints: int
    threshold: float
    reference: str

    @property
    def has_regressions(self) -> bool:
        return len(self.flags) > 0

    def report(self) -> dict[str, Any]:
        return {
            "n_checkpoints": self.n_checkpoints,
            "threshold": self.threshold,
            "reference": self.reference,
            "n_regressions": len(self.flags),
            "regressions": [
                {
                    "task": f.task,
                    "checkpoint_id": f.checkpoint_id,
                    "checkpoint_index": f.checkpoint_index,
                    "current_score": f.current_score,
                    "reference_score": f.reference_score,
                    "reference_checkpoint_id": f.reference_checkpoint_id,
                    "reference_index": f.reference_index,
                    "relative_delta": f.relative_delta,
                }
                for f in self.flags
            ],
        }


def track_regression(
    reports: Sequence[EvalReport],
    *,
    threshold: float = 0.05,
    reference: str = "best",
) -> RegressionReport:
    """Flag metrics that regressed by more than ``threshold`` (relative) across a sequence of checkpoints.

    ``reports`` is an ordered sequence -- successive training rungs, or successive steps of a J2 compression
    ladder. For each task, each report (from the second onward) is compared against either the best score seen
    so far in the sequence (``reference="best"``, the default -- catches slow drift, not just one-step drops)
    or the immediately-prior report (``reference="prior"`` -- catches only step-to-step drops). A task is
    flagged when it moved worse than the reference by more than ``threshold`` as a fraction of the reference's
    magnitude, direction-aware (a *drop* for accuracy-like metrics, a *rise* for perplexity-like metrics).
    """
    if reference not in ("best", "prior"):
        raise ValueError("reference must be 'best' or 'prior'")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    flags: list[RegressionFlag] = []
    # best_so_far[task] = (score, checkpoint_id, index); direction resolved from the first report's TaskResult
    best_so_far: dict[str, tuple[float, str, int]] = {}
    directions: dict[str, bool] = {}

    for idx, rep in enumerate(reports):
        for t in rep.tasks:
            directions.setdefault(t.name, t.higher_is_better)
            if t.name not in best_so_far:
                best_so_far[t.name] = (t.score, rep.checkpoint_id, idx)
                continue

            ref_score, ref_id, ref_idx = best_so_far[t.name]
            higher_is_better = directions[t.name]
            denom = abs(ref_score) if abs(ref_score) > 1e-12 else 1e-12
            raw_delta = (t.score - ref_score) / denom
            # normalize so "negative" always means worse, regardless of metric direction
            signed = raw_delta if higher_is_better else -raw_delta
            if signed < -threshold:
                flags.append(
                    RegressionFlag(
                        task=t.name,
                        checkpoint_id=rep.checkpoint_id,
                        checkpoint_index=idx,
                        current_score=t.score,
                        reference_score=ref_score,
                        reference_checkpoint_id=ref_id,
                        reference_index=ref_idx,
                        relative_delta=signed,
                        threshold=threshold,
                    )
                )

            if reference == "prior":
                best_so_far[t.name] = (t.score, rep.checkpoint_id, idx)
            else:  # "best": keep whichever of (current, reference) is better
                is_better = t.score > ref_score if higher_is_better else t.score < ref_score
                if is_better:
                    best_so_far[t.name] = (t.score, rep.checkpoint_id, idx)

    return RegressionReport(flags=flags, n_checkpoints=len(reports), threshold=threshold, reference=reference)
