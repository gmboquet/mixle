"""E7: the long-context referee -- one evaluation suite every Track-E mechanism (E1's baseline and every
E2-E6 challenger) is measured against on the same terms. The graduation rule requires matched compute;
this module now rejects a comparison that has neither equal measured training FLOPs nor a shared enforced
training-compute ceiling.

``evaluate(mechanism, ...)`` drives any :class:`~mixle.experimental.context_spine.ContextMechanism` through
four kinds of controlled-dependency-distance probes at every requested range:

- **needle** -- a (key, value) pair planted once near the start; the key recurs ``distance`` tokens later
  and the mechanism must recall the associated value (classic needle-in-a-haystack fact retrieval).
- **copy** -- a purely positional dependency: the token ``distance`` steps back must be reproduced, no
  key/value indirection (isolates raw positional recall from associative recall).
- **multi-hop** -- ``hops`` independent anchor values scattered across ``[0, distance)`` must all be
  retained and combined (sum mod vocab) to answer a single probe at the end -- a dependency that cannot be
  satisfied by remembering only the most recent anchor.
- **multi-scale perplexity** -- a fixed, learnable order-1 Markov rule (a random token permutation) is
  trained and measured at each range, to see whether streaming quality degrades with total length,
  independent of any single controlled dependency.

Every controlled-dependency probe trains a fresh clone of the same initial mechanism only on the final
dependency position, then measures a paired held-out ``loss_threshold_success_rate``. The
:class:`~mixle.experimental.context_spine.ContextMechanism` protocol returns scalar loss rather than
logits, so this module does not call that threshold statistic classification accuracy.

**Calibrated forgetting curves ("does it know what it forgot?").** The mechanism's OWN per-probe loss is
its only self-reported signal (the protocol exposes nothing else). :func:`evaluate` overlays that signal
against the needle threshold-success curve and reports ``self_knowledge_correlation`` -- the correlation
between ``1 - loss_threshold_success_rate`` and probe loss across ranges.

**Compute/state protocols.** :func:`evaluate` accounts for training and evaluation tokens separately and
can enforce a shared ``suite_training_budget_flops`` across the isolated cells. :func:`comparison_table`
rejects multi-mechanism output unless seed, data protocol, ranges, vocabulary, compute contract, and state
budget match; it also rejects any result that exceeded its state budget. Merely using the same token counts
is not described as matched compute.

**Length curriculum as a bandit.** :func:`length_curriculum` (also run internally by :func:`evaluate`) uses
:class:`mixle.task.bandit.ThompsonBernoulli` (reused, not reimplemented) with one arm per length bucket in
``ranges``. Reward is the fraction of the maximum possible loss reduction achieved by one training step on
that bucket (``clip(improvement / chance_loss, 0, 1)``, so it lives in ``[0, 1]`` as ``ThompsonBernoulli``
requires), divided by that bucket's FLOP cost relative to the cheapest bucket -- literally "loss improvement
per FLOP", normalized to be dimensionless and bounded. Ultra-long buckets are additionally rationed by a
shared FLOP ledger seeded once from ``compute_budget_flops`` (split evenly across buckets): an arm whose
next pull would exceed its remaining ledger is masked out of selection for the rest of the run, so the
policy cannot simply spend the whole compute box on the longest bucket even if its posterior looks best.

**Honest scale note (see also this module's test file):** at ``distance=1e6`` a single real training run
here is computationally enormous -- ``evaluate``'s ``ranges`` default matches the roadmap card literally
(``(1e3, 1e4, 1e5, 1e6)``) and the function accepts genuinely large ranges from any caller. The test suite
that exercises this module does NOT use those literal values; it uses small stand-in ranges (documented in
``mixle/tests/long_context_eval_test.py``) so the suite runs in a few seconds while exercising the exact
same code path a caller would use at card scale.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.experimental.context_spine import ContextMechanism, train_tbptt
from mixle.task.bandit import ThompsonBernoulli

# Same "6ND" FLOPs-per-token-per-param heuristic as mixle.ppl.scaling_laws.FLOPS_PER_TOKEN_PARAM,
# duplicated (not imported) so this module doesn't import upward from mixle.ppl -- core modules
# must stay ppl -> core, never the reverse (see ppl_separation_test.py).
_FLOPS_PER_TOKEN_PARAM = 6.0

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "DEFAULT_VOCAB",
    "needle_suite",
    "copy_suite",
    "multi_hop_suite",
    "length_curriculum",
    "evaluate",
    "comparison_table",
]

DEFAULT_VOCAB = 17
"""Default alphabet size for synthetic suites when ``mechanism`` doesn't expose its own ``.vocab``."""


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("mixle.experimental.long_context_eval requires torch (mechanisms train via TBPTT).")


def _to_tensors(x: np.ndarray, y: np.ndarray) -> tuple[Any, Any]:
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


def _chunks(x: Any, y: Any, chunk_size: int) -> list[tuple[Any, Any]]:
    return [(x[:, i : i + chunk_size], y[:, i : i + chunk_size]) for i in range(0, x.shape[1], chunk_size)]


def _exact_positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}.")
    return int(value)


def _finite_nonnegative(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


def _cell_seed(seed: int, distance: int, suite_index: int) -> int:
    return int(np.random.SeedSequence([seed, distance, suite_index]).generate_state(1, dtype=np.uint32)[0])


def _fresh_mechanism(mechanism: ContextMechanism, *, seed: int) -> ContextMechanism:
    """Return an independent copy at the caller-supplied initial weights."""
    try:
        cloned = copy.deepcopy(mechanism)
    except Exception as exc:
        raise TypeError("mechanism must support deepcopy so every evaluation cell starts identically.") from exc
    torch.manual_seed(seed)
    return cloned


def _choose_chunk_size(distance: int) -> int:
    """Keep chunks small enough that streaming genuinely crosses several carried-state boundaries
    (the whole point of testing a ``ContextMechanism`` rather than a single-shot full-attention forward),
    but large enough that a range in the millions doesn't require millions of Python-level steps."""
    return max(2, min(64, distance // 4))


# ---------------------------------------------------------------------------------------------------------
# Synthetic suites: needle, copy, multi-hop -- see module docstring for what each isolates.
# ---------------------------------------------------------------------------------------------------------


def needle_suite(rng: np.random.RandomState, *, distance: int, vocab: int) -> tuple[Any, Any]:
    """(key, value) planted at positions 0/1; the key recurs at position ``distance`` and the target
    there is the value -- associative recall at a controlled range. Requires ``distance >= 2``."""
    if distance < 2:
        raise ValueError(f"needle_suite needs distance >= 2, got {distance}.")
    length = distance + 1
    x = rng.randint(0, vocab, size=(1, length))
    key = vocab - 1
    value = int(rng.randint(0, vocab - 1))
    x[0, 0] = key
    x[0, 1] = value
    x[0, distance] = key
    y = x.copy()
    y[0, distance] = value
    return _to_tensors(x, y)


def copy_suite(rng: np.random.RandomState, *, distance: int, vocab: int) -> tuple[Any, Any]:
    """Pure positional recall: the target at position ``distance`` is the token that appeared at
    position 0, with no key/value cue -- isolates raw positional memory from associative lookup."""
    if distance < 1:
        raise ValueError(f"copy_suite needs distance >= 1, got {distance}.")
    length = distance + 1
    x = rng.randint(0, vocab, size=(1, length))
    y = x.copy()
    y[0, distance] = x[0, 0]
    return _to_tensors(x, y)


def multi_hop_suite(rng: np.random.RandomState, *, distance: int, vocab: int, hops: int = 3) -> tuple[Any, Any]:
    """``hops`` anchor values scattered across ``[0, distance)``; the probe at ``distance`` must equal
    their sum mod ``vocab - 1``. Answering correctly requires retaining EVERY anchor, not just the most
    recent one -- a dependency a single-needle test can't distinguish from short-range recall."""
    if distance < 1:
        raise ValueError(f"multi_hop_suite needs distance >= 1, got {distance}.")
    hops = max(1, min(hops, distance))
    length = distance + 1
    x = rng.randint(0, vocab, size=(1, length))
    anchor_vocab = max(vocab - 1, 2)
    positions = sorted(set(int(p) for p in np.linspace(0, distance - 1, num=hops, endpoint=False)))
    values = rng.randint(0, anchor_vocab, size=len(positions))
    for pos, val in zip(positions, values):
        x[0, pos] = int(val)
    target = int(values.sum() % anchor_vocab)
    y = x.copy()
    y[0, distance] = target
    return _to_tensors(x, y)


def _markov_sequence(rng: np.random.RandomState, *, length: int, vocab: int, perm: np.ndarray) -> tuple[Any, Any]:
    """A fixed, learnable order-1 rule (``y[i] = perm[x[i-1]]``) -- used only for the multi-scale
    perplexity probe, where the point is streaming quality at scale, not a single controlled dependency."""
    x = rng.randint(0, vocab, size=(1, length))
    y = np.empty_like(x)
    y[0, 0] = x[0, 0]
    y[0, 1:] = perm[x[0, :-1]]
    return _to_tensors(x, y)


# ---------------------------------------------------------------------------------------------------------
# Train-then-probe: shared driver for needle / copy / multi-hop.
# ---------------------------------------------------------------------------------------------------------


def _train_and_probe(
    mechanism: ContextMechanism,
    opt: Any,
    suite_fn: Any,
    *,
    distance: int,
    vocab: int,
    chunk_size: int,
    n_train_steps: int,
    n_eval_trials: int,
    rng: np.random.RandomState,
    **suite_kwargs: Any,
) -> dict[str, Any]:
    distance = _exact_positive_int(distance, "distance")
    vocab = _exact_positive_int(vocab, "vocab", minimum=2)
    chunk_size = _exact_positive_int(chunk_size, "chunk_size")
    n_train_steps = _exact_positive_int(n_train_steps, "n_train_steps", minimum=0)
    n_eval_trials = _exact_positive_int(n_eval_trials, "n_eval_trials")
    chance_loss = math.log(vocab)
    threshold = 0.5 * chance_loss
    was_training = getattr(mechanism, "training", None)

    if hasattr(mechanism, "train"):
        mechanism.train()
    for _ in range(n_train_steps):
        x, y = suite_fn(rng, distance=distance, vocab=vocab, **suite_kwargs)
        state = mechanism.init_state(1)
        # Warm the prefix without an optimizer step, then train only on the controlled dependency target.
        # Otherwise identity targets at every uninteresting prefix position dominate the scientific probe.
        with torch.no_grad():
            for chunk in _chunks(x[:, :-1], y[:, :-1], chunk_size):
                state, _ = mechanism.step(state, chunk)
        state = mechanism.detach(state)
        train_tbptt(mechanism, state, [(x[:, -1:], y[:, -1:])], opt, detach_horizon=1)

    solved: list[bool] = []
    probe_losses: list[float] = []
    if hasattr(mechanism, "eval"):
        mechanism.eval()
    with torch.no_grad():
        for _ in range(n_eval_trials):
            x, y = suite_fn(rng, distance=distance, vocab=vocab, **suite_kwargs)
            state = mechanism.init_state(1)
            for chunk in _chunks(x[:, :-1], y[:, :-1], chunk_size):
                state, _ = mechanism.step(state, chunk)
            # A length-1 probe's mean loss IS the exact per-position loss at the controlled distance.
            _, probe_loss = mechanism.step(state, (x[:, -1:], y[:, -1:]))
            loss_v = float(probe_loss)
            probe_losses.append(loss_v)
            solved.append(loss_v < threshold)

    result = {
        "distance": distance,
        "metric": "loss_threshold_success_rate",
        "loss_threshold_success_rate": float(np.mean(solved)),
        "loss_threshold": threshold,
        "mean_probe_loss": float(np.mean(probe_losses)),
        "chance_loss": chance_loss,
        "training_steps": n_train_steps,
        "training_tokens": n_train_steps * (distance + 1),
        "evaluation_trials": n_eval_trials,
        "evaluation_tokens": n_eval_trials * (distance + 1),
    }
    if was_training is not None:
        mechanism.train(was_training)
    return result


def multi_scale_perplexity(
    mechanism: ContextMechanism,
    opt: Any,
    *,
    length: int,
    vocab: int,
    chunk_size: int,
    n_steps: int,
    n_eval_trials: int,
    rng: np.random.RandomState,
    perm: np.ndarray,
) -> dict[str, Any]:
    """Train on fresh Markov instances, then report loss/perplexity on separate held-out sequences."""
    length = _exact_positive_int(length, "length", minimum=2)
    vocab = _exact_positive_int(vocab, "vocab", minimum=2)
    chunk_size = _exact_positive_int(chunk_size, "chunk_size")
    n_steps = _exact_positive_int(n_steps, "n_steps", minimum=0)
    n_eval_trials = _exact_positive_int(n_eval_trials, "n_eval_trials")
    was_training = getattr(mechanism, "training", None)
    if hasattr(mechanism, "train"):
        mechanism.train()
    for _ in range(n_steps):
        x, y = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
        state = mechanism.init_state(1)
        with torch.no_grad():
            state, _ = mechanism.step(state, (x[:, :1], y[:, :1]))
        state = mechanism.detach(state)
        chunks = _chunks(x[:, 1:], y[:, 1:], chunk_size)
        train_tbptt(mechanism, state, chunks, opt, detach_horizon=len(chunks))

    losses: list[float] = []
    if hasattr(mechanism, "eval"):
        mechanism.eval()
    with torch.no_grad():
        for _ in range(n_eval_trials):
            x, y = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
            state = mechanism.init_state(1)
            state, _ = mechanism.step(state, (x[:, :1], y[:, :1]))
            trial_losses: list[float] = []
            for chunk in _chunks(x[:, 1:], y[:, 1:], chunk_size):
                state, loss = mechanism.step(state, chunk)
                trial_losses.append(float(loss))
            losses.append(float(np.mean(trial_losses)))
    mean_loss = float(np.mean(losses))
    result = {
        "length": length,
        "mean_loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 50.0))),
        "training_steps": n_steps,
        "training_tokens": n_steps * length,
        "training_scored_tokens": n_steps * (length - 1),
        "evaluation_trials": n_eval_trials,
        "evaluation_tokens": n_eval_trials * length,
        "evaluation_scored_tokens": n_eval_trials * (length - 1),
    }
    if was_training is not None:
        mechanism.train(was_training)
    return result


def _forgetting_curve(needle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    distances = [r["distance"] for r in needle_rows]
    success_rate = np.array([r["loss_threshold_success_rate"] for r in needle_rows])
    self_loss = np.array([r["mean_probe_loss"] for r in needle_rows])
    forgetting = 1.0 - success_rate
    if len(distances) >= 2 and np.std(self_loss) > 0 and np.std(forgetting) > 0:
        corr = float(np.corrcoef(forgetting, self_loss)[0, 1])
    else:
        corr = float("nan")
    return {
        "distances": distances,
        "metric": "loss_threshold_success_rate",
        "success_rate": success_rate.tolist(),
        "self_reported_loss": self_loss.tolist(),
        "self_knowledge_correlation": corr,
    }


# ---------------------------------------------------------------------------------------------------------
# Matched-FLOPs / matched-state-bytes bookkeeping.
# ---------------------------------------------------------------------------------------------------------


def _n_params(mechanism: ContextMechanism) -> int:
    if not hasattr(mechanism, "parameters"):
        return 0
    return int(sum(p.numel() for p in mechanism.parameters()))


def _flops_for(mechanism: ContextMechanism, n_tokens: int) -> float:
    """``6 * n_params * n_tokens`` -- the same dense-Transformer FLOPs approximation
    :mod:`mixle.ppl.scaling_laws` uses for training-compute allocation (Kaplan et al. 2020)."""
    return _FLOPS_PER_TOKEN_PARAM * float(_n_params(mechanism)) * float(n_tokens)


def _state_bytes(state: Any) -> int:
    """Best-effort recursive byte-count of a carried state: anything duck-typed as a tensor
    (``numel()``/``element_size()``) contributes ``numel() * element_size()``; dataclasses, dicts, lists,
    and tuples are walked; anything else contributes zero. Generic over ANY ``ContextMechanism`` state
    shape -- not hard-coded to :class:`~mixle.experimental.context_spine.SlidingWindowState`."""
    seen: set[int] = set()
    total = 0

    def walk(obj: Any) -> None:
        nonlocal total
        if obj is None or id(obj) in seen:
            return
        if hasattr(obj, "numel") and hasattr(obj, "element_size"):
            seen.add(id(obj))
            total += int(obj.numel()) * int(obj.element_size())
            return
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            seen.add(id(obj))
            for f in dataclasses.fields(obj):
                walk(getattr(obj, f.name))
            return
        if isinstance(obj, dict):
            seen.add(id(obj))
            for v in obj.values():
                walk(v)
            return
        if isinstance(obj, (list, tuple)):
            seen.add(id(obj))
            for v in obj:
                walk(v)
            return

    walk(state)
    return total


# ---------------------------------------------------------------------------------------------------------
# Length curriculum: ThompsonBernoulli bandit over length buckets.
# ---------------------------------------------------------------------------------------------------------


def length_curriculum(
    mechanism: ContextMechanism,
    opt: Any,
    ranges: tuple[int, ...],
    *,
    vocab: int,
    n_rounds: int,
    compute_budget_flops: float,
    seed: int,
    perm: np.ndarray,
) -> dict[str, Any]:
    """A :class:`~mixle.task.bandit.ThompsonBernoulli` arm per length bucket in ``ranges``.

    Reward = fraction of the maximum possible loss reduction achieved by one training step on that
    bucket, divided by the bucket's FLOP cost relative to the cheapest bucket ("loss improvement per
    FLOP", normalized to ``[0, 1]``). A shared FLOP ledger (``compute_budget_flops`` split evenly across
    buckets) additionally masks out any arm whose next pull would exceed its remaining share -- ultra-long
    buckets are rationed by construction, since each of their pulls costs proportionally more.
    """
    if not ranges:
        raise ValueError("ranges must be non-empty.")
    ranges = tuple(_exact_positive_int(value, "range", minimum=2) for value in ranges)
    vocab = _exact_positive_int(vocab, "vocab", minimum=2)
    n_rounds = _exact_positive_int(n_rounds, "n_rounds", minimum=0)
    compute_budget_flops = _finite_nonnegative(compute_budget_flops, "compute_budget_flops")
    seed = _exact_positive_int(seed, "seed", minimum=0)
    was_training = getattr(mechanism, "training", None)
    bandit = ThompsonBernoulli(len(ranges), seed=seed) if len(ranges) >= 2 else None
    single_rng = np.random.RandomState(seed)
    single_alpha = 1.0
    single_beta = 1.0
    single_pulls = 0
    ledger = np.full(len(ranges), float(compute_budget_flops) / len(ranges))
    costs = np.array([_flops_for(mechanism, r) for r in ranges])
    cost_ratio = costs / max(float(np.min(costs)), 1.0)
    rng = np.random.RandomState(seed + 1)

    pulled_lengths: list[int] = []
    for _ in range(n_rounds):
        affordable = ledger >= costs
        if not affordable.any():
            break  # the compute box is exhausted; stop rather than overspend any bucket.
        if bandit is None:
            single_rng.beta(single_alpha, single_beta)  # advance deterministically like a Thompson draw
            arm = 0
        else:
            draws = bandit.rng.beta(bandit.alpha, bandit.beta)
            draws = np.where(affordable, draws, -np.inf)
            arm = int(np.argmax(draws))
        length = int(ranges[arm])
        chunk_size = _choose_chunk_size(length)

        if hasattr(mechanism, "eval"):
            mechanism.eval()
        with torch.no_grad():
            # Hold the evaluation sample fixed across the update. Comparing two independent draws adds
            # sampling noise to the reward and can credit a harmful update merely because x2 was easier.
            x_eval, y_eval = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
            state0 = mechanism.init_state(1)
            state0, _ = mechanism.step(state0, (x_eval[:, :1], y_eval[:, :1]))
            before_chunks = _chunks(x_eval[:, 1:], y_eval[:, 1:], chunk_size)
            loss_before = 0.0
            for chunk in before_chunks:
                state0, loss = mechanism.step(state0, chunk)
                loss_before += float(loss)
            loss_before /= len(before_chunks)

        x1, y1 = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
        if hasattr(mechanism, "train"):
            mechanism.train()
        state1 = mechanism.init_state(1)
        with torch.no_grad():
            state1, _ = mechanism.step(state1, (x1[:, :1], y1[:, :1]))
        state1 = mechanism.detach(state1)
        chunks1 = _chunks(x1[:, 1:], y1[:, 1:], chunk_size)
        train_tbptt(mechanism, state1, chunks1, opt, detach_horizon=len(chunks1))

        if hasattr(mechanism, "eval"):
            mechanism.eval()
        with torch.no_grad():
            state2 = mechanism.init_state(1)
            state2, _ = mechanism.step(state2, (x_eval[:, :1], y_eval[:, :1]))
            after_chunks = _chunks(x_eval[:, 1:], y_eval[:, 1:], chunk_size)
            loss_after = 0.0
            for chunk in after_chunks:
                state2, loss = mechanism.step(state2, chunk)
                loss_after += float(loss)
            loss_after /= len(after_chunks)

        chance_loss = math.log(vocab)
        improvement = max(loss_before - loss_after, 0.0)
        normalized_improvement = min(improvement / chance_loss, 1.0)
        reward = float(np.clip(normalized_improvement / cost_ratio[arm], 0.0, 1.0))
        if bandit is None:
            single_alpha += reward
            single_beta += 1.0 - reward
            single_pulls += 1
        else:
            bandit.update(arm, reward)
        ledger[arm] -= costs[arm]
        pulled_lengths.append(length)

    result = {
        "bucket_ranges": [int(r) for r in ranges],
        "pulls": [single_pulls] if bandit is None else bandit.pulls.tolist(),
        "posterior_means": ([single_alpha / (single_alpha + single_beta)] if bandit is None else bandit.means.tolist()),
        "ledger_remaining": ledger.tolist(),
        "pulled_lengths": pulled_lengths,
        "paired_before_after": True,
        "compute_budget_flops": float(compute_budget_flops),
        "compute_flops_used": float(np.sum(np.full(len(ranges), compute_budget_flops / len(ranges)) - ledger)),
    }
    if was_training is not None:
        mechanism.train(was_training)
    return result


# ---------------------------------------------------------------------------------------------------------
# The one-command entry point.
# ---------------------------------------------------------------------------------------------------------


def evaluate(
    mechanism: ContextMechanism,
    *,
    ranges: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000),
    state_budget_bytes: float,
    seed: int,
    hops: int = 3,
    n_train_steps: int = 6,
    n_eval_trials: int = 8,
    perplexity_steps: int = 6,
    curriculum_rounds: int = 12,
    compute_budget_flops: float | None = None,
    suite_training_budget_flops: float | None = None,
) -> dict[str, Any]:
    """Run the full E7 referee suite against ``mechanism`` end-to-end. See the module docstring for what
    each piece measures and honestly claims. Requires a torch-trainable mechanism (``.parameters()``
    exposed, as every Track-E mechanism in :mod:`mixle.experimental.context_spine` is). The supplied
    mechanism is not trained or mutated: every range/suite, state probe, and curriculum receives an
    independent deep copy of its initial weights.

    ``suite_training_budget_flops`` is a total ceiling split evenly across every range/suite cell. When it
    is supplied, each cell uses as many of the requested training steps as fit within its share. The
    separate ``compute_budget_flops`` controls only the curriculum and defaults to ``20x`` one pass over
    the largest range.
    """
    _require_torch()
    if not hasattr(mechanism, "parameters"):
        raise ValueError("evaluate() requires a torch-trainable mechanism exposing .parameters().")

    if not ranges:
        raise ValueError("ranges must be non-empty.")
    ranges = tuple(_exact_positive_int(r, "range", minimum=2) for r in ranges)
    if len(set(ranges)) != len(ranges):
        raise ValueError("ranges must not contain duplicate distances.")
    state_budget_bytes = _finite_nonnegative(state_budget_bytes, "state_budget_bytes")
    seed = _exact_positive_int(seed, "seed", minimum=0)
    hops = _exact_positive_int(hops, "hops")
    n_train_steps = _exact_positive_int(n_train_steps, "n_train_steps", minimum=0)
    n_eval_trials = _exact_positive_int(n_eval_trials, "n_eval_trials")
    perplexity_steps = _exact_positive_int(perplexity_steps, "perplexity_steps", minimum=0)
    curriculum_rounds = _exact_positive_int(curriculum_rounds, "curriculum_rounds", minimum=0)
    if suite_training_budget_flops is not None:
        suite_training_budget_flops = _finite_nonnegative(
            suite_training_budget_flops,
            "suite_training_budget_flops",
            positive=True,
        )

    n_params = _n_params(mechanism)
    if n_params <= 0:
        raise ValueError("evaluate() requires a mechanism with at least one trainable parameter.")
    vocab = _exact_positive_int(getattr(mechanism, "vocab", DEFAULT_VOCAB), "mechanism.vocab", minimum=2)
    perm = np.random.RandomState(_cell_seed(seed, max(ranges), 99)).permutation(vocab)

    n_cells = 4 * len(ranges)
    cell_budget = None if suite_training_budget_flops is None else suite_training_budget_flops / n_cells

    def steps_within_budget(requested: int, model: ContextMechanism, tokens_per_step: int) -> int:
        if cell_budget is None:
            return requested
        cost = _flops_for(model, tokens_per_step)
        return min(requested, int(math.floor(cell_budget / cost))) if cost > 0 else 0

    training_flops_used = 0.0
    evaluation_flops_used = 0.0

    suites: dict[int, dict[str, Any]] = {}
    needle_rows: list[dict[str, Any]] = []
    for distance in ranges:
        chunk_size = _choose_chunk_size(distance)
        dependency_rows: list[dict[str, Any]] = []
        for suite_index, (suite_fn, suite_kwargs) in enumerate(
            ((needle_suite, {}), (copy_suite, {}), (multi_hop_suite, {"hops": hops}))
        ):
            cell_seed = _cell_seed(seed, distance, suite_index)
            cell_model = _fresh_mechanism(mechanism, seed=cell_seed)
            effective_steps = steps_within_budget(n_train_steps, cell_model, distance + 1)
            cell_opt = torch.optim.Adam(cell_model.parameters(), lr=1e-2)
            row = _train_and_probe(
                cell_model,
                cell_opt,
                suite_fn,
                distance=distance,
                vocab=vocab,
                chunk_size=chunk_size,
                n_train_steps=effective_steps,
                n_eval_trials=n_eval_trials,
                rng=np.random.RandomState(cell_seed),
                **suite_kwargs,
            )
            row["data_seed"] = cell_seed
            row["training_flops"] = _flops_for(cell_model, row["training_tokens"])
            row["evaluation_flops"] = _flops_for(cell_model, row["evaluation_tokens"])
            training_flops_used += row["training_flops"]
            evaluation_flops_used += row["evaluation_flops"]
            dependency_rows.append(row)
        needle, copy_, multi_hop = dependency_rows

        perplexity_seed = _cell_seed(seed, distance, 3)
        perplexity_model = _fresh_mechanism(mechanism, seed=perplexity_seed)
        effective_perplexity_steps = steps_within_budget(perplexity_steps, perplexity_model, distance)
        perplexity_opt = torch.optim.Adam(perplexity_model.parameters(), lr=1e-2)
        perplexity = multi_scale_perplexity(
            perplexity_model,
            perplexity_opt,
            length=distance,
            vocab=vocab,
            chunk_size=chunk_size,
            n_steps=effective_perplexity_steps,
            n_eval_trials=n_eval_trials,
            rng=np.random.RandomState(perplexity_seed),
            perm=perm,
        )
        perplexity["data_seed"] = perplexity_seed
        perplexity["training_flops"] = _flops_for(perplexity_model, perplexity["training_tokens"])
        perplexity["evaluation_flops"] = _flops_for(perplexity_model, perplexity["evaluation_tokens"])
        training_flops_used += perplexity["training_flops"]
        evaluation_flops_used += perplexity["evaluation_flops"]
        needle_rows.append(needle)
        suites[distance] = {
            "needle": needle,
            "copy": copy_,
            "multi_hop": multi_hop,
            "perplexity": perplexity,
            "training_flops": sum(row["training_flops"] for row in (needle, copy_, multi_hop, perplexity)),
            "evaluation_flops": sum(row["evaluation_flops"] for row in (needle, copy_, multi_hop, perplexity)),
        }

    forgetting_curve = _forgetting_curve(needle_rows)

    largest = max(ranges)
    chunk_size = _choose_chunk_size(largest)
    state_model = _fresh_mechanism(mechanism, seed=_cell_seed(seed, largest, 4))
    state_rng = np.random.RandomState(_cell_seed(seed, largest, 4))
    x, y = copy_suite(state_rng, distance=largest, vocab=vocab)
    state = state_model.init_state(1)
    if hasattr(state_model, "eval"):
        state_model.eval()
    with torch.no_grad():
        for chunk in _chunks(x, y, chunk_size):
            state, _ = state_model.step(state, chunk)
    state_bytes_used = _state_bytes(state)

    curriculum_model = _fresh_mechanism(mechanism, seed=_cell_seed(seed, largest, 5))
    curriculum_opt = torch.optim.Adam(curriculum_model.parameters(), lr=1e-2)
    if compute_budget_flops is None:
        compute_budget_flops = 20.0 * _flops_for(curriculum_model, largest)
    else:
        compute_budget_flops = _finite_nonnegative(compute_budget_flops, "compute_budget_flops")
    curriculum = length_curriculum(
        curriculum_model,
        curriculum_opt,
        ranges,
        vocab=vocab,
        n_rounds=curriculum_rounds,
        compute_budget_flops=compute_budget_flops,
        seed=seed,
        perm=perm,
    )

    protocol_config = {
        "version": 2,
        "seed": seed,
        "ranges": ranges,
        "vocab": vocab,
        "hops": hops,
        "n_train_steps": n_train_steps,
        "n_eval_trials": n_eval_trials,
        "perplexity_steps": perplexity_steps,
        "curriculum_rounds": curriculum_rounds,
    }
    paired_dataset_id = hashlib.sha256(
        json.dumps(protocol_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "ranges": ranges,
        "seed": seed,
        "vocab": vocab,
        "n_params": n_params,
        "evaluation_protocol": "paired-isolated-dependency-v2",
        "paired_dataset_id": paired_dataset_id,
        "suite_training_budget_flops": suite_training_budget_flops,
        "training_flops_used": training_flops_used,
        "evaluation_flops_used": evaluation_flops_used,
        "curriculum_compute_budget_flops": float(compute_budget_flops),
        "state_budget_bytes": state_budget_bytes,
        "state_bytes_used": int(state_bytes_used),
        "within_state_budget": bool(state_bytes_used <= state_budget_bytes),
        "suites": suites,
        "forgetting_curve": forgetting_curve,
        "curriculum": curriculum,
    }


def comparison_table(results: dict[str, Any]) -> str:
    """Render :func:`evaluate` output as a plain-text table. Accepts either a single ``evaluate()``
    return value, or a ``{name: evaluate(...)}`` mapping. Multi-mechanism comparisons fail closed unless
    their paired-data, compute, and state contracts match."""
    if "suites" in results:
        results = {"mechanism": results}
    if not isinstance(results, dict) or not results:
        raise ValueError("results must contain at least one evaluate() receipt.")

    receipts = list(results.values())
    required = {
        "ranges",
        "seed",
        "vocab",
        "evaluation_protocol",
        "paired_dataset_id",
        "suite_training_budget_flops",
        "training_flops_used",
        "curriculum_compute_budget_flops",
        "state_budget_bytes",
        "state_bytes_used",
        "within_state_budget",
        "suites",
    }
    for name, receipt in results.items():
        if not isinstance(receipt, dict):
            raise TypeError(f"{name!r} must be an evaluate() receipt dictionary.")
        missing = required - set(receipt)
        if missing:
            raise ValueError(f"{name!r} is missing evaluate() receipt fields: {sorted(missing)}")
        if not receipt["within_state_budget"]:
            raise ValueError(f"{name!r} exceeded the declared state budget and cannot enter the comparison.")

    if len(receipts) > 1:
        reference = receipts[0]
        paired_fields = (
            "ranges",
            "seed",
            "vocab",
            "evaluation_protocol",
            "paired_dataset_id",
            "curriculum_compute_budget_flops",
            "state_budget_bytes",
        )
        for field in paired_fields:
            if any(receipt[field] != reference[field] for receipt in receipts[1:]):
                raise ValueError(f"comparison requires identical {field}.")

        budgets = [receipt["suite_training_budget_flops"] for receipt in receipts]
        if all(budget is None for budget in budgets):
            used = [float(receipt["training_flops_used"]) for receipt in receipts]
            tolerance = max(1.0, max(used)) * 1e-12
            if max(used) - min(used) > tolerance:
                raise ValueError("training FLOPs differ; rerun with one shared suite_training_budget_flops ceiling.")
        elif any(budget is None for budget in budgets) or len(set(float(budget) for budget in budgets)) != 1:
            raise ValueError("comparison requires the same suite_training_budget_flops contract.")

    lines: list[str] = []
    for name, r in results.items():
        budget_note = "OK" if r["within_state_budget"] else "OVER BUDGET"
        compute_note = (
            f"used_flops={r['training_flops_used']:.3e}"
            if r["suite_training_budget_flops"] is None
            else f"used_flops={r['training_flops_used']:.3e}/{r['suite_training_budget_flops']:.3e}"
        )
        lines.append(
            f"== {name} (seed={r['seed']}, n_params={r['n_params']}, "
            f"{compute_note}, state_bytes={r['state_bytes_used']}/{int(r['state_budget_bytes'])} "
            f"{budget_note}) =="
        )
        w = max(len(str(d)) for d in r["ranges"])
        lines.append(
            f"{'range'.rjust(w)}   {'needle_ok':>10}   {'copy_ok':>10}   {'multihop_ok':>12}   "
            f"{'ppl':>10}   {'train_flops':>12}"
        )
        for d in r["ranges"]:
            s = r["suites"][d]
            lines.append(
                f"{str(d).rjust(w)}   {s['needle']['loss_threshold_success_rate']:>10.3f}   "
                f"{s['copy']['loss_threshold_success_rate']:>10.3f}   "
                f"{s['multi_hop']['loss_threshold_success_rate']:>12.3f}   "
                f"{s['perplexity']['perplexity']:>10.3f}   {s['training_flops']:>12.3e}"
            )
        fc = r["forgetting_curve"]
        lines.append(
            f"  self-knowledge correlation (needle loss vs forgetting): {fc['self_knowledge_correlation']:.3f}"
        )
        cur = r["curriculum"]
        lines.append(f"  curriculum pulls per bucket: {dict(zip(r['ranges'], cur['pulls']))}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")
