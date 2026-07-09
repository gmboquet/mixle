"""E7: the long-context referee -- one evaluation suite every Track-E mechanism (E1's baseline and every
E2-E6 challenger) is measured against on the same terms (see ``mixle/experimental/README.md``'s graduation
rule: "beats the E1 baseline on the E7 evaluation suite at matched FLOPs" + misfit receipts).

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

Every probe trains ``mechanism`` briefly (via :func:`~mixle.experimental.context_spine.train_tbptt`) on
fresh random instances of its suite, then measures held-out accuracy: the protocol
(:class:`~mixle.experimental.context_spine.ContextMechanism`) only returns a scalar mean loss per step, not
logits, so exact argmax accuracy is unavailable by design -- "solved" is instead a chance-normalized loss
threshold (probe loss below half the uniform-guess loss ``0.5 * ln(vocab)``), which is a real, documented
proxy rather than a silently-approximate one.

**Calibrated forgetting curves ("does it know what it forgot?").** The mechanism's OWN per-probe loss is
its only self-reported signal (the protocol exposes nothing else). :func:`evaluate` overlays that signal
against the needle accuracy curve and reports ``self_knowledge_correlation`` -- the correlation between
"how much it forgot" (``1 - accuracy``) and "how surprised it says it was" (its own probe loss) across
ranges. A mechanism that is well-calibrated about its own forgetting scores near +1; a mechanism that is
confidently wrong scores near 0.

**Matched-FLOPs / matched-state-bytes protocols.** :func:`evaluate` reports, per range, the FLOPs spent
(``6 * n_params * n_tokens``, the same Kaplan/Hoffmann approximation :mod:`mixle.ppl.scaling_laws` uses)
and, once, the carried-state byte footprint at the largest tested range against the caller-supplied
``state_budget_bytes``. Two mechanisms compared with :func:`comparison_table` on the SAME ``ranges`` and
``state_budget_bytes`` are, by construction, being compared at matched FLOPs and matched state bytes --
that comparison is the caller's job (pass a ``{name: evaluate(...)}`` mapping); this module only makes the
numbers honest and side-by-side.

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

import dataclasses
import math
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
    chance_loss = math.log(vocab)
    threshold = 0.5 * chance_loss

    for _ in range(n_train_steps):
        x, y = suite_fn(rng, distance=distance, vocab=vocab, **suite_kwargs)
        state = mechanism.init_state(1)
        chunks = _chunks(x, y, chunk_size)
        train_tbptt(mechanism, state, chunks, opt, detach_horizon=len(chunks))

    solved: list[bool] = []
    probe_losses: list[float] = []
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

    return {
        "distance": distance,
        "accuracy": float(np.mean(solved)),
        "mean_probe_loss": float(np.mean(probe_losses)),
        "chance_loss": chance_loss,
    }


def multi_scale_perplexity(
    mechanism: ContextMechanism,
    opt: Any,
    *,
    length: int,
    vocab: int,
    chunk_size: int,
    n_steps: int,
    rng: np.random.RandomState,
    perm: np.ndarray,
) -> dict[str, Any]:
    """Train ``n_steps`` fresh order-1-Markov instances of ``length`` and report mean loss / perplexity."""
    losses: list[float] = []
    for _ in range(n_steps):
        x, y = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
        state = mechanism.init_state(1)
        chunks = _chunks(x, y, chunk_size)
        receipt = train_tbptt(mechanism, state, chunks, opt, detach_horizon=len(chunks))
        losses.append(float(np.mean(receipt["losses"])))
    mean_loss = float(np.mean(losses))
    return {"length": length, "mean_loss": mean_loss, "perplexity": float(math.exp(min(mean_loss, 50.0)))}


def _forgetting_curve(needle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    distances = [r["distance"] for r in needle_rows]
    accuracy = np.array([r["accuracy"] for r in needle_rows])
    self_loss = np.array([r["mean_probe_loss"] for r in needle_rows])
    forgetting = 1.0 - accuracy
    if len(distances) >= 2 and np.std(self_loss) > 0 and np.std(forgetting) > 0:
        corr = float(np.corrcoef(forgetting, self_loss)[0, 1])
    else:
        corr = float("nan")
    return {
        "distances": distances,
        "accuracy": accuracy.tolist(),
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
    bandit = ThompsonBernoulli(len(ranges), seed=seed)
    ledger = np.full(len(ranges), float(compute_budget_flops) / len(ranges))
    costs = np.array([_flops_for(mechanism, r) for r in ranges])
    cost_ratio = costs / max(float(np.min(costs)), 1.0)
    rng = np.random.RandomState(seed + 1)

    pulled_lengths: list[int] = []
    for _ in range(n_rounds):
        affordable = ledger >= costs
        if not affordable.any():
            break  # the compute box is exhausted; stop rather than overspend any bucket.
        draws = bandit.rng.beta(bandit.alpha, bandit.beta)
        draws = np.where(affordable, draws, -np.inf)
        arm = int(np.argmax(draws))
        length = int(ranges[arm])
        chunk_size = _choose_chunk_size(length)

        with torch.no_grad():
            x0, y0 = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
            state0 = mechanism.init_state(1)
            before_chunks = _chunks(x0, y0, chunk_size)
            loss_before = 0.0
            for chunk in before_chunks:
                state0, loss = mechanism.step(state0, chunk)
                loss_before += float(loss)
            loss_before /= len(before_chunks)

        x1, y1 = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
        state1 = mechanism.init_state(1)
        chunks1 = _chunks(x1, y1, chunk_size)
        train_tbptt(mechanism, state1, chunks1, opt, detach_horizon=len(chunks1))

        with torch.no_grad():
            x2, y2 = _markov_sequence(rng, length=length, vocab=vocab, perm=perm)
            state2 = mechanism.init_state(1)
            after_chunks = _chunks(x2, y2, chunk_size)
            loss_after = 0.0
            for chunk in after_chunks:
                state2, loss = mechanism.step(state2, chunk)
                loss_after += float(loss)
            loss_after /= len(after_chunks)

        chance_loss = math.log(vocab)
        improvement = max(loss_before - loss_after, 0.0)
        normalized_improvement = min(improvement / chance_loss, 1.0)
        reward = float(np.clip(normalized_improvement / cost_ratio[arm], 0.0, 1.0))
        bandit.update(arm, reward)
        ledger[arm] -= costs[arm]
        pulled_lengths.append(length)

    return {
        "bucket_ranges": [int(r) for r in ranges],
        "pulls": bandit.pulls.tolist(),
        "posterior_means": bandit.means.tolist(),
        "ledger_remaining": ledger.tolist(),
        "pulled_lengths": pulled_lengths,
    }


# ---------------------------------------------------------------------------------------------------------
# The one-command entry point.
# ---------------------------------------------------------------------------------------------------------


def evaluate(
    mechanism: ContextMechanism,
    *,
    ranges: tuple[float, ...] = (1e3, 1e4, 1e5, 1e6),
    state_budget_bytes: float,
    seed: int,
    hops: int = 3,
    n_train_steps: int = 6,
    n_eval_trials: int = 8,
    perplexity_steps: int = 6,
    curriculum_rounds: int = 12,
    compute_budget_flops: float | None = None,
) -> dict[str, Any]:
    """Run the full E7 referee suite against ``mechanism`` end-to-end. See the module docstring for what
    each piece measures and honestly claims. Requires a torch-trainable mechanism (``.parameters()``
    exposed, as every Track-E mechanism in :mod:`mixle.experimental.context_spine` is) -- trains it IN
    PLACE via TBPTT, so pass a freshly-initialized instance for a clean baseline measurement.

    ``compute_budget_flops`` defaults to ``20x`` the FLOP cost of one full pass over the largest range,
    generous enough that the length-curriculum bandit (:func:`length_curriculum`) gets a meaningful number
    of rounds without the caller having to reason about FLOPs by hand.
    """
    _require_torch()
    if not hasattr(mechanism, "parameters"):
        raise ValueError("evaluate() requires a torch-trainable mechanism exposing .parameters().")

    ranges = tuple(int(r) for r in ranges)
    if not ranges:
        raise ValueError("ranges must be non-empty.")
    if any(r < 2 for r in ranges):
        raise ValueError(f"every range must be >= 2 (needle_suite's minimum controlled distance): {ranges}")

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    vocab = int(getattr(mechanism, "vocab", DEFAULT_VOCAB))
    opt = torch.optim.Adam(mechanism.parameters(), lr=1e-2)
    perm = rng.permutation(vocab)

    suites: dict[int, dict[str, Any]] = {}
    needle_rows: list[dict[str, Any]] = []
    for distance in ranges:
        chunk_size = _choose_chunk_size(distance)
        needle = _train_and_probe(
            mechanism,
            opt,
            needle_suite,
            distance=distance,
            vocab=vocab,
            chunk_size=chunk_size,
            n_train_steps=n_train_steps,
            n_eval_trials=n_eval_trials,
            rng=rng,
        )
        copy_ = _train_and_probe(
            mechanism,
            opt,
            copy_suite,
            distance=distance,
            vocab=vocab,
            chunk_size=chunk_size,
            n_train_steps=n_train_steps,
            n_eval_trials=n_eval_trials,
            rng=rng,
        )
        multi_hop = _train_and_probe(
            mechanism,
            opt,
            multi_hop_suite,
            distance=distance,
            vocab=vocab,
            chunk_size=chunk_size,
            n_train_steps=n_train_steps,
            n_eval_trials=n_eval_trials,
            rng=rng,
            hops=hops,
        )
        perplexity = multi_scale_perplexity(
            mechanism,
            opt,
            length=distance,
            vocab=vocab,
            chunk_size=chunk_size,
            n_steps=perplexity_steps,
            rng=rng,
            perm=perm,
        )
        needle_rows.append(needle)
        suites[distance] = {
            "needle": needle,
            "copy": copy_,
            "multi_hop": multi_hop,
            "perplexity": perplexity,
            "flops": _flops_for(mechanism, distance),
        }

    forgetting_curve = _forgetting_curve(needle_rows)

    largest = max(ranges)
    chunk_size = _choose_chunk_size(largest)
    x, y = copy_suite(rng, distance=largest, vocab=vocab)
    state = mechanism.init_state(1)
    with torch.no_grad():
        for chunk in _chunks(x, y, chunk_size):
            state, _ = mechanism.step(state, chunk)
    state_bytes_used = _state_bytes(state)

    if compute_budget_flops is None:
        compute_budget_flops = 20.0 * _flops_for(mechanism, largest)
    curriculum = length_curriculum(
        mechanism,
        opt,
        ranges,
        vocab=vocab,
        n_rounds=curriculum_rounds,
        compute_budget_flops=compute_budget_flops,
        seed=seed,
        perm=perm,
    )

    return {
        "ranges": ranges,
        "seed": seed,
        "vocab": vocab,
        "n_params": _n_params(mechanism),
        "state_budget_bytes": float(state_budget_bytes),
        "state_bytes_used": int(state_bytes_used),
        "within_state_budget": bool(state_bytes_used <= state_budget_bytes),
        "suites": suites,
        "forgetting_curve": forgetting_curve,
        "curriculum": curriculum,
    }


def comparison_table(results: dict[str, Any]) -> str:
    """Render :func:`evaluate` output as a plain-text table. Accepts either a single ``evaluate()``
    return value, or a ``{name: evaluate(...)}`` mapping for a matched-FLOPs / matched-state-bytes
    side-by-side comparison (e.g. an E2-E6 challenger against the E1 baseline, both evaluated with the
    same ``ranges`` and ``state_budget_bytes``)."""
    if "suites" in results:
        results = {"mechanism": results}

    lines: list[str] = []
    for name, r in results.items():
        budget_note = "OK" if r["within_state_budget"] else "OVER BUDGET"
        lines.append(
            f"== {name} (seed={r['seed']}, n_params={r['n_params']}, "
            f"state_bytes={r['state_bytes_used']}/{int(r['state_budget_bytes'])} {budget_note}) =="
        )
        w = max(len(str(d)) for d in r["ranges"])
        lines.append(
            f"{'range'.rjust(w)}   {'needle_acc':>10}   {'copy_acc':>10}   {'multihop_acc':>12}   "
            f"{'ppl':>10}   {'flops':>12}"
        )
        for d in r["ranges"]:
            s = r["suites"][d]
            lines.append(
                f"{str(d).rjust(w)}   {s['needle']['accuracy']:>10.3f}   {s['copy']['accuracy']:>10.3f}   "
                f"{s['multi_hop']['accuracy']:>12.3f}   {s['perplexity']['perplexity']:>10.3f}   "
                f"{s['flops']:>12.3e}"
            )
        fc = r["forgetting_curve"]
        lines.append(
            f"  self-knowledge correlation (needle loss vs forgetting): {fc['self_knowledge_correlation']:.3f}"
        )
        cur = r["curriculum"]
        lines.append(f"  curriculum pulls per bucket: {dict(zip(r['ranges'], cur['pulls']))}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")
