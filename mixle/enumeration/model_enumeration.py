"""Enumerate the outputs of arbitrary scoring models (neural nets, transformers, ...) by score.

The mixle distribution enumerators (``dist.enumerator()``) need a mixle distribution. These utilities instead
work with any model that can *score* candidates, supplied as a plain Python callable -- so a transformer, RNN,
n-gram language model, or any classifier can be enumerated in descending log-probability order without being a
mixle distribution. Nothing here imports a deep-learning framework; the caller's callable bridges to their model.

Four entry points, from most general to most specific:

- ``best_first(...)`` -- the generic engine: best-first / A* search that yields goal states in descending score,
  given ``successors``, an ``is_goal`` test, a ``score``, and an optional admissible ``heuristic``.
- ``best_first_decode(next_logprobs, ...)`` -- EXACT descending-probability enumeration of sequences from an
  autoregressive model. ``next_logprobs(prefix)`` returns ``(token, log_prob)`` continuations (e.g. a
  transformer's ``log_softmax`` of the next-token logits). Yields ``(sequence, total_log_prob)`` lazily, best
  first. Exact because each step's log-prob is <= 0, so a prefix's score upper-bounds every completion.
- ``beam_search(next_logprobs, beam_width, ...)`` -- the classic approximate top-k decode (fixed beam), for
  when exact best-first explores too much.
- ``top_k_scored(candidates, score, k)`` -- top-k over a finite candidate set scored by a callable (e.g. a
  classifier's class log-probabilities).

Example (transformer-style next-token decoding)::

    import numpy as np
    def next_logprobs(prefix):
        logits = my_transformer(prefix)            # (vocab,) numpy/torch -> numpy
        lp = logits - logsumexp(logits)            # log_softmax
        return list(enumerate(lp))                 # [(token_id, log_prob), ...]
    for seq, total_lp in best_first_decode(next_logprobs, eos=EOS, max_len=20, max_results=5):
        ...                                        # the 5 highest-probability sequences, best first
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Iterable, Iterator
from typing import Any


def _validate_step_logprob(log_prob: float) -> float:
    """Reject a continuation log-probability that would break best-first's exactness proof.

    ``best_first_decode`` (and the pruning inside ``quantized_best_first_decode``) is sound only
    because every step's log-probability is <= 0, so a prefix's running score never increases as it is
    extended -- the property that makes "everything still in the frontier bounds everything not yet
    generated" hold, which is what makes "already-popped is the true top-k" a valid conclusion. A
    positive increment breaks that bound silently (results stay labeled exact while actually being
    wrong); NaN breaks every downstream comparison the same increment feeds (heap order, top-p
    cumulative mass) the same way. ``-inf`` is accepted: it is the standard zero-probability
    ("impossible token") sentinel used throughout this codebase (see e.g.
    ``quantization.core.Quantizer.bits``) and does not threaten monotonicity -- extending with a -inf
    step only explains why a branch is deprioritized, never why one should be promoted above an
    already-returned result.
    """
    lp = float(log_prob)
    if math.isnan(lp):
        raise ValueError(f"continuation log-probability must not be NaN, got {log_prob!r}.")
    if lp > 0.0:
        raise ValueError(f"continuation log-probability must be <= 0 (a probability > 1 is invalid), got {log_prob!r}.")
    return lp


def _positive_int(value: Any, name: str) -> int:
    """Coerce ``value`` to a positive ``int``, raising a clear ``ValueError`` if it is not one."""
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    if n < 1 or n != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return n


def _validate_pruning_bounds(top_k: int | None, top_p: float | None) -> None:
    """Reject a nonsensical top-k / top-p (nucleus) pruning bound before it can silently mis-prune a step."""
    if top_k is not None:
        _positive_int(top_k, "top_k")
    if top_p is not None and not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"top_p must be a value in (0, 1], got {top_p!r}.")


# 2**1023 is the largest integer power of two that still converts to a finite float64 (2**1024 raises
# OverflowError) -- well beyond any bucket resolution a caller could sensibly want (the default is 12),
# but validated explicitly rather than left to surface as an opaque OverflowError deep inside push().
_MAX_BUCKET_BITS = 1023


def _validate_bucket_bits(bucket_bits: Any) -> int:
    """Coerce ``bucket_bits`` to an ``int`` in ``[0, 1023]`` so ``2**bucket_bits`` is a finite float."""
    try:
        n = int(bucket_bits)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"bucket_bits must be an integer, got {bucket_bits!r}.") from exc
    if n != bucket_bits or not (0 <= n <= _MAX_BUCKET_BITS):
        raise ValueError(f"bucket_bits must be an integer in [0, {_MAX_BUCKET_BITS}], got {bucket_bits!r}.")
    return n


def best_first(
    start: Any,
    successors: Callable[[Any], Iterable[Any]],
    is_goal: Callable[[Any], bool],
    score: Callable[[Any], float],
    heuristic: Callable[[Any], float] | None = None,
    max_results: int | None = None,
) -> Iterator[tuple[Any, float]]:
    """Best-first / A* search yielding goal states in descending score (log-probability) order.

    Yields ``(goal_state, score(goal_state))`` lazily, highest first. The ordering is exact when
    ``f(state) = score(state) + heuristic(state)`` is an admissible upper bound on the score of every goal
    reachable from ``state`` -- in particular when ``heuristic`` is omitted (treated as 0) and ``score`` never
    increases along a path (the usual case for cumulative log-probabilities, which add terms <= 0).

    CALLER-CERTIFIED primitive: this is the fully generic engine, so ``score``/``heuristic`` are not
    validated here. Whether ``f = score + heuristic`` is a genuine admissible bound is not something
    this function can check generically -- verifying it would mean solving the search first -- so an
    unsound pair silently yields results in the wrong order with no error, exactly like a caller-bug in
    ``successors``/``is_goal`` would. Positive per-step score increases are NOT rejected here (unlike
    :func:`best_first_decode`'s narrower h=0 contract) because they are sometimes legitimate: e.g.
    ``hmm_paths.hmm_best_paths`` calls this with a possibly-positive continuous-density emission score,
    made exact by a backward-Viterbi ``heuristic`` that bounds it, not by the score alone being
    non-increasing. Callers that only have a plain non-increasing log-probability score (the common
    case) should prefer :func:`best_first_decode`, which DOES validate that contract because it is
    actually checkable there.

    Args:
        start: the initial (typically partial) state.
        successors: expand a state into its child states.
        is_goal: True when a state is a complete output to yield (and not expanded further).
        score: the (partial) log-score of a state.
        heuristic: optional admissible upper bound on the best completion score reachable from a state.
        max_results: stop after yielding this many goals; ``None`` enumerates until the frontier is empty.

    Yields:
        ``(goal_state, score)`` in nonincreasing score.
    """
    h = (lambda _s: 0.0) if heuristic is None else heuristic
    counter = itertools.count()  # tiebreaker so the heap never compares states
    heap: list[tuple[float, int, Any]] = [(-(score(start) + h(start)), next(counter), start)]
    emitted = 0
    while heap and (max_results is None or emitted < max_results):
        _, _, state = heapq.heappop(heap)
        if is_goal(state):
            yield state, score(state)
            emitted += 1
            continue
        for child in successors(state):
            heapq.heappush(heap, (-(score(child) + h(child)), next(counter), child))


def best_first_decode(
    next_logprobs: Callable[[tuple], Iterable[tuple[Any, float]]],
    eos: Any = None,
    max_len: int | None = None,
    start: tuple = (),
    heuristic: Callable[[tuple], float] | None = None,
    max_results: int | None = None,
) -> Iterator[tuple[tuple, float]]:
    """Exactly enumerate an autoregressive model's sequences in descending total log-probability.

    Args:
        next_logprobs: ``next_logprobs(prefix)`` returns an iterable of ``(token, log_prob)`` continuations of
            ``prefix`` (e.g. the log-softmax of a transformer's next-token logits). Log-probs must be <= 0
            (NaN or positive values are rejected with ``ValueError`` -- see below).
        eos: end-of-sequence token; a prefix whose last token is ``eos`` is complete (and not extended).
        max_len: maximum sequence length; a prefix of this length is complete. At least one of ``eos`` /
            ``max_len`` should be given or enumeration may not terminate.
        start: the initial prefix (default empty).
        heuristic: optional admissible upper bound on the remaining log-probability from a prefix (e.g.
            ``remaining_steps * max_step_logprob``); tightens the search. Omit for the exact h=0 search.
        max_results: stop after this many complete sequences.

    Yields:
        ``(sequence_tuple, total_log_prob)`` in nonincreasing total log-probability.

    Raises:
        ValueError: if ``next_logprobs`` returns a NaN or positive log-probability. Unlike the fully
            generic :func:`best_first`, this function's whole exactness claim rests on every step's
            log-prob being <= 0 (see the module docstring), so that contract IS validated here, at the
            point each continuation is consumed -- a caller's scoring bug (e.g. passing raw logits
            instead of a log-softmax) fails loudly instead of silently shipping results that are
            labeled exact but are not.
    """
    if eos is None and max_len is None:
        raise ValueError("best_first_decode needs eos and/or max_len to know when a sequence is complete.")

    def _is_complete(prefix: tuple) -> bool:
        if eos is not None and len(prefix) > 0 and prefix[-1] == eos:
            return True
        return max_len is not None and len(prefix) >= max_len

    # state = (prefix_tuple, cumulative_log_prob)
    def successors(state: tuple) -> Iterator[tuple[tuple, float]]:
        prefix, lp = state
        for token, token_lp in next_logprobs(prefix):
            yield (prefix + (token,), lp + _validate_step_logprob(token_lp))

    h = None if heuristic is None else (lambda state: heuristic(state[0]))
    for (prefix, lp), _score in best_first(
        (start, 0.0),
        successors=successors,
        is_goal=lambda state: _is_complete(state[0]),
        score=lambda state: state[1],
        heuristic=h,
        max_results=max_results,
    ):
        yield prefix, lp


def beam_search(
    next_logprobs: Callable[[tuple], Iterable[tuple[Any, float]]],
    beam_width: int,
    eos: Any = None,
    max_len: int | None = None,
    start: tuple = (),
    num_results: int | None = None,
) -> list[tuple[tuple, float]]:
    """Approximate top sequences of an autoregressive model by beam search.

    Keeps at most ``beam_width`` live prefixes per step (the highest-scoring ones); a prefix that emits ``eos``
    or reaches ``max_len`` is finalized. Returns the finalized sequences sorted by total log-probability. This
    is the standard heuristic decode -- faster than exact best-first but not guaranteed to return the true
    top-k.

    Args:
        next_logprobs: ``next_logprobs(prefix) -> [(token, log_prob), ...]`` (see ``best_first_decode``).
        beam_width: number of prefixes kept per step.
        eos: end-of-sequence token (optional).
        max_len: maximum length (optional, but recommended to bound the search).
        start: the initial prefix.
        num_results: number of sequences to return (default ``beam_width``).

    Returns:
        A list of ``(sequence_tuple, total_log_prob)`` sorted by nonincreasing log-probability.
    """
    if eos is None and max_len is None:
        raise ValueError("beam_search needs eos and/or max_len to terminate.")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")

    beam: list[tuple[tuple, float]] = [(start, 0.0)]
    finished: list[tuple[tuple, float]] = []
    step = 0
    while beam and (max_len is None or step < max_len):
        candidates: list[tuple[tuple, float]] = []
        for prefix, lp in beam:
            for token, token_lp in next_logprobs(prefix):
                new_prefix = prefix + (token,)
                new_lp = lp + token_lp
                if (eos is not None and token == eos) or (max_len is not None and len(new_prefix) >= max_len):
                    finished.append((new_prefix, new_lp))
                else:
                    candidates.append((new_prefix, new_lp))
        candidates.sort(key=lambda u: -u[1])
        beam = candidates[:beam_width]
        step += 1

    finished.sort(key=lambda u: -u[1])
    return finished[: (beam_width if num_results is None else num_results)]


def top_k_scored(
    candidates: Iterable[Any], score: Callable[[Any], float], k: int | None = None
) -> list[tuple[Any, float]]:
    """Return a finite candidate set scored by ``score``, sorted in descending score.

    For a classifier: ``candidates`` are the class labels and ``score`` is ``lambda c: model.log_prob(c | x)``.

    Args:
        candidates: a finite iterable of candidate outputs.
        score: the (log-)score of a candidate.
        k: keep only the top ``k`` (uses a bounded heap); ``None`` returns all, sorted. Must be a
            positive integer when given -- ``heapq.nlargest`` silently returns ``[]`` for ``k <= 0``,
            which reads as "no candidates qualify" rather than the caller-bound error it actually is.

    Returns:
        A list of ``(candidate, score)`` in nonincreasing score.
    """
    scored = ((c, float(score(c))) for c in candidates)
    if k is None:
        return sorted(scored, key=lambda u: -u[1])
    k = _positive_int(k, "k")
    return heapq.nlargest(k, scored, key=lambda u: u[1])


def _prune_step(items: Iterable[tuple[Any, float]], top_k: int | None, top_p: float | None) -> list[tuple[Any, float]]:
    """Restrict a step's (token, log_prob) continuations to the top-k / top-p (nucleus).

    This structural pruning keeps peaked neural distributions tractable to enumerate.

    ``top_k``/``top_p`` are validated (see :func:`_validate_pruning_bounds`) and every ``(token,
    log_prob)`` pair is validated (see :func:`_validate_step_logprob`) before ranking. Both matter for
    the same reason: an out-of-range ``top_p`` or a NaN log-prob silently corrupts the cumulative-mass
    nucleus loop below (``cum >= top_p`` compares False against NaN either way, so the loop never stops
    early -- it just keeps whatever is left), and an unvalidated positive log-prob breaks the same
    non-increasing-score monotonicity :func:`best_first_decode` depends on, since every value pruned
    here is what :func:`quantized_best_first_decode` pushes back onto its frontier.
    """
    _validate_pruning_bounds(top_k, top_p)
    ranked = sorted(((token, _validate_step_logprob(lp)) for token, lp in items), key=lambda u: -u[1])
    if top_k is not None:
        ranked = ranked[:top_k]
    if top_p is not None and ranked:
        kept: list[tuple[Any, float]] = []
        cum = 0.0
        for token, lp in ranked:
            kept.append((token, lp))
            cum += math.exp(lp)
            if cum >= top_p:
                break
        ranked = kept
    return ranked


def quantized_best_first_decode(
    next_logprobs: Callable[[tuple], Iterable[tuple[Any, float]]] | None = None,
    eos: Any = None,
    max_len: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    bucket_bits: int = 12,
    batch_next_logprobs: Callable[[list[tuple]], list[Iterable[tuple[Any, float]]]] | None = None,
    batch_size: int = 64,
    start: tuple = (),
    max_results: int | None = None,
    min_mass: float | None = None,
) -> Iterator[tuple[tuple, float]]:
    """Fast descending-probability sequence enumeration specialized for neural / transformer decoders.

    Three structure-aware accelerations over :func:`best_first_decode`:

    1. **Nucleus / top-k pruning.** Neural next-token distributions are sharply peaked, so each step is
       restricted to its ``top_k`` tokens or its ``top_p`` nucleus -- dropping the long low-probability tail
       collapses the branching factor (a ~50k vocab down to a handful) at negligible mass loss.
    2. **Quantized bucket priority queue.** Cumulative log-probs only decrease, so instead of an O(log n)
       comparison heap the frontier is bucketed by quantized score (``bucket = floor(score * 2**bucket_bits)``)
       and drained highest-bucket first -- O(1) pushes/pops, and prefixes of near-equal score are grouped.
       Buckets are disjoint score ranges, so order is exact across buckets and within ~2**-bucket_bits inside one.
    3. **Batched scoring.** The cost is dominated by model forward passes. Pass ``batch_next_logprobs`` to
       score up to ``batch_size`` frontier prefixes in one call (one padded GPU forward) instead of one at a time.

    With ``top_k=top_p=None`` and a large ``bucket_bits`` this reduces to the exact enumeration; pruning is the
    only approximation (report ``min_mass`` to stop once enough probability is covered).

    Args:
        next_logprobs: ``next_logprobs(prefix) -> [(token, log_prob), ...]`` (used when ``batch_next_logprobs``
            is not given). Log-probs must be <= 0.
        eos: end-of-sequence token (a prefix ending in ``eos`` is complete).
        max_len: maximum sequence length. Give ``eos`` and/or ``max_len``.
        top_k: keep only the ``top_k`` highest-probability tokens per step.
        top_p: keep the smallest set of tokens per step whose probability sums to >= ``top_p`` (nucleus).
        bucket_bits: score-quantization resolution; larger = finer ordering, slower bookkeeping. Must be
            an integer in ``[0, 1023]`` (so ``2**bucket_bits`` is a finite float).
        batch_next_logprobs: optional ``batch_next_logprobs([prefix, ...]) -> [[(token, log_prob), ...], ...]``
            scoring a batch of prefixes in one forward pass. MUST return exactly one step table per
            requested prefix, in the same order -- a short or long return is a caller bug and raises
            ``ValueError`` naming the mismatch, rather than silently pairing up (and dropping) prefixes
            with ``zip()``.
        batch_size: number of frontier prefixes expanded per (batched) scoring call. Must be a positive
            integer -- a non-positive value previously could take an empty slice of a non-empty bucket
            without consuming it, hanging the search forever (MXR-080-0227).
        start: initial prefix.
        max_results: stop after this many complete sequences. Must be a positive integer when given.
        min_mass: stop once the yielded sequences cover at least this much probability mass. Must be a
            finite value >= 0.0 when given.

    Yields:
        ``(sequence_tuple, total_log_prob)``, highest probability first (exact across score buckets).

    Raises:
        ValueError: for any invalid control (``batch_size``, ``bucket_bits``, ``max_results``,
            ``min_mass``, ``top_k``, ``top_p``), or if a batched callback returns a different number of
            step tables than prefixes requested.
    """
    if next_logprobs is None and batch_next_logprobs is None:
        raise ValueError("provide next_logprobs or batch_next_logprobs.")
    if eos is None and max_len is None:
        raise ValueError("quantized_best_first_decode needs eos and/or max_len to know when a sequence is complete.")
    # Validate every control up front, before any work happens, so a bad caller input fails fast and
    # atomically instead of raising (or hanging -- MXR-080-0227) partway through an already-started
    # search that has silently yielded some results.
    batch_size = _positive_int(batch_size, "batch_size")
    bucket_bits = _validate_bucket_bits(bucket_bits)
    if max_results is not None:
        max_results = _positive_int(max_results, "max_results")
    if min_mass is not None:
        min_mass = float(min_mass)
        if not math.isfinite(min_mass) or min_mass < 0.0:
            raise ValueError(f"min_mass must be a finite value >= 0.0, got {min_mass!r}.")
    _validate_pruning_bounds(top_k, top_p)

    scale = float(2**bucket_bits)

    def _is_complete(prefix: tuple) -> bool:
        if eos is not None and len(prefix) > 0 and prefix[-1] == eos:
            return True
        return max_len is not None and len(prefix) >= max_len

    buckets: dict[int, list[tuple[float, tuple]]] = {}
    bucket_heap: list[int] = []  # min-heap of -bucket so the highest bucket pops first

    def push(prefix: tuple, score: float) -> None:
        b = math.floor(score * scale)
        lst = buckets.get(b)
        if lst is None:
            buckets[b] = [(score, prefix)]
            heapq.heappush(bucket_heap, -b)
        else:
            lst.append((score, prefix))

    push(start, 0.0)
    emitted = 0
    covered = 0.0
    while bucket_heap:
        b = -bucket_heap[0]
        lst = buckets.get(b)
        if not lst:
            heapq.heappop(bucket_heap)
            buckets.pop(b, None)
            continue
        # Peek (do NOT yet remove) a batch from the current (highest) bucket. Removal from the live
        # bucket is deferred until any batched scoring call for this batch has returned and been
        # validated, so a callback that raises or under/over-returns leaves the frontier exactly as it
        # was. The previous code did this the other way around: remove the batch first, then pair it
        # with the callback's results via zip() -- which silently stops at the shorter of the two
        # lists, dropping any prefix the callback failed to score with no error at all (MXR-080-0227;
        # the reproduced case was a single live prefix and an under-returning callback, which came back
        # as an empty enumeration -- the whole search silently vanished).
        before = len(lst)
        take_all = before <= batch_size
        take = lst if take_all else lst[-batch_size:]
        to_expand = [u for u in take if not _is_complete(u[1])]
        steps: list[Iterable[tuple[Any, float]]] = []
        if to_expand:
            prefixes = [pf for _, pf in to_expand]
            if batch_next_logprobs is not None:
                steps = list(batch_next_logprobs(prefixes))
            else:
                steps = [next_logprobs(pf) for pf in prefixes]
            if len(steps) != len(prefixes):
                raise ValueError(
                    "batch_next_logprobs must return exactly one step table per requested prefix "
                    f"(expected {len(prefixes)}, got {len(steps)})."
                )
        # Commit: scoring for this batch succeeded (or was not needed), so it is now safe to remove it
        # from the live bucket. batch_size is validated positive and `lst` is non-empty here, so this
        # must always strictly shrink the bucket -- checked explicitly rather than trusted, so a similar
        # slicing bug can never again hang the loop silently: a negative batch_size previously took an
        # empty slice right here without shrinking the live bucket at all, so the loop revisited the
        # same non-empty bucket forever (MXR-080-0227).
        if take_all:
            buckets[b] = []
            after = 0
        else:
            del lst[-batch_size:]
            after = len(lst)
        if after >= before:
            raise RuntimeError(
                f"quantized_best_first_decode: bucket {b} did not shrink this iteration "
                f"({before} -> {after}); refusing to loop forever."
            )
        # yield completed sequences in this batch, exact order within the bucket
        for sc, pf in sorted((u for u in take if _is_complete(u[1])), key=lambda u: -u[0]):
            yield pf, sc
            emitted += 1
            covered += math.exp(sc)
            if (max_results is not None and emitted >= max_results) or (min_mass is not None and covered >= min_mass):
                return
        for (sc, pf), step in zip(to_expand, steps):
            for token, token_lp in _prune_step(step, top_k, top_p):
                push(pf + (token,), sc + token_lp)
