"""Descending-probability enumeration AND unranking of HMM state paths (list-Viterbi / A* + count-DP).

The count-budget index handles the *decomposable* families (Sequence / Composite / MarkovChain). An HMM
is non-decomposable -- the latent state couples emissions across time -- so it is not served by the
count semiring. Two complementary tools live here:

* :func:`hmm_best_paths` -- **exact, lazy** enumeration in nonincreasing joint log-probability by A*
  search with the backward Viterbi value as an admissible (in fact tight) heuristic. The right tool for
  the head, but rank ``k`` costs ``O(k)`` expansions: there is no random access.
* :class:`HMMPathIndex` -- the **quantized precomputation structure**: a forward count DP over integer
  score buckets (one ``O(T * K^2 * W)`` build, ``W`` = bit budget in fine buckets) that then answers
  ``count`` / ``unrank`` / ``threshold`` / ``mass_above`` in ``O(T * K)`` per query -- random access into
  the ranked path list at any depth, which A* structurally cannot do. Ordering is exact up to the fine
  bucket width (``bin_width_bits / oversample`` bits; raise ``oversample`` to sharpen); every returned
  path carries its **exact** joint log-probability.

For a path ``z_{1..T}`` the joint log-probability of the latent path with the observed emissions is
``log_pi[z_1] + log_b[0, z_1] + sum_{t>1} (log_A[z_{t-1}, z_t] + log_b[t, z_t])`` where ``log_b[t, k]``
is the emission log-likelihood ``log p(x_t | z_t = k)``. The completion heuristic
``h[t, s] = max over z_{t+1..T} of the remaining transition+emission score`` is computed once by a
backward max-product pass, so ``g + h`` is the exact score of the best path through ``(t, s)`` -- an
admissible bound that makes best-first emit paths in exactly nonincreasing joint log-probability.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from mixle.enumeration.model_enumeration import best_first
from mixle.enumeration.quantization.core import _TOL, Quantizer

_LOG2 = math.log(2.0)


def _backward_viterbi(log_pi: np.ndarray, log_A: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """``h[t, s]`` = best (max) completion score after occupying state ``s`` at time ``t``.

    ``h[T-1, s] = 0`` (nothing left to add); ``h[t, s] = max_{s'} log_A[s, s'] + log_b[t+1, s'] +
    h[t+1, s']``. This is the standard backward max-product recursion.
    """
    t_len, k = log_b.shape
    h = np.zeros((t_len, k), dtype=float)
    for t in range(t_len - 2, -1, -1):
        # m[s, s'] = score of stepping s -> s' at time t+1 then completing optimally
        m = log_A + (log_b[t + 1] + h[t + 1])[None, :]
        h[t] = m.max(axis=1)
    return h


def hmm_best_paths(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_b: np.ndarray,
    k: int | None = None,
) -> Iterator[tuple[tuple[int, ...], float]]:
    """Enumerate HMM state paths in nonincreasing joint log-probability.

    Args:
        log_pi: ``(K,)`` log initial-state distribution.
        log_A: ``(K, K)`` log transition matrix (row ``j`` -> column ``k`` is ``log p(z_t=k | z_{t-1}=j)``).
        log_b: ``(T, K)`` per-position emission log-likelihoods ``log p(x_t | z_t=k)``.
        k: stop after the ``k`` best paths; ``None`` enumerates all ``K**T`` lazily.

    Yields:
        ``(path, joint_log_prob)`` with ``path`` a length-``T`` tuple of state indices, highest first.
        The first yield is the Viterbi (MAP) path.
    """
    log_pi = np.asarray(log_pi, dtype=float)
    log_A = np.asarray(log_A, dtype=float)
    log_b = np.asarray(log_b, dtype=float)
    t_len, n_states = log_b.shape
    if t_len == 0:
        return
    h = _backward_viterbi(log_pi, log_A, log_b)

    # State: (t, last_state, path, g). The synthetic root (-1, -1, (), 0.0) fans out into the T=0 states
    # so best_first has a single start. score = prefix log-prob g; heuristic = best completion h[t, s].
    root = (-1, -1, (), 0.0)

    def successors(state: Any) -> Iterator[Any]:
        t, s, path, g = state
        if t == -1:  # root -> choose z_1
            for s1 in range(n_states):
                g1 = float(log_pi[s1] + log_b[0, s1])
                yield (0, s1, (s1,), g1)
            return
        if t >= t_len - 1:  # complete path: no successors
            return
        row = log_A[s]
        nxt = log_b[t + 1]
        for s2 in range(n_states):
            g2 = g + float(row[s2] + nxt[s2])
            yield (t + 1, s2, path + (s2,), g2)

    def is_goal(state: Any) -> bool:
        return state[0] == t_len - 1

    def score(state: Any) -> float:
        return state[3]

    def heuristic(state: Any) -> float:
        t, s = state[0], state[1]
        if t < 0:
            return 0.0  # root is popped first regardless of its f
        return float(h[t, s])

    for state, g in best_first(root, successors, is_goal, score, heuristic, max_results=k):
        yield state[2], g


class HMMPathIndex:
    """Quantized random-access index over an HMM's state paths for one observation sequence.

    **Precompute** (once, ``O(T * K^2 * W)``): quantize every step score -- ``log_pi[s] + log_b[0, s]``
    and ``log_A[s, s'] + log_b[t, s']``, one floor per step so the accumulated smear is at most ``T``
    fine buckets -- and run a forward count DP: ``C_t[s']`` is the histogram, over integer total-score
    buckets, of the number of length-``t+1`` prefixes ending in state ``s'``. Counts are float64 (exact
    below 2**53; the number of paths is ``K**T``, so deep problems carry the documented ~1e-16/op
    relative error instead of overflowing).

    **Query** (each ``O(T * K)``): ``unrank(i)`` walks the stored tables backward -- pick the final
    state whose bucket count covers the offset, then repeatedly the predecessor whose shifted prefix
    bucket does -- returning the ``i``-th best path *by quantized score* and its exact joint
    log-probability. ``count`` / ``threshold`` / ``mass_above`` read the pooled final histogram.

    Ordering contract: paths are ordered by their quantized bucket (width ``bin_width_bits/oversample``
    bits); within a bucket the order is deterministic but unspecified -- exactly the count-index
    semantics elsewhere in :mod:`mixle.enumeration`. :func:`hmm_best_paths` remains the exact-order
    tool for the head; this index is the random-access tool for depth (rank 1e6 costs one table walk,
    not 1e6 A* expansions).
    """

    def __init__(
        self,
        log_pi: np.ndarray,
        log_A: np.ndarray,
        log_b: np.ndarray,
        *,
        bin_width_bits: float = 1.0,
        oversample: int = 8,
        budget_bits: float | None = None,
    ) -> None:
        self.log_pi = np.asarray(log_pi, dtype=float)
        self.log_A = np.asarray(log_A, dtype=float)
        self.log_b = np.asarray(log_b, dtype=float)
        self.T, self.K = self.log_b.shape
        if self.T == 0:
            raise ValueError("log_b must cover at least one position")
        self.quantizer = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)

        # Emission log-likelihoods may be POSITIVE (continuous densities), so raw joint scores are not
        # bounded by 0. Shift each position's step scores by that position's best (max) score: shifted
        # scores are <= 0, buckets measure "bits behind the per-step optimum", and ordering is unchanged
        # (the same constant shifts every path). ``total_offset`` converts bucket <-> true joint score.
        init_scores = self.log_pi + self.log_b[0]
        self._off = np.zeros(self.T, dtype=float)
        self._off[0] = float(np.max(init_scores[np.isfinite(init_scores)]))
        step_scores = []
        for t in range(1, self.T):
            sc = self.log_A + self.log_b[t][None, :]
            self._off[t] = float(np.max(sc[np.isfinite(sc)]))
            step_scores.append(sc - self._off[t])
        self.total_offset = float(self._off.sum())
        self._init_shifted = init_scores - self._off[0]
        self._step_shifted = step_scores

        # default budget: enough for every finite-score path (max shifted step bits summed) -> no truncation
        if budget_bits is None:
            worst = 0.0
            for sc in (self._init_shifted, *self._step_shifted):
                finite = sc[np.isfinite(sc)]
                worst += float(np.max(-finite / _LOG2)) if finite.size else 0.0
            budget_bits = worst + 1.0
        self._budget_fb = max(1, int(math.ceil(float(budget_bits) * self.quantizer.fine_per_bit())))
        self._build()

    def _fine(self, shifted_score: np.ndarray) -> np.ndarray:
        scale = self.quantizer.oversample / self.quantizer.bin_width_bits
        bits = np.where(np.isfinite(shifted_score), np.maximum(0.0, -shifted_score / _LOG2), np.inf)
        fb = np.floor(bits * scale + _TOL)
        return np.where(np.isfinite(fb), fb, -1).astype(np.int64)  # -1 marks an impossible move

    def bucket_of(self, log_joint: float) -> int:
        """The index's bucket frame for a true joint score (bits behind the per-position optimum)."""
        bits = max(0.0, -(float(log_joint) - self.total_offset) / _LOG2)
        return int(math.floor(bits * self.quantizer.fine_per_bit() + _TOL))

    def _build(self) -> None:
        W = self._budget_fb + 1
        # quantized step buckets: init0[s]; step[t, s, s'] for the move s -> s' at position t
        self._init_fb = self._fine(self._init_shifted)
        self._step_fb = (
            np.stack([self._fine(sc) for sc in self._step_shifted], axis=0)
            if self.T > 1
            else np.zeros((0, self.K, self.K), dtype=np.int64)
        )
        # forward count DP over buckets; alpha[t][s] is a length-W float64 vector (bucket -> #prefixes)
        alpha = np.zeros((self.T, self.K, W), dtype=np.float64)
        for s in range(self.K):
            fb = int(self._init_fb[s])
            if 0 <= fb <= self._budget_fb:
                alpha[0, s, fb] = 1.0
        self.truncated = bool((self._init_fb > self._budget_fb).any())
        for t in range(1, self.T):
            for sp in range(self.K):
                acc = alpha[t, sp]
                for s in range(self.K):
                    fb = int(self._step_fb[t - 1, s, sp])
                    if fb < 0:
                        continue  # impossible move
                    if fb > self._budget_fb:
                        self.truncated = True
                        continue
                    src = alpha[t - 1, s]
                    if fb:
                        acc[fb:] += src[: W - fb]
                        if src[W - fb :].any():
                            self.truncated = True
                    else:
                        acc += src
        self._alpha = alpha
        self._final = alpha[self.T - 1].sum(axis=0)  # pooled bucket counts over final states
        self._cum = np.cumsum(self._final)

    # -- whole-index reads ----------------------------------------------------------------------------------

    def total(self) -> float:
        """Number of state paths within the budget (== K**T when nothing truncated; float64 counts)."""
        return float(self._final.sum())

    def count(self, min_log_joint: float) -> float:
        """How many paths have quantized joint log-probability at least ``min_log_joint``.

        Counts every true qualifier (the structural bucket never over-states a path's bits) plus at
        most the paths within the ``T``-floor smear band below the threshold.
        """
        fb = min(self.bucket_of(min_log_joint), self._budget_fb)
        return float(self._cum[fb])

    def mass_above(self, min_log_joint: float) -> tuple[float, float]:
        """A ``(lower, upper)`` bracket on the total joint probability/density of paths above the threshold.

        Bucket arithmetic with the ``T``-floor smear: a path in bucket ``b`` carries between
        ``exp(total_offset) * 2**-((b + T) / fpb)`` and ``exp(total_offset) * 2**-(b / fpb)`` of joint
        mass (the offset restores the per-position shift, so unnormalized emission likelihoods work).
        """
        fb = min(self.bucket_of(min_log_joint), self._budget_fb)
        per_bit = self.quantizer.fine_per_bit()
        buckets = np.arange(fb + 1, dtype=float)
        c = self._final[: fb + 1]
        hi = float((c * np.exp(self.total_offset - buckets / per_bit * _LOG2)).sum())
        lo = float((c * np.exp(self.total_offset - (buckets + self.T) / per_bit * _LOG2)).sum())
        return lo, hi

    # -- random access --------------------------------------------------------------------------------------

    def unrank(self, i: int) -> tuple[tuple[int, ...], float]:
        """The ``i``-th best state path by quantized score (0-based) and its exact joint log-probability.

        One backward table walk -- ``O(T * K)`` -- regardless of how deep ``i`` is.
        """
        if i < 0:
            raise IndexError("rank must be >= 0")
        if i >= self._cum[-1]:
            raise IndexError("rank %d beyond the indexed paths (total %.6g)" % (i, float(self._cum[-1])))
        bucket = int(np.searchsorted(self._cum, float(i), side="right"))
        offset = float(i) - (float(self._cum[bucket - 1]) if bucket else 0.0)

        # final state: walk states in index order inside this bucket
        state = -1
        for s in range(self.K):
            c = float(self._alpha[self.T - 1, s, bucket])
            if offset < c:
                state = s
                break
            offset -= c
        path = [state]
        remaining = bucket
        for t in range(self.T - 1, 0, -1):
            chosen = -1
            for s in range(self.K):
                fb = int(self._step_fb[t - 1, s, state])
                if fb < 0 or fb > remaining:
                    continue
                c = float(self._alpha[t - 1, s, remaining - fb])
                if offset < c:
                    chosen = s
                    break
                offset -= c
            if chosen < 0:  # numerical crumbs from float counts: take the last viable predecessor
                for s in range(self.K - 1, -1, -1):
                    fb = int(self._step_fb[t - 1, s, state])
                    if 0 <= fb <= remaining and self._alpha[t - 1, s, remaining - fb] > 0:
                        chosen = s
                        break
            remaining -= int(self._step_fb[t - 1, chosen, state])
            path.append(chosen)
            state = chosen
        path.reverse()
        lp = float(self.log_pi[path[0]] + self.log_b[0, path[0]])
        for t in range(1, self.T):
            lp += float(self.log_A[path[t - 1], path[t]] + self.log_b[t, path[t]])
        return tuple(path), lp

    def threshold(self, rank: int) -> float:
        """Exact joint log-probability of the ``rank``-th best (quantized-order) path."""
        if rank < 1:
            raise ValueError("rank must be >= 1")
        _path, lp = self.unrank(rank - 1)
        return lp

    def iter_paths(self, start: int = 0) -> Iterator[tuple[tuple[int, ...], float]]:
        """Iterate paths from quantized rank ``start`` (sequential unranks over the stored tables)."""
        n = int(self._cum[-1])
        for i in range(int(start), n):
            yield self.unrank(i)
