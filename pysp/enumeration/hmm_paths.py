"""Exact descending-probability enumeration of HMM state paths (list-Viterbi / A*).

The count-budget index handles the *decomposable* families (Sequence / Composite / MarkovChain). An HMM
is non-decomposable -- the latent state couples emissions across time -- so it is not served by the
count semiring. But its *most probable paths* can still be enumerated exactly and lazily in descending
joint log-probability by A* search, with the backward Viterbi value as an admissible (in fact tight)
heuristic. This is the k-best / list-Viterbi algorithm phrased as best-first search, reusing the generic
engine in :mod:`pysp.enumeration.model_enumeration`.

For a path ``z_{1..T}`` the joint log-probability of the latent path with the observed emissions is
``log_pi[z_1] + log_b[0, z_1] + sum_{t>1} (log_A[z_{t-1}, z_t] + log_b[t, z_t])`` where ``log_b[t, k]``
is the emission log-likelihood ``log p(x_t | z_t = k)``. The completion heuristic
``h[t, s] = max over z_{t+1..T} of the remaining transition+emission score`` is computed once by a
backward max-product pass, so ``g + h`` is the exact score of the best path through ``(t, s)`` -- an
admissible bound that makes best-first emit paths in exactly nonincreasing joint log-probability.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from pysp.enumeration.model_enumeration import best_first


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
