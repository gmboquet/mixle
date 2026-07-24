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
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.enumeration.model_enumeration import best_first
from mixle.enumeration.quantization.core import _TOL, Quantizer
from mixle.enumeration.quantization.seek import _require_index

_LOG2 = math.log(2.0)
_NORM_TOL = 1.0e-6  # absolute tolerance on logsumexp(log_pi | a log_A row) deviating from 0 (normalized)


def _logsumexp_1d(scores: np.ndarray) -> float:
    """``log(sum(exp(scores)))``, robust to -inf entries (an all-impossible slice is exactly -inf)."""
    m = float(np.max(scores)) if scores.size else -math.inf
    if not math.isfinite(m):
        return m  # all -inf (or empty): the sum is 0, so the log is exactly -inf (NaN/+inf rejected earlier)
    return m + math.log(float(np.sum(np.exp(scores - m))))


def _check_finite_or_impossible(name: str, arr: np.ndarray) -> None:
    """Reject NaN and +inf; -inf is the valid "impossible" sentinel (MXR-080-0228).

    Every score in this module is either a genuine, finite log-probability/log-likelihood or the
    explicit -inf marker for "cannot happen". NaN is neither -- silently treating it as -inf, as a
    bare ``np.isfinite`` filter would, launders a caller's corrupted/garbage input into a confident
    "impossible" claim instead of surfacing the bug. +inf is not a valid log-anything (it would
    assert infinite likelihood).
    """
    if np.isnan(arr).any():
        raise ValueError(f"{name} must not contain NaN.")
    if np.isposinf(arr).any():
        raise ValueError(f"{name} must be finite or -inf (impossible), never +inf.")


def _check_log_probabilities(name: str, arr: np.ndarray) -> None:
    """Reject entries > 0: a log-probability's exponential is a probability, which cannot exceed 1."""
    if (arr > 0.0).any():
        raise ValueError(f"{name} entries must be <= 0 (a probability cannot exceed 1); found a positive entry.")


def _check_not_superstochastic(name: str, log_probs: np.ndarray) -> None:
    """Reject a categorical row that describes MORE than one unit of total probability.

    ``log_probs`` is one distribution's worth of log-probabilities (``log_pi`` itself, or one row of
    ``log_A``). Summing to more than 1 (``logsumexp > 0``) is a genuine impossibility -- exactly like
    a single entry > 0 (:func:`_check_log_probabilities`) -- and always rejected.

    Summing to LESS than 1 is deliberately allowed: this package already relies on sub-stochastic
    rows as the supported way to forbid individual transitions without renormalizing the rest of the
    row (set one entry to -inf and leave the others as they were -- see e.g. the impossible-transition
    tests, which remove a substantial, non-numerical-noise fraction of a row's mass this way). A row
    that is entirely -inf is simply the zero-probability extreme of that same allowed range -- an
    explicit "empty support"/"dead end" marker (MXR-080-0228; see :class:`HMMPathIndex`'s
    ``empty_support``), not a case requiring its own branch: ``_logsumexp_1d`` returns exactly -inf
    for it, which is <= the tolerance below like any other sub-stochastic row.
    """
    lse = _logsumexp_1d(log_probs)
    if lse > _NORM_TOL:
        raise ValueError(
            f"{name} must not describe more than one unit of total probability (logsumexp <= 0); "
            f"got logsumexp={lse!r} (total probability {math.exp(lse)!r})."
        )


def _validate_hmm_model(log_pi: np.ndarray, log_A: np.ndarray, log_b: np.ndarray) -> None:
    """Validate ``log_pi``/``log_A``/``log_b`` form a coherent, well-posed HMM (MXR-080-0228).

    Shape: ``log_pi`` is ``(K,)``, ``log_A`` is square ``(K, K)``, ``log_b`` is ``(T, K)``, all
    sharing the same ``K`` -- checked explicitly and reported with the actual shapes on mismatch,
    instead of deferring to whatever opaque numpy broadcast error (or, worse, a silently
    broadcast-compatible-but-wrong result) a mismatched combination happens to trigger deeper in the
    pipeline.

    Numeric contract: ``log_pi`` and ``log_A`` are genuine categorical log-probabilities over
    discrete latent states -- every entry <= 0, never NaN/+inf, and describing at most one unit of
    total probability (via :func:`_check_not_superstochastic`: sub-stochastic rows, including an
    all-impossible row/vector, are allowed -- see that function's docstring for why). ``log_b`` is an
    emission LOG-LIKELIHOOD, not a probability: the class docstring and :func:`hmm_best_paths` are
    explicit that emission densities may be positive (unnormalized/continuous densities > 1 are
    legitimate), so ``log_b`` is only required to be finite-or-impossible (never NaN, never +inf)
    with no <= 0 or normalization requirement.
    """
    if log_pi.ndim != 1:
        raise ValueError(f"log_pi must be 1-D (K,), got shape {log_pi.shape}.")
    if log_A.ndim != 2 or log_A.shape[0] != log_A.shape[1]:
        raise ValueError(f"log_A must be square 2-D (K, K), got shape {log_A.shape}.")
    if log_b.ndim != 2:
        raise ValueError(f"log_b must be 2-D (T, K), got shape {log_b.shape}.")
    k_pi, k_a, k_b = log_pi.shape[0], log_A.shape[0], log_b.shape[1]
    if not (k_pi == k_a == k_b):
        raise ValueError(
            "log_pi, log_A, and log_b disagree on the number of states K: "
            f"log_pi is {log_pi.shape} (K={k_pi}), log_A is {log_A.shape} (K={k_a}), "
            f"log_b is {log_b.shape} (T, K={k_b}); all three must share the same K."
        )
    if k_pi == 0:
        raise ValueError("log_pi/log_A/log_b must have at least one state (K >= 1).")

    _check_finite_or_impossible("log_pi", log_pi)
    _check_finite_or_impossible("log_A", log_A)
    _check_finite_or_impossible("log_b", log_b)
    _check_log_probabilities("log_pi", log_pi)
    _check_log_probabilities("log_A", log_A)
    _check_not_superstochastic("log_pi", log_pi)
    for s in range(k_a):
        _check_not_superstochastic(f"log_A row {s}", log_A[s])


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
    _validate_hmm_model(log_pi, log_A, log_b)
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


@dataclass
class CertifiedCount:
    """A path count from :class:`HMMPathIndex`, honest about whether it is the complete answer.

    ``HMMPathIndex`` is built to a finite quantized-bucket budget; when a query needs to look beyond
    that budget on a genuinely truncated index, the stored tables cannot answer it exactly -- paths
    beyond the budget were never counted. Returning the in-budget count as a bare number would look
    identical to a complete answer with no way to tell the two apart (MXR-080-0230); this return type
    makes the distinction explicit instead.

    Attributes:
        value: the count of indexed (in-budget) paths satisfying the query. Always a sound LOWER
            bound on the true count -- every path counted here genuinely qualifies, so this can only
            under-, never over-, state the truth. Exact (up to the ordinary quantization smear
            documented on :meth:`HMMPathIndex.count`, unrelated to truncation) when ``certified``.
        certified: True when ``value`` is provably the complete answer -- no path beyond the built
            budget could also qualify. False when the query reaches beyond the built budget on a
            truncated index, so additional qualifying paths may exist beyond ``value``.
    """

    value: int
    certified: bool


@dataclass
class CertifiedMassBound:
    """A joint probability/density mass bracket from :meth:`HMMPathIndex.mass_above`.

    Attributes:
        lower: sound lower bound on the true mass above the threshold. Valid regardless of
            ``certified`` -- paths beyond a truncated budget can only ADD mass, never invalidate a
            lower bound derived purely from the paths this index actually has.
        upper: sound upper bound when ``certified``; ``math.inf`` when not. A truncated index's
            stored histogram cannot certify any FINITE upper bound on mass that might live beyond the
            built budget without deepening it -- reporting the in-budget figure anyway would be an
            unsound bound wearing a sound bound's shape (MXR-080-0230).
        certified: True when both bounds are provably sound answers to the exact query; False when
            the threshold reaches beyond the built budget on a truncated index.
    """

    lower: float
    upper: float
    certified: bool


class HMMPathIndex:
    """Quantized random-access index over an HMM's state paths for one observation sequence.

    ``log_pi``/``log_A``/``log_b`` are validated at construction (MXR-080-0228): shapes must be
    ``(K,)``/``(K, K)``/``(T, K)`` with a consistent ``K``, every score must be finite-or-impossible
    (``-inf``, never NaN or ``+inf``), and ``log_pi``/``log_A`` must describe at most one unit of
    total probability each (sub-stochastic rows -- e.g. from forbidding individual transitions
    without renormalizing -- are allowed; ``log_b`` is an emission LOG-LIKELIHOOD and may be
    positive/unnormalized -- see :func:`hmm_best_paths`). An all-impossible initial vector or a
    position with no finite score at all is not an error: it is represented explicitly via
    ``empty_support`` (and ``total()`` reporting a certified zero), rather than crashing.

    **Precompute** (once, ``O(T * K^2 * W)``): quantize every step score -- ``log_pi[s] + log_b[0, s]``
    and ``log_A[s, s'] + log_b[t, s']``, one floor per step so the accumulated smear is at most ``T``
    fine buckets -- and run a forward count DP: ``C_t[s']`` is the histogram, over integer total-score
    buckets, of the number of length-``t+1`` prefixes ending in state ``s'``. Counts are EXACT Python
    arbitrary-precision integers, never float64 (MXR-080-0229): the number of paths is ``K**T``, which
    exceeds float64's 2**53 exact-integer range for even modest ``T``, and every operation this DP
    performs on a count is addition -- no rounding is ever actually needed here.

    **Query** (each ``O(T * K)``): ``unrank(i)`` walks the stored tables backward -- pick the final
    state whose bucket count covers the offset, then repeatedly the predecessor whose shifted prefix
    bucket does -- returning the ``i``-th best path *by quantized score* and its exact joint
    log-probability, exactly for any ``i`` within the built support. ``total`` / ``count`` /
    ``mass_above`` read the pooled final histogram and report a :class:`CertifiedCount` /
    :class:`CertifiedMassBound`: a query that reaches beyond a truncated index's built budget gets a
    sound bound with ``certified=False`` rather than a bare number indistinguishable from a complete
    answer (MXR-080-0230).

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
        _validate_hmm_model(self.log_pi, self.log_A, self.log_b)
        self.T, self.K = self.log_b.shape
        if self.T == 0:
            raise ValueError("log_b must cover at least one position")
        self.quantizer = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)

        # Emission log-likelihoods may be POSITIVE (continuous densities), so raw joint scores are not
        # bounded by 0. Shift each position's step scores by that position's best (max) score: shifted
        # scores are <= 0, buckets measure "bits behind the per-step optimum", and ordering is unchanged
        # (the same constant shifts every path). ``total_offset`` converts bucket <-> true joint score.
        #
        # A position can have NO finite score at all -- e.g. log_b[t] is -inf for every state the
        # model could plausibly occupy there -- which makes every path through this index have
        # probability exactly zero (MXR-080-0228). np.max has no identity for an empty array and
        # would crash; guard it explicitly and record ``empty_support`` instead, leaving that
        # position's offset at its zero-initialized default (harmless: the unshifted scores are
        # already all -inf, and -inf - 0.0 stays -inf, so _fine() below still correctly marks every
        # move through this position "impossible", which is exactly what propagates to total()==0).
        init_scores = self.log_pi + self.log_b[0]
        self._off = np.zeros(self.T, dtype=float)
        self.empty_support = False
        init_finite = np.isfinite(init_scores)
        if init_finite.any():
            self._off[0] = float(np.max(init_scores[init_finite]))
        else:
            self.empty_support = True
        step_scores = []
        for t in range(1, self.T):
            sc = self.log_A + self.log_b[t][None, :]
            finite = np.isfinite(sc)
            if finite.any():
                self._off[t] = float(np.max(sc[finite]))
            else:
                self.empty_support = True
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
        # forward count DP over buckets; alpha[t][s] is a length-W vector of EXACT Python-int counts
        # (arbitrary precision, never float64 -- MXR-080-0229). The number of paths is up to K**T:
        # for K=2 that already exceeds float64's 2**53 exact-integer range past T=53, at which point
        # adjacent integer ranks collapse onto the same float64 value -- total()/unrank() would then
        # disagree about the support, and random access would misroute. Every operation this DP
        # performs on a count -- seeding 0/1, elementwise +=, sum, cumsum -- is exact addition, never
        # multiplication or rounding, so exactness costs only the constant factor of Python-int
        # arithmetic (the same tradeoff CountHistogram's exact=True mode makes structurally elsewhere
        # in this package; see quantization/core.py).
        alpha = np.zeros((self.T, self.K, W), dtype=object)
        for s in range(self.K):
            fb = int(self._init_fb[s])
            if 0 <= fb <= self._budget_fb:
                alpha[0, s, fb] = 1
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
        self._final = alpha[self.T - 1].sum(axis=0)  # pooled bucket counts over final states (exact ints)
        self._cum = np.cumsum(self._final)

    # -- whole-index reads ----------------------------------------------------------------------------------

    def total(self) -> CertifiedCount:
        """Number of state paths within the budget (== K**T when nothing truncated; exact integer count).

        ``certified`` is False when the index is truncated: ``value`` is then only a sound LOWER
        bound on the true number of paths -- some paths exceed the built budget and were never
        counted (MXR-080-0230). Deepen ``budget_bits`` (or omit it for the default, which is sized to
        cover every finite-score path) to certify a larger/complete total.
        """
        return CertifiedCount(value=int(self._final.sum()), certified=not self.truncated)

    def count(self, min_log_joint: float) -> CertifiedCount:
        """How many paths have quantized joint log-probability at least ``min_log_joint``.

        Counts every true qualifier (the structural bucket never over-states a path's bits) plus at
        most the paths within the ``T``-floor smear band below the threshold -- this ordinary
        quantization smear applies regardless of ``certified``.

        ``certified`` is True whenever this query's own bucket stays within the built budget: bucket
        costs only accumulate along a path (never decrease), so every bucket the index does hold is a
        complete, exact count regardless of the index's global ``truncated`` flag. It is False only
        when the query needs to look beyond the built budget on a genuinely truncated index
        (MXR-080-0230): ``value`` is then a sound LOWER bound (every indexed path is a real
        qualifier), not the complete answer -- more qualifying paths may live beyond the budget.
        """
        fb_raw = self.bucket_of(min_log_joint)
        clamped = fb_raw > self._budget_fb
        fb = min(fb_raw, self._budget_fb)
        certified = (not clamped) or (not self.truncated)
        return CertifiedCount(value=int(self._cum[fb]), certified=certified)

    def mass_above(self, min_log_joint: float) -> CertifiedMassBound:
        """A bracket on the total joint probability/density of paths above the threshold.

        Bucket arithmetic with the ``T``-floor smear: a path in bucket ``b`` carries between
        ``exp(total_offset) * 2**-((b + T) / fpb)`` and ``exp(total_offset) * 2**-(b / fpb)`` of joint
        mass (the offset restores the per-position shift, so unnormalized emission likelihoods work).

        ``certified`` follows the same rule as :meth:`count`: True unless this query reaches beyond
        the built budget on a truncated index. When not certified, ``lower`` remains a sound bound
        (unindexed paths can only add mass) but ``upper`` is replaced with ``math.inf`` -- the
        in-budget figure is not a valid upper bound once qualifying mass may exist beyond the budget,
        so reporting it as one would be unsound, not just imprecise (MXR-080-0230).
        """
        fb_raw = self.bucket_of(min_log_joint)
        clamped = fb_raw > self._budget_fb
        fb = min(fb_raw, self._budget_fb)
        certified = (not clamped) or (not self.truncated)
        per_bit = self.quantizer.fine_per_bit()
        buckets = np.arange(fb + 1, dtype=float)
        c = np.array(self._final[: fb + 1], dtype=float)  # counts -> float64 for this inherently-approximate mass
        hi = float((c * np.exp(self.total_offset - buckets / per_bit * _LOG2)).sum())
        lo = float((c * np.exp(self.total_offset - (buckets + self.T) / per_bit * _LOG2)).sum())
        if not certified:
            hi = math.inf
        return CertifiedMassBound(lower=lo, upper=hi, certified=certified)

    # -- random access --------------------------------------------------------------------------------------

    def unrank(self, i: int) -> tuple[tuple[int, ...], float]:
        """The ``i``-th best state path by quantized score (0-based) and its exact joint log-probability.

        One backward table walk -- ``O(T * K)`` -- regardless of how deep ``i`` is. Every table read
        along the walk is an exact Python int (MXR-080-0229), so ``i`` resolves exactly for supports
        far beyond float64's 2**53 exact-integer range -- adjacent ranks never alias to the same
        stored offset the way they would under float64 accumulation.
        """
        i = _require_index(i, label="rank")
        if i < 0:
            raise IndexError("rank must be >= 0")
        if i >= self._cum[-1]:
            raise IndexError("rank %d beyond the indexed paths (total %d)" % (i, self._cum[-1]))
        bucket = int(np.searchsorted(self._cum, i, side="right"))
        offset = i - (self._cum[bucket - 1] if bucket else 0)

        # final state: walk states in index order inside this bucket
        state = -1
        for s in range(self.K):
            c = self._alpha[self.T - 1, s, bucket]
            if offset < c:
                state = s
                break
            offset -= c
        if state < 0:
            # alpha[T-1, :, bucket].sum() == _final[bucket] exactly (both are exact-integer sums over
            # the same terms), and `offset` was bounded by that very sum via the cumsum/searchsorted
            # step above -- so exhausting every state without consuming `offset` means the
            # exact-integer bookkeeping itself is inconsistent, not an expected rounding crumb.
            raise RuntimeError(
                f"internal invariant violated: rank {i} not resolved by any final state in bucket "
                f"{bucket} (exact-integer counts should make this unreachable)"
            )
        path = [state]
        remaining = bucket
        for t in range(self.T - 1, 0, -1):
            chosen = -1
            for s in range(self.K):
                fb = int(self._step_fb[t - 1, s, state])
                if fb < 0 or fb > remaining:
                    continue
                c = self._alpha[t - 1, s, remaining - fb]
                if offset < c:
                    chosen = s
                    break
                offset -= c
            if chosen < 0:
                # same exact-integer invariant as above, one position earlier in the walk -- see there.
                raise RuntimeError(
                    f"internal invariant violated: rank {i} not resolved by any predecessor at "
                    f"position {t} (exact-integer counts should make this unreachable)"
                )
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
        rank = _require_index(rank, label="rank")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        _path, lp = self.unrank(rank - 1)
        return lp

    def iter_paths(self, start: int = 0) -> Iterator[tuple[tuple[int, ...], float]]:
        """Iterate paths from quantized rank ``start`` (sequential unranks over the stored tables)."""
        start = _require_index(start, label="start")
        if start < 0:
            raise IndexError("start must be non-negative")
        n = int(self._cum[-1])
        for i in range(start, n):
            yield self.unrank(i)
