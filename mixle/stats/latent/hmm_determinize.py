"""Weighted determinization of a quantized terminal HMM, and exact n-best-strings over the result.

An ambiguous HMM assigns a sequence's probability as a sum over its state paths, so naively ranking by
best path gives the n-best *paths*, not the n-best *sequences* (the same sequence recurs on many paths).
The fix is the standard one: determinize first, then rank. Determinization rebuilds the machine over
*belief states* (normalized forward vectors) -- new states that factor the duplicated mass out of the
originals -- so the result is deterministic (one path per sequence) and each edge weight is the exact
conditional probability; products of edge weights are the exact marginals. Ranking the deterministic
machine then yields duplicate-free, exact n-best *sequences*.

This is textbook weighted-automata theory, implemented natively so mixle stays self-contained:
  * weighted determinization + the twins property characterizing when it terminates --
    Mohri, "On the Determinization of Weighted Finite Automata", SIAM J. Comput. 1997;
  * removing duplicate hypotheses by determinizing *before* the n-best step --
    Mohri & Riley, "An Efficient Algorithm for the n-Best-Strings Problem", ICSLP 2002.

Termination: the belief orbit is finite iff the automaton satisfies the twins property (always true for
acyclic / bounded-length HMMs). When it is not -- e.g. an ergodic self-loop chain whose belief drifts
through a new point per prefix -- the expansion does not terminate; we cap it and raise EnumerationError
(the caller then keeps the exact O(index) enumerate-and-bin path on the original HMM).

Exact arithmetic: belief states are compared for equality, so the expansion is done in exact rationals
(``fractions.Fraction``) derived from the quantized HMM's integer exponents; float beliefs would never
compare equal and the expansion would never close.
"""

from __future__ import annotations

import heapq
import itertools
import math
from fractions import Fraction
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    EnumerationError,
    SequenceEncodableProbabilityDistribution,
    child_enumerator,
)

STRUCTURAL_ZERO = -1


def _row_probs(exponents: np.ndarray, theta: Fraction) -> list[list[Fraction]]:
    """Exact row-normalized probabilities theta^k / sum_j theta^{k_j} (negative exponent -> 0)."""
    out = []
    for row in np.asarray(exponents):
        terms = [theta ** int(k) if int(k) >= 0 else Fraction(0) for k in row]
        z = sum(terms)
        out.append([t / z if z != 0 else Fraction(0) for t in terms])
    return out


def determinize_quantized_terminal(dist, max_states: int = 1 << 16):
    """Determinize a terminal-value quantized HMM into a :class:`DeterminizedSequenceDistribution`.

    Raises EnumerationError if the HMM has no terminal_values, or if the belief expansion exceeds
    ``max_states`` (the twins property fails -- not finitely determinizable)."""
    tv = getattr(dist, "terminal_values", None)
    if tv is None:
        raise EnumerationError(dist, reason="determinization here is for the terminal_values (stopping-time) HMM")
    tv = set(tv)
    n = dist.n_states
    theta = Fraction(dist.theta).limit_denominator(10**12)
    A = _row_probs(dist.transition_exponents, theta)
    E = _row_probs(dist.emission_exponents, theta)
    levels = list(dist.levels)
    if getattr(dist, "initial_exponents", None) is None:
        raise EnumerationError(dist, reason="determinization requires quantized initial_exponents")
    init = _row_probs([dist.initial_exponents], theta)[0]

    return _build_machine(A, E, init, levels, tv, getattr(dist, "name", None), max_states, dist)


def determinize_terminal_hmm(dist, max_states: int = 1 << 16, max_denominator: int = 10**9):
    """Determinize a GENERAL terminal-value HMM (any transitions/initial, finite-discrete emissions).

    Belief equality must be decidable, so the model's float probabilities are first rationalized
    (``Fraction(p).limit_denominator(max_denominator)``) and the determinization is exact on that
    rationalized model -- which equals the original to the rationalization precision. Requires
    terminal_values and enumerable finite emissions. Raises EnumerationError if not finitely
    determinizable within ``max_states`` or if an emission support is not finite/enumerable."""
    tv = getattr(dist, "terminal_values", None)
    if tv is None:
        raise EnumerationError(dist, reason="determinization here is for the terminal_values (stopping-time) HMM")
    tv = set(tv)
    n = dist.n_states
    md = max_denominator
    A = [[Fraction(float(dist.transitions[s][sp])).limit_denominator(md) for sp in range(n)] for s in range(n)]
    init = [Fraction(float(w)).limit_denominator(md) for w in dist.w]

    cap = 1 << 16
    supports: list[dict] = []
    level_ix: dict[Any, int] = {}
    for s in range(n):
        sup: dict[Any, Fraction] = {}
        cnt = 0
        try:
            for v, lp in child_enumerator(dist.topics[s], "topics[%d]" % s):
                cnt += 1
                if cnt > cap:
                    raise EnumerationError(dist, reason="emission support too large to determinize")
                if lp != -np.inf:
                    sup[v] = Fraction(float(np.exp(lp))).limit_denominator(md)
                    level_ix.setdefault(v, len(level_ix))
        except EnumerationError:
            raise
        except Exception as exc:  # non-enumerable / continuous emission
            raise EnumerationError(dist, reason="emission is not finite/enumerable: %s" % exc) from exc
        supports.append(sup)
    levels = list(level_ix)
    E = [[supports[s].get(v, Fraction(0)) for v in levels] for s in range(n)]
    return _build_machine(A, E, init, levels, tv, getattr(dist, "name", None), max_states, dist)


def _build_machine(A, E, init, levels, tv, name, max_states, src):
    """Mohri belief-state determinization over exact rationals -> DeterminizedSequenceDistribution.

    Belief state = predictive distribution P(z_t | x_<t); emit (no transition) for the conditional, then
    transition the posterior for the next belief. New belief states factor the duplicated mass out of the
    originals; the result is deterministic. Raises EnumerationError past ``max_states``."""
    n = len(init)
    start = tuple(init)
    ids: dict[tuple, int] = {start: 0}
    trans: list[dict[Any, tuple[float, int]]] = []
    accept: list[dict[Any, float]] = []
    frontier = [start]
    order = [start]
    while frontier:
        nxt = []
        for q in frontier:
            tq: dict[Any, tuple[float, int]] = {}
            aq: dict[Any, float] = {}
            for vi, x in enumerate(levels):
                ap = [q[s] * E[s][vi] for s in range(n)]
                c = sum(ap)
                if c == 0:
                    continue
                if x in tv:
                    aq[x] = math.log(float(c))  # terminal emission completes the sequence
                else:
                    post = [a / c for a in ap]
                    nb = tuple(sum(post[s] * A[s][sp] for s in range(n)) for sp in range(n))
                    if nb not in ids:
                        if len(ids) >= max_states:
                            raise EnumerationError(
                                src, reason="belief expansion exceeded max_states (not finitely determinizable)"
                            )
                        ids[nb] = len(order)
                        order.append(nb)
                        nxt.append(nb)
                    tq[x] = (math.log(float(c)), ids[nb])
            trans.append(tq)
            accept.append(aq)
        frontier = nxt
    return DeterminizedSequenceDistribution(trans, accept, name=name)


class DeterminizedSequenceDistribution(SequenceEncodableProbabilityDistribution):
    """A deterministic weighted machine over terminal-ended sequences (one path per sequence).

    ``trans[q][x] = (log_weight, next_state)`` for a non-terminal symbol; ``accept[q][x] = log_weight``
    for a terminal symbol that completes the sequence. ``log_density(x)`` is the unique path's summed
    log-weight (== the original HMM's exact marginal); enumeration yields exact, duplicate-free
    n-best-strings."""

    def __init__(self, trans, accept, name: str | None = None) -> None:
        self.trans = trans
        self.accept = accept
        self.n_det_states = len(trans)
        self.name = name

    def __str__(self) -> str:
        return "DeterminizedSequenceDistribution(states=%d, name=%s)" % (self.n_det_states, repr(self.name))

    def density(self, x) -> float:
        """Return the probability of a terminal-ended sequence."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x) -> float:
        """Return the unique deterministic path log-weight for a sequence."""
        if not x:
            return -np.inf
        q = 0
        lp = 0.0
        for sym in x[:-1]:
            edge = self.trans[q].get(sym)
            if edge is None:
                return -np.inf
            w, q = edge
            lp += w
        a = self.accept[q].get(x[-1])
        return -np.inf if a is None else lp + a

    def seq_log_density(self, x) -> np.ndarray:
        """Return log-densities for a batch of encoded sequences."""
        return np.array([self.log_density(s) for s in x], dtype=float)

    def dist_to_encoder(self) -> DeterminizedDataEncoder:
        """Return the passthrough sequence encoder for determinized machines."""
        return DeterminizedDataEncoder()

    def enumerator(self) -> DeterminizedEnumerator:
        """Return an exact descending-probability enumerator."""
        return DeterminizedEnumerator(self)

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """Structural count index over the deterministic machine -> sub-linear exact seek/rank.

        The machine is deterministic (one path per sequence), so a forward reach DP over states counts
        each sequence exactly once at its (floored) cost: reach_by_len[t][q] is the count histogram of
        length-t non-terminal prefixes reaching state q; an accept edge then completes a length-(t+1)
        sequence. Unranking walks the reach DP backward through the (precomputed) incoming edges. Exact up
        to quantization tie granularity; no path over-count (the determinization already removed it)."""
        from mixle.enumeration.quantization.core import CountHistogram, CountIndex

        mfb = max_fine_bucket
        fb = quantizer.fine_bucket
        reach_by_len: list[dict[int, CountHistogram]] = [{0: CountHistogram.delta(0, 1)}]
        total = CountHistogram.empty()
        contributing: list[tuple[int, int, Any, CountHistogram]] = []
        cap = 1 << 20
        truncated = False  # reliable: True iff any sequence was dropped beyond mfb (more exist deeper)
        t = 0
        while t < cap and reach_by_len[t]:
            cur = reach_by_len[t]
            for q, h in cur.items():
                for sym, w in self.accept[q].items():  # terminal completion -> a sequence of length t+1
                    sh = h.shift(fb(w))
                    mb = sh.max_bucket()
                    if mb is not None and mb > mfb:
                        truncated = True
                    comp = sh.truncate(mfb)
                    if not comp.is_empty():
                        total = total.add(comp)
                        contributing.append((t, q, sym, comp))
            nxt: dict[int, CountHistogram] = {}
            for q, h in cur.items():
                for _sym, (w, nb) in self.trans[q].items():
                    sh = h.shift(fb(w))
                    mb = sh.max_bucket()
                    if mb is not None and mb > mfb:
                        truncated = True
                    hh = sh.truncate(mfb)
                    if hh.is_empty():
                        continue
                    nxt[nb] = nxt[nb].add(hh) if nb in nxt else hh
            reach_by_len.append(nxt)
            t += 1
        if t >= cap:
            truncated = True

        incoming: dict[int, list[tuple[int, Any, float]]] = {}
        for qp in range(self.n_det_states):
            for sym, (w, nb) in self.trans[qp].items():
                incoming.setdefault(nb, []).append((qp, sym, w))

        def unrank(length: int, q: int, sym: Any, b: int, o: int) -> tuple[list[Any], float]:
            seq: list[Any] = [None] * (length + 1)
            seq[length] = sym
            cb = b - fb(self.accept[q][sym])
            cq, ct, co = q, length, o
            while ct > 0:
                picked = False
                for qp, s, w in incoming.get(cq, ()):  # walk one non-terminal symbol backward
                    ph = reach_by_len[ct - 1].get(qp)
                    if ph is None:
                        continue
                    c = ph.count_at(cb - fb(w))
                    if c == 0:
                        continue
                    if co < c:
                        seq[ct - 1] = s
                        cq, cb, ct, co = qp, cb - fb(w), ct - 1, co
                        picked = True
                        break
                    co -= c
                if not picked:
                    raise IndexError("offset outside determinized prefix")
            return seq, self.log_density(seq)

        def getter(fbk: int, off: int) -> tuple[Any, float]:
            o = int(off)
            for length, q, sym, comp in contributing:
                c = comp.count_at(fbk)
                if o < c:
                    return unrank(length, q, sym, fbk, o)
                o -= c
            raise IndexError("offset outside determinized fine bucket %d" % fbk)

        return CountIndex(total, getter), truncated

    def sampler(self, seed: int | None = None) -> DeterminizedSampler:
        """Return a sampler that walks the deterministic weighted machine."""
        return DeterminizedSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None):
        """Raise because determinized machines are derived views, not fitted families."""
        # a determinized machine is derived from a fitted HMM, not estimated from data
        raise NotImplementedError("DeterminizedSequenceDistribution is a derived view; fit the source HMM instead")


class DeterminizedSampler:
    """Generative sampler: at each state the accept+transition edge weights are the conditional next-symbol
    distribution (they sum to 1), so walk it until a terminal (accept) edge is taken."""

    def __init__(self, dist: DeterminizedSequenceDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def _one(self) -> list[Any]:
        q, out = 0, []
        while True:
            syms, probs, nxt = [], [], []
            for x, w in self.dist.accept[q].items():
                syms.append(x)
                probs.append(math.exp(w))
                nxt.append(-1)
            for x, (w, nb) in self.dist.trans[q].items():
                syms.append(x)
                probs.append(math.exp(w))
                nxt.append(nb)
            p = np.array(probs)
            i = self.rng.choice(len(syms), p=p / p.sum())
            out.append(syms[i])
            if nxt[i] == -1:
                return out
            q = nxt[i]

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one terminal-ended sequence or a list of independent sequences."""
        if size is None:
            return self._one()
        return [self._one() for _ in range(size)]


class DeterminizedDataEncoder(DataSequenceEncoder):
    """Passthrough encoder for determinized sequence observations."""

    def __str__(self) -> str:
        return "DeterminizedDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DeterminizedDataEncoder)

    def seq_encode(self, x):
        """Return sequence observations as a list without transforming symbols."""
        return list(x)


class DeterminizedEnumerator(DistributionEnumerator):
    """Exact descending-probability enumeration of sequences over the deterministic machine (n-best
    strings). Best-first with an admissible per-state best-completion bound (Viterbi over the machine)."""

    def __init__(self, dist: DeterminizedSequenceDistribution) -> None:
        super().__init__(dist)
        self.trans = dist.trans
        self.accept = dist.accept
        n = dist.n_det_states
        # beta[q] = best (max) completion log-prob from state q; value iteration (converges for a finite,
        # i.e. determinizable, machine).
        beta = [max(aq.values(), default=-np.inf) for aq in self.accept]
        for _ in range(n + 1):
            changed = False
            for q in range(n):
                b = beta[q]
                for _x, (w, nb) in self.trans[q].items():
                    cand = w + beta[nb]
                    if cand > b:
                        b = cand
                if b > beta[q]:
                    beta[q] = b
                    changed = True
            if not changed:
                break
        self._beta = beta
        self._gen = self._iter()

    def _iter(self):
        counter = itertools.count()
        heap = []

        def push(state, prefix, lp):
            bound = lp + self._beta[state]
            if bound > -np.inf:
                heapq.heappush(heap, (-bound, next(counter), state, prefix, lp))

        push(0, (), 0.0)
        while heap:
            neg, _, state, prefix, lp = heapq.heappop(heap)
            if state == -1:  # a completed sequence; popped in exact descending-probability order
                yield list(prefix), lp
                continue
            for x, w in self.accept[state].items():  # terminal emissions complete the sequence
                heapq.heappush(heap, (-(lp + w), next(counter), -1, prefix + (x,), lp + w))
            for x, (w, nb) in self.trans[state].items():  # non-terminal emissions continue it
                push(nb, prefix + (x,), lp + w)

    def __next__(self):
        return next(self._gen)

    def seek(self, index: int):
        """Sub-linear exact seek over the deterministic machine (no prefix enumeration).

        Deepens the structural count index until ``index`` is covered or the support is provably exhausted
        (the count-DP's reliable ``truncated`` flag), then unranks within the located fine bucket. Robust to
        gaps in the cost spectrum, unlike the shared total-growth deepening heuristic."""
        from mixle.enumeration.quantization.core import Quantizer

        if index < 0:
            raise IndexError("index must be non-negative")
        q = Quantizer(bin_width_bits=1.0, oversample=64)
        mfb = 128
        while True:
            ci, truncated = self.dist.quantized_count_index(q, mfb)
            hist = ci.hist
            cum = 0
            for b in range(hist.base, hist.base + len(hist.data)):  # ascending bucket = descending prob
                cnt = hist.count_at(b)
                if cum + cnt > index:
                    value, _lp = ci.get_in_bucket(b, index - cum)
                    return _DetSeekResult(value)
                cum += cnt
            if not truncated:
                raise IndexError("index %d is beyond the support count %d" % (index, cum))
            mfb *= 2


class _DetSeekResult:
    __slots__ = ("value",)

    def __init__(self, value) -> None:
        self.value = value
