"""Structured HMMs: a composable transition layer (dense / low-rank / combinators) + forward-backward.

A standard HMM stores a dense K x K transition matrix and the forward-backward does O(K^2) work per step.
Rich structure -- a low-rank transition, a block of independent chains, a factorial (Kronecker) product --
is hard to express that way. This module factors the transition behind a small :class:`TransitionOperator`
interface so the forward-backward only needs two primitives:

    forward(alpha)  = alpha @ A         (push a state-belief forward one step)
    backward(v)     = A @ v             (pull an emission-weighted belief back one step)

and an M-step that re-estimates the operator from expected transition mass. Any operator that implements
those plugs into the SAME forward-backward / EM. Implementations:

    * :class:`DenseTransition`     -- the usual K x K matrix (O(K^2)).
    * :class:`LowRankTransition`   -- A = G @ Phi with an inner rank r (K x r, r x K row-stochastic): each
      state mixes over r shared "transition profiles". Forward/backward and the M-step are O(K r), and the
      parameter count drops from K^2 to 2 K r. (Combinators -- block-diagonal, Kronecker/factorial -- are
      the same interface; see TransitionOperator subclasses.)

The forgetting / mixing property of an ergodic chain (beliefs forget the distant past) is what lets the
forward-backward be split into chunks and run in parallel; see ``parallel`` in this package's estimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TransitionOperator:
    """A row-stochastic state-transition operator behind the HMM forward-backward.

    Subclasses provide the two linear maps the recursions need plus an M-step from expected transition
    mass. ``forward``/``backward`` must be consistent with ``as_matrix`` (``forward(a) == a @ A``,
    ``backward(v) == A @ v``); the low-overhead operators never materialize ``A``.
    """

    n_states: int

    # Operators are required components of the serializable StructuredHMM / InputOutputHMM but are not
    # distributions or estimators themselves, so they opt in to mixle JSON serialization explicitly
    # (state round-trips via __dict__; SparseTransition's csr matrix uses the sparse codec).
    __pysp_serializable__ = True

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this operator."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransitionOperator:
        """Reconstruct an operator from ``to_dict`` output."""
        from mixle.utils.serialization import from_serializable

        rv = from_serializable(payload)
        if not isinstance(rv, cls):
            raise TypeError("decoded object is %s, not %s" % (type(rv).__name__, cls.__name__))
        return rv

    def forward(self, alpha: np.ndarray) -> np.ndarray:  # alpha @ A
        """Push a state-belief row vector one transition forward."""
        raise NotImplementedError

    def backward(self, v: np.ndarray) -> np.ndarray:  # A @ v
        """Pull an emission-weighted belief vector one transition backward."""
        raise NotImplementedError

    def as_matrix(self) -> np.ndarray:
        """Materialize the transition matrix when an explicit matrix is needed."""
        raise NotImplementedError

    # --- M-step: accumulate expected transition mass over a sequence, then re-estimate ---
    def new_accumulator(self) -> Any:
        """Create an empty transition sufficient-statistic accumulator."""
        raise NotImplementedError

    def accumulate(self, acc: Any, alpha_t: np.ndarray, w_next: np.ndarray, scale: float) -> None:
        """Add one transition's expected mass. ``alpha_t`` is the (normalized) forward belief at t,
        ``w_next = b_{t+1} * beta_{t+1}`` the emission-weighted backward belief at t+1, ``scale`` the
        forward normalizer ``c_{t+1}`` (so the per-step posterior transition mass is exact)."""
        raise NotImplementedError

    def estimate(self, acc: Any) -> TransitionOperator:
        """Estimate a transition operator of the same structural family from accumulated statistics."""
        raise NotImplementedError

    def random_accumulator(self, rng) -> Any:
        """A randomly-filled accumulator whose ``estimate`` yields a random (structured) transition --
        used to seed EM when there is no warm start. Fills ``new_accumulator`` shapes (nested) with noise."""

        def fill(a):
            return a + rng.random(a.shape) if isinstance(a, np.ndarray) else [fill(x) for x in a]

        return fill(self.new_accumulator())


def _row_normalize(m: np.ndarray) -> np.ndarray:
    m = np.maximum(m, 0.0)
    s = m.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return m / s


class DenseTransition(TransitionOperator):
    """The usual dense K x K row-stochastic transition (O(K^2) forward-backward).

    ``prior`` (a K x K pseudocount matrix) is added to the expected counts before each M-step
    re-normalization -- a Dirichlet/MAP transition. A diagonal prior is a *sticky* self-transition bias
    (see :func:`sticky_transition`); a flat prior is symmetric-Dirichlet smoothing.
    """

    def __init__(self, a: np.ndarray, prior: np.ndarray | None = None) -> None:
        self.a = np.asarray(a, dtype=float)
        self.n_states = self.a.shape[0]
        self.prior = None if prior is None else np.asarray(prior, dtype=float)

    def forward(self, alpha):
        """Push a state-belief row vector forward with the dense matrix."""
        return alpha @ self.a

    def backward(self, v):
        """Pull a vector backward with the dense matrix."""
        return self.a @ v

    def as_matrix(self):
        """Return the dense row-stochastic transition matrix."""
        return self.a

    def new_accumulator(self):
        """Create dense expected-transition-count storage."""
        return np.zeros_like(self.a)

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate one dense expected-transition-count contribution."""
        acc += np.outer(alpha_t, w_next) * (self.a / max(scale, 1e-300))

    def estimate(self, acc):
        """Estimate a row-normalized dense transition from expected counts."""
        return DenseTransition(_row_normalize(acc if self.prior is None else acc + self.prior), self.prior)


def sticky_transition(a, kappa: float) -> DenseTransition:
    """A dense transition with a STICKY self-transition prior: ``kappa`` pseudocounts on the diagonal
    favor staying in a state (longer dwell times, cleaner segmentation -- the sticky-HMM idea)."""
    a = np.asarray(a, dtype=float)
    return DenseTransition(a, prior=float(kappa) * np.eye(a.shape[0]))


def dirichlet_transition(a, alpha: float) -> DenseTransition:
    """A dense transition with a symmetric Dirichlet(``alpha``) smoothing prior on every row (MAP)."""
    a = np.asarray(a, dtype=float)
    return DenseTransition(a, prior=np.full((a.shape[0], a.shape[0]), float(alpha)))


def kron_initial(pi1, pi2) -> np.ndarray:
    """Factorized initial distribution ``pi1 (x) pi2`` for a factorial (Kronecker) HMM -- the two chains
    start independently. Matches a :class:`KroneckerTransition` so the joint initial respects the factors."""
    return np.kron(np.asarray(pi1, dtype=float), np.asarray(pi2, dtype=float))


class LowRankTransition(TransitionOperator):
    """A = G @ Phi: each state's next-state distribution is a mix of ``r`` shared transition profiles.

    ``G`` is K x r row-stochastic (state -> profile mixing), ``Phi`` is r x K row-stochastic (profile ->
    next-state). ``A = G @ Phi`` is K x K row-stochastic with rank <= r. Forward (``(alpha @ G) @ Phi``),
    backward (``G @ (Phi @ v)``) and the M-step are all O(K r) -- never forming A -- and the parameter
    count is 2 K r instead of K^2.
    """

    def __init__(self, g: np.ndarray, phi: np.ndarray) -> None:
        self.g = np.asarray(g, dtype=float)  # (K, r)
        self.phi = np.asarray(phi, dtype=float)  # (r, K)
        self.n_states = self.g.shape[0]
        self.rank = self.g.shape[1]

    def forward(self, alpha):
        """Push a state-belief row vector through the low-rank transition."""
        return (alpha @ self.g) @ self.phi  # (alpha^T A)

    def backward(self, v):
        """Pull a vector backward through the low-rank transition."""
        return self.g @ (self.phi @ v)  # (A v)

    def as_matrix(self):
        """Materialize the implied dense transition matrix."""
        return self.g @ self.phi

    def new_accumulator(self):
        """Create state-profile and profile-next sufficient-statistic storage."""
        return [np.zeros_like(self.g), np.zeros_like(self.phi)]  # [n (K,r), m (r,K)]

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate one exact low-rank transition contribution."""
        # exact expected mass of the latent profile r on this transition, in O(K r) (no K x K matrix):
        #   u[r]   = sum_i alpha_t[i] G[i,r]            (alpha into profiles)
        #   v[r]   = sum_j Phi[r,j] w_next[j]           (profiles' emission-weighted reach)
        #   n[i,r] += alpha_t[i] G[i,r] v[r] / scale    (state->profile counts, for G)
        #   m[r,j] += u[r] Phi[r,j] w_next[j] / scale   (profile->next counts, for Phi)
        inv = 1.0 / max(scale, 1e-300)
        u = alpha_t @ self.g  # (r,)
        v = self.phi @ w_next  # (r,)
        acc[0] += (alpha_t[:, None] * self.g) * v[None, :] * inv
        acc[1] += (u[:, None] * self.phi) * w_next[None, :] * inv

    def estimate(self, acc):
        """Estimate low-rank transition factors from accumulated statistics."""
        return LowRankTransition(_row_normalize(acc[0]), _row_normalize(acc[1]))


class SparseTransition(TransitionOperator):
    """Only the given ``(from, to)`` edges are allowed (left-to-right / banded HMMs). Forward, backward
    and the M-step are O(#edges) -- transitions outside the edge set stay exactly zero through EM, so the
    structure is preserved. Build edges yourself or with :func:`left_to_right_edges` / :func:`banded_edges`."""

    def __init__(self, n_states: int, edges, values=None) -> None:
        from scipy.sparse import csr_matrix

        self.n_states = int(n_states)
        self.rows = np.asarray([e[0] for e in edges], dtype=int)
        self.cols = np.asarray([e[1] for e in edges], dtype=int)
        vals = np.ones(len(self.rows)) if values is None else np.asarray(values, dtype=float)
        a = csr_matrix((np.maximum(vals, 0.0), (self.rows, self.cols)), shape=(self.n_states, self.n_states))
        rs = np.asarray(a.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        from scipy.sparse import diags

        self.a = diags(1.0 / rs) @ a  # row-normalized csr
        self._edge_vals = np.asarray(self.a[self.rows, self.cols]).ravel()

    def forward(self, alpha):
        """Push a state-belief row vector through the sparse transition."""
        return alpha @ self.a

    def backward(self, v):
        """Pull a vector backward through the sparse transition."""
        return self.a @ v

    def as_matrix(self):
        """Materialize the sparse transition as a dense matrix."""
        return np.asarray(self.a.todense())

    def new_accumulator(self):
        """Create one expected-count slot per allowed edge."""
        return np.zeros(len(self.rows))  # one expected count per allowed edge

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate one sparse expected-transition contribution."""
        acc += alpha_t[self.rows] * w_next[self.cols] * self._edge_vals / max(scale, 1e-300)

    def estimate(self, acc):
        """Estimate a sparse transition over the same allowed edge set."""
        return SparseTransition(self.n_states, list(zip(self.rows.tolist(), self.cols.tolist())), acc)


def left_to_right_edges(n_states: int, skip: int = 1):
    """Edges for a left-to-right (Bakis) HMM: each state may stay or advance up to ``skip`` states."""
    return [(i, j) for i in range(n_states) for j in range(i, min(n_states, i + skip + 1))]


def banded_edges(n_states: int, bandwidth: int = 1):
    """Edges for a banded transition: state i connects to i-bandwidth .. i+bandwidth (local time-series)."""
    return [(i, j) for i in range(n_states) for j in range(max(0, i - bandwidth), min(n_states, i + bandwidth + 1))]


def _final_state_enumerate(hmm, len_dist, max_results=50):
    """Best-first enumeration of observation sequences in descending marginal probability for a
    StructuredHMM whose sequences must END in a ``final_states`` state (e.g. an HSMM expansion).
    Admissible A*: a prefix's forward log-vector + a backward upper bound UB[r][s] (logsumexp of r further
    steps from s using each state's best emission, ending in a final state) bounds every completion, so a
    popped complete sequence is in true descending order. Needs discrete (Categorical) emissions + a
    Categorical-like ``len_dist`` (a .pmap over lengths)."""
    import heapq

    from scipy.special import logsumexp

    from mixle.enumeration import EnumerationError

    try:
        symbols = sorted(set().union(*(set(e.pmap.keys()) for e in hmm.emissions)))
        lengths = sorted(int(x) for x in len_dist.pmap.keys())
    except AttributeError as exc:
        raise EnumerationError(hmm, reason="final-state enumeration needs Categorical emissions + len_dist") from exc
    final_mask = hmm.final_mask
    log_emit = {v: np.array([float(e.log_density(v)) for e in hmm.emissions]) for v in symbols}
    max_emit = np.max(np.stack([log_emit[v] for v in symbols]), axis=0)
    log_pi = np.log(hmm.pi + 1e-300)
    log_a = np.log(hmm.transition.as_matrix() + 1e-300)
    log_len = {x: float(len_dist.log_density(x)) for x in lengths}
    l_max = max(lengths)
    ub = [np.where(final_mask, 0.0, -np.inf)]
    for _ in range(1, l_max):
        ub.append(logsumexp(log_a + (max_emit + ub[-1])[None, :], axis=1))

    def complete_score(fwd, t):
        return (logsumexp(fwd[final_mask]) + log_len[t]) if (t in log_len and final_mask.any()) else -np.inf

    def extend_ub(fwd, t):
        best = -np.inf
        for x in lengths:
            r = x - t
            if r == 0:
                best = max(best, complete_score(fwd, t))
            elif r > 0:
                best = max(best, logsumexp(fwd + ub[r]) + log_len[x])
        return best

    heap = []
    counter = 0
    for v in symbols:
        fwd = log_pi + log_emit[v]
        pri = extend_ub(fwd, 1)
        if np.isfinite(pri):
            heapq.heappush(heap, (-pri, counter, "p", (v,), fwd, 1))
            counter += 1
    out = []
    while heap and len(out) < max_results:
        neg, _, kind, prefix, fwd, t = heapq.heappop(heap)
        if kind == "c":
            out.append((list(prefix), -neg))
            continue
        sc = complete_score(fwd, t)
        if np.isfinite(sc):
            heapq.heappush(heap, (-sc, counter, "c", prefix, fwd, t))
            counter += 1
        if t < l_max:
            base = logsumexp(fwd[:, None] + log_a, axis=0)
            for v in symbols:
                fwd2 = base + log_emit[v]
                pri = extend_ub(fwd2, t + 1)
                if np.isfinite(pri):
                    heapq.heappush(heap, (-pri, counter, "p", prefix + (v,), fwd2, t + 1))
                    counter += 1
    return out


class FinalStateEnumeration:
    """Result of :func:`_final_state_enumerate`: ``top_k(k)`` -> [(sequence, log_prob), ...] descending."""

    def __init__(self, hmm, len_dist):
        self._hmm, self._len_dist = hmm, len_dist

    def top_k(self, k):
        """Return the top ``k`` final-state-constrained sequences."""
        return _final_state_enumerate(self._hmm, self._len_dist, max_results=int(k))


_DENSE_FB_NUMBA = None


def _dense_fb_numba():
    """Lazily build a numba-jitted scaled dense forward-backward returning (loglik, gamma, xi_sum)."""
    global _DENSE_FB_NUMBA
    if _DENSE_FB_NUMBA is not None:
        return _DENSE_FB_NUMBA
    from numba import njit

    @njit(cache=True)
    def fb(log_b, pi, a):  # log_b (T,K), pi (K,), a (K,K)
        t_len, k = log_b.shape
        b = np.empty((t_len, k))
        mxsum = 0.0
        for t in range(t_len):
            mx = log_b[t].max()
            mxsum += mx
            for j in range(k):
                b[t, j] = np.exp(log_b[t, j] - mx)
        alpha = np.empty((t_len, k))
        c = np.empty(t_len)
        s = 0.0
        for j in range(k):
            alpha[0, j] = pi[j] * b[0, j]
            s += alpha[0, j]
        c[0] = s
        for j in range(k):
            alpha[0, j] /= s
        for t in range(1, t_len):
            s = 0.0
            for j in range(k):
                acc = 0.0
                for i in range(k):
                    acc += alpha[t - 1, i] * a[i, j]
                alpha[t, j] = acc * b[t, j]
                s += alpha[t, j]
            c[t] = s
            for j in range(k):
                alpha[t, j] /= s
        loglik = mxsum
        for t in range(t_len):
            loglik += np.log(c[t])
        beta = np.empty((t_len, k))
        for j in range(k):
            beta[t_len - 1, j] = 1.0
        for t in range(t_len - 2, -1, -1):
            for i in range(k):
                acc = 0.0
                for j in range(k):
                    acc += a[i, j] * b[t + 1, j] * beta[t + 1, j]
                beta[t, i] = acc / c[t + 1]
        gamma = np.empty((t_len, k))
        for t in range(t_len):
            gs = 0.0
            for j in range(k):
                gamma[t, j] = alpha[t, j] * beta[t, j]
                gs += gamma[t, j]
            for j in range(k):
                gamma[t, j] /= gs
        xi = np.zeros((k, k))
        for t in range(t_len - 1):
            for i in range(k):
                ai = alpha[t, i]
                if ai == 0.0:
                    continue
                for j in range(k):
                    xi[i, j] += ai * a[i, j] * b[t + 1, j] * beta[t + 1, j] / c[t + 1]
        return loglik, gamma, xi

    _DENSE_FB_NUMBA = fb
    return fb


class StructuredHMM:
    """An HMM whose transition is a :class:`TransitionOperator` (dense / low-rank / a combinator).

    ``emissions`` is one observation distribution per state; ``pi`` the initial-state distribution;
    ``transition`` any ``TransitionOperator``. The scaled forward-backward and EM call the operator's
    ``forward``/``backward``/``accumulate``/``estimate``, so a low-rank or factorial transition runs the
    SAME inference at its own cost (O(K r) for low-rank). ``emission_estimators`` (one per state) drives
    the emission M-step; default reuses ``emissions[k].estimator()``.
    """

    def __init__(
        self,
        emissions,
        pi,
        transition: TransitionOperator,
        emission_estimators=None,
        keys=(None, None),
        name=None,
        len_dist=None,
        terminal_states=None,
        final_states=None,
    ) -> None:
        self.emissions = list(emissions)
        self.pi = np.asarray(pi, dtype=float)
        self.transition = transition
        self.K = len(self.emissions)
        self.keys = tuple(keys)  # (init_key, trans_key) for parameter tying across models
        self.name = name
        self.len_dist = len_dist  # optional distribution over sequence length (needed for enumeration)
        # terminal (absorbing) states: when set, the sequence length is a STOPPING TIME -- the chain only
        # transitions FROM non-terminal states and the sequence must END in a terminal state (no len_dist).
        self.terminal_states = None if terminal_states is None else set(int(s) for s in terminal_states)
        self.term_mask = None
        if self.terminal_states:
            self.term_mask = np.zeros(self.K, dtype=bool)
            self.term_mask[list(self.terminal_states)] = True
        # final states: the sequence may END only in one of these (a NON-absorbing boundary -- unlike
        # terminal_states, the chain still transitions through them mid-sequence). Used by the HSMM->HMM
        # expansion to require the final segment to complete. terminal_states takes precedence if both set.
        self.final_states = None if final_states is None else set(int(s) for s in final_states)
        self.final_mask = None
        if self.final_states:
            self.final_mask = np.zeros(self.K, dtype=bool)
            self.final_mask[list(self.final_states)] = True
        # coupling invariant: emissions[k] <-> pi[k] <-> transition row/col k all index the SAME state k,
        # so the three counts must agree (for a Kronecker op, n_states == K1*K2 emissions).
        if not (self.K == len(self.pi) == transition.n_states):
            raise ValueError(
                f"state-count mismatch: {self.K} emissions, len(pi)={len(self.pi)}, "
                f"transition.n_states={transition.n_states} must be equal."
            )
        s = self.pi.sum()
        if s > 0:
            self.pi = self.pi / s
        self._emit_est = emission_estimators or [e.estimator() for e in self.emissions]

    def _log_b(self, seq) -> np.ndarray:
        return np.array([[float(e.log_density(x)) for e in self.emissions] for x in seq])

    def _forward_backward(self, log_b, pi=None):
        if self.term_mask is not None:
            return self._terminal_forward_backward(log_b, pi=pi)
        if self.final_mask is not None:
            return self._final_forward_backward(log_b, pi=pi)
        T, _ = log_b.shape
        op = self.transition
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - mx)  # (T,K) scaled emissions
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = (self.pi if pi is None else pi) * b[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = op.forward(alpha[t - 1]) * b[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        loglik = float(np.sum(np.log(c)) + np.sum(mx))  # add the per-step maxima back
        beta = np.zeros((T, self.K))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = op.backward(b[t + 1] * beta[t + 1]) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return alpha, beta, c, b, gamma, loglik

    def _final_forward_backward(self, log_b, pi=None):
        """Standard forward, but the sequence may end only in a ``final_states`` state (non-absorbing): the
        likelihood sums the FINAL position over final_states and the backward boundary is the final mask.
        Transitions are unrestricted (the chain passes through final states normally mid-sequence)."""
        T, _ = log_b.shape
        op = self.transition
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - mx)
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = (self.pi if pi is None else pi) * b[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = op.forward(alpha[t - 1]) * b[t]
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t] if c[t] > 0 else alpha[t]
        final_mass = float(alpha[T - 1][self.final_mask].sum())
        loglik = float(np.sum(np.log(c)) + np.sum(mx) + np.log(final_mass + 1e-300))
        beta = np.zeros((T, self.K))
        beta[T - 1] = np.where(self.final_mask, 1.0, 0.0)
        for t in range(T - 2, -1, -1):
            beta[t] = op.backward(b[t + 1] * beta[t + 1]) / c[t + 1]
        gamma = alpha * beta
        gs = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gs, out=np.zeros_like(gamma), where=gs > 0)
        return alpha, beta, c, b, gamma, loglik

    def _terminal_forward_backward(self, log_b, pi=None):
        """Forward-backward when states are terminal (absorbing): the chain transitions only FROM
        non-terminal states (mask the belief at terminal states before each step) and the sequence must end
        in a terminal state. Works through the operator interface, so any transition structure is supported.
        Returns the SAME tuple as the standard path; the alpha returned is already terminal-masked for the
        transition accumulate, and the loglik is the terminal-stopping-time likelihood."""
        T, _ = log_b.shape
        op = self.transition
        nonterm = ~self.term_mask
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - mx)
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = (self.pi if pi is None else pi) * b[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = op.forward(np.where(nonterm, alpha[t - 1], 0.0)) * b[t]  # only leave non-terminal states
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t] if c[t] > 0 else alpha[t]
        term_mass = float(alpha[T - 1][self.term_mask].sum())  # must end in a terminal state
        loglik = float(np.sum(np.log(c)) + np.sum(mx) + np.log(term_mass + 1e-300))
        beta = np.zeros((T, self.K))
        beta[T - 1] = np.where(self.term_mask, 1.0, 0.0)  # only terminal states close a sequence
        for t in range(T - 2, -1, -1):
            beta[t] = np.where(nonterm, op.backward(b[t + 1] * beta[t + 1]), 0.0) / c[t + 1]
        gamma = alpha * beta
        gs = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gs, out=np.zeros_like(gamma), where=gs > 0)
        alpha_masked = alpha * nonterm[None, :]  # terminal states contribute no outgoing transition mass
        return alpha_masked, beta, c, b, gamma, loglik

    def viterbi(self, seq):
        """Most-likely state path (Viterbi / max-product). Uses the transition matrix, so it works for any
        operator; O(T K^2) -- a read-out, not the EM hot loop."""
        log_b = self._log_b(seq)
        log_a = np.log(self.transition.as_matrix() + 1e-300)
        log_pi = np.log(self.pi + 1e-300)
        t_len, k = log_b.shape
        delta = np.zeros((t_len, k))
        psi = np.zeros((t_len, k), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, t_len):
            m = delta[t - 1][:, None] + log_a  # (from, to)
            psi[t] = np.argmax(m, axis=0)
            delta[t] = m[psi[t], np.arange(k)] + log_b[t]
        path = np.zeros(t_len, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(t_len - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posterior_decode(self, seq):
        """Per-position MAP state argmax_k P(z_t = k | x) from the forward-backward posteriors gamma."""
        return np.argmax(self._forward_backward(self._log_b(seq))[4], axis=1)

    def enumerator(self):
        """Enumerate observation sequences in descending marginal probability (top_k / rank / seek /
        nucleus / certified estimates). Enumeration depends only on pi, the transition MATRIX, the
        emissions and a length distribution -- not on the operator's internal structure -- so it reuses
        the built-in HMM enumerator (an A*-style best-first search over the trellis) on the dense matrix.
        Requires ``len_dist`` (a distribution over sequence length) and enumerable (discrete) emissions."""
        from mixle.enumeration import EnumerationError
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

        if self.len_dist is None:
            raise EnumerationError(self, reason="StructuredHMM needs a len_dist to enumerate sequence length")
        if self.final_mask is not None:
            # sequences must end in a final state (e.g. an HSMM expansion) -- the built-in enumerator does
            # not honor a final-state mask, so use the dedicated final-state best-first enumerator.
            return FinalStateEnumeration(self, self.len_dist)
        dense = HiddenMarkovModelDistribution(
            self.emissions,
            w=self.pi.tolist(),
            transitions=self.transition.as_matrix().tolist(),
            len_dist=self.len_dist,
        )
        return dense.enumerator()

    def dist_to_enumerator(self):
        """Return the sequence enumerator for this HMM."""
        return self.enumerator()

    def state_posteriors(self, seq):
        """The full smoothing posteriors gamma[t,k] = P(z_t = k | x)."""
        return self._forward_backward(self._log_b(seq))[4]

    def seq_log_density(self, seqs) -> np.ndarray:
        """Score a batch of observation sequences."""
        return np.array([self._forward_backward(self._log_b(s))[5] for s in seqs])

    def sampler(self, seed=None):
        """Return a sampler for observation sequences."""
        return _StructuredHMMSampler(self, seed)

    def _can_fast_fb(self):
        """The numba dense forward-backward applies for a plain dense transition with no terminal states."""
        from mixle.utils.optional_deps import HAS_NUMBA

        return HAS_NUMBA and self.term_mask is None and type(self.transition) is DenseTransition

    def fit(self, seqs, *, max_its: int = 50, tol: float = 1e-6, fast: bool = True):
        """EM (Baum-Welch) through the transition operator. Returns ``(fitted_hmm, loglik_trace)``.

        ``fast=True`` uses the numba-jitted dense forward-backward (~30x over the numpy Python loop) when
        the transition is a plain ``DenseTransition`` with no terminal states; structured operators
        (low-rank / sparse / combinator) use the operator's per-step accumulate as before."""
        seqs = [list(s) for s in seqs]
        use_fast = fast and self._can_fast_fb()
        fb = _dense_fb_numba() if use_fast else None
        ll_trace = []
        for _ in range(int(max_its)):
            trans_acc = self.transition.new_accumulator()
            pi_acc = np.zeros(self.K)
            emit_accs = [est.accumulator_factory().make() for est in self._emit_est]
            nk = np.zeros(self.K)
            total_ll = 0.0
            a_mat = self.transition.as_matrix() if use_fast else None
            for seq in seqs:
                if not seq:
                    continue
                log_b = self._log_b(seq)
                if use_fast:
                    ll, gamma, xi = fb(log_b, self.pi, a_mat)
                    trans_acc += xi  # expected K x K transition counts, computed in numba
                else:
                    alpha, beta, c, b, gamma, ll = self._forward_backward(log_b)
                    for t in range(len(seq) - 1):
                        self.transition.accumulate(trans_acc, alpha[t], b[t + 1] * beta[t + 1], c[t + 1])
                total_ll += ll
                pi_acc += gamma[0]
                for k in range(self.K):
                    enc = self.emissions[k].dist_to_encoder().seq_encode(seq)
                    emit_accs[k].seq_update(enc, gamma[:, k], self.emissions[k])
                    nk[k] += gamma[:, k].sum()
            self.transition = self.transition.estimate(trans_acc)
            self.pi = pi_acc / pi_acc.sum()
            self.emissions = [self._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(self.K)]
            ll_trace.append(total_ll)
            if len(ll_trace) > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tol * max(1.0, abs(ll_trace[-2])):
                break
        return self, ll_trace


class _StructuredHMMSampler:
    def __init__(self, hmm: StructuredHMM, seed=None):
        self.hmm = hmm
        self.rng = np.random.RandomState(seed)

    def sample(self, length: int):
        h = self.hmm
        a = h.transition.as_matrix()
        s = self.rng.choice(h.K, p=h.pi)
        out = []
        for _ in range(int(length)):
            out.append(h.emissions[s].sampler(seed=int(self.rng.randint(1, 2**31))).sample())
            if h.term_mask is not None and h.term_mask[s]:
                break  # terminal (absorbing) state ends the sequence -- length is the stopping time
            row = a[s]
            rs = row.sum()
            s = self.rng.choice(h.K, p=row / rs) if rs > 0 else s
        return out


class BlockDiagonalTransition(TransitionOperator):
    """Independent sub-chains: the states partition into blocks and transitions stay within a block.

    A model whose initial state picks a block and then evolves inside it -- a mixture of regimes that do
    not switch. Build it from any sub-operators (each block can itself be dense or low-rank). Exact,
    block-local forward-backward and M-step.
    """

    def __init__(self, blocks) -> None:
        self.blocks = list(blocks)
        self.sizes = [b.n_states for b in self.blocks]
        self.offsets = np.cumsum([0] + self.sizes)
        self.n_states = int(self.offsets[-1])

    def _slices(self):
        return [slice(int(self.offsets[i]), int(self.offsets[i + 1])) for i in range(len(self.blocks))]

    def forward(self, alpha):
        """Push each block's belief mass forward independently."""
        out = np.zeros(self.n_states)
        for b, sl in zip(self.blocks, self._slices()):
            out[sl] = b.forward(alpha[sl])
        return out

    def backward(self, v):
        """Pull each block's vector backward independently."""
        out = np.zeros(self.n_states)
        for b, sl in zip(self.blocks, self._slices()):
            out[sl] = b.backward(v[sl])
        return out

    def as_matrix(self):
        """Materialize the block-diagonal transition matrix."""
        from scipy.linalg import block_diag

        return block_diag(*[b.as_matrix() for b in self.blocks])

    def new_accumulator(self):
        """Create one transition accumulator per block."""
        return [b.new_accumulator() for b in self.blocks]

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate block-local expected transition mass."""
        for b, a, sl in zip(self.blocks, acc, self._slices()):
            b.accumulate(a, alpha_t[sl], w_next[sl], scale)

    def estimate(self, acc):
        """Estimate each block and return a new block-diagonal transition."""
        return BlockDiagonalTransition([b.estimate(a) for b, a in zip(self.blocks, acc)])


class KroneckerTransition(TransitionOperator):
    """Factorial HMM: the state is the pair ``(s1, s2)`` of two chains evolving in parallel, with
    ``A = A1 (x) A2`` (Kronecker). State index is ``i1 * K2 + i2``.

    Forward-backward uses the reshape identity (``alpha @ (A1 (x) A2)`` reshapes to ``A1^T @ M @ A2``),
    so a step is O(K1 K2 (K1 + K2)) instead of O((K1 K2)^2) -- the whole point of a factorial HMM. The
    E-step is *exact* over the joint state; the M-step is the standard factorial marginal update (each
    factor re-estimated from the marginalized joint transition mass), verified to keep EM monotone.
    """

    def __init__(self, op1: TransitionOperator, op2: TransitionOperator) -> None:
        self.op1, self.op2 = op1, op2
        self.k1, self.k2 = op1.n_states, op2.n_states
        self.n_states = self.k1 * self.k2

    def _a1(self):
        return self.op1.as_matrix()

    def _a2(self):
        return self.op2.as_matrix()

    def forward(self, alpha):
        """Push a joint belief through the Kronecker-factorized transition."""
        m = alpha.reshape(self.k1, self.k2)
        return (self._a1().T @ m @ self._a2()).reshape(-1)  # alpha @ (A1 (x) A2)

    def backward(self, v):
        """Pull a joint vector backward through the Kronecker transition."""
        m = v.reshape(self.k1, self.k2)
        return (self._a1() @ m @ self._a2().T).reshape(-1)  # (A1 (x) A2) @ v

    def as_matrix(self):
        """Materialize the dense Kronecker transition matrix."""
        return np.kron(self._a1(), self._a2())

    def new_accumulator(self):
        """Create marginal transition-count accumulators for both factors."""
        return [np.zeros((self.k1, self.k1)), np.zeros((self.k2, self.k2))]  # marginal factor counts

    def accumulate(self, acc, alpha_t, w_next, scale):
        """Accumulate exact marginalized factor transition statistics."""
        a1, a2 = self._a1(), self._a2()
        am = alpha_t.reshape(self.k1, self.k2)
        wm = w_next.reshape(self.k1, self.k2)
        inv = 1.0 / max(scale, 1e-300)
        acc[0] += a1 * (am @ (a2 @ wm.T)) * inv  # n1[i1,j1] = A1[i1,j1] * sum_{i2,j2} xi
        acc[1] += a2 * (am.T @ (a1 @ wm)) * inv  # n2[i2,j2] = A2[i2,j2] * sum_{i1,j1} xi

    def estimate(self, acc):
        """Estimate both Kronecker factors from marginalized expected counts."""
        return KroneckerTransition(DenseTransition(_row_normalize(acc[0])), DenseTransition(_row_normalize(acc[1])))


def _chunk_spans(t_len: int, chunk: int, overlap: int):
    """Yield (ctx_lo, ctx_hi, keep_lo, keep_hi): a window [ctx_lo:ctx_hi] whose interior [keep_lo:keep_hi]
    (relative to the window) is kept; the ``overlap`` context on each side is run only to forget the
    boundary, then discarded."""
    for start in range(0, t_len, chunk):
        end = min(start + chunk, t_len)
        ctx_lo, ctx_hi = max(0, start - overlap), min(t_len, end + overlap)
        yield ctx_lo, ctx_hi, start - ctx_lo, end - ctx_lo


def chunked_state_posteriors(hmm: StructuredHMM, seq, *, chunk: int, overlap: int) -> np.ndarray:
    """State posteriors gamma for one long sequence via overlapping chunks, each run INDEPENDENTLY
    (embarrassingly parallel). The first chunk uses the model's pi; interior chunks start from the uniform
    belief and the ``overlap`` context lets the chain *forget* that wrong boundary -- so the kept interior
    matches the exact forward-backward up to an error that decays at the mixing rate in ``overlap``."""
    log_b_full = hmm._log_b(seq)
    t_len = len(seq)
    out = np.zeros((t_len, hmm.K))
    uniform = np.ones(hmm.K) / hmm.K
    for ctx_lo, ctx_hi, keep_lo, keep_hi in _chunk_spans(t_len, chunk, overlap):
        pi = hmm.pi if ctx_lo == 0 else uniform
        _, _, _, _, gamma, _ = hmm._forward_backward(log_b_full[ctx_lo:ctx_hi], pi=pi)
        out[ctx_lo + keep_lo : ctx_lo + keep_hi] = gamma[keep_lo:keep_hi]
    return out


def fit_chunked(
    hmm: StructuredHMM, seqs, *, chunk: int, overlap: int, max_its: int = 50, workers: int = 0, tol: float = 1e-6
):
    """Baum-Welch where each long sequence's forward-backward is split into overlapping chunks run in
    PARALLEL (the forgetting property bounds the boundary error). ``workers>0`` runs the per-chunk E-steps
    on a thread pool (NumPy releases the GIL in its array kernels); ``workers=0`` runs them serially. The
    interior suff-statistics are accumulated exactly as in :meth:`StructuredHMM.fit`; this only changes
    *how* the E-step is computed, trading a small, overlap-controlled approximation for intra-sequence
    parallelism. Returns ``(fitted_hmm, loglik_trace)`` (LL is the chunk-summed approximation)."""
    from concurrent.futures import ThreadPoolExecutor

    seqs = [list(s) for s in seqs]
    uniform = np.ones(hmm.K) / hmm.K
    ll_trace = []

    def chunk_estep(args):
        seq, ctx_lo, ctx_hi, keep_lo, keep_hi = args
        log_b = hmm._log_b(seq)[ctx_lo:ctx_hi]
        pi = hmm.pi if ctx_lo == 0 else uniform
        alpha, beta, c, b, gamma, ll = hmm._forward_backward(log_b, pi=pi)
        # transition mass over kept interior transitions only
        contrib_ll = float(np.sum(np.log(c[keep_lo : max(keep_lo + 1, keep_hi)])))
        return seq, ctx_lo, keep_lo, keep_hi, alpha, beta, c, b, gamma, contrib_ll

    for _ in range(int(max_its)):
        tasks = [
            (seq, lo, hi, klo, khi)
            for seq in seqs
            if seq
            for (lo, hi, klo, khi) in _chunk_spans(len(seq), chunk, overlap)
        ]
        if workers and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(chunk_estep, tasks))  # independent chunks -> parallel
        else:
            results = [chunk_estep(t) for t in tasks]

        trans_acc = hmm.transition.new_accumulator()
        pi_acc = np.zeros(hmm.K)
        emit_accs = [est.accumulator_factory().make() for est in hmm._emit_est]
        nk = np.zeros(hmm.K)
        total_ll = 0.0
        for seq, ctx_lo, keep_lo, keep_hi, alpha, beta, c, b, gamma, contrib_ll in results:
            total_ll += contrib_ll
            if ctx_lo == 0:
                pi_acc += gamma[0]
            for t in range(keep_lo, keep_hi):  # kept interior transitions
                if t + 1 < len(c):
                    hmm.transition.accumulate(trans_acc, alpha[t], b[t + 1] * beta[t + 1], c[t + 1])
            seg = seq[ctx_lo + keep_lo : ctx_lo + keep_hi]
            for k in range(hmm.K):
                enc = hmm.emissions[k].dist_to_encoder().seq_encode(seg)
                w = gamma[keep_lo:keep_hi, k]
                emit_accs[k].seq_update(enc, w, hmm.emissions[k])
                nk[k] += w.sum()
        hmm.transition = hmm.transition.estimate(trans_acc)
        hmm.pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else hmm.pi
        hmm.emissions = [hmm._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(hmm.K)]
        ll_trace.append(total_ll)
        if len(ll_trace) > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tol * max(1.0, abs(ll_trace[-2])):
            break
    return hmm, ll_trace


# ===================================================================================================
# The 5-part estimator contract: makes StructuredHMM a SequenceEncodableProbabilityDistribution that
# optimize()/run_em() can fit directly (optimize(seqs, hmm.estimator())). The E-step (forward-backward
# per sequence) lives in the accumulator; the M-step (pi / transition-operator / emission re-estimation)
# in the estimator. Keys (init_key, trans_key) let two HMMs TIE their initial / transition parameters.
# ===================================================================================================
from mixle.stats.compute.pdist import (  # noqa: E402
    DataSequenceEncoder,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _add_nested(a, b):
    return a + b if isinstance(a, np.ndarray) else [_add_nested(x, y) for x, y in zip(a, b)]


def _scale_nested(a, f):
    return a * f if isinstance(a, np.ndarray) else [_scale_nested(x, f) for x in a]


class StructuredHMMDataEncoder(DataSequenceEncoder):
    """Sequences pass through as lists -- the structured forward-backward scores raw observations through
    the per-state emission ``log_density`` (no flattened columnar encoding; composability over raw speed)."""

    def seq_encode(self, x):
        """Encode sequences as lists without changing their observations."""
        return [list(s) for s in x]

    def __eq__(self, other):
        return isinstance(other, StructuredHMMDataEncoder)

    def __hash__(self):
        return hash("StructuredHMMDataEncoder")


class StructuredHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """Baum-Welch E-step accumulator: per-sequence forward-backward, accumulating initial-state mass,
    transition-operator mass, and per-state weighted emission statistics."""

    def __init__(self, emission_accumulators, transition_proto, keys=(None, None)) -> None:
        self.emit = list(emission_accumulators)
        self.K = len(self.emit)
        self.transition_proto = transition_proto
        self.pi_acc = np.zeros(self.K)
        self.trans_acc = transition_proto.new_accumulator()
        self.nk = np.zeros(self.K)
        self.init_key, self.trans_key = keys

    def update(self, x, weight, estimate):
        """Accumulate sufficient statistics from one weighted sequence."""
        self.seq_update([x], np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights, estimate):
        """Run forward-backward and accumulate weighted sufficient statistics for a batch."""
        for seq, w in zip(x, np.asarray(weights, dtype=float)):
            if not seq:
                continue
            log_b = estimate._log_b(seq)
            alpha, beta, c, b, gamma, _ = estimate._forward_backward(log_b)
            self.pi_acc += w * gamma[0]
            for t in range(len(seq) - 1):
                estimate.transition.accumulate(self.trans_acc, alpha[t], b[t + 1] * beta[t + 1] * w, c[t + 1])
            for k in range(self.K):
                enc = estimate.emissions[k].dist_to_encoder().seq_encode(seq)
                wk = gamma[:, k] * w
                self.emit[k].seq_update(enc, wk, estimate.emissions[k])
                self.nk[k] += float(wk.sum())

    def seq_initialize(self, x, weights, rng):
        """Initialize sufficient statistics with random soft state responsibilities."""
        # no model yet: seed with random soft responsibilities + a random transition accumulator
        self.trans_acc = _add_nested(self.trans_acc, self.transition_proto.random_accumulator(rng))
        for seq, w in zip(x, np.asarray(weights, dtype=float)):
            if not seq:
                continue
            g = rng.dirichlet(np.ones(self.K), len(seq))
            self.pi_acc += w * g[0]
            for k in range(self.K):
                enc = self.emit[k].acc_to_encoder().seq_encode(seq)
                wk = g[:, k] * w
                self.emit[k].seq_initialize(enc, wk, rng)
                self.nk[k] += float(wk.sum())

    def combine(self, suff_stat):
        """Merge serialized HMM sufficient statistics."""
        pi_acc, trans_acc, emit_vals, nk = suff_stat
        self.pi_acc += pi_acc
        self.trans_acc = _add_nested(self.trans_acc, trans_acc)
        self.nk += nk
        for k in range(self.K):
            self.emit[k].combine(emit_vals[k])
        return self

    def value(self):
        """Return serialized HMM sufficient statistics."""
        return (self.pi_acc.copy(), self.trans_acc, [e.value() for e in self.emit], self.nk.copy())

    def from_value(self, x):
        """Restore accumulator state from serialized sufficient statistics."""
        self.pi_acc, self.trans_acc, emit_vals, self.nk = x[0].copy(), x[1], x[2], x[3].copy()
        for k in range(self.K):
            self.emit[k].from_value(emit_vals[k])
        return self

    def scale(self, factor):
        """Multiply the running statistics by ``factor`` -- the decay primitive online/streaming
        Baum-Welch (StreamingEstimator) uses to fold a new batch into a forgetting running estimate."""
        f = float(factor)
        self.pi_acc *= f
        self.trans_acc = _scale_nested(self.trans_acc, f)
        self.nk *= f
        for e in self.emit:
            if hasattr(e, "scale"):
                e.scale(f)
        return self

    def acc_to_encoder(self):
        """Return the encoder associated with this accumulator."""
        return StructuredHMMDataEncoder()

    # parameter tying: pool initial / transition counts across accumulators sharing a key
    def key_merge(self, store):
        """Merge tied initial or transition sufficient statistics into ``store``."""
        if self.init_key is not None:
            store[self.init_key] = self.pi_acc + store[self.init_key] if self.init_key in store else self.pi_acc
        if self.trans_key is not None:
            store[self.trans_key] = (
                _add_nested(store[self.trans_key], self.trans_acc) if self.trans_key in store else self.trans_acc
            )
        for e in self.emit:
            if hasattr(e, "key_merge"):
                e.key_merge(store)

    def key_replace(self, store):
        """Replace tied initial or transition statistics from ``store``."""
        if self.init_key is not None and self.init_key in store:
            self.pi_acc = store[self.init_key]
        if self.trans_key is not None and self.trans_key in store:
            self.trans_acc = store[self.trans_key]
        for e in self.emit:
            if hasattr(e, "key_replace"):
                e.key_replace(store)


class StructuredHMMAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for structured HMM accumulators."""

    def __init__(self, emission_estimators, transition_proto, keys):
        self.emission_estimators = emission_estimators
        self.transition_proto = transition_proto
        self.keys = keys

    def make(self):
        """Create a fresh structured HMM accumulator."""
        emit = [est.accumulator_factory().make() for est in self.emission_estimators]
        return StructuredHMMAccumulator(emit, self.transition_proto, self.keys)


class StructuredHMMEstimator(ParameterEstimator):
    """Estimator (M-step) for a :class:`StructuredHMM`: re-estimates pi, the transition OPERATOR (any
    structure -- dense/low-rank/combinator), and each state's emission from the Baum-Welch statistics.
    ``keys=(init_key, trans_key)`` tie the initial / transition parameters across HMMs that share them."""

    def __init__(
        self, emission_estimators, transition_proto, keys=(None, None), name=None, len_dist=None, terminal_states=None
    ):
        self.emission_estimators = list(emission_estimators)
        self.transition_proto = transition_proto
        self.keys = tuple(keys)
        self.name = name
        self.len_dist = len_dist  # carried (not fit) so fitted models retain it for enumeration
        self.terminal_states = terminal_states

    def accumulator_factory(self):
        """Return the accumulator factory used by this estimator."""
        return StructuredHMMAccumulatorFactory(self.emission_estimators, self.transition_proto, self.keys)

    def estimate(self, nobs, suff_stat):
        """Estimate an HMM from Baum-Welch sufficient statistics."""
        pi_acc, trans_acc, emit_vals, nk = suff_stat
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(len(pi_acc)) / len(pi_acc)
        transition = self.transition_proto.estimate(trans_acc)
        emissions = [self.emission_estimators[k].estimate(float(nk[k]), emit_vals[k]) for k in range(len(emit_vals))]
        return StructuredHMM(
            emissions,
            pi,
            transition,
            self.emission_estimators,
            self.keys,
            self.name,
            self.len_dist,
            self.terminal_states,
        )


# --- make StructuredHMM satisfy the distribution side of the contract -------------------------------
def _structured_hmm_log_density(self, x):
    return self._forward_backward(self._log_b(x))[5]


def _structured_hmm_dist_to_encoder(self):
    return StructuredHMMDataEncoder()


def _structured_hmm_estimator(self, pseudo_count=None):
    return StructuredHMMEstimator(
        self._emit_est, self.transition, self.keys, self.name, self.len_dist, self.terminal_states
    )


StructuredHMM.log_density = _structured_hmm_log_density
StructuredHMM.dist_to_encoder = _structured_hmm_dist_to_encoder
StructuredHMM.estimator = _structured_hmm_estimator
SequenceEncodableProbabilityDistribution.register(StructuredHMM)


def stationary_initial(op: TransitionOperator, *, iters: int = 2000, tol: float = 1e-13) -> np.ndarray:
    """The transition's stationary distribution (pi @ A == pi), by power iteration through ``op.forward``
    -- so it is O(K r) for a low-rank op, never forming A. Use it to COUPLE a StructuredHMM's initial
    state to its transition (``pi = stationary_initial(transition)``): the chain starts in its long-run
    distribution instead of a free, separately-estimated pi. Answers "do the initial states match the
    transition?" -- they can, by construction."""
    pi = np.ones(op.n_states) / op.n_states
    for _ in range(int(iters)):
        nxt = np.maximum(op.forward(pi), 0.0)
        s = nxt.sum()
        nxt = nxt / s if s > 0 else pi
        if np.max(np.abs(nxt - pi)) < tol:
            return nxt
        pi = nxt
    return pi


class InputOutputHMM:
    """Input-output HMM (IOHMM): an exogenous discrete input ``u_t`` selects which transition governs each
    step. Holds one :class:`TransitionOperator` per input symbol; the emission is per-state. Data is
    ``(obs_seq, input_seq)`` pairs where ``input_seq[t]`` in {0..M-1} drives the transition from t to t+1.

    Lets a covariate steer the dynamics -- regime switching driven by an observed control, the difference
    between a plain HMM and a controlled Markov model. (Input-dependent emissions are a natural extension;
    here emissions depend on state only.)
    """

    def __init__(self, emissions, pi, transitions, emission_estimators=None, name=None, terminal_states=None) -> None:
        self.emissions = list(emissions)
        self.pi = np.asarray(pi, dtype=float)
        self.pi = self.pi / self.pi.sum() if self.pi.sum() > 0 else self.pi
        self.transitions = list(transitions)  # one operator per input symbol
        self.K = len(self.emissions)
        self.M = len(self.transitions)
        self.name = name
        self.terminal_states = None if terminal_states is None else set(int(s) for s in terminal_states)
        self.term_mask = None
        if self.terminal_states:
            self.term_mask = np.zeros(self.K, dtype=bool)
            self.term_mask[list(self.terminal_states)] = True
        if not all(t.n_states == self.K == len(self.pi) for t in self.transitions):
            raise ValueError("every input's transition must have n_states == #emissions == len(pi).")
        self._emit_est = emission_estimators or [e.estimator() for e in self.emissions]

    def _log_b(self, seq):
        return np.array([[float(e.log_density(x)) for e in self.emissions] for x in seq])

    def _forward_backward(self, log_b, inputs):
        t_len, _ = log_b.shape
        term = self.term_mask
        nonterm = None if term is None else ~term
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - mx)
        alpha = np.zeros((t_len, self.K))
        c = np.zeros(t_len)
        alpha[0] = self.pi * b[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, t_len):
            prev = alpha[t - 1] if nonterm is None else np.where(nonterm, alpha[t - 1], 0.0)
            alpha[t] = self.transitions[inputs[t - 1]].forward(prev) * b[t]
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t] if c[t] > 0 else alpha[t]
        if term is None:
            loglik = float(np.sum(np.log(c)) + np.sum(mx))
        else:
            tm = float(alpha[t_len - 1][term].sum())
            loglik = float(np.sum(np.log(c)) + np.sum(mx) + np.log(tm + 1e-300))
        beta = np.zeros((t_len, self.K))
        beta[t_len - 1] = 1.0 if term is None else np.where(term, 1.0, 0.0)
        for t in range(t_len - 2, -1, -1):
            back = self.transitions[inputs[t]].backward(b[t + 1] * beta[t + 1])
            beta[t] = (back if nonterm is None else np.where(nonterm, back, 0.0)) / c[t + 1]
        gamma = alpha * beta
        gs = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gs, out=np.zeros_like(gamma), where=gs > 0)
        if nonterm is not None:
            alpha = alpha * nonterm[None, :]  # mask outgoing transition mass from terminal states
        return alpha, beta, c, b, gamma, loglik

    def seq_log_density(self, x, input_seqs=None):
        """Per-sequence forward log-likelihood. Two call forms:
        - ``seq_log_density(obs_seqs, input_seqs)`` -- the explicit two-list API; or
        - ``seq_log_density(records)`` -- one list of ``(obs, input)``-pair sequences (the 5-part contract)."""
        if input_seqs is None:
            out = []
            for seq in x:
                obs = [p[0] for p in seq]
                inputs = [int(p[1]) for p in seq]
                out.append(self._forward_backward(self._log_b(obs), inputs)[5])
            return np.array(out)
        return np.array([self._forward_backward(self._log_b(o), list(u))[5] for o, u in zip(x, input_seqs)])

    def _obs_inputs(self, seq, inputs):
        """Split one record into (observations, inputs), accepting both call forms the scoring API uses:
        ``(record)`` -- one sequence of ``(observation, input)`` pairs -- or ``(obs_seq, input_seq)``."""
        if inputs is None:
            return [p[0] for p in seq], [int(p[1]) for p in seq]
        return list(seq), [int(u) for u in inputs]

    def viterbi(self, seq, inputs=None):
        """Most-likely state path (Viterbi / max-product), conditioned on the input/control sequence:
        step t -> t+1 maximizes over the transition ``inputs[t]`` selects. Call as ``viterbi(record)``
        on one ``(observation, input)``-pair sequence or as ``viterbi(obs_seq, input_seq)``. Uses the
        per-input transition matrices, so it works for any operator; O(T K^2) -- a read-out, not the
        EM hot loop."""
        obs, u = self._obs_inputs(seq, inputs)
        log_b = self._log_b(obs)
        log_as = [np.log(t.as_matrix() + 1e-300) for t in self.transitions]
        log_pi = np.log(self.pi + 1e-300)
        t_len, k = log_b.shape
        delta = np.zeros((t_len, k))
        psi = np.zeros((t_len, k), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, t_len):
            m = delta[t - 1][:, None] + log_as[u[t - 1]]  # (from, to) under the input driving this step
            psi[t] = np.argmax(m, axis=0)
            delta[t] = m[psi[t], np.arange(k)] + log_b[t]
        path = np.zeros(t_len, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(t_len - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posterior_decode(self, seq, inputs=None):
        """Per-position MAP state argmax_k P(z_t = k | x, u) from the forward-backward posteriors gamma."""
        obs, u = self._obs_inputs(seq, inputs)
        return np.argmax(self._forward_backward(self._log_b(obs), u)[4], axis=1)

    def state_posteriors(self, seq, inputs=None):
        """The full smoothing posteriors gamma[t,k] = P(z_t = k | x, u), conditioned on the inputs."""
        obs, u = self._obs_inputs(seq, inputs)
        return self._forward_backward(self._log_b(obs), u)[4]

    def sampler(self, seed=None):
        """Return a sampler for ``(observation, input)``-pair records along a given control sequence."""
        return _IOHMMSampler(self, seed)

    def fit(self, obs_seqs, input_seqs, *, max_its: int = 50, tol: float = 1e-6):
        """Fit the IOHMM with Baum-Welch using the supplied observation and input sequences."""
        obs_seqs = [list(o) for o in obs_seqs]
        input_seqs = [list(u) for u in input_seqs]
        ll_trace = []
        for _ in range(int(max_its)):
            trans_accs = [t.new_accumulator() for t in self.transitions]
            pi_acc = np.zeros(self.K)
            emit_accs = [est.accumulator_factory().make() for est in self._emit_est]
            nk = np.zeros(self.K)
            total_ll = 0.0
            for o, u in zip(obs_seqs, input_seqs):
                if not o:
                    continue
                log_b = self._log_b(o)
                alpha, beta, c, b, gamma, ll = self._forward_backward(log_b, u)
                total_ll += ll
                pi_acc += gamma[0]
                for t in range(len(o) - 1):
                    m = u[t]
                    self.transitions[m].accumulate(trans_accs[m], alpha[t], b[t + 1] * beta[t + 1], c[t + 1])
                for k in range(self.K):
                    enc = self.emissions[k].dist_to_encoder().seq_encode(o)
                    emit_accs[k].seq_update(enc, gamma[:, k], self.emissions[k])
                    nk[k] += gamma[:, k].sum()
            self.transitions = [self.transitions[m].estimate(trans_accs[m]) for m in range(self.M)]
            self.pi = pi_acc / pi_acc.sum()
            self.emissions = [self._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(self.K)]
            ll_trace.append(total_ll)
            if len(ll_trace) > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tol * max(1.0, abs(ll_trace[-2])):
                break
        return self, ll_trace

    # --- 5-part contract: a record is one (obs, input) sequence = a list of (observation, input) pairs ---
    def log_density(self, seq):
        """Return the log likelihood of one ``(observation, input)`` sequence."""
        obs = [p[0] for p in seq]
        inputs = [int(p[1]) for p in seq]
        return self._forward_backward(self._log_b(obs), inputs)[5]

    def dist_to_encoder(self):
        """Return the pass-through IOHMM sequence encoder."""
        return IOHMMDataEncoder()

    def estimator(self, pseudo_count=None):
        """Return the estimator for this IOHMM structure."""
        return IOHMMEstimator(self._emit_est, list(self.transitions), self.name)

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible representation of this IOHMM (decodes via ``load_models``)."""
        from mixle.utils.serialization import to_serializable

        return to_serializable(self)


class _IOHMMSampler:
    """Samples one IOHMM record -- a list of ``(observation, input)`` pairs -- along a given control
    sequence. The inputs are exogenous, so the caller supplies them; the sampler draws the state path
    through the per-input transitions and one emission per state visited."""

    def __init__(self, hmm: InputOutputHMM, seed=None):
        self.hmm = hmm
        self.rng = np.random.RandomState(seed)

    def sample(self, inputs):
        h = self.hmm
        u = [int(v) for v in inputs]
        mats = [t.as_matrix() for t in h.transitions]
        s = self.rng.choice(h.K, p=h.pi)
        out = []
        for t, m in enumerate(u):
            out.append((h.emissions[s].sampler(seed=int(self.rng.randint(1, 2**31))).sample(), m))
            if h.term_mask is not None and h.term_mask[s]:
                break  # terminal (absorbing) state ends the sequence -- length is the stopping time
            if t + 1 < len(u):
                row = mats[m][s]
                rs = row.sum()
                s = self.rng.choice(h.K, p=row / rs) if rs > 0 else s
        return out


class IOHMMDataEncoder(DataSequenceEncoder):
    """An IOHMM record is one ``(obs, input)`` sequence -- a list of ``(observation, input_symbol)`` pairs."""

    def seq_encode(self, x):
        """Encode IOHMM records as lists of ``(observation, input)`` pairs."""
        return [list(s) for s in x]

    def __eq__(self, other):
        return isinstance(other, IOHMMDataEncoder)

    def __hash__(self):
        return hash("IOHMMDataEncoder")


class IOHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for IOHMM Baum-Welch sufficient statistics."""

    def __init__(self, emission_accumulators, transition_protos):
        self.emit = list(emission_accumulators)
        self.K = len(self.emit)
        self.transition_protos = list(transition_protos)
        self.M = len(self.transition_protos)
        self.pi_acc = np.zeros(self.K)
        self.trans_accs = [t.new_accumulator() for t in self.transition_protos]
        self.nk = np.zeros(self.K)

    def update(self, x, weight, estimate):
        """Accumulate sufficient statistics from one weighted IOHMM record."""
        self.seq_update([x], np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights, estimate):
        """Accumulate weighted sufficient statistics from a batch of IOHMM records."""
        for seq, w in zip(x, np.asarray(weights, dtype=float)):
            if not seq:
                continue
            obs = [p[0] for p in seq]
            inputs = [int(p[1]) for p in seq]
            log_b = estimate._log_b(obs)
            alpha, beta, c, b, gamma, _ = estimate._forward_backward(log_b, inputs)
            self.pi_acc += w * gamma[0]
            for t in range(len(seq) - 1):
                m = inputs[t]
                estimate.transitions[m].accumulate(self.trans_accs[m], alpha[t], b[t + 1] * beta[t + 1] * w, c[t + 1])
            for k in range(self.K):
                enc = estimate.emissions[k].dist_to_encoder().seq_encode(obs)
                wk = gamma[:, k] * w
                self.emit[k].seq_update(enc, wk, estimate.emissions[k])
                self.nk[k] += float(wk.sum())

    def seq_initialize(self, x, weights, rng):
        """Initialize IOHMM sufficient statistics with random soft responsibilities."""
        for m in range(self.M):
            self.trans_accs[m] = _add_nested(self.trans_accs[m], self.transition_protos[m].random_accumulator(rng))
        for seq, w in zip(x, np.asarray(weights, dtype=float)):
            if not seq:
                continue
            obs = [p[0] for p in seq]
            g = rng.dirichlet(np.ones(self.K), len(seq))
            self.pi_acc += w * g[0]
            for k in range(self.K):
                enc = self.emit[k].acc_to_encoder().seq_encode(obs)
                wk = g[:, k] * w
                self.emit[k].seq_initialize(enc, wk, rng)
                self.nk[k] += float(wk.sum())

    def combine(self, suff_stat):
        """Merge serialized IOHMM sufficient statistics."""
        pi_acc, trans_accs, emit_vals, nk = suff_stat
        self.pi_acc += pi_acc
        self.trans_accs = [_add_nested(a, b) for a, b in zip(self.trans_accs, trans_accs)]
        self.nk += nk
        for k in range(self.K):
            self.emit[k].combine(emit_vals[k])
        return self

    def value(self):
        """Return serialized IOHMM sufficient statistics."""
        return (self.pi_acc.copy(), self.trans_accs, [e.value() for e in self.emit], self.nk.copy())

    def from_value(self, x):
        """Restore accumulator state from serialized IOHMM statistics."""
        self.pi_acc, self.trans_accs, emit_vals, self.nk = x[0].copy(), x[1], x[2], x[3].copy()
        for k in range(self.K):
            self.emit[k].from_value(emit_vals[k])
        return self

    def acc_to_encoder(self):
        """Return the encoder associated with this accumulator."""
        return IOHMMDataEncoder()


class IOHMMAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for IOHMM accumulators."""

    def __init__(self, emission_estimators, transition_protos):
        self.emission_estimators = emission_estimators
        self.transition_protos = transition_protos

    def make(self):
        """Create a fresh IOHMM accumulator."""
        emit = [est.accumulator_factory().make() for est in self.emission_estimators]
        return IOHMMAccumulator(emit, self.transition_protos)


class IOHMMEstimator(ParameterEstimator):
    """Estimator (M-step) for an :class:`InputOutputHMM`: re-estimates pi, one transition operator per input
    symbol (from the per-input expected counts), and each state's emission."""

    def __init__(self, emission_estimators, transition_protos, name=None):
        self.emission_estimators = list(emission_estimators)
        self.transition_protos = list(transition_protos)
        self.name = name

    def accumulator_factory(self):
        """Return the accumulator factory used by this estimator."""
        return IOHMMAccumulatorFactory(self.emission_estimators, self.transition_protos)

    def estimate(self, nobs, suff_stat):
        """Estimate an IOHMM from Baum-Welch sufficient statistics."""
        pi_acc, trans_accs, emit_vals, nk = suff_stat
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(len(pi_acc)) / len(pi_acc)
        transitions = [self.transition_protos[m].estimate(trans_accs[m]) for m in range(len(trans_accs))]
        emissions = [self.emission_estimators[k].estimate(float(nk[k]), emit_vals[k]) for k in range(len(emit_vals))]
        return InputOutputHMM(emissions, pi, transitions, self.emission_estimators, self.name)


SequenceEncodableProbabilityDistribution.register(InputOutputHMM)


class ExplicitDurationHMM:
    """Hidden semi-Markov model (explicit-duration HMM): each state emits for a random *duration* drawn
    from a per-state duration distribution, then switches state (the transition matrix has a zero diagonal
    -- dwell time is modeled explicitly, not as a self-loop). This captures non-geometric state durations a
    plain HMM cannot.

    ``durations`` is one length-``max_duration`` probability vector per state (over d = 1..max_duration).
    The forward variable alpha_t(j) = P(obs_1:t, a segment ends at t in state j); the likelihood is
    sum_j alpha_T(j). Forward/EM are O(T * K * max_duration). Verified against brute-force segmentation.
    """

    def __init__(self, emissions, pi, transition_matrix, durations, max_duration, name=None) -> None:
        self.emissions = list(emissions)
        self.K = len(self.emissions)
        self.pi = np.asarray(pi, dtype=float)
        self.pi = self.pi / self.pi.sum()
        a = np.asarray(transition_matrix, dtype=float).copy()
        np.fill_diagonal(a, 0.0)  # EDHMM: must switch state after a segment
        self.a = _row_normalize(a)
        self.D = int(max_duration)
        self.dur = _row_normalize(np.asarray(durations, dtype=float))  # (K, D), over d=1..D
        self.name = name
        self._emit_est = [e.estimator() for e in self.emissions]

    def _log_b(self, seq):
        return np.array([[float(e.log_density(x)) for e in self.emissions] for x in seq])

    def _seg_loglik(self, log_b):
        """seg[t, d, j] = log P(obs_{t-d+1 .. t} | state j) for a length-(d+1) segment ENDING at t."""
        t_len = log_b.shape[0]
        csum = np.vstack([np.zeros(self.K), np.cumsum(log_b, axis=0)])  # (T+1, K)
        seg = np.full((t_len, self.D, self.K), -np.inf)
        for d in range(self.D):  # duration index d -> actual duration d+1
            for t in range(d, t_len):
                seg[t, d] = csum[t + 1] - csum[t - d]
        return seg

    def _forward(self, log_b):
        t_len = log_b.shape[0]
        seg = self._seg_loglik(log_b)
        log_dur, log_a, log_pi = np.log(self.dur + 1e-300), np.log(self.a + 1e-300), np.log(self.pi + 1e-300)
        log_alpha = np.full((t_len, self.K), -np.inf)  # segment ends at t in j
        log_e = np.full((t_len + 1, self.K), -np.inf)  # entry into j at time tau (segment starts at tau)
        log_e[0] = log_pi
        for t in range(t_len):
            for j in range(self.K):
                terms = [
                    log_e[t - d, j] + log_dur[j, d] + seg[t, d, j]
                    for d in range(min(t + 1, self.D))
                    if np.isfinite(log_e[t - d, j])
                ]
                if terms:
                    log_alpha[t, j] = _logsumexp(terms)
            for j in range(self.K):
                log_e[t + 1, j] = _logsumexp(log_alpha[t] + log_a[:, j])
        return log_alpha, log_e, seg

    def forward_loglik(self, seq):
        """Total log-likelihood log sum_j alpha_T(j) via the scaled explicit-duration forward."""
        log_alpha, _, _ = self._forward(self._log_b(seq))
        return float(_logsumexp(log_alpha[-1]))

    def _backward(self, log_b, seg):
        t_len = log_b.shape[0]
        log_dur, log_a = np.log(self.dur + 1e-300), np.log(self.a + 1e-300)
        log_beta = np.full((t_len, self.K), -np.inf)  # P(obs_{t+1:} | segment ends at t in j)
        log_bstar = np.full((t_len + 1, self.K), -np.inf)  # P(obs_{tau:} | segment starts at tau in j)
        log_beta[t_len - 1] = 0.0
        for tau in range(t_len - 1, -1, -1):
            for j in range(self.K):
                terms = [
                    log_dur[j, d] + seg[tau + d, d, j] + log_beta[tau + d, j] for d in range(min(self.D, t_len - tau))
                ]
                log_bstar[tau, j] = _logsumexp(terms) if terms else -np.inf
            if tau > 0:
                for j in range(self.K):
                    log_beta[tau - 1, j] = _logsumexp(log_a[j, :] + log_bstar[tau, :])
        return log_beta, log_bstar

    def _estep(self, seq):
        """Per-sequence E-step. Returns (loglik, pi_contrib (K,), trans_contrib (K,K), dur_contrib (K,D),
        occ (T,K)) -- the segment-posterior statistics that drive the duration/transition/emission M-step."""
        log_b = self._log_b(seq)
        t_len = len(seq)
        log_dur = np.log(self.dur + 1e-300)
        log_a = np.log(self.a + 1e-300)
        log_pi = np.log(self.pi + 1e-300)
        log_alpha, log_e, seg = self._forward(log_b)
        log_beta, log_bstar = self._backward(log_b, seg)
        z = _logsumexp(log_alpha[-1])
        pi_contrib = np.exp(log_pi + log_bstar[0] - z)
        dur_contrib = np.zeros((self.K, self.D))
        trans_contrib = np.zeros((self.K, self.K))
        occ = np.zeros((t_len, self.K))
        for t in range(t_len):
            for j in range(self.K):
                for d in range(min(t + 1, self.D)):
                    lp = log_e[t - d, j] + log_dur[j, d] + seg[t, d, j] + log_beta[t, j] - z
                    if np.isfinite(lp):
                        p = np.exp(lp)
                        dur_contrib[j, d] += p
                        occ[t - d : t + 1, j] += p
            if t < t_len - 1:
                for i in range(self.K):
                    trans_contrib[i] += np.exp(log_alpha[t, i] + log_a[i, :] + log_bstar[t + 1, :] - z)
        return float(z), pi_contrib, trans_contrib, dur_contrib, occ

    def fit(self, seqs, *, max_its: int = 50, tol: float = 1e-6):
        """Baum-Welch (EM) for the explicit-duration HMM: re-estimates emissions, the per-state duration
        distributions, the (zero-diagonal) transition, and pi. Returns (fitted_hmm, loglik_trace)."""
        seqs = [list(s) for s in seqs]
        ll_trace = []
        for _ in range(int(max_its)):
            dur_acc = np.zeros((self.K, self.D))
            trans_acc = np.zeros((self.K, self.K))
            pi_acc = np.zeros(self.K)
            emit_accs = [est.accumulator_factory().make() for est in self._emit_est]
            nk = np.zeros(self.K)
            total_ll = 0.0
            for seq in seqs:
                if not seq:
                    continue
                ll, pic, trc, drc, occ = self._estep(seq)
                total_ll += ll
                pi_acc += pic
                trans_acc += trc
                dur_acc += drc
                for k in range(self.K):
                    enc = self.emissions[k].dist_to_encoder().seq_encode(seq)
                    emit_accs[k].seq_update(enc, occ[:, k], self.emissions[k])
                    nk[k] += occ[:, k].sum()
            self.pi = pi_acc / pi_acc.sum()
            np.fill_diagonal(trans_acc, 0.0)
            self.a = _row_normalize(trans_acc) if trans_acc.sum() > 0 else self.a
            self.dur = _row_normalize(dur_acc)
            self.emissions = [self._emit_est[k].estimate(float(nk[k]), emit_accs[k].value()) for k in range(self.K)]
            ll_trace.append(total_ll)
            if len(ll_trace) > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tol * max(1.0, abs(ll_trace[-2])):
                break
        return self, ll_trace

    # --- 5-part contract: a record is one observation sequence ---
    def log_density(self, seq):
        """Return the explicit-duration HMM log likelihood for one sequence."""
        return self.forward_loglik(seq)

    def seq_log_density(self, x):
        """Score a batch of observation sequences."""
        return np.array([self.forward_loglik(list(seq)) for seq in x])

    def dist_to_encoder(self):
        """Return the pass-through explicit-duration sequence encoder."""
        return EDHMMDataEncoder()

    def estimator(self, pseudo_count=None):
        """Return the estimator for this explicit-duration HMM structure."""
        return EDHMMEstimator(self._emit_est, self.K, self.D, self.name)

    def to_structured_hmm(self, len_dist=None):
        """The HSMM as an EQUIVALENT StructuredHMM via the remaining-duration expansion: K*D sub-states
        (k, r) = "state k with r steps left in the segment". The expanded chain emits from state k at every
        sub-state, decrements deterministically (k,r)->(k,r-1), and at (k,1) switches segment with
        A[k,k']*dur[k'](d'). ``final_states`` = the (k,1) sub-states require the last segment to COMPLETE, so
        the expanded forward log-likelihood EQUALS this EDHMM's exactly. This hands the HSMM the full
        StructuredHMM read-out API -- Viterbi (recover state+remaining-duration), posterior decoding, the
        standard forward -- and, with ``len_dist``, enumeration. O(K*D) states."""

        def idx(k, r):  # r in 1..D
            return k * self.D + (r - 1)

        n = self.K * self.D
        emissions = [self.emissions[k] for k in range(self.K) for _ in range(self.D)]
        pi = np.zeros(n)
        for k in range(self.K):
            for d in range(1, self.D + 1):
                pi[idx(k, d)] = self.pi[k] * self.dur[k, d - 1]
        a = np.zeros((n, n))
        for k in range(self.K):
            for r in range(2, self.D + 1):
                a[idx(k, r), idx(k, r - 1)] = 1.0  # decrement remaining duration
            for kp in range(self.K):
                if kp == k:
                    continue
                for dp in range(1, self.D + 1):
                    a[idx(k, 1), idx(kp, dp)] = self.a[k, kp] * self.dur[kp, dp - 1]  # switch segment
        final = {idx(k, 1) for k in range(self.K)}
        return StructuredHMM(emissions, pi, DenseTransition(a), len_dist=len_dist, final_states=final)

    def enumerator(self, len_dist):
        """Enumerate observation sequences in descending marginal probability under this HSMM (complete
        final segment), given a ``len_dist`` over total sequence length. Built on the exact HMM expansion +
        the final-state best-first enumerator; ``.top_k(k)`` -> [(sequence, log_prob), ...]. Needs discrete
        (Categorical) emissions and a Categorical-like ``len_dist``."""
        return self.to_structured_hmm(len_dist=len_dist).enumerator()

    def state_posteriors(self, seq):
        """Per-position smoothing posteriors gamma[t, j] = P(z_t = j | obs), marginalizing the durations
        (sum the posterior of every segment that covers position t). Rows sum to 1."""
        log_b = self._log_b(seq)
        t_len = len(seq)
        log_alpha, log_e, seg = self._forward(log_b)
        log_beta, _ = self._backward(log_b, seg)
        z = _logsumexp(log_alpha[-1])
        log_dur = np.log(self.dur + 1e-300)
        occ = np.zeros((t_len, self.K))
        for t in range(t_len):
            for j in range(self.K):
                for d in range(min(t + 1, self.D)):
                    lp = log_e[t - d, j] + log_dur[j, d] + seg[t, d, j] + log_beta[t, j] - z
                    if np.isfinite(lp):
                        occ[t - d : t + 1, j] += np.exp(lp)
        return occ

    def posterior_decode(self, seq):
        """Per-position MAP state argmax_j P(z_t = j | obs)."""
        return np.argmax(self.state_posteriors(seq), axis=1)

    def viterbi_segments(self, seq):
        """Most-likely segmentation (max-product over the segment lattice): a list of (state, start,
        duration) segments covering the sequence, O(T K D). The HSMM analog of Viterbi decoding."""
        log_b = self._log_b(seq)
        t_len = len(seq)
        seg = self._seg_loglik(log_b)
        log_dur, log_a, log_pi = np.log(self.dur + 1e-300), np.log(self.a + 1e-300), np.log(self.pi + 1e-300)
        delta = np.full((t_len, self.K), -np.inf)  # best score of a segment ending at t in j
        bp_d = np.zeros((t_len, self.K), dtype=int)  # chosen duration index
        entry = np.full((t_len, self.K), -np.inf)  # best score to START a segment at t in j
        bp_prev = np.full((t_len, self.K), -1, dtype=int)  # previous state at a segment boundary
        entry[0] = log_pi
        for t in range(t_len):
            for j in range(self.K):
                best, bd = -np.inf, 0
                for d in range(min(t + 1, self.D)):
                    e = entry[t - d, j]
                    if np.isfinite(e):
                        val = e + log_dur[j, d] + seg[t, d, j]
                        if val > best:
                            best, bd = val, d
                delta[t, j], bp_d[t, j] = best, bd
            if t + 1 < t_len:
                for j in range(self.K):
                    vals = delta[t] + log_a[:, j]
                    entry[t + 1, j], bp_prev[t + 1, j] = float(vals.max()), int(vals.argmax())
        j = int(delta[t_len - 1].argmax())
        segments, t = [], t_len - 1
        while t >= 0:
            d = int(bp_d[t, j])
            start = t - d
            segments.append((int(j), int(start), d + 1))  # (state, start, duration)
            if start == 0:
                break
            j = int(bp_prev[start, j])
            t = start - 1
        segments.reverse()
        return segments

    def sampler(self, seed=None):
        """Return a sampler for explicit-duration HMM sequences."""
        return _EDHMMSampler(self, seed)


def _logsumexp(v):
    v = np.asarray(v, dtype=float)
    m = v.max()
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(v - m))))


class _EDHMMSampler:
    def __init__(self, hmm, seed=None):
        self.hmm = hmm
        self.rng = np.random.RandomState(seed)

    def sample(self, length):
        h = self.hmm
        out = []
        s = self.rng.choice(h.K, p=h.pi)
        while len(out) < length:
            d = self.rng.choice(h.D, p=h.dur[s]) + 1
            for _ in range(d):
                if len(out) >= length:
                    break
                out.append(h.emissions[s].sampler(seed=int(self.rng.randint(1, 2**31))).sample())
            s = self.rng.choice(h.K, p=h.a[s])
        return out


class EDHMMDataEncoder(DataSequenceEncoder):
    """An ExplicitDurationHMM record is one observation sequence (the durations are latent)."""

    def seq_encode(self, x):
        """Encode EDHMM records as observation-sequence lists."""
        return [list(s) for s in x]

    def __eq__(self, other):
        return isinstance(other, EDHMMDataEncoder)

    def __hash__(self):
        return hash("EDHMMDataEncoder")


class EDHMMAccumulator(SequenceEncodableStatisticAccumulator):
    """E-step accumulator for an explicit-duration HMM: per-sequence segment posteriors -> initial /
    transition / per-state DURATION counts + emission occupancy statistics."""

    def __init__(self, emission_accumulators, k, d):
        self.emit = list(emission_accumulators)
        self.K, self.D = int(k), int(d)
        self.pi_acc = np.zeros(self.K)
        self.trans_acc = np.zeros((self.K, self.K))
        self.dur_acc = np.zeros((self.K, self.D))
        self.nk = np.zeros(self.K)

    def update(self, x, weight, estimate):
        """Accumulate sufficient statistics from one weighted sequence."""
        self.seq_update([x], np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights, estimate):
        """Accumulate weighted explicit-duration HMM statistics from a batch."""
        for seq, w in zip(x, np.asarray(weights, dtype=float)):
            if not seq:
                continue
            _, pic, trc, drc, occ = estimate._estep(list(seq))
            self.pi_acc += w * pic
            self.trans_acc += w * trc
            self.dur_acc += w * drc
            for k in range(self.K):
                enc = estimate.emissions[k].dist_to_encoder().seq_encode(seq)
                wk = occ[:, k] * w
                self.emit[k].seq_update(enc, wk, estimate.emissions[k])
                self.nk[k] += float(wk.sum())

    def seq_initialize(self, x, weights, rng):
        """Initialize explicit-duration HMM statistics with random soft responsibilities."""
        self.trans_acc += rng.random((self.K, self.K))
        np.fill_diagonal(self.trans_acc, 0.0)
        self.dur_acc += rng.random((self.K, self.D))
        for seq, w in zip(x, np.asarray(weights, dtype=float)):
            if not seq:
                continue
            g = rng.dirichlet(np.ones(self.K), len(seq))
            self.pi_acc += w * g[0]
            for k in range(self.K):
                enc = self.emit[k].acc_to_encoder().seq_encode(seq)
                wk = g[:, k] * w
                self.emit[k].seq_initialize(enc, wk, rng)
                self.nk[k] += float(wk.sum())

    def combine(self, suff_stat):
        """Merge serialized explicit-duration HMM sufficient statistics."""
        pi_acc, trans_acc, dur_acc, emit_vals, nk = suff_stat
        self.pi_acc += pi_acc
        self.trans_acc += trans_acc
        self.dur_acc += dur_acc
        self.nk += nk
        for k in range(self.K):
            self.emit[k].combine(emit_vals[k])
        return self

    def value(self):
        """Return serialized explicit-duration HMM sufficient statistics."""
        return (
            self.pi_acc.copy(),
            self.trans_acc.copy(),
            self.dur_acc.copy(),
            [e.value() for e in self.emit],
            self.nk.copy(),
        )

    def from_value(self, x):
        """Restore accumulator state from serialized EDHMM statistics."""
        self.pi_acc, self.trans_acc, self.dur_acc, emit_vals, self.nk = (
            x[0].copy(),
            x[1].copy(),
            x[2].copy(),
            x[3],
            x[4].copy(),
        )
        for k in range(self.K):
            self.emit[k].from_value(emit_vals[k])
        return self

    def acc_to_encoder(self):
        """Return the encoder associated with this accumulator."""
        return EDHMMDataEncoder()


class EDHMMAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for explicit-duration HMM accumulators."""

    def __init__(self, emission_estimators, k, d):
        self.emission_estimators = emission_estimators
        self.k, self.d = k, d

    def make(self):
        """Create a fresh explicit-duration HMM accumulator."""
        emit = [est.accumulator_factory().make() for est in self.emission_estimators]
        return EDHMMAccumulator(emit, self.k, self.d)


class EDHMMEstimator(ParameterEstimator):
    """Estimator (M-step) for an :class:`ExplicitDurationHMM`: re-estimates pi, the zero-diagonal transition,
    the per-state DURATION distributions, and each state's emission from the segment-posterior statistics."""

    def __init__(self, emission_estimators, k, d, name=None):
        self.emission_estimators = list(emission_estimators)
        self.k, self.d = int(k), int(d)
        self.name = name

    def accumulator_factory(self):
        """Return the accumulator factory used by this estimator."""
        return EDHMMAccumulatorFactory(self.emission_estimators, self.k, self.d)

    def estimate(self, nobs, suff_stat):
        """Estimate an explicit-duration HMM from segment posterior statistics."""
        pi_acc, trans_acc, dur_acc, emit_vals, nk = suff_stat
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(self.k) / self.k
        a = trans_acc.copy()
        np.fill_diagonal(a, 0.0)
        a = _row_normalize(a) if a.sum() > 0 else np.full((self.k, self.k), 1.0 / max(self.k - 1, 1))
        dur = _row_normalize(dur_acc)
        emissions = [self.emission_estimators[i].estimate(float(nk[i]), emit_vals[i]) for i in range(self.k)]
        return ExplicitDurationHMM(emissions, pi, a, dur, self.d, name=self.name)


SequenceEncodableProbabilityDistribution.register(ExplicitDurationHMM)


def jit_forward_loglik(hmm: StructuredHMM):
    """Compile the scaled forward log-likelihood recursion to a single jax.jit XLA program (lax.scan over
    time). Returns a callable ``score(seq) -> float``: emission log-densities are evaluated on the host
    (arbitrary emissions), then the forward scan runs jitted on the transition matrix. Works for any
    operator (uses ``as_matrix()``); the win is large T / K. Requires the JAX optional extra."""
    import jax
    import jax.numpy as jnp

    a_mat = jnp.asarray(hmm.transition.as_matrix())
    pi = jnp.asarray(hmm.pi)

    @jax.jit
    def _fwd(log_b):
        mx = log_b.max(axis=1, keepdims=True)
        b = jnp.exp(log_b - mx)
        a0 = pi * b[0]
        c0 = a0.sum()

        def step(alpha, bt):
            a2 = (alpha @ a_mat) * bt
            c = a2.sum()
            return a2 / c, jnp.log(c)

        _, logc = jax.lax.scan(step, a0 / c0, b[1:])
        return jnp.sum(logc) + jnp.log(c0) + jnp.sum(mx)

    def score(seq):
        return float(_fwd(jnp.asarray(hmm._log_b(seq))))

    return score
