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
    ``backward(v) == A @ v``); the cheap operators never materialize ``A``.
    """

    n_states: int

    def forward(self, alpha: np.ndarray) -> np.ndarray:  # alpha @ A
        raise NotImplementedError

    def backward(self, v: np.ndarray) -> np.ndarray:  # A @ v
        raise NotImplementedError

    def as_matrix(self) -> np.ndarray:
        raise NotImplementedError

    # --- M-step: accumulate expected transition mass over a sequence, then re-estimate ---
    def new_accumulator(self) -> Any:
        raise NotImplementedError

    def accumulate(self, acc: Any, alpha_t: np.ndarray, w_next: np.ndarray, scale: float) -> None:
        """Add one transition's expected mass. ``alpha_t`` is the (normalized) forward belief at t,
        ``w_next = b_{t+1} * beta_{t+1}`` the emission-weighted backward belief at t+1, ``scale`` the
        forward normalizer ``c_{t+1}`` (so the per-step posterior transition mass is exact)."""
        raise NotImplementedError

    def estimate(self, acc: Any) -> TransitionOperator:
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
        return alpha @ self.a

    def backward(self, v):
        return self.a @ v

    def as_matrix(self):
        return self.a

    def new_accumulator(self):
        return np.zeros_like(self.a)

    def accumulate(self, acc, alpha_t, w_next, scale):
        acc += np.outer(alpha_t, w_next) * (self.a / max(scale, 1e-300))

    def estimate(self, acc):
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
        return (alpha @ self.g) @ self.phi  # (alpha^T A)

    def backward(self, v):
        return self.g @ (self.phi @ v)  # (A v)

    def as_matrix(self):
        return self.g @ self.phi

    def new_accumulator(self):
        return [np.zeros_like(self.g), np.zeros_like(self.phi)]  # [n (K,r), m (r,K)]

    def accumulate(self, acc, alpha_t, w_next, scale):
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
        return alpha @ self.a

    def backward(self, v):
        return self.a @ v

    def as_matrix(self):
        return np.asarray(self.a.todense())

    def new_accumulator(self):
        return np.zeros(len(self.rows))  # one expected count per allowed edge

    def accumulate(self, acc, alpha_t, w_next, scale):
        acc += alpha_t[self.rows] * w_next[self.cols] * self._edge_vals / max(scale, 1e-300)

    def estimate(self, acc):
        return SparseTransition(self.n_states, list(zip(self.rows.tolist(), self.cols.tolist())), acc)


def left_to_right_edges(n_states: int, skip: int = 1):
    """Edges for a left-to-right (Bakis) HMM: each state may stay or advance up to ``skip`` states."""
    return [(i, j) for i in range(n_states) for j in range(i, min(n_states, i + skip + 1))]


def banded_edges(n_states: int, bandwidth: int = 1):
    """Edges for a banded transition: state i connects to i-bandwidth .. i+bandwidth (local time-series)."""
    return [(i, j) for i in range(n_states) for j in range(max(0, i - bandwidth), min(n_states, i + bandwidth + 1))]


class StructuredHMM:
    """An HMM whose transition is a :class:`TransitionOperator` (dense / low-rank / a combinator).

    ``emissions`` is one observation distribution per state; ``pi`` the initial-state distribution;
    ``transition`` any ``TransitionOperator``. The scaled forward-backward and EM call the operator's
    ``forward``/``backward``/``accumulate``/``estimate``, so a low-rank or factorial transition runs the
    SAME inference at its own cost (O(K r) for low-rank). ``emission_estimators`` (one per state) drives
    the emission M-step; default reuses ``emissions[k].estimator()``.
    """

    def __init__(
        self, emissions, pi, transition: TransitionOperator, emission_estimators=None, keys=(None, None), name=None
    ) -> None:
        self.emissions = list(emissions)
        self.pi = np.asarray(pi, dtype=float)
        self.transition = transition
        self.K = len(self.emissions)
        self.keys = tuple(keys)  # (init_key, trans_key) for parameter tying across models
        self.name = name
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

    def state_posteriors(self, seq):
        """The full smoothing posteriors gamma[t,k] = P(z_t = k | x)."""
        return self._forward_backward(self._log_b(seq))[4]

    def seq_log_density(self, seqs) -> np.ndarray:
        return np.array([self._forward_backward(self._log_b(s))[5] for s in seqs])

    def sampler(self, seed=None):
        return _StructuredHMMSampler(self, seed)

    def fit(self, seqs, *, max_its: int = 50, tol: float = 1e-6):
        """EM (Baum-Welch) through the transition operator. Returns ``(fitted_hmm, loglik_trace)``."""
        seqs = [list(s) for s in seqs]
        ll_trace = []
        for _ in range(int(max_its)):
            trans_acc = self.transition.new_accumulator()
            pi_acc = np.zeros(self.K)
            emit_accs = [est.accumulator_factory().make() for est in self._emit_est]
            nk = np.zeros(self.K)
            total_ll = 0.0
            for seq in seqs:
                if not seq:
                    continue
                log_b = self._log_b(seq)
                alpha, beta, c, b, gamma, ll = self._forward_backward(log_b)
                total_ll += ll
                pi_acc += gamma[0]
                for t in range(len(seq) - 1):
                    self.transition.accumulate(trans_acc, alpha[t], b[t + 1] * beta[t + 1], c[t + 1])
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
            s = self.rng.choice(h.K, p=a[s])
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
        out = np.zeros(self.n_states)
        for b, sl in zip(self.blocks, self._slices()):
            out[sl] = b.forward(alpha[sl])
        return out

    def backward(self, v):
        out = np.zeros(self.n_states)
        for b, sl in zip(self.blocks, self._slices()):
            out[sl] = b.backward(v[sl])
        return out

    def as_matrix(self):
        from scipy.linalg import block_diag

        return block_diag(*[b.as_matrix() for b in self.blocks])

    def new_accumulator(self):
        return [b.new_accumulator() for b in self.blocks]

    def accumulate(self, acc, alpha_t, w_next, scale):
        for b, a, sl in zip(self.blocks, acc, self._slices()):
            b.accumulate(a, alpha_t[sl], w_next[sl], scale)

    def estimate(self, acc):
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
        m = alpha.reshape(self.k1, self.k2)
        return (self._a1().T @ m @ self._a2()).reshape(-1)  # alpha @ (A1 (x) A2)

    def backward(self, v):
        m = v.reshape(self.k1, self.k2)
        return (self._a1() @ m @ self._a2().T).reshape(-1)  # (A1 (x) A2) @ v

    def as_matrix(self):
        return np.kron(self._a1(), self._a2())

    def new_accumulator(self):
        return [np.zeros((self.k1, self.k1)), np.zeros((self.k2, self.k2))]  # marginal factor counts

    def accumulate(self, acc, alpha_t, w_next, scale):
        a1, a2 = self._a1(), self._a2()
        am = alpha_t.reshape(self.k1, self.k2)
        wm = w_next.reshape(self.k1, self.k2)
        inv = 1.0 / max(scale, 1e-300)
        acc[0] += a1 * (am @ (a2 @ wm.T)) * inv  # n1[i1,j1] = A1[i1,j1] * sum_{i2,j2} xi
        acc[1] += a2 * (am.T @ (a1 @ wm)) * inv  # n2[i2,j2] = A2[i2,j2] * sum_{i1,j1} xi

    def estimate(self, acc):
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
        self.seq_update([x], np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights, estimate):
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
        pi_acc, trans_acc, emit_vals, nk = suff_stat
        self.pi_acc += pi_acc
        self.trans_acc = _add_nested(self.trans_acc, trans_acc)
        self.nk += nk
        for k in range(self.K):
            self.emit[k].combine(emit_vals[k])
        return self

    def value(self):
        return (self.pi_acc.copy(), self.trans_acc, [e.value() for e in self.emit], self.nk.copy())

    def from_value(self, x):
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
        return StructuredHMMDataEncoder()

    # parameter tying: pool initial / transition counts across accumulators sharing a key
    def key_merge(self, store):
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
        if self.init_key is not None and self.init_key in store:
            self.pi_acc = store[self.init_key]
        if self.trans_key is not None and self.trans_key in store:
            self.trans_acc = store[self.trans_key]
        for e in self.emit:
            if hasattr(e, "key_replace"):
                e.key_replace(store)


class StructuredHMMAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, emission_estimators, transition_proto, keys):
        self.emission_estimators = emission_estimators
        self.transition_proto = transition_proto
        self.keys = keys

    def make(self):
        emit = [est.accumulator_factory().make() for est in self.emission_estimators]
        return StructuredHMMAccumulator(emit, self.transition_proto, self.keys)


class StructuredHMMEstimator(ParameterEstimator):
    """Estimator (M-step) for a :class:`StructuredHMM`: re-estimates pi, the transition OPERATOR (any
    structure -- dense/low-rank/combinator), and each state's emission from the Baum-Welch statistics.
    ``keys=(init_key, trans_key)`` tie the initial / transition parameters across HMMs that share them."""

    def __init__(self, emission_estimators, transition_proto, keys=(None, None), name=None):
        self.emission_estimators = list(emission_estimators)
        self.transition_proto = transition_proto
        self.keys = tuple(keys)
        self.name = name

    def accumulator_factory(self):
        return StructuredHMMAccumulatorFactory(self.emission_estimators, self.transition_proto, self.keys)

    def estimate(self, nobs, suff_stat):
        pi_acc, trans_acc, emit_vals, nk = suff_stat
        pi = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(len(pi_acc)) / len(pi_acc)
        transition = self.transition_proto.estimate(trans_acc)
        emissions = [self.emission_estimators[k].estimate(float(nk[k]), emit_vals[k]) for k in range(len(emit_vals))]
        return StructuredHMM(emissions, pi, transition, self.emission_estimators, self.keys, self.name)


# --- make StructuredHMM satisfy the distribution side of the contract -------------------------------
def _structured_hmm_log_density(self, x):
    return self._forward_backward(self._log_b(x))[5]


def _structured_hmm_dist_to_encoder(self):
    return StructuredHMMDataEncoder()


def _structured_hmm_estimator(self, pseudo_count=None):
    return StructuredHMMEstimator(self._emit_est, self.transition, self.keys, self.name)


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
