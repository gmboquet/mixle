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


def _row_normalize(m: np.ndarray) -> np.ndarray:
    m = np.maximum(m, 0.0)
    s = m.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return m / s


class DenseTransition(TransitionOperator):
    """The usual dense K x K row-stochastic transition (O(K^2) forward-backward)."""

    def __init__(self, a: np.ndarray) -> None:
        self.a = np.asarray(a, dtype=float)
        self.n_states = self.a.shape[0]

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
        return DenseTransition(_row_normalize(acc))


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


class StructuredHMM:
    """An HMM whose transition is a :class:`TransitionOperator` (dense / low-rank / a combinator).

    ``emissions`` is one observation distribution per state; ``pi`` the initial-state distribution;
    ``transition`` any ``TransitionOperator``. The scaled forward-backward and EM call the operator's
    ``forward``/``backward``/``accumulate``/``estimate``, so a low-rank or factorial transition runs the
    SAME inference at its own cost (O(K r) for low-rank). ``emission_estimators`` (one per state) drives
    the emission M-step; default reuses ``emissions[k].estimator()``.
    """

    def __init__(self, emissions, pi, transition: TransitionOperator, emission_estimators=None) -> None:
        self.emissions = list(emissions)
        self.pi = np.asarray(pi, dtype=float)
        self.transition = transition
        self.K = len(self.emissions)
        self._emit_est = emission_estimators or [e.estimator() for e in self.emissions]

    def _log_b(self, seq) -> np.ndarray:
        return np.array([[float(e.log_density(x)) for e in self.emissions] for x in seq])

    def _forward_backward(self, log_b):
        T, _ = log_b.shape
        op = self.transition
        mx = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - mx)  # (T,K) scaled emissions
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = self.pi * b[0]
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
