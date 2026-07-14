"""Variational multi-hop attention: a 2-hop chain over TIED latent embeddings, with prior annealing.

This combines the two hard pieces: the multi-hop chain (:mod:`chained_attention`) and tied latent
embeddings (:mod:`variational_embedding_attention`). Each context position is a ``(key, value)``; a
single latent embedding ``e_s`` per symbol is used in every role (query, key, value). Hop 1 attends the
query embedding to the key embeddings; the attended position's *value* embedding becomes the hop-2
query; hop 2 attends again; the target is emitted from the final attended value. The two hop latents are
summed exactly (an ``N x N`` table); the embeddings are latent with a mean-field posterior
``q(e_s)=N(m_s, v_s)`` fit by a reparameterized-ELBO gradient step (the embedding M-step has no closed
form -- the softmax partition supplies the repulsion that prevents collapse, and it is not quadratic).

Because tying makes identity matching trivial, the ``N(0,I)`` prior would otherwise collapse the unused
embeddings; the estimator **anneals** the prior weight from ~0 upward over EM iterations so the data
spreads the embeddings first. Observation: ``(context_keys, context_values, query_symbol, target)``.

References: multi-hop attention = Memory Networks (Sukhbaatar et al. 2015); attention as a variational
latent variable = Deng et al. 2018. The annealing is the practical face of Deterministic Annealing EM
(Ueda & Nakano 1998) -- tempering the objective to escape the collapsed fixed point and reach an
initialization-independent solution. (We checked: principled DAEM tempering does not improve the
*closed-form* chained head, which is already at its initialization-independent global optimum; the
annealing is only load-bearing here, where the latent-embedding prior creates the collapse basin.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _softmax(s: np.ndarray, axis: int) -> np.ndarray:
    s = s - s.max(axis=axis, keepdims=True)
    w = np.exp(s)
    return w / w.sum(axis=axis, keepdims=True)


def _two_hop(embed, keys, vals, q, t, emission, sigma2):
    """Forward + per-symbol gradient + emission-aware final responsibilities for the 2-hop chain."""
    eq = embed[q]  # (n, D)
    ek = embed[keys]  # (n, N, D)
    d1 = eq[:, None, :] - ek
    a1 = _softmax(-np.sum(d1 * d1, axis=2) / (2 * sigma2), axis=1)  # (n, N)
    ev = embed[vals]  # (n, N, D)
    d2 = ev[:, :, None, :] - ek[:, None, :, :]  # (n, i, j, D)
    a2 = _softmax(-np.sum(d2 * d2, axis=3) / (2 * sigma2), axis=2)  # (n, i, j)
    bj = emission[vals, t[:, None]]  # (n, N)  emission of each final value at the target
    h = np.einsum("nij,nj->ni", a2, bj)  # (n, N)
    p = np.clip(np.einsum("ni,ni->n", a1, h), 1e-12, None)
    gs1 = a1 * (h - p[:, None]) / p[:, None]  # (n, N)
    gs2 = a1[:, :, None] * a2 * (bj[:, None, :] - h[:, :, None]) / p[:, None, None]  # (n, i, j)
    grad = np.zeros_like(embed)
    np.add.at(grad, q, -np.einsum("ni,nij->nj", gs1, d1) / sigma2)
    np.add.at(grad, keys.reshape(-1), (gs1[:, :, None] * d1 / sigma2).reshape(-1, embed.shape[1]))
    np.add.at(grad, vals.reshape(-1), (-np.einsum("nij,nijd->nid", gs2, d2) / sigma2).reshape(-1, embed.shape[1]))
    np.add.at(grad, keys.reshape(-1), (np.einsum("nij,nijd->njd", gs2, d2) / sigma2).reshape(-1, embed.shape[1]))
    rj = (a1[:, :, None] * a2 * bj[:, None, :]).sum(axis=1) / p[:, None]  # (n, N) final-position posterior
    return p, grad, rj


class VariationalMultiHopAttentionDistribution(SequenceEncodableProbabilityDistribution):
    """A 2-hop chain over tied latent embeddings (mean-field posterior)."""

    def __init__(self, mean, log_var, emission, sigma2: float = 0.3, name: str | None = None) -> None:
        """Args:
        mean / log_var: ``(S, D)`` posterior mean / log-variance of the tied embeddings.
        emission: ``(S, T)`` per-(value-symbol) categorical over targets.
        sigma2: gate variance (attention temperature).
        name: optional name.
        """
        self.mean = np.asarray(mean, dtype=float)
        self.log_var = np.asarray(log_var, dtype=float)
        self.emission = np.asarray(emission, dtype=float)
        self.num_symbols, self.embed_dim = self.mean.shape
        self.num_targets = self.emission.shape[1]
        self.sigma2 = float(sigma2)
        self.name = name

    def __str__(self) -> str:
        return "VariationalMultiHopAttentionDistribution(S=%d, D=%d, T=%d, name=%s)" % (
            self.num_symbols,
            self.embed_dim,
            self.num_targets,
            repr(self.name),
        )

    def density(self, x) -> float:
        """Return the probability of one context/query/target observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x) -> float:
        """Return the log-probability of one context/query/target observation."""
        enc = self.dist_to_encoder().seq_encode([x])
        return float(self.seq_log_density(enc)[0])

    def seq_log_density(self, x) -> np.ndarray:
        """Return vectorized log-probabilities for encoded two-hop attention observations."""
        keys, vals, q, t = x
        p, _, _ = _two_hop(self.mean, keys, vals, q, t, self.emission, self.sigma2)
        return np.log(p)

    def predict_proba(self, context_keys, context_values, query) -> np.ndarray:
        """Predictive target distribution (posterior-mean embeddings); ``(T,)`` or ``(n, T)``."""
        single = np.ndim(context_keys) == 1
        keys = np.atleast_2d(np.asarray(context_keys, dtype=int))
        vals = np.atleast_2d(np.asarray(context_values, dtype=int))
        q = np.atleast_1d(np.asarray(query, dtype=int))
        m = self.mean
        eq, ek, ev = m[q], m[keys], m[vals]
        a1 = _softmax(-np.sum((eq[:, None] - ek) ** 2, 2) / (2 * self.sigma2), 1)
        d2 = ev[:, :, None, :] - ek[:, None, :, :]
        a2 = _softmax(-np.sum(d2 * d2, 3) / (2 * self.sigma2), 2)
        pred = np.einsum("ni,nij,njt->nt", a1, a2, self.emission[vals])
        return pred[0] if single else pred

    def embeddings(self) -> np.ndarray:
        """Return posterior mean embeddings for the tied latent symbols."""
        return self.mean

    def sampler(self, seed: int | None = None) -> VariationalMultiHopAttentionSampler:
        """Return a sampler for synthetic two-hop attention observations."""
        return VariationalMultiHopAttentionSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> VariationalMultiHopAttentionEstimator:
        """Return a variational EM estimator initialized with this model's dimensions."""
        return VariationalMultiHopAttentionEstimator(
            num_symbols=self.num_symbols,
            embed_dim=self.embed_dim,
            num_targets=self.num_targets,
            sigma2=self.sigma2,
            name=self.name,
        )

    def dist_to_encoder(self) -> VariationalMultiHopAttentionDataEncoder:
        """Return the encoder for context keys, values, query symbols, and targets."""
        return VariationalMultiHopAttentionDataEncoder()


class VariationalMultiHopAttentionSampler(DistributionSampler):
    """Sample two-hop attention observations from posterior-mean embeddings plus embedding noise."""

    def __init__(self, dist, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one observation or ``size`` iid synthetic observations."""
        n = 1 if size is None else size
        d = self.dist
        N = 6
        embed = d.mean + np.exp(0.5 * d.log_var) * self.rng.randn(*d.mean.shape)
        out = []
        for _ in range(n):
            keys = self.rng.randint(0, d.num_symbols, N)
            vals = self.rng.randint(0, d.num_symbols, N)
            q = int(self.rng.randint(0, d.num_symbols))
            a1 = _softmax(-np.sum((embed[q][None, None] - embed[keys][None]) ** 2, 2) / (2 * d.sigma2), 1)[0]
            i = self.rng.choice(N, p=a1)
            a2 = _softmax(-np.sum((embed[vals[i]][None, None] - embed[keys][None]) ** 2, 2) / (2 * d.sigma2), 1)[0]
            j = self.rng.choice(N, p=a2)
            t = int(self.rng.choice(d.num_targets, p=d.emission[vals[j]]))
            out.append((keys, vals, q, t))
        return out[0] if size is None else out


class VariationalMultiHopAttentionAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate Monte-Carlo ELBO gradients and emission responsibilities for variational EM."""

    def __init__(self, num_symbols, embed_dim, num_targets, mc, seed, keys=None, name=None) -> None:
        self.num_symbols = num_symbols
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        self.mc = mc
        self._rng = RandomState(seed)
        self.grad_m = np.zeros((num_symbols, embed_dim))
        self.grad_logv = np.zeros((num_symbols, embed_dim))
        self.emission_count = np.zeros((num_symbols, num_targets))
        self.ll = 0.0
        self.n = 0.0
        self.keys = keys
        self.name = name

    def seq_update(self, x, weights, estimate) -> None:
        """Update ELBO gradients and emission counts from encoded observations."""
        keys, vals, q, t = x
        w = np.asarray(weights, dtype=float)
        m, log_v, sig = estimate.mean, estimate.log_var, estimate.sigma2
        s = np.exp(0.5 * log_v)
        for _ in range(self.mc):
            eps = self._rng.randn(*m.shape)
            embed = m + s * eps
            p, gE, rj = _two_hop(embed, keys, vals, q, t, estimate.emission, sig)
            self.grad_m += gE / self.mc
            self.grad_logv += (gE * eps * s * 0.5) / self.mc
            self.ll += float(np.dot(w, np.log(p))) / self.mc
            np.add.at(
                self.emission_count,
                (vals.reshape(-1), np.repeat(t, keys.shape[1])),
                (rj * w[:, None] / self.mc).reshape(-1),
            )
        self.n += float(w.sum())

    def seq_initialize(self, x, weights, rng: RandomState) -> None:
        """Initialize emission counts with random final-hop responsibilities."""
        keys, vals, q, t = x
        n, N = keys.shape
        w = np.asarray(weights, dtype=float)
        rj = rng.dirichlet(np.ones(N), size=n) * w[:, None]
        np.add.at(self.emission_count, (vals.reshape(-1), np.repeat(t, N)), rj.reshape(-1))
        self.n += float(w.sum())

    def update(self, x, weight, estimate) -> None:
        """Update from one weighted two-hop attention observation."""
        enc = VariationalMultiHopAttentionDataEncoder().seq_encode([x])
        self.seq_update(enc, np.array([weight], dtype=float), estimate)

    def initialize(self, x, weight, rng) -> None:
        """Initialize from one weighted two-hop attention observation."""
        enc = VariationalMultiHopAttentionDataEncoder().seq_encode([x])
        self.seq_initialize(enc, np.array([weight], dtype=float), rng)

    def combine(self, suff_stat):
        """Merge variational gradients, emission counts, log-likelihood, and weight totals."""
        gm, glv, ec, ll, n = suff_stat
        self.grad_m += gm
        self.grad_logv += glv
        self.emission_count += ec
        self.ll += ll
        self.n += n
        return self

    def value(self):
        """Return accumulated gradients, emission counts, log-likelihood, and total weight."""
        return (self.grad_m.copy(), self.grad_logv.copy(), self.emission_count.copy(), self.ll, self.n)

    def from_value(self, x):
        """Restore accumulator state from ``value`` output."""
        self.grad_m, self.grad_logv, self.emission_count = (np.asarray(v, dtype=float) for v in x[:3])
        self.ll = float(x[3])
        self.n = float(x[4])
        return self

    def acc_to_encoder(self):
        """Return the encoder compatible with this attention accumulator."""
        return VariationalMultiHopAttentionDataEncoder()


class VariationalMultiHopAttentionAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for variational multi-hop attention EM steps."""

    def __init__(self, estimator, keys=None, name=None) -> None:
        self.est = estimator
        self.keys = keys
        self.name = name

    def make(self):
        """Create an accumulator with a deterministic per-iteration Monte-Carlo seed."""
        e = self.est
        seed = (e.seed * 1_000_003 + e._t) % (2**31)
        return VariationalMultiHopAttentionAccumulator(
            e.num_symbols, e.embed_dim, e.num_targets, e.mc, seed, keys=self.keys, name=self.name
        )


class VariationalMultiHopAttentionEstimator(ParameterEstimator):
    """Variational-EM estimator with prior annealing (KL weight ramped over EM iterations)."""

    def __init__(
        self,
        num_symbols: int,
        embed_dim: int,
        num_targets: int,
        *,
        sigma2: float = 0.3,
        lr: float = 0.05,
        mc: int = 5,
        prior_strength: float = 0.1,
        anneal_iters: int = 100,
        emission_smoothing: float = 1e-4,
        seed: int = 0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Args:
        num_symbols, embed_dim, num_targets: dimensions ``S, D, T``.
        sigma2: gate variance.
        lr: Adam learning rate for the embedding E-step.
        mc: Monte-Carlo samples for the reparameterized ELBO gradient.
        prior_strength: final weight on the ``N(0, I)`` prior (KL term).
        anneal_iters: ramp the prior weight linearly from 0 to ``prior_strength`` over this many EM
            iterations (prevents the unused embeddings collapsing before the data spreads them).
        emission_smoothing / seed / name / keys: standard controls.
        """
        self.num_symbols = num_symbols
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        self.sigma2 = float(sigma2)
        self.lr = float(lr)
        self.mc = int(mc)
        self.prior_strength = float(prior_strength)
        self.anneal_iters = int(anneal_iters)
        self.emission_smoothing = float(emission_smoothing)
        self.seed = int(seed)
        self.name = name
        self.keys = keys
        self.mean = self.log_var = None
        self._am = self._av = self._bm = self._bv = None
        self._t = 0

    def accumulator_factory(self):
        """Return a factory for variational multi-hop attention accumulators."""
        return VariationalMultiHopAttentionAccumulatorFactory(self, keys=self.keys, name=self.name)

    def _adam(self, param, grad, m1, m2):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        return param + self.lr * (m1 / (1 - b1**self._t)) / (np.sqrt(m2 / (1 - b2**self._t)) + eps), m1, m2

    def estimate(self, nobs, suff_stat):
        """Apply one variational EM update and return the updated attention distribution."""
        grad_m, grad_logv, emission_count, _ll, _n = suff_stat
        if self.mean is None:
            rng = RandomState(self.seed)
            self.mean = rng.randn(self.num_symbols, self.embed_dim)
            self.log_var = np.full((self.num_symbols, self.embed_dim), np.log(0.3))
            self._am = np.zeros_like(self.mean)
            self._av = np.zeros_like(self.mean)
            self._bm = np.zeros_like(self.log_var)
            self._bv = np.zeros_like(self.log_var)
        else:
            self._t += 1
            ps = self.prior_strength * min(1.0, self._t / max(1, self.anneal_iters))  # annealed KL weight
            v = np.exp(self.log_var)
            g_m = grad_m - ps * self.mean
            g_logv = grad_logv - ps * 0.5 * (v - 1.0)
            self.mean, self._am, self._av = self._adam(self.mean, g_m, self._am, self._av)
            self.log_var, self._bm, self._bv = self._adam(self.log_var, g_logv, self._bm, self._bv)
            self.log_var = np.clip(self.log_var, -8.0, 2.0)
        em = emission_count + self.emission_smoothing
        emission = em / em.sum(axis=1, keepdims=True)
        return VariationalMultiHopAttentionDistribution(
            self.mean.copy(), self.log_var.copy(), emission, sigma2=self.sigma2, name=self.name
        )


class VariationalMultiHopAttentionDataEncoder(DataSequenceEncoder):
    """Encode context keys, context values, query symbols, and targets as integer arrays."""

    def __str__(self) -> str:
        return "VariationalMultiHopAttentionDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VariationalMultiHopAttentionDataEncoder)

    def seq_encode(self, x: Sequence[tuple[Any, Any, int, int]]):
        """Encode ``(context_keys, context_values, query, target)`` observations."""
        keys = np.asarray([np.asarray(xi[0], dtype=int) for xi in x], dtype=int)
        vals = np.asarray([np.asarray(xi[1], dtype=int) for xi in x], dtype=int)
        q = np.asarray([int(xi[2]) for xi in x], dtype=int)
        t = np.asarray([int(xi[3]) for xi in x], dtype=int)
        return keys, vals, q, t


__all__ = [
    "VariationalMultiHopAttentionDistribution",
    "VariationalMultiHopAttentionSampler",
    "VariationalMultiHopAttentionAccumulator",
    "VariationalMultiHopAttentionAccumulatorFactory",
    "VariationalMultiHopAttentionEstimator",
    "VariationalMultiHopAttentionDataEncoder",
]
