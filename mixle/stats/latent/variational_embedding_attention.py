"""Variational-EM responsibility attention with latent, tied token embeddings.

The plain :mod:`responsibility_attention` head learns a *key* vector per symbol from an *observed*
query, by closed-form EM. To learn a genuinely **tied** embedding -- one shared latent vector ``e_s``
used in *both* the query and the key role -- the embedding has to become a latent variable, because as
a parameter its M-step is non-closed-form (it appears on both sides of the gate). Its posterior is
intractable, so it is approximated **variationally** with a mean-field ``q(e_s) = N(m_s, v_s)``. The
discrete attention stays an exact-EM latent.

Generative model for one observation ``(context_symbols, query_symbol, target)``:

    e_s  ~ N(0, I)                                       # latent tied embedding per symbol
    a_i  ∝ pi_i exp(-||e_q - e_{c_i}||^2 / 2 sigma2)     # generative gate: attend where embeddings match
    t    ~ Categorical( sum_i a_i emission[c_i, :] )     # target read from the attended symbol

Fitting is **variational EM**, run through the ordinary :func:`mixle.inference.optimize` loop:
  * the *variational E-step* takes one reparameterized-ELBO gradient step on the embedding posterior
    ``(m, v)`` -- the per-observation ELBO-gradient is additive, so it accumulates like any sufficient
    statistic; the Adam optimizer state lives on the estimator (which persists across EM iterations),
    keeping the accumulator purely additive;
  * the *M-step* updates the emission and position prior in closed form.

Because tying ties the query and key roles, a symbol's embedding learned from its appearances as a key
(in context) also serves as its query -- so the model transfers a representation across roles, which a
lookup/one-hot query cannot. Caveats: the objective is an ELBO (a bound, monotone in the bound,
not the exact likelihood); the embedding posterior is a *global* latent fit by an inner gradient step,
so this estimator is single-process for the embedding update (the discrete-attention / emission /
prior parts remain ordinary additive EM).

References: variational attention as a latent variable (Deng, Kim, Chiu, Guo & Rush 2018); latent
coordinates / embeddings via variational inference (Titsias & Lawrence 2010, Bayesian GP-LVM, which
likewise makes nonlinearly-appearing latents tractable variationally). The reparameterized-ELBO E-step
follows their recommendation of low-variance variational gradients over REINFORCE-style hard attention.
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


def _softmax_rows(s: np.ndarray) -> np.ndarray:
    s = s - s.max(axis=1, keepdims=True)
    w = np.exp(s)
    return w / w.sum(axis=1, keepdims=True)


def _attention(embed: np.ndarray, ctx: np.ndarray, q: np.ndarray, sigma2: float, log_pi: np.ndarray):
    """Attention weights ``a`` (n,N) and the (query-key) difference tensor ``d`` (n,N,D)."""
    eq = embed[q]  # (n, D)
    ek = embed[ctx]  # (n, N, D)
    d = eq[:, None, :] - ek
    sq = np.einsum("nij,nij->ni", d, d)
    a = _softmax_rows(log_pi[None, :] - sq / (2.0 * sigma2))
    return a, d


def _data_term(embed, ctx, q, t, emission, sigma2, log_pi):
    """Return (a, p, log p sum, gradient of sum log p wrt the embedding table)."""
    eq = embed[q]
    ek = embed[ctx]
    d = eq[:, None, :] - ek
    sq = np.einsum("nij,nij->ni", d, d)
    a = _softmax_rows(log_pi[None, :] - sq / (2.0 * sigma2))
    bvec = emission[ctx, t[:, None]]  # (n, N)
    p = np.clip(np.sum(a * bvec, axis=1), 1e-12, None)  # (n,)
    gscore = a * (bvec - p[:, None]) / p[:, None]  # d log p / d score  (n, N)
    g_eq = -np.einsum("ni,nij->nj", gscore, d) / sigma2
    g_ek = (gscore[:, :, None] * d) / sigma2
    grad = np.zeros_like(embed)
    np.add.at(grad, q, g_eq)
    np.add.at(grad, ctx.reshape(-1), g_ek.reshape(-1, embed.shape[1]))
    return a, p, float(np.sum(np.log(p))), grad


class VariationalEmbeddingAttentionDistribution(SequenceEncodableProbabilityDistribution):
    """Responsibility-attention head over tied latent embeddings (mean-field posterior)."""

    def __init__(
        self,
        mean: np.ndarray,
        log_var: np.ndarray,
        emission: np.ndarray,
        position_prior: np.ndarray,
        sigma2: float = 0.5,
        name: str | None = None,
    ) -> None:
        """Args:
        mean: ``(S, D)`` posterior means ``m_s`` of the tied embeddings.
        log_var: ``(S, D)`` posterior log-variances ``log v_s``.
        emission: ``(S, T)`` per-symbol categorical over targets.
        position_prior: ``(N,)`` prior over context positions.
        sigma2: gate variance (fixed).
        name: optional name.
        """
        self.mean = np.asarray(mean, dtype=float)
        self.log_var = np.asarray(log_var, dtype=float)
        self.emission = np.asarray(emission, dtype=float)
        self.position_prior = np.asarray(position_prior, dtype=float)
        self.num_symbols, self.embed_dim = self.mean.shape
        self.num_targets = self.emission.shape[1]
        self.context_length = self.position_prior.shape[0]
        self.sigma2 = float(sigma2)
        self.name = name
        self.log_position_prior = np.log(np.clip(self.position_prior, 1e-300, None))

    def __str__(self) -> str:
        return "VariationalEmbeddingAttentionDistribution(S=%d, N=%d, D=%d, T=%d, name=%s)" % (
            self.num_symbols,
            self.context_length,
            self.embed_dim,
            self.num_targets,
            repr(self.name),
        )

    def density(self, x: tuple[Any, int, int]) -> float:
        """Return the posterior-mean probability of one attention observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[Any, int, int]) -> float:
        """Plug-in (posterior-mean) log conditional density ``log p(target | context, query)``."""
        ctx = np.asarray(x[0], dtype=int)[None, :]
        q = np.asarray([x[1]], dtype=int)
        t = np.asarray([x[2]], dtype=int)
        return float(self.seq_log_density((ctx, q, t))[0])

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Posterior-mean log density over an encoded batch -> ``(n,)`` (uses ``e = m``)."""
        ctx, q, t = x
        a, _ = _attention(self.mean, ctx, q, self.sigma2, self.log_position_prior)
        bvec = self.emission[ctx, t[:, None]]
        return np.log(np.clip(np.sum(a * bvec, axis=1), 1e-300, None))

    def predict_proba(self, context: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Predictive target distribution from posterior-mean attention; ``(T,)`` or ``(n, T)``."""
        single = np.ndim(context) == 1
        ctx = np.atleast_2d(np.asarray(context, dtype=int))
        q = np.atleast_1d(np.asarray(query, dtype=int))
        a, _ = _attention(self.mean, ctx, q, self.sigma2, self.log_position_prior)
        pred = np.einsum("ni,nit->nt", a, self.emission[ctx])
        return pred[0] if single else pred

    def embeddings(self) -> np.ndarray:
        """The learned (posterior-mean) tied embedding table ``(S, D)``."""
        return self.mean

    def sampler(self, seed: int | None = None) -> VariationalEmbeddingAttentionSampler:
        """Return a sampler for synthetic latent-embedding attention observations."""
        return VariationalEmbeddingAttentionSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> VariationalEmbeddingAttentionEstimator:
        """Return a variational EM estimator initialized with this model's dimensions."""
        return VariationalEmbeddingAttentionEstimator(
            num_symbols=self.num_symbols,
            context_length=self.context_length,
            embed_dim=self.embed_dim,
            num_targets=self.num_targets,
            sigma2=self.sigma2,
            name=self.name,
        )

    def dist_to_encoder(self) -> VariationalEmbeddingAttentionDataEncoder:
        """Return the encoder for contexts, queries, and targets."""
        return VariationalEmbeddingAttentionDataEncoder()


class VariationalEmbeddingAttentionSampler(DistributionSampler):
    """Joint generative sampler (draws embeddings from the posterior + a uniform-distinct context)."""

    def __init__(self, dist: VariationalEmbeddingAttentionDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one observation or ``size`` iid synthetic observations."""
        n = 1 if size is None else size
        d = self.dist
        embed = d.mean + np.exp(0.5 * d.log_var) * self.rng.randn(*d.mean.shape)
        out = []
        for _ in range(n):
            ctx = self.rng.choice(d.num_symbols, size=d.context_length, replace=False)
            q = int(self.rng.choice(d.num_symbols))
            a, _ = _attention(embed, ctx[None, :], np.array([q]), d.sigma2, d.log_position_prior)
            z = self.rng.choice(d.context_length, p=a[0])
            t = int(self.rng.choice(d.num_targets, p=d.emission[ctx[z]]))
            out.append((ctx, q, t))
        return out[0] if size is None else out


class VariationalEmbeddingAttentionAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates the additive ELBO-gradient + emission/position sufficient statistics."""

    def __init__(self, num_symbols, context_length, embed_dim, num_targets, mc, seed, keys=None, name=None) -> None:
        self.num_symbols = num_symbols
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        self.mc = mc
        self._rng = RandomState(seed)
        self.grad_m = np.zeros((num_symbols, embed_dim))
        self.grad_logv = np.zeros((num_symbols, embed_dim))
        self.emission_count = np.zeros((num_symbols, num_targets))
        self.position_count = np.zeros(context_length)
        self.ll = 0.0
        self.n = 0.0
        self.keys = keys
        self.name = name

    def seq_update(self, x, weights, estimate: VariationalEmbeddingAttentionDistribution) -> None:
        """Update ELBO gradients and closed-form count statistics from encoded observations."""
        ctx, q, t = x
        w = np.asarray(weights, dtype=float)
        m, log_v, sigma2 = estimate.mean, estimate.log_var, estimate.sigma2
        log_pi = estimate.log_position_prior
        s = np.exp(0.5 * log_v)
        for _ in range(self.mc):
            eps = self._rng.randn(*m.shape)
            embed = m + s * eps
            a, _p, ll, gE = _data_term(embed, ctx, q, t, estimate.emission, sigma2, log_pi)
            self.grad_m += gE / self.mc
            self.grad_logv += (gE * eps * s * 0.5) / self.mc
            self.ll += ll / self.mc
            # emission / position counts (weighted attention responsibilities)
            aw = a * w[:, None]
            np.add.at(self.emission_count, (ctx.reshape(-1), np.repeat(t, ctx.shape[1])), (aw / self.mc).reshape(-1))
            self.position_count += aw.sum(axis=0) / self.mc
        self.n += float(w.sum())

    def update(self, x, weight: float, estimate) -> None:
        """Update from one weighted latent-embedding attention observation."""
        enc = VariationalEmbeddingAttentionDataEncoder().seq_encode([x])
        self.seq_update(enc, np.array([weight], dtype=float), estimate)

    def seq_initialize(self, x, weights, rng: RandomState) -> None:
        """Initialize emission and position counts with random attention responsibilities."""
        # no gradient yet (embeddings are initialised by the estimator); seed emission/position from
        # random attention so the first M-step is non-degenerate
        ctx, q, t = x
        w = np.asarray(weights, dtype=float)
        n, N = ctx.shape
        a = rng.dirichlet(np.ones(N), size=n) * w[:, None]
        np.add.at(self.emission_count, (ctx.reshape(-1), np.repeat(t, N)), a.reshape(-1))
        self.position_count += a.sum(axis=0)
        self.n += float(w.sum())

    def initialize(self, x, weight: float, rng: RandomState) -> None:
        """Initialize from one weighted latent-embedding attention observation."""
        enc = VariationalEmbeddingAttentionDataEncoder().seq_encode([x])
        self.seq_initialize(enc, np.array([weight], dtype=float), rng)

    def combine(self, suff_stat) -> VariationalEmbeddingAttentionAccumulator:
        """Merge ELBO gradients, emission counts, position counts, and scalar totals."""
        gm, glv, ec, pc, ll, n = suff_stat
        self.grad_m += gm
        self.grad_logv += glv
        self.emission_count += ec
        self.position_count += pc
        self.ll += ll
        self.n += n
        return self

    def value(self):
        """Return accumulated gradients, count statistics, log-likelihood, and total weight."""
        return (
            self.grad_m.copy(),
            self.grad_logv.copy(),
            self.emission_count.copy(),
            self.position_count.copy(),
            self.ll,
            self.n,
        )

    def from_value(self, x) -> VariationalEmbeddingAttentionAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.grad_m, self.grad_logv, self.emission_count, self.position_count = (
            np.asarray(v, dtype=float) for v in x[:4]
        )
        self.ll = float(x[4])
        self.n = float(x[5])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.combine(stats_dict[self.keys])
                # write the POOL back: without this the dict keeps the FIRST site's value and
                # key_replace hands that truncated pool to every tied site (later sites' data
                # silently discarded -- caught by the keyed-protocol sweep)
                stats_dict[self.keys] = self.value()
            else:
                stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> VariationalEmbeddingAttentionDataEncoder:
        """Return the encoder compatible with this accumulator."""
        return VariationalEmbeddingAttentionDataEncoder()


class VariationalEmbeddingAttentionAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for variational embedding attention EM steps."""

    def __init__(self, estimator: VariationalEmbeddingAttentionEstimator, keys=None, name=None) -> None:
        self.est = estimator
        self.keys = keys
        self.name = name

    def make(self) -> VariationalEmbeddingAttentionAccumulator:
        """Create an accumulator with a deterministic per-iteration Monte-Carlo seed."""
        e = self.est
        # vary the reparameterization noise across EM iterations (e._t advances each estimate)
        seed = (e.seed * 1_000_003 + e._t) % (2**31)
        return VariationalEmbeddingAttentionAccumulator(
            e.num_symbols, e.context_length, e.embed_dim, e.num_targets, e.mc, seed, keys=self.keys, name=self.name
        )


class VariationalEmbeddingAttentionEstimator(ParameterEstimator):
    """Variational-EM estimator; holds the embedding posterior + Adam state across EM iterations."""

    def __init__(
        self,
        num_symbols: int,
        context_length: int,
        embed_dim: int,
        num_targets: int,
        *,
        sigma2: float = 0.5,
        lr: float = 0.05,
        mc: int = 6,
        prior_strength: float = 1.0,
        emission_smoothing: float = 1e-4,
        seed: int = 0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Args:
        num_symbols, context_length, embed_dim, num_targets: dimensions ``S, N, D, T``.
        sigma2: fixed gate variance.
        lr: Adam learning rate for the embedding-posterior E-step.
        mc: Monte-Carlo samples for the reparameterized ELBO gradient.
        prior_strength: weight on the ``N(0, I)`` embedding prior (KL term).
        emission_smoothing: additive smoothing on the emission M-step.
        seed: RNG seed (embedding init + reparameterization noise).
        name / keys: standard controls.
        """
        self.num_symbols = num_symbols
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        self.sigma2 = float(sigma2)
        self.lr = float(lr)
        self.mc = int(mc)
        self.prior_strength = float(prior_strength)
        self.emission_smoothing = float(emission_smoothing)
        self.seed = int(seed)
        self.name = name
        self.keys = keys
        # mutable embedding-optimizer state (persists across EM iterations)
        self.mean: np.ndarray | None = None
        self.log_var: np.ndarray | None = None
        self._am = self._av = self._bm = self._bv = None
        self._t = 0

    def accumulator_factory(self) -> VariationalEmbeddingAttentionAccumulatorFactory:
        """Return a factory for variational embedding attention accumulators."""
        return VariationalEmbeddingAttentionAccumulatorFactory(self, keys=self.keys, name=self.name)

    def _init_state(self) -> None:
        rng = RandomState(self.seed)
        self.mean = 0.1 * rng.randn(self.num_symbols, self.embed_dim)
        self.log_var = np.full((self.num_symbols, self.embed_dim), np.log(0.3))
        self._am = np.zeros_like(self.mean)
        self._av = np.zeros_like(self.mean)
        self._bm = np.zeros_like(self.log_var)
        self._bv = np.zeros_like(self.log_var)
        self._t = 0

    def _adam(self, param, grad, m1, m2):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        mhat = m1 / (1 - b1**self._t)
        vhat = m2 / (1 - b2**self._t)
        return param + self.lr * mhat / (np.sqrt(vhat) + eps), m1, m2

    def estimate(self, nobs: float | None, suff_stat) -> VariationalEmbeddingAttentionDistribution:
        """Apply one variational EM update and return the updated attention distribution."""
        grad_m, grad_logv, emission_count, position_count, _ll, _n = suff_stat
        first = self.mean is None
        if first:
            self._init_state()
        else:
            # variational E-step: one Adam step on the ELBO (data gradient - KL gradient)
            self._t += 1
            v = np.exp(self.log_var)
            g_m = grad_m - self.prior_strength * self.mean
            g_logv = grad_logv - self.prior_strength * 0.5 * (v - 1.0)
            self.mean, self._am, self._av = self._adam(self.mean, g_m, self._am, self._av)
            self.log_var, self._bm, self._bv = self._adam(self.log_var, g_logv, self._bm, self._bv)
            self.log_var = np.clip(self.log_var, -8.0, 2.0)
        # M-step: emission + position prior in closed form
        em = emission_count + self.emission_smoothing
        emission = em / em.sum(axis=1, keepdims=True)
        total = position_count.sum()
        position_prior = position_count / total if total > 0 else np.ones(self.context_length) / self.context_length
        return VariationalEmbeddingAttentionDistribution(
            self.mean.copy(), self.log_var.copy(), emission, position_prior, sigma2=self.sigma2, name=self.name
        )


class VariationalEmbeddingAttentionDataEncoder(DataSequenceEncoder):
    """Encodes ``(context_symbols, query_symbol, target)`` triples into stacked integer arrays."""

    def __str__(self) -> str:
        return "VariationalEmbeddingAttentionDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VariationalEmbeddingAttentionDataEncoder)

    def seq_encode(self, x: Sequence[tuple[Any, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode ``(context_symbols, query, target)`` observations."""
        ctx = np.asarray([np.asarray(xi[0], dtype=int) for xi in x], dtype=int)
        q = np.asarray([int(xi[1]) for xi in x], dtype=int)
        t = np.asarray([int(xi[2]) for xi in x], dtype=int)
        return ctx, q, t


__all__ = [
    "VariationalEmbeddingAttentionDistribution",
    "VariationalEmbeddingAttentionSampler",
    "VariationalEmbeddingAttentionAccumulator",
    "VariationalEmbeddingAttentionAccumulatorFactory",
    "VariationalEmbeddingAttentionEstimator",
    "VariationalEmbeddingAttentionDataEncoder",
]
