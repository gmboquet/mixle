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
    statistic; the Adam optimizer state is carried explicitly by the model and sufficient-statistic
    artifact, so distributed and resumed updates have the same semantics;
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
from mixle.stats.latent._attention_contracts import (
    AttentionOptimizerState,
    exact_ids,
    finite_matrix,
    merge_optimizer_state,
    normalize_log_rows,
    observation_weights,
    positive_finite,
    positive_integer,
    row_simplex,
    safe_log_probabilities,
    simplex,
    weighted_log_probability_sum,
)
from mixle.utils.vector import ImpossibleEvidenceError

_OPTIMIZER_FAMILY = "variational_embedding_attention"


def _softmax_rows(s: np.ndarray) -> np.ndarray:
    return normalize_log_rows(s)[1]


def _attention(embed: np.ndarray, ctx: np.ndarray, q: np.ndarray, sigma2: float, log_pi: np.ndarray):
    """Attention weights ``a`` (n,N) and the (query-key) difference tensor ``d`` (n,N,D)."""
    eq = embed[q]  # (n, D)
    ek = embed[ctx]  # (n, N, D)
    d = eq[:, None, :] - ek
    sq = np.einsum("nij,nij->ni", d, d)
    a = _softmax_rows(log_pi[None, :] - sq / (2.0 * sigma2))
    return a, d


def _data_term(embed, ctx, q, t, emission, sigma2, log_pi, weights):
    """Return weighted likelihood terms and their embedding-table gradient."""
    eq = embed[q]
    ek = embed[ctx]
    d = eq[:, None, :] - ek
    sq = np.einsum("nij,nij->ni", d, d)
    a = _softmax_rows(log_pi[None, :] - sq / (2.0 * sigma2))
    bvec = emission[ctx, t[:, None]]  # (n, N)
    p = np.sum(a * bvec, axis=1)  # (n,)
    possible = p > 0.0
    gscore = np.zeros_like(a)
    gscore[possible] = weights[possible, None] * a[possible] * (bvec[possible] - p[possible, None]) / p[possible, None]
    g_eq = -np.einsum("ni,nij->nj", gscore, d) / sigma2
    g_ek = (gscore[:, :, None] * d) / sigma2
    grad = np.zeros_like(embed)
    np.add.at(grad, q, g_eq)
    np.add.at(grad, ctx.reshape(-1), g_ek.reshape(-1, embed.shape[1]))
    return a, p, weighted_log_probability_sum(p, weights), grad


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
        optimizer_state: AttentionOptimizerState | dict[str, Any] | None = None,
    ) -> None:
        """Args:
        mean: ``(S, D)`` posterior means ``m_s`` of the tied embeddings.
        log_var: ``(S, D)`` posterior log-variances ``log v_s``.
        emission: ``(S, T)`` per-symbol categorical over targets.
        position_prior: ``(N,)`` prior over context positions.
        sigma2: gate variance (fixed).
        name: optional name.
        """
        self.mean = finite_matrix(mean, "mean")
        self.log_var = finite_matrix(log_var, "log_var")
        if self.log_var.shape != self.mean.shape:
            raise ValueError("mean and log_var must have the same shape")
        with np.errstate(over="ignore", invalid="ignore"):
            variance = np.exp(self.log_var)
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
            raise ValueError("log_var must encode positive finite variances")
        self.emission = row_simplex(emission, "emission")
        self.position_prior = simplex(position_prior, "position_prior")
        self.num_symbols, self.embed_dim = self.mean.shape
        if self.emission.shape[0] != self.num_symbols:
            raise ValueError("emission must have one row per symbol")
        self.num_targets = self.emission.shape[1]
        self.context_length = self.position_prior.shape[0]
        self.sigma2 = positive_finite(sigma2, "sigma2")
        self.name = name
        self.log_position_prior = safe_log_probabilities(self.position_prior)
        if optimizer_state is None:
            state = AttentionOptimizerState.fresh(_OPTIMIZER_FAMILY, self.mean, self.log_var, 0)
        else:
            state = (
                AttentionOptimizerState.from_dict(optimizer_state)
                if isinstance(optimizer_state, dict)
                else optimizer_state
            )
            if state.family != _OPTIMIZER_FAMILY:
                raise ValueError("optimizer_state belongs to a different attention family")
            if state.mean.shape != self.mean.shape:
                raise ValueError("optimizer_state geometry does not match the distribution")
            if not np.array_equal(state.mean, self.mean) or not np.array_equal(state.log_var, self.log_var):
                raise ValueError("optimizer_state posterior does not match mean and log_var")
        self.optimizer_state = state

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
        return safe_log_probabilities(np.sum(a * bvec, axis=1))

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
        """Sample iid observations from the posterior-mean plug-in law used by scoring."""
        return VariationalEmbeddingAttentionSampler(self, seed, posterior_predictive=False)

    def posterior_predictive_sampler(self, seed: int | None = None) -> VariationalEmbeddingAttentionSampler:
        """Sample iid observations, independently drawing an embedding table for every event."""
        return VariationalEmbeddingAttentionSampler(self, seed, posterior_predictive=True)

    def estimator(self, pseudo_count: float | None = None) -> VariationalEmbeddingAttentionEstimator:
        """Return a variational EM estimator initialized with this model's dimensions."""
        return VariationalEmbeddingAttentionEstimator(
            num_symbols=self.num_symbols,
            context_length=self.context_length,
            embed_dim=self.embed_dim,
            num_targets=self.num_targets,
            sigma2=self.sigma2,
            seed=self.optimizer_state.seed,
            name=self.name,
        )

    def dist_to_encoder(self) -> VariationalEmbeddingAttentionDataEncoder:
        """Return the encoder for contexts, queries, and targets."""
        return VariationalEmbeddingAttentionDataEncoder(self.num_symbols, self.context_length, self.num_targets)


class VariationalEmbeddingAttentionSampler(DistributionSampler):
    """Sampler for either the plug-in or explicitly requested posterior-predictive law."""

    def __init__(
        self,
        dist: VariationalEmbeddingAttentionDistribution,
        seed: int | None = None,
        *,
        posterior_predictive: bool = False,
    ) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.posterior_predictive = bool(posterior_predictive)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one observation or ``size`` iid synthetic observations."""
        n = 1 if size is None else size
        d = self.dist
        out = []
        for _ in range(n):
            embed = d.mean
            if self.posterior_predictive:
                embed = d.mean + np.exp(0.5 * d.log_var) * self.rng.randn(*d.mean.shape)
            ctx = self.rng.choice(d.num_symbols, size=d.context_length, replace=d.context_length > d.num_symbols)
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
        self.seed = int(seed)
        self.grad_m = np.zeros((num_symbols, embed_dim))
        self.grad_logv = np.zeros((num_symbols, embed_dim))
        self.emission_count = np.zeros((num_symbols, num_targets))
        self.position_count = np.zeros(context_length)
        self.ll = 0.0
        self.n = 0.0
        self.optimizer_state: AttentionOptimizerState | None = None
        self.keys = keys
        self.name = name

    def seq_update(self, x, weights, estimate: VariationalEmbeddingAttentionDistribution) -> None:
        """Update ELBO gradients and closed-form count statistics from encoded observations."""
        ctx, q, t = x
        w = observation_weights(weights, ctx.shape[0])
        m, log_v, sigma2 = estimate.mean, estimate.log_var, estimate.sigma2
        log_pi = estimate.log_position_prior
        supported = np.any(
            (estimate.position_prior[None, :] > 0.0) & (estimate.emission[ctx, t[:, None]] > 0.0),
            axis=1,
        )
        impossible = np.flatnonzero(~supported & (w > 0.0))
        if impossible.size:
            raise ImpossibleEvidenceError(
                f"variational embedding attention encountered impossible evidence at rows {impossible.tolist()}"
            )
        self.optimizer_state = merge_optimizer_state(
            self.optimizer_state, estimate.optimizer_state, family=_OPTIMIZER_FAMILY
        )
        iteration = estimate.optimizer_state.iteration
        rng = RandomState(
            (estimate.optimizer_state.seed * 1_000_003 + iteration) % (2**31)
        )
        s = np.exp(0.5 * log_v)
        for _ in range(self.mc):
            eps = rng.randn(*m.shape)
            embed = m + s * eps
            a, _p, ll, gE = _data_term(embed, ctx, q, t, estimate.emission, sigma2, log_pi, w)
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
        w = observation_weights(weights, ctx.shape[0])
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
        gm, glv, ec, pc, ll, n, state = suff_stat
        self.grad_m += gm
        self.grad_logv += glv
        self.emission_count += ec
        self.position_count += pc
        self.ll += ll
        self.n += n
        self.optimizer_state = merge_optimizer_state(self.optimizer_state, state, family=_OPTIMIZER_FAMILY)
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
            None if self.optimizer_state is None else self.optimizer_state.to_dict(),
        )

    def from_value(self, x) -> VariationalEmbeddingAttentionAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.grad_m, self.grad_logv, self.emission_count, self.position_count = (
            np.asarray(v, dtype=float) for v in x[:4]
        )
        self.ll = float(x[4])
        self.n = float(x[5])
        self.optimizer_state = merge_optimizer_state(None, x[6], family=_OPTIMIZER_FAMILY)
        return self

    def acc_to_encoder(self) -> VariationalEmbeddingAttentionDataEncoder:
        """Return the encoder compatible with this accumulator."""
        return VariationalEmbeddingAttentionDataEncoder(self.num_symbols, self.context_length, self.num_targets)


class VariationalEmbeddingAttentionAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for variational embedding attention EM steps."""

    def __init__(self, estimator: VariationalEmbeddingAttentionEstimator, keys=None, name=None) -> None:
        self.est = estimator
        self.keys = keys
        self.name = name

    def make(self) -> VariationalEmbeddingAttentionAccumulator:
        """Create an accumulator with a deterministic per-iteration Monte-Carlo seed."""
        e = self.est
        return VariationalEmbeddingAttentionAccumulator(
            e.num_symbols,
            e.context_length,
            e.embed_dim,
            e.num_targets,
            e.mc,
            e.seed,
            keys=self.keys,
            name=self.name,
        )


class VariationalEmbeddingAttentionEstimator(ParameterEstimator):
    """Stateless variational-EM estimator consuming explicit model-carried Adam state."""

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
        self.num_symbols = positive_integer(num_symbols, "num_symbols")
        self.context_length = positive_integer(context_length, "context_length")
        self.embed_dim = positive_integer(embed_dim, "embed_dim")
        self.num_targets = positive_integer(num_targets, "num_targets")
        self.sigma2 = positive_finite(sigma2, "sigma2")
        self.lr = positive_finite(lr, "lr")
        self.mc = positive_integer(mc, "mc")
        self.prior_strength = float(prior_strength)
        self.emission_smoothing = float(emission_smoothing)
        if not np.isfinite(self.prior_strength) or self.prior_strength < 0.0:
            raise ValueError("prior_strength must be finite and non-negative")
        if not np.isfinite(self.emission_smoothing) or self.emission_smoothing < 0.0:
            raise ValueError("emission_smoothing must be finite and non-negative")
        self.seed = int(seed)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> VariationalEmbeddingAttentionAccumulatorFactory:
        """Return a factory for variational embedding attention accumulators."""
        return VariationalEmbeddingAttentionAccumulatorFactory(self, keys=self.keys, name=self.name)

    def _init_state(self) -> AttentionOptimizerState:
        rng = RandomState(self.seed)
        mean = 0.1 * rng.randn(self.num_symbols, self.embed_dim)
        log_var = np.full((self.num_symbols, self.embed_dim), np.log(0.3))
        return AttentionOptimizerState.fresh(_OPTIMIZER_FAMILY, mean, log_var, self.seed)

    def _adam(self, param, grad, m1, m2, iteration):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        mhat = m1 / (1 - b1**iteration)
        vhat = m2 / (1 - b2**iteration)
        return param + self.lr * mhat / (np.sqrt(vhat) + eps), m1, m2

    def estimate(self, nobs: float | None, suff_stat) -> VariationalEmbeddingAttentionDistribution:
        """Apply one variational EM update and return the updated attention distribution."""
        grad_m, grad_logv, emission_count, position_count, _ll, _n = suff_stat[:6]
        state_data = suff_stat[6] if len(suff_stat) > 6 else None
        state = (
            self._init_state()
            if state_data is None
            else merge_optimizer_state(None, state_data, family=_OPTIMIZER_FAMILY)
        )
        if state_data is not None:
            iteration = state.iteration + 1
            variance = np.exp(state.log_var)
            g_m = grad_m - self.prior_strength * state.mean
            g_logv = grad_logv - self.prior_strength * 0.5 * (variance - 1.0)
            mean, am, av = self._adam(
                state.mean,
                g_m,
                state.mean_first_moment,
                state.mean_second_moment,
                iteration,
            )
            log_var, bm, bv = self._adam(
                state.log_var,
                g_logv,
                state.log_var_first_moment,
                state.log_var_second_moment,
                iteration,
            )
            log_var = np.clip(log_var, -8.0, 2.0)
            state = AttentionOptimizerState(_OPTIMIZER_FAMILY, mean, log_var, am, av, bm, bv, iteration, state.seed)
        # M-step: emission + position prior in closed form
        em = emission_count + self.emission_smoothing
        row_totals = em.sum(axis=1, keepdims=True)
        emission = np.divide(
            em,
            row_totals,
            out=np.full_like(em, 1.0 / self.num_targets),
            where=row_totals > 0.0,
        )
        total = position_count.sum()
        position_prior = position_count / total if total > 0 else np.ones(self.context_length) / self.context_length
        return VariationalEmbeddingAttentionDistribution(
            state.mean,
            state.log_var,
            emission,
            position_prior,
            sigma2=self.sigma2,
            name=self.name,
            optimizer_state=state,
        )


class VariationalEmbeddingAttentionDataEncoder(DataSequenceEncoder):
    """Encodes ``(context_symbols, query_symbol, target)`` triples into stacked integer arrays."""

    def __init__(self, num_symbols=None, context_length=None, num_targets=None) -> None:
        self.num_symbols = num_symbols
        self.context_length = context_length
        self.num_targets = num_targets

    def __str__(self) -> str:
        return "VariationalEmbeddingAttentionDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VariationalEmbeddingAttentionDataEncoder) and (
            self.num_symbols,
            self.context_length,
            self.num_targets,
        ) == (other.num_symbols, other.context_length, other.num_targets)

    def seq_encode(self, x: Sequence[tuple[Any, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode ``(context_symbols, query, target)`` observations."""
        ctx = exact_ids(
            [xi[0] for xi in x],
            "variational-attention context IDs",
            upper=self.num_symbols,
            ndim=2,
        )
        if ctx.shape[1] == 0 or (self.context_length is not None and ctx.shape[1] != self.context_length):
            raise ValueError("variational-attention contexts must match the declared positive width")
        q = exact_ids(
            [xi[1] for xi in x],
            "variational-attention query IDs",
            upper=self.num_symbols,
            ndim=1,
        )
        t = exact_ids(
            [xi[2] for xi in x],
            "variational-attention target IDs",
            upper=self.num_targets,
            ndim=1,
        )
        return ctx, q, t

    def row_count(self, x) -> int:
        return int(np.asarray(x[0]).shape[0])


__all__ = [
    "VariationalEmbeddingAttentionDistribution",
    "VariationalEmbeddingAttentionSampler",
    "VariationalEmbeddingAttentionAccumulator",
    "VariationalEmbeddingAttentionAccumulatorFactory",
    "VariationalEmbeddingAttentionEstimator",
    "VariationalEmbeddingAttentionDataEncoder",
]
