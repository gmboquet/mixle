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
from mixle.stats.latent._attention_contracts import (
    AttentionOptimizerState,
    exact_ids,
    finite_matrix,
    merge_optimizer_state,
    observation_weights,
    positive_finite,
    positive_integer,
    row_simplex,
    safe_log_probabilities,
    weighted_log_probability_sum,
)
from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_statistic_tuple,
)
from mixle.utils.vector import ImpossibleEvidenceError

_OPTIMIZER_FAMILY = "variational_multihop_attention"


def _validated_multihop_statistics(values, num_symbols, embed_dim, num_targets):
    """Validate variational multi-hop statistics and conserved mass."""
    grad_m, grad_logv, emission_count, ll, n, state = validated_statistic_tuple(
        values, 6, "variational multi-hop attention sufficient statistics"
    )
    grad_m = finite_matrix(grad_m, "variational multi-hop mean gradients")
    grad_logv = finite_matrix(grad_logv, "variational multi-hop log-variance gradients")
    expected_gradient = (num_symbols, embed_dim)
    if grad_m.shape != expected_gradient or grad_logv.shape != expected_gradient:
        raise ValueError("variational multi-hop attention gradients have incorrect geometry")
    emission_count = validated_count_array(
        emission_count,
        (num_symbols, num_targets),
        "variational multi-hop emission counts",
    )
    n = validated_observation_weight(n, "variational multi-hop total weight")
    validate_effective_sample_mass(n, emission_count.sum(), label="variational multi-hop emission mass")
    if isinstance(ll, (bool, np.bool_)):
        raise TypeError("variational multi-hop likelihood must be a real scalar")
    try:
        ll = float(ll)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("variational multi-hop likelihood must be a real scalar") from exc
    if not np.isfinite(ll):
        raise ValueError("variational multi-hop likelihood must be finite")
    state = merge_optimizer_state(None, state, family=_OPTIMIZER_FAMILY)
    return grad_m, grad_logv, emission_count, ll, n, state


def _softmax(s: np.ndarray, axis: int) -> np.ndarray:
    s = s - s.max(axis=axis, keepdims=True)
    w = np.exp(s)
    return w / w.sum(axis=axis, keepdims=True)


def _two_hop(embed, keys, vals, q, t, emission, sigma2, weights=None):
    """Forward + per-symbol gradient + emission-aware final responsibilities for the 2-hop chain."""
    if weights is None:
        weights = np.ones(keys.shape[0], dtype=np.float64)
    eq = embed[q]  # (n, D)
    ek = embed[keys]  # (n, N, D)
    d1 = eq[:, None, :] - ek
    a1 = _softmax(-np.sum(d1 * d1, axis=2) / (2 * sigma2), axis=1)  # (n, N)
    ev = embed[vals]  # (n, N, D)
    d2 = ev[:, :, None, :] - ek[:, None, :, :]  # (n, i, j, D)
    a2 = _softmax(-np.sum(d2 * d2, axis=3) / (2 * sigma2), axis=2)  # (n, i, j)
    bj = emission[vals, t[:, None]]  # (n, N)  emission of each final value at the target
    h = np.einsum("nij,nj->ni", a2, bj)  # (n, N)
    p = np.einsum("ni,ni->n", a1, h)
    possible = p > 0.0
    gs1 = np.zeros_like(a1)
    gs2 = np.zeros_like(a2)
    gs1[possible] = weights[possible, None] * a1[possible] * (h[possible] - p[possible, None]) / p[possible, None]
    gs2[possible] = (
        weights[possible, None, None]
        * a1[possible, :, None]
        * a2[possible]
        * (bj[possible, None, :] - h[possible, :, None])
        / p[possible, None, None]
    )
    grad = np.zeros_like(embed)
    np.add.at(grad, q, -np.einsum("ni,nij->nj", gs1, d1) / sigma2)
    np.add.at(grad, keys.reshape(-1), (gs1[:, :, None] * d1 / sigma2).reshape(-1, embed.shape[1]))
    np.add.at(grad, vals.reshape(-1), (-np.einsum("nij,nijd->nid", gs2, d2) / sigma2).reshape(-1, embed.shape[1]))
    np.add.at(grad, keys.reshape(-1), (np.einsum("nij,nijd->njd", gs2, d2) / sigma2).reshape(-1, embed.shape[1]))
    rj = np.zeros_like(a1)
    rj[possible] = (a1[possible, :, None] * a2[possible] * bj[possible, None, :]).sum(axis=1) / p[possible, None]
    return p, grad, rj


class VariationalMultiHopAttentionDistribution(SequenceEncodableProbabilityDistribution):
    """A 2-hop chain over tied latent embeddings (mean-field posterior)."""

    def __init__(
        self,
        mean,
        log_var,
        emission,
        sigma2: float = 0.3,
        name: str | None = None,
        optimizer_state: AttentionOptimizerState | dict[str, Any] | None = None,
    ) -> None:
        """Args:
        mean / log_var: ``(S, D)`` posterior mean / log-variance of the tied embeddings.
        emission: ``(S, T)`` per-(value-symbol) categorical over targets.
        sigma2: gate variance (attention temperature).
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
        self.num_symbols, self.embed_dim = self.mean.shape
        if self.emission.shape[0] != self.num_symbols:
            raise ValueError("emission must have one row per symbol")
        self.num_targets = self.emission.shape[1]
        self.sigma2 = positive_finite(sigma2, "sigma2")
        self.name = name
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
        return safe_log_probabilities(p)

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
        """Sample iid observations from the posterior-mean plug-in law used by scoring."""
        return VariationalMultiHopAttentionSampler(self, seed, posterior_predictive=False)

    def posterior_predictive_sampler(self, seed: int | None = None) -> VariationalMultiHopAttentionSampler:
        """Sample iid observations, independently drawing an embedding table for every event."""
        return VariationalMultiHopAttentionSampler(self, seed, posterior_predictive=True)

    def estimator(self, pseudo_count: float | None = None) -> VariationalMultiHopAttentionEstimator:
        """Return a variational EM estimator initialized with this model's dimensions."""
        return VariationalMultiHopAttentionEstimator(
            num_symbols=self.num_symbols,
            embed_dim=self.embed_dim,
            num_targets=self.num_targets,
            sigma2=self.sigma2,
            seed=self.optimizer_state.seed,
            name=self.name,
        )

    def dist_to_encoder(self) -> VariationalMultiHopAttentionDataEncoder:
        """Return the encoder for context keys, values, query symbols, and targets."""
        return VariationalMultiHopAttentionDataEncoder(self.num_symbols, self.num_targets)


class VariationalMultiHopAttentionSampler(DistributionSampler):
    """Sampler for either the plug-in or explicitly requested posterior-predictive law."""

    def __init__(self, dist, seed: int | None = None, *, posterior_predictive: bool = False) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.posterior_predictive = bool(posterior_predictive)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one observation or ``size`` iid synthetic observations."""
        n = 1 if size is None else size
        d = self.dist
        N = 6
        out = []
        for _ in range(n):
            embed = d.mean
            if self.posterior_predictive:
                embed = d.mean + np.exp(0.5 * d.log_var) * self.rng.randn(*d.mean.shape)
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
        self.seed = int(seed)
        self.grad_m = np.zeros((num_symbols, embed_dim))
        self.grad_logv = np.zeros((num_symbols, embed_dim))
        self.emission_count = np.zeros((num_symbols, num_targets))
        self.ll = 0.0
        self.n = 0.0
        self.optimizer_state: AttentionOptimizerState | None = None
        self.keys = keys
        self.name = name

    def seq_update(self, x, weights, estimate) -> None:
        """Update ELBO gradients and emission counts from encoded observations."""
        keys, vals, q, t = x
        w = observation_weights(weights, keys.shape[0])
        m, log_v, sig = estimate.mean, estimate.log_var, estimate.sigma2
        supported = np.any(estimate.emission[vals, t[:, None]] > 0.0, axis=1)
        impossible = np.flatnonzero(~supported & (w > 0.0))
        if impossible.size:
            raise ImpossibleEvidenceError(
                f"variational multi-hop attention encountered impossible evidence at rows {impossible.tolist()}"
            )
        self.optimizer_state = merge_optimizer_state(
            self.optimizer_state, estimate.optimizer_state, family=_OPTIMIZER_FAMILY
        )
        rng = RandomState((estimate.optimizer_state.seed * 1_000_003 + estimate.optimizer_state.iteration) % (2**31))
        s = np.exp(0.5 * log_v)
        for _ in range(self.mc):
            eps = rng.randn(*m.shape)
            embed = m + s * eps
            p, gE, rj = _two_hop(embed, keys, vals, q, t, estimate.emission, sig, w)
            self.grad_m += gE / self.mc
            self.grad_logv += (gE * eps * s * 0.5) / self.mc
            self.ll += weighted_log_probability_sum(p, w) / self.mc
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
        w = observation_weights(weights, n)
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
        gm, glv, ec, ll, n, state = _validated_multihop_statistics(
            suff_stat,
            self.num_symbols,
            self.embed_dim,
            self.num_targets,
        )
        self.grad_m += gm
        self.grad_logv += glv
        self.emission_count += ec
        self.ll += ll
        self.n += n
        self.optimizer_state = merge_optimizer_state(self.optimizer_state, state, family=_OPTIMIZER_FAMILY)
        return self

    def value(self):
        """Return accumulated gradients, emission counts, log-likelihood, and total weight."""
        return (
            self.grad_m.copy(),
            self.grad_logv.copy(),
            self.emission_count.copy(),
            self.ll,
            self.n,
            None if self.optimizer_state is None else self.optimizer_state.to_dict(),
        )

    def from_value(self, x):
        """Restore accumulator state from ``value`` output."""
        (
            self.grad_m,
            self.grad_logv,
            self.emission_count,
            self.ll,
            self.n,
            self.optimizer_state,
        ) = _validated_multihop_statistics(
            x,
            self.num_symbols,
            self.embed_dim,
            self.num_targets,
        )
        return self

    def acc_to_encoder(self):
        """Return the encoder compatible with this attention accumulator."""
        return VariationalMultiHopAttentionDataEncoder(self.num_symbols, self.num_targets)


class VariationalMultiHopAttentionAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for variational multi-hop attention EM steps."""

    def __init__(self, estimator, keys=None, name=None) -> None:
        self.est = estimator
        self.keys = keys
        self.name = name

    def make(self):
        """Create an accumulator with a deterministic per-iteration Monte-Carlo seed."""
        e = self.est
        return VariationalMultiHopAttentionAccumulator(
            e.num_symbols,
            e.embed_dim,
            e.num_targets,
            e.mc,
            e.seed,
            keys=self.keys,
            name=self.name,
        )


class VariationalMultiHopAttentionEstimator(ParameterEstimator):
    """Stateless variational-EM estimator with model-carried Adam state and prior annealing."""

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
        self.num_symbols = positive_integer(num_symbols, "num_symbols")
        self.embed_dim = positive_integer(embed_dim, "embed_dim")
        self.num_targets = positive_integer(num_targets, "num_targets")
        self.sigma2 = positive_finite(sigma2, "sigma2")
        self.lr = positive_finite(lr, "lr")
        self.mc = positive_integer(mc, "mc")
        self.prior_strength = float(prior_strength)
        self.anneal_iters = positive_integer(anneal_iters, "anneal_iters")
        self.emission_smoothing = float(emission_smoothing)
        if not np.isfinite(self.prior_strength) or self.prior_strength < 0.0:
            raise ValueError("prior_strength must be finite and non-negative")
        if not np.isfinite(self.emission_smoothing) or self.emission_smoothing < 0.0:
            raise ValueError("emission_smoothing must be finite and non-negative")
        self.seed = int(seed)
        self.name = name
        self.keys = keys

    def accumulator_factory(self):
        """Return a factory for variational multi-hop attention accumulators."""
        return VariationalMultiHopAttentionAccumulatorFactory(self, keys=self.keys, name=self.name)

    def _adam(self, param, grad, m1, m2, iteration):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        return (
            param + self.lr * (m1 / (1 - b1**iteration)) / (np.sqrt(m2 / (1 - b2**iteration)) + eps),
            m1,
            m2,
        )

    def _init_state(self) -> AttentionOptimizerState:
        rng = RandomState(self.seed)
        mean = rng.randn(self.num_symbols, self.embed_dim)
        log_var = np.full((self.num_symbols, self.embed_dim), np.log(0.3))
        return AttentionOptimizerState.fresh(_OPTIMIZER_FAMILY, mean, log_var, self.seed)

    def estimate(self, nobs, suff_stat):
        """Apply one variational EM update and return the updated attention distribution."""
        grad_m, grad_logv, emission_count, _ll, observed_mass, state_data = _validated_multihop_statistics(
            suff_stat,
            self.num_symbols,
            self.embed_dim,
            self.num_targets,
        )
        validate_effective_sample_mass(
            nobs,
            observed_mass,
            label="variational multi-hop attention effective sample",
        )
        state = self._init_state() if state_data is None else state_data
        if state_data is not None:
            iteration = state.iteration + 1
            ps = self.prior_strength * min(1.0, iteration / self.anneal_iters)
            variance = np.exp(state.log_var)
            g_m = grad_m - ps * state.mean
            g_logv = grad_logv - ps * 0.5 * (variance - 1.0)
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
        em = emission_count + self.emission_smoothing
        row_totals = em.sum(axis=1, keepdims=True)
        emission = np.divide(
            em,
            row_totals,
            out=np.full_like(em, 1.0 / self.num_targets),
            where=row_totals > 0.0,
        )
        return VariationalMultiHopAttentionDistribution(
            state.mean,
            state.log_var,
            emission,
            sigma2=self.sigma2,
            name=self.name,
            optimizer_state=state,
        )


class VariationalMultiHopAttentionDataEncoder(DataSequenceEncoder):
    """Encode context keys, context values, query symbols, and targets as integer arrays."""

    def __init__(self, num_symbols=None, num_targets=None) -> None:
        self.num_symbols = num_symbols
        self.num_targets = num_targets

    def __str__(self) -> str:
        return "VariationalMultiHopAttentionDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VariationalMultiHopAttentionDataEncoder) and (
            self.num_symbols,
            self.num_targets,
        ) == (other.num_symbols, other.num_targets)

    def seq_encode(self, x: Sequence[tuple[Any, Any, int, int]]):
        """Encode ``(context_keys, context_values, query, target)`` observations."""
        keys = exact_ids(
            [xi[0] for xi in x],
            "variational multi-hop keys",
            upper=self.num_symbols,
            ndim=2,
        )
        vals = exact_ids(
            [xi[1] for xi in x],
            "variational multi-hop values",
            upper=self.num_symbols,
            ndim=2,
        )
        if keys.shape != vals.shape or keys.shape[1] == 0:
            raise ValueError("variational multi-hop contexts must share one positive width")
        q = exact_ids(
            [xi[2] for xi in x],
            "variational multi-hop query IDs",
            upper=self.num_symbols,
            ndim=1,
        )
        t = exact_ids(
            [xi[3] for xi in x],
            "variational multi-hop target IDs",
            upper=self.num_targets,
            ndim=1,
        )
        return keys, vals, q, t

    def row_count(self, x) -> int:
        return int(np.asarray(x[0]).shape[0])


__all__ = [
    "VariationalMultiHopAttentionDistribution",
    "VariationalMultiHopAttentionSampler",
    "VariationalMultiHopAttentionAccumulator",
    "VariationalMultiHopAttentionAccumulatorFactory",
    "VariationalMultiHopAttentionEstimator",
    "VariationalMultiHopAttentionDataEncoder",
]
