"""Responsibility attention: one attention head as an EM-able mixture over context positions.

A single attention head, written as a generative latent-variable model. The latent ``z`` is *which
context position you attend to*; the "gate" is generative (the query is generated from the attended
token's key) so the attention weights are posterior responsibilities rather than a discriminative
dot-product softmax. That one change is what makes the head fully EM-able: every M-step is a
closed-form, additively-mergeable sufficient-statistic update (a GMM mean for the keys, a weighted
categorical for the emission, counts for the position prior), so the head trains by expectation-
maximization and composes with the rest of ``pysp`` (drop it into a mixture, an HMM emission, a
composite) instead of needing gradient descent.

Generative story for one observation ``(context, query, target)`` with context tokens ``c_1..c_N``:

    z ~ Categorical(position_prior)
    query  ~ Normal(key_means[c_z], sigma2 I)        # query generated from the attended token's key
    target ~ Categorical(emission[c_z, :])           # target emitted from the attended token

so  ``p(query, target | context) = sum_i position_prior_i N(query; key_means[c_i]) emission[c_i, target]``
and the attention weight on position ``i`` is the posterior ``p(z = i | query, target, context)``.

The observation is the triple ``(context, query, target)``; the context is conditioned on (it is the
covariate that defines the positions), so the density is the conditional ``p(query, target | context)``
that EM maximizes. This is the single-hop leaf; deeper / stackable multi-hop versions compose several
of these latents (a chain is forward--backward, an HME-style tree is nested responsibilities).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _log_weights(
    enc: tuple[np.ndarray, np.ndarray, np.ndarray], dist: ResponsibilityAttentionDistribution
) -> np.ndarray:
    """Per-(observation, position) log joint weight ``log[ pi_i N(y;K[c_i]) B[c_i,t] ]`` -> ``(n, N)``."""
    ctx, y, t = enc
    keys = dist.key_means[ctx]  # (n, N, D)
    diff = y[:, None, :] - keys
    sq = np.einsum("nij,nij->ni", diff, diff)  # (n, N)
    gate = -0.5 * sq / dist.sigma2 + dist.gate_const
    log_emit = np.log(dist.emission[ctx, t[:, None]])  # (n, N)
    return dist.log_position_prior[None, :] + gate + log_emit


def _normalize_rows(log_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (row log-sum-exp ``(n,)``, responsibilities ``(n, N)``)."""
    m = log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w - m)
    z = w.sum(axis=1, keepdims=True)
    return (np.log(z[:, 0]) + m[:, 0]), w / z


class ResponsibilityAttentionDistribution(SequenceEncodableProbabilityDistribution):
    """One generative-gate attention head; a mixture over context positions (EM-able)."""

    def __init__(
        self,
        key_means: np.ndarray,
        emission: np.ndarray,
        position_prior: np.ndarray | None = None,
        sigma2: float = 1.0,
        name: str | None = None,
    ) -> None:
        """Args:
        key_means: ``(S, D)`` per-symbol gate means (the "keys").
        emission: ``(S, T)`` per-symbol categorical over targets (rows sum to 1).
        position_prior: ``(N,)`` prior over context positions (defaults to uniform; ``N`` is the
            context length the head expects).
        sigma2: gate variance (fixed; the responsibility temperature).
        name: optional name.
        """
        self.key_means = np.asarray(key_means, dtype=float)
        self.emission = np.asarray(emission, dtype=float)
        self.num_symbols, self.query_dim = self.key_means.shape
        self.num_targets = self.emission.shape[1]
        if position_prior is None:
            raise ValueError("position_prior (length = context length N) is required.")
        self.position_prior = np.asarray(position_prior, dtype=float)
        self.context_length = self.position_prior.shape[0]
        self.sigma2 = float(sigma2)
        self.name = name
        self.log_position_prior = np.log(np.clip(self.position_prior, 1e-300, None))
        self.gate_const = -0.5 * self.query_dim * np.log(2.0 * np.pi * self.sigma2)

    def __str__(self) -> str:
        return "ResponsibilityAttentionDistribution(S=%d, N=%d, D=%d, T=%d, sigma2=%s, name=%s)" % (
            self.num_symbols,
            self.context_length,
            self.query_dim,
            self.num_targets,
            repr(self.sigma2),
            repr(self.name),
        )

    def density(self, x: tuple[Any, Any, int]) -> float:
        """Density ``p(query, target | context)`` at one observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[Any, Any, int]) -> float:
        """Log conditional density ``log p(query, target | context)`` at one observation ``(ctx, y, t)``."""
        ctx = np.asarray(x[0], dtype=int)[None, :]
        y = np.asarray(x[1], dtype=float)[None, :]
        t = np.asarray([x[2]], dtype=int)
        ll, _ = _normalize_rows(_log_weights((ctx, y, t), self))
        return float(ll[0])

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized ``log p(query, target | context)`` over an encoded batch -> ``(n,)``."""
        ll, _ = _normalize_rows(_log_weights(x, self))
        return ll

    def predict_proba(self, context: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Predictive target distribution ``p(target | context, query)`` (target marginalized over z).

        Attention here does *not* see the target: ``a_i ∝ pi_i N(query; K[c_i])``, then
        ``p(target) = sum_i a_i emission[c_i, :]``. Accepts a single example or a batch.

        Returns:
            ``(T,)`` for a single example, or ``(n, T)`` for a batch.
        """
        ctx = np.atleast_2d(np.asarray(context, dtype=int))
        y = np.atleast_2d(np.asarray(query, dtype=float))
        keys = self.key_means[ctx]
        diff = y[:, None, :] - keys
        sq = np.einsum("nij,nij->ni", diff, diff)
        log_a = self.log_position_prior[None, :] + (-0.5 * sq / self.sigma2)
        log_a -= log_a.max(axis=1, keepdims=True)
        a = np.exp(log_a)
        a /= a.sum(axis=1, keepdims=True)
        pred = np.einsum("ni,nit->nt", a, self.emission[ctx])
        return pred[0] if np.ndim(context) == 1 else pred

    def sampler(self, seed: int | None = None) -> ResponsibilityAttentionSampler:
        return ResponsibilityAttentionSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ResponsibilityAttentionEstimator:
        return ResponsibilityAttentionEstimator(
            num_symbols=self.num_symbols,
            context_length=self.context_length,
            query_dim=self.query_dim,
            num_targets=self.num_targets,
            sigma2=self.sigma2,
            pseudo_count=pseudo_count,
            name=self.name,
        )

    def dist_to_encoder(self) -> ResponsibilityAttentionDataEncoder:
        return ResponsibilityAttentionDataEncoder()


class ResponsibilityAttentionSampler(DistributionSampler):
    """Joint generative sampler (adds a uniform-distinct context prior so it is a full generator)."""

    def __init__(self, dist: ResponsibilityAttentionDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        n = 1 if size is None else size
        d = self.dist
        out = []
        for _ in range(n):
            ctx = self.rng.choice(d.num_symbols, size=d.context_length, replace=False)
            z = self.rng.choice(d.context_length, p=d.position_prior / d.position_prior.sum())
            sym = ctx[z]
            y = d.key_means[sym] + np.sqrt(d.sigma2) * self.rng.randn(d.query_dim)
            t = self.rng.choice(d.num_targets, p=d.emission[sym])
            out.append((ctx, y, int(t)))
        return out[0] if size is None else out


class ResponsibilityAttentionAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates the additive sufficient statistics for the responsibility-attention M-step."""

    def __init__(
        self,
        num_symbols: int,
        context_length: int,
        query_dim: int,
        num_targets: int,
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        self.num_symbols = num_symbols
        self.context_length = context_length
        self.query_dim = query_dim
        self.num_targets = num_targets
        self.key_sum = np.zeros((num_symbols, query_dim))  # Σ r·query per symbol  (GMM mean numerator)
        self.key_mass = np.zeros(num_symbols)  # Σ r per symbol        (GMM mean denominator)
        self.emission_count = np.zeros((num_symbols, num_targets))  # Σ r·1[target] per symbol
        self.position_count = np.zeros(context_length)  # Σ r per position
        self.keys = keys
        self.name = name
        self._rng: RandomState | None = None

    def _accumulate(self, enc, r: np.ndarray) -> None:
        ctx, y, t = enc
        n, N = ctx.shape
        flat_sym = ctx.reshape(-1)
        flat_r = r.reshape(-1)
        np.add.at(self.key_mass, flat_sym, flat_r)
        np.add.at(self.key_sum, flat_sym, (r[:, :, None] * y[:, None, :]).reshape(-1, self.query_dim))
        np.add.at(self.emission_count, (flat_sym, np.repeat(t, N)), flat_r)
        self.position_count += r.sum(axis=0)

    def update(self, x, weight: float, estimate: ResponsibilityAttentionDistribution | None) -> None:
        enc = ResponsibilityAttentionDataEncoder().seq_encode([x])
        self.seq_update(enc, np.array([weight], dtype=float), estimate)

    def seq_update(self, x, weights: np.ndarray, estimate: ResponsibilityAttentionDistribution | None) -> None:
        _, r = _normalize_rows(_log_weights(x, estimate))
        self._accumulate(x, r * weights[:, None])

    def initialize(self, x, weight: float, rng: RandomState) -> None:
        enc = ResponsibilityAttentionDataEncoder().seq_encode([x])
        self.seq_initialize(enc, np.array([weight], dtype=float), rng)

    def seq_initialize(self, x, weights: np.ndarray, rng: RandomState) -> None:
        # random responsibilities break the symmetry the M-step would otherwise preserve
        ctx = x[0]
        n, N = ctx.shape
        r = rng.dirichlet(np.ones(N), size=n)
        self._accumulate(x, r * weights[:, None])

    def combine(self, suff_stat) -> ResponsibilityAttentionAccumulator:
        ks, km, ec, pc = suff_stat
        self.key_sum += ks
        self.key_mass += km
        self.emission_count += ec
        self.position_count += pc
        return self

    def value(self):
        # copies: value() is a snapshot, so combine()/key_merge()/distributed reduction can never
        # alias an accumulator's own arrays
        return (self.key_sum.copy(), self.key_mass.copy(), self.emission_count.copy(), self.position_count.copy())

    def from_value(self, x) -> ResponsibilityAttentionAccumulator:
        self.key_sum, self.key_mass, self.emission_count, self.position_count = (np.asarray(v, dtype=float) for v in x)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.combine(stats_dict[self.keys])
            else:
                stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> ResponsibilityAttentionDataEncoder:
        return ResponsibilityAttentionDataEncoder()


class ResponsibilityAttentionAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(
        self,
        num_symbols: int,
        context_length: int,
        query_dim: int,
        num_targets: int,
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        self.num_symbols = num_symbols
        self.context_length = context_length
        self.query_dim = query_dim
        self.num_targets = num_targets
        self.keys = keys
        self.name = name

    def make(self) -> ResponsibilityAttentionAccumulator:
        return ResponsibilityAttentionAccumulator(
            self.num_symbols, self.context_length, self.query_dim, self.num_targets, keys=self.keys, name=self.name
        )


class ResponsibilityAttentionEstimator(ParameterEstimator):
    """Estimate a :class:`ResponsibilityAttentionDistribution` by closed-form EM M-steps."""

    def __init__(
        self,
        num_symbols: int,
        context_length: int,
        query_dim: int,
        num_targets: int,
        *,
        sigma2: float = 1.0,
        emission_smoothing: float = 1e-6,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Args:
        num_symbols, context_length, query_dim, num_targets: model dimensions ``S, N, D, T``.
        sigma2: fixed gate variance carried into the fitted distribution.
        emission_smoothing: additive smoothing on the emission categorical M-step.
        pseudo_count / name / keys: standard estimator controls.
        """
        self.num_symbols = num_symbols
        self.context_length = context_length
        self.query_dim = query_dim
        self.num_targets = num_targets
        self.sigma2 = float(sigma2)
        self.emission_smoothing = float(emission_smoothing)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ResponsibilityAttentionAccumulatorFactory:
        return ResponsibilityAttentionAccumulatorFactory(
            self.num_symbols, self.context_length, self.query_dim, self.num_targets, keys=self.keys, name=self.name
        )

    def estimate(self, nobs: float | None, suff_stat) -> ResponsibilityAttentionDistribution:
        key_sum, key_mass, emission_count, position_count = suff_stat
        # keys: responsibility-weighted mean query per symbol (GMM mean update)
        denom = np.clip(key_mass, 1e-12, None)
        key_means = key_sum / denom[:, None]
        # emission: smoothed weighted categorical per symbol
        em = emission_count + self.emission_smoothing
        emission = em / em.sum(axis=1, keepdims=True)
        # position prior: normalized responsibility counts
        total = position_count.sum()
        position_prior = position_count / total if total > 0 else np.ones(self.context_length) / self.context_length
        return ResponsibilityAttentionDistribution(
            key_means, emission, position_prior=position_prior, sigma2=self.sigma2, name=self.name
        )


class ResponsibilityAttentionDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of ``(context, query, target)`` triples into stacked arrays."""

    def __str__(self) -> str:
        return "ResponsibilityAttentionDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ResponsibilityAttentionDataEncoder)

    def seq_encode(self, x: Sequence[tuple[Any, Any, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = np.asarray([np.asarray(xi[0], dtype=int) for xi in x], dtype=int)
        y = np.asarray([np.asarray(xi[1], dtype=float) for xi in x], dtype=float)
        t = np.asarray([int(xi[2]) for xi in x], dtype=int)
        return ctx, y, t


__all__ = [
    "ResponsibilityAttentionDistribution",
    "ResponsibilityAttentionSampler",
    "ResponsibilityAttentionAccumulator",
    "ResponsibilityAttentionAccumulatorFactory",
    "ResponsibilityAttentionEstimator",
    "ResponsibilityAttentionDataEncoder",
]
