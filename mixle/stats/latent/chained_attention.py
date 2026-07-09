"""Chained (multi-hop) attention: an L-hop stack of responsibility-attention, EM-able via forward-backward.

A single attention head answers one content lookup. *Chaining* L of them answers a multi-step lookup --
"find b that a points to, then find c that b points to" -- which one hop provably cannot do. Each
position in the context carries a ``(key_symbol, value_symbol)`` pair. The initial query is the one-hot
of the query symbol; at hop ``l`` it attends over positions by matching the current query to the hop's
key table ``K^(l)``, and the attended position's *value* (as a one-hot) becomes the query for hop
``l+1``. After ``L`` hops the target is emitted from the final attended value.

The hop latents ``z_1..z_L`` (which position each hop lands on) form a time-inhomogeneous chain over
the ``N`` context positions -- a content-addressed HMM with an emission only at the last hop -- so
**forward-backward** gives exact responsibilities in ``O(L N^2)`` and every M-step stays closed-form
and additively mergeable. Crucially the queries are *one-hot* (observed symbols / retrieved values), so
they anchor the key tables and the closed-form M-step does not collapse (unlike a tied latent
embedding, which needs the variational treatment).

Observation: ``(context_keys, context_values, query_symbol, target)``. ``n_hops = 1`` recovers a single
responsibility-attention head; ``n_hops = 2`` does transitive (``a->b->c``) lookup. The gate variance
``sigma2`` is the attention temperature -- keep it small (the one-hot query/key separation is only ``2``,
so a large ``sigma2`` blurs the chained prediction).

References: multi-hop content-addressed attention is the End-to-End Memory Network (Sukhbaatar, Szlam,
Weston & Fergus 2015); the transitive lookup is a bAbI-style reasoning task and the 2-hop copy is the
induction-head circuit (Olsson et al. 2022). Exact forward-backward over the alignment chain mirrors
HMM-based word alignment (Vogel & Ney 1996), which -- like here -- keeps the marginal likelihood exact.
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


def _gate(query_oh: np.ndarray, key_table: np.ndarray, keys: np.ndarray, sigma2: float) -> np.ndarray:
    """Attention over positions: softmax_i(-||query - K[key_i]||^2 / 2 sigma2). -> (n, N)."""
    d = query_oh[:, None, :] - key_table[keys]
    return _softmax(-np.sum(d * d, axis=2) / (2.0 * sigma2), axis=1)


def _forward_backward(K: np.ndarray, emission: np.ndarray, sigma2: float, enc, eye: np.ndarray):
    """Return (p, gamma[L], xi[L-1], transitions) for a batch (forward-backward over the hop chain)."""
    keys, vals, q, t = enc
    n, N = keys.shape
    L = K.shape[0]
    e_final = emission[vals, t[:, None]]  # (n, N)
    trans = []
    for ell in range(L - 1):
        valoh = eye[vals]  # (n, N, S)
        dd = valoh[:, :, None, :] - K[ell + 1][keys][:, None, :, :]  # (n, i, j, S)
        trans.append(_softmax(-np.sum(dd * dd, axis=3) / (2.0 * sigma2), axis=2))
    alpha = [_gate(eye[q], K[0], keys, sigma2)]
    for ell in range(L - 1):
        alpha.append(np.einsum("ni,nij->nj", alpha[ell], trans[ell]))
    p = np.clip(np.einsum("nk,nk->n", alpha[L - 1], e_final), 1e-300, None)
    beta = [None] * L
    beta[L - 1] = e_final
    for ell in range(L - 2, -1, -1):
        beta[ell] = np.einsum("nij,nj->ni", trans[ell], beta[ell + 1])
    gamma = [alpha[ell] * beta[ell] / p[:, None] for ell in range(L)]
    xi = [alpha[ell][:, :, None] * trans[ell] * beta[ell + 1][:, None, :] / p[:, None, None] for ell in range(L - 1)]
    return p, gamma, xi, trans


class ChainedAttentionDistribution(SequenceEncodableProbabilityDistribution):
    """An L-hop stack of responsibility-attention heads (chained, content-addressed)."""

    def __init__(self, keys: np.ndarray, emission: np.ndarray, sigma2: float = 0.1, name: str | None = None) -> None:
        """Args:
        keys: ``(L, S, S)`` per-hop key tables (one-hot query space, so dim ``S``).
        emission: ``(S, T)`` per-(value-symbol) categorical over targets.
        sigma2: gate variance / attention temperature.
        name: optional name.
        """
        self.key_tables = np.asarray(keys, dtype=float)
        self.emission = np.asarray(emission, dtype=float)
        self.n_hops, self.num_symbols, _ = self.key_tables.shape
        self.num_targets = self.emission.shape[1]
        self.sigma2 = float(sigma2)
        self.name = name
        self._eye = np.eye(self.num_symbols)

    def __str__(self) -> str:
        return "ChainedAttentionDistribution(L=%d, S=%d, T=%d, sigma2=%s, name=%s)" % (
            self.n_hops,
            self.num_symbols,
            self.num_targets,
            repr(self.sigma2),
            repr(self.name),
        )

    def density(self, x: tuple[Any, Any, int, int]) -> float:
        """Return the probability of one chained-attention observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[Any, Any, int, int]) -> float:
        """Return the log-probability of one context/query/target observation."""
        enc = self.dist_to_encoder().seq_encode([x])
        return float(self.seq_log_density(enc)[0])

    def seq_log_density(self, x) -> np.ndarray:
        """Return vectorized log-probabilities for encoded chained-attention observations."""
        p, _, _, _ = _forward_backward(self.key_tables, self.emission, self.sigma2, x, self._eye)
        return np.log(p)

    def predict_proba(self, context_keys: np.ndarray, context_values: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Predictive target distribution (target marginalized); ``(T,)`` or ``(n, T)``."""
        single = np.ndim(context_keys) == 1
        keys = np.atleast_2d(np.asarray(context_keys, dtype=int))
        vals = np.atleast_2d(np.asarray(context_values, dtype=int))
        q = np.atleast_1d(np.asarray(query, dtype=int))
        alpha = _gate(self._eye[q], self.key_tables[0], keys, self.sigma2)
        for ell in range(self.n_hops - 1):
            valoh = self._eye[vals]
            dd = valoh[:, :, None, :] - self.key_tables[ell + 1][keys][:, None, :, :]
            tr = _softmax(-np.sum(dd * dd, axis=3) / (2.0 * self.sigma2), axis=2)
            alpha = np.einsum("ni,nij->nj", alpha, tr)
        pred = np.einsum("nk,nkt->nt", alpha, self.emission[vals])
        return pred[0] if single else pred

    def sampler(self, seed: int | None = None) -> ChainedAttentionSampler:
        """Return a sampler for synthetic chained-attention observations."""
        return ChainedAttentionSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ChainedAttentionEstimator:
        """Return a closed-form EM estimator for this attention chain."""
        return ChainedAttentionEstimator(
            n_hops=self.n_hops,
            num_symbols=self.num_symbols,
            num_targets=self.num_targets,
            sigma2=self.sigma2,
            name=self.name,
        )

    def dist_to_encoder(self) -> ChainedAttentionDataEncoder:
        """Return the encoder for context keys, values, query symbols, and targets."""
        return ChainedAttentionDataEncoder()


class ChainedAttentionSampler(DistributionSampler):
    """Generative sampler (uniform context + the chained gate)."""

    def __init__(self, dist: ChainedAttentionDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one observation or ``size`` iid synthetic observations."""
        n = 1 if size is None else size
        d = self.dist
        N = 6
        out = []
        for _ in range(n):
            keys = self.rng.randint(0, d.num_symbols, size=N)
            vals = self.rng.randint(0, d.num_symbols, size=N)
            q = int(self.rng.randint(0, d.num_symbols))
            cur = d._eye[q]
            pos = None
            for ell in range(d.n_hops):
                a = _gate(cur[None, :], d.key_tables[ell], keys[None, :], d.sigma2)[0]
                pos = int(self.rng.choice(N, p=a))
                cur = d._eye[vals[pos]]
            t = int(self.rng.choice(d.num_targets, p=d.emission[vals[pos]]))
            out.append((keys, vals, q, t))
        return out[0] if size is None else out


class ChainedAttentionAccumulator(SequenceEncodableStatisticAccumulator):
    """Forward-backward sufficient statistics (additive): per-hop key numerators + emission counts."""

    def __init__(self, n_hops, num_symbols, num_targets, keys=None, name=None) -> None:
        self.n_hops = n_hops
        self.num_symbols = num_symbols
        self.num_targets = num_targets
        self.key_num = np.zeros((n_hops, num_symbols, num_symbols))
        self.key_mass = np.zeros((n_hops, num_symbols))
        self.emission_count = np.zeros((num_symbols, num_targets))
        self.ll = 0.0
        self.n = 0.0
        self.keys = keys
        self.name = name
        self._eye = np.eye(num_symbols)

    def _accumulate(self, enc, gamma, xi, weights) -> None:
        keys, vals, q, t = enc
        N = keys.shape[1]
        w = weights[:, None]
        # hop 0: incoming query = one-hot(q), weighted by occupancy gamma[0]
        g0 = gamma[0] * w
        np.add.at(
            self.key_num[0], keys.reshape(-1), (g0[:, :, None] * self._eye[q][:, None, :]).reshape(-1, self.num_symbols)
        )
        np.add.at(self.key_mass[0], keys.reshape(-1), g0.reshape(-1))
        for ell in range(1, self.n_hops):
            xw = xi[ell - 1] * w[:, :, None]
            inq = np.einsum("nij,nis->njs", xw, self._eye[vals])  # expected incoming one-hot at position j
            wj = xw.sum(axis=1)
            np.add.at(self.key_num[ell], keys.reshape(-1), inq.reshape(-1, self.num_symbols))
            np.add.at(self.key_mass[ell], keys.reshape(-1), wj.reshape(-1))
        gL = gamma[self.n_hops - 1] * w
        np.add.at(self.emission_count, (vals.reshape(-1), np.repeat(t, N)), gL.reshape(-1))

    def seq_update(self, x, weights, estimate: ChainedAttentionDistribution) -> None:
        """Update forward-backward sufficient statistics from encoded observations."""
        w = np.asarray(weights, dtype=float)
        p, gamma, xi, _ = _forward_backward(estimate.key_tables, estimate.emission, estimate.sigma2, x, estimate._eye)
        self._accumulate(x, gamma, xi, w)
        self.ll += float(np.dot(w, np.log(p)))
        self.n += float(w.sum())

    def seq_initialize(self, x, weights, rng: RandomState) -> None:
        """Initialize sufficient statistics with random hop responsibilities."""
        keys, vals, q, t = x
        n, N = keys.shape
        w = np.asarray(weights, dtype=float)
        gamma = [rng.dirichlet(np.ones(N), size=n) for _ in range(self.n_hops)]
        xi = [rng.dirichlet(np.ones(N * N), size=n).reshape(n, N, N) for _ in range(self.n_hops - 1)]
        self._accumulate(x, gamma, xi, w)
        self.n += float(w.sum())

    def update(self, x, weight: float, estimate) -> None:
        """Update from one weighted chained-attention observation."""
        enc = ChainedAttentionDataEncoder().seq_encode([x])
        self.seq_update(enc, np.array([weight], dtype=float), estimate)

    def initialize(self, x, weight: float, rng: RandomState) -> None:
        """Initialize from one weighted chained-attention observation."""
        enc = ChainedAttentionDataEncoder().seq_encode([x])
        self.seq_initialize(enc, np.array([weight], dtype=float), rng)

    def combine(self, suff_stat) -> ChainedAttentionAccumulator:
        """Merge key-table, emission, likelihood, and weight statistics."""
        kn, km, ec, ll, n = suff_stat
        self.key_num += kn
        self.key_mass += km
        self.emission_count += ec
        self.ll += ll
        self.n += n
        return self

    def value(self):
        """Return key statistics, emission counts, log-likelihood, and total weight."""
        return (self.key_num.copy(), self.key_mass.copy(), self.emission_count.copy(), self.ll, self.n)

    def from_value(self, x) -> ChainedAttentionAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.key_num, self.key_mass, self.emission_count = (np.asarray(v, dtype=float) for v in x[:3])
        self.ll = float(x[3])
        self.n = float(x[4])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.combine(stats_dict[self.keys])
            else:
                stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> ChainedAttentionDataEncoder:
        """Return the encoder compatible with this accumulator."""
        return ChainedAttentionDataEncoder()


class ChainedAttentionAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for chained-attention EM statistics."""

    def __init__(self, n_hops, num_symbols, num_targets, keys=None, name=None) -> None:
        self.n_hops = n_hops
        self.num_symbols = num_symbols
        self.num_targets = num_targets
        self.keys = keys
        self.name = name

    def make(self) -> ChainedAttentionAccumulator:
        """Create an empty chained-attention accumulator."""
        return ChainedAttentionAccumulator(
            self.n_hops, self.num_symbols, self.num_targets, keys=self.keys, name=self.name
        )


class ChainedAttentionEstimator(ParameterEstimator):
    """Closed-form EM estimator: per-hop key tables (GMM means of one-hot queries) + emission counts."""

    def __init__(
        self,
        n_hops: int,
        num_symbols: int,
        num_targets: int,
        *,
        sigma2: float = 0.1,
        emission_smoothing: float = 1e-4,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Args:
        n_hops, num_symbols, num_targets: model dimensions ``L, S, T``.
        sigma2: fixed gate variance (attention temperature).
        emission_smoothing: additive smoothing on the emission M-step.
        pseudo_count / name / keys: standard controls.
        """
        self.n_hops = n_hops
        self.num_symbols = num_symbols
        self.num_targets = num_targets
        self.sigma2 = float(sigma2)
        self.emission_smoothing = float(emission_smoothing)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ChainedAttentionAccumulatorFactory:
        """Return a factory for chained-attention sufficient-statistic accumulators."""
        return ChainedAttentionAccumulatorFactory(
            self.n_hops, self.num_symbols, self.num_targets, keys=self.keys, name=self.name
        )

    def estimate(self, nobs: float | None, suff_stat) -> ChainedAttentionDistribution:
        """Estimate key tables and emissions from accumulated forward-backward statistics."""
        key_num, key_mass, emission_count, _ll, _n = suff_stat
        key_tables = key_num / np.clip(key_mass, 1e-9, None)[:, :, None]
        em = emission_count + self.emission_smoothing
        emission = em / em.sum(axis=1, keepdims=True)
        return ChainedAttentionDistribution(key_tables, emission, sigma2=self.sigma2, name=self.name)


class ChainedAttentionDataEncoder(DataSequenceEncoder):
    """Encodes ``(context_keys, context_values, query_symbol, target)`` into stacked integer arrays."""

    def __str__(self) -> str:
        return "ChainedAttentionDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChainedAttentionDataEncoder)

    def seq_encode(
        self, x: Sequence[tuple[Any, Any, int, int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode ``(context_keys, context_values, query, target)`` observations."""
        keys = np.asarray([np.asarray(xi[0], dtype=int) for xi in x], dtype=int)
        vals = np.asarray([np.asarray(xi[1], dtype=int) for xi in x], dtype=int)
        q = np.asarray([int(xi[2]) for xi in x], dtype=int)
        t = np.asarray([int(xi[3]) for xi in x], dtype=int)
        return keys, vals, q, t


__all__ = [
    "ChainedAttentionDistribution",
    "ChainedAttentionSampler",
    "ChainedAttentionAccumulator",
    "ChainedAttentionAccumulatorFactory",
    "ChainedAttentionEstimator",
    "ChainedAttentionDataEncoder",
]
