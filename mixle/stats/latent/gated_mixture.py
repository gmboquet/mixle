"""Gated mixture (mixture of experts): mixing weights are a learned function of a covariate, not constants.

``MixtureDistribution`` mixes ``K`` components with FIXED weights ``w_k``. A gated mixture replaces those
constants with a *gate* ``p(k | z)`` -- a function of a per-observation covariate ``z`` -- so the mixture
that explains ``y`` shifts smoothly as ``z`` moves. That is the classic mixture-of-experts (Jacobs et al.
1991): each component is an "expert" over ``y``, the gate routes probability mass among them by ``z``.

An observation is a pair ``(z, y)``: ``z`` drives the gate, ``y`` is scored by the experts. The density is

    p(y | z) = sum_k gate_k(z) * f_k(y),      gate_k(z) = softmax over experts of the gate's logits at z.

EM is the same responsibility loop as a plain mixture, with the gate in place of the constant prior:
the E-step forms ``r_nk ∝ gate_k(z_n) f_k(y_n)``; the M-step (a) refits each expert on ``y`` weighted by
its responsibilities (exactly as a plain mixture does) and (b) refits the gate to predict ``r_nk`` from
``z_n`` (a soft-target multinomial regression). Unlike a plain mixture's closed-form weight update, the
gate step is an optimization, so the accumulator buffers ``(z, responsibilities)`` -- the same
buffer-the-rows pattern the neural leaves and :class:`~mixle.stats.combinator.copula.CopulaDistribution`
use.

The gate is pluggable (any object implementing the small ``Gate`` protocol below). The default
:class:`SoftmaxGate` is a torch-free multinomial logistic regression, so a gated mixture needs no torch;
a :class:`~mixle.models.softmax_leaf.NeuralCategorical`-backed gate can be substituted for a deep gate.

Reference: Jacobs, Jordan, Nowlan & Hinton, "Adaptive Mixtures of Local Experts" (Neural Computation, 1991).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class SoftmaxGate:
    """A torch-free multinomial-logistic gate ``p(k | z) = softmax(W z + b)_k``, fit on soft targets.

    ``fit(Z, R)`` minimizes the soft cross-entropy ``-sum_{n,k} R_{n,k} log p(k | z_n)`` by gradient
    descent -- ``R`` are the responsibilities (rows need not sum to 1; the sample weight is folded in).
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = np.asarray(weight, dtype=np.float64)  # (K, p)
        self.bias = np.asarray(bias, dtype=np.float64)  # (K,)
        self.n_classes = self.weight.shape[0]
        self.n_features = self.weight.shape[1]

    @classmethod
    def zeros(cls, n_classes: int, n_features: int) -> SoftmaxGate:
        """Create a zero-logit gate with uniform initial class probabilities."""
        return cls(np.zeros((n_classes, n_features)), np.zeros(n_classes))

    def log_prob_batch(self, z: np.ndarray) -> np.ndarray:
        """``(n, K)`` log-gate ``log p(k | z_n)`` for each row of ``z`` (shape ``(n, p)``)."""
        logits = np.asarray(z, dtype=np.float64) @ self.weight.T + self.bias  # (n, K)
        logits -= logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return logits - logsumexp

    def fit(self, z: np.ndarray, resp: np.ndarray, *, steps: int = 200, lr: float = 0.1) -> SoftmaxGate:
        """Fit a softmax gate to responsibility-weighted soft targets."""
        z = np.asarray(z, dtype=np.float64)
        r = np.asarray(resp, dtype=np.float64)
        n = max(len(z), 1)
        w, b = self.weight.copy(), self.bias.copy()
        row_mass = r.sum(axis=1, keepdims=True)  # per-row total responsibility (== sample weight)
        for _ in range(int(steps)):
            logits = z @ w.T + b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad_logits = probs * row_mass - r  # (n, K): d(soft-CE)/d(logits)
            gw = grad_logits.T @ z / n
            gb = grad_logits.sum(axis=0) / n
            w -= lr * gw
            b -= lr * gb
        return SoftmaxGate(w, b)


class GatedMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """A mixture whose weights are a gate ``p(k | z)``; observations are ``(z, y)`` pairs."""

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        gate: Any,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.components = list(components)
        self.num_components = len(self.components)
        if self.num_components < 2:
            raise ValueError("GatedMixtureDistribution needs at least 2 experts; got %d" % self.num_components)
        if getattr(gate, "n_classes", self.num_components) != self.num_components:
            raise ValueError(
                "gate.n_classes (%d) must match the number of experts (%d)" % (gate.n_classes, self.num_components)
            )
        self.gate = gate
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "GatedMixtureDistribution([%s], gate=%s)" % (
            ", ".join(map(str, self.components)),
            type(self.gate).__name__,
        )

    def log_density(self, x: tuple[Any, Any]) -> float:
        """Return ``log p(y | z)`` for one covariate/response pair."""
        z, y = x
        gate_lp = self.gate.log_prob_batch(np.atleast_2d(np.asarray(z, dtype=np.float64)))[0]
        comp_lp = np.array([gate_lp[k] + float(self.components[k].log_density(y)) for k in range(self.num_components)])
        m = comp_lp.max()
        return float(m + np.log(np.exp(comp_lp - m).sum())) if np.isfinite(m) else float("-inf")

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return vectorized conditional log-densities for encoded ``(z, y)`` pairs."""
        z_arr, comp_encs = enc
        gate_lp = self.gate.log_prob_batch(z_arr)  # (n, K)
        ll_mat = np.full((z_arr.shape[0], self.num_components), -np.inf, dtype=np.float64)
        for k in range(self.num_components):
            ll_mat[:, k] = gate_lp[:, k] + np.asarray(
                self.components[k].seq_log_density(comp_encs[k]), dtype=np.float64
            )
        m = ll_mat.max(axis=1, keepdims=True)
        finite = np.isfinite(m.ravel())
        out = np.full(z_arr.shape[0], -np.inf, dtype=np.float64)
        out[finite] = m.ravel()[finite] + np.log(np.exp(ll_mat[finite] - m[finite]).sum(axis=1))
        return out

    def posterior(self, x: tuple[Any, Any]) -> np.ndarray:
        """Return posterior expert responsibilities for one ``(z, y)`` observation."""
        z, y = x
        gate_lp = self.gate.log_prob_batch(np.atleast_2d(np.asarray(z, dtype=np.float64)))[0]
        lp = np.array([gate_lp[k] + float(self.components[k].log_density(y)) for k in range(self.num_components)])
        m = lp.max()
        if not np.isfinite(m):
            return np.exp(gate_lp)
        p = np.exp(lp - m)
        return p / p.sum()

    def sampler(self, seed: int | None = None) -> GatedMixtureSampler:
        """Return a conditional sampler that requires a covariate ``z``."""
        return GatedMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GatedMixtureEstimator:
        """Return an EM estimator for experts and the covariate-dependent gate."""
        return GatedMixtureEstimator(
            [c.estimator() for c in self.components], self.gate, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> GatedMixtureDataEncoder:
        """Return the encoder for covariates plus expert response encodings."""
        return GatedMixtureDataEncoder([c.dist_to_encoder() for c in self.components])


class GatedMixtureSampler(DistributionSampler):
    """Sample ``y`` given a supplied ``z``: draw a component from ``gate(z)``, then sample that expert."""

    def __init__(self, dist: GatedMixtureDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self._comp_samplers = [c.sampler(seed) for c in dist.components]

    def sample_given(self, z: Any) -> Any:
        """Sample a response from the gated mixture conditional on covariate ``z``."""
        gate_p = np.exp(self.dist.gate.log_prob_batch(np.atleast_2d(np.asarray(z, dtype=np.float64)))[0])
        k = int(self.rng.choice(self.dist.num_components, p=gate_p / gate_p.sum()))
        return self._comp_samplers[k].sample()

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because unconditional sampling requires caller-supplied covariates."""
        raise NotImplementedError("GatedMixture is conditional p(y|z); use sampler().sample_given(z).")


class GatedMixtureDataEncoder(DataSequenceEncoder):
    """Encode ``[(z, y), ...]`` as ``(z array (n, p), per-expert encodings of the y column)``."""

    def __init__(self, component_encoders: Sequence[DataSequenceEncoder]) -> None:
        self.component_encoders = list(component_encoders)

    def __str__(self) -> str:
        return "GatedMixtureDataEncoder([%s])" % ", ".join(map(str, self.component_encoders))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GatedMixtureDataEncoder) and self.component_encoders == other.component_encoders

    def seq_encode(self, data: Sequence[tuple[Any, Any]]) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Encode covariates as a dense matrix and responses for every expert."""
        z = np.asarray([np.atleast_1d(np.asarray(row[0], dtype=np.float64)) for row in data], dtype=np.float64)
        ys = [row[1] for row in data]
        comp_encs = tuple(enc.seq_encode(ys) for enc in self.component_encoders)
        return z, comp_encs


class GatedMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """E-step responsibilities route weight to expert sub-accumulators; buffer ``(z, resp)`` for the gate."""

    def __init__(self, component_accumulators: Sequence[Any], num_components: int, keys: str | None = None) -> None:
        self.component_accumulators = list(component_accumulators)
        self.num_components = num_components
        self.keys = keys
        self._z: list[np.ndarray] = []
        self._resp: list[np.ndarray] = []

    def _responsibilities(
        self, enc: Any, weights: np.ndarray, estimate: GatedMixtureDistribution | None
    ) -> tuple[np.ndarray, np.ndarray]:
        z_arr, comp_encs = enc
        n = z_arr.shape[0]
        if estimate is None:
            r = np.full((n, self.num_components), 1.0 / self.num_components)
        else:
            gate_lp = estimate.gate.log_prob_batch(z_arr)
            ll = np.empty((n, self.num_components))
            for k in range(self.num_components):
                ll[:, k] = gate_lp[:, k] + np.asarray(
                    estimate.components[k].seq_log_density(comp_encs[k]), dtype=np.float64
                )
            m = ll.max(axis=1, keepdims=True)
            bad = ~np.isfinite(m.ravel())
            ll[bad] = gate_lp[bad] if estimate is not None else 0.0
            m[bad.nonzero()[0]] = ll[bad].max(axis=1, keepdims=True) if bad.any() else m[bad.nonzero()[0]]
            p = np.exp(ll - m)
            r = p / p.sum(axis=1, keepdims=True)
        r = r * np.asarray(weights, dtype=np.float64)[:, None]
        return z_arr, r

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: GatedMixtureDistribution | None) -> None:
        """Update expert accumulators and gate buffers from encoded observations."""
        z_arr, r = self._responsibilities(enc, weights, estimate)
        _, comp_encs = enc
        for k in range(self.num_components):
            self.component_accumulators[k].seq_update(
                comp_encs[k], r[:, k], None if estimate is None else estimate.components[k]
            )
        self._z.append(z_arr)
        self._resp.append(r)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize expert accumulators with random responsibility allocations."""
        z_arr, comp_encs = enc
        n = z_arr.shape[0]
        r = rng.dirichlet(np.ones(self.num_components), size=n) * np.asarray(weights, dtype=np.float64)[:, None]
        for k in range(self.num_components):
            self.component_accumulators[k].seq_initialize(comp_encs[k], r[:, k], rng)
        self._z.append(z_arr)
        self._resp.append(r)

    def update(self, x: tuple[Any, Any], weight: float, estimate: GatedMixtureDistribution | None) -> None:
        """Update from one weighted ``(z, y)`` observation."""
        enc = GatedMixtureDataEncoder([a.acc_to_encoder() for a in self.component_accumulators]).seq_encode([x])
        self.seq_update(enc, np.asarray([weight], dtype=np.float64), estimate)

    def initialize(self, x: tuple[Any, Any], weight: float, rng: np.random.RandomState) -> None:
        """Initialize from one weighted ``(z, y)`` observation."""
        enc = GatedMixtureDataEncoder([a.acc_to_encoder() for a in self.component_accumulators]).seq_encode([x])
        self.seq_initialize(enc, np.asarray([weight], dtype=np.float64), rng)

    def combine(self, suff_stat: tuple[tuple[Any, ...], np.ndarray, np.ndarray]) -> GatedMixtureAccumulator:
        """Merge expert sufficient statistics and buffered gate training data."""
        comp_stats, z, r = suff_stat
        for k in range(self.num_components):
            self.component_accumulators[k].combine(comp_stats[k])
        if len(z):
            self._z.append(np.asarray(z, dtype=np.float64))
            self._resp.append(np.asarray(r, dtype=np.float64))
        return self

    def value(self) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
        """Return expert statistics, buffered covariates, and responsibility targets."""
        comp_vals = tuple(a.value() for a in self.component_accumulators)
        z = np.concatenate(self._z, axis=0) if self._z else np.zeros((0, 0))
        r = np.concatenate(self._resp, axis=0) if self._resp else np.zeros((0, self.num_components))
        return comp_vals, z, r

    def from_value(self, x: tuple[tuple[Any, ...], np.ndarray, np.ndarray]) -> GatedMixtureAccumulator:
        """Restore expert statistics and gate training buffers."""
        comp_vals, z, r = x
        for k in range(self.num_components):
            self.component_accumulators[k].from_value(comp_vals[k])
        self._z = [np.asarray(z, dtype=np.float64)] if len(z) else []
        self._resp = [np.asarray(r, dtype=np.float64)] if len(z) else []
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed merges to expert accumulators."""
        for a in self.component_accumulators:
            if hasattr(a, "key_merge"):
                a.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed replacements to expert accumulators."""
        for a in self.component_accumulators:
            if hasattr(a, "key_replace"):
                a.key_replace(stats_dict)

    def acc_to_encoder(self) -> GatedMixtureDataEncoder:
        """Return the encoder composed from expert accumulator encoders."""
        return GatedMixtureDataEncoder([a.acc_to_encoder() for a in self.component_accumulators])


class GatedMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for gated-mixture EM."""

    def __init__(self, component_factories: Sequence[Any], num_components: int, keys: str | None = None) -> None:
        self.component_factories = list(component_factories)
        self.num_components = num_components
        self.keys = keys

    def make(self) -> GatedMixtureAccumulator:
        """Create an empty gated-mixture accumulator."""
        return GatedMixtureAccumulator(
            [f.make() for f in self.component_factories], self.num_components, keys=self.keys
        )


class GatedMixtureEstimator(ParameterEstimator):
    """M-step: refit each expert from its responsibility-weighted stats, refit the gate on ``(z, resp)``."""

    def __init__(
        self,
        component_estimators: Sequence[ParameterEstimator],
        gate: Any,
        gate_steps: int = 200,
        gate_lr: float = 0.1,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.component_estimators = list(component_estimators)
        self.num_components = len(self.component_estimators)
        self.gate = gate
        self.gate_steps = int(gate_steps)
        self.gate_lr = float(gate_lr)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GatedMixtureAccumulatorFactory:
        """Return a factory for gated-mixture sufficient-statistic accumulators."""
        return GatedMixtureAccumulatorFactory(
            [e.accumulator_factory() for e in self.component_estimators], self.num_components, keys=self.keys
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[tuple[Any, ...], np.ndarray, np.ndarray]
    ) -> GatedMixtureDistribution:
        """Estimate experts from responsibility-weighted stats and refit the gate."""
        comp_stats, z, r = suff_stat
        components = [self.component_estimators[k].estimate(nobs, comp_stats[k]) for k in range(self.num_components)]
        if len(z) and hasattr(self.gate, "fit"):
            gate = self.gate.fit(z, r, steps=self.gate_steps, lr=self.gate_lr)
        else:
            gate = self.gate
        return GatedMixtureDistribution(components, gate, name=self.name, keys=self.keys)
