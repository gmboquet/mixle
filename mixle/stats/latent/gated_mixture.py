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
gate step is an optimization, so the accumulator retains a bounded deterministic coreset of
``(z, responsibilities)`` rows. The coreset is selected by stable content priority, which makes it
mergeable across partitioned accumulators without unbounded memory growth; a receipt reports how
many rows were retained or dropped.

The gate is pluggable (any object implementing the small ``Gate`` protocol below). The default
:class:`SoftmaxGate` is a torch-free multinomial logistic regression, so a gated mixture needs no torch;
a :class:`~mixle.models.softmax_leaf.NeuralCategorical`-backed gate can be substituted for a deep gate.

Reference: Jacobs, Jordan, Nowlan & Hinton, "Adaptive Mixtures of Local Experts" (Neural Computation, 1991).
"""

from __future__ import annotations

import copy
import hashlib
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.stats.compute.mixture_evidence import normalize_mixture_log_scores
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_count_array,
    validated_observation_weight,
    validated_observation_weights,
    validated_weighted_responsibilities,
)
from mixle.stats.latent.mixture import _owned_generative_components


def _positive_integer(value: Any, label: str) -> int:
    """Return a non-boolean positive exact integer."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a positive exact integer.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be a positive exact integer.") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive.")
    return result


def _positive_finite(value: Any, label: str) -> float:
    """Return a positive finite floating-point control."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a positive finite number.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite.")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    """Return a non-negative finite floating-point control."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a non-negative finite number.") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be non-negative and finite.")
    return result


def _validated_covariates(z: Any, n_features: int | None, *, label: str) -> np.ndarray:
    """Return an owned finite ``(rows, features)`` covariate matrix."""
    try:
        values = np.asarray(z, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric matrix.") from exc
    if values.ndim != 2:
        raise ValueError(f"{label} must have shape (rows, features), got {values.shape}.")
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError(f"{label} must have {n_features} features, got {values.shape[1]}.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values.")
    return values.copy()


def _validated_gate_log_probabilities(gate: Any, z: Any, n_classes: int) -> np.ndarray:
    """Validate a gate's declared geometry and normalized batch probabilities."""
    n_features = getattr(gate, "n_features", None)
    values = _validated_covariates(z, n_features, label="gate covariates")
    if not callable(getattr(gate, "log_prob_batch", None)):
        raise TypeError("gate must define a callable log_prob_batch(z) method.")
    try:
        log_probabilities = np.asarray(gate.log_prob_batch(values), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("gate.log_prob_batch(z) must return a numeric matrix.") from exc
    expected = (values.shape[0], n_classes)
    if log_probabilities.shape != expected:
        raise ValueError(f"gate.log_prob_batch(z) must return shape {expected}, got {log_probabilities.shape}.")
    if np.any(np.isnan(log_probabilities)) or np.any(np.isposinf(log_probabilities)):
        raise ValueError("gate log probabilities cannot contain NaN or positive infinity.")
    probabilities = np.exp(log_probabilities)
    if np.any(~np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("gate probabilities must be finite, non-negative rows that sum to one.")
    return log_probabilities


@dataclass(frozen=True)
class GateOptimizationReceipt:
    """Outcome of a bounded softmax-gate optimization."""

    converged: bool
    steps_completed: int
    initial_loss: float
    final_loss: float
    gradient_norm: float
    termination: str


@dataclass(frozen=True)
class GateBufferReceipt:
    """Bounded deterministic coreset status for gate-training rows."""

    rows_seen: int
    rows_retained: int
    rows_dropped: int
    capacity: int
    selection: str = "deterministic_hash_top_k"


@dataclass(frozen=True)
class GatedMixtureStatistics(Sequence[Any]):
    """Versioned gated-mixture statistics with backward four-slot iteration."""

    component_statistics: tuple[Any, ...]
    covariates: np.ndarray
    responsibilities: np.ndarray
    buffer_receipt: GateBufferReceipt
    component_counts: np.ndarray
    schema_version: int = 1

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        legacy = (
            self.component_statistics,
            self.covariates,
            self.responsibilities,
            self.buffer_receipt,
        )
        return legacy[index]


def _unpack_gated_statistics(
    values: Any,
    *,
    num_components: int,
    max_buffer_rows: int,
) -> tuple[tuple[Any, ...], Any, Any, GateBufferReceipt, np.ndarray]:
    """Validate current statistics or reconcile an exact legacy buffer."""
    if isinstance(values, GatedMixtureStatistics):
        comp_stats = values.component_statistics
        z = values.covariates
        responsibilities = values.responsibilities
        receipt = values.buffer_receipt
        counts = validated_count_array(values.component_counts, (num_components,), "gated-mixture component counts")
    else:
        if not isinstance(values, (tuple, list)) or len(values) not in (3, 4):
            raise ValueError("gated-mixture sufficient statistics must contain three or four legacy entries")
        comp_stats, z, responsibilities = values[:3]
        receipt = GateBufferReceipt(len(z), len(z), 0, max_buffer_rows) if len(values) == 3 else values[3]
        if not isinstance(receipt, GateBufferReceipt):
            raise TypeError("gated-mixture buffer metadata must be a GateBufferReceipt")
        if receipt.rows_dropped:
            raise ValueError("legacy gated-mixture statistics with dropped rows have no recoverable component mass")
        raw_responsibilities = validated_count_array(
            responsibilities,
            (len(z), num_components),
            "gated-mixture buffered responsibilities",
        )
        counts = raw_responsibilities.sum(axis=0)
    if not isinstance(receipt, GateBufferReceipt):
        raise TypeError("gated-mixture buffer metadata must be a GateBufferReceipt")
    if not isinstance(comp_stats, (tuple, list)) or len(comp_stats) != num_components:
        raise ValueError("gated-mixture component statistics must have one item per component")
    return tuple(comp_stats), z, responsibilities, receipt, counts


class SoftmaxGate:
    """A torch-free multinomial-logistic gate ``p(k | z) = softmax(W z + b)_k``, fit on soft targets.

    ``fit(Z, R)`` minimizes the soft cross-entropy ``-sum_{n,k} R_{n,k} log p(k | z_n)`` by gradient
    descent -- ``R`` are the responsibilities (rows need not sum to 1; the sample weight is folded in).
    """

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray,
        *,
        fit_receipt: GateOptimizationReceipt | None = None,
    ) -> None:
        try:
            owned_weight = np.asarray(weight, dtype=np.float64)
            owned_bias = np.asarray(bias, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("SoftmaxGate weight and bias must be numeric arrays.") from exc
        if owned_weight.ndim != 2 or owned_weight.shape[0] == 0 or owned_weight.shape[1] == 0:
            raise ValueError("SoftmaxGate weight must have nonempty shape (classes, features).")
        if owned_bias.shape != (owned_weight.shape[0],):
            raise ValueError(
                "SoftmaxGate bias must have shape (%d,), got %r." % (owned_weight.shape[0], owned_bias.shape)
            )
        if np.any(~np.isfinite(owned_weight)) or np.any(~np.isfinite(owned_bias)):
            raise ValueError("SoftmaxGate parameters must be finite.")
        self.weight = owned_weight.copy()
        self.bias = owned_bias.copy()
        self.n_classes = self.weight.shape[0]
        self.n_features = self.weight.shape[1]
        self.fit_receipt = fit_receipt

    @classmethod
    def zeros(cls, n_classes: int, n_features: int) -> SoftmaxGate:
        """Create a zero-logit gate with uniform initial class probabilities."""
        n_classes = _positive_integer(n_classes, "n_classes")
        n_features = _positive_integer(n_features, "n_features")
        return cls(np.zeros((n_classes, n_features)), np.zeros(n_classes))

    def log_prob_batch(self, z: np.ndarray) -> np.ndarray:
        """``(n, K)`` log-gate ``log p(k | z_n)`` for each row of ``z`` (shape ``(n, p)``)."""
        values = _validated_covariates(z, self.n_features, label="SoftmaxGate covariates")
        logits = values @ self.weight.T + self.bias
        if np.any(~np.isfinite(logits)):
            raise ValueError("SoftmaxGate logits must remain finite.")
        logits -= logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return logits - logsumexp

    @staticmethod
    def _loss_and_gradient(
        z: np.ndarray,
        r: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return normalized soft cross-entropy and its parameter gradients."""
        logits = z @ w.T + b
        if np.any(~np.isfinite(logits)):
            return np.inf, np.full_like(w, np.nan), np.full_like(b, np.nan)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(exp_logits.sum(axis=1, keepdims=True))
        n = max(len(z), 1)
        loss = float(-np.sum(r * log_probabilities) / n)
        grad_logits = probabilities * r.sum(axis=1, keepdims=True) - r
        return loss, grad_logits.T @ z / n, grad_logits.sum(axis=0) / n

    def fit_with_receipt(
        self,
        z: np.ndarray,
        resp: np.ndarray,
        *,
        steps: int = 200,
        lr: float = 0.1,
        tol: float = 1.0e-8,
    ) -> tuple[SoftmaxGate, GateOptimizationReceipt]:
        """Fit a softmax gate and return explicit convergence/failure status."""
        steps = _positive_integer(steps, "SoftmaxGate steps")
        lr = _positive_finite(lr, "SoftmaxGate learning rate")
        tol = _nonnegative_finite(tol, "SoftmaxGate convergence tolerance")
        z = _validated_covariates(z, self.n_features, label="SoftmaxGate training covariates")
        try:
            r = np.asarray(resp, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("SoftmaxGate responsibilities must be a numeric matrix.") from exc
        if r.shape != (z.shape[0], self.n_classes):
            raise ValueError(
                "SoftmaxGate responsibilities must have shape %r, got %r." % ((z.shape[0], self.n_classes), r.shape)
            )
        if np.any(~np.isfinite(r)) or np.any(r < 0.0):
            raise ValueError("SoftmaxGate responsibilities must be finite and non-negative.")

        w, b = self.weight.copy(), self.bias.copy()
        if len(z) == 0 or float(r.sum()) == 0.0:
            receipt = GateOptimizationReceipt(True, 0, 0.0, 0.0, 0.0, "zero_mass")
            return SoftmaxGate(w, b, fit_receipt=receipt), receipt
        initial_loss, gw, gb = self._loss_and_gradient(z, r, w, b)
        gradient_norm = float(np.sqrt(np.sum(gw * gw) + np.sum(gb * gb)))

        final_loss = initial_loss
        converged = gradient_norm <= tol
        termination = "gradient_tolerance" if converged else "step_limit"
        completed = 0
        for step in range(steps):
            if converged:
                break
            step_size = lr
            accepted = False
            for _ in range(12):
                candidate_w = w - step_size * gw
                candidate_b = b - step_size * gb
                candidate_loss, candidate_gw, candidate_gb = self._loss_and_gradient(z, r, candidate_w, candidate_b)
                if (
                    np.isfinite(candidate_loss)
                    and np.all(np.isfinite(candidate_gw))
                    and np.all(np.isfinite(candidate_gb))
                    and candidate_loss <= final_loss
                ):
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                termination = "line_search_failed"
                break
            improvement = final_loss - candidate_loss
            w, b = candidate_w, candidate_b
            final_loss, gw, gb = candidate_loss, candidate_gw, candidate_gb
            completed = step + 1
            gradient_norm = float(np.sqrt(np.sum(gw * gw) + np.sum(gb * gb)))
            if gradient_norm <= tol:
                converged = True
                termination = "gradient_tolerance"
            elif abs(improvement) <= tol * max(1.0, abs(final_loss)):
                converged = True
                termination = "loss_tolerance"

        receipt = GateOptimizationReceipt(
            converged,
            completed,
            initial_loss,
            final_loss,
            gradient_norm,
            termination,
        )
        return SoftmaxGate(w, b, fit_receipt=receipt), receipt

    def fit(
        self,
        z: np.ndarray,
        resp: np.ndarray,
        *,
        steps: int = 200,
        lr: float = 0.1,
        tol: float = 1.0e-8,
    ) -> SoftmaxGate:
        """Fit a softmax gate; the returned gate carries ``fit_receipt``."""
        fitted, _ = self.fit_with_receipt(z, resp, steps=steps, lr=lr, tol=tol)
        return fitted


class GatedMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """A mixture whose weights are a gate ``p(k | z)``; observations are ``(z, y)`` pairs."""

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        gate: Any,
        name: str | None = None,
        keys: str | None = None,
        *,
        gate_fit_receipt: GateOptimizationReceipt | None = None,
        gate_buffer_receipt: GateBufferReceipt | None = None,
    ) -> None:
        self.components = _owned_generative_components(
            components,
            "GatedMixtureDistribution",
            minimum=2,
        )
        self.num_components = len(self.components)
        if not callable(getattr(gate, "log_prob_batch", None)):
            raise TypeError("gate must define a callable log_prob_batch(z) method.")
        try:
            gate_classes = operator.index(gate.n_classes)
        except (AttributeError, TypeError) as exc:
            raise TypeError("gate.n_classes must be an exact integer.") from exc
        if gate_classes != self.num_components:
            raise ValueError(
                "gate.n_classes (%d) must match the number of experts (%d)" % (gate_classes, self.num_components)
            )
        if hasattr(gate, "n_features"):
            gate_features = _positive_integer(gate.n_features, "gate.n_features")
            _validated_gate_log_probabilities(gate, np.zeros((1, gate_features)), self.num_components)
        self.gate = gate
        self.gate_fit_receipt = gate_fit_receipt or getattr(gate, "fit_receipt", None)
        self.gate_buffer_receipt = gate_buffer_receipt
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
        gate_lp = _validated_gate_log_probabilities(
            self.gate,
            np.atleast_2d(np.asarray(z, dtype=np.float64)),
            self.num_components,
        )[0]
        comp_lp = np.array([gate_lp[k] + float(self.components[k].log_density(y)) for k in range(self.num_components)])
        return float(normalize_mixture_log_scores(comp_lp[None, :]).log_evidence[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return vectorized conditional log-densities for encoded ``(z, y)`` pairs."""
        z_arr, comp_encs = enc
        gate_lp = _validated_gate_log_probabilities(self.gate, z_arr, self.num_components)
        ll_mat = np.full((z_arr.shape[0], self.num_components), -np.inf, dtype=np.float64)
        for k in range(self.num_components):
            ll_mat[:, k] = gate_lp[:, k] + np.asarray(
                self.components[k].seq_log_density(comp_encs[k]), dtype=np.float64
            )
        return normalize_mixture_log_scores(ll_mat).log_evidence

    def posterior(self, x: tuple[Any, Any]) -> np.ndarray:
        """Return posterior expert responsibilities for one ``(z, y)`` observation."""
        z, y = x
        gate_lp = _validated_gate_log_probabilities(
            self.gate,
            np.atleast_2d(np.asarray(z, dtype=np.float64)),
            self.num_components,
        )[0]
        lp = np.array([gate_lp[k] + float(self.components[k].log_density(y)) for k in range(self.num_components)])
        return normalize_mixture_log_scores(lp[None, :]).responsibilities[0]

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
        return GatedMixtureDataEncoder(
            [c.dist_to_encoder() for c in self.components],
            n_features=getattr(self.gate, "n_features", None),
        )


class GatedMixtureSampler(DistributionSampler):
    """Sample ``y`` given a supplied ``z``: draw a component from ``gate(z)``, then sample that expert."""

    def __init__(self, dist: GatedMixtureDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self._comp_samplers = [c.sampler(seed) for c in dist.components]

    def sample_given(self, z: Any) -> Any:
        """Sample a response from the gated mixture conditional on covariate ``z``."""
        gate_lp = _validated_gate_log_probabilities(
            self.dist.gate,
            np.atleast_2d(np.asarray(z, dtype=np.float64)),
            self.dist.num_components,
        )[0]
        gate_p = np.exp(gate_lp)
        k = int(self.rng.choice(self.dist.num_components, p=gate_p))
        return self._comp_samplers[k].sample()

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because unconditional sampling requires caller-supplied covariates."""
        raise NotImplementedError("GatedMixture is conditional p(y|z); use sampler().sample_given(z).")


class GatedMixtureDataEncoder(DataSequenceEncoder):
    """Encode ``[(z, y), ...]`` as ``(z array (n, p), per-expert encodings of the y column)``."""

    def __init__(
        self,
        component_encoders: Sequence[DataSequenceEncoder],
        n_features: int | None = None,
    ) -> None:
        self.component_encoders = list(component_encoders)
        self.n_features = None if n_features is None else _positive_integer(n_features, "n_features")

    def __str__(self) -> str:
        return "GatedMixtureDataEncoder([%s])" % ", ".join(map(str, self.component_encoders))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GatedMixtureDataEncoder)
            and self.component_encoders == other.component_encoders
            and self.n_features == other.n_features
        )

    def seq_encode(self, data: Sequence[tuple[Any, Any]]) -> tuple[np.ndarray, tuple[Any, ...]]:
        """Encode covariates as a dense matrix and responses for every expert."""
        covariates = [np.atleast_1d(np.asarray(row[0], dtype=np.float64)) for row in data]
        if covariates:
            z = np.asarray(covariates, dtype=np.float64)
            if z.ndim != 2:
                raise ValueError("gated-mixture covariates must all have one consistent feature width.")
        else:
            z = np.zeros((0, self.n_features or 0), dtype=np.float64)
        z = _validated_covariates(z, self.n_features, label="gated-mixture covariates")
        ys = [row[1] for row in data]
        comp_encs = tuple(enc.seq_encode(ys) for enc in self.component_encoders)
        return z, comp_encs


class GatedMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """Route expert statistics and retain a bounded mergeable gate-training coreset."""

    def __init__(
        self,
        component_accumulators: Sequence[Any],
        num_components: int,
        keys: str | None = None,
        *,
        max_buffer_rows: int = 10_000,
        n_features: int | None = None,
    ) -> None:
        self.component_accumulators = list(component_accumulators)
        self.num_components = _positive_integer(num_components, "num_components")
        if len(self.component_accumulators) != self.num_components:
            raise ValueError("gated-mixture accumulator count must match num_components.")
        self.keys = keys
        self.max_buffer_rows = _positive_integer(max_buffer_rows, "max_buffer_rows")
        self.n_features = None if n_features is None else _positive_integer(n_features, "n_features")
        self._z = np.zeros((0, self.n_features or 0), dtype=np.float64)
        self._resp = np.zeros((0, self.num_components), dtype=np.float64)
        self._priorities = np.zeros(0, dtype=np.uint64)
        self._rows_seen = 0
        self.component_counts = np.zeros(self.num_components, dtype=np.float64)

    @staticmethod
    def _row_priorities(z: np.ndarray, resp: np.ndarray) -> np.ndarray:
        """Return stable content priorities so top-k coresets merge associatively."""
        priorities = np.empty(len(z), dtype=np.uint64)
        for index, (z_row, r_row) in enumerate(zip(z, resp)):
            digest = hashlib.blake2b(digest_size=8, person=b"mixle-gate")
            digest.update(np.ascontiguousarray(z_row, dtype=np.float64).tobytes())
            digest.update(np.ascontiguousarray(r_row, dtype=np.float64).tobytes())
            priorities[index] = int.from_bytes(digest.digest(), byteorder="big", signed=False)
        return priorities

    @property
    def buffer_receipt(self) -> GateBufferReceipt:
        """Return current bounded-buffer retention status."""
        retained = len(self._z)
        return GateBufferReceipt(
            rows_seen=self._rows_seen,
            rows_retained=retained,
            rows_dropped=max(self._rows_seen - retained, 0),
            capacity=self.max_buffer_rows,
        )

    def _append_gate_rows(
        self,
        z: Any,
        resp: Any,
        *,
        rows_seen: int | None = None,
    ) -> None:
        """Merge rows into the deterministic bounded top-k coreset."""
        values = _validated_covariates(z, self.n_features, label="gated-mixture buffered covariates")
        if self.n_features is None and values.shape[1] > 0:
            self.n_features = values.shape[1]
            if self._z.shape[1] == 0:
                self._z = np.zeros((0, self.n_features), dtype=np.float64)
        try:
            responsibilities = np.asarray(resp, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("gated-mixture buffered responsibilities must be numeric.") from exc
        if responsibilities.shape != (len(values), self.num_components):
            raise ValueError(
                "gated-mixture buffered responsibilities must have shape %r, got %r."
                % ((len(values), self.num_components), responsibilities.shape)
            )
        if np.any(~np.isfinite(responsibilities)) or np.any(responsibilities < 0.0):
            raise ValueError("gated-mixture buffered responsibilities must be finite and non-negative.")

        seen_increment = len(values) if rows_seen is None else operator.index(rows_seen)
        if seen_increment < len(values):
            raise ValueError("rows_seen cannot be smaller than the number of retained gate rows.")
        self._rows_seen += seen_increment
        if len(values) == 0:
            return

        priorities = self._row_priorities(values, responsibilities)
        all_z = np.concatenate((self._z, values), axis=0)
        all_resp = np.concatenate((self._resp, responsibilities), axis=0)
        all_priorities = np.concatenate((self._priorities, priorities), axis=0)
        if len(all_z) > self.max_buffer_rows:
            keep = np.argpartition(all_priorities, self.max_buffer_rows - 1)[: self.max_buffer_rows]
            keep = keep[np.argsort(all_priorities[keep], kind="stable")]
            all_z = all_z[keep]
            all_resp = all_resp[keep]
            all_priorities = all_priorities[keep]
        self._z = all_z.copy()
        self._resp = all_resp.copy()
        self._priorities = all_priorities.copy()

    def _responsibilities(
        self, enc: Any, weights: np.ndarray, estimate: GatedMixtureDistribution | None
    ) -> tuple[np.ndarray, np.ndarray]:
        z_arr, comp_encs = enc
        n = z_arr.shape[0]
        checked_weights = validated_observation_weights(weights, n, "gated-mixture weights")
        if estimate is None:
            r = np.full((n, self.num_components), 1.0 / self.num_components)
        else:
            gate_lp = _validated_gate_log_probabilities(estimate.gate, z_arr, self.num_components)
            ll = np.empty((n, self.num_components))
            for k in range(self.num_components):
                ll[:, k] = gate_lp[:, k] + np.asarray(
                    estimate.components[k].seq_log_density(comp_encs[k]), dtype=np.float64
                )
            r = normalize_mixture_log_scores(ll).responsibilities
        r = validated_weighted_responsibilities(
            r * checked_weights[:, None],
            checked_weights,
            self.num_components,
            label="gated-mixture responsibilities",
            allow_unassigned=True,
        )
        return z_arr, r

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: GatedMixtureDistribution | None) -> None:
        """Update expert accumulators and gate buffers from encoded observations."""
        z_arr, r = self._responsibilities(enc, weights, estimate)
        _, comp_encs = enc
        self.component_counts += r.sum(axis=0)
        for k in range(self.num_components):
            self.component_accumulators[k].seq_update(
                comp_encs[k], r[:, k], None if estimate is None else estimate.components[k]
            )
        self._append_gate_rows(z_arr, r)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize expert accumulators with random responsibility allocations."""
        z_arr, comp_encs = enc
        n = z_arr.shape[0]
        checked_weights = validated_observation_weights(weights, n, "gated-mixture weights")
        r = validated_weighted_responsibilities(
            rng.dirichlet(np.ones(self.num_components), size=n) * checked_weights[:, None],
            checked_weights,
            self.num_components,
            label="gated-mixture initialization responsibilities",
        )
        self.component_counts += r.sum(axis=0)
        for k in range(self.num_components):
            self.component_accumulators[k].seq_initialize(comp_encs[k], r[:, k], rng)
        self._append_gate_rows(z_arr, r)

    def update(self, x: tuple[Any, Any], weight: float, estimate: GatedMixtureDistribution | None) -> None:
        """Update from one weighted ``(z, y)`` observation."""
        weight = validated_observation_weight(weight)
        enc = GatedMixtureDataEncoder([a.acc_to_encoder() for a in self.component_accumulators]).seq_encode([x])
        self.seq_update(enc, np.asarray([weight], dtype=np.float64), estimate)

    def initialize(self, x: tuple[Any, Any], weight: float, rng: np.random.RandomState) -> None:
        """Initialize from one weighted ``(z, y)`` observation."""
        weight = validated_observation_weight(weight)
        enc = GatedMixtureDataEncoder([a.acc_to_encoder() for a in self.component_accumulators]).seq_encode([x])
        self.seq_initialize(enc, np.asarray([weight], dtype=np.float64), rng)

    def combine(self, suff_stat: tuple[Any, ...]) -> GatedMixtureAccumulator:
        """Merge expert sufficient statistics and buffered gate training data."""
        comp_stats, z, r, receipt, counts = _unpack_gated_statistics(
            suff_stat,
            num_components=self.num_components,
            max_buffer_rows=self.max_buffer_rows,
        )
        self.component_counts += counts
        for k in range(self.num_components):
            self.component_accumulators[k].combine(copy.deepcopy(comp_stats[k]))
        if len(z):
            self._append_gate_rows(z, r, rows_seen=receipt.rows_seen)
        else:
            self._rows_seen += receipt.rows_seen
        return self

    def value(self) -> GatedMixtureStatistics:
        """Return owned expert statistics, bounded gate rows, and retention status."""
        comp_vals = tuple(copy.deepcopy(a.value()) for a in self.component_accumulators)
        return GatedMixtureStatistics(
            comp_vals,
            self._z.copy(),
            self._resp.copy(),
            self.buffer_receipt,
            self.component_counts.copy(),
        )

    def from_value(self, x: tuple[Any, ...]) -> GatedMixtureAccumulator:
        """Restore expert statistics and gate training buffers."""
        comp_vals, z, r, receipt, counts = _unpack_gated_statistics(
            x,
            num_components=self.num_components,
            max_buffer_rows=self.max_buffer_rows,
        )
        for k in range(self.num_components):
            self.component_accumulators[k].from_value(copy.deepcopy(comp_vals[k]))
        self._z = np.zeros((0, self.n_features or 0), dtype=np.float64)
        self._resp = np.zeros((0, self.num_components), dtype=np.float64)
        self._priorities = np.zeros(0, dtype=np.uint64)
        self._rows_seen = 0
        self.component_counts = counts
        if len(z):
            self._append_gate_rows(z, r, rows_seen=receipt.rows_seen)
        else:
            self._rows_seen = receipt.rows_seen
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
        return GatedMixtureDataEncoder(
            [a.acc_to_encoder() for a in self.component_accumulators],
            n_features=self.n_features,
        )


class GatedMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for gated-mixture EM."""

    def __init__(
        self,
        component_factories: Sequence[Any],
        num_components: int,
        keys: str | None = None,
        *,
        max_buffer_rows: int = 10_000,
        n_features: int | None = None,
    ) -> None:
        self.component_factories = list(component_factories)
        self.num_components = _positive_integer(num_components, "num_components")
        self.keys = keys
        self.max_buffer_rows = _positive_integer(max_buffer_rows, "max_buffer_rows")
        self.n_features = None if n_features is None else _positive_integer(n_features, "n_features")

    def make(self) -> GatedMixtureAccumulator:
        """Create an empty gated-mixture accumulator."""
        return GatedMixtureAccumulator(
            [f.make() for f in self.component_factories],
            self.num_components,
            keys=self.keys,
            max_buffer_rows=self.max_buffer_rows,
            n_features=self.n_features,
        )


class GatedMixtureEstimator(ParameterEstimator):
    """M-step: refit each expert from its responsibility-weighted stats, refit the gate on ``(z, resp)``."""

    def __init__(
        self,
        component_estimators: Sequence[ParameterEstimator],
        gate: Any,
        gate_steps: int = 200,
        gate_lr: float = 0.1,
        gate_tol: float = 1.0e-8,
        max_buffer_rows: int = 10_000,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.component_estimators = list(component_estimators)
        self.num_components = len(self.component_estimators)
        if self.num_components < 2:
            raise ValueError("GatedMixtureEstimator requires at least two component estimators.")
        try:
            gate_classes = operator.index(gate.n_classes)
        except (AttributeError, TypeError) as exc:
            raise TypeError("gate.n_classes must be an exact integer.") from exc
        if gate_classes != self.num_components:
            raise ValueError("gate.n_classes must match the number of component estimators.")
        if hasattr(gate, "n_features"):
            _positive_integer(gate.n_features, "gate.n_features")
        self.gate = gate
        self.gate_steps = _positive_integer(gate_steps, "gate_steps")
        self.gate_lr = _positive_finite(gate_lr, "gate_lr")
        self.gate_tol = _nonnegative_finite(gate_tol, "gate_tol")
        self.max_buffer_rows = _positive_integer(max_buffer_rows, "max_buffer_rows")
        self.name = name
        self.keys = keys
        self.last_gate_fit_receipt: GateOptimizationReceipt | None = None
        self.last_gate_buffer_receipt: GateBufferReceipt | None = None

    def accumulator_factory(self) -> GatedMixtureAccumulatorFactory:
        """Return a factory for gated-mixture sufficient-statistic accumulators."""
        return GatedMixtureAccumulatorFactory(
            [e.accumulator_factory() for e in self.component_estimators],
            self.num_components,
            keys=self.keys,
            max_buffer_rows=self.max_buffer_rows,
            n_features=getattr(self.gate, "n_features", None),
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[Any, ...]) -> GatedMixtureDistribution:
        """Estimate experts from responsibility-weighted stats and refit the gate."""
        comp_stats, z, r, buffer_receipt, component_counts = _unpack_gated_statistics(
            suff_stat,
            num_components=self.num_components,
            max_buffer_rows=self.max_buffer_rows,
        )
        validate_effective_sample_mass(
            nobs,
            component_counts.sum(),
            label="gated-mixture effective sample",
            allow_unassigned=True,
        )
        self.last_gate_buffer_receipt = buffer_receipt
        components = [
            self.component_estimators[k].estimate(component_counts[k], comp_stats[k])
            for k in range(self.num_components)
        ]
        if len(z) and callable(getattr(self.gate, "fit_with_receipt", None)):
            gate, receipt = self.gate.fit_with_receipt(
                z,
                r,
                steps=self.gate_steps,
                lr=self.gate_lr,
                tol=self.gate_tol,
            )
        elif len(z) and callable(getattr(self.gate, "fit", None)):
            gate = self.gate.fit(z, r, steps=self.gate_steps, lr=self.gate_lr)
            receipt = GateOptimizationReceipt(
                False,
                self.gate_steps,
                np.nan,
                np.nan,
                np.nan,
                "external_gate_status_unavailable",
            )
        else:
            gate = self.gate
            receipt = GateOptimizationReceipt(True, 0, 0.0, 0.0, 0.0, "no_gate_rows")
        if len(z):
            _validated_gate_log_probabilities(gate, z, self.num_components)
        self.last_gate_fit_receipt = receipt
        return GatedMixtureDistribution(
            components,
            gate,
            name=self.name,
            keys=self.keys,
            gate_fit_receipt=receipt,
            gate_buffer_receipt=buffer_receipt,
        )
