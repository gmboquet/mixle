"""Probabilistic PCA latent-factor distributions and estimators.

Observations are length-``d`` real vectors represented as ``np.ndarray`` or a
compatible sequence of floats.

Probabilistic PCA is the latent linear-Gaussian model

    z ~ N(0, I_q),    x | z ~ N(W z + mu, sigma2 * I_d),

so marginally ``x ~ N(mu, C)`` with the structured covariance ``C = W W^T + sigma2 * I_d`` (a rank-q
factor structure plus isotropic noise). It is the probabilistic foundation of PCA / factor analysis and
gives a generative model, a likelihood, and a posterior over the latent factors
``E[z | x] = M^{-1} W^T (x - mu)`` (the low-dimensional embedding, exposed by ``transform``), with
``M = W^T W + sigma2 * I_q``.

Scoring uses the Woodbury identity, so the d-by-d inverse and log-determinant are obtained from a small
q-by-q solve (``C^{-1} = (I_d - W M^{-1} W^T) / sigma2`` and ``log|C| = (d-q) log sigma2 + log|M|``); the
reduction is engine-neutral, so the model scores on NumPy and Torch. Estimation is the **closed-form**
maximum-likelihood solution of Tipping & Bishop (1999): ``sigma2`` is the mean of the discarded
eigenvalues of the sample covariance and ``W`` is built from its top-q eigenpairs -- no EM iteration.
"""

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
from mixle.stats.latent.effective_sample import (
    validate_effective_sample_mass,
    validated_observation_weights,
    validated_statistic_tuple,
)

_MIN_SIGMA2 = 1.0e-12
_LOG_2PI = float(np.log(2.0 * np.pi))


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive.")
    return result


def _finite_nonnegative_weight(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _event_vector(value: Any, dim: int, label: str) -> np.ndarray:
    try:
        event = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric vector.") from exc
    if event.shape != (dim,):
        raise ValueError(f"{label} must have shape ({dim},), got {event.shape}.")
    if np.any(~np.isfinite(event)):
        raise ValueError(f"{label} must contain finite values.")
    return event


def _event_matrix(value: Any, dim: int, label: str) -> np.ndarray:
    try:
        events = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric matrix.") from exc
    if events.ndim != 2 or events.shape[1] != dim:
        raise ValueError(f"{label} must have shape (n, {dim}), got {events.shape}.")
    if np.any(~np.isfinite(events)):
        raise ValueError(f"{label} must contain finite values.")
    return events


class ProbabilisticPCADistribution(SequenceEncodableProbabilityDistribution):
    """Probabilistic PCA: x ~ N(mu, W W^T + sigma2 I) with q latent factors."""

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for PPCA scoring."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    def __init__(
        self,
        w: Sequence[Sequence[float]] | np.ndarray,
        mu: Sequence[float] | np.ndarray,
        sigma2: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a probabilistic PCA distribution.

        Args:
            w: ``d`` by ``q`` factor-loading matrix.
            mu: Length-``d`` mean vector.
            sigma2: Positive isotropic noise variance.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            w: Factor-loading matrix.
            mu: Mean vector.
            sigma2: Isotropic noise variance.
            dim: Observation dimension.
            latent_dim: Number of latent factors.
            inv_covar: Cached covariance inverse via Woodbury.
            log_det: Cached covariance log-determinant.

        """
        try:
            loadings = np.asarray(w, dtype=np.float64)
            mean = np.asarray(mu, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("ProbabilisticPCADistribution parameters must be numeric arrays.") from exc
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("ProbabilisticPCADistribution requires a non-empty one-dimensional mu.")
        if loadings.ndim != 2 or loadings.shape[0] != len(mean) or loadings.shape[1] == 0:
            raise ValueError("ProbabilisticPCADistribution requires W of shape (d, q) matching mu of length d.")
        if np.any(~np.isfinite(loadings)) or np.any(~np.isfinite(mean)):
            raise ValueError("ProbabilisticPCADistribution parameters must be finite.")
        if isinstance(sigma2, (bool, np.bool_)) or not isinstance(
            sigma2,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("ProbabilisticPCADistribution sigma2 must be a real number.")
        noise_variance = float(sigma2)
        if noise_variance <= 0.0 or not np.isfinite(noise_variance):
            raise ValueError("ProbabilisticPCADistribution requires sigma2 > 0.")
        self._w = loadings.copy()
        self._mu = mean.copy()
        self._sigma2 = noise_variance
        self.dim = loadings.shape[0]
        self.latent_dim = loadings.shape[1]

        q = self.latent_dim
        m = self._w.T @ self._w + self._sigma2 * np.eye(q)  # (q, q)
        self._m_inv = np.linalg.inv(m)
        # Woodbury: C^{-1} = (I_d - W M^{-1} W^T) / sigma2
        self._inv_covar = (np.eye(self.dim) - self._w @ self._m_inv @ self._w.T) / self._sigma2
        sign, log_det_m = np.linalg.slogdet(m)
        if sign != 1.0 or not np.isfinite(log_det_m):
            raise ValueError("ProbabilisticPCADistribution covariance factorization is not positive definite.")
        self.log_det = float((self.dim - q) * np.log(self._sigma2) + log_det_m)
        self.name = name
        self.keys = keys

    @property
    def w(self) -> np.ndarray:
        """Return an owned snapshot of the factor loadings."""
        return self._w.copy()

    @property
    def mu(self) -> np.ndarray:
        """Return an owned snapshot of the mean."""
        return self._mu.copy()

    @property
    def sigma2(self) -> float:
        """Return the immutable isotropic noise variance."""
        return self._sigma2

    @property
    def inv_covar(self) -> np.ndarray:
        """Return an owned snapshot of the cached covariance inverse."""
        return self._inv_covar.copy()

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        return "ProbabilisticPCADistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr([[float(v) for v in row] for row in self.w]),
            repr([float(v) for v in self.mu]),
            repr(self.sigma2),
            repr(self.name),
            repr(self.keys),
        )

    def transform(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the posterior mean of the latent factors E[z | x] = M^{-1} W^T (x - mu)."""
        raw = np.asarray(x)
        if raw.ndim == 1:
            diff = _event_vector(x, self.dim, "ProbabilisticPCA transform event") - self._mu
            return self._m_inv @ (self._w.T @ diff)
        if raw.ndim == 2:
            diff = _event_matrix(x, self.dim, "ProbabilisticPCA transform events") - self._mu
            return self._m_inv @ (self._w.T @ diff.T)
        raise ValueError(f"ProbabilisticPCA transform input must have shape ({self.dim},) or (n, {self.dim}).")

    def density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the probability density at a single observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the log-density at a single observation."""
        diff = _event_vector(x, self.dim, "ProbabilisticPCA event") - self._mu
        mahal = float(diff @ self._inv_covar @ diff)
        return -0.5 * (self.dim * _LOG_2PI + self.log_det + mahal)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        diff = _event_matrix(x, self.dim, "ProbabilisticPCA encoded events") - self._mu
        mahal = np.einsum("ij,jk,ik->i", diff, self._inv_covar, diff)
        return -0.5 * (self.dim * _LOG_2PI + self.log_det + mahal)

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        shape = tuple(getattr(x, "shape", ()))
        if len(shape) != 2 or shape[1] != self.dim:
            raise ValueError(f"ProbabilisticPCA backend events must have shape (n, {self.dim}), got {shape}.")
        events = _event_matrix(engine.to_numpy(x), self.dim, "ProbabilisticPCA backend events")
        diff = engine.asarray(events) - engine.asarray(self._mu)
        mahal = engine.sum(engine.matmul(diff, engine.asarray(self._inv_covar)) * diff, axis=-1)
        const = engine.asarray(self.dim * _LOG_2PI + self.log_det)
        return engine.asarray(-0.5) * (const + mahal)

    def sampler(self, seed: int | None = None) -> "ProbabilisticPCASampler":
        """Return a sampler for drawing observations from this distribution."""
        return ProbabilisticPCASampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ProbabilisticPCAEstimator":
        """Return a closed-form ML estimator with the latent dimension fixed at this model's q."""
        return ProbabilisticPCAEstimator(latent_dim=self.latent_dim, dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ProbabilisticPCADataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return ProbabilisticPCADataEncoder(self.dim)


class ProbabilisticPCASampler(DistributionSampler):
    """Draw iid observations x = mu + W z + sigma * eps from a PPCA model."""

    def __init__(self, dist: ProbabilisticPCADistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw ``size`` iid vectors (shape (d,) when size is None, else (size, d))."""
        sz = 1 if size is None else size
        d, q = self.dist.dim, self.dist.latent_dim
        z = self.rng.standard_normal(size=(sz, q))
        noise = np.sqrt(self.dist._sigma2) * self.rng.standard_normal(size=(sz, d))
        rv = self.dist._mu[None, :] + z @ self.dist._w.T + noise
        return rv[0] if size is None else rv


class ProbabilisticPCAAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted count, mean, and second-moment matrix (the PPCA sufficient statistics)."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        self.dim = None if dim is None else _positive_integer(dim, "ProbabilisticPCA accumulator dimension")
        self.count = 0.0
        self.sum = np.zeros(self.dim) if self.dim is not None else None
        self.sum2 = np.zeros((self.dim, self.dim)) if self.dim is not None else None
        self.keys = keys

    def _ensure_dim(self, d: int) -> None:
        d = _positive_integer(d, "ProbabilisticPCA event dimension")
        if self.dim is None:
            self.dim = d
        elif self.dim != d:
            raise ValueError(f"ProbabilisticPCA event dimension {d} does not match accumulator dimension {self.dim}.")
        if self.sum is None:
            self.sum = np.zeros(self.dim)
            self.sum2 = np.zeros((self.dim, self.dim))

    def update(self, x: np.ndarray, weight: float, estimate: ProbabilisticPCADistribution | None) -> None:
        """Accumulate weighted count, sum, and second moment for one vector."""
        if self.dim is None:
            raw = np.asarray(x)
            if raw.ndim != 1 or raw.size == 0:
                raise ValueError("ProbabilisticPCA accumulator event must be a non-empty vector.")
            self._ensure_dim(len(raw))
        xx = _event_vector(x, self.dim, "ProbabilisticPCA accumulator event")
        weight = _finite_nonnegative_weight(weight, "ProbabilisticPCA accumulator weight")
        self.count += weight
        self.sum += weight * xx
        self.sum2 += weight * np.outer(xx, xx)

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: ProbabilisticPCADistribution | None) -> None:
        """Accumulate weighted count, sum, and second moment for encoded vectors."""
        raw = np.asarray(x)
        if raw.ndim != 2 or raw.shape[1] == 0:
            raise ValueError("ProbabilisticPCA accumulator events must be a non-empty-width matrix.")
        self._ensure_dim(raw.shape[1])
        events = _event_matrix(x, self.dim, "ProbabilisticPCA accumulator events")
        weights = validated_observation_weights(
            weights,
            len(events),
            "ProbabilisticPCA accumulator weights",
        )
        self.count += float(np.sum(weights, dtype=np.float64))
        self.sum += events.T @ weights
        self.sum2 += (events * weights[:, None]).T @ events

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray | None, np.ndarray | None]) -> "ProbabilisticPCAAccumulator":
        """Merge serialized PPCA sufficient statistics into this accumulator."""
        incoming = ProbabilisticPCAAccumulator()
        incoming.from_value(validated_statistic_tuple(suff_stat, 3, "ProbabilisticPCA sufficient statistics"))
        if incoming.sum is not None:
            self._ensure_dim(incoming.dim)
            self.sum += incoming.sum
            self.sum2 += incoming.sum2
        self.count += incoming.count
        return self

    def value(self) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        """Return the total weight, weighted sum, and weighted second moment."""
        return (
            self.count,
            None if self.sum is None else self.sum.copy(),
            None if self.sum2 is None else self.sum2.copy(),
        )

    def from_value(self, x: tuple[float, np.ndarray | None, np.ndarray | None]) -> "ProbabilisticPCAAccumulator":
        """Restore the accumulator from serialized PPCA sufficient statistics."""
        count, first_value, second_value = validated_statistic_tuple(x, 3, "ProbabilisticPCA sufficient statistics")
        self.count = _finite_nonnegative_weight(count, "ProbabilisticPCA sufficient-statistic count")
        if first_value is None or second_value is None:
            if first_value is not None or second_value is not None:
                raise ValueError("ProbabilisticPCA sufficient-statistic moments must both be present or absent.")
            if self.count != 0.0:
                raise ValueError("positive-count ProbabilisticPCA statistics require moments")
            self.sum = self.sum2 = None
            self.dim = None
            return self
        first = np.asarray(first_value, dtype=np.float64)
        if first.ndim != 1 or first.size == 0 or np.any(~np.isfinite(first)):
            raise ValueError("ProbabilisticPCA sufficient-statistic sum must be a finite non-empty vector.")
        second = np.asarray(second_value, dtype=np.float64)
        if second.shape != (len(first), len(first)) or np.any(~np.isfinite(second)):
            raise ValueError("ProbabilisticPCA second moment has invalid shape or values.")
        self.sum = first.copy()
        self.sum2 = second.copy()
        self.dim = len(first)
        if self.count == 0.0 and (np.any(self.sum != 0.0) or np.any(self.sum2 != 0.0)):
            raise ValueError("zero-count ProbabilisticPCA moments must be exact zero arrays")
        return self

    def acc_to_encoder(self) -> "ProbabilisticPCADataEncoder":
        """Return an encoder compatible with PPCA vector observations."""
        return ProbabilisticPCADataEncoder(self.dim)


class ProbabilisticPCAAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ProbabilisticPCAAccumulator."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> ProbabilisticPCAAccumulator:
        """Create an empty PPCA accumulator."""
        return ProbabilisticPCAAccumulator(dim=self.dim, keys=self.keys)


class ProbabilisticPCAEstimator(ParameterEstimator):
    """Closed-form maximum-likelihood estimator for PPCA (Tipping & Bishop eigen-solution)."""

    def __init__(
        self,
        latent_dim: int,
        dim: int | None = None,
        min_sigma2: float = _MIN_SIGMA2,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.latent_dim = _positive_integer(latent_dim, "ProbabilisticPCA latent_dim")
        self.dim = None if dim is None else _positive_integer(dim, "ProbabilisticPCA dimension")
        if isinstance(min_sigma2, (bool, np.bool_)) or not isinstance(
            min_sigma2,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("ProbabilisticPCA min_sigma2 must be a real number.")
        self.min_sigma2 = float(min_sigma2)
        if not np.isfinite(self.min_sigma2) or self.min_sigma2 <= 0.0:
            raise ValueError("ProbabilisticPCA min_sigma2 must be finite and positive.")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ProbabilisticPCAAccumulatorFactory:
        """Return a factory for PPCA sufficient-statistic accumulators."""
        return ProbabilisticPCAAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, np.ndarray | None, np.ndarray | None]
    ) -> ProbabilisticPCADistribution:
        """Estimate PPCA parameters from weighted first and second moments."""
        count, s, s2 = validated_statistic_tuple(suff_stat, 3, "ProbabilisticPCA sufficient statistics")
        count = _finite_nonnegative_weight(count, "ProbabilisticPCA sufficient-statistic count")
        validate_effective_sample_mass(nobs, count, label="ProbabilisticPCA effective sample")
        if count == 0.0:
            if s is None and s2 is None:
                d = self.dim if self.dim is not None else self.latent_dim
            elif s is None or s2 is None:
                raise ValueError("zero-count ProbabilisticPCA moments must both be present or absent.")
            else:
                empty_sum = np.asarray(s, dtype=np.float64)
                empty_second = np.asarray(s2, dtype=np.float64)
                if (
                    empty_sum.ndim != 1
                    or empty_sum.size == 0
                    or empty_second.shape != (len(empty_sum), len(empty_sum))
                    or np.any(empty_sum != 0.0)
                    or np.any(empty_second != 0.0)
                ):
                    raise ValueError("zero-count ProbabilisticPCA moments must be exact zero arrays.")
                d = len(empty_sum)
                if self.dim is not None and d != self.dim:
                    raise ValueError(f"ProbabilisticPCA statistics have dimension {d}, expected {self.dim}.")
            return ProbabilisticPCADistribution(
                np.zeros((d, self.latent_dim)), np.zeros(d), 1.0, name=self.name, keys=self.keys
            )
        if s is None or s2 is None:
            raise ValueError("positive-count ProbabilisticPCA statistics require both moments.")

        first = np.asarray(s, dtype=np.float64)
        if first.ndim != 1 or first.size == 0 or np.any(~np.isfinite(first)):
            raise ValueError("ProbabilisticPCA sufficient-statistic sum must be a finite non-empty vector.")
        d = len(first)
        if self.dim is not None and d != self.dim:
            raise ValueError(f"ProbabilisticPCA statistics have dimension {d}, expected {self.dim}.")
        second = np.asarray(s2, dtype=np.float64)
        if second.shape != (d, d) or np.any(~np.isfinite(second)):
            raise ValueError("ProbabilisticPCA second moment has invalid shape or values.")
        q = min(self.latent_dim, d)
        mu = first / count
        cov = second / count - np.outer(mu, mu)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]

        # sigma2 = mean of the discarded eigenvalues (the isotropic residual variance).
        sigma2 = float(np.mean(eigvals[q:])) if q < d else 0.0
        sigma2 = max(sigma2, self.min_sigma2)
        # W = U_q (Lambda_q - sigma2 I)^{1/2}; padded with zero columns if q < latent_dim.
        scale = np.sqrt(np.clip(eigvals[:q] - sigma2, 0.0, None))
        w = eigvecs[:, :q] * scale[None, :]
        if q < self.latent_dim:
            w = np.hstack([w, np.zeros((d, self.latent_dim - q))])
        return ProbabilisticPCADistribution(w, mu, sigma2, name=self.name, keys=self.keys)


class ProbabilisticPCADataEncoder(DataSequenceEncoder):
    """Encode a sequence of length-d real vectors into an (n, d) float array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else _positive_integer(dim, "ProbabilisticPCA encoder dimension")

    def __str__(self) -> str:
        return "ProbabilisticPCADataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProbabilisticPCADataEncoder) and other.dim == self.dim

    def seq_encode(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Validate and encode observations as a two-dimensional float array."""
        if self.dim is not None and isinstance(x, Sequence) and len(x) == 0:
            return np.empty((0, self.dim), dtype=np.float64)
        try:
            events = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("ProbabilisticPCA observations must be a numeric matrix.") from exc
        if events.ndim != 2 or events.shape[1] == 0:
            expected = "(n, d)" if self.dim is None else f"(n, {self.dim})"
            raise ValueError(f"ProbabilisticPCA observations must have shape {expected}, got {events.shape}.")
        if self.dim is not None and events.shape[1] != self.dim:
            raise ValueError(f"ProbabilisticPCA observations must have shape (n, {self.dim}), got {events.shape}.")
        if np.any(~np.isfinite(events)):
            raise ValueError("ProbabilisticPCA observations must contain finite values.")
        return events.copy()
