"""Wrapped normal distribution -- a Gaussian wrapped around the circle.

Wrapping ``X ~ N(mu, sigma^2)`` onto angles gives the wrapped normal ``WN(mu, sigma^2)``, the most
common symmetric circular law and the maximum-entropy distribution for a fixed circular mean and
resultant. Its density is the periodic sum of Gaussians,

    f(theta; mu, sigma^2) = (1 / (sigma sqrt(2 pi))) sum_k exp(-(wrap(theta - mu) + 2 pi k)^2 / (2 sigma^2)),

which is uniform as ``sigma -> inf`` and concentrates at ``mu`` as ``sigma -> 0``. Its first
trigonometric moment is ``rho e^{i mu}`` with ``rho = exp(-sigma^2 / 2)``, so the mean direction and
concentration are estimated in closed form from the mean resultant: ``mu = atan2(sum sin, sum cos)`` and
``sigma^2 = -2 log Rbar``. It samples exactly by wrapping ``N(mu, sigma^2)``.

The density is summed over enough wraps (``K`` set from ``sigma``) that every retained Gaussian is
machine-negligible past the window, which keeps the truncated sum strictly positive (unlike the
trigonometric-series form, which can dip negative when truncated).

Reference: Mardia & Jupp, *Directional Statistics* (2000), sec. 3.5.7.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.directional._circular import (
    validated_angle,
    validated_angles,
    validated_circular_statistics,
    validated_sample_size,
    validated_weight,
    validated_weights,
)

_TWO_PI = 2.0 * math.pi


class WrappedNormalFitError(RuntimeError):
    """Raised when circular moments have no finite wrapped-normal fit."""


def _wrap(theta: Any) -> Any:
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(theta) + math.pi) % _TWO_PI - math.pi


class WrappedNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Wrapped normal distribution with mean direction ``mu`` and wrapped variance ``sigma2`` > 0."""

    def __init__(self, mu: float, sigma2: float, name: str | None = None, keys: str | None = None) -> None:
        checked_mu = validated_angle(mu, "wrapped-normal mean direction")
        checked_sigma2 = validated_angle(sigma2, "wrapped-normal variance")
        if checked_sigma2 <= 0.0:
            raise ValueError("WrappedNormalDistribution requires sigma2 > 0.")
        if checked_sigma2 > 1.0e6:
            raise ValueError(
                "WrappedNormalDistribution sigma2 exceeds the supported finite branch range"
            )
        self.mu = float(
            math.atan2(math.sin(checked_mu), math.cos(checked_mu))
        )
        self.sigma2 = checked_sigma2
        self.name = name
        self.keys = keys
        self._sigma = math.sqrt(self.sigma2)
        self._log_norm = math.log(self._sigma * math.sqrt(_TWO_PI))
        # enough wraps that the outermost retained Gaussian is ~6 sigma out of the window
        self._k = np.arange(-(self.K), self.K + 1, dtype=np.float64)

    @property
    def K(self) -> int:
        """Number of wraps summed on each side of the window."""
        return max(3, int(math.ceil(6.0 * math.sqrt(self.sigma2) / _TWO_PI)) + 1)

    def __str__(self) -> str:
        return "WrappedNormalDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.sigma2),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single angle (radians)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single angle (radians)."""
        d = _wrap(validated_angle(x) - self.mu) + _TWO_PI * self._k
        return float(logsumexp(-0.5 * d * d / self.sigma2) - self._log_norm)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of angles."""
        d0 = _wrap(validated_angles(x) - self.mu)  # (n,)
        d = d0[:, None] + _TWO_PI * self._k[None, :]  # (n, 2K+1)
        return logsumexp(-0.5 * d * d / self.sigma2, axis=1) - self._log_norm

    # --- compute-engine backend (numpy + torch/GPU), leaf path only. The wrap-branch count K is a
    # per-instance truncation, so mixtures score component-by-component (each with its exact K) through
    # the generic kernel rather than a stacked trio. Sufficient statistics use the engines' trig tier. ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated wrapped-normal kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for wrapped normal distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="wrapped_normal",
            distribution_type=cls,
            parameters=(ParameterSpec("mu"), ParameterSpec("sigma2", constraint="positive")),
            statistics=(StatisticSpec("sum_cos"), StatisticSpec("sum_sin"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row circular moments ``(sum_cos, sum_sin, count)`` — uses the engine trig tier."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        checked = x if is_symbolic_payload(x) else validated_angles(x)
        theta = engine.asarray(checked)
        return engine.cos(theta), engine.sin(theta), theta * 0.0 + engine.asarray(1.0)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density: the (2K+1)-branch wrapped logsumexp on engine ops."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        checked = x if is_symbolic_payload(x) else validated_angles(x)
        theta = engine.asarray(checked)
        raw = theta - self.mu
        d0 = raw - _TWO_PI * engine.floor((raw + math.pi) / _TWO_PI)  # wrap to (-pi, pi]
        d = d0[:, None] + engine.asarray(_TWO_PI * self._k)[None, :]
        return engine.logsumexp(-0.5 * d * d / self.sigma2, axis=1) - self._log_norm

    def sampler(self, seed: int | None = None) -> "WrappedNormalSampler":
        """Return a sampler that wraps ``N(mu, sigma2)`` onto the circle."""
        return WrappedNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WrappedNormalEstimator":
        """Return a closed-form (mean-resultant) estimator for ``mu`` and ``sigma2``."""
        if pseudo_count is not None:
            raise ValueError("wrapped-normal pseudo-count regularization is not implemented")
        return WrappedNormalEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WrappedNormalDataEncoder":
        """Return the data encoder used by this distribution (the angle itself)."""
        return WrappedNormalDataEncoder()


class WrappedNormalSampler(DistributionSampler):
    """Draw angles by wrapping ``N(mu, sigma2)`` around the circle."""

    def __init__(self, dist: WrappedNormalDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one angle or an array of iid angles."""
        d = self.dist
        n = 1 if size is None else validated_sample_size(size)
        theta = _wrap(d.mu + d._sigma * self.rng.standard_normal(n))
        return float(theta[0]) if size is None else theta


class WrappedNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted circular resultant ``(sum cos, sum sin, count)``."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: WrappedNormalDistribution | None) -> None:
        """Accumulate one weighted circular resultant contribution."""
        theta = validated_angle(x)
        checked_weight = validated_weight(weight)
        self.sum_cos += checked_weight * math.cos(theta)
        self.sum_sin += checked_weight * math.sin(theta)
        self.count += checked_weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one angle."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate circular resultant statistics from encoded angles."""
        theta = validated_angles(x)
        w = validated_weights(weights, len(theta))
        self.sum_cos += float(np.dot(np.cos(theta), w))
        self.sum_sin += float(np.dot(np.sin(theta), w))
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded angles."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "WrappedNormalAccumulator":
        """Merge another wrapped-normal sufficient-statistic tuple."""
        count, sum_cos, sum_sin = validated_circular_statistics(
            suff_stat,
            count_index=2,
        )
        combined = (
            self.sum_cos + sum_cos,
            self.sum_sin + sum_sin,
            self.count + count,
        )
        self.count, self.sum_cos, self.sum_sin = (
            validated_circular_statistics(combined, count_index=2)
        )
        return self

    def value(self) -> tuple[float, float, float]:
        """Return cosine sum, sine sum, and total weight."""
        return self.sum_cos, self.sum_sin, self.count

    def from_value(self, x: tuple[float, float, float]) -> "WrappedNormalAccumulator":
        """Replace accumulator contents from circular-resultant statistics."""
        self.count, self.sum_cos, self.sum_sin = (
            validated_circular_statistics(x, count_index=2)
        )
        return self

    def scale(self, c: float) -> "WrappedNormalAccumulator":
        """Scale linear wrapped-normal sufficient statistics."""
        checked_scale = validated_weight(c)
        self.sum_cos *= checked_scale
        self.sum_sin *= checked_scale
        self.count *= checked_scale
        return self

    def acc_to_encoder(self) -> "WrappedNormalDataEncoder":
        """Return the encoder used by this accumulator."""
        return WrappedNormalDataEncoder()


class WrappedNormalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WrappedNormalAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> WrappedNormalAccumulator:
        """Create a fresh wrapped-normal accumulator."""
        return WrappedNormalAccumulator(name=self.name, keys=self.keys)


class WrappedNormalEstimator(ParameterEstimator):
    """Estimate ``mu`` and ``sigma2`` from the mean resultant (``rho = exp(-sigma2/2)``)."""

    def __init__(
        self,
        sigma2_max: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if sigma2_max is not None:
            raise ValueError(
                "wrapped-normal boundary clipping is not a valid fit contract"
            )
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WrappedNormalAccumulatorFactory:
        """Return an accumulator factory for circular-resultant statistics."""
        return WrappedNormalAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> WrappedNormalDistribution:
        """Estimate mean direction and variance from the mean resultant."""
        count, sum_cos, sum_sin = validated_circular_statistics(
            suff_stat,
            count_index=2,
        )
        if count == 0.0:
            raise WrappedNormalFitError(
                "wrapped-normal fitting requires positive observation weight"
            )
        mu = math.atan2(sum_sin, sum_cos)
        rbar = math.hypot(sum_cos, sum_sin) / count
        if rbar <= 1.0e-12:
            raise WrappedNormalFitError(
                "wrapped-normal zero resultant has no finite variance or mean direction"
            )
        if rbar >= 1.0 - 1.0e-12:
            raise WrappedNormalFitError(
                "wrapped-normal unit resultant has no positive-variance fit"
            )
        sigma2 = -2.0 * math.log(rbar)
        result = WrappedNormalDistribution(
            mu,
            sigma2,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "circular-resultant",
            "identifiable_direction": True,
            "resultant_length": rbar,
            "repairs": (),
        }
        return result


class WrappedNormalDataEncoder(DataSequenceEncoder):
    """Encode angles as a float array (wrapped to (-pi, pi])."""

    def __str__(self) -> str:
        return "WrappedNormalDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WrappedNormalDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode angles after wrapping them to ``(-pi, pi]``."""
        return _wrap(validated_angles(x))

    def row_count(self, x: np.ndarray) -> int:
        """Return the encoded row count after finite-angle validation."""
        return len(validated_angles(x))
