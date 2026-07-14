"""Projected normal distribution -- a flexible circular law from projecting a 2-D Gaussian.

The angle ``theta = atan2(Z2, Z1)`` of a bivariate normal ``Z ~ N(mu, I_2)`` follows the (isotropic)
projected normal ``PN(mu)`` on the circle. With ``mu = (mu_x, mu_y)`` and ``a(theta) = mu_x cos theta
+ mu_y sin theta = mu . u(theta)``, its density is

    f(theta; mu) = (1 / 2 pi) exp(-||mu||^2 / 2) [1 + a Phi(a) / phi(a)],

where ``phi``/``Phi`` are the standard normal pdf/cdf. It is uniform at ``mu = 0`` and concentrates in
the direction of ``mu`` as ``||mu||`` grows; unlike the von Mises it can be asymmetric/peaked depending
on ``mu``. It samples exactly (draw ``N(mu, I_2)``, take the angle). Parameters are fit by EM with the
latent radius (Nunez-Antonio & Gutierrez-Pena 2005): given the current ``mu``, ``E[r | theta]`` has a
closed form and the M-step is ``mu = mean(E[r | theta] u(theta))``.

The ratio ``a Phi(a) / phi(a)`` is evaluated stably for all ``a`` via the scaled complementary error
function: ``Phi(a)/phi(a) = sqrt(pi/2) erfcx(-a/sqrt2)``, so ``M(a) := sqrt(pi/2) erfcx(-a/sqrt2)`` and
``a Phi(a)/phi(a) = a M(a)``.

References:
  - Mardia & Jupp, *Directional Statistics* (2000), sec. 3.5.6 (projected/offset normal).
  - Nunez-Antonio & Gutierrez-Pena, "A Bayesian analysis of directional data using the projected normal
    distribution", *J. Applied Statistics* (2005).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import erfcx

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_HALF_PI = math.sqrt(math.pi / 2.0)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _mills(a: Any) -> Any:
    """``M(a) = Phi(a)/phi(a) = sqrt(pi/2) * erfcx(-a/sqrt2)`` -- stable for all ``a``."""
    return _SQRT_HALF_PI * erfcx(-np.asarray(a, dtype=np.float64) * _INV_SQRT2)


class ProjectedNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Isotropic projected normal ``PN(mu)`` on the circle, ``mu = (mu_x, mu_y)``."""

    def __init__(self, mu_x: float, mu_y: float, name: str | None = None, keys: str | None = None) -> None:
        self.mu_x = float(mu_x)
        self.mu_y = float(mu_y)
        self.name = name
        self.keys = keys
        self._half_sq = 0.5 * (self.mu_x * self.mu_x + self.mu_y * self.mu_y)

    def __str__(self) -> str:
        return "ProjectedNormalDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu_x),
            repr(self.mu_y),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single angle (radians)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single angle (radians)."""
        a = self.mu_x * math.cos(float(x)) + self.mu_y * math.sin(float(x))
        return -_LOG_2PI - self._half_sq + math.log1p(a * float(_mills(a)))

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density for sequence-encoded ``(cos, sin)`` observations."""
        cos_t, sin_t = x
        a = self.mu_x * cos_t + self.mu_y * sin_t
        return -_LOG_2PI - self._half_sq + np.log1p(a * _mills(a))

    # --- compute-engine backend (numpy + torch/GPU). Density AND the latent-radius E-step lower to
    # engine ops via erfcx from the special tier: M(a) = sqrt(pi/2) erfcx(-a/sqrt2),
    # E[r|theta] = (a + (1+a^2) M) / (1 + a M) — the estimate-dependent stats come from the params dict. ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated projected-normal kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for projected normal distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="projected_normal",
            distribution_type=cls,
            parameters=(ParameterSpec("mu_x"), ParameterSpec("mu_y")),
            statistics=(StatisticSpec("sum_x"), StatisticSpec("sum_y"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def _engine_mills(a: Any, engine: Any) -> Any:
        return _SQRT_HALF_PI * engine.erfcx(-a * _INV_SQRT2)

    @classmethod
    def backend_legacy_sufficient_statistics(cls, x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row E-step stats ``(E[r] cos, E[r] sin, 1)`` under the CURRENT estimate's parameters."""
        cos_t = engine.asarray(x[0])
        sin_t = engine.asarray(x[1])
        a = engine.asarray(params["mu_x"]) * cos_t + engine.asarray(params["mu_y"]) * sin_t
        m = cls._engine_mills(a, engine)
        r = (a + (1.0 + a * a) * m) / (1.0 + a * m)
        return r * cos_t, r * sin_t, cos_t * 0.0 + engine.asarray(1.0)

    @classmethod
    def backend_log_density_from_params(cls, cos_t: Any, sin_t: Any, mu_x: Any, mu_y: Any, engine: Any) -> Any:
        """Engine-neutral projected-normal log-density from ``(cos, sin)`` and the mean vector."""
        a = mu_x * cos_t + mu_y * sin_t
        half_sq = 0.5 * (mu_x * mu_x + mu_y * mu_y)
        return -_LOG_2PI - half_sq + engine.log(1.0 + a * cls._engine_mills(a, engine))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded ``(cos, sin)`` data."""
        return self.backend_log_density_from_params(
            engine.asarray(x[0]),
            engine.asarray(x[1]),
            engine.asarray(self.mu_x),
            engine.asarray(self.mu_y),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["ProjectedNormalDistribution"], engine: Any) -> dict[str, Any]:
        """Stacked projected-normal mean vectors for a homogeneous mixture kernel."""
        return {
            "mu_x": engine.asarray([d.mu_x for d in dists]),
            "mu_y": engine.asarray([d.mu_y for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of projected-normal log densities."""
        cos_t = engine.asarray(x[0])
        sin_t = engine.asarray(x[1])
        return cls.backend_log_density_from_params(
            cos_t[:, None], sin_t[:, None], params["mu_x"][None, :], params["mu_y"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Stacked E-step resultants ``(sum r cos, sum r sin, count)`` under per-component parameters."""
        cos_t = engine.asarray(x[0])
        sin_t = engine.asarray(x[1])
        ww = engine.asarray(weights)
        a = cos_t[:, None] * params["mu_x"][None, :] + sin_t[:, None] * params["mu_y"][None, :]
        m = cls._engine_mills(a, engine)
        r = (a + (1.0 + a * a) * m) / (1.0 + a * m)
        return (
            engine.sum(ww * r * cos_t[:, None], axis=0),
            engine.sum(ww * r * sin_t[:, None], axis=0),
            engine.sum(ww, axis=0),
        )

    def sampler(self, seed: int | None = None) -> "ProjectedNormalSampler":
        """Return a sampler that draws ``N(mu, I_2)`` and returns the angle."""
        return ProjectedNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ProjectedNormalEstimator":
        """Return an EM (latent-radius) estimator for ``mu``."""
        return ProjectedNormalEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ProjectedNormalDataEncoder":
        """Return the data encoder used by this distribution (cos/sin of the angle)."""
        return ProjectedNormalDataEncoder()


class ProjectedNormalSampler(DistributionSampler):
    """Draw angles as ``atan2(Z2, Z1)`` for ``Z ~ N(mu, I_2)``."""

    def __init__(self, dist: ProjectedNormalDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one angle or an array of iid angles."""
        d = self.dist
        n = 1 if size is None else int(size)
        z1 = d.mu_x + self.rng.standard_normal(n)
        z2 = d.mu_y + self.rng.standard_normal(n)
        theta = np.arctan2(z2, z1)
        return float(theta[0]) if size is None else theta


def _expected_radius(a: np.ndarray) -> np.ndarray:
    """``E[r | theta] = (a + (1 + a^2) M(a)) / (1 + a M(a))`` with ``M(a) = Phi(a)/phi(a)``."""
    m = _mills(a)
    return (a + (1.0 + a * a) * m) / (1.0 + a * m)


class ProjectedNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the ``E[r|theta]``-weighted resultant ``(sum r*cos, sum r*sin, count)`` (EM E-step)."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def _radius_for(self, cos_t: np.ndarray, sin_t: np.ndarray, estimate: Any) -> np.ndarray:
        if estimate is None:  # first pass: E[r] ~ 1 -> resultant points in the data mean direction
            return np.ones_like(cos_t)
        a = estimate.mu_x * cos_t + estimate.mu_y * sin_t
        return _expected_radius(a)

    def update(self, x: float, weight: float, estimate: ProjectedNormalDistribution | None) -> None:
        """Accumulate one EM radius-weighted directional contribution."""
        cos_t, sin_t = math.cos(float(x)), math.sin(float(x))
        r = float(self._radius_for(np.array([cos_t]), np.array([sin_t]), estimate)[0])
        self.sum_x += weight * r * cos_t
        self.sum_y += weight * r * sin_t
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one angle."""
        self.update(x, weight, None)

    def seq_update(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: Any) -> None:
        """Accumulate EM radius-weighted direction statistics from encoded angles."""
        cos_t, sin_t = x
        w = np.asarray(weights, dtype=np.float64)
        r = self._radius_for(cos_t, sin_t, estimate)
        self.sum_x += float(np.dot(w * r, cos_t))
        self.sum_y += float(np.dot(w * r, sin_t))
        self.count += float(w.sum())

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded angles."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "ProjectedNormalAccumulator":
        """Merge another projected-normal sufficient-statistic tuple."""
        self.sum_x += suff_stat[0]
        self.sum_y += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return radius-weighted x/y sums and total weight."""
        return self.sum_x, self.sum_y, self.count

    def from_value(self, x: tuple[float, float, float]) -> "ProjectedNormalAccumulator":
        """Replace accumulator contents from radius-weighted statistics."""
        self.sum_x, self.sum_y, self.count = float(x[0]), float(x[1]), float(x[2])
        return self

    def acc_to_encoder(self) -> "ProjectedNormalDataEncoder":
        """Return the encoder used by this accumulator."""
        return ProjectedNormalDataEncoder()


class ProjectedNormalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ProjectedNormalAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> ProjectedNormalAccumulator:
        """Create a fresh projected-normal accumulator."""
        return ProjectedNormalAccumulator(name=self.name, keys=self.keys)


class ProjectedNormalEstimator(ParameterEstimator):
    """EM estimator: ``mu = mean(E[r|theta] u(theta))`` (one M-step per accumulated E-step)."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ProjectedNormalAccumulatorFactory:
        """Return an accumulator factory for projected-normal EM statistics."""
        return ProjectedNormalAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> ProjectedNormalDistribution:
        """Estimate the projected-normal mean vector from EM sufficient statistics."""
        sum_x, sum_y, count = suff_stat
        if count <= 0.0:
            return ProjectedNormalDistribution(0.0, 0.0, name=self.name, keys=self.keys)
        return ProjectedNormalDistribution(sum_x / count, sum_y / count, name=self.name, keys=self.keys)


class ProjectedNormalDataEncoder(DataSequenceEncoder):
    """Encode angles as their cosine and sine."""

    def __str__(self) -> str:
        return "ProjectedNormalDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProjectedNormalDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode angles as cosine and sine arrays."""
        theta = np.asarray(x, dtype=np.float64)
        return np.cos(theta), np.sin(theta)
