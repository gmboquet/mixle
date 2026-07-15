"""Rician (Rice) distribution -- the envelope of a sinusoid in additive Gaussian noise.

The amplitude ``X = sqrt((nu + sigma Z1)^2 + (sigma Z2)^2)`` of a 2-D Gaussian offset from the origin
by ``nu`` (the line-of-sight / signal component), ``Z1, Z2 ~ N(0, 1)``. Models fading envelopes with a
dominant path (wireless/radar/sonar), MRI magnitude noise, and wind speed. With ``nu >= 0`` and scale
``sigma > 0``,

    f(x; nu, sigma) = (x / sigma^2) exp(-(x^2 + nu^2) / (2 sigma^2)) I0(x nu / sigma^2),  x > 0,

where ``I0`` is the modified Bessel function (evaluated stably via the exponentially scaled ``ive``).
At ``nu = 0`` it reduces to the Rayleigh; for large ``nu/sigma`` it approaches a Gaussian. It samples
exactly from the 2-D Gaussian envelope and has a closed-form method-of-moments fit from the second and
fourth moments: ``sigma^2 = (m2 - sqrt(2 m2^2 - m4))/2`` and ``nu^2 = m2 - 2 sigma^2``.

Reference: Rice, "Mathematical analysis of random noise", *Bell System Tech. J.* (1944/1945).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import ive

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class RicianDistribution(SequenceEncodableProbabilityDistribution):
    """Rician distribution with non-centrality ``nu >= 0`` and scale ``sigma > 0``."""

    def __init__(self, nu: float, sigma: float, name: str | None = None, keys: str | None = None) -> None:
        if nu < 0.0 or not np.isfinite(nu):
            raise ValueError("RicianDistribution requires finite nu >= 0.")
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError("RicianDistribution requires finite sigma > 0.")
        self.nu = float(nu)
        self.sigma = float(sigma)
        self.name = name
        self.keys = keys
        self._sig2 = self.sigma * self.sigma
        self._log_sig2 = math.log(self._sig2)

    def __str__(self) -> str:
        return "RicianDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.nu),
            repr(self.sigma),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at ``x`` (-inf for x <= 0)."""
        xv = float(x)
        if xv <= 0.0:
            return -math.inf
        z = xv * self.nu / self._sig2
        # I0(z) = ive(0, z) * exp(z), so log I0(z) = log ive(0, z) + z
        return (
            math.log(xv) - self._log_sig2 - (xv * xv + self.nu * self.nu) / (2.0 * self._sig2) + math.log(ive(0, z)) + z
        )

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of observations."""
        xv = np.asarray(x, dtype=np.float64)
        z = xv * self.nu / self._sig2
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                np.log(xv) - self._log_sig2 - (xv * xv + self.nu * self.nu) / (2.0 * self._sig2) + np.log(ive(0, z)) + z
            )
        return np.where(xv > 0.0, out, -np.inf)

    # --- compute-engine backend (numpy + torch/GPU): scoring + sufficient statistics in engine ops.
    # log I0(z) = log i0e(z) + z (the exponentially-scaled Bessel from the engines' special tier). ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Rician kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Rician distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="rician",
            distribution_type=cls,
            parameters=(ParameterSpec("nu"), ParameterSpec("sigma", constraint="positive")),
            statistics=(StatisticSpec("count"), StatisticSpec("sum_x2"), StatisticSpec("sum_x4")),
            support="positive",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row Rician power sums in accumulator order ``(count, sum x^2, sum x^4)``."""
        xx = engine.asarray(x)
        x2 = xx * xx
        return xx * 0.0 + engine.asarray(1.0), x2, x2 * x2

    @staticmethod
    def backend_log_density_from_params(x: Any, nu: Any, sigma: Any, engine: Any) -> Any:
        """Engine-neutral Rician log-density (``-inf`` for ``x <= 0``)."""
        sig2 = sigma * sigma
        x_pos = engine.where(x > 0.0, x, engine.asarray(1.0))  # keep log NaN-free off-support
        z = x_pos * nu / sig2
        out = (
            engine.log(x_pos)
            - engine.log(sig2)
            - (x_pos * x_pos + nu * nu) / (2.0 * sig2)
            + engine.log(engine.i0e(z))
            + z
        )
        return engine.where(x > 0.0, out, engine.asarray(float("-inf")))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.nu), engine.asarray(self.sigma), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["RicianDistribution"], engine: Any) -> dict[str, Any]:
        """Stacked Rician parameters for a homogeneous mixture kernel."""
        return {"nu": engine.asarray([d.nu for d in dists]), "sigma": engine.asarray([d.sigma for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Rician log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params["nu"][None, :], params["sigma"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Stacked Rician power sums ``(count, sum x^2, sum x^4)`` using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        x2 = xx * xx
        return engine.sum(ww, axis=0), engine.sum(ww * x2[:, None], axis=0), engine.sum(ww * (x2 * x2)[:, None], axis=0)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) (Marcum-Q, via scipy rice)."""
        from scipy.stats import rice

        xv = float(x)
        return float(rice.cdf(xv, self.nu / self.sigma, scale=self.sigma)) if xv > 0.0 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) (via scipy rice)."""
        from scipy.stats import rice

        return float(rice.ppf(float(q), self.nu / self.sigma, scale=self.sigma))

    def mean(self) -> float:
        """Mean sigma sqrt(pi/2) L_{1/2}(-nu^2/(2 sigma^2)) (stable via the scaled Bessel ive)."""
        kappa = self.nu * self.nu / (2.0 * self._sig2)
        laguerre = (1.0 + kappa) * ive(0, kappa / 2.0) + kappa * ive(1, kappa / 2.0)
        return float(self.sigma * math.sqrt(math.pi / 2.0) * laguerre)

    def variance(self) -> float:
        """Variance E[X^2] - mean^2 with E[X^2] = nu^2 + 2 sigma^2."""
        mu = self.mean()
        return float(self.nu * self.nu + 2.0 * self._sig2 - mu * mu)

    def sampler(self, seed: int | None = None) -> "RicianSampler":
        """Return a sampler (the envelope of a 2-D Gaussian offset by ``nu``)."""
        return RicianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "RicianEstimator":
        """Return a closed-form method-of-moments estimator."""
        return RicianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "RicianDataEncoder":
        """Return the data encoder used by this distribution (the raw value)."""
        return RicianDataEncoder()


class RicianSampler(DistributionSampler):
    """Draw ``X = sqrt((nu + sigma Z1)^2 + (sigma Z2)^2)`` for ``Z1, Z2 ~ N(0, 1)``."""

    def __init__(self, dist: RicianDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        d = self.dist
        n = 1 if size is None else int(size)
        z1 = d.nu + d.sigma * self.rng.standard_normal(n)
        z2 = d.sigma * self.rng.standard_normal(n)
        x = np.sqrt(z1 * z1 + z2 * z2)
        return float(x[0]) if size is None else x


class RicianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted power sums ``(count, sum x^2, sum x^4)`` for the moment fit."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.s2 = 0.0
        self.s4 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: RicianDistribution | None) -> None:
        """Accumulate weighted second and fourth power sums for one observation."""
        x2 = float(x) ** 2
        self.count += weight
        self.s2 += weight * x2
        self.s4 += weight * x2 * x2

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted second and fourth power sums from encoded data."""
        x2 = np.asarray(x, dtype=np.float64) ** 2
        w = np.asarray(weights, dtype=np.float64)
        self.count += float(w.sum())
        self.s2 += float(np.dot(w, x2))
        self.s4 += float(np.dot(w, x2 * x2))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "RicianAccumulator":
        """Merge another Rician sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.s2 += suff_stat[1]
        self.s4 += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, second power sum, and fourth power sum."""
        return self.count, self.s2, self.s4

    def from_value(self, x: tuple[float, float, float]) -> "RicianAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.s2, self.s4 = float(x[0]), float(x[1]), float(x[2])
        return self

    def acc_to_encoder(self) -> "RicianDataEncoder":
        """Return the encoder used by this accumulator."""
        return RicianDataEncoder()


class RicianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for RicianAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> RicianAccumulator:
        """Create a fresh Rician accumulator."""
        return RicianAccumulator(name=self.name, keys=self.keys)


class RicianEstimator(ParameterEstimator):
    """Method-of-moments estimator from ``E[X^2]`` and ``E[X^4]`` (closed-form quadratic in sigma^2)."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> RicianAccumulatorFactory:
        """Return an accumulator factory for Rician power-sum statistics."""
        return RicianAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> RicianDistribution:
        """Estimate Rician noncentrality and scale from second and fourth moments."""
        count, s2, s4 = suff_stat
        if count <= 0.0:
            return RicianDistribution(0.0, 1.0, name=self.name, keys=self.keys)
        m2 = s2 / count
        m4 = s4 / count
        disc = 2.0 * m2 * m2 - m4
        sig2 = (m2 - math.sqrt(disc)) / 2.0 if disc > 0.0 else m2 / 2.0
        sig2 = min(max(sig2, 1.0e-12), m2 / 2.0)  # keep nu^2 = m2 - 2 sig2 >= 0
        nu = math.sqrt(max(m2 - 2.0 * sig2, 0.0))
        return RicianDistribution(nu, math.sqrt(sig2), name=self.name, keys=self.keys)


class RicianDataEncoder(DataSequenceEncoder):
    """Encode observations as a float array."""

    def __str__(self) -> str:
        return "RicianDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RicianDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return np.asarray(x, dtype=np.float64)
