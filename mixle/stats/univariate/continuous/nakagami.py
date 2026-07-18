"""Nakagami distribution -- the amplitude/envelope law of Nakagami-m fading.

A positive-support family for signal envelopes (wireless/radar/sonar fading, also reliability and
hydrology). With shape ``m >= 1/2`` and spread ``omega = E[X^2] > 0``,

    f(x; m, omega) = 2 m^m / (Gamma(m) omega^m) * x^(2m-1) * exp(-m x^2 / omega),  x > 0,

so ``X^2 ~ Gamma(m, omega/m)``; ``m = 1/2`` is the half-normal and ``m = 1`` the Rayleigh. The CDF is the
regularized lower incomplete gamma, it samples exactly via a Gamma draw, and it has a clean closed-form
method-of-moments fit: ``omega = E[X^2]`` and ``m = E[X^2]^2 / Var[X^2]``.

Reference: Nakagami, "The m-distribution -- a general formula of intensity distribution of rapid
fading", in *Statistical Methods in Radio Wave Propagation* (1960).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gammainc, gammaincinv, gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class NakagamiDistribution(SequenceEncodableProbabilityDistribution):
    """Nakagami distribution with shape ``m >= 1/2`` and spread ``omega = E[X^2] > 0``."""

    def __init__(self, m: float, omega: float, name: str | None = None, keys: str | None = None) -> None:
        if m < 0.5 or not np.isfinite(m):
            raise ValueError("NakagamiDistribution requires finite m >= 1/2.")
        if omega <= 0.0 or not np.isfinite(omega):
            raise ValueError("NakagamiDistribution requires finite omega > 0.")
        self.m = float(m)
        self.omega = float(omega)
        self.name = name
        self.keys = keys
        self._log_const = math.log(2.0) + self.m * math.log(self.m) - gammaln(self.m) - self.m * math.log(self.omega)
        self._m_over_omega = self.m / self.omega

    def __str__(self) -> str:
        return "NakagamiDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.m),
            repr(self.omega),
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
        return self._log_const + (2.0 * self.m - 1.0) * math.log(xv) - self._m_over_omega * xv * xv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of observations."""
        xv = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self._log_const + (2.0 * self.m - 1.0) * np.log(xv) - self._m_over_omega * xv * xv
        return np.where(xv > 0.0, out, -np.inf)

    # --- compute-engine backend (numpy + torch/GPU): scoring + sufficient statistics in engine ops ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Nakagami kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Nakagami distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="nakagami",
            distribution_type=cls,
            parameters=(ParameterSpec("m", constraint="positive"), ParameterSpec("omega", constraint="positive")),
            statistics=(StatisticSpec("count"), StatisticSpec("sum_x2"), StatisticSpec("sum_x4")),
            support="positive",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row Nakagami power sums in accumulator order ``(count, sum x^2, sum x^4)``."""
        xx = engine.asarray(x)
        x2 = xx * xx
        return xx * 0.0 + engine.asarray(1.0), x2, x2 * x2

    @staticmethod
    def backend_log_density_from_params(x: Any, m: Any, omega: Any, engine: Any) -> Any:
        """Engine-neutral Nakagami log-density from explicit parameters (``-inf`` for ``x <= 0``)."""
        log_const = engine.log(engine.asarray(2.0)) + m * engine.log(m) - engine.gammaln(m) - m * engine.log(omega)
        out = log_const + (2.0 * m - 1.0) * engine.log(x) - (m / omega) * x * x
        return engine.where(x > 0.0, out, engine.asarray(float("-inf")))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.m), engine.asarray(self.omega), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["NakagamiDistribution"], engine: Any) -> dict[str, Any]:
        """Stacked Nakagami parameters for a homogeneous mixture kernel."""
        return {"m": engine.asarray([d.m for d in dists]), "omega": engine.asarray([d.omega for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Nakagami log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params["m"][None, :], params["omega"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Stacked Nakagami power sums ``(count, sum x^2, sum x^4)`` using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        x2 = xx * xx
        return engine.sum(ww, axis=0), engine.sum(ww * x2[:, None], axis=0), engine.sum(ww * (x2 * x2)[:, None], axis=0)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) = P(m, m x^2 / omega) (0 for x <= 0)."""
        xv = float(x)
        return float(gammainc(self.m, self._m_over_omega * xv * xv)) if xv > 0.0 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        return float(math.sqrt(self.omega * gammaincinv(self.m, float(q)) / self.m))

    def mean(self) -> float:
        """Mean (Gamma(m+1/2)/Gamma(m)) sqrt(omega/m)."""
        return float(math.exp(gammaln(self.m + 0.5) - gammaln(self.m)) * math.sqrt(self.omega / self.m))

    def variance(self) -> float:
        """Variance omega - mean^2 (since E[X^2] = omega)."""
        mu = self.mean()
        return float(self.omega - mu * mu)

    def entropy(self) -> float:
        """Differential entropy m + lgamma(m) + (1/2 - m) psi(m) + (1/2) log(omega/m) - log(2).

        Derived via the entropy of a monotone transform: ``X = sqrt(Y)`` with
        ``Y = X^2 ~ Gamma(m, omega/m)`` (Nakagami, 'The m-distribution', 1960), so
        ``h(X) = h(Y) - log(2) - E[log X] = h(Y) - log(2) - (1/2)(E[log Y])`` and both ``h(Y)``
        and ``E[log Y]`` are standard Gamma-distribution identities. Reduces exactly to the
        Rayleigh entropy at ``m = 1``.
        """
        from scipy.special import digamma

        return float(
            self.m
            + gammaln(self.m)
            + (0.5 - self.m) * digamma(self.m)
            + 0.5 * math.log(self.omega / self.m)
            - math.log(2.0)
        )

    def sampler(self, seed: int | None = None) -> "NakagamiSampler":
        """Return a sampler (``X = sqrt(Gamma(m, omega/m))``)."""
        return NakagamiSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "NakagamiEstimator":
        """Return a closed-form method-of-moments estimator."""
        return NakagamiEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "NakagamiDataEncoder":
        """Return the data encoder used by this distribution (the raw value)."""
        return NakagamiDataEncoder()


class NakagamiSampler(DistributionSampler):
    """Draw ``X = sqrt(G)`` with ``G ~ Gamma(shape=m, scale=omega/m)`` (so ``X^2 ~ Gamma(m, omega/m)``)."""

    def __init__(self, dist: NakagamiDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        d = self.dist
        n = 1 if size is None else int(size)
        x = np.sqrt(self.rng.gamma(d.m, d.omega / d.m, size=n))
        return float(x[0]) if size is None else x


class NakagamiAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted power sums ``(count, sum x^2, sum x^4)`` for the moment fit."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.s2 = 0.0
        self.s4 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: NakagamiDistribution | None) -> None:
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

    def combine(self, suff_stat: tuple[float, float, float]) -> "NakagamiAccumulator":
        """Merge another Nakagami sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.s2 += suff_stat[1]
        self.s4 += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, second power sum, and fourth power sum."""
        return self.count, self.s2, self.s4

    def from_value(self, x: tuple[float, float, float]) -> "NakagamiAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.s2, self.s4 = float(x[0]), float(x[1]), float(x[2])
        return self

    def acc_to_encoder(self) -> "NakagamiDataEncoder":
        """Return the encoder used by this accumulator."""
        return NakagamiDataEncoder()


class NakagamiAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for NakagamiAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> NakagamiAccumulator:
        """Create a fresh Nakagami accumulator."""
        return NakagamiAccumulator(name=self.name, keys=self.keys)


class NakagamiEstimator(ParameterEstimator):
    """Method-of-moments estimator: ``omega = E[X^2]``, ``m = E[X^2]^2 / Var[X^2]`` (clamped m >= 1/2)."""

    def __init__(self, m_min: float = 0.5, name: str | None = None, keys: str | None = None) -> None:
        self.m_min = m_min
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> NakagamiAccumulatorFactory:
        """Return an accumulator factory for Nakagami power-sum statistics."""
        return NakagamiAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> NakagamiDistribution:
        """Estimate shape and spread from weighted second and fourth moments."""
        count, s2, s4 = suff_stat
        if count <= 0.0:
            return NakagamiDistribution(1.0, 1.0, name=self.name, keys=self.keys)
        omega = s2 / count
        var_x2 = s4 / count - omega * omega
        m = (omega * omega / var_x2) if var_x2 > 0.0 else 1.0e6
        m = max(m, self.m_min)
        return NakagamiDistribution(m, omega, name=self.name, keys=self.keys)


class NakagamiDataEncoder(DataSequenceEncoder):
    """Encode observations as a float array."""

    def __str__(self) -> str:
        return "NakagamiDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NakagamiDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return np.asarray(x, dtype=np.float64)
