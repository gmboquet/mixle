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
from mixle.stats.univariate.continuous._observation_contracts import (
    SquaredPowerSumTrack,
    anchored_pooled_variance,
    consistent_anchored_triple,
    finite_observations,
    scored_observation,
    warn_uncorrectable_raw_moments,
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

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep the cached ``_log_const`` and ``_m_over_omega`` tied to ``m`` and ``omega``.

        BOTH derived constants must refresh, not just the normalizer: ``log_density`` reads
        ``_m_over_omega`` in its exponent and ``cdf`` uses it as the gamma-integral scale, so
        refreshing only ``_log_const`` still scored the new normalizer against the OLD shape.

        The constant is computed once in ``__init__`` and read by ``log_density``, so a later
        assignment used to leave it stale and the scorer kept reporting the *previous*
        parameter's density with no error at all (MXR-080-1192).

        Recompute rather than validate: callers legitimately install out-of-domain or non-finite
        parameters -- deserialized legacy states and NaN-propagation checks both do -- so a value
        outside the domain yields a NaN constant that propagates honestly instead of rejecting a
        state the library is expected to be able to hold.
        """
        object.__setattr__(self, name, value)
        if name not in ("m", "omega"):
            return
        try:
            object.__setattr__(
                self,
                "_log_const",
                math.log(2.0) + self.m * math.log(self.m) - gammaln(self.m) - self.m * math.log(self.omega),
            )
            object.__setattr__(self, "_m_over_omega", self.m / self.omega)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, AttributeError, FloatingPointError):
            # AttributeError covers __init__, where the first parameter is assigned before the rest.
            object.__setattr__(self, "_log_const", float("nan"))
            object.__setattr__(self, "_m_over_omega", float("nan"))

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
        """Return the log-density at ``x``, including the exact limit at zero."""
        xv = scored_observation(x, label="Nakagami observations")
        if xv < 0.0:
            return -math.inf
        if xv == 0.0:
            return self._log_const if self.m == 0.5 else -math.inf
        return self._log_const + (2.0 * self.m - 1.0) * math.log(xv) - self._m_over_omega * xv * xv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of observations."""
        xv = np.asarray(x, dtype=np.float64)
        exponent = 2.0 * self.m - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_x = np.where(exponent == 0.0, 0.0, np.log(xv))
            out = self._log_const + exponent * log_x - self._m_over_omega * xv * xv
        return np.where(xv >= 0.0, out, -np.inf)

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
            support="nonnegative_real",
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
        """Engine-neutral Nakagami log-density with the exact limit at zero."""
        log_const = engine.log(engine.asarray(2.0)) + m * engine.log(m) - engine.gammaln(m) - m * engine.log(omega)
        exponent = 2.0 * m - 1.0
        safe_x = engine.where(x > 0.0, x, engine.asarray(1.0))
        safe_log_x = engine.log(safe_x)
        out = log_const + exponent * safe_log_x - (m / omega) * x * x
        neg_inf = engine.asarray(float("-inf"))
        at_zero = engine.where(exponent == 0.0, log_const, neg_inf)
        return engine.where(x > 0.0, out, engine.where(x == 0.0, at_zero, neg_inf))

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
        if pseudo_count is None:
            return NakagamiEstimator(name=self.name, keys=self.keys)
        # X^2 ~ Gamma(shape=m, scale=omega/m), so E[X^2] = omega and
        # E[X^4] = Var(X^2) + E[X^2]^2 = omega^2/m + omega^2 = omega^2*(1+m)/m -- the raw
        # power-sum space estimate() accumulates in (s2, s4) -- so pseudo_count can blend a prior
        # pseudo-sample toward them (mirrors GumbelEstimator / WeibullEstimator's suff_stat pattern).
        s2_0 = self.omega
        s4_0 = self.omega * self.omega * (1.0 + self.m) / self.m
        return NakagamiEstimator(pseudo_count=pseudo_count, suff_stat=(s2_0, s4_0), name=self.name, keys=self.keys)

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


class NakagamiAccumulator(SquaredPowerSumTrack, SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted power sums ``(count, sum x^2, sum x^4)`` for the moment fit.

    The whole accumulator is :class:`SquaredPowerSumTrack` -- shape, validation and the
    conditioning-gated shift-anchored track on ``y = x**2``, which is the quantity this family's
    M-step differences. It is shared with the Rician family, whose accumulator was character-for-
    character the same; the duplicate-body gate caught the pair and de-duplicating it is what keeps
    a fix landing on one from missing the other.
    """

    _OBSERVATION_LABEL = "Nakagami"

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
    """Method-of-moments estimator: ``omega = E[X^2]``, ``m = E[X^2]^2 / Var[X^2]`` (clamped m >= 1/2).

    ``Var[X^2]`` comes from the shift-anchored moment track whenever the accumulated statistics
    carry that payload (see :class:`NakagamiAccumulator`). The shape is the reciprocal of the
    relative spread of ``X^2``, so the raw ``E[X^4] - E[X^2]^2`` form loses about ``eps * m``:
    measured against exact rational arithmetic, 2.0e-10 relative at m=1e6, 2.6e-8 at 1e8 and 2.6e-6
    at 1e10, now 2.0e-10, 4.7e-16 and 1.0e-13. With a plain raw tuple the historical path is used
    bit-identically, and ``estimate`` warns rather than returning a shape it cannot stand behind.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        m_min: float = 0.5,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.m_min = m_min
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> NakagamiAccumulatorFactory:
        """Return an accumulator factory for Nakagami power-sum statistics."""
        return NakagamiAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> NakagamiDistribution:
        """Estimate shape and spread from weighted second and fourth moments."""
        count, s2, s4 = float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2])
        # The anchored payload describes the RAW data only, so read it before any prior blend.
        anchored = consistent_anchored_triple(suff_stat, s2, count)
        raw_count = count
        prior_mean: float | None = None
        prior_variance: float | None = None
        pc = self.pseudo_count
        if pc is not None and self.suff_stat is not None:
            prior_mean, s4_0 = self.suff_stat
            prior_variance = max(s4_0 - prior_mean * prior_mean, 0.0)
            s2 += pc * prior_mean
            s4 += pc * s4_0
            count += pc
        if count <= 0.0:
            return NakagamiDistribution(1.0, 1.0, name=self.name, keys=self.keys)
        omega = s2 / count
        if anchored is None:
            # Historical raw path, bit-identical: statistics the conditioning gate never needed to
            # anchor, or ones that arrived already reduced and can no longer be corrected.
            warn_uncorrectable_raw_moments(s2, s4, count, family="Nakagami")
            var_x2 = s4 / count - omega * omega
        else:
            var_x2, _ = anchored_pooled_variance(
                anchored[0], anchored[1], anchored[2], raw_count, omega, pc, prior_mean, prior_variance
            )
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
        return finite_observations(x, label="Nakagami observations", minimum=0.0)
