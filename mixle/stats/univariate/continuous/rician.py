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
from mixle.stats.univariate.continuous._observation_contracts import (
    SquaredPowerSumTrack,
    anchored_pooled_variance,
    consistent_anchored_triple,
    finite_observations,
    masked_stacked_fourth_power_sums,
    scored_observation,
    warn_uncorrectable_raw_moments,
)

# ``scipy.special.ive(0, z)`` returns NaN for ``z`` above roughly 1e10 (measured on scipy 1.17.1:
# finite at 1e9, NaN at 1e10; the sibling entry point ``i0e`` is exact out to 1e300, which is why
# only this scorer was affected), and the Rician's Bessel argument is ``x*nu/sigma^2``,
# which passes 1e10 at an ordinary ``nu/sigma`` of about 1e5. Every log-density then came back NaN
# and ``optimize`` reported only "fused EM did not produce a finite objective from its non-finite
# initial model" -- the same opaque internal error the Gumbel family used to raise, on data that is
# genuinely Rician (large ``nu/sigma`` is the near-Gaussian limit, not degenerate input).
#
# Above the crossover the asymptotic expansion is used instead. ``I0(z) ~ e^z/sqrt(2 pi z) *
# (1 + 1/(8z) + 9/(128 z^2) + 225/(3072 z^3) + ...)``, so the first omitted term is ~``0.11/z^4``;
# at the crossover below that is ~1e-21, well under a double's resolution, and the scaled Bessel is
# still exact there, so the two branches agree to the last ulp across the seam.
_I0E_ASYMPTOTIC_FROM = 1.0e5
_LOG_2PI = math.log(2.0 * math.pi)


def _log_i0e(z: Any) -> Any:
    """Return ``log(I0(z) * exp(-z))`` for ``z >= 0``, finite at every representable magnitude."""
    zz = np.asarray(z, dtype=np.float64)
    small = np.minimum(zz, _I0E_ASYMPTOTIC_FROM)  # keeps ive NaN-free in the unused branch
    with np.errstate(divide="ignore", invalid="ignore"):
        near = np.log(ive(0, small))
    big = np.maximum(zz, _I0E_ASYMPTOTIC_FROM)
    inv = 1.0 / big
    far = -0.5 * (_LOG_2PI + np.log(big)) + np.log1p(inv * (0.125 + inv * (0.0703125 + inv * 0.0732421875)))
    out = np.where(zz < _I0E_ASYMPTOTIC_FROM, near, far)
    return out if out.ndim else float(out)


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
        xv = scored_observation(x, label="Rician observations")
        if xv <= 0.0:
            return -math.inf
        z = xv * self.nu / self._sig2
        # I0(z) = ive(0, z) * exp(z), so log I0(z) = log ive(0, z) + z
        return math.log(xv) - self._log_sig2 - (xv * xv + self.nu * self.nu) / (2.0 * self._sig2) + _log_i0e(z) + z

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of observations."""
        xv = np.asarray(x, dtype=np.float64)
        z = xv * self.nu / self._sig2
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(xv) - self._log_sig2 - (xv * xv + self.nu * self.nu) / (2.0 * self._sig2) + _log_i0e(z) + z
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
        # This path needs NO asymptotic branch, unlike the NumPy scorer above, and the difference is
        # a scipy entry-point quirk rather than a design choice: measured on scipy 1.17.1, ``ive(0, z)``
        # returns NaN for z >= 1e10 while ``i0e(z)`` -- what the engines expose, and what
        # ``torch.special.i0e`` implements -- stays exact out to 1e300. The two agree with
        # :func:`_log_i0e` to within 1.6e-16 at every z tested from 1e5 to 1e300, so backend parity
        # holds across the crossover without duplicating the expansion into a kernel.
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
        return masked_stacked_fourth_power_sums(x, weights, engine)

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

    def entropy(self) -> float:
        """Differential entropy in nats, by adaptive quadrature of -f(x) log f(x) over x > 0.

        At ``nu = 0`` the Rician is exactly the Rayleigh, whose entropy is the closed form
        ``1 + log(sigma / sqrt(2)) + euler_gamma / 2``. For ``nu > 0`` the density's
        ``log I0(x nu / sigma^2)`` term has no closed-form expectation (Rice, 'Mathematical
        analysis of random noise', 1944/1945), so the integral is evaluated numerically against
        the exact log-density.
        """
        if self.nu == 0.0:
            return float(1.0 + math.log(self.sigma / math.sqrt(2.0)) + np.euler_gamma / 2.0)

        from scipy import integrate

        def integrand(x: float) -> float:
            logf = self.log_density(x)
            return 0.0 if not np.isfinite(logf) else -math.exp(logf) * logf

        val, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
        return float(val)

    def sampler(self, seed: int | None = None) -> "RicianSampler":
        """Return a sampler (the envelope of a 2-D Gaussian offset by ``nu``)."""
        return RicianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "RicianEstimator":
        """Return a closed-form method-of-moments estimator."""
        if pseudo_count is None:
            return RicianEstimator(name=self.name, keys=self.keys)
        # X^2 = Z1^2 + Z2^2 with Z1 ~ N(nu, sigma^2), Z2 ~ N(0, sigma^2) is a scaled noncentral
        # chi-squared (df=2, noncentrality nu^2/sigma^2), giving E[X^2] = nu^2 + 2*sigma^2 and
        # Var(X^2) = 4*sigma^2*(sigma^2 + nu^2) in closed form -- the raw power-sum space
        # estimate() accumulates in (s2, s4). Cross-checked against estimate()'s own inversion:
        # substituting nu^2 = m2 - 2*sig2 into E[X^4] reproduces its "disc = 2*m2^2 - m4" quadratic
        # exactly, confirming this is the same moment relationship in reverse.
        m2_0 = self.nu * self.nu + 2.0 * self._sig2
        var_x2_0 = 4.0 * self._sig2 * (self._sig2 + self.nu * self.nu)
        m4_0 = var_x2_0 + m2_0 * m2_0
        return RicianEstimator(pseudo_count=pseudo_count, suff_stat=(m2_0, m4_0), name=self.name, keys=self.keys)

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


class RicianAccumulator(SquaredPowerSumTrack, SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted power sums ``(count, sum x^2, sum x^4)`` for the moment fit.

    The whole accumulator is :class:`SquaredPowerSumTrack` -- shape, validation and the
    conditioning-gated shift-anchored track on ``y = x**2``, which is the quantity this family's
    M-step differences. It is shared with the Nakagami family, whose accumulator was character-for-
    character the same; the duplicate-body gate caught the pair and de-duplicating it is what keeps
    a fix landing on one from missing the other.
    """

    _OBSERVATION_LABEL = "Rician"

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
    """Method-of-moments estimator from ``E[X^2]`` and ``E[X^4]`` (closed-form quadratic in sigma^2).

    The quadratic is solved in its RATIONALIZED form,
    ``sigma^2 = Var(X^2) / (2 (E[X^2] + sqrt(E[X^2]^2 - Var(X^2))))``, whenever the accumulated
    statistics carry the shift-anchored payload (see :class:`RicianAccumulator`). Algebraically it
    is the same root as ``(m2 - sqrt(2 m2^2 - m4)) / 2``, but it differences raw moments once
    instead of twice, and the one difference it needs is supplied exactly by the anchored track.
    That makes the fit accurate at any ``nu/sigma``: measured against exact rational arithmetic on
    the same 2000-point samples, the historical form lost 5.4e-9 relative at ``nu/sigma`` 1e4,
    4.1e-5 at 1e6 and 44% at 1e8, and now measures 3.5e-15, 7.1e-13 and 3.5e-11. With a plain raw
    tuple -- statistics the conditioning gate never needed to anchor, or ones that arrived already
    reduced from an engine kernel or an older serialization -- the historical path is used
    bit-identically, and ``estimate`` warns rather than returning a scale it cannot stand behind.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> RicianAccumulatorFactory:
        """Return an accumulator factory for Rician power-sum statistics."""
        return RicianAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> RicianDistribution:
        """Estimate Rician noncentrality and scale from second and fourth moments."""
        count, s2, s4 = float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2])
        # The anchored payload describes the RAW data only, so read it before any prior blend.
        anchored = consistent_anchored_triple(suff_stat, s2, count)
        raw_count = count
        prior_mean: float | None = None
        prior_variance: float | None = None
        pc = self.pseudo_count
        if pc is not None and self.suff_stat is not None:
            prior_mean, m4_0 = self.suff_stat
            prior_variance = max(m4_0 - prior_mean * prior_mean, 0.0)
            s2 += pc * prior_mean
            s4 += pc * m4_0
            count += pc
        if count <= 0.0:
            return RicianDistribution(0.0, 1.0, name=self.name, keys=self.keys)
        m2 = s2 / count
        if anchored is None:
            # Historical raw path, bit-identical: statistics the conditioning gate never needed to
            # anchor, or ones that arrived already reduced and can no longer be corrected.
            warn_uncorrectable_raw_moments(s2, s4, count, family="Rician")
            m4 = s4 / count
            disc = 2.0 * m2 * m2 - m4
            sig2 = (m2 - math.sqrt(disc)) / 2.0 if disc > 0.0 else m2 / 2.0
        else:
            # ``m2 - sqrt(2 m2^2 - m4)`` differences two ``O(nu^4)`` quantities to reach an
            # ``O(sigma^2)`` answer twice over, so it loses ~``eps * (nu/sigma)^2``. Rationalizing
            # it, ``m2 - sqrt(disc) = (m4 - m2^2)/(m2 + sqrt(disc))``, replaces the outer
            # subtraction with a same-sign sum and leaves exactly one differenced quantity:
            # ``Var(X^2) = m4 - m2^2``, which the anchored track supplies directly. ``disc`` is then
            # ``m2^2 - Var(X^2)``, a subtraction of a small quantity from a large one rather than of
            # two large ones. Algebraically identical, and exact where the historical form was not.
            variance, _ = anchored_pooled_variance(
                anchored[0], anchored[1], anchored[2], raw_count, m2, pc, prior_mean, prior_variance
            )
            disc = m2 * m2 - variance
            sig2 = variance / (2.0 * (m2 + math.sqrt(disc))) if disc > 0.0 and m2 > 0.0 else m2 / 2.0
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
        return finite_observations(x, label="Rician observations", minimum=0.0)
