"""Exponentially-modified Gaussian distributions over real values.

Observations are real-valued floats. The EMG models ``X = N(mu, sigma2) + Exp(rate=lam)`` -- a Gaussian
    convolved with a positive-shifting exponential, giving a right-skewed density. Its stable log-density is

        log f(x) = log(lam/2) - 0.5*u^2 + log(erfcx(z)),

    where ``u = (x - mu)/sigma`` and ``z = (lam*sigma - u)/sqrt(2)`` and ``sigma = sqrt(sigma2)``.
    Using ``log_erfcx`` keeps the right tail (large ``z``) from underflowing.

The MLE has no closed form (the score equations couple mu, sigma2, lam), so the estimator uses a
method-of-moments fit from the accumulated first three moments, which is consistent and the usual
practical choice for the EMG.
"""

import math
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import log_erfcx

_MIN_EMG_PARAM = 1.0e-12


class ExponentiallyModifiedGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Exponentially-modified Gaussian: ``X = N(mu, sigma2) + Exp(rate=lam)``."""

    def __init__(
        self,
        mu: float,
        sigma2: float,
        lam: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an EMG with Gaussian mean ``mu``, Gaussian variance ``sigma2`` and exponential rate ``lam``.

        Args:
            mu (float): Real-valued mean of the Gaussian component.
            sigma2 (float): Positive variance of the Gaussian component.
            lam (float): Positive rate of the exponential component (its mean is ``1/lam``).
            name (Optional[str]): Optional object name.
            keys (Optional[str]): Optional parameter key.

        Attributes:
            mu (float): Gaussian mean.
            sigma2 (float): Gaussian variance.
            sigma (float): Gaussian standard deviation.
            lam (float): Exponential rate.

        """
        if not np.isfinite(mu):
            raise ValueError("ExponentiallyModifiedGaussianDistribution requires finite mu.")
        if sigma2 <= 0.0 or not np.isfinite(sigma2):
            raise ValueError("ExponentiallyModifiedGaussianDistribution requires finite sigma2 > 0.")
        if lam <= 0.0 or not np.isfinite(lam):
            raise ValueError("ExponentiallyModifiedGaussianDistribution requires finite lam > 0.")
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.sigma = float(math.sqrt(sigma2))
        self.lam = float(lam)
        self.name = name
        self.keys = keys
        self.log_lam_half = math.log(self.lam / 2.0)

    def __str__(self) -> str:
        """Return a constructor-style representation of the exponentially modified Gaussian distribution."""
        return "ExponentiallyModifiedGaussianDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.sigma2),
            repr(self.lam),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Density of the EMG at observation ``x`` (see ``log_density``)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Stable log-density of the EMG at ``x``.

        ``log f(x) = log(lam/2) - 0.5*u^2 + log_erfcx(z)`` with ``u = (x - mu)/sigma`` and
        ``z = (lam*sigma - u)/sqrt(2)``.
        """
        u = (x - self.mu) / self.sigma
        z = (self.lam * self.sigma - u) / math.sqrt(2.0)
        return self.log_lam_half - 0.5 * u * u + float(log_erfcx(z))

    def seq_ld_lambda(self) -> list[Callable]:
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized EMG log-density at sequence-encoded input ``x``."""
        xx = np.asarray(x, dtype=np.float64)
        u = (xx - self.mu) / self.sigma
        z = (self.lam * self.sigma - u) / math.sqrt(2.0)
        return self.log_lam_half - 0.5 * u * u + np.asarray(log_erfcx(z), dtype=np.float64)

    # --- compute-engine backend (numpy + torch/GPU), SCORING only: the Welford running-moment
    # accumulator stays host-side (a bit-correct E-step fallback), so torch accelerates likelihood
    # evaluation while estimation statistics remain exactly the legacy path. ---
    @classmethod
    def compute_capabilities(cls):
        """Declare NumPy/Torch scoring capabilities for EMG log-density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @staticmethod
    def _engine_log_erfcx(z: Any, engine: Any) -> Any:
        """Stable ``log erfcx(z)`` on engine ops: direct for ``z >= 0``; ``z^2 + log(2 - erfc(-z))``
        below (``erfc(-z) = exp(-z^2) erfcx(-z)`` is in (0, 1] there), so neither side overflows."""
        zp = engine.maximum(z, engine.asarray(0.0))
        zn = engine.maximum(-z, engine.asarray(0.0))
        pos = engine.log(engine.erfcx(zp))
        neg = z * z + engine.log(2.0 - engine.exp(-zn * zn) * engine.erfcx(zn))
        return engine.where(z >= 0.0, pos, neg)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized EMG log-density for encoded data."""
        xx = engine.asarray(x)
        u = (xx - self.mu) / self.sigma
        z = (self.lam * self.sigma - u) / math.sqrt(2.0)
        return self.log_lam_half - 0.5 * u * u + self._engine_log_erfcx(z, engine)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact, via scipy's exponnorm)."""
        from scipy.stats import exponnorm

        return float(exponnorm.cdf(x, 1.0 / (self.lam * self.sigma), loc=self.mu, scale=self.sigma))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``."""
        from scipy.stats import exponnorm

        return float(exponnorm.ppf(q, 1.0 / (self.lam * self.sigma), loc=self.mu, scale=self.sigma))

    def sampler(self, seed: int | None = None) -> "ExponentiallyModifiedGaussianSampler":
        """Return an ExponentiallyModifiedGaussianSampler for this distribution."""
        return ExponentiallyModifiedGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ExponentiallyModifiedGaussianEstimator":
        """Return an ExponentiallyModifiedGaussianEstimator (method-of-moments)."""
        return ExponentiallyModifiedGaussianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ExponentiallyModifiedGaussianDataEncoder":
        """Return the encoder for exponentially modified Gaussian observations."""
        return ExponentiallyModifiedGaussianDataEncoder()


class ExponentiallyModifiedGaussianSampler(DistributionSampler):
    """Sample an EMG by adding independent Gaussian and exponential draws."""

    def __init__(self, dist: ExponentiallyModifiedGaussianDistribution, seed: int | None = None) -> None:
        """Sampler: draw a Gaussian and add an independent Exponential.

        Args:
            dist (ExponentiallyModifiedGaussianDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random sampler.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw ``size`` iid EMG samples (a single float if ``size`` is None)."""
        d = self.dist
        normal = self.rng.normal(loc=d.mu, scale=d.sigma, size=size)
        expo = self.rng.exponential(scale=1.0 / d.lam, size=size)
        return normal + expo


class ExponentiallyModifiedGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted central moments for method-of-moments EMG estimation."""

    def __init__(self, keys: str | None = None, name: str | None = None) -> None:
        """Accumulate the first three (weighted) central moments needed for the MoM fit.

        Stored as ``(count, mean, M2, M3)`` with ``M2 = sum_i w_i (x_i-mean)^2`` and
        ``M3 = sum_i w_i (x_i-mean)^3``. Centering each batch before squaring/cubing and
        merging via the Pébay parallel-moment recurrence avoids the ``E[x^2]-E[x]^2``
        cancellation that destroyed the skewness (and hence ``tau``) for large-``|mean|`` data.
        EMG is a host-only leaf, so this representation change has no engine-swap implications.

        Attributes:
            count (float): sum_i w_i.
            mean (float): weighted mean.
            m2 (float): weighted second central moment sum.
            m3 (float): weighted third central moment sum.
        """
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0
        self.m3 = 0.0
        self.keys = keys
        self.name = name

    def _merge(self, c_b: float, mean_b: float, m2_b: float, m3_b: float) -> None:
        """Merge a second weighted central-moment batch (parallel/Pébay form)."""
        c_a, mean_a, m2_a, m3_a = self.count, self.mean, self.m2, self.m3
        count = c_a + c_b
        if count <= 0.0:
            return
        delta = mean_b - mean_a
        self.mean = mean_a + delta * (c_b / count)
        self.m2 = m2_a + m2_b + delta * delta * (c_a * c_b / count)
        self.m3 = (
            m3_a
            + m3_b
            + delta**3 * (c_a * c_b * (c_a - c_b) / (count * count))
            + 3.0 * delta * (c_a * m2_b - c_b * m2_a) / count
        )
        self.count = count

    def update(self, x: float, weight: float, estimate: Optional["ExponentiallyModifiedGaussianDistribution"]) -> None:
        """Update weighted central moments from one observation."""
        self._merge(float(weight), float(x), 0.0, 0.0)

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize weighted central moments from one observation."""
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize weighted central moments from encoded observations."""
        self.seq_update(x, weights, None)

    def seq_update(
        self, x: np.ndarray, weights: np.ndarray, estimate: ExponentiallyModifiedGaussianDistribution | None
    ) -> None:
        """Update weighted central moments from encoded observations."""
        xx = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        c_b = float(ww.sum())
        if c_b <= 0.0:
            return
        mean_b = float(np.dot(xx, ww) / c_b)
        dx = xx - mean_b  # center before squaring/cubing -> no cancellation
        m2_b = float(np.dot(ww, dx * dx))
        m3_b = float(np.dot(ww, dx * dx * dx))
        self._merge(c_b, mean_b, m2_b, m3_b)

    def combine(self, suff_stat: tuple[float, float, float, float]) -> "ExponentiallyModifiedGaussianAccumulator":
        """Merge another accumulator's central-moment summary."""
        self._merge(float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2]), float(suff_stat[3]))
        return self

    def value(self) -> tuple[float, float, float, float]:
        """Return count, mean, second central moment, and third central moment."""
        return self.count, self.mean, self.m2, self.m3

    def from_value(self, x: tuple[float, float, float, float]) -> "ExponentiallyModifiedGaussianAccumulator":
        """Restore count and central-moment state from ``value`` output."""
        self.count, self.mean, self.m2, self.m3 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        return self

    def scale(self, c: float) -> "ExponentiallyModifiedGaussianAccumulator":
        """Scale the weighted moment summary by a constant."""
        # Scaling all weights by c multiplies the total weight and the central-moment sums by c;
        # the mean is invariant.
        self.count *= c
        self.m2 *= c
        self.m3 *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "ExponentiallyModifiedGaussianDataEncoder":
        """Return the encoder compatible with EMG moment statistics."""
        return ExponentiallyModifiedGaussianDataEncoder()


class ExponentiallyModifiedGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Create EMG central-moment accumulators."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.keys = keys
        self.name = name

    def make(self) -> "ExponentiallyModifiedGaussianAccumulator":
        """Create an empty EMG accumulator."""
        return ExponentiallyModifiedGaussianAccumulator(name=self.name, keys=self.keys)


class ExponentiallyModifiedGaussianEstimator(ParameterEstimator):
    """Estimate EMG parameters from weighted central moments."""

    def __init__(
        self,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Method-of-moments EMG estimator.

        The EMG MLE is iterative with no closed form (the score equations couple ``mu``, ``sigma2``
        and ``lam``). This estimator uses the consistent method-of-moments fit from the first three
        central moments: with sample variance ``v`` and skewness ``g``, set the exponential mean
        ``tau = 1/lam = (g*v^{3/2}/2)^{1/3}``, then ``mu = mean - tau`` and ``sigma2 = v - tau^2``.
        Degenerate moments (non-positive skew or ``tau^2 >= v``) fall back to a small positive
        exponential component so the result stays a valid EMG.
        """
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "ExponentiallyModifiedGaussianAccumulatorFactory":
        """Return a factory for EMG central-moment accumulators."""
        return ExponentiallyModifiedGaussianAccumulatorFactory(self.name, self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float]
    ) -> "ExponentiallyModifiedGaussianDistribution":
        """Estimate an EMG from the accumulated (count, mean, M2, M3) via method of moments."""
        n, m1, sum_m2, sum_m3 = suff_stat

        if n <= 0.0:
            return ExponentiallyModifiedGaussianDistribution(0.0, 1.0, 1.0, name=self.name, keys=self.keys)

        # ``sum_m2``/``sum_m3`` are weighted central moments (see the accumulator), so the sample
        # variance and third central moment are read off directly without E[x^2]-E[x]^2 cancellation.
        var = sum_m2 / n
        if var <= _MIN_EMG_PARAM or not np.isfinite(var):
            var = _MIN_EMG_PARAM

        # third central moment and skewness
        mu3 = sum_m3 / n
        skew = mu3 / (var**1.5)

        if skew > _MIN_EMG_PARAM:
            tau = (0.5 * skew) ** (1.0 / 3.0) * math.sqrt(var)
        else:
            # near-symmetric data: give the exponential component a small positive share
            tau = math.sqrt(var) * 1e-3

        # keep a strictly positive Gaussian variance
        sigma2 = var - tau * tau
        if sigma2 <= _MIN_EMG_PARAM or not np.isfinite(sigma2):
            sigma2 = max(var * 0.5, _MIN_EMG_PARAM)
            tau = math.sqrt(max(var - sigma2, _MIN_EMG_PARAM))

        if tau < _MIN_EMG_PARAM:
            tau = _MIN_EMG_PARAM
        lam = 1.0 / tau
        mu = m1 - tau

        return ExponentiallyModifiedGaussianDistribution(mu, sigma2, lam, name=self.name, keys=self.keys)


class ExponentiallyModifiedGaussianDataEncoder(DataSequenceEncoder):
    """Encode sequences of iid EMG observations (data type float)."""

    def __str__(self) -> str:
        return "ExponentiallyModifiedGaussianDataEncoder"

    def __eq__(self, other) -> bool:
        return isinstance(other, ExponentiallyModifiedGaussianDataEncoder)

    def seq_encode(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Validate and encode EMG observations as a finite float array."""
        rv = np.asarray(x, dtype=float)
        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise ValueError("ExponentiallyModifiedGaussianDistribution requires support x in (-inf, inf).")
        return rv
