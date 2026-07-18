"""Tweedie compound Poisson-Gamma distributions for ``1 < p < 2``.

Data type (float >= 0): the Tweedie exponential-dispersion model with mean ``mu``, dispersion
``phi``, and **fixed** power ``p`` in ``(1, 2)`` is the compound Poisson-Gamma law

    Y = sum_{i=1}^N G_i,   N ~ Poisson(lam),   G_i ~ Gamma(shape=a, scale=theta)  (iid),

with ``lam = mu**(2-p) / (phi*(2-p))``, ``a = (2-p)/(p-1)``, ``theta = phi*(p-1)*mu**(p-1)``. There
is a point mass ``P(Y=0) = exp(-lam)``; for ``y > 0`` the density is the (convergent) series

    f(y) = sum_{n>=1} Poisson(n; lam) * Gamma(y; shape=n*a, scale=theta),

evaluated here in log-space via a windowed log-sum-exp over ``n``. ``E[Y] = mu`` and
``Var[Y] = phi * mu**p``, so the method of moments (mean for ``mu``, Pearson for ``phi``) is exact;
``p`` is a fixed hyperparameter (the profile likelihood over ``p`` is left to the caller).


Reference: Jorgensen, *The Theory of Dispersion Models* (Chapman & Hall, 1997).
"""

import math
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
from mixle.utils.vector import gammaln

_MIN_TWEEDIE = 1.0e-12


def _tweedie_params(mu: float, phi: float, p: float) -> tuple[float, float, float]:
    """Return the compound Poisson-Gamma ``(lam, a, theta)`` for mean ``mu``, dispersion ``phi``."""
    lam = mu ** (2.0 - p) / (phi * (2.0 - p))
    a = (2.0 - p) / (p - 1.0)
    theta = phi * (p - 1.0) * mu ** (p - 1.0)
    return lam, a, theta


def _tweedie_positive_logpdf(y: np.ndarray, mu: float, phi: float, p: float) -> np.ndarray:
    """Return ``log f(y)`` for strictly-positive ``y`` (the compound Poisson-Gamma series)."""
    lam, a, theta = _tweedie_params(mu, phi, p)
    log_lam = math.log(lam)
    log_theta = math.log(theta)
    log_y = np.log(y)

    # The series terms in n are unimodal. The summand's log is
    #   A_n + (a*log y_i)*n, with A_n = n*log lam - lgamma(n+1) - lgamma(n*a) - n*a*log theta,
    # whose Stirling-approximate stationary point grows like n* ~ y_i**(2-p) / (phi*(2-p)) (i.e. ~sqrt(y)
    # in the dispersion-units that matter), NOT linearly in y. A fixed cap (the old n_max=20000) truncates
    # the peak for large lambda (large mu / small phi); instead center a window on the actual peak and
    # widen it until the boundary log-terms fall ``tol`` below the peak.
    def _log_terms(n: np.ndarray) -> np.ndarray:
        # log term[i, n] = c_i + A_n + (a*log y_i)*n
        a_n = n * log_lam - gammaln(n + 1.0) - gammaln(n * a) - n * a * log_theta
        c_i = -lam - log_y - y / theta
        return c_i[:, None] + a_n[None, :] + (a * log_y)[:, None] * n[None, :]

    # Peak location of the saddlepoint series (Dunn & Smyth): n*_i ~ y_i**(2-p) / (phi*(2-p)).
    n_peak = np.power(np.maximum(y, _MIN_TWEEDIE), 2.0 - p) / (phi * (2.0 - p))
    n_peak_max = float(np.max(n_peak)) if y.size else 1.0
    # Initial window: [1, n_hi]. Widen geometrically until both the lower boundary (n=1) and the upper
    # boundary (n=n_hi) are ``tol`` below the per-row peak across every observation.
    tol = 50.0  # in log-units; exp(-50) ~ 2e-22 relative contribution, far below float64 round-off
    n_hi = max(50.0, 2.0 * n_peak_max + 10.0 * math.sqrt(n_peak_max + 1.0) + 50.0)
    for _ in range(64):
        n = np.arange(1, int(math.ceil(n_hi)) + 1, dtype=np.float64)
        terms = _log_terms(n)
        m = np.max(terms, axis=1)
        # Boundary log-terms relative to the per-row peak; if the upper edge is still within ``tol`` of
        # the peak for any row the window is too narrow, so widen and retry.
        upper_gap = m - terms[:, -1]
        if not y.size or np.all(upper_gap >= tol):
            break
        n_hi = n_hi * 2.0
    return m + np.log(np.sum(np.exp(terms - m[:, None]), axis=1))


class TweedieDistribution(SequenceEncodableProbabilityDistribution):
    """Tweedie (compound Poisson-Gamma) distribution on ``[0, inf)`` with fixed power ``p in (1, 2)``."""

    def __init__(self, mu: float, phi: float, p: float = 1.5, name: str | None = None, keys: str | None = None) -> None:
        """Create a Tweedie with mean ``mu``, dispersion ``phi``, and fixed power ``p``.

        Args:
            mu (float): Positive mean ``E[Y]``.
            phi (float): Positive dispersion (``Var[Y] = phi * mu**p``).
            p (float): Power parameter, strictly in ``(1, 2)`` (compound Poisson-Gamma). Fixed.
            name (Optional[str]): Optional object name.
            keys (Optional[str]): Optional parameter key.
        """
        if mu <= 0.0 or not np.isfinite(mu):
            raise ValueError("TweedieDistribution requires finite mu > 0.")
        if phi <= 0.0 or not np.isfinite(phi):
            raise ValueError("TweedieDistribution requires finite phi > 0.")
        if not (1.0 < p < 2.0):
            raise ValueError("TweedieDistribution requires power p strictly in (1, 2).")
        self.mu = float(mu)
        self.phi = float(phi)
        self.p = float(p)
        self.name = name
        self.keys = keys
        self.lam, self.gamma_shape, self.gamma_scale = _tweedie_params(self.mu, self.phi, self.p)

    def __str__(self) -> str:
        """Return a constructor-style representation of the Tweedie distribution."""
        return "TweedieDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.phi),
            repr(self.p),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Probability density (or the point mass at 0) at ``x`` (see ``log_density``)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Tweedie log-density: ``log P(Y=0) = -lam`` at 0, the series for ``x > 0``, ``-inf`` for ``x < 0``."""
        try:
            xx = float(x)
        except (TypeError, ValueError):
            return -np.inf
        if not np.isfinite(xx) or xx < 0.0:
            return -np.inf
        if xx == 0.0:
            return -self.lam
        return float(_tweedie_positive_logpdf(np.array([xx], dtype=np.float64), self.mu, self.phi, self.p)[0])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Tweedie log-density at sequence-encoded non-negative observations ``x``."""
        xx = np.asarray(x, dtype=np.float64)
        rv = np.full(xx.shape, -np.inf, dtype=np.float64)
        zero = xx == 0.0
        rv[zero] = -self.lam
        pos = xx > 0.0
        if np.any(pos):
            rv[pos] = _tweedie_positive_logpdf(xx[pos], self.mu, self.phi, self.p)
        return rv

    # --- compute-engine backend (numpy + torch/GPU), SCORING only: the moment accumulator stays
    # host-side. The compound Poisson-Gamma series has all-POSITIVE terms, so the logsumexp
    # accumulation is cancellation-free; the ``n``-window mirrors the numpy path (peak-centered,
    # widened until the upper boundary sits 50 log-units below every row's peak). Zeros/negatives
    # are handled by masking (the numpy path slices instead): ``y`` is clamped to 1e-300 inside the
    # series so no ``-inf - (-inf)`` NaN can form, then ``where`` restores the point mass / -inf. ---
    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for Tweedie scoring."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized Tweedie log-density for encoded data (see class backend note)."""
        lam, a, theta = _tweedie_params(self.mu, self.phi, self.p)
        log_lam, log_theta = math.log(lam), math.log(theta)

        xx = engine.asarray(x)
        ys = engine.maximum(xx, engine.asarray(1.0e-300))  # keep the series finite on masked rows
        log_y = engine.log(ys)
        c_i = -lam - log_y - ys / theta

        y_max = float(engine.to_numpy(engine.max(xx))) if np.prod(np.shape(engine.to_numpy(xx))) else 1.0
        n_peak_max = max(y_max, _MIN_TWEEDIE) ** (2.0 - self.p) / (self.phi * (2.0 - self.p))
        n_hi = max(50.0, 2.0 * n_peak_max + 10.0 * math.sqrt(n_peak_max + 1.0) + 50.0)
        for _ in range(64):
            n = engine.asarray(np.arange(1, int(math.ceil(n_hi)) + 1, dtype=np.float64))
            a_n = n * log_lam - engine.gammaln(n + 1.0) - engine.gammaln(n * a) - n * a * log_theta
            terms = c_i[:, None] + a_n[None, :] + (a * log_y)[:, None] * n[None, :]
            m = engine.max(terms, axis=1)
            gap = float(engine.to_numpy(engine.max(terms[:, -1] - m)))  # worst upper-boundary gap
            if gap <= -50.0:
                break
            n_hi = n_hi * 2.0
        pos_val = engine.logsumexp(terms, axis=1)

        neg_inf = engine.asarray(-np.inf)
        return engine.where(xx == 0.0, engine.asarray(-lam), engine.where(xx > 0.0, pos_val, neg_inf))

    def sampler(self, seed: int | None = None) -> "TweedieSampler":
        """Return a TweedieSampler for this distribution."""
        return TweedieSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "TweedieEstimator":
        """Return a TweedieEstimator (method of moments at the fixed power ``p``)."""
        return TweedieEstimator(p=self.p, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "TweedieDataEncoder":
        """Return the encoder for Tweedie observations."""
        return TweedieDataEncoder()


class TweedieSampler(DistributionSampler):
    """Draw iid Tweedie observations exactly as a compound Poisson-Gamma sum."""

    def __init__(self, dist: TweedieDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw ``size`` iid Tweedie samples (a single float if ``size`` is None).

        ``Y | N`` is a sum of ``N`` iid ``Gamma(shape, scale)``, which is ``Gamma(N*shape, scale)``;
        ``N = 0`` yields an exact zero.
        """
        n = int(size) if size is not None else 1
        counts = self.rng.poisson(lam=self.dist.lam, size=n)
        out = np.zeros(n, dtype=np.float64)
        nz = counts > 0
        if np.any(nz):
            out[nz] = self.rng.gamma(shape=counts[nz] * self.dist.gamma_shape, scale=self.dist.gamma_scale)
        return float(out[0]) if size is None else out


class TweedieAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted count, sum, and sum-of-squares for the moment fit."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: TweedieDistribution | None) -> None:
        """Accumulate weighted count, sum, and second moment for one observation."""
        xx = float(x)
        if not np.isfinite(xx) or xx < 0.0:
            raise ValueError("TweedieDistribution requires non-negative observations.")
        xw = xx * weight
        self.count += weight
        self.sum += xw
        self.sum2 += xx * xw

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: TweedieDistribution | None) -> None:
        """Accumulate weighted count, sum, and second moment from encoded observations."""
        xx = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        self.count += ww.sum()
        self.sum += np.dot(xx, ww)
        self.sum2 += np.dot(xx * xx, ww)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "TweedieAccumulator":
        """Merge serialized moment statistics into this accumulator."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum2 += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return the total weight, weighted sum, and weighted second moment."""
        return self.count, self.sum, self.sum2

    def from_value(self, x: tuple[float, float, float]) -> "TweedieAccumulator":
        """Restore the accumulator from serialized moment statistics."""
        self.count, self.sum, self.sum2 = x
        return self

    def scale(self, c: float) -> "TweedieAccumulator":
        """Scale accumulated moment statistics by a constant."""
        self.count *= c
        self.sum *= c
        self.sum2 *= c
        return self

    def acc_to_encoder(self) -> "TweedieDataEncoder":
        """Return an encoder for non-negative Tweedie observations."""
        return TweedieDataEncoder()


class TweedieAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for TweedieAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> "TweedieAccumulator":
        """Create an empty Tweedie accumulator."""
        return TweedieAccumulator(name=self.name, keys=self.keys)


class TweedieEstimator(ParameterEstimator):
    """Estimate ``(mu, phi)`` at fixed power ``p`` by the (exact) method of moments."""

    def __init__(self, p: float = 1.5, name: str | None = None, keys: str | None = None) -> None:
        """Method-of-moments Tweedie estimator at fixed power ``p``.

        ``E[Y] = mu`` and ``Var[Y] = phi * mu**p``, so ``mu`` is the sample mean and
        ``phi = sample_var / mu**p`` (both floored to stay positive).
        """
        if not (1.0 < p < 2.0):
            raise ValueError("TweedieEstimator requires power p strictly in (1, 2).")
        self.p = float(p)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "TweedieAccumulatorFactory":
        """Return a factory for Tweedie moment accumulators."""
        return TweedieAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> "TweedieDistribution":
        """Estimate a Tweedie from the accumulated ``(count, sum, sum2)`` via method of moments."""
        count, xsum, xsum2 = suff_stat
        if count <= 0.0:
            return TweedieDistribution(1.0, 1.0, self.p, name=self.name, keys=self.keys)
        mean = max(xsum / count, _MIN_TWEEDIE)
        var = xsum2 / count - mean * mean
        phi = var / mean**self.p
        if not np.isfinite(phi) or phi <= 0.0:
            phi = _MIN_TWEEDIE
        return TweedieDistribution(mean, phi, self.p, name=self.name, keys=self.keys)


class TweedieDataEncoder(DataSequenceEncoder):
    """Encode sequences of iid Tweedie observations (non-negative float data type)."""

    def __str__(self) -> str:
        return "TweedieDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TweedieDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Validate and encode observations as a non-negative float array."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(np.isnan(rv)) or np.any(np.isinf(rv)) or np.any(rv < 0.0)):
            raise ValueError("TweedieDistribution requires finite non-negative observations.")
        return rv
