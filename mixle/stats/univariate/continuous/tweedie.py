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
_DEFAULT_MAX_SERIES_TERMS = 100_000
_DEFAULT_MAX_SERIES_WORK = 10_000_000
_DEFAULT_SERIES_TOLERANCE = 50.0


class TweedieSeriesResourceError(RuntimeError):
    """A Tweedie density series exceeded its declared term or work budget."""

    def __init__(self, *, required_terms: int, rows: int, max_terms: int, max_work: int) -> None:
        self.required_terms = required_terms
        self.rows = rows
        self.max_terms = max_terms
        self.max_work = max_work
        super().__init__(
            "Tweedie density series requires at least %d terms across %d rows; "
            "budgets are max_series_terms=%d and max_series_work=%d." % (required_terms, rows, max_terms, max_work)
        )


def _series_budget(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _series_target(n_peak_max: float) -> int:
    return int(
        math.ceil(
            max(
                50.0,
                2.0 * n_peak_max + 10.0 * math.sqrt(n_peak_max + 1.0) + 50.0,
            )
        )
    )


def _check_series_budget(*, terms: int, rows: int, max_terms: int, max_work: int) -> None:
    if terms > max_terms or terms * rows > max_work:
        raise TweedieSeriesResourceError(
            required_terms=terms,
            rows=rows,
            max_terms=max_terms,
            max_work=max_work,
        )


def _tweedie_params(mu: float, phi: float, p: float) -> tuple[float, float, float]:
    """Return the compound Poisson-Gamma ``(lam, a, theta)`` for mean ``mu``, dispersion ``phi``."""
    lam = mu ** (2.0 - p) / (phi * (2.0 - p))
    a = (2.0 - p) / (p - 1.0)
    theta = phi * (p - 1.0) * mu ** (p - 1.0)
    return lam, a, theta


def _tweedie_positive_logpdf(
    y: np.ndarray,
    mu: float,
    phi: float,
    p: float,
    *,
    max_terms: int,
    max_work: int,
    tolerance: float,
) -> np.ndarray:
    """Return ``log f(y)`` using bounded, streaming log-sum accumulation."""
    lam, a, theta = _tweedie_params(mu, phi, p)
    log_lam = math.log(lam)
    log_theta = math.log(theta)
    log_y = np.log(y)
    rows = len(y)
    if rows == 0:
        return np.empty(0, dtype=np.float64)
    c_i = -lam - log_y - y / theta
    n_peak = np.power(np.maximum(y, _MIN_TWEEDIE), 2.0 - p) / (phi * (2.0 - p))
    target = _series_target(float(np.max(n_peak)))
    _check_series_budget(terms=target, rows=rows, max_terms=max_terms, max_work=max_work)

    log_sum = np.full(rows, -np.inf, dtype=np.float64)
    peak = np.full(rows, -np.inf, dtype=np.float64)
    completed = 0
    while True:
        for n_int in range(completed + 1, target + 1):
            n = float(n_int)
            a_n = n * log_lam - gammaln(n + 1.0) - gammaln(n * a) - n * a * log_theta
            term = c_i + a_n + (a * log_y) * n
            log_sum = np.logaddexp(log_sum, term)
            peak = np.maximum(peak, term)
        completed = target
        if np.all(peak - term >= tolerance):
            return log_sum
        target = min(max_terms, target * 2)
        if target == completed:
            raise TweedieSeriesResourceError(
                required_terms=completed + 1,
                rows=rows,
                max_terms=max_terms,
                max_work=max_work,
            )
        _check_series_budget(terms=target, rows=rows, max_terms=max_terms, max_work=max_work)


class TweedieDistribution(SequenceEncodableProbabilityDistribution):
    """Tweedie (compound Poisson-Gamma) distribution on ``[0, inf)`` with fixed power ``p in (1, 2)``."""

    def __init__(
        self,
        mu: float,
        phi: float,
        p: float = 1.5,
        name: str | None = None,
        keys: str | None = None,
        *,
        max_series_terms: int = _DEFAULT_MAX_SERIES_TERMS,
        max_series_work: int = _DEFAULT_MAX_SERIES_WORK,
        series_tolerance: float = _DEFAULT_SERIES_TOLERANCE,
    ) -> None:
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
        checked_max_terms = _series_budget(max_series_terms, "max_series_terms")
        checked_max_work = _series_budget(max_series_work, "max_series_work")
        if not np.isfinite(series_tolerance) or series_tolerance <= 0.0:
            raise ValueError("series_tolerance must be finite and positive")
        self.mu = float(mu)
        self.phi = float(phi)
        self.p = float(p)
        self.name = name
        self.keys = keys
        self.max_series_terms = checked_max_terms
        self.max_series_work = checked_max_work
        self.series_tolerance = float(series_tolerance)
        self.lam, self.gamma_shape, self.gamma_scale = _tweedie_params(self.mu, self.phi, self.p)

    def __str__(self) -> str:
        """Return a constructor-style representation of the Tweedie distribution."""
        return (
            "TweedieDistribution(%s, %s, %s, name=%s, keys=%s, "
            "max_series_terms=%s, max_series_work=%s, series_tolerance=%s)"
        ) % (
            repr(self.mu),
            repr(self.phi),
            repr(self.p),
            repr(self.name),
            repr(self.keys),
            repr(self.max_series_terms),
            repr(self.max_series_work),
            repr(self.series_tolerance),
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
        return float(
            _tweedie_positive_logpdf(
                np.array([xx], dtype=np.float64),
                self.mu,
                self.phi,
                self.p,
                max_terms=self.max_series_terms,
                max_work=self.max_series_work,
                tolerance=self.series_tolerance,
            )[0]
        )

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Tweedie log-density at sequence-encoded non-negative observations ``x``."""
        xx = np.asarray(x, dtype=np.float64)
        rv = np.full(xx.shape, -np.inf, dtype=np.float64)
        zero = xx == 0.0
        rv[zero] = -self.lam
        pos = xx > 0.0
        if np.any(pos):
            rv[pos] = _tweedie_positive_logpdf(
                xx[pos],
                self.mu,
                self.phi,
                self.p,
                max_terms=self.max_series_terms,
                max_work=self.max_series_work,
                tolerance=self.series_tolerance,
            )
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
        rows = int(np.size(engine.to_numpy(xx)))
        if rows == 0:
            return engine.asarray(np.empty(0, dtype=np.float64))
        ys = engine.maximum(xx, engine.asarray(1.0e-300))  # keep the series finite on masked rows
        log_y = engine.log(ys)
        c_i = -lam - log_y - ys / theta

        y_max = float(engine.to_numpy(engine.max(xx))) if np.prod(np.shape(engine.to_numpy(xx))) else 1.0
        n_peak_max = max(y_max, _MIN_TWEEDIE) ** (2.0 - self.p) / (self.phi * (2.0 - self.p))
        target = _series_target(n_peak_max)
        _check_series_budget(
            terms=target,
            rows=rows,
            max_terms=self.max_series_terms,
            max_work=self.max_series_work,
        )
        log_sum = None
        peak = None
        completed = 0
        while True:
            term = None
            for n_int in range(completed + 1, target + 1):
                n = engine.asarray(float(n_int))
                a_n = n * log_lam - engine.gammaln(n + 1.0) - engine.gammaln(n * a) - n * a * log_theta
                term = c_i + a_n + (a * log_y) * n
                if log_sum is None:
                    log_sum = term
                    peak = term
                else:
                    log_sum = engine.logsumexp(engine.stack((log_sum, term), axis=1), axis=1)
                    peak = engine.maximum(peak, term)
            completed = target
            gap = float(engine.to_numpy(engine.max(term - peak)))
            if gap <= -self.series_tolerance:
                break
            target = min(self.max_series_terms, target * 2)
            if target == completed:
                raise TweedieSeriesResourceError(
                    required_terms=completed + 1,
                    rows=rows,
                    max_terms=self.max_series_terms,
                    max_work=self.max_series_work,
                )
            _check_series_budget(
                terms=target,
                rows=rows,
                max_terms=self.max_series_terms,
                max_work=self.max_series_work,
            )
        pos_val = log_sum

        neg_inf = engine.asarray(-np.inf)
        return engine.where(xx == 0.0, engine.asarray(-lam), engine.where(xx > 0.0, pos_val, neg_inf))

    def sampler(self, seed: int | None = None) -> "TweedieSampler":
        """Return a TweedieSampler for this distribution."""
        return TweedieSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "TweedieEstimator":
        """Return a TweedieEstimator (method of moments at the fixed power ``p``)."""
        if pseudo_count is None:
            return TweedieEstimator(p=self.p, name=self.name, keys=self.keys)
        # E[Y] = mu, Var[Y] = phi*mu^p, so E[Y^2] = phi*mu^p + mu^2 -- the raw moment space
        # estimate() accumulates in (sum, sum2) -- so pseudo_count can blend a prior pseudo-sample
        # toward them (mirrors GumbelEstimator / WeibullEstimator's suff_stat pattern).
        mean0 = self.mu
        second0 = self.phi * self.mu**self.p + self.mu * self.mu
        return TweedieEstimator(
            p=self.p, pseudo_count=pseudo_count, suff_stat=(mean0, second0), name=self.name, keys=self.keys
        )

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

    def __init__(
        self,
        p: float = 1.5,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Method-of-moments Tweedie estimator at fixed power ``p``.

        ``E[Y] = mu`` and ``Var[Y] = phi * mu**p``, so ``mu`` is the sample mean and
        ``phi = sample_var / mu**p`` (both floored to stay positive).
        """
        if not (1.0 < p < 2.0):
            raise ValueError("TweedieEstimator requires power p strictly in (1, 2).")
        self.p = float(p)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "TweedieAccumulatorFactory":
        """Return a factory for Tweedie moment accumulators."""
        return TweedieAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> "TweedieDistribution":
        """Estimate a Tweedie from the accumulated ``(count, sum, sum2)`` via method of moments."""
        count, xsum, xsum2 = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            mean0, second0 = self.suff_stat
            xsum += self.pseudo_count * mean0
            xsum2 += self.pseudo_count * second0
            count += self.pseudo_count
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
