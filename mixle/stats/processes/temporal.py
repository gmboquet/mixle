"""Time and date modelling on raw timestamps: periodic (cyclic) distributions and seasonal time series.

Real temporal data arrives as raw timestamps -- Python ``datetime``/``date``, ``numpy.datetime64``, ISO
strings, or POSIX seconds. This module consumes any of those directly. Two capabilities:

* :class:`PeriodicTimeDistribution` -- a distribution over *where in a recurring cycle* events fall
  (time-of-day, day-of-week, season), a von Mises on the cycle phase. Captures recurring timing: "events
  cluster around 9am", "activity peaks on weekends", "blooms in spring".
* :class:`SeasonalTimeSeries` -- a *conditional distribution* ``value | time`` on ``(timestamp, value)``
  data: a Gaussian whose mean is a linear trend plus Fourier seasonal harmonics at one or more periods.
  Like any distribution it has ``conditional`` (returns the predictive distribution at a time), ``mean``,
  ``log_density`` and a ``sampler`` -- not a ``predict``. ``decompose`` splits the mean into trend +
  seasonal parts.

Part of the earth-science/multiphysics/UQ work and generally useful (paleo records are time series;
event catalogues -- earthquakes, blooms, drilling -- have strong calendar/seasonal structure).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.directional.von_mises import VonMisesDistribution, VonMisesEstimator

__all__ = [
    "to_unix_seconds",
    "cyclic_phase",
    "PERIODS",
    "PeriodicTimeDistribution",
    "PeriodicTimeEstimator",
    "SeasonalTimeSeries",
]

# Named cycle lengths in seconds (year = mean Gregorian year).
PERIODS: dict[str, float] = {
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "month": 365.2425 * 86400.0 / 12.0,
    "year": 365.2425 * 86400.0,
}


def to_unix_seconds(x: Any) -> np.ndarray:
    """Convert raw date/time data to POSIX seconds (float), accepting datetime/date, ``datetime64``,
    ISO strings, or numbers (already seconds). Returns a 1-D float array; NaT/None become NaN."""
    arr = np.asarray(x)
    if arr.dtype == object or np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.datetime64):
        sec = np.asarray(x, dtype="datetime64[ns]").astype("datetime64[ns]")
        return np.atleast_1d(sec.astype("int64").astype(float) / 1e9)
    if np.issubdtype(arr.dtype, np.number):
        return np.atleast_1d(arr.astype(float))
    sec = np.asarray(x, dtype="datetime64[ns]")
    return np.atleast_1d(sec.astype("int64").astype(float) / 1e9)


def _period_seconds(period: float | str) -> float:
    if isinstance(period, str):
        if period not in PERIODS:
            raise ValueError(f"unknown period {period!r}; use one of {list(PERIODS)} or seconds.")
        return PERIODS[period]
    return float(period)


def cyclic_phase(times: Any, period: float | str) -> np.ndarray:
    """Map timestamps to a cycle phase in ``[0, 2pi)`` for the given ``period`` (named or seconds)."""
    p = _period_seconds(period)
    return 2.0 * np.pi * np.mod(to_unix_seconds(times), p) / p


class PeriodicTimeDistribution(SequenceEncodableProbabilityDistribution):
    """A distribution over raw timestamps whose recurring-cycle *phase* is von Mises.

    The cycle phase ``phi = 2*pi*(t mod P)/P`` is distributed von Mises around a peak; this is a proper
    density over *time* within one period (a constant Jacobian ``log(2*pi/P)`` makes it integrate to 1).
    It follows the mixle Distribution / Sampler / Estimator / Accumulator / DataEncoder contract by
    delegating the circular density to :class:`~mixle.stats.directional.von_mises.VonMisesDistribution`, so
    recurring time-of-day / day-of-week / seasonal patterns compose with the rest of the library.

    ``period`` is the cycle (``'day'``, ``'week'``, ``'year'``, ... or seconds); ``loc`` is the peak phase
    (radians) and ``conc`` the concentration (0 = uniform over the cycle, large = sharply peaked).
    """

    def __init__(self, period: float | str = "day", loc: float = 0.0, conc: float = 0.0, name=None, keys=None):
        self.period = period
        self.period_s = _period_seconds(period)
        self.von_mises = VonMisesDistribution(float(loc), float(conc))
        self._jac = float(np.log(2.0 * np.pi / self.period_s))  # d(phase)/d(time), so it integrates to 1 over a period
        self.name = name
        self.keys = keys

    @property
    def loc(self) -> float:
        """Return the peak phase location in radians."""
        return float(self.von_mises.mu)

    @property
    def conc(self) -> float:
        """Return the von Mises concentration for the recurring phase."""
        return float(self.von_mises.kappa)

    def __str__(self) -> str:
        return "PeriodicTimeDistribution(%r, loc=%r, conc=%r)" % (self.period, self.loc, self.conc)

    def density(self, t: Any) -> float:
        """Return the time density at one timestamp."""
        return float(np.exp(self.log_density(t)))

    def log_density(self, t: Any) -> float:
        """Log-density at one timestamp (the von Mises phase density plus the time Jacobian)."""
        return float(self.von_mises.log_density(float(cyclic_phase(t, self.period)[0])) + self._jac)

    def seq_log_density(self, x) -> np.ndarray:
        """Return vectorized log-densities for phase-encoded timestamps."""
        return self.von_mises.seq_log_density(x) + self._jac  # x is phase-encoded by the PeriodicTime encoder

    def peak_phase_fraction(self) -> float:
        """The peak location as a fraction of the cycle (e.g. 0.375 of a day = 09:00)."""
        return float(np.mod(self.loc, 2.0 * np.pi) / (2.0 * np.pi))

    def sampler(self, seed: int | None = None) -> PeriodicTimeSampler:
        """Return a sampler for recurring cycle positions."""
        return PeriodicTimeSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> PeriodicTimeEstimator:
        """Return an estimator for the recurring timestamp phase."""
        return PeriodicTimeEstimator(self.period, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> PeriodicTimeDataEncoder:
        """Return the timestamp-to-phase encoder used by vectorized methods."""
        return PeriodicTimeDataEncoder(self.period, self.von_mises.dist_to_encoder())


class PeriodicTimeSampler(DistributionSampler):
    """Sample timestamps as positions within one recurring period."""

    def __init__(self, dist: PeriodicTimeDistribution, seed: int | None = None):
        self.dist = dist
        self.von_mises_sampler = dist.von_mises.sampler(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray | float:
        """Draw cycle position(s) in seconds, ``[0, period_seconds)`` (the phase mapped back to time)."""
        phi = self.von_mises_sampler.sample(size)
        secs = np.mod(np.asarray(phi), 2.0 * np.pi) / (2.0 * np.pi) * self.dist.period_s
        return float(secs) if size is None else secs


class PeriodicTimeDataEncoder(DataSequenceEncoder):
    """Encode timestamps by the cycle phase, then defer to the von Mises encoder over the angles."""

    def __init__(self, period, von_mises_encoder):
        self.period = period
        self.von_mises_encoder = von_mises_encoder

    def __str__(self) -> str:
        return "PeriodicTimeDataEncoder(%r)" % (self.period,)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PeriodicTimeDataEncoder)
            and self.period == other.period
            and self.von_mises_encoder == other.von_mises_encoder
        )

    def seq_encode(self, x):
        """Encode raw timestamps as von Mises phase observations."""
        return self.von_mises_encoder.seq_encode(cyclic_phase(x, self.period))


class PeriodicTimeEstimator(ParameterEstimator):
    """Maximum-likelihood estimator: the von Mises MLE of the cycle phase (delegated to VonMises)."""

    def __init__(self, period: float | str = "day", name=None, keys=None):
        self.period = period
        self.von_mises_estimator = VonMisesEstimator()
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StatisticAccumulatorFactory:
        """Return an accumulator factory that converts timestamps to cycle phases."""
        period = self.period
        keys = self.keys
        von_mises_factory = self.von_mises_estimator.accumulator_factory()

        class _Factory(StatisticAccumulatorFactory):
            def make(self):
                """Create a periodic-time accumulator."""
                return PeriodicTimeAccumulator(period, von_mises_factory.make(), keys=keys)

        return _Factory()

    def estimate(self, nobs, suff_stat) -> PeriodicTimeDistribution:
        """Estimate a periodic-time distribution from von Mises phase statistics."""
        vm = self.von_mises_estimator.estimate(nobs, suff_stat)
        return PeriodicTimeDistribution(self.period, vm.mu, vm.kappa, name=self.name, keys=self.keys)


class PeriodicTimeAccumulator(SequenceEncodableStatisticAccumulator):
    """Wrap a von Mises accumulator; timestamps are mapped to cycle phases, then the stats are delegated."""

    def __init__(self, period, von_mises_acc, keys=None):
        self.period = period
        self.von_mises_acc = von_mises_acc
        self.keys = keys

    def update(self, t, weight, estimate):
        """Update delegated von Mises statistics from one raw timestamp."""
        self.von_mises_acc.update(
            float(cyclic_phase(t, self.period)[0]), weight, None if estimate is None else estimate.von_mises
        )

    def initialize(self, t, weight, rng):
        """Initialize delegated von Mises statistics from one raw timestamp."""
        self.von_mises_acc.initialize(float(cyclic_phase(t, self.period)[0]), weight, rng)

    def seq_update(self, x, weights, estimate):
        """Update delegated von Mises statistics from phase-encoded timestamps."""
        self.von_mises_acc.seq_update(x, weights, None if estimate is None else estimate.von_mises)

    def seq_initialize(self, x, weights, rng):
        """Initialize delegated von Mises statistics from phase-encoded timestamps."""
        self.von_mises_acc.seq_initialize(x, weights, rng)

    def combine(self, suff_stat):
        """Merge delegated von Mises sufficient statistics."""
        self.von_mises_acc.combine(suff_stat)
        return self

    def value(self):
        """Return delegated von Mises sufficient statistics."""
        return self.von_mises_acc.value()

    def from_value(self, x):
        """Restore delegated von Mises sufficient statistics."""
        self.von_mises_acc.from_value(x)
        return self

    def scale(self, c):
        """Scale delegated von Mises sufficient statistics."""
        self.von_mises_acc.scale(c)
        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict):
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self):
        """Return the timestamp encoder compatible with this accumulator."""
        return PeriodicTimeDataEncoder(self.period, self.von_mises_acc.acc_to_encoder())


class SeasonalTimeSeries:
    """A conditional distribution ``value | time`` for raw ``(timestamp, value)`` series: a Gaussian whose
    mean is a linear trend plus Fourier seasonal harmonics.

    Models ``value | time ~ N(mu(time), s^2)`` with ``mu(t) = b0 + b1 t + sum over periods/harmonics of
    [a sin(2pi k t / P) + b cos(...)]``, fit by least squares (``t`` in days from the first timestamp).
    Being a distribution, it has no ``predict`` -- you ask for the conditional distribution at a time and
    read its mean, sample it, or score data: ``conditional(t)`` returns a
    :class:`~mixle.stats.GaussianDistribution` (the posterior-predictive at ``t``, parameter uncertainty +
    noise), ``mean(times)`` is ``E[value | time]``, ``log_density(times, values)`` scores observations,
    and ``sampler(seed).sample(times)`` draws values. Captures several seasonalities at once (daily +
    weekly + yearly); ``decompose`` splits the mean into trend + per-period parts.
    """

    def __init__(self, periods: Sequence[float | str] = ("year",), harmonics: int = 3, trend: bool = True):
        self.periods = [_period_seconds(p) for p in periods]
        self.period_names = list(periods)
        self.harmonics = int(harmonics)
        self.trend = bool(trend)

    def _design(self, secs: np.ndarray) -> np.ndarray:
        t_days = (secs - self._t0) / 86400.0
        cols = [np.ones_like(t_days)]
        if self.trend:
            cols.append(t_days)
        for p in self.periods:
            for k in range(1, self.harmonics + 1):
                w = 2.0 * np.pi * k * secs / p
                cols.append(np.sin(w))
                cols.append(np.cos(w))
        return np.column_stack(cols)

    def fit(self, times: Any, values: Any) -> SeasonalTimeSeries:
        """Fit trend, seasonal coefficients, noise variance, and predictive covariance terms."""
        secs = to_unix_seconds(times)
        y = np.asarray(values, dtype=float).ravel()
        order = np.argsort(secs)
        secs, y = secs[order], y[order]
        self._t0 = secs[0]
        x = self._design(secs)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        dof = max(len(y) - x.shape[1], 1)
        self.beta = beta
        self.sigma = float(np.sqrt(np.sum(resid**2) / dof))
        self._xtx_inv = np.linalg.pinv(x.T @ x)
        return self

    def mean(self, times: Any) -> np.ndarray:
        """The conditional expectation ``E[value | time]`` -- the fitted trend + seasonality."""
        return self._design(to_unix_seconds(times)) @ self.beta

    def _predictive_var(self, times: Any) -> np.ndarray:
        """Posterior-predictive variance at ``times`` (observation noise + parameter uncertainty)."""
        x = self._design(to_unix_seconds(times))
        return self.sigma**2 * (1.0 + np.einsum("ij,jk,ik->i", x, self._xtx_inv, x))

    def conditional(self, time: Any):
        """The conditional distribution ``p(value | time)`` -- a :class:`GaussianDistribution` (or a list,
        for an array of times). Prediction returns a distribution to sample,
        score, or read ``.mu`` / ``.sigma2`` from, not a bare point estimate."""
        from mixle.stats import GaussianDistribution

        m, v = self.mean(time), self._predictive_var(time)
        if np.ndim(time) == 0:
            return GaussianDistribution(float(m[0]), float(v[0]))
        return [GaussianDistribution(float(mi), float(vi)) for mi, vi in zip(m, v)]

    def log_density(self, times: Any, values: Any) -> np.ndarray | float:
        """Conditional log-density of ``(time, value)`` observations under the model."""
        m, v = self.mean(times), self._predictive_var(times)
        y = np.asarray(values, dtype=float).ravel()
        ld = -0.5 * ((y - m) ** 2 / v + np.log(2.0 * np.pi * v))
        return float(ld[0]) if np.ndim(values) == 0 else ld

    def sampler(self, seed: int | None = None) -> SeasonalTimeSeriesSampler:
        """Return a sampler for conditional values at requested timestamps."""
        return SeasonalTimeSeriesSampler(self, seed)

    def decompose(self, times: Any) -> dict[str, np.ndarray]:
        """Split the prediction into ``trend`` and one component per period (the seasonal contributions)."""
        secs = to_unix_seconds(times)
        t_days = (secs - self._t0) / 86400.0
        out = {"trend": self.beta[0] + (self.beta[1] * t_days if self.trend else 0.0)}
        j = 2 if self.trend else 1
        for name, p in zip(self.period_names, self.periods):
            part = np.zeros_like(secs, dtype=float)
            for k in range(1, self.harmonics + 1):
                w = 2.0 * np.pi * k * secs / p
                part = part + self.beta[j] * np.sin(w) + self.beta[j + 1] * np.cos(w)
                j += 2
            out[str(name)] = part
        return out


class SeasonalTimeSeriesSampler:
    """Draws values from the conditional ``p(value | time)`` of a fitted :class:`SeasonalTimeSeries`."""

    def __init__(self, dist: SeasonalTimeSeries, seed: int | None = None):
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, times: Any) -> np.ndarray:
        """Sample one value at each timestamp in ``times`` from its conditional distribution."""
        m = self.dist.mean(times)
        sd = np.sqrt(self.dist._predictive_var(times))
        return m + sd * self.rng.standard_normal(len(m))
