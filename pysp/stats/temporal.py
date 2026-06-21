"""Time and date modelling on raw timestamps: periodic (cyclic) distributions and seasonal time series.

Real temporal data arrives as raw timestamps -- Python ``datetime``/``date``, ``numpy.datetime64``, ISO
strings, or POSIX seconds. This module consumes any of those directly. Two capabilities:

* :class:`PeriodicTime` -- a distribution over *where in a recurring cycle* events fall (time-of-day,
  day-of-week, season), via a von Mises on the cycle phase. Captures recurring timing: "events cluster
  around 9am", "activity peaks on weekends", "blooms in spring".
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

__all__ = ["to_unix_seconds", "cyclic_phase", "PERIODS", "PeriodicTime", "SeasonalTimeSeries"]

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


def _solve_kappa(rbar: float) -> float:
    """Invert the von Mises mean-resultant length to a concentration (Fisher's approximation)."""
    rbar = float(np.clip(rbar, 0.0, 1.0 - 1e-10))
    if rbar < 0.53:
        return 2.0 * rbar + rbar**3 + 5.0 * rbar**5 / 6.0
    if rbar < 0.85:
        return -0.4 + 1.39 * rbar + 0.43 / (1.0 - rbar)
    return 1.0 / (rbar**3 - 4.0 * rbar**2 + 3.0 * rbar)


class PeriodicTime:
    """A von Mises distribution over the phase of a recurring cycle -- recurring-timing on raw timestamps.

    ``period`` is the cycle (``'day'``, ``'week'``, ``'year'``, ... or seconds). ``loc`` is the peak phase
    (radians) and ``conc`` the concentration (0 = uniform over the cycle, large = sharply peaked).
    """

    def __init__(self, period: float | str = "day", loc: float = 0.0, conc: float = 0.0):
        self.period = period
        self.period_s = _period_seconds(period)
        self.loc = float(loc)
        self.conc = float(conc)

    def log_density(self, t: Any) -> np.ndarray | float:
        """Log-density at timestamp(s), as a proper density over time within one period."""
        from scipy.special import i0e

        phi = cyclic_phase(t, self.period)
        jac = np.log(2.0 * np.pi / self.period_s)  # d(phase)/d(time), so the density integrates to 1 over a period
        vm = self.conc * (np.cos(phi - self.loc) - 1.0) - np.log(2.0 * np.pi) - np.log(i0e(self.conc))
        ld = vm + jac
        return float(ld[0]) if np.ndim(t) == 0 else ld

    def peak_phase_fraction(self) -> float:
        """The peak location as a fraction of the cycle (e.g. 0.375 of a day = 09:00)."""
        return float(np.mod(self.loc, 2.0 * np.pi) / (2.0 * np.pi))

    @classmethod
    def fit(cls, times: Any, period: float | str = "day") -> PeriodicTime:
        """Maximum-likelihood von Mises fit of the cycle phase of ``times``."""
        phi = cyclic_phase(times, period)
        c, s = np.cos(phi).mean(), np.sin(phi).mean()
        rbar = np.hypot(c, s)
        loc = np.arctan2(s, c)
        return cls(period, loc=loc, conc=_solve_kappa(rbar))

    def sampler(self, seed: int | None = None) -> PeriodicTimeSampler:
        return PeriodicTimeSampler(self, seed)


class PeriodicTimeSampler:
    def __init__(self, dist: PeriodicTime, seed: int | None = None):
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None) -> np.ndarray | float:
        """Draw phase fraction(s) of the cycle in ``[0, period_seconds)`` (von Mises rejection sampler)."""
        n = 1 if size is None else size
        phi = self._vonmises(n)
        secs = np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi) * self.dist.period_s
        return float(secs[0]) if size is None else secs

    def _vonmises(self, n: int) -> np.ndarray:
        # Best-Fisher rejection sampler for von Mises(loc, conc).
        kappa = self.dist.conc
        if kappa < 1e-8:
            return self.rng.uniform(0, 2 * np.pi, n)
        tau = 1.0 + np.sqrt(1.0 + 4.0 * kappa**2)
        rho = (tau - np.sqrt(2.0 * tau)) / (2.0 * kappa)
        r = (1.0 + rho**2) / (2.0 * rho)
        out = np.empty(n)
        i = 0
        while i < n:
            u1, u2, u3 = self.rng.uniform(size=3)
            z = np.cos(np.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)
            if c * (2.0 - c) - u2 > 0 or np.log(c / u2) + 1.0 - c >= 0:
                out[i] = np.mod(self.dist.loc + np.sign(u3 - 0.5) * np.arccos(f), 2.0 * np.pi)
                i += 1
        return out


class SeasonalTimeSeries:
    """A conditional distribution ``value | time`` for raw ``(timestamp, value)`` series: a Gaussian whose
    mean is a linear trend plus Fourier seasonal harmonics.

    Models ``value | time ~ N(mu(time), s^2)`` with ``mu(t) = b0 + b1 t + sum over periods/harmonics of
    [a sin(2pi k t / P) + b cos(...)]``, fit by least squares (``t`` in days from the first timestamp).
    Being a distribution, it has no ``predict`` -- you ask for the conditional distribution at a time and
    read its mean, sample it, or score data: ``conditional(t)`` returns a
    :class:`~pysp.stats.GaussianDistribution` (the posterior-predictive at ``t``, parameter uncertainty +
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
        for an array of times). This is how you 'predict' the pysp way: you get a distribution to sample,
        score, or read ``.mu`` / ``.sigma2`` from, not a bare point estimate."""
        from pysp.stats import GaussianDistribution

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
