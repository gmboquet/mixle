"""Multivariate (mutually-exciting) Hawkes process with an exponential kernel.

A ``D``-dimensional Hawkes process over marked events ``(t, mark)`` on a fixed window ``[0, T]``. Each
mark has a baseline intensity ``mu_d`` and an event of mark ``j`` excites the intensity of mark ``d`` by
``alpha_{dj} exp(-beta (t - t_k))``, so

    lambda_d(t) = mu_d + sum_{t_k < t} alpha_{d, mark_k} exp(-beta (t - t_k)),

coupling the dimensions through the ``D x D`` excitation matrix ``alpha`` (a shared decay ``beta``). The
exact log-likelihood ``sum_n log lambda_{mark_n}(t_n) - sum_d \\int_0^T lambda_d`` is computed in
``O(n D)`` by a per-mark excitation recursion, sampling is by multivariate Ogata thinning, and the
parameters are fit by the Veen-Schoenberg branching EM (each event is an immigrant of its mark or the
offspring of an earlier event, with the excitation kernel as the parent likelihood). The process is
stationary when the spectral radius of ``alpha / beta`` is below 1.


Reference: Hawkes, 'Spectra of some self-exciting and mutually exciting point processes', Biometrika (1971).
"""

import math
import warnings
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

_MIN = 1.0e-12


def _split(events: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split a realization (sequence of ``(time, mark)``) into sorted time and mark arrays."""
    if len(events) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    arr = np.asarray(events, dtype=np.float64)
    return arr[:, 0].astype(np.float64), arr[:, 1].astype(np.int64)


class MultivariateHawkesProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Multivariate Hawkes process: baselines ``mu`` (D), excitation ``alpha`` (D, D), decay ``beta``."""

    def __init__(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: float,
        window: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        m = np.asarray(mu, dtype=np.float64).reshape(-1)
        a = np.asarray(alpha, dtype=np.float64)
        self.dim = m.shape[0]
        if a.shape != (self.dim, self.dim):
            raise ValueError("alpha must be a (D, D) matrix matching len(mu)")
        if np.any(m <= 0.0) or np.any(a < 0.0) or beta <= 0.0 or window <= 0.0:
            raise ValueError("multivariate Hawkes requires mu>0, alpha>=0, beta>0, window>0.")
        self.mu = m
        self.alpha = a
        self.beta = float(beta)
        self.window = float(window)
        self.name = name
        self.keys = keys
        self._col_alpha = a.sum(axis=0)  # col j: total excitation an event of mark j sends to all marks
        self.spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a / self.beta))))

    def __str__(self) -> str:
        return "MultivariateHawkesProcessDistribution(%s, %s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mu.tolist()),
            repr(self.alpha.tolist()),
            repr(self.beta),
            repr(self.window),
            repr(self.name),
            repr(self.keys),
        )

    def intensity(self, t: float, times: Any, marks: Any) -> np.ndarray:
        """Per-mark conditional rate vector (the vector-valued variant of ``intensity``).

        Returns ``lambda(t)`` of shape ``(D,)`` with
        ``lambda_k(t) = mu_k + sum_{(t_i, m_i) < t} alpha[k, m_i] exp(-beta (t - t_i))``.
        """
        ti = np.asarray(times, dtype=np.float64).reshape(-1)
        mi = np.asarray(marks, dtype=np.int64).reshape(-1)
        past = ti < t
        lam = self.mu.copy()
        if np.any(past):
            decay = np.exp(-self.beta * (t - ti[past]))
            # s[j] = sum_{past, mark=j} exp(-beta (t - t_i)); lambda = mu + alpha @ s
            s = np.zeros(self.dim)
            np.add.at(s, mi[past], decay)
            lam = lam + self.alpha @ s
        return lam

    def expected_count(self, t_start: float, t_end: float, times: Any, marks: Any) -> np.ndarray:
        """Per-mark compensator vector (the vector-valued variant of ``expected_count``).

        Returns ``(D,)`` with the integral of ``lambda_k`` over ``[t_start, t_end]`` given the history.
        """
        ti = np.asarray(times, dtype=np.float64).reshape(-1)
        mi = np.asarray(marks, dtype=np.int64).reshape(-1)
        rel = ti < t_end
        comp = self.mu * (t_end - t_start)
        if np.any(rel):
            tp, mp = ti[rel], mi[rel]
            lo = np.maximum(t_start, tp)
            kernel = (np.exp(-self.beta * (lo - tp)) - np.exp(-self.beta * (t_end - tp))) / self.beta
            # per-parent-mark integrated kernel mass, then route through the excitation matrix
            mass = np.zeros(self.dim)
            np.add.at(mass, mp, kernel)
            comp = comp + self.alpha @ mass
        return comp

    def density(self, x: Any) -> float:
        """Probability density of one realization (a sequence of ``(time, mark)`` events)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Exact log-likelihood of one realization of marked events sorted by time."""
        times, marks = _split(x)
        n = times.size
        w = self.window
        if n and (np.any(~np.isfinite(times)) or times[0] < 0.0 or times[-1] > w or np.any(np.diff(times) < 0.0)):
            return -np.inf
        if n and (np.any(marks < 0) or np.any(marks >= self.dim)):
            return -np.inf
        mu, alpha, beta = self.mu, self.alpha, self.beta
        loglam = 0.0
        s = np.zeros(self.dim)  # s[j] = sum_{k<i, mark_k=j} exp(-beta (t_i - t_k))
        prev = 0.0
        for i in range(n):
            if i > 0:
                s *= math.exp(-beta * (times[i] - prev))
            lam = mu[marks[i]] + float(alpha[marks[i]] @ s)
            loglam += math.log(lam)
            s[marks[i]] += 1.0
            prev = times[i]
        comp = w * float(mu.sum())
        if n:
            comp += (1.0 / beta) * float(np.sum(self._col_alpha[marks] * (1.0 - np.exp(-beta * (w - times)))))
        return float(loglam - comp)

    def seq_log_density(self, x: list[Any]) -> np.ndarray:
        """Log-likelihood for a list of realizations."""
        return np.array([self.log_density(ev) for ev in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> "MultivariateHawkesProcessSampler":
        """Return a sampler (multivariate Ogata thinning)."""
        return MultivariateHawkesProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "MultivariateHawkesProcessEstimator":
        """Return a branching-EM estimator over the same window and dimension."""
        return MultivariateHawkesProcessEstimator(self.dim, self.window, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "MultivariateHawkesProcessDataEncoder":
        """Return the data encoder (passes realizations through; the likelihood is per-realization)."""
        return MultivariateHawkesProcessDataEncoder(self.window, self.dim)


class MultivariateHawkesProcessSampler(DistributionSampler):
    """Draw realizations by multivariate Ogata thinning."""

    def __init__(self, dist: MultivariateHawkesProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        if dist.spectral_radius >= 1.0:
            warnings.warn(
                "super-critical multivariate Hawkes (spectral radius of alpha/beta = %g >= 1): the process "
                "is non-stationary and may explode." % dist.spectral_radius,
                stacklevel=2,
            )

    def _sample_one(self) -> list[tuple[float, int]]:
        d = self.dist
        mu, alpha, beta, w = d.mu, d.alpha, d.beta, d.window
        cap = 10_000_000
        events: list[tuple[float, int]] = []
        s = np.zeros(d.dim)
        t = 0.0
        last = 0.0
        while len(events) < cap:
            lam_bar = float(mu.sum() + d._col_alpha @ s)  # total intensity decays between events -> upper bound
            t = t + self.rng.exponential(1.0 / lam_bar)
            if t >= w:
                break
            s = s * math.exp(-beta * (t - last))
            last = t
            lam_d = mu + alpha @ s  # per-mark intensities at the candidate time
            lam_total = float(lam_d.sum())
            if self.rng.uniform() <= lam_total / lam_bar:
                m = int(self.rng.choice(d.dim, p=lam_d / lam_total))
                events.append((t, m))
                s[m] += 1.0
        return events

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one marked-event realization, or ``size`` iid realizations."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(int(size))]


class MultivariateHawkesProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the multivariate branching sufficient statistics."""

    def __init__(self, dim: int, window: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.window = float(window)
        self.s0 = np.zeros(dim)  # per-mark expected immigrant counts
        self.g = np.zeros((dim, dim))  # g[d,j] = expected # of d-events triggered by j-events
        self.w_delay = 0.0  # sum over all offspring pairs of p_ik (t_i - t_k)  (for beta)
        self.mass = np.zeros(dim)  # per-mark integrated excitation mass sum_k (1-exp(-beta(W-t_k)))/beta
        self.total_window = 0.0
        self.name = name
        self.keys = keys

    def _accumulate(self, events: Any, weight: float, estimate: MultivariateHawkesProcessDistribution | None) -> None:
        times, marks = _split(events)
        n = times.size
        self.total_window += weight * self.window
        if n == 0:
            return
        if estimate is None:
            # branching-ratio-0.5 heuristic init: half immigrants, offspring split uniformly over marks
            for d in range(self.dim):
                cnt = float(np.sum(marks == d))
                self.s0[d] += weight * 0.5 * cnt
                self.g[d, :] += weight * 0.5 * cnt / self.dim
            self.w_delay += weight * 0.5 * float(times[-1] - times[0]) * max(n - 1, 0) / max(n, 1)
            self.mass += weight * np.array([np.sum(marks == j) for j in range(self.dim)]) * (self.window / 2.0)
            return
        mu, alpha, beta = estimate.mu, estimate.alpha, estimate.beta
        s = np.zeros(self.dim)
        s_delay = np.zeros(self.dim)
        prev = 0.0
        for i in range(n):
            if i > 0:
                dt = times[i] - prev
                e = math.exp(-beta * dt)
                s_delay = e * (s_delay + dt * s)
                s = e * s
            mi = marks[i]
            lam = mu[mi] + float(alpha[mi] @ s)
            self.s0[mi] += weight * mu[mi] / lam  # immigrant responsibility
            self.g[mi, :] += weight * (alpha[mi] * s) / lam  # offspring responsibilities by parent mark
            self.w_delay += weight * float(alpha[mi] @ s_delay) / lam
            s[mi] += 1.0
            prev = times[i]
        # integrated excitation mass available from each mark (edge-corrected denominator for alpha)
        contrib = (1.0 - np.exp(-beta * (self.window - times))) / beta
        for j in range(self.dim):
            self.mass[j] += weight * float(np.sum(contrib[marks == j]))

    def update(self, x: Any, weight: float, estimate: MultivariateHawkesProcessDistribution | None) -> None:
        """Accumulate branching EM statistics for one marked-event realization."""
        self._accumulate(x, weight, estimate)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize branching statistics for one marked-event realization."""
        self._accumulate(x, weight, None)

    def seq_update(self, x: list[Any], weights: np.ndarray, estimate: MultivariateHawkesProcessDistribution) -> None:
        """Accumulate branching EM statistics from encoded realizations."""
        for ev, wt in zip(x, np.asarray(weights, dtype=np.float64)):
            self._accumulate(ev, float(wt), estimate)

    def seq_initialize(self, x: list[Any], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize branching statistics from encoded realizations."""
        for ev, wt in zip(x, np.asarray(weights, dtype=np.float64)):
            self._accumulate(ev, float(wt), None)

    def combine(self, suff_stat: tuple) -> "MultivariateHawkesProcessAccumulator":
        """Merge another multivariate-Hawkes sufficient-statistic tuple."""
        s0, g, wd, mass, tw = suff_stat
        self.s0 += s0
        self.g += g
        self.w_delay += wd
        self.mass += mass
        self.total_window += tw
        return self

    def value(self) -> tuple:
        """Return immigrant, offspring, delay, mass, and window statistics."""
        return self.s0.copy(), self.g.copy(), self.w_delay, self.mass.copy(), self.total_window

    def from_value(self, x: tuple) -> "MultivariateHawkesProcessAccumulator":
        """Replace accumulator contents from branching sufficient statistics."""
        s0, g, wd, mass, tw = x
        self.s0 = np.asarray(s0, dtype=np.float64).copy()
        self.g = np.asarray(g, dtype=np.float64).copy()
        self.w_delay = float(wd)
        self.mass = np.asarray(mass, dtype=np.float64).copy()
        self.total_window = float(tw)
        self.dim = self.s0.shape[0]
        return self

    def scale(self, c: float) -> "MultivariateHawkesProcessAccumulator":
        """Scale all weight-linear sufficient statistics by ``c``."""
        self.s0 *= c
        self.g *= c
        self.w_delay *= c
        self.mass *= c
        self.total_window *= c
        return self

    def acc_to_encoder(self) -> "MultivariateHawkesProcessDataEncoder":
        """Return the marked-event encoder used by this accumulator."""
        return MultivariateHawkesProcessDataEncoder(self.window, self.dim)


class MultivariateHawkesProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MultivariateHawkesProcessAccumulator."""

    def __init__(self, dim: int, window: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.window = window
        self.name = name
        self.keys = keys

    def make(self) -> MultivariateHawkesProcessAccumulator:
        """Create a fresh multivariate-Hawkes accumulator."""
        return MultivariateHawkesProcessAccumulator(self.dim, self.window, name=self.name, keys=self.keys)


class MultivariateHawkesProcessEstimator(ParameterEstimator):
    """Veen-Schoenberg branching-EM estimator for the multivariate Hawkes parameters."""

    def __init__(self, dim: int, window: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.window = float(window)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MultivariateHawkesProcessAccumulatorFactory:
        """Return an accumulator factory for branching EM statistics."""
        return MultivariateHawkesProcessAccumulatorFactory(self.dim, self.window, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> MultivariateHawkesProcessDistribution:
        """Estimate baseline, excitation, and decay from branching statistics."""
        s0, g, w_delay, mass, total_window = suff_stat
        mu = np.maximum(s0, _MIN) / max(total_window, _MIN)
        beta = max(float(g.sum()), _MIN) / max(w_delay, _MIN)  # offspring count / delay-weighted
        alpha = g / np.maximum(mass[None, :], _MIN)  # alpha_{dj} = E[# d<-j triggers] / j integrated mass
        # keep sub-critical: scale alpha down if the spectral radius of alpha/beta reaches 1
        radius = float(np.max(np.abs(np.linalg.eigvals(alpha / beta))))
        if radius >= 1.0:
            alpha *= (1.0 - 1.0e-6) / radius
        alpha = np.maximum(alpha, 0.0)
        return MultivariateHawkesProcessDistribution(mu, alpha, beta, self.window, name=self.name, keys=self.keys)


class MultivariateHawkesProcessDataEncoder(DataSequenceEncoder):
    """Validate and pass through realizations of marked events."""

    def __init__(self, window: float, dim: int) -> None:
        self.window = float(window)
        self.dim = int(dim)

    def __str__(self) -> str:
        return "MultivariateHawkesProcessDataEncoder(%s, %s)" % (repr(self.window), repr(self.dim))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MultivariateHawkesProcessDataEncoder)
            and self.window == other.window
            and self.dim == other.dim
        )

    def seq_encode(self, x: Sequence[Any]) -> list[Any]:
        """Validate and normalize marked-event realizations."""
        out = []
        for events in x:
            times, marks = _split(events)
            if times.size and (
                np.any(~np.isfinite(times))
                or times[0] < 0.0
                or times[-1] > self.window
                or np.any(np.diff(times) < 0.0)
                or np.any(marks < 0)
                or np.any(marks >= self.dim)
            ):
                raise ValueError("events must be finite, sorted, within [0, window], with marks in [0, dim).")
            out.append([(float(t), int(m)) for t, m in zip(times, marks)])
        return out
