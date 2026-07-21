"""A self-exciting (Hawkes) point process with a power-law triggering kernel and productivity marks.

The library's :class:`~mixle.stats.HawkesProcessDistribution` uses an *exponential* triggering kernel,
whose memorylessness gives an O(n) recursion. Many self-exciting processes instead trigger with a heavy-
tailed **power-law** kernel ``g(s) = (1 + s/c)^{-p}`` (long memory; the events keep mattering far into the
future), and many are **marked** -- each event carries a value ``m_i`` that scales how strongly it excites
the future, ``productivity = A * exp(alpha * m_i)``. This distribution covers that general case. The
conditional intensity

    lambda(t) = mu + sum_{t_j < t} A e^{alpha m_j} (1 + (t - t_j)/c)^{-p}

is the forecast rate; ``log_density`` is the exact realization likelihood, ``sampler`` draws catalogues by
branching, and the estimator fits ``(mu, A, alpha, c, p)`` by maximum likelihood. Domain-neutral: an event
catalogue is just ``(times, marks)`` on a window ``[0, T]``.


Reference: Hawkes, 'Spectra of some self-exciting and mutually exciting point processes', Biometrika (1971).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.optimize import minimize

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

__all__ = ["PowerLawHawkesDistribution", "PowerLawHawkesEstimator"]


class PowerLawHawkesDistribution(SequenceEncodableProbabilityDistribution):
    """Marked power-law-kernel Hawkes process on a fixed window ``[0, window]``.

    A realization is ``(times, marks)`` -- a sorted event-time array and a matching mark array (use zeros,
    or omit, for the unmarked process). ``mu > 0`` is the background rate, ``A >= 0`` the productivity,
    ``alpha`` the mark sensitivity, and ``c > 0``, ``p > 1`` the Omori-Utsu kernel scale and exponent.
    """

    def __init__(self, mu, A, c, p, window, *, alpha=0.0, mark_dist=None, name=None, keys=None):
        self.mu, self.A, self.alpha, self.c, self.p, self.window = (float(v) for v in (mu, A, alpha, c, p, window))
        if not (self.mu > 0 and self.A >= 0 and self.c > 0 and self.p > 1 and self.window > 0):
            raise ValueError("PowerLawHawkes requires mu>0, A>=0, c>0, p>1, window>0.")
        self.mark_dist = mark_dist  # an optional mixle distribution the marks are drawn from (None => unmarked)
        self.name = name
        self.keys = keys

    def __str__(self):
        return "PowerLawHawkesDistribution(%r, %r, %r, %r, %r, alpha=%r)" % (
            self.mu,
            self.A,
            self.c,
            self.p,
            self.window,
            self.alpha,
        )

    @staticmethod
    def _unpack(x) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(x, tuple):
            t, m = x
            return np.asarray(t, dtype=float).reshape(-1), np.asarray(m, dtype=float).reshape(-1)
        t = np.asarray(x, dtype=float).reshape(-1)
        return t, np.zeros_like(t)

    def intensity(self, t: float, times, marks=None) -> float:
        """The conditional rate ``lambda(t)`` given the catalogue so far -- the instantaneous forecast."""
        times = np.asarray(times, dtype=float)
        marks = np.zeros_like(times) if marks is None else np.asarray(marks, dtype=float)
        past = times < t
        trig = self.A * np.exp(self.alpha * marks[past]) * (1.0 + (t - times[past]) / self.c) ** (-self.p)
        return float(self.mu + trig.sum())

    def expected_count(self, t_start: float, t_end: float, times, marks=None) -> float:
        """Expected number of events in ``[t_start, t_end]`` given the catalogue -- the window forecast."""
        times = np.asarray(times, dtype=float)
        marks = np.zeros_like(times) if marks is None else np.asarray(marks, dtype=float)
        rel = times < t_end
        tp, mp = times[rel], marks[rel]
        prod = self.A * np.exp(self.alpha * mp) * self.c / (self.p - 1.0)
        lo = np.maximum(t_start, tp)
        omori = (1.0 + (lo - tp) / self.c) ** (1.0 - self.p) - (1.0 + (t_end - tp) / self.c) ** (1.0 - self.p)
        return float(self.mu * (t_end - t_start) + np.sum(prod * omori))

    def branching_ratio(self, mean_mark: float = 0.0) -> float:
        """Expected direct offspring per event ``A c/(p-1) e^{alpha * mean_mark}`` -- criticality (<1 stable)."""
        return self.A * self.c / (self.p - 1.0) * np.exp(self.alpha * mean_mark)

    def density(self, x) -> float:
        """Return the realization likelihood as a density on event times."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x) -> float:
        """Exact log-likelihood of one realization on ``[0, window]``."""
        t, m = self._unpack(x)
        if t.size and (np.any(~np.isfinite(t)) or t[0] < 0 or t[-1] > self.window or np.any(np.diff(t) < 0)):
            return -np.inf
        prod = self.A * np.exp(self.alpha * m)
        lam = np.full(t.size, self.mu)
        for j in range(1, t.size):  # O(n^2): the power-law kernel has no finite-state recursion
            lam[j] += np.sum(prod[:j] * (1.0 + (t[j] - t[:j]) / self.c) ** (-self.p))
        integral = self.mu * self.window + np.sum(
            prod * self.c / (self.p - 1.0) * (1.0 - (1.0 + (self.window - t) / self.c) ** (1.0 - self.p))
        )
        return float(np.sum(np.log(lam)) - integral)

    def seq_log_density(self, x) -> np.ndarray:
        """Return log-likelihoods for a batch of point-process realizations."""
        return np.array([self.log_density(r) for r in x])

    def sampler(self, seed: int | None = None) -> PowerLawHawkesSampler:
        """Return a branching-process sampler for event catalogues."""
        return PowerLawHawkesSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> PowerLawHawkesEstimator:
        """Return the maximum-likelihood estimator for realizations on this window."""
        return PowerLawHawkesEstimator(self.window, alpha_fixed=self.alpha if self.alpha == 0.0 else None)

    def dist_to_encoder(self) -> PowerLawHawkesDataEncoder:
        """Return the pass-through realization encoder used by vectorized methods."""
        return PowerLawHawkesDataEncoder()


class PowerLawHawkesSampler(DistributionSampler):
    """Draw catalogues by branching: Poisson background + power-law-distributed offspring."""

    def __init__(self, dist: PowerLawHawkesDistribution, seed: int | None = None):
        self.dist = dist
        self.rng = RandomState(seed)
        self._mark = dist.mark_dist.sampler(self.rng.randint(0, 2**31 - 1)) if dist.mark_dist is not None else None

    def _draw_mark(self) -> float:
        return 0.0 if self._mark is None else float(self._mark.sample())

    def _sample_one(self):
        d = self.dist
        times = list(self.rng.uniform(0, d.window, self.rng.poisson(d.mu * d.window)))
        marks = [self._draw_mark() for _ in times]
        queue = list(zip(times, marks))
        while queue:
            ti, mi = queue.pop()
            expected = d.A * np.exp(d.alpha * mi) * d.c / (d.p - 1.0)
            for _ in range(self.rng.poisson(expected)):
                tau = d.c * ((1 - self.rng.uniform()) ** (-1.0 / (d.p - 1.0)) - 1.0)  # power-law inter-time
                tc = ti + tau
                if tc < d.window:
                    mc = self._draw_mark()
                    times.append(tc)
                    marks.append(mc)
                    queue.append((tc, mc))
        order = np.argsort(times)
        return np.asarray(times)[order], np.asarray(marks)[order]

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one catalogue or a list of catalogues by the branching construction."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(int(size))]


class PowerLawHawkesDataEncoder(DataSequenceEncoder):
    """Pass-through encoder: a realization is a ``(times, marks)`` tuple; a batch is a list of them."""

    def __str__(self) -> str:
        return "PowerLawHawkesDataEncoder()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PowerLawHawkesDataEncoder)

    def seq_encode(self, x):
        """Return the batch of realizations as a list."""
        return list(x)


class PowerLawHawkesEstimator(ParameterEstimator):
    """Maximum-likelihood estimator of ``(mu, A, alpha, c, p)`` over realizations on a common window."""

    def __init__(self, window: float, *, alpha_fixed: float | None = None, name=None, keys=None):
        self.window = float(window)
        self.alpha_fixed = alpha_fixed
        self.name = name
        self.keys = keys

    def accumulator_factory(self):
        """Return a factory for raw-realization Hawkes accumulators."""
        return PowerLawHawkesAccumulatorFactory(self.window, self.alpha_fixed, name=self.name, keys=self.keys)

    def estimate(self, nobs, suff_stat) -> PowerLawHawkesDistribution:
        """Fit Hawkes parameters by numerical maximum likelihood."""
        realizations, window, alpha_fixed = suff_stat
        unmarked = alpha_fixed is not None
        unpack = PowerLawHawkesDistribution._unpack
        data = [unpack(r) for r in realizations]
        n_total = sum(len(t) for t, _ in data)

        def negll(theta):
            mu, a, c, pm1 = np.exp(theta[[0, 1, 3, 4]])
            alpha = alpha_fixed if unmarked else theta[2]
            d = PowerLawHawkesDistribution(mu, a, c, 1.0 + pm1, window, alpha=alpha)
            ll = sum(d.log_density(r) for r in data)
            return -ll if np.isfinite(ll) else 1e12

        x0 = [np.log(max(n_total / (len(data) * window), 1e-3)), np.log(0.5), 0.0, np.log(0.02), np.log(0.3)]
        bounds = [
            (np.log(1e-4), np.log(1e3)),
            (np.log(1e-4), np.log(1e3)),
            (-4.0, 4.0),
            (np.log(1e-4), np.log(10.0)),
            (np.log(1e-3), np.log(5.0)),
        ]
        res = minimize(negll, x0, method="L-BFGS-B", bounds=bounds)
        mu, a, c, pm1 = np.exp(res.x[[0, 1, 3, 4]])
        alpha = alpha_fixed if unmarked else res.x[2]
        return PowerLawHawkesDistribution(mu, a, c, 1.0 + pm1, window, alpha=alpha, name=self.name)


class PowerLawHawkesAccumulator(SequenceEncodableStatisticAccumulator):
    """Collects realizations (the MLE needs the full event times, not closed-form sufficient statistics)."""

    def __init__(self, window: float, alpha_fixed: float | None, name=None, keys=None):
        self.window = float(window)
        self.alpha_fixed = alpha_fixed
        self.realizations: list[Any] = []
        self.name = name
        self.keys = keys

    def update(self, x, weight, estimate):
        """Store one realization for maximum-likelihood fitting."""
        self.realizations.append(x)

    def initialize(self, x, weight, rng):
        """Store one realization during initialization."""
        self.realizations.append(x)

    def seq_update(self, x, weights, estimate):
        """Store a batch of realizations for maximum-likelihood fitting."""
        self.realizations.extend(x)

    def seq_initialize(self, x, weights, rng):
        """Store a batch of realizations during initialization."""
        self.realizations.extend(x)

    def combine(self, suff_stat):
        """Merge stored realization catalogues."""
        self.realizations.extend(suff_stat[0])
        return self

    def scale(self, c: float) -> PowerLawHawkesAccumulator:
        """Leave raw catalogues unchanged because they are not weighted sufficient statistics."""
        # The accumulator stores raw realizations (no weighted sufficient statistics), so there is
        # nothing meaningful to rescale; keep it a safe no-op rather than corrupting the catalogue.
        return self

    def value(self):
        """Return stored realizations with the shared window and alpha constraint."""
        return (self.realizations, self.window, self.alpha_fixed)

    def from_value(self, x):
        """Restore stored realizations, window, and alpha constraint."""
        # copy on restore: assigning the pooled LIST reference would alias every tied site to one
        # list, so a later in-place extend at any site mutates all of them (copy-on-adopt precedent)
        self.realizations = list(x[0])
        self.window, self.alpha_fixed = x[1], x[2]
        return self

    def acc_to_encoder(self):
        """Return the encoder compatible with raw Hawkes realizations."""
        return PowerLawHawkesDataEncoder()


class PowerLawHawkesAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for PowerLawHawkesAccumulator."""

    def __init__(self, window: float, alpha_fixed: float | None, name=None, keys=None):
        self.window = float(window)
        self.alpha_fixed = alpha_fixed
        self.name = name
        self.keys = keys

    def make(self) -> PowerLawHawkesAccumulator:
        """Create an empty power-law Hawkes accumulator."""
        return PowerLawHawkesAccumulator(self.window, self.alpha_fixed, name=self.name, keys=self.keys)
