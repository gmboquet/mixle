"""Earthquake forecasting: Gutenberg-Richter magnitudes and the ETAS self-exciting point process.

Earthquakes are not deterministically predictable, but they are *probabilistically forecastable*: the
operational model is ETAS (Epidemic-Type Aftershock Sequence), a self-exciting marked point process where
each event raises the rate of future events through an Omori-Utsu aftershock decay scaled by the parent's
magnitude, on top of a constant background rate. Its conditional intensity ``lambda(t | history)`` *is* the
forecast -- the instantaneous rate, hence the probability of an event in any time-magnitude window.
Magnitudes follow the Gutenberg-Richter law. Part of the earth-science/UQ work.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["GutenbergRichter", "ETAS"]

LOG10 = np.log(10.0)


class GutenbergRichter:
    """The magnitude-frequency law ``P(M >= m) = 10^{-b (m - m0)}`` for ``m >= m0`` (completeness).

    Equivalently an exponential of rate ``beta = b ln 10`` above the threshold ``m0``. The ``b``-value
    (typically near 1) controls how fast large events become rarer.
    """

    def __init__(self, b: float = 1.0, m0: float = 0.0):
        self.b = float(b)
        self.m0 = float(m0)
        self.beta = self.b * LOG10

    def log_density(self, m):
        m = np.asarray(m, dtype=float)
        return np.where(m >= self.m0, np.log(self.beta) - self.beta * (m - self.m0), -np.inf)

    def sampler(self, seed: int | None = None) -> GutenbergRichterSampler:
        return GutenbergRichterSampler(self, seed)

    @classmethod
    def fit(cls, magnitudes, m0: float | None = None) -> GutenbergRichter:
        """Aki maximum-likelihood ``b``-value: ``b = log10(e) / (mean(m) - m0)``."""
        m = np.asarray(magnitudes, dtype=float)
        m0 = float(m.min()) if m0 is None else float(m0)
        b = np.log10(np.e) / (m.mean() - m0)
        return cls(b, m0)


class GutenbergRichterSampler:
    def __init__(self, dist: GutenbergRichter, seed=None):
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None):
        n = 1 if size is None else size
        m = self.dist.m0 + self.rng.exponential(1.0 / self.dist.beta, n)
        return float(m[0]) if size is None else m


class ETAS:
    """Temporal ETAS earthquake-forecasting model (with magnitude marks).

    Conditional intensity ``lambda(t) = mu + sum_{t_i < t} A e^{alpha (m_i - m0)} (1 + (t - t_i)/c)^{-p}``:
    a constant background rate ``mu`` plus an Omori-Utsu aftershock burst from every past event, whose size
    grows with the parent magnitude (productivity ``A``, magnitude scaling ``alpha``) and decays in time
    (``c``, ``p``). Fit it to a catalogue, read the forecast off :meth:`intensity` / :meth:`expected_count`,
    and simulate synthetic catalogues.
    """

    def __init__(self, mu=1.0, A=0.5, alpha=1.0, c=0.01, p=1.2, m0=0.0):
        self.mu, self.A, self.alpha, self.c, self.p, self.m0 = (float(v) for v in (mu, A, alpha, c, p, m0))

    def intensity(self, t: float, times, mags) -> float:
        """The conditional rate at time ``t`` given the catalogue ``(times, mags)`` -- the forecast rate."""
        times, mags = np.asarray(times, dtype=float), np.asarray(mags, dtype=float)
        past = times < t
        tp, mp = times[past], mags[past]
        trig = self.A * np.exp(self.alpha * (mp - self.m0)) * (1.0 + (t - tp) / self.c) ** (-self.p)
        return float(self.mu + trig.sum())

    def expected_count(self, t_start: float, t_end: float, times, mags) -> float:
        """Forecast: the expected number of events in ``[t_start, t_end]`` given the catalogue so far
        (the integral of the conditional intensity over the window, holding history fixed)."""
        times, mags = np.asarray(times, dtype=float), np.asarray(mags, dtype=float)
        rel = times < t_end  # every event before the window's end can contribute aftershocks to it
        tp, mp = times[rel], mags[rel]
        prod = self.A * np.exp(self.alpha * (mp - self.m0)) * self.c / (self.p - 1.0)
        lo = np.maximum(t_start, tp)  # integrate each event's Omori kernel over its overlap with the window
        omori = (1.0 + (lo - tp) / self.c) ** (1.0 - self.p) - (1.0 + (t_end - tp) / self.c) ** (1.0 - self.p)
        return float(self.mu * (t_end - t_start) + np.sum(prod * omori))

    def branching_ratio(self, mean_magnitude: float | None = None) -> float:
        """Mean offspring per event ``n = A c/(p-1) E[e^{alpha(m-m0)}]`` -- the criticality (must be < 1)."""
        fac = 1.0 if mean_magnitude is None else np.exp(self.alpha * (mean_magnitude - self.m0))
        return self.A * self.c / (self.p - 1.0) * fac

    def log_likelihood(self, times, mags, t_end: float) -> float:
        times, mags = np.asarray(times, dtype=float), np.asarray(mags, dtype=float)
        order = np.argsort(times)
        times, mags = times[order], mags[order]
        prod_m = self.A * np.exp(self.alpha * (mags - self.m0))  # per-event productivity
        lam = np.full(len(times), self.mu)
        for j in range(1, len(times)):  # sum-of-log-intensities (lower-triangular trigger sum)
            dt = times[j] - times[:j]
            lam[j] += np.sum(prod_m[:j] * (1.0 + dt / self.c) ** (-self.p))
        integral = self.mu * t_end + np.sum(
            prod_m * self.c / (self.p - 1.0) * (1.0 - (1.0 + (t_end - times) / self.c) ** (1.0 - self.p))
        )
        return float(np.sum(np.log(lam)) - integral)

    @classmethod
    def fit(cls, times, mags, t_end: float, m0: float = 0.0) -> ETAS:
        """Maximum-likelihood fit of ``(mu, A, alpha, c, p)`` to a catalogue.

        Parameters are optimized in a bounded log space (positivity + a stationarity-friendly range); the
        objective is guarded so the optimizer never wanders into overflow."""
        times, mags = np.asarray(times, dtype=float), np.asarray(mags, dtype=float)

        def negll(theta):
            lmu, la, alpha, lc, lpm1 = theta
            model = cls(np.exp(lmu), np.exp(la), alpha, np.exp(lc), 1.0 + np.exp(lpm1), m0)
            nll = -model.log_likelihood(times, mags, t_end)
            return nll if np.isfinite(nll) else 1e12

        x0 = [np.log(max(len(times) / t_end, 1e-3)), np.log(0.5), 1.0, np.log(0.02), np.log(0.3)]
        bounds = [
            (np.log(1e-4), np.log(1e3)),  # mu
            (np.log(1e-4), np.log(1e3)),  # A
            (0.0, 4.0),  # alpha
            (np.log(1e-4), np.log(10.0)),  # c
            (np.log(1e-3), np.log(5.0)),  # p - 1
        ]
        res = minimize(negll, x0, method="L-BFGS-B", bounds=bounds)
        lmu, la, alpha, lc, lpm1 = res.x
        return cls(np.exp(lmu), np.exp(la), alpha, np.exp(lc), 1.0 + np.exp(lpm1), m0)

    def simulate(self, t_end: float, *, b: float = 1.0, seed: int | None = None):
        """Simulate a catalogue on ``[0, t_end]`` by branching: Poisson background + Omori-distributed
        aftershocks (magnitudes from Gutenberg-Richter). Returns ``(times, magnitudes)`` time-sorted."""
        rng = np.random.RandomState(seed)
        gr = GutenbergRichter(b, self.m0)
        n_bg = rng.poisson(self.mu * t_end)
        times = list(rng.uniform(0, t_end, n_bg))
        mags = list(self.m0 + rng.exponential(1.0 / (b * LOG10), n_bg))
        queue = list(zip(times, mags))
        while queue:
            t_i, m_i = queue.pop()
            expected = self.A * np.exp(self.alpha * (m_i - self.m0)) * self.c / (self.p - 1.0)
            for _ in range(rng.poisson(expected)):
                tau = self.c * ((1 - rng.uniform()) ** (-1.0 / (self.p - 1.0)) - 1.0)  # Omori inter-time
                t_child = t_i + tau
                if t_child < t_end:
                    m_child = self.m0 + rng.exponential(1.0 / (b * LOG10))
                    times.append(t_child)
                    mags.append(m_child)
                    queue.append((t_child, m_child))
        order = np.argsort(times)
        return np.asarray(times)[order], np.asarray(mags)[order]
