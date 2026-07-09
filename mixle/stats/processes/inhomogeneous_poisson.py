"""Evaluate, estimate, and sample from an inhomogeneous Poisson process with piecewise-constant rate.

Defines InhomogeneousPoissonProcessDistribution, InhomogeneousPoissonProcessSampler,
InhomogeneousPoissonProcessAccumulatorFactory, InhomogeneousPoissonProcessAccumulator,
InhomogeneousPoissonProcessEstimator, and InhomogeneousPoissonProcessDataEncoder.

Data type: each observation is a 1-D array/list of event times within the window
``[edges[0], edges[-1]]``. The intensity ``lambda(t)`` is constant ``rates[b]`` on bin ``b`` (the
bins are given by ``edges``; uniform bins on ``[0, t_max]`` by default). The log-likelihood of one
realization with per-bin event counts ``n_b`` is

    sum_b n_b * log(rates[b])  -  sum_b rates[b] * width[b],

i.e. the standard Poisson-process log-likelihood ``sum_i log lambda(t_i) - integral lambda``. The
MLE is closed-form: ``rates[b] = (total events in bin b) / (width[b] * n_realizations)``.


Reference: Daley & Vere-Jones, *An Introduction to the Theory of Point Processes* (Springer, 2003).
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

_MIN_RATE = 1.0e-12


def _resolve_edges(num_bins: int | None, t_max: float | None, edges: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if edges is not None:
        e = np.asarray(edges, dtype=np.float64)
        if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0.0):
            raise ValueError("edges must be a strictly increasing 1-D array of length >= 2.")
        return e
    if t_max is None or num_bins is None or num_bins < 1 or not (t_max > 0.0):
        raise ValueError("provide either edges, or t_max > 0 and num_bins >= 1.")
    return np.linspace(0.0, float(t_max), int(num_bins) + 1)


class InhomogeneousPoissonProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Inhomogeneous Poisson process with piecewise-constant intensity on a fixed window."""

    def __init__(
        self,
        rates: Sequence[float] | np.ndarray,
        t_max: float | None = None,
        edges: Sequence[float] | np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a piecewise-constant-rate Poisson process.

        Args:
            rates (Sequence[float]): Non-negative per-bin intensities (length K).
            t_max (Optional[float]): Window upper bound for uniform bins on ``[0, t_max]`` (used when
                ``edges`` is not given); ``K`` equal-width bins are created.
            edges (Optional[Sequence[float]]): Explicit strictly-increasing bin edges (length K+1),
                overriding ``t_max``.
            name (Optional[str]): Optional object name.
            keys (Optional[str]): Optional parameter key.
        """
        self.rates = np.asarray(rates, dtype=np.float64)
        if (
            self.rates.ndim != 1
            or self.rates.size < 1
            or np.any(self.rates < 0.0)
            or not np.all(np.isfinite(self.rates))
        ):
            raise ValueError("rates must be a 1-D array of finite non-negative intensities.")
        self.edges = _resolve_edges(self.rates.size, t_max, edges)
        if self.edges.size - 1 != self.rates.size:
            raise ValueError("len(edges) must equal len(rates) + 1.")
        self.widths = np.diff(self.edges)
        self.num_bins = self.rates.size
        self.t_min = float(self.edges[0])
        self.t_max = float(self.edges[-1])
        self.name = name
        self.keys = keys
        with np.errstate(divide="ignore"):
            self._log_rates = np.where(self.rates > 0.0, np.log(self.rates), -np.inf)
        self._integral = float(np.sum(self.rates * self.widths))

    def __str__(self) -> str:
        """Return a constructor-style representation of the inhomogeneous Poisson process."""
        return "InhomogeneousPoissonProcessDistribution(%s, edges=%s, name=%s, keys=%s)" % (
            repr(list(self.rates)),
            repr(list(self.edges)),
            repr(self.name),
            repr(self.keys),
        )

    def intensity(self, t: float, times: Any = None, marks: Any = None) -> float:
        """Conditional rate ``lambda(t) = rates[bin containing t]``.

        The inhomogeneous Poisson process is **not** self-exciting, so the rate depends only on ``t``.
        ``times``/``marks`` are accepted for ``TemporalPointProcess`` signature parity and ignored.
        Raises ``ValueError`` for ``t`` outside the support ``[edges[0], edges[-1]]``.
        """
        t = float(t)
        if not (self.t_min <= t <= self.t_max):
            raise ValueError("intensity queried outside the process window [edges[0], edges[-1]].")
        # bin b is [edges[b], edges[b+1]); clamp the right endpoint into the last bin
        b = int(np.searchsorted(self.edges, t, side="right") - 1)
        b = min(max(b, 0), self.num_bins - 1)
        return float(self.rates[b])

    def expected_count(self, t_start: float, t_end: float, times: Any = None, marks: Any = None) -> float:
        """Compensator ``integral_{t_start}^{t_end} lambda(s) ds`` -- the piecewise-rate integral.

        Computed as ``sum_b rate_b * width(overlap([t_start, t_end], bin_b))``. ``times``/``marks`` are
        accepted for signature parity and ignored. With ``t_start=edges[0], t_end=edges[-1]`` this returns
        the full integral ``sum_b rate_b width_b`` used by ``log_density``.
        """
        a, b = float(t_start), float(t_end)
        if b <= a:
            return 0.0
        lo = np.maximum(self.edges[:-1], a)
        hi = np.minimum(self.edges[1:], b)
        overlap = np.clip(hi - lo, 0.0, None)
        return float(np.sum(self.rates * overlap))

    def _bin_counts(self, events: Any) -> np.ndarray | None:
        ev = np.asarray(events, dtype=np.float64).reshape(-1)
        if ev.size and (np.any(~np.isfinite(ev)) or np.any(ev < self.t_min) or np.any(ev > self.t_max)):
            return None
        counts, _ = np.histogram(ev, bins=self.edges)
        return counts.astype(np.float64)

    def density(self, x: Any) -> float:
        """Probability density of one realization ``x`` (a sequence of event times)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Log-likelihood of one realization: ``sum_b n_b log rate_b - sum_b rate_b width_b``."""
        counts = self._bin_counts(x)
        if counts is None:
            return -np.inf
        return self._counts_log_density(counts)

    def _counts_log_density(self, counts: np.ndarray) -> float:
        emitted = np.where(counts > 0.0, counts * self._log_rates, 0.0)
        if np.any(~np.isfinite(emitted)):  # an event landed in a zero-rate bin
            return -np.inf
        return float(np.sum(emitted) - self._integral)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-likelihood for a ``(num_realizations, num_bins)`` matrix of per-bin counts."""
        counts = np.asarray(x, dtype=np.float64)
        emitted = np.where(counts > 0.0, counts * self._log_rates[None, :], 0.0)
        rv = np.sum(emitted, axis=1) - self._integral
        rv[np.any(~np.isfinite(emitted), axis=1)] = -np.inf
        return rv

    def sampler(self, seed: int | None = None) -> "InhomogeneousPoissonProcessSampler":
        """Return an InhomogeneousPoissonProcessSampler for this distribution."""
        return InhomogeneousPoissonProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "InhomogeneousPoissonProcessEstimator":
        """Return an InhomogeneousPoissonProcessEstimator over the same bin edges."""
        return InhomogeneousPoissonProcessEstimator(edges=self.edges, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "InhomogeneousPoissonProcessDataEncoder":
        """Returns an InhomogeneousPoissonProcessDataEncoder bound to these bin edges."""
        return InhomogeneousPoissonProcessDataEncoder(self.edges)


class InhomogeneousPoissonProcessSampler(DistributionSampler):
    """Draw realizations by binwise thinning: ``n_b ~ Poisson(rate_b * width_b)`` then uniform times."""

    def __init__(self, dist: InhomogeneousPoissonProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self) -> np.ndarray:
        d = self.dist
        counts = self.rng.poisson(lam=d.rates * d.widths)
        times = []
        for b in range(d.num_bins):
            if counts[b] > 0:
                times.append(self.rng.uniform(d.edges[b], d.edges[b + 1], size=int(counts[b])))
        events = np.concatenate(times) if times else np.empty(0, dtype=np.float64)
        events.sort()
        return events

    def sample(self, size: int | None = None) -> np.ndarray | list[np.ndarray]:
        """Draw one realization (event-time array) or a list of ``size`` realizations."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(int(size))]


class InhomogeneousPoissonProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted per-bin event counts and the weighted number of realizations."""

    def __init__(self, edges: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        self.edges = np.asarray(edges, dtype=np.float64)
        self.num_bins = self.edges.size - 1
        self.bin_counts = np.zeros(self.num_bins, dtype=np.float64)
        self.n_realizations = 0.0
        self.name = name
        self.keys = keys

    def _counts(self, x: Any) -> np.ndarray:
        ev = np.asarray(x, dtype=np.float64).reshape(-1)
        counts, _ = np.histogram(ev, bins=self.edges)
        return counts.astype(np.float64)

    def update(self, x: Any, weight: float, estimate: InhomogeneousPoissonProcessDistribution | None) -> None:
        """Accumulate weighted per-bin counts for one event-time realization."""
        self.bin_counts += weight * self._counts(x)
        self.n_realizations += weight

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted realization."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted per-bin counts from encoded realizations."""
        counts = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        self.bin_counts += ww @ counts
        self.n_realizations += ww.sum()

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded realizations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "InhomogeneousPoissonProcessAccumulator":
        """Merge serialized bin-count statistics into this accumulator."""
        self.bin_counts += suff_stat[0]
        self.n_realizations += suff_stat[1]
        return self

    def value(self) -> tuple[np.ndarray, float]:
        """Return the weighted bin counts and weighted realization count."""
        return self.bin_counts.copy(), self.n_realizations

    def from_value(self, x: tuple[np.ndarray, float]) -> "InhomogeneousPoissonProcessAccumulator":
        """Restore the accumulator from serialized bin-count statistics."""
        self.bin_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.n_realizations = float(x[1])
        self.num_bins = self.bin_counts.size
        return self

    def scale(self, c: float) -> "InhomogeneousPoissonProcessAccumulator":
        """Scale accumulated sufficient statistics by a constant."""
        self.bin_counts *= c
        self.n_realizations *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into a keyed statistics dictionary."""
        if self.keys is not None:
            if self.keys in stats_dict:
                bc, nr = stats_dict[self.keys]
                self.bin_counts += bc
                self.n_realizations += nr
            else:
                stats_dict[self.keys] = (self.bin_counts.copy(), self.n_realizations)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from a keyed statistics dictionary."""
        if self.keys is not None and self.keys in stats_dict:
            bc, nr = stats_dict[self.keys]
            self.bin_counts = np.asarray(bc, dtype=np.float64).copy()
            self.n_realizations = float(nr)

    def acc_to_encoder(self) -> "InhomogeneousPoissonProcessDataEncoder":
        """Return an encoder that bins event times on this accumulator's edges."""
        return InhomogeneousPoissonProcessDataEncoder(self.edges)


class InhomogeneousPoissonProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for InhomogeneousPoissonProcessAccumulator."""

    def __init__(self, edges: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        self.edges = np.asarray(edges, dtype=np.float64)
        self.name = name
        self.keys = keys

    def make(self) -> "InhomogeneousPoissonProcessAccumulator":
        """Create an empty inhomogeneous Poisson process accumulator."""
        return InhomogeneousPoissonProcessAccumulator(self.edges, name=self.name, keys=self.keys)


class InhomogeneousPoissonProcessEstimator(ParameterEstimator):
    """Closed-form MLE: ``rate_b = (weighted events in bin b) / (width_b * weighted realizations)``."""

    def __init__(
        self,
        num_bins: int | None = None,
        t_max: float | None = None,
        edges: Sequence[float] | np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.edges = _resolve_edges(num_bins, t_max, edges)
        self.widths = np.diff(self.edges)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "InhomogeneousPoissonProcessAccumulatorFactory":
        """Return a factory for inhomogeneous Poisson sufficient-statistic accumulators."""
        return InhomogeneousPoissonProcessAccumulatorFactory(self.edges, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, float]
    ) -> "InhomogeneousPoissonProcessDistribution":
        """Estimate per-bin rates from accumulated ``(bin_counts, n_realizations)``."""
        bin_counts, n_realizations = suff_stat
        denom = max(float(n_realizations), _MIN_RATE)
        rates = np.asarray(bin_counts, dtype=np.float64) / (self.widths * denom)
        return InhomogeneousPoissonProcessDistribution(rates, edges=self.edges, name=self.name, keys=self.keys)


class InhomogeneousPoissonProcessDataEncoder(DataSequenceEncoder):
    """Encode a list of realizations (event-time arrays) into a ``(num_realizations, num_bins)`` count matrix."""

    def __init__(self, edges: Sequence[float] | np.ndarray) -> None:
        self.edges = np.asarray(edges, dtype=np.float64)

    def __str__(self) -> str:
        return "InhomogeneousPoissonProcessDataEncoder(%s)" % repr(list(self.edges))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InhomogeneousPoissonProcessDataEncoder) and np.array_equal(self.edges, other.edges)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Encode event-time realizations as per-bin count rows."""
        t_min = float(self.edges[0])
        t_max = float(self.edges[-1])
        rows = []
        for events in x:
            ev = np.asarray(events, dtype=np.float64).reshape(-1)
            if ev.size and (np.any(~np.isfinite(ev)) or np.any(ev < t_min) or np.any(ev > t_max)):
                raise ValueError("event times must be finite and within the process window [edges[0], edges[-1]].")
            counts, _ = np.histogram(ev, bins=self.edges)
            rows.append(counts.astype(np.float64))
        return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, self.edges.size - 1), dtype=np.float64)
