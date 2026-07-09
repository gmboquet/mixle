"""Evaluate, estimate, and sample from a univariate Hawkes process with an exponential kernel.

Defines HawkesProcessDistribution, HawkesProcessSampler, HawkesProcessAccumulatorFactory,
HawkesProcessAccumulator, HawkesProcessEstimator, and HawkesProcessDataEncoder.

A Hawkes process is a *self-exciting* temporal point process: every event transiently raises the
intensity of future events. With the exponential triggering kernel ``g(s) = alpha * exp(-beta s)``
the conditional intensity given the history is

    lambda(t) = mu + sum_{t_j < t} alpha * exp(-beta (t - t_j)),

with background rate ``mu > 0``, excitation jump ``alpha >= 0``, and decay rate ``beta > 0``. The
branching ratio ``alpha / beta`` is the expected number of direct offspring per event; the process
is sub-critical (stationary) when ``alpha < beta``.

Data type: each observation is a 1-D array/list of event times, sorted, lying in the fixed window
``[0, window]``. The exact log-likelihood of one realization ``t_1 < ... < t_n`` is the standard
point-process log-likelihood ``sum_i log lambda(t_i) - integral_0^window lambda(s) ds``:

    sum_i log(mu + alpha R_i) - mu*window - (alpha/beta) sum_i (1 - exp(-beta (window - t_i))),

where ``R_i = sum_{j<i} exp(-beta (t_i - t_j))`` obeys the O(n) recursion
``R_i = exp(-beta (t_i - t_{i-1})) (R_{i-1} + 1)`` (``R_1 = 0``).

Fitting uses the Veen-Schoenberg / Lewis-Mohler EM over the latent branching structure (each event
is an immigrant from ``mu`` or an offspring of an earlier event). The E-step responsibilities and
the offspring-delay statistic are accumulated in O(n) via a companion recursion, giving finite
sufficient statistics ``(S0, G, W, n_events, total_window)`` and the closed-form M-step
``mu = S0/total_window``, ``beta = G/W``, ``alpha = beta * G / n_events`` (the standard
edge-effect-free branching estimator; exact as ``window -> inf``).


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


class HawkesProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Univariate Hawkes process with an exponential excitation kernel on a fixed window."""

    def __init__(
        self,
        mu: float,
        alpha: float,
        beta: float,
        window: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an exponential-kernel Hawkes process.

        Args:
            mu (float): Background (immigrant) intensity, ``mu > 0``.
            alpha (float): Excitation jump of the triggering kernel, ``alpha >= 0``.
            beta (float): Exponential decay rate of the triggering kernel, ``beta > 0``.
            window (float): Length ``T`` of the observation window ``[0, T]``, ``window > 0``.
            name (Optional[str]): Optional object name.
            keys (Optional[str]): Optional parameter key.
        """
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.window = float(window)
        if not (self.mu > 0.0 and self.alpha >= 0.0 and self.beta > 0.0 and self.window > 0.0):
            raise ValueError("Hawkes process requires mu>0, alpha>=0, beta>0, window>0.")
        self.name = name
        self.keys = keys
        self.branching_ratio = self.alpha / self.beta

    def __str__(self) -> str:
        """Return a constructor-style representation of the Hawkes process distribution."""
        return "HawkesProcessDistribution(%s, %s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.alpha),
            repr(self.beta),
            repr(self.window),
            repr(self.name),
            repr(self.keys),
        )

    def intensity(self, t: float, times: Any, marks: Any = None) -> float:
        """Conditional rate ``lambda(t) = mu + alpha sum_{t_i < t} exp(-beta (t - t_i))`` given the history.

        ``times`` is the event history; ``marks`` is accepted for ``TemporalPointProcess`` signature
        parity (the univariate Hawkes process is unmarked) and is ignored.
        """
        ti = np.asarray(times, dtype=np.float64).reshape(-1)
        past = ti[ti < t]
        return float(self.mu + self.alpha * np.sum(np.exp(-self.beta * (t - past))))

    def expected_count(self, t_start: float, t_end: float, times: Any, marks: Any = None) -> float:
        """Compensator ``integral_{t_start}^{t_end} lambda(s) ds`` of the intensity given the history.

        ``marks`` is accepted for signature parity and ignored (the univariate process is unmarked).
        """
        ti = np.asarray(times, dtype=np.float64).reshape(-1)
        tp = ti[ti < t_end]
        lo = np.maximum(t_start, tp)
        kernel = np.exp(-self.beta * (lo - tp)) - np.exp(-self.beta * (t_end - tp))
        return float(self.mu * (t_end - t_start) + (self.alpha / self.beta) * np.sum(kernel))

    def density(self, x: Any) -> float:
        """Probability density of one realization ``x`` (a sequence of event times)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Exact log-likelihood of one realization (a sorted event-time sequence in ``[0, window]``)."""
        t = np.asarray(x, dtype=np.float64).reshape(-1)
        if t.size and (np.any(~np.isfinite(t)) or t[0] < 0.0 or t[-1] > self.window or np.any(np.diff(t) < 0.0)):
            return -np.inf
        mu, alpha, beta, w = self.mu, self.alpha, self.beta, self.window
        loglam = 0.0
        r = 0.0
        prev = 0.0
        for i in range(t.size):
            r = math.exp(-beta * (t[i] - prev)) * (r + 1.0) if i > 0 else 0.0
            loglam += math.log(mu + alpha * r)
            prev = t[i]
        compensator = mu * w + (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (w - t)))) if t.size else mu * w
        return float(loglam - compensator)

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
        """Vectorized exact log-likelihood over a padded ``(num_realizations, max_len)`` time matrix."""
        times, lengths, window = x
        times = np.asarray(times, dtype=np.float64)
        lengths = np.asarray(lengths, dtype=np.int64)
        n = lengths.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        mu, alpha, beta = self.mu, self.alpha, self.beta
        max_len = times.shape[1]
        r = np.zeros(n, dtype=np.float64)
        loglam = np.zeros(n, dtype=np.float64)
        idx = np.arange(max_len)
        for i in range(max_len):
            active = idx[i] < lengths
            if i == 0:
                ri = np.zeros(n, dtype=np.float64)
            else:
                dt = times[:, i] - times[:, i - 1]
                ri = np.exp(-beta * dt) * (r + 1.0)
            loglam += np.where(active, np.log(mu + alpha * ri), 0.0)
            r = ri
        # padding entries are set to ``window`` by the encoder, so window-t == 0 contributes nothing
        compensator = mu * window + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (window - times)), axis=1)
        return loglam - compensator

    def sampler(self, seed: int | None = None) -> "HawkesProcessSampler":
        """Return a HawkesProcessSampler (Ogata thinning) for this distribution."""
        return HawkesProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HawkesProcessEstimator":
        """Return a HawkesProcessEstimator over the same observation window."""
        return HawkesProcessEstimator(window=self.window, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "HawkesProcessDataEncoder":
        """Returns a HawkesProcessDataEncoder bound to this window."""
        return HawkesProcessDataEncoder(self.window)


class HawkesProcessSampler(DistributionSampler):
    """Draw realizations on ``[0, window]`` by Ogata's thinning algorithm."""

    def __init__(self, dist: HawkesProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        if dist.branching_ratio >= 1.0:
            warnings.warn(
                "super-critical Hawkes process (branching_ratio = alpha/beta = %g >= 1): the process is "
                "non-stationary and self-perpetuating, so realizations may explode and hit the event cap."
                % dist.branching_ratio,
                stacklevel=2,
            )

    def _sample_one(self) -> np.ndarray:
        # Ogata thinning: between events the intensity only decays, so lam(t+) = mu + alpha*excitation
        # is a valid upper bound until the next accepted event. ``excitation`` tracks
        # sum_j exp(-beta (t - t_j)) incrementally.
        d = self.dist
        mu, alpha, beta, w = d.mu, d.alpha, d.beta, d.window
        cap = 10_000_000  # runaway guard for near/super-critical processes
        events: list[float] = []
        t = 0.0
        last = 0.0
        excitation = 0.0
        while len(events) < cap:
            lam_bar = mu + alpha * excitation
            t = t + self.rng.exponential(1.0 / lam_bar)
            if t >= w:
                break
            excitation *= math.exp(-beta * (t - last))  # decay to the candidate time
            last = t
            lam_t = mu + alpha * excitation
            if self.rng.uniform() <= lam_t / lam_bar:
                events.append(t)
                excitation += 1.0  # the new event contributes exp(0) = 1 to future excitation
        if len(events) >= cap:
            warnings.warn(
                "Hawkes realization hit the %d-event cap and was truncated before the window end; the "
                "process is likely near- or super-critical." % cap,
                stacklevel=2,
            )
        return np.asarray(events, dtype=np.float64)

    def sample(self, size: int | None = None) -> np.ndarray | list[np.ndarray]:
        """Draw one realization (event-time array) or a list of ``size`` realizations."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(int(size))]


class HawkesProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the EM branching sufficient statistics ``(S0, G, W, n_events, total_window)``."""

    def __init__(self, window: float, name: str | None = None, keys: str | None = None) -> None:
        self.window = float(window)
        self.s0 = 0.0  # expected immigrant count  (sum of immigrant responsibilities)
        self.g = 0.0  # expected offspring count   (sum of offspring responsibilities)
        self.w = 0.0  # responsibility-weighted offspring delay  (sum p_ij (t_i - t_j))
        self.n_events = 0.0
        self.total_window = 0.0
        self.name = name
        self.keys = keys

    def _accumulate_seq(
        self, times: np.ndarray, length: int, weight: float, estimate: HawkesProcessDistribution
    ) -> None:
        mu, alpha, beta = estimate.mu, estimate.alpha, estimate.beta
        r = 0.0  # R_i = sum_{j<i} exp(-beta (t_i - t_j))
        s = 0.0  # S_i = sum_{j<i} exp(-beta (t_i - t_j)) (t_i - t_j)
        prev = 0.0
        for i in range(length):
            if i > 0:
                dt = times[i] - prev
                e = math.exp(-beta * dt)
                s = e * (dt * (r + 1.0) + s)
                r = e * (r + 1.0)
            lam = mu + alpha * r
            self.s0 += weight * mu / lam  # p_i0
            self.g += weight * alpha * r / lam  # sum_{j<i} p_ij
            self.w += weight * alpha * s / lam  # sum_{j<i} p_ij (t_i - t_j)
            prev = times[i]
        self.n_events += weight * length
        self.total_window += weight * self.window

    def update(self, x: Any, weight: float, estimate: HawkesProcessDistribution | None) -> None:
        """Accumulate EM branching statistics for one event-time realization."""
        t = np.asarray(x, dtype=np.float64).reshape(-1)
        if estimate is None:
            self._initialize_seq(t, weight)
        else:
            self._accumulate_seq(t, t.size, weight, estimate)

    def _initialize_seq(self, times: np.ndarray, weight: float) -> None:
        # branching-ratio-0.5 heuristic: half the events are immigrants, offspring delays ~ the span.
        n = times.size
        span = float(times[-1] - times[0]) if n > 1 else self.window
        self.s0 += weight * 0.5 * n
        self.g += weight * 0.5 * n
        self.w += weight * 0.5 * span
        self.n_events += weight * n
        self.total_window += weight * self.window

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize branching statistics with one weighted realization."""
        self._initialize_seq(np.asarray(x, dtype=np.float64).reshape(-1), weight)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray, float], weights: np.ndarray, estimate: HawkesProcessDistribution
    ) -> None:
        """Accumulate EM branching statistics from encoded realizations."""
        # Vectorized across sequences: one loop over event index (<= max_len), each step updating the
        # R_i / S_i recursion and the (s0, g, w) responsibility sums for all sequences at once -- the
        # same scheme as seq_log_density. Bit-identical to summing _accumulate_seq per sequence.
        times, lengths, _ = x
        times = np.asarray(times, dtype=np.float64)
        lengths = np.asarray(lengths, dtype=np.int64)
        n_seq = lengths.shape[0]
        if n_seq == 0:
            return
        w = np.asarray(weights, dtype=np.float64)
        mu, alpha, beta = estimate.mu, estimate.alpha, estimate.beta
        max_len = times.shape[1]
        r = np.zeros(n_seq, dtype=np.float64)  # R_i per sequence
        s = np.zeros(n_seq, dtype=np.float64)  # S_i per sequence
        for i in range(max_len):
            active = i < lengths
            if i == 0:
                ri = np.zeros(n_seq, dtype=np.float64)
                si = np.zeros(n_seq, dtype=np.float64)
            else:
                dt = times[:, i] - times[:, i - 1]
                e = np.exp(-beta * dt)
                si = e * (dt * (r + 1.0) + s)
                ri = e * (r + 1.0)
            lam = mu + alpha * ri
            base = np.where(active, w / lam, 0.0)
            self.s0 += float(np.sum(base * mu))
            self.g += float(np.sum(base * alpha * ri))
            self.w += float(np.sum(base * alpha * si))
            r, s = ri, si
        self.n_events += float(np.dot(w, lengths))
        self.total_window += float(np.sum(w) * self.window)

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray, float], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize branching statistics from encoded realizations."""
        times, lengths, _ = x
        for k in range(len(lengths)):
            n = int(lengths[k])
            self._initialize_seq(times[k, :n], float(weights[k]))

    def combine(self, suff_stat: tuple[float, float, float, float, float]) -> "HawkesProcessAccumulator":
        """Merge serialized Hawkes EM sufficient statistics into this accumulator."""
        s0, g, w, ne, tw = suff_stat
        self.s0 += s0
        self.g += g
        self.w += w
        self.n_events += ne
        self.total_window += tw
        return self

    def value(self) -> tuple[float, float, float, float, float]:
        """Return the EM branching statistics and exposure totals."""
        return self.s0, self.g, self.w, self.n_events, self.total_window

    def from_value(self, x: tuple[float, float, float, float, float]) -> "HawkesProcessAccumulator":
        """Restore the accumulator from serialized Hawkes EM statistics."""
        self.s0, self.g, self.w, self.n_events, self.total_window = (float(v) for v in x)
        return self

    def scale(self, c: float) -> "HawkesProcessAccumulator":
        """Scale accumulated sufficient statistics by a constant."""
        self.s0 *= c
        self.g *= c
        self.w *= c
        self.n_events *= c
        self.total_window *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into a keyed statistics dictionary."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys] = tuple(a + b for a, b in zip(stats_dict[self.keys], self.value()))
            else:
                stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from a keyed statistics dictionary."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> "HawkesProcessDataEncoder":
        """Return an encoder that pads event-time realizations for this window."""
        return HawkesProcessDataEncoder(self.window)


class HawkesProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for HawkesProcessAccumulator."""

    def __init__(self, window: float, name: str | None = None, keys: str | None = None) -> None:
        self.window = float(window)
        self.name = name
        self.keys = keys

    def make(self) -> "HawkesProcessAccumulator":
        """Create an empty Hawkes process accumulator."""
        return HawkesProcessAccumulator(self.window, name=self.name, keys=self.keys)


class HawkesProcessEstimator(ParameterEstimator):
    """EM branching M-step: ``mu=S0/total_window``, ``beta=G/W``, ``alpha=beta*G/n_events``."""

    def __init__(self, window: float, name: str | None = None, keys: str | None = None) -> None:
        self.window = float(window)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "HawkesProcessAccumulatorFactory":
        """Return a factory for Hawkes EM sufficient-statistic accumulators."""
        return HawkesProcessAccumulatorFactory(self.window, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float, float]
    ) -> "HawkesProcessDistribution":
        """Estimate ``(mu, alpha, beta)`` from the accumulated branching sufficient statistics."""
        s0, g, w, n_events, total_window = suff_stat
        mu = max(s0, _MIN) / max(total_window, _MIN)
        beta = max(g, _MIN) / max(w, _MIN)
        # Branching ratio alpha/beta = expected offspring per event. Floor it away from 0: alpha == 0 is
        # an absorbing fixed point of the EM (no excitation -> no offspring responsibilities -> alpha
        # stays 0), so a tiny floor lets EM escape a Poisson-like start. Cap below 1 to stay sub-critical.
        branching = min(max(g / max(n_events, _MIN), 1.0e-3), 1.0 - 1.0e-6)
        alpha = beta * branching
        return HawkesProcessDistribution(mu, alpha, beta, self.window, name=self.name, keys=self.keys)


class HawkesProcessDataEncoder(DataSequenceEncoder):
    """Encode realizations (event-time arrays) into a padded ``(num_realizations, max_len)`` matrix."""

    def __init__(self, window: float) -> None:
        self.window = float(window)

    def __str__(self) -> str:
        return "HawkesProcessDataEncoder(%s)" % repr(self.window)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HawkesProcessDataEncoder) and self.window == other.window

    def seq_encode(self, x: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, float]:
        """Encode event-time realizations as padded times, lengths, and window."""
        seqs = []
        for events in x:
            t = np.asarray(events, dtype=np.float64).reshape(-1)
            if t.size and (np.any(~np.isfinite(t)) or t[0] < 0.0 or t[-1] > self.window or np.any(np.diff(t) < 0.0)):
                raise ValueError("event times must be finite, sorted, and within [0, window].")
            seqs.append(t)
        lengths = np.asarray([s.size for s in seqs], dtype=np.int64)
        max_len = int(lengths.max()) if lengths.size and lengths.max() > 0 else 0
        # pad with ``window`` so the compensator term (1 - exp(-beta (window - t))) vanishes on padding
        times = np.full((len(seqs), max_len), self.window, dtype=np.float64)
        for k, s in enumerate(seqs):
            times[k, : s.size] = s
        return times, lengths, self.window
