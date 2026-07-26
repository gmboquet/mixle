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

Fitting retains the weighted realizations and maximizes this exact finite-window
likelihood, including the edge-dependent integrated-kernel term. This is not
the common edge-free branching approximation: a Poisson boundary fit
(``alpha=0``) and an unconstrained finite-window supercritical fit are both
representable.


Reference: Hawkes, 'Spectra of some self-exciting and mutually exciting point processes', Biometrika (1971).
"""

import math
import operator
from collections.abc import Sequence
from typing import Any, NamedTuple

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

_DEFAULT_MAX_EVENTS = 100_000
_WINDOW_TOLERANCE = 1.0e-12


class HawkesProcessStatistics(NamedTuple):
    """Versioned weighted event evidence for exact finite-window fitting."""

    realizations: tuple[np.ndarray, ...]
    weights: np.ndarray
    window: float

    @property
    def schema_version(self) -> int:
        """Return the serialized-statistic schema version."""
        return 1


def _exact_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an exact integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an exact integer") from exc


def _finite_nonnegative_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _validated_events(
    value: Any,
    *,
    window: float,
    fail_closed: bool,
) -> np.ndarray | None:
    try:
        events = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        if fail_closed:
            raise ValueError("Hawkes event times must be numeric") from exc
        return None
    invalid = (
        events.ndim != 1
        or np.any(~np.isfinite(events))
        or np.any(events < 0.0)
        or np.any(events > window)
        or (events.size > 1 and np.any(np.diff(events) <= 0.0))
    )
    if invalid:
        if fail_closed:
            raise ValueError(
                "Hawkes event times must be a one-dimensional, finite, "
                "strictly increasing sequence inside [0, window]"
            )
        return None
    result = events.copy()
    result.setflags(write=False)
    return result


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hawkes weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"Hawkes weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Hawkes weights must be finite and non-negative")
    return weights


def _validated_payload(
    value: Any,
    *,
    window: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(
            "Hawkes encoded payload must be (times, lengths, window)"
        )
    times_raw, lengths_raw, encoded_window = value
    try:
        times = np.asarray(times_raw, dtype=np.float64)
        lengths = np.asarray(lengths_raw)
        encoded_window = float(encoded_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hawkes encoded payload must be numeric") from exc
    if (
        not np.isfinite(encoded_window)
        or not math.isclose(
            encoded_window,
            window,
            rel_tol=0.0,
            abs_tol=_WINDOW_TOLERANCE * max(1.0, window),
        )
    ):
        raise ValueError("Hawkes encoded window does not match the model window")
    if times.ndim != 2:
        raise ValueError("Hawkes encoded times must be a two-dimensional matrix")
    if (
        lengths.ndim != 1
        or lengths.shape[0] != times.shape[0]
        or lengths.dtype == np.bool_
        or not np.issubdtype(lengths.dtype, np.integer)
    ):
        raise ValueError(
            "Hawkes encoded lengths must be an aligned exact-integer vector"
        )
    lengths = lengths.astype(np.int64, copy=False)
    if np.any(lengths < 0) or np.any(lengths > times.shape[1]):
        raise ValueError("Hawkes encoded length is outside the padded matrix")
    for row, length in zip(times, lengths):
        checked = _validated_events(
            row[: int(length)],
            window=window,
            fail_closed=True,
        )
        if checked is None:  # pragma: no cover - fail_closed always raises
            raise AssertionError("unreachable")
        padding = row[int(length) :]
        if np.any(~np.isfinite(padding)) or np.any(padding != window):
            raise ValueError(
                "Hawkes padding must equal the fixed observation window"
            )
    return times, lengths


def _validated_statistics(
    value: Any,
    *,
    window: float,
) -> HawkesProcessStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(
            "Hawkes statistics must contain realizations, weights, and window"
        )
    raw_realizations, raw_weights, raw_window = value
    try:
        statistic_window = float(raw_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hawkes statistic window must be numeric") from exc
    if (
        not np.isfinite(statistic_window)
        or not math.isclose(
            statistic_window,
            window,
            rel_tol=0.0,
            abs_tol=_WINDOW_TOLERANCE * max(1.0, window),
        )
    ):
        raise ValueError("Hawkes statistic window does not match the estimator")
    if not isinstance(raw_realizations, (tuple, list)):
        raise ValueError("Hawkes realizations must be a sequence")
    realizations = tuple(
        _validated_events(item, window=window, fail_closed=True)
        for item in raw_realizations
    )
    if any(item is None for item in realizations):  # pragma: no cover
        raise AssertionError("unreachable")
    weights = _validated_weights(raw_weights, len(realizations)).copy()
    weights.setflags(write=False)
    return HawkesProcessStatistics(realizations, weights, window)


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
        max_events: int = _DEFAULT_MAX_EVENTS,
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
        if not (
            np.all(np.isfinite([self.mu, self.alpha, self.beta, self.window]))
            and self.mu > 0.0
            and self.alpha >= 0.0
            and self.beta > 0.0
            and self.window > 0.0
        ):
            raise ValueError(
                "Hawkes process requires finite mu>0, alpha>=0, beta>0, "
                "and window>0."
            )
        self.max_events = _exact_integer(
            max_events,
            label="Hawkes maximum event count",
        )
        if self.max_events < 0:
            raise ValueError("Hawkes maximum event count must be non-negative")
        self.name = name
        self.keys = keys
        self.branching_ratio = self.alpha / self.beta

    def __str__(self) -> str:
        """Return a constructor-style representation of the Hawkes process distribution."""
        return (
            "HawkesProcessDistribution(%s, %s, %s, %s, name=%s, keys=%s, "
            "max_events=%s)"
        ) % (
            repr(self.mu),
            repr(self.alpha),
            repr(self.beta),
            repr(self.window),
            repr(self.name),
            repr(self.keys),
            repr(self.max_events),
        )

    def intensity(self, t: float, times: Any, marks: Any = None) -> float:
        """Conditional rate ``lambda(t) = mu + alpha sum_{t_i < t} exp(-beta (t - t_i))`` given the history.

        ``times`` is the event history; ``marks`` is accepted for ``TemporalPointProcess`` signature
        parity (the univariate Hawkes process is unmarked) and is ignored.
        """
        query = _finite_nonnegative_scalar(t, label="Hawkes query time")
        if query > self.window:
            raise ValueError("Hawkes query time must lie inside [0, window]")
        ti = _validated_events(
            times,
            window=self.window,
            fail_closed=True,
        )
        if ti is None:  # pragma: no cover
            raise AssertionError("unreachable")
        past = ti[ti < query]
        return float(
            self.mu
            + self.alpha * np.sum(np.exp(-self.beta * (query - past)))
        )

    def expected_count(self, t_start: float, t_end: float, times: Any, marks: Any = None) -> float:
        """Compensator ``integral_{t_start}^{t_end} lambda(s) ds`` of the intensity given the history.

        ``marks`` is accepted for signature parity and ignored (the univariate process is unmarked).
        """
        start = _finite_nonnegative_scalar(
            t_start,
            label="Hawkes interval start",
        )
        end = _finite_nonnegative_scalar(t_end, label="Hawkes interval end")
        if start > end or end > self.window:
            raise ValueError(
                "Hawkes interval must satisfy 0 <= start <= end <= window"
            )
        ti = _validated_events(
            times,
            window=self.window,
            fail_closed=True,
        )
        if ti is None:  # pragma: no cover
            raise AssertionError("unreachable")
        tp = ti[ti < end]
        lo = np.maximum(start, tp)
        kernel = np.exp(-self.beta * (lo - tp)) - np.exp(
            -self.beta * (end - tp)
        )
        return float(
            self.mu * (end - start)
            + (self.alpha / self.beta) * np.sum(kernel)
        )

    def density(self, x: Any) -> float:
        """Probability density of one realization ``x`` (a sequence of event times)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Exact log-likelihood of one realization (a strictly increasing event-time sequence in ``[0, window]``).

        The conditional intensity is defined over the strict history ``t_j < t`` (see the class docstring),
        so a realization with a tied timestamp pair (``t_j == t_i`` for ``j != i``) is not a valid ``t_1 <
        ... < t_n`` history -- like any other malformed ordering, it scores ``-inf`` rather than silently
        letting one of the tied events excite the other.
        """
        t = _validated_events(
            x,
            window=self.window,
            fail_closed=False,
        )
        if t is None:
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
        times, lengths = _validated_payload(x, window=self.window)
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
        active = np.arange(max_len)[None, :] < lengths[:, None]
        compensator = mu * self.window + (alpha / beta) * np.sum(
            np.where(
                active,
                1.0 - np.exp(-beta * (self.window - times)),
                0.0,
            ),
            axis=1,
        )
        return loglam - compensator

    def sampler(self, seed: int | None = None) -> "HawkesProcessSampler":
        """Return a HawkesProcessSampler (Ogata thinning) for this distribution."""
        return HawkesProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HawkesProcessEstimator":
        """Return a HawkesProcessEstimator over the same observation window."""
        if pseudo_count is not None:
            checked_pseudo = _finite_nonnegative_scalar(
                pseudo_count,
                label="Hawkes pseudo-count",
            )
            if checked_pseudo != 0.0:
                raise NotImplementedError(
                    "HawkesProcessDistribution does not define an implicit "
                    "pseudo-count prior; fit explicit penalized likelihood "
                    "instead"
                )
        return HawkesProcessEstimator(
            window=self.window,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
        )

    def dist_to_encoder(self) -> "HawkesProcessDataEncoder":
        """Returns a HawkesProcessDataEncoder bound to this window."""
        return HawkesProcessDataEncoder(self.window)


class HawkesProcessSampler(DistributionSampler):
    """Draw realizations on ``[0, window]`` by Ogata's thinning algorithm."""

    def __init__(self, dist: HawkesProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.last_receipt: dict[str, Any] | None = None

    def _sample_one(self) -> np.ndarray:
        # Ogata thinning: between events the intensity only decays, so lam(t+) = mu + alpha*excitation
        # is a valid upper bound until the next accepted event. ``excitation`` tracks
        # sum_j exp(-beta (t - t_j)) incrementally.
        d = self.dist
        mu, alpha, beta, w = d.mu, d.alpha, d.beta, d.window
        events: list[float] = []
        t = 0.0
        last = 0.0
        excitation = 0.0
        candidates = 0
        while True:
            lam_bar = mu + alpha * excitation
            t = t + self.rng.exponential(1.0 / lam_bar)
            candidates += 1
            if t >= w:
                self.last_receipt = {
                    "complete": True,
                    "events_generated": len(events),
                    "candidate_draws": candidates,
                    "event_budget": d.max_events,
                    "reached_time": t,
                    "window": w,
                    "branching_ratio": d.branching_ratio,
                    "termination_reason": "window_crossed",
                }
                break
            excitation *= math.exp(-beta * (t - last))  # decay to the candidate time
            last = t
            lam_t = mu + alpha * excitation
            if self.rng.uniform() <= lam_t / lam_bar:
                if len(events) >= d.max_events:
                    self.last_receipt = {
                        "complete": False,
                        "events_generated": len(events),
                        "candidate_draws": candidates,
                        "event_budget": d.max_events,
                        "reached_time": t,
                        "window": w,
                        "branching_ratio": d.branching_ratio,
                        "termination_reason": "event_budget_exhausted",
                    }
                    raise RuntimeError(
                        "Hawkes sampling event budget exhausted before the "
                        "observation window was completed"
                    )
                events.append(t)
                excitation += 1.0  # the new event contributes exp(0) = 1 to future excitation
        return np.asarray(events, dtype=np.float64)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray | list[np.ndarray]:
        """Draw one realization (event-time array) or a list of ``size`` realizations."""
        if size is None:
            return self._sample_one()
        checked_size = _exact_integer(size, label="Hawkes sample size")
        if checked_size < 0:
            raise ValueError("Hawkes sample size must be non-negative")
        return [self._sample_one() for _ in range(checked_size)]


class HawkesProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Retain validated weighted realizations for exact finite-window MLE."""

    def __init__(self, window: float, name: str | None = None, keys: str | None = None) -> None:
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("Hawkes accumulator requires a finite window > 0")
        self.window = float(window)
        self.realizations: list[np.ndarray] = []
        self.weights: list[float] = []
        self.name = name
        self.keys = keys

    def _store(self, value: Any, weight: Any) -> None:
        events = _validated_events(
            value,
            window=self.window,
            fail_closed=True,
        )
        if events is None:  # pragma: no cover
            raise AssertionError("unreachable")
        checked_weight = _finite_nonnegative_scalar(
            weight,
            label="Hawkes weight",
        )
        self.realizations.append(events)
        self.weights.append(checked_weight)

    def update(self, x: Any, weight: float, estimate: HawkesProcessDistribution | None) -> None:
        """Store one validated weighted realization."""
        self._store(x, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Store one validated weighted realization during initialization."""
        self._store(x, weight)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray, float], weights: np.ndarray, estimate: HawkesProcessDistribution
    ) -> None:
        """Store validated encoded realizations and aligned weights."""
        times, lengths = _validated_payload(x, window=self.window)
        checked_weights = _validated_weights(weights, len(lengths))
        for row, length, weight in zip(times, lengths, checked_weights):
            self._store(row[: int(length)], float(weight))

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray, float], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Store validated encoded realizations during initialization."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Any) -> "HawkesProcessAccumulator":
        """Merge validated serialized Hawkes evidence."""
        checked = _validated_statistics(suff_stat, window=self.window)
        self.realizations.extend(checked.realizations)
        self.weights.extend(checked.weights.tolist())
        return self

    def value(self) -> HawkesProcessStatistics:
        """Return owned, versioned weighted realization evidence."""
        weights = np.asarray(self.weights, dtype=np.float64)
        weights.setflags(write=False)
        return HawkesProcessStatistics(
            tuple(self.realizations),
            weights,
            self.window,
        )

    def from_value(self, x: Any) -> "HawkesProcessAccumulator":
        """Restore validated serialized Hawkes evidence."""
        checked = _validated_statistics(x, window=self.window)
        self.realizations = list(checked.realizations)
        self.weights = checked.weights.tolist()
        return self

    def scale(self, c: float) -> "HawkesProcessAccumulator":
        """Scale realization weights by a finite non-negative constant."""
        checked = _finite_nonnegative_scalar(c, label="Hawkes scale")
        self.weights = [checked * weight for weight in self.weights]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this evidence under its shared parameter key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this evidence from its shared parameter key."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

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
    """Exact weighted finite-window maximum-likelihood estimator."""

    def __init__(
        self,
        window: float,
        name: str | None = None,
        keys: str | None = None,
        max_events: int = _DEFAULT_MAX_EVENTS,
    ) -> None:
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("Hawkes estimator requires a finite window > 0")
        self.window = float(window)
        self.name = name
        self.keys = keys
        self.max_events = _exact_integer(
            max_events,
            label="Hawkes maximum event count",
        )
        if self.max_events < 0:
            raise ValueError("Hawkes maximum event count must be non-negative")

    def accumulator_factory(self) -> "HawkesProcessAccumulatorFactory":
        """Return a factory for Hawkes EM sufficient-statistic accumulators."""
        return HawkesProcessAccumulatorFactory(self.window, name=self.name, keys=self.keys)

    def estimate(
        self,
        nobs: float | None,
        suff_stat: Any,
    ) -> "HawkesProcessDistribution":
        """Maximize the exact compensator-aware weighted log-likelihood."""
        checked = _validated_statistics(suff_stat, window=self.window)
        total_weight = float(np.sum(checked.weights))
        event_weight = float(
            sum(
                weight * events.size
                for events, weight in zip(
                    checked.realizations,
                    checked.weights,
                )
            )
        )
        if total_weight <= 0.0:
            raise ValueError("Hawkes fitting requires positive realization weight")
        if event_weight <= 0.0:
            raise ValueError(
                "Hawkes evidence with no positively weighted events has no "
                "finite interior baseline-rate MLE"
            )

        empirical_rate = event_weight / (total_weight * self.window)
        gap_arrays = [
            np.diff(events)
            for events in checked.realizations
            if events.size > 1
        ]
        positive_gaps = (
            np.concatenate(gap_arrays)
            if gap_arrays
            else np.empty(0, dtype=np.float64)
        )
        beta_seed = (
            1.0 / float(np.median(positive_gaps))
            if positive_gaps.size
            else 1.0 / self.window
        )
        beta_seed = max(beta_seed, 1.0 / self.window)

        def objective(parameters: np.ndarray) -> float:
            mu = math.exp(float(parameters[0]))
            alpha = float(parameters[1])
            beta = math.exp(float(parameters[2]))
            total = 0.0
            for events, weight in zip(
                checked.realizations,
                checked.weights,
            ):
                if weight == 0.0:
                    continue
                loglam = 0.0
                excitation = 0.0
                previous = 0.0
                for index, event in enumerate(events):
                    if index:
                        excitation = math.exp(
                            -beta * (float(event) - previous)
                        ) * (excitation + 1.0)
                    intensity = mu + alpha * excitation
                    if not np.isfinite(intensity) or intensity <= 0.0:
                        return np.inf
                    loglam += math.log(intensity)
                    previous = float(event)
                compensator = mu * self.window
                if events.size and alpha:
                    compensator += (alpha / beta) * float(
                        np.sum(
                            1.0
                            - np.exp(
                                -beta * (self.window - events)
                            )
                        )
                    )
                total += float(weight) * (loglam - compensator)
            return -total if np.isfinite(total) else np.inf

        starts = [
            np.asarray(
                [
                    math.log(max(empirical_rate * immigrant_share, 1.0e-12)),
                    beta_seed * branching,
                    math.log(beta_seed),
                ],
                dtype=np.float64,
            )
            for immigrant_share, branching in (
                (1.0, 0.0),
                (0.75, 0.25),
                (0.5, 0.5),
                (0.25, 0.9),
            )
        ]
        candidates = []
        bounds = [(-40.0, 40.0), (0.0, None), (-40.0, 40.0)]
        for start in starts:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1.0e-12},
            )
            if result.success and np.isfinite(result.fun):
                candidates.append(result)
        if not candidates:
            raise RuntimeError(
                "Hawkes finite-window likelihood optimization failed to "
                "converge to a finite candidate"
            )
        best = min(candidates, key=lambda result: float(result.fun))
        mu = math.exp(float(best.x[0]))
        alpha = max(0.0, float(best.x[1]))
        beta = math.exp(float(best.x[2]))
        return HawkesProcessDistribution(
            mu,
            alpha,
            beta,
            self.window,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
        )


class HawkesProcessDataEncoder(DataSequenceEncoder):
    """Encode realizations (event-time arrays) into a padded ``(num_realizations, max_len)`` matrix."""

    def __init__(self, window: float) -> None:
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("Hawkes encoder requires a finite window > 0")
        self.window = float(window)

    def __str__(self) -> str:
        return "HawkesProcessDataEncoder(%s)" % repr(self.window)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HawkesProcessDataEncoder) and self.window == other.window

    def seq_encode(self, x: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, float]:
        """Encode event-time realizations as padded times, lengths, and window.

        Raises:
            ValueError: If any realization has a non-finite, out-of-window, or non-strictly-increasing
                time (including a tied timestamp pair) -- the conditional intensity is only defined over
                the strict history ``t_j < t``, so ``t_j == t_i`` is not a valid history.
        """
        seqs = []
        for events in x:
            t = _validated_events(
                events,
                window=self.window,
                fail_closed=True,
            )
            if t is None:  # pragma: no cover
                raise AssertionError("unreachable")
            seqs.append(t)
        lengths = np.asarray([s.size for s in seqs], dtype=np.int64)
        max_len = int(lengths.max()) if lengths.size and lengths.max() > 0 else 0
        # pad with ``window`` so the compensator term (1 - exp(-beta (window - t))) vanishes on padding
        times = np.full((len(seqs), max_len), self.window, dtype=np.float64)
        for k, s in enumerate(seqs):
            times[k, : s.size] = s
        return times, lengths, self.window

    def row_count(self, x: Any) -> int:
        """Return the validated number of encoded realizations."""
        _, lengths = _validated_payload(x, window=self.window)
        return int(lengths.size)
