"""Multivariate (mutually-exciting) Hawkes process with an exponential kernel.

A ``D``-dimensional Hawkes process over marked events ``(t, mark)`` on a fixed window ``[0, T]``. Each
mark has a baseline intensity ``mu_d`` and an event of mark ``j`` excites the intensity of mark ``d`` by
``alpha_{dj} exp(-beta (t - t_k))``, so

    lambda_d(t) = mu_d + sum_{t_k < t} alpha_{d, mark_k} exp(-beta (t - t_k)),

coupling the dimensions through the ``D x D`` excitation matrix ``alpha`` (a shared decay ``beta``). The
exact log-likelihood ``sum_n log lambda_{mark_n}(t_n) - sum_d \\int_0^T lambda_d`` is computed in
``O(n D)`` by a per-mark excitation recursion, sampling is by multivariate Ogata thinning, and the
parameters are fit by direct maximization of this exact finite-window
likelihood. The fit does not silently project the excitation matrix into the
stationary region: finite-window supercritical models are representable. The
process is stationary when the spectral radius of ``alpha / beta`` is below 1.


Reference: Hawkes, 'Spectra of some self-exciting and mutually exciting point processes', Biometrika (1971).
"""

import math
import operator
import warnings
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


class MultivariateHawkesProcessStatistics(NamedTuple):
    """Versioned weighted marked-event evidence."""

    realizations: tuple[np.ndarray, ...]
    weights: np.ndarray
    dim: int
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


def _split(events: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split a realization (sequence of ``(time, mark)``) into sorted time and mark arrays.

    Raises:
        ValueError: If any mark is not integer-valued. Marks index the process dimension in
            ``{0, ..., D-1}``, so a fractional mark like ``0.9`` would otherwise be silently
            truncated to ``0`` by the int cast before any downstream range check ever saw it.
    """
    try:
        size = len(events)
    except TypeError as exc:
        raise ValueError(
            "events must be a two-dimensional sequence of (time, mark) pairs"
        ) from exc
    if size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    try:
        arr = np.asarray(events, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("event time/mark pairs must be numeric") from exc
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "events must have exact shape (num_events, 2)"
        )
    marks = arr[:, 1]
    if np.any(~np.isfinite(marks)) or not np.array_equal(marks, np.round(marks)):
        raise ValueError("event marks must be integer-valued (they index the process dimension).")
    return arr[:, 0].astype(np.float64), marks.astype(np.int64)


def _validated_realization(
    events: Any,
    *,
    dim: int,
    window: float,
    fail_closed: bool,
) -> np.ndarray | None:
    times, marks = _split(events)
    invalid = (
        np.any(~np.isfinite(times))
        or np.any(times < 0.0)
        or np.any(times > window)
        or (times.size > 1 and np.any(np.diff(times) <= 0.0))
        or np.any(marks < 0)
        or np.any(marks >= dim)
    )
    if invalid:
        if fail_closed:
            raise ValueError(
                "multivariate Hawkes events must have finite, strictly "
                "increasing times inside [0, window] and exact marks in "
                "[0, dim)"
            )
        return None
    result = np.column_stack((times, marks)).astype(np.float64, copy=False)
    result.setflags(write=False)
    return result


def _validated_history(
    times: Any,
    marks: Any,
    *,
    dim: int,
    window: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        time_array = np.asarray(times, dtype=np.float64)
        mark_array = np.asarray(marks)
    except (TypeError, ValueError) as exc:
        raise ValueError("multivariate Hawkes history must be numeric") from exc
    if time_array.ndim != 1 or mark_array.ndim != 1:
        raise ValueError("multivariate Hawkes history arrays must be one-dimensional")
    if time_array.shape != mark_array.shape:
        raise ValueError("multivariate Hawkes history times and marks must align")
    if (
        mark_array.dtype == np.bool_
        or not np.issubdtype(mark_array.dtype, np.number)
        or np.any(~np.isfinite(mark_array.astype(np.float64)))
        or not np.array_equal(mark_array, np.round(mark_array))
    ):
        raise ValueError("multivariate Hawkes history marks must be exact integers")
    realization = _validated_realization(
        np.column_stack((time_array, mark_array)),
        dim=dim,
        window=window,
        fail_closed=True,
    )
    if realization is None:  # pragma: no cover
        raise AssertionError("unreachable")
    return realization[:, 0], realization[:, 1].astype(np.int64)


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("multivariate Hawkes weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(
            "multivariate Hawkes weights must have exact shape "
            f"({rows},)"
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(
            "multivariate Hawkes weights must be finite and non-negative"
        )
    return weights


def _validated_statistics(
    value: Any,
    *,
    dim: int,
    window: float,
) -> MultivariateHawkesProcessStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(
            "multivariate Hawkes statistics must contain realizations, "
            "weights, dimension, and window"
        )
    raw_realizations, raw_weights, raw_dim, raw_window = value
    statistic_dim = _exact_integer(
        raw_dim,
        label="multivariate Hawkes statistic dimension",
    )
    try:
        statistic_window = float(raw_window)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "multivariate Hawkes statistic window must be numeric"
        ) from exc
    if statistic_dim != dim:
        raise ValueError(
            "multivariate Hawkes statistic dimension does not match estimator"
        )
    if (
        not np.isfinite(statistic_window)
        or not math.isclose(
            statistic_window,
            window,
            rel_tol=0.0,
            abs_tol=_WINDOW_TOLERANCE * max(1.0, window),
        )
    ):
        raise ValueError(
            "multivariate Hawkes statistic window does not match estimator"
        )
    if not isinstance(raw_realizations, (tuple, list)):
        raise ValueError("multivariate Hawkes realizations must be a sequence")
    realizations = tuple(
        _validated_realization(
            events,
            dim=dim,
            window=window,
            fail_closed=True,
        )
        for events in raw_realizations
    )
    if any(item is None for item in realizations):  # pragma: no cover
        raise AssertionError("unreachable")
    weights = _validated_weights(raw_weights, len(realizations)).copy()
    weights.setflags(write=False)
    return MultivariateHawkesProcessStatistics(
        realizations,
        weights,
        dim,
        window,
    )


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
        max_events: int = _DEFAULT_MAX_EVENTS,
    ) -> None:
        m = np.asarray(mu, dtype=np.float64)
        a = np.asarray(alpha, dtype=np.float64)
        if m.ndim != 1 or m.size == 0:
            raise ValueError("mu must be a nonempty one-dimensional vector")
        self.dim = m.shape[0]
        if a.shape != (self.dim, self.dim):
            raise ValueError("alpha must be a (D, D) matrix matching len(mu)")
        checked_beta = float(beta)
        checked_window = float(window)
        if (
            np.any(~np.isfinite(m))
            or np.any(~np.isfinite(a))
            or not np.isfinite(checked_beta)
            or not np.isfinite(checked_window)
            or np.any(m <= 0.0)
            or np.any(a < 0.0)
            or checked_beta <= 0.0
            or checked_window <= 0.0
        ):
            raise ValueError(
                "multivariate Hawkes requires finite mu>0, alpha>=0, "
                "beta>0, and window>0"
            )
        self.mu = m.copy()
        self.alpha = a.copy()
        self.mu.setflags(write=False)
        self.alpha.setflags(write=False)
        self.beta = float(beta)
        self.window = float(window)
        self.max_events = _exact_integer(
            max_events,
            label="multivariate Hawkes maximum event count",
        )
        if self.max_events < 0:
            raise ValueError(
                "multivariate Hawkes maximum event count must be non-negative"
            )
        self.name = name
        self.keys = keys
        self._col_alpha = self.alpha.sum(axis=0)
        self._col_alpha.setflags(write=False)
        self.spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(self.alpha / self.beta)))
        )

    def __str__(self) -> str:
        return (
            "MultivariateHawkesProcessDistribution(%s, %s, %s, %s, "
            "name=%s, keys=%s, max_events=%s)"
        ) % (
            repr(self.mu.tolist()),
            repr(self.alpha.tolist()),
            repr(self.beta),
            repr(self.window),
            repr(self.name),
            repr(self.keys),
            repr(self.max_events),
        )

    def intensity(self, t: float, times: Any, marks: Any) -> np.ndarray:
        """Per-mark conditional rate vector (the vector-valued variant of ``intensity``).

        Returns ``lambda(t)`` of shape ``(D,)`` with
        ``lambda_k(t) = mu_k + sum_{(t_i, m_i) < t} alpha[k, m_i] exp(-beta (t - t_i))``.
        """
        query = _finite_nonnegative_scalar(
            t,
            label="multivariate Hawkes query time",
        )
        if query > self.window:
            raise ValueError(
                "multivariate Hawkes query time must lie inside [0, window]"
            )
        ti, mi = _validated_history(
            times,
            marks,
            dim=self.dim,
            window=self.window,
        )
        past = ti < query
        lam = self.mu.copy()
        if np.any(past):
            decay = np.exp(-self.beta * (query - ti[past]))
            # s[j] = sum_{past, mark=j} exp(-beta (t - t_i)); lambda = mu + alpha @ s
            s = np.zeros(self.dim)
            np.add.at(s, mi[past], decay)
            lam = lam + self.alpha @ s
        return lam

    def expected_count(self, t_start: float, t_end: float, times: Any, marks: Any) -> np.ndarray:
        """Per-mark compensator vector (the vector-valued variant of ``expected_count``).

        Returns ``(D,)`` with the integral of ``lambda_k`` over ``[t_start, t_end]`` given the history.
        """
        start = _finite_nonnegative_scalar(
            t_start,
            label="multivariate Hawkes interval start",
        )
        end = _finite_nonnegative_scalar(
            t_end,
            label="multivariate Hawkes interval end",
        )
        if start > end or end > self.window:
            raise ValueError(
                "multivariate Hawkes interval must satisfy "
                "0 <= start <= end <= window"
            )
        ti, mi = _validated_history(
            times,
            marks,
            dim=self.dim,
            window=self.window,
        )
        rel = ti < end
        comp = self.mu * (end - start)
        if np.any(rel):
            tp, mp = ti[rel], mi[rel]
            lo = np.maximum(start, tp)
            kernel = (
                np.exp(-self.beta * (lo - tp))
                - np.exp(-self.beta * (end - tp))
            ) / self.beta
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
        realization = _validated_realization(
            x,
            dim=self.dim,
            window=self.window,
            fail_closed=False,
        )
        if realization is None:
            return -np.inf
        times = realization[:, 0]
        marks = realization[:, 1].astype(np.int64)
        n = times.size
        w = self.window
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
        """Return an exact finite-window estimator."""
        if pseudo_count is not None:
            checked_pseudo = _finite_nonnegative_scalar(
                pseudo_count,
                label="multivariate Hawkes pseudo-count",
            )
            if checked_pseudo != 0.0:
                raise NotImplementedError(
                    "multivariate Hawkes does not define an implicit "
                    "pseudo-count prior; fit explicit penalized likelihood "
                    "instead"
                )
        return MultivariateHawkesProcessEstimator(
            self.dim,
            self.window,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
        )

    def dist_to_encoder(self) -> "MultivariateHawkesProcessDataEncoder":
        """Return the data encoder (passes realizations through; the likelihood is per-realization)."""
        return MultivariateHawkesProcessDataEncoder(self.window, self.dim)


class MultivariateHawkesProcessSampler(DistributionSampler):
    """Draw realizations by multivariate Ogata thinning."""

    def __init__(self, dist: MultivariateHawkesProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.last_receipt: dict[str, Any] | None = None
        if dist.spectral_radius >= 1.0:
            warnings.warn(
                "super-critical multivariate Hawkes (spectral radius of alpha/beta = %g >= 1): the process "
                "is non-stationary and may explode." % dist.spectral_radius,
                stacklevel=2,
            )

    def _sample_one(self) -> list[tuple[float, int]]:
        d = self.dist
        mu, alpha, beta, w = d.mu, d.alpha, d.beta, d.window
        events: list[tuple[float, int]] = []
        s = np.zeros(d.dim)
        t = 0.0
        last = 0.0
        candidates = 0
        while True:
            lam_bar = float(mu.sum() + d._col_alpha @ s)  # total intensity decays between events -> upper bound
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
                    "spectral_radius": d.spectral_radius,
                    "termination_reason": "window_crossed",
                }
                break
            s = s * math.exp(-beta * (t - last))
            last = t
            lam_d = mu + alpha @ s  # per-mark intensities at the candidate time
            lam_total = float(lam_d.sum())
            if self.rng.uniform() <= lam_total / lam_bar:
                if len(events) >= d.max_events:
                    self.last_receipt = {
                        "complete": False,
                        "events_generated": len(events),
                        "candidate_draws": candidates,
                        "event_budget": d.max_events,
                        "reached_time": t,
                        "window": w,
                        "spectral_radius": d.spectral_radius,
                        "termination_reason": "event_budget_exhausted",
                    }
                    raise RuntimeError(
                        "multivariate Hawkes sampling event budget exhausted "
                        "before the observation window was completed"
                    )
                m = int(self.rng.choice(d.dim, p=lam_d / lam_total))
                events.append((t, m))
                s[m] += 1.0
        return events

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one marked-event realization, or ``size`` iid realizations."""
        if size is None:
            return self._sample_one()
        checked_size = _exact_integer(
            size,
            label="multivariate Hawkes sample size",
        )
        if checked_size < 0:
            raise ValueError(
                "multivariate Hawkes sample size must be non-negative"
            )
        return [self._sample_one() for _ in range(checked_size)]


class MultivariateHawkesProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Retain validated weighted realizations for exact finite-window MLE."""

    def __init__(self, dim: int, window: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _exact_integer(
            dim,
            label="multivariate Hawkes dimension",
        )
        if self.dim <= 0:
            raise ValueError(
                "multivariate Hawkes dimension must be strictly positive"
            )
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError(
                "multivariate Hawkes accumulator requires a finite window > 0"
            )
        self.window = float(window)
        self.realizations: list[np.ndarray] = []
        self.weights: list[float] = []
        self.name = name
        self.keys = keys

    def _store(self, events: Any, weight: Any) -> None:
        checked_events = _validated_realization(
            events,
            dim=self.dim,
            window=self.window,
            fail_closed=True,
        )
        if checked_events is None:  # pragma: no cover
            raise AssertionError("unreachable")
        checked_weight = _finite_nonnegative_scalar(
            weight,
            label="multivariate Hawkes weight",
        )
        self.realizations.append(checked_events)
        self.weights.append(checked_weight)

    def update(self, x: Any, weight: float, estimate: MultivariateHawkesProcessDistribution | None) -> None:
        """Store one validated weighted realization."""
        self._store(x, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Store one validated weighted realization during initialization."""
        self._store(x, weight)

    def seq_update(self, x: list[Any], weights: np.ndarray, estimate: MultivariateHawkesProcessDistribution) -> None:
        """Store an exactly aligned batch of weighted realizations."""
        rows = list(x)
        checked_weights = _validated_weights(weights, len(rows))
        for events, weight in zip(rows, checked_weights):
            self._store(events, float(weight))

    def seq_initialize(self, x: list[Any], weights: np.ndarray, rng: RandomState | None) -> None:
        """Store an aligned batch during initialization."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Any) -> "MultivariateHawkesProcessAccumulator":
        """Merge validated serialized marked-event evidence."""
        checked = _validated_statistics(
            suff_stat,
            dim=self.dim,
            window=self.window,
        )
        self.realizations.extend(checked.realizations)
        self.weights.extend(checked.weights.tolist())
        return self

    def value(self) -> MultivariateHawkesProcessStatistics:
        """Return owned, versioned marked-event evidence."""
        weights = np.asarray(self.weights, dtype=np.float64)
        weights.setflags(write=False)
        return MultivariateHawkesProcessStatistics(
            tuple(self.realizations),
            weights,
            self.dim,
            self.window,
        )

    def from_value(self, x: Any) -> "MultivariateHawkesProcessAccumulator":
        """Replace contents from validated serialized evidence."""
        checked = _validated_statistics(
            x,
            dim=self.dim,
            window=self.window,
        )
        self.realizations = list(checked.realizations)
        self.weights = checked.weights.tolist()
        return self

    def scale(self, c: float) -> "MultivariateHawkesProcessAccumulator":
        """Scale realization weights by a finite non-negative constant."""
        checked = _finite_nonnegative_scalar(
            c,
            label="multivariate Hawkes scale",
        )
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
    """Exact weighted finite-window maximum-likelihood estimator."""

    def __init__(
        self,
        dim: int,
        window: float,
        name: str | None = None,
        keys: str | None = None,
        max_events: int = _DEFAULT_MAX_EVENTS,
    ) -> None:
        self.dim = _exact_integer(
            dim,
            label="multivariate Hawkes dimension",
        )
        if self.dim <= 0:
            raise ValueError(
                "multivariate Hawkes dimension must be strictly positive"
            )
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError(
                "multivariate Hawkes estimator requires a finite window > 0"
            )
        self.window = float(window)
        self.name = name
        self.keys = keys
        self.max_events = _exact_integer(
            max_events,
            label="multivariate Hawkes maximum event count",
        )
        if self.max_events < 0:
            raise ValueError(
                "multivariate Hawkes maximum event count must be non-negative"
            )

    def accumulator_factory(self) -> MultivariateHawkesProcessAccumulatorFactory:
        """Return an accumulator factory for branching EM statistics."""
        return MultivariateHawkesProcessAccumulatorFactory(self.dim, self.window, name=self.name, keys=self.keys)

    def estimate(
        self,
        nobs: float | None,
        suff_stat: Any,
    ) -> MultivariateHawkesProcessDistribution:
        """Maximize the exact compensator-aware weighted log-likelihood."""
        checked = _validated_statistics(
            suff_stat,
            dim=self.dim,
            window=self.window,
        )
        total_weight = float(np.sum(checked.weights))
        event_counts = np.zeros(self.dim, dtype=np.float64)
        gap_arrays = []
        for events, weight in zip(checked.realizations, checked.weights):
            if weight == 0.0:
                continue
            marks = events[:, 1].astype(np.int64)
            event_counts += float(weight) * np.bincount(
                marks,
                minlength=self.dim,
            )
            if events.shape[0] > 1:
                gap_arrays.append(np.diff(events[:, 0]))
        if total_weight <= 0.0:
            raise ValueError(
                "multivariate Hawkes fitting requires positive realization "
                "weight"
            )
        if np.any(event_counts <= 0.0):
            raise ValueError(
                "multivariate Hawkes fitting requires positively weighted "
                "events in every dimension for an interior baseline-rate MLE"
            )

        empirical_mu = event_counts / (total_weight * self.window)
        gaps = (
            np.concatenate(gap_arrays)
            if gap_arrays
            else np.empty(0, dtype=np.float64)
        )
        beta_seed = (
            1.0 / float(np.median(gaps))
            if gaps.size
            else 1.0 / self.window
        )
        beta_seed = max(beta_seed, 1.0 / self.window)
        alpha_offset = self.dim
        beta_offset = alpha_offset + self.dim * self.dim

        def objective(parameters: np.ndarray) -> float:
            mu = np.exp(parameters[: self.dim])
            alpha = parameters[
                alpha_offset:beta_offset
            ].reshape(self.dim, self.dim)
            beta = math.exp(float(parameters[beta_offset]))
            column_mass = np.sum(alpha, axis=0)
            total = 0.0
            for events, weight in zip(
                checked.realizations,
                checked.weights,
            ):
                if weight == 0.0:
                    continue
                times = events[:, 0]
                marks = events[:, 1].astype(np.int64)
                excitation = np.zeros(self.dim, dtype=np.float64)
                previous = 0.0
                loglam = 0.0
                for index, (event, mark) in enumerate(zip(times, marks)):
                    if index:
                        excitation *= math.exp(
                            -beta * (float(event) - previous)
                        )
                    intensity = float(mu[mark] + alpha[mark] @ excitation)
                    if not np.isfinite(intensity) or intensity <= 0.0:
                        return np.inf
                    loglam += math.log(intensity)
                    excitation[mark] += 1.0
                    previous = float(event)
                compensator = self.window * float(np.sum(mu))
                if times.size and np.any(alpha):
                    compensator += (1.0 / beta) * float(
                        np.sum(
                            column_mass[marks]
                            * (
                                1.0
                                - np.exp(
                                    -beta * (self.window - times)
                                )
                            )
                        )
                    )
                total += float(weight) * (loglam - compensator)
            return -total if np.isfinite(total) else np.inf

        starts = []
        for immigrant_share, branching in (
            (1.0, 0.0),
            (0.7, 0.3),
            (0.4, 0.8),
        ):
            alpha = np.full(
                (self.dim, self.dim),
                beta_seed * branching / self.dim,
                dtype=np.float64,
            )
            starts.append(
                np.concatenate(
                    (
                        np.log(
                            np.maximum(
                                empirical_mu * immigrant_share,
                                1.0e-12,
                            )
                        ),
                        alpha.reshape(-1),
                        [math.log(beta_seed)],
                    )
                )
            )
        bounds = (
            [(-40.0, 40.0)] * self.dim
            + [(0.0, None)] * (self.dim * self.dim)
            + [(-40.0, 40.0)]
        )
        candidates = []
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
                "multivariate Hawkes finite-window likelihood optimization "
                "failed to converge to a finite candidate"
            )
        best = min(candidates, key=lambda result: float(result.fun))
        mu = np.exp(best.x[: self.dim])
        alpha = np.maximum(
            best.x[alpha_offset:beta_offset].reshape(
                self.dim,
                self.dim,
            ),
            0.0,
        )
        beta = math.exp(float(best.x[beta_offset]))
        return MultivariateHawkesProcessDistribution(
            mu,
            alpha,
            beta,
            self.window,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
        )


class MultivariateHawkesProcessDataEncoder(DataSequenceEncoder):
    """Validate and pass through realizations of marked events."""

    def __init__(self, window: float, dim: int) -> None:
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError(
                "multivariate Hawkes encoder requires a finite window > 0"
            )
        self.window = float(window)
        self.dim = _exact_integer(
            dim,
            label="multivariate Hawkes dimension",
        )
        if self.dim <= 0:
            raise ValueError(
                "multivariate Hawkes dimension must be strictly positive"
            )

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
            realization = _validated_realization(
                events,
                dim=self.dim,
                window=self.window,
                fail_closed=True,
            )
            if realization is None:  # pragma: no cover
                raise AssertionError("unreachable")
            times = realization[:, 0]
            marks = realization[:, 1].astype(np.int64)
            out.append([(float(t), int(m)) for t, m in zip(times, marks)])
        return out

    def row_count(self, x: Any) -> int:
        """Return the number of validated encoded realizations."""
        rows = list(x)
        for events in rows:
            _validated_realization(
                events,
                dim=self.dim,
                window=self.window,
                fail_closed=True,
            )
        return len(rows)
