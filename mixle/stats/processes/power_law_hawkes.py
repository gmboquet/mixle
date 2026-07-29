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

import math
import operator
import warnings
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

__all__ = ["PowerLawHawkesDistribution", "PowerLawHawkesEstimator"]

_DEFAULT_MAX_EVENTS = 100_000
_DEFAULT_MAX_WORK = 1_000_000
_WINDOW_TOLERANCE = 1.0e-12


class PowerLawHawkesStatistics(NamedTuple):
    """Versioned weighted marked-event evidence."""

    realizations: tuple[tuple[np.ndarray, np.ndarray], ...]
    weights: np.ndarray
    window: float
    alpha_fixed: float | None
    mark_signature: str | None

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


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_nonnegative_scalar(value: Any, *, label: str) -> float:
    result = _finite_scalar(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _mark_mgf(mark_dist: Any, argument: float) -> float:
    """Return a verified exponential moment for supported mark laws."""
    if argument == 0.0:
        return 1.0
    mgf = getattr(mark_dist, "mgf", None)
    if callable(mgf):
        value = float(mgf(argument))
    else:
        from mixle.stats.univariate.continuous.exponential import (
            ExponentialDistribution,
        )
        from mixle.stats.univariate.continuous.gaussian import (
            GaussianDistribution,
        )

        if isinstance(mark_dist, GaussianDistribution):
            value = math.exp(mark_dist.mu * argument + 0.5 * mark_dist.sigma2 * argument * argument)
        elif isinstance(mark_dist, ExponentialDistribution):
            denominator = 1.0 - mark_dist.beta * argument
            value = 1.0 / denominator if denominator > 0.0 else np.inf
        else:
            raise TypeError(
                "marked PowerLawHawkes requires a mark distribution with a "
                "verified mgf(argument), or a built-in Gaussian/Exponential "
                "mark law"
            )
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("mark distribution must have a finite positive MGF at alpha")
    return value


def _validated_realization(
    value: Any,
    *,
    window: float,
    mark_dist: Any,
    fail_closed: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("PowerLawHawkes realization must be (times, marks)")
            times = np.asarray(value[0], dtype=np.float64)
            marks = np.asarray(value[1], dtype=np.float64)
        else:
            times = np.asarray(value, dtype=np.float64)
            marks = np.zeros(times.shape, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        if fail_closed:
            raise ValueError("PowerLawHawkes times and marks must be numeric") from exc
        return None
    structural = times.ndim != 1 or marks.ndim != 1 or times.shape != marks.shape
    if structural:
        raise ValueError("PowerLawHawkes times and marks must be aligned one-dimensional arrays")
    invalid = (
        np.any(~np.isfinite(times))
        or np.any(~np.isfinite(marks))
        or np.any(times < 0.0)
        or np.any(times > window)
        or (times.size > 1 and np.any(np.diff(times) <= 0.0))
        or (mark_dist is None and np.any(marks != 0.0))
    )
    if invalid:
        if fail_closed:
            raise ValueError(
                "PowerLawHawkes requires finite, strictly increasing times "
                "inside [0, window] and finite supported marks; unmarked "
                "models require zero marks"
            )
        return None
    if mark_dist is not None:
        for mark in marks:
            score = float(mark_dist.log_density(float(mark)))
            if np.isnan(score) or score == np.inf or score == -np.inf:
                if fail_closed:
                    raise ValueError("PowerLawHawkes mark lies outside the mark law support")
                return None
    owned_times = times.copy()
    owned_marks = marks.copy()
    owned_times.setflags(write=False)
    owned_marks.setflags(write=False)
    return owned_times, owned_marks


def _validated_history(
    times: Any,
    marks: Any,
    *,
    window: float,
    mark_dist: Any,
) -> tuple[np.ndarray, np.ndarray]:
    value = times if marks is None else (times, marks)
    checked = _validated_realization(
        value,
        window=window,
        mark_dist=mark_dist,
        fail_closed=True,
    )
    if checked is None:  # pragma: no cover
        raise AssertionError("unreachable")
    return checked


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    if value is None:
        return np.ones(rows, dtype=np.float64)
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("PowerLawHawkes weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"PowerLawHawkes weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("PowerLawHawkes weights must be finite and non-negative")
    return weights


def _validated_statistics(
    value: Any,
    *,
    window: float,
    alpha_fixed: float | None,
    mark_dist: Any,
) -> PowerLawHawkesStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise ValueError(
            "PowerLawHawkes statistics must contain realizations, weights, "
            "window, alpha constraint, and mark-law signature"
        )
    (
        raw_realizations,
        raw_weights,
        raw_window,
        raw_alpha,
        raw_mark_signature,
    ) = value
    statistic_window = _finite_scalar(
        raw_window,
        label="PowerLawHawkes statistic window",
    )
    if not math.isclose(
        statistic_window,
        window,
        rel_tol=0.0,
        abs_tol=_WINDOW_TOLERANCE * max(1.0, window),
    ):
        raise ValueError("PowerLawHawkes statistic window does not match estimator")
    checked_alpha = (
        None
        if raw_alpha is None
        else _finite_scalar(
            raw_alpha,
            label="PowerLawHawkes fixed alpha",
        )
    )
    if checked_alpha != alpha_fixed:
        raise ValueError("PowerLawHawkes statistic alpha constraint does not match estimator")
    expected_signature = None if mark_dist is None else str(mark_dist)
    if raw_mark_signature != expected_signature:
        raise ValueError("PowerLawHawkes statistic mark law does not match estimator")
    if not isinstance(raw_realizations, (tuple, list)):
        raise ValueError("PowerLawHawkes realizations must be a sequence")
    realizations = tuple(
        _validated_realization(
            realization,
            window=window,
            mark_dist=mark_dist,
            fail_closed=True,
        )
        for realization in raw_realizations
    )
    if any(item is None for item in realizations):  # pragma: no cover
        raise AssertionError("unreachable")
    weights = _validated_weights(
        raw_weights,
        len(realizations),
    ).copy()
    weights.setflags(write=False)
    return PowerLawHawkesStatistics(
        realizations,
        weights,
        window,
        alpha_fixed,
        expected_signature,
    )


class PowerLawHawkesDistribution(SequenceEncodableProbabilityDistribution):
    """Marked power-law-kernel Hawkes process on a fixed window ``[0, window]``.

    A realization is ``(times, marks)`` -- a sorted event-time array and a matching mark array (use zeros,
    or omit, for the unmarked process). ``mu > 0`` is the background rate, ``A >= 0`` the productivity,
    ``alpha`` the mark sensitivity, and ``c > 0``, ``p > 1`` the Omori-Utsu kernel scale and exponent.
    """

    def __init__(
        self,
        mu,
        A,
        c,
        p,
        window,
        *,
        alpha=0.0,
        mark_dist=None,
        name=None,
        keys=None,
        max_events: int = _DEFAULT_MAX_EVENTS,
        max_work: int = _DEFAULT_MAX_WORK,
    ):
        (
            self.mu,
            self.A,
            self.alpha,
            self.c,
            self.p,
            self.window,
        ) = (
            _finite_scalar(v, label=label)
            for v, label in zip(
                (mu, A, alpha, c, p, window),
                ("mu", "A", "alpha", "c", "p", "window"),
            )
        )
        if not (self.mu > 0 and self.A >= 0 and self.c > 0 and self.p > 1 and self.window > 0):
            raise ValueError("PowerLawHawkes requires finite mu>0, A>=0, c>0, p>1, and window>0")
        if mark_dist is None and self.alpha != 0.0:
            raise ValueError("unmarked PowerLawHawkes requires alpha=0 because every mark is identically zero")
        if mark_dist is not None:
            for method in ("log_density", "sampler"):
                if not callable(getattr(mark_dist, method, None)):
                    raise TypeError(f"PowerLawHawkes mark distribution must expose {method}()")
            _mark_mgf(mark_dist, self.alpha)
        self.mark_dist = mark_dist
        self.max_events = _exact_integer(
            max_events,
            label="PowerLawHawkes maximum event count",
        )
        self.max_work = _exact_integer(
            max_work,
            label="PowerLawHawkes maximum branching work",
        )
        if self.max_events < 0 or self.max_work < 0:
            raise ValueError("PowerLawHawkes sampling budgets must be non-negative")
        self.name = name
        self.keys = keys

    def __str__(self):
        return (
            "PowerLawHawkesDistribution(%r, %r, %r, %r, %r, alpha=%r, "
            "mark_dist=%s, name=%r, keys=%r, max_events=%r, max_work=%r)"
        ) % (
            self.mu,
            self.A,
            self.c,
            self.p,
            self.window,
            self.alpha,
            str(self.mark_dist) if self.mark_dist is not None else "None",
            self.name,
            self.keys,
            self.max_events,
            self.max_work,
        )

    @staticmethod
    def _unpack(x) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(x, tuple):
            if len(x) != 2:
                raise ValueError("PowerLawHawkes realization must be (times, marks)")
            t, m = x
            return np.asarray(t, dtype=float), np.asarray(m, dtype=float)
        t = np.asarray(x, dtype=float)
        return t, np.zeros_like(t)

    def intensity(self, t: float, times, marks=None) -> float:
        """The conditional rate ``lambda(t)`` given the catalogue so far -- the instantaneous forecast."""
        query = _finite_nonnegative_scalar(
            t,
            label="PowerLawHawkes query time",
        )
        if query > self.window:
            raise ValueError("PowerLawHawkes query time must lie inside [0, window]")
        times, marks = _validated_history(
            times,
            marks,
            window=self.window,
            mark_dist=self.mark_dist,
        )
        past = times < query
        trig = self.A * np.exp(self.alpha * marks[past]) * (1.0 + (query - times[past]) / self.c) ** (-self.p)
        if np.any(~np.isfinite(trig)):
            raise OverflowError("PowerLawHawkes productivity overflowed for query history")
        return float(self.mu + trig.sum())

    def expected_count(self, t_start: float, t_end: float, times, marks=None) -> float:
        """Expected number of events in ``[t_start, t_end]`` given the catalogue -- the window forecast."""
        start = _finite_nonnegative_scalar(
            t_start,
            label="PowerLawHawkes interval start",
        )
        end = _finite_nonnegative_scalar(
            t_end,
            label="PowerLawHawkes interval end",
        )
        if start > end or end > self.window:
            raise ValueError("PowerLawHawkes interval must satisfy 0 <= start <= end <= window")
        times, marks = _validated_history(
            times,
            marks,
            window=self.window,
            mark_dist=self.mark_dist,
        )
        rel = times < end
        tp, mp = times[rel], marks[rel]
        prod = self.A * np.exp(self.alpha * mp) * self.c / (self.p - 1.0)
        if np.any(~np.isfinite(prod)):
            raise OverflowError("PowerLawHawkes productivity overflowed for query history")
        lo = np.maximum(start, tp)
        omori = (1.0 + (lo - tp) / self.c) ** (1.0 - self.p) - (1.0 + (end - tp) / self.c) ** (1.0 - self.p)
        return float(self.mu * (end - start) + np.sum(prod * omori))

    def branching_ratio(self) -> float:
        """Return exact expected direct offspring using the mark-law MGF."""
        mark_factor = 1.0 if self.mark_dist is None else _mark_mgf(self.mark_dist, self.alpha)
        return self.A * self.c / (self.p - 1.0) * mark_factor

    def density(self, x) -> float:
        """Return the realization likelihood as a density on event times."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x) -> float:
        """Exact log-likelihood of one realization on ``[0, window]``.

        For a marked process (``mark_dist is not None``) this is the temporal Hawkes
        log-likelihood plus ``sum_i log mark_dist.density(m_i)``: marks are modeled as iid draws
        given their event times, so their log-densities simply add on to the point-process term.
        """
        checked = _validated_realization(
            x,
            window=self.window,
            mark_dist=self.mark_dist,
            fail_closed=False,
        )
        if checked is None:
            return -np.inf
        t, m = checked
        with np.errstate(over="ignore"):
            prod = self.A * np.exp(self.alpha * m)
        if np.any(~np.isfinite(prod)):
            return -np.inf
        lam = np.full(t.size, self.mu)
        for j in range(1, t.size):  # O(n^2): the power-law kernel has no finite-state recursion
            lam[j] += np.sum(prod[:j] * (1.0 + (t[j] - t[:j]) / self.c) ** (-self.p))
        integral = self.mu * self.window + np.sum(
            prod * self.c / (self.p - 1.0) * (1.0 - (1.0 + (self.window - t) / self.c) ** (1.0 - self.p))
        )
        mark_ll = sum(self.mark_dist.log_density(float(mi)) for mi in m) if self.mark_dist is not None else 0.0
        return float(np.sum(np.log(lam)) - integral + mark_ll)

    def seq_log_density(self, x) -> np.ndarray:
        """Return log-likelihoods for a batch of point-process realizations (mark term included; see log_density)."""
        return np.array([self.log_density(r) for r in x])

    def sampler(self, seed: int | None = None) -> PowerLawHawkesSampler:
        """Return a branching-process sampler for event catalogues."""
        return PowerLawHawkesSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> PowerLawHawkesEstimator:
        """Return the maximum-likelihood estimator for realizations on this window."""
        if pseudo_count is not None:
            checked_pseudo = _finite_nonnegative_scalar(
                pseudo_count,
                label="PowerLawHawkes pseudo-count",
            )
            if checked_pseudo != 0.0:
                raise NotImplementedError(
                    "PowerLawHawkes does not define an implicit pseudo-count "
                    "prior; fit explicit penalized likelihood instead"
                )
        return PowerLawHawkesEstimator(
            self.window,
            alpha_fixed=self.alpha if self.alpha == 0.0 else None,
            mark_dist=self.mark_dist,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
            max_work=self.max_work,
        )

    def dist_to_encoder(self) -> PowerLawHawkesDataEncoder:
        """Return the pass-through realization encoder used by vectorized methods."""
        return PowerLawHawkesDataEncoder(
            self.window,
            self.mark_dist,
        )


class PowerLawHawkesSampler(DistributionSampler):
    """Draw catalogues by branching: Poisson background + power-law-distributed offspring."""

    def __init__(self, dist: PowerLawHawkesDistribution, seed: int | None = None):
        self.dist = dist
        self.rng = RandomState(seed)
        self._mark = dist.mark_dist.sampler(self.rng.randint(0, 2**31 - 1)) if dist.mark_dist is not None else None
        self.last_receipt: dict[str, Any] | None = None
        if dist.branching_ratio() >= 1.0:
            warnings.warn(
                "super-critical PowerLawHawkes process may exhaust its explicit simulation budgets",
                stacklevel=2,
            )

    def _draw_mark(self) -> float:
        if self._mark is None:
            return 0.0
        mark = _finite_scalar(
            self._mark.sample(),
            label="PowerLawHawkes sampled mark",
        )
        score = float(self.dist.mark_dist.log_density(mark))
        if not np.isfinite(score):
            raise ValueError("PowerLawHawkes mark sampler produced a value outside its declared mark law")
        return mark

    def _sample_one(self):
        d = self.dist
        background_mean = d.mu * d.window
        if not np.isfinite(background_mean):
            raise OverflowError("PowerLawHawkes background event expectation overflowed")
        background_count = int(self.rng.poisson(background_mean))
        if background_count > d.max_events:
            self.last_receipt = {
                "complete": False,
                "events_generated": 0,
                "work_used": 0,
                "event_budget": d.max_events,
                "work_budget": d.max_work,
                "branching_ratio": d.branching_ratio(),
                "termination_reason": "background_event_budget_exhausted",
            }
            raise RuntimeError("PowerLawHawkes background draw exceeded the event budget")
        times = list(self.rng.uniform(0, d.window, background_count))
        marks = [self._draw_mark() for _ in times]
        queue = list(zip(times, marks))
        work = 0
        while queue:
            ti, mi = queue.pop()
            try:
                expected = d.A * math.exp(d.alpha * mi) * d.c / (d.p - 1.0)
            except OverflowError:
                expected = np.inf
            if not np.isfinite(expected):
                self.last_receipt = {
                    "complete": False,
                    "events_generated": len(times),
                    "work_used": work,
                    "event_budget": d.max_events,
                    "work_budget": d.max_work,
                    "branching_ratio": d.branching_ratio(),
                    "termination_reason": "offspring_intensity_overflow",
                }
                raise RuntimeError("PowerLawHawkes offspring expectation overflowed")
            offspring = int(self.rng.poisson(expected))
            if work + offspring > d.max_work:
                self.last_receipt = {
                    "complete": False,
                    "events_generated": len(times),
                    "work_used": work,
                    "event_budget": d.max_events,
                    "work_budget": d.max_work,
                    "branching_ratio": d.branching_ratio(),
                    "termination_reason": "work_budget_exhausted",
                }
                raise RuntimeError("PowerLawHawkes branching work budget exhausted")
            work += offspring
            for _ in range(offspring):
                tau = d.c * ((1 - self.rng.uniform()) ** (-1.0 / (d.p - 1.0)) - 1.0)  # power-law inter-time
                tc = ti + tau
                if tc < d.window:
                    if len(times) >= d.max_events:
                        self.last_receipt = {
                            "complete": False,
                            "events_generated": len(times),
                            "work_used": work,
                            "event_budget": d.max_events,
                            "work_budget": d.max_work,
                            "branching_ratio": d.branching_ratio(),
                            "termination_reason": "event_budget_exhausted",
                        }
                        raise RuntimeError("PowerLawHawkes event budget exhausted")
                    mc = self._draw_mark()
                    times.append(tc)
                    marks.append(mc)
                    queue.append((tc, mc))
        order = np.argsort(times)
        self.last_receipt = {
            "complete": True,
            "events_generated": len(times),
            "work_used": work,
            "event_budget": d.max_events,
            "work_budget": d.max_work,
            "branching_ratio": d.branching_ratio(),
            "termination_reason": "queue_exhausted",
        }
        return np.asarray(times)[order], np.asarray(marks)[order]

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one catalogue or a list of catalogues by the branching construction."""
        if size is None:
            return self._sample_one()
        checked_size = _exact_integer(
            size,
            label="PowerLawHawkes sample size",
        )
        if checked_size < 0:
            raise ValueError("PowerLawHawkes sample size must be non-negative")
        return [self._sample_one() for _ in range(checked_size)]


class PowerLawHawkesDataEncoder(DataSequenceEncoder):
    """Pass-through encoder: a realization is a ``(times, marks)`` tuple; a batch is a list of them."""

    def __init__(self, window: float, mark_dist: Any) -> None:
        self.window = _finite_scalar(
            window,
            label="PowerLawHawkes encoder window",
        )
        if self.window <= 0.0:
            raise ValueError("PowerLawHawkes encoder requires a window > 0")
        self.mark_dist = mark_dist

    def __str__(self) -> str:
        return "PowerLawHawkesDataEncoder(%r, %s)" % (
            self.window,
            str(self.mark_dist) if self.mark_dist is not None else "None",
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PowerLawHawkesDataEncoder)
            and self.window == other.window
            and str(self.mark_dist) == str(other.mark_dist)
        )

    def seq_encode(self, x):
        """Return validated owned realization snapshots."""
        result = []
        for realization in x:
            checked = _validated_realization(
                realization,
                window=self.window,
                mark_dist=self.mark_dist,
                fail_closed=True,
            )
            if checked is None:  # pragma: no cover
                raise AssertionError("unreachable")
            result.append(checked)
        return result

    def row_count(self, x: Any) -> int:
        """Return the number of validated encoded realizations."""
        rows = list(x)
        self.seq_encode(rows)
        return len(rows)


class PowerLawHawkesEstimator(ParameterEstimator):
    """Maximum-likelihood estimator of ``(mu, A, alpha, c, p)`` over realizations on a common window."""

    def __init__(
        self,
        window: float,
        *,
        alpha_fixed: float | None = None,
        mark_dist: Any = None,
        name=None,
        keys=None,
        max_events: int = _DEFAULT_MAX_EVENTS,
        max_work: int = _DEFAULT_MAX_WORK,
    ):
        self.window = _finite_scalar(
            window,
            label="PowerLawHawkes estimator window",
        )
        if self.window <= 0.0:
            raise ValueError("PowerLawHawkes estimator requires a window > 0")
        self.alpha_fixed = (
            None
            if alpha_fixed is None
            else _finite_scalar(
                alpha_fixed,
                label="PowerLawHawkes fixed alpha",
            )
        )
        self.mark_dist = mark_dist
        if self.mark_dist is None and self.alpha_fixed is None:
            raise ValueError("unmarked PowerLawHawkes estimation must fix alpha=0")
        self.name = name
        self.keys = keys
        self.max_events = _exact_integer(
            max_events,
            label="PowerLawHawkes maximum event count",
        )
        self.max_work = _exact_integer(
            max_work,
            label="PowerLawHawkes maximum branching work",
        )
        if self.max_events < 0 or self.max_work < 0:
            raise ValueError("PowerLawHawkes sampling budgets must be non-negative")

    def accumulator_factory(self):
        """Return a factory for raw-realization Hawkes accumulators."""
        return PowerLawHawkesAccumulatorFactory(
            self.window,
            self.alpha_fixed,
            self.mark_dist,
            name=self.name,
            keys=self.keys,
        )

    def estimate(self, nobs, suff_stat) -> PowerLawHawkesDistribution:
        """Fit Hawkes parameters by weighted numerical maximum likelihood.

        Each realization's log-likelihood is scaled by its accumulated weight, so this is the
        weighted MLE ``argmax sum_i weight_i * log_density(realization_i)`` -- the objective EM
        responsibilities and other weighted ``optimize()`` callers rely on.
        """
        checked = _validated_statistics(
            suff_stat,
            window=self.window,
            alpha_fixed=self.alpha_fixed,
            mark_dist=self.mark_dist,
        )
        alpha_fixed = checked.alpha_fixed
        unmarked = alpha_fixed is not None
        n_total = sum(
            weight * len(times)
            for (times, _), weight in zip(
                checked.realizations,
                checked.weights,
            )
        )
        total_weight = float(np.sum(checked.weights))
        if total_weight <= 0.0:
            raise ValueError("PowerLawHawkes fitting requires positive realization weight")
        if n_total <= 0.0:
            raise ValueError(
                "PowerLawHawkes evidence with no positively weighted events has no finite interior baseline-rate MLE"
            )

        def negll(theta):
            mu, a, c, pm1 = np.exp(theta[[0, 1, 3, 4]])
            alpha = alpha_fixed if unmarked else theta[2]
            if self.mark_dist is not None:
                try:
                    _mark_mgf(self.mark_dist, float(alpha))
                except (TypeError, ValueError, OverflowError):
                    return np.inf
            total = 0.0
            for (times, marks), weight in zip(
                checked.realizations,
                checked.weights,
            ):
                if weight == 0.0:
                    continue
                with np.errstate(over="ignore"):
                    productivity = a * np.exp(alpha * marks)
                if np.any(~np.isfinite(productivity)):
                    return np.inf
                intensity = np.full(times.size, mu)
                for index in range(1, times.size):
                    intensity[index] += np.sum(
                        productivity[:index] * (1.0 + (times[index] - times[:index]) / c) ** (-(1.0 + pm1))
                    )
                if np.any(~np.isfinite(intensity)) or np.any(intensity <= 0.0):
                    return np.inf
                compensator = mu * self.window + np.sum(
                    productivity * c / pm1 * (1.0 - (1.0 + (self.window - times) / c) ** (-pm1))
                )
                total += float(weight) * (float(np.sum(np.log(intensity))) - compensator)
            return -total if np.isfinite(total) else np.inf

        empirical_rate = n_total / (total_weight * self.window)
        alpha_seed = 0.0 if alpha_fixed is not None else 1.0
        starts = [
            [
                np.log(max(empirical_rate, 1e-3)),
                np.log(1.0e-4),
                0.0,
                np.log(0.02),
                np.log(0.3),
            ],
            [
                np.log(max(0.7 * empirical_rate, 1e-3)),
                np.log(1.0),
                alpha_seed,
                np.log(0.02),
                np.log(0.3),
            ],
            [
                np.log(max(0.5 * empirical_rate, 1e-3)),
                np.log(4.0),
                alpha_seed,
                np.log(0.02),
                np.log(0.3),
            ],
            [
                np.log(max(0.3 * empirical_rate, 1e-3)),
                np.log(2.0),
                alpha_seed,
                np.log(0.1),
                np.log(0.5),
            ],
        ]
        bounds = [
            (np.log(1e-4), np.log(1e3)),
            (np.log(1e-4), np.log(1e3)),
            (-4.0, 4.0),
            (np.log(1e-4), np.log(10.0)),
            (np.log(1e-3), np.log(5.0)),
        ]
        candidates = []
        messages = []
        for start in starts:
            result = minimize(
                negll,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1.0e-12},
            )
            messages.append(str(result.message))
            if result.success and np.isfinite(result.fun):
                candidates.append(result)
        if not candidates:
            raise RuntimeError("PowerLawHawkes likelihood optimization failed: " + "; ".join(messages))
        res = min(candidates, key=lambda result: float(result.fun))
        mu, a, c, pm1 = np.exp(res.x[[0, 1, 3, 4]])
        alpha = alpha_fixed if unmarked else res.x[2]
        return PowerLawHawkesDistribution(
            mu,
            a,
            c,
            1.0 + pm1,
            self.window,
            alpha=alpha,
            mark_dist=self.mark_dist,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
            max_work=self.max_work,
        )


class PowerLawHawkesAccumulator(SequenceEncodableStatisticAccumulator):
    """Collects weighted realizations (the MLE needs the full event times, not closed-form sufficient statistics).

    Each stored entry is a ``(realization, weight)`` pair; ``PowerLawHawkesEstimator.estimate`` fits by
    weighted maximum likelihood, ``argmax sum_i weight_i * log_density(realization_i)``, so a
    realization's weight controls how much it contributes -- e.g. EM component responsibilities or
    ``optimize()``-supplied sample weights -- exactly like a closed-form accumulator's weighted sufficient
    statistics would.
    """

    def __init__(
        self,
        window: float,
        alpha_fixed: float | None,
        mark_dist: Any,
        name=None,
        keys=None,
    ):
        self.window = _finite_scalar(
            window,
            label="PowerLawHawkes accumulator window",
        )
        if self.window <= 0.0:
            raise ValueError("PowerLawHawkes accumulator requires a window > 0")
        self.alpha_fixed = alpha_fixed
        self.mark_dist = mark_dist
        self.realizations: list[tuple[np.ndarray, np.ndarray]] = []
        self.weights: list[float] = []
        self.name = name
        self.keys = keys

    def _store(self, x: Any, weight: Any) -> None:
        realization = _validated_realization(
            x,
            window=self.window,
            mark_dist=self.mark_dist,
            fail_closed=True,
        )
        if realization is None:  # pragma: no cover
            raise AssertionError("unreachable")
        checked_weight = _finite_nonnegative_scalar(
            weight,
            label="PowerLawHawkes weight",
        )
        self.realizations.append(realization)
        self.weights.append(checked_weight)

    def update(self, x, weight, estimate):
        """Store one realization together with its weight for maximum-likelihood fitting."""
        self._store(x, weight)

    def initialize(self, x, weight, rng):
        """Store one realization together with its weight during initialization."""
        self._store(x, weight)

    def seq_update(self, x, weights, estimate):
        """Store a batch of realizations together with their weights for maximum-likelihood fitting."""
        rows = list(x)
        checked_weights = _validated_weights(weights, len(rows))
        for realization, weight in zip(rows, checked_weights):
            self._store(realization, float(weight))

    def seq_initialize(self, x, weights, rng):
        """Store a batch of realizations together with their weights during initialization."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):
        """Merge stored weighted realization catalogues."""
        checked = _validated_statistics(
            suff_stat,
            window=self.window,
            alpha_fixed=self.alpha_fixed,
            mark_dist=self.mark_dist,
        )
        self.realizations.extend(checked.realizations)
        self.weights.extend(checked.weights.tolist())
        return self

    def scale(self, c: float) -> PowerLawHawkesAccumulator:
        """Multiply every stored realization's weight by ``c``.

        The raw event catalogue (times/marks) is data, not a sufficient statistic, and must stay
        unscaled -- only the weight that says how much each realization counts is linear and gets
        rescaled, matching the sibling Hawkes accumulators' ``scale()`` contract of multiplying
        every accumulated weighted quantity by ``c`` (see e.g. HawkesProcessAccumulator.scale).
        """
        checked = _finite_nonnegative_scalar(
            c,
            label="PowerLawHawkes scale",
        )
        self.weights = [weight * checked for weight in self.weights]
        return self

    def value(self) -> PowerLawHawkesStatistics:
        """Return stored weighted realizations with the shared window and alpha constraint."""
        weights = np.asarray(self.weights, dtype=np.float64)
        weights.setflags(write=False)
        return PowerLawHawkesStatistics(
            tuple(self.realizations),
            weights,
            self.window,
            self.alpha_fixed,
            None if self.mark_dist is None else str(self.mark_dist),
        )

    def from_value(self, x):
        """Restore stored weighted realizations, window, and alpha constraint."""
        checked = _validated_statistics(
            x,
            window=self.window,
            alpha_fixed=self.alpha_fixed,
            mark_dist=self.mark_dist,
        )
        self.realizations = list(checked.realizations)
        self.weights = checked.weights.tolist()
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

    def acc_to_encoder(self):
        """Return the encoder compatible with raw Hawkes realizations."""
        return PowerLawHawkesDataEncoder(
            self.window,
            self.mark_dist,
        )


class PowerLawHawkesAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for PowerLawHawkesAccumulator."""

    def __init__(
        self,
        window: float,
        alpha_fixed: float | None,
        mark_dist: Any,
        name=None,
        keys=None,
    ):
        self.window = float(window)
        self.alpha_fixed = alpha_fixed
        self.mark_dist = mark_dist
        self.name = name
        self.keys = keys

    def make(self) -> PowerLawHawkesAccumulator:
        """Create an empty power-law Hawkes accumulator."""
        return PowerLawHawkesAccumulator(
            self.window,
            self.alpha_fixed,
            self.mark_dist,
            name=self.name,
            keys=self.keys,
        )
