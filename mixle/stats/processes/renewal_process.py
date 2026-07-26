"""Renewal process -- a point process whose inter-arrival times are i.i.d. from a base distribution.

A renewal process on the window ``[0, T]`` generates event times ``0 < t_1 < t_2 < ... < t_n <= T`` such
that the gaps ``g_1 = t_1``, ``g_i = t_i - t_{i-1}`` are i.i.d. draws from a positive-support
*inter-arrival* distribution ``f`` (e.g. Gamma, Weibull, LogGaussian, InverseGaussian, Exponential -- the
Poisson process is the Exponential special case). The exact log-likelihood of one realization is

    log L = sum_i log f(g_i) + log S(T - t_n),     S(x) = 1 - F(x)  (the survival of the censored last gap)

where the final term is the probability that no further event occurred before the window closed (with
``t_0 = 0`` and ``t_n = 0`` when there are no events, so an empty realization scores ``log S(T)``). Scoring
is therefore exact, including the right-censored boundary.

Estimation retains both completed gaps and every realization's right-censored
tail, and fits the full finite-window likelihood. Exponential inter-arrivals
use an exact closed-form censored MLE/update; Gamma, Weibull, LogGaussian, and
InverseGaussian inter-arrivals use bounded numerical likelihood optimization.
Other child estimators fail closed instead of silently discarding censoring.
The window ``T`` is fixed, known configuration and is never estimated.
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

_CDF_TOLERANCE = 1.0e-12
_DEFAULT_MAX_EVENTS = 100_000


class RenewalProcessStatistics(NamedTuple):
    """Versioned renewal evidence including right-censoring."""

    gap_statistics: Any
    completed_gaps: np.ndarray
    completed_weights: np.ndarray
    censored_times: np.ndarray
    censored_weights: np.ndarray

    @property
    def schema_version(self) -> int:
        """Return the serialized statistic schema version."""
        return 1


def _exact_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an exact integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an exact integer") from exc


def _nonnegative_scalar(value: Any, *, label: str) -> float:
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
        events = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        if fail_closed:
            raise ValueError("RenewalProcess event times must be numeric") from exc
        return None
    invalid = (
        np.any(~np.isfinite(events))
        or np.any(events <= 0.0)
        or np.any(events > window)
        or (events.size > 1 and np.any(np.diff(events) <= 0.0))
    )
    if invalid:
        if fail_closed:
            raise ValueError(
                "RenewalProcess events must be finite, strictly increasing, "
                "and inside (0, window]"
            )
        return None
    return events


def _gaps_and_remaining(
    value: Any,
    *,
    window: float,
    fail_closed: bool,
) -> tuple[np.ndarray, float] | None:
    events = _validated_events(
        value,
        window=window,
        fail_closed=fail_closed,
    )
    if events is None:
        return None
    gaps = (
        np.diff(np.concatenate(([0.0], events)))
        if events.size
        else np.empty(0, dtype=np.float64)
    )
    last = float(events[-1]) if events.size else 0.0
    return gaps, window - last


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("RenewalProcess weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"RenewalProcess weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("RenewalProcess weights must be finite and non-negative")
    return weights


def _validated_statistics(value: Any) -> RenewalProcessStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise ValueError(
            "RenewalProcess statistics must contain gap and censoring evidence"
        )
    arrays = []
    for index, item in enumerate(value[1:]):
        try:
            array = np.asarray(item, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RenewalProcess evidence field {index + 1} must be numeric"
            ) from exc
        if array.ndim != 1:
            raise ValueError("RenewalProcess evidence arrays must be one-dimensional")
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(
                "RenewalProcess gaps, censoring times, and weights must be "
                "finite and non-negative"
            )
        arrays.append(array.copy())
    gaps, gap_weights, censored, censor_weights = arrays
    if gaps.shape != gap_weights.shape:
        raise ValueError(
            "RenewalProcess completed gaps and weights must align"
        )
    if censored.shape != censor_weights.shape:
        raise ValueError(
            "RenewalProcess censoring times and weights must align"
        )
    if np.any(gaps <= 0.0):
        raise ValueError("RenewalProcess completed gaps must be strictly positive")
    return RenewalProcessStatistics(
        value[0],
        gaps,
        gap_weights,
        censored,
        censor_weights,
    )


def _validated_payload(
    value: Any,
    *,
    gap_encoder: DataSequenceEncoder,
    window: float,
) -> tuple[Any, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError(
            "RenewalProcess encoded payload must contain six fields"
        )
    child_enc, raw_gaps, seg_ids, num_real, remaining, ok = value
    num_real = _exact_integer(
        num_real,
        label="RenewalProcess encoded realization count",
    )
    if num_real < 0:
        raise ValueError(
            "RenewalProcess encoded realization count must be non-negative"
        )
    try:
        gaps = np.asarray(raw_gaps, dtype=np.float64)
        segments_raw = np.asarray(seg_ids)
        remaining = np.asarray(remaining, dtype=np.float64)
        ok = np.asarray(ok)
    except (TypeError, ValueError) as exc:
        raise ValueError("RenewalProcess encoded metadata must be numeric") from exc
    if gaps.ndim != 1 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.0):
        raise ValueError(
            "RenewalProcess encoded gaps must be finite and strictly positive"
        )
    if (
        segments_raw.ndim != 1
        or segments_raw.shape != gaps.shape
        or segments_raw.dtype == np.bool_
        or not np.issubdtype(segments_raw.dtype, np.integer)
    ):
        raise ValueError(
            "RenewalProcess segment ids must be aligned exact integers"
        )
    segments = segments_raw.astype(np.int64, copy=False)
    if np.any(segments < 0) or np.any(segments >= num_real):
        raise ValueError("RenewalProcess segment id is outside realization range")
    if remaining.shape != (num_real,):
        raise ValueError(
            "RenewalProcess remaining times must match realization count"
        )
    if (
        np.any(~np.isfinite(remaining))
        or np.any(remaining < 0.0)
        or np.any(remaining > window)
    ):
        raise ValueError(
            "RenewalProcess remaining times must lie inside the fixed window"
        )
    if ok.dtype != np.bool_ or ok.shape != (num_real,):
        raise ValueError(
            "RenewalProcess validity flags must be a Boolean realization vector"
        )
    if int(gap_encoder.row_count(child_enc)) != gaps.size:
        raise ValueError(
            "RenewalProcess child encoding does not match completed gaps"
        )
    elapsed = np.bincount(
        segments,
        weights=gaps,
        minlength=num_real,
    )
    if np.any(
        ok
        & ~np.isclose(
            elapsed + remaining,
            window,
            rtol=0.0,
            atol=1.0e-10 * max(1.0, window),
        )
    ):
        raise ValueError(
            "RenewalProcess gaps and remaining time contradict the fixed window"
        )
    if np.any(~ok & ((elapsed != 0.0) | (remaining != 0.0))):
        raise ValueError(
            "Invalid RenewalProcess rows cannot carry likelihood evidence"
        )
    return child_enc, gaps, segments, num_real, remaining.copy(), ok.copy()

def _survival_logprob(interarrival: Any, remaining: np.ndarray) -> np.ndarray:
    """Return validated ``log(1 - cdf(remaining))`` values."""
    cdf = getattr(interarrival, "cdf", None)
    if not callable(cdf):
        raise TypeError(
            "RenewalProcess requires the inter-arrival distribution to expose cdf() for the censored "
            "boundary term (e.g. Gamma/Weibull/Exponential/LogGaussian/InverseGaussian)."
        )
    out = np.empty(remaining.shape[0], dtype=np.float64)
    for i, r in enumerate(remaining):
        f = float(cdf(float(r)))
        if not np.isfinite(f) or f < -_CDF_TOLERANCE or f > 1.0 + _CDF_TOLERANCE:
            raise ValueError(
                f"RenewalProcess inter-arrival cdf({float(r)!r}) returned "
                f"non-probability value {f!r}"
            )
        bounded = min(1.0, max(0.0, f))
        out[i] = math.log1p(-bounded) if bounded < 1.0 else -np.inf
    return out


class RenewalProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Renewal process with i.i.d. inter-arrivals ``interarrival`` observed on ``[0, window]``."""

    def __init__(
        self,
        interarrival: Any,
        window: float,
        name: str | None = None,
        keys: str | None = None,
        max_events: int = _DEFAULT_MAX_EVENTS,
    ) -> None:
        if (
            isinstance(window, (bool, np.bool_))
            or not (np.isfinite(window) and window > 0.0)
        ):
            raise ValueError("RenewalProcessDistribution requires a finite window > 0.")
        for method in ("log_density", "seq_log_density", "cdf", "sampler", "estimator", "dist_to_encoder"):
            if not callable(getattr(interarrival, method, None)):
                raise TypeError(
                    "RenewalProcess inter-arrival distribution must expose "
                    f"{method}()"
                )
        from mixle.stats.compute.declarations import declaration_for

        declaration = declaration_for(interarrival)
        if declaration is not None and declaration.support not in {
            "positive_real",
            "non_negative_real",
        }:
            raise TypeError(
                "RenewalProcess inter-arrival distributions must declare "
                "positive or non-negative real support"
            )
        checked_budget = _exact_integer(
            max_events,
            label="RenewalProcess maximum event count",
        )
        if checked_budget < 0:
            raise ValueError(
                "RenewalProcess maximum event count must be non-negative"
            )
        self.interarrival = interarrival
        self.window = float(window)
        self.name = name
        self.keys = keys
        self.max_events = checked_budget
        _survival_logprob(
            self.interarrival,
            np.asarray([0.0, self.window], dtype=np.float64),
        )

    def __str__(self) -> str:
        return "RenewalProcessDistribution(%s, %s, name=%s, keys=%s, max_events=%s)" % (
            str(self.interarrival),
            repr(self.window),
            repr(self.name),
            repr(self.keys),
            repr(self.max_events),
        )

    def _gaps_and_remaining(self, x: Any) -> tuple[np.ndarray, float] | None:
        return _gaps_and_remaining(
            x,
            window=self.window,
            fail_closed=False,
        )

    def log_density(self, x: Any) -> float:
        """Exact log-likelihood of one realization (observed gaps + censored survival)."""
        gr = self._gaps_and_remaining(x)
        if gr is None:
            return -np.inf
        gaps, remaining = gr
        ll = float(np.sum([self.interarrival.log_density(float(g)) for g in gaps])) if gaps.size else 0.0
        return ll + float(_survival_logprob(self.interarrival, np.array([remaining]))[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Vectorized log-likelihood for encoded realizations (flattened gaps + per-realization survival)."""
        child_enc, _, seg_ids, num_real, remaining, ok = _validated_payload(
            x,
            gap_encoder=self.interarrival.dist_to_encoder(),
            window=self.window,
        )
        rv = np.full(num_real, -np.inf, dtype=np.float64)
        per_real = np.zeros(num_real, dtype=np.float64)
        if seg_ids.size:
            gap_ll = np.asarray(self.interarrival.seq_log_density(child_enc), dtype=np.float64)
            np.add.at(per_real, seg_ids, gap_ll)
        per_real += _survival_logprob(self.interarrival, remaining)
        rv[ok] = per_real[ok]
        return rv

    def sampler(self, seed: int | None = None) -> "RenewalProcessSampler":
        """Return a sampler that draws gaps until the cumulative time exceeds ``window``."""
        return RenewalProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "RenewalProcessEstimator":
        """Return an estimator that fits the inter-arrival distribution to the observed gaps."""
        return RenewalProcessEstimator(
            self.interarrival.estimator(pseudo_count=pseudo_count),
            self.window,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
        )

    def dist_to_encoder(self) -> "RenewalProcessDataEncoder":
        """Return the data encoder (delegates gap encoding to the inter-arrival encoder)."""
        return RenewalProcessDataEncoder(self.interarrival.dist_to_encoder(), self.window)


class RenewalProcessSampler(DistributionSampler):
    """Draw inter-arrival gaps from ``interarrival`` until the cumulative time passes ``window``."""

    def __init__(self, dist: RenewalProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        # seed the inter-arrival sampler deterministically from this rng
        self._gap_sampler = dist.interarrival.sampler(seed=int(self.rng.randint(2**31)))
        if not callable(getattr(self._gap_sampler, "sample", None)):
            raise TypeError(
                "RenewalProcess inter-arrival sampler must expose sample()"
            )
        self.last_receipt: dict[str, Any] | None = None

    def _sample_one(self) -> np.ndarray:
        events: list[float] = []
        t = 0.0
        gap_draws = 0
        while True:
            raw_gap = self._gap_sampler.sample()
            g = _nonnegative_scalar(
                raw_gap,
                label="RenewalProcess sampled inter-arrival",
            )
            if g <= 0.0:
                raise ValueError(
                    "RenewalProcess sampled inter-arrivals must be strictly positive"
                )
            gap_draws += 1
            t += g
            if t > self.dist.window:
                self.last_receipt = {
                    "complete": True,
                    "events_generated": len(events),
                    "gap_draws": gap_draws,
                    "event_budget": self.dist.max_events,
                    "termination_reason": "window_crossed",
                }
                break
            if len(events) >= self.dist.max_events:
                self.last_receipt = {
                    "complete": False,
                    "events_generated": len(events),
                    "gap_draws": gap_draws,
                    "event_budget": self.dist.max_events,
                    "termination_reason": "event_budget_exhausted",
                }
                raise RuntimeError(
                    "RenewalProcess sampling event budget exhausted before "
                    "the observation window was crossed"
                )
            events.append(t)
        return np.asarray(events, dtype=np.float64)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray | list[np.ndarray]:
        """Draw one realization, or ``size`` iid realizations, on the fixed window."""
        if size is None:
            return self._sample_one()
        checked_size = _exact_integer(
            size,
            label="RenewalProcess sample size",
        )
        if checked_size < 0:
            raise ValueError("RenewalProcess sample size must be non-negative")
        return [self._sample_one() for _ in range(checked_size)]


class RenewalProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Feed the inter-arrival gaps to the inter-arrival distribution's accumulator (renewal MLE)."""

    def __init__(
        self,
        gap_accumulator: SequenceEncodableStatisticAccumulator,
        window: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.gap_accumulator = gap_accumulator
        self.window = float(window)
        self.name = name
        self.keys = keys
        self.completed_gaps: list[float] = []
        self.completed_weights: list[float] = []
        self.censored_times: list[float] = []
        self.censored_weights: list[float] = []

    def update(self, x: Any, weight: float, estimate: RenewalProcessDistribution | None) -> None:
        """Accumulate observed inter-arrival gaps from one realization."""
        gaps, remaining = _gaps_and_remaining(
            x,
            window=self.window,
            fail_closed=True,
        )
        checked_weight = _nonnegative_scalar(
            weight,
            label="RenewalProcess weight",
        )
        gap_est = estimate.interarrival if estimate is not None else None
        for g in gaps:
            self.gap_accumulator.update(float(g), checked_weight, gap_est)
        self.completed_gaps.extend(float(gap) for gap in gaps)
        self.completed_weights.extend([checked_weight] * len(gaps))
        self.censored_times.append(remaining)
        self.censored_weights.append(checked_weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the gap accumulator from one realization."""
        gaps, remaining = _gaps_and_remaining(
            x,
            window=self.window,
            fail_closed=True,
        )
        checked_weight = _nonnegative_scalar(
            weight,
            label="RenewalProcess weight",
        )
        for g in gaps:
            self.gap_accumulator.initialize(float(g), checked_weight, rng)
        self.completed_gaps.extend(float(gap) for gap in gaps)
        self.completed_weights.extend([checked_weight] * len(gaps))
        self.censored_times.append(remaining)
        self.censored_weights.append(checked_weight)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: RenewalProcessDistribution | None) -> None:
        """Accumulate encoded observed gaps with realization-level weights."""
        (
            child_enc,
            raw_gaps,
            seg_ids,
            num_real,
            remaining,
            ok,
        ) = _validated_payload(
            x,
            gap_encoder=self.gap_accumulator.acc_to_encoder(),
            window=self.window,
        )
        if not np.all(ok):
            raise ValueError(
                "RenewalProcess cannot accumulate invalid realizations"
            )
        checked_weights = _validated_weights(weights, num_real)
        if seg_ids.size:
            gap_weights = checked_weights[seg_ids]
            gap_est = estimate.interarrival if estimate is not None else None
            self.gap_accumulator.seq_update(child_enc, gap_weights, gap_est)
            self.completed_gaps.extend(raw_gaps.tolist())
            self.completed_weights.extend(gap_weights.tolist())
        self.censored_times.extend(remaining.tolist())
        self.censored_weights.extend(checked_weights.tolist())

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize from encoded observed gaps."""
        (
            child_enc,
            raw_gaps,
            seg_ids,
            num_real,
            remaining,
            ok,
        ) = _validated_payload(
            x,
            gap_encoder=self.gap_accumulator.acc_to_encoder(),
            window=self.window,
        )
        if not np.all(ok):
            raise ValueError(
                "RenewalProcess cannot initialize from invalid realizations"
            )
        checked_weights = _validated_weights(weights, num_real)
        if seg_ids.size:
            gap_weights = checked_weights[seg_ids]
            self.gap_accumulator.seq_initialize(child_enc, gap_weights, rng)
            self.completed_gaps.extend(raw_gaps.tolist())
            self.completed_weights.extend(gap_weights.tolist())
        self.censored_times.extend(remaining.tolist())
        self.censored_weights.extend(checked_weights.tolist())

    def combine(self, suff_stat: Any) -> "RenewalProcessAccumulator":
        """Merge another inter-arrival sufficient-statistic value."""
        checked = _validated_statistics(suff_stat)
        self.gap_accumulator.combine(checked.gap_statistics)
        self.completed_gaps.extend(checked.completed_gaps.tolist())
        self.completed_weights.extend(checked.completed_weights.tolist())
        self.censored_times.extend(checked.censored_times.tolist())
        self.censored_weights.extend(checked.censored_weights.tolist())
        return self

    def value(self) -> Any:
        """Return the wrapped inter-arrival accumulator value."""
        return RenewalProcessStatistics(
            self.gap_accumulator.value(),
            np.asarray(self.completed_gaps, dtype=np.float64),
            np.asarray(self.completed_weights, dtype=np.float64),
            np.asarray(self.censored_times, dtype=np.float64),
            np.asarray(self.censored_weights, dtype=np.float64),
        )

    def from_value(self, x: Any) -> "RenewalProcessAccumulator":
        """Replace the wrapped inter-arrival accumulator from ``x``."""
        checked = _validated_statistics(x)
        self.gap_accumulator.from_value(checked.gap_statistics)
        self.completed_gaps = checked.completed_gaps.tolist()
        self.completed_weights = checked.completed_weights.tolist()
        self.censored_times = checked.censored_times.tolist()
        self.censored_weights = checked.censored_weights.tolist()
        return self

    def scale(self, c: float) -> "RenewalProcessAccumulator":
        """Scale the wrapped inter-arrival sufficient statistics by ``c``."""
        checked_scale = _nonnegative_scalar(
            c,
            label="RenewalProcess scale",
        )
        self.gap_accumulator.scale(checked_scale)
        self.completed_weights = [
            weight * checked_scale for weight in self.completed_weights
        ]
        self.censored_weights = [
            weight * checked_scale for weight in self.censored_weights
        ]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic merging to the gap accumulator."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self
        else:
            self.gap_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic replacement to the gap accumulator."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())
        else:
            self.gap_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "RenewalProcessDataEncoder":
        """Return an encoder that converts event times into inter-arrival gaps."""
        return RenewalProcessDataEncoder(
            self.gap_accumulator.acc_to_encoder(),
            self.window,
        )


class RenewalProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for RenewalProcessAccumulator (wraps the inter-arrival accumulator factory)."""

    def __init__(
        self,
        gap_factory: StatisticAccumulatorFactory,
        window: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.gap_factory = gap_factory
        self.window = window
        self.name = name
        self.keys = keys

    def make(self) -> "RenewalProcessAccumulator":
        """Create a fresh renewal-process accumulator."""
        return RenewalProcessAccumulator(
            self.gap_factory.make(),
            self.window,
            name=self.name,
            keys=self.keys,
        )


def _estimate_censored_interarrival(
    estimator: ParameterEstimator,
    statistics: RenewalProcessStatistics,
    nobs: float | None,
) -> Any:
    """Fit supported inter-arrival families to completed and censored gaps."""
    from scipy.special import gammaincc, gammaln, log_ndtr
    from scipy.stats import invgauss

    from mixle.stats.univariate.continuous.exponential import (
        ExponentialEstimator,
    )
    from mixle.stats.univariate.continuous.gamma import (
        GammaDistribution,
    )
    from mixle.stats.univariate.continuous.inverse_gaussian import (
        InverseGaussianDistribution,
    )
    from mixle.stats.univariate.continuous.log_gaussian import (
        LogGaussianDistribution,
    )
    from mixle.stats.univariate.continuous.weibull import (
        WeibullDistribution,
    )

    if isinstance(estimator, ExponentialEstimator):
        gap_stat = statistics.gap_statistics
        if not isinstance(gap_stat, (tuple, list)) or len(gap_stat) != 2:
            raise ValueError(
                "Exponential renewal gap statistics must be (count, sum)"
            )
        exposure = float(
            np.dot(statistics.censored_times, statistics.censored_weights)
        )
        adjusted = (gap_stat[0], gap_stat[1] + exposure)
        if float(adjusted[0]) <= 0.0:
            raise ValueError(
                "A fully censored exponential renewal sample has no finite MLE"
            )
        return estimator.estimate(nobs, adjusted)

    initial = estimator.estimate(nobs, statistics.gap_statistics)
    gaps = statistics.completed_gaps
    gap_weights = statistics.completed_weights
    censored = statistics.censored_times
    censor_weights = statistics.censored_weights
    if float(gap_weights.sum()) <= 0.0:
        raise ValueError(
            "Fully censored renewal evidence has no finite identifiable fit"
        )

    name = getattr(estimator, "name", None)
    keys = getattr(estimator, "keys", None)
    log_gaps = np.log(gaps)

    if isinstance(initial, GammaDistribution):
        x0 = np.log([initial.k, initial.theta])

        def objective(parameters):
            shape, scale = np.exp(parameters)
            completed = (
                (shape - 1.0) * log_gaps
                - gaps / scale
                - gammaln(shape)
                - shape * math.log(scale)
            )
            survival = gammaincc(shape, censored / scale)
            if np.any(survival <= 0.0) or np.any(~np.isfinite(survival)):
                return np.inf
            return -float(
                np.dot(gap_weights, completed)
                + np.dot(censor_weights, np.log(survival))
            )

        def build(parameters):
            shape, scale = np.exp(parameters)
            return GammaDistribution(
                shape,
                scale,
                name=name,
                keys=keys,
            )

    elif isinstance(initial, WeibullDistribution):
        x0 = np.log([initial.shape, initial.scale])

        def objective(parameters):
            shape, scale = np.exp(parameters)
            powered_gaps = np.power(gaps / scale, shape)
            completed = (
                math.log(shape)
                - math.log(scale)
                + (shape - 1.0) * (log_gaps - math.log(scale))
                - powered_gaps
            )
            log_survival = -np.power(censored / scale, shape)
            return -float(
                np.dot(gap_weights, completed)
                + np.dot(censor_weights, log_survival)
            )

        def build(parameters):
            shape, scale = np.exp(parameters)
            return WeibullDistribution(
                shape,
                scale,
                name=name,
                keys=keys,
            )

    elif isinstance(initial, LogGaussianDistribution):
        x0 = np.asarray(
            [initial.mu, 0.5 * math.log(initial.sigma2)],
            dtype=np.float64,
        )

        def objective(parameters):
            mu, log_sigma = parameters
            sigma = math.exp(log_sigma)
            z = (log_gaps - mu) / sigma
            completed = (
                -log_gaps
                - log_sigma
                - 0.5 * math.log(2.0 * math.pi)
                - 0.5 * z * z
            )
            with np.errstate(divide="ignore"):
                censor_z = (np.log(censored) - mu) / sigma
            log_survival = log_ndtr(-censor_z)
            return -float(
                np.dot(gap_weights, completed)
                + np.dot(censor_weights, log_survival)
            )

        def build(parameters):
            mu, log_sigma = parameters
            return LogGaussianDistribution(
                mu,
                math.exp(2.0 * log_sigma),
                name=name,
                keys=keys,
            )

    elif isinstance(initial, InverseGaussianDistribution):
        x0 = np.log([initial.mu, initial.lam])

        def objective(parameters):
            mu, lam = np.exp(parameters)
            shape = mu / lam
            completed = invgauss.logpdf(gaps, shape, scale=lam)
            log_survival = invgauss.logsf(censored, shape, scale=lam)
            if np.any(~np.isfinite(completed)) or np.any(
                ~np.isfinite(log_survival)
            ):
                return np.inf
            return -float(
                np.dot(gap_weights, completed)
                + np.dot(censor_weights, log_survival)
            )

        def build(parameters):
            mu, lam = np.exp(parameters)
            return InverseGaussianDistribution(
                mu,
                lam,
                name=name,
                keys=keys,
            )

    else:
        raise TypeError(
            "RenewalProcess censor-aware fitting currently supports "
            "Exponential, Gamma, Weibull, LogGaussian, and InverseGaussian "
            "inter-arrival estimators"
        )

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1.0e-12},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            "RenewalProcess censor-aware likelihood optimization failed: "
            f"{result.message}"
        )
    return build(result.x)


class RenewalProcessEstimator(ParameterEstimator):
    """Fit supported inter-arrival families to the full censored likelihood."""

    def __init__(
        self,
        interarrival_estimator: ParameterEstimator,
        window: float,
        name: str | None = None,
        keys: str | None = None,
        max_events: int = _DEFAULT_MAX_EVENTS,
    ) -> None:
        self.interarrival_estimator = interarrival_estimator
        if (
            isinstance(window, (bool, np.bool_))
            or not np.isfinite(window)
            or window <= 0.0
        ):
            raise ValueError("RenewalProcessEstimator requires a finite window > 0")
        self.window = float(window)
        self.name = name
        self.keys = keys
        self.max_events = _exact_integer(
            max_events,
            label="RenewalProcess maximum event count",
        )
        if self.max_events < 0:
            raise ValueError(
                "RenewalProcess maximum event count must be non-negative"
            )

    def accumulator_factory(self) -> "RenewalProcessAccumulatorFactory":
        """Return an accumulator factory for observed inter-arrival gaps."""
        return RenewalProcessAccumulatorFactory(
            self.interarrival_estimator.accumulator_factory(),
            self.window,
            name=self.name,
            keys=self.keys,
        )

    def estimate(self, nobs: float | None, suff_stat: Any) -> "RenewalProcessDistribution":
        """Estimate the inter-arrival distribution and keep the fixed window."""
        checked = _validated_statistics(suff_stat)
        rebuilt = self.interarrival_estimator.accumulator_factory().make()
        for gap, weight in zip(
            checked.completed_gaps,
            checked.completed_weights,
        ):
            rebuilt.update(float(gap), float(weight), None)
        checked = RenewalProcessStatistics(
            rebuilt.value(),
            checked.completed_gaps,
            checked.completed_weights,
            checked.censored_times,
            checked.censored_weights,
        )
        interarrival = _estimate_censored_interarrival(
            self.interarrival_estimator,
            checked,
            nobs,
        )
        return RenewalProcessDistribution(
            interarrival,
            self.window,
            name=self.name,
            keys=self.keys,
            max_events=self.max_events,
        )


class RenewalProcessDataEncoder(DataSequenceEncoder):
    """Encode realizations with raw gaps, child encoding, and censoring metadata."""

    def __init__(self, gap_encoder: DataSequenceEncoder, window: float) -> None:
        self.gap_encoder = gap_encoder
        if (
            isinstance(window, (bool, np.bool_))
            or not np.isfinite(window)
            or window <= 0.0
        ):
            raise ValueError(
                "RenewalProcessDataEncoder requires a finite window > 0"
            )
        self.window = float(window)

    def __str__(self) -> str:
        return "RenewalProcessDataEncoder(%s, %s)" % (str(self.gap_encoder), repr(self.window))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RenewalProcessDataEncoder)
            and self.gap_encoder == other.gap_encoder
            and self.window == other.window
        )

    def seq_encode(self, x: Sequence[Any]) -> Any:
        """Encode event-time realizations into flattened gaps and censoring metadata."""
        window = self.window
        flat_gaps: list[float] = []
        seg_ids: list[int] = []
        remaining: list[float] = []
        ok: list[bool] = []
        for i, events in enumerate(x):
            result = _gaps_and_remaining(
                events,
                window=window,
                fail_closed=False,
            )
            valid = result is not None
            ok.append(valid)
            if not valid:
                remaining.append(0.0)
                continue
            gaps, censored = result
            flat_gaps.extend(float(g) for g in gaps)
            seg_ids.extend([i] * gaps.size)
            remaining.append(censored)
        child_enc = self.gap_encoder.seq_encode(flat_gaps)
        return (
            child_enc,
            np.asarray(flat_gaps, dtype=np.float64),
            np.asarray(seg_ids, dtype=np.int64),
            len(ok),  # number of realizations
            np.asarray(remaining, dtype=np.float64),
            np.asarray(ok, dtype=bool),
        )

    def row_count(self, x: Any) -> int:
        """Return the validated number of encoded realizations."""
        return _validated_payload(
            x,
            gap_encoder=self.gap_encoder,
            window=self.window,
        )[3]
