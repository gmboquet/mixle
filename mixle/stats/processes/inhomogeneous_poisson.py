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
import operator
from collections.abc import Sequence
from typing import Any, NamedTuple

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


class InhomogeneousPoissonProcessStatistics(NamedTuple):
    """Versioned, tuple-compatible process statistics."""

    bin_counts: np.ndarray
    realization_weight: float

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
    edges: np.ndarray,
    fail_closed: bool,
) -> np.ndarray | None:
    try:
        events = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        if fail_closed:
            raise ValueError("InhomogeneousPoisson event times must be numeric") from exc
        return None
    if np.any(~np.isfinite(events)) or np.any(events < edges[0]) or np.any(events > edges[-1]):
        if fail_closed:
            raise ValueError("InhomogeneousPoisson event times must be finite and inside the process window")
        return None
    return events


def _validated_count_rows(value: Any, *, num_bins: int) -> np.ndarray:
    try:
        counts = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("InhomogeneousPoisson encoded counts must be numeric") from exc
    if counts.shape == (0,):
        counts = counts.reshape((0, num_bins))
    if counts.ndim != 2 or counts.shape[1] != num_bins:
        raise ValueError(f"InhomogeneousPoisson encoded counts must have shape (N, {num_bins})")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0) or np.any(counts != np.floor(counts)):
        raise ValueError("InhomogeneousPoisson per-realization counts must be finite non-negative exact integers")
    return counts.copy()


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("InhomogeneousPoisson weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"InhomogeneousPoisson weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("InhomogeneousPoisson weights must be finite and non-negative")
    return weights


def _validated_statistics(
    value: Any,
    *,
    num_bins: int,
) -> InhomogeneousPoissonProcessStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("InhomogeneousPoisson statistics must be (bin_counts, realization_weight)")
    try:
        counts = np.asarray(value[0], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("InhomogeneousPoisson aggregate counts must be numeric") from exc
    if counts.shape != (num_bins,):
        raise ValueError(f"InhomogeneousPoisson aggregate counts must have shape ({num_bins},)")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("InhomogeneousPoisson aggregate counts must be finite and non-negative")
    realization_weight = _nonnegative_scalar(
        value[1],
        label="InhomogeneousPoisson realization weight",
    )
    if realization_weight == 0.0 and np.any(counts != 0.0):
        raise ValueError("Positive InhomogeneousPoisson counts require realization exposure")
    return InhomogeneousPoissonProcessStatistics(
        counts.copy(),
        realization_weight,
    )


def _prior_vector(value: Any, *, num_bins: int, label: str, minimum: float) -> np.ndarray:
    if np.ndim(value) == 0:
        scalar = _nonnegative_scalar(value, label=f"InhomogeneousPoisson {label}")
        result = np.full(num_bins, scalar, dtype=np.float64)
    else:
        try:
            result = np.asarray(value, dtype=np.float64).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"InhomogeneousPoisson {label} must be numeric") from exc
        if result.shape != (num_bins,):
            raise ValueError(f"InhomogeneousPoisson {label} must have shape ({num_bins},)")
    if np.any(~np.isfinite(result)) or np.any(result < minimum):
        raise ValueError(f"InhomogeneousPoisson {label} must be finite and at least {minimum}")
    result.setflags(write=False)
    return result


def _resolve_edges(num_bins: int | None, t_max: float | None, edges: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if edges is not None:
        e = np.array(edges, dtype=np.float64, copy=True)
        if e.ndim != 1 or e.size < 2 or np.any(~np.isfinite(e)) or np.any(np.diff(e) <= 0.0):
            raise ValueError("edges must be a finite strictly increasing 1-D array of length >= 2.")
        e.setflags(write=False)
        return e
    if t_max is None or num_bins is None:
        raise ValueError("provide either edges, or t_max > 0 and num_bins >= 1.")
    bins = _exact_integer(
        num_bins,
        label="InhomogeneousPoisson bin count",
    )
    checked_t_max = _nonnegative_scalar(
        t_max,
        label="InhomogeneousPoisson t_max",
    )
    if bins < 1 or checked_t_max <= 0.0:
        raise ValueError("provide either edges, or t_max > 0 and num_bins >= 1.")
    result = np.linspace(0.0, checked_t_max, bins + 1)
    result.setflags(write=False)
    return result


class InhomogeneousPoissonProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Inhomogeneous Poisson process with piecewise-constant intensity on a fixed window."""

    def __init__(
        self,
        rates: Sequence[float] | np.ndarray,
        t_max: float | None = None,
        edges: Sequence[float] | np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior_shape: Any = 1.0,
        prior_rate: Any = 0.0,
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
        self.rates = np.array(rates, dtype=np.float64, copy=True)
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
        self.prior_shape = _prior_vector(
            prior_shape,
            num_bins=self.num_bins,
            label="Gamma prior shape",
            minimum=1.0,
        )
        self.prior_rate = _prior_vector(
            prior_rate,
            num_bins=self.num_bins,
            label="Gamma prior rate",
            minimum=0.0,
        )
        with np.errstate(divide="ignore"):
            self._log_rates = np.where(self.rates > 0.0, np.log(self.rates), -np.inf)
        self._integral = float(np.sum(self.rates * self.widths))
        self.rates.setflags(write=False)
        self.widths.setflags(write=False)
        self._log_rates.setflags(write=False)

    def __str__(self) -> str:
        """Return a constructor-style representation of the inhomogeneous Poisson process."""
        return (
            "InhomogeneousPoissonProcessDistribution(%s, edges=%s, name=%s, keys=%s, prior_shape=%s, prior_rate=%s)"
            % (
                repr(list(self.rates)),
                repr(list(self.edges)),
                repr(self.name),
                repr(self.keys),
                repr(self.prior_shape.tolist()),
                repr(self.prior_rate.tolist()),
            )
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
        ev = _validated_events(
            events,
            edges=self.edges,
            fail_closed=False,
        )
        if ev is None:
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
        counts = _validated_count_rows(x, num_bins=self.num_bins)
        emitted = np.where(counts > 0.0, counts * self._log_rates[None, :], 0.0)
        rv = np.sum(emitted, axis=1) - self._integral
        rv[np.any(~np.isfinite(emitted), axis=1)] = -np.inf
        return rv

    def sampler(self, seed: int | None = None) -> "InhomogeneousPoissonProcessSampler":
        """Return an InhomogeneousPoissonProcessSampler for this distribution."""
        return InhomogeneousPoissonProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "InhomogeneousPoissonProcessEstimator":
        """Return an InhomogeneousPoissonProcessEstimator over the same bin edges."""
        if pseudo_count is None:
            return InhomogeneousPoissonProcessEstimator(
                edges=self.edges,
                name=self.name,
                keys=self.keys,
                prior_shape=self.prior_shape,
                prior_rate=self.prior_rate,
            )
        return InhomogeneousPoissonProcessEstimator(
            edges=self.edges,
            name=self.name,
            keys=self.keys,
            pseudo_count=pseudo_count,
        )

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

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray | list[np.ndarray]:
        """Draw one realization (event-time array) or a list of ``size`` realizations."""
        if size is None:
            return self._sample_one()
        checked_size = _exact_integer(
            size,
            label="InhomogeneousPoisson sample size",
        )
        if checked_size < 0:
            raise ValueError("InhomogeneousPoisson sample size must be non-negative")
        return [self._sample_one() for _ in range(checked_size)]


class InhomogeneousPoissonProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted per-bin event counts and the weighted number of realizations."""

    def __init__(self, edges: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        self.edges = _resolve_edges(None, None, edges)
        self.num_bins = self.edges.size - 1
        self.bin_counts = np.zeros(self.num_bins, dtype=np.float64)
        self.n_realizations = 0.0
        self.name = name
        self.keys = keys

    def _counts(self, x: Any) -> np.ndarray:
        ev = _validated_events(
            x,
            edges=self.edges,
            fail_closed=True,
        )
        counts, _ = np.histogram(ev, bins=self.edges)
        return counts.astype(np.float64)

    def update(self, x: Any, weight: float, estimate: InhomogeneousPoissonProcessDistribution | None) -> None:
        """Accumulate weighted per-bin counts for one event-time realization."""
        counts = self._counts(x)
        checked_weight = _nonnegative_scalar(
            weight,
            label="InhomogeneousPoisson weight",
        )
        self.bin_counts += checked_weight * counts
        self.n_realizations += checked_weight

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted realization."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted per-bin counts from encoded realizations."""
        counts = _validated_count_rows(x, num_bins=self.num_bins)
        ww = _validated_weights(weights, counts.shape[0])
        self.bin_counts += ww @ counts
        self.n_realizations += ww.sum()

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded realizations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "InhomogeneousPoissonProcessAccumulator":
        """Merge serialized bin-count statistics into this accumulator."""
        checked = _validated_statistics(suff_stat, num_bins=self.num_bins)
        self.bin_counts += checked.bin_counts
        self.n_realizations += checked.realization_weight
        return self

    def value(self) -> tuple[np.ndarray, float]:
        """Return the weighted bin counts and weighted realization count."""
        return InhomogeneousPoissonProcessStatistics(
            self.bin_counts.copy(),
            self.n_realizations,
        )

    def from_value(self, x: tuple[np.ndarray, float]) -> "InhomogeneousPoissonProcessAccumulator":
        """Restore the accumulator from serialized bin-count statistics."""
        checked = _validated_statistics(x, num_bins=self.num_bins)
        self.bin_counts = checked.bin_counts.copy()
        self.n_realizations = checked.realization_weight
        return self

    def scale(self, c: float) -> "InhomogeneousPoissonProcessAccumulator":
        """Scale accumulated sufficient statistics by a constant."""
        checked_scale = _nonnegative_scalar(
            c,
            label="InhomogeneousPoisson scale",
        )
        self.bin_counts *= checked_scale
        self.n_realizations *= checked_scale
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into its configured shared statistic."""
        if self.keys is None:
            return
        if self.keys in stats_dict:
            stats_dict[self.keys].combine(self.value())
        else:
            stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator with its configured shared statistic."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "InhomogeneousPoissonProcessDataEncoder":
        """Return an encoder that bins event times on this accumulator's edges."""
        return InhomogeneousPoissonProcessDataEncoder(self.edges)


class InhomogeneousPoissonProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for InhomogeneousPoissonProcessAccumulator."""

    def __init__(self, edges: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        self.edges = _resolve_edges(None, None, edges)
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
        pseudo_count: float | None = None,
        prior_shape: Any | None = None,
        prior_rate: Any | None = None,
    ) -> None:
        self.edges = _resolve_edges(num_bins, t_max, edges)
        self.widths = np.diff(self.edges)
        self.name = name
        self.keys = keys
        if pseudo_count is not None and (prior_shape is not None or prior_rate is not None):
            raise ValueError("InhomogeneousPoisson pseudo_count cannot be combined with explicit Gamma priors")
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _nonnegative_scalar(
                pseudo_count,
                label="InhomogeneousPoisson pseudo-count",
            )
        )
        if self.pseudo_count is not None:
            shape_value: Any = 1.0 + self.pseudo_count
            rate_value: Any = self.pseudo_count
        else:
            shape_value = 1.0 if prior_shape is None else prior_shape
            rate_value = 0.0 if prior_rate is None else prior_rate
        self.prior_shape = _prior_vector(
            shape_value,
            num_bins=self.widths.size,
            label="Gamma prior shape",
            minimum=1.0,
        )
        self.prior_rate = _prior_vector(
            rate_value,
            num_bins=self.widths.size,
            label="Gamma prior rate",
            minimum=0.0,
        )
        self.widths.setflags(write=False)

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Serialize fixed edges and the effective explicit Gamma prior."""
        return {
            "edges": self.edges,
            "name": self.name,
            "keys": self.keys,
            "prior_shape": self.prior_shape,
            "prior_rate": self.prior_rate,
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Validate and restore estimator configuration."""
        required = {"edges", "name", "keys", "prior_shape", "prior_rate"}
        if set(state) != required:
            raise ValueError("invalid InhomogeneousPoisson estimator state fields")
        self.__init__(
            edges=state["edges"],
            name=state["name"],
            keys=state["keys"],
            prior_shape=state["prior_shape"],
            prior_rate=state["prior_rate"],
        )

    def accumulator_factory(self) -> "InhomogeneousPoissonProcessAccumulatorFactory":
        """Return a factory for inhomogeneous Poisson sufficient-statistic accumulators."""
        return InhomogeneousPoissonProcessAccumulatorFactory(self.edges, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, float]
    ) -> "InhomogeneousPoissonProcessDistribution":
        """Estimate per-bin rates from accumulated ``(bin_counts, n_realizations)``."""
        checked = _validated_statistics(
            suff_stat,
            num_bins=self.widths.size,
        )
        if checked.realization_weight <= 0.0:
            raise ValueError("Cannot estimate InhomogeneousPoisson without realization exposure")
        if nobs is not None:
            _nonnegative_scalar(
                nobs,
                label="InhomogeneousPoisson observation count",
            )
        numerator = checked.bin_counts + self.prior_shape - 1.0
        denominator = self.widths * checked.realization_weight + self.prior_rate
        rates = numerator / denominator
        return InhomogeneousPoissonProcessDistribution(
            rates,
            edges=self.edges,
            name=self.name,
            keys=self.keys,
            prior_shape=self.prior_shape,
            prior_rate=self.prior_rate,
        )


class InhomogeneousPoissonProcessDataEncoder(DataSequenceEncoder):
    """Encode a list of realizations (event-time arrays) into a ``(num_realizations, num_bins)`` count matrix."""

    def __init__(self, edges: Sequence[float] | np.ndarray) -> None:
        self.edges = _resolve_edges(None, None, edges)

    def __str__(self) -> str:
        return "InhomogeneousPoissonProcessDataEncoder(%s)" % repr(list(self.edges))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InhomogeneousPoissonProcessDataEncoder) and np.array_equal(self.edges, other.edges)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Encode event-time realizations as per-bin count rows."""
        rows = []
        for events in x:
            ev = _validated_events(
                events,
                edges=self.edges,
                fail_closed=True,
            )
            counts, _ = np.histogram(ev, bins=self.edges)
            rows.append(counts.astype(np.float64))
        return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, self.edges.size - 1), dtype=np.float64)

    def row_count(self, x: np.ndarray) -> int:
        """Return the number of validated encoded realizations."""
        return int(
            _validated_count_rows(
                x,
                num_bins=self.edges.size - 1,
            ).shape[0]
        )
