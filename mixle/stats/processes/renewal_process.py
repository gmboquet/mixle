"""Renewal process -- a point process whose inter-arrival times are i.i.d. from a base distribution.

A renewal process on the window ``[0, T]`` generates event times ``0 < t_1 < t_2 < ... < t_n <= T`` such
that the gaps ``g_1 = t_1``, ``g_i = t_i - t_{i-1}`` are i.i.d. draws from a positive-support
*inter-arrival* distribution ``f`` (e.g. Gamma, Weibull, LogGaussian, InverseGaussian, Exponential -- the
Poisson process is the Exponential special case). The exact log-likelihood of one realization is

    log L = sum_i log f(g_i) + log S(T - t_n),     S(x) = 1 - F(x)  (the survival of the censored last gap)

where the final term is the probability that no further event occurred before the window closed (with
``t_0 = 0`` and ``t_n = 0`` when there are no events, so an empty realization scores ``log S(T)``). Scoring
is therefore exact, including the right-censored boundary.

Estimation fits the inter-arrival distribution to the *observed* gaps via that distribution's own
estimator -- the standard renewal-process MLE. The censored-boundary term contributes to the likelihood
(scoring) but not to the M-step; its effect on the fitted parameters is ``O(1/n_events)`` and vanishes as
the window spans many inter-arrivals, so the estimator is consistent. (A fully boundary-corrected M-step
would require a censored-data estimator for the inter-arrival family; that is a deliberate, documented
follow-up.) The window ``T`` is a fixed, known observation parameter, not estimated.
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


def _survival_logprob(interarrival: Any, remaining: np.ndarray) -> np.ndarray:
    """``log S(remaining) = log(1 - cdf(remaining))`` per realization, stable and clamped."""
    cdf = getattr(interarrival, "cdf", None)
    if not callable(cdf):
        raise TypeError(
            "RenewalProcess requires the inter-arrival distribution to expose cdf() for the censored "
            "boundary term (e.g. Gamma/Weibull/Exponential/LogGaussian/InverseGaussian)."
        )
    out = np.empty(remaining.shape[0], dtype=np.float64)
    for i, r in enumerate(remaining):
        f = float(cdf(float(r)))
        out[i] = math.log1p(-f) if f < 1.0 else -np.inf
    return out


class RenewalProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Renewal process with i.i.d. inter-arrivals ``interarrival`` observed on ``[0, window]``."""

    def __init__(self, interarrival: Any, window: float, name: str | None = None, keys: str | None = None) -> None:
        if not (np.isfinite(window) and window > 0.0):
            raise ValueError("RenewalProcessDistribution requires a finite window > 0.")
        self.interarrival = interarrival
        self.window = float(window)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "RenewalProcessDistribution(%s, %s, name=%s, keys=%s)" % (
            str(self.interarrival),
            repr(self.window),
            repr(self.name),
            repr(self.keys),
        )

    def _gaps_and_remaining(self, x: Any) -> tuple[np.ndarray, float] | None:
        ev = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
        if ev.size and (np.any(~np.isfinite(ev)) or ev[0] < 0.0 or ev[-1] > self.window):
            return None
        gaps = np.diff(np.concatenate(([0.0], ev))) if ev.size else np.empty(0, dtype=np.float64)
        last = float(ev[-1]) if ev.size else 0.0
        return gaps, self.window - last

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
        child_enc, seg_ids, num_real, remaining, ok = x
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
            self.interarrival.estimator(pseudo_count=pseudo_count), self.window, name=self.name, keys=self.keys
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

    def _sample_one(self) -> np.ndarray:
        events: list[float] = []
        t = 0.0
        while True:
            g = float(self._gap_sampler.sample())
            t += g
            if t > self.dist.window:
                break
            events.append(t)
        return np.asarray(events, dtype=np.float64)

    def sample(self, size: int | None = None) -> np.ndarray | list[np.ndarray]:
        """Draw one realization, or ``size`` iid realizations, on the fixed window."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(int(size))]


class RenewalProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Feed the inter-arrival gaps to the inter-arrival distribution's accumulator (renewal MLE)."""

    def __init__(
        self, gap_accumulator: SequenceEncodableStatisticAccumulator, name: str | None = None, keys: str | None = None
    ) -> None:
        self.gap_accumulator = gap_accumulator
        self.name = name
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: RenewalProcessDistribution | None) -> None:
        """Accumulate observed inter-arrival gaps from one realization."""
        ev = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
        gaps = np.diff(np.concatenate(([0.0], ev))) if ev.size else np.empty(0, dtype=np.float64)
        gap_est = estimate.interarrival if estimate is not None else None
        for g in gaps:
            self.gap_accumulator.update(float(g), weight, gap_est)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the gap accumulator from one realization."""
        ev = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
        gaps = np.diff(np.concatenate(([0.0], ev))) if ev.size else np.empty(0, dtype=np.float64)
        for g in gaps:
            self.gap_accumulator.initialize(float(g), weight, rng)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: RenewalProcessDistribution | None) -> None:
        """Accumulate encoded observed gaps with realization-level weights."""
        child_enc, seg_ids, num_real, remaining, ok = x
        if seg_ids.size:
            gap_weights = np.asarray(weights, dtype=np.float64)[seg_ids]
            gap_est = estimate.interarrival if estimate is not None else None
            self.gap_accumulator.seq_update(child_enc, gap_weights, gap_est)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize from encoded observed gaps."""
        child_enc, seg_ids, num_real, remaining, ok = x
        if seg_ids.size:
            gap_weights = np.asarray(weights, dtype=np.float64)[seg_ids]
            self.gap_accumulator.seq_initialize(child_enc, gap_weights, rng)

    def combine(self, suff_stat: Any) -> "RenewalProcessAccumulator":
        """Merge another inter-arrival sufficient-statistic value."""
        self.gap_accumulator.combine(suff_stat)
        return self

    def value(self) -> Any:
        """Return the wrapped inter-arrival accumulator value."""
        return self.gap_accumulator.value()

    def from_value(self, x: Any) -> "RenewalProcessAccumulator":
        """Replace the wrapped inter-arrival accumulator from ``x``."""
        self.gap_accumulator.from_value(x)
        return self

    def scale(self, c: float) -> "RenewalProcessAccumulator":
        """Scale the wrapped inter-arrival sufficient statistics by ``c``."""
        self.gap_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic merging to the gap accumulator."""
        self.gap_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Delegate keyed statistic replacement to the gap accumulator."""
        self.gap_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "RenewalProcessDataEncoder":
        """Return an encoder that converts event times into inter-arrival gaps."""
        # window is recovered by the estimator; the encoder only needs the gap encoder + a window for
        # the survival term, which the estimator supplies via dist_to_encoder on the fitted model.
        return RenewalProcessDataEncoder(self.gap_accumulator.acc_to_encoder(), None)


class RenewalProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for RenewalProcessAccumulator (wraps the inter-arrival accumulator factory)."""

    def __init__(
        self, gap_factory: StatisticAccumulatorFactory, name: str | None = None, keys: str | None = None
    ) -> None:
        self.gap_factory = gap_factory
        self.name = name
        self.keys = keys

    def make(self) -> "RenewalProcessAccumulator":
        """Create a fresh renewal-process accumulator."""
        return RenewalProcessAccumulator(self.gap_factory.make(), name=self.name, keys=self.keys)


class RenewalProcessEstimator(ParameterEstimator):
    """Fit the inter-arrival distribution to observed gaps (standard renewal-process MLE)."""

    def __init__(
        self,
        interarrival_estimator: ParameterEstimator,
        window: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.interarrival_estimator = interarrival_estimator
        self.window = float(window)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "RenewalProcessAccumulatorFactory":
        """Return an accumulator factory for observed inter-arrival gaps."""
        return RenewalProcessAccumulatorFactory(
            self.interarrival_estimator.accumulator_factory(), name=self.name, keys=self.keys
        )

    def estimate(self, nobs: float | None, suff_stat: Any) -> "RenewalProcessDistribution":
        """Estimate the inter-arrival distribution and keep the fixed window."""
        interarrival = self.interarrival_estimator.estimate(nobs, suff_stat)
        return RenewalProcessDistribution(interarrival, self.window, name=self.name, keys=self.keys)


class RenewalProcessDataEncoder(DataSequenceEncoder):
    """Encode realizations into (flattened-gap child encoding, segment ids, count, remaining, validity)."""

    def __init__(self, gap_encoder: DataSequenceEncoder, window: float | None) -> None:
        self.gap_encoder = gap_encoder
        self.window = window

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
            ev = np.sort(np.asarray(events, dtype=np.float64).reshape(-1))
            valid = not (
                ev.size and (np.any(~np.isfinite(ev)) or ev[0] < 0.0 or (window is not None and ev[-1] > window))
            )
            ok.append(valid)
            if not valid:
                remaining.append(0.0)
                continue
            gaps = np.diff(np.concatenate(([0.0], ev))) if ev.size else np.empty(0, dtype=np.float64)
            flat_gaps.extend(float(g) for g in gaps)
            seg_ids.extend([i] * gaps.size)
            last = float(ev[-1]) if ev.size else 0.0
            remaining.append((window - last) if window is not None else 0.0)
        child_enc = self.gap_encoder.seq_encode(flat_gaps)
        return (
            child_enc,
            np.asarray(seg_ids, dtype=np.int64),
            len(ok),  # number of realizations
            np.asarray(remaining, dtype=np.float64),
            np.asarray(ok, dtype=bool),
        )
