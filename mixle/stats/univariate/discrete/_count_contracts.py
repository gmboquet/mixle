"""Shared validation for count observations, weights, and cached law parameters."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


class CachedParameterLaw:
    """Keep derived parameter caches consistent with the parameters they came from.

    Several count families validate a parameter in ``__init__`` and then cache
    derived quantities (logs, normalizers, gamma terms) that every scorer reads.
    Assigning the parameter afterwards used to leave those caches stale, so the
    sampler drew from the new parameter while scalar, sequence, and backend
    scoring all kept using the old cache.

    A subclass lists its parameters in ``_cached_parameters`` and performs the
    validation plus the whole cache computation in ``_rebuild_parameter_caches``.
    Constructors set the raw parameters, call the rebuild once, and then set
    ``_parameter_caches_ready``; assigning any listed parameter afterwards re-runs
    that same validated rebuild, so the parameters and their caches can never
    disagree. Writes that bypass ``__setattr__`` entirely -- pickling, ``deepcopy``
    and the inference-transaction rollback all restore ``__dict__`` directly --
    are untouched because they replay a consistent snapshot.
    """

    _cached_parameters: tuple[str, ...] = ()

    def _rebuild_parameter_caches(self) -> None:
        """Validate this law's parameters and rebuild every derived cache."""
        raise NotImplementedError("cached parameter laws must implement _rebuild_parameter_caches.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, rebuilding derived caches when a parameter changes."""
        if not (name in self._cached_parameters and self.__dict__.get("_parameter_caches_ready", False)):
            object.__setattr__(self, name, value)
            return
        previous = self.__dict__.copy()
        object.__setattr__(self, name, value)
        try:
            self._rebuild_parameter_caches()
        except BaseException:
            # A rejected parameter must leave the law exactly as it was rather than
            # stranding it with a new parameter beside its old caches.
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise


def exact_integer_observations(
    values: Sequence[Any] | np.ndarray,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    """Return an exact int64 array after validating every observation."""
    raw = np.asarray(values, dtype=object)
    encoded = np.empty(raw.shape, dtype=np.int64)
    for index, value in np.ndenumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must be exact integers, not booleans.")
        if isinstance(value, numbers.Integral):
            integer = int(value)
        elif isinstance(value, numbers.Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"{label} must be finite exact integers.")
            integer = int(numeric)
        else:
            raise ValueError(f"{label} must be finite exact integers.")
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise ValueError(f"{label} must fit in signed 64-bit integer storage.")
        if minimum is not None and integer < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
        if maximum is not None and integer > maximum:
            raise ValueError(f"{label} must be at most {maximum}.")
        encoded[index] = integer
    return encoded


def nonnegative_weights(values: Any, *, shape: tuple[int, ...], label: str = "weights") -> np.ndarray:
    """Return finite, non-negative float64 weights aligned to an observation array."""
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != shape:
        raise ValueError(f"{label} must have the same shape as the encoded observations.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{label} must be finite and non-negative.")
    return weights


# Block size for every blocked count-series sum below: bounded memory (0.5 MB of float64 per
# block) however far into the support a call has to reach.
ENTROPY_BLOCK = 65_536
# Hard cap on the number of terms summed exactly before the smooth tail takes over. Past this
# many terms the series is a slowly varying function of ``k`` on the scale of one count, so
# Euler-Maclaurin resolves the remainder to double rounding at a small constant cost. One
# array over the whole effective support was the shape that OOM-killed CI on the log-series
# quantile scan (0.8.0) and that P01-F02 found still standing in Poisson and negative binomial.
ENTROPY_TERM_CAP = 1_000_000


def blocked_entropy(
    log_pmf_terms: Callable[[int, int], np.ndarray],
    log_pmf_at: Callable[[float], float],
    *,
    support_start: int,
    effective_support_end: float,
    mean: float,
    sd: float,
    decay_length: float | None = None,
    term_cap: int = ENTROPY_TERM_CAP,
) -> float:
    """Return ``-sum_k p_k log p_k`` over ``k >= support_start`` in bounded memory.

    ``log_pmf_terms(start, stop)`` returns ``log p_k`` for the half-open integer block
    ``[start, stop)`` and ``log_pmf_at(x)`` the same quantity at a real ``x`` (the smooth
    extension through ``gammaln``). The series is summed exactly, one bounded block at a
    time, over the first ``term_cap`` terms; a support wider than that has its remainder
    resolved by :func:`smooth_entropy_tail`, which needs the law's ``mean``, ``sd`` and --
    for a geometric-tailed law whose decay outruns its standard deviation -- its
    ``decay_length`` to place the quadrature.

    ``effective_support_end`` is the count past which the remaining terms are below double
    rounding (typically the law's own quantile at ``1 - 1e-16`` plus a margin). It may be
    infinite or NaN: any value that is not a finite count is treated as wider than the cap,
    which routes the remainder through the tail rather than through an unbounded ``arange``.
    """
    total = 0.0
    start = int(support_start)
    last_exact = start + int(term_cap) - 1
    end = last_exact + 1 if not math.isfinite(effective_support_end) else int(effective_support_end)
    while start <= end and start <= last_exact:
        stop = min(end, last_exact, start + ENTROPY_BLOCK - 1) + 1
        terms = log_pmf_terms(start, stop)
        total += float(-np.sum(np.exp(terms) * terms))
        start = stop
    if start <= end:
        total += smooth_entropy_tail(log_pmf_at, start, mean=mean, sd=sd, decay_length=decay_length)
    return float(total)


def smooth_entropy_tail(
    log_pmf_at: Callable[[float], float],
    start: int,
    *,
    mean: float,
    sd: float,
    decay_length: float | None = None,
) -> float:
    """``-sum_{k>=start} p_k log p_k`` by Euler-Maclaurin: ``integral_start^inf f + f(start)/2``.

    Past the exact-summation cap ``f(x) = -p_x log p_x`` varies slowly on the scale of one
    count -- the standard deviation of any law that reaches this branch is far larger than one --
    so the sum is its integral plus half the first term, with a remainder of order
    ``f'(start)/12``. The quadrature runs in units of the law's own width: the interval is broken
    at the mean and at standard deviations either side of it so the bulk is never missed when the
    cap falls below it, then walked out in decay lengths so a geometric tail is integrated to
    exhaustion rather than truncated at a Gaussian width. Integrating the rescaled variable keeps
    every segment O(1) wide, which a distribution whose support runs to 1e10 counts otherwise
    makes ``quad`` subdivide to its limit.
    """
    from scipy.integrate import quad

    scale = max(float(sd), 0.0)
    if decay_length is not None and math.isfinite(decay_length):
        scale = max(scale, float(decay_length))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    lo = float(start)

    def integrand(x: float) -> float:
        log_p = log_pmf_at(x)
        if not math.isfinite(log_p) or log_p < -745.0:
            return 0.0
        return -math.exp(log_p) * log_p

    def rescaled(t: float) -> float:
        return scale * integrand(lo + scale * t)

    offsets = [0.0]
    for point in (-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        t = (float(mean) + point * scale - lo) / scale
        if math.isfinite(t) and t > offsets[-1]:
            offsets.append(t)
    for multiple in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        offsets.append(offsets[-1] + multiple)
    integral = sum(quad(rescaled, a, b, limit=200)[0] for a, b in zip(offsets, offsets[1:]))
    return float(integral + 0.5 * integrand(lo))


def gaussian_limit_entropy(variance: float, skewness: float, excess_kurtosis: float) -> float:
    """Entropy of a lattice law that has become Gaussian, with its Edgeworth correction.

    ``H = 0.5 log(2 pi e sigma^2) - gamma_1^2 / 12 - gamma_2^2 / 48``: the entropy of the
    Edgeworth expansion of a standardized lattice variable, whose leading negentropy terms are
    the squared skewness and excess kurtosis (Comon, *Independent component analysis*, 1994,
    eq. 27). It reproduces the classical Poisson expansion's ``-1/(12 lambda)`` term and is used
    only where the shape parameters put both corrections below double rounding -- the regime
    where the support is too wide to sum and the log-pmf's own cancellation error would swamp a
    quadrature of it.
    """
    return float(0.5 * math.log(2.0 * math.pi * math.e * variance) - skewness**2 / 12.0 - excess_kurtosis**2 / 48.0)


def validated_quantile_probability(q: Any, *, label: str) -> float:
    """Return ``q`` as a float after refusing anything outside ``[0, 1]`` (NaN included)."""
    value = float(q)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label}: q must be in [0, 1].")
    return value
