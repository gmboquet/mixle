"""Logarithmic (log-series) distributions over positive integers.

Observations are positive integers ``k >= 1``. A log-series distribution with shape ``p in (0, 1)`` has
log-mass

        log(P(k; p)) = k*log(p) - log(k) - log(-log(1 - p)),    k = 1, 2, ...,

a one-parameter exponential family used for over-dispersed positive counts (species abundance,
word-frequency / type-token models). The mean is ``-p / ((1 - p) * log(1 - p))`` (a value in
``(1, inf)`` that increases with p), which the estimator inverts.

The per-row score is linear in the encoded ``k`` and ``log(k)`` fields once the scalar normalizer
``log(-log(1 - p))`` is precomputed, so the family gets generated NumPy, Torch, and Numba kernels.


Reference: Fisher, Corbet & Williams, 'The relation between the number of species and the number of individuals...', J. Animal Ecology (1943).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.univariate.continuous._observation_contracts import (
    is_whole_number,
    refuse_unsupported_observation,
    refuse_unsupported_observations,
    validated_quantile_probability,
)
from mixle.stats.univariate.discrete._count_contracts import exact_integer_observations
from mixle.utils.special import valid_integer

_MIN_P = 1.0e-12
_MAX_P = 1.0 - 1.0e-12


def _mean_from_p(p: float) -> float:
    """Return the log-series mean -p / ((1 - p) * log(1 - p))."""
    return -p / ((1.0 - p) * math.log1p(-p))


def _solve_p(mean: float) -> float:
    """Invert the mean -> p for the log-series distribution (the mean increases monotonically in p)."""
    if mean <= 1.0:
        return _MIN_P
    lo, hi = _MIN_P, _MAX_P
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _mean_from_p(mid) < mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Below this shape the closed-form variance is 0/0 in double precision and its series stands in;
# the series truncation error is O(p^4), far below rounding at the switch point.
_VARIANCE_SERIES_P = 1.0e-4


class LogSeriesDistribution(SequenceEncodableProbabilityDistribution):
    """Logarithmic (log-series) distribution on k = 1, 2, ... with shape parameter p in (0, 1)."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated log-series kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for log-series distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="log_series",
            distribution_type=cls,
            parameters=(
                ParameterSpec("log_p"),
                ParameterSpec("log_norm", constraint="real", differentiable=False),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="positive_integer",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure=cls.exp_family_base_measure,
                legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row (count, k) sufficient statistics in accumulator order."""
        k = engine.asarray(x[0])
        return k * 0.0 + engine.asarray(1.0), k

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return the log-series sufficient statistic ``T(k) = (k,)``."""
        return (engine.asarray(x[0]),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the log-series natural parameter ``eta = log(p)``."""
        return (params["log_p"],)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the log-series log partition ``A = log(-log(1 - p))``."""
        return params["log_norm"]

    @staticmethod
    def exp_family_base_measure(x: tuple[Any, Any], engine: Any) -> Any:
        """Return the log-series base measure ``log h(k) = -log(k)`` (independent of p, fixed base)."""
        return -engine.asarray(x[1])

    @staticmethod
    def exp_family_from_natural(eta: Any) -> "LogSeriesDistribution":
        """Return the log-series with natural parameter ``eta = log(p)`` (so ``p = exp(eta)``)."""
        import numpy as _np

        return LogSeriesDistribution(float(_np.exp(float(eta[0]))))

    @staticmethod
    def backend_log_density_from_params(k: Any, log_k: Any, log_p: Any, log_norm: Any, engine: Any) -> Any:
        """Engine-neutral log-series log-mass from explicit parameters (linear in k and log k)."""
        return k * log_p - log_k - log_norm

    def __init__(self, p: float, name: str | None = None, keys: str | None = None) -> None:
        """LogSeriesDistribution for shape parameter p.

        Args:
            p (float): Shape parameter in (0, 1).
            name (Optional[str]): Assign a name to LogSeriesDistribution instance.
            keys (Optional[str]): Assign keys for merging sufficient statistics.

        Attributes:
            p (float): Shape parameter.
            log_p (float): Cached log(p).
            log_norm (float): Cached log(-log(1 - p)).

        """
        if not (0.0 < p < 1.0) or not np.isfinite(p):
            raise ValueError("LogSeriesDistribution requires p in (0, 1).")
        self.p = float(p)
        self.log_p = math.log(self.p)
        self.log_norm = math.log(-math.log1p(-self.p))
        self.name = name
        self.keys = keys

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep the cached ``log_p``, ``log_norm`` tied to the parameter ``p`` it derives from.

        The constants are computed once in ``__init__`` and read by ``log_density``, so a later
        assignment used to leave them stale and the scorer kept reporting the *previous*
        parameter's density with no error at all (MXR-080-1192).

        Recompute rather than validate: callers legitimately install out-of-domain or non-finite
        parameters -- deserialized legacy states and NaN-propagation checks both do -- so a value
        outside the domain yields a NaN constant that propagates honestly instead of rejecting a
        state the library is expected to be able to hold.
        """
        object.__setattr__(self, name, value)
        if name not in ("p",):
            return
        try:
            object.__setattr__(self, "log_p", math.log(self.p))
            object.__setattr__(self, "log_norm", math.log(-math.log1p(-self.p)))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, AttributeError, FloatingPointError):
            # AttributeError covers __init__, where the first parameter is assigned before the rest.
            object.__setattr__(self, "log_p", float("nan"))
            object.__setattr__(self, "log_norm", float("nan"))

    def __str__(self) -> str:
        """Return a constructor-style representation of the log-series distribution."""
        return "LogSeriesDistribution(%s, name=%s, keys=%s)" % (repr(self.p), repr(self.name), repr(self.keys))

    def density(self, x: int) -> float:
        """Return the probability mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Return the log-mass at a single positive integer (or -inf off support)."""
        if not valid_integer(x) or float(x) < 1.0:
            return -np.inf
        k = int(x)
        return k * self.log_p - math.log(k) - self.log_norm

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-mass values for sequence-encoded (k, log k) observations."""
        k, log_k = x
        with np.errstate(invalid="ignore"):
            rv = k * self.log_p - log_k - self.log_norm
        return np.where(np.asarray(k) >= 1, rv, -np.inf)

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-mass for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x[0]),
            engine.asarray(x[1]),
            engine.asarray(self.log_p),
            engine.asarray(self.log_norm),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["LogSeriesDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for a homogeneous mixture kernel."""
        return {
            "log_p": engine.asarray([d.log_p for d in dists]),
            "log_norm": engine.asarray([d.log_norm for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of log-series log densities."""
        k = engine.asarray(x[0])[:, None]
        log_k = engine.asarray(x[1])[:, None]
        return cls.backend_log_density_from_params(
            k, log_k, params["log_p"][None, :], params["log_norm"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return stacked sufficient statistics using engine-resident arrays."""
        k = engine.asarray(x[0])
        ww = engine.asarray(weights)
        return engine.sum(ww, axis=0), engine.sum(ww * k[:, None], axis=0)

    def mean(self) -> float:
        """Mean E[X] = -p / ((1-p) log(1-p))."""
        import math

        # ``log1p(-p)``, not ``log(1 - p)``: the latter rounds to exactly zero for p below the
        # float epsilon and divided by it (P01-F10), where the mean's limit is a plain 1.
        l1m = math.log1p(-self.p)
        return float(-self.p / ((1.0 - self.p) * l1m))

    def variance(self) -> float:
        """Variance Var[X] = -p (p + log(1-p)) / ((1-p)^2 log(1-p)^2)."""
        import math

        p = self.p
        if p < _VARIANCE_SERIES_P:
            # Both ``p + log1p(-p)`` and ``log1p(-p) ** 2`` collapse to zero well before p does,
            # so the direct quotient is 0/0 for small p (P01-F10). Factoring the leading p^2 out
            # of each leaves ``p * S / ((1-p)^2 T^2)`` with S -> 1/2 and T -> 1, i.e. the limit
            # ``p / 2 -> 0`` scipy reports.
            s = 0.5 + p / 3.0 + p * p / 4.0 + p * p * p / 5.0
            t = 1.0 + p / 2.0 + p * p / 3.0 + p * p * p / 4.0
            return float(p * s / ((1.0 - p) ** 2 * t * t))
        l1m = math.log1p(-p)
        return float(-p * (p + l1m) / ((1.0 - p) ** 2 * l1m * l1m))

    def entropy(self) -> float:
        """Shannon entropy in nats, by exact summation of the standard series.

        The log-series entropy ``-sum_k p_k log p_k`` has no closed form (Fisher, Corbet &
        Williams, 1943). The series is summed from ``k = 1`` up to this distribution's own
        quantile at ``1 - 1e-16`` (plus a safety margin), beyond which the remaining tail mass is
        far below double rounding.
        """
        kmax = int(self.quantile(1.0 - 1.0e-16)) + 50
        total = 0.0
        start = 1
        # Summed in bounded blocks, never as one array: at p within ~1e-9 of 1 the quantile is
        # ~3e10 and a single arange was a 226 GiB allocation. Past the term cap the remaining
        # tail is a smooth, slowly decaying function of k, and Euler-Maclaurin (integral plus
        # half the first term) resolves it to double rounding.
        while start <= kmax and start <= self._SERIES_CAP:
            stop = min(kmax, start + self._SERIES_BLOCK - 1) + 1
            lp = self._log_pmf_terms(start, stop)
            total += float(-np.sum(np.exp(lp) * lp))
            start = stop
        if start <= kmax:
            total += self._entropy_tail(start)
        return float(total)

    def _entropy_tail(self, start: int) -> float:
        """``-sum_{k>=start} p_k log p_k`` by Euler-Maclaurin: ``integral_start^inf f + f(start)/2``."""
        import math

        from scipy.integrate import quad

        log_p, log_norm = self.log_p, self.log_norm

        def f(x: float) -> float:
            lp = x * log_p - math.log(x) - log_norm
            return -math.exp(lp) * lp

        # the integrand decays like p^x: integrate over a few decay lengths, in pieces
        scale = 1.0 / max(-log_p, 1.0e-300)
        edges = [float(start)] + [start + scale * m for m in (1, 2, 4, 8, 16, 32, 64)]
        integral = sum(quad(f, a, b, limit=200)[0] for a, b in zip(edges, edges[1:]))
        return float(integral + 0.5 * f(float(start)))

    # Block size for the finite series sums below: bounded memory (0.5 MB of float64 per block)
    # however far into the tail a call has to go.
    _SERIES_BLOCK = 65_536
    # Hard cap on scanned terms before the closed-form tail bound takes over (about a second of
    # numpy time); p within ~1e-8 of 1 needs more terms than that to reach q = 1 - 1e-16.
    _SERIES_CAP = 50_000_000

    def _log_pmf_terms(self, start: int, stop: int) -> np.ndarray:
        k = np.arange(start, stop, dtype=np.float64)
        return k * self.log_p - np.log(k) - self.log_norm

    def _tail_mass(self, k: int) -> float:
        """``P(X > k) = sum_{j>k} p^j / (j * -log(1-p))`` by Euler-Maclaurin from ``a = k + 1``.

        With ``f(x) = p^x / x = exp(-lambda x) / x`` and ``lambda = -log p``, the sum from ``a`` is
        the integral of ``f`` over ``[a, inf)`` plus ``f(a)/2``, minus the first derivative over
        12, plus the third over 720, and so on; the integral is the exponential integral
        ``E1(lambda a)``. The neglected term is of order ``f(a) (lambda + 1/a)^5``, below double
        rounding wherever this stands in for direct summation (``a`` past the term cap, or where
        the terms no longer move a double sum). The geometric bound ``p^a / (a (1-p))`` this
        replaces was loose by a factor of about ``1 / (a (1-p))`` and, unclamped, sent the CDF
        negative past the cap at ``p`` within ``1e-8`` of one.
        """
        import math

        from scipy.special import exp1

        a = k + 1
        lam = -self.log_p
        if not lam > 0.0:  # p == 1 rounds log p to zero: the tail is all the mass
            return 1.0
        x = lam * a
        f_a = math.exp(-x - math.log(a))
        if f_a == 0.0:
            return 0.0
        inv_a = 1.0 / a
        euler_maclaurin = (
            float(exp1(x))
            + 0.5 * f_a
            + f_a * (lam + inv_a) / 12.0
            - f_a * (lam**3 + 3.0 * lam**2 * inv_a + 6.0 * lam * inv_a**2 + 6.0 * inv_a**3) / 720.0
        )
        return float(min(1.0, max(0.0, euler_maclaurin * math.exp(-self.log_norm))))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)``, support ``x >= 1``.

        Summed directly in bounded blocks (the series has no closed form); past ``_SERIES_CAP``
        terms one minus the Euler-Maclaurin tail mass stands in, exact to double rounding by
        then and never below the partial sum at the cap, so the result is monotone in ``x`` and
        within ``[0, 1]``.
        """
        import math

        k = math.floor(float(x))
        if k < 1:
            return 0.0
        total = 0.0
        start = 1
        while start <= k and start <= self._SERIES_CAP:
            stop = min(k, start + self._SERIES_BLOCK - 1) + 1
            total += float(np.sum(np.exp(self._log_pmf_terms(start, stop))))
            start = stop
        if start <= k:
            return float(min(1.0, max(total, 1.0 - self._tail_mass(k))))
        return float(min(1.0, total))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the smallest ``k`` with ``P(X <= k) >= q``.

        Summed directly in bounded blocks, never through scipy's generic discrete ``ppf``. That
        search brackets the answer by doubling an upper bound until ``cdf(bound) >= q``, and the
        summed CDF can saturate one ULP short of a ``q`` this close to one (numpy's SIMD ``exp``
        and ``log`` round the last bit differently per CPU: the sum reached ``q = 1 - 1e-16``
        at ``k = 5210`` on arm64 and emulated x86-64 and never did on AVX-512 hosted CI runners),
        so the bound doubled without limit and the vectorized CDF sums grew until the 16 GB
        runner was OOM-killed -- through ``entropy()``, which calls this at exactly that ``q``.
        Here a block whose terms no longer move the running sum, or the term cap, ends the
        scan, and the Euler-Maclaurin tail mass answers for whatever lies beyond: the smallest
        ``k`` past the scan whose tail mass is at most ``1 - q``.
        """
        import math

        # Same rule as every other family, and the same message: this one raised without naming
        # itself, so a level computed wrong somewhere up a composite fit said only "q must be in
        # [0, 1]" with nothing to say which law was asked (R05-F07).
        q = validated_quantile_probability(q, label="LogSeriesDistribution.quantile")
        if q <= 0.0:
            return 1.0
        if q >= 1.0:
            return math.inf
        total = 0.0
        start = 1
        scanned = 0  # every k <= scanned has a summed CDF below q
        while start <= self._SERIES_CAP:
            stop = start + self._SERIES_BLOCK
            terms = np.exp(self._log_pmf_terms(start, stop))
            cumulative = total + np.cumsum(terms)
            hit = np.flatnonzero(cumulative >= q)
            if hit.size:
                return float(start + int(hit[0]))
            scanned = stop - 1
            new_total = float(cumulative[-1])
            if new_total == total or terms[-1] < 1.0e-300:  # the series can no longer move the sum
                break
            total = new_total
            start = stop
        # saturated or capped: bisect the monotone tail mass for the smallest k beyond the scan
        # with P(X > k) <= 1 - q; never below the scan's end, so the answer is monotone in q
        # across the seam between the two methods
        remaining = max(1.0 - q, 5.0e-324)
        lo = scanned
        hi = max(2 * lo, 1)
        while self._tail_mass(hi) > remaining and hi < 2**62:
            hi *= 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._tail_mass(mid) > remaining:
                lo = mid
            else:
                hi = mid
        return float(hi)

    def sampler(self, seed: int | None = None) -> "LogSeriesSampler":
        """Return a sampler for drawing observations from this distribution."""
        return LogSeriesSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LogSeriesEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return LogSeriesEstimator(name=self.name, keys=self.keys)
        return LogSeriesEstimator(
            pseudo_count=pseudo_count, suff_stat=_mean_from_p(self.p), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "LogSeriesDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return LogSeriesDataEncoder()

    def enumerator(self) -> "LogSeriesEnumerator":
        """Return an enumerator over k = 1, 2, ... in descending probability order."""
        return LogSeriesEnumerator(self)


class LogSeriesEnumerator(DistributionEnumerator):
    """Enumerate log-series support values in descending probability order.

    The log-series pmf ``p^k / (k * -log(1 - p))`` is strictly decreasing in ``k`` (each step
    multiplies by ``p * k / (k + 1) < 1``), so value order 1, 2, 3, ... IS descending-probability
    order. The iterator is infinite.
    """

    def __init__(self, dist: LogSeriesDistribution) -> None:
        super().__init__(dist)
        self._k = 1

    def __next__(self) -> tuple[int, float]:
        k = self._k
        self._k += 1
        return (k, self.dist.log_density(k))


class LogSeriesSampler(DistributionSampler):
    """Draw iid log-series observations."""

    def __init__(self, dist: LogSeriesDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None, *, batched: bool = True) -> int | np.ndarray:
        """Draw ``size`` iid positive integers (an int when ``size`` is None)."""
        rv = self.rng.logseries(self.dist.p, size=size)
        return int(rv) if size is None else rv


_LOGSERIES_SUPPORT_MESSAGE = (
    "LogSeriesDistribution has support k in {1, 2, 3, ...}, but at least %d observation(s) "
    "carrying weight are below one, fractional, NaN, or infinite (estimation encodes in chunks; "
    "the first offending chunk refuses). NaN and infinity used to surface as a raw int() "
    "conversion error naming neither the family nor the support."
)


class LogSeriesAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum for log-series estimation."""

    def __init__(self, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.keys = keys

    def update(self, x: int, weight: float, estimate: LogSeriesDistribution | None) -> None:
        """Accumulate weighted count and total for one positive integer."""
        refuse_unsupported_observation(
            is_whole_number(x) and x >= 1,
            weight,
            message=_LOGSERIES_SUPPORT_MESSAGE % 1,
        )
        self.count += weight
        self.sum += float(x) * weight

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: LogSeriesDistribution | None
    ) -> None:
        """Accumulate weighted count and total from encoded observations."""
        refuse_unsupported_observations(
            np.isfinite(x[0]) & (np.asarray(x[0]) >= 1),
            weights,
            message=_LOGSERIES_SUPPORT_MESSAGE,
        )
        self.count += np.sum(weights, dtype=np.float64)
        self.sum += np.dot(x[0], weights)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "LogSeriesAccumulator":
        """Merge another log-series sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        """Return accumulated count and integer total."""
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "LogSeriesAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.sum = x
        return self

    def acc_to_encoder(self) -> "LogSeriesDataEncoder":
        """Return the encoder used by this accumulator."""
        return LogSeriesDataEncoder()


class LogSeriesAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LogSeriesAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> LogSeriesAccumulator:
        """Create a fresh log-series accumulator."""
        return LogSeriesAccumulator(keys=self.keys)


class LogSeriesEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the log-series shape p (inverts the mean -> p relation)."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LogSeriesAccumulatorFactory:
        """Return an accumulator factory for log-series count statistics."""
        return LogSeriesAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> LogSeriesDistribution:
        """Estimate the shape parameter by inverting the weighted mean."""
        count, total = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            total += self.pseudo_count * self.suff_stat
            count += self.pseudo_count
        if count <= 0.0:
            return LogSeriesDistribution(0.5, name=self.name, keys=self.keys)
        mean = total / count
        p = _solve_p(mean)
        dist = LogSeriesDistribution(p, name=self.name, keys=self.keys)
        if p <= _MIN_P:
            # The mean of a log-series is at least one and rises with p; a sample mean at (or
            # below) one asks for p = 0, which is not a distribution, so the solver returns its
            # floor. It used to do so silently (P01-F09).
            dist._numerical_repairs = ("logseries-p-floored(sample mean %.6g -> p=%.6g)" % (mean, _MIN_P),)
        return dist


class LogSeriesDataEncoder(DataSequenceEncoder):
    """Encode log-series observations as (k, log k) pairs."""

    def __str__(self) -> str:
        return "LogSeriesDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogSeriesDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Encode observations as integer values and log-values."""
        rv = exact_integer_observations(
            x,
            label="Log-series observations",
            # An out-of-support count is scored, not refused: the scalar path returns -inf for it
            # and a mixture whose other component owns that value has to be able to encode the
            # whole batch (P02-F03). The integer contract itself still holds.
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return rv, np.log(rv)
