"""Exponential-tilting combinator: reweight a base distribution by ``exp(theta . T(x))``.

``ExponentialTiltedDistribution`` wraps a base distribution and applies an exponential change
of measure (an Esscher transform):

    p_theta(x) = exp(theta . T(x)) * p_base(x) / Z(theta),   Z(theta) = E_base[exp(theta . T(X))],

where ``T`` is a sufficient statistic and ``Z(theta)`` the moment/cumulant generating function.
This single operation covers several classic use cases:

* **general tilt** -- an arbitrary statistic ``T`` (default identity ``T(x) = x``);
* **tempering / power** -- ``T(x) = log p_base(x)`` gives ``p_theta ~ p_base^{1+theta}`` (annealed
  importance sampling, parallel tempering);
* **Esscher / MGF tilt** -- continuous exponential-family bases (Gaussian, Gamma, ...) whose
  ``Z(theta)`` is closed form (rare-event importance sampling, actuarial pricing).

The normalizer ``Z(theta)`` is resolved in three ways, in order: an explicit user
``log_normalizer`` (the CGF); a registered analytic tilt for the base family (identity statistic
only -- these also yield the exact in-family tilted distribution, so sampling is exact); otherwise
exact enumeration ``Z = sum_a p(a) exp(theta . T(a))`` for an enumerable base. It pairs with the
support tools (the tilted normalizer is the same sum the enumeration bounds reason about) and sits
beside :class:`~mixle.stats.combinator.truncated.TruncatedDistribution` -- truncation is the
degenerate tilt whose statistic is a ``{0, -inf}`` indicator.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.stats.combinator._base import MaskedBaseEncoder, SingleChildAccumulator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


@dataclass
class TiltResult:
    """Result of resolving the tilt normalizer: ``log Z(theta)`` and, when the tilt closes in the
    base family, the exact tilted distribution (same family) for exact sampling."""

    log_normalizer: float
    closed_form: SequenceEncodableProbabilityDistribution | None = None


# --- analytic tilt registry ("register, don't branch") -----------------------------------
# Each entry maps a base distribution type to ``fn(base, theta) -> TiltResult`` for the identity
# statistic ``T(x) = x``. Returning ``closed_form`` (the in-family tilted distribution) gives both
# an exact normalizer and an exact sampler. New exponential families register their CGF here
# without touching the combinator.
_TILT_REGISTRY: dict[type, Callable[[Any, float], TiltResult]] = {}


def register_exponential_tilt(dist_type: type, fn: Callable[[Any, float], TiltResult]) -> None:
    """Register an analytic identity-statistic tilt ``fn(base, theta) -> TiltResult`` for ``dist_type``."""
    if not callable(fn):
        raise TypeError("tilt factory must be callable.")
    _TILT_REGISTRY[dist_type] = fn


def registered_tilt_families() -> list[str]:
    """Return the names of base families with a registered analytic tilt."""
    return sorted(t.__name__ for t in _TILT_REGISTRY)


def _register_builtin_tilts() -> None:
    from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
    from mixle.stats.univariate.continuous.gamma import GammaDistribution
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
    from mixle.stats.univariate.discrete.poisson import PoissonDistribution

    def gaussian_tilt(base: Any, theta: float) -> TiltResult:
        # N(mu, s2) tilted by theta (identity stat) -> N(mu + theta s2, s2); logZ = theta mu + theta^2 s2 / 2.
        mu, s2 = float(base.mu), float(base.sigma2)
        logz = theta * mu + 0.5 * theta * theta * s2
        return TiltResult(logz, GaussianDistribution(mu + theta * s2, s2))

    def poisson_tilt(base: Any, theta: float) -> TiltResult:
        # Poisson(lam) -> Poisson(lam e^theta); logZ = lam (e^theta - 1).
        lam = float(base.lam)
        logz = lam * (np.expm1(theta))
        return TiltResult(logz, PoissonDistribution(lam * np.exp(theta)))

    def gamma_tilt(base: Any, theta: float) -> TiltResult:
        # Gamma(k, scale) -> Gamma(k, scale/(1 - theta scale)) for theta < 1/scale; logZ = -k log(1 - theta scale).
        k, scale = float(base.k), float(base.theta)
        if theta * scale >= 1.0:
            return TiltResult(float("inf"))
        logz = -k * np.log1p(-theta * scale)
        return TiltResult(logz, GammaDistribution(k, scale / (1.0 - theta * scale)))

    def exponential_tilt(base: Any, theta: float) -> TiltResult:
        # Exponential(beta = mean) -> Exponential(beta/(1 - theta beta)) for theta < 1/beta; logZ = -log(1 - theta beta).
        beta = float(base.beta)
        if theta * beta >= 1.0:
            return TiltResult(float("inf"))
        logz = -np.log1p(-theta * beta)
        return TiltResult(logz, ExponentialDistribution(beta / (1.0 - theta * beta)))

    register_exponential_tilt(GaussianDistribution, gaussian_tilt)
    register_exponential_tilt(PoissonDistribution, poisson_tilt)
    register_exponential_tilt(GammaDistribution, gamma_tilt)
    register_exponential_tilt(ExponentialDistribution, exponential_tilt)


_register_builtin_tilts()


def _coerce_theta(theta: Any) -> Any:
    arr = np.asarray(theta, dtype=float)
    return float(arr) if arr.ndim == 0 else arr


def tilt_log_normalizer(
    base: SequenceEncodableProbabilityDistribution,
    theta: Any,
    statistic_kind: str,
    statistic_fn: Callable[[Any], Any] | None,
    user_normalizer: Callable[[Any], float] | float | None,
) -> TiltResult:
    """Resolve ``log Z(theta)`` (and any closed-form tilted distribution) for a base + statistic.

    Resolution order: explicit ``user_normalizer`` -> registered analytic tilt (identity statistic
    only) -> exact enumeration over an enumerable base. Raises if none applies.
    """
    if user_normalizer is not None:
        logz = float(user_normalizer(theta) if callable(user_normalizer) else user_normalizer)
        return TiltResult(logz)

    if statistic_kind == "identity":
        fn = _TILT_REGISTRY.get(type(base))
        if fn is not None:
            return fn(base, theta)

    # Exact enumeration: Z = sum_a p(a) exp(theta . T(a)).
    try:
        items = list(base.enumerator())
    except EnumerationError as e:
        raise ValueError(
            "tilt normalizer needs an explicit log_normalizer, a registered analytic family "
            "(%s) under the identity statistic, or an enumerable base: %s"
            % (", ".join(registered_tilt_families()), e.reason)
        ) from None
    lps = np.asarray([float(lp) for _, lp in items], dtype=float)
    tvals = np.asarray([_statistic_value(statistic_kind, statistic_fn, base, v) for v, _ in items], dtype=float)
    tilt_terms = tvals @ np.atleast_1d(theta) if tvals.ndim > 1 else np.atleast_1d(theta)[0] * tvals
    return TiltResult(float(logsumexp(lps + tilt_terms)))


def _statistic_value(kind: str, fn: Callable[[Any], Any] | None, base: Any, x: Any) -> Any:
    if kind == "identity":
        return float(np.asarray(x, dtype=float)) if np.ndim(x) == 0 else np.asarray(x, dtype=float)
    if kind == "log_density":
        return float(base.log_density(x))
    return fn(x)


class ExponentialTiltedDistribution(SequenceEncodableProbabilityDistribution):
    """A base distribution reweighted by ``exp(theta . T(x))`` and renormalized by ``Z(theta)``."""

    def __init__(
        self,
        base: SequenceEncodableProbabilityDistribution,
        theta: Any,
        statistic: str | Callable[[Any], Any] | None = None,
        log_normalizer: Callable[[Any], float] | float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an exponentially tilted distribution.

        Args:
            base: the base distribution to tilt.
            theta: tilt parameter -- a scalar (identity / tempering / scalar statistic) or a vector
                matching a vector-valued statistic.
            statistic: ``None`` for the identity ``T(x) = x``; ``"log_density"`` for tempering
                ``T(x) = log p_base(x)`` (giving ``p ~ p_base^{1+theta}``); or a callable ``T(x)``.
            log_normalizer: optional CGF -- a callable ``theta -> log Z`` or a constant. Required
                when the base is neither registered (identity statistic) nor enumerable.
            name, keys: optional instance name / parameter key.
        """
        self.base = base
        self.theta = _coerce_theta(theta)
        self.name = name
        self.keys = keys
        self._user_normalizer = log_normalizer
        if statistic is None:
            self._stat_kind = "identity"
            self._stat_fn = None
        elif statistic == "log_density":
            self._stat_kind = "log_density"
            self._stat_fn = None
        elif callable(statistic):
            self._stat_kind = "callable"
            self._stat_fn = statistic
        else:
            raise ValueError("statistic must be None, 'log_density', or a callable.")

        result = tilt_log_normalizer(base, self.theta, self._stat_kind, self._stat_fn, log_normalizer)
        if not np.isfinite(result.log_normalizer):
            raise ValueError("tilt normalizer Z(theta) is not finite; theta is outside the family's domain.")
        self.log_z = float(result.log_normalizer)
        self._closed_form = result.closed_form

    def __str__(self) -> str:
        stat = self._stat_kind if self._stat_kind != "callable" else "callable"
        return "ExponentialTiltedDistribution(%s, theta=%s, statistic=%s, name=%s, keys=%s)" % (
            str(self.base),
            repr(self.theta),
            stat,
            repr(self.name),
            repr(self.keys),
        )

    def _statistic(self, x: Any) -> Any:
        return _statistic_value(self._stat_kind, self._stat_fn, self.base, x)

    def _tilt_term(self, t: Any) -> float:
        if np.ndim(self.theta) == 0:
            return float(self.theta) * float(t)
        return float(np.dot(self.theta, np.asarray(t, dtype=float)))

    def density(self, x: Any) -> float:
        """Return the tilted probability/density at ``x``."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return ``log p_base(x) + theta . T(x) - log Z``."""
        base_lp = float(self.base.log_density(x))
        if not np.isfinite(base_lp):
            return base_lp
        return base_lp + self._tilt_term(self._statistic(x)) - self.log_z

    def seq_log_density(self, x: tuple[Any, np.ndarray]) -> np.ndarray:
        """Return per-row tilted log-densities for an encoded batch."""
        base_enc, tvals = x
        base_lp = np.asarray(self.base.seq_log_density(base_enc), dtype=np.float64)
        if np.ndim(self.theta) == 0:
            tilt = float(self.theta) * np.asarray(tvals, dtype=np.float64)
        else:
            tilt = np.asarray(tvals, dtype=np.float64) @ np.asarray(self.theta, dtype=np.float64)
        return base_lp + tilt - self.log_z

    def closed_form(self) -> SequenceEncodableProbabilityDistribution | None:
        """Return the in-family tilted distribution when the tilt closes in the base family, else None."""
        return self._closed_form

    def sampler(self, seed: int | None = None) -> "ExponentialTiltedSampler":
        """Return a sampler (exact for registered/enumerable bases; SIR otherwise)."""
        return ExponentialTiltedSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None, fit: str = "theta") -> "ExponentialTiltedEstimator":
        """Return an estimator. ``fit='theta'`` fits the tilt by the exponential-family score equation
        ``E_theta[T] = mean(T(data))`` (base + statistic fixed); ``fit='base'`` holds ``theta`` fixed and
        refits the base on the data, re-tilting (the fixed-tilt analogue of the truncation estimator)."""
        if fit not in ("theta", "base"):
            raise ValueError("fit must be 'theta' or 'base'.")
        return ExponentialTiltedEstimator(self, pseudo_count=pseudo_count, fit=fit)

    def dist_to_encoder(self) -> "ExponentialTiltedDataEncoder":
        """Return the data encoder (base encoding plus the precomputed statistic per row)."""
        return ExponentialTiltedDataEncoder(self)

    def support_size(self) -> int | None:
        """Tilting preserves the support, so the cardinality is the base's."""
        return self.base.support_size()

    def enumerator(self) -> "ExponentialTiltedEnumerator":
        """Enumerate the support in descending tilted-probability order."""
        return ExponentialTiltedEnumerator(self)


class ExponentialTiltedEnumerator(DistributionEnumerator):
    """Reweight the base enumeration by ``theta . T(a) - log Z`` and re-sort descending."""

    def __init__(self, dist: ExponentialTiltedDistribution) -> None:
        super().__init__(dist)
        try:
            base_items = list(dist.base.enumerator())
        except EnumerationError as e:
            raise EnumerationError(dist, reason="tilt requires an enumerable base: %s" % e.reason) from None
        tilted = [(v, float(lp) + dist._tilt_term(dist._statistic(v)) - dist.log_z) for v, lp in base_items]
        tilted.sort(key=lambda vl: vl[1], reverse=True)
        self._items = iter(tilted)

    def __next__(self) -> tuple[Any, float]:
        return next(self._items)


class ExponentialTiltedSampler(DistributionSampler):
    """Sample the tilted distribution: exact via the closed-form / enumerated tilted pmf, else SIR."""

    def __init__(self, dist: ExponentialTiltedDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.rng = RandomState(seed)
        self._values: list[Any] | None = None
        self._probs: np.ndarray | None = None
        if dist.closed_form() is not None:
            self.exact = dist.closed_form().sampler(seed=self.rng.randint(0, 2**31 - 1))
        else:
            self.exact = None
            try:
                items = list(dist.base.enumerator())
            except EnumerationError:
                items = None
            if items is not None:
                lps = np.asarray([float(lp) + dist._tilt_term(dist._statistic(v)) for v, lp in items], dtype=float)
                w = np.exp(lps - logsumexp(lps))
                self._values = [v for v, _ in items]
                self._probs = w / w.sum()
            else:
                self.base_sampler = dist.base.sampler(seed=self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one tilted value (or a list of ``size``)."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        if self.exact is not None:
            return self.exact.sample()
        if self._values is not None:
            idx = self.rng.choice(len(self._values), p=self._probs)
            return self._values[idx]
        # Sampling-importance-resampling fallback (approximate; for non-enumerable bases w/o closed form).
        pool = [self.base_sampler.sample() for _ in range(1024)]
        lw = np.asarray([self.dist._tilt_term(self.dist._statistic(v)) for v in pool], dtype=float)
        w = np.exp(lw - logsumexp(lw))
        return pool[int(self.rng.choice(len(pool), p=w / w.sum()))]


class ExponentialTiltedAccumulator(SingleChildAccumulator):
    """Accumulate the count and the weighted sum of the statistic ``T`` (the exp-family score data).

    Carries extra scalar statistics ``(sum_t, count)`` alongside the bare child value, so it overrides
    the delegation trio; only ``key_merge``/``key_replace`` (which forward to the child) are inherited.
    """

    def __init__(self, dim: int, base_accumulator: SequenceEncodableStatisticAccumulator, keys: str | None = None):
        self.dim = dim
        self.base_accumulator = base_accumulator
        self.keys = keys
        self.sum_t = np.zeros(dim, dtype=float)
        self.count = 0.0

    def update(self, x: Any, weight: float, estimate: ExponentialTiltedDistribution | None) -> None:
        """Accumulate one observation's tilt statistic and child sufficient statistics."""
        t = np.atleast_1d(np.asarray(estimate._statistic(x) if estimate is not None else x, dtype=float))
        self.sum_t += weight * t
        self.count += weight
        self.base_accumulator.update(x, weight, None if estimate is None else estimate.base)

    def seq_update(
        self, x: tuple[Any, np.ndarray], weights: np.ndarray, estimate: ExponentialTiltedDistribution | None
    ) -> None:
        """Accumulate encoded tilt statistics and child sufficient statistics."""
        base_enc, tvals = x
        w = np.asarray(weights, dtype=float)
        tv = np.asarray(tvals, dtype=float)
        tv2 = tv.reshape(len(w), -1)
        self.sum_t += tv2.T @ w
        self.count += float(w.sum())
        self.base_accumulator.seq_update(base_enc, w, None if estimate is None else estimate.base)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the child sufficient statistics for one weighted observation."""
        self.base_accumulator.initialize(x, weight, rng)

    def seq_initialize(self, x: tuple[Any, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize child sufficient statistics from encoded observations."""
        base_enc, _ = x
        self.base_accumulator.seq_initialize(base_enc, np.asarray(weights, dtype=float), rng)

    def combine(self, suff_stat: Any) -> "ExponentialTiltedAccumulator":
        """Merge serialized tilt and child sufficient statistics into this accumulator."""
        sum_t, count, base_ss = suff_stat
        self.sum_t += np.asarray(sum_t, dtype=float)
        self.count += float(count)
        self.base_accumulator.combine(base_ss)
        return self

    def value(self) -> Any:
        """Return tilt statistic totals, total weight, and child sufficient statistics."""
        return (self.sum_t.copy(), self.count, self.base_accumulator.value())

    def from_value(self, x: Any) -> "ExponentialTiltedAccumulator":
        """Restore the accumulator from serialized exponential-tilt statistics."""
        sum_t, count, base_ss = x
        self.sum_t = np.asarray(sum_t, dtype=float).copy()
        self.count = float(count)
        self.base_accumulator.from_value(base_ss)
        return self

    def scale(self, c: float) -> "ExponentialTiltedAccumulator":
        """Scale tilt statistic totals and child sufficient statistics by a constant."""
        self.sum_t *= c
        self.count *= c
        self.base_accumulator.scale(c)
        return self

    def acc_to_encoder(self) -> "DataSequenceEncoder":
        """Return the child encoder used by the delegated base accumulator."""
        return self.base_accumulator.acc_to_encoder()


class ExponentialTiltedAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for :class:`ExponentialTiltedAccumulator`."""

    def __init__(self, dim: int, base_factory: StatisticAccumulatorFactory, keys: str | None = None) -> None:
        self.dim = dim
        self.base_factory = base_factory
        self.keys = keys

    def make(self) -> ExponentialTiltedAccumulator:
        """Create an empty exponential-tilt accumulator."""
        return ExponentialTiltedAccumulator(self.dim, self.base_factory.make(), keys=self.keys)


class ExponentialTiltedEstimator(ParameterEstimator):
    """Fit the tilt by the exp-family score equation (``fit='theta'``) or refit the base (``fit='base'``)."""

    def __init__(
        self,
        prototype: ExponentialTiltedDistribution,
        pseudo_count: float | None = None,
        fit: str = "theta",
        max_iter: int = 60,
        tol: float = 1e-8,
    ) -> None:
        self.prototype = prototype
        self.base = prototype.base
        self.theta = prototype.theta
        self.dim = int(np.atleast_1d(prototype.theta).shape[0])
        self.pseudo_count = pseudo_count
        self.fit = fit
        self.max_iter = max_iter
        self.tol = tol

    def accumulator_factory(self) -> ExponentialTiltedAccumulatorFactory:
        """Return a factory for exponential-tilt sufficient-statistic accumulators."""
        return ExponentialTiltedAccumulatorFactory(
            self.dim,
            self.base.estimator(pseudo_count=self.pseudo_count).accumulator_factory(),
            keys=self.prototype.keys,
        )

    def _mean_grad_logz(self, theta_scalar: float, h: float = 1e-5) -> float:
        # A'(theta) = E_theta[T]; central finite difference of the cumulant generating function.
        proto = self.prototype
        up = tilt_log_normalizer(self.base, theta_scalar + h, proto._stat_kind, proto._stat_fn, proto._user_normalizer)
        dn = tilt_log_normalizer(self.base, theta_scalar - h, proto._stat_kind, proto._stat_fn, proto._user_normalizer)
        return (up.log_normalizer - dn.log_normalizer) / (2.0 * h)

    def _solve_theta(self, mean_t: float) -> float:
        # Solve A'(theta) = mean_t. A is convex (A'' = Var_theta[T] >= 0) so A' is monotone; bracket + bisect.
        def g(th: float) -> float:
            try:
                return self._mean_grad_logz(th) - mean_t
            except Exception:  # noqa: BLE001
                return float("nan")

        lo, hi = -1.0, 1.0
        g0 = g(0.0)
        if not np.isfinite(g0):
            return 0.0
        for _ in range(60):
            glo, ghi = g(lo), g(hi)
            if np.isfinite(glo) and glo > 0:
                lo *= 2.0
                continue
            if np.isfinite(ghi) and ghi < 0:
                hi *= 2.0
                continue
            if np.isfinite(glo) and np.isfinite(ghi) and glo <= 0 <= ghi:
                break
            # A non-finite endpoint marks the family's domain edge -- pull it back toward 0.
            if not np.isfinite(ghi):
                hi = 0.5 * hi
            if not np.isfinite(glo):
                lo = 0.5 * lo
            if abs(hi - lo) < 1e-9:
                return 0.0
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if not np.isfinite(gm):
                hi = mid
                continue
            if abs(gm) < self.tol:
                return mid
            if gm < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def estimate(self, nobs: float | None, suff_stat: Any) -> ExponentialTiltedDistribution:
        """Estimate either the tilt parameter or the tilted base distribution."""
        sum_t, count, base_ss = suff_stat
        base = self.base.estimator(pseudo_count=self.pseudo_count).estimate(count if nobs is None else nobs, base_ss)
        if self.fit == "base":
            # Fixed-tilt: keep theta, re-wrap the refit base.
            return ExponentialTiltedDistribution(
                base,
                self.theta,
                statistic=self._statistic_arg(),
                log_normalizer=self.prototype._user_normalizer,
                name=self.prototype.name,
                keys=self.prototype.keys,
            )
        mean_t = np.asarray(sum_t, dtype=float) / max(count, 1e-12)
        if self.dim == 1:
            theta_hat = self._solve_theta(float(mean_t[0]))
        else:
            raise NotImplementedError("vector-theta MLE is not implemented; pass fit='base' or a scalar statistic.")
        return ExponentialTiltedDistribution(
            self.base,
            theta_hat,
            statistic=self._statistic_arg(),
            log_normalizer=self.prototype._user_normalizer,
            name=self.prototype.name,
            keys=self.prototype.keys,
        )

    def _statistic_arg(self):
        proto = self.prototype
        if proto._stat_kind == "identity":
            return None
        if proto._stat_kind == "log_density":
            return "log_density"
        return proto._stat_fn


class ExponentialTiltedDataEncoder(MaskedBaseEncoder):
    """Encode observations via the base encoder, plus the precomputed statistic ``T`` per row."""

    def __init__(self, dist: ExponentialTiltedDistribution) -> None:
        self.base_encoder = dist.base.dist_to_encoder()
        self._dist = dist

    def _extra_columns(self, x: Sequence[Any]) -> tuple[np.ndarray]:
        return (np.asarray([self._dist._statistic(v) for v in x], dtype=float),)
