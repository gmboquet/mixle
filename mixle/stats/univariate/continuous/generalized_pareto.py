"""Generalized Pareto distribution (GPD): the peaks-over-threshold law of extreme exceedances.

By the Pickands-Balkema-de Haan theorem the distribution of exceedances over a high threshold of
almost any distribution converges to a GPD, which makes it the workhorse for modelling tail risk
(hydrology, finance, reliability). With threshold ``loc = mu``, scale ``sigma > 0`` and shape ``xi``,
for ``y = x - mu >= 0``:

    f(x) = (1/sigma) (1 + xi * y / sigma) ** (-1/xi - 1)   (xi != 0),
    f(x) = (1/sigma) exp(-y / sigma)                       (xi == 0, the exponential tail).

``xi > 0`` is a heavy (Pareto) tail, ``xi = 0`` exponential, ``xi < 0`` a tail with a finite upper
endpoint at ``mu - sigma/xi``. The threshold ``mu`` is treated as a *fixed, known* level (chosen, not
fit -- the standard peaks-over-threshold setup); ``sigma`` and ``xi`` are fit by method of moments,
which is closed-form: ``xi = (1 - m^2/v)/2`` and ``sigma = m (1 - xi)`` from the exceedance mean ``m``
and variance ``v`` (valid for ``xi < 1/2``, where the variance is finite).


Reference: Pickands, 'Statistical inference using extreme order statistics', Ann. Statist. (1975).
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
from mixle.stats.univariate.continuous._observation_contracts import (
    AnchoredMomentTrack,
    anchored_pooled_variance,
    consistent_anchored_triple,
    finite_observation,
    finite_observations,
    needs_anchor,
    scored_observation,
    warn_uncorrectable_raw_moments,
)

_XI_TOL = 1.0e-8  # abs(xi) below this is treated as the exponential limit


class GeneralizedParetoSuffStat(tuple):
    """A ``(sum, sum2, count)`` sufficient statistic that also carries two side payloads.

    Behaves exactly like the plain 3-tuple everywhere it is indexed, unpacked, or iterated (it *is*
    one); ``anchored`` is the shift-anchored moment payload
    ``(anchor, sum_i w_i*(x_i - anchor), sum_i w_i*(x_i - anchor)^2)`` that
    :class:`GeneralizedParetoAccumulator` maintains alongside the raw moments so the moment
    inversion survives a threshold far from zero. ``max_x`` is the largest observation actually
    folded into these moments with positive weight (``None`` when no such max is known to be
    trustworthy -- see :meth:`GeneralizedParetoAccumulator.combine`); :meth:`GeneralizedParetoEstimator.estimate`
    uses it to keep a shape<0 fit's implied support consistent with the data it was fit from (T1-01).
    ``max_unverified`` disambiguates WHY ``max_x`` is ``None`` for a prior-carrying accumulator: it is
    ``True`` when a foreign/pre-payload ``combine()`` left real observations with no tracked max, so
    the T1-01 protection could not be applied though it is supposed to be unconditional there, versus
    the ordinary ``False`` case where the max was never relevant to begin with. ``estimate`` reads it
    to disclose that ambiguity via ``numerical_repairs()`` rather than silently falling back to the
    pre-T1-01 unclamped behaviour (T1-02).
    Code that doesn't know about any of these payloads (generic ``scale_suff_stat``, engine kernels,
    ...) sees an ordinary tuple and the estimate falls back to the historical raw path.
    """

    def __new__(
        cls,
        sum_: float,
        sum2_: float,
        count_: float,
        anchored: tuple[float, float, float] | None = None,
        max_x: float | None = None,
        max_unverified: bool = False,
    ):
        obj = super().__new__(cls, (sum_, sum2_, count_))
        obj.anchored = anchored
        obj.max_x = max_x
        obj.max_unverified = max_unverified
        return obj

    def __reduce__(self):
        # A plain tuple subclass with a payload-bearing __new__ does not pickle by default; the
        # Spark/mp reducers round-trip accumulator values through pickle, so keep every payload.
        return (_rebuild_gpd_suff_stat, (tuple(self), self.anchored, self.max_x, self.max_unverified))


def _rebuild_gpd_suff_stat(
    values: tuple, anchored: tuple | None, max_x: float | None = None, max_unverified: bool = False
) -> "GeneralizedParetoSuffStat":
    """Unpickle helper for :class:`GeneralizedParetoSuffStat` (module-level so pickle can import it)."""
    return GeneralizedParetoSuffStat(
        values[0], values[1], values[2], anchored=anchored, max_x=max_x, max_unverified=max_unverified
    )


class GeneralizedParetoPriorMoments(tuple):
    """The ``(mean, second_moment)`` prior pair, carrying the variance it was built from.

    This family is the only one of its siblings whose ``pseudo_count`` prior is encoded as RAW
    MOMENTS: Gumbel, Student-t and Logistic all take ``(loc, scale)`` parameters, from which the
    prior variance is computed exactly at any magnitude. Recovering it here as
    ``second_moment - mean**2`` is the same cancellation this module repairs everywhere else, and it
    is worse than the data path because it is a two-number encoding with no second chance: a prior
    at a threshold of 1.7e9 recovers a variance of 0.0 from a true 10.4167, and at 1e8 recovers
    10.0 -- 4% low.

    Keeping the documented pair AS the tuple and hanging the exact variance beside it fixes the
    library's own :meth:`GeneralizedParetoDistribution.estimator` path without changing the shape
    any caller passes: a plain 2-tuple still means exactly what it always meant.
    """

    def __new__(cls, mean: float, second_moment: float, variance: float | None = None):
        obj = super().__new__(cls, (mean, second_moment))
        obj.variance = variance
        return obj

    def __reduce__(self):
        # Estimators are pickled by the Spark/mp reducers, so the payload has to survive the trip.
        return (_rebuild_gpd_prior_moments, (tuple(self), self.variance))


def _rebuild_gpd_prior_moments(values: tuple, variance: float | None) -> "GeneralizedParetoPriorMoments":
    """Unpickle helper for :class:`GeneralizedParetoPriorMoments` (module-level so pickle can import it)."""
    return GeneralizedParetoPriorMoments(values[0], values[1], variance=variance)


def _prior_variance(suff_stat: Any, loc: float, xi_min: float) -> float:
    """The prior variance of a ``(mean, second_moment)`` pair, exactly when the pair carries it.

    Falls back to the historical ``second_moment - mean**2`` reconstruction for a plain tuple, and
    warns when that reconstruction cannot be trusted. The threshold is the FAMILY's, not float64's:
    a generalized Pareto with exceedance mean ``m`` has variance ``m**2 / (1 - 2*xi)``, so over the
    whole admissible shape range the SMALLEST variance any such prior can have is
    ``m**2 / (1 - 2*xi_min)``. A reconstruction landing below that floor, or a pair whose float64
    resolution is coarse enough to swallow three digits of even the floor, is reporting a prior no
    generalized Pareto has.

    Taking the floor from ``xi_min`` rather than from ``m**2`` is what keeps this off legitimate
    input twice over. A bounded (``xi < 0``) tail genuinely has variance below ``m**2`` -- ``xi =
    -0.3`` gives ``0.625 * m**2`` -- so an ``m**2`` floor would warn on a perfectly ordinary prior;
    and a resolution-only test would fire on a deliberately zero-variance prior at an ORDINARY mean,
    where the encoding represents zero perfectly well and there is nothing to say. Two cases stay
    deliberately quiet: a prior at a mean around 1e6, where the reconstruction still keeps ~5 digits
    (measured 3.9e-6 relative), and any pair carrying the payload.
    """
    mean, second_moment = float(suff_stat[0]), float(suff_stat[1])
    carried = getattr(suff_stat, "variance", None)
    if carried is not None and np.isfinite(carried) and carried >= 0.0:
        return float(carried)
    recovered = max(second_moment - mean * mean, 0.0)
    floor = (mean - float(loc)) ** 2 / (1.0 - 2.0 * min(float(xi_min), 0.0))
    resolution = float(np.spacing(abs(second_moment)))
    if floor > 0.0 and (recovered < floor or 1.0e3 * resolution > floor):
        import warnings

        warnings.warn(
            "generalized-Pareto pseudo_count prior was supplied as the raw pair "
            "(mean=%.6g, second_moment=%.6g) and its variance has to be recovered as "
            "second_moment - mean**2, which gives %.6g. Every generalized Pareto with exceedance "
            "mean %.6g and a shape at or above xi_min has variance at least %.6g, so that prior is "
            "not one this family can hold: "
            "the variance did not survive the encoding at this magnitude (float64 resolves it only "
            "to %.3g). Build the prior with GeneralizedParetoDistribution(...).estimator("
            "pseudo_count), which carries the exact variance, or express the prior relative to the "
            "threshold." % (mean, second_moment, recovered, mean - float(loc), floor, resolution),
            RuntimeWarning,
            stacklevel=3,
        )
    return recovered


def _consistent_gpd_max(suff_stat: Any, raw_count: float) -> float | None:
    """The tracked observation max carried by ``suff_stat``, when it can be trusted, else ``None``.

    Mirrors :func:`consistent_anchored_triple`'s trust rule for the sibling ``.anchored`` payload: a
    ``max_x`` is only usable when finite and the statistic represents at least one real (raw, not
    prior-pseudo) observation. A plain tuple, or an accumulator whose max was tainted by a
    foreign/pre-payload merge (see :meth:`GeneralizedParetoAccumulator.combine`), carries no max, and
    :meth:`GeneralizedParetoEstimator.estimate` falls back to never clamping -- the pre-T1-01 behavior
    for that (rare) case, unless :func:`_gpd_max_unverified` says the ambiguity needs disclosing
    (T1-02).
    """
    max_x = getattr(suff_stat, "max_x", None)
    if max_x is None or raw_count <= 0.0 or not np.isfinite(max_x):
        return None
    return float(max_x)


def _gpd_max_unverified(suff_stat: Any) -> bool:
    """Whether ``suff_stat``'s missing ``max_x`` is a disclosed ambiguity rather than "not relevant".

    A prior-carrying accumulator's support-consistency protection (T1-01) is supposed to be
    unconditional (see :meth:`GeneralizedParetoAccumulator._max_x_relevant`), but a foreign/pre-payload
    ``combine()`` can still leave it with real observations and no trustworthy max (see
    :meth:`GeneralizedParetoAccumulator.value`). ``True`` only in exactly that case, so
    :meth:`GeneralizedParetoEstimator.estimate` can disclose the ambiguity via ``numerical_repairs()``
    (T1-02) instead of conflating it with the ordinary, quiet "max was never relevant" case a plain
    tuple or a no-prior accumulator's ``value()`` also reports as ``max_x=None``.
    """
    return bool(getattr(suff_stat, "max_unverified", False))


def _gpd_observations(
    value: Any,
    *,
    loc: float,
    scale: float | None = None,
    shape: float | None = None,
) -> np.ndarray:
    upper = None
    if shape is not None and scale is not None and shape < -_XI_TOL:
        upper = loc - scale / shape
    return finite_observations(
        value,
        label="generalized-Pareto observations",
        minimum=loc,
        maximum=upper,
    )


class GeneralizedParetoDistribution(SequenceEncodableProbabilityDistribution):
    """Generalized Pareto distribution with threshold ``loc``, scale ``> 0`` and shape ``xi``."""

    def __init__(
        self, scale: float, shape: float, loc: float = 0.0, name: str | None = None, keys: str | None = None
    ) -> None:
        if scale <= 0.0 or not np.isfinite(scale) or not np.isfinite(shape) or not np.isfinite(loc):
            raise ValueError("GeneralizedParetoDistribution requires finite parameters and scale > 0.")
        self.scale = float(scale)
        self.shape = float(shape)  # xi
        self.loc = float(loc)  # mu (threshold)
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "GeneralizedParetoDistribution(%s, %s, loc=%s, name=%s, keys=%s)" % (
            repr(self.scale),
            repr(self.shape),
            repr(self.loc),
            repr(self.name),
            repr(self.keys),
        )

    def _upper(self) -> float:
        """Upper endpoint of the support (``inf`` unless ``xi < 0``)."""
        return self.loc - self.scale / self.shape if self.shape < -_XI_TOL else math.inf

    def density(self, x: float) -> float:
        """Return the probability density at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single observation (``-inf`` outside the support)."""
        xx = scored_observation(x, label="generalized-Pareto observations")
        y = xx - self.loc
        if y < 0.0 or xx > self._upper():
            return -np.inf
        if abs(self.shape) < _XI_TOL:
            return -self.log_scale - y / self.scale
        t = 1.0 + self.shape * y / self.scale
        if t <= 0.0:
            return -np.inf
        return -self.log_scale - (1.0 / self.shape + 1.0) * math.log(t)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        y = np.asarray(x, dtype=np.float64) - self.loc
        if abs(self.shape) < _XI_TOL:
            rv = -self.log_scale - y / self.scale
        else:
            t = 1.0 + self.shape * y / self.scale
            with np.errstate(divide="ignore", invalid="ignore"):
                rv = -self.log_scale - (1.0 / self.shape + 1.0) * np.log(t)
            rv = np.where(t <= 0.0, -np.inf, rv)
        return np.where(y < 0.0, -np.inf, rv)

    # --- compute-engine backend (numpy + torch/GPU): scoring + sufficient statistics in engine ops ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated generalized-Pareto kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for generalized Pareto distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="generalized_pareto",
            distribution_type=cls,
            parameters=(ParameterSpec("scale", constraint="positive"), ParameterSpec("shape"), ParameterSpec("loc")),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row GPD moment sums in accumulator order ``(sum, sum2, count)``."""
        xx = engine.asarray(x)
        return xx, xx * xx, xx * 0.0 + engine.asarray(1.0)

    @staticmethod
    def backend_log_density_from_params(x: Any, scale: Any, shape: Any, loc: Any, engine: Any) -> Any:
        """Engine-neutral GPD log-density; the ``abs(xi) < tol`` exponential limit is selected per element."""
        y = x - loc
        neg_inf = engine.asarray(float("-inf"))
        is_limit = engine.abs(shape) < _XI_TOL
        xi_safe = engine.where(is_limit, engine.asarray(1.0), shape)  # keep the general branch NaN-free
        t = 1.0 + xi_safe * y / scale
        t_pos = engine.where(t > 0.0, t, engine.asarray(1.0))
        general = -engine.log(scale) - (1.0 / xi_safe + 1.0) * engine.log(t_pos)
        general = engine.where(t > 0.0, general, neg_inf)
        limit = -engine.log(scale) - y / scale
        rv = engine.where(is_limit, limit, general)
        return engine.where(y < 0.0, neg_inf, rv)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.scale),
            engine.asarray(self.shape),
            engine.asarray(self.loc),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["GeneralizedParetoDistribution"], engine: Any) -> dict[str, Any]:
        """Stacked GPD parameters for a homogeneous mixture kernel."""
        return {
            "scale": engine.asarray([d.scale for d in dists]),
            "shape": engine.asarray([d.shape for d in dists]),
            "loc": engine.asarray([d.loc for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of GPD log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["scale"][None, :], params["shape"][None, :], params["loc"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Stacked GPD moment sums ``(sum, sum2, count)`` using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        return (
            engine.sum(ww * xx[:, None], axis=0),
            engine.sum(ww * (xx * xx)[:, None], axis=0),
            engine.sum(ww, axis=0),
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact)."""
        from scipy.stats import genpareto as _sp

        return float(_sp.cdf(x, self.shape, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``."""
        from scipy.stats import genpareto as _sp

        return float(_sp.ppf(q, self.shape, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        """Mean loc + scale/(1-xi) for xi < 1, else inf."""
        xi = self.shape
        return float(self.loc + self.scale / (1.0 - xi)) if xi < 1.0 else float("inf")

    def variance(self) -> float:
        """Variance scale^2 / ((1-xi)^2 (1-2xi)) for xi < 1/2, else inf."""
        xi = self.shape
        if xi < 0.5:
            return float(self.scale * self.scale / ((1.0 - xi) ** 2 * (1.0 - 2.0 * xi)))
        return float("inf")

    def entropy(self) -> float:
        """Differential entropy log(scale) + xi + 1."""
        return float(self.log_scale + self.shape + 1.0)

    def sampler(self, seed: int | None = None) -> "GeneralizedParetoSampler":
        """Return a sampler for drawing observations from this distribution."""
        return GeneralizedParetoSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GeneralizedParetoEstimator":
        """Return a method-of-moments estimator for ``scale`` and ``shape`` at the fixed threshold ``loc``."""
        if pseudo_count is None:
            return GeneralizedParetoEstimator(loc=self.loc, name=self.name, keys=self.keys)
        # Convert this distribution's own (scale, shape) into the raw first two moments -- the
        # space estimate() accumulates in -- so pseudo_count can blend a prior pseudo-sample toward
        # them (mirrors GumbelEstimator / WeibullEstimator's suff_stat pattern).
        mean0 = self.mean()
        var0 = self.variance()
        second0 = var0 + mean0 * mean0
        # The pair is still exactly ``(mean, second_moment)``; the variance rides beside it so that
        # ``estimate`` never has to recover it by differencing at threshold magnitude, where it is
        # destroyed (0.0 recovered from 10.4167 at a threshold of 1.7e9). See
        # :class:`GeneralizedParetoPriorMoments`.
        return GeneralizedParetoEstimator(
            loc=self.loc,
            pseudo_count=pseudo_count,
            suff_stat=GeneralizedParetoPriorMoments(mean0, second0, variance=var0),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "GeneralizedParetoDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return GeneralizedParetoDataEncoder(
            loc=self.loc,
            scale=self.scale,
            shape=self.shape,
        )


class GeneralizedParetoSampler(DistributionSampler):
    """Draw iid GPD observations by inverse-CDF transform."""

    def __init__(self, dist: GeneralizedParetoDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples by inverse CDF."""
        d = self.dist
        u = self.rng.uniform(size=size)  # uniform; 1-U is also uniform, so use U directly below
        if abs(d.shape) < _XI_TOL:
            y = -d.scale * np.log(u)
        else:
            y = (d.scale / d.shape) * (np.power(u, -d.shape) - 1.0)
        return d.loc + y


class GeneralizedParetoAccumulator(AnchoredMomentTrack, SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for GPD estimation.

    Alongside the raw ``(sum, sum2, count)`` the accumulator keeps the conditioning-gated
    shift-anchored moment track of :class:`AnchoredMomentTrack`. Peaks-over-threshold data is the
    case that needs it most: BOTH moments the inversion uses are differences taken at data
    magnitude -- the exceedance mean ``sum_x/count - loc`` and the variance ``E[x^2]-E[x]^2`` -- so
    a threshold at epoch seconds turned a shape of ``+0.16`` into a flat ``0`` and the exponential
    limit, silently. Well-conditioned data never activates the track and accumulates bit-identically
    to the historical single-pass path.
    """

    def __init__(
        self, loc: float = 0.0, name: str | None = None, keys: str | None = None, has_prior: bool = False
    ) -> None:
        if not np.isfinite(loc):
            raise ValueError("generalized-Pareto accumulator threshold must be finite")
        self.loc = float(loc)
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys
        # Whether the estimator this accumulator was built for carries a pseudo_count prior: a
        # prior blend is invisible to the cheap raw-moment relevance pre-check in
        # `_max_x_relevant` (it can push the FINAL fitted shape negative even when raw data alone
        # would not, see estimate()), so a prior-carrying estimator always carries the max forward
        # rather than risk dropping it before the blend that turns out to need it.
        self.has_prior = has_prior
        self._init_anchor()
        # Largest observation folded in with positive weight -- see `GeneralizedParetoSuffStat`.
        # `_max_tainted` marks the rare case where a combined-in partial carried real observations
        # but no max payload (a foreign/pre-payload tuple): from that point the true max can no
        # longer be vouched for, so `value()` reports `max_x=None` and the M-step falls back to its
        # historical, unclamped behaviour rather than clamping against an undercounted max.
        self._max_x = -math.inf
        self._max_tainted = False

    def update(self, x: float, weight: float, estimate: GeneralizedParetoDistribution | None) -> None:
        """Accumulate weighted first and second moments for one observation."""
        # `estimate.scale`/`estimate.shape` are deliberately NOT forwarded to the observation
        # validator here (contrast `GeneralizedParetoDataEncoder.seq_encode`, which validates an
        # externally-supplied array against a FIXED distribution's support). This call re-accumulates
        # the SAME training data through the CURRENT mid-EM model on every iteration, and for a
        # shape<0 method-of-moments fit that model's implied upper endpoint (loc - scale/xi) can
        # legitimately sit below the data's own max -- an ordinary, expected transient of MoM
        # refinement (T1-01), not a data problem. The accumulated value never depends on scale/shape,
        # so skipping that bound here does not change the resulting fit; only the structural
        # `x >= loc` invariant, which does not change across iterations, is still enforced.
        loc = estimate.loc if estimate is not None else self.loc
        xx = finite_observation(
            x,
            label="generalized-Pareto observation",
            minimum=loc,
        )
        if weight > 0.0:
            self._max_x = max(self._max_x, xx)
        # BEFORE the raw fold, so an activation only converts content the gate already vouched for.
        self._anchor_scalar(xx, weight)
        self.sum += xx * weight
        self.sum2 += xx * xx * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: GeneralizedParetoDistribution | None) -> None:
        """Accumulate weighted first and second moments from encoded data."""
        # See `update()` above: `estimate.scale`/`estimate.shape` are deliberately not forwarded to
        # the validator here -- this re-accumulates the SAME training data through the current
        # mid-EM model, whose implied upper endpoint can legitimately (and transiently) sit below
        # the data's own max for a shape<0 method-of-moments fit (T1-01). Only the fixed threshold
        # `loc` is a real invariant of the data across iterations.
        loc = estimate.loc if estimate is not None else self.loc
        xx = _gpd_observations(x, loc=loc)
        positive = weights > 0.0
        if xx.size and np.any(positive):
            self._max_x = max(self._max_x, float(np.max(xx[positive])))
        self._anchor_fold_chunk(xx, weights)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "GeneralizedParetoAccumulator":
        """Merge another generalized-Pareto sufficient-statistic tuple."""
        self._anchor_absorb(suff_stat)
        incoming_max = getattr(suff_stat, "max_x", None)
        if incoming_max is not None and np.isfinite(incoming_max):
            self._max_x = max(self._max_x, float(incoming_max))
        elif suff_stat[2] > 0.0:
            # Real observations arrived with no tracked max (a foreign/pre-payload tuple): this
            # accumulator's own max can no longer be vouched for as the training data's true max.
            self._max_tainted = True
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def _max_x_relevant(self) -> bool:
        """Whether the RAW moments even admit a shape<0 fit, so a max is worth carrying.

        A cheap pre-check on the same (sum, sum2, count) moments the raw M-step path inverts. Only
        consulted when ``self.has_prior`` is False: a caller whose estimator carries a
        ``pseudo_count`` prior always carries the max regardless of this check, because the prior
        blend -- invisible here -- can push the FINAL fitted shape negative even when the raw data
        alone would not (an adversarial review of T1-01 found this: dropping the max for
        raw-positive-shape data reintroduced the exact self-inconsistent-fit crash T1-01 exists to
        prevent, whenever a strong enough prior pulled the blended shape negative). For a no-prior
        estimator the raw moments ARE the final moments, so this check is exact, not an
        approximation: a positive-or-zero-shape fit has infinite support and can never need the
        clamp, so ordinary heavy-tailed data -- the common GPD case -- never carries the payload at
        all, matching the anchored track's own conditioning gate and the pinned
        ``test_ordinary_data_never_activates_the_anchored_track`` invariant.
        """
        if self.count <= 0.0:
            return False
        mean_x = self.sum / self.count
        var = self.sum2 / self.count - mean_x * mean_x
        m = mean_x - self.loc
        if m <= 0.0 or var <= 0.0:
            return False
        return 0.5 * (1.0 - m * m / var) < -_XI_TOL

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count.

        A plain 3-tuple for every consumer that treats it as one; once the shift-anchored moment
        track is live, or a trustworthy observation max is known AND relevant, it is a
        :class:`GeneralizedParetoSuffStat` additionally carrying the anchored moments in its
        ``.anchored`` attribute and/or the max in ``.max_x``, so :meth:`combine` can fold them in and
        :meth:`GeneralizedParetoEstimator.estimate` can invert threshold-invariant moments and keep a
        shape<0 fit's implied support consistent with the data (T1-01). "Relevant" is always true
        for a prior-carrying estimator (see :meth:`_max_x_relevant`); for one without a prior it is
        the same cheap raw-moment pre-check as before, so ordinary well-conditioned data still
        returns the historical plain tuple, byte for byte.

        For a prior-carrying estimator that protection is supposed to be unconditional, but a foreign
        (pre-payload) ``combine()`` can still leave real observations with no trustworthy max
        (``self._max_tainted``); rather than silently degrade to the plain tuple that hides this from
        ``estimate``, that case is still reported as a :class:`GeneralizedParetoSuffStat` with
        ``max_x=None`` and ``max_unverified=True``, so ``estimate`` can disclose the ambiguity instead
        of reproducing the T1-01 crash class with zero signal (T1-02).
        """
        anchored = self._anchor_payload()
        max_x = None if (self._max_tainted or not np.isfinite(self._max_x)) else self._max_x
        if max_x is not None and not self.has_prior and not self._max_x_relevant():
            max_x = None
        max_unverified = self.has_prior and self._max_tainted and self.count > 0.0
        if anchored is None and max_x is None and not max_unverified:
            return self.sum, self.sum2, self.count
        return GeneralizedParetoSuffStat(
            self.sum, self.sum2, self.count, anchored=anchored, max_x=max_x, max_unverified=max_unverified
        )

    def from_value(self, x: tuple[float, float, float]) -> "GeneralizedParetoAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum, self.sum2, self.count = float(x[0]), float(x[1]), float(x[2])
        self._anchor_restore(x)
        incoming_max = getattr(x, "max_x", None)
        if incoming_max is not None and np.isfinite(incoming_max):
            self._max_x = float(incoming_max)
            self._max_tainted = False
        else:
            self._max_x = -math.inf
            self._max_tainted = self.count > 0.0
        return self

    def scale(self, c: float) -> "GeneralizedParetoAccumulator":
        """Scale the accumulated statistics in place by ``c``, anchored track included.

        The structural default round-trips through ``value()``/``from_value()`` and
        ``scale_suff_stat`` rebuilds a PLAIN tuple, which would drop the payload and undo the
        repair; ``loc`` is a fixed threshold, not a statistic, and is left alone. The tracked max is
        likewise left alone: uniform weight scaling reweights the aggregate contribution of the same
        observations, it does not change which values were actually observed.
        """
        self._anchor_scale(c)
        self.sum *= c
        self.sum2 *= c
        self.count *= c
        return self

    def acc_to_encoder(self) -> "GeneralizedParetoDataEncoder":
        """Return the encoder used by this accumulator."""
        return GeneralizedParetoDataEncoder(loc=self.loc)


class GeneralizedParetoAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GeneralizedParetoAccumulator."""

    def __init__(
        self, loc: float = 0.0, name: str | None = None, keys: str | None = None, has_prior: bool = False
    ) -> None:
        if not np.isfinite(loc):
            raise ValueError("generalized-Pareto accumulator threshold must be finite")
        self.loc = float(loc)
        self.name = name
        self.keys = keys
        self.has_prior = has_prior

    def make(self) -> GeneralizedParetoAccumulator:
        """Create a fresh generalized-Pareto accumulator."""
        return GeneralizedParetoAccumulator(loc=self.loc, name=self.name, keys=self.keys, has_prior=self.has_prior)


class GeneralizedParetoEstimator(ParameterEstimator):
    """Method-of-moments estimator for GPD scale and shape at a fixed threshold ``loc``.

    Both inverted moments -- the exceedance mean ``sum_x/count - loc`` and the variance
    ``E[x^2]-E[x]^2`` -- are differences taken at DATA magnitude, so a threshold far from zero
    destroys them: peaks over a threshold at epoch seconds fitted ``shape=0`` (the exponential
    limit) instead of ``+0.16``, silently. Whenever the accumulated statistics carry the
    shift-anchored payload (see :class:`GeneralizedParetoAccumulator`) both moments are formed
    about a data anchor instead, which makes the fit THRESHOLD-EQUIVARIANT: estimating on ``x + c``
    with the threshold at ``loc + c`` returns the same ``scale`` and ``shape``, for any constant
    ``c`` the data can carry. With a plain raw tuple -- statistics the conditioning gate never
    needed to anchor, or ones that arrived already reduced from an engine kernel or an older
    serialization -- the historical raw path is used bit-identically, and ``estimate`` warns rather
    than returning moments it cannot stand behind.

    The ``pseudo_count`` prior is carried as the raw pair ``suff_stat = (mean0, second0)``, so its
    variance is recovered by differencing (``second0 - mean0**2``) exactly as before. That
    reconstruction is the prior's own encoding and is unaffected by the anchored track: a prior
    whose mean sits far from zero loses precision in the reconstructed variance, so express such a
    prior in threshold-relative coordinates.

    ``loc`` is a property, not a bare attribute: reassigning it re-targets this SAME estimator at a
    new threshold (the "threshold stability" / mean-residual-life re-thresholding workflow) and, when
    a ``pseudo_count`` prior is configured, re-anchors its frozen ``(mean0, second0)`` moments to the
    new origin so the prior keeps meaning what it meant when built instead of being silently
    reinterpreted relative to the wrong ``loc`` (T1-01).
    """

    def __init__(
        self,
        loc: float = 0.0,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_scale: float = 1.0e-12,
        xi_max: float = 0.5 - 1.0e-6,
        xi_min: float = -10.0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self._loc = float(loc)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        # The threshold `suff_stat`'s (mean, second_moment) prior pair is expressed relative to --
        # always `self._loc` itself at construction, since `.estimator()` builds both together (T1-01).
        self._prior_loc = self._loc
        self.min_scale = min_scale
        self.xi_max = xi_max  # method of moments needs a finite variance (xi < 1/2)
        self.xi_min = xi_min
        self.name = name
        self.keys = keys

    @property
    def loc(self) -> float:
        """The fixed threshold ``mu`` observations are exceedances over."""
        return self._loc

    @loc.setter
    def loc(self, value: float) -> None:
        """Re-target this estimator at a new threshold, re-anchoring any frozen prior with it.

        A natural "threshold stability" workflow (mean-residual-life re-thresholding) re-targets the
        SAME estimator at a new ``loc`` in place. ``.estimator(pseudo_count=...)`` bakes the prior's
        mean/second moment as ABSOLUTE values relative to the distribution's OWN ``loc`` at
        construction time (see :meth:`GeneralizedParetoDistribution.estimator`); left untouched, a
        later re-target would silently reinterpret those absolute moments relative to the wrong
        origin, corrupting the blend with no diagnostic signal (T1-01).

        The exceedance mean ``mean0 - loc`` and the variance are properties of the prior
        DISTRIBUTION, not of ``loc``, so they are threshold-invariant; only the absolute mean shifts
        by exactly the same amount the threshold does. Re-deriving the pair that way keeps the prior
        meaning what it meant when built, at any new origin, without needing to remember more than
        the origin it was last expressed relative to.
        """
        new_loc = float(value)
        if not np.isfinite(new_loc):
            raise ValueError("generalized-Pareto estimator threshold must be finite")
        if self.pseudo_count is not None and self.suff_stat is not None:
            variance = _prior_variance(self.suff_stat, self._prior_loc, self.xi_min)
            mean0 = float(self.suff_stat[0]) + (new_loc - self._prior_loc)
            second0 = variance + mean0 * mean0
            self.suff_stat = GeneralizedParetoPriorMoments(mean0, second0, variance=variance)
        self._prior_loc = new_loc
        self._loc = new_loc

    def accumulator_factory(self) -> GeneralizedParetoAccumulatorFactory:
        """Return an accumulator factory for generalized-Pareto moments."""
        has_prior = self.pseudo_count is not None and self.suff_stat is not None
        return GeneralizedParetoAccumulatorFactory(loc=self.loc, name=self.name, keys=self.keys, has_prior=has_prior)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> GeneralizedParetoDistribution:
        """Estimate scale and shape from exceedance moments at the fixed location."""
        sum_x, sum_x2, count = float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2])
        # The anchored payload and the observation max (when present) both describe the RAW data
        # only -- captured before the prior blend below -- so both are read off the untouched
        # (sum_x, count) pair. The prior contributes a fictitious pseudo-sample with no observed
        # max of its own, so it must never enter the support-consistency clamp below.
        anchored = consistent_anchored_triple(suff_stat, sum_x, count)
        max_x = _consistent_gpd_max(suff_stat, count)
        max_unverified = _gpd_max_unverified(suff_stat)
        raw_count = count

        prior_mean: float | None = None
        prior_variance: float | None = None
        pc = self.pseudo_count
        if pc is not None and self.suff_stat is not None:
            mean0, second0 = self.suff_stat[0], self.suff_stat[1]
            prior_mean = float(mean0)
            prior_variance = _prior_variance(self.suff_stat, self.loc, self.xi_min)
            sum_x += pc * mean0
            sum_x2 += pc * second0
            count += pc
        if count <= 0.0:
            return GeneralizedParetoDistribution(1.0, 0.0, loc=self.loc, name=self.name, keys=self.keys)
        # Whether the raw (non-anchored) path below actually hit cancellation, as opposed to just
        # being the historical well-conditioned path it always was for small-magnitude data (T4-01):
        # only meaningful when that path is the one taken, so it stays False on the anchored branch.
        raw_ill_conditioned = False
        if anchored is not None:
            # Offset space: ``anchor`` is a data value and ``anchor - loc`` is exact (Sterbenz --
            # the threshold is at or below every observation and within a factor of two of one), so
            # the exceedance mean is built from two quantities that are each accurate at EXCEEDANCE
            # scale. Forming it as ``sum_x/count - loc`` instead carries the rounding of a
            # data-magnitude sum into a small difference, which is half of what this fix repairs;
            # the variance below is the other half.
            offset = anchored[1] / raw_count
            if prior_mean is not None:
                offset = (anchored[1] + pc * (prior_mean - anchored[0])) / count
            mean_x = anchored[0] + offset
            m = (anchored[0] - self.loc) + offset
            var, notes = anchored_pooled_variance(
                anchored[0], anchored[1], anchored[2], raw_count, mean_x, pc, prior_mean, prior_variance
            )
        else:
            notes = ()
            mean_x = sum_x / count
            m = mean_x - self.loc  # exceedance mean
            if raw_count <= 0.0 and prior_variance is not None:
                # No real observations at all: `sum_x`/`sum_x2` are pure prior pseudo-moments
                # (`pc*mean0`, `pc*second0`), so `sum_x2/count - mean_x**2` here reduces exactly to
                # the raw-differencing reconstruction `_prior_variance` above already computed
                # correctly (and cancellation-free, via the carried `.variance` payload when the
                # prior was built by `.estimator(pseudo_count)`). Recomputing it by differencing
                # throws that unused signal away and re-introduces the very threshold-magnitude
                # cancellation `_prior_variance` exists to avoid (T1-01: at loc=1.7e9 this recovered
                # shape=+0.499 from a true prior shape of -0.3 with zero raw observations). No warning
                # either: the exact prior variance was used, so there is nothing uncorrectable here.
                var = prior_variance
            else:
                warn_uncorrectable_raw_moments(sum_x, sum_x2, count, family="generalized-Pareto")
                var = sum_x2 / count - mean_x * mean_x
            raw_ill_conditioned = needs_anchor(sum_x, sum_x2, count)
        if m <= 0.0 or var <= 0.0:
            degenerate = GeneralizedParetoDistribution(
                max(m, self.min_scale), 0.0, loc=self.loc, name=self.name, keys=self.keys
            )
            if max_x is None and max_unverified and raw_ill_conditioned:
                # T4-01: the same foreign-combine ambiguity `support-consistency-unverified` below
                # discloses, but reached here because raw-moment cancellation ALSO collapsed the
                # variance to non-positive before the shape<0 clamp ever got a chance to matter --
                # an earlier, unrelated short-circuit must not silently swallow this disclosure.
                notes = notes + (
                    "support-consistency-unverified(raw sufficient statistics were too ill-conditioned "
                    "for a reliable fit at this magnitude and collapsed to a degenerate result: a "
                    "foreign partial statistic was combined into this prior-carrying accumulator "
                    "without a tracked max, so this could not be corroborated)",
                )
            if notes:
                degenerate._numerical_repairs = notes
            return degenerate
        xi = 0.5 * (1.0 - m * m / var)
        xi = min(max(xi, self.xi_min), self.xi_max)
        scale = max(m * (1.0 - xi), self.min_scale)
        if max_x is not None and xi < -_XI_TOL:
            # A shape<0 method-of-moments fit only matches the first two moments; unlike an MLE
            # respecting the finite-support constraint, it has no mechanism to keep its implied
            # upper endpoint (loc - scale/xi) at or above the data's own max. Left alone, every
            # later use of this fit -- another EM iteration re-scoring the SAME training data, a
            # held-out score, a caller's own log_density call -- sees -inf on the excluded points
            # (T1-01: this is what made optimize()/fit() crash on data estimate() itself accepted).
            # Raise scale (never xi, the moment-matched shape) to the smallest value that restores
            # support consistency; a small relative margin absorbs the rounding
            # ``loc - scale/xi`` picks back up on the way out.
            required_scale = abs(xi) * (max_x - self.loc) * (1.0 + 1.0e-9)
            if required_scale > scale:
                notes = notes + ("scale-floored-for-support(%.6g -> %.6g)" % (scale, required_scale),)
                scale = required_scale
        elif max_x is None and max_unverified and (xi < -_XI_TOL or raw_ill_conditioned):
            # T1-02: a prior-carrying accumulator's max is supposed to be unconditionally trustworthy
            # (see `_max_x_relevant`), but a foreign/pre-payload `combine()` can still leave it with
            # real observations and no tracked max. The clamp above can't run without a max to clamp
            # against, so this fit's implied upper endpoint is left unverified against the data it was
            # fit from -- the same silent gap T1-01 closed for the ordinary case, reached here through
            # a merge instead of the moment blend. Disclose it rather than reproduce that silence.
            #
            # T4-01: `xi < -_XI_TOL` alone missed the sibling case where the SAME foreign-combine
            # cancellation instead pushed the fitted shape to the *positive* boundary (or anywhere
            # non-negative) -- a result the ordinary "shape>=0 means infinite support, no clamp
            # needed" reasoning silently trusted even though it came from raw moments too
            # ill-conditioned to trust at all. `raw_ill_conditioned` catches that sibling without
            # touching the ordinary, well-conditioned foreign-combine case (T1-02's own test), which
            # never sets it.
            if xi < -_XI_TOL:
                detail = "implied upper endpoint %.6g not checked against the training data's true max" % (
                    self.loc - scale / xi
                )
            else:
                detail = (
                    "raw sufficient statistics were also too ill-conditioned for a reliable fit at "
                    "this magnitude (fitted shape=%.6g, scale=%.6g)" % (xi, scale)
                )
            notes = notes + (
                "support-consistency-unverified(%s: a foreign partial statistic was combined into "
                "this prior-carrying accumulator without a tracked max)" % detail,
            )
        dist = GeneralizedParetoDistribution(scale, xi, loc=self.loc, name=self.name, keys=self.keys)
        if notes:
            dist._numerical_repairs = notes
        return dist


class GeneralizedParetoDataEncoder(DataSequenceEncoder):
    """Encode GPD observations as a float array."""

    def __init__(
        self,
        loc: float = 0.0,
        scale: float | None = None,
        shape: float | None = None,
    ) -> None:
        if not np.isfinite(loc):
            raise ValueError("generalized-Pareto encoder threshold must be finite")
        if (scale is None) != (shape is None):
            raise ValueError("generalized-Pareto encoder scale and shape must be supplied together")
        if scale is not None and (not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(shape)):
            raise ValueError("generalized-Pareto encoder requires finite shape and scale > 0")
        self.loc = float(loc)
        self.scale = None if scale is None else float(scale)
        self.shape = None if shape is None else float(shape)

    def __str__(self) -> str:
        return "GeneralizedParetoDataEncoder(loc=%r, scale=%r, shape=%r)" % (
            self.loc,
            self.scale,
            self.shape,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GeneralizedParetoDataEncoder)
            and other.loc == self.loc
            and other.scale == self.scale
            and other.shape == self.shape
        )

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return _gpd_observations(
            x,
            loc=self.loc,
            scale=self.scale,
            shape=self.shape,
        )
