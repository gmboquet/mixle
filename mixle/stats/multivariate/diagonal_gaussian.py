"""Diagonal Gaussian distributions, estimators, accumulators, and encoders.

The log-density of an ``n``-dimensional diagonal Gaussian observation
``x = (x_1, x_2, ..., x_n)`` with mean ``mu`` and diagonal covariance
``covar = diag(s2_1, s2_2, ..., s2_n)`` is:

    log(p_mat(x)) = -0.5*sum_{i=1}^{n} (x_i-m_i)^2 / s2_i - 0.5*log(s2_i) - (n/2)*log(2*pi).

Reference: Mardia, Kent & Bibby, *Multivariate Analysis* (Academic Press, 1979).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import *
from mixle.inference.fisher import FixedFisherView
from mixle.stats.bayes.multivariate_normal_gamma import MultivariateNormalGammaDistribution
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.multivariate._vector_contracts import (
    batch as vector_batch,
)
from mixle.stats.multivariate._vector_contracts import (
    dimension as vector_dimension,
)
from mixle.stats.multivariate._vector_contracts import (
    event as vector_event,
)
from mixle.stats.multivariate._vector_contracts import (
    finite_scalar,
    gaussian_moments,
    gaussian_prior_statistics,
    marginal_indices,
    pooled_gaussian_covariance,
    pseudo_counts,
    require_pseudo_moments,
)
from mixle.stats.multivariate._vector_contracts import (
    vector as vector_parameter,
)
from mixle.stats.multivariate._vector_contracts import (
    weight as observation_weight,
)
from mixle.stats.multivariate._vector_contracts import (
    weights as observation_weights,
)
from mixle.utils.aliasing import MISSING, coalesce_alias
from mixle.utils.special import digamma
from mixle.utils.vector import owned_backend_parameter

_FLOOR_MATERIAL_RATIO = 1.0e-3
"""Relative inflation of the smallest raw variance above which the M-step's variance floor is
recorded as a numerical repair. Same threshold, and the same reporting-only role, as the full
covariance estimator's ``_RIDGE_MATERIAL_RATIO``: 1000x above the ~1e-6 design point, so a
scale-homogeneous fit stays quiet while a 0.1% inflation -- roughly where likelihood/AIC comparisons
start to move -- is reported. Crossing it records a string; it never rejects or alters the fit."""


# Conditioning threshold for the shift-anchored moment gate, and the same constant the univariate
# GaussianAccumulator uses (mixle.stats.univariate.continuous.gaussian._ANCHOR_CONDITION_RATIO):
# the raw ``E[x^2] - mu^2`` variance loses about ``eps * (mean/sd)^2`` relative accuracy, so a
# (mean/sd)^2 up to 4e6 (ratio ~2000) keeps the raw form within ~1e-9 relative error and the
# historical single-pass statistics are bit-preserved there. Beyond it the anchored track takes over.
_ANCHOR_CONDITION_RATIO = 4.0e6

_ANCHOR_MEAN_ULP = 8.8817841970012523e-16
"""4 * eps -- the granularity of recomputing a mean at a given magnitude. A scatter below
``count * (_ANCHOR_MEAN_ULP * |mean|)^2`` is the mean's own rounding, not data (see
``_anchored_pooled_variances``)."""


def _needs_anchor(chunk_sum: np.ndarray, chunk_sum2: np.ndarray, w_sum: float) -> bool:
    """Whether a chunk's weighted moments are too ill-conditioned for the raw variance form.

    The per-coordinate version of the univariate gate; a single coordinate needing the anchor
    activates it for all of them, because the anchor is one vector. ``spread2`` computed here is
    itself the cancellation-prone estimate, but as a GATE it is reliable: when cancellation has
    corrupted it, the corruption is bounded by ``eps * m^2``, which still leaves ``m*m`` orders of
    magnitude above ``_ANCHOR_CONDITION_RATIO * spread2``. A non-positive computed spread activates
    the anchor outright (constant or near-constant data in that coordinate).
    """
    m = chunk_sum / w_sum
    spread2 = chunk_sum2 / w_sum - m * m
    m2 = m * m
    return bool(np.any(spread2 <= 0.0) or np.any(m2 > _ANCHOR_CONDITION_RATIO * spread2))


class DiagonalGaussianSuffStat(tuple):
    """A ``(sum, sum_squares, count)`` sufficient statistic that also carries shift-anchored moments.

    Behaves exactly like the plain 3-tuple everywhere it is indexed, unpacked, or iterated (it *is*
    one), so generic consumers -- ``scale_suff_stat``, the declared ``StatisticSpec`` reader, engine
    kernels -- see nothing new. ``anchored`` is extra payload:
    ``(anchor, sum_i w_i*(x_i - anchor), sum_i w_i*(x_i - anchor)^2)`` per coordinate, which
    :meth:`DiagonalGaussianAccumulator.combine` folds in and
    :meth:`DiagonalGaussianEstimator.estimate` uses to compute shift-invariant variances. Mirrors
    :class:`mixle.stats.univariate.continuous.gaussian.GaussianSuffStat`.
    """

    def __new__(cls, sum_: np.ndarray, sum2_: np.ndarray, count_: float, anchored: tuple | None = None):
        obj = super().__new__(cls, (sum_, sum2_, count_))
        obj.anchored = anchored
        return obj

    def __reduce__(self):
        # A tuple subclass with a payload-bearing __new__ does not pickle by default, and the
        # Spark/multiprocessing reducers round-trip accumulator values through pickle.
        return (_rebuild_diagonal_suff_stat, (tuple(self), self.anchored))


def _rebuild_diagonal_suff_stat(values: tuple, anchored: tuple | None) -> "DiagonalGaussianSuffStat":
    """Unpickle helper for :class:`DiagonalGaussianSuffStat` (module-level so pickle can import it)."""
    return DiagonalGaussianSuffStat(values[0], values[1], values[2], anchored=anchored)


def _consistent_anchored_moments(
    suff_stat: Any, sum_x: np.ndarray | None, count: float, dim: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the anchored moment payload of ``suff_stat`` when it is usable, else ``None``.

    ``None`` falls back to the raw reduced-moment M-step, so a payload is only trusted when it is
    shaped right, finite, non-negative in its second moment, and agrees with the raw first moment it
    claims to describe -- a hand-built :class:`DiagonalGaussianSuffStat` whose payload contradicts
    its tuple must not silently change the estimate the tuple alone would have produced.
    """
    anchored = getattr(suff_stat, "anchored", None)
    if anchored is None or sum_x is None or count <= 0.0 or dim is None:
        return None
    try:
        anchor, a_sum, a_sum2 = anchored
        anchor = np.asarray(anchor, dtype=float).reshape(-1)
        a_sum = np.asarray(a_sum, dtype=float).reshape(-1)
        a_sum2 = np.asarray(a_sum2, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if anchor.shape != (dim,) or a_sum.shape != (dim,) or a_sum2.shape != (dim,):
        return None
    if not (np.isfinite(anchor).all() and np.isfinite(a_sum).all() and np.isfinite(a_sum2).all()):
        return None
    if np.any(a_sum2 < 0.0):
        return None
    implied_sum = a_sum + count * anchor
    tolerance = 1.0e-6 * np.maximum.reduce(
        (np.abs(sum_x), np.abs(count * anchor), np.ones_like(anchor)),
    )
    if np.any(np.abs(implied_sum - sum_x) > tolerance):
        return None
    return anchor, a_sum, a_sum2


def _anchored_mean_offset(
    anchored: tuple[np.ndarray, np.ndarray, np.ndarray],
    count: float,
    pseudo_count: float | None,
    prior_mean: np.ndarray | None,
) -> np.ndarray:
    """The pooled mean MINUS the anchor, computed entirely in offset space.

    Same estimator as ``(sum_x + pc*prior_mu) / (count + pc)`` -- ``a_sum`` is ``sum_x - count*anchor``
    -- but every term is O(count * spread) instead of O(count * |mean|), so the result does not
    inherit the ~5-ulp-of-the-offset error that summing 1e3 values near 1.7e9 puts into ``sum_x``.
    That error is harmless in the mean itself and NOT harmless in the variance: the scatter's
    sensitivity to the mean is second order, so a mean off by ``e`` inflates every variance by
    ``e^2`` -- 5e-12 relative on a unit-scale variance at offset 1.7e9, an order of magnitude above
    what float64 makes unavoidable. Keeping the mean in offset space removes it.
    """
    _, a_sum, _ = anchored
    if pseudo_count not in (None, 0.0) and prior_mean is not None:
        prior_offset = np.asarray(prior_mean, dtype=float) - anchored[0]
        return (a_sum + pseudo_count * prior_offset) / (count + pseudo_count)
    return a_sum / count


def _anchored_pooled_variances(
    anchored: tuple[np.ndarray, np.ndarray, np.ndarray],
    count: float,
    mean_offset: np.ndarray,
    pseudo_count: float | None,
    prior_mean: np.ndarray | None,
    prior_covar: np.ndarray | None,
) -> np.ndarray:
    """:func:`pooled_gaussian_covariance` (diagonal branch) computed from shift-anchored moments.

    Same pooling contract as the raw-moment form, but the observed scatter is expanded about the
    data anchor, so every term is O(count * spread^2) and the result is shift-invariant instead of
    losing ~2*log2(|mean|/sd) bits to cancellation. ``mean_offset`` is the pooled mean relative to
    the anchor (see :func:`_anchored_mean_offset`), never the mean itself -- the whole point is that
    no quantity here carries the data's offset.
    """
    anchor, a_sum, a_sum2 = anchored
    # The scatter is SPLIT rather than accumulated in one array, which is what lets the noise clamp
    # stay off the data (mirrors the univariate ``_anchored_pooled_variance`` split). ``core`` is the
    # scatter about the sample's OWN mean (``a_sum/count``): every term is O(count * spread^2),
    # computed entirely at spread scale, and it carries all of the data. ``count * gap * gap`` is the
    # displacement of the mean actually REPORTED (``mean_offset``) from that sample mean -- genuine
    # under a pseudo-count prior, pure rounding of ``sum_x / count`` at data magnitude on the plain
    # ML path, and the ONLY place the large magnitude enters. Clamping each piece with the noise
    # source that actually applies to it -- instead of one combined array against one combined
    # threshold -- leaves genuine spread at extreme magnitude untouched: the old combined form's
    # ulp-scale threshold had to be crossed by the spread as well as by cancellation noise, so any
    # per-coordinate spread below ~4 eps |mean| per observation read as constant even when
    # ``count * spread^2`` -- the actual scatter -- was orders of magnitude above it.
    centroid_offset = a_sum / count if count > 0.0 else np.zeros_like(a_sum)
    core = a_sum2 - centroid_offset * a_sum
    # Mathematically >= 0; only last-ulp rounding of the two O(count * spread^2) terms can undershoot
    # -- or overshoot: a degenerate coordinate's scatter must come out EXACTLY zero on every
    # algebraically equivalent path, or the scale-relative variance floor reads the +O(eps) residue as
    # a genuine spread and two equivalent fits disagree (the accumulator/reweighted-seq_update
    # invariant catches that). Same clamp, same rationale, as the univariate ``_anchored_pooled_variance``.
    #
    # This bound is RELATIVE to the terms differenced, which is what makes it safe here: both terms
    # are weighted the same way, so it scales with the responsibilities instead of competing with
    # them, and it no longer has to be crossed by the spread. Data whose spread the grid cannot carry
    # lands with ``a_sum2`` and ``centroid_offset * a_sum`` EXACTLY equal (every observation rounded
    # to the same float), so this form reports exactly zero for it on its own, from the data rather
    # than from a threshold.
    noise_scale = np.maximum(np.abs(a_sum2), np.maximum(np.abs(centroid_offset * a_sum), 1.0e-300))
    core = np.where(core < 1.0e-12 * noise_scale, 0.0, core)
    core = np.maximum(core, 0.0)
    # Displacement of the mean actually reported from the sample mean, per coordinate. Below that
    # coordinate's own rounding granularity it is not a displacement at all, just which order the
    # large-magnitude sum was accumulated in, and squaring it would turn that into variance. Zeroing
    # individual coordinates of ``gap`` (rather than the combined scatter) is safe here because
    # ``count * gap * gap`` only ever enters ADDITIVELY and elementwise -- unlike the full covariance's
    # outer product, no coordinate's shift term depends on another's, so there is no symmetry/PSD
    # structure a per-coordinate zero could break.
    gap = mean_offset - centroid_offset
    mean_ulp = _ANCHOR_MEAN_ULP * np.maximum(np.abs(anchor + mean_offset), np.abs(anchor))
    gap = np.where(np.abs(gap) <= mean_ulp, 0.0, gap)
    observed_scatter = core + count * gap * gap
    if pseudo_count not in (None, 0.0) and prior_covar is not None:
        if prior_mean is None:
            offset = 0.0
        else:
            offset = ((np.asarray(prior_mean, dtype=float) - anchor) - mean_offset) ** 2
        prior_scatter = pseudo_count * (prior_covar + offset)
        return (observed_scatter + prior_scatter) / (count + pseudo_count)
    if count == 0.0:
        return observed_scatter
    return observed_scatter / count


def _record_variance_floor(dist: Any, floor: float, raw_covar: np.ndarray) -> Any:
    """Note on ``dist`` when the M-step variance floor lifted a coordinate, so a fit can report it.

    This estimator floored silently: on 10 uV noise recorded in volts it returned variances inflated
    114x with an empty ``numerical_repairs()`` and an empty ``fit_provenance().repairs``, while the
    scalar and full-covariance estimators both recorded the same clamp (T4-6). A repair means the
    parameters are not the ones the data implied, so it has to be visible on every surface that
    applies one. An ordinary fit lifts nothing and records nothing.
    """
    smallest = float(np.min(raw_covar)) if raw_covar.size else 0.0
    if smallest >= floor:
        return dist
    if smallest <= 0.0:
        note = "variance-floored(%.3g; onto a non-positive variance)" % floor
    elif (floor - smallest) <= _FLOOR_MATERIAL_RATIO * smallest:
        return dist
    else:
        note = "variance-floored(%.3g; %.3gx the smallest variance)" % (floor, floor / smallest)
    dist._numerical_repairs = (note,) + tuple(getattr(dist, "_numerical_repairs", ()))
    return dist


class DiagonalGaussianFisherView(FixedFisherView):
    """Fisher view over per-dimension first and second moments for a diagonal Gaussian."""

    def __init__(self, dist: Any) -> None:
        self.dim = int(dist.dim if hasattr(dist, "dim") else len(dist.mu))
        labels = [("sum", str(i)) for i in range(self.dim)]
        labels.extend(("sum2", str(i)) for i in range(self.dim))
        labels.append(("count",))
        super().__init__(dist, labels)

    def _as_matrix(self, data: Any) -> np.ndarray:
        return np.asarray(data, dtype=np.float64).reshape((-1, self.dim))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        x = self._as_matrix(data)
        return np.hstack((x, x * x, np.ones((x.shape[0], 1), dtype=np.float64)))

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        x = enc_data[0] if isinstance(enc_data, tuple) else enc_data
        return self._statistics_from_data(np.asarray(x, dtype=np.float64), estimate=estimate)

    def _model_mean(self) -> np.ndarray:
        mu = np.asarray(self.dist.mu, dtype=np.float64).reshape(-1)
        var = np.asarray(self.dist.covar, dtype=np.float64).reshape(-1)
        return np.concatenate((mu, mu * mu + var, np.asarray([1.0])))

    def _model_fisher(self) -> np.ndarray:
        mu = np.asarray(self.dist.mu, dtype=np.float64).reshape(-1)
        var = np.asarray(self.dist.covar, dtype=np.float64).reshape(-1)
        dim = self.dim
        out = np.zeros((2 * dim + 1, 2 * dim + 1), dtype=np.float64)
        out[:dim, :dim] = np.diag(var)
        diag = 2.0 * mu * var
        out[np.arange(dim), dim + np.arange(dim)] = diag
        out[dim + np.arange(dim), np.arange(dim)] = diag
        out[dim + np.arange(dim), dim + np.arange(dim)] = 2.0 * var * var + 4.0 * mu * mu * var
        return out


class DiagonalGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Multivariate Gaussian distribution with independent components (diagonal covariance matrix)."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for diagonal Gaussian generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the diagonal Gaussian."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="diagonal_gaussian",
            distribution_type=cls,
            parameters=(
                ParameterSpec("mu", constraint="real_vector"),
                ParameterSpec("covar", constraint="positive_vector"),
            ),
            statistics=(
                StatisticSpec("sum", kind="vector_moment"),
                StatisticSpec("sum2", kind="vector_moment"),
                StatisticSpec("count"),
            ),
            support="real_vector",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return vector sufficient statistics for generated diagonal-Gaussian scoring."""
        xx = engine.asarray(x)
        return xx, xx * xx

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return vector natural parameters for generated diagonal-Gaussian scoring."""
        covar = params["covar"]
        return params["mu"] / covar, -0.5 / covar

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the diagonal-Gaussian log partition for generated scoring."""
        mu = params["mu"]
        covar = params["covar"]
        return 0.5 * engine.sum(
            engine.log(engine.asarray(2.0 * np.pi) * covar) + (mu * mu / covar),
            axis=-1,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return row-wise legacy accumulator statistics for generated resident reductions."""
        xx = engine.asarray(x)
        one = engine.sum(xx * 0.0, axis=1) + engine.asarray(1.0)
        return xx, xx * xx, one

    def __init__(
        self,
        mu: Sequence[float] | np.ndarray,
        covar: Sequence[float] | np.ndarray = MISSING,
        name: str | None = None,
        keys: str | None = None,
        covariance: Sequence[float] | np.ndarray = MISSING,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a diagonal Gaussian distribution.

        Args:
            mu: Mean vector.
            covar: Per-coordinate variances. ``covariance`` is accepted as an
                alias.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior (Optional): Conjugate parameter prior over (mu, tau=1/covar). A
                :class:`~mixle.stats.bayes.multivariate_normal_gamma.MultivariateNormalGammaDistribution` enables the
                Bayesian/variational machinery (``expected_log_density`` and the conjugate
                posterior update); ``None`` (default) is a plain point model.

        Attributes:
             dim: Dimension of the Gaussian.
             mu: Mean vector.
             covar: Per-coordinate variances.
             name: Optional diagnostic name.
             log_c: Log-normalization constant.
             ca: Quadratic scoring coefficient of the EXPANDED form (kept for the fused kernels,
                 which read it directly; the Python scoring path no longer uses it -- see
                 :meth:`log_density`).
             cb: Linear scoring coefficient of the expanded form (as ``ca``).
             cc: Constant scoring coefficient of the expanded form (as ``ca``).
             keys: Optional sufficient-statistic key.

        """
        covar = coalesce_alias("covar", covar, "covariance", covariance, default=MISSING)
        self.mu = vector_parameter(mu, label="diagonal Gaussian mean")
        self.dim = len(self.mu)
        self.covar = vector_parameter(
            covar,
            label="diagonal Gaussian covariance",
            dim=self.dim,
            positive=True,
        )
        self.name = name
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())

        # ``ca``/``cb``/``cc`` are the EXPANDED quadratic form's coefficients
        # (``x^2*ca + x*cb + cc``). They are retained because
        # :mod:`mixle.stats.compute.fused_kernels` stacks them directly off the distribution, but the
        # Python scoring path uses the centered form instead: see :meth:`log_density` for why the
        # expanded form cannot be used on offset data. The centered path derives its per-coordinate
        # precision per call rather than caching it here, because this class's serialized state is
        # its whole ``__dict__`` and the codec requires the decoded field set to match the artifact
        # exactly -- a new cached field would refuse every artifact written before it.
        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c
        for parameter in (self.mu, self.covar, self.ca, self.cb):
            parameter.setflags(write=False)
        self.keys = keys

        self.set_prior(prior)

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a parameter prior and precompute conjugate-prior expectations.

        With a MultivariateNormalGamma(mu0, lam, a, b) prior over (mu, tau=1/covar) this
        caches the expected natural parameters [ea, eb, e1, e2] with e1 = E[mu*tau] and
        e2 = -0.5*E[tau] per component (ea, eb scalars summed over components), so that
        ``expected_log_density(x) = x.e1 + (x*x).e2 - ea + eb``. Any other prior
        (including ``None``) leaves the distribution a plain point model.
        """
        self.prior = prior

        if isinstance(prior, MultivariateNormalGammaDistribution):
            mu, lam, a, b = prior.get_parameters()
            if any(parameter.shape != (self.dim,) for parameter in (mu, lam, a, b)):
                raise ValueError("diagonal Gaussian conjugate prior dimension must match the distribution")

            ea = np.sum((mu * mu) * (a / b) * 0.5 + (0.5 / lam) + 0.5 * (np.log(b) - digamma(a)))
            e1 = mu * a / b
            e2 = -0.5 * a / b
            eb = -0.5 * np.log(2 * np.pi) * self.dim

            self.conj_prior_params = [mu, lam, a, b]
            self.expected_nparams = [ea, eb, e1, e2]
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def expected_log_density(self, x) -> float:
        """Variational expectation E_q[log p(x | mu, tau)] under the prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return np.dot(x, e1) + np.dot(np.power(x, 2), e2) - ea + eb
        return self.log_density(x)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return np.dot(x, e1) + np.dot(x * x, e2) - ea + eb
        return self.seq_log_density(x)

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = repr(list(self.mu.flatten()))
        s2 = repr(list(self.covar.flatten()))
        s3 = repr(self.name)
        return "DiagonalGaussianDistribution(%s, %s, name=%s)" % (s1, s2, s3)

    def density(self, x: Sequence[float] | np.ndarray):
        """Evaluate the density at observation x.

        See log_density() for details.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-dim observation vector.

        Returns:
            Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: Sequence[float] | np.ndarray):
        """Evaluate the log-density at observation x.

        The log-density is given by

            log(p(x)) = -0.5*sum_{i=1}^{n} (x_i-m_i)^2 / s2_i - 0.5*log(s2_i) - (n/2)*log(2*pi).

        Computed in the CENTERED form the formula above states, ``-0.5*sum (x-mu)^2/covar``, not in
        the algebraically equal expanded form ``x^2*ca + x*cb + cc``. The expanded form is three
        terms of size ``O(|x|^2/covar)`` summing to an ``O(1)`` answer, so at ``|x| ~ 1e9`` with unit
        variance it cancels away every significant digit: the scorer returned round-number garbage
        (measured: log-likelihoods of exactly 66560.0 and -135168.0 on unit-variance data at offset
        1.7e9), and because EM selects the model with the best objective, an ``optimize``/``fit``
        run at that offset returned the WORSE of its own iterates. ``x - mu`` is exact for operands
        of the same magnitude, so the centered form is accurate at any offset; it is also one gemv
        instead of two. The full-covariance estimator's scorer always used the centered form -- this
        is the diagonal one catching up, and it makes the documented formula true (T1-F1).
        The expanded coefficients remain available as ``ca``/``cb``/``cc`` for the fused kernels.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-dim observation vector.

        Returns:
            Log-density at x.

        """
        checked = vector_event(x, self.dim, label="diagonal Gaussian observation")
        diff = checked - self.mu
        return np.dot(diff * diff, self.ca) + self.log_c

    def condition(self, observed: dict[int, float]) -> "DiagonalGaussianDistribution":
        """Return the conditional over the unobserved dimensions given ``observed``.

        A diagonal Gaussian has independent coordinates, so conditioning on some of them leaves the rest
        unchanged: the result is just ``DiagonalGaussian(mu[unobserved], covar[unobserved])`` (the
        observed values do not shift the unobserved mean or variance). Provided so diagonal-covariance
        Gaussian mixtures support :meth:`MixtureDistribution.conditional` -- there the *responsibilities*
        still update from how well each component explains the observed coordinates, even though the
        within-component coordinates are independent. Raises if no dimension is left unobserved.
        """
        if observed and (min(observed) < 0 or max(observed) >= self.dim):
            raise ValueError("observed indices must be in [0, dim)")
        unobs = np.array([i for i in range(self.dim) if i not in observed], dtype=int)
        if unobs.size == 0:
            raise ValueError("at least one dimension must be left unobserved")
        prior = self._coordinate_prior(unobs)
        return DiagonalGaussianDistribution(
            self.mu[unobs],
            self.covar[unobs],
            name=self.name,
            keys=self.keys,
            prior=prior,
        )

    def marginal(self, keep: Sequence[int]) -> "DiagonalGaussianDistribution":
        """Return the marginal over the dimensions ``keep``: ``DiagonalGaussian(mu[keep], covar[keep])``.

        Marginalizing a diagonal Gaussian simply drops the other independent coordinates (order kept).
        """
        idx = marginal_indices(keep, self.dim)
        prior = self._coordinate_prior(idx)
        return DiagonalGaussianDistribution(
            self.mu[idx],
            self.covar[idx],
            name=self.name,
            keys=self.keys,
            prior=prior,
        )

    def _coordinate_prior(
        self,
        indices: np.ndarray,
    ) -> SequenceEncodableProbabilityDistribution | None:
        """Restrict an independent conjugate prior to selected coordinates."""
        if not isinstance(self.prior, MultivariateNormalGammaDistribution):
            if len(indices) == self.dim and np.array_equal(indices, np.arange(self.dim)):
                return self.prior
            return None
        mu, lam, a, b = self.prior.get_parameters()
        return MultivariateNormalGammaDistribution(
            mu[indices],
            lam[indices],
            a[indices],
            b[indices],
            name=self.prior.name,
            prior=self.prior.prior,
        )

    def density_cumulative(self, x: Sequence[float] | np.ndarray) -> float:
        """Exact probability-ordered cumulative ``G(x) = P(p(Y) >= p(x))`` -- the highest-density-region
        mass through ``x`` (multivariate analogue of a CDF). For a diagonal Gaussian the squared
        Mahalanobis distance ``sum_i (x_i-mu_i)^2/var_i`` is chi-square(dim), so ``G = chi2.cdf(maha2, dim)``.
        Used by :func:`mixle.enumeration.density_rank.density_rank` to return an EXACT cumulative.
        """
        from scipy.stats import chi2

        diff = np.asarray(x, dtype=float) - self.mu
        maha2 = float(np.sum(diff * diff / self.covar))
        return float(chi2.cdf(maha2, df=self.dim))

    def density_quantile(self, q: float) -> np.ndarray:
        """Inverse of :meth:`density_cumulative`: a representative point at cumulative-density index ``q``.

        ``q`` is the highest-density-region mass, whose boundary is the squared-Mahalanobis level
        ``chi2.ppf(q, dim)``; a representative point on that contour offsets the first coordinate by
        ``sqrt(level * var_0)`` (Mahalanobis distance exactly the level). Sweeping ``q`` enumerates the
        support in descending density.
        """
        from scipy.stats import chi2

        qf = float(q)
        if not 0.0 <= qf <= 1.0:
            raise ValueError("q must be in [0, 1].")
        level = float(chi2.ppf(qf, df=self.dim))
        point = self.mu.copy()
        point[0] = point[0] + float(np.sqrt(level * self.covar[0]))
        return point

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of the log-density at a sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, dim) from
                DiagonalGaussianDataEncoder.seq_encode().

        Returns:
            Numpy array of length sz containing the log-density of each encoded observation.

        """
        # Centered, not expanded -- see log_density for why the expanded form is unusable on offset
        # data. ``ca`` is exactly the ``-0.5/covar`` the centered form contracts against, so this is
        # ONE gemv where the expanded form needed two, and it caches nothing new.
        checked = vector_batch(x, self.dim, label="diagonal Gaussian observations")
        diff = checked - self.mu
        rv = np.dot(diff * diff, self.ca)
        rv += self.log_c
        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, covar: Any, engine: Any) -> Any:
        """Engine-neutral diagonal Gaussian log-density from explicit parameters."""
        dim = engine.asarray(float(tuple(getattr(covar, "shape", (len(covar),)))[-1]))
        log_c = -0.5 * (engine.log(engine.asarray(2.0 * np.pi)) * dim + engine.sum(engine.log(covar), axis=-1))
        return log_c - 0.5 * engine.sum((x - mu) * (x - mu) / covar, axis=-1)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(owned_backend_parameter(self.mu)),
            engine.asarray(owned_backend_parameter(self.covar)),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["DiagonalGaussianDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked diagonal-Gaussian parameters for a homogeneous mixture kernel."""
        dim = dists[0].dim
        if any(d.dim != dim for d in dists):
            raise ValueError("Stacked DiagonalGaussianDistribution components require a shared dimension.")
        return {
            "mu": engine.asarray(np.stack([d.mu for d in dists], axis=0)),
            "covar": engine.asarray(np.stack([d.covar for d in dists], axis=0)),
            "dim": engine.asarray(float(dim)),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of diagonal-Gaussian log densities."""
        xx = engine.asarray(x)
        mu = params["mu"]
        covar = params["covar"]
        log_c = -0.5 * (engine.log(engine.asarray(2.0 * np.pi)) * params["dim"] + engine.sum(engine.log(covar), axis=1))
        quad = engine.sum(
            (xx[:, None, :] - mu[None, :, :]) * (xx[:, None, :] - mu[None, :, :]) / covar[None, :, :], axis=2
        )
        return log_c[None, :] - 0.5 * quad

    def to_fisher(self, **kwargs):
        """Return this distribution's own Fisher view."""
        return DiagonalGaussianFisherView(self)

    def sampler(self, seed: int | None = None) -> "DiagonalGaussianSampler":
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional seed for the sampler's random state.

        Returns:
            A configured ``DiagonalGaussianSampler``.

        """
        return DiagonalGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "DiagonalGaussianEstimator":
        """Return an estimator initialized from this distribution's shape.

        Args:
            pseudo_count: Optional smoothing count applied to mean and variance
                estimates.

        Returns:
            A ``DiagonalGaussianEstimator``.

        """
        if pseudo_count is None:
            return DiagonalGaussianEstimator(name=self.name, keys=self.keys, prior=self.prior)
        else:
            return DiagonalGaussianEstimator(
                pseudo_count=(pseudo_count, pseudo_count),
                suff_stat=(self.mu, self.covar),
                name=self.name,
                keys=self.keys,
                prior=self.prior,
            )

    def dist_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Return an encoder for iid diagonal Gaussian observations."""
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianSampler(DistributionSampler):
    """Sampler for iid diagonal Gaussian observations."""

    def __init__(self, dist: DiagonalGaussianDistribution, seed: int | None = None) -> None:
        """Create a sampler bound to ``dist``.

        Args:
            dist: Distribution to sample from.
            seed: Optional seed for the sampler's random state.

        Attributes:
            dist: Distribution being sampled.
            rng: Random state used for draws.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> Sequence[np.ndarray] | np.ndarray:
        """Draw iid samples from the diagonal Gaussian distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, a single sample is drawn.

        Returns:
            Numpy array with shape (dim,) if size is None, else a list of 'size' such arrays.

        """
        if size is None:
            rv = self.rng.randn(self.dist.dim)
            rv *= np.sqrt(self.dist.covar)
            rv += self.dist.mu
            return rv
        # Vectorized: randn(size, dim) fills row-major, so row i equals the i-th per-draw randn(dim);
        # bit-identical to the loop, far faster.
        rv = self.rng.randn(int(size), self.dist.dim) * np.sqrt(self.dist.covar) + self.dist.mu
        return list(rv)


class DiagonalGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for diagonal Gaussian sufficient statistics.

    Alongside the declared raw moments ``(sum, sum2, count)`` this keeps a SHIFT-ANCHORED moment
    track, for the reason the univariate :class:`~mixle.stats.univariate.continuous.gaussian.
    GaussianAccumulator` keeps one: the variance computed from raw reduced moments is the classic
    cancellation-prone ``E[x^2] - mu^2`` form, which loses ~2*log2(|mean|/sd) bits, so unit-spread
    data at offset 1e7 fits variances tens of percent wrong and data at offset 1.7e9 fits variances
    thousands of times too large (or collapsed onto the floor -- the sign of the garbage is
    data-dependent), on data whose variance a two-pass computation returns exactly. Epoch seconds
    are ~1.7e9. Anchoring at the first value seen keeps every term of the scatter
    O(count * spread^2), making the M-step variances shift-invariant. The track is
    CONDITIONING-GATED (see :func:`_needs_anchor`): a chunk the raw form handles to ~1e-9 relative
    error accumulates exactly the historical single-pass way -- bit-identical statistics, no second
    pass -- and the anchor activates only when a chunk (or a scalar ``update``) would corrupt a
    variance.
    """

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        """Create an accumulator for weighted first and second moments.

        Args:
            dim: Optional Gaussian dimension. Inferred from data when omitted.
            keys: Optional key for merging sufficient statistics.

        Attributes:
             dim: Gaussian dimension.
             count: Sum of observation weights.
             sum: Weighted sum of observation vectors.
             sum2: Weighted sum of squared observation vectors.
             keys: Optional sufficient-statistic key.

        """
        self.dim = vector_dimension(dim, label="diagonal Gaussian dimension", allow_none=True)
        self.count = 0.0
        self.sum = vec.zeros(dim) if dim is not None else None
        self.sum2 = vec.zeros(dim) if dim is not None else None
        self.keys = keys

        # Shift-anchored moments, kept alongside the raw (sum, sum2) when the data needs them; see
        # the class docstring. ``None`` means the track has never activated, which is the ordinary
        # well-conditioned case and the state in which ``value()`` returns the historical plain tuple.
        self._anchor: np.ndarray | None = None
        self._anchored_sum: np.ndarray | None = None
        self._anchored_sum2: np.ndarray | None = None

    def _activate_anchor(self, anchor: np.ndarray) -> None:
        """Start the shift-anchored moment track at ``anchor``.

        Any content already accumulated raw-only is converted about the new anchor. The conversion
        is the cancellation-prone form, but it is only ever applied to content that accumulated
        WITHOUT activating the gate -- i.e. content the gate certified as well-conditioned -- or to
        pre-existing raw statistics restored through ``from_value``/``combine``, where the
        conversion is no less accurate than the raw-only estimate those statistics supported before.
        """
        a = np.asarray(anchor, dtype=float).reshape(-1).copy()
        self._anchor = a
        self._anchored_sum = vec.zeros(self.dim)
        self._anchored_sum2 = vec.zeros(self.dim)
        if self.sum is not None and (self.count != 0.0 or np.any(self.sum != 0.0) or np.any(self.sum2 != 0.0)):
            self._anchored_sum += self.sum - a * self.count
            self._anchored_sum2 += np.maximum(
                self.sum2 - 2.0 * a * self.sum + a * a * self.count,
                0.0,
            )

    def update(
        self, x: Sequence[float] | np.ndarray, weight: float, estimate: DiagonalGaussianDistribution | None
    ) -> None:
        """Update sufficient statistics with a single weighted observation.

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-dim observation vector.
            weight (float): Weight for the observation.
            estimate (Optional[DiagonalGaussianDistribution]): Kept for consistency with
                SequenceEncodableStatisticAccumulator (not used).

        Returns:
            None.

        """
        if self.dim is None:
            checked = vector_parameter(x, label="diagonal Gaussian observation")
            self.dim = len(checked)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)
        else:
            checked = vector_event(x, self.dim, label="diagonal Gaussian observation")
        # ``estimate`` is unused by this accumulator (see the Args note), so the only wiring
        # mistake worth refusing is a DIMENSION mismatch. The former isinstance guard also refused
        # an estimate the library itself produces: ``GaussianMixtureEstimator`` repacks every
        # fitted component -- diagonal ones included -- as a ``MultivariateGaussianDistribution``,
        # so diagonal components inside a Gaussian mixture died here on EM iteration two.
        estimate_dim = getattr(estimate, "dim", None)
        if estimate is not None and estimate_dim is not None and estimate_dim != self.dim:
            raise ValueError(
                "diagonal Gaussian accumulator estimate must have the configured dimension "
                "(accumulator dim %d, estimate dim %r)" % (self.dim, estimate_dim)
            )
        checked_weight = observation_weight(weight, label="diagonal Gaussian observation weight")
        # Scalar updates carry no chunk to assess conditioning from, so the anchor activates on the
        # first observation (an O(d) bookkeeping track on a path that is already O(d)). Activation
        # happens BEFORE the raw fold so any pre-anchor content is converted from statistics the
        # conditioning gate has already vouched for.
        if self._anchor is None:
            self._activate_anchor(checked)
        dx = checked - self._anchor
        self._anchored_sum += dx * checked_weight
        self._anchored_sum2 += dx * dx * checked_weight
        x_weight = checked * checked_weight
        self.count += checked_weight
        self.sum += x_weight
        x_weight *= checked
        self.sum2 += x_weight

    def initialize(self, x: Sequence[float] | np.ndarray, weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with a weighted observation. Calls update().

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-dim observation vector.
            weight (float): Weight for the observation.
            rng (RandomState): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: DiagonalGaussianDistribution | None) -> None:
        """Vectorized update of sufficient statistics with an encoded sequence of observations.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, dim).
            weights (np.ndarray): Numpy array of sz observation weights.
            estimate (Optional[DiagonalGaussianDistribution]): Kept for consistency (not used).

        Returns:
            None.

        """
        if self.dim is None:
            raw = np.asarray(x, dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] == 0:
                raise ValueError("diagonal Gaussian observations must have exact shape (N, D) with D > 0")
            self.dim = raw.shape[1]
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)
        checked = vector_batch(x, self.dim, label="diagonal Gaussian observations")
        checked_weights = observation_weights(
            weights,
            len(checked),
            label="diagonal Gaussian observation weights",
        )
        # Dimension-only check, as in ``update`` above: ``estimate`` is unused here, and the former
        # isinstance guard refused the ``MultivariateGaussianDistribution`` components that
        # ``GaussianMixtureEstimator``'s repack legitimately hands back to diagonal accumulators.
        estimate_dim = getattr(estimate, "dim", None)
        if estimate is not None and estimate_dim is not None and estimate_dim != self.dim:
            raise ValueError(
                "diagonal Gaussian accumulator estimate must have the configured dimension "
                "(accumulator dim %d, estimate dim %r)" % (self.dim, estimate_dim)
            )
        x_weight = np.multiply(checked.T, checked_weights)
        w_sum = float(checked_weights.sum())
        chunk_sum = x_weight.sum(axis=1)
        x_weight *= checked.T
        chunk_sum2 = x_weight.sum(axis=1)
        # Conditioning gate: activate the anchored track only when this chunk's raw moments would
        # corrupt a variance (or the anchor is already live). BEFORE the raw fold, so activation
        # converts only pre-chunk content -- content the gate has already passed as well-conditioned.
        # The chunk's own moments are the ones just computed for the fold, so gating costs a
        # d-length test, not a second pass over the data.
        if len(checked) > 0 and (
            self._anchor is not None or (w_sum > 0.0 and _needs_anchor(chunk_sum, chunk_sum2, w_sum))
        ):
            if self._anchor is None:
                self._activate_anchor(checked[0])
            dx = checked - self._anchor
            wdx = dx * checked_weights[:, None]
            self._anchored_sum += wdx.sum(axis=0)
            self._anchored_sum2 += (wdx * dx).sum(axis=0)
        self.count += w_sum
        self.sum += chunk_sum
        self.sum2 += chunk_sum2

    def seq_initialize(self, x, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of the accumulator. Calls seq_update().

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, dim).
            weights (np.ndarray): Numpy array of sz observation weights.
            rng (RandomState): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> "DiagonalGaussianAccumulator":
        """Merge sufficient statistics into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of squared observations, sum of weights).

        Returns:
            This accumulator.

        """
        sum_x, sum_xx, count, inferred_dim = gaussian_moments(
            suff_stat,
            self.dim,
            diagonal=True,
        )
        if sum_x is None:
            return self
        if self.sum is None:
            # copy on adopt: value() hands out the LIVE arrays, so adopting the caller's reference
            # makes every later in-place += here mutate the DONOR accumulator too (chunk combines
            # and keyed pooling both hit this -- caught by the keyed-protocol sweep). Zeroing and
            # folding below is that copy, and it also gives the anchored merge a live dim to work in.
            self.dim = inferred_dim
            self.sum = vec.zeros(inferred_dim)
            self.sum2 = vec.zeros(inferred_dim)
            self.count = 0.0
        anchored = _consistent_anchored_moments(suff_stat, sum_x, count, inferred_dim)
        if anchored is not None:
            # Chan's parallel-merge: re-express the incoming anchored moments about this
            # accumulator's anchor. The anchor gap is between two data values, so every term stays
            # O(count * spread^2) -- no large-offset cancellation is reintroduced. Activation (when
            # this side has no anchor yet) runs BEFORE the raw fold below so it converts only this
            # side's pre-existing content.
            b_anchor, b_asum, b_asum2 = anchored
            if self._anchor is None:
                self._activate_anchor(b_anchor)
            gap = b_anchor - self._anchor
            self._anchored_sum += b_asum + count * gap
            self._anchored_sum2 += b_asum2 + 2.0 * gap * b_asum + count * gap * gap
        elif self._anchor is not None and (count != 0.0 or np.any(sum_x != 0.0) or np.any(sum_xx != 0.0)):
            # Raw-only statistics (an engine kernel, a hand-built tuple, a gate-passing peer)
            # joining an anchored pool: convert about our anchor. See _activate_anchor for why the
            # cancellation-prone conversion is acceptable exactly here.
            a = self._anchor
            self._anchored_sum += sum_x - a * count
            self._anchored_sum2 += np.maximum(sum_xx - 2.0 * a * sum_x + a * a * count, 0.0)

        self.sum += sum_x
        self.sum2 += sum_xx
        self.count += count

        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(sum, sum_squares, count)`` sufficient statistics.

        Once the shift-anchored moment track is live the returned value is a
        :class:`DiagonalGaussianSuffStat` -- a drop-in 3-tuple (indexing/unpacking/iteration all
        behave identically) that additionally carries those moments in its ``anchored`` attribute,
        so :meth:`combine` can fold them in and :meth:`DiagonalGaussianEstimator.estimate` can
        compute shift-invariant variances. Well-conditioned data never activates the track and gets
        the historical plain tuple.
        """
        sum_copy = None if self.sum is None else self.sum.copy()
        sum2_copy = None if self.sum2 is None else self.sum2.copy()
        if self._anchor is None:
            return (sum_copy, sum2_copy, self.count)
        return DiagonalGaussianSuffStat(
            sum_copy,
            sum2_copy,
            self.count,
            anchored=(self._anchor.copy(), self._anchored_sum.copy(), self._anchored_sum2.copy()),
        )

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "DiagonalGaussianAccumulator":
        """Replace this accumulator's sufficient statistics.

        Args:
            x (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of squared observations, sum of weights).

        Returns:
            This accumulator.

        """
        self.sum, self.sum2, self.count, self.dim = gaussian_moments(
            x,
            self.dim,
            diagonal=True,
        )
        anchored = _consistent_anchored_moments(x, self.sum, self.count, self.dim)
        if anchored is not None:
            self._anchor, self._anchored_sum, self._anchored_sum2 = (
                anchored[0].copy(),
                anchored[1].copy(),
                anchored[2].copy(),
            )
        else:
            # Raw-only statistics replace the state: the anchored track restarts unactivated, and a
            # later activation (first update / anchored merge) converts this content then.
            self._anchor = None
            self._anchored_sum = None
            self._anchored_sum2 = None
        return self

    def scale(self, c: float) -> "DiagonalGaussianAccumulator":
        """Scale the accumulated statistics in-place by ``c``, anchored track included.

        The structural default routes through ``from_value(scale_suff_stat(self.value(), c))``,
        which rebuilds from a PLAIN tuple and would therefore drop the anchored payload -- turning a
        scaled large-offset accumulator back into a cancellation-prone one. Scaling every weight by
        ``c`` scales both anchored moments by ``c`` and leaves the anchor (a data value) alone.
        """
        factor = float(c)
        if self.sum is not None:
            self.sum = self.sum * factor
            self.sum2 = self.sum2 * factor
            self.count = self.count * factor
        if self._anchor is not None:
            self._anchored_sum = self._anchored_sum * factor
            self._anchored_sum2 = self._anchored_sum2 * factor
        return self

    def acc_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Return an encoder compatible with this accumulator's dimension."""
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for diagonal Gaussian accumulators."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        """Create an accumulator factory.

        Args:
            dim: Optional Gaussian dimension.
            keys: Optional key for merging sufficient statistics.

        Attributes:
             dim: Optional Gaussian dimension.
             keys: Optional sufficient-statistic key.

        """
        self.dim = vector_dimension(dim, label="diagonal Gaussian dimension", allow_none=True)
        self.keys = keys

    def make(self) -> "DiagonalGaussianAccumulator":
        """Return a fresh accumulator with the factory configuration."""
        return DiagonalGaussianAccumulator(dim=self.dim, keys=self.keys)


class DiagonalGaussianEstimator(ParameterEstimator):
    """Estimator for diagonal Gaussian distributions.

    The fit is SHIFT-EQUIVARIANT: ``fit(x + c)`` returns the variances of ``x + c`` to within a few
    ulps for any constant ``c`` the data can carry, epoch seconds (~1.7e9) included. That is not
    free from the declared ``(sum, sum2, count)`` statistics -- their variance is the
    cancellation-prone ``E[x^2] - mu^2``, which at offset 1e7 was tens of percent wrong and at 1.7e9
    thousands-fold wrong or collapsed onto the floor -- so
    :class:`DiagonalGaussianAccumulator` carries a conditioning-gated shift-anchored moment track
    and this estimator reads it. Statistics that arrive already reduced and WITHOUT that track (an
    engine kernel's stacked moments, a hand-built tuple) cannot be corrected here and take the
    historical path; when they are too ill-conditioned for it, ``estimate`` warns rather than
    returning variances it cannot stand behind.
    """

    def __init__(
        self,
        dim: int | None = None,
        pseudo_count: float | tuple[float | None, float | None] = (None, None),
        suff_stat: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
        min_covar: float | None = None,
        ridge: float | None = None,
    ) -> None:
        """Create an estimator for weighted diagonal Gaussian statistics.

        Args:
            dim: Optional Gaussian dimension.
            pseudo_count: Optional smoothing counts for mean and variance. A scalar is
                broadcast to both slots.
            suff_stat: Optional prior mean and variance used for smoothing.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior (Optional): Conjugate MultivariateNormalGamma prior over (mu, tau=1/covar). When present,
                ``estimate`` performs the closed-form per-component conjugate posterior update (returning the
                joint MAP estimate and carrying the posterior forward as the fitted model's prior) instead
                of the maximum-likelihood / pseudo-count update.
            min_covar (Optional[float]): Absolute per-coordinate variance floor applied in the MLE M-step.
                ``None`` (default) uses a tiny ``1e-8`` as a FALLBACK only -- it applies when the
                relative floor has no scale to work from (every coordinate variance non-positive, or
                ``ridge=0.0``). Passing a value explicitly restores it as a hard lower bound, for
                callers that want a fixed regularizer. Invalid/non-finite reduced statistics are
                rejected rather than treated as numerical noise.
            ridge (Optional[float]): Relative variance floor coefficient. ``None`` (default) uses ``1e-6``;
                each coordinate variance is floored at ``ridge * mean(var)`` so the safeguard is
                data-scaled and the fit is equivariant under a change of units. Bias is negligible at
                the defaults when the coordinates share a scale; because the floor is set by the MEAN
                variance, heterogeneous-unit columns can still have their smallest variances inflated
                materially, and a floor that lifts a coordinate is recorded on the fitted distribution
                as a ``variance-floored(...)`` entry in ``numerical_repairs()``.

        Attributes:
            name: Optional diagnostic name.
            dim: Gaussian dimension.
            prior_mu: Prior mean used for smoothing.
            prior_covar: Prior variance used for smoothing.
            pseudo_count: Smoothing counts for mean and variance.
            keys: Optional sufficient-statistic key.

        """
        self.name = name
        self.prior_mu, self.prior_covar, self.dim = gaussian_prior_statistics(
            suff_stat,
            dim,
            diagonal=True,
        )
        self.pseudo_count = pseudo_counts(
            pseudo_count,
            label="diagonal Gaussian pseudo-count",
        )
        require_pseudo_moments(self.pseudo_count, self.prior_mu, self.prior_covar)
        self.keys = keys
        self.prior = prior
        self.has_conj_prior = isinstance(prior, MultivariateNormalGammaDistribution)
        if self.has_conj_prior:
            prior_dim = len(prior.get_parameters()[0])
            if self.dim is None:
                self.dim = prior_dim
            elif self.dim != prior_dim:
                raise ValueError("diagonal Gaussian estimator prior dimension must match its configured dimension")
        self.min_covar = finite_scalar(
            1.0e-8 if min_covar is None else min_covar,
            label="diagonal Gaussian min_covar",
            positive=True,
        )
        self.ridge = finite_scalar(
            1.0e-6 if ridge is None else ridge,
            label="diagonal Gaussian ridge",
            nonnegative=True,
        )
        # An explicitly requested floor is absolute and keeps winning the max() in
        # ``_variance_floor``; the default is only the fallback for data carrying no scale of its own.
        self._absolute_min_covar = min_covar is not None

    def _variance_floor(self, covar: np.ndarray) -> float:
        """Effective per-coordinate variance floor for the raw variances ``covar``.

        Left as ``max(min_covar, ridge * mean(var))`` the absolute term also won on data whose whole
        scale sits below it -- 10 uV of noise recorded in volts -- so the same data in volts and in
        microvolts fitted different variances (T4-6). By default the absolute floor is now the
        fallback for variances with no scale at all; an explicitly configured ``min_covar`` is still a
        hard lower bound, which is how ``mixle.task`` regularizes its diagonal Gaussians.
        """
        positive = covar[covar > 0.0]
        relative = self.ridge * float(np.mean(positive)) if positive.size else 0.0
        if self._absolute_min_covar:
            return max(self.min_covar, relative)
        return relative if relative > 0.0 else self.min_covar

    @staticmethod
    def _warn_if_uncorrectable(sum_x: np.ndarray, sum_xx: np.ndarray, count: float) -> None:
        """Warn when raw-only statistics are too ill-conditioned for the variances they imply.

        The anchored track fixes the estimator's OWN accumulation. Statistics that arrive already
        reduced and without an anchor -- an engine/GPU kernel's stacked moments, a hand-built tuple,
        a legacy artifact -- cannot be corrected here: the information cancellation destroyed is not
        in them any more. Before this, that case returned variances thousands of times too large (or
        collapsed onto the floor) with an empty ``numerical_repairs()`` (T1-F1). It is now named.

        Deliberately NOT a raise: these statistics are the declared exchange format, the raw M-step
        is what the library has always done with them, and a fit that is imprecise is not a fit that
        must be refused. Deliberately NOT the full ``_needs_anchor`` gate either -- that gate also
        fires on a non-positive computed spread, which is the ordinary degenerate/single-point EM
        component the variance floor exists for and already discloses. Only the genuine large-offset
        cancellation regime warns.
        """
        if count <= 0.0 or sum_x is None or sum_xx is None:
            return
        m = sum_x / count
        spread2 = sum_xx / count - m * m
        risky = (spread2 > 0.0) & (m * m > _ANCHOR_CONDITION_RATIO * spread2)
        if not np.any(risky):
            return
        import warnings

        ratio = np.where(risky, m * m / np.where(spread2 > 0.0, spread2, 1.0), -np.inf)
        worst = int(np.argmax(ratio))
        warnings.warn(
            "diagonal Gaussian sufficient statistics arrived without shift-anchored moments and are "
            "too ill-conditioned for the raw E[x^2] - mu^2 variance: coordinate %d has "
            "mean^2/variance %.3g, so the fitted variance loses roughly %.0f%% of its significant "
            "digits to cancellation. Accumulate through DiagonalGaussianAccumulator (which anchors "
            "automatically), or center the data before fitting."
            % (
                worst,
                float(ratio[worst]),
                min(100.0, 100.0 * np.log10(float(ratio[worst])) / 16.0),
            ),
            RuntimeWarning,
            stacklevel=3,
        )

    def accumulator_factory(self) -> "DiagonalGaussianAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        return DiagonalGaussianAccumulatorFactory(dim=self.dim, keys=self.keys)

    def model_log_density(self, model: "DiagonalGaussianDistribution") -> float:
        """Log-density of the model parameters under the MultivariateNormalGamma prior (ELBO global term).

        The prior is over (mu, tau=1/covar), so the model's covariance is inverted before scoring.
        """
        if self.has_conj_prior:
            return float(self.prior.log_density((model.mu, 1.0 / model.covar)))
        return 0.0

    def _estimate_conjugate(
        self,
        suff_stat: tuple[np.ndarray, np.ndarray, float],
        anchored: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> "DiagonalGaussianDistribution":
        """Closed-form per-component NormalGamma conjugate posterior update returning the joint MAP estimate."""
        sum_x, sum_xx, nobs_loc1 = suff_stat
        sum_xxx = sum_x
        nobs_loc2 = nobs_loc1

        old_mu, old_lam, old_a, old_b = self.prior.get_parameters()

        new_n = old_lam + nobs_loc1
        new_a = old_a + (nobs_loc2 / 2.0)

        # With shift-anchored moments available, every mean here is recomputed in offset space --
        # ``a_sum`` is ``sum_x - count*anchor``, so the same estimator without any term carrying the
        # data's offset. ``sample_mean1`` enters the posterior scale QUADRATICALLY through
        # ``new_b1`` below, so the ~5-ulp-of-the-offset error in ``sum_x`` would otherwise land in
        # the fitted variance at ~1e-9 relative even after the scatter itself was repaired.
        anchored_mean = None if anchored is None or nobs_loc1 <= 0 else anchored[0] + anchored[1] / nobs_loc1

        if anchored_mean is not None:
            sample_mean1 = anchored_mean
        elif nobs_loc1 > 0:
            sample_mean1 = sum_x / nobs_loc1
        else:
            sample_mean1 = 0

        if anchored_mean is not None:
            sample_mean2 = anchored_mean
        elif nobs_loc2 > 0:
            sample_mean2 = sum_xxx / nobs_loc2
        else:
            sample_mean2 = 0

        if anchored_mean is not None:
            new_mu = anchored[0] + (anchored[1] + old_lam * (old_mu - anchored[0])) / (old_lam + nobs_loc1)
        else:
            new_mu = (sum_x + old_mu * old_lam) / (old_lam + nobs_loc1)

        # Per-coordinate scatter ``sum_xx - (sum_x)^2/n`` is cancellation-prone (see GaussianEstimator):
        # floor it at 0 so a near-constant coordinate cannot drive ``new_b``/variance negative, matching
        # the MLE path and the scalar/full-covariance conjugate estimators. (The diagonal Gaussian's
        # constructor now validates covar > 0; the final min_covar floor below is what actually backstops
        # against a ValueError there, but flooring the scatter here too keeps intermediate quantities
        # well-defined rather than relying solely on that backstop.)
        # When the accumulator's shift-anchored moments are available, use the same scatter expanded
        # about the anchor instead: mathematically the same quantity, but shift-invariant.
        if anchored is not None and nobs_loc2 > 0:
            # The centroid offset IS the sample mean relative to the anchor, so this is the scatter
            # about ``sample_mean2`` with no offset-carrying term anywhere in it.
            new_b0 = nobs_loc2 * _anchored_pooled_variances(
                anchored, nobs_loc2, _anchored_mean_offset(anchored, nobs_loc2, None, None), None, None, None
            )
        else:
            new_b0 = np.maximum(sum_xx - sample_mean2 * sum_xxx, 0.0)
        new_b1 = (old_lam * nobs_loc1 / new_n) * np.power(sample_mean1 - old_mu, 2)
        new_b = old_b + 0.5 * (new_b0 + new_b1)

        denom = new_a - 0.5  # per-coordinate array
        safe_denom = np.where(denom > 0.0, denom, 1.0)
        raw_sigma2 = np.where(denom > 0.0, new_b / safe_denom, self.min_covar)
        floor = self._variance_floor(raw_sigma2)  # match the MLE-path variance floor

        new_prior = MultivariateNormalGammaDistribution(new_mu, new_n, new_a, new_b)
        dist = DiagonalGaussianDistribution(
            new_mu,
            np.maximum(raw_sigma2, floor),
            name=self.name,
            keys=self.keys,
            prior=new_prior,
        )
        return _record_variance_floor(dist, floor, raw_sigma2)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, float]
    ) -> "DiagonalGaussianDistribution":
        """Estimate a diagonal Gaussian distribution from aggregated sufficient statistics.

        Suff_stat is a Tuple of size 3 containing:
            suff_stat[0] (np.ndarray): Component-wise sum of weighted observation values.
            suff_stat[1] (np.ndarray): Component-wise sum of weighted squared observation values.
            suff_stat[2] (float): Sum of weights for each observation.

        Args:
            nobs (Optional[float]): Weighted number of observations used in aggregation of suff stats.
            suff_stat (Tuple[np.ndarray, np.ndarray, float]): See above for details.

        Returns:
            DiagonalGaussianDistribution object.

        """
        sum_x, sum_xx, count, inferred_dim = gaussian_moments(
            suff_stat,
            self.dim,
            diagonal=True,
        )
        if sum_x is None:
            if self.dim is None:
                raise ValueError("cannot infer diagonal Gaussian dimension from empty sufficient statistics")
            sum_x = vec.zeros(self.dim)
            sum_xx = vec.zeros(self.dim)
            inferred_dim = self.dim
        anchored = _consistent_anchored_moments(suff_stat, sum_x, count, inferred_dim)
        checked_stat = (sum_x, sum_xx, count)
        if self.has_conj_prior:
            return self._estimate_conjugate(checked_stat, anchored)

        nobs = count
        pc1, pc2 = self.pseudo_count

        if nobs <= 0:
            d = inferred_dim
            mu = np.asarray(self.prior_mu, dtype=float) if self.prior_mu is not None else vec.zeros(d)
            raw_covar = np.asarray(self.prior_covar, dtype=float) if self.prior_covar is not None else np.ones(d)
            floor = self._variance_floor(raw_covar)
            dist = DiagonalGaussianDistribution(
                mu,
                np.maximum(raw_covar, floor),
                name=self.name,
                keys=self.keys,
                prior=self.prior,
            )
            return _record_variance_floor(dist, floor, raw_covar)

        # A mean pseudo-count is only usable when its prior mean was supplied; unpaired counts
        # mean "no pseudo-observations" and fall through to the plain maximum-likelihood mean.
        # This mirrors the univariate contract in gaussian.py -- without the prior_mu guard the
        # branch below evaluates ``pc1 * None`` and raises TypeError.
        #
        # When the accumulator carried shift-anchored moments, BOTH the mean and the variances are
        # computed in offset space: the same two estimators, but with no term carrying the data's
        # offset, so neither loses digits to cancellation. Raw-only producers keep the historical
        # path -- and are told when that path cannot be trusted, rather than being handed silently
        # wrong variances.
        if anchored is not None:
            mean_offset = _anchored_mean_offset(anchored, nobs, pc1, self.prior_mu)
            mu = anchored[0] + mean_offset
            covar = _anchored_pooled_variances(
                anchored,
                nobs,
                mean_offset,
                pc2,
                self.prior_mu,
                self.prior_covar,
            )
        else:
            if pc1 not in (None, 0.0) and self.prior_mu is not None:
                mu = (sum_x + pc1 * self.prior_mu) / (nobs + pc1)
            else:
                mu = sum_x / nobs
            self._warn_if_uncorrectable(sum_x, sum_xx, nobs)
            covar = pooled_gaussian_covariance(
                sum_x,
                sum_xx,
                nobs,
                mu,
                pc2,
                self.prior_mu,
                self.prior_covar,
                diagonal=True,
            )

        # P1 variance floor: clamp non-positive coordinates to a data-scaled floor (see
        # ``_variance_floor``) so a component holding few points cannot produce zero or negative
        # variances, and record the clamp when it actually lifted a coordinate.
        raw_covar = np.asarray(covar, dtype=float)
        floor = self._variance_floor(raw_covar)

        dist = DiagonalGaussianDistribution(
            mu,
            np.maximum(raw_covar, floor),
            name=self.name,
            keys=self.keys,
            prior=self.prior,
        )
        return _record_variance_floor(dist, floor, raw_covar)


class DiagonalGaussianDataEncoder(DataSequenceEncoder):
    """Encoder for iid diagonal Gaussian observations."""

    def __init__(self, dim: int | None = None) -> None:
        """Create an encoder with an optional fixed dimension.

        Args:
            dim: Optional Gaussian dimension. Inferred from data when omitted.

        """
        self.dim = vector_dimension(dim, label="diagonal Gaussian dimension", allow_none=True)

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "DiagonalGaussianDataEncoder(dim=" + str(self.dim) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is an encoder with the same dimension.

        Args:
            other (object): Object to compare against.

        Returns:
            bool.

        """
        if isinstance(other, DiagonalGaussianDataEncoder):
            return self.dim == other.dim
        else:
            return False

    def seq_encode(self, x: Sequence[list[float] | np.ndarray]) -> np.ndarray:
        """Encode a sequence of iid length-dim observations for vectorized ``seq_`` calls.

        Args:
            x (Sequence[Union[List[float], np.ndarray]]): Sequence of length-dim observation vectors.

        Returns:
            Encoded data matrix with shape (len(x), dim).

        """
        raw = np.asarray(x, dtype=np.float64)
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[1] == 0:
                raise ValueError("diagonal Gaussian observations must have exact shape (N, D) with D > 0")
            self.dim = raw.shape[1]
        return vector_batch(raw, self.dim, label="diagonal Gaussian observations").copy()
