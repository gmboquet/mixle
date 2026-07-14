"""Gaussian distributions, estimators, accumulators, samplers, and encoders.

For real-valued observations, ``GaussianDistribution(mu, sigma2)`` has
``sigma2 > 0`` and log-density:

    log(f(x;mu, sigma2)) = -0.5*log(2*pi*sigma2) - 0.5*(x-mu)^2/sigma2, for real-valued x.

Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.inference.fisher import FixedFisherView
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
from mixle.stats.compute.error_receipts import CompensatedAccumulator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.aliasing import broadcast_pseudo_count
from mixle.utils.special import digamma


class GaussianFisherView(FixedFisherView):
    """Fisher view over the Gaussian's (sum, sum2, count, count2) sufficient statistics."""

    def __init__(self, dist: Any) -> None:
        super().__init__(dist, [("sum",), ("sum2",), ("count",), ("count2",)])

    @staticmethod
    def _matrix(x: Any) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        one = np.ones_like(xx, dtype=np.float64)
        return np.column_stack((xx, xx * xx, one, one))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        return self._matrix(data)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        return self._matrix(enc_data)

    def _model_mean(self) -> np.ndarray:
        mu = float(self.dist.mu)
        var = float(self.dist.sigma2)
        return np.asarray([mu, mu * mu + var, 1.0, 1.0], dtype=np.float64)

    def _model_fisher(self) -> np.ndarray:
        mu = float(self.dist.mu)
        var = float(self.dist.sigma2)
        ex1 = mu
        ex2 = mu * mu + var
        ex3 = mu * mu * mu + 3.0 * mu * var
        ex4 = mu**4 + 6.0 * mu * mu * var + 3.0 * var * var
        info = np.zeros((4, 4), dtype=np.float64)
        info[0, 0] = ex2 - ex1 * ex1
        info[0, 1] = ex3 - ex1 * ex2
        info[1, 0] = info[0, 1]
        info[1, 1] = ex4 - ex2 * ex2
        return info


class GaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Univariate Gaussian distribution."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for generated Gaussian density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Gaussian distribution."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="gaussian",
            distribution_type=cls,
            parameters=(ParameterSpec("mu"), ParameterSpec("sigma2", constraint="positive")),
            statistics=(
                StatisticSpec("sum"),
                StatisticSpec("sum2"),
                StatisticSpec("count"),
                StatisticSpec("count2"),
            ),
            support="real",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return Gaussian sufficient statistics for generated scoring."""
        xx = engine.asarray(x)
        return xx, xx * xx

    @staticmethod
    def exp_family_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Gaussian sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        one = xx * 0.0 + engine.asarray(1.0)
        return xx, xx * xx, one, one

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Gaussian natural parameters for generated scoring."""
        sigma2 = params["sigma2"]
        return params["mu"] / sigma2, -0.5 / sigma2

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Gaussian log partition for generated scoring."""
        mu = params["mu"]
        sigma2 = params["sigma2"]
        return 0.5 * engine.log(engine.asarray(2.0 * pi) * sigma2) + 0.5 * mu * mu / sigma2

    def __init__(
        self,
        mu: float,
        sigma2: float,
        name: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a univariate Gaussian distribution.

        Args:
            mu: Mean of the Gaussian.
            sigma2: Positive finite variance.
            name: Optional diagnostic name.
            prior (Optional): Conjugate parameter prior over (mu, tau=1/sigma2). A
                :class:`~mixle.stats.bayes.normal_gamma.NormalGammaDistribution` enables the
                Bayesian/variational machinery (``expected_log_density`` and the
                conjugate posterior update); ``None`` (default) is a plain point model.

        Attributes:
            mu: Mean of the Gaussian.
            sigma2: Variance of the Gaussian.
            name: Optional diagnostic name.
            const: Density normalizing constant.
            log_const: Log normalizing constant.

        """
        if not np.isfinite(mu):
            raise ValueError("GaussianDistribution requires finite mu.")
        if sigma2 <= 0.0 or not np.isfinite(sigma2):
            raise ValueError("GaussianDistribution requires finite sigma2 > 0.")
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)
        self.name = name
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        return "GaussianDistribution(%s, %s, name=%s)" % (repr(self.mu), repr(self.sigma2), repr(self.name))

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a parameter prior and precompute conjugate-prior expectations.

        With a NormalGamma(mu0, lam, a, b) prior over (mu, tau=1/sigma2) this caches the
        variational expected natural parameters [ea, eb, e1, e2] so that
        ``expected_log_density(x) = x*(e1 + x*e2) - ea + eb`` (the VB E-step term).
        Any other prior (including ``None``) leaves the distribution a plain point model.
        """
        self.prior = prior
        if isinstance(prior, NormalGammaDistribution):
            mu, lam, a, b = prior.get_parameters()
            ea = (mu * mu) * (a / b) * 0.5 + (0.5 / lam) + 0.5 * (np.log(b) - digamma(a))
            e1 = mu * a / b
            e2 = -0.5 * a / b
            eb = -0.5 * np.log(2 * np.pi)
            self.expected_nparams = [ea, eb, e1, e2]
            self.has_conj_prior = True
        else:
            self.expected_nparams = None
            self.has_conj_prior = False

    def expected_log_density(self, x: float) -> float:
        """Variational expectation E_q[log p(x | mu, tau)] under the NormalGamma prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return x * (e1 + x * e2) - ea + eb
        return self.log_density(x)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return x * (e1 + x * e2) - ea + eb
        return self.seq_log_density(x)

    def density(self, x: float) -> float:
        """Density of Gaussian distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            Density of Gaussian at x.

        """
        return self.const * exp(-0.5 * (x - self.mu) * (x - self.mu) / self.sigma2)

    def log_density(self, x: float) -> float:
        """Log-density of Gaussian distribution at observation x.

        Log-density of Gaussian with mean mu and variance sigma2 given by,
            log(f(x;mu, sigma2)) = -0.5*log(2*pi*sigma2) - 0.5*(x-mu)^2/sigma2, for real-valued x.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            Log-density at observation x.

        """
        return self.log_const - 0.5 * (x - self.mu) * (x - self.mu) / self.sigma2

    def seq_ld_lambda(self) -> list[Callable]:
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): Numpy array of floats.

        Returns:
            Numpy array of log-density (float) of len(x).

        """
        # out-of-place so torch tensors with requires_grad pass through the
        # generic engine path without breaking the autograd graph
        rv = x - self.mu
        rv = rv * rv
        rv = rv * (-0.5 / self.sigma2)
        rv = rv + self.log_const

        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, sigma2: Any, engine: Any) -> Any:
        """Engine-neutral Gaussian log-density from explicit parameters."""
        return -0.5 * engine.log(engine.asarray(2.0 * pi) * sigma2) - 0.5 * (x - mu) * (x - mu) / sigma2

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        xx = engine.asarray(x)
        mu = engine.asarray(self.mu)
        sigma2 = engine.asarray(self.sigma2)
        return self.backend_log_density_from_params(xx, mu, sigma2, engine)

    def gradient_log_prior(self, priors: Any, prior_strength: float, torch: Any, engine: Any) -> Any:
        """Distribution-owned MAP prior contribution for Gaussian parameters."""
        from mixle.stats.compute.gradient import normal_gamma_log_prior

        return normal_gamma_log_prior(self.mu, self.sigma2, priors, torch)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["GaussianDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Gaussian parameters for a homogeneous mixture kernel."""
        return {
            "mu": engine.asarray([d.mu for d in dists]),
            "sigma2": engine.asarray([d.sigma2 for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Gaussian log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["mu"][None, :], params["sigma2"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Return stacked Gaussian sufficient statistics using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        xx_col = xx[:, None]
        count = engine.sum(ww, axis=0)
        weighted_x = ww * xx_col
        return (
            engine.sum(weighted_x, axis=0),
            engine.sum(weighted_x * xx_col, axis=0),
            count,
            count,
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import norm

        return float(norm.cdf(x, loc=self.mu, scale=self.sigma2**0.5))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import norm

        return float(norm.ppf(q, loc=self.mu, scale=self.sigma2**0.5))

    def to_fisher(self, **kwargs):
        """Return the Gaussian's own Fisher view."""
        return GaussianFisherView(self)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.mu)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(self.sigma2)

    def entropy(self) -> float:
        """Differential entropy 0.5*log(2*pi*e*sigma2)."""
        import math

        return float(0.5 * (math.log(2.0 * math.pi * self.sigma2) + 1.0))

    def skewness(self) -> float:
        """Skewness (0)."""
        return 0.0

    def kurtosis(self) -> float:
        """Excess kurtosis (0)."""
        return 0.0

    def mode(self) -> float:
        """Mode (= the mean mu)."""
        return float(self.mu)

    def sampler(self, seed: int | None = None) -> "GaussianSampler":
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional seed for the sampler's random state.

        Returns:
            A configured ``GaussianSampler``.

        """
        return GaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GaussianEstimator":
        """Return an estimator initialized from this distribution's shape.

        Args:
            pseudo_count: Optional smoothing count applied to the current mean
                and variance.

        Returns:
            A ``GaussianEstimator``.

        """
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return GaussianEstimator(
                pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name, prior=self.prior
            )
        else:
            return GaussianEstimator(name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "GaussianDataEncoder":
        """Return an encoder for iid scalar Gaussian observations."""
        return GaussianDataEncoder()


class GaussianSampler(DistributionSampler):
    """Draw independent samples from a :class:`GaussianDistribution`."""

    def __init__(self, dist: GaussianDistribution, seed: int | None = None) -> None:
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

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw iid samples from the Gaussian distribution.

        Args:
            size: Number of iid samples to draw. ``None`` returns a scalar sample.

        Returns:
            A scalar draw when ``size`` is ``None``; otherwise an array of draws.

        """
        return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianSuffStat(tuple):
    """A ``(sum, sum2, count, count2)`` sufficient statistic that also carries a numerics receipt.

    Behaves exactly like the plain 4-tuple everywhere it is indexed, unpacked, or iterated (it *is*
    one); ``receipt`` is extra payload that :meth:`GaussianAccumulator.combine` reads to fold the
    Kahan error-bound bookkeeping (``abs_total``, ``n`` for ``sum`` and ``sum2``) into the receiving
    accumulator when both sides are ``compensated``. Code that doesn't know about ``compensated``
    accumulation (serialization, generic ``scale_suff_stat``, ...) sees an ordinary tuple.
    """

    def __new__(cls, sum_: float, sum2_: float, count_: float, count2_: float, receipt: dict | None = None):
        obj = super().__new__(cls, (sum_, sum2_, count_, count2_))
        obj.receipt = receipt
        return obj


class GaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for Gaussian estimation."""

    def __init__(self, keys: str | None = None, name: str | None = None, compensated: bool = False) -> None:
        """Create an accumulator for weighted Gaussian moments.

        Args:
            keys: Optional key for merging sufficient statistics.
            name: Optional diagnostic name.
            compensated: Opt-in Kahan-compensated accumulation of ``sum``/``sum2`` with a
                running numerics-error bound (see :mod:`mixle.stats.compute.error_receipts`), read
                back via :meth:`error_bound`. ``False`` (the default) is the plain float64
                accumulation this class always used -- that code path is untouched, so it carries no
                measurable overhead over the pre-existing behavior.

        Attributes:
            sum: Weighted sum of observations.
            sum2: Weighted sum of squared observations.
            count: Sum of weights for the first moment.
            count2: Sum of weights for the second moment.
            keys: Optional sufficient-statistic key.
            name: Optional diagnostic name.

        """
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name
        self.compensated = compensated
        self._sum_acc = CompensatedAccumulator(compensated=True) if compensated else None
        self._sum2_acc = CompensatedAccumulator(compensated=True) if compensated else None

    def update(self, x: float, weight: float, estimate: Optional["GaussianDistribution"]) -> None:
        """Update sufficient statistics for GaussianAccumulator with one weighted observation.

        Args:
            x (float): Observation from Gaussian distribution.
            weight (float): Weight for observation.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        x_weight = x * weight
        if self.compensated:
            self._sum_acc.add(x_weight)
            self._sum2_acc.add(x * x_weight)
            self.sum = self._sum_acc.total
            self.sum2 = self._sum2_acc.total
        else:
            self.sum += x_weight
            self.sum2 += x * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize with a weighted observation.

        Args:
            x (float): Observation from Gaussian distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization from encoded weighted observations.

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: GaussianDistribution | None) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        if self.compensated:
            for xi, wi in zip(x, weights):
                xw = float(xi) * float(wi)
                self._sum_acc.add(xw)
                self._sum2_acc.add(float(xi) * xw)
            self.sum = self._sum_acc.total
            self.sum2 = self._sum2_acc.total
            w_sum = float(np.sum(weights))
        else:
            self.sum += np.dot(x, weights)
            self.sum2 += np.dot(x * x, weights)
            w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: tuple[float, float, float, float]) -> "GaussianAccumulator":
        """Merge sufficient statistics into this accumulator.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i).

        When this accumulator is ``compensated`` and ``suff_stat`` carries a numerics-error
        receipt (see :meth:`value` / :class:`GaussianSuffStat`), the receipt is folded in too --
        its ``(abs_total, n)`` fields add exactly, just like ``sum``/``count`` above, so
        :meth:`error_bound` composes correctly across combined partitions.

        Args:
            suff_stat (Tuple[float, float, float, float]): See above for details.

        Returns:
            This accumulator.

        """
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        if self.compensated:
            receipt = getattr(suff_stat, "receipt", None)
            if receipt is not None:
                abs_s, n_s = receipt["sum"]
                abs_s2, n_s2 = receipt["sum2"]
                self._sum_acc.abs_total += abs_s
                self._sum_acc.n += n_s
                self._sum2_acc.abs_total += abs_s2
                self._sum2_acc.n += n_s2
            # keep the running Kahan total (and its compensation) in sync with the merged sum
            self._sum_acc.total = self.sum
            self._sum2_acc.total = self.sum2

        return self

    def error_bound(self) -> dict[str, float] | None:
        """Return the running numerics-error bound receipt for ``sum``/``sum2``.

        ``None`` when this accumulator was not constructed with ``compensated=True`` -- the
        disabled default carries no receipt to report.
        """
        if not self.compensated:
            return None
        return {"sum": self._sum_acc.bound(), "sum2": self._sum2_acc.bound()}

    def value(self) -> tuple[float, float, float, float]:
        """Returns sufficient statistics of GaussianAccumulator object (Tuple[float, float, float, float]).

        When ``compensated``, the returned value is a :class:`GaussianSuffStat` -- a drop-in 4-tuple
        (indexing/unpacking/iteration all behave identically) that additionally carries the
        numerics-error receipt in its ``.receipt`` attribute, so :meth:`combine` can fold it in.
        """
        if self.compensated:
            receipt = {
                "sum": (self._sum_acc.abs_total, self._sum_acc.n),
                "sum2": (self._sum2_acc.abs_total, self._sum2_acc.n),
            }
            return GaussianSuffStat(self.sum, self.sum2, self.count, self.count2, receipt=receipt)
        return self.sum, self.sum2, self.count, self.count2

    def from_value(self, x: tuple[float, float, float, float]) -> "GaussianAccumulator":
        """Replace this accumulator's sufficient statistics.

        Arg passed x is tuple of four floats:
            x[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            x[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            x[2] (float): Sum of weighted observations (sum_i w_i),
            x[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            x: Tuple of ``(sum, sum2, count, count2)``.

        Returns:
            This accumulator.

        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        if self.compensated:
            receipt = getattr(x, "receipt", None)
            if receipt is not None:
                abs_s, n_s = receipt["sum"]
                abs_s2, n_s2 = receipt["sum2"]
                self._sum_acc = CompensatedAccumulator(total=self.sum, abs_total=abs_s, n=n_s, compensated=True)
                self._sum2_acc = CompensatedAccumulator(total=self.sum2, abs_total=abs_s2, n=n_s2, compensated=True)
            else:
                self._sum_acc = CompensatedAccumulator(total=self.sum, compensated=True)
                self._sum2_acc = CompensatedAccumulator(total=self.sum2, compensated=True)

        return self

    def acc_to_encoder(self) -> "GaussianDataEncoder":
        """Return an encoder compatible with Gaussian scalar observations."""
        return GaussianDataEncoder()


class GaussianAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, name: str | None = None, keys: str | None = None, compensated: bool = False) -> None:
        """GaussianAccumulatorFactory object for creating GaussianAccumulator.

        Args:
            name (Optional[str]): Assign a name to GaussianAccumulatorFactory object.
            keys (Optional[str]): Assign keys member for GaussianAccumulators.
            compensated (bool): Passed through to each made :class:`GaussianAccumulator`; see there.

        Attributes:
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.

        """
        self.keys = keys
        self.name = name
        self.compensated = compensated

    def make(self) -> "GaussianAccumulator":
        """Return a GaussianAccumulator object with name and keys passed."""
        return GaussianAccumulator(name=self.name, keys=self.keys, compensated=self.compensated)


class GaussianEstimator(ParameterEstimator):
    """Estimate Gaussian mean and variance from accumulated sufficient statistics."""

    def __init__(
        self,
        pseudo_count: float | tuple[float | None, float | None] = (None, None),
        suff_stat: tuple[float | None, float | None] = (None, None),
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
        min_covar: float | None = None,
        compensated: bool = False,
    ):
        """Create an estimator for Gaussian sufficient statistics.

        Args:
            pseudo_count: Optional smoothing weights for the prior mean and variance statistics.
                A scalar is broadcast to both slots.
            suff_stat: Optional prior mean and variance statistics used with ``pseudo_count``.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior (Optional): Conjugate NormalGamma prior over (mu, tau=1/sigma2). When present,
                ``estimate`` performs the closed-form conjugate posterior update (returning the joint
                MAP estimate and carrying the posterior forward as the fitted model's prior) instead
                of the maximum-likelihood / pseudo-count update.
            min_covar (Optional[float]): Absolute variance floor applied in the MLE M-step. ``None``
                (default) uses a tiny ``1e-8`` floor; the estimated variance is also floored at a
                relative ``1e-6 * sigma2`` to keep the safeguard data-scaled. Set explicitly to widen
                the floor for hard / high-dimensional cases. Bias is negligible at the default.
            compensated (bool): Opt-in Kahan-compensated accumulation with a running numerics-error
                bound for the accumulators this estimator makes; see
                :class:`GaussianAccumulator`. ``False`` by default (no overhead).

        Attributes:
            pseudo_count: Smoothing weights for ``suff_stat``.
            suff_stat: Prior mean and variance statistics.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.

        """
        pseudo_count = broadcast_pseudo_count(pseudo_count, 2)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalGammaDistribution)
        self.min_covar = 1.0e-8 if min_covar is None else float(min_covar)
        self.compensated = compensated

    def accumulator_factory(self) -> "GaussianAccumulatorFactory":
        """Return GaussianAccumulatorFactory with name and keys passed."""
        return GaussianAccumulatorFactory(self.name, self.keys, compensated=self.compensated)

    def model_log_density(self, model: "GaussianDistribution") -> float:
        """Log-density of the model parameters under the NormalGamma prior (ELBO global term).

        The prior is over (mu, tau=1/sigma2), so the model's (mu, sigma2) is mapped accordingly.
        """
        if self.has_conj_prior:
            return float(self.prior.log_density((model.mu, 1.0 / model.sigma2)))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[float, float, float, float]) -> "GaussianDistribution":
        """Closed-form NormalGamma conjugate posterior update returning the joint MAP estimate."""
        sum_x, sum_xx, nobs_loc1, nobs_loc2 = suff_stat
        sum_xxx = sum_x  # the variance-count scatter uses the same weighted sum of x
        old_mu, old_lam, old_a, old_b = self.prior.get_parameters()

        new_n = old_lam + nobs_loc1
        new_a = old_a + (nobs_loc2 / 2.0)

        sample_mean1 = sum_x / nobs_loc1 if nobs_loc1 > 0 else 0.0
        sample_mean2 = sum_xxx / nobs_loc2 if nobs_loc2 > 0 else 0.0

        new_mu = (sum_x + old_mu * old_lam) / (old_lam + nobs_loc1)

        # The scatter ``sum_xx - (sum_x)^2/n`` from reduced sufficient statistics is the classic
        # cancellation-prone form: on near-constant / large-offset data it can round slightly negative,
        # driving ``new_b`` (and hence the variance) negative -- a ValueError for the scalar Gaussian, a
        # silent NaN log-density for the diagonal one. Floor it at 0 (the MLE path floors equivalently).
        new_b0 = max(sum_xx - sample_mean2 * sum_xxx, 0.0)
        new_b1 = (old_lam * nobs_loc1 / new_n) * np.power(sample_mean1 - old_mu, 2)
        new_b = old_b + 0.5 * (new_b0 + new_b1)

        denom = new_a - 0.5
        new_sigma2 = new_b / denom if denom > 0.0 else self.min_covar
        new_sigma2 = max(new_sigma2, self.min_covar)  # match the MLE-path variance floor
        new_prior = NormalGammaDistribution(new_mu, new_n, new_a, new_b)
        return GaussianDistribution(new_mu, new_sigma2, name=self.name, prior=new_prior)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float, float]) -> "GaussianDistribution":
        """Estimate a Gaussian distribution from aggregated sufficient statistics.

        The tuple is interpreted as ``(sum_x, sum_x2, count_for_mean,
        count_for_variance)``. Optional pseudo-counts smooth the corresponding
        mean and variance estimates.

        Args:
            nobs: Unused; accepted for the ``ParameterEstimator`` interface.
            suff_stat: Aggregated Gaussian sufficient statistics.

        Returns:
            A fitted Gaussian distribution.

        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        nobs_loc1 = suff_stat[2]
        nobs_loc2 = suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (suff_stat[0] + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1]) / (
                nobs_loc2 + self.pseudo_count[1]
            )
        else:
            # E[x^2] - E[x]^2 from the (count, sum, sum_sq) exponential-family sufficient statistic.
            # This is the *required* form for engine-swap parity: the streaming/stacked (numpy/torch)
            # reduction produces sum_sq in a single pass, so the M-step cannot use a centered
            # (Welford) scatter without breaking numpy<->torch accumulate parity. The subtraction can
            # lose precision when |mu| >> sigma (large-offset data); the floor below caps the
            # degenerate tail, and pre-centering the data is the recommended remedy for extreme offsets.
            sigma2 = suff_stat[1] / nobs_loc2 - mu * mu

        # P1 variance floor: clamp non-finite / non-positive variance and apply a
        # data-scaled floor max(abs_floor, rel * sigma2) so a degenerate component
        # cannot produce a zero/negative/NaN variance. Bias is negligible at the
        # default abs_floor=1e-8 / rel=1e-6.
        if not np.isfinite(sigma2) or sigma2 <= 0.0:
            sigma2 = self.min_covar
        else:
            sigma2 = max(sigma2, self.min_covar, 1.0e-6 * sigma2)

        return GaussianDistribution(mu, sigma2, name=self.name)


class GaussianDataEncoder(DataSequenceEncoder):
    """Encoder for iid scalar Gaussian observations."""

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "GaussianDataEncoder"

    def __eq__(self, other) -> bool:
        """Return whether ``other`` is a Gaussian data encoder.

        Args:
            other: Object to compare.

        Returns:
            True if other is an instance of a GaussianDataEncoder, else False.

        """
        return isinstance(other, GaussianDataEncoder)

    def seq_encode(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Encode sequence of iid Gaussian observations.

        Data type must be List[float] or np.ndarray[float].

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid Gaussian observations.

        Returns:
            A numpy array of floats.

        """
        rv = np.asarray(x, dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise ValueError("GaussianDistribution requires support x in (-inf,inf).")
        return rv
