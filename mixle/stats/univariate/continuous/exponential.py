"""Exponential distributions over non-negative real values.

Observations are floats ``x >= 0``. An exponential distribution with scale ``beta > 0`` has log-density

    log(f(x; beta)) = -log(beta) - x/beta,

and assigns ``-np.inf`` outside the support.

Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.utils.special import digamma


def _fisher_mean_var(dist):
    mean = float(dist.beta) if hasattr(dist, "beta") else 1.0 / float(dist.lam)
    return mean, mean * mean


class ExponentialDistribution(SequenceEncodableProbabilityDistribution):
    """Exponential distribution on non-negative real values with scale ``beta``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for generated Exponential density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Exponential distribution."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="exponential",
            distribution_type=cls,
            parameters=(ParameterSpec("beta", constraint="positive"),),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="non_negative_real",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure=cls.exp_family_base_measure,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return Exponential sufficient statistics for generated scoring."""
        return (engine.asarray(x),)

    @staticmethod
    def exp_family_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Exponential sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        return xx * 0.0 + engine.asarray(1.0), xx

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Exponential natural parameters for generated scoring."""
        return (-engine.asarray(1.0) / params["beta"],)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Exponential log partition for generated scoring."""
        return engine.log(params["beta"])

    @staticmethod
    def exp_family_base_measure(x: Any, engine: Any) -> Any:
        """Return Exponential support base measure for generated scoring."""
        xx = engine.asarray(x)
        return engine.where(xx >= 0.0, xx * 0.0, engine.asarray(-np.inf))

    def __init__(
        self,
        beta: float,
        name: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ):
        """Create an exponential distribution with scale ``beta``.

        Observations are non-negative floats. The log-density is
        ``-log(beta) - x / beta`` for ``x >= 0`` and ``-inf`` otherwise.

        Args:
            beta (float): Positive scale parameter.
            name (Optional[str]): Optional distribution name.
            prior (Optional): Conjugate parameter prior over the rate ``1/beta``. A
                :class:`~mixle.stats.univariate.continuous.gamma.GammaDistribution` enables the Bayesian/variational
                machinery (``expected_log_density`` and the conjugate posterior update); ``None``
                (default) is a plain point model.

        Attributes:
            beta (float): Positive scale parameter.
            log_beta (float): ``log(beta)``.
            name (Optional[str]): Optional distribution name.

        """
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("ExponentialDistribution requires beta > 0.")
        self.beta = float(beta)
        self.log_beta = np.log(self.beta)
        self.name = name
        self.set_prior(prior)

    def __str__(self) -> str:
        """Return a constructor-style representation of the exponential distribution."""
        return "ExponentialDistribution(%s, name=%s)" % (repr(self.beta), repr(self.name))

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a parameter prior and precompute the conjugate Gamma expectations.

        The exponential rate is ``1/beta``; with a Gamma(k, theta) prior on that rate this
        caches the variational expected natural parameters so that
        ``expected_log_density(x) = e1*x - ea`` where the natural parameter is ``eta = -rate``,
        ``E[eta] = -k*theta`` and ``E[-log(-eta)] = psi(k) + ln theta`` (mapping the prior's
        ``(k, theta)`` to the bstats ``[a, b] = [k, 1/theta]`` form). Any other prior (including
        ``None``) leaves the distribution a plain point model.
        """
        self.prior = prior
        if isinstance(prior, GammaDistribution):
            k, theta = prior.get_parameters()
            a, b = k, 1.0 / theta
            self.conj_prior_params = (a, b)
            e1 = -a / b
            ea = -(digamma(a) - log(b))
            self.expected_nparams = (ea, 0.0, e1)
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def expected_log_density(self, x: float) -> float:
        """Variational expectation E_q[log p(x | rate)] under the Gamma prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if not self.has_conj_prior:
            return self.log_density(x)
        ea, eb, e1 = self.expected_nparams
        return -inf if x < 0 else e1 * x + (eb - ea)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if not self.has_conj_prior:
            return self.seq_log_density(x)
        ea, eb, e1 = self.expected_nparams
        rv = e1 * x + (eb - ea)
        return np.where(x >= 0.0, rv, -np.inf)

    def density(self, x: float) -> float:
        """Evaluate the density of exponential distribution with scale beta.

        See log_density() for details.

        Args:
            x (float): Non-negative real-valued number.

        Returns:
            Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Evaluate the log-density of exponential distribution with scale beta.

        log(f(x;beta)) = -log(beta) - x/beta, for x >= 0, else -np.inf.

        Args:
            x (float): Non-negative real-valued number.

        Returns:
            Log-density evaluated at x.

        """
        if x < 0:
            return -inf
        else:
            return -x / self.beta - self.log_beta

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Vectorized call to log-density on each observation value x.

        Args:
            x (np.ndarray): Numpy array of floats.

        Returns:
            Numpy array of log-density (float) of len(x).

        """
        # out-of-place arithmetic keeps the autograd graph intact under torch
        rv = x * (-1.0 / self.beta)
        rv = rv - self.log_beta
        rv = np.where(x >= 0.0, rv, -np.inf)
        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, beta: Any, engine: Any) -> Any:
        """Engine-neutral exponential log-density from explicit parameters."""
        rv = -x / beta - engine.log(beta)
        return engine.where(x >= 0.0, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        xx = engine.asarray(x)
        beta = engine.asarray(self.beta)
        return self.backend_log_density_from_params(xx, beta, engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["ExponentialDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Exponential parameters for a homogeneous mixture kernel."""
        return {"beta": engine.asarray([d.beta for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Exponential log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params["beta"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return stacked Exponential sufficient statistics using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        return engine.sum(ww, axis=0), engine.sum(ww * xx[:, None], axis=0)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import expon as _sp

        return float(_sp.cdf(x, scale=self.beta))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import expon as _sp

        return float(_sp.ppf(q, scale=self.beta))

    def to_fisher(self, **kwargs):
        """Return the Exponential's count-family Fisher view."""
        from mixle.inference.fisher import CountFisherView, _count_data, _identity_encoded

        return CountFisherView(self, _fisher_mean_var, _count_data, _identity_encoded)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.beta)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(self.beta * self.beta)

    def entropy(self) -> float:
        """Differential entropy 1 + log(scale)."""
        import math

        return float(1.0 + math.log(self.beta))

    def skewness(self) -> float:
        """Skewness (2)."""
        return 2.0

    def kurtosis(self) -> float:
        """Excess kurtosis (6)."""
        return 6.0

    def mode(self) -> float:
        """Mode (0)."""
        return 0.0

    def sampler(self, seed: int | None = None) -> "ExponentialSampler":
        """Create a sampler with scale ``beta``.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            ExponentialSampler object.

        """
        return ExponentialSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: float | None = None) -> "ExponentialEstimator":
        """Return an estimator initialized with this distribution's scale.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            ExponentialEstimator.

        """
        if pseudo_count is None:
            return ExponentialEstimator(name=self.name, prior=self.prior)
        else:
            return ExponentialEstimator(
                pseudo_count=pseudo_count, suff_stat=self.beta, name=self.name, prior=self.prior
            )

    def dist_to_encoder(self) -> "ExponentialDataEncoder":
        """Return the encoder for exponential observations."""
        return ExponentialDataEncoder()


class ExponentialSampler(DistributionSampler):
    """Draw independent samples from an :class:`ExponentialDistribution`."""

    def __init__(self, dist: "ExponentialDistribution", seed: int | None = None) -> None:
        """ExponentialSampler for drawing samples from ExponentialSampler instance.

        Args:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            rng (RandomState): RandomState with seed set to seed if passed in args.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw iid samples from the exponential distribution.

        Args:
            size (Optional[int]): Number of samples. ``None`` returns one float.

        Returns:
            NumPy array when ``size`` is provided; otherwise one sample as a float.
        """
        return self.rng.exponential(scale=self.dist.beta, size=size)


class ExponentialAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum statistics for Exponential estimation."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator for exponential sufficient statistics.

        Args:
            keys (Optional[str]): Aggregate all sufficient statistics with same keys values.

        Attributes:
            sum (float): Tracks the sum of observation values.
            count (float): Tracks the sum of weighted observations used to form sum.
            key (Optional[str]): Aggregate all sufficient statistics with same key.

        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys

    def update(self, x: float, weight: float, estimate: Optional["ExponentialDistribution"]) -> None:
        """Update sufficient statistics for ExponentialAccumulator with one weighted observation.

        Args:
            x (float): Observation from exponential distribution.
            weight (float): Weight for observation.
            estimate (Optional['ExponentialDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional["ExponentialDistribution"]) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        sum increased by sum of weighted observations.
        count increased by sum of weights.

        Args:
            x (ndarray): Numpy array of positvie floats.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional['ExponentialDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def initialize(self, x: float, weight: float, rng: Optional["np.random.RandomState"]) -> None:
        """Initialize sufficient statistics of ExponentialAccumulator with weighted observation.

        This delegates to :meth:`update`.

        Args:
            x (float): Positive real-valued observation of exponential.
            weight (float): Positive real-valued weight for observation x.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of ExponentialAccumulator sufficient statistics with weighted observations.

        This delegates to :meth:`seq_update`.

        Args:
            x (ndarray): Numpy array of positive floats.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "ExponentialAccumulator":
        """Aggregates sufficient statistics with ExponentialAccumulator member sufficient statistics.

        Args:
            suff_stat (Tuple[float, float]): Aggregated count and sum.

        Returns:
            ExponentialAccumulator

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> tuple[float, float]:
        """Return ``(count, sum)`` sufficient statistics."""
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "ExponentialAccumulator":
        """Sets sufficient statistics (count and sum) of ExponentialAccumulator to x.

        Args:
            x (Tuple[float, float]): Sufficient statistics tuple (count, sum).

        Returns:
            ExponentialAccumulator

        """
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merges ExponentialAccumulator sufficient statistics with sufficient statistics contained in suff_stat dict
        that share the same key.

        Args:
            stats_dict (Dict[str, Any]): Dict containing 'key' string for ExponentialAccumulator
                objects that represent the same distribution.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1 = stats_dict[self.keys]
                self.count += x0
                self.sum += x1
                # write the POOL back: without this, the dict keeps the FIRST site's stats and
                # key_replace hands every tied site that truncated pool -- later sites' data was
                # silently discarded (order-dependent wrong fits; found by the compiler review's
                # keyed-tying probe, present in 8 families vs the combine-into-dict families)
                stats_dict[self.keys] = (self.count, self.sum)
            else:
                stats_dict[self.keys] = (self.count, self.sum)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Set the sufficient statistics of ExponentialAccumulator to stats_key sufficient statistics if key is in
            stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Map key to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.count = stats_dict[self.keys][0]
                self.sum = stats_dict[self.keys][1]

    def acc_to_encoder(self) -> "ExponentialDataEncoder":
        """Return the encoder associated with this accumulator."""
        return ExponentialDataEncoder()


class ExponentialAccumulatorFactory(StatisticAccumulatorFactory):
    """Create Exponential accumulators with a shared optional merge key."""

    def __init__(self, keys: str | None = None) -> None:
        """Create a factory for exponential accumulators.

        Args:
            keys (Optional[str]): Used for merging sufficient statistics of ExponentialAccumulator.

        Attributes:
            keys (Optional[str]): Used for merging sufficient statistics of ExponentialAccumulator.

        """
        self.keys = keys

    def make(self) -> "ExponentialAccumulator":
        """Create an accumulator with this factory's merge key."""
        return ExponentialAccumulator(keys=self.keys)


class ExponentialEstimator(ParameterEstimator):
    """Estimate Exponential scale parameters from accumulated sufficient statistics."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create an estimator for exponential scale parameters.

        Args:
            pseudo_count (Optional[float]): Used to weight sufficient statistics.
            suff_stat (Optional[float]): Positive float value for scale of exponential distribution.
            name (Optional[str]): Assign a name to ExponentialEstimator.
            keys (Optional[str]): Assign keys to ExponentialEstimator for combining sufficient statistics.
            prior (Optional): Conjugate Gamma prior over the rate ``1/beta``. When present,
                ``estimate`` performs the closed-form conjugate posterior update (returning the
                Gamma-posterior-mode rate and carrying the posterior forward as the fitted model's
                prior) instead of the maximum-likelihood / pseudo-count update.

        Attributes:
            pseudo_count (Optional[float]): Used to weight sufficient statistics.
            suff_stat (Optional[float]): Positive float value for scale of exponential distribution.
            name (Optional[str]): Assign a name to ExponentialEstimator.
            keys (Optional[str]): Assign keys to ExponentialEstimator for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.prior = prior
        self.has_conj_prior = isinstance(prior, GammaDistribution)

    def accumulator_factory(self) -> "ExponentialAccumulatorFactory":
        """Create an exponential accumulator factory with this estimator's keys."""
        return ExponentialAccumulatorFactory(self.keys)

    def model_log_density(self, model: "ExponentialDistribution") -> float:
        """Log-density of the model's rate (1/beta) under the Gamma prior (ELBO global term)."""
        if self.has_conj_prior:
            return float(self.prior.log_density(1.0 / model.beta))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[float, float]) -> "ExponentialDistribution":
        """Closed-form Gamma conjugate posterior update returning the posterior-mode estimate.

        The Gamma prior is over the rate; ``[a, b] = [k, 1/theta]`` maps the prior's (k, theta).
        The posterior is Gamma(n, 1/s) with ``n = count + a`` and ``s = sum + b``, the posterior-mode
        rate is ``(n - 1)/s``, and the returned scale is its reciprocal ``s/(n - 1)``.
        """
        k, theta = self.prior.get_parameters()
        a, b = k, 1.0 / theta

        n = suff_stat[0] + a
        s = suff_stat[1] + b

        rate = (n - 1.0) / s
        return ExponentialDistribution(1.0 / rate, name=self.name, prior=GammaDistribution(n, 1.0 / s))

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> "ExponentialDistribution":
        """Estimate ExponentialDistribution from suff_stat arg.

        Estimate ExponentialDistribution from sufficient statistic tuple suff_stat, counting a float value for
        count and sum. If pseudo_count is set, this is used to re-weight the member value "suff_stat", which is the
        scale of ExponentialEstimator object.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat (Tuple[float, float]): Tuple of count and sum. Both are positive real-valued floats.

        Returns:
            ExponentialDistribution object.

        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[1] + self.suff_stat * self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[1] + self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        else:
            if suff_stat[0] > 0:
                p = suff_stat[1] / suff_stat[0]
            else:
                p = 1.0

        return ExponentialDistribution(beta=p, name=self.name)


class ExponentialDataEncoder(DataSequenceEncoder):
    """Data encoder for iid non-negative exponential observations."""

    def __str__(self) -> str:
        """Return the exponential encoder's display name."""
        return "ExponentialDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an exponential data encoder."""
        return isinstance(other, ExponentialDataEncoder)

    def seq_encode(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Encode sequence of iid exponential observations.

        Data type must be a float.
        Data must also be non-negative real-valued numbers.

        Args:
            x (Union[List[float], np.ndarray]): IID numpy array or list of non-negative real-valued floats.

        Returns:
            Numpy array of floats.

        """
        rv = np.asarray(x, dtype=float)

        if np.any(rv < 0) or np.any(np.isnan(rv)):
            raise ValueError("Exponential requires x >= 0.")

        return rv
