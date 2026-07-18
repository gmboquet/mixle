"""Log-Gaussian distributions over positive real values.

Observations are floats ``x > 0``. A log-Gaussian distribution with location ``mu`` and variance
``sigma2 > 0`` has log-density

    log(f(x; mu, sigma2)) = -log(2*pi*sigma2) - log(x) - (log(x)-mu)^2/sigma2.

Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.inference.fisher import FixedFisherView
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
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


class LogGaussianFisherView(FixedFisherView):
    """Expose log-Gaussian sufficient statistics for Fisher-information utilities."""

    def __init__(self, dist: Any) -> None:
        super().__init__(dist, [("log_sum",), ("log_sum2",), ("count",), ("count2",)])

    @staticmethod
    def _matrix(x: Any, already_log: bool = False) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        if not already_log:
            xx = np.log(xx)
        one = np.ones_like(xx, dtype=np.float64)
        return np.column_stack((xx, xx * xx, one, one))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        return self._matrix(data, already_log=False)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        return self._matrix(enc_data, already_log=True)

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


class LogGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Log-normal distribution where ``log(X)`` is Gaussian with mean ``mu`` and variance ``sigma2``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for generated log-Gaussian density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the log-Gaussian distribution."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="log_gaussian",
            distribution_type=cls,
            parameters=(ParameterSpec("mu"), ParameterSpec("sigma2", constraint="positive")),
            statistics=(
                StatisticSpec("log_sum"),
                StatisticSpec("log_sum2"),
                StatisticSpec("count"),
                StatisticSpec("count2"),
            ),
            support="positive_real",
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
        """Return log-Gaussian sufficient statistics for generated scoring."""
        xx = engine.asarray(x)
        return xx, xx * xx

    @staticmethod
    def exp_family_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row log-Gaussian sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        one = xx * 0.0 + engine.asarray(1.0)
        return xx, xx * xx, one, one

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return log-Gaussian natural parameters for generated scoring."""
        sigma2 = params["sigma2"]
        return params["mu"] / sigma2, -0.5 / sigma2

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return log-Gaussian log partition for generated scoring."""
        mu = params["mu"]
        sigma2 = params["sigma2"]
        return 0.5 * engine.log(engine.asarray(2.0 * pi) * sigma2) + 0.5 * mu * mu / sigma2

    @staticmethod
    def exp_family_base_measure(x: Any, engine: Any) -> Any:
        """Return log-Gaussian base measure for generated scoring."""
        return -engine.asarray(x)

    def __init__(
        self,
        mu: float,
        sigma2: float,
        name: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a log-Gaussian distribution.

        Args:
            mu (float): Mean of ``log(X)``.
            sigma2 (float): Variance of ``log(X)``.
            name (Optional[str]): Optional distribution name.
            prior (Optional): Conjugate parameter prior over (mu, tau=1/sigma2) of log(X). A
                :class:`~mixle.stats.bayes.normal_gamma.NormalGammaDistribution` enables the
                Bayesian/variational machinery (``expected_log_density`` and the conjugate
                posterior update); ``None`` (default) is a plain point model.

        Attributes:
            mu (float): Location parameter for log-Gaussian distribution.
            sigma2 (float): Scale for log-Gaussian distribution.
            name (Optional[str]): Optional distribution name.
            cont (float): Normalizing constant (depends on sigma2).
            log_const (float): Log of above.

        """
        if isnan(mu) or isinf(mu):
            raise ValueError("LogGaussianDistribution requires finite mu.")
        if sigma2 <= 0.0 or isnan(sigma2) or isinf(sigma2):
            raise ValueError("LogGaussianDistribution requires finite sigma2 > 0.")
        self.mu = mu
        self.sigma2 = sigma2
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)
        self.name = name
        self.set_prior(prior)

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a parameter prior and precompute conjugate-prior expectations.

        With a NormalGamma(mu0, lam, a, b) prior over (mu, tau=1/sigma2) of log(X) this caches
        the variational expected natural parameters [ea, eb, e1, e2] exactly as in the Gaussian
        case, so that ``expected_log_density(x) = y*(e1 + y*e2) - ea + eb - y`` with y = log(x).
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

        With a conjugate prior this is the Gaussian VB expected log-likelihood evaluated at
        log(x), plus the Jacobian term -log(x); without a prior it falls back to log_density(x).
        """
        if self.has_conj_prior:
            if x <= 0:
                return -np.inf
            y = np.log(x)
            ea, eb, e1, e2 = self.expected_nparams
            return y * (e1 + y * e2) - ea + eb - y
        return self.log_density(x)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded (logged) observations."""
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return x * (e1 + x * e2) - ea + eb - x
        return self.seq_log_density(x)

    def __str__(self) -> str:
        """Return a constructor-style representation of the log-Gaussian distribution."""
        return "LogGaussianDistribution(%s, %s, name=%s)" % (repr(self.mu), repr(self.sigma2), repr(self.name))

    def density(self, x: float) -> float:
        """Density of Log-Gaussian distribution at observation x.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            Density of Log-Gaussian at x.

        """
        if x <= 0.0:
            return 0.0
        return self.const * exp(-0.5 * (np.log(x) - self.mu) ** 2 / self.sigma2) / x

    def log_density(self, x: float) -> float:
        """Log-density of log-Gaussian distribution at observation x.

        Log-density of log-Gaussian with log-mean mu and log-variance sigma2 given by,
            log(f(x;mu, sigma2)) = -0.5*log(2*pi*sigma2) - log(x) - 0.5*(log(x)-mu)^2/sigma2, for positive x.

        Args:
            x (float): Positive valued observation of log-Gaussian.

        Returns:
            Log-density at observation x.

        """
        if x <= 0.0:
            return -np.inf
        y = np.log(x)
        return self.log_const - 0.5 * (y - self.mu) ** 2 / self.sigma2 - y

    def seq_ld_lambda(self) -> list[Callable]:
        """Return vectorized log-density callables for fast scoring."""
        return [self.seq_log_density]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): Numpy array of floats.

        Returns:
            Numpy array of log-density (float) of len(x).

        """
        # out-of-place so torch tensors with requires_grad pass through cleanly
        rv = x - self.mu
        rv = rv * rv
        rv = rv * (-0.5 / self.sigma2)
        rv = rv + self.log_const
        rv = rv - x

        return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, sigma2: Any, engine: Any) -> Any:
        """Engine-neutral log-Gaussian log-density on log-encoded data."""
        return -0.5 * engine.log(engine.asarray(2.0 * pi) * sigma2) - 0.5 * (x - mu) * (x - mu) / sigma2 - x

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for log-encoded data."""
        xx = engine.asarray(x)
        mu = engine.asarray(self.mu)
        sigma2 = engine.asarray(self.sigma2)
        return self.backend_log_density_from_params(xx, mu, sigma2, engine)

    def gradient_log_prior(self, priors: Any, prior_strength: float, torch: Any, engine: Any) -> Any:
        """Distribution-owned MAP prior contribution for log-Gaussian parameters."""
        from mixle.stats.compute.gradient import normal_gamma_log_prior

        return normal_gamma_log_prior(self.mu, self.sigma2, priors, torch)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["LogGaussianDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked log-Gaussian parameters for a homogeneous mixture kernel."""
        return {
            "mu": engine.asarray([d.mu for d in dists]),
            "sigma2": engine.asarray([d.sigma2 for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of log-Gaussian log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["mu"][None, :], params["sigma2"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Return stacked log-Gaussian sufficient statistics using engine-resident arrays."""
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
        import math

        from scipy.stats import lognorm as _sp

        return float(_sp.cdf(x, self.sigma2**0.5, scale=math.exp(self.mu)))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        import math

        from scipy.stats import lognorm as _sp

        return float(_sp.ppf(q, self.sigma2**0.5, scale=math.exp(self.mu)))

    def to_fisher(self, **kwargs):
        """Return this distribution's own Fisher view."""
        return LogGaussianFisherView(self)

    def mean(self) -> float:
        """Mean exp(mu + sigma2/2)."""
        import math

        return float(math.exp(self.mu + 0.5 * self.sigma2))

    def variance(self) -> float:
        """Variance (exp(sigma2) - 1) * exp(2 mu + sigma2)."""
        import math

        return float((math.exp(self.sigma2) - 1.0) * math.exp(2.0 * self.mu + self.sigma2))

    def entropy(self) -> float:
        """Differential entropy mu + 0.5*log(2*pi*e*sigma2)."""
        import math

        return float(self.mu + 0.5 * (math.log(2.0 * math.pi * self.sigma2) + 1.0))

    def sampler(self, seed: int | None = None) -> "LogGaussianSampler":
        """Create a sampler from this log-Gaussian distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            LogGaussianSampler object.

        """
        return LogGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LogGaussianEstimator":
        """Return an estimator initialized from this distribution's parameters.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            LogGaussianEstimator object.

        """
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return LogGaussianEstimator(
                pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name, prior=self.prior
            )
        else:
            return LogGaussianEstimator(name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "LogGaussianDataEncoder":
        """Return the encoder for log-Gaussian observations."""
        return LogGaussianDataEncoder()


class LogGaussianSampler(DistributionSampler):
    """Draw independent samples from a :class:`LogGaussianDistribution`."""

    def __init__(self, dist: LogGaussianDistribution, seed: int | None = None) -> None:
        """LogGaussianSampler for drawing samples from LogGaussianSampler instance.

        Args:
            dist (LogGaussianDistribution): LogGaussianDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (LogGaussianDistribution): LogGaussianDistribution instance to sample from.
            rng (RandomState): RandomState with seed set to seed if passed in args.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw iid samples from the log-Gaussian distribution.

        ``None`` returns a single float. A positive ``size`` returns a NumPy
        array of that many samples on the original positive scale.

        Args:
            size (Optional[int]): Number of samples.

        Returns:
            Log-Gaussian samples.

        """
        return np.exp(self.rng.normal(loc=self.dist.mu, scale=np.sqrt(self.dist.sigma2), size=size))


class LogGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted log-scale moments for log-Gaussian estimation."""

    def __init__(self, keys: str | None = None, name: str | None = None) -> None:
        """Create an accumulator for log-Gaussian sufficient statistics.

        Args:
            keys (Optional[str]): Optional merge key for sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        Attributes:
            log_sum (float): Sum of weighted ``log(x)`` values.
            log_sum2 (float): Sum of weighted ``log(x) ** 2`` values.
            count (float): Sum of weights for first moments.
            count2 (float): Sum of weights for second moments.
            keys (Optional[str]): Optional merge key.
            name (Optional[str]): Optional accumulator name.
        """
        self.log_sum = 0.0
        self.log_sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional["LogGaussianDistribution"]) -> None:
        """Update sufficient statistics for LogGaussianAccumulator with one weighted observation.

        Args:
            x (float): Observation from log-Gaussian distribution.
            weight (float): Weight for observation.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        x_weight = np.log(x) * weight
        self.log_sum += x_weight
        self.log_sum2 += np.log(x) * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics with one weighted observation.

        Args:
            x (float): Observation from log-Gaussian distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of LogGaussianAccumulator sufficient statistics with weighted observations.

        This delegates to :meth:`seq_update`.

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: LogGaussianDistribution | None) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        Args:
            x (ndarray): Numpy array of floats.
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional['GaussianDistribution']): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.log_sum += np.dot(x, weights)
        self.log_sum2 += np.dot(x * x, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: tuple[float, float, float, float]) -> "LogGaussianAccumulator":
        """Aggregates sufficient statistics with LogGaussianAccumulator member sufficient statistics.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*log(X_i)),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*log(X_i)^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            suff_stat (Tuple[float, float, float, float]): See above for details.

        Returns:
            GaussianAccumulator object.

        """
        self.log_sum += suff_stat[0]
        self.log_sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self) -> tuple[float, float, float, float]:
        """Return sufficient statistics as ``(count, sum, sumsq, log_sum)``."""
        return self.log_sum, self.log_sum2, self.count, self.count2

    def from_value(self, x: tuple[float, float, float, float]) -> "LogGaussianAccumulator":
        """Assigns sufficient statistics of LogGaussianAccumulator instance to x.

        Arg passed x is tuple of four floats:
            x[0] (float): Sum of weighted observations (sum_i w_i*log(X_i)),
            x[1] (float): Sum of weighted observations (sum_i w_i*log(X_i)^2),
            x[2] (float): Sum of weighted observations (sum_i w_i),
            x[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            x: See above for details

        Returns:
            LogGaussianAccumulator object.

        """
        self.log_sum = x[0]
        self.log_sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def acc_to_encoder(self) -> "LogGaussianDataEncoder":
        """Return the encoder associated with this accumulator."""
        return LogGaussianDataEncoder()


class LogGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Create log-Gaussian accumulators with shared name and merge-key metadata."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        """Create a factory for log-Gaussian accumulators.

        Args:
            name (Optional[str]): Optional name assigned to created accumulators.
            keys (Optional[str]): Optional merge key for created accumulators.

        Attributes:
            name (Optional[str]): Optional accumulator name.
            keys (Optional[str]): Optional merge key.
        """
        self.keys = keys
        self.name = name

    def make(self) -> "LogGaussianAccumulator":
        """Return a fresh log-Gaussian accumulator with this factory's name and keys."""
        return LogGaussianAccumulator(name=self.name, keys=self.keys)


class LogGaussianEstimator(ParameterEstimator):
    """Estimate log-Gaussian location and variance from accumulated log moments."""

    def __init__(
        self,
        pseudo_count: float | tuple[float | None, float | None] = (None, None),
        suff_stat: tuple[float | None, float | None] = (None, None),
        min_covar: float | None = None,
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ):
        """Create an estimator for log-Gaussian location and variance.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two positive floats.
                A scalar is broadcast to both slots.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of float and positive float.
            name (Optional[str]): Assign a name to LogGaussianEstimator.
            keys (Optional[str]): Assign keys to LogGaussianEstimator for combining sufficient statistics.
            prior (Optional): Conjugate NormalGamma prior over (mu, tau=1/sigma2) of log(X). When present,
                ``estimate`` performs the closed-form conjugate posterior update (returning the joint
                MAP estimate and carrying the posterior forward as the fitted model's prior) instead
                of the maximum-likelihood / pseudo-count update.

        Attributes:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for suff_stat.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and variance (sigma2).
            name (Optional[str]): String name of LogGaussianEstimator instance.
            keys (Optional[str]): String keys of LogGaussianEstimator instance for combining sufficient statistics.

        """
        pseudo_count = broadcast_pseudo_count(pseudo_count, 2)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_covar = 1.0e-8 if min_covar is None else float(min_covar)
        self.keys = keys
        self.name = name
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalGammaDistribution)

    def accumulator_factory(self) -> "LogGaussianAccumulatorFactory":
        """Return GaussianAccumulatorFactory with name and keys passed."""
        return LogGaussianAccumulatorFactory(self.name, self.keys)

    def model_log_density(self, model: "LogGaussianDistribution") -> float:
        """Log-density of the model parameters under the NormalGamma prior (ELBO global term).

        The prior is over (mu, tau=1/sigma2) of log(X), so the model's (mu, sigma2) is mapped.
        """
        if self.has_conj_prior:
            return float(self.prior.log_density((model.mu, 1.0 / model.sigma2)))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[float, float, float, float]) -> "LogGaussianDistribution":
        """Closed-form NormalGamma conjugate posterior update on log-scale statistics."""
        sum_x, sum_xx, nobs_loc1, nobs_loc2 = suff_stat
        sum_xxx = sum_x
        old_mu, old_lam, old_a, old_b = self.prior.get_parameters()

        new_n = old_lam + nobs_loc1
        new_a = old_a + (nobs_loc2 / 2.0)

        sample_mean1 = sum_x / nobs_loc1 if nobs_loc1 > 0 else 0.0
        sample_mean2 = sum_xxx / nobs_loc2 if nobs_loc2 > 0 else 0.0

        new_mu = (sum_x + old_mu * old_lam) / (old_lam + nobs_loc1)

        new_b0 = sum_xx - sample_mean2 * sum_xxx
        new_b1 = (old_lam * nobs_loc1 / new_n) * np.power(sample_mean1 - old_mu, 2)
        new_b = old_b + 0.5 * (new_b0 + new_b1)

        new_sigma2 = new_b / (new_a - 0.5)
        new_sigma2 = new_sigma2 if new_sigma2 > 0 else 1.0
        new_prior = NormalGammaDistribution(new_mu, new_n, new_a, new_b)
        return LogGaussianDistribution(new_mu, new_sigma2, name=self.name, prior=new_prior)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float, float]) -> "LogGaussianDistribution":
        """Estimate a log-Gaussian distribution from accumulated log moments.

        ``suff_stat`` is ``(sum_log_x, sum_log_x2, count, count2)``. ``nobs``
        is accepted for estimator API consistency but is not used.
        """
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        log_x, log_x2 = suff_stat[0], suff_stat[1]
        nobs_loc1, nobs_loc2 = suff_stat[2], suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (log_x + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = self.min_covar
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1]) / (
                nobs_loc2 + self.pseudo_count[1]
            )
        else:
            # E[y^2] - E[y]^2 on the log scale (matches GaussianEstimator; the previous form was only
            # correct when the two observation counts were equal)
            sigma2 = log_x2 / nobs_loc2 - mu * mu

        sigma2 = max(sigma2, self.min_covar, 1.0e-6 * sigma2)
        return LogGaussianDistribution(mu, sigma2, name=self.name)


class LogGaussianDataEncoder(DataSequenceEncoder):
    """Data encoder for iid positive log-Gaussian observations."""

    def __str__(self) -> str:
        """Return the log-Gaussian encoder's display name."""
        return "LogGaussianDataEncoder"

    def __eq__(self, other) -> bool:
        """Return true when ``other`` is a log-Gaussian data encoder.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is an instance of a LogGaussianDataEncoder, else False.

        """
        return isinstance(other, LogGaussianDataEncoder)

    def seq_encode(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Encode sequence of iid Log-Gaussian observations.

        Data type must be List[float] or np.ndarray[float].

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid log-Gaussian observations.

        Returns:
            A numpy array of floats.

        """
        rv = np.asarray(np.log(x), dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise ValueError("LogGaussianDistribution requires support x in (0,inf).")
        return rv
