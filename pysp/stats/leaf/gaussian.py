"""Evaluate, estimate, and sample from a gaussian distribution with mean mu and variance sigma2.

Defines the GaussianDistribution, GaussianSampler, GaussianAccumulatorFactory, GaussianAccumulator,
GaussianEstimator, and the GaussianDataEncoder classes for use with pysparkplug.

Data type: (float): The GaussianDistribution with mean mu and variance sigma2 > 0.0, has log-density
    log(f(x;mu, sigma2)) = -0.5*log(2*pi*sigma2) - 0.5*(x-mu)^2/sigma2, for real-valued x.

"""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.stats.bayes.normal_gamma import NormalGammaDistribution
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.fisher import FixedFisherView
from pysp.utils.special import digamma


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
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.compute.declarations import (
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
        """GaussianDistribution object defines Gaussian distribution with mean mu and variance sigma2.

        Args:
            mu (float): Real-valued number.
            sigma2 (float): Positive real-valued number.
            name (Optional[str]): String for name of object.
            prior (Optional): Conjugate parameter prior over (mu, tau=1/sigma2). A
                :class:`~pysp.stats.bayes.normal_gamma.NormalGammaDistribution` enables the
                Bayesian/variational machinery (``expected_log_density`` and the
                conjugate posterior update); ``None`` (default) is a plain point model.

        Attributes:
            mu (float): Mean of gaussian distribution.
            sigma2 (float): Variance of Gaussian distribution.
            name (Optional[str]): String for name of object.
            cont (float): Normalizing constant of Gaussian (depends on sigma2).
            log_const (float): Log of above.

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
        """Returns string representation of GaussianDistribution object."""
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
        from pysp.stats.compute.gradient import normal_gamma_log_prior

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

    def sampler(self, seed: int | None = None) -> "GaussianSampler":
        """Create an GaussianSampler object from parameters of GaussianDistribution instance.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            GaussianSampler object.

        """
        return GaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GaussianEstimator":
        """Create GaussianEstimator with mu and sigma2 passed if pseudo_count is not None.

        Arg variable pseudo_count is used to pass and re-weight mu and sigma2 of GaussianDistribution instance. Simply
        creates a GaussianEstimator with name passed if pseudo_count is None.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            GaussianEstimator object.

        """
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return GaussianEstimator(
                pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name, prior=self.prior
            )
        else:
            return GaussianEstimator(name=self.name, prior=self.prior)

    def dist_to_encoder(self) -> "GaussianDataEncoder":
        """Returns a GaussianDataEncoder object for encoding sequences of data."""
        return GaussianDataEncoder()


class GaussianSampler(DistributionSampler):
    def __init__(self, dist: GaussianDistribution, seed: int | None = None) -> None:
        """GaussianSampler for drawing samples from GaussianSampler instance.

        Args:
            dist (GaussianDistribution): GaussianDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        Attributes:
            dist (GaussianDistribution): GaussianDistribution instance to sample from.
            rng (RandomState): RandomState with seed set to seed if passed in args.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw 'size' iid samples from GaussianSampler object.

        Numpy array of length 'size' from Gaussian distribution with mean mu and scale sigma2 if size not None.
        Else a single sample is returned as float.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            'size' iid samples from Gaussian distribution.

        """
        return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self, keys: str | None = None, name: str | None = None) -> None:
        """GaussianAccumulator object used to accumulate sufficient statistics from observed data.

        Args:
            keys (Optional[str]): Set key for GaussianAccumulator object.
            name (Optional[str]): Set name for GaussianAccumulator object.

        Attributes:
            sum (float): Sum of weighted observations (sum_i w_i*X_i).
            sum2 (float): Sum of weighted squared observations (sum_i w_i*X_i^2)
            count (float): Sum of weights for observations (sum_i w_i).
            count2 (float): Sum of weights for squared observations (sum_i w_i).
            count (float): Tracks the sum of weighted observations used to form sum.
            key (Optional[str]): Key string used to aggregate all sufficient statistics with same keys values.
            name (Optional[str]): Name for GaussianAccumulator object.

        """
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

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
        self.sum += x_weight
        self.sum2 += x * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize GaussianAccumulator object with weighted observation

        Note: Just calls update().

        Args:
            x (float): Observation from Gaussian distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of GaussianAccumulator sufficient statistics with weighted observations.

        Note: Just calls seq_update().

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
        self.sum += np.dot(x, weights)
        self.sum2 += np.dot(x * x, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: tuple[float, float, float, float]) -> "GaussianAccumulator":
        """Aggregates sufficient statistics with GaussianAccumulator member sufficient statistics.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            suff_stat (Tuple[float, float, float, float]): See above for details.

        Returns:
            GaussianAccumulator object.

        """
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self) -> tuple[float, float, float, float]:
        """Returns sufficient statistics of GaussianAccumulator object (Tuple[float, float, float, float])."""
        return self.sum, self.sum2, self.count, self.count2

    def from_value(self, x: tuple[float, float, float, float]) -> "GaussianAccumulator":
        """Assigns sufficient statistics of GaussianAccumulator instance to x.

        Arg passed x is tuple of four floats:
            x[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            x[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            x[2] (float): Sum of weighted observations (sum_i w_i),
            x[3] (float): Sum of weighted observations (sum_i w_i).

        Args:
            x: See above for deatils.

        Returns:
            GaussianAccumulator object.

        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2, x3 = stats_dict[self.keys]
                self.sum += x0
                self.sum2 += x1
                self.count += x2
                self.count2 += x3

            else:
                stats_dict[self.keys] = (self.sum, self.sum2, self.count, self.count2)

    def key_replace(self, stats_dict: dict[str, "GaussianAccumulator"]) -> None:
        """Set sufficient statistics of object instance to suff_stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.sum, self.sum2, self.count, self.count2 = stats_dict[self.keys]

    def acc_to_encoder(self) -> "GaussianDataEncoder":
        """Returns a GaussianDataEncoder object for encoding sequences of data."""
        return GaussianDataEncoder()


class GaussianAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        """GaussianAccumulatorFactory object for creating GaussianAccumulator.

        Args:
            name (Optional[str]): Assign a name to GaussianAccumulatorFactory object.
            keys (Optional[str]): Assign keys member for GaussianAccumulators.

        Attributes:
            name (Optional[str]): Name of the GaussianAccumulatorFactory obejct.
            keys (Optional[str]): String id for merging sufficient statistics of GaussianAccumulator.

        """
        self.keys = keys
        self.name = name

    def make(self) -> "GaussianAccumulator":
        """Return a GaussianAccumulator object with name and keys passed."""
        return GaussianAccumulator(name=self.name, keys=self.keys)


class GaussianEstimator(ParameterEstimator):
    def __init__(
        self,
        pseudo_count: tuple[float | None, float | None] = (None, None),
        suff_stat: tuple[float | None, float | None] = (None, None),
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
        min_covar: float | None = None,
    ):
        """GaussianEstimator object used to estimate GaussianDistribution from aggregated sufficient statistics.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two positive floats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of float and positive float.
            name (Optional[str]): Assign a name to GaussianEstimator.
            keys (Optional[str]): Assign keys to GaussianEstimator for combining sufficient statistics.
            prior (Optional): Conjugate NormalGamma prior over (mu, tau=1/sigma2). When present,
                ``estimate`` performs the closed-form conjugate posterior update (returning the joint
                MAP estimate and carrying the posterior forward as the fitted model's prior) instead
                of the maximum-likelihood / pseudo-count update.
            min_covar (Optional[float]): Absolute variance floor applied in the MLE M-step. ``None``
                (default) uses a tiny ``1e-8`` floor; the estimated variance is also floored at a
                relative ``1e-6 * sigma2`` to keep the safeguard data-scaled. Set explicitly to widen
                the floor for hard / high-dimensional cases. Bias is negligible at the default.

        Attributes:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for suff_stat.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and variance (sigma2).
            name (Optional[str]): String name of GaussianEstimator instance.
            keys (Optional[str]): String keys of GaussianEstimator instance for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalGammaDistribution)
        self.min_covar = 1.0e-8 if min_covar is None else float(min_covar)

    def accumulator_factory(self) -> "GaussianAccumulatorFactory":
        """Return GaussianAccumulatorFactory with name and keys passed."""
        return GaussianAccumulatorFactory(self.name, self.keys)

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

        new_b0 = sum_xx - sample_mean2 * sum_xxx
        new_b1 = (old_lam * nobs_loc1 / new_n) * np.power(sample_mean1 - old_mu, 2)
        new_b = old_b + 0.5 * (new_b0 + new_b1)

        new_sigma2 = new_b / (new_a - 0.5)
        new_prior = NormalGammaDistribution(new_mu, new_n, new_a, new_b)
        return GaussianDistribution(new_mu, new_sigma2, name=self.name, prior=new_prior)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float, float]) -> "GaussianDistribution":
        """Estimate a GaussianDistribution object from sufficient statistics aggregated from data.

        Arg passed suff_stat is tuple of four floats:
            suff_stat[0] (float): Sum of weighted observations (sum_i w_i*X_i),
            suff_stat[1] (float): Sum of weighted observations (sum_i w_i*X_i^2),
            suff_stat[2] (float): Sum of weighted observations (sum_i w_i),
            suff_stat[3] (float): Sum of weighted observations (sum_i w_i),\

        obtained from aggregation of observations.

        If member variable pseudo_count is not None, suff_stat is combined with re-weighted member instance variables
        suff_stat. If pseudo_count is None, arg suff_stat is used to form maximum likelihood estimates for mu and
        sigma2 of GaussianDistribution object.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See above for details.

        Returns:
            GaussianDistribution object.

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
    """GaussianDataEncoder object for encoding sequences of iid Gaussian observations with data type float."""

    def __str__(self) -> str:
        """Returns string representation of GaussianDataEncoder object."""
        return "GaussianDataEncoder"

    def __eq__(self, other) -> bool:
        """Checks if other object is an instance of a GaussianDataEncoder.

        Args:
            other (object): Obejct to compare.

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
            raise Exception("GaussianDistribution requires support x in (-inf,inf).")
        return rv
