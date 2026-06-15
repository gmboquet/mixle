"""Create, estimate, and sample from a diagonal Gaussian distribution (independent-multivariate Gaussian).

Defines the DiagonalGaussianDistribution, DiagonalGaussianSampler, DiagonalGaussianAccumulatorFactory,
DiagonalGaussianAccumulator, DiagonalGaussianEstimator, and the DiagonalGaussianDataEncoder classes for use with
pysparkplug.

The log-density of an 'n' dimensional diagonal-gaussian observation x = (x_1,x_2,...,x_n) with mean mu=(m_1,m_2,..,m_n),
and diagonal covariance matrix given by covar = diag(s2_1, s2_2,...,s2_n).

    log(p_mat(x)) = -0.5*sum_{i=1}^{n} (x_i-m_i)^2 / s2_i - 0.5*log(s2_i) - (n/2)*log(pi).

Data type: x (List[float], np.ndarray).

"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.mvngamma import MultivariateNormalGammaDistribution
from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.aliasing import MISSING, coalesce_alias
from pysp.utils.special import digamma


class DiagonalGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Multivariate Gaussian distribution with independent components (diagonal covariance matrix)."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.declarations import DistributionDeclaration, ExponentialFamilySpec, ParameterSpec, StatisticSpec

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
        """Create a DiagonalGaussianDistribution object with mean mu and covariance covar.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean of Gaussian distribution.
            covar (Union[Sequence[float], np.ndarray]): Variance of each component.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for object isntance.
            prior (Optional): Conjugate parameter prior over (mu, tau=1/covar). A
                :class:`~pysp.stats.mvngamma.MultivariateNormalGammaDistribution` enables the
                Bayesian/variational machinery (``expected_log_density`` and the conjugate
                posterior update); ``None`` (default) is a plain point model.

        Attributes:
             dim (int): Dimension of the multivariate Gaussian. Determined by mean length.
             mu (np.ndarray): Mean of the Gaussian.
             covar (np.ndarray): Variance for each component.
             name (Optional[str]): Name of object instance.
             log_c (float): Normalizing constant for diagonal Gausisan.
             ca (np.ndarray): Term for likelihood-calc.
             cb (np.ndarray): Term for likelihood-calc.
             cc (np.ndarray): Term for likelihood-calc.
             key (Optional[str]): Key for merging sufficient statistics.

        """
        covar = coalesce_alias("covar", covar, "covariance", covariance, default=MISSING)
        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.name = name
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c
        self.key = keys

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
        """Returns string representation of DiagonalGaussianDistribution object."""
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

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-dim observation vector.

        Returns:
            Log-density at x.

        """
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of the log-density at a sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, dim) from
                DiagonalGaussianDataEncoder.seq_encode().

        Returns:
            Numpy array of length sz containing the log-density of each encoded observation.

        """
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
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
            engine.asarray(x), engine.asarray(self.mu), engine.asarray(self.covar), engine
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

    def sampler(self, seed: int | None = None) -> "DiagonalGaussianSampler":
        """Create a DiagonalGaussianSampler for sampling from this distribution.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            DiagonalGaussianSampler object.

        """
        return DiagonalGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "DiagonalGaussianEstimator":
        """Create a DiagonalGaussianEstimator for estimating this distribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            DiagonalGaussianEstimator object.

        """
        if pseudo_count is None:
            return DiagonalGaussianEstimator(name=self.name, keys=self.key, prior=self.prior)
        else:
            return DiagonalGaussianEstimator(
                pseudo_count=(pseudo_count, pseudo_count), name=self.name, keys=self.key, prior=self.prior
            )

    def dist_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Returns a DiagonalGaussianDataEncoder object for encoding sequences of iid observations."""
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianSampler(DistributionSampler):
    """DiagonalGaussianSampler object for sampling from a DiagonalGaussianDistribution."""

    def __init__(self, dist: DiagonalGaussianDistribution, seed: int | None = None) -> None:
        """DiagonalGaussianSampler object for sampling from DiagonalGaussian instance.

        Args:
            dist (DiagonalGaussianDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            dist (DiagonalGaussianDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> Sequence[np.ndarray] | np.ndarray:
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
        else:
            return [self.sample() for i in range(size)]


class DiagonalGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """DiagonalGaussianAccumulator object for aggregating sufficient statistics from iid observations."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        """DiagonalGaussianAccumulator object for aggregating sufficient statistics from iid observations.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
             dim (Optional[int]): Optional dimension of Gaussian.
             count (float): Used for tracking weighted observations counts.
             sum (np.ndarray): Sum of observation vectors.
             sum2 (np.ndarray): Sum of squared observation vectors.
             key (Optional[str]): If set, merge sufficient statistics with objects containing matching keys.

        """
        self.dim = dim
        self.count = 0.0
        self.sum = vec.zeros(dim) if dim is not None else None
        self.sum2 = vec.zeros(dim) if dim is not None else None
        self.key = keys

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
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = x * weight
        self.count += weight
        self.sum += x_weight
        x_weight *= x
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
            self.dim = len(x[0])
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = np.multiply(x.T, weights)
        self.count += weights.sum()
        self.sum += x_weight.sum(axis=1)
        x_weight *= x.T
        self.sum2 += x_weight.sum(axis=1)

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
        """Merge the sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of squared observations, sum of weights).

        Returns:
            DiagonalGaussianAccumulator object.

        """
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns the sufficient statistics (sum, sum of squares, count) of the accumulator."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "DiagonalGaussianAccumulator":
        """Set the sufficient statistics of the accumulator to x.

        Args:
            x (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of squared observations, sum of weights).

        Returns:
            DiagonalGaussianAccumulator object.

        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Combine sufficient statistics with other accumulators sharing a matching key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to aggregated statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key].value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace sufficient statistics with values from stats_dict for a matching key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to aggregated statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key])

    def acc_to_encoder(self) -> "DiagonalGaussianDataEncoder":
        """Returns a DiagonalGaussianDataEncoder object for encoding sequences of iid observations."""
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """DiagonalGaussianAccumulatorFactory object for creating DiagonalGaussianAccumulator objects."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        """DiagonalGaussianAccumulatorFactory object for creating DiagonalGaussianAccumulator objects.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.

        Attributes:
             dim (Optional[int]): Optional dimension of Gaussian.
             key (Optional[str]): If set, merge sufficient statistics with objects containing matching keys.

        """
        self.dim = dim
        self.key = keys

    def make(self) -> "DiagonalGaussianAccumulator":
        """Returns a new DiagonalGaussianAccumulator with the factory's dim and keys."""
        return DiagonalGaussianAccumulator(dim=self.dim, keys=self.key)


class DiagonalGaussianEstimator(ParameterEstimator):
    """DiagonalGaussianEstimator object for estimating a diagonal Gaussian distribution from
    aggregated sufficient statistics."""

    def __init__(
        self,
        dim: int | None = None,
        pseudo_count: tuple[float | None, float | None] = (None, None),
        suff_stat: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """DiagonalGaussianEstimator object for estimating diagonal Gaussian distributions from aggregated sufficient
            statistics.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the sum of observations and sum of
                squared observations in estimation.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray]]): Sum of observations and sum of squared
                observations both having same dimension.
            name (Optinal[str]): Set name for object instance.
            keys (Optional[str]): Set keys for merging sufficient statistics.
            prior (Optional): Conjugate MultivariateNormalGamma prior over (mu, tau=1/covar). When present,
                ``estimate`` performs the closed-form per-component conjugate posterior update (returning the
                joint MAP estimate and carrying the posterior forward as the fitted model's prior) instead
                of the maximum-likelihood / pseudo-count update.

        Attributes:
            name (Optinal[str]): Name for object instance.
            dim (int): Dimension of Gaussian, either set of determined from suff_stat arg.
            prior_mu (Optional[np.ndarray]): Set from suff_stat[0].
            prior_covar ((Optional[np.ndarray]): Set from suff_stat[1].
            pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the sum of observations and sum of
                squared observations in estimation.
            keys (Optional[str]): Key for merging sufficient statistics.

        """
        dim_loc = (
            dim
            if dim is not None
            else (
                (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1]))))
                if suff_stat[0] is None
                else len(suff_stat[0])
            )
        )

        self.name = name
        self.dim = dim_loc
        self.pseudo_count = pseudo_count
        self.prior_mu = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.prior_covar = None if suff_stat[1] is None else np.reshape(suff_stat[1], dim_loc)
        self.key = keys
        self.prior = prior
        self.has_conj_prior = isinstance(prior, MultivariateNormalGammaDistribution)

    def accumulator_factory(self) -> "DiagonalGaussianAccumulatorFactory":
        """Returns a DiagonalGaussianAccumulatorFactory built from the estimator's attributes."""
        return DiagonalGaussianAccumulatorFactory(dim=self.dim, keys=self.key)

    def model_log_density(self, model: "DiagonalGaussianDistribution") -> float:
        """Log-density of the model parameters under the MultivariateNormalGamma prior (ELBO global term).

        The prior is over (mu, tau=1/covar), so the model's covariance is inverted before scoring.
        """
        if self.has_conj_prior:
            return float(self.prior.log_density((model.mu, 1.0 / model.covar)))
        return 0.0

    def _estimate_conjugate(self, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> "DiagonalGaussianDistribution":
        """Closed-form per-component NormalGamma conjugate posterior update returning the joint MAP estimate."""
        sum_x, sum_xx, nobs_loc1 = suff_stat
        sum_xxx = sum_x
        nobs_loc2 = nobs_loc1

        old_mu, old_lam, old_a, old_b = self.prior.get_parameters()

        new_n = old_lam + nobs_loc1
        new_a = old_a + (nobs_loc2 / 2.0)

        if nobs_loc1 > 0:
            sample_mean1 = sum_x / nobs_loc1
        else:
            sample_mean1 = 0

        if nobs_loc2 > 0:
            sample_mean2 = sum_xxx / nobs_loc2
        else:
            sample_mean2 = 0

        new_mu = (sum_x + old_mu * old_lam) / (old_lam + nobs_loc1)

        new_b0 = sum_xx - sample_mean2 * sum_xxx
        new_b1 = (old_lam * nobs_loc1 / new_n) * np.power(sample_mean1 - old_mu, 2)
        new_b = old_b + 0.5 * (new_b0 + new_b1)

        new_sigma2 = new_b / (new_a - 0.5)

        new_prior = MultivariateNormalGammaDistribution(new_mu, new_n, new_a, new_b)
        return DiagonalGaussianDistribution(new_mu, new_sigma2, name=self.name, prior=new_prior)

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
        if self.has_conj_prior:
            return self._estimate_conjugate(suff_stat)

        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (suff_stat[1] + (pc2 * self.prior_covar) - (mu * mu * nobs)) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - (mu * mu)

        return DiagonalGaussianDistribution(mu, covar, name=self.name)


class DiagonalGaussianDataEncoder(DataSequenceEncoder):
    """DiagonalGaussianDataEncoder object for encoding sequences of iid diagonal-Gaussian observations."""

    def __init__(self, dim: int | None = None) -> None:
        """DiagonalGaussianDataEncoder object.

        Args:
            dim (Optional[int]): Optional dimension of the Gaussian. Inferred from data if None.

        """
        self.dim = dim

    def __str__(self) -> str:
        """Returns string representation of DiagonalGaussianDataEncoder object."""
        return "DiagonalGaussianDataEncoder(dim=" + str(self.dim) + ")"

    def __eq__(self, other: object) -> bool:
        """Checks if other object is a DiagonalGaussianDataEncoder with the same dim.

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
        """Encode a sequence of iid length-dim observations for vectorized 'seq_' calls.

        Args:
            x (Sequence[Union[List[float], np.ndarray]]): Sequence of length-dim observation vectors.

        Returns:
            Encoded data matrix with shape (len(x), dim).

        """
        if self.dim is None:
            self.dim = len(x[0])
        xv = np.reshape(x, (-1, self.dim))
        return xv
