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
from mixle.utils.aliasing import MISSING, broadcast_pseudo_count, coalesce_alias
from mixle.utils.special import digamma


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
             ca: Quadratic scoring coefficient.
             cb: Linear scoring coefficient.
             cc: Constant scoring coefficient.
             keys: Optional sufficient-statistic key.

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

        Args:
            x (Union[Sequence[float], np.ndarray]): Length-dim observation vector.

        Returns:
            Log-density at x.

        """
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

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
        return DiagonalGaussianDistribution(self.mu[unobs], self.covar[unobs])

    def marginal(self, keep: Sequence[int]) -> "DiagonalGaussianDistribution":
        """Return the marginal over the dimensions ``keep``: ``DiagonalGaussian(mu[keep], covar[keep])``.

        Marginalizing a diagonal Gaussian simply drops the other independent coordinates (order kept).
        """
        idx = np.asarray(list(keep), dtype=int)
        if idx.size == 0:
            raise ValueError("keep at least one dimension")
        if idx.min() < 0 or idx.max() >= self.dim:
            raise ValueError("kept indices must be in [0, dim)")
        return DiagonalGaussianDistribution(self.mu[idx], self.covar[idx])

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
                pseudo_count=(pseudo_count, pseudo_count), name=self.name, keys=self.keys, prior=self.prior
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
        # Vectorized: randn(size, dim) fills row-major, so row i equals the i-th per-draw randn(dim);
        # bit-identical to the loop, far faster.
        rv = self.rng.randn(int(size), self.dist.dim) * np.sqrt(self.dist.covar) + self.dist.mu
        return list(rv)


class DiagonalGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for diagonal Gaussian sufficient statistics."""

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
        self.dim = dim
        self.count = 0.0
        self.sum = vec.zeros(dim) if dim is not None else None
        self.sum2 = vec.zeros(dim) if dim is not None else None
        self.keys = keys

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
        """Merge sufficient statistics into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of squared observations, sum of weights).

        Returns:
            This accumulator.

        """
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            # copy on adopt: value() hands out the LIVE arrays, so adopting the caller's reference
            # makes every later in-place += here mutate the DONOR accumulator too (chunk combines
            # and keyed pooling both hit this -- caught by the keyed-protocol sweep)
            self.sum = np.asarray(suff_stat[0], dtype=np.float64).copy()
            self.sum2 = np.asarray(suff_stat[1], dtype=np.float64).copy()
            self.count = suff_stat[2]

        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(sum, sum_squares, count)`` sufficient statistics."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "DiagonalGaussianAccumulator":
        """Replace this accumulator's sufficient statistics.

        Args:
            x (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of squared observations, sum of weights).

        Returns:
            This accumulator.

        """
        self.sum = None if x[0] is None else np.asarray(x[0], dtype=np.float64).copy()
        self.sum2 = None if x[1] is None else np.asarray(x[1], dtype=np.float64).copy()
        self.count = x[2]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Combine sufficient statistics with other accumulators sharing a matching key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to aggregated statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                self.combine(stats_dict[self.keys].value())
                # write the POOL back: the dict must end holding the pooled accumulator, else
                # key_replace hands every tied site the FIRST site's statistics (later sites'
                # data silently discarded -- caught by the keyed-protocol sweep)
                stats_dict[self.keys] = self
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace sufficient statistics with values from stats_dict for a matching key.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to aggregated statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                # the dict holds the pooled ACCUMULATOR (see key_merge); passing it whole made
                # from_value subscript an accumulator object -- keyed use crashed with TypeError
                self.from_value(stats_dict[self.keys].value())

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
        self.dim = dim
        self.keys = keys

    def make(self) -> "DiagonalGaussianAccumulator":
        """Return a fresh accumulator with the factory configuration."""
        return DiagonalGaussianAccumulator(dim=self.dim, keys=self.keys)


class DiagonalGaussianEstimator(ParameterEstimator):
    """Estimator for diagonal Gaussian distributions."""

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
                ``None`` (default) uses a tiny ``1e-8``. Negatives / NaNs are clamped to this floor.
            ridge (Optional[float]): Relative variance floor coefficient. ``None`` (default) uses ``1e-6``;
                each coordinate variance is floored at ``max(min_covar, ridge * mean(var))`` so the
                safeguard is data-scaled. Bias is negligible at the defaults.

        Attributes:
            name: Optional diagnostic name.
            dim: Gaussian dimension.
            prior_mu: Prior mean used for smoothing.
            prior_covar: Prior variance used for smoothing.
            pseudo_count: Smoothing counts for mean and variance.
            keys: Optional sufficient-statistic key.

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
        pseudo_count = broadcast_pseudo_count(pseudo_count, 2)
        self.pseudo_count = pseudo_count
        self.prior_mu = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.prior_covar = None if suff_stat[1] is None else np.reshape(suff_stat[1], dim_loc)
        self.keys = keys
        self.prior = prior
        self.has_conj_prior = isinstance(prior, MultivariateNormalGammaDistribution)
        self.min_covar = 1.0e-8 if min_covar is None else float(min_covar)
        self.ridge = 1.0e-6 if ridge is None else float(ridge)

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

        # Per-coordinate scatter ``sum_xx - (sum_x)^2/n`` is cancellation-prone (see GaussianEstimator):
        # floor it at 0 so a near-constant coordinate cannot drive ``new_b``/variance negative, which the
        # diagonal Gaussian's constructor does NOT validate (silent NaN log-density otherwise).
        new_b0 = np.maximum(sum_xx - sample_mean2 * sum_xxx, 0.0)
        new_b1 = (old_lam * nobs_loc1 / new_n) * np.power(sample_mean1 - old_mu, 2)
        new_b = old_b + 0.5 * (new_b0 + new_b1)

        denom = new_a - 0.5  # per-coordinate array
        safe_denom = np.where(denom > 0.0, denom, 1.0)
        new_sigma2 = np.where(denom > 0.0, new_b / safe_denom, self.min_covar)
        new_sigma2 = np.maximum(new_sigma2, self.min_covar)  # match the MLE-path variance floor

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

        if nobs <= 0:
            # zero-responsibility component: fall back to the prior mean (or zeros)
            # rather than dividing by zero and emitting a NaN mean.
            d = self.dim if self.dim is not None else len(suff_stat[0])
            mu = np.asarray(self.prior_mu, dtype=float) if self.prior_mu is not None else vec.zeros(d)
            covar = np.asarray(self.prior_covar, dtype=float) if self.prior_covar is not None else np.ones(d)
            floor = max(self.min_covar, self.ridge * float(np.mean(covar[covar > 0.0])) if np.any(covar > 0.0) else 0.0)
            covar = np.maximum(covar, floor)
            return DiagonalGaussianDistribution(mu, covar, name=self.name)

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (suff_stat[1] + (pc2 * self.prior_covar) - (mu * mu * nobs)) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - (mu * mu)

        # P1 variance floor: clamp non-finite / non-positive coordinates and apply a
        # data-scaled floor max(min_covar, ridge * mean(var)) so a component holding
        # few points cannot produce zero/negative/NaN variances. Bias is negligible.
        covar = np.asarray(covar, dtype=float)
        finite = np.isfinite(covar)
        if not finite.all():
            covar = np.where(finite, covar, self.min_covar)
        floor = max(self.min_covar, self.ridge * float(np.mean(covar[covar > 0.0])) if np.any(covar > 0.0) else 0.0)
        covar = np.maximum(covar, floor)

        return DiagonalGaussianDistribution(mu, covar, name=self.name)


class DiagonalGaussianDataEncoder(DataSequenceEncoder):
    """Encoder for iid diagonal Gaussian observations."""

    def __init__(self, dim: int | None = None) -> None:
        """Create an encoder with an optional fixed dimension.

        Args:
            dim: Optional Gaussian dimension. Inferred from data when omitted.

        """
        self.dim = dim

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
