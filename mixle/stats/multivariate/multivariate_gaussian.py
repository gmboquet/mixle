"""Multivariate Gaussian distributions over real-valued vectors.

Data type: np.ndarray[float]

x = (x_1,x_2,..,x_n) ~ MVN(mu, covar), where mu is a length n numpy array, and covar is an n by n positive definite
covariance matrix.

The log-density is given by
    log(p(x)) = -0.5*k*log(2*pi) - 0.5*log|covar| - 0.5*(x-mu)' covar^{-1} (x-mu).

Reference: Mardia, Kent & Bibby, *Multivariate Analysis* (Academic Press, 1979).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.linalg
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import *
from mixle.inference.fisher import FixedFisherView
from mixle.stats.bayes.normal_wishart import NormalWishartDistribution
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.aliasing import MISSING, broadcast_pseudo_count, coalesce_alias


def _robust_cho_factor(covar: np.ndarray):
    """Cholesky-factor a covariance, self-healing a covariance that lost positive-definiteness.

    A sample covariance ``E[xx^T] - mu mu^T`` is PD in exact arithmetic, but float32 accumulation (GPU /
    MPS engines) can lose PD-ness to catastrophic cancellation, and a near-empty EM component can go
    singular. The fast path (a genuinely PD float64 covariance) is UNCHANGED. On failure, symmetrize
    (fixes float32 asymmetry) and add a trace-scaled jitter, escalating until PD -- so the fit proceeds
    instead of crashing at ``cho_factor``. The jitter is minimal (starts at 1e-10 * mean-diagonal)."""
    try:
        return scipy.linalg.cho_factor(covar)
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        sym = 0.5 * (covar + covar.T)
        scale = float(np.trace(sym)) / max(sym.shape[0], 1)
        scale = scale if np.isfinite(scale) and scale > 0 else 1.0
        eye = np.eye(sym.shape[0])
        jitter = 1e-10 * scale
        for _ in range(12):
            try:
                return scipy.linalg.cho_factor(sym + jitter * eye)
            except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
                jitter *= 10.0
        raise


class MultivariateGaussianFisherView(FixedFisherView):
    """Fisher view over first and upper-triangular second moments for a full Gaussian."""

    def __init__(self, dist: Any) -> None:
        self.dim = int(dist.dim if hasattr(dist, "dim") else len(dist.mu))
        self._tri = np.triu_indices(self.dim)
        labels = [("sum", str(i)) for i in range(self.dim)]
        labels.extend(("sum2", str(i), str(j)) for i, j in zip(self._tri[0], self._tri[1]))
        labels.append(("count",))
        super().__init__(dist, labels)

    def _as_matrix(self, data: Any) -> np.ndarray:
        return np.asarray(data, dtype=np.float64).reshape((-1, self.dim))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        x = self._as_matrix(data)
        i, j = self._tri
        xx = x[:, i] * x[:, j]
        return np.hstack((x, xx, np.ones((x.shape[0], 1), dtype=np.float64)))

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        return self._statistics_from_data(np.asarray(enc_data, dtype=np.float64), estimate=estimate)

    def _model_mean(self) -> np.ndarray:
        mu = np.asarray(self.dist.mu, dtype=np.float64).reshape(-1)
        cov = np.asarray(self.dist.covar, dtype=np.float64).reshape((self.dim, self.dim))
        second = cov + np.outer(mu, mu)
        return np.concatenate((mu, second[self._tri], np.asarray([1.0])))

    def _model_fisher(self) -> np.ndarray:
        mu = np.asarray(self.dist.mu, dtype=np.float64).reshape(-1)
        cov = np.asarray(self.dist.covar, dtype=np.float64).reshape((self.dim, self.dim))
        dim = self.dim
        pairs = list(zip(self._tri[0], self._tri[1]))
        m = len(pairs)
        out = np.zeros((dim + m + 1, dim + m + 1), dtype=np.float64)
        out[:dim, :dim] = cov

        for a, (j, k) in enumerate(pairs):
            col = dim + a
            cross = mu[j] * cov[:, k] + mu[k] * cov[:, j]
            out[:dim, col] = cross
            out[col, :dim] = cross

        for a, (i, j) in enumerate(pairs):
            ia = dim + a
            for b, (k, l) in enumerate(pairs):
                ib = dim + b
                out[ia, ib] = (
                    cov[i, k] * cov[j, l]
                    + cov[i, l] * cov[j, k]
                    + mu[i] * mu[k] * cov[j, l]
                    + mu[i] * mu[l] * cov[j, k]
                    + mu[j] * mu[k] * cov[i, l]
                    + mu[j] * mu[l] * cov[i, k]
                )

        return 0.5 * (out + out.T)


class MultivariateGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Multivariate normal distribution with mean vector mu and full covariance matrix covar."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for multivariate Gaussian generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the multivariate Gaussian."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="multivariate_gaussian",
            distribution_type=cls,
            parameters=(
                ParameterSpec("mu", constraint="real_vector"),
                ParameterSpec("inv_covar", constraint="positive_matrix", differentiable=False),
                ParameterSpec("log_det", differentiable=False),
                ParameterSpec("dim", constraint="fixed", differentiable=False),
            ),
            statistics=(
                StatisticSpec("sum", kind="vector_moment"),
                StatisticSpec("sum2", kind="matrix_moment"),
                StatisticSpec("count"),
            ),
            support="real_vector",
            differentiable=False,
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: Any, engine: Any) -> tuple[Any, ...]:
        """Return vector/matrix sufficient statistics for generated MVN scoring."""
        xx = engine.asarray(x)
        return xx, xx[:, :, None] * xx[:, None, :]

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return natural parameters for generated MVN scoring."""
        mu = params["mu"]
        inv_covar = params["inv_covar"]
        eta1 = engine.matmul(inv_covar, mu[..., None])[..., 0]
        eta2 = engine.asarray(-0.5) * inv_covar
        return eta1, eta2

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the full-covariance Gaussian log partition."""
        mu = params["mu"]
        eta1 = engine.matmul(params["inv_covar"], mu[..., None])[..., 0]
        quad = engine.sum(mu * eta1, axis=-1)
        return engine.asarray(0.5) * (
            quad + params["log_det"] + engine.asarray(float(params["dim"])) * engine.log(engine.asarray(2.0 * pi))
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return row-wise legacy accumulator statistics for generated resident reductions."""
        xx = engine.asarray(x)
        one = engine.sum(xx * 0.0, axis=1) + engine.asarray(1.0)
        return xx, xx[:, :, None] * xx[:, None, :], one

    def __init__(
        self,
        mu: list[float] | np.ndarray,
        covar: list[list[float]] | np.ndarray = MISSING,
        name: str | None = None,
        keys: str | None = None,
        covariance: list[list[float]] | np.ndarray = MISSING,
        prior: SequenceEncodableProbabilityDistribution | None = None,
    ) -> None:
        """Create a multivariate Gaussian distribution.

        Args:
            mu: Mean vector.
            covar: Positive-definite covariance matrix. ``covariance`` is
                accepted as an alias.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior (Optional): Conjugate parameter prior over (mu, Lambda=covar^-1). A
                :class:`~mixle.stats.bayes.normal_wishart.NormalWishartDistribution` enables the
                Bayesian/variational machinery (``expected_log_density`` and the conjugate
                posterior update); ``None`` (default) is a plain point model.

        Attributes:
            dim: Dimension of the Gaussian.
            mu: Mean vector.
            covar: Covariance matrix.
            chol: Cholesky factor when available.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.
            use_lstsq: Whether scoring falls back to least-squares solves.
            chol_const: Log-normalization term used by the scoring path.

        """
        covar = coalesce_alias("covar", covar, "covariance", covariance, default=MISSING)
        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.covar = np.reshape(self.covar, (len(self.mu), len(self.mu)))
        self.chol = _robust_cho_factor(self.covar)
        self.name = name
        self.keys = keys

        if self.chol is None:
            raise RuntimeError("Cannot obtain Choleskey factorization for covariance matrix.")
        else:
            self.use_lstsq = False
            self.log_det = float(2.0 * np.log(vec.diag(self.chol[0])).sum())
            self.inv_covar = scipy.linalg.cho_solve(self.chol, np.eye(self.dim))
            self.chol_const = -0.5 * (len(self.mu) * np.log(2.0 * pi) + self.log_det)

        self.set_prior(prior)

    def set_prior(self, prior: SequenceEncodableProbabilityDistribution | None) -> None:
        """Attach a parameter prior and precompute conjugate-prior expectations.

        With a NormalWishart(m0, kappa, W, nu) prior over (mu, Lambda=covar^-1) this
        caches the prior parameters and E[ln|Lambda|], the quantities needed by
        ``expected_log_density``. Any other prior (including ``None``) leaves the
        distribution a plain point model.
        """
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalWishartDistribution)

        if self.has_conj_prior:
            self.conj_prior_params = prior.get_parameters()
            self.e_log_det = prior.expected_log_det()
        else:
            self.conj_prior_params = None
            self.e_log_det = None

    def expected_log_density(self, x) -> float:
        """Variational expectation E_q[log p(x | mu, Lambda)] under the NormalWishart prior.

        Falls back to the plug-in ``log_density(x)`` when no conjugate prior is attached.
        """
        if self.has_conj_prior:
            m0, kappa, w_mat, nu = self.conj_prior_params
            diff = np.asarray(x, dtype=float) - m0
            e_quad = self.dim / kappa + nu * float(np.dot(diff, np.dot(w_mat, diff)))
            return 0.5 * self.e_log_det - 0.5 * self.dim * np.log(2.0 * np.pi) - 0.5 * e_quad
        return self.log_density(x)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``expected_log_density`` over sequence-encoded observations."""
        if self.has_conj_prior:
            m0, kappa, w_mat, nu = self.conj_prior_params
            diff = x - m0
            e_quad = self.dim / kappa + nu * np.sum(np.dot(diff, w_mat) * diff, axis=1)
            return 0.5 * self.e_log_det - 0.5 * self.dim * np.log(2.0 * np.pi) - 0.5 * e_quad
        return self.seq_log_density(x)

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        s1 = repr(list(self.mu))
        s2 = repr([list(u) for u in self.covar])
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return "MultivariateGaussianDistribution(%s, %s, name=%s, keys=%s)" % (s1, s2, s3, s4)

    def density(self, x: np.ndarray) -> float:
        """Evaluate the density at x.

        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Evaluate the log-density at x.

        The log-density is given by
            log(p(x)) = -0.5*k*log(2*pi) - 0.5*log|covar| - 0.5*(x-mu)' covar^{-1} (x-mu).
        Args:
            x (np.ndarray): Observation from multivariate Gaussian distribution.

        Returns:
            Log-density at x.

        """
        if self.use_lstsq:
            raise RuntimeError("Least-squares log-likelihood evaluation not supported.")
        else:
            try:
                diff = self.mu - x
                soln = scipy.linalg.cho_solve(self.chol, diff.T).T
                rv = self.chol_const - 0.5 * ((diff * soln).sum())
                return rv
            except Exception as e:
                raise e

    def density_cumulative(self, x: np.ndarray) -> float:
        """Exact probability-ordered cumulative ``G(x) = P(p(Y) >= p(x))`` -- the highest-density-region
        mass whose boundary passes through ``x`` (the multivariate analogue of a CDF; a coordinate-wise
        CDF is undefined without a total order on R^d).

        For a multivariate Gaussian ``p(y) >= p(x)`` iff the squared Mahalanobis distance is no larger,
        and that distance is chi-square with ``dim`` degrees of freedom, so ``G(x) = chi2.cdf(maha2, dim)``.
        Used by :func:`mixle.enumeration.density_rank.density_rank` to return an EXACT cumulative for the MVN.
        """
        from scipy.stats import chi2

        diff = self.mu - np.asarray(x, dtype=float)
        soln = scipy.linalg.cho_solve(self.chol, diff.T).T
        maha2 = float((diff * soln).sum())
        return float(chi2.cdf(maha2, df=self.dim))

    def density_quantile(self, q: float) -> np.ndarray:
        """Inverse of :meth:`density_cumulative`: a representative point at cumulative-density index ``q``.

        The multivariate analogue of a quantile / inverse-CDF: a coordinate-wise quantile is undefined
        without a total order on R^d, but the density ordering gives one. ``q`` is the highest-density
        region mass, so the boundary is the squared-Mahalanobis level ``chi2.ppf(q, dim)``; we return a
        representative point on that contour, ``mu + sqrt(level) * L[:, 0]`` where ``covar = L L^T``
        (so its Mahalanobis distance is exactly the level). Sweeping ``q`` enumerates the support in
        descending density.
        """
        from scipy.stats import chi2

        qf = float(q)
        if not 0.0 <= qf <= 1.0:
            raise ValueError("q must be in [0, 1].")
        radius = float(np.sqrt(chi2.ppf(qf, df=self.dim)))
        chol_lower = np.linalg.cholesky(self.covar)
        return self.mu + radius * chol_lower[:, 0]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of the log-density at a sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, dim) from
                MultivariateGaussianDataEncoder.seq_encode().

        Returns:
            Numpy array of length sz containing the log-density of each encoded observation.

        """
        if self.use_lstsq:
            return np.ones(x.shape[0])
        else:
            diff = self.mu - x
            soln = scipy.linalg.cho_solve(self.chol, diff.T).T
            rv = self.chol_const - 0.5 * ((diff * soln).sum(axis=1))
            return rv

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, inv_covar: Any, log_det: Any, engine: Any) -> Any:
        """Engine-neutral multivariate Gaussian log-density from inverse covariance."""
        diff = engine.asarray(x) - mu
        soln = engine.matmul(diff, inv_covar)
        quad = engine.sum(diff * soln, axis=-1)
        dim = float(mu.shape[-1])
        return -0.5 * (engine.asarray(dim) * engine.log(engine.asarray(2.0 * pi)) + log_det + quad)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.mu),
            engine.asarray(self.inv_covar),
            engine.asarray(self.log_det),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["MultivariateGaussianDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked full-covariance Gaussian parameters for a homogeneous mixture kernel."""
        dim = dists[0].dim
        if any(d.dim != dim for d in dists):
            raise ValueError("Stacked MultivariateGaussianDistribution components require a shared dimension.")
        return {
            "__pysp_component_axis__": {"mu": 0, "inv_covar": 0, "log_det": 0},
            "mu": np.stack([d.mu for d in dists], axis=0),
            "inv_covar": np.stack([d.inv_covar for d in dists], axis=0),
            "log_det": np.asarray([d.log_det for d in dists], dtype=float),
            "dim": dim,
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of full-covariance Gaussian log densities."""
        xx = engine.asarray(x)
        diff = xx[:, None, :] - params["mu"][None, :, :]
        soln = engine.matmul(diff[:, :, None, :], params["inv_covar"][None, :, :, :])[:, :, 0, :]
        quad = engine.sum(diff * soln, axis=2)
        return -0.5 * (
            engine.asarray(float(params["dim"])) * engine.log(engine.asarray(2.0 * pi))
            + params["log_det"][None, :]
            + quad
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return component-stacked legacy sufficient statistics on the active engine.

        The weighted second moment ``sum_n w[n,k] x_n x_n^T`` is accumulated per
        component as a gemm ``(x * w[:, k]).T @ x`` instead of reducing a per-sample,
        per-component ``(n, k, dim, dim)`` outer-product tensor. That intermediate is
        ``N*K*dim*dim`` (~20 GB at n=2e4, k=8, dim=128 — it OOMs a GPU); the per-component
        gemm holds only an ``(n, dim)`` temporary and hands the reduction to BLAS/cuBLAS.
        The first moment is likewise ``w.T @ x`` rather than a reduced ``(n, k, dim)`` tensor.
        """
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        k = int(ww.shape[1])
        sum_x = engine.matmul(ww.T, xx)
        sum_xx = engine.stack([engine.matmul((xx * ww[:, j : j + 1]).T, xx) for j in range(k)], axis=0)
        counts = engine.sum(ww, axis=0)
        return sum_x, sum_xx, counts

    def to_fisher(self, **kwargs):
        """Return this distribution's own Fisher view."""
        return MultivariateGaussianFisherView(self)

    def sampler(self, seed: int | None = None):
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional seed for the sampler's random state.

        Returns:
            A configured ``MultivariateGaussianSampler``.

        """
        return MultivariateGaussianSampler(self, seed)

    def condition(self, observed: dict[int, float]) -> "MultivariateGaussianDistribution":
        """Return the conditional distribution over the unobserved dimensions given ``observed``.

        ``observed`` maps dimension index to its fixed value; the result is the closed-form Gaussian
        conditional over the remaining dimensions (in increasing index order):

            mu_{u|o} = mu_u + Sigma_uo Sigma_oo^{-1} (x_o - mu_o),
            Sigma_{u|o} = Sigma_uu - Sigma_uo Sigma_oo^{-1} Sigma_ou.

        Sampling the result is ``given=``-style conditional sampling (draw the unobserved coordinates
        consistent with the observed ones). Raises if no dimension is left unobserved.
        """
        obs_idx = np.array(sorted(observed), dtype=int)
        if obs_idx.size and (obs_idx.min() < 0 or obs_idx.max() >= self.dim):
            raise ValueError("observed indices must be in [0, dim)")
        unobs_idx = np.array([i for i in range(self.dim) if i not in observed], dtype=int)
        if unobs_idx.size == 0:
            raise ValueError("at least one dimension must be left unobserved")
        if obs_idx.size == 0:
            return MultivariateGaussianDistribution(self.mu.copy(), self.covar.copy())
        x_o = np.array([observed[i] for i in obs_idx], dtype=np.float64)
        cov = np.asarray(self.covar, dtype=np.float64)
        s_oo = cov[np.ix_(obs_idx, obs_idx)]
        s_uo = cov[np.ix_(unobs_idx, obs_idx)]
        s_uu = cov[np.ix_(unobs_idx, unobs_idx)]
        solve = np.linalg.solve(s_oo, np.concatenate([(x_o - self.mu[obs_idx])[:, None], s_uo.T], axis=1))
        mu_cond = self.mu[unobs_idx] + s_uo @ solve[:, 0]
        cov_cond = s_uu - s_uo @ solve[:, 1:]
        cov_cond = 0.5 * (cov_cond + cov_cond.T)
        return MultivariateGaussianDistribution(mu_cond, cov_cond)

    def marginal(self, keep: Sequence[int]) -> "MultivariateGaussianDistribution":
        """Return the marginal Gaussian over dimensions ``keep``.

        ``N(mu, Sigma)`` marginalized to index set ``keep`` is ``N(mu[keep], Sigma[keep, keep])``. The
        order of ``keep`` is preserved, so the result's dimensions follow the given order.
        """
        idx = np.asarray(list(keep), dtype=int)
        if idx.size == 0:
            raise ValueError("keep at least one dimension")
        if idx.min() < 0 or idx.max() >= self.dim:
            raise ValueError("kept indices must be in [0, dim)")
        return MultivariateGaussianDistribution(self.mu[idx], np.asarray(self.covar)[np.ix_(idx, idx)])

    def estimator(self, pseudo_count: float | None = None):
        """Return an estimator initialized from this distribution's shape.

        If pseudo_count is passed, the current mean and covariance are used to regularize the estimate.

        Args:
            pseudo_count: Optional smoothing count applied to mean and
                covariance estimates.

        Returns:
            A ``MultivariateGaussianEstimator``.

        """
        if pseudo_count is None:
            return MultivariateGaussianEstimator(name=self.name, prior=self.prior)
        else:
            pseudo_count = (pseudo_count, pseudo_count)
            return MultivariateGaussianEstimator(
                pseudo_count=pseudo_count, suff_stat=(self.mu, self.covar), name=self.name, prior=self.prior
            )

    def dist_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        """Return an encoder for iid multivariate Gaussian observations."""
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianSampler(DistributionSampler):
    """Sampler for iid multivariate Gaussian observations."""

    def __init__(self, dist: "MultivariateGaussianDistribution", seed: int | None = None) -> None:
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

    def sample(self, size: int | None = None) -> np.ndarray:
        """Draw iid samples from the multivariate Gaussian distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, a single sample is drawn.

        Returns:
            Numpy array with shape (dim,) if size is None, else with shape (size, dim).

        """
        return self.rng.multivariate_normal(mean=self.dist.mu, cov=self.dist.covar, size=size)


class MultivariateGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for multivariate Gaussian sufficient statistics."""

    def __init__(self, dim: int | None = None, keys: str | None = None, name: str | None = None) -> None:
        """Create an accumulator for weighted first and second moments.

        Args:
            dim: Optional dimension of the Gaussian. Inferred from data when
                omitted.
            keys: Optional key for merging sufficient statistics.
            name: Optional diagnostic name.

        Attributes:
            dim: Dimension of the Gaussian.
            count: Sum of observation weights.
            sum: Weighted sum of observation vectors.
            sum2: Weighted sum of observation outer products.
            keys: Optional key for merging sufficient statistics.
            name: Optional diagnostic name.

        """
        self.dim = dim
        self.count = 0.0
        self.keys = keys
        self.name = name

        if dim is not None:
            self.sum = vec.zeros(dim)
            self.sum2 = vec.zeros((dim, dim))
        else:
            self.sum = None
            self.sum2 = None

    def update(self, x: np.ndarray, weight: float, estimate: MultivariateGaussianDistribution | None) -> None:
        """Update sufficient statistics with a single weighted observation.

        Args:
            x (np.ndarray): Length-dim observation vector.
            weight (float): Weight for the observation.
            estimate (Optional[MultivariateGaussianDistribution]): Kept for consistency with
                SequenceEncodableStatisticAccumulator (not used).

        Returns:
            None.

        """
        x = np.asarray(x, dtype=float)
        if self.dim is None:
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros((self.dim, self.dim))

        x_weight = x * weight
        self.sum += x_weight
        self.sum2 += vec.outer(x, x_weight)
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the accumulator with a weighted observation. Calls update().

        Args:
            x (np.ndarray): Length-dim observation vector.
            weight (float): Weight for the observation.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: RandomState | None) -> None:
        """Vectorized update of sufficient statistics with an encoded sequence of observations.

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, dim).
            weights (np.ndarray): Numpy array of sz observation weights.
            estimate (Optional[MultivariateGaussianDistribution]): Kept for consistency (not used).

        Returns:
            None.

        """
        if self.dim is None:
            self.dim = x.shape[1]
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros((self.dim, self.dim))

        x_weight = np.multiply(x.T, weights)
        self.count += weights.sum()
        self.sum += x_weight.sum(axis=1)
        # the weighted second moment sum_i w_i x_i x_i^T is (x.T * w) @ x -- a single BLAS gemm.
        # np.einsum runs the naive C loop here (no BLAS), which dominated MVN EM at ~76% of fit time
        # (20-36x slower than matmul on this contraction); the plain matmul is exact and multithreaded.
        self.sum2 += x_weight @ x

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of the accumulator. Calls seq_update().

        Args:
            x (np.ndarray): Encoded data matrix with shape (sz, dim).
            weights (np.ndarray): Numpy array of sz observation weights.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> "MultivariateGaussianAccumulator":
        """Merge sufficient statistics into this accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of outer products, sum of weights).

        Returns:
            This accumulator.

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
        """Return ``(sum, sum_outer, count)`` sufficient statistics."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "MultivariateGaussianAccumulator":
        """Replace this accumulator's sufficient statistics.

        Args:
            x (Tuple[np.ndarray, np.ndarray, float]): Tuple of (weighted sum of observations,
                weighted sum of outer products, sum of weights).

        Returns:
            This accumulator.

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
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
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
                self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "MultivariateGaussianDataEncoder":
        """Return an encoder compatible with this accumulator's dimension."""
        return MultivariateGaussianDataEncoder(dim=self.dim)


class MultivariateGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for multivariate Gaussian accumulators."""

    def __init__(self, dim: int | None, keys: str | None = None, name: str | None = None) -> None:
        """Create an accumulator factory.

        Args:
            dim: Optional Gaussian dimension.
            keys: Optional key for merging sufficient statistics.
            name: Optional diagnostic name.

        Attributes:
            dim: Optional Gaussian dimension.
            keys: Optional sufficient-statistic key.
            name: Optional diagnostic name.

        """
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> "MultivariateGaussianAccumulator":
        """Return a fresh accumulator with the factory configuration."""
        return MultivariateGaussianAccumulator(dim=self.dim, keys=self.keys, name=self.name)


class MultivariateGaussianEstimator(ParameterEstimator):
    """Estimator for multivariate Gaussian distributions."""

    def __init__(
        self,
        dim: int | None = None,
        pseudo_count: float | tuple[float | None, float | None] | None = (None, None),
        suff_stat: tuple[np.ndarray | None, np.ndarray | None] | None = (None, None),
        name: str | None = None,
        keys: str | None = None,
        prior: SequenceEncodableProbabilityDistribution | None = None,
        min_covar: float | None = None,
        ridge: float | None = None,
        track_conditioning: bool = False,
        degenerate_ratio: float = 1.0e-6,
    ) -> None:
        """Create an estimator for weighted multivariate Gaussian statistics.

        Args:
            dim: Gaussian dimension. Inferred from ``suff_stat`` when omitted.
            pseudo_count: Optional smoothing counts for mean and covariance. A scalar is
                broadcast to both slots.
            suff_stat: Optional prior mean and covariance used for smoothing.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.
            prior (Optional): Conjugate NormalWishart prior over (mu, Lambda=covar^-1). When present,
                ``estimate`` performs the closed-form conjugate posterior update (returning the joint
                MAP estimate and carrying the posterior forward as the fitted model's prior) instead
                of the maximum-likelihood / pseudo-count update.
            min_covar (Optional[float]): Absolute diagonal ridge floor applied in the MLE M-step.
                ``None`` (default) uses a tiny ``1e-8``.
            ridge (Optional[float]): Relative ridge coefficient. ``None`` (default) uses ``1e-6``;
                the covariance is regularized as ``cov + eps * I`` with
                ``eps = max(min_covar, ridge * trace(cov) / d)`` so a singular / non-finite
                covariance (a component holding < d points) cannot break the Cholesky factor.
                Bias is negligible at the defaults.
            track_conditioning (bool): Opt-in numerics-conditioning receipt. When ``True``,
                ``estimate`` computes the eigenspectrum of the RAW (pre-ridge) empirical covariance
                and attaches it to the returned distribution as ``.conditioning_receipt`` -- a
                :class:`~mixle.stats.compute.error_receipts.ConditioningReceipt` with the eigenvalues,
                condition number, and a near-degenerate-variance flag. Computed on the raw scatter
                (not the ridge-regularized one) so the receipt reports the DATA's true conditioning
                rather than the numerical safety net masking it. ``False`` by default (no overhead).
            degenerate_ratio (float): Smallest/largest covariance-eigenvalue ratio below which
                ``track_conditioning`` flags the fit as near-degenerate. Only used when
                ``track_conditioning=True``.

        Attributes:
            dim: Gaussian dimension.
            pseudo_count: Smoothing counts for mean and covariance.
            prior_mu: Prior mean used for smoothing.
            prior_covar: Prior covariance used for smoothing.
            name: Optional diagnostic name.
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

        self.dim = dim_loc
        pseudo_count = broadcast_pseudo_count(pseudo_count, 2)
        self.pseudo_count = pseudo_count
        self.prior_mu = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.prior_covar = None if suff_stat[1] is None else np.reshape(suff_stat[1], (dim_loc, dim_loc))
        self.name = name
        self.keys = keys
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalWishartDistribution)
        self.min_covar = 1.0e-8 if min_covar is None else float(min_covar)
        self.ridge = 1.0e-6 if ridge is None else float(ridge)
        self.track_conditioning = track_conditioning
        self.degenerate_ratio = float(degenerate_ratio)

    def accumulator_factory(self) -> "MultivariateGaussianAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        return MultivariateGaussianAccumulatorFactory(dim=self.dim, keys=self.keys, name=self.name)

    def model_log_density(self, model: "MultivariateGaussianDistribution") -> float:
        """Log-density of the model parameters under the NormalWishart prior (ELBO global term).

        The prior is over (mu, Lambda=covar^-1), so the model's covariance is inverted before scoring.
        """
        if self.has_conj_prior:
            return float(self.prior.log_density((model.mu, np.linalg.inv(model.covar))))
        return 0.0

    def _estimate_conjugate(
        self, suff_stat: tuple[np.ndarray, np.ndarray, float]
    ) -> "MultivariateGaussianDistribution":
        """Closed-form NormalWishart conjugate posterior update returning the joint MAP estimate."""
        xsum, outer_sum, count = suff_stat
        d = self.dim if self.dim is not None else len(xsum)

        m0, kappa0, w0, nu0 = self.prior.get_parameters()

        kappa_n = kappa0 + count
        nu_n = nu0 + count
        m_n = (kappa0 * m0 + xsum) / kappa_n

        if count > 0:
            xbar = xsum / count
            scatter = outer_sum - count * np.outer(xbar, xbar)
            dmu = xbar - m0
            w_n_inv = np.linalg.inv(w0) + scatter + (kappa0 * count / kappa_n) * np.outer(dmu, dmu)
        else:
            w_n_inv = np.linalg.inv(w0)

        # keep the inverse-scale symmetric despite accumulation round-off
        w_n_inv = 0.5 * (w_n_inv + w_n_inv.T)
        w_n = np.linalg.inv(w_n_inv)

        # joint MAP precision is (nu_n - d) W_n for nu_n > d; fall back to
        # the posterior mean nu_n W_n at the boundary
        if nu_n > d:
            covar = w_n_inv / (nu_n - d)
        else:
            covar = w_n_inv / nu_n

        posterior = NormalWishartDistribution(m_n, kappa_n, w_n, nu_n)
        return MultivariateGaussianDistribution(m_n, covar, name=self.name, prior=posterior)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, float]
    ) -> "MultivariateGaussianDistribution":
        """Estimate a multivariate normal distribution with from aggregated sufficient statistics.

        Suff_stat is a Tuple of size 3 containing:
            suff_stat[0] (np.ndarray): Component-wise sum of weighted observation values.
            suff_stat[1] (np.ndarray): Component-wise sum of weighted squared observation values.
            suff_stat[2] (float): Sum of weights for each observation.

        Args:
            nobs (Optional[float]): Weighted number of observations used in aggregation of suff stats.
            suff_stat (Tuple[np.ndarray, np.ndarray, float]): See above for details.

        Returns:
            MultivariateGaussianDistribution

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
            raw_covar = np.asarray(self.prior_covar, dtype=float) if self.prior_covar is not None else np.eye(d)
            covar = self._regularize_covar(raw_covar)
            dist = MultivariateGaussianDistribution(mu, covar, name=self.name, keys=self.keys)
            self._attach_conditioning_receipt(dist, raw_covar)
            return dist

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            raw_covar = (suff_stat[1] + (pc2 * self.prior_covar) - vec.outer(mu, mu * nobs)) / (nobs + pc2)
        else:
            raw_covar = (suff_stat[1] / nobs) - vec.outer(mu, mu)

        covar = self._regularize_covar(raw_covar)

        dist = MultivariateGaussianDistribution(mu, covar, name=self.name, keys=self.keys)
        self._attach_conditioning_receipt(dist, raw_covar)
        return dist

    def _attach_conditioning_receipt(self, dist: "MultivariateGaussianDistribution", raw_covar: np.ndarray) -> None:
        """Opt-in: compute and attach a numerics-conditioning receipt to ``dist`` (see ``__init__``)."""
        if not self.track_conditioning:
            return
        from mixle.stats.compute.error_receipts import conditioning_receipt

        dist.conditioning_receipt = conditioning_receipt(raw_covar, degenerate_ratio=self.degenerate_ratio)

    def _regularize_covar(self, covar: np.ndarray) -> np.ndarray:
        """P1 covariance ridge: cov <- cov + eps*I with eps = max(min_covar, ridge*trace/d).

        Clamps non-finite entries to zero first so a singular / NaN covariance from a
        component holding fewer than d points cannot break the Cholesky factorization.
        Symmetrizes to absorb accumulation round-off. Bias is negligible at the defaults.
        """
        covar = np.asarray(covar, dtype=float)
        d = covar.shape[0]
        if not np.isfinite(covar).all():
            covar = np.where(np.isfinite(covar), covar, 0.0)
        covar = 0.5 * (covar + covar.T)
        trace = float(np.trace(covar))
        eps = max(self.min_covar, self.ridge * trace / d if trace > 0.0 else 0.0)
        return covar + eps * np.eye(d)


class MultivariateGaussianDataEncoder(DataSequenceEncoder):
    """Encoder for iid multivariate Gaussian observations."""

    def __init__(self, dim: int | None = None) -> None:
        """Create an encoder with an optional fixed dimension.

        Args:
            dim: Optional Gaussian dimension. Inferred from data when omitted.

        """
        self.dim = dim

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "MultivariateGaussianDataEncoder(dim=" + str(self.dim) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is an encoder with the same dimension.

        Args:
            other (object): Object to compare against.

        Returns:
            bool.

        """
        return other.dim == self.dim if isinstance(other, MultivariateGaussianDataEncoder) else False

    def seq_encode(self, x: Sequence[list[float]] | Sequence[list[np.ndarray]] | np.ndarray):
        """Encode a sequence of iid length-dim observations for vectorized 'seq_' calls.

        Args:
            x (Union[Sequence[List[float]], Sequence[List[np.ndarray]], np.ndarray]): Sequence of
                length-dim observation vectors.

        Returns:
            Encoded data matrix with shape (len(x), dim).

        """
        self.dim = len(x[0]) if self.dim is None else self.dim
        return np.reshape(np.asarray(x), (-1, self.dim))
