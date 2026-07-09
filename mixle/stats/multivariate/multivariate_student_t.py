"""Multivariate Student's t distributions over real-valued vectors.

Data type: np.ndarray[float] (a length-p real vector).

x ~ MVT(dof, loc, shape) with degrees of freedom nu = dof > 0, location vector mu (length p), and a
p-by-p symmetric positive-definite scale matrix Sigma. The log-density is

    log(f(x)) = gammaln((nu + p)/2) - gammaln(nu/2) - 0.5*p*log(nu*pi) - 0.5*log|Sigma|
                - 0.5*(nu + p)*log(1 + delta(x)/nu),

where delta(x) = (x - mu)' Sigma^{-1} (x - mu) is the squared Mahalanobis distance. As nu -> inf the
distribution converges to MVN(mu, Sigma); for nu > 2 the covariance is nu/(nu - 2) * Sigma. The heavy
tails make it a robust alternative to the multivariate normal.

Estimation keeps nu fixed and runs the EM / iteratively-reweighted update (each observation gets the
latent-scale weight u_i = (nu + p)/(nu + delta_i) under the current estimate), which is the standard
maximum-likelihood scheme for a known degrees of freedom. The engine-neutral
``backend_log_density_from_params`` gives the family generated NumPy and Torch scoring.


Reference: Kotz & Nadarajah, *Multivariate t Distributions and Their Applications* (Cambridge, 2004).
"""

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
from mixle.utils.special import gammaln

_MIN_RIDGE = 1.0e-12


def _safe_inverse_and_logdet(shape: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (Sigma^{-1}, log|Sigma|) with a tiny ridge fallback for near-singular Sigma."""
    mat = np.asarray(shape, dtype=float)
    sign, log_det = np.linalg.slogdet(mat)
    if sign <= 0.0 or not np.isfinite(log_det):
        mat = mat + np.eye(mat.shape[0]) * _MIN_RIDGE
        sign, log_det = np.linalg.slogdet(mat)
    return np.linalg.inv(mat), float(log_det)


class MultivariateStudentTDistribution(SequenceEncodableProbabilityDistribution):
    """Multivariate Student's t distribution with degrees of freedom dof, location mu, and scale Sigma."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated multivariate Student-t kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for multivariate Student-t distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="multivariate_student_t",
            distribution_type=cls,
            parameters=(
                ParameterSpec("mu", constraint="real_vector"),
                ParameterSpec("inv_shape", constraint="positive_matrix", differentiable=False),
                ParameterSpec("log_const", constraint="real", differentiable=False),
                ParameterSpec("dof", constraint="positive", differentiable=False),
                ParameterSpec("dim", constraint="fixed", differentiable=False),
            ),
            # The accumulator carries the EM/IRLS reweighted statistics (count, sum_u, sum_ux, sum_uxx).
            # Each is reweighted by u_i, which depends on the current estimate, so the family deliberately
            # exposes no row-wise generated sufficient-statistic hook (no resident reduction shortcut).
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum_u"),
                StatisticSpec("sum_ux", kind="vector_moment"),
                StatisticSpec("sum_uxx", kind="matrix_moment"),
            ),
            support="real_vector",
            differentiable=False,
        )

    @staticmethod
    def backend_log_density_from_params(
        x: Any, mu: Any, inv_shape: Any, log_const: Any, dof: Any, dim: Any, engine: Any
    ) -> Any:
        """Engine-neutral multivariate Student's t log-density from fitted parameters."""
        xx = engine.asarray(x)
        diff = xx - mu
        # delta = (x - mu)' Sigma^{-1} (x - mu), batched over the leading observation axis.
        mahal = engine.sum(engine.matmul(diff, inv_shape) * diff, axis=-1)
        p = engine.asarray(float(dim))
        return log_const - engine.asarray(0.5) * (dof + p) * engine.log(engine.asarray(1.0) + mahal / dof)

    def __init__(
        self,
        dof: float,
        loc: Sequence[float] | np.ndarray,
        shape: Sequence[Sequence[float]] | np.ndarray,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a multivariate Student-t distribution.

        Args:
            dof: Degrees of freedom, which must be positive and finite.
            loc: Location vector of length ``p``.
            shape: ``p`` by ``p`` symmetric positive-definite scale matrix.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            dof: Degrees of freedom.
            mu: Location vector.
            shape: Scale matrix.
            inv_shape: Cached inverse scale matrix.
            log_det: Cached scale log-determinant.
            log_const: Cached log normalizer.
            dim: Observation dimension.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.

        """
        if dof <= 0.0 or not np.isfinite(dof):
            raise ValueError("MultivariateStudentTDistribution requires dof > 0.")
        mu = np.asarray(loc, dtype=float).copy()
        shape = np.asarray(shape, dtype=float).copy()
        dim = len(mu)
        if shape.shape != (dim, dim):
            raise ValueError("MultivariateStudentTDistribution shape must be a (p, p) matrix matching loc.")

        self.dof = float(dof)
        self.mu = mu
        self.shape = shape
        self.dim = dim
        self.inv_shape, self.log_det = _safe_inverse_and_logdet(shape)
        self.log_const = (
            gammaln((self.dof + dim) / 2.0)
            - gammaln(self.dof / 2.0)
            - 0.5 * dim * np.log(self.dof * np.pi)
            - 0.5 * self.log_det
        )
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        return "MultivariateStudentTDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.dof),
            repr([float(v) for v in self.mu]),
            repr([[float(v) for v in row] for row in self.shape]),
            repr(self.name),
            repr(self.keys),
        )

    def condition(self, observed: dict[int, float]) -> "MultivariateStudentTDistribution":
        """Return the conditional distribution over the unobserved dimensions given ``observed``.

        The conditional of a multivariate Student-t is again a multivariate Student-t. With observed
        dimensions ``o`` (Mahalanobis ``d_o``) and unobserved ``u``:

            dof'   = dof + |o|,
            mu'    = mu_u + S_uo S_oo^{-1} (x_o - mu_o),
            shape' = (dof + d_o)/(dof + |o|) * (S_uu - S_uo S_oo^{-1} S_ou),

        i.e. the location shifts like the Gaussian conditional but the scale is inflated by how far the
        observed coordinates fall in the tails (``given=``-style conditional sampling). Raises if no
        dimension is left unobserved.
        """
        mu = np.asarray(self.mu, dtype=np.float64)
        shape = np.asarray(self.shape, dtype=np.float64)
        obs_idx = np.array(sorted(observed), dtype=int)
        if obs_idx.size and (obs_idx.min() < 0 or obs_idx.max() >= self.dim):
            raise ValueError("observed indices must be in [0, dim)")
        unobs_idx = np.array([i for i in range(self.dim) if i not in observed], dtype=int)
        if unobs_idx.size == 0:
            raise ValueError("at least one dimension must be left unobserved")
        if obs_idx.size == 0:
            return MultivariateStudentTDistribution(self.dof, mu.copy(), shape.copy())
        x_o = np.array([observed[i] for i in obs_idx], dtype=np.float64) - mu[obs_idx]
        s_oo = shape[np.ix_(obs_idx, obs_idx)]
        s_uo = shape[np.ix_(unobs_idx, obs_idx)]
        s_uu = shape[np.ix_(unobs_idx, unobs_idx)]
        solve = np.linalg.solve(s_oo, np.concatenate([x_o[:, None], s_uo.T], axis=1))
        mu_cond = mu[unobs_idx] + s_uo @ solve[:, 0]
        d_o = float(x_o @ solve[:, 0])  # observed Mahalanobis distance
        p_o = obs_idx.size
        scale_cond = ((self.dof + d_o) / (self.dof + p_o)) * (s_uu - s_uo @ solve[:, 1:])
        scale_cond = 0.5 * (scale_cond + scale_cond.T)
        return MultivariateStudentTDistribution(self.dof + p_o, mu_cond, scale_cond)

    def marginal(self, keep: Sequence[int]) -> "MultivariateStudentTDistribution":
        """Return the marginal over the dimensions ``keep``: ``MVT(dof, mu[keep], shape[keep, keep])``.

        A multivariate Student-t marginal keeps the same degrees of freedom and simply restricts the
        location and shape to the kept coordinates (order preserved).
        """
        idx = np.asarray(list(keep), dtype=int)
        if idx.size == 0:
            raise ValueError("keep at least one dimension")
        if idx.min() < 0 or idx.max() >= self.dim:
            raise ValueError("kept indices must be in [0, dim)")
        mu = np.asarray(self.mu, dtype=np.float64)
        shape = np.asarray(self.shape, dtype=np.float64)
        return MultivariateStudentTDistribution(self.dof, mu[idx], shape[np.ix_(idx, idx)])

    def density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the probability density at a single observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the log-density at a single observation."""
        diff = np.asarray(x, dtype=float) - self.mu
        mahal = float(diff @ self.inv_shape @ diff)
        return self.log_const - 0.5 * (self.dof + self.dim) * np.log1p(mahal / self.dof)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        diff = x - self.mu
        mahal = np.einsum("ij,jk,ik->i", diff, self.inv_shape, diff)
        return self.log_const - 0.5 * (self.dof + self.dim) * np.log1p(mahal / self.dof)

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.mu),
            engine.asarray(self.inv_shape),
            engine.asarray(self.log_const),
            engine.asarray(self.dof),
            self.dim,
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["MultivariateStudentTDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for equal-dimensional multivariate Student's t mixtures."""
        dim = int(dists[0].dim)
        if any(int(dist.dim) != dim for dist in dists):
            raise ValueError("Stacked MultivariateStudentTDistribution components require equal dimension.")
        return {
            "__pysp_component_axis__": {"mu": 0, "inv_shape": 0, "log_const": 0, "dof": 0},
            "mu": engine.asarray([dist.mu for dist in dists]),
            "inv_shape": engine.asarray([dist.inv_shape for dist in dists]),
            "log_const": engine.asarray([dist.log_const for dist in dists]),
            "dof": engine.asarray([dist.dof for dist in dists]),
            "dim": dim,
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of multivariate Student's t component log densities."""
        xx = engine.asarray(x)
        mu = params["mu"]
        inv_shape = params["inv_shape"]
        dof = params["dof"]
        p = engine.asarray(float(params["dim"]))
        # diff[n, k, j] = x[n, j] - mu[k, j]; mahal[n, k] = diff' inv_shape[k] diff, contracted as
        # sum_{l, j} diff[n,k,l] * inv_shape[k,l,j] * diff[n,k,j] (engine-safe, no batched matmul).
        diff = xx[:, None, :] - mu[None, :, :]
        outer = diff[:, :, :, None] * diff[:, :, None, :]
        mahal = engine.sum(engine.sum(outer * inv_shape[None, :, :, :], axis=-1), axis=-1)
        return params["log_const"][None, :] - engine.asarray(0.5) * (dof[None, :] + p) * engine.log(
            engine.asarray(1.0) + mahal / dof[None, :]
        )

    def sampler(self, seed: int | None = None) -> "MultivariateStudentTSampler":
        """Return a sampler for drawing observations from this distribution."""
        return MultivariateStudentTSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "MultivariateStudentTEstimator":
        """Return an EM estimator that keeps dof fixed at this distribution's value."""
        return MultivariateStudentTEstimator(dof=self.dof, dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "MultivariateStudentTDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return MultivariateStudentTDataEncoder()


class MultivariateStudentTSampler(DistributionSampler):
    """Draw iid multivariate Student's t observations as mu + Z * sqrt(nu / G)."""

    def __init__(self, dist: MultivariateStudentTDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.chol = np.linalg.cholesky(dist.shape)

    def sample(self, size: int | None = None) -> np.ndarray:
        """Draw ``size`` iid vectors (shape (p,) when size is None, else (size, p))."""
        sz = 1 if size is None else size
        p = self.dist.dim
        z = self.rng.standard_normal(size=(sz, p)) @ self.chol.T
        g = self.rng.chisquare(self.dist.dof, size=sz)
        rv = self.dist.mu[None, :] + z * np.sqrt(self.dist.dof / g)[:, None]
        return rv[0] if size is None else rv


class MultivariateStudentTAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the EM/IRLS sufficient statistics for multivariate Student's t estimation.

    The reweighting u_i = (nu + p)/(nu + delta_i) is computed from the previous ``estimate``; with no
    estimate (initialization) every u_i = 1, which seeds the fit with the Gaussian moment statistics.
    """

    def __init__(self, dof: float, dim: int | None = None, keys: str | None = None) -> None:
        self.dof = float(dof)
        self.dim = dim
        self.count = 0.0
        self.sum_u = 0.0
        self.sum_ux = np.zeros(dim) if dim is not None else None
        self.sum_uxx = np.zeros((dim, dim)) if dim is not None else None
        self.keys = keys

    def _ensure_dim(self, p: int) -> None:
        if self.dim is None:
            self.dim = p
        if self.sum_ux is None:
            self.sum_ux = np.zeros(self.dim)
            self.sum_uxx = np.zeros((self.dim, self.dim))

    def _weight_for(self, diff: np.ndarray, estimate: MultivariateStudentTDistribution | None) -> float:
        if estimate is None:
            return 1.0
        mahal = float(diff @ estimate.inv_shape @ diff)
        return (estimate.dof + estimate.dim) / (estimate.dof + mahal)

    def update(
        self, x: Sequence[float] | np.ndarray, weight: float, estimate: MultivariateStudentTDistribution | None
    ) -> None:
        """Accumulate one EM reweighted vector observation."""
        xx = np.asarray(x, dtype=float)
        self._ensure_dim(len(xx))
        u = self._weight_for(xx - estimate.mu, estimate) if estimate is not None else 1.0
        wu = weight * u
        self.count += weight
        self.sum_u += wu
        self.sum_ux += wu * xx
        self.sum_uxx += wu * np.outer(xx, xx)

    def initialize(self, x: Sequence[float] | np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one vector observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MultivariateStudentTDistribution | None) -> None:
        """Accumulate EM reweighted statistics from encoded vectors."""
        self._ensure_dim(x.shape[1])
        if estimate is None:
            u = np.ones(x.shape[0])
        else:
            diff = x - estimate.mu
            mahal = np.einsum("ij,jk,ik->i", diff, estimate.inv_shape, diff)
            u = (estimate.dof + estimate.dim) / (estimate.dof + mahal)
        wu = weights * u
        self.count += float(np.sum(weights, dtype=np.float64))
        self.sum_u += float(np.sum(wu, dtype=np.float64))
        self.sum_ux += x.T @ wu
        self.sum_uxx += (x * wu[:, None]).T @ x

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, np.ndarray, np.ndarray]) -> "MultivariateStudentTAccumulator":
        """Merge another multivariate Student-t sufficient-statistic tuple."""
        count, sum_u, sum_ux, sum_uxx = suff_stat
        if sum_ux is not None:
            self._ensure_dim(len(sum_ux))
            self.sum_ux += sum_ux
            self.sum_uxx += sum_uxx
        self.count += count
        self.sum_u += sum_u
        return self

    def value(self) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
        """Return count, latent-weight total, weighted sum, and weighted second moment."""
        return self.count, self.sum_u, self.sum_ux, self.sum_uxx

    def from_value(
        self, x: tuple[float, float, np.ndarray | None, np.ndarray | None]
    ) -> "MultivariateStudentTAccumulator":
        """Replace accumulator contents from sufficient statistics."""
        self.count, self.sum_u, self.sum_ux, self.sum_uxx = x
        self.dim = None if x[2] is None else len(x[2])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed statistics into ``stats_dict`` when keys are configured."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from keyed statistics when available."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "MultivariateStudentTDataEncoder":
        """Return the encoder used by this accumulator."""
        return MultivariateStudentTDataEncoder()


class MultivariateStudentTAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MultivariateStudentTAccumulator."""

    def __init__(self, dof: float, dim: int | None = None, keys: str | None = None) -> None:
        self.dof = dof
        self.dim = dim
        self.keys = keys

    def make(self) -> MultivariateStudentTAccumulator:
        """Create a fresh multivariate Student-t accumulator."""
        return MultivariateStudentTAccumulator(dof=self.dof, dim=self.dim, keys=self.keys)


class MultivariateStudentTEstimator(ParameterEstimator):
    """Fixed-dof EM estimator for the multivariate Student's t location and scale matrix."""

    def __init__(
        self,
        dof: float = 5.0,
        dim: int | None = None,
        min_ridge: float = _MIN_RIDGE,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dof <= 0.0 or not np.isfinite(dof):
            raise ValueError("MultivariateStudentTEstimator requires dof > 0.")
        self.dof = float(dof)
        self.dim = dim
        self.min_ridge = min_ridge
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MultivariateStudentTAccumulatorFactory:
        """Return an accumulator factory for fixed-dof Student-t EM statistics."""
        return MultivariateStudentTAccumulatorFactory(dof=self.dof, dim=self.dim, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, np.ndarray | None, np.ndarray | None]
    ) -> MultivariateStudentTDistribution:
        """Estimate location and scale from EM reweighted statistics."""
        count, sum_u, sum_ux, sum_uxx = suff_stat
        if sum_ux is None or count <= 0.0 or sum_u <= 0.0:
            p = self.dim if self.dim is not None else 1
            return MultivariateStudentTDistribution(self.dof, np.zeros(p), np.eye(p), name=self.name, keys=self.keys)

        mu = sum_ux / sum_u
        # Sigma = sum_i w_i u_i (x_i - mu)(x_i - mu)' / sum_i w_i
        scatter = sum_uxx - np.outer(mu, sum_ux) - np.outer(sum_ux, mu) + sum_u * np.outer(mu, mu)
        shape = scatter / count
        shape = 0.5 * (shape + shape.T)
        shape = shape + np.eye(len(mu)) * self.min_ridge
        return MultivariateStudentTDistribution(self.dof, mu, shape, name=self.name, keys=self.keys)


class MultivariateStudentTDataEncoder(DataSequenceEncoder):
    """Encode a sequence of length-p real vectors into an (n, p) float array."""

    def __str__(self) -> str:
        return "MultivariateStudentTDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultivariateStudentTDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Encode observations as an ``(n, p)`` floating-point matrix."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.ndim != 2:
            rv = rv.reshape((len(x), -1))
        if rv.size and not np.all(np.isfinite(rv)):
            raise ValueError("MultivariateStudentTDistribution requires finite real-vector observations.")
        return rv
