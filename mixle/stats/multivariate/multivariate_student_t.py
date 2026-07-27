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
    marginal_indices,
    student_t_moments,
)
from mixle.stats.multivariate._vector_contracts import (
    matrix as matrix_parameter,
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
from mixle.utils.special import gammaln
from mixle.utils.vector import cholesky_logdet

_MIN_RIDGE = 1.0e-12


def _safe_inverse_and_logdet(shape: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (Sigma^{-1}, log|Sigma|), retrying once with a tiny ridge for a near-singular Sigma.

    Positive-definiteness is checked via Cholesky (``cholesky_logdet``), not a determinant-sign
    check: a matrix can have a positive determinant while being negative definite or indefinite
    (e.g. -I in an even dimension has determinant +1 while every eigenvalue is negative), so
    ``np.linalg.slogdet``'s sign is not a valid substitute. Raises if Sigma is still not
    positive-definite after the ridge retry, rather than silently returning a bogus log|Sigma|.

    ``MultivariateStudentTDistribution.__init__`` now rejects a non-positive-definite ``shape`` up
    front (see its own ``cholesky_logdet`` check), so for that call site ``mat`` here has already been
    proven PD and the ridge-retry branch below is unreachable in practice. It is kept -- rather than
    removed -- as a defensive fallback for any other caller of this function, and because the ridge
    retry alone cannot safely distinguish "PD in exact arithmetic, singular only from float rounding"
    from "exactly, structurally singular": both look identical in their eigenvalues, which is exactly
    why the up-front check in ``__init__`` treats any Cholesky failure on the raw ``shape`` as fatal
    rather than something to try to ridge through.
    """
    mat = np.asarray(shape, dtype=float)
    log_det = cholesky_logdet(mat)
    if log_det is None:
        mat = mat + np.eye(mat.shape[0]) * _MIN_RIDGE
        log_det = cholesky_logdet(mat)
    if log_det is None:
        raise ValueError("MultivariateStudentTDistribution requires a positive-definite scale matrix.")
    return np.linalg.inv(mat), log_det


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
            shape: ``p`` by ``p`` symmetric positive-definite scale matrix. Raises if ``shape`` is
                not symmetric or not positive-definite.
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
        checked_dof = finite_scalar(dof, label="multivariate Student-t dof", positive=True)
        mu = vector_parameter(loc, label="multivariate Student-t location")
        dim = len(mu)
        shape = matrix_parameter(
            shape,
            label="multivariate Student-t scale",
            dim=dim,
            symmetric=True,
        )
        # Positive-DEFINITE, not just semi-definite: unlike e.g. a Gaussian covariance, a singular
        # shape (a genuine zero eigenvalue, not merely a negative one) makes log|Sigma| = -inf and
        # Sigma^{-1} undefined, so the density is not well-defined. Checked directly on the raw shape
        # here -- not left to _safe_inverse_and_logdet below -- because that helper's tiny ridge retry
        # is meant only to absorb float rounding on a matrix that is PD in exact arithmetic (e.g. the
        # EM estimator's own reweighted scatter matrix, which is PSD by construction and pre-ridged by
        # the estimator itself before it ever reaches this constructor); it cannot distinguish that
        # from a shape that is exactly, structurally singular by construction (eigenvalues are
        # identical either way), so it previously let a singular shape build a distribution that
        # scored a finite density while its own sampler raised LinAlgError from cholesky.
        if cholesky_logdet(shape) is None:
            raise ValueError("MultivariateStudentTDistribution requires a positive-definite scale matrix.")

        self.dof = checked_dof
        self.mu = mu
        self.shape = shape
        self.dim = dim
        self.inv_shape, self.log_det = _safe_inverse_and_logdet(shape)
        for parameter in (self.mu, self.shape, self.inv_shape):
            parameter.setflags(write=False)
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
            return MultivariateStudentTDistribution(
                self.dof,
                mu.copy(),
                shape.copy(),
                name=self.name,
                keys=self.keys,
            )
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
        return MultivariateStudentTDistribution(
            self.dof + p_o,
            mu_cond,
            scale_cond,
            name=self.name,
            keys=self.keys,
        )

    def marginal(self, keep: Sequence[int]) -> "MultivariateStudentTDistribution":
        """Return the marginal over the dimensions ``keep``: ``MVT(dof, mu[keep], shape[keep, keep])``.

        A multivariate Student-t marginal keeps the same degrees of freedom and simply restricts the
        location and shape to the kept coordinates (order preserved).
        """
        idx = marginal_indices(keep, self.dim)
        mu = np.asarray(self.mu, dtype=np.float64)
        shape = np.asarray(self.shape, dtype=np.float64)
        return MultivariateStudentTDistribution(
            self.dof,
            mu[idx],
            shape[np.ix_(idx, idx)],
            name=self.name,
            keys=self.keys,
        )

    def density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the probability density at a single observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the log-density at a single observation."""
        diff = vector_event(x, self.dim, label="multivariate Student-t observation") - self.mu
        mahal = float(diff @ self.inv_shape @ diff)
        return self.log_const - 0.5 * (self.dof + self.dim) * np.log1p(mahal / self.dof)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        checked = vector_batch(x, self.dim, label="multivariate Student-t observations")
        diff = checked - self.mu
        mahal = np.einsum("ij,jk,ik->i", diff, self.inv_shape, diff)
        return self.log_const - 0.5 * (self.dof + self.dim) * np.log1p(mahal / self.dof)

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.mu.copy()),
            engine.asarray(self.inv_shape.copy()),
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
        return MultivariateStudentTEstimator(
            dof=self.dof, dim=self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "MultivariateStudentTDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return MultivariateStudentTDataEncoder(self.dim)


class MultivariateStudentTSampler(DistributionSampler):
    """Draw iid multivariate Student's t observations as mu + Z * sqrt(nu / G)."""

    def __init__(self, dist: MultivariateStudentTDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.chol = np.linalg.cholesky(dist.shape)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
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
        self.dof = finite_scalar(dof, label="multivariate Student-t dof", positive=True)
        self.dim = vector_dimension(dim, label="multivariate Student-t dimension", allow_none=True)
        self.count = 0.0
        self.sum_u = 0.0
        self.sum_ux = np.zeros(dim) if dim is not None else None
        self.sum_uxx = np.zeros((dim, dim)) if dim is not None else None
        self.keys = keys

    def _ensure_dim(self, p: int) -> None:
        p = vector_dimension(p, label="multivariate Student-t dimension")
        if self.dim is None:
            self.dim = p
        if self.sum_ux is None:
            self.sum_ux = np.zeros(self.dim)
            self.sum_uxx = np.zeros((self.dim, self.dim))

    def _weight_for(self, diff: np.ndarray, estimate: MultivariateStudentTDistribution | None) -> float:
        if estimate is None:
            return 1.0
        if (
            not isinstance(estimate, MultivariateStudentTDistribution)
            or estimate.dim != self.dim
            or estimate.dof != self.dof
        ):
            raise ValueError("Student-t accumulator estimate must match its configured dimension and dof")
        mahal = float(diff @ estimate.inv_shape @ diff)
        return (estimate.dof + estimate.dim) / (estimate.dof + mahal)

    def update(
        self, x: Sequence[float] | np.ndarray, weight: float, estimate: MultivariateStudentTDistribution | None
    ) -> None:
        """Accumulate one EM reweighted vector observation."""
        if self.dim is None:
            xx = vector_parameter(x, label="multivariate Student-t observation")
            self._ensure_dim(len(xx))
        else:
            xx = vector_event(x, self.dim, label="multivariate Student-t observation")
        checked_weight = observation_weight(weight, label="multivariate Student-t observation weight")
        u = self._weight_for(xx - estimate.mu, estimate) if estimate is not None else 1.0
        wu = checked_weight * u
        self.count += checked_weight
        self.sum_u += wu
        self.sum_ux += wu * xx
        self.sum_uxx += wu * np.outer(xx, xx)

    def initialize(self, x: Sequence[float] | np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one vector observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MultivariateStudentTDistribution | None) -> None:
        """Accumulate EM reweighted statistics from encoded vectors."""
        if self.dim is None:
            raw = np.asarray(x, dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] == 0:
                raise ValueError("multivariate Student-t observations must have exact shape (N, D) with D > 0")
            self._ensure_dim(raw.shape[1])
        checked = vector_batch(x, self.dim, label="multivariate Student-t observations")
        checked_weights = observation_weights(
            weights,
            len(checked),
            label="multivariate Student-t observation weights",
        )
        if estimate is None:
            u = np.ones(checked.shape[0])
        else:
            if (
                not isinstance(estimate, MultivariateStudentTDistribution)
                or estimate.dim != self.dim
                or estimate.dof != self.dof
            ):
                raise ValueError("Student-t accumulator estimate must match its configured dimension and dof")
            diff = checked - estimate.mu
            mahal = np.einsum("ij,jk,ik->i", diff, estimate.inv_shape, diff)
            u = (estimate.dof + estimate.dim) / (estimate.dof + mahal)
        wu = checked_weights * u
        self.count += float(np.sum(checked_weights, dtype=np.float64))
        self.sum_u += float(np.sum(wu, dtype=np.float64))
        self.sum_ux += checked.T @ wu
        self.sum_uxx += (checked * wu[:, None]).T @ checked

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, np.ndarray, np.ndarray]) -> "MultivariateStudentTAccumulator":
        """Merge another multivariate Student-t sufficient-statistic tuple."""
        count, sum_u, sum_ux, sum_uxx, inferred_dim = student_t_moments(suff_stat, self.dim)
        if sum_ux is not None:
            self._ensure_dim(inferred_dim)
            self.sum_ux += sum_ux
            self.sum_uxx += sum_uxx
        self.count += count
        self.sum_u += sum_u
        return self

    def value(self) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
        """Return count, latent-weight total, weighted sum, and weighted second moment."""
        return (
            self.count,
            self.sum_u,
            None if self.sum_ux is None else self.sum_ux.copy(),
            None if self.sum_uxx is None else self.sum_uxx.copy(),
        )

    def from_value(
        self, x: tuple[float, float, np.ndarray | None, np.ndarray | None]
    ) -> "MultivariateStudentTAccumulator":
        """Replace accumulator contents from sufficient statistics."""
        self.count, self.sum_u, self.sum_ux, self.sum_uxx, self.dim = student_t_moments(
            x,
            self.dim,
        )
        return self

    def acc_to_encoder(self) -> "MultivariateStudentTDataEncoder":
        """Return the encoder used by this accumulator."""
        return MultivariateStudentTDataEncoder(self.dim)


class MultivariateStudentTAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MultivariateStudentTAccumulator."""

    def __init__(self, dof: float, dim: int | None = None, keys: str | None = None) -> None:
        self.dof = finite_scalar(dof, label="multivariate Student-t dof", positive=True)
        self.dim = vector_dimension(dim, label="multivariate Student-t dimension", allow_none=True)
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
        pseudo_count: float | None = None,
        min_ridge: float = _MIN_RIDGE,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        checked_dof = finite_scalar(dof, label="multivariate Student-t dof", positive=True)
        if pseudo_count is not None:
            # Unlike the raw-moment method-of-moments estimators (Gumbel, Weibull, ...), this is an
            # EM/IRLS estimator: the accumulated sufficient statistic (sum_u, sum_ux, sum_uxx) is
            # reweighted by the latent u_i = (dof+p)/(dof+mahal_i), which depends on the PREVIOUS
            # iterate's own fitted mu/shape, not a fixed closed form. Blending a prior pseudo-sample
            # into it correctly would require E[u], E[u*x], E[u*x*x'] under this distribution's own
            # generating process, i.e. moments of a ratio of independent chi-squared variables
            # (mahal = (dof/G)*W, G ~ Chi2(dof), W ~ Chi2(p)) -- not a small, safe change to bolt
            # onto this fixed point. Refuse explicitly rather than silently ignoring pseudo_count.
            raise ValueError(
                "MultivariateStudentTEstimator does not support pseudo_count smoothing: its EM/IRLS "
                "fit reweights the sufficient statistic by a latent factor with no simple closed-form "
                "expectation under the model, so there is no small, safe way to blend a prior "
                "pseudo-sample into it. Pass pseudo_count=None (the default)."
            )
        self.dof = checked_dof
        self.dim = vector_dimension(dim, label="multivariate Student-t dimension", allow_none=True)
        self.min_ridge = finite_scalar(min_ridge, label="multivariate Student-t min_ridge", positive=True)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MultivariateStudentTAccumulatorFactory:
        """Return an accumulator factory for fixed-dof Student-t EM statistics."""
        return MultivariateStudentTAccumulatorFactory(dof=self.dof, dim=self.dim, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, np.ndarray | None, np.ndarray | None]
    ) -> MultivariateStudentTDistribution:
        """Estimate location and scale from EM reweighted statistics."""
        count, sum_u, sum_ux, sum_uxx, inferred_dim = student_t_moments(suff_stat, self.dim)
        if sum_ux is None or count <= 0.0 or sum_u <= 0.0:
            if self.dim is None:
                raise ValueError("cannot infer Student-t dimension from an empty sufficient statistic")
            p = self.dim
            return MultivariateStudentTDistribution(self.dof, np.zeros(p), np.eye(p), name=self.name, keys=self.keys)
        if self.dim is not None and inferred_dim != self.dim:
            raise ValueError("Student-t sufficient statistic dimension does not match estimator")

        mu = sum_ux / sum_u
        # Sigma = sum_i w_i u_i (x_i - mu)(x_i - mu)' / sum_i w_i
        scatter = sum_uxx - np.outer(mu, sum_ux) - np.outer(sum_ux, mu) + sum_u * np.outer(mu, mu)
        shape = scatter / count
        shape = 0.5 * (shape + shape.T)
        scale = max(
            float(np.linalg.norm(sum_uxx, ord=2)),
            float(np.linalg.norm(np.outer(mu, sum_ux), ord=2)),
            float(np.linalg.norm(sum_u * np.outer(mu, mu), ord=2)),
            1.0,
        ) / count
        minimum_eigenvalue = float(np.linalg.eigvalsh(shape)[0])
        if minimum_eigenvalue < -1.0e-6 * scale:
            raise ValueError("Student-t sufficient statistics imply a non-positive-semidefinite scale")
        if minimum_eigenvalue < 0.0:
            shape = shape + np.eye(len(mu)) * -minimum_eigenvalue
        shape = shape + np.eye(len(mu)) * self.min_ridge
        return MultivariateStudentTDistribution(self.dof, mu, shape, name=self.name, keys=self.keys)


class MultivariateStudentTDataEncoder(DataSequenceEncoder):
    """Encode a sequence of length-p real vectors into an (n, p) float array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = vector_dimension(dim, label="multivariate Student-t dimension", allow_none=True)

    def __str__(self) -> str:
        return "MultivariateStudentTDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultivariateStudentTDataEncoder) and other.dim == self.dim

    def seq_encode(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Encode observations as an ``(n, p)`` floating-point matrix."""
        raw = np.asarray(x, dtype=np.float64)
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[1] == 0:
                raise ValueError("multivariate Student-t observations must have exact shape (N, D) with D > 0")
            self.dim = raw.shape[1]
        return vector_batch(raw, self.dim, label="multivariate Student-t observations").copy()
