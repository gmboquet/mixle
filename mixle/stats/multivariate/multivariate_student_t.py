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
    needs_vector_anchor,
    student_t_moments,
    warn_uncorrectable_vector_moments,
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
from mixle.utils.vector import cholesky_logdet, owned_backend_parameter

_MIN_RIDGE = 1.0e-12


class MultivariateStudentTSuffStat(tuple):
    """A ``(count, sum_u, sum_ux, sum_uxx)`` statistic that also carries shift-anchored moments.

    Behaves exactly like the plain 4-tuple everywhere it is indexed, unpacked, iterated or validated
    by :func:`student_t_moments` (it *is* one); ``anchored`` is the extra payload
    ``(anchor, sum_i w_i u_i (x_i - anchor), sum_i w_i u_i (x_i - anchor)(x_i - anchor)^T)``, the
    same EM-reweighted moments expressed about a data anchor instead of about the origin. Code that
    doesn't know about the payload sees an ordinary tuple and the estimate falls back to the
    historical raw path.
    """

    def __new__(
        cls,
        count: float,
        sum_u: float,
        sum_ux: np.ndarray | None,
        sum_uxx: np.ndarray | None,
        anchored: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ):
        obj = super().__new__(cls, (count, sum_u, sum_ux, sum_uxx))
        obj.anchored = anchored
        return obj

    def __reduce__(self):
        # A plain tuple subclass with a payload-bearing __new__ does not pickle by default; the
        # Spark/mp reducers round-trip accumulator values through pickle, so keep the payload.
        return (_rebuild_student_t_suff_stat, (tuple(self), self.anchored))


def _rebuild_student_t_suff_stat(values: tuple, anchored: tuple | None) -> "MultivariateStudentTSuffStat":
    """Unpickle helper for :class:`MultivariateStudentTSuffStat` (module-level so pickle can import it)."""
    return MultivariateStudentTSuffStat(values[0], values[1], values[2], values[3], anchored=anchored)


def _consistent_anchored_student_t(
    suff_stat: Any,
    sum_ux: np.ndarray | None,
    sum_u: float,
    dim: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the anchored payload of ``suff_stat`` when it is usable, else ``None``.

    ``None`` falls back to the raw reduced-moment M-step, so a payload is only trusted when it is
    shaped right, finite, and agrees with the raw first moment it claims to describe -- a hand-built
    statistic whose payload contradicts its tuple must not silently change the estimate the tuple
    alone would have produced.
    """
    anchored = getattr(suff_stat, "anchored", None)
    if anchored is None or sum_ux is None or dim is None or not sum_u > 0.0:
        return None
    try:
        anchor, a_ux, a_uxx = anchored
        anchor = np.asarray(anchor, dtype=float).reshape(-1)
        a_ux = np.asarray(a_ux, dtype=float).reshape(-1)
        a_uxx = np.asarray(a_uxx, dtype=float)
    except (TypeError, ValueError):
        return None
    if anchor.shape != (dim,) or a_ux.shape != (dim,) or a_uxx.shape != (dim, dim):
        return None
    if not (np.isfinite(anchor).all() and np.isfinite(a_ux).all() and np.isfinite(a_uxx).all()):
        return None
    implied = a_ux + sum_u * anchor
    tolerance = 1.0e-6 * np.maximum.reduce(
        (np.abs(sum_ux), np.abs(sum_u * anchor), np.ones_like(anchor)),
    )
    if np.any(np.abs(implied - sum_ux) > tolerance):
        return None
    return anchor, a_ux, 0.5 * (a_uxx + a_uxx.T)


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
        dimensions ``o`` (Mahalanobis ``d_o``) and unobserved ``u``::

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
            engine.asarray(owned_backend_parameter(self.mu)),
            engine.asarray(owned_backend_parameter(self.inv_shape)),
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

    Alongside those raw moments the accumulator keeps a CONDITIONING-GATED shift-anchored track.
    The M-step forms ``sum_uxx - mu sum_ux' - sum_ux mu' + sum_u mu mu'``, which loses
    ~``2*log2(abs(mean)/sd)`` bits to cancellation, so sd ~2 data at the Unix-epoch offset fitted a
    scale matrix collapsed to ~1e-6 (and a ValueError at 1e10) with no warning at all. Anchoring at
    a data value keeps every term of the scatter ``O(sum_u * spread^2)``, which makes the scale
    matrix shift-invariant. The track is gated: a chunk whose ``abs(mean)/spread`` ratio the raw
    form handles to ~1e-9 relative error (see :func:`needs_vector_anchor`) accumulates exactly the
    historical single-pass way -- bit-identical statistics, no second pass. The raw moments remain
    the exchange format, so the anchored track rides along as a payload on
    :class:`MultivariateStudentTSuffStat`.
    """

    def __init__(self, dof: float, dim: int | None = None, keys: str | None = None) -> None:
        self.dof = finite_scalar(dof, label="multivariate Student-t dof", positive=True)
        self.dim = vector_dimension(dim, label="multivariate Student-t dimension", allow_none=True)
        self.count = 0.0
        self.sum_u = 0.0
        self.sum_ux = np.zeros(dim) if dim is not None else None
        self.sum_uxx = np.zeros((dim, dim)) if dim is not None else None
        self.keys = keys
        self._anchor: np.ndarray | None = None
        self._anchored_ux: np.ndarray | None = None
        self._anchored_uxx: np.ndarray | None = None

    def _ensure_dim(self, p: int) -> None:
        p = vector_dimension(p, label="multivariate Student-t dimension")
        if self.dim is None:
            self.dim = p
        if self.sum_ux is None:
            self.sum_ux = np.zeros(self.dim)
            self.sum_uxx = np.zeros((self.dim, self.dim))

    def _activate_anchor(self, anchor: np.ndarray) -> None:
        """Start the shift-anchored moment track at ``anchor``.

        Any content already accumulated raw-only is converted about the new anchor. The conversion
        is the cancellation-prone form, but it is only ever applied to content that accumulated
        WITHOUT activating the gate -- content the gate certified as well-conditioned -- or to
        pre-existing raw statistics restored through ``from_value``/``combine``, where the
        conversion is no less accurate than the raw-only estimate those statistics supported before.
        """
        a = np.asarray(anchor, dtype=float).reshape(-1).copy()
        self._anchor = a
        self._anchored_ux = np.zeros(self.dim)
        self._anchored_uxx = np.zeros((self.dim, self.dim))
        if self.sum_u != 0.0 or np.any(self.sum_ux != 0.0):
            self._anchored_ux += self.sum_ux - a * self.sum_u
            scatter = self.sum_uxx - np.outer(a, self.sum_ux) - np.outer(self.sum_ux, a) + self.sum_u * np.outer(a, a)
            self._anchored_uxx += 0.5 * (scatter + scatter.T)

    def _anchor_rows(self, rows: np.ndarray, wu: np.ndarray, chunk_ux: np.ndarray, chunk_uxx: np.ndarray) -> None:
        """Fold reweighted rows into the anchored track when they need one. Call BEFORE the raw fold.

        The chunk's own moments are the ones the caller already computed for the raw fold, so the
        conditioning gate costs one test rather than a second pass over the data. The gate is
        assessed on the LATENT-WEIGHTED totals, because those are the moments the M-step differences.
        """
        if len(rows) == 0:
            return
        # An accumulator restored from an EMPTY statistic keeps its dimension but has no moment
        # arrays yet -- a starved EM component is a normal state, and the next update has to be able
        # to grow it back. Both the anchored track and the raw fold below need the arrays, and
        # without this both raised ``TypeError: unsupported operand type(s) for +: 'NoneType'``.
        self._ensure_dim(rows.shape[1])
        u_sum = float(np.sum(wu, dtype=np.float64))
        if self._anchor is None and not needs_vector_anchor(chunk_ux, np.diag(chunk_uxx), u_sum):
            return
        if self._anchor is None:
            self._activate_anchor(rows[0])
        dx = rows - self._anchor
        wdx = dx * wu[:, None]
        self._anchored_ux += wdx.sum(axis=0)
        scatter = wdx.T @ dx
        self._anchored_uxx += 0.5 * (scatter + scatter.T)

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
        # Scalar updates carry no chunk to assess conditioning from, so the anchor activates on the
        # first observation -- a zero-cost O(1) bookkeeping track on this path. BEFORE the raw fold,
        # so activation converts only content the gate has already vouched for.
        self._anchor_rows(xx[None, :], np.asarray([wu], dtype=float), wu * xx, wu * np.outer(xx, xx))
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
        chunk_ux = checked.T @ wu
        chunk_uxx = (checked * wu[:, None]).T @ checked
        # Conditioning gate BEFORE the raw fold, so an activation converts only pre-chunk content.
        self._anchor_rows(checked, wu, chunk_ux, chunk_uxx)
        self.count += float(np.sum(checked_weights, dtype=np.float64))
        self.sum_u += float(np.sum(wu, dtype=np.float64))
        self.sum_ux += chunk_ux
        self.sum_uxx += chunk_uxx

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, np.ndarray, np.ndarray]) -> "MultivariateStudentTAccumulator":
        """Merge another multivariate Student-t sufficient-statistic tuple."""
        count, sum_u, sum_ux, sum_uxx, inferred_dim = student_t_moments(suff_stat, self.dim)
        if sum_ux is not None:
            self._ensure_dim(inferred_dim)
            self._absorb_anchored(suff_stat, sum_ux, sum_uxx, sum_u)
            self.sum_ux += sum_ux
            self.sum_uxx += sum_uxx
        self.count += count
        self.sum_u += sum_u
        return self

    def _absorb_anchored(
        self,
        suff_stat: Any,
        sum_ux: np.ndarray,
        sum_uxx: np.ndarray,
        sum_u: float,
    ) -> None:
        """Fold another statistic into the anchored track. Call BEFORE the raw fold.

        An incoming payload merges by Chan's parallel form: re-express its moments about this
        accumulator's anchor. The anchor gap is between two data values, so every term stays
        ``O(sum_u * spread^2)`` and no large-offset cancellation is reintroduced. A raw-only
        statistic joining an already-anchored pool converts about our anchor instead -- see
        :meth:`_activate_anchor` for why that cancellation-prone conversion is acceptable here.
        """
        anchored = _consistent_anchored_student_t(suff_stat, sum_ux, sum_u, self.dim)
        if anchored is not None:
            b_anchor, b_ux, b_uxx = anchored
            if self._anchor is None:
                self._activate_anchor(b_anchor)
            gap = b_anchor - self._anchor
            cross = np.outer(gap, b_ux)
            self._anchored_ux += b_ux + sum_u * gap
            self._anchored_uxx += b_uxx + cross + cross.T + sum_u * np.outer(gap, gap)
        elif self._anchor is not None and (sum_u != 0.0 or np.any(sum_ux != 0.0) or np.any(sum_uxx != 0.0)):
            a = self._anchor
            self._anchored_ux += sum_ux - a * sum_u
            scatter = sum_uxx - np.outer(a, sum_ux) - np.outer(sum_ux, a) + sum_u * np.outer(a, a)
            self._anchored_uxx += 0.5 * (scatter + scatter.T)

    def value(self) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
        """Return count, latent-weight total, weighted sum, and weighted second moment.

        A plain 4-tuple for every consumer that treats it as one -- ``student_t_moments`` included;
        once the shift-anchored track is live it is a :class:`MultivariateStudentTSuffStat` that
        additionally carries those moments in its ``anchored`` attribute, so :meth:`combine` can
        fold them in and :meth:`MultivariateStudentTEstimator.estimate` can form a shift-invariant
        scale matrix.
        """
        raw = (
            self.count,
            self.sum_u,
            None if self.sum_ux is None else self.sum_ux.copy(),
            None if self.sum_uxx is None else self.sum_uxx.copy(),
        )
        if self._anchor is None:
            return raw
        return MultivariateStudentTSuffStat(
            *raw,
            anchored=(self._anchor.copy(), self._anchored_ux.copy(), self._anchored_uxx.copy()),
        )

    def from_value(
        self, x: tuple[float, float, np.ndarray | None, np.ndarray | None]
    ) -> "MultivariateStudentTAccumulator":
        """Replace accumulator contents from sufficient statistics."""
        self.count, self.sum_u, self.sum_ux, self.sum_uxx, self.dim = student_t_moments(
            x,
            self.dim,
        )
        anchored = _consistent_anchored_student_t(x, self.sum_ux, self.sum_u, self.dim)
        if anchored is None:
            # Raw-only statistics replace the state: the anchored track restarts unactivated, and a
            # later activation (first update / anchored merge) converts this content then.
            self._anchor = None
            self._anchored_ux = None
            self._anchored_uxx = None
        else:
            self._anchor, self._anchored_ux, self._anchored_uxx = (
                anchored[0].copy(),
                anchored[1].copy(),
                anchored[2].copy(),
            )
        return self

    def scale(self, c: float) -> "MultivariateStudentTAccumulator":
        """Scale the accumulated statistics in place by ``c``, anchored track included.

        The structural default round-trips through ``value()``/``from_value()``, and
        ``scale_suff_stat`` rebuilds the payload as a PLAIN tuple -- which drops the anchored track
        and undoes the repair. Scaling every weight by ``c`` scales both anchored moments by ``c``
        and leaves the anchor (a data value, not a statistic) alone.
        """
        factor = float(c)
        self.count *= factor
        self.sum_u *= factor
        if self.sum_ux is not None:
            self.sum_ux = self.sum_ux * factor
            self.sum_uxx = self.sum_uxx * factor
        if self._anchor is not None:
            self._anchored_ux = self._anchored_ux * factor
            self._anchored_uxx = self._anchored_uxx * factor
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
    """Fixed-dof EM estimator for the multivariate Student's t location and scale matrix.

    The scale matrix is formed from shift-anchored statistics whenever the accumulated sufficient
    statistics carry that payload (see :class:`MultivariateStudentTAccumulator`), which makes the
    fit SHIFT-EQUIVARIANT: estimating on ``x + c`` returns ``loc + c`` with the scale matrix
    unchanged, for any constant ``c`` the data can carry -- epoch seconds (~1.7e9) and 2**31
    included. With a plain raw tuple -- statistics the conditioning gate never needed to anchor, or
    ones that arrived already reduced from an engine kernel or an older serialization -- the
    historical raw path is used bit-identically, and ``estimate`` warns rather than returning a
    scale matrix it cannot stand behind.

    Residual limit, and it is representational rather than algorithmic. One M-step is
    shift-equivariant to ~1e-15, but a full EM run is not exactly so: the latent reweighting
    ``u_i = (nu + p)/(nu + delta_i)`` is a function of ``x_i - mu``, and ``mu`` is a float64 at the
    data's own magnitude, so it can only be placed on that grid -- 1.2e-4 apart at offset 1e12.
    Snapping the un-shifted fit's ``mu`` onto exactly that grid reproduces the shifted fit to
    3e-14, which is what identifies the mechanism. The effect on the fitted scale matrix is bounded
    by roughly ``ulp(abs(mu)) / spread`` per iteration: 4e-8 at epoch seconds, 1e-7 at 2**31, 7e-5
    at 1e12. Every consumer of a fitted location faces the same granularity, so there is no more
    precise answer to return; subtract a constant origin before fitting if a tighter one is needed.
    """

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

        # Sigma = sum_i w_i u_i (x_i - mu)(x_i - mu)' / sum_i w_i.
        #
        # ``scale`` below is the magnitude of the terms this M-step actually differences, and the
        # PSD guard's tolerance is relative to it. On the anchored path those terms are the
        # anchor-relative ones, so the tolerance tracks the cancellation that is really possible
        # rather than being inflated by the data's distance from the origin -- at offset 1.7e9 the
        # raw-path tolerance was ~1e24 and waved through a scale matrix that had lost every digit.
        anchored = _consistent_anchored_student_t(suff_stat, sum_ux, sum_u, inferred_dim)
        if anchored is not None:
            anchor, a_ux, a_uxx = anchored
            offset = a_ux / sum_u
            mu = anchor + offset
            scatter = a_uxx - np.outer(a_ux, offset)
            terms = (
                float(np.linalg.norm(a_uxx, ord=2)),
                float(np.linalg.norm(np.outer(a_ux, offset), ord=2)),
            )
        else:
            warn_uncorrectable_vector_moments(sum_ux, np.diag(sum_uxx), sum_u, family="multivariate Student-t")
            mu = sum_ux / sum_u
            scatter = sum_uxx - np.outer(mu, sum_ux) - np.outer(sum_ux, mu) + sum_u * np.outer(mu, mu)
            terms = (
                float(np.linalg.norm(sum_uxx, ord=2)),
                float(np.linalg.norm(np.outer(mu, sum_ux), ord=2)),
                float(np.linalg.norm(sum_u * np.outer(mu, mu), ord=2)),
            )
        shape = scatter / count
        shape = 0.5 * (shape + shape.T)
        scale = max(*terms, 1.0) / count
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
