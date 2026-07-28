"""Von Mises-Fisher distributions on unit spheres.

Data type: Union[Sequence[float], np.ndarray] (a unit-norm vector on the (p-1)-sphere in R^p).

The von Mises-Fisher (vMF) distribution is defined on the (p-1)-sphere in R^{p}. Assume x_mat = (X_1,..,X_p) follows a vMF
distribution with mean direction vector mu = (mu_1, mu_2, ..., mu_p) s.t. ||mu||=1 and concentration parameter
kappa > 0. The vMF log-density is

    log(f(x; mu, kappa)) = log(c_p(kappa)) + kappa * dot(mu, x),

where dot is a dot product and
    log(c_p(kappa)) = (p/2-1)log(kappa) - (p/2)*log(2*pi) + log(B_{p/2-1}(kappa)), where

log(B_{p/2-1}(kappa)) = denotes the modified Bessel function of the first kind at order p/2-1.

Numerical notes:
    Evaluating log I_v(kappa) directly with scipy.special.iv overflows for large kappa, and the
    exponentially scaled scipy.special.ive underflows when the order v = p/2 - 1 is large relative to
    kappa (high dimension with modest concentration). The helper lniv() therefore uses log(ive) + kappa
    where ive has support and falls back to the uniform large-order asymptotic expansion
    (Abramowitz & Stegun 9.7.7) implemented in lniv_uniform() when ive underflows. Both the normalizing
    constant and the Bessel-ratio Newton iteration in VonMisesFisherEstimator.estimate() rely on lniv().



Reference: Mardia & Jupp, *Directional Statistics* (Wiley, 2000).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.linalg
import scipy.special
from numpy.random import RandomState
from scipy.special import gammaln

import mixle.utils.vector as vec
from mixle.engines.arithmetic import *
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.matrix.wishart import (
    _validated_dimension,
    _validated_sample_size,
    _validated_weight,
    _validated_weights,
)
from mixle.utils.vector import owned_backend_parameter

_UNIT_NORM_ATOL = 1.0e-8
_RESULTANT_ATOL = 1.0e-10


class VonMisesFisherFitError(RuntimeError):
    """Raised when vMF sufficient statistics have no finite identifiable fit."""


def _unit_vector(value: Any, dim: int, name: str = "vMF observation") -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if result.shape != (dim,):
        raise ValueError("%s must have exact shape (%d,)" % (name, dim))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must be finite" % name)
    norm = float(np.linalg.norm(result))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=_UNIT_NORM_ATOL):
        raise ValueError("%s must have unit norm" % name)
    return result.copy()


def _unit_batch(value: Any, dim: int, name: str = "vMF observations") -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if result.shape == (0,):
        return np.empty((0, dim), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != dim:
        raise ValueError("%s must have exact shape (N, %d)" % (name, dim))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must be finite" % name)
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=_UNIT_NORM_ATOL):
        raise ValueError("%s must contain only unit vectors" % name)
    return result.copy()


def _unit_backend_array(
    value: Any,
    dim: int,
    name: str = "vMF backend observations",
) -> np.ndarray:
    """Validate row or broadcast-expanded backend vectors on their last axis."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if result.ndim not in (2, 3) or result.shape[-1] != dim:
        raise ValueError("%s must have rank two or three with final dimension %d" % (name, dim))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must be finite" % name)
    norms = np.linalg.norm(result, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=_UNIT_NORM_ATOL):
        raise ValueError("%s must contain only unit vectors" % name)
    return result.copy()


def _validated_vmf_statistics(
    value: Any,
    expected_dim: int | None,
) -> tuple[float, np.ndarray | None, int | None]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("vMF sufficient statistics must be a two-item tuple")
    if isinstance(value[0], (bool, np.bool_)) or np.ndim(value[0]) != 0:
        raise TypeError("vMF sufficient-statistic count must be a real scalar")
    try:
        count = float(value[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("vMF sufficient-statistic count must be numeric") from exc
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("vMF sufficient-statistic count must be finite and non-negative")
    if value[1] is None:
        if count != 0.0:
            raise ValueError("vMF non-empty statistics require a vector sum")
        return count, None, expected_dim
    try:
        vector_sum = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("vMF vector sum must be numeric") from exc
    if vector_sum.ndim != 1 or vector_sum.size < 2:
        raise ValueError("vMF vector sum must be a vector of dimension at least two")
    if expected_dim is not None and vector_sum.shape != (expected_dim,):
        raise ValueError("vMF vector sum must have exact shape (%d,)" % expected_dim)
    if np.any(~np.isfinite(vector_sum)):
        raise ValueError("vMF vector sum must be finite")
    if count == 0.0 and np.any(vector_sum != 0.0):
        raise ValueError("empty vMF statistics must have zero vector sum")
    norm = float(np.linalg.norm(vector_sum))
    if norm > count + _RESULTANT_ATOL * max(1.0, count):
        raise ValueError("vMF vector sum violates the unit-vector resultant bound")
    return count, vector_sum.copy(), len(vector_sum)


def _mean_resultant_length(dim: int, kappa: float) -> float:
    if kappa == 0.0:
        return 0.0
    log_kappa = np.log(kappa)
    result = np.exp(lniv(dim / 2.0, log_kappa) - lniv((dim / 2.0) - 1.0, log_kappa))
    if not np.isfinite(result) or result < 0.0 or result >= 1.0:
        raise VonMisesFisherFitError("vMF Bessel ratio is not finite and interior")
    return float(result)


def lniv_uniform(v, ln_z):
    """log I_v(z) by the uniform large-order asymptotic (A&S 9.7.7):

        I_v(v t) ~ exp(v eta) / (sqrt(2 pi v) (1 + t^2)^{1/4}),
        eta = sqrt(1 + t^2) + log(t / (1 + sqrt(1 + t^2))).

    Valid uniformly in t = z/v for large v, including t -> 0 where it reduces
    to the small-argument form (z/2)^v / Gamma(v+1) via Stirling.

    Args:
        v (float): Order of the modified Bessel function. Must be positive.
        ln_z (float): Log of the (positive) argument z.

    Returns:
        Approximate value of log I_v(z) as a float.
    """
    if v == 0:
        if not np.isfinite(ln_z):
            return 0.0
        z = np.exp(ln_z)
        if z == 0.0:
            return 0.0
        rv0 = scipy.special.i0e(z)
        if rv0 > 0.0 and np.isfinite(rv0):
            return np.log(rv0) + z
        return z - 0.5 * np.log(2.0 * np.pi * z)

    t = np.exp(ln_z - np.log(v))
    s = np.sqrt(1.0 + t * t)
    eta = s + np.log(t) - np.log1p(s)
    return v * eta - 0.5 * np.log(2.0 * np.pi * v) - 0.25 * np.log1p(t * t)


def lniv(v, ln_z):
    """Numerically stable log I_v(e^{ln_z}).

    Uses the exponentially scaled Bessel function where it has support and the
    uniform large-order expansion where ive underflows (large v relative to z;
    ive cannot underflow for v = 0, so that branch always has v > 0).

    Args:
        v (float): Order of the modified Bessel function. Must be non-negative.
        ln_z (float): Log of the argument z. May be -inf (z = 0).

    Returns:
        log I_v(z) as a float (-inf when z = 0 and v > 0).
    """
    if not np.isfinite(ln_z):
        return 0.0 if v == 0 else -np.inf

    z = np.exp(ln_z)
    rv0 = scipy.special.ive(v, z)

    if rv0 > 0 and np.isfinite(rv0):
        return np.log(rv0) + z

    return lniv_uniform(v, ln_z)


class VonMisesFisherDistribution(SequenceEncodableProbabilityDistribution):
    """Von Mises-Fisher distribution on the (p-1)-sphere with mean direction mu and concentration kappa.

    Data type: Union[Sequence[float], np.ndarray] (a unit-norm vector in R^p).
    """

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for von Mises-Fisher generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the von Mises-Fisher distribution."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="von_mises_fisher",
            distribution_type=cls,
            parameters=(
                ParameterSpec("mu", constraint="real_vector"),
                ParameterSpec("kappa"),
                ParameterSpec("log_const", constraint="real", differentiable=False),
            ),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum", kind="vector_moment"),
            ),
            support="unit_vector",
            differentiable=False,
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return row-wise legacy sufficient statistics for resident reductions."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x):
            xx = engine.asarray(x)
            one = engine.sum(xx * 0.0, axis=1) + engine.asarray(1.0)
            return one, xx
        mu = np.asarray(params["mu"], dtype=np.float64)
        xx = engine.asarray(_unit_backend_array(x, mu.shape[-1]))
        one = engine.sum(xx * 0.0, axis=1) + engine.asarray(1.0)
        return one, xx

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, kappa: Any, log_const: Any, engine: Any) -> Any:
        """Engine-neutral von Mises-Fisher log-density from fitted parameters."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x) or is_symbolic_payload(mu):
            xx = engine.asarray(x)
            return engine.sum(xx * mu, axis=-1) * kappa + log_const
        checked_mu = np.asarray(mu, dtype=np.float64)
        if checked_mu.ndim not in (1, 2, 3) or checked_mu.shape[-1] < 2:
            raise ValueError("vMF backend mean direction has invalid geometry")
        if np.any(~np.isfinite(checked_mu)) or not np.allclose(
            np.linalg.norm(checked_mu, axis=-1),
            1.0,
            rtol=0.0,
            atol=_UNIT_NORM_ATOL,
        ):
            raise ValueError("vMF backend mean direction must contain unit vectors")
        checked_kappa = np.asarray(kappa, dtype=np.float64)
        checked_log_const = np.asarray(log_const, dtype=np.float64)
        if np.any(~np.isfinite(checked_kappa)) or np.any(checked_kappa < 0.0):
            raise ValueError("vMF backend concentration must be finite and non-negative")
        if np.any(~np.isfinite(checked_log_const)):
            raise ValueError("vMF backend normalizer must be finite")
        xx = engine.asarray(_unit_backend_array(x, checked_mu.shape[-1]))
        return engine.sum(xx * mu, axis=-1) * kappa + log_const

    def __init__(
        self,
        mu: Sequence[float] | np.ndarray,
        kappa: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a von Mises-Fisher distribution on the unit sphere.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean direction vector. Norm should be 1.0.
            kappa (float): Positive valued concentration parameter.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            name (Optional[str]): Optional distribution name.
            dim (int): Length of mu (dimension for vmf-distribution).
            mu (np.ndarray): Mean direction vector. Norm should be 1.0.
            kappa (float): Positive valued concentration parameter.
            log_const (float): Normalizing constant for vmf distribution.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        try:
            raw_mu = np.asarray(mu, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("VonMisesFisherDistribution requires numeric mu.") from exc
        if raw_mu.ndim != 1 or raw_mu.size < 2:
            raise ValueError("VonMisesFisherDistribution requires a vector of dimension at least two.")
        dim = len(raw_mu)
        checked_mu = _unit_vector(raw_mu, dim, "vMF mean direction")
        if isinstance(kappa, (bool, np.bool_)) or np.ndim(kappa) != 0:
            raise TypeError("VonMisesFisherDistribution requires scalar concentration.")
        try:
            checked_kappa = float(kappa)
        except (TypeError, ValueError) as exc:
            raise TypeError("VonMisesFisherDistribution requires scalar concentration.") from exc
        if not np.isfinite(checked_kappa) or checked_kappa < 0:
            # kappa == 0 is the legitimate uniform-density limit (handled below); only reject what a
            # bare `kappa > 0` comparison silently lets through the "else: uniform" branch as if it were
            # that limit -- most importantly NaN, since `nan > 0` is False just like a genuine 0 is.
            raise ValueError(
                "VonMisesFisherDistribution requires kappa to be finite and non-negative, got %r." % (kappa,)
            )

        if checked_kappa > 0:
            # log c_p(kappa) = (p/2 - 1) log kappa - (p/2) log(2 pi) - log I_{p/2-1}(kappa)
            v = (dim / 2.0) - 1.0
            log_kappa = np.log(checked_kappa)
            self.log_const = v * log_kappa - (dim / 2.0) * np.log(2.0 * pi) - lniv(v, log_kappa)
        else:
            # uniform density on the (p-1)-sphere: Gamma(p/2) / (2 pi^{p/2})
            self.log_const = gammaln(dim / 2.0) - np.log(2.0) - (dim / 2.0) * np.log(pi)

        if not np.isfinite(self.log_const):
            raise ValueError("VonMisesFisherDistribution normalizer must be finite.")
        self.name = name
        self.dim = None if dim is None else _validated_dimension(dim, "vMF accumulator dimension")
        if self.dim is not None and self.dim < 2:
            raise ValueError("vMF accumulator dimension must be at least two")
        self.mu = checked_mu
        self.mu.setflags(write=False)
        self.kappa = checked_kappa
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the von Mises-Fisher distribution."""
        s1 = repr(self.mu.tolist())
        s2 = repr(self.kappa)
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return "VonMisesFisherDistribution(%s, %s, name=%s, keys=%s)" % (s1, s2, s3, s4)

    def density(self, x: Sequence[float] | np.ndarray) -> float:
        """Density of von Mises-Fisher distribution at observation x.

        See log_density() for details.

        Args:
            x (Union[Sequence[float], np.ndarray]): Unit-norm vector in R^p.

        Returns:
            Density at observation x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: Sequence[float] | np.ndarray) -> float:
        """Log-density of von Mises-Fisher distribution at observation x.

        The log-density is given by

            log(f(x; mu, kappa)) = log(c_p(kappa)) + kappa * dot(mu, x),

        for x on the (p-1)-sphere. When kappa = 0 this reduces to the uniform density on the sphere.

        Args:
            x (Union[Sequence[float], np.ndarray]): Unit-norm vector in R^p.

        Returns:
            Log-density at observation x.

        """
        z = _unit_vector(x, self.dim)
        return np.dot(z, self.mu) * self.kappa + self.log_const

    def density_cumulative(self, x: Sequence[float] | np.ndarray) -> float:
        """Exact probability-ordered cumulative ``G(x) = P(p(Y) >= p(x))`` (the HDR mass at x).

        A coordinate-wise CDF is undefined on the sphere (no total order), but since the density is
        monotone in the cosine ``t = mu . x`` (``p(y) >= p(x)`` iff ``mu.y >= mu.x`` for ``kappa >= 0``),
        the highest-density-region mass is the upper tail of the cosine marginal, whose density is
        ``f(s) proportional to exp(kappa s) (1 - s^2)^((p-3)/2)`` on ``[-1, 1]``. ``G`` is that tail
        integral (computed by quadrature; the ``exp(kappa(s-1))`` shift keeps it stable for large
        kappa and cancels in the ratio). Returned to density_rank as method ``exact-analytic``.
        """
        from scipy.integrate import quad

        xx = _unit_vector(x, self.dim)
        if self.kappa == 0.0:
            return 1.0
        t = float(np.clip(np.dot(self.mu, xx), -1.0, 1.0))
        k = float(self.kappa)
        a = (self.dim - 3.0) / 2.0

        def f(s: float) -> float:
            return exp(k * (s - 1.0)) * (max(1.0 - s * s, 0.0) ** a)

        num, _ = quad(f, t, 1.0, limit=200)
        den, _ = quad(f, -1.0, 1.0, limit=200)
        return float(min(1.0, max(0.0, num / den))) if den > 0.0 else 0.0

    def density_quantile(self, q: float) -> np.ndarray:
        """Inverse of :meth:`density_cumulative`: a representative unit vector at cumulative-density ``q``.

        ``q`` is the highest-density-region mass; since the density is monotone in the cosine
        ``t = mu . y``, the boundary is the cosine ``t_q`` with tail mass ``q``, found by bisection on
        the cosine marginal. The returned representative is a unit vector at that cosine from ``mu``
        (``t_q * mu + sqrt(1 - t_q^2) * perp`` for a fixed ``perp`` orthogonal to ``mu``). Sweeping ``q``
        enumerates the sphere in descending density (concentric caps about ``mu``).
        """
        from scipy.integrate import quad

        qf = float(q)
        if not 0.0 <= qf <= 1.0:
            raise ValueError("q must be in [0, 1].")
        if self.kappa == 0.0:
            if qf != 1.0:
                raise ValueError("uniform vMF density has no probability-ordered quantile below one")
            return self.mu.copy()
        k = float(self.kappa)
        a = (self.dim - 3.0) / 2.0

        def f(s: float) -> float:
            return exp(k * (s - 1.0)) * (max(1.0 - s * s, 0.0) ** a)

        den, _ = quad(f, -1.0, 1.0, limit=200)

        def tail(t: float) -> float:
            return quad(f, t, 1.0, limit=200)[0] / den if den > 0.0 else 0.0

        # tail(t) decreases from 1 at t=-1 to 0 at t=1; bisect for tail(t_q) = q.
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if tail(mid) > qf:
                lo = mid
            else:
                hi = mid
        t_q = 0.5 * (lo + hi)
        # A unit direction orthogonal to mu (use whichever axis is least aligned with mu).
        axis = int(np.argmin(np.abs(self.mu)))
        e = np.zeros(self.dim)
        e[axis] = 1.0
        perp = e - np.dot(e, self.mu) * self.mu
        norm = float(np.linalg.norm(perp))
        perp = perp / norm if norm > 0.0 else e
        return t_q * self.mu + float(np.sqrt(max(0.0, 1.0 - t_q * t_q))) * perp

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): 2-d numpy array of N unit-norm vectors with p columns.

        Returns:
            Numpy array of log-density (float) of length N.

        """
        checked = _unit_batch(x, self.dim)
        return np.dot(checked, self.mu) * self.kappa + self.log_const

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded unit-vector observations."""
        return self.backend_log_density_from_params(
            _unit_batch(x, self.dim),
            engine.asarray(owned_backend_parameter(self.mu)),
            engine.asarray(self.kappa),
            engine.asarray(self.log_const),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["VonMisesFisherDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for equal-dimensional von Mises-Fisher mixtures."""
        if not dists:
            raise ValueError("stacked vMF parameters require at least one component")
        if any(not isinstance(dist, cls) for dist in dists):
            raise TypeError("stacked vMF parameters require matching distributions")
        dim = int(dists[0].dim)
        if any(int(dist.dim) != dim for dist in dists):
            raise ValueError("Stacked VonMisesFisherDistribution components require equal dimension.")
        return {
            "__pysp_component_axis__": {"mu": 0, "kappa": 0, "log_const": 0},
            "mu": engine.asarray([dist.mu for dist in dists]),
            "kappa": engine.asarray([dist.kappa for dist in dists]),
            "log_const": engine.asarray([dist.log_const for dist in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of von Mises-Fisher component log densities."""
        raw_mu = np.asarray(params["mu"], dtype=np.float64)
        if raw_mu.ndim != 2 or raw_mu.shape[1] < 2:
            raise ValueError("stacked vMF means must have shape (components, dimension)")
        if np.any(~np.isfinite(raw_mu)) or not np.allclose(
            np.linalg.norm(raw_mu, axis=1),
            1.0,
            rtol=0.0,
            atol=_UNIT_NORM_ATOL,
        ):
            raise ValueError("stacked vMF means must be finite unit vectors")
        xx = engine.asarray(_unit_batch(x, raw_mu.shape[1]))
        return engine.matmul(xx, params["mu"].T) * params["kappa"][None, :] + params["log_const"][None, :]

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked legacy ``(count, weighted_vector_sum)`` statistics."""
        dim = int(params["mu"].shape[1])
        checked_x = _unit_batch(x, dim)
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.ndim != 2 or raw_weights.shape[0] != len(checked_x):
            raise ValueError("vMF stacked weights must have exact shape (observations, components)")
        if np.any(~np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
            raise ValueError("vMF stacked weights must be finite and non-negative")
        xx = engine.asarray(checked_x)
        ww = engine.asarray(raw_weights)
        return engine.sum(ww, axis=0), engine.matmul(ww.T, xx)

    def sampler(self, seed: int | None = None) -> "VonMisesFisherSampler":
        """Create a sampler from this von Mises-Fisher distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            VonMisesFisherSampler configured from this distribution.

        """
        return VonMisesFisherSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "VonMisesFisherEstimator":
        """Create an estimator for a von Mises-Fisher distribution.

        Args:
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).

        Returns:
            VonMisesFisherEstimator configured with this distribution's name and keys.

        """
        if pseudo_count is not None:
            raise ValueError("vMF pseudo-count regularization is not implemented")
        return VonMisesFisherEstimator(
            dim=self.dim,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "VonMisesFisherDataEncoder":
        """Return the encoder for von Mises-Fisher observations."""
        return VonMisesFisherDataEncoder(self.dim)


class VonMisesFisherSampler(DistributionSampler):
    """Sampler for the VonMisesFisherDistribution using Wood's rejection sampling scheme."""

    def __init__(self, dist: "VonMisesFisherDistribution", seed: int | None = None) -> None:
        """Create a sampler for a von Mises-Fisher distribution.

        Args:
            dist (VonMisesFisherDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw iid unit-norm vectors from the von Mises-Fisher distribution.

        Args:
            size (Optional[int]): Number of samples to draw. If None, a single vector is returned.

        Returns:
            Numpy array of shape (dim,) if size is None, else of shape (size, dim).

        """
        rng1 = np.random.RandomState(self.rng.randint(maxrandint))
        rng2 = np.random.RandomState(self.rng.randint(maxrandint))
        rng3 = np.random.RandomState(self.rng.randint(maxrandint))

        d = self.dist.dim
        mu = self.dist.mu
        k = self.dist.kappa

        t1 = np.sqrt(4.0 * k * k + (d - 1.0) * (d - 1.0))
        # b = (d-1.0)/(t1 + 2*k)
        b = (t1 - 2 * k) / (d - 1.0)
        x0 = (1.0 - b) / (1.0 + b)

        m = (d - 1.0) / 2.0
        c = k * x0 + (d - 1.0) * np.log(1 - x0 * x0)

        sz = 1 if size is None else _validated_sample_size(size)
        if sz == 0:
            return np.empty((0, d), dtype=np.float64)

        QQ = np.zeros((d, d), dtype=float)
        QQ[0, :] = mu
        _, s, vh = scipy.linalg.svd(QQ)
        QQ = vh[np.abs(s) < 0.1, :].T  # (d, d-1) orthonormal complement of mu

        # Wood's tangent coordinate w, drawn by *batched* rejection: draw blocks of (z, u), accept
        # where t - c >= log u, and accumulate sz accepted values. Wood's scheme accepts in O(1)
        # expected draws, so the budget below is only a guard against a pathological non-terminating
        # loop (the per-draw `while True` it replaces had no such guard).
        w = np.empty(sz)
        filled = 0
        for _ in range(10_000):
            if filled >= sz:
                break
            block = max(sz - filled, 64)
            z = rng1.beta(m, m, size=block)
            u = rng2.rand(block)
            ww = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            t = k * ww + (d - 1) * np.log(1.0 - x0 * ww)
            acc = (t - c) >= np.log(u)
            take = min(int(acc.sum()), sz - filled)
            if take:
                w[filled : filled + take] = ww[acc][:take]
                filled += take
        if filled < sz:
            raise RuntimeError(
                "VonMisesFisherSampler exceeded the rejection budget (dim=%d, kappa=%g); acceptance was "
                "near zero." % (d, k)
            )

        # tangential directions: sz unit vectors in the complement of mu, then combine with w
        v = rng3.randn(sz, d - 1) @ QQ.T  # (sz, d)
        v /= np.sqrt(np.einsum("ij,ij->i", v, v))[:, None]
        rv = np.sqrt(1.0 - w * w)[:, None] * v + w[:, None] * mu[None, :]

        return rv[0, :] if size is None else rv


class VonMisesFisherAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for the VonMisesFisherDistribution. Tracks the weighted vector sum and total weight."""

    def __init__(self, dim: int | None = None, name: str | None = None, keys: str | None = None) -> None:
        """Create an accumulator for von Mises-Fisher sufficient statistics.

        Args:
            dim (Optional[int]): Dimension p of the observations. If None, set from data on first update.
            name (Optional[str]): Optional accumulator name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            dim (Optional[int]): Dimension p of the observations.
            count (float): Sum of observation weights.
            ssum (Optional[np.ndarray]): Weighted sum of observation vectors. None until dim is known.
            key (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        """
        self.dim = dim
        self.count = 0.0

        if dim is not None:
            self.ssum = vec.zeros(self.dim)
        else:
            self.ssum = None

        self.keys = keys
        self.name = name

    def update(
        self, x: Sequence[float] | np.ndarray, weight: float, estimate: VonMisesFisherDistribution | None
    ) -> None:
        """Update sufficient statistics with a weighted observation.

        Args:
            x (Union[Sequence[float], np.ndarray]): Unit-norm vector in R^p.
            weight (float): Weight for observation.
            estimate (Optional[VonMisesFisherDistribution]): Previous estimate (unused).

        """
        if self.dim is None:
            try:
                candidate = np.asarray(x, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("vMF observation must be numeric") from exc
            if candidate.ndim != 1 or candidate.size < 2:
                raise ValueError("vMF observation dimension must be at least two")
            dim = len(candidate)
        else:
            dim = self.dim
        checked_x = _unit_vector(x, dim)
        checked_weight = _validated_weight(weight)
        if self.dim is None:
            self.dim = dim
            self.ssum = vec.zeros(dim)
        self.ssum += checked_x * checked_weight
        self.count += checked_weight

    def initialize(self, x: Sequence[float] | np.ndarray, weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with a weighted observation.

        Args:
            x (Union[Sequence[float], np.ndarray]): Unit-norm vector in R^p.
            weight (float): Weight for observation.
            rng (RandomState): Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: VonMisesFisherDistribution | None) -> None:
        """Vectorized update of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N unit-norm vectors with p columns.
            weights (np.ndarray): Weights for each of the N observations.
            estimate (Optional[VonMisesFisherDistribution]): Previous estimate (unused).

        """
        raw_x = np.asarray(x)
        if self.dim is None:
            if raw_x.ndim != 2 or raw_x.shape[1] < 2:
                raise ValueError("vMF observations require dimension at least two")
            dim = raw_x.shape[1]
        else:
            dim = self.dim
        checked_x = _unit_batch(x, dim)
        checked_weights = _validated_weights(weights, len(checked_x))
        if self.dim is None:
            self.dim = dim
            self.ssum = vec.zeros(dim)
        self.ssum += np.matmul(checked_weights, checked_x)
        self.count += float(checked_weights.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N unit-norm vectors with p columns.
            weights (np.ndarray): Weights for each of the N observations.
            rng (RandomState): Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "VonMisesFisherAccumulator":
        """Combine sufficient statistics from another accumulator into this one.

        Args:
            suff_stat (Tuple[float, np.ndarray]): Tuple of count and weighted vector sum.

        Returns:
            Self, with aggregated sufficient statistics.

        """
        count, vector_sum, dim = _validated_vmf_statistics(suff_stat, self.dim)
        if vector_sum is not None and self.ssum is not None:
            combined_count = self.count + count
            combined_sum = self.ssum + vector_sum
            _validated_vmf_statistics(
                (combined_count, combined_sum),
                self.dim,
            )
            self.ssum = combined_sum
            self.count = combined_count
        elif vector_sum is not None:
            self.dim = dim
            self.ssum = vector_sum
            self.count = count

        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Returns sufficient statistics as a Tuple of count and weighted vector sum."""
        return self.count, None if self.ssum is None else self.ssum.copy()

    def from_value(self, x: tuple[float, np.ndarray]) -> "VonMisesFisherAccumulator":
        """Set sufficient statistics of accumulator from value x.

        Args:
            x (Tuple[float, np.ndarray]): Tuple of count and weighted vector sum.

        """
        count, vector_sum, dim = _validated_vmf_statistics(x, self.dim)
        self.ssum = vector_sum
        self.count = count
        if dim is not None:
            self.dim = dim
        return self

    def scale(self, c: float) -> "VonMisesFisherAccumulator":
        """Scale linear vMF sufficient statistics."""
        checked_scale = _validated_weight(c)
        self.count *= checked_scale
        if self.ssum is not None:
            self.ssum *= checked_scale
        return self

    def acc_to_encoder(self) -> "VonMisesFisherDataEncoder":
        """Return the encoder associated with this accumulator."""
        return VonMisesFisherDataEncoder(self.dim)


class VonMisesFisherAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating VonMisesFisherAccumulator objects."""

    def __init__(self, dim: int | None = None, name: str | None = None, keys: str | None = None) -> None:
        """Create a factory for von Mises-Fisher accumulators.

        Args:
            dim (Optional[int]): Dimension p of the observations. If None, set from data.
            name (Optional[str]): Optional name assigned to created accumulators.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.dim = None if dim is None else _validated_dimension(dim, "vMF accumulator dimension")
        if self.dim is not None and self.dim < 2:
            raise ValueError("vMF accumulator dimension must be at least two")
        self.keys = keys
        self.name = name

    def make(self) -> "SequenceEncodableStatisticAccumulator":
        """Return a fresh von Mises-Fisher accumulator."""
        return VonMisesFisherAccumulator(
            dim=self.dim,
            name=self.name,
            keys=self.keys,
        )


class VonMisesFisherEstimator(ParameterEstimator):
    """Estimator for the VonMisesFisherDistribution using the Banerjee et al. approximation for kappa."""

    def __init__(
        self,
        dim: int | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an estimator for von Mises-Fisher parameters.

        Args:
            dim (Optional[int]): Dimension p of the observations. If None, set from data.
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.dim = None if dim is None else _validated_dimension(dim, "vMF estimator dimension")
        if self.dim is not None and self.dim < 2:
            raise ValueError("vMF estimator dimension must be at least two")
        if pseudo_count is not None:
            raise ValueError("vMF pseudo-count regularization is not implemented")
        self.name = name
        self.pseudo_count = None
        self.keys = keys

    def accumulator_factory(self):
        """Return a factory for von Mises-Fisher accumulators."""
        return VonMisesFisherAccumulatorFactory(dim=self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> "VonMisesFisherDistribution":
        """Estimate a VonMisesFisherDistribution from sufficient statistics.

        The mean direction is the normalized weighted vector sum. The
        concentration solves ``A_p(kappa) = rhat`` using a finite bracket and
        Brent root certificate. Boundary resultants with ``rhat=1`` have only
        an infinite-concentration MLE and raise a typed fit error.

        Args:
            nobs (Optional[float]): Number of observations (unused).
            suff_stat (Tuple[float, np.ndarray]): Tuple of count and weighted vector sum.

        Returns:
            A fitted distribution carrying convergence and identifiability metadata.

        """
        from scipy.optimize import brentq

        count, ssum, dim = _validated_vmf_statistics(suff_stat, self.dim)
        if count == 0.0 or ssum is None or dim is None:
            raise VonMisesFisherFitError("vMF fitting requires positive observation weight")
        ssum_norm = float(np.linalg.norm(ssum))
        if ssum_norm == 0.0:
            mu = np.ones(dim) / np.sqrt(dim)
            fitted = VonMisesFisherDistribution(
                mu,
                0.0,
                name=self.name,
                keys=self.keys,
            )
            fitted.fit_metadata = {
                "converged": True,
                "solver": "uniform-resultant",
                "identifiable_direction": False,
                "resultant_length": 0.0,
                "score": 0.0,
                "repairs": (),
            }
            return fitted

        rhat = ssum_norm / count
        if rhat >= 1.0 - _RESULTANT_ATOL:
            raise VonMisesFisherFitError("vMF resultant is on the unit boundary and has no finite concentration MLE")
        mu = ssum / ssum_norm

        def score(kappa: float) -> float:
            return _mean_resultant_length(dim, kappa) - rhat

        lower = 0.0
        upper = max(
            1.0,
            rhat * (dim - rhat * rhat) / (1.0 - rhat * rhat),
        )
        upper_score = score(upper)
        bracket_iterations = 0
        while upper_score < 0.0 and upper < 1.0e12:
            upper *= 2.0
            upper_score = score(upper)
            bracket_iterations += 1
        if not np.isfinite(upper_score) or upper_score < 0.0:
            raise VonMisesFisherFitError("vMF concentration fit has no certified finite bracket")
        try:
            kappa, result = brentq(
                score,
                lower,
                upper,
                xtol=1.0e-10,
                full_output=True,
                disp=False,
            )
        except (ValueError, RuntimeError) as exc:
            raise VonMisesFisherFitError("vMF concentration root solver failed") from exc
        residual = float(score(float(kappa)))
        if not result.converged or not np.isfinite(residual) or abs(residual) > 1.0e-8:
            raise VonMisesFisherFitError("vMF concentration fit lacks an optimality certificate")
        fitted = VonMisesFisherDistribution(
            mu,
            float(kappa),
            name=self.name,
            keys=self.keys,
        )
        fitted.fit_metadata = {
            "converged": True,
            "solver": "brentq",
            "identifiable_direction": True,
            "resultant_length": rhat,
            "score": residual,
            "iterations": bracket_iterations + int(result.iterations),
            "bracket": (lower, upper),
            "repairs": (),
        }
        return fitted


class VonMisesFisherDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of unit-norm vector observations."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else _validated_dimension(dim, "vMF encoder dimension")
        if self.dim is not None and self.dim < 2:
            raise ValueError("vMF encoder dimension must be at least two")

    def __str__(self) -> str:
        """Return the von Mises-Fisher encoder's display name."""
        return "VonMisesFisherDataEncoder(%s)" % repr(self.dim)

    def __eq__(self, other) -> bool:
        """Return true when ``other`` is a von Mises-Fisher data encoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a VonMisesFisherDataEncoder instance, else False.

        """
        return isinstance(other, VonMisesFisherDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Encode a sequence of N unit-norm vectors for vectorized functions.

        Args:
            x (Union[Sequence[float], np.ndarray]): Sequence of N unit-norm vectors in R^p.

        Returns:
            2-d numpy array with N rows and p columns.

        """
        if self.dim is None:
            try:
                raw = np.asarray(x, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("vMF observations must be numeric") from exc
            if raw.shape == (0,):
                raise ValueError("cannot infer vMF dimension from an empty observation batch")
            if raw.ndim != 2 or raw.shape[1] < 2:
                raise ValueError("vMF observations require dimension at least two")
            return _unit_batch(raw, raw.shape[1])
        return _unit_batch(x, self.dim)

    def row_count(self, x: np.ndarray) -> int:
        """Return the row count after validating unit-vector geometry."""
        if self.dim is None:
            encoded = self.seq_encode(x)
            return len(encoded)
        return len(_unit_batch(x, self.dim))
