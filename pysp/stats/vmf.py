"""Create, estimate, and sample from a von Mises-Fisher distribution.

Defines the VonMisesFisherDistribution, VonMisesFisherSampler, VonMisesFisherAccumulatorFactory,
VonMisesFisherAccumulator, VonMisesFisherEstimator, and the VonMisesFisherDataEncoder classes for use with pysparkplug.

Data type: Union[Sequence[float], np.ndarray] (a unit-norm vector on the (p-1)-sphere in R^p).

The von Mises-Fisher (vmf) distribution on the (p-1) sphere in R^{p}. Assume x_mat = (X_1,..,X_p) follows a vmf
distribution with mean direction vector mu = (mu_1, mu_2, ..., mu_p) s.t. ||mu||=1 and concentration parameter
kappa > 0. The vmf log-density if given by

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

"""

import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.linalg
import scipy.special
from numpy.random import RandomState
from scipy.special import gammaln

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


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
        from pysp.stats.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

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
        xx = engine.asarray(x)
        one = engine.sum(xx * 0.0, axis=1) + engine.asarray(1.0)
        return one, xx

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, kappa: Any, log_const: Any, engine: Any) -> Any:
        """Engine-neutral von Mises-Fisher log-density from fitted parameters."""
        xx = engine.asarray(x)
        return engine.sum(xx * mu, axis=-1) * kappa + log_const

    def __init__(
        self,
        mu: Sequence[float] | np.ndarray,
        kappa: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """VonMisesFisherDistribution object.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean direction vector. Norm should be 1.0.
            kappa (float): Positive valued concentration parameter.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional keys for object instance.

        Attributes:
            name (Optional[str]): Optional name for object instance.
            dim (int): Length of mu (dimension for vmf-distribution).
            mu (np.ndarray): Mean direction vector. Norm should be 1.0.
            kappa (float): Positive valued concentration parameter.
            log_const (float): Normalizing constant for vmf distribution.
            keys (Optional[str]): Optional keys for object instance.

        """
        dim = len(mu)
        mu = np.asarray(mu).copy()

        if kappa > 0:
            # log c_p(kappa) = (p/2 - 1) log kappa - (p/2) log(2 pi) - log I_{p/2-1}(kappa)
            v = (dim / 2.0) - 1.0
            log_kappa = np.log(kappa)
            self.log_const = v * log_kappa - (dim / 2.0) * np.log(2.0 * pi) - lniv(v, log_kappa)
        else:
            # uniform density on the (p-1)-sphere: Gamma(p/2) / (2 pi^{p/2})
            self.log_const = gammaln(dim / 2.0) - np.log(2.0) - (dim / 2.0) * np.log(pi)

        self.name = name
        self.dim = dim
        self.mu = mu
        self.kappa = kappa
        self.key = keys

    def __str__(self) -> str:
        """Returns string representation of VonMisesFisherDistribution object."""
        s1 = repr(list(self.mu))
        s2 = repr(self.kappa)
        s3 = repr(self.name)
        s4 = self.key
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
        z = np.asarray(x).copy()
        return np.dot(z, self.mu) * self.kappa + self.log_const

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): 2-d numpy array of N unit-norm vectors with p columns.

        Returns:
            Numpy array of log-density (float) of length N.

        """
        return np.dot(x, self.mu) * self.kappa + self.log_const

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded unit-vector observations."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.mu),
            engine.asarray(self.kappa),
            engine.asarray(self.log_const),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["VonMisesFisherDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for equal-dimensional von Mises-Fisher mixtures."""
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
        xx = engine.asarray(x)
        return engine.matmul(xx, params["mu"].T) * params["kappa"][None, :] + params["log_const"][None, :]

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked legacy ``(count, weighted_vector_sum)`` statistics."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        return engine.sum(ww, axis=0), engine.matmul(ww.T, xx)

    def sampler(self, seed: int | None = None) -> "VonMisesFisherSampler":
        """Create a VonMisesFisherSampler object from parameters of VonMisesFisherDistribution instance.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            VonMisesFisherSampler object.

        """
        return VonMisesFisherSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "VonMisesFisherEstimator":
        """Create a VonMisesFisherEstimator object.

        Args:
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).

        Returns:
            VonMisesFisherEstimator object.

        """
        if pseudo_count is None:
            return VonMisesFisherEstimator(name=self.name, keys=self.key)
        else:
            return VonMisesFisherEstimator(name=self.name, keys=self.key)

    def dist_to_encoder(self) -> "VonMisesFisherDataEncoder":
        """Returns a VonMisesFisherDataEncoder object for encoding sequences of data."""
        return VonMisesFisherDataEncoder()


class VonMisesFisherSampler(DistributionSampler):
    """Sampler for the VonMisesFisherDistribution using Wood's rejection sampling scheme."""

    def __init__(self, dist: "VonMisesFisherDistribution", seed: int | None = None) -> None:
        """VonMisesFisherSampler object.

        Args:
            dist (VonMisesFisherDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> np.ndarray:
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

        sz = 1 if size is None else size
        rv = np.zeros((sz, d))

        QQ = np.zeros((d, d), dtype=float)
        QQ[0, :] = mu
        _, s, vh = scipy.linalg.svd(QQ)
        QQ = vh[np.abs(s) < 0.1, :].T

        for i in range(sz):
            t = c - 1
            u = 1

            while (t - c) < np.log(u):
                z = rng1.beta(m, m)
                u = rng2.rand()
                w = (1.0 - (1.0 + b) * z) / (1.0 - (1 - b) * z)
                t = k * w + (d - 1) * np.log(1.0 - x0 * w)

            v = rng3.randn(d - 1)
            v = np.dot(QQ, v)
            v /= np.sqrt(np.dot(v, v))
            rv[i, :] = np.sqrt(1 - w * w) * v + w * mu

        if size is None:
            return rv[0, :]
        else:
            return rv


class VonMisesFisherAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for the VonMisesFisherDistribution. Tracks the weighted vector sum and total weight."""

    def __init__(self, dim: int | None = None, name: str | None = None, keys: str | None = None) -> None:
        """VonMisesFisherAccumulator object.

        Args:
            dim (Optional[int]): Dimension p of the observations. If None, set from data on first update.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            dim (Optional[int]): Dimension p of the observations.
            count (float): Sum of observation weights.
            ssum (Optional[np.ndarray]): Weighted sum of observation vectors. None until dim is known.
            key (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional name for object instance.

        """
        self.dim = dim
        self.count = 0.0

        if dim is not None:
            self.ssum = vec.zeros(dim)
        else:
            self.ssum = None

        self.key = keys
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
            self.dim = len(x)
            self.ssum = vec.zeros(self.dim)

        self.ssum += x * weight
        self.count += weight

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

        Non-finite or negative weights are dropped from the vector sum.

        Args:
            x (np.ndarray): 2-d numpy array of N unit-norm vectors with p columns.
            weights (np.ndarray): Weights for each of the N observations.
            estimate (Optional[VonMisesFisherDistribution]): Previous estimate (unused).

        """
        if self.dim is None:
            self.dim = x.shape[1]
            self.ssum = vec.zeros(self.dim)

        good_w = np.bitwise_and(np.isfinite(weights), weights >= 0)
        if np.all(good_w):
            x_weight = np.multiply(x.T, weights)
        else:
            x_weight = np.multiply(x[good_w, :].T, weights[good_w])

        self.count += weights.sum()
        self.ssum += x_weight.sum(axis=1)

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
        if suff_stat[1] is not None and self.ssum is not None:
            self.ssum += suff_stat[1]
            self.count += suff_stat[0]

        elif suff_stat[1] is not None and self.ssum is None:
            self.ssum = suff_stat[1]
            self.count = suff_stat[0]

        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Returns sufficient statistics as a Tuple of count and weighted vector sum."""
        return self.count, self.ssum

    def from_value(self, x: tuple[float, np.ndarray]) -> "VonMisesFisherAccumulator":
        """Set sufficient statistics of accumulator from value x.

        Args:
            x (Tuple[float, np.ndarray]): Tuple of count and weighted vector sum.

        """
        self.ssum = x[1]
        self.count = x[0]
        self.dim = None if self.ssum is None else len(self.ssum)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to accumulators with shared sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.combine(stats_dict[self.key].value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Set sufficient statistics of object instance to suff stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to accumulators with shared sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "VonMisesFisherDataEncoder":
        """Returns a VonMisesFisherDataEncoder object for encoding sequences of data."""
        return VonMisesFisherDataEncoder()


class VonMisesFisherAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for creating VonMisesFisherAccumulator objects."""

    def __init__(self, dim: int | None = None, name: str | None = None, keys: str | None = None) -> None:
        """VonMisesFisherAccumulatorFactory object.

        Args:
            dim (Optional[int]): Dimension p of the observations. If None, set from data.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.dim = dim
        self.key = keys
        self.name = name

    def make(self) -> "SequenceEncodableStatisticAccumulator":
        """Returns a new VonMisesFisherAccumulator object."""
        return VonMisesFisherAccumulator(dim=self.dim, keys=self.key)


class VonMisesFisherEstimator(ParameterEstimator):
    """Estimator for the VonMisesFisherDistribution using the Banerjee et al. approximation for kappa."""

    def __init__(
        self,
        dim: int | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """VonMisesFisherEstimator object.

        Args:
            dim (Optional[int]): Dimension p of the observations. If None, set from data.
            pseudo_count (Optional[float]): Kept for interface consistency (has no effect on estimation).
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.dim = dim
        self.name = name
        self.pseudo_count = pseudo_count
        self.name = name
        self.key = keys

    def accumulator_factory(self):
        """Returns a VonMisesFisherAccumulatorFactory for creating VonMisesFisherAccumulator objects."""
        return VonMisesFisherAccumulatorFactory(dim=self.dim, name=self.name, keys=self.key)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> "VonMisesFisherDistribution":
        """Estimate a VonMisesFisherDistribution from sufficient statistics.

        The mean direction mu is the normalized weighted vector sum. The concentration kappa solves
        A_p(kappa) = rhat (rhat = ||ssum|| / count), initialized with the closed-form Banerjee et al.
        approximation and refined with up to three Newton steps. The Bessel-function ratio A_p(kappa)
        is evaluated through lniv() so large orders p/2 fall back to the uniform large-order asymptotic
        instead of underflowing. Two guards keep the solution finite: rhat is clamped below 1 (rhat -> 1
        sends kappa -> inf), and Newton refinement is skipped within 1e-9 of 1 where A_p'(kappa) -> 0
        makes the iteration ill-conditioned while the initializer is already accurate.

        Args:
            nobs (Optional[float]): Number of observations (unused).
            suff_stat (Tuple[float, np.ndarray]): Tuple of count and weighted vector sum.

        Returns:
            VonMisesFisherDistribution object (uniform on the sphere, kappa = 0, if no data observed).

        """
        count, ssum = suff_stat
        dim = len(ssum)

        def _newton(p, r, k):
            k = max(sys.float_info.min, k)
            # apk = scipy.special.iv(p/2.0, k)/scipy.special.iv((p/2.0)-1.0, k)
            apk = np.exp(lniv(p / 2.0, np.log(k)) - lniv((p / 2.0) - 1.0, np.log(k)))

            rv = k - (apk - r) / (1.0 - apk * apk - ((p - 1.0) / k) * apk)
            rv = max(sys.float_info.min, rv)
            return rv

        ssum_norm = np.sqrt(np.dot(ssum, ssum))

        if ssum_norm > 0 and count > 0:
            # rhat -> 1 means kappa -> inf; clamp so the Banerjee initializer
            # and Newton refinement stay finite
            rhat = min(ssum_norm / count, 1.0 - 1.0e-10)
            mu = ssum / ssum_norm

            k = rhat * (dim - (rhat * rhat)) / (1.0 - (rhat * rhat))

            # Newton refinement of A_p(k) = rhat; near rhat = 1 the Banerjee
            # initializer is already accurate and Newton is ill-conditioned
            # (A_p'(k) -> 0), so leave the closed-form value
            if rhat < 1.0 - 1.0e-9:
                for i in range(3):
                    k = _newton(dim, rhat, k)

        else:
            mu = np.ones(dim) / np.sqrt(dim)
            k = 0.0

        return VonMisesFisherDistribution(mu, k, name=self.name)


class VonMisesFisherDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of unit-norm vector observations."""

    def __str__(self) -> str:
        """Returns string representation of VonMisesFisherDataEncoder object."""
        return "VonMisesFisherDataEncoder"

    def __eq__(self, other) -> bool:
        """Checks if other object is an instance of a VonMisesFisherDataEncoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is a VonMisesFisherDataEncoder instance, else False.

        """
        return isinstance(other, VonMisesFisherDataEncoder)

    def seq_encode(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Encode a sequence of N unit-norm vectors for vectorized functions.

        Args:
            x (Union[Sequence[float], np.ndarray]): Sequence of N unit-norm vectors in R^p.

        Returns:
            2-d numpy array with N rows and p columns.

        """
        rv = np.asarray(x).copy()
        return rv
