"""Watson distribution -- a rotationally symmetric distribution for *axial* data on the sphere.

Axial data are unit vectors identified with their antipodes (``x`` and ``-x`` are the same), e.g. fibre
or crystal orientations, where the von Mises-Fisher (which distinguishes ``x`` from ``-x``) does not
apply. The Watson distribution on ``S^{p-1}`` concentrates around an axis ``mu`` with shape ``kappa``:

    f(x; mu, kappa) = M(1/2, p/2, kappa)^{-1} / omega_p * exp(kappa (mu^T x)^2),

where ``M`` is Kummer's confluent hypergeometric function and ``omega_p = 2 pi^{p/2} / Gamma(p/2)`` is
the sphere's surface area. ``kappa > 0`` is *bipolar* (mass near the axis +/-mu), ``kappa < 0``
*girdle* (mass on the equator orthogonal to mu); both are antipodally symmetric. It is fit by maximum
likelihood: ``mu`` is the leading (kappa>0) or trailing (kappa<0) eigenvector of the scatter matrix,
and ``kappa`` solves ``E[(mu^T x)^2] = mu^T S mu`` (a monotone 1-D equation in the Kummer ratio).


Reference: Mardia & Jupp, *Directional Statistics* (Wiley, 2000).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import hyp1f1

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _kummer_ratio(kappa: float, p: int) -> float:
    """``E[(mu^T x)^2]`` under Watson = ``M'(1/2,p/2,k)/M(1/2,p/2,k) = (1/p) M(3/2,(p+2)/2,k)/M(1/2,p/2,k)``."""
    return (1.0 / p) * hyp1f1(1.5, (p + 2) / 2.0, kappa) / hyp1f1(0.5, p / 2.0, kappa)


def _solve_kappa(r: float, p: int, lo: float = -700.0, hi: float = 700.0) -> float:
    """Solve the monotone ``E[(mu^T x)^2](kappa) = r`` for ``kappa`` by bisection (``r`` in (0, 1))."""
    if r <= _kummer_ratio(lo, p):
        return lo
    if r >= _kummer_ratio(hi, p):
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _kummer_ratio(mid, p) < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class WatsonDistribution(SequenceEncodableProbabilityDistribution):
    """Watson distribution on the unit sphere ``S^{p-1}`` with axis ``mu`` and concentration ``kappa``."""

    def __init__(self, mu: np.ndarray, kappa: float, name: str | None = None, keys: str | None = None) -> None:
        m = np.asarray(mu, dtype=np.float64)
        if m.ndim != 1 or not np.isfinite(kappa):
            raise ValueError("mu must be a 1-D unit vector and kappa finite")
        norm = np.linalg.norm(m)
        if norm == 0.0:
            raise ValueError("mu must be non-zero")
        self.mu = m / norm
        self.dim = m.shape[0]
        self.kappa = float(kappa)
        self.name = name
        self.keys = keys
        log_omega = math.log(2.0) + (self.dim / 2.0) * math.log(math.pi) - math.lgamma(self.dim / 2.0)
        self._log_const = -log_omega - math.log(hyp1f1(0.5, self.dim / 2.0, self.kappa))

    def __str__(self) -> str:
        return "WatsonDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu.tolist()),
            repr(self.kappa),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the density at a single unit vector ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the log-density at a single unit vector ``x``."""
        dot = float(np.dot(np.asarray(x, dtype=np.float64), self.mu))
        return self._log_const + self.kappa * dot * dot

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of unit vectors, shape ``(N, p)``."""
        dots = np.asarray(x, dtype=np.float64) @ self.mu
        return self._log_const + self.kappa * dots * dots

    # --- compute-engine backend (numpy + torch/GPU), SCORING only: the normalizer is a host scalar
    # (Kummer / Bingham constants via scipy), the data math is engine matmul + quadratics. The scatter
    # accumulator stays host-side, so torch accelerates mixture E-step scoring with a bit-correct M-step. ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Watson scoring kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for ``(N, p)`` unit vectors."""
        dots = engine.matmul(engine.asarray(x), engine.asarray(self.mu))
        return self._log_const + self.kappa * dots * dots

    def sampler(self, seed: int | None = None) -> "WatsonSampler":
        """Return a sampler for drawing unit vectors from this distribution."""
        return WatsonSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WatsonEstimator":
        """Return a maximum-likelihood estimator (scatter eigenvector + Kummer-ratio kappa solve)."""
        return WatsonEstimator(self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WatsonDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return WatsonDataEncoder()


class WatsonSampler(DistributionSampler):
    """Draw axes by sampling ``s = |mu^T x|`` via a numerical inverse-CDF, then a random orthogonal tangent."""

    def __init__(self, dist: WatsonDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        # density of s in [0,1] (the substitution t = s^2 removes the t^{-1/2} singularity):
        #   h(s) propto exp(kappa s^2) (1 - s^2)^{(p-3)/2}
        s = np.linspace(0.0, 1.0, 4000)
        log_h = dist.kappa * s * s + ((dist.dim - 3) / 2.0) * np.log1p(-np.clip(s * s, 0.0, 1.0 - 1e-15))
        h = np.exp(log_h - log_h.max())
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(s))])
        self._s_grid = s
        self._cdf = cdf / cdf[-1]
        # orthonormal complement of mu (columns span the tangent space)
        q = np.eye(dist.dim) - np.outer(dist.mu, dist.mu)
        u_, sv_, _ = np.linalg.svd(q)
        self._tangent = u_[:, sv_ > 1e-9]  # (p, p-1)

    def sample(self, size: int | None = None) -> np.ndarray:
        """Draw one unit vector or a stack of iid unit vectors."""
        d = self.dist
        n = 1 if size is None else int(size)
        s = np.interp(self.rng.uniform(size=n), self._cdf, self._s_grid)  # |mu^T x|
        sign = np.where(self.rng.uniform(size=n) < 0.5, 1.0, -1.0)
        v = self.rng.randn(n, d.dim - 1) @ self._tangent.T  # (n, p) in the tangent space
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        x = (s * sign)[:, None] * d.mu[None, :] + np.sqrt(1.0 - s * s)[:, None] * v
        return x[0] if size is None else x


class WatsonAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted scatter matrix ``S = sum_i w_i x_i x_i^T`` and total weight."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.scatter = np.zeros((dim, dim), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: WatsonDistribution | None) -> None:
        """Accumulate one weighted outer product into the scatter matrix."""
        xx = np.asarray(x, dtype=np.float64)
        self.scatter += weight * np.outer(xx, xx)
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one unit vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: WatsonDistribution | None) -> None:
        """Accumulate weighted scatter statistics from encoded unit vectors."""
        xx = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        self.scatter += (xx * w[:, None]).T @ xx
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded unit vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "WatsonAccumulator":
        """Merge another Watson sufficient-statistic tuple."""
        self.scatter += suff_stat[0]
        self.count += suff_stat[1]
        return self

    def value(self) -> tuple[np.ndarray, float]:
        """Return the scatter matrix and total weight."""
        return self.scatter.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, float]) -> "WatsonAccumulator":
        """Replace accumulator contents from scatter statistics."""
        self.scatter = np.asarray(x[0], dtype=np.float64).copy()
        self.count = float(x[1])
        self.dim = self.scatter.shape[0]
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

    def acc_to_encoder(self) -> "WatsonDataEncoder":
        """Return the encoder used by this accumulator."""
        return WatsonDataEncoder()


class WatsonAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WatsonAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.name = name
        self.keys = keys

    def make(self) -> WatsonAccumulator:
        """Create a fresh Watson accumulator."""
        return WatsonAccumulator(self.dim, name=self.name, keys=self.keys)


class WatsonEstimator(ParameterEstimator):
    """Maximum-likelihood estimator: scatter eigenvector for the axis, Kummer-ratio solve for kappa."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WatsonAccumulatorFactory:
        """Return an accumulator factory for Watson scatter statistics."""
        return WatsonAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float]) -> WatsonDistribution:
        """Estimate the Watson axis and concentration from weighted scatter."""
        scatter, count = suff_stat
        p = self.dim
        if count <= 0.0:
            return WatsonDistribution(np.eye(p)[0], 0.0, name=self.name, keys=self.keys)
        s_mat = scatter / count  # mean scatter; eigenvalues in [0,1], sum to 1
        eigval, eigvec = np.linalg.eigh(s_mat)
        # bipolar (kappa>0): the data align with the top eigenvector; girdle (kappa<0): the bottom one
        r_top, r_bot = float(eigval[-1]), float(eigval[0])
        if abs(r_top - 1.0 / p) >= abs(r_bot - 1.0 / p):
            mu, r = eigvec[:, -1], r_top
        else:
            mu, r = eigvec[:, 0], r_bot
        kappa = _solve_kappa(min(max(r, 1.0e-6), 1.0 - 1.0e-6), p)
        return WatsonDistribution(mu, kappa, name=self.name, keys=self.keys)


class WatsonDataEncoder(DataSequenceEncoder):
    """Encode a sequence of unit vectors as an ``(N, p)`` float array."""

    def __str__(self) -> str:
        return "WatsonDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WatsonDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode unit vectors as an ``(N, p)`` floating-point array."""
        return np.asarray(x, dtype=np.float64)
