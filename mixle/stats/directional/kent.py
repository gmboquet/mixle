"""Kent (Fisher-Bingham FB5) distribution on the 2-sphere.

The Kent distribution is the elliptical analogue of the von Mises-Fisher law on ``S^2`` (unit vectors in
``R^3``). Its density at a unit vector ``x`` is

    f(x) = c(kappa, beta)^{-1} exp( kappa * (g1 . x) + beta * [(g2 . x)^2 - (g3 . x)^2] ),

where ``G = [g1, g2, g3]`` is a ``3 x 3`` orthonormal orientation (``g1`` the mean direction, ``g2`` the
major axis, ``g3`` the minor axis), ``kappa > 0`` is the concentration and ``0 <= 2 beta < kappa`` the
ovalness. ``beta = 0`` recovers von Mises-Fisher (circular contours); ``beta -> kappa/2`` gives highly
elliptical (girdle-like) contours. It is the standard model for asymmetric clusters of orientations
(palaeomagnetism, structural geology, spherical data).

Normalizer (verified to 1e-12 against arbitrary-precision integration over ``S^2``):

    c(kappa, beta) = 2 pi sum_{j>=0} [Gamma(j+1/2)/Gamma(j+1)] beta^{2j} (2/kappa)^{2j+1/2} I_{2j+1/2}(kappa),

evaluated in log space with exponentially scaled Bessel functions for numerical stability. Sampling is by
exact von Mises-Fisher-envelope rejection; ``kappa, beta`` and the orientation are fit by Kent's moment
method (mean direction + tangential scatter eigenvectors) followed by a maximum-likelihood refinement of
``(kappa, beta)``.

Reference: Kent, "The Fisher-Bingham distribution on the sphere", *J. Royal Statistical Society B* 44
(1982); Mardia & Jupp, *Directional Statistics* (2000), ch. 9.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln, ive, logsumexp

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_LOG_2PI = math.log(2.0 * math.pi)


def _log_kent_norm(kappa: float, beta: float) -> float:
    """Return ``log c(kappa, beta)`` via the stable log-space Bessel series (``ive`` + log-sum-exp)."""
    log_terms = []
    j = 0
    while j <= 500:
        log_coef = gammaln(j + 0.5) - gammaln(j + 1) + (2 * j + 0.5) * math.log(2.0 / kappa)
        if beta > 0.0:
            log_coef += 2 * j * math.log(beta)
        elif j > 0:
            break  # beta == 0 (von Mises-Fisher): only the j = 0 term survives
        iv = ive(2 * j + 0.5, kappa)
        if iv > 0.0:
            log_terms.append(log_coef + math.log(iv))
            if j > 2 and log_terms[-1] < max(log_terms) - 38.0:  # term ~ 1e-16 of the running max
                break
        j += 1
    return _LOG_2PI + kappa + float(logsumexp(log_terms))


class KentDistribution(SequenceEncodableProbabilityDistribution):
    """Kent (FB5) distribution on ``S^2`` with orientation ``gamma`` (3x3), concentration and ovalness."""

    def __init__(
        self, gamma: np.ndarray, kappa: float, beta: float, name: str | None = None, keys: str | None = None
    ) -> None:
        g = np.asarray(gamma, dtype=np.float64)
        if g.shape != (3, 3):
            raise ValueError("KentDistribution gamma must be a 3x3 orthonormal matrix (columns g1, g2, g3).")
        if kappa <= 0.0 or not np.isfinite(kappa):
            raise ValueError("KentDistribution requires finite kappa > 0.")
        if beta < 0.0 or 2.0 * beta >= kappa:
            raise ValueError("KentDistribution requires 0 <= 2*beta < kappa.")
        self.gamma = g
        self.kappa = float(kappa)
        self.beta = float(beta)
        self.name = name
        self.keys = keys
        self._log_c = _log_kent_norm(self.kappa, self.beta)

    def __str__(self) -> str:
        return "KentDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.gamma.tolist()),
            repr(self.kappa),
            repr(self.beta),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Return the Kent density at one unit 3-vector."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the log-density at a unit 3-vector ``x``."""
        v = np.asarray(x, dtype=np.float64)
        p = v @ self.gamma  # (g1.x, g2.x, g3.x)
        return -self._log_c + self.kappa * p[0] + self.beta * (p[1] * p[1] - p[2] * p[2])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded ``(n, 3)`` array of unit vectors."""
        p = np.asarray(x, dtype=np.float64) @ self.gamma  # (n, 3) projections onto the frame
        return -self._log_c + self.kappa * p[:, 0] + self.beta * (p[:, 1] ** 2 - p[:, 2] ** 2)

    # --- compute-engine backend (numpy + torch/GPU), SCORING only: the normalizer is a host scalar
    # (Kummer / Bingham constants via scipy), the data math is engine matmul + quadratics. The scatter
    # accumulator stays host-side, so torch accelerates mixture E-step scoring with a bit-correct M-step. ---
    @classmethod
    def compute_capabilities(cls):
        """Declare NumPy/Torch scoring capabilities for Kent log-density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for ``(N, 3)`` unit vectors."""
        p = engine.matmul(engine.asarray(x), engine.asarray(self.gamma))
        p1, p2 = p[:, 1], p[:, 2]
        return -self._log_c + self.kappa * p[:, 0] + self.beta * (p1 * p1 - p2 * p2)

    def sampler(self, seed: int | None = None) -> "KentSampler":
        """Return an exact rejection sampler for this Kent distribution."""
        return KentSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "KentEstimator":
        """Return Kent's moment/ML estimator for orientation, concentration, and ovalness."""
        return KentEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "KentDataEncoder":
        """Return the unit-vector encoder used by vectorized methods."""
        return KentDataEncoder()


class KentSampler(DistributionSampler):
    """Sample by von Mises-Fisher-envelope rejection (exact)."""

    def __init__(self, dist: KentDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _batch(self, n: int) -> np.ndarray:
        kappa, beta = self.dist.kappa, self.dist.beta
        g1, g2, g3 = self.dist.gamma[:, 0], self.dist.gamma[:, 1], self.dist.gamma[:, 2]
        out = np.empty((n, 3))
        filled = 0
        while filled < n:
            m = (n - filled) * 2 + 8  # oversample to amortize rejection
            u = self.rng.uniform(size=m)
            w = 1.0 + np.log(u + (1.0 - u) * math.exp(-2.0 * kappa)) / kappa  # vMF cos-angle from g1
            phi = self.rng.uniform(0.0, 2.0 * math.pi, size=m)
            # vMF(g1, kappa) envelope; accept with prob exp(beta[(1-w^2) cos 2phi - 1])
            accept = self.rng.uniform(size=m) < np.exp(beta * ((1.0 - w * w) * np.cos(2.0 * phi) - 1.0))
            wa, pa = w[accept], phi[accept]
            k = min(len(wa), n - filled)
            if k == 0:
                continue
            s = np.sqrt(np.maximum(1.0 - wa[:k] ** 2, 0.0))
            out[filled : filled + k] = (
                wa[:k, None] * g1 + (s * np.cos(pa[:k]))[:, None] * g2 + (s * np.sin(pa[:k]))[:, None] * g3
            )
            filled += k
        return out

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one unit vector or ``size`` iid unit vectors."""
        if size is None:
            return self._batch(1)[0]
        return list(self._batch(int(size)))


class KentAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate ``(count, sum_x (3,), sum_xx (3,3))`` -- the sufficient statistics for the moment fit."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum_x = np.zeros(3)
        self.sum_xx = np.zeros((3, 3))
        self.name = name
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: KentDistribution | None) -> None:
        """Update first- and second-moment statistics from one weighted vector."""
        v = np.asarray(x, dtype=np.float64)
        self.count += weight
        self.sum_x += weight * v
        self.sum_xx += weight * np.outer(v, v)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize moment statistics from one weighted vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update moment statistics from encoded unit vectors."""
        v = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        self.count += float(w.sum())
        self.sum_x += v.T @ w
        self.sum_xx += (v * w[:, None]).T @ v

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize moment statistics from encoded unit vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> "KentAccumulator":
        """Merge weighted count, vector sum, and scatter matrix statistics."""
        self.count += suff_stat[0]
        self.sum_x += suff_stat[1]
        self.sum_xx += suff_stat[2]
        return self

    def value(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Return weighted count, vector sum, and scatter matrix."""
        return self.count, self.sum_x, self.sum_xx

    def from_value(self, x: tuple[float, np.ndarray, np.ndarray]) -> "KentAccumulator":
        """Restore weighted count, vector sum, and scatter matrix."""
        self.count = float(x[0])
        self.sum_x = np.asarray(x[1], dtype=np.float64).copy()
        self.sum_xx = np.asarray(x[2], dtype=np.float64).copy()
        return self

    def acc_to_encoder(self) -> "KentDataEncoder":
        """Return the encoder compatible with Kent moment statistics."""
        return KentDataEncoder()


class KentAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for KentAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> KentAccumulator:
        """Create an empty Kent accumulator."""
        return KentAccumulator(name=self.name, keys=self.keys)


class KentEstimator(ParameterEstimator):
    """Kent's moment estimator for the orientation, with an ML refinement of ``(kappa, beta)``."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> KentAccumulatorFactory:
        """Return a factory for Kent sufficient-statistic accumulators."""
        return KentAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> KentDistribution:
        """Estimate orientation, concentration, and ovalness from moment statistics."""
        from scipy.optimize import minimize

        count, sum_x, sum_xx = suff_stat
        if count <= 0.0:
            return KentDistribution(np.eye(3), 1.0, 0.0, name=self.name, keys=self.keys)
        xbar = sum_x / count
        scatter = sum_xx / count
        r1 = float(np.linalg.norm(xbar))
        g1 = xbar / r1 if r1 > 1e-12 else np.array([0.0, 0.0, 1.0])

        # build any orthonormal tangent basis {h2, h3} perpendicular to g1
        seed_vec = np.array([1.0, 0.0, 0.0]) if abs(g1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        h2 = seed_vec - (seed_vec @ g1) * g1
        h2 /= np.linalg.norm(h2)
        h3 = np.cross(g1, h2)
        # diagonalize the 2x2 tangential scatter -> the Kent rotation angle psi
        t22, t23, t33 = h2 @ scatter @ h2, h2 @ scatter @ h3, h3 @ scatter @ h3
        psi = 0.5 * math.atan2(2.0 * t23, t22 - t33)
        g2 = math.cos(psi) * h2 + math.sin(psi) * h3
        g3 = -math.sin(psi) * h2 + math.cos(psi) * h3
        # ensure g2 is the major axis (larger tangential variance)
        if g2 @ scatter @ g2 < g3 @ scatter @ g3:
            g2, g3 = g3, -g2
        gamma = np.column_stack([g1, g2, g3])

        # moment sufficient statistics in the fitted frame, then ML refine (kappa, beta)
        r2 = float(g2 @ scatter @ g2 - g3 @ scatter @ g3)

        def neg_ll(theta: np.ndarray) -> float:
            kappa = math.exp(theta[0])
            beta = 0.5 * kappa / (1.0 + math.exp(-theta[1]))  # 0 <= 2 beta < kappa via a logistic link
            return _log_kent_norm(kappa, beta) - kappa * r1 - beta * r2

        # initialize from the large-concentration Kent moment approximation
        q = max(t22 + t33 - 2.0 * min(t22, t33), 1e-6)
        k0 = max(1.0 / max(2.0 - 2.0 * r1, 1e-3), 1.0)
        res = minimize(
            neg_ll,
            np.array([math.log(k0), 0.0]),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
        )
        kappa = math.exp(res.x[0])
        beta = 0.5 * kappa / (1.0 + math.exp(-res.x[1]))
        beta = min(beta, 0.4999 * kappa)
        return KentDistribution(gamma, kappa, beta, name=self.name, keys=self.keys)


class KentDataEncoder(DataSequenceEncoder):
    """Encode unit vectors as a normalized ``(n, 3)`` float array."""

    def __str__(self) -> str:
        return "KentDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KentDataEncoder)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Normalize and encode observations as an ``(n, 3)`` array."""
        v = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
