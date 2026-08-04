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
from scipy.special import expit, gammaln, ive, logsumexp

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.directional.von_mises_fisher import (
    _unit_batch,
    _unit_vector,
)
from mixle.stats.matrix.wishart import (
    _validated_sample_size,
    _validated_weight,
    _validated_weights,
)
from mixle.utils.vector import owned_backend_parameter

from ._circular import symmetrized_scatter

_MOMENT_ATOL = 1.0e-8


class KentFitError(RuntimeError):
    """Raised when Kent moments have no certified finite fit."""


class KentSamplingError(RuntimeError):
    """Raised when Kent rejection sampling exhausts its proposal budget."""

    def __init__(
        self,
        accepted: int,
        proposed: int,
        kappa: float,
        beta: float,
    ):
        self.accepted = accepted
        self.proposed = proposed
        self.kappa = kappa
        self.beta = beta
        super().__init__(
            "Kent rejection sampler accepted %d of %d proposals (kappa=%g, beta=%g)" % (accepted, proposed, kappa, beta)
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
    if not log_terms:
        raise ValueError("Kent normalizer series produced no finite terms")
    result = _LOG_2PI + kappa + float(logsumexp(log_terms))
    if not np.isfinite(result):
        raise ValueError("Kent normalizer series is non-finite")
    return result


def _validated_kent_statistics(
    value: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Kent sufficient statistics must be a three-item tuple")
    if isinstance(value[0], (bool, np.bool_)) or np.ndim(value[0]) != 0:
        raise TypeError("Kent count must be a real scalar")
    try:
        count = float(value[0])
        vector_sum = np.asarray(value[1], dtype=np.float64)
        scatter = np.asarray(value[2], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Kent sufficient statistics must be numeric") from exc
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("Kent count must be finite and non-negative")
    if vector_sum.shape != (3,) or np.any(~np.isfinite(vector_sum)):
        raise ValueError("Kent vector sum must be a finite length-three vector")
    if scatter.shape != (3, 3) or np.any(~np.isfinite(scatter)):
        raise ValueError("Kent scatter must be a finite 3x3 matrix")
    scatter = symmetrized_scatter(scatter, "Kent")
    if count == 0.0:
        if np.any(vector_sum != 0.0) or np.any(scatter != 0.0):
            raise ValueError("empty Kent statistics must have zero moments")
    else:
        tolerance = _MOMENT_ATOL * max(1.0, count)
        if float(np.linalg.norm(vector_sum)) > count + tolerance:
            raise ValueError("Kent vector sum violates the unit-vector resultant bound")
        if abs(float(np.trace(scatter)) - count) > tolerance:
            raise ValueError("Kent scatter trace must equal its observation weight")
        if float(np.linalg.eigvalsh(scatter).min()) < -tolerance:
            raise ValueError("Kent scatter must be positive semidefinite")
        centered = scatter - np.outer(vector_sum, vector_sum) / count
        if float(np.linalg.eigvalsh(centered).min()) < -tolerance:
            raise ValueError("Kent centered scatter must be positive semidefinite")
    return count, vector_sum.copy(), scatter.copy()


class KentDistribution(SequenceEncodableProbabilityDistribution):
    """Kent (FB5) distribution on ``S^2`` with orientation ``gamma`` (3x3), concentration and ovalness."""

    def __init__(
        self, gamma: np.ndarray, kappa: float, beta: float, name: str | None = None, keys: str | None = None
    ) -> None:
        try:
            g = np.asarray(gamma, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("KentDistribution gamma must be numeric.") from exc
        if g.shape != (3, 3):
            raise ValueError("KentDistribution gamma must be a 3x3 orthonormal matrix (columns g1, g2, g3).")
        if np.any(~np.isfinite(g)):
            raise ValueError("KentDistribution gamma must be finite.")
        if not np.allclose(g.T @ g, np.eye(3), rtol=0.0, atol=1.0e-8):
            raise ValueError("KentDistribution gamma must be orthonormal.")
        if not np.isclose(np.linalg.det(g), 1.0, rtol=0.0, atol=1.0e-8):
            raise ValueError("KentDistribution gamma must be right-handed.")
        if (
            isinstance(kappa, (bool, np.bool_))
            or np.ndim(kappa) != 0
            or isinstance(beta, (bool, np.bool_))
            or np.ndim(beta) != 0
        ):
            raise TypeError("KentDistribution kappa and beta must be real scalars.")
        try:
            checked_kappa = float(kappa)
            checked_beta = float(beta)
        except (TypeError, ValueError) as exc:
            raise TypeError("KentDistribution kappa and beta must be real scalars.") from exc
        if checked_kappa <= 0.0 or not np.isfinite(checked_kappa):
            raise ValueError("KentDistribution requires finite kappa > 0.")
        if not np.isfinite(checked_beta):
            # NaN beta satisfies neither `beta < 0.0` nor `2.0 * beta >= kappa` below (every comparison
            # against NaN is False), so it would otherwise pass straight through as if valid -- and then
            # KentSampler._batch's rejection loop computes an all-NaN, therefore always-False accept
            # mask every iteration and spins forever, since beta never changes between iterations.
            raise ValueError("KentDistribution requires beta to be finite.")
        if checked_beta < 0.0 or 2.0 * checked_beta >= checked_kappa:
            raise ValueError("KentDistribution requires 0 <= 2*beta < kappa.")
        self.gamma = g.copy()
        self.gamma.setflags(write=False)
        self.kappa = checked_kappa
        self.beta = checked_beta
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
        v = _unit_vector(x, 3, "Kent observation")
        p = v @ self.gamma  # (g1.x, g2.x, g3.x)
        return -self._log_c + self.kappa * p[0] + self.beta * (p[1] * p[1] - p[2] * p[2])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded ``(n, 3)`` array of unit vectors."""
        p = _unit_batch(x, 3, "Kent observations") @ self.gamma
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
        checked = _unit_batch(x, 3, "Kent backend observations")
        p = engine.matmul(engine.asarray(checked), engine.asarray(owned_backend_parameter(self.gamma)))
        p1, p2 = p[:, 1], p[:, 2]
        return -self._log_c + self.kappa * p[:, 0] + self.beta * (p1 * p1 - p2 * p2)

    def sampler(self, seed: int | None = None) -> "KentSampler":
        """Return an exact rejection sampler for this Kent distribution."""
        return KentSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "KentEstimator":
        """Return Kent's moment/ML estimator for orientation, concentration, and ovalness."""
        if pseudo_count is not None:
            raise ValueError("Kent pseudo-count regularization is not implemented")
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
        proposed = 0
        for _ in range(10_000):
            if filled >= n:
                break
            m = (n - filled) * 2 + 8  # oversample to amortize rejection
            proposed += m
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
        if filled < n:
            raise KentSamplingError(
                filled,
                proposed,
                self.dist.kappa,
                self.dist.beta,
            )
        return out

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one unit vector or ``size`` iid unit vectors."""
        if size is None:
            return self._batch(1)[0]
        return list(self._batch(_validated_sample_size(size)))


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
        v = _unit_vector(x, 3, "Kent observation")
        checked_weight = _validated_weight(weight)
        self.count += checked_weight
        self.sum_x += checked_weight * v
        self.sum_xx += checked_weight * np.outer(v, v)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize moment statistics from one weighted vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update moment statistics from encoded unit vectors."""
        v = _unit_batch(x, 3, "Kent observations")
        w = _validated_weights(weights, len(v))
        self.count += float(w.sum())
        self.sum_x += v.T @ w
        self.sum_xx += (v * w[:, None]).T @ v

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize moment statistics from encoded unit vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray, np.ndarray]) -> "KentAccumulator":
        """Merge weighted count, vector sum, and scatter matrix statistics."""
        count, vector_sum, scatter = _validated_kent_statistics(suff_stat)
        combined = (
            self.count + count,
            self.sum_x + vector_sum,
            self.sum_xx + scatter,
        )
        self.count, self.sum_x, self.sum_xx = _validated_kent_statistics(combined)
        return self

    def value(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Return weighted count, vector sum, and scatter matrix."""
        return self.count, self.sum_x.copy(), self.sum_xx.copy()

    def from_value(self, x: tuple[float, np.ndarray, np.ndarray]) -> "KentAccumulator":
        """Restore weighted count, vector sum, and scatter matrix."""
        self.count, self.sum_x, self.sum_xx = _validated_kent_statistics(x)
        return self

    def scale(self, c: float) -> "KentAccumulator":
        """Scale linear Kent sufficient statistics."""
        checked_scale = _validated_weight(c)
        self.count *= checked_scale
        self.sum_x *= checked_scale
        self.sum_xx *= checked_scale
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

        count, sum_x, sum_xx = _validated_kent_statistics(suff_stat)
        if count == 0.0:
            raise KentFitError("Kent fitting requires positive observation weight")
        xbar = sum_x / count
        scatter = sum_xx / count
        r1 = float(np.linalg.norm(xbar))
        if r1 <= _MOMENT_ATOL:
            raise KentFitError("Kent mean direction is non-identifiable at zero resultant")
        if r1 >= 1.0 - _MOMENT_ATOL:
            raise KentFitError("Kent resultant is on a boundary with no finite concentration fit")
        g1 = xbar / r1

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
            beta = 0.5 * kappa * float(expit(theta[1]))
            try:
                value = _log_kent_norm(kappa, beta) - kappa * r1 - beta * r2
            except (ValueError, OverflowError):
                return np.inf
            return value if np.isfinite(value) else np.inf

        # initialize from the large-concentration Kent moment approximation
        k0 = max(1.0 / max(2.0 - 2.0 * r1, 1e-3), 1.0)
        res = minimize(
            neg_ll,
            np.array([math.log(k0), 0.0]),
            method="Nelder-Mead",
            bounds=((-20.0, math.log(1.0e4)), (-30.0, 30.0)),
            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
        )
        if (
            not res.success
            or np.asarray(res.x).shape != (2,)
            or np.any(~np.isfinite(res.x))
            or not np.isfinite(res.fun)
        ):
            raise KentFitError("Kent optimizer failed: %s" % res.message)
        kappa = math.exp(res.x[0])
        beta = 0.5 * kappa * float(expit(res.x[1]))
        result = KentDistribution(
            gamma,
            kappa,
            beta,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "Nelder-Mead",
            "iterations": int(res.nit),
            "objective": float(res.fun),
            "resultant_length": r1,
            "repairs": (),
        }
        return result


class KentDataEncoder(DataSequenceEncoder):
    """Validate and encode unit vectors as an ``(n, 3)`` float array."""

    def __str__(self) -> str:
        return "KentDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KentDataEncoder)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Validate and encode observations as an ``(n, 3)`` array."""
        return _unit_batch(x, 3, "Kent observations")

    def row_count(self, x: np.ndarray) -> int:
        """Return the encoded row count after sphere validation."""
        return len(_unit_batch(x, 3, "Kent observations"))
