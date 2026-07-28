"""Bingham distribution on the 2-sphere (antipodally symmetric axial data).

The Bingham distribution models *axes* (unit vectors identified with their antipodes, ``x ~ -x``) on
``S^2``. Its density is

    f(x) = c(Z)^{-1} exp( sum_i z_i * (m_i . x)^2 ),

where ``M = [m_1, m_2, m_3]`` is a ``3 x 3`` orthonormal orientation and ``z = (z_1, z_2, z_3)`` are the
concentrations (defined only up to a common additive shift, since ``sum_i (m_i . x)^2 = 1``; by convention
the largest is taken to be ``0``). It is the antipodally symmetric analogue of the von Mises-Fisher /
Kent laws and the standard model for undirected orientation data (crystallography, geology, principal
axes). Equal concentrations give the uniform distribution on the sphere; a large gap concentrates mass on
a great circle (girdle) or a pair of poles (bipolar).

Normalizer (reduced to a stable 1-D integral, verified to <1e-7 against arbitrary-precision integration
over ``S^2``):

    c(Z) = 2 pi integral_0^pi exp(z_3 cos^2 t + (z_1+z_2)/2 sin^2 t) I_0((z_1-z_2)/2 sin^2 t) sin t dt.

Sampling is by exact angular-central-Gaussian-envelope rejection (Kent, Ganeiber & Mardia 2013); the
orientation and concentrations are fit by maximum likelihood (the orientation is the eigenbasis of the
scatter matrix; the concentrations solve the concave moment-matching problem ``d log c / d z_i =
E[(m_i . x)^2]``).

Reference: Bingham, "An antipodally symmetric distribution on the sphere", *Annals of Statistics* 2
(1974); Kent, Ganeiber & Mardia, "A new method to simulate the Bingham and related distributions",
(2013); Mardia & Jupp, *Directional Statistics* (2000).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.integrate import quad
from scipy.special import i0e

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

_MOMENT_ATOL = 1.0e-8


class BinghamFitError(RuntimeError):
    """Raised when Bingham moments have no certified finite fit."""


class BinghamSamplingError(RuntimeError):
    """Raised when Bingham rejection sampling exhausts its proposal budget."""

    def __init__(self, accepted: int, proposed: int, concentrations: np.ndarray):
        self.accepted = accepted
        self.proposed = proposed
        self.concentrations = concentrations.copy()
        super().__init__(
            "Bingham rejection sampler accepted %d of %d proposals for z=%r"
            % (accepted, proposed, concentrations.tolist())
        )


def _bingham_norm(z: np.ndarray) -> float:
    """Return ``c(Z)`` through a scaled-Bessel one-dimensional integral."""
    z1, z2, z3 = float(z[0]), float(z[1]), float(z[2])

    def integrand(u: float) -> float:
        sin2 = max(0.0, 1.0 - u * u)
        argument = 0.5 * (z1 - z2) * sin2
        exponent = z3 * u * u + 0.5 * (z1 + z2) * sin2 + abs(argument)
        return math.exp(exponent) * float(i0e(argument))

    val, error = quad(
        integrand,
        -1.0,
        1.0,
        epsabs=1.0e-11,
        epsrel=1.0e-10,
        limit=400,
    )
    result = 2.0 * math.pi * val
    if not np.isfinite(result) or result <= 0.0 or not np.isfinite(error) or error > max(1.0e-9, 1.0e-7 * abs(val)):
        raise ValueError("Bingham normalizer integration failed its finite error contract")
    return result


def _validated_bingham_statistics(value: Any) -> tuple[float, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Bingham sufficient statistics must be a two-item tuple")
    if isinstance(value[0], (bool, np.bool_)) or np.ndim(value[0]) != 0:
        raise TypeError("Bingham count must be a real scalar")
    try:
        count = float(value[0])
        scatter = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bingham sufficient statistics must be numeric") from exc
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("Bingham count must be finite and non-negative")
    if scatter.shape != (3, 3) or np.any(~np.isfinite(scatter)):
        raise ValueError("Bingham scatter must be a finite 3x3 matrix")
    if not np.array_equal(scatter, scatter.T):
        raise ValueError("Bingham scatter must be exactly symmetric")
    if count == 0.0:
        if np.any(scatter != 0.0):
            raise ValueError("empty Bingham statistics must have zero scatter")
    else:
        tolerance = _MOMENT_ATOL * max(1.0, count)
        if abs(float(np.trace(scatter)) - count) > tolerance:
            raise ValueError("Bingham scatter trace must equal its observation weight")
        if float(np.linalg.eigvalsh(scatter).min()) < -tolerance:
            raise ValueError("Bingham scatter must be positive semidefinite")
    return count, scatter.copy()


class BinghamDistribution(SequenceEncodableProbabilityDistribution):
    """Bingham distribution on ``S^2`` with orientation ``m`` (3x3) and concentrations ``z`` (length 3)."""

    def __init__(self, m: np.ndarray, z: Sequence[float], name: str | None = None, keys: str | None = None) -> None:
        try:
            mm = np.asarray(m, dtype=np.float64)
            zz = np.asarray(z, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Bingham parameters must be numeric") from exc
        if mm.shape != (3, 3):
            raise ValueError("BinghamDistribution m must be a 3x3 orthonormal matrix (columns m1, m2, m3).")
        if zz.shape != (3,):
            raise ValueError("BinghamDistribution z must have exact shape (3,).")
        if np.any(~np.isfinite(mm)) or np.any(~np.isfinite(zz)):
            raise ValueError("BinghamDistribution parameters must be finite.")
        if not np.allclose(mm.T @ mm, np.eye(3), atol=1e-6):
            raise ValueError("BinghamDistribution m must be orthonormal.")
        self.m = mm.copy()
        self.z = (zz - zz.max()).copy()
        self.m.setflags(write=False)
        self.z.setflags(write=False)
        self.name = name
        self.keys = keys
        self._log_c = math.log(_bingham_norm(self.z))
        if not np.isfinite(self._log_c):
            raise ValueError("Bingham normalizer must be finite.")

    def __str__(self) -> str:
        return "BinghamDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.m.tolist()),
            repr(self.z.tolist()),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Return the Bingham density at one unit 3-vector."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the log-density at a unit 3-vector ``x`` (the same value at ``x`` and ``-x``)."""
        p = _unit_vector(x, 3, "Bingham observation") @ self.m
        return -self._log_c + float(np.dot(self.z, p * p))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded ``(n, 3)`` array of unit vectors."""
        p = _unit_batch(x, 3, "Bingham observations") @ self.m
        return -self._log_c + (p * p) @ self.z

    # --- compute-engine backend (numpy + torch/GPU), SCORING only: the normalizer is a host scalar
    # (Kummer / Bingham constants via scipy), the data math is engine matmul + quadratics. The scatter
    # accumulator stays host-side, so torch accelerates mixture E-step scoring with a bit-correct M-step. ---
    @classmethod
    def compute_capabilities(cls):
        """Declare NumPy/Torch scoring capabilities for Bingham log-density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for ``(N, 3)`` unit vectors."""
        checked = _unit_batch(x, 3, "Bingham backend observations")
        p = engine.matmul(engine.asarray(checked), engine.asarray(owned_backend_parameter(self.m)))
        return -self._log_c + engine.matmul(p * p, engine.asarray(self.z))

    def sampler(self, seed: int | None = None) -> "BinghamSampler":
        """Return an exact rejection sampler for this Bingham distribution."""
        return BinghamSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "BinghamEstimator":
        """Return a maximum-likelihood estimator for Bingham orientation and concentration."""
        if pseudo_count is not None:
            raise ValueError("Bingham pseudo-count regularization is not implemented")
        return BinghamEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "BinghamDataEncoder":
        """Return the unit-vector encoder used by vectorized methods."""
        return BinghamDataEncoder()


class BinghamSampler(DistributionSampler):
    """Sample by angular-central-Gaussian-envelope rejection (Kent, Ganeiber & Mardia 2013)."""

    def __init__(self, dist: BinghamDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _batch(self, n: int) -> np.ndarray:
        from scipy.optimize import brentq

        m = self.dist.m
        a = self.dist.z.max() - self.dist.z  # eigenvalues of A = -shift(Z) >= 0, one is 0
        big_a = (m * a) @ m.T
        # optimal envelope tuning b solves sum_i 1/(b + 2 a_i) = 1
        b = brentq(lambda bb: float(np.sum(1.0 / (bb + 2.0 * a))) - 1.0, 1e-9, 200.0)
        omega = np.eye(3) + 2.0 * big_a / b
        w, v = np.linalg.eigh(omega)
        omega_inv_sqrt = (v / np.sqrt(w)) @ v.T
        log_bound = -(3.0 - b) / 2.0 + 1.5 * math.log(3.0 / b)
        out = np.empty((n, 3))
        filled = 0
        proposed = 0
        for _ in range(10_000):
            if filled >= n:
                break
            k = (n - filled) * 2 + 8
            proposed += k
            z = self.rng.standard_normal((k, 3))
            y = z @ omega_inv_sqrt.T
            y /= np.linalg.norm(y, axis=1, keepdims=True)  # ACG(Omega^{-1}) draw
            t = np.einsum("ni,ij,nj->n", y, big_a, y)
            log_acc = -t + 1.5 * np.log1p(2.0 * t / b) - log_bound
            acc = np.log(self.rng.uniform(size=k)) < log_acc
            ya = y[acc]
            take = min(len(ya), n - filled)
            out[filled : filled + take] = ya[:take]
            filled += take
        if filled < n:
            raise BinghamSamplingError(filled, proposed, self.dist.z)
        return out

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one axial unit vector or ``size`` iid vectors."""
        if size is None:
            return self._batch(1)[0]
        return list(self._batch(_validated_sample_size(size)))


class BinghamAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate ``(count, sum_xx (3,3))`` -- the scatter matrix is sufficient for the Bingham fit."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum_xx = np.zeros((3, 3))
        self.name = name
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: BinghamDistribution | None) -> None:
        """Update scatter statistics from one weighted unit vector."""
        v = _unit_vector(x, 3, "Bingham observation")
        checked_weight = _validated_weight(weight)
        self.count += checked_weight
        self.sum_xx += checked_weight * np.outer(v, v)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize scatter statistics from one weighted unit vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update scatter statistics from encoded unit vectors."""
        v = _unit_batch(x, 3, "Bingham observations")
        w = _validated_weights(weights, len(v))
        self.count += float(w.sum())
        self.sum_xx += (v * w[:, None]).T @ v

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize scatter statistics from encoded unit vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "BinghamAccumulator":
        """Merge weighted count and scatter statistics."""
        count, scatter = _validated_bingham_statistics(suff_stat)
        combined = (self.count + count, self.sum_xx + scatter)
        self.count, self.sum_xx = _validated_bingham_statistics(combined)
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return weighted count and scatter matrix."""
        return self.count, self.sum_xx.copy()

    def from_value(self, x: tuple[float, np.ndarray]) -> "BinghamAccumulator":
        """Restore weighted count and scatter matrix."""
        self.count, self.sum_xx = _validated_bingham_statistics(x)
        return self

    def scale(self, c: float) -> "BinghamAccumulator":
        """Scale linear Bingham sufficient statistics."""
        checked_scale = _validated_weight(c)
        self.count *= checked_scale
        self.sum_xx *= checked_scale
        return self

    def acc_to_encoder(self) -> "BinghamDataEncoder":
        """Return the encoder compatible with Bingham scatter statistics."""
        return BinghamDataEncoder()


class BinghamAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for BinghamAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> BinghamAccumulator:
        """Create an empty Bingham accumulator."""
        return BinghamAccumulator(name=self.name, keys=self.keys)


class BinghamEstimator(ParameterEstimator):
    """Maximum-likelihood Bingham fit: scatter eigenbasis + concave concentration moment-matching."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> BinghamDistribution:
        """Estimate orientation and concentrations from scatter statistics."""
        from scipy.optimize import minimize

        count, sum_xx = _validated_bingham_statistics(suff_stat)
        if count == 0.0:
            raise BinghamFitError("Bingham fitting requires positive observation weight")
        scatter = sum_xx / count
        omega, vecs = np.linalg.eigh(scatter)  # ascending eigenvalues; columns are the axes
        m = vecs  # orientation: column i is the axis with mean square projection omega[i]

        # concave ML moment-matching: maximize z.omega - log c(z) with z3 = 0 (largest axis fixed)
        def neg_g(z12: np.ndarray) -> float:
            z = np.array([z12[0], z12[1], 0.0])
            return math.log(_bingham_norm(z)) - float(np.dot(z, omega))

        res = minimize(
            neg_g,
            np.array([-1.0, -0.5]),
            method="Nelder-Mead",
            bounds=((-1.0e5, 0.0), (-1.0e5, 0.0)),
            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
        )
        if (
            not res.success
            or np.asarray(res.x).shape != (2,)
            or np.any(~np.isfinite(res.x))
            or not np.isfinite(res.fun)
        ):
            raise BinghamFitError("Bingham concentration optimizer failed: %s" % res.message)
        z = np.array([res.x[0], res.x[1], 0.0])
        result = BinghamDistribution(m, z, name=self.name, keys=self.keys)
        result.fit_metadata = {
            "converged": True,
            "solver": "Nelder-Mead",
            "iterations": int(res.nit),
            "objective": float(res.fun),
            "identifiable_orientation": bool(np.min(np.diff(np.sort(omega))) > _MOMENT_ATOL),
            "repairs": (),
        }
        return result

    def accumulator_factory(self) -> BinghamAccumulatorFactory:
        """Return a factory for Bingham sufficient-statistic accumulators."""
        return BinghamAccumulatorFactory(name=self.name, keys=self.keys)


class BinghamDataEncoder(DataSequenceEncoder):
    """Validate and encode axial data as an ``(n, 3)`` unit-vector array."""

    def __str__(self) -> str:
        return "BinghamDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BinghamDataEncoder)

    def seq_encode(self, x: Sequence[Any]) -> np.ndarray:
        """Validate and encode axial observations as an ``(n, 3)`` array."""
        return _unit_batch(x, 3, "Bingham observations")

    def row_count(self, x: np.ndarray) -> int:
        """Return the encoded row count after sphere validation."""
        return len(_unit_batch(x, 3, "Bingham observations"))
