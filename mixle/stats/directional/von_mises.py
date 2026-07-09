"""Von Mises distributions for circular angular data.

Observations are angles in radians. A Von Mises distribution with mean direction ``mu`` and
concentration ``kappa >= 0`` has log-density

        log(f(theta; mu, kappa)) = kappa * cos(theta - mu) - log(2*pi*I_0(kappa)),

the circular analogue of a Gaussian (kappa = 0 is uniform on the circle; large kappa concentrates near
mu). This is the one-dimensional companion to :class:`~mixle.stats.directional.von_mises_fisher` (the von
Mises-Fisher distribution on a sphere).

It is a two-parameter exponential family with sufficient statistics ``(cos theta, sin theta)``:

        log(f) = eta1*cos(theta) + eta2*sin(theta) + log_const,
                eta1 = kappa*cos(mu),  eta2 = kappa*sin(mu),  log_const = -log(2*pi*I_0(kappa)).

The natural parameters and normalizer are precomputed (the Bessel term ``I_0`` lives only in the scalar
``log_const``), so the per-row score is linear in the encoded ``cos``/``sin`` fields and lowers cleanly
to generated NumPy, Torch, and Numba kernels.


Reference: Mardia & Jupp, *Directional Statistics* (Wiley, 2000).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import i0e, ive

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_LOG_2PI = math.log(2.0 * math.pi)
_MAX_KAPPA = 1.0e8


def _log_i0(kappa: float) -> float:
    """Return log(I_0(kappa)) stably via the exponentially-scaled Bessel function i0e."""
    return float(math.log(i0e(kappa)) + kappa)


def _bessel_ratio(kappa: float) -> float:
    """Return A(kappa) = I_1(kappa) / I_0(kappa), the mean resultant length of von Mises(mu, kappa).

    The exponential scaling in ``ive`` cancels in the ratio, keeping it stable for large kappa.
    """
    if kappa <= 0.0:
        return 0.0
    return float(ive(1.0, kappa) / ive(0.0, kappa))


def _solve_kappa(r: float) -> float:
    """Invert A(kappa) = r for the concentration (Best & Fisher initializer + Newton refinement)."""
    if r <= 0.0:
        return 0.0
    if r >= 1.0 - 1.0e-12:
        return _MAX_KAPPA
    if r < 0.53:
        kappa = 2.0 * r + r**3 + 5.0 * r**5 / 6.0
    elif r < 0.85:
        kappa = -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    else:
        kappa = 1.0 / (r**3 - 4.0 * r**2 + 3.0 * r)
    # Newton on A(kappa) = r, with A'(kappa) = 1 - A/kappa - A^2.
    for _ in range(5):
        if kappa <= 0.0 or kappa >= _MAX_KAPPA:
            break
        a = _bessel_ratio(kappa)
        deriv = 1.0 - a / kappa - a * a
        if deriv <= 1.0e-12:
            break
        kappa = kappa - (a - r) / deriv
    return float(min(max(kappa, 0.0), _MAX_KAPPA))


class VonMisesDistribution(SequenceEncodableProbabilityDistribution):
    """Von Mises distribution on the circle with mean direction mu and concentration kappa >= 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated von Mises kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for von Mises distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="von_mises",
            distribution_type=cls,
            parameters=(
                ParameterSpec("eta1"),
                ParameterSpec("eta2"),
                ParameterSpec("log_const", constraint="real", differentiable=False),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum_cos"), StatisticSpec("sum_sin")),
            support="real",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row (count, cos, sin) sufficient statistics in accumulator order."""
        cos_t = engine.asarray(x[0])
        sin_t = engine.asarray(x[1])
        return cos_t * 0.0 + engine.asarray(1.0), cos_t, sin_t

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return von Mises sufficient statistics ``T(x) = (cos x, sin x)``."""
        return engine.asarray(x[0]), engine.asarray(x[1])

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return von Mises natural parameters ``eta = (kappa cos mu, kappa sin mu)``."""
        return params["eta1"], params["eta2"]

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return von Mises log partition ``A = log(2 pi I_0(kappa)) = -log_const``."""
        return -params["log_const"]

    @staticmethod
    def exp_family_from_natural(eta: Any) -> "VonMisesDistribution":
        """Return the von Mises with natural parameters ``eta = (kappa cos mu, kappa sin mu)``."""
        eta1 = float(eta[0])
        eta2 = float(eta[1])
        kappa = math.hypot(eta1, eta2)
        mu = math.atan2(eta2, eta1)
        return VonMisesDistribution(mu, kappa)

    @staticmethod
    def backend_log_density_from_params(
        cos_t: Any, sin_t: Any, eta1: Any, eta2: Any, log_const: Any, engine: Any
    ) -> Any:
        """Engine-neutral von Mises log-density from natural parameters (linear in cos/sin)."""
        return eta1 * cos_t + eta2 * sin_t + log_const

    def __init__(self, mu: float, kappa: float, name: str | None = None, keys: str | None = None) -> None:
        """VonMisesDistribution for mean direction mu and concentration kappa.

        Args:
            mu (float): Mean direction in radians.
            kappa (float): Non-negative concentration. kappa = 0 is uniform on the circle.
            name (Optional[str]): Assign a name to VonMisesDistribution instance.
            keys (Optional[str]): Assign keys for merging sufficient statistics.

        Attributes:
            mu (float): Mean direction (wrapped to (-pi, pi]).
            kappa (float): Concentration parameter.
            eta1, eta2 (float): Natural parameters kappa*cos(mu), kappa*sin(mu).
            log_const (float): Cached -log(2*pi*I_0(kappa)).

        """
        if kappa < 0.0 or not np.isfinite(kappa):
            raise ValueError("VonMisesDistribution requires kappa >= 0.")
        if not np.isfinite(mu):
            raise ValueError("VonMisesDistribution requires a finite mu.")
        self.mu = float(math.atan2(math.sin(mu), math.cos(mu)))  # wrap to (-pi, pi]
        self.kappa = float(kappa)
        self.eta1 = self.kappa * math.cos(self.mu)
        self.eta2 = self.kappa * math.sin(self.mu)
        if self.kappa <= 0.0:
            self.log_const = -_LOG_2PI
        else:
            self.log_const = -_LOG_2PI - _log_i0(self.kappa)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the von Mises distribution."""
        return "VonMisesDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.kappa),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single angle."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single angle (radians)."""
        return self.kappa * math.cos(float(x) - self.mu) + self.log_const

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded (cos, sin) observations."""
        cos_t, sin_t = x
        return self.eta1 * cos_t + self.eta2 * sin_t + self.log_const

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x[0]),
            engine.asarray(x[1]),
            engine.asarray(self.eta1),
            engine.asarray(self.eta2),
            engine.asarray(self.log_const),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["VonMisesDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked natural parameters for a homogeneous mixture kernel."""
        return {
            "eta1": engine.asarray([d.eta1 for d in dists]),
            "eta2": engine.asarray([d.eta2 for d in dists]),
            "log_const": engine.asarray([d.log_const for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of von Mises log densities."""
        cos_t = engine.asarray(x[0])[:, None]
        sin_t = engine.asarray(x[1])[:, None]
        return cls.backend_log_density_from_params(
            cos_t, sin_t, params["eta1"][None, :], params["eta2"][None, :], params["log_const"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked sufficient statistics using engine-resident arrays."""
        cos_t = engine.asarray(x[0])
        sin_t = engine.asarray(x[1])
        ww = engine.asarray(weights)
        return (
            engine.sum(ww, axis=0),
            engine.sum(ww * cos_t[:, None], axis=0),
            engine.sum(ww * sin_t[:, None], axis=0),
        )

    def sampler(self, seed: int | None = None) -> "VonMisesSampler":
        """Return a sampler for drawing angles from this distribution."""
        return VonMisesSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "VonMisesEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return VonMisesEstimator(name=self.name, keys=self.keys)
        return VonMisesEstimator(
            pseudo_count=pseudo_count,
            suff_stat=(math.cos(self.mu) * _bessel_ratio(self.kappa), math.sin(self.mu) * _bessel_ratio(self.kappa)),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "VonMisesDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return VonMisesDataEncoder()


class VonMisesSampler(DistributionSampler):
    """Draw iid angles from a von Mises distribution."""

    def __init__(self, dist: VonMisesDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw ``size`` iid angles in (-pi, pi] (a float when ``size`` is None)."""
        return self.rng.vonmises(self.dist.mu, self.dist.kappa, size=size)


class VonMisesAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and circular moments (sum of cos / sin) for von Mises estimation."""

    def __init__(self, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.keys = keys

    def update(self, x: float, weight: float, estimate: VonMisesDistribution | None) -> None:
        """Accumulate one weighted circular moment contribution."""
        self.count += weight
        self.sum_cos += math.cos(float(x)) * weight
        self.sum_sin += math.sin(float(x)) * weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one angle."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: VonMisesDistribution | None
    ) -> None:
        """Accumulate circular moments from encoded cos/sin values."""
        cos_t, sin_t = x
        self.count += np.sum(weights, dtype=np.float64)
        self.sum_cos += np.dot(cos_t, weights)
        self.sum_sin += np.dot(sin_t, weights)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded angles."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "VonMisesAccumulator":
        """Merge another von Mises sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum_cos += suff_stat[1]
        self.sum_sin += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, cosine sum, and sine sum."""
        return self.count, self.sum_cos, self.sum_sin

    def from_value(self, x: tuple[float, float, float]) -> "VonMisesAccumulator":
        """Replace accumulator contents from circular-moment statistics."""
        self.count, self.sum_cos, self.sum_sin = x
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

    def acc_to_encoder(self) -> "VonMisesDataEncoder":
        """Return the encoder used by this accumulator."""
        return VonMisesDataEncoder()


class VonMisesAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for VonMisesAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> VonMisesAccumulator:
        """Create a fresh von Mises accumulator."""
        return VonMisesAccumulator(keys=self.keys)


class VonMisesEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the von Mises mean direction and concentration.

    The MLE is ``mu = atan2(sum sin, sum cos)`` and ``kappa = A^{-1}(R)`` where ``R`` is the mean
    resultant length and ``A(kappa) = I_1(kappa) / I_0(kappa)``.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> VonMisesAccumulatorFactory:
        """Return an accumulator factory for von Mises circular moments."""
        return VonMisesAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> VonMisesDistribution:
        """Estimate mean direction and concentration from weighted circular moments."""
        count, sum_cos, sum_sin = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            mean_cos0, mean_sin0 = self.suff_stat
            sum_cos += self.pseudo_count * mean_cos0
            sum_sin += self.pseudo_count * mean_sin0
            count += self.pseudo_count

        if count <= 0.0:
            return VonMisesDistribution(0.0, 0.0, name=self.name, keys=self.keys)

        mean_cos = sum_cos / count
        mean_sin = sum_sin / count
        r = math.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
        mu = math.atan2(mean_sin, mean_cos)
        kappa = _solve_kappa(r)
        return VonMisesDistribution(mu, kappa, name=self.name, keys=self.keys)


class VonMisesDataEncoder(DataSequenceEncoder):
    """Encode angle observations as (cos, sin) pairs."""

    def __str__(self) -> str:
        return "VonMisesDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VonMisesDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode angles as cosine and sine arrays."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(~np.isfinite(rv)):
            raise ValueError("VonMisesDistribution requires finite angle observations.")
        return np.cos(rv), np.sin(rv)
