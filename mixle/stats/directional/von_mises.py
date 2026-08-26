"""Von Mises distributions for circular angular data.

Observations are angles in radians. A Von Mises distribution with mean direction ``mu`` and
concentration ``kappa >= 0`` has log-density

        log(f(theta; mu, kappa)) = kappa * cos(theta - mu) - log(2*pi*I_0(kappa)),

the circular analogue of a Gaussian (kappa = 0 is uniform on the circle; large kappa concentrates near
mu). This is the one-dimensional companion to :class:`~mixle.stats.directional.von_mises_fisher` (the von
Mises-Fisher distribution on a sphere).

Wrap-around semantics -- read this before comparing scores across families. Any finite real ``x``
is accepted and interpreted *modulo* ``2*pi``: only ``cos(x)`` and ``sin(x)`` ever enter the
density, the sufficient statistics, and the estimator, so ``x``, ``x + 2*pi`` and ``x + 200*pi``
are indistinguishable observations of the same angle. The density integrates to one over a single
period ``(-pi, pi]``, not over the real line -- integrated over a wider interval it exceeds one.
Two consequences for non-angular data:

* Fitting a real-line column (money, durations, counts) silently wraps it onto the circle; the
  fit is a valid circular density of ``x mod 2*pi`` but says nothing about ``x`` itself.
* The log-density is **not comparable** to real-line families' log-densities. A column spanning
  ``k`` periods overlays its mass onto one period, so the wrapped model can "win" a likelihood
  comparison against any real-line family by roughly ``log(k)`` nats per observation without
  modeling the data at all. Compare von Mises scores only against other circular models, on data
  that is genuinely angular.

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
from scipy.optimize import brentq
from scipy.special import i0e, ive

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.directional._circular import (
    validated_angle,
    validated_angles,
    validated_circular_statistics,
    validated_sample_size,
    validated_trig,
    validated_weight,
    validated_weights,
)

_LOG_2PI = math.log(2.0 * math.pi)


class VonMisesFitError(RuntimeError):
    """Raised when circular moments have no finite von Mises fit."""


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
    """Invert ``I1(kappa)/I0(kappa)`` with an adaptive certified bracket."""
    if not np.isfinite(r) or r < 0.0 or r >= 1.0:
        raise VonMisesFitError("von Mises resultant must lie in [0, 1) for a finite fit")
    if r == 0.0:
        return 0.0
    upper = max(1.0, 1.0 / max(2.0 * (1.0 - r), 1.0e-12))
    while _bessel_ratio(upper) < r:
        upper *= 2.0
        if upper > 1.0e12:
            raise VonMisesFitError("von Mises concentration could not be bracketed")
    try:
        result = brentq(
            lambda value: _bessel_ratio(value) - r,
            0.0,
            upper,
            xtol=1.0e-12,
            rtol=1.0e-12,
            maxiter=200,
        )
    except (RuntimeError, ValueError) as exc:
        raise VonMisesFitError("von Mises concentration solve failed") from exc
    if not np.isfinite(result):
        raise VonMisesFitError("von Mises concentration solve returned a non-finite value")
    return float(result)


class VonMisesDistribution(SequenceEncodableProbabilityDistribution):
    """Von Mises distribution on the circle with mean direction mu and concentration kappa >= 0.

    Observations are angles in radians, interpreted modulo ``2*pi``: ``log_density(x)`` equals
    ``log_density(x + 2*pi*k)`` for every integer ``k``, and the density normalizes over one
    period, not over the real line. Scores are therefore comparable only to other circular
    models -- see the module docstring for why a wrapped fit of real-line data can spuriously
    out-score every real-line family.
    """

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
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        cos_t = engine.asarray(cosine)
        sin_t = engine.asarray(sine)
        return cos_t * 0.0 + engine.asarray(1.0), cos_t, sin_t

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return von Mises sufficient statistics ``T(x) = (cos x, sin x)``."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        return engine.asarray(cosine), engine.asarray(sine)

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
        checked_kappa = validated_angle(kappa, "von Mises concentration")
        checked_mu = validated_angle(mu, "von Mises mean direction")
        if checked_kappa < 0.0:
            raise ValueError("VonMisesDistribution requires kappa >= 0.")
        self.mu = float(math.atan2(math.sin(checked_mu), math.cos(checked_mu)))
        self.kappa = checked_kappa
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
        """Return the log-density at a single angle (radians), wrapping ``x`` modulo ``2*pi``.

        Any finite real is accepted; values outside ``(-pi, pi]`` are folded onto the circle, so
        this is the density of ``x mod 2*pi``, normalized over one period only.
        """
        theta = validated_angle(x)
        return self.kappa * math.cos(theta - self.mu) + self.log_const

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded (cos, sin) observations."""
        cos_t, sin_t = validated_trig(x)
        return self.eta1 * cos_t + self.eta2 * sin_t + self.log_const

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        return self.backend_log_density_from_params(
            engine.asarray(cosine),
            engine.asarray(sine),
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
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        cos_t = engine.asarray(cosine)[:, None]
        sin_t = engine.asarray(sine)[:, None]
        return cls.backend_log_density_from_params(
            cos_t, sin_t, params["eta1"][None, :], params["eta2"][None, :], params["log_const"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked sufficient statistics using engine-resident arrays."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        cos_t = engine.asarray(cosine)
        sin_t = engine.asarray(sine)
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

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw ``size`` iid angles in (-pi, pi] (a float when ``size`` is None)."""
        checked_size = None if size is None else validated_sample_size(size)
        return self.rng.vonmises(
            self.dist.mu,
            self.dist.kappa,
            size=checked_size,
        )


class VonMisesAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and circular moments (sum of cos / sin) for von Mises estimation."""

    def __init__(
        self,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.count = 0.0
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: VonMisesDistribution | None) -> None:
        """Accumulate one weighted circular moment contribution."""
        theta = validated_angle(x)
        checked_weight = validated_weight(weight)
        self.count += checked_weight
        self.sum_cos += math.cos(theta) * checked_weight
        self.sum_sin += math.sin(theta) * checked_weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one angle."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: VonMisesDistribution | None
    ) -> None:
        """Accumulate circular moments from encoded cos/sin values."""
        cos_t, sin_t = validated_trig(x)
        checked_weights = validated_weights(weights, len(cos_t))
        self.count += float(checked_weights.sum())
        self.sum_cos += float(np.dot(cos_t, checked_weights))
        self.sum_sin += float(np.dot(sin_t, checked_weights))

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded angles."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "VonMisesAccumulator":
        """Merge another von Mises sufficient-statistic tuple."""
        count, sum_cos, sum_sin = validated_circular_statistics(
            suff_stat,
            count_index=0,
        )
        combined = (
            self.count + count,
            self.sum_cos + sum_cos,
            self.sum_sin + sum_sin,
        )
        self.count, self.sum_cos, self.sum_sin = validated_circular_statistics(combined, count_index=0)
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, cosine sum, and sine sum."""
        return self.count, self.sum_cos, self.sum_sin

    def from_value(self, x: tuple[float, float, float]) -> "VonMisesAccumulator":
        """Replace accumulator contents from circular-moment statistics."""
        self.count, self.sum_cos, self.sum_sin = validated_circular_statistics(x, count_index=0)
        return self

    def scale(self, c: float) -> "VonMisesAccumulator":
        """Scale linear von Mises sufficient statistics."""
        checked_scale = validated_weight(c)
        self.count *= checked_scale
        self.sum_cos *= checked_scale
        self.sum_sin *= checked_scale
        return self

    def acc_to_encoder(self) -> "VonMisesDataEncoder":
        """Return the encoder used by this accumulator."""
        return VonMisesDataEncoder()


class VonMisesAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for VonMisesAccumulator."""

    def __init__(
        self,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> VonMisesAccumulator:
        """Create a fresh von Mises accumulator."""
        return VonMisesAccumulator(name=self.name, keys=self.keys)


class VonMisesEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the von Mises mean direction and concentration.

    The MLE is ``mu = atan2(sum sin, sum cos)`` and ``kappa = A^{-1}(R)`` where ``R`` is the mean
    resultant length and ``A(kappa) = I_1(kappa) / I_0(kappa)``.

    The data must be angles in radians. Any finite real is accepted and wrapped modulo ``2*pi``
    (only ``cos``/``sin`` enter the statistics), so fitting a non-angular real-line column does not
    fail -- it silently fits the circular law of ``x mod 2*pi``, whose log-likelihood is normalized
    over one period and is not comparable to real-line families (see the module docstring).
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if pseudo_count is None:
            if suff_stat is not None:
                raise ValueError("von Mises prior moments require a pseudo-count")
            self.pseudo_count = None
            self.suff_stat = None
        else:
            self.pseudo_count = validated_weight(pseudo_count)
            if not isinstance(suff_stat, (tuple, list)) or len(suff_stat) != 2:
                raise ValueError("von Mises pseudo-count requires two prior moments")
            mean_cos = validated_angle(
                suff_stat[0],
                "von Mises prior cosine moment",
            )
            mean_sin = validated_angle(
                suff_stat[1],
                "von Mises prior sine moment",
            )
            if math.hypot(mean_cos, mean_sin) > 1.0 + 1.0e-8:
                raise ValueError("von Mises prior resultant cannot exceed one")
            self.suff_stat = (mean_cos, mean_sin)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> VonMisesAccumulatorFactory:
        """Return an accumulator factory for von Mises circular moments."""
        return VonMisesAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> VonMisesDistribution:
        """Estimate mean direction and concentration from weighted circular moments."""
        count, sum_cos, sum_sin = validated_circular_statistics(
            suff_stat,
            count_index=0,
        )
        if self.pseudo_count is not None and self.suff_stat is not None:
            mean_cos0, mean_sin0 = self.suff_stat
            sum_cos += self.pseudo_count * mean_cos0
            sum_sin += self.pseudo_count * mean_sin0
            count += self.pseudo_count

        count, sum_cos, sum_sin = validated_circular_statistics(
            (count, sum_cos, sum_sin),
            count_index=0,
        )
        if count == 0.0:
            raise VonMisesFitError("von Mises fitting requires positive observation weight")

        mean_cos = sum_cos / count
        mean_sin = sum_sin / count
        r = math.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
        mu = math.atan2(mean_sin, mean_cos)
        kappa = _solve_kappa(r)
        result = VonMisesDistribution(
            mu,
            kappa,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "brentq-resultant",
            "identifiable_direction": bool(r > 1.0e-12),
            "resultant_length": r,
            "regularized": self.pseudo_count is not None,
            "repairs": (),
        }
        return result


class VonMisesDataEncoder(DataSequenceEncoder):
    """Encode angle observations as (cos, sin) pairs."""

    def __str__(self) -> str:
        return "VonMisesDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VonMisesDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode angles as cosine and sine arrays."""
        rv = validated_angles(x)
        return np.cos(rv), np.sin(rv)

    def row_count(self, x: tuple[np.ndarray, np.ndarray]) -> int:
        """Return the encoded row count after unit-circle validation."""
        cosine, _ = validated_trig(x)
        return len(cosine)
