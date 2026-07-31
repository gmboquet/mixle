"""Wrapped Cauchy distribution -- a heavy-tailed circular law on angles.

Wrapping a Cauchy around the circle gives the wrapped Cauchy, the circular analogue of the Cauchy and
a heavier-tailed alternative to the von Mises. With mean direction ``mu`` and mean-resultant length
``rho`` in ``[0, 1)``,

    f(theta; mu, rho) = (1 - rho^2) / (2 pi (1 + rho^2 - 2 rho cos(theta - mu))),

uniform on the circle at ``rho = 0`` and increasingly peaked at ``mu`` as ``rho -> 1``. Its first
trigonometric moment is exactly ``rho e^{i mu}``, so the mean direction and concentration are estimated
in closed form from the mean resultant (the circular analogue of the sample mean). It samples exactly:
``rho`` corresponds to wrapping a Cauchy of scale ``gamma = -log rho``.


Reference: Mardia & Jupp, *Directional Statistics* (Wiley, 2000).
"""

import math
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


class WrappedCauchyFitError(RuntimeError):
    """Raised when circular moments have no finite wrapped-Cauchy fit."""


def _wrap(theta: Any) -> Any:
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(theta) + math.pi) % (2.0 * math.pi) - math.pi


class WrappedCauchyDistribution(SequenceEncodableProbabilityDistribution):
    """Wrapped Cauchy distribution with mean direction ``mu`` and concentration ``rho`` in ``[0, 1)``."""

    def __init__(self, mu: float, rho: float, name: str | None = None, keys: str | None = None) -> None:
        checked_mu = validated_angle(mu, "wrapped-Cauchy mean direction")
        checked_rho = validated_angle(rho, "wrapped-Cauchy concentration")
        if not (0.0 <= checked_rho < 1.0):
            raise ValueError("WrappedCauchyDistribution requires concentration rho in [0, 1).")
        self.mu = float(math.atan2(math.sin(checked_mu), math.cos(checked_mu)))
        self.rho = checked_rho
        self.name = name
        self.keys = keys
        self._log_num = math.log1p(-self.rho) + math.log1p(self.rho) - _LOG_2PI
        # parameter-side trig, cached as attributes: the compute declaration exposes these (not mu) so the
        # generated scorer sees scalar parameters and the (cos, sin) encoding needs no engine trig
        self.cos_mu = math.cos(self.mu)
        self.sin_mu = math.sin(self.mu)

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep ``_log_num`` tied to ``rho`` and ``cos_mu``/``sin_mu`` tied to ``mu``.

        All three are computed once in ``__init__`` and read by ``log_density``, so a later
        assignment used to leave them stale and the scorer kept reporting the *previous*
        parameters' density with no error at all (MXR-080-1192).

        Recompute rather than validate: callers legitimately install out-of-domain or non-finite
        parameters -- deserialized legacy states and NaN-propagation checks both do -- so a value
        outside the domain yields a NaN constant that propagates honestly instead of rejecting a
        state the library is expected to be able to hold.
        """
        object.__setattr__(self, name, value)
        if name not in ("mu", "rho"):
            return
        try:
            object.__setattr__(self, "_log_num", math.log1p(-self.rho) + math.log1p(self.rho) - _LOG_2PI)
            object.__setattr__(self, "cos_mu", math.cos(self.mu))
            object.__setattr__(self, "sin_mu", math.sin(self.mu))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, AttributeError, FloatingPointError):
            # AttributeError covers __init__, where the first parameter is assigned before the rest.
            object.__setattr__(self, "_log_num", float("nan"))
            object.__setattr__(self, "cos_mu", float("nan"))
            object.__setattr__(self, "sin_mu", float("nan"))

    def __str__(self) -> str:
        return "WrappedCauchyDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.rho),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single angle (radians)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single angle (radians)."""
        theta = validated_angle(x)
        deviation = theta - self.mu
        denominator = (1.0 - self.rho) ** 2 + 4.0 * self.rho * math.sin(0.5 * deviation) ** 2
        return self._log_num - math.log(denominator)

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded ``(cos, sin)`` observations."""
        cos_t, sin_t = validated_trig(x)
        cos_dev = np.clip(
            cos_t * self.cos_mu + sin_t * self.sin_mu,
            -1.0,
            1.0,
        )
        denominator = (1.0 - self.rho) ** 2 + 2.0 * self.rho * (1.0 - cos_dev)
        return self._log_num - np.log(denominator)

    # --- compute-engine backend (numpy + torch/GPU). The encoder pre-computes (cos, sin) and the
    # parameter-side trig is host-scalar math, so scoring needs no engine trig at all. ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated wrapped-Cauchy kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for wrapped Cauchy distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="wrapped_cauchy",
            distribution_type=cls,
            parameters=(ParameterSpec("cos_mu"), ParameterSpec("sin_mu"), ParameterSpec("rho")),
            statistics=(StatisticSpec("sum_cos"), StatisticSpec("sum_sin"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row circular moments in accumulator order ``(sum_cos, sum_sin, count)``."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        cos_t = engine.asarray(cosine)
        sin_t = engine.asarray(sine)
        return cos_t, sin_t, cos_t * 0.0 + engine.asarray(1.0)

    @staticmethod
    def backend_log_density_from_params(cos_t: Any, sin_t: Any, cos_mu: Any, sin_mu: Any, rho: Any, engine: Any) -> Any:
        """Engine-neutral wrapped-Cauchy log-density from pre-computed observation/parameter trig."""
        cos_dev = cos_t * cos_mu + sin_t * sin_mu
        one_minus_cosine = engine.maximum(
            1.0 - cos_dev,
            engine.asarray(0.0),
        )
        log_num = engine.log(1.0 - rho) + engine.log(1.0 + rho) - engine.log(engine.asarray(2.0 * math.pi))
        denominator = (1.0 - rho) * (1.0 - rho) + 2.0 * rho * one_minus_cosine
        return log_num - engine.log(denominator)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded ``(cos, sin)`` data."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        return self.backend_log_density_from_params(
            engine.asarray(cosine),
            engine.asarray(sine),
            engine.asarray(self.cos_mu),
            engine.asarray(self.sin_mu),
            engine.asarray(self.rho),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["WrappedCauchyDistribution"], engine: Any) -> dict[str, Any]:
        """Stacked wrapped-Cauchy parameters (trig computed host-side) for a homogeneous mixture kernel."""
        return {
            "cos_mu": engine.asarray([math.cos(d.mu) for d in dists]),
            "sin_mu": engine.asarray([math.sin(d.mu) for d in dists]),
            "rho": engine.asarray([d.rho for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of wrapped-Cauchy log densities."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        cos_t = engine.asarray(cosine)
        sin_t = engine.asarray(sine)
        return cls.backend_log_density_from_params(
            cos_t[:, None],
            sin_t[:, None],
            params["cos_mu"][None, :],
            params["sin_mu"][None, :],
            params["rho"][None, :],
            engine,
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Stacked circular moments ``(sum_cos, sum_sin, count)`` using engine-resident arrays."""
        from mixle.engines.symbolic_engine import is_symbolic_payload

        if is_symbolic_payload(x[0]) or is_symbolic_payload(x[1]):
            cosine, sine = x
        else:
            cosine, sine = validated_trig(x)
        cos_t = engine.asarray(cosine)
        sin_t = engine.asarray(sine)
        ww = engine.asarray(weights)
        return (
            engine.sum(ww * cos_t[:, None], axis=0),
            engine.sum(ww * sin_t[:, None], axis=0),
            engine.sum(ww, axis=0),
        )

    def sampler(self, seed: int | None = None) -> "WrappedCauchySampler":
        """Return a sampler for drawing angles from this distribution."""
        return WrappedCauchySampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WrappedCauchyEstimator":
        """Return a closed-form (mean-resultant) estimator for ``mu`` and ``rho``."""
        if pseudo_count is not None:
            raise ValueError("wrapped-Cauchy pseudo-count regularization is not implemented")
        return WrappedCauchyEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WrappedCauchyDataEncoder":
        """Return the data encoder used by this distribution (cos/sin of the angle)."""
        return WrappedCauchyDataEncoder()


class WrappedCauchySampler(DistributionSampler):
    """Draw angles by wrapping a Cauchy of scale ``-log rho`` around the circle."""

    def __init__(self, dist: WrappedCauchyDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one angle or an array of iid angles."""
        d = self.dist
        n = 1 if size is None else validated_sample_size(size)
        if d.rho == 0.0:  # uniform on the circle
            theta = self.rng.uniform(-math.pi, math.pi, size=n)
        else:
            gamma = -math.log(d.rho)  # Cauchy scale of the un-wrapped variable
            c = d.mu + gamma * np.tan(math.pi * (self.rng.uniform(size=n) - 0.5))
            theta = _wrap(c)
        return float(theta[0]) if size is None else theta


class WrappedCauchyAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted circular resultant ``(sum cos, sum sin, count)``."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: WrappedCauchyDistribution | None) -> None:
        """Accumulate one weighted circular resultant contribution."""
        theta = validated_angle(x)
        checked_weight = validated_weight(weight)
        self.sum_cos += checked_weight * math.cos(theta)
        self.sum_sin += checked_weight * math.sin(theta)
        self.count += checked_weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one angle."""
        self.update(x, weight, None)

    def seq_update(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: Any) -> None:
        """Accumulate circular-resultant statistics from encoded cos/sin values."""
        cos_t, sin_t = validated_trig(x)
        w = validated_weights(weights, len(cos_t))
        self.sum_cos += float(np.dot(cos_t, w))
        self.sum_sin += float(np.dot(sin_t, w))
        self.count += float(w.sum())

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded angles."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "WrappedCauchyAccumulator":
        """Merge another wrapped-Cauchy sufficient-statistic tuple."""
        count, sum_cos, sum_sin = validated_circular_statistics(
            suff_stat,
            count_index=2,
        )
        combined = (
            self.sum_cos + sum_cos,
            self.sum_sin + sum_sin,
            self.count + count,
        )
        self.count, self.sum_cos, self.sum_sin = validated_circular_statistics(combined, count_index=2)
        return self

    def value(self) -> tuple[float, float, float]:
        """Return cosine sum, sine sum, and total weight."""
        return self.sum_cos, self.sum_sin, self.count

    def from_value(self, x: tuple[float, float, float]) -> "WrappedCauchyAccumulator":
        """Replace accumulator contents from circular-resultant statistics."""
        self.count, self.sum_cos, self.sum_sin = validated_circular_statistics(x, count_index=2)
        return self

    def scale(self, c: float) -> "WrappedCauchyAccumulator":
        """Scale linear wrapped-Cauchy sufficient statistics."""
        checked_scale = validated_weight(c)
        self.sum_cos *= checked_scale
        self.sum_sin *= checked_scale
        self.count *= checked_scale
        return self

    def acc_to_encoder(self) -> "WrappedCauchyDataEncoder":
        """Return the encoder used by this accumulator."""
        return WrappedCauchyDataEncoder()


class WrappedCauchyAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WrappedCauchyAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> WrappedCauchyAccumulator:
        """Create a fresh wrapped-Cauchy accumulator."""
        return WrappedCauchyAccumulator(name=self.name, keys=self.keys)


class WrappedCauchyEstimator(ParameterEstimator):
    """Estimate ``mu`` and ``rho`` from the mean resultant (the first trigonometric moment)."""

    def __init__(
        self,
        rho_max: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if rho_max is not None:
            raise ValueError("wrapped-Cauchy boundary clipping is not a valid fit contract")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WrappedCauchyAccumulatorFactory:
        """Return an accumulator factory for circular-resultant statistics."""
        return WrappedCauchyAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> WrappedCauchyDistribution:
        """Estimate mean direction and concentration from the mean resultant."""
        count, sum_cos, sum_sin = validated_circular_statistics(
            suff_stat,
            count_index=2,
        )
        if count == 0.0:
            raise WrappedCauchyFitError("wrapped-Cauchy fitting requires positive observation weight")
        mu = math.atan2(sum_sin, sum_cos)
        rho = math.hypot(sum_cos, sum_sin) / count
        if rho >= 1.0 - 1.0e-12:
            raise WrappedCauchyFitError("wrapped-Cauchy resultant is on a boundary with no finite fit")
        result = WrappedCauchyDistribution(
            mu,
            rho,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "circular-resultant",
            "identifiable_direction": bool(rho > 1.0e-12),
            "resultant_length": rho,
            "repairs": (),
        }
        return result


class WrappedCauchyDataEncoder(DataSequenceEncoder):
    """Encode angles as their cosine and sine."""

    def __str__(self) -> str:
        return "WrappedCauchyDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WrappedCauchyDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode angles as cosine and sine arrays."""
        theta = validated_angles(x)
        return np.cos(theta), np.sin(theta)

    def row_count(self, x: tuple[np.ndarray, np.ndarray]) -> int:
        """Return the encoded row count after unit-circle validation."""
        cosine, _ = validated_trig(x)
        return len(cosine)
