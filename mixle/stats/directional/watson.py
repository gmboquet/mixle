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
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.optimize import brentq
from scipy.special import gammaln, hyp1f1

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


class WatsonFitError(RuntimeError):
    """Raised when Watson moments have no certified finite fit."""


class WatsonSamplingError(RuntimeError):
    """Raised when exact Watson rejection sampling exhausts its proposal budget."""

    def __init__(
        self,
        accepted: int,
        proposed: int,
        kappa: float,
        dim: int,
    ) -> None:
        self.accepted = accepted
        self.proposed = proposed
        self.kappa = kappa
        self.dim = dim
        super().__init__(
            "Watson rejection sampler accepted %d of %d proposals (kappa=%g, dim=%d)" % (accepted, proposed, kappa, dim)
        )


def _validated_dimension(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Watson dimension must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("Watson dimension must be an integer") from exc
    if result < 2:
        raise ValueError("Watson dimension must be an integer of at least two")
    return result


def _watson_log_normalizer_and_ratio(
    kappa: float,
    p: int,
) -> tuple[float, float]:
    """Return ``(log M(1/2,p/2,kappa), d log M / d kappa)``.

    Positive concentration uses Kummer's transformation, so the exponentially
    large factor is represented as an additive ``kappa`` rather than passed to
    ``hyp1f1``.  Negative concentration is already on the non-growing branch.
    """
    p = _validated_dimension(p)
    if not np.isfinite(kappa):
        raise ValueError("Watson concentration must be finite")
    if kappa == 0.0:
        return 0.0, 1.0 / p
    a = 0.5
    b = p / 2.0

    def asymptotic_series(
        first: float,
        second: float,
        magnitude: float,
    ) -> tuple[float, float]:
        total = 1.0
        derivative = 0.0
        term = 1.0
        previous_abs = math.inf
        for order in range(1, 65):
            term *= (first + order - 1.0) * (second + order - 1.0) / (order * magnitude)
            if not np.isfinite(term) or abs(term) > previous_abs:
                break
            total += term
            derivative -= order * term / magnitude
            if abs(term) <= 1.0e-15 * max(1.0, abs(total)):
                break
            previous_abs = abs(term)
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("Watson Kummer asymptotic series did not converge")
        return total, derivative

    if kappa > 0.0:
        c = b - a
        denominator = float(hyp1f1(c, b, -kappa))
        numerator = float(hyp1f1(c + 1.0, b + 1.0, -kappa))
        if denominator > 0.0 and np.isfinite(denominator) and np.isfinite(numerator):
            log_normalizer = kappa + math.log(denominator)
            ratio = 1.0 - (c / b) * numerator / denominator
        else:
            series, derivative = asymptotic_series(c, 1.0 - a, kappa)
            log_normalizer = kappa + (a - b) * math.log(kappa) + float(gammaln(b) - gammaln(a)) + math.log(series)
            ratio = 1.0 + (a - b) / kappa + derivative / series
    else:
        denominator = float(hyp1f1(a, b, kappa))
        numerator = float(hyp1f1(a + 1.0, b + 1.0, kappa))
        if denominator > 0.0 and np.isfinite(denominator) and np.isfinite(numerator):
            log_normalizer = math.log(denominator)
            ratio = (a / b) * numerator / denominator
        else:
            magnitude = -kappa
            series, derivative = asymptotic_series(
                a,
                1.0 + a - b,
                magnitude,
            )
            log_normalizer = float(gammaln(b) - gammaln(b - a)) - a * math.log(magnitude) + math.log(series)
            ratio = a / magnitude - derivative / series
    if not np.isfinite(log_normalizer) or not np.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("Watson Kummer evaluation violated its finite moment contract")
    return log_normalizer, ratio


def _kummer_ratio(kappa: float, p: int) -> float:
    """``E[(mu^T x)^2]`` under Watson = ``M'(1/2,p/2,k)/M(1/2,p/2,k) = (1/p) M(3/2,(p+2)/2,k)/M(1/2,p/2,k)``."""
    return _watson_log_normalizer_and_ratio(float(kappa), p)[1]


def _solve_kappa(r: float, p: int) -> float:
    """Solve the monotone Watson moment equation with an adaptive finite bracket."""
    p = _validated_dimension(p)
    if not np.isfinite(r) or r <= 0.0 or r >= 1.0:
        raise WatsonFitError("Watson moment must lie strictly between zero and one")
    uniform_moment = 1.0 / p
    if abs(r - uniform_moment) <= 1.0e-12:
        return 0.0
    if r < uniform_moment:
        lo, hi = -1.0, 0.0
        while _kummer_ratio(lo, p) > r:
            lo *= 2.0
            if lo < -1.0e12:
                raise WatsonFitError("Watson concentration could not be bracketed on the girdle branch")
    else:
        lo, hi = 0.0, 1.0
        while _kummer_ratio(hi, p) < r:
            hi *= 2.0
            if hi > 1.0e12:
                raise WatsonFitError("Watson concentration could not be bracketed on the bipolar branch")
    try:
        result = brentq(
            lambda value: _kummer_ratio(value, p) - r,
            lo,
            hi,
            xtol=1.0e-10,
            rtol=1.0e-12,
            maxiter=200,
        )
    except (RuntimeError, ValueError) as exc:
        raise WatsonFitError("Watson concentration solve failed") from exc
    if not np.isfinite(result):
        raise WatsonFitError("Watson concentration solve returned a non-finite value")
    return float(result)


def _validated_watson_statistics(
    value: Any,
    dim: int,
) -> tuple[np.ndarray, float]:
    dim = _validated_dimension(dim)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Watson sufficient statistics must be a two-item tuple")
    try:
        scatter = np.asarray(value[0], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Watson scatter must be numeric") from exc
    if isinstance(value[1], (bool, np.bool_)) or np.ndim(value[1]) != 0:
        raise TypeError("Watson count must be a real scalar")
    try:
        count = float(value[1])
    except (TypeError, ValueError) as exc:
        raise TypeError("Watson count must be a real scalar") from exc
    if scatter.shape != (dim, dim) or np.any(~np.isfinite(scatter)):
        raise ValueError("Watson scatter must be a finite %dx%d matrix" % (dim, dim))
    if not np.array_equal(scatter, scatter.T):
        raise ValueError("Watson scatter must be exactly symmetric")
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("Watson count must be finite and non-negative")
    if count == 0.0:
        if np.any(scatter != 0.0):
            raise ValueError("empty Watson statistics must have zero scatter")
    else:
        tolerance = _MOMENT_ATOL * max(1.0, count)
        if abs(float(np.trace(scatter)) - count) > tolerance:
            raise ValueError("Watson scatter trace must equal its observation weight")
        if float(np.linalg.eigvalsh(scatter).min()) < -tolerance:
            raise ValueError("Watson scatter must be positive semidefinite")
    return scatter.copy(), count


class WatsonDistribution(SequenceEncodableProbabilityDistribution):
    """Watson distribution on the unit sphere ``S^{p-1}`` with axis ``mu`` and concentration ``kappa``."""

    def __init__(self, mu: np.ndarray, kappa: float, name: str | None = None, keys: str | None = None) -> None:
        try:
            raw_mu = np.asarray(mu, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Watson axis must be numeric") from exc
        if raw_mu.ndim != 1 or raw_mu.size < 2:
            raise ValueError("Watson axis must have dimension of at least two")
        self.dim = _validated_dimension(raw_mu.size)
        self.mu = _unit_vector(raw_mu, self.dim, "Watson axis")
        self.mu.setflags(write=False)
        if isinstance(kappa, (bool, np.bool_)) or np.ndim(kappa) != 0:
            raise TypeError("Watson concentration must be a real scalar")
        try:
            self.kappa = float(kappa)
        except (TypeError, ValueError) as exc:
            raise TypeError("Watson concentration must be a real scalar") from exc
        if not np.isfinite(self.kappa):
            raise ValueError("Watson concentration must be finite")
        self.name = name
        self.keys = keys
        log_omega = math.log(2.0) + (self.dim / 2.0) * math.log(math.pi) - math.lgamma(self.dim / 2.0)
        log_kummer, _ = _watson_log_normalizer_and_ratio(
            self.kappa,
            self.dim,
        )
        self._log_const = -log_omega - log_kummer
        if not np.isfinite(self._log_const):
            raise ValueError("Watson normalizer must be finite")

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
        dot = float(np.dot(_unit_vector(x, self.dim, "Watson observation"), self.mu))
        return self._log_const + self.kappa * dot * dot

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of unit vectors, shape ``(N, p)``."""
        dots = _unit_batch(x, self.dim, "Watson observations") @ self.mu
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
        from mixle.engines.symbolic_engine import is_symbolic_payload

        checked = x if is_symbolic_payload(x) else _unit_batch(x, self.dim, "Watson backend observations")
        dots = engine.matmul(
            engine.asarray(checked),
            engine.asarray(owned_backend_parameter(self.mu)),
        )
        return self._log_const + self.kappa * dots * dots

    def sampler(self, seed: int | None = None) -> "WatsonSampler":
        """Return an exact rejection sampler for this Watson distribution."""
        return WatsonSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WatsonEstimator":
        """Return a maximum-likelihood estimator (scatter eigenvector + Kummer-ratio kappa solve)."""
        if pseudo_count is not None:
            raise ValueError("Watson pseudo-count regularization is not implemented")
        return WatsonEstimator(self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WatsonDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return WatsonDataEncoder(self.dim)


class WatsonSampler(DistributionSampler):
    """Draw Watson axes with exact, bounded rejection sampling."""

    def __init__(self, dist: WatsonDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self._positive_proposal = None
        self._positive_log_bound = 0.0
        if dist.kappa > 0.0:
            from mixle.stats.directional.von_mises_fisher import (
                VonMisesFisherDistribution,
            )

            proposal_seed = int(self.rng.randint(0, np.iinfo(np.int32).max))
            self._positive_proposal = VonMisesFisherDistribution(
                dist.mu,
                dist.kappa,
            ).sampler(seed=proposal_seed)
            log_cosh = float(np.logaddexp(dist.kappa, -dist.kappa)) - math.log(2.0)
            self._positive_log_bound = max(0.0, dist.kappa - log_cosh)
        self.sampling_metadata = {
            "method": "exact-rejection",
            "exact": True,
            "accepted": 0,
            "proposed": 0,
            "acceptance_rate": None,
        }

    def _uniform_sphere(self, size: int) -> np.ndarray:
        values = self.rng.standard_normal((size, self.dist.dim))
        norms = np.linalg.norm(values, axis=1)
        while np.any(norms == 0.0):
            zero = norms == 0.0
            values[zero] = self.rng.standard_normal((int(zero.sum()), self.dist.dim))
            norms[zero] = np.linalg.norm(values[zero], axis=1)
        return values / norms[:, None]

    def _propose(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        kappa = self.dist.kappa
        if kappa > 0.0:
            if self._positive_proposal is None:
                raise RuntimeError("Watson positive-concentration proposal is missing")
            values = np.asarray(
                self._positive_proposal.sample(size=size),
                dtype=np.float64,
            )
            signs = np.where(self.rng.uniform(size=size) < 0.5, 1.0, -1.0)
            values *= signs[:, None]
            projection = values @ self.dist.mu
            log_cosh = np.logaddexp(kappa * projection, -kappa * projection) - math.log(2.0)
            log_acceptance = kappa * projection * projection - log_cosh - self._positive_log_bound
            return values, log_acceptance
        values = self._uniform_sphere(size)
        projection = values @ self.dist.mu
        return values, kappa * projection * projection

    def _batch(self, size: int) -> np.ndarray:
        if size == 0:
            return np.empty((0, self.dist.dim), dtype=np.float64)
        if self.dist.kappa == 0.0:
            output = self._uniform_sphere(size)
            self.sampling_metadata = {
                "method": "exact-uniform-sphere",
                "exact": True,
                "accepted": size,
                "proposed": size,
                "acceptance_rate": 1.0,
            }
            return output
        proposal_budget = max(10_000, size * 1_000)
        output = np.empty((size, self.dist.dim), dtype=np.float64)
        accepted = 0
        proposed = 0
        while accepted < size and proposed < proposal_budget:
            batch_size = min(
                (size - accepted) * 2 + 8,
                proposal_budget - proposed,
            )
            values, log_acceptance = self._propose(batch_size)
            proposed += batch_size
            mask = np.log(self.rng.uniform(size=batch_size)) < np.minimum(
                log_acceptance,
                0.0,
            )
            selected = values[mask]
            take = min(len(selected), size - accepted)
            output[accepted : accepted + take] = selected[:take]
            accepted += take
        self.sampling_metadata = {
            "method": "exact-rejection",
            "exact": True,
            "accepted": accepted,
            "proposed": proposed,
            "acceptance_rate": accepted / proposed if proposed else None,
        }
        if accepted < size:
            raise WatsonSamplingError(
                accepted,
                proposed,
                self.dist.kappa,
                self.dist.dim,
            )
        return output

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one unit vector or a stack of iid unit vectors."""
        if size is None:
            return self._batch(1)[0]
        return self._batch(_validated_sample_size(size))


class WatsonAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted scatter matrix ``S = sum_i w_i x_i x_i^T`` and total weight."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim)
        self.scatter = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: WatsonDistribution | None) -> None:
        """Accumulate one weighted outer product into the scatter matrix."""
        xx = _unit_vector(x, self.dim, "Watson observation")
        checked_weight = _validated_weight(weight)
        self.scatter += checked_weight * np.outer(xx, xx)
        self.count += checked_weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one unit vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: WatsonDistribution | None) -> None:
        """Accumulate weighted scatter statistics from encoded unit vectors."""
        xx = _unit_batch(x, self.dim, "Watson observations")
        w = _validated_weights(weights, len(xx))
        self.scatter += (xx * w[:, None]).T @ xx
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded unit vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "WatsonAccumulator":
        """Merge another Watson sufficient-statistic tuple."""
        scatter, count = _validated_watson_statistics(suff_stat, self.dim)
        combined = (self.scatter + scatter, self.count + count)
        self.scatter, self.count = _validated_watson_statistics(
            combined,
            self.dim,
        )
        return self

    def value(self) -> tuple[np.ndarray, float]:
        """Return the scatter matrix and total weight."""
        return self.scatter.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, float]) -> "WatsonAccumulator":
        """Replace accumulator contents from scatter statistics."""
        self.scatter, self.count = _validated_watson_statistics(x, self.dim)
        return self

    def scale(self, c: float) -> "WatsonAccumulator":
        """Scale linear Watson sufficient statistics."""
        checked_scale = _validated_weight(c)
        self.scatter *= checked_scale
        self.count *= checked_scale
        return self

    def acc_to_encoder(self) -> "WatsonDataEncoder":
        """Return the encoder used by this accumulator."""
        return WatsonDataEncoder(self.dim)


class WatsonAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WatsonAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim)
        self.name = name
        self.keys = keys

    def make(self) -> WatsonAccumulator:
        """Create a fresh Watson accumulator."""
        return WatsonAccumulator(self.dim, name=self.name, keys=self.keys)


class WatsonEstimator(ParameterEstimator):
    """Maximum-likelihood estimator: scatter eigenvector for the axis, Kummer-ratio solve for kappa."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WatsonAccumulatorFactory:
        """Return an accumulator factory for Watson scatter statistics."""
        return WatsonAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float]) -> WatsonDistribution:
        """Estimate the Watson axis and concentration from weighted scatter."""
        scatter, count = _validated_watson_statistics(suff_stat, self.dim)
        p = self.dim
        if count == 0.0:
            raise WatsonFitError("Watson fitting requires positive observation weight")
        s_mat = scatter / count  # mean scatter; eigenvalues in [0,1], sum to 1
        eigval, eigvec = np.linalg.eigh(s_mat)
        if eigval[0] <= _MOMENT_ATOL or eigval[-1] >= 1.0 - _MOMENT_ATOL:
            raise WatsonFitError("Watson moments lie on a boundary with no finite concentration fit")
        uniform_moment = 1.0 / p
        if float(np.max(np.abs(eigval - uniform_moment))) <= _MOMENT_ATOL:
            result = WatsonDistribution(
                np.eye(p)[0],
                0.0,
                name=self.name,
                keys=self.keys,
            )
            result.fit_metadata = {
                "converged": True,
                "solver": "isotropic-limit",
                "identifiable_axis": False,
                "selected_moment": uniform_moment,
                "repairs": (),
            }
            return result
        # bipolar (kappa>0): the data align with the top eigenvector; girdle (kappa<0): the bottom one
        r_top, r_bot = float(eigval[-1]), float(eigval[0])
        if abs(r_top - uniform_moment) >= abs(r_bot - uniform_moment):
            mu, r = eigvec[:, -1], r_top
            eigen_gap = r_top - float(eigval[-2])
        else:
            mu, r = eigvec[:, 0], r_bot
            eigen_gap = float(eigval[1]) - r_bot
        kappa = _solve_kappa(r, p)
        result = WatsonDistribution(
            mu,
            kappa,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "brentq-moment-equation",
            "identifiable_axis": bool(eigen_gap > _MOMENT_ATOL),
            "selected_moment": r,
            "moment_residual": abs(_kummer_ratio(kappa, p) - r),
            "repairs": (),
        }
        return result


class WatsonDataEncoder(DataSequenceEncoder):
    """Encode a sequence of unit vectors as an ``(N, p)`` float array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else _validated_dimension(dim)

    def __str__(self) -> str:
        return "WatsonDataEncoder(dim=%r)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WatsonDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode unit vectors as an ``(N, p)`` floating-point array."""
        if self.dim is None:
            try:
                raw = np.asarray(x, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Watson observations must be numeric") from exc
            if raw.shape == (0,):
                raise ValueError("cannot infer Watson dimension from an empty observation batch")
            if raw.ndim != 2 or raw.shape[1] < 2:
                raise ValueError("Watson observations require dimension at least two")
            return _unit_batch(raw, raw.shape[1], "Watson observations")
        return _unit_batch(x, self.dim, "Watson observations")

    def row_count(self, x: np.ndarray) -> int:
        """Return the encoded row count after sphere validation."""
        return len(self.seq_encode(x))
