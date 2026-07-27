"""Compositional transforms and a Jacobian-correct logratio-normal density.

Compositions are strictly positive vectors on the unit simplex. ``clr`` and
``ilr`` map positive rays to logratio coordinates; ``AitchisonNormalDistribution``
uses an ilr Gaussian and reports density with respect to ordinary Lebesgue
measure on the first ``D-1`` simplex coordinates. Consequently its score
includes the absolute ilr change-of-variables Jacobian

``|d ilr(x) / d(x_1,...,x_{D-1})| = 1 / (sqrt(D) * product_i x_i)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)

__all__ = [
    "closure",
    "clr",
    "clr_inv",
    "ilr",
    "ilr_inv",
    "ilr_basis",
    "AitchisonNormalDistribution",
    "AitchisonNormalSampler",
    "AitchisonNormalDataEncoder",
    "AitchisonNormalEstimator",
    "AitchisonNormalAccumulator",
]


def _matrix(value: Any, *, label: str, width: int | None = None) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 1:
        raise ValueError(f"{label} must be a non-empty vector or matrix of row observations.")
    if width is not None and raw.shape[1] != width:
        raise ValueError(f"{label} rows must have width {width}.")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.array(raw, dtype=np.float64, copy=True)


def _positive_matrix(value: Any, *, label: str, width: int | None = None) -> np.ndarray:
    result = _matrix(value, label=label, width=width)
    if result.shape[1] < 2:
        raise ValueError(f"{label} must contain at least two compositional parts.")
    if np.any(result <= 0.0):
        raise ValueError(f"{label} must contain strictly positive parts.")
    totals = result.sum(axis=1)
    if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError(f"{label} must have positive finite row totals.")
    return result


def _simplex_matrix(value: Any, *, label: str, width: int) -> np.ndarray:
    result = _positive_matrix(value, label=label, width=width)
    if not np.allclose(result.sum(axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{label} rows must sum to one.")
    return result


def _basis(value: Any, *, parts: int | None = None) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != raw.shape[0] - 1:
        raise ValueError("ilr basis must have shape (D, D-1) with D >= 2.")
    if parts is not None and raw.shape[0] != parts:
        raise ValueError(f"ilr basis must have {parts} rows.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("ilr basis must contain only finite values.")
    tolerance = 1.0e-10
    if not np.allclose(raw.T @ raw, np.eye(raw.shape[1]), rtol=tolerance, atol=tolerance):
        raise ValueError("ilr basis columns must be orthonormal.")
    if not np.allclose(raw.T @ np.ones(raw.shape[0]), 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("ilr basis columns must be zero-sum contrasts.")
    return np.array(raw, dtype=np.float64, copy=True)


def _exact_parts(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("part count must be an exact integer.")
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw):
        raise TypeError("part count must be an exact integer.")
    try:
        parts = int(raw.item())
        numeric = float(raw.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("part count must be an exact integer.") from exc
    if not math.isfinite(numeric) or numeric != parts or parts < 2:
        raise ValueError("part count must be an exact integer at least two.")
    return parts


def _pseudo_count(value: Any | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("pseudo_count must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("pseudo_count must be finite and nonnegative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("pseudo_count must be finite and nonnegative.")
    return result


def closure(x: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Normalize strictly positive finite rows to a positive finite total."""
    if isinstance(total, (bool, np.bool_)):
        raise TypeError("closure total must be finite and positive.")
    try:
        checked_total = float(total)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("closure total must be finite and positive.") from exc
    if not math.isfinite(checked_total) or checked_total <= 0.0:
        raise ValueError("closure total must be finite and positive.")
    values = _positive_matrix(x, label="composition")
    return checked_total * values / values.sum(axis=1, keepdims=True)


def clr(x: np.ndarray) -> np.ndarray:
    """Return centered logratios for strictly positive finite compositions."""
    values = _positive_matrix(x, label="composition")
    log_values = np.log(values)
    return log_values - log_values.mean(axis=1, keepdims=True)


def clr_inv(y: np.ndarray) -> np.ndarray:
    """Map finite clr-coordinate rows to the unit simplex."""
    values = _matrix(y, label="clr coordinates")
    if values.shape[1] < 2:
        raise ValueError("clr coordinates must have at least two parts.")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def ilr_basis(d: int) -> np.ndarray:
    """Return the Helmert ``(D, D-1)`` orthonormal contrast basis."""
    parts = _exact_parts(d)
    result = np.zeros((parts, parts - 1))
    for index in range(parts - 1):
        count = index + 1
        result[:count, index] = 1.0 / count
        result[count, index] = -1.0
        result[:, index] *= math.sqrt(count / (count + 1.0))
    return result


def ilr(x: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Map strictly positive ``D``-part rows to ``D-1`` ilr coordinates."""
    values = _positive_matrix(x, label="composition")
    contrasts = ilr_basis(values.shape[1]) if basis is None else _basis(basis, parts=values.shape[1])
    return clr(values) @ contrasts


def ilr_inv(y: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Map finite ``D-1`` ilr rows to strictly positive unit compositions."""
    values = _matrix(y, label="ilr coordinates")
    contrasts = ilr_basis(values.shape[1] + 1) if basis is None else _basis(basis)
    if values.shape[1] != contrasts.shape[1]:
        raise ValueError(f"ilr coordinate rows must have width {contrasts.shape[1]}.")
    return clr_inv(values @ contrasts.T)


def _simplex_log_jacobian(x: np.ndarray) -> np.ndarray:
    return -0.5 * math.log(x.shape[1]) - np.sum(np.log(x), axis=1)


def _encoded_payload(value: Any, n_parts: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("encoded Aitchison-normal data must be an (ilr_coordinates, log_jacobian) tuple.")
    coordinates = np.asarray(value[0], dtype=np.float64)
    log_jacobian = np.asarray(value[1], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != n_parts - 1:
        raise ValueError(f"encoded ilr coordinates must have shape (N, {n_parts - 1}).")
    if log_jacobian.shape != (coordinates.shape[0],):
        raise ValueError("encoded log-Jacobian must have one value per ilr row.")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(log_jacobian)):
        raise ValueError("encoded Aitchison-normal data must be finite.")
    return coordinates, log_jacobian


class AitchisonNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Logratio-normal density on ordinary unit-simplex coordinates."""

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.gaussian = MultivariateGaussianDistribution(
            np.asarray(mean, dtype=np.float64),
            np.asarray(cov, dtype=np.float64),
            name=name,
            keys=keys,
        )
        self.n_parts = self.gaussian.dim + 1
        self.name = name
        self.keys = keys

    @property
    def mean(self) -> np.ndarray:
        """Return the Gaussian mean in ilr coordinates."""
        return np.asarray(self.gaussian.mu)

    @property
    def cov(self) -> np.ndarray:
        """Return the Gaussian covariance in ilr coordinates."""
        return np.asarray(self.gaussian.covar)

    def __str__(self) -> str:
        return "AitchisonNormalDistribution(%r, %r, name=%r, keys=%r)" % (
            list(self.gaussian.mu),
            [list(row) for row in self.cov],
            self.name,
            self.keys,
        )

    def density(self, x: np.ndarray) -> float:
        """Return ordinary-coordinate simplex density."""
        return float(math.exp(self.log_density(x)))

    def log_density(self, x: np.ndarray) -> float:
        """Return density relative to the first ``D-1`` simplex coordinates."""
        composition = _simplex_matrix(x, label="Aitchison-normal observation", width=self.n_parts)
        coordinates = ilr(composition)[0]
        return float(self.gaussian.log_density(coordinates) + _simplex_log_jacobian(composition)[0])

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Score an encoded ``(ilr coordinates, log Jacobian)`` payload."""
        coordinates, log_jacobian = _encoded_payload(x, self.n_parts)
        return self.gaussian.seq_log_density(coordinates) + log_jacobian

    def mean_composition(self) -> np.ndarray:
        """Return the ilr-mean mapped to the unit simplex."""
        return ilr_inv(self.gaussian.mu)[0]

    def sampler(self, seed: int | None = None) -> AitchisonNormalSampler:
        """Return a push-forward sampler from the ilr Gaussian."""
        return AitchisonNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> AitchisonNormalEstimator:
        """Return an estimator preserving keys and optional current-model smoothing."""
        checked_pseudo_count = _pseudo_count(pseudo_count)
        return AitchisonNormalEstimator(
            dim=self.gaussian.dim,
            pseudo_count=checked_pseudo_count,
            suff_stat=(self.mean, self.cov) if checked_pseudo_count is not None else (None, None),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> AitchisonNormalDataEncoder:
        """Return a fixed-width composition encoder."""
        return AitchisonNormalDataEncoder(self.gaussian.dist_to_encoder(), self.n_parts)


class AitchisonNormalSampler(DistributionSampler):
    """Sample compositions by inverting Gaussian ilr draws."""

    def __init__(self, dist: AitchisonNormalDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.gaussian_sampler = dist.gaussian.sampler(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one composition or ``size`` iid compositions."""
        coordinates = self.gaussian_sampler.sample(size)
        return ilr_inv(np.atleast_2d(coordinates))[0] if size is None else ilr_inv(np.asarray(coordinates))


class AitchisonNormalDataEncoder(DataSequenceEncoder):
    """Encode simplex rows as ilr coordinates plus their log Jacobians."""

    def __init__(self, gaussian_encoder: DataSequenceEncoder, n_parts: int) -> None:
        self.gaussian_encoder = gaussian_encoder
        self.n_parts = _exact_parts(n_parts)

    def __str__(self) -> str:
        return "AitchisonNormalDataEncoder(n_parts=%d, gaussian_encoder=%s)" % (
            self.n_parts,
            self.gaussian_encoder,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AitchisonNormalDataEncoder)
            and self.n_parts == other.n_parts
            and self.gaussian_encoder == other.gaussian_encoder
        )

    def seq_encode(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """Validate unit-simplex rows and encode coordinates with Jacobians."""
        compositions = _simplex_matrix(x, label="Aitchison-normal observations", width=self.n_parts)
        coordinates = self.gaussian_encoder.seq_encode(ilr(compositions))
        return coordinates, _simplex_log_jacobian(compositions)

    def row_count(self, x: tuple[np.ndarray, np.ndarray]) -> int:
        """Return the number of aligned encoded rows."""
        coordinates, _ = _encoded_payload(x, self.n_parts)
        return len(coordinates)


class AitchisonNormalEstimator(ParameterEstimator):
    """Fit a Gaussian in ilr coordinates with explicit smoothing and keys."""

    def __init__(
        self,
        dim: int | None = None,
        pseudo_count: float | None = None,
        suff_stat: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        checked_pseudo_count = _pseudo_count(pseudo_count)
        self.gaussian_estimator = MultivariateGaussianEstimator(
            dim=dim,
            pseudo_count=(
                (None, None)
                if checked_pseudo_count is None
                else (checked_pseudo_count, checked_pseudo_count)
            ),
            suff_stat=suff_stat,
            name=name,
            keys=keys,
        )
        self.dim = dim
        self.pseudo_count = checked_pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StatisticAccumulatorFactory:
        """Return a keyed accumulator factory delegated to the Gaussian estimator."""
        gaussian_factory = self.gaussian_estimator.accumulator_factory()
        keys = self.keys
        n_parts = None if self.dim is None else self.dim + 1

        class _Factory(StatisticAccumulatorFactory):
            def make(self) -> AitchisonNormalAccumulator:
                return AitchisonNormalAccumulator(gaussian_factory.make(), n_parts=n_parts, keys=keys)

        return _Factory()

    def estimate(self, nobs: float | None, suff_stat: Any) -> AitchisonNormalDistribution:
        """Estimate ilr Gaussian parameters from delegated sufficient statistics."""
        gaussian = self.gaussian_estimator.estimate(nobs, suff_stat)
        return AitchisonNormalDistribution(
            gaussian.mu,
            gaussian.covar,
            name=self.name,
            keys=self.keys,
        )


class AitchisonNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Delegate ilr-coordinate moments while preserving wrapper key semantics."""

    def __init__(self, gaussian_acc: Any, n_parts: int | None, keys: str | None = None) -> None:
        self.gaussian_acc = gaussian_acc
        self.n_parts = n_parts
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: AitchisonNormalDistribution | None) -> None:
        """Update delegated Gaussian statistics from one raw composition."""
        width = self.n_parts if self.n_parts is not None else _positive_matrix(x, label="composition").shape[1]
        composition = _simplex_matrix(x, label="Aitchison-normal observation", width=width)
        self.gaussian_acc.update(ilr(composition)[0], weight, None if estimate is None else estimate.gaussian)
        if self.n_parts is None:
            self.n_parts = width

    def initialize(self, x: np.ndarray, weight: float, rng: Any) -> None:
        """Initialize delegated statistics from one raw composition."""
        self.update(x, weight, None)

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: AitchisonNormalDistribution | None,
    ) -> None:
        """Update delegated moments from encoded ilr coordinates."""
        if self.n_parts is None:
            coordinates = np.asarray(x[0], dtype=np.float64)
            if coordinates.ndim != 2:
                raise ValueError("encoded ilr coordinates must be a matrix.")
            self.n_parts = coordinates.shape[1] + 1
        coordinates, _ = _encoded_payload(x, self.n_parts)
        self.gaussian_acc.seq_update(coordinates, weights, None if estimate is None else estimate.gaussian)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: Any) -> None:
        """Initialize delegated moments from encoded ilr coordinates."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Any) -> AitchisonNormalAccumulator:
        """Merge delegated Gaussian sufficient statistics."""
        self.gaussian_acc.combine(suff_stat)
        return self

    def value(self) -> Any:
        """Return delegated Gaussian sufficient statistics."""
        return self.gaussian_acc.value()

    def from_value(self, x: Any) -> AitchisonNormalAccumulator:
        """Restore delegated Gaussian sufficient statistics."""
        self.gaussian_acc.from_value(x)
        if self.n_parts is None and getattr(self.gaussian_acc, "dim", None) is not None:
            self.n_parts = self.gaussian_acc.dim + 1
        return self

    def acc_to_encoder(self) -> AitchisonNormalDataEncoder:
        """Return an encoder compatible with accumulated ilr dimension."""
        if self.n_parts is None:
            raise ValueError("cannot create an Aitchison-normal encoder before the part count is known.")
        return AitchisonNormalDataEncoder(self.gaussian_acc.acc_to_encoder(), self.n_parts)
