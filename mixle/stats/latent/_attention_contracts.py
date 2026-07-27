"""Shared probability and observation contracts for attention distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.stats.latent.effective_sample import validated_observation_weights


@dataclass(frozen=True)
class AttentionOptimizerState:
    """Serializable Adam state carried by variational attention models."""

    family: str
    mean: np.ndarray
    log_var: np.ndarray
    mean_first_moment: np.ndarray
    mean_second_moment: np.ndarray
    log_var_first_moment: np.ndarray
    log_var_second_moment: np.ndarray
    iteration: int
    seed: int
    schema_version: int = 1
    algorithm: str = "adam"

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("attention optimizer family must be a nonempty string")
        if self.schema_version != 1 or self.algorithm != "adam":
            raise ValueError("unsupported attention optimizer-state schema")
        if isinstance(self.iteration, (bool, np.bool_)) or int(self.iteration) != self.iteration:
            raise TypeError("attention optimizer iteration must be a non-negative integer")
        if self.iteration < 0:
            raise ValueError("attention optimizer iteration must be non-negative")
        if isinstance(self.seed, (bool, np.bool_)) or int(self.seed) != self.seed:
            raise TypeError("attention optimizer seed must be an integer")
        arrays = {
            "mean": self.mean,
            "log_var": self.log_var,
            "mean_first_moment": self.mean_first_moment,
            "mean_second_moment": self.mean_second_moment,
            "log_var_first_moment": self.log_var_first_moment,
            "log_var_second_moment": self.log_var_second_moment,
        }
        normalized = {name: finite_matrix(value, name) for name, value in arrays.items()}
        shape = normalized["mean"].shape
        if any(value.shape != shape for value in normalized.values()):
            raise ValueError("all attention optimizer-state arrays must share one shape")
        for name, value in normalized.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def fresh(cls, family: str, mean: np.ndarray, log_var: np.ndarray, seed: int) -> AttentionOptimizerState:
        """Create a zero-moment state around an existing posterior."""
        mean_array = finite_matrix(mean, "mean")
        log_var_array = finite_matrix(log_var, "log_var")
        if mean_array.shape != log_var_array.shape:
            raise ValueError("mean and log_var must have the same shape")
        zeros = np.zeros_like(mean_array)
        return cls(family, mean_array, log_var_array, zeros, zeros, zeros, zeros, 0, seed)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data snapshot suitable for receipts and sufficient statistics."""
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "family": self.family,
            "mean": self.mean.tolist(),
            "log_var": self.log_var.tolist(),
            "mean_first_moment": self.mean_first_moment.tolist(),
            "mean_second_moment": self.mean_second_moment.tolist(),
            "log_var_first_moment": self.log_var_first_moment.tolist(),
            "log_var_second_moment": self.log_var_second_moment.tolist(),
            "iteration": self.iteration,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AttentionOptimizerState:
        """Restore a state from :meth:`to_dict` output."""
        if not isinstance(value, dict):
            raise TypeError("attention optimizer state must be a dictionary")
        required = {
            "family",
            "mean",
            "log_var",
            "mean_first_moment",
            "mean_second_moment",
            "log_var_first_moment",
            "log_var_second_moment",
            "iteration",
            "seed",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"attention optimizer state is missing fields: {sorted(missing)}")
        return cls(
            family=value["family"],
            mean=value["mean"],
            log_var=value["log_var"],
            mean_first_moment=value["mean_first_moment"],
            mean_second_moment=value["mean_second_moment"],
            log_var_first_moment=value["log_var_first_moment"],
            log_var_second_moment=value["log_var_second_moment"],
            iteration=value["iteration"],
            seed=value["seed"],
            schema_version=value.get("schema_version", 1),
            algorithm=value.get("algorithm", "adam"),
        )


def merge_optimizer_state(
    current: AttentionOptimizerState | None,
    incoming: AttentionOptimizerState | dict[str, Any] | None,
    *,
    family: str,
) -> AttentionOptimizerState | None:
    """Merge a non-additive optimizer snapshot, rejecting mixed model generations."""
    other = AttentionOptimizerState.from_dict(incoming) if isinstance(incoming, dict) else incoming
    if other is not None and other.family != family:
        raise ValueError(f"optimizer state belongs to {other.family!r}, not {family!r}")
    if current is None:
        return other
    if other is None:
        return current
    scalar_equal = (
        current.family,
        current.iteration,
        current.seed,
        current.schema_version,
        current.algorithm,
    ) == (other.family, other.iteration, other.seed, other.schema_version, other.algorithm)
    array_equal = all(
        np.array_equal(getattr(current, field), getattr(other, field))
        for field in (
            "mean",
            "log_var",
            "mean_first_moment",
            "mean_second_moment",
            "log_var_first_moment",
            "log_var_second_moment",
        )
    )
    if not scalar_equal or not array_equal:
        raise ValueError("cannot combine attention statistics from different optimizer states")
    return current


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_, float, np.floating)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite real scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def finite_matrix(value: Any, name: str, *, ndim: int = 2) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real-valued") from exc
    if result.ndim != ndim or any(size <= 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def row_simplex(value: Any, name: str) -> np.ndarray:
    result = finite_matrix(value, name)
    if np.any(result < 0.0) or not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} rows must be non-negative and sum to one")
    return result


def simplex(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real-valued") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if (
        np.any(~np.isfinite(result))
        or np.any(result < 0.0)
        or not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"{name} must be a finite non-negative simplex")
    return result.copy()


def exact_ids(value: Any, name: str, *, upper: int | None = None, ndim: int | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular array of exact integers") from exc
    if raw.dtype.kind == "b" or (ndim is not None and raw.ndim != ndim):
        raise TypeError(f"{name} must contain exact non-boolean integers")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain exact integers") from exc
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must contain finite exact integers")
    result = numeric.astype(np.intp)
    if np.any(result < 0) or (upper is not None and np.any(result >= upper)):
        raise ValueError(f"{name} contains an out-of-range ID")
    return result


def observation_weights(value: Any, n: int) -> np.ndarray:
    return validated_observation_weights(value, n, "attention observation weights")


def safe_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Preserve structural zeros while taking stable logarithms."""
    values = np.asarray(probabilities, dtype=np.float64)
    result = np.full(values.shape, -np.inf, dtype=np.float64)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result


def weighted_log_probability_sum(probabilities: np.ndarray, weights: np.ndarray) -> float:
    """Sum weighted log probabilities without producing ``0 * -inf``."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive_weight = weights > 0.0
    if np.any((probabilities <= 0.0) & positive_weight):
        return float("-inf")
    return float(np.dot(weights[positive_weight], np.log(probabilities[positive_weight])))


def normalize_log_rows(log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize finite log rows and leave impossible rows at zero responsibility."""
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 2 or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("attention log weights must be a finite-or-minus-infinity matrix")
    possible = np.any(np.isfinite(values), axis=1)
    log_total = np.full(values.shape[0], -np.inf, dtype=np.float64)
    responsibilities = np.zeros_like(values)
    if np.any(possible):
        selected = values[possible]
        maxima = selected.max(axis=1, keepdims=True)
        weights = np.exp(selected - maxima)
        totals = weights.sum(axis=1, keepdims=True)
        responsibilities[possible] = weights / totals
        log_total[possible] = np.log(totals[:, 0]) + maxima[:, 0]
    return log_total, responsibilities
