"""Dirichlet-process mixture experiment helpers.

This module keeps nonparametric-mixture logic in the model layer.  It exposes
small stick-breaking utilities and a dependency-free truncated variational
mixture loop over ordinary ``pysp.stats`` component estimators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import pysp.utils.vector as vec
from pysp.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution
from pysp.utils.special import digamma
from pysp.utils.special import softmax_rows as _softmax_rows

_EPS = 1.0e-300


@dataclass
class TruncatedDirichletProcessMixtureFitResult:
    """Fitted truncated DPM plus variational responsibilities and history."""

    model: TruncatedDirichletProcessMixtureModel
    responsibilities: np.ndarray
    history: list[float]


class TruncatedDirichletProcessMixtureModel:
    """Truncated stick-breaking mixture over existing pysp component models."""

    def __init__(
        self,
        components: Sequence[SequenceEncodableProbabilityDistribution],
        alpha: float = 1.0,
        gamma: Any | None = None,
        weights: Any | None = None,
        name: str | None = None,
    ) -> None:
        if len(components) == 0:
            raise ValueError("TruncatedDirichletProcessMixtureModel requires at least one component.")
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError("alpha must be finite and positive.")
        self.components = list(components)
        self.num_components = len(self.components)
        self.alpha = float(alpha)
        self.name = name
        if gamma is None:
            self.gamma = np.column_stack(
                [
                    np.ones(self.num_components, dtype=np.float64),
                    np.full(self.num_components, self.alpha, dtype=np.float64),
                ]
            )
        else:
            self.gamma = _as_gamma(gamma, self.num_components)
        if weights is None:
            self.weights = mean_stick_weights(self.gamma)
        else:
            self.weights = _as_simplex(weights, self.num_components, "weights")
        self.log_weights = np.log(np.clip(self.weights, _EPS, 1.0))

    def __str__(self) -> str:
        return "TruncatedDirichletProcessMixtureModel(num_components=%d, alpha=%r, name=%r)" % (
            self.num_components,
            self.alpha,
            self.name,
        )

    @property
    def expected_log_weights(self) -> np.ndarray:
        """Return E_q[log pi_k] under the variational stick posteriors."""
        return expected_log_stick_weights(self.gamma)

    def component_log_density(self, x: Any) -> np.ndarray:
        """Return component log densities for one observation."""
        return np.asarray([d.log_density(x) for d in self.components], dtype=np.float64)

    def log_density(self, x: Any) -> float:
        """Return the finite-truncation mixture log density for one observation."""
        return vec.log_sum(self.component_log_density(x) + self.log_weights)

    def density(self, x: Any) -> float:
        """Return the finite-truncation mixture density for one observation."""
        return float(np.exp(self.log_density(x)))

    def responsibilities(self, data: Sequence[Any], expected: bool = True) -> np.ndarray:
        """Return posterior component probabilities for observations."""
        scores = _component_log_density_matrix(self.components, data)
        log_prior = self.expected_log_weights if expected else self.log_weights
        return _softmax_rows(scores + log_prior[None, :])

    def effective_components(self, threshold: float = 0.01) -> int:
        """Count components with posterior mean stick weight above ``threshold``."""
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")
        return int(np.count_nonzero(self.weights > threshold))

    def sample(self, size: int | None = None, seed: int | None = None) -> Any | list[Any]:
        """Draw observations from the finite truncation."""
        rng = np.random.RandomState(seed)
        samplers = [d.sampler(seed=int(rng.randint(0, 2**31 - 1))) for d in self.components]
        states = rng.choice(self.num_components, size=size, replace=True, p=self.weights)
        if size is None:
            return samplers[int(states)].sample()
        return [samplers[int(k)].sample() for k in states]


def stick_breaking_weights(stick_fractions: Any, residual: bool = True) -> np.ndarray:
    """Convert stick fractions into mixture weights.

    When ``residual`` is true, the returned vector has one extra final entry
    containing the remaining stick mass.  This is the usual finite truncation.
    """
    v = np.asarray(stick_fractions, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("stick_fractions must be one-dimensional.")
    if np.any(~np.isfinite(v)) or np.any(v < 0.0) or np.any(v > 1.0):
        raise ValueError("stick fractions must be finite values in [0, 1].")
    remaining = 1.0
    weights = []
    for frac in v:
        weights.append(remaining * float(frac))
        remaining *= 1.0 - float(frac)
    if residual:
        weights.append(remaining)
    return np.asarray(weights, dtype=np.float64)


def expected_log_stick_weights(gamma: Any) -> np.ndarray:
    """Return E_q[log pi_k] for truncated Beta stick posteriors."""
    gam = _as_gamma(gamma)
    if gam.shape[0] == 1:
        return np.zeros(1, dtype=np.float64)
    total = gam[:, 0] + gam[:, 1]
    exp_log_v = digamma(gam[:, 0]) - digamma(total)
    exp_log_not_v = digamma(gam[:, 1]) - digamma(total)
    rv = np.empty(gam.shape[0], dtype=np.float64)
    remaining = 0.0
    for i in range(gam.shape[0] - 1):
        rv[i] = remaining + exp_log_v[i]
        remaining += exp_log_not_v[i]
    rv[-1] = remaining
    return rv


def mean_stick_weights(gamma: Any) -> np.ndarray:
    """Return E_q[pi_k] under independent Beta stick posteriors."""
    gam = _as_gamma(gamma)
    if gam.shape[0] == 1:
        return np.ones(1, dtype=np.float64)
    mean_v = gam[:, 0] / (gam[:, 0] + gam[:, 1])
    weights = []
    remaining = 1.0
    for i in range(gam.shape[0] - 1):
        weights.append(remaining * mean_v[i])
        remaining *= 1.0 - mean_v[i]
    weights.append(remaining)
    return _as_simplex(weights, gam.shape[0], "mean stick weights")


def sample_crp_assignments(num_obs: int, alpha: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample Chinese-restaurant-process assignments and table counts."""
    if num_obs < 0:
        raise ValueError("num_obs must be non-negative.")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive.")
    rng = np.random.RandomState(seed)
    assignments = np.empty(int(num_obs), dtype=np.int64)
    counts: list[int] = []
    for i in range(int(num_obs)):
        probs = np.asarray(counts + [float(alpha)], dtype=np.float64)
        probs /= probs.sum()
        k = int(rng.choice(len(probs), p=probs))
        if k == len(counts):
            counts.append(1)
        else:
            counts[k] += 1
        assignments[i] = k
    return assignments, np.asarray(counts, dtype=np.int64)


def fit_truncated_dpm(
    data: Sequence[Any],
    initial_components: Sequence[SequenceEncodableProbabilityDistribution],
    component_estimator: ParameterEstimator | Sequence[ParameterEstimator],
    alpha: float = 1.0,
    max_its: int = 50,
    tol: float | None = 1.0e-8,
    sort_components: bool = True,
    name: str | None = None,
) -> TruncatedDirichletProcessMixtureFitResult:
    """Fit a truncated DP mixture by coordinate-ascent variational updates.

    The component M-steps are delegated to ordinary ``pysp.stats`` estimators.
    This keeps component likelihood math and sufficient statistics in their
    distribution modules.
    """
    if len(data) == 0:
        raise ValueError("fit_truncated_dpm requires at least one observation.")
    if len(initial_components) == 0:
        raise ValueError("initial_components must not be empty.")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive.")
    k = len(initial_components)
    components = list(initial_components)
    estimators = _component_estimators(component_estimator, k)
    gamma = np.column_stack(
        [
            np.ones(k, dtype=np.float64),
            np.full(k, float(alpha), dtype=np.float64),
        ]
    )
    history: list[float] = []
    responsibilities = np.full((len(data), k), 1.0 / k, dtype=np.float64)

    for _ in range(max(1, int(max_its))):
        log_scores = _component_log_density_matrix(components, data)
        responsibilities = _softmax_rows(log_scores + expected_log_stick_weights(gamma)[None, :])
        counts = responsibilities.sum(axis=0)
        components = _estimate_components(data, components, estimators, responsibilities, counts)

        if sort_components and k > 1:
            order = np.argsort(-counts)
            components = [components[i] for i in order]
            responsibilities = responsibilities[:, order]
            counts = counts[order]

        gamma = _posterior_stick_gamma(counts, float(alpha))
        model = TruncatedDirichletProcessMixtureModel(components, alpha=alpha, gamma=gamma, name=name)
        objective = _variational_predictive_objective(model, data)
        history.append(objective)
        if len(history) > 1 and tol is not None and abs(history[-1] - history[-2]) < tol:
            break

    responsibilities = model.responsibilities(data, expected=True)
    return TruncatedDirichletProcessMixtureFitResult(model, responsibilities, history)


def _as_gamma(gamma: Any, expected_rows: int | None = None) -> np.ndarray:
    gam = np.asarray(gamma, dtype=np.float64)
    if gam.ndim != 2 or gam.shape[1] != 2:
        raise ValueError("gamma must have shape (num_components, 2).")
    if expected_rows is not None and gam.shape[0] != expected_rows:
        raise ValueError("gamma row count must match the number of components.")
    if np.any(~np.isfinite(gam)) or np.any(gam <= 0.0):
        raise ValueError("gamma entries must be finite and positive.")
    return gam


def _as_simplex(values: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != size:
        raise ValueError("%s must have length %d." % (name, size))
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("%s must contain finite non-negative values." % name)
    total = arr.sum()
    if total <= 0.0:
        raise ValueError("%s must have positive total mass." % name)
    return arr / total


def _component_estimators(
    component_estimator: ParameterEstimator | Sequence[ParameterEstimator], k: int
) -> list[ParameterEstimator]:
    if isinstance(component_estimator, (list, tuple)):
        if len(component_estimator) != k:
            raise ValueError("component_estimator sequence length must match initial_components.")
        return list(component_estimator)
    return [component_estimator for _ in range(k)]


def _component_log_density_matrix(
    components: Sequence[SequenceEncodableProbabilityDistribution], data: Sequence[Any]
) -> np.ndarray:
    rv = np.empty((len(data), len(components)), dtype=np.float64)
    for j, comp in enumerate(components):
        rv[:, j] = [comp.log_density(x) for x in data]
    return rv


def _estimate_components(
    data: Sequence[Any],
    old_components: Sequence[SequenceEncodableProbabilityDistribution],
    estimators: Sequence[ParameterEstimator],
    responsibilities: np.ndarray,
    counts: np.ndarray,
) -> list[SequenceEncodableProbabilityDistribution]:
    new_components = []
    for k, estimator in enumerate(estimators):
        if counts[k] <= 1.0e-12:
            new_components.append(old_components[k])
            continue
        acc = estimator.accumulator_factory().make()
        for x, w in zip(data, responsibilities[:, k]):
            if w != 0.0:
                acc.update(x, float(w), old_components[k])
        new_components.append(estimator.estimate(float(counts[k]), acc.value()))
    return new_components


def _posterior_stick_gamma(counts: np.ndarray, alpha: float) -> np.ndarray:
    gam = np.zeros((counts.shape[0], 2), dtype=np.float64)
    remaining = np.cumsum(counts[::-1])[::-1] - counts
    gam[:, 0] = 1.0 + counts
    gam[:, 1] = alpha + remaining
    return gam


def _variational_predictive_objective(model: TruncatedDirichletProcessMixtureModel, data: Sequence[Any]) -> float:
    log_scores = _component_log_density_matrix(model.components, data)
    weighted = log_scores + model.expected_log_weights[None, :]
    return float(np.sum([vec.log_sum(row) for row in weighted]))
