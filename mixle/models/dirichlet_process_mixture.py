"""Dirichlet-process mixture experiment helpers.

This module keeps nonparametric-mixture logic in the model layer.  It exposes
small stick-breaking utilities and a dependency-free truncated variational
mixture loop over ordinary ``mixle.stats`` component estimators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import mixle.utils.vector as vec
from mixle.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution
from mixle.utils.special import digamma, gammaln
from mixle.utils.special import softmax_rows as _softmax_rows

_EPS = 1.0e-300


@dataclass
class TruncatedDirichletProcessMixtureFitResult:
    """Fitted truncated DPM plus variational responsibilities and history."""

    model: TruncatedDirichletProcessMixtureModel
    responsibilities: np.ndarray
    history: list[float]
    objective_name: str = "truncated_dp_elbo"


@dataclass
class _ComponentSlot:
    """Identity-preserving component family and its latest effective count."""

    component: SequenceEncodableProbabilityDistribution
    estimator: ParameterEstimator
    count: float = 0.0
    stick_gamma: tuple[float, float] = (1.0, 1.0)


class TruncatedDirichletProcessMixtureModel:
    """Truncated stick-breaking mixture over existing mixle component models."""

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
        alpha = _positive_finite(alpha, "alpha")
        self.components = list(components)
        self.num_components = len(self.components)
        self.alpha = alpha
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
        scores = np.asarray([d.log_density(x) for d in self.components], dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise ValueError("every truncated-DPM component log density must be finite")
        return scores

    def log_density(self, x: Any) -> float:
        """Return the finite-truncation mixture log density for one observation."""
        return vec.log_sum(self.component_log_density(x) + self.log_weights)

    def density(self, x: Any) -> float:
        """Return the finite-truncation mixture density for one observation."""
        return float(np.exp(self.log_density(x)))

    def responsibilities(self, data: Sequence[Any], expected: bool = True) -> np.ndarray:
        """Return posterior component probabilities for observations."""
        if len(data) == 0:
            raise ValueError("responsibilities require at least one observation")
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
    alpha = _positive_finite(alpha, "alpha")
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

    The component M-steps are delegated to ordinary ``mixle.stats`` estimators.
    Components are point-estimated parameters; ``history`` is the complete
    truncated-stick ELBO over assignments and Beta sticks conditional on those
    point estimates (expected likelihood + assignment entropy - stick KL).
    """
    if len(data) == 0:
        raise ValueError("fit_truncated_dpm requires at least one observation.")
    if len(initial_components) == 0:
        raise ValueError("initial_components must not be empty.")
    alpha = _positive_finite(alpha, "alpha")
    max_its = _positive_int(max_its, "max_its")
    if tol is not None:
        if isinstance(tol, (bool, np.bool_)):
            raise TypeError("tol must be a finite non-negative scalar or None")
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative or None")
    if not isinstance(sort_components, (bool, np.bool_)):
        raise TypeError("sort_components must be a boolean")
    k = len(initial_components)
    estimators = _component_estimators(component_estimator, k)
    gamma = np.column_stack(
        [
            np.ones(k, dtype=np.float64),
            np.full(k, float(alpha), dtype=np.float64),
        ]
    )
    slots = [
        _ComponentSlot(component, estimator, stick_gamma=tuple(gamma[index]))
        for index, (component, estimator) in enumerate(zip(initial_components, estimators))
    ]
    history: list[float] = []
    responsibilities = np.full((len(data), k), 1.0 / k, dtype=np.float64)

    for _ in range(max_its):
        components = [slot.component for slot in slots]
        log_scores = _component_log_density_matrix(components, data)
        responsibilities = _softmax_rows(log_scores + expected_log_stick_weights(gamma)[None, :])
        counts = responsibilities.sum(axis=0)
        fitted_components = _estimate_components(
            data,
            components,
            [slot.estimator for slot in slots],
            responsibilities,
            counts,
        )
        slots = [
            _ComponentSlot(component, slot.estimator, float(count), slot.stick_gamma)
            for component, slot, count in zip(fitted_components, slots, counts)
        ]

        # Recompute q(z) against the newly fitted component likelihoods before updating q(v).
        updated_scores = _component_log_density_matrix([slot.component for slot in slots], data)
        responsibilities = _softmax_rows(updated_scores + expected_log_stick_weights(gamma)[None, :])
        counts = responsibilities.sum(axis=0)
        for slot, count in zip(slots, counts):
            slot.count = float(count)

        if sort_components and k > 1:
            order = np.argsort(-counts)
            slots = [slots[i] for i in order]
            responsibilities = responsibilities[:, order]
            counts = counts[order]

        gamma = _posterior_stick_gamma(counts, float(alpha))
        for slot, stick_gamma in zip(slots, gamma):
            slot.stick_gamma = tuple(stick_gamma)
        gamma = np.asarray([slot.stick_gamma for slot in slots], dtype=np.float64)
        components = [slot.component for slot in slots]
        model = TruncatedDirichletProcessMixtureModel(components, alpha=alpha, gamma=gamma, name=name)
        objective = _truncated_dpm_elbo(components, data, responsibilities, gamma, float(alpha))
        history.append(objective)
        if len(history) > 1 and tol is not None and abs(history[-1] - history[-2]) < tol:
            break

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
        estimators = list(component_estimator)
    else:
        estimators = [component_estimator for _ in range(k)]
    for index, estimator in enumerate(estimators):
        if not callable(getattr(estimator, "accumulator_factory", None)) or not callable(
            getattr(estimator, "estimate", None)
        ):
            raise TypeError(f"component estimator {index} does not implement the estimation contract")
    return estimators


def _component_log_density_matrix(
    components: Sequence[SequenceEncodableProbabilityDistribution], data: Sequence[Any]
) -> np.ndarray:
    if len(components) == 0 or len(data) == 0:
        raise ValueError("component log-density evaluation requires components and observations")
    rv = np.empty((len(data), len(components)), dtype=np.float64)
    for j, comp in enumerate(components):
        if not callable(getattr(comp, "log_density", None)):
            raise TypeError(f"component {j} does not implement log_density")
        try:
            rv[:, j] = [comp.log_density(x) for x in data]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"component {j} could not score the supplied observations") from exc
    if not np.all(np.isfinite(rv)):
        bad = np.argwhere(~np.isfinite(rv))[0]
        raise ValueError(
            f"component log density must be finite; observation {int(bad[0])}, component {int(bad[1])} was not"
        )
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


def _truncated_dpm_elbo(
    components: Sequence[SequenceEncodableProbabilityDistribution],
    data: Sequence[Any],
    responsibilities: Any,
    gamma: Any,
    alpha: float,
) -> float:
    """Return the variational lower bound conditional on point-estimated component parameters."""
    log_scores = _component_log_density_matrix(components, data)
    r = np.asarray(responsibilities, dtype=np.float64)
    if r.shape != log_scores.shape or not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("responsibilities must be a finite non-negative matrix matching component scores")
    if not np.allclose(r.sum(axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("every responsibility row must sum to one")
    gam = _as_gamma(gamma, len(components))
    expected_log_weights = expected_log_stick_weights(gam)
    entropy = -float(np.sum(np.where(r > 0.0, r * np.log(np.clip(r, _EPS, 1.0)), 0.0)))
    expected_complete_log_likelihood = float(
        np.sum(r * (log_scores + expected_log_weights[None, :]))
    )
    stick_kl = sum(_beta_kl(gam[index, 0], gam[index, 1], 1.0, alpha) for index in range(len(gam) - 1))
    elbo = expected_complete_log_likelihood + entropy - stick_kl
    if not np.isfinite(elbo):
        raise RuntimeError("truncated-DPM ELBO became non-finite")
    return float(elbo)


def _beta_kl(a: float, b: float, prior_a: float, prior_b: float) -> float:
    log_beta_prior = gammaln(prior_a) + gammaln(prior_b) - gammaln(prior_a + prior_b)
    log_beta_q = gammaln(a) + gammaln(b) - gammaln(a + b)
    value = (
        log_beta_prior
        - log_beta_q
        + (a - prior_a) * digamma(a)
        + (b - prior_b) * digamma(b)
        + (prior_a + prior_b - a - b) * digamma(a + b)
    )
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result
