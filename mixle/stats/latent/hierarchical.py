"""Hierarchical (partial-pooling) normal model -- the plate / random-effects structure.

Grouped data (many drill sites, survey lines, lab batches, taxa) is best modelled with a *plate*: each
group has its own parameter drawn from a shared population distribution. Fitting each group alone (no
pooling) overfits small groups; pooling everything (complete pooling) ignores real between-group
variation. Partial pooling -- the hierarchical model -- learns the population spread and shrinks each
group's estimate toward the population mean by an amount set by its sample size.

``HierarchicalNormalDistribution`` is a first-class mixle leaf whose *observation is a whole group* (a
sequence of values): ``y[g,i] ~ N(theta[g], sigma^2)`` with ``theta[g] ~ N(mu, tau^2)``. Marginalizing the
latent group mean gives a closed-form group likelihood ``y_g ~ N(mu*1, sigma^2 I + tau^2 11^T)``, so it
follows the Distribution / Sampler / Estimator / Accumulator / DataEncoder contract: it fits through
``estimate(groups, dist.estimator())`` (empirical-Bayes EM over the latent means), scores groups with
``log_density`` / ``seq_log_density``, and exposes the per-group shrinkage posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass
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

__all__ = [
    "HierarchicalNormalDistribution",
    "HierarchicalNormalEstimator",
    "HierarchicalNormalFitDiagnostics",
]

_DEFAULT_MAX_GROUP_SIZES = 10_000
_MIN_VARIANCE = 1.0e-10


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive.")
    return result


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _finite_nonnegative_real(value: Any, label: str) -> float:
    result = _finite_real(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return result


def _finite_positive_real(value: Any, label: str) -> float:
    result = _finite_real(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def _validated_group(group: Any, label: str) -> np.ndarray:
    try:
        values = np.asarray(group, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric vector.") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional vector.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{label} must contain finite values.")
    return values


def _validated_encoded_groups(encoded: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 3:
        raise ValueError("hierarchical-normal encoding must contain size, mean, and within-SSE arrays.")
    try:
        n, ybar, sse = (np.asarray(value, dtype=np.float64) for value in encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("hierarchical-normal encoded statistics must be numeric arrays.") from exc
    if n.ndim != 1 or ybar.shape != n.shape or sse.shape != n.shape:
        raise ValueError("hierarchical-normal encoded statistics must be aligned one-dimensional arrays.")
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(ybar)) or np.any(~np.isfinite(sse)):
        raise ValueError("hierarchical-normal encoded statistics must be finite.")
    if np.any(n <= 0.0) or np.any(n != np.floor(n)):
        raise ValueError("hierarchical-normal group sizes must be exact positive integers.")
    if np.any(sse < 0.0):
        raise ValueError("hierarchical-normal within-group SSE must be non-negative.")
    return n.astype(np.int64), ybar, sse


@dataclass(frozen=True)
class HierarchicalNormalFitDiagnostics:
    """Machine-readable receipt for the empirical-Bayes EM fit."""

    converged: bool
    identifiable: bool
    iterations: int
    termination_reason: str
    objective_trace: tuple[float, ...]
    final_parameter_delta: float | None
    n_group_sizes: int
    total_group_weight: float
    total_observation_weight: float


class HierarchicalNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Two-level normal hierarchy over groups: ``y[g,i] ~ N(theta[g], sigma^2)``, ``theta[g] ~ N(mu, tau^2)``.

    Each observation is one group (a sequence of values). ``mu`` is the population mean, ``tau`` the
    between-group sd and ``sigma`` the within-group sd. The latent group mean is marginalized out, so a
    group's likelihood is the multivariate normal ``N(mu*1, sigma^2 I + tau^2 11^T)`` (computed in closed
    form). ``group_posterior`` / ``shrinkage`` give the partial-pooling estimates.
    """

    def __init__(
        self,
        mu: float,
        tau: float,
        sigma: float,
        name: str | None = None,
        keys: str | None = None,
        fit_diagnostics: HierarchicalNormalFitDiagnostics | None = None,
    ):
        self.mu = _finite_real(mu, "hierarchical-normal mu")
        self.tau = _finite_positive_real(tau, "hierarchical-normal tau")
        self.sigma = _finite_positive_real(sigma, "hierarchical-normal sigma")
        self.name = name
        self.keys = keys
        self.fit_diagnostics = fit_diagnostics

    def __str__(self) -> str:
        return "HierarchicalNormalDistribution(mu=%r, tau=%r, sigma=%r)" % (self.mu, self.tau, self.sigma)

    def _group_log_density(self, n: np.ndarray, ybar: np.ndarray, sse: np.ndarray) -> np.ndarray:
        """Marginal log-likelihood of groups from their sufficient stats ``(size, mean, within-SSE)``."""
        s2, t2 = self.sigma**2, self.tau**2
        dev = ybar - self.mu
        logdet = (n - 1.0) * np.log(s2) + np.log(s2 + n * t2)  # |sigma^2 I + tau^2 11^T|
        quad = (sse + n * dev**2) / s2 - (t2 / (s2 * (s2 + n * t2))) * (n * dev) ** 2
        return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)

    @staticmethod
    def _suff(group) -> tuple[float, float, float]:
        y = _validated_group(group, "hierarchical-normal group")
        n = float(len(y))
        ybar = float(y.mean())
        return n, ybar, float(np.sum((y - ybar) ** 2))

    def density(self, group) -> float:
        """Return the marginal density of one observed group."""
        return float(np.exp(self.log_density(group)))

    def log_density(self, group) -> float:
        """Marginal log-likelihood of one group (latent group mean integrated out)."""
        n, ybar, sse = self._suff(group)
        return float(self._group_log_density(np.array([n]), np.array([ybar]), np.array([sse]))[0])

    def seq_log_density(self, x) -> np.ndarray:
        """Return vectorized marginal log likelihoods for encoded groups."""
        n, ybar, sse = _validated_encoded_groups(x)
        return self._group_log_density(n, ybar, sse)

    def group_posterior(self, ybar: float, n: int) -> tuple[float, float]:
        """Posterior ``(mean, sd)`` of a group's true mean given its sample mean ``ybar`` and size ``n``.

        The shrinkage estimate ``mu + shrink*(ybar - mu)`` with ``shrink = tau^2/(tau^2 + sigma^2/n)``.
        """
        ybar = _finite_real(ybar, "hierarchical-normal group mean")
        n = _positive_integer(n, "hierarchical-normal group size")
        post_var = 1.0 / (n / self.sigma**2 + 1.0 / self.tau**2)
        post_mean = (n * ybar / self.sigma**2 + self.mu / self.tau**2) * post_var
        return float(post_mean), float(np.sqrt(post_var))

    def shrinkage(self, n: int) -> float:
        """The shrinkage weight for a size-``n`` group (0 = full pooling to ``mu``, 1 = its own mean)."""
        n = _positive_integer(n, "hierarchical-normal group size")
        return float(self.tau**2 / (self.tau**2 + self.sigma**2 / n))

    def sampler(self, seed: int | None = None) -> HierarchicalNormalSampler:
        """Return a sampler for grouped observations from this hierarchy."""
        return HierarchicalNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> HierarchicalNormalEstimator:
        """Return an empirical-Bayes EM estimator for the hierarchy."""
        return HierarchicalNormalEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> HierarchicalNormalDataEncoder:
        """Return the data encoder used by this distribution."""
        return HierarchicalNormalDataEncoder()


class HierarchicalNormalSampler(DistributionSampler):
    """Draw grouped observations from the two-level normal hierarchy."""

    def __init__(self, dist: HierarchicalNormalDistribution, seed: int | None = None):
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, sizes, *, batched: bool = True):
        """Draw group(s) of given size(s): an int draws one group; a sequence draws one group per entry."""
        d = self.dist
        if np.ndim(sizes) == 0:
            size = _positive_integer(sizes, "hierarchical-normal sample group size")
            theta = self.rng.normal(d.mu, d.tau)
            return self.rng.normal(theta, d.sigma, size)
        try:
            group_sizes = list(sizes)
        except TypeError as exc:
            raise TypeError("hierarchical-normal sample sizes must be an integer or iterable of integers.") from exc
        return [
            self.rng.normal(
                self.rng.normal(d.mu, d.tau),
                d.sigma,
                _positive_integer(size, f"hierarchical-normal sample group size {index}"),
            )
            for index, size in enumerate(group_sizes)
        ]


class HierarchicalNormalDataEncoder(DataSequenceEncoder):
    """Encode each group to its sufficient statistics ``(size, mean, within-group SSE)``."""

    def seq_encode(self, x):
        """Encode groups as arrays of size, mean, and within-group SSE."""
        try:
            groups = list(x)
        except TypeError as exc:
            raise TypeError("hierarchical-normal data must be an iterable of groups.") from exc
        if not groups:
            empty = np.empty(0, dtype=np.float64)
            return empty.copy(), empty.copy(), empty.copy()
        suff = np.array([HierarchicalNormalDistribution._suff(group) for group in groups], dtype=np.float64)
        return suff[:, 0], suff[:, 1], suff[:, 2]

    def row_count(self, x) -> int:
        """Return the number of encoded groups."""
        return len(_validated_encoded_groups(x)[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HierarchicalNormalDataEncoder)


def _canonical_bucket_statistics(suff_stat: Any) -> tuple[np.ndarray, ...]:
    """Validate current five-array statistics or lower the legacy three-array group layout."""
    if not isinstance(suff_stat, (tuple, list)):
        raise ValueError("hierarchical-normal sufficient statistics must be a tuple of arrays.")
    if len(suff_stat) == 3:
        n, ybar, sse = _validated_encoded_groups(suff_stat)
        weights = np.ones(len(n), dtype=np.float64)
        arrays = (n, weights, ybar.copy(), ybar**2, sse)
    elif len(suff_stat) == 5:
        try:
            arrays = tuple(np.asarray(value, dtype=np.float64) for value in suff_stat)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("hierarchical-normal sufficient statistics must be numeric arrays.") from exc
        sizes, weights, sum_y, sum_y2, sum_sse = arrays
        if any(array.ndim != 1 or array.shape != sizes.shape for array in arrays):
            raise ValueError("hierarchical-normal sufficient-statistic arrays must be aligned and one-dimensional.")
        if np.any(~np.isfinite(np.stack(arrays))):
            raise ValueError("hierarchical-normal sufficient statistics must be finite.")
        if np.any(sizes <= 0.0) or np.any(sizes != np.floor(sizes)):
            raise ValueError("hierarchical-normal sufficient-statistic sizes must be exact positive integers.")
        if np.any(weights < 0.0) or np.any(sum_y2 < 0.0) or np.any(sum_sse < 0.0):
            raise ValueError("hierarchical-normal weights, squared sums, and SSE must be non-negative.")
        zero_weight = weights == 0.0
        if np.any(sum_y[zero_weight] != 0.0) or np.any(sum_y2[zero_weight] != 0.0) or np.any(sum_sse[zero_weight] != 0.0):
            raise ValueError("zero-weight hierarchical-normal buckets must have zero weighted moments.")
        positive = weights > 0.0
        tolerance = 1.0e-10 * np.maximum(1.0, weights[positive] * sum_y2[positive])
        if np.any(sum_y[positive] ** 2 > weights[positive] * sum_y2[positive] + tolerance):
            raise ValueError("hierarchical-normal weighted moments violate non-negative variance.")
        arrays = (sizes.astype(np.int64), weights, sum_y, sum_y2, sum_sse)
    else:
        raise ValueError("hierarchical-normal sufficient statistics must contain three or five arrays.")

    buckets: dict[int, np.ndarray] = {}
    for size, weight, sum_y, sum_y2, sum_sse in zip(*arrays):
        key = int(size)
        if key not in buckets:
            buckets[key] = np.zeros(4, dtype=np.float64)
        buckets[key] += np.array([weight, sum_y, sum_y2, sum_sse], dtype=np.float64)
    ordered = sorted(buckets)
    if not ordered:
        empty = np.empty(0, dtype=np.float64)
        return empty.astype(np.int64), empty.copy(), empty.copy(), empty.copy(), empty.copy()
    values = np.stack([buckets[size] for size in ordered])
    return (
        np.asarray(ordered, dtype=np.int64),
        values[:, 0],
        values[:, 1],
        values[:, 2],
        values[:, 3],
    )


class HierarchicalNormalEstimator(ParameterEstimator):
    """Weighted empirical-Bayes EM over size-bucketed group sufficient statistics."""

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-9,
        max_group_sizes: int = _DEFAULT_MAX_GROUP_SIZES,
        name: str | None = None,
        keys: str | None = None,
    ):
        self.max_iter = _positive_integer(max_iter, "hierarchical-normal max_iter")
        self.tol = _finite_nonnegative_real(tol, "hierarchical-normal tol")
        self.max_group_sizes = _positive_integer(max_group_sizes, "hierarchical-normal max_group_sizes")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StatisticAccumulatorFactory:
        """Return an accumulator factory for bounded size-bucketed statistics."""
        maximum = self.max_group_sizes

        class _Factory(StatisticAccumulatorFactory):
            def make(self):
                return HierarchicalNormalAccumulator(maximum)

        return _Factory()

    @staticmethod
    def _objective(
        sizes: np.ndarray,
        group_weights: np.ndarray,
        sum_y: np.ndarray,
        sum_y2: np.ndarray,
        sum_sse: np.ndarray,
        mu: float,
        tau2: float,
        sigma2: float,
    ) -> float:
        centered2 = sum_y2 - 2.0 * mu * sum_y + group_weights * mu**2
        value = np.sum(
            group_weights
            * (
                sizes * np.log(2.0 * np.pi)
                + (sizes - 1.0) * np.log(sigma2)
                + np.log(sigma2 + sizes * tau2)
            )
            + sum_sse / sigma2
            + sizes * centered2 / (sigma2 + sizes * tau2)
        )
        return float(-0.5 * value)

    def estimate(self, nobs, suff_stat) -> HierarchicalNormalDistribution:
        """Fit ``mu``, ``tau``, and ``sigma`` with a weighted EM receipt."""
        sizes, group_weights, sum_y, sum_y2, sum_sse = _canonical_bucket_statistics(suff_stat)
        positive = group_weights > 0.0
        sizes = sizes[positive]
        group_weights = group_weights[positive]
        sum_y = sum_y[positive]
        sum_y2 = sum_y2[positive]
        sum_sse = sum_sse[positive]
        if len(sizes) > self.max_group_sizes:
            raise ValueError(
                f"hierarchical-normal statistics contain {len(sizes)} group sizes; "
                f"the configured limit is {self.max_group_sizes}."
            )
        if not len(sizes):
            receipt = HierarchicalNormalFitDiagnostics(
                converged=True,
                identifiable=False,
                iterations=0,
                termination_reason="no_data",
                objective_trace=(),
                final_parameter_delta=None,
                n_group_sizes=0,
                total_group_weight=0.0,
                total_observation_weight=0.0,
            )
            return HierarchicalNormalDistribution(
                0.0,
                1.0,
                1.0,
                name=self.name,
                keys=self.keys,
                fit_diagnostics=receipt,
            )

        total_group_weight = float(group_weights.sum())
        total_observation_weight = float(np.dot(group_weights, sizes))
        total_within_df = float(np.dot(group_weights, sizes - 1))
        mu = float(sum_y.sum() / total_group_weight)
        between = max(float(sum_y2.sum() / total_group_weight - mu**2), _MIN_VARIANCE)
        if total_within_df > 0.0:
            sigma2 = max(float(sum_sse.sum() / total_observation_weight), _MIN_VARIANCE)
            tau2 = max(between, _MIN_VARIANCE)
        else:
            sigma2 = max(0.5 * between, _MIN_VARIANCE)
            tau2 = max(0.5 * between, _MIN_VARIANCE)
        identifiable = total_group_weight > 1.0 and total_within_df > 0.0
        objective_trace = [
            self._objective(sizes, group_weights, sum_y, sum_y2, sum_sse, mu, tau2, sigma2)
        ]
        converged = False
        termination_reason = "max_iterations"
        final_delta = None
        iterations = 0

        for _ in range(self.max_iter):
            post_var = 1.0 / (sizes / sigma2 + 1.0 / tau2)
            slope = (sizes / sigma2) * post_var
            intercept = (mu / tau2) * post_var
            sum_post_mean = slope * sum_y + intercept * group_weights
            sum_post_mean2 = (
                slope**2 * sum_y2
                + 2.0 * slope * intercept * sum_y
                + intercept**2 * group_weights
            )
            mu_new = float(sum_post_mean.sum() / total_group_weight)
            tau_numerator = np.sum(
                sum_post_mean2
                - 2.0 * mu_new * sum_post_mean
                + group_weights * mu_new**2
                + group_weights * post_var
            )
            tau2_new = max(float(tau_numerator / total_group_weight), _MIN_VARIANCE)
            residual_slope = 1.0 - slope
            residual2 = (
                residual_slope**2 * sum_y2
                - 2.0 * residual_slope * intercept * sum_y
                + intercept**2 * group_weights
            )
            sigma_numerator = np.sum(sum_sse + sizes * residual2 + sizes * group_weights * post_var)
            sigma2_new = max(float(sigma_numerator / total_observation_weight), _MIN_VARIANCE)
            candidate_objective = self._objective(
                sizes,
                group_weights,
                sum_y,
                sum_y2,
                sum_sse,
                mu_new,
                tau2_new,
                sigma2_new,
            )
            if not np.isfinite(candidate_objective):
                termination_reason = "non_finite_update_rejected"
                break
            allowance = 1.0e-8 * max(1.0, abs(objective_trace[-1]))
            if candidate_objective < objective_trace[-1] - allowance:
                termination_reason = "non_monotone_update_rejected"
                break
            final_delta = abs(mu_new - mu) + abs(tau2_new - tau2) + abs(sigma2_new - sigma2)
            mu, tau2, sigma2 = mu_new, tau2_new, sigma2_new
            objective_trace.append(candidate_objective)
            iterations += 1
            if final_delta <= self.tol:
                converged = True
                termination_reason = "converged"
                break

        receipt = HierarchicalNormalFitDiagnostics(
            converged=converged,
            identifiable=identifiable,
            iterations=iterations,
            termination_reason=termination_reason,
            objective_trace=tuple(objective_trace),
            final_parameter_delta=final_delta,
            n_group_sizes=len(sizes),
            total_group_weight=total_group_weight,
            total_observation_weight=total_observation_weight,
        )
        return HierarchicalNormalDistribution(
            mu,
            np.sqrt(tau2),
            np.sqrt(sigma2),
            name=self.name,
            keys=self.keys,
            fit_diagnostics=receipt,
        )


class HierarchicalNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Bounded exact weighted statistics grouped by integer group size."""

    def __init__(self, max_group_sizes: int = _DEFAULT_MAX_GROUP_SIZES):
        self.max_group_sizes = _positive_integer(max_group_sizes, "hierarchical-normal max_group_sizes")
        self.buckets: dict[int, np.ndarray] = {}

    def _add(self, size: int, ybar: float, sse: float, weight: float) -> None:
        if weight == 0.0:
            return
        if size not in self.buckets and len(self.buckets) >= self.max_group_sizes:
            raise ValueError(
                f"hierarchical-normal distinct group-size limit {self.max_group_sizes} would be exceeded."
            )
        if size not in self.buckets:
            self.buckets[size] = np.zeros(4, dtype=np.float64)
        self.buckets[size] += weight * np.array([1.0, ybar, ybar**2, sse], dtype=np.float64)

    def update(self, x, weight, estimate):
        """Accumulate one weighted observed group."""
        weight = _finite_nonnegative_real(weight, "hierarchical-normal group weight")
        size, ybar, sse = HierarchicalNormalDistribution._suff(x)
        self._add(int(size), ybar, sse, weight)

    def initialize(self, x, weight, rng):
        """Initialize statistics from one weighted observed group."""
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):
        """Accumulate an encoded weighted batch of groups."""
        sizes, ybar, sse = _validated_encoded_groups(x)
        try:
            weights = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("hierarchical-normal group weights must be numeric.") from exc
        if weights.shape != sizes.shape:
            raise ValueError(f"hierarchical-normal group weights must have shape {sizes.shape}.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("hierarchical-normal group weights must be finite and non-negative.")
        new_sizes = {int(size) for size, weight in zip(sizes, weights) if weight > 0.0} - set(self.buckets)
        if len(self.buckets) + len(new_sizes) > self.max_group_sizes:
            raise ValueError(
                f"hierarchical-normal distinct group-size limit {self.max_group_sizes} would be exceeded."
            )
        for size, mean, within_sse, weight in zip(sizes, ybar, sse, weights):
            self._add(int(size), float(mean), float(within_sse), float(weight))

    def seq_initialize(self, x, weights, rng):
        """Initialize from encoded weighted group statistics."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):
        """Merge another size-bucketed sufficient statistic."""
        sizes, weights, sum_y, sum_y2, sum_sse = _canonical_bucket_statistics(suff_stat)
        new_sizes = {int(size) for size, weight in zip(sizes, weights) if weight > 0.0} - set(self.buckets)
        if len(self.buckets) + len(new_sizes) > self.max_group_sizes:
            raise ValueError(
                f"hierarchical-normal distinct group-size limit {self.max_group_sizes} would be exceeded."
            )
        for size, weight, weighted_y, weighted_y2, weighted_sse in zip(
            sizes,
            weights,
            sum_y,
            sum_y2,
            sum_sse,
        ):
            if weight == 0.0:
                continue
            size = int(size)
            if size not in self.buckets:
                self.buckets[size] = np.zeros(4, dtype=np.float64)
            self.buckets[size] += np.array(
                [weight, weighted_y, weighted_y2, weighted_sse],
                dtype=np.float64,
            )
        return self

    def value(self):
        """Return five aligned arrays over sorted distinct group sizes."""
        if not self.buckets:
            empty = np.empty(0, dtype=np.float64)
            return empty.astype(np.int64), empty.copy(), empty.copy(), empty.copy(), empty.copy()
        sizes = np.asarray(sorted(self.buckets), dtype=np.int64)
        values = np.stack([self.buckets[int(size)] for size in sizes])
        return sizes, values[:, 0].copy(), values[:, 1].copy(), values[:, 2].copy(), values[:, 3].copy()

    def from_value(self, x):
        """Replace this accumulator from validated size-bucketed statistics."""
        self.buckets = {}
        return self.combine(x)

    def acc_to_encoder(self):
        """Return the encoder used by this accumulator."""
        return HierarchicalNormalDataEncoder()
