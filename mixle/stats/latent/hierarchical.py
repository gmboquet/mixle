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

import numpy as np

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    StatisticAccumulatorFactory,
)

__all__ = ["HierarchicalNormalDistribution", "HierarchicalNormalEstimator"]


class HierarchicalNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Two-level normal hierarchy over groups: ``y[g,i] ~ N(theta[g], sigma^2)``, ``theta[g] ~ N(mu, tau^2)``.

    Each observation is one group (a sequence of values). ``mu`` is the population mean, ``tau`` the
    between-group sd and ``sigma`` the within-group sd. The latent group mean is marginalized out, so a
    group's likelihood is the multivariate normal ``N(mu*1, sigma^2 I + tau^2 11^T)`` (computed in closed
    form). ``group_posterior`` / ``shrinkage`` give the partial-pooling estimates.
    """

    def __init__(self, mu: float, tau: float, sigma: float, name: str | None = None, keys: str | None = None):
        self.mu, self.tau, self.sigma = float(mu), float(tau), float(sigma)
        self.name = name
        self.keys = keys

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
        y = np.asarray(group, dtype=float).ravel()
        n = float(len(y))
        ybar = float(y.mean()) if n else 0.0
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
        n, ybar, sse = x  # the encoder yields per-group (size, mean, within-SSE) arrays
        return self._group_log_density(n, ybar, sse)

    def group_posterior(self, ybar: float, n: int) -> tuple[float, float]:
        """Posterior ``(mean, sd)`` of a group's true mean given its sample mean ``ybar`` and size ``n``.

        The shrinkage estimate ``mu + shrink*(ybar - mu)`` with ``shrink = tau^2/(tau^2 + sigma^2/n)``.
        """
        post_var = 1.0 / (n / self.sigma**2 + 1.0 / self.tau**2)
        post_mean = (n * ybar / self.sigma**2 + self.mu / self.tau**2) * post_var
        return float(post_mean), float(np.sqrt(post_var))

    def shrinkage(self, n: int) -> float:
        """The shrinkage weight for a size-``n`` group (0 = full pooling to ``mu``, 1 = its own mean)."""
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
            theta = self.rng.normal(d.mu, d.tau)
            return self.rng.normal(theta, d.sigma, int(sizes))
        return [self.rng.normal(self.rng.normal(d.mu, d.tau), d.sigma, int(n)) for n in sizes]


class HierarchicalNormalDataEncoder:
    """Encode each group to its sufficient statistics ``(size, mean, within-group SSE)``."""

    def seq_encode(self, x):
        """Encode groups as arrays of size, mean, and within-group SSE."""
        suff = np.array([HierarchicalNormalDistribution._suff(g) for g in x], dtype=float)
        return suff[:, 0], suff[:, 1], suff[:, 2]


class HierarchicalNormalEstimator(ParameterEstimator):
    """Empirical-Bayes estimator: fits ``(mu, tau, sigma)`` by EM over the latent group means."""

    def __init__(self, max_iter: int = 500, tol: float = 1e-9, name: str | None = None, keys: str | None = None):
        self.max_iter = max_iter
        self.tol = tol
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StatisticAccumulatorFactory:
        """Return an accumulator factory for group sufficient statistics."""

        class _Factory(StatisticAccumulatorFactory):
            def make(self):
                return HierarchicalNormalAccumulator()

        return _Factory()

    def estimate(self, nobs, suff_stat) -> HierarchicalNormalDistribution:
        """Fit ``mu``, ``tau``, and ``sigma`` by EM over latent group means."""
        n, ybar, sse_within = (np.asarray(a, dtype=float) for a in suff_stat)
        total = n.sum()
        mu = ybar.mean()
        tau2 = max(ybar.var(), 1e-6)
        sigma2 = max(np.sum(sse_within) / max(total, 1.0), 1e-6)
        for _ in range(self.max_iter):
            prec = n / sigma2 + 1.0 / tau2  # posterior precision of each theta_g
            post_var = 1.0 / prec
            post_mean = (n * ybar / sigma2 + mu / tau2) * post_var  # posterior mean (shrinkage)
            mu_new = post_mean.mean()
            tau2_new = max(np.mean((post_mean - mu_new) ** 2 + post_var), 1e-10)
            sigma2_new = max((np.sum(sse_within + n * (ybar - post_mean) ** 2) + np.sum(n * post_var)) / total, 1e-10)
            done = abs(mu_new - mu) + abs(tau2_new - tau2) + abs(sigma2_new - sigma2) < self.tol
            mu, tau2, sigma2 = mu_new, tau2_new, sigma2_new
            if done:
                break
        return HierarchicalNormalDistribution(mu, np.sqrt(tau2), np.sqrt(sigma2), name=self.name, keys=self.keys)


class HierarchicalNormalAccumulator:
    """Collects per-group sufficient statistics ``(size, mean, within-SSE)`` for the EM fit."""

    def __init__(self):
        self.groups: list[tuple[float, float, float]] = []

    def update(self, x, weight, estimate):
        """Append sufficient statistics for one observed group."""
        self.groups.append(HierarchicalNormalDistribution._suff(x))

    def initialize(self, x, weight, rng):
        """Initialize statistics from one observed group."""
        self.groups.append(HierarchicalNormalDistribution._suff(x))

    def seq_update(self, x, weights, estimate):
        """Append encoded sufficient statistics for a batch of groups."""
        n, ybar, sse = x
        self.groups.extend(zip(n.tolist(), ybar.tolist(), sse.tolist()))

    def seq_initialize(self, x, weights, rng):
        """Initialize from encoded group sufficient statistics."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):
        """Merge another collection of group sufficient statistics."""
        self.groups.extend(zip(*(a.tolist() for a in suff_stat)))
        return self

    def value(self):
        """Return grouped sufficient statistics as three aligned arrays."""
        arr = np.array(self.groups, dtype=float).reshape(-1, 3)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def from_value(self, x):
        """Replace this accumulator from grouped sufficient-statistic arrays."""
        self.groups = list(zip(*(np.asarray(a, dtype=float).tolist() for a in x)))
        return self

    def acc_to_encoder(self):
        """Return the encoder used by this accumulator."""
        return HierarchicalNormalDataEncoder()
