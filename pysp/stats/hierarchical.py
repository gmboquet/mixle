"""Hierarchical (partial-pooling) models -- the plate / random-effects structure.

Grouped data (many drill sites, survey lines, lab batches, taxa) is best modelled with a *plate*: each
group has its own parameter drawn from a shared population distribution. Fitting each group alone
(no pooling) overfits small groups; pooling everything (complete pooling) ignores real between-group
variation. Partial pooling -- the hierarchical model -- learns the population spread and shrinks each
group's estimate toward the population mean by an amount set by its sample size. Part of the
earth-science/multiphysics/UQ plan (Phase 7, composition expressiveness: plates / hierarchy).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["HierarchicalNormal"]


class HierarchicalNormal:
    """Two-level normal hierarchy: ``y[g,i] ~ N(theta[g], sigma^2)`` with ``theta[g] ~ N(mu, tau^2)``.

    Fits the population mean ``mu``, between-group sd ``tau`` and within-group sd ``sigma`` by EM
    (marginal-likelihood / empirical Bayes). The per-group posteriors are the *shrinkage* estimates: a
    group is pulled toward ``mu`` by ``tau^2 / (tau^2 + sigma^2/n_g)`` of the way from ``mu`` to its own
    mean -- small / noisy groups shrink more, large groups barely.
    """

    def __init__(self, mu: float, tau: float, sigma: float):
        self.mu, self.tau, self.sigma = float(mu), float(tau), float(sigma)

    @classmethod
    def fit(cls, groups: Sequence[Sequence[float]], *, max_iter: int = 500, tol: float = 1e-9) -> HierarchicalNormal:
        """Fit by EM over the latent group means ``theta[g]``. ``groups`` is a list of per-group samples."""
        gs = [np.asarray(g, dtype=float).ravel() for g in groups if len(g) > 0]
        n = np.array([len(g) for g in gs], dtype=float)
        ybar = np.array([g.mean() for g in gs])
        sse_within = np.array([np.sum((g - g.mean()) ** 2) for g in gs])
        total = n.sum()
        mu = ybar.mean()
        tau2 = max(ybar.var(), 1e-6)
        sigma2 = max(np.sum(sse_within) / max(total, 1.0), 1e-6)
        for _ in range(max_iter):
            prec = n / sigma2 + 1.0 / tau2  # posterior precision of each theta_g
            post_var = 1.0 / prec
            post_mean = (n * ybar / sigma2 + mu / tau2) * post_var  # posterior mean (shrinkage)
            mu_new = post_mean.mean()
            tau2_new = np.mean((post_mean - mu_new) ** 2 + post_var)  # E[(theta-mu)^2]
            # within-group variance: residual SSE about the posterior group means + posterior uncertainty
            sigma2_new = (np.sum(sse_within + n * (ybar - post_mean) ** 2) + np.sum(n * post_var)) / total
            tau2_new, sigma2_new = max(tau2_new, 1e-10), max(sigma2_new, 1e-10)
            if abs(mu_new - mu) + abs(tau2_new - tau2) + abs(sigma2_new - sigma2) < tol:
                mu, tau2, sigma2 = mu_new, tau2_new, sigma2_new
                break
            mu, tau2, sigma2 = mu_new, tau2_new, sigma2_new
        self = cls(mu, np.sqrt(tau2), np.sqrt(sigma2))
        self._ybar, self._n = ybar, n
        return self

    def group_posterior(self, ybar: float, n: int) -> tuple[float, float]:
        """Posterior ``(mean, sd)`` for a group's true mean given its sample mean ``ybar`` and size ``n``.

        The shrinkage estimate ``mu + shrink*(ybar - mu)`` with ``shrink = tau^2/(tau^2 + sigma^2/n)``.
        """
        post_var = 1.0 / (n / self.sigma**2 + 1.0 / self.tau**2)
        post_mean = (n * ybar / self.sigma**2 + self.mu / self.tau**2) * post_var
        return float(post_mean), float(np.sqrt(post_var))

    def shrinkage(self, n: int) -> float:
        """The shrinkage weight for a size-``n`` group (0 = full pooling to ``mu``, 1 = its own mean)."""
        return float(self.tau**2 / (self.tau**2 + self.sigma**2 / n))
