"""Max-stable processes for spatial extremes: the Smith (Gaussian-storm) model.

Block maxima of a spatial field (annual flood peaks, peak seismic amplitude, extreme porosity) are
spatially *dependent*, and that dependence has its own limit law -- a max-stable process -- which the
ordinary GEV/GPD (treated independently per site) misses. The Smith model is the canonical one:
``Z(s) = max_i xi_i * phi_Sigma(s - U_i)`` over a Poisson storm process, giving unit-Frechet margins and a
closed-form pairwise dependence. The extremal coefficient ``theta(h) in [1, 2]`` summarizes it: 1 = full
dependence (extremes always co-occur), 2 = independence. Part of the earth-science/UQ work (Phase 6).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["SmithMaxStable"]


class SmithMaxStable:
    """The Smith max-stable process with Gaussian storm-profile covariance ``sigma`` (d x d, SPD).

    Margins are unit Frechet; spatial dependence is set by ``sigma`` (larger -> longer-range extremal
    dependence). ``extremal_coefficient(h)`` and the bivariate CDF have closed forms; ``sampler`` draws
    fields on a set of locations.
    """

    def __init__(self, sigma: np.ndarray):
        self.sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
        self._inv = np.linalg.inv(self.sigma)

    def _mahalanobis(self, h: np.ndarray) -> float:
        h = np.atleast_1d(np.asarray(h, dtype=float))
        return float(np.sqrt(h @ self._inv @ h))

    def extremal_coefficient(self, h: np.ndarray) -> float:
        """``theta(h) = 2 * Phi(a/2)`` with ``a`` the Mahalanobis lag length -- 1 at h=0 (full dependence)
        rising to 2 as the lag grows (independence)."""
        return 2.0 * norm.cdf(self._mahalanobis(h) / 2.0)

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        """``P(Z(s) <= z1, Z(s+h) <= z2) = exp(-V(z1, z2))`` -- the Smith bivariate distribution."""
        a = self._mahalanobis(h)
        if a < 1e-12:
            return float(np.exp(-1.0 / min(z1, z2)))  # fully dependent limit
        v = (1.0 / z1) * norm.cdf(a / 2.0 + np.log(z2 / z1) / a) + (1.0 / z2) * norm.cdf(a / 2.0 + np.log(z1 / z2) / a)
        return float(np.exp(-v))

    def sampler(self, locations: np.ndarray, seed: int | None = None) -> SmithMaxStableSampler:
        return SmithMaxStableSampler(self, np.atleast_2d(np.asarray(locations, dtype=float)), seed)

    @classmethod
    def fit_isotropic(cls, locations: np.ndarray, fields: np.ndarray) -> SmithMaxStable:
        """Fit an isotropic Smith model (``sigma = s^2 I``) by matching the binned extremal coefficient
        (estimated from the madogram) to ``2 Phi(|h| / (2 s))``."""
        from scipy.optimize import minimize_scalar

        loc = np.atleast_2d(np.asarray(locations, dtype=float))
        z = np.asarray(fields, dtype=float)  # (n_replicates, n_locations), unit-Frechet-ish
        d = loc.shape[1]
        # empirical extremal coefficient per pair via the F-madogram: theta = (1 + nu) / (1 - nu) with
        # nu = E|F(Z1) - F(Z2)| on uniform margins (nu = 1/3 at independence -> theta = 2).
        u = np.argsort(np.argsort(z, axis=0), axis=0) / (z.shape[0] + 1.0)  # rank-transform to uniform margins
        pairs = [(i, j) for i in range(len(loc)) for j in range(i + 1, len(loc))]
        lags = np.array([np.linalg.norm(loc[i] - loc[j]) for i, j in pairs])
        nu = np.array([np.mean(np.abs(u[:, i] - u[:, j])) for i, j in pairs])
        theta_emp = np.clip((1 + nu) / (1 - nu + 1e-9), 1.0, 2.0)

        def obj(s):
            theta_model = 2.0 * norm.cdf(lags / (2.0 * max(s, 1e-3)))
            return np.mean((theta_model - theta_emp) ** 2)

        s = minimize_scalar(obj, bounds=(0.05, 10 * (lags.max() + 1e-9)), method="bounded").x
        return cls(s**2 * np.eye(d))


class SmithMaxStableSampler:
    def __init__(self, dist: SmithMaxStable, locations: np.ndarray, seed: int | None = None):
        self.dist = dist
        self.loc = locations
        self.rng = np.random.RandomState(seed)
        self._chol = np.linalg.cholesky(dist.sigma)
        self._logdet = 2.0 * np.sum(np.log(np.diag(self._chol)))

    def _storm(self, u: np.ndarray) -> np.ndarray:
        """Gaussian storm profile phi_Sigma(loc - u) at every location."""
        diff = self.loc - u
        sol = np.linalg.solve(self._chol, diff.T)
        d = self.loc.shape[1]
        return np.exp(-0.5 * np.sum(sol**2, axis=0) - 0.5 * self._logdet - 0.5 * d * np.log(2 * np.pi))

    def sample(self, size: int | None = None, *, n_storms: int = 200) -> np.ndarray:
        """Draw max-stable field(s) at the locations (unit Frechet margins) via the Schlather algorithm."""
        n = 1 if size is None else size
        lo, hi = (
            self.loc.min(0) - 5 * np.sqrt(np.diag(self.dist.sigma)),
            self.loc.max(0) + 5 * np.sqrt(np.diag(self.dist.sigma)),
        )
        out = np.zeros((n, len(self.loc)))
        for r in range(n):
            z = np.zeros(len(self.loc))
            gamma = 0.0
            for _ in range(n_storms):
                gamma += self.rng.exponential()  # Poisson arrival of storm intensity 1/gamma
                xi = 1.0 / gamma
                u = lo + self.rng.uniform(size=self.loc.shape[1]) * (hi - lo)
                vol = np.prod(hi - lo)
                z = np.maximum(z, xi * vol * self._storm(u))
            out[r] = z
        return out[0] if size is None else out
