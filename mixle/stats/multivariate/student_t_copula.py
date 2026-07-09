"""Student-t copula: an elliptical copula like the Gaussian but with heavy, SYMMETRIC tail dependence.

The Gaussian copula is tail-INDEPENDENT: extreme joint moves are asymptotically as rare as independence would
predict -- the flaw widely blamed for underpricing joint defaults in the 2008 crisis. The Student-t copula
keeps the Gaussian's correlation matrix ``R`` but pulls each uniform back through a ``t_nu`` quantile instead
of a normal one, giving symmetric UPPER- and lower-tail dependence controlled by the degrees of freedom
``nu`` (small ``nu`` = heavy joint tails; ``nu -> inf`` recovers the Gaussian copula). Its density on
``(0,1)^d`` is the multivariate-``t`` density at the ``t``-scores ``z_i = t_nu^{-1}(u_i)`` divided by the
product of univariate-``t`` densities:

    c(u) = f_mvt(z; R, nu) / prod_i f_t(z_i; nu).

Fit by the elliptical inversion estimator (``R`` = the correlation of the ``t``-scores, consistent since a
multivariate ``t`` has correlation ``R``) with ``nu`` chosen by a profile likelihood over a small grid.

Reference: Demarta & McNeil, "The t Copula and Related Copulas" (Int. Stat. Review, 2005).
"""

from __future__ import annotations

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln
from scipy.stats import t as _t

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulatorFactory,
    UScoreEncoder,
)

_CLIP = 1.0e-12
_NU_GRID = (2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0)  # profile-likelihood grid for the tail-heaviness


class StudentTCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Student-t copula on ``(0,1)^d`` with correlation ``corr`` and degrees of freedom ``df``."""

    def __init__(self, corr: np.ndarray, df: float, name: str | None = None, keys: str | None = None) -> None:
        r = np.asarray(corr, dtype=np.float64)
        if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] < 2:
            raise ValueError("corr must be a square correlation matrix of size >= 2")
        sign, logdet = np.linalg.slogdet(r)
        if sign <= 0:
            raise ValueError("corr must be positive definite")
        self.corr = r
        self.dim = r.shape[0]
        self.df = float(df)
        self.name = name
        self.keys = keys
        self._logdet = float(logdet)
        self._inv = np.linalg.inv(r)

    def __str__(self) -> str:
        return "StudentTCopulaDistribution(dim=%d, df=%.4g)" % (self.dim, self.df)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64), _CLIP, 1.0 - _CLIP)
        nu, d = self.df, self.dim
        z = _t.ppf(u, nu)  # t-scores (n, d)
        quad = np.einsum("ni,ij,nj->n", z, self._inv, z)  # z^T R^{-1} z
        # log f_mvt(z; R, nu) - sum_i log f_t(z_i; nu); the (nu*pi) constants cancel between the two.
        log_mvt_kernel = -0.5 * self._logdet - 0.5 * (nu + d) * np.log1p(quad / nu)
        const = gammaln(0.5 * (nu + d)) + (d - 1) * gammaln(0.5 * nu) - d * gammaln(0.5 * (nu + 1.0))
        sum_uni = 0.5 * (nu + 1.0) * np.sum(np.log1p(z * z / nu), axis=1)
        return const + log_mvt_kernel + sum_uni

    def sampler(self, seed: int | None = None) -> StudentTCopulaSampler:
        return StudentTCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> StudentTCopulaEstimator:
        return StudentTCopulaEstimator(self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class StudentTCopulaSampler(DistributionSampler):
    """Draw ``z ~ mvt(R, nu)`` (Gaussian scaled by a chi-square) then map through the univariate ``t_nu`` CDF."""

    def __init__(self, dist: StudentTCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None) -> np.ndarray:
        n = 1 if size is None else int(size)
        nu, d = self.dist.df, self.dist.dim
        g = self.rng.multivariate_normal(np.zeros(d), self.dist.corr, size=n)
        chi = self.rng.chisquare(nu, size=(n, 1))
        z = g / np.sqrt(chi / nu)  # multivariate-t with dispersion R
        u = np.clip(_t.cdf(z, nu), _CLIP, 1.0 - _CLIP)
        return u[0] if size is None else u


class StudentTCopulaEstimator(ParameterEstimator):
    """Inversion for ``R`` (correlation of the ``t``-scores) + profile likelihood over ``nu`` on a grid."""

    def __init__(self, dim: int, min_eig: float = 1.0e-8, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.min_eig = min_eig
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def _corr_from_scores(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        wsum = float(w.sum())
        mean = (z * w[:, None]).sum(axis=0) / wsum
        zc = z - mean
        cov = (zc * w[:, None]).T @ zc / wsum
        dd = np.sqrt(np.clip(np.diag(cov), 1.0e-12, None))
        corr = cov / np.outer(dd, dd)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        eig, vec = np.linalg.eigh(corr)  # project to a valid (PD) correlation matrix if needed
        if eig.min() < self.min_eig:
            corr = vec @ np.diag(np.clip(eig, self.min_eig, None)) @ vec.T
            d2 = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d2, d2)
            np.fill_diagonal(corr, 1.0)
        return corr

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> StudentTCopulaDistribution:
        u, w = suff_stat
        if len(u) < 2:
            return StudentTCopulaDistribution(np.eye(self.dim), _NU_GRID[2], name=self.name, keys=self.keys)
        u = np.clip(np.asarray(u, dtype=np.float64), _CLIP, 1.0 - _CLIP)
        w = np.asarray(w, dtype=np.float64)
        best = None
        for nu in _NU_GRID:  # profile nu: refit R at each nu (t-scores depend on nu), keep the best likelihood
            corr = self._corr_from_scores(_t.ppf(u, nu), w)
            cand = StudentTCopulaDistribution(corr, nu, name=self.name, keys=self.keys)
            ll = float(np.dot(w, cand.seq_log_density(u)))
            if best is None or ll > best[0]:
                best = (ll, cand)
        return best[1]
