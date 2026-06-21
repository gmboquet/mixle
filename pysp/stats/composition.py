"""Compositional data analysis: Aitchison logratio transforms and the logratio-normal distribution.

Geochemistry (and many earth-science) measurements are *compositions* -- vectors of non-negative parts
that sum to a constant (element abundances, mineral fractions, isotope splits). Ordinary statistics on
them is wrong: they live on the simplex, not in real space. Aitchison's logratio transforms (clr/ilr)
map the simplex isometrically to real coordinates where standard multivariate-Gaussian modelling applies;
the isometric logratio (ilr) uses an orthonormal basis so distances/covariances are preserved. Part of
the earth-science/multiphysics/UQ plan (Phase 6, modality breadth).
"""

from __future__ import annotations

import numpy as np

__all__ = ["closure", "clr", "clr_inv", "ilr", "ilr_inv", "ilr_basis", "AitchisonNormal"]


def closure(x: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Normalize each row to sum to ``total`` (project onto the simplex)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return total * x / x.sum(axis=1, keepdims=True)


def clr(x: np.ndarray) -> np.ndarray:
    """Centered logratio: ``clr(x)_i = log(x_i) - mean_j log(x_j)``. Maps the simplex to the zero-sum
    hyperplane in ``R^D`` (the parts stay labelled, but the result is singular -- use ilr for modelling)."""
    lx = np.log(np.atleast_2d(np.asarray(x, dtype=float)))
    return lx - lx.mean(axis=1, keepdims=True)


def clr_inv(y: np.ndarray) -> np.ndarray:
    """Inverse clr (softmax onto the simplex)."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    e = np.exp(y - y.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ilr_basis(d: int) -> np.ndarray:
    """A ``(D, D-1)`` orthonormal contrast basis (Helmert) for the isometric logratio of ``D`` parts."""
    v = np.zeros((d, d - 1))
    for i in range(d - 1):
        n = i + 1
        v[:n, i] = 1.0 / n
        v[n, i] = -1.0
        v[:, i] *= np.sqrt(n / (n + 1.0))
    return v


def ilr(x: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Isometric logratio: ``D``-part composition -> ``D-1`` real coordinates (orthonormal, so Euclidean
    distance in ilr space equals Aitchison distance on the simplex)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    v = ilr_basis(x.shape[1]) if basis is None else basis
    return clr(x) @ v


def ilr_inv(y: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Inverse isometric logratio: ``D-1`` real coordinates -> ``D``-part composition on the simplex."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    v = ilr_basis(y.shape[1] + 1) if basis is None else basis
    return clr_inv(y @ v.T)


class AitchisonNormal:
    """A logratio-normal distribution on the simplex: ``ilr(x) ~ N(mean, cov)``.

    The natural Gaussian for compositions -- model in the orthonormal ilr coordinates, interpret on the
    simplex. ``log_density`` is the Gaussian density in ilr space (the compositional density w.r.t. the
    Aitchison measure); fitting is the Gaussian MLE on the ilr-transformed data; sampling draws in ilr
    space and maps back. ``mean`` (length ``D-1``) and ``cov`` (``(D-1, D-1)``) are the ilr parameters of
    a ``D``-part composition.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.atleast_2d(np.asarray(cov, dtype=float))
        self.n_parts = len(self.mean) + 1
        self._chol = np.linalg.cholesky(self.cov)
        self._logdet = 2.0 * np.sum(np.log(np.diag(self._chol)))
        self._inv = np.linalg.inv(self.cov)

    def log_density(self, x: np.ndarray) -> np.ndarray | float:
        """Log-density at composition(s) ``x`` (the ilr-space Gaussian log-density)."""
        y = ilr(x)
        diff = y - self.mean
        maha = np.einsum("ij,jk,ik->i", diff, self._inv, diff)
        k = len(self.mean)
        ld = -0.5 * (maha + self._logdet + k * np.log(2.0 * np.pi))
        return float(ld[0]) if np.ndim(x) == 1 else ld

    def sampler(self, seed: int | None = None) -> AitchisonNormalSampler:
        return AitchisonNormalSampler(self, seed)

    @classmethod
    def fit(cls, x: np.ndarray) -> AitchisonNormal:
        """Maximum-likelihood fit: the Gaussian MLE in ilr coordinates."""
        y = ilr(np.atleast_2d(np.asarray(x, dtype=float)))
        return cls(y.mean(axis=0), np.cov(y.T, bias=True) if y.shape[1] > 1 else np.array([[np.var(y)]]))

    def mean_composition(self) -> np.ndarray:
        """The center of the distribution as a composition (the ilr-mean mapped back to the simplex)."""
        return ilr_inv(self.mean)[0]


class AitchisonNormalSampler:
    def __init__(self, dist: AitchisonNormal, seed: int | None = None):
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None) -> np.ndarray:
        n = 1 if size is None else size
        z = self.rng.standard_normal((n, len(self.dist.mean)))
        y = self.dist.mean[None, :] + z @ self.dist._chol.T
        comp = ilr_inv(y)
        return comp[0] if size is None else comp
