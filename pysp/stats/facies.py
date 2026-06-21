"""Latent facies field: a spatial Hidden Markov Random Field for subsurface composition and structure.

The subsurface is a discrete set of rock types / facies (sand, shale, carbonate, reservoir, ...), each
with its own continuous physical properties; geology is spatially coherent, so neighbouring locations tend
to share a facies. This models a grid of locations with a *latent discrete facies field* ``z`` carrying a
Potts (Ising-style) smoothness prior, where each facies ``k`` emits the observed property vector from its
own multivariate Gaussian. Stacking heterogeneous measurements into that vector -- seismic-derived Vp /
density / impedance, hyperspectral mineral indices, log readings -- fuses the modalities, and fitting
returns the posterior over **composition** (per-location facies probabilities) and **structure** (the MAP
facies map), plus each facies' property distribution.

This is the latent-switching graphical model the composition-expressiveness work pointed at: a latent
label switches which emission generates the data, with spatial coupling. Inference is mean-field
variational EM (the Potts partition function is intractable); the mean-field responsibilities are the
calibrated composition posterior.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LatentFaciesField"]


def _neighbor_offsets(ndim: int) -> list[tuple[int, ...]]:
    """First-order (von Neumann) neighbour offsets on an ndim grid: +/-1 along each axis."""
    offs = []
    for ax in range(ndim):
        for d in (-1, 1):
            o = [0] * ndim
            o[ax] = d
            offs.append(tuple(o))
    return offs


class LatentFaciesField:
    """Spatial mixture of multivariate Gaussians with a Potts smoothness prior over the latent labels.

    Args:
        shape: grid shape, e.g. ``(nx, ny)`` or ``(nx, ny, nz)`` -- sets the neighbour structure.
        n_facies: number of facies (latent classes).
        beta: Potts coupling (>=0); larger = smoother facies (more spatial coherence). 0 = independent GMM.
    """

    def __init__(self, shape, n_facies: int, beta: float = 1.0):
        self.shape = tuple(int(s) for s in np.atleast_1d(shape))
        self.k = int(n_facies)
        self.beta = float(beta)
        self.n = int(np.prod(self.shape))
        self._neighbors = self._build_neighbors()

    def _build_neighbors(self) -> list[np.ndarray]:
        """For each node, the flat indices of its in-grid grid neighbours."""
        idx = np.arange(self.n).reshape(self.shape)
        neigh: list[list[int]] = [[] for _ in range(self.n)]
        for off in _neighbor_offsets(len(self.shape)):
            shifted = np.roll(idx, shift=[-o for o in off], axis=tuple(range(len(self.shape))))
            # mask out wrapped (out-of-grid) edges
            valid = np.ones(self.shape, dtype=bool)
            for ax, o in enumerate(off):
                sl = [slice(None)] * len(self.shape)
                if o == 1:
                    sl[ax] = -1
                elif o == -1:
                    sl[ax] = 0
                if o != 0:
                    valid[tuple(sl)] = False
            flat_idx, flat_sh, flat_va = idx.ravel(), shifted.ravel(), valid.ravel()
            for i in np.where(flat_va)[0]:
                neigh[flat_idx[i]].append(int(flat_sh[i]))
        return [np.array(v, dtype=int) for v in neigh]

    def _emission_loglik(self, y: np.ndarray) -> np.ndarray:
        """``(n, K)`` Gaussian log-likelihood of each location under each facies."""
        ll = np.empty((self.n, self.k))
        for c in range(self.k):
            diff = y - self.means[c]
            chol = np.linalg.cholesky(self.covs[c])
            sol = np.linalg.solve(chol, diff.T)
            logdet = 2.0 * np.sum(np.log(np.diag(chol)))
            ll[:, c] = -0.5 * (np.sum(sol**2, axis=0) + logdet + y.shape[1] * np.log(2.0 * np.pi))
        return ll

    def _kmeans_init(self, y: np.ndarray, rng: np.random.RandomState, iters: int = 25):
        """k-means++ initialization -- well-separated facies before the Potts coupling is annealed in."""
        c = [y[rng.randint(len(y))]]
        for _ in range(self.k - 1):
            d2 = np.min([np.sum((y - ci) ** 2, axis=1) for ci in c], axis=0)
            c.append(y[rng.choice(len(y), p=d2 / d2.sum())])
        c = np.array(c)
        lab = np.zeros(len(y), dtype=int)
        for _ in range(iters):
            lab = np.argmin(((y[:, None] - c[None]) ** 2).sum(2), axis=1)
            for j in range(self.k):
                if (lab == j).any():
                    c[j] = y[lab == j].mean(0)
        return c, lab

    def fit(
        self, observations: np.ndarray, *, max_iter: int = 40, mf_iter: int = 3, seed: int = 0
    ) -> LatentFaciesField:
        """Fit by mean-field variational EM. ``observations`` is ``(n_locations, n_features)`` (row order
        matches ``shape.ravel()``); stack any per-location measurements into the feature vector.

        Robustness: facies are k-means++-initialized and the Potts coupling is annealed from 0 to ``beta``
        over the first iterations, so clusters form before spatial smoothing is applied (a strong prior on
        a random init otherwise collapses every location into one facies)."""
        y = np.asarray(observations, dtype=float).reshape(self.n, -1)
        d = y.shape[1]
        rng = np.random.RandomState(seed)
        c0, lab = self._kmeans_init(y, rng)
        self.means = c0
        self.covs = np.stack(
            [
                np.cov(y[lab == j].T) + 1e-3 * np.eye(d) if (lab == j).sum() > d else np.cov(y.T) + 1e-3 * np.eye(d)
                for j in range(self.k)
            ]
        )
        self.weights = np.array([max((lab == j).mean(), 1e-3) for j in range(self.k)])
        q = np.eye(self.k)[lab]
        for t in range(max_iter):
            beta_t = self.beta * min(1.0, (t + 1) / max(1.0, 0.3 * max_iter))  # anneal the coupling in
            emis = self._emission_loglik(y)
            for _ in range(mf_iter):  # mean-field fixed point for the Potts posterior
                field = np.array([q[nb].sum(axis=0) if nb.size else np.zeros(self.k) for nb in self._neighbors])
                logq = emis + np.log(self.weights + 1e-300) + beta_t * field
                logq -= logq.max(axis=1, keepdims=True)
                q = np.exp(logq)
                q /= q.sum(axis=1, keepdims=True)
            nk = q.sum(axis=0)
            for c in range(self.k):  # empty-cluster guard: reseed a dead facies at the worst-fit location
                if nk[c] < 1.0:
                    worst = np.argmin(emis[np.arange(self.n), q.argmax(1)])
                    self.means[c] = y[worst]
            self.weights = (nk + 1e-3) / (self.n + self.k * 1e-3)
            for c in range(self.k):
                w = q[:, c]
                if nk[c] >= 1.0:
                    self.means[c] = (w[:, None] * y).sum(axis=0) / nk[c]
                    diff = y - self.means[c]
                    self.covs[c] = (w[:, None] * diff).T @ diff / nk[c] + 1e-3 * np.eye(d)
        self._q = q
        self._y = y
        return self

    def posterior(self) -> np.ndarray:
        """The composition posterior: ``(n_locations, n_facies)`` facies probabilities at each location."""
        return self._q

    def map_facies(self) -> np.ndarray:
        """The MAP facies map (the structure), reshaped to the grid ``shape``."""
        return self._q.argmax(axis=1).reshape(self.shape)

    def entropy(self) -> np.ndarray:
        """Per-location posterior entropy (uncertainty in composition), reshaped to ``shape``."""
        q = np.clip(self._q, 1e-12, 1.0)
        return (-(q * np.log(q)).sum(axis=1)).reshape(self.shape)

    def facies_distribution(self, k: int):
        """The fitted property distribution of facies ``k`` as a :class:`MultivariateGaussianDistribution`."""
        from pysp.stats import MultivariateGaussianDistribution

        return MultivariateGaussianDistribution(self.means[k], self.covs[k])
