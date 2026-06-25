"""Spatial mixture: a mixture whose latent labels live on a grid under a Markov-random-field prior.

A plain mixture treats observations as exchangeable. When the observations sit on a grid (an image, a
field of measurements, a map) the latent component labels are *spatially coherent* -- neighbouring cells
tend to share a component. This adds a Potts / Ising smoothness prior over the label field,
``P(z) proportional to exp(beta * sum_{i~j} 1[z_i == z_j])``, on top of an arbitrary per-component pysp
emission distribution. It generalizes :class:`~pysp.stats.MixtureDistribution` with spatial coupling and
reduces to it at ``beta = 0``; inference is mean-field variational EM. The emission family is any pysp
estimator (Gaussian, multivariate Gaussian, categorical, ...), so the spatial structure is the only thing
this class adds -- everything about *what* each component emits is delegated to the library.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["SpatialMixture"]


def _grid_neighbors(shape: tuple[int, ...]) -> list[np.ndarray]:
    """For each node, the flat indices of its first-order (von Neumann) in-grid neighbours."""
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    neigh: list[list[int]] = [[] for _ in range(n)]
    for ax in range(len(shape)):
        for d in (-1, 1):
            sl_src = [slice(None)] * len(shape)
            sl_dst = [slice(None)] * len(shape)
            sl_src[ax] = slice(1, None) if d == 1 else slice(0, -1)
            sl_dst[ax] = slice(0, -1) if d == 1 else slice(1, None)
            a, b = idx[tuple(sl_src)].ravel(), idx[tuple(sl_dst)].ravel()
            for u, v in zip(a, b):
                neigh[int(u)].append(int(v))
    return [np.array(v, dtype=int) for v in neigh]


class SpatialMixture:
    """A grid-structured mixture with a Potts prior on the latent labels and pluggable pysp emissions.

    Args:
        shape: grid shape, e.g. ``(nx, ny)`` or ``(nx, ny, nz)`` -- defines the neighbour structure.
        n_components: number of mixture components (latent classes).
        emission: a pysp ``ParameterEstimator`` for the per-component family, e.g.
            ``MultivariateGaussianEstimator()`` -- this is what makes the class domain-agnostic.
        beta: Potts coupling (``>= 0``); larger smooths the labels more. ``0`` is an ordinary mixture.
    """

    def __init__(self, shape, n_components: int, emission, beta: float = 1.0):
        self.shape = tuple(int(s) for s in np.atleast_1d(shape))
        self.k = int(n_components)
        self.emission = emission
        self.beta = float(beta)
        self.n = int(np.prod(self.shape))
        self._neighbors = _grid_neighbors(self.shape)

    def _emission_loglik(self, data_enc) -> np.ndarray:
        """``(n, K)`` log-likelihood of every cell under each component, via the emissions' encoders."""
        return np.column_stack([c.seq_log_density(data_enc) for c in self.components])

    def _reestimate(self, acc_enc, q: np.ndarray, current: list | None = None) -> list:
        """Responsibility-weighted M-step: drive each component's accumulator and re-estimate (pysp contract).

        ``current`` is the previous component list (``None`` on the first, initialization call)."""
        out = []
        for j in range(self.k):
            acc = self.emission.accumulator_factory().make()
            acc.seq_update(acc_enc, q[:, j], None if current is None else current[j])
            out.append(self.emission.estimate(None, acc.value()))
        return out

    def fit(self, observations, *, max_iter: int = 40, mf_iter: int = 3, seed: int = 0) -> SpatialMixture:
        """Fit by mean-field variational EM. ``observations`` is a length-``prod(shape)`` sequence of
        per-cell observations (row order matches ``shape.ravel()``); each is a single emission datum.

        Robustness: components are initialized by a short hard-assignment pass and the Potts coupling is
        annealed from 0 to ``beta`` over the first iterations, so components form before the smoothness
        prior is applied (a strong prior on a degenerate init otherwise collapses every cell into one)."""
        data = list(observations)
        rng = np.random.RandomState(seed)
        acc_enc = self.emission.accumulator_factory().make().acc_to_encoder().seq_encode(data)

        # init: random partition -> estimate each component -> a few hard-EM steps to separate them
        lab = rng.randint(self.k, size=self.n)
        self.components = self._reestimate(acc_enc, np.eye(self.k)[lab], current=None)
        for _ in range(5):
            lab = self._emission_loglik(self.components[0].dist_to_encoder().seq_encode(data)).argmax(axis=1)
            counts = np.bincount(lab, minlength=self.k)
            if (counts == 0).any():  # reseed any empty component at a random cell
                for j in np.where(counts == 0)[0]:
                    lab[rng.randint(self.n)] = j
            self.components = self._reestimate(acc_enc, np.eye(self.k)[lab], current=self.components)

        q = np.eye(self.k)[lab]
        for t in range(max_iter):
            beta_t = self.beta * min(1.0, (t + 1) / max(1.0, 0.3 * max_iter))  # anneal the coupling in
            emis = self._emission_loglik(self.components[0].dist_to_encoder().seq_encode(data))
            for _ in range(mf_iter):  # mean-field fixed point for the Potts posterior
                field = np.array([q[nb].sum(axis=0) if nb.size else np.zeros(self.k) for nb in self._neighbors])
                logq = emis + beta_t * field
                logq -= logq.max(axis=1, keepdims=True)
                q = np.exp(logq)
                q /= q.sum(axis=1, keepdims=True)
            self.components = self._reestimate(acc_enc, q, current=self.components)
        self._q = q
        return self

    def responsibilities(self) -> np.ndarray:
        """The posterior label probabilities, ``(prod(shape), n_components)`` -- a simplex per cell."""
        return self._q

    def labels(self) -> np.ndarray:
        """The MAP label field, reshaped to ``shape``."""
        return self._q.argmax(axis=1).reshape(self.shape)

    def entropy(self) -> np.ndarray:
        """Per-cell posterior entropy (label uncertainty), reshaped to ``shape``."""
        q = np.clip(self._q, 1e-12, 1.0)
        return (-(q * np.log(q)).sum(axis=1)).reshape(self.shape)

    def component(self, j: int) -> Any:
        """The fitted pysp emission distribution of component ``j``."""
        return self.components[j]
