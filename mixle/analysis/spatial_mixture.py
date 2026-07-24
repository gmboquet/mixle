"""Spatial mixture: a mixture whose latent labels live on a grid under a Markov-random-field prior.

A plain mixture treats observations as exchangeable. When the observations sit on a grid (an image, a
field of measurements, a map) the latent component labels are *spatially coherent* -- neighbouring cells
tend to share a component. This adds a Potts / Ising smoothness prior over the label field,
``P(z) proportional to exp(beta * sum_{i~j} 1[z_i == z_j])``, on top of an arbitrary per-component mixle
emission distribution. It generalizes :class:`~mixle.stats.MixtureDistribution` with spatial coupling and
reduces to it at ``beta = 0``; inference is mean-field variational EM. The emission family is any mixle
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
    """A grid-structured mixture with a Potts prior on the latent labels and pluggable mixle emissions.

    Args:
        shape: grid shape, e.g. ``(nx, ny)`` or ``(nx, ny, nz)`` -- defines the neighbour structure.
        n_components: number of mixture components (latent classes).
        emission: a mixle ``ParameterEstimator`` for the per-component family, e.g.
            ``MultivariateGaussianEstimator()`` -- this is what makes the class domain-agnostic.
        beta: Potts coupling (``>= 0``); larger smooths the labels more. ``0`` is an ordinary mixture.

    Raises:
        ValueError: if any ``shape`` dimension is not positive, if ``n_components`` is not in
            ``[1, prod(shape)]`` (MXR-080-0115: more components than grid cells guarantees at least
            one permanently empty component by pigeonhole -- rejected here rather than left to fail
            unpredictably during fitting), or if ``beta`` is not finite and non-negative.
    """

    def __init__(self, shape, n_components: int, emission, beta: float = 1.0):
        shape = tuple(int(s) for s in np.atleast_1d(shape))
        if len(shape) == 0 or any(s <= 0 for s in shape):
            raise ValueError(f"shape must be one or more positive grid dimensions, got {shape}")
        n = int(np.prod(shape))
        k = int(n_components)
        if not (1 <= k <= n):
            reason = (
                "n_components > n_cells guarantees at least one permanently empty component by pigeonhole"
                if k > n
                else "n_components must be at least 1"
            )
            raise ValueError(
                f"n_components must satisfy 1 <= n_components <= prod(shape); shape {shape} has {n} "
                f"cells, got n_components={k} ({reason})"
            )
        beta = float(beta)
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError(f"beta must be finite and non-negative, got {beta!r}")
        self.shape = shape
        self.k = k
        self.emission = emission
        self.beta = beta
        self.n = n
        self._neighbors = _grid_neighbors(self.shape)

    def _emission_loglik(self, data) -> np.ndarray:
        """``(n, K)`` log-likelihood of every cell under each component.

        MXR-080-0116: each component is encoded and scored with its OWN ``dist_to_encoder()``, never a
        different component's -- the class's pluggable-emission design explicitly allows components
        whose fitted encoders differ (e.g. a family whose encoder captures per-fit discovered support),
        so reusing one shared encoding across components would silently score most of them with the
        wrong encoder. Every component must still honor the structurally compatible encoder contract
        the shared responsibilities/neighbor bookkeeping requires: one log-density value per cell,
        regardless of which emission family produced it. That is enforced here, directly on each
        component's output, rather than left to an incidental ``column_stack`` shape error.
        """
        cols = []
        for j, c in enumerate(self.components):
            col = np.asarray(c.seq_log_density(c.dist_to_encoder().seq_encode(data)))
            if col.shape != (self.n,):
                raise ValueError(
                    f"component {j} ({type(c).__name__}) violates the encoder contract: scoring "
                    f"{self.n} cells produced a log-density array of shape {col.shape}, expected "
                    f"({self.n},) -- every component's encoder must produce exactly one log-density "
                    "value per cell for the shared spatial responsibilities/neighbor bookkeeping to "
                    "stay valid"
                )
            cols.append(col)
        return np.column_stack(cols)

    def _reestimate(self, acc_enc, q: np.ndarray, current: list | None = None) -> list:
        """Responsibility-weighted M-step: drive each component's accumulator and re-estimate (mixle contract).

        ``current`` is the previous component list (``None`` on the first, initialization call)."""
        out = []
        for j in range(self.k):
            acc = self.emission.accumulator_factory().make()
            acc.seq_update(acc_enc, q[:, j], None if current is None else current[j])
            out.append(self.emission.estimate(None, acc.value()))
        return out

    def _repair_empty_components(self, lab: np.ndarray) -> np.ndarray:
        """Return a copy of ``lab`` with every component assigned at least one cell.

        MXR-080-0115: called before every re-estimate -- both on the initial random partition and on
        every hard-EM refinement -- so an empty component is never re-estimated as-is (that silently
        produces an invalid/degenerate placeholder, e.g. a Gaussian stuck at its zero-accumulator
        fallback, instead of an error).

        One cell is donated to each empty component from the currently largest component (ties broken
        by lowest component id, then lowest cell index): fully deterministic, so the same ``lab``
        always repairs the same way, regardless of incidental array/iteration order. Counts are updated
        after each donation, so a later donation within the same call always sees the up-to-date state
        -- it can never re-empty a component this call just fixed, nor steal the sole cell of a
        pre-existing one-cell component, because a donor must have at least 2 cells before it can give
        one up. ``1 <= k <= n`` (enforced at construction) guarantees a >=2-cell donor always exists
        whenever an empty component does, however many components are empty at once.
        """
        lab = np.asarray(lab).copy()
        counts = np.bincount(lab, minlength=self.k)
        for j in np.where(counts == 0)[0]:
            donors = np.where(counts >= 2)[0]
            if donors.size == 0:  # pragma: no cover -- unreachable given 1 <= k <= n at construction
                raise RuntimeError("no donor component available to repair an empty component")
            donor = donors[np.argmax(counts[donors])]
            donor_cell = np.where(lab == donor)[0][0]
            lab[donor_cell] = j
            counts[donor] -= 1
            counts[j] += 1
        return lab

    def fit(self, observations, *, max_iter: int = 40, mf_iter: int = 3, seed: int = 0) -> SpatialMixture:
        """Fit by mean-field variational EM. ``observations`` is a length-``prod(shape)`` sequence of
        per-cell observations (row order matches ``shape.ravel()``); each is a single emission datum.

        Robustness: components are initialized by a short hard-assignment pass and the Potts coupling is
        annealed from 0 to ``beta`` over the first iterations, so components form before the smoothness
        prior is applied (a strong prior on a degenerate init otherwise collapses every cell into one).
        Every partition is repaired to be nonempty (see :meth:`_repair_empty_components`) BEFORE it is
        re-estimated (MXR-080-0115).

        Raises:
            ValueError: if ``max_iter`` or ``mf_iter`` is not a positive integer, or if
                ``observations`` does not have exactly ``prod(shape)`` entries (MXR-080-0116: a
                mismatched count would otherwise let responsibilities, neighbors, and encoded data
                describe different numbers of cells, surfacing later as an unrelated shape-mismatch
                crash deep inside the M-step instead of a clear error here).
        """
        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
        mf_iter = int(mf_iter)
        if mf_iter < 1:
            raise ValueError(f"mf_iter must be a positive integer, got {mf_iter!r}")

        data = list(observations)
        if len(data) != self.n:
            raise ValueError(
                f"observations must have exactly prod(shape)={self.n} entries, one per grid cell in "
                f"shape {self.shape} (row order matching shape.ravel()); got {len(data)}"
            )
        rng = np.random.RandomState(seed)
        acc_enc = self.emission.accumulator_factory().make().acc_to_encoder().seq_encode(data)

        # init: random partition -> repair -> estimate each component -> a few hard-EM steps to
        # separate them, repairing before every re-estimate (never after).
        lab = self._repair_empty_components(rng.randint(self.k, size=self.n))
        self.components = self._reestimate(acc_enc, np.eye(self.k)[lab], current=None)
        for _ in range(5):
            lab = self._emission_loglik(data).argmax(axis=1)
            lab = self._repair_empty_components(lab)
            self.components = self._reestimate(acc_enc, np.eye(self.k)[lab], current=self.components)

        q = np.eye(self.k)[lab]
        for t in range(max_iter):
            beta_t = self.beta * min(1.0, (t + 1) / max(1.0, 0.3 * max_iter))  # anneal the coupling in
            emis = self._emission_loglik(data)
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
        """The fitted mixle emission distribution of component ``j``."""
        return self.components[j]
