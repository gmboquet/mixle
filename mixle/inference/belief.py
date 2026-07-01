"""Belief states: a distribution over a latent, updated by evidence.

A *belief state* is the answer-side representation for reasoning: the posterior over a scientific
latent given all evidence seen so far. Unlike an embedding vector it is *distributional* -- it
carries its own uncertainty -- and unlike an LLM's hidden state it *updates by Bayesian
conditioning*, so folding in a new modality (or a retrieved datum) is a principled evidence step,
not a concatenation. That makes multi-source reasoning an **assimilation loop**: start from the
prior, fold in evidence one piece at a time, and watch the posterior entropy shrink.

This module provides the exact, canonical realization -- :class:`GaussianBelief`, a multivariate
Gaussian over a continuous latent with a linear-Gaussian (Kalman) measurement update. Two beliefs
about the same latent fuse as a **product of experts** (:meth:`GaussianBelief.fuse`), which is the
cross-modal fusion the reasoning layer is built on. Sequential updates are exact and
order-independent: folding in evidence one datum at a time equals conditioning on all of it at
once -- the property the tests check.

Non-Gaussian belief states (mixture/HMM responsibilities, mean-field fields) already exist as
``LatentPosterior`` realizations in :mod:`mixle.stats.compute.posterior`; :func:`as_belief` adapts
any object exposing ``mean``/``cov`` into this interface. Nonlinear/EKF and particle updates are
future work (see notes/mixle-cross-modal-reasoning-design.md, Phase 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import ndtri


def _as_rng(rng: Any) -> RandomState:
    return rng if isinstance(rng, RandomState) else RandomState(rng)


class BeliefState(ABC):
    """A distribution over a latent, exposing a uniform query + update interface.

    Realizations answer where they are defined: :meth:`mean`, :meth:`cov`, :meth:`var`, :meth:`sd`,
    :meth:`entropy`, :meth:`interval`, :meth:`sample`, :meth:`marginal`, and -- the point of a
    belief state -- :meth:`update`, which returns a *new* belief conditioned on fresh evidence.
    """

    @abstractmethod
    def mean(self) -> np.ndarray:
        """The posterior mean of the latent."""

    @abstractmethod
    def entropy(self) -> float:
        """The differential/Shannon entropy ``H[q]`` (nats) -- watch it shrink as evidence arrives."""

    @abstractmethod
    def sample(self, n: int = 1, rng: Any = None) -> np.ndarray:
        """Draw ``n`` latent samples from the belief."""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> BeliefState:
        """Return a new belief conditioned on fresh evidence (the assimilation step)."""

    def cov(self) -> np.ndarray:
        """The posterior covariance (not defined for every realization)."""
        raise NotImplementedError(f"{type(self).__name__} does not define cov()")

    def var(self) -> np.ndarray:
        """Per-coordinate posterior variance."""
        return np.diag(np.atleast_2d(self.cov()))

    def sd(self) -> np.ndarray:
        """Per-coordinate posterior standard deviation."""
        return np.sqrt(self.var())

    def interval(self, level: float = 0.9) -> np.ndarray:
        """Per-coordinate central credible interval at ``level`` -- an ``(d, 2)`` array of ``[lo, hi]``."""
        raise NotImplementedError(f"{type(self).__name__} does not define interval()")

    def marginal(self, indices: Any) -> BeliefState:
        """The belief restricted to a subset of latent coordinates."""
        raise NotImplementedError(f"{type(self).__name__} does not define marginal()")


class GaussianBelief(BeliefState):
    """A multivariate-Gaussian belief ``N(mean, cov)`` over a continuous latent.

    Evidence is a linear-Gaussian observation ``y = H z + noise``, ``noise ~ N(0, R)``; :meth:`update`
    applies the exact Kalman measurement update (Joseph form, so the covariance stays symmetric
    positive-definite). :meth:`fuse` combines two beliefs about the same latent as a product of
    Gaussian experts. :meth:`condition` does noiseless Gaussian conditioning on a coordinate subset.
    """

    def __init__(self, mean: Any, cov: Any) -> None:
        m = np.atleast_1d(np.asarray(mean, dtype=float))
        P = np.atleast_2d(np.asarray(cov, dtype=float))
        if P.shape != (m.size, m.size):
            raise ValueError(f"cov shape {P.shape} must be ({m.size}, {m.size}) to match mean of size {m.size}")
        self._mean = m
        self._cov = 0.5 * (P + P.T)  # symmetrize defensively
        self._dim = m.size

    @property
    def dim(self) -> int:
        return self._dim

    def mean(self) -> np.ndarray:
        return self._mean.copy()

    def cov(self) -> np.ndarray:
        return self._cov.copy()

    def entropy(self) -> float:
        # H[N(m,P)] = 0.5 (d log(2 pi e) + log|P|). Use the symmetric eigenvalues (clipped to a tiny
        # floor) rather than slogdet: eigvalsh is stable and warning-free for the wide dynamic range a
        # multi-modal latent produces (e.g. density ~1e2 alongside susceptibility ~1e-2).
        evals = np.clip(np.linalg.eigvalsh(self._cov), 1e-300, None)
        logdet = float(np.log(evals).sum())
        return float(0.5 * (self._dim * np.log(2.0 * np.pi * np.e) + logdet))

    def interval(self, level: float = 0.9) -> np.ndarray:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        z = float(ndtri(0.5 * (1.0 + level)))
        half = z * self.sd()
        return np.stack([self._mean - half, self._mean + half], axis=1)

    def sample(self, n: int = 1, rng: Any = None) -> np.ndarray:
        return _as_rng(rng).multivariate_normal(self._mean, self._cov, size=int(n))

    def update(self, H: Any, y: Any, R: Any) -> GaussianBelief:
        """Kalman measurement update: condition on ``y = H z + noise``, ``noise ~ N(0, R)``.

        Args:
            H: ``(k, d)`` observation matrix (or ``(d,)`` / scalar for a single linear readout).
            y: ``(k,)`` observed value (or scalar).
            R: ``(k, k)`` observation-noise covariance (or ``(k,)`` diagonal / scalar).
        """
        Hm = np.atleast_2d(np.asarray(H, dtype=float))
        if Hm.shape[1] != self._dim:
            Hm = Hm.reshape(-1, self._dim)
        k = Hm.shape[0]
        yv = np.atleast_1d(np.asarray(y, dtype=float)).reshape(k)
        Rm = np.asarray(R, dtype=float)
        if Rm.ndim == 0:
            Rm = Rm * np.eye(k)
        elif Rm.ndim == 1:
            Rm = np.diag(Rm)

        P = self._cov
        S = Hm @ P @ Hm.T + Rm  # innovation covariance
        K = np.linalg.solve(S, Hm @ P).T  # gain = P Hᵀ S⁻¹  (via solve for stability)
        innovation = yv - Hm @ self._mean
        m_new = self._mean + K @ innovation
        ImKH = np.eye(self._dim) - K @ Hm
        P_new = ImKH @ P @ ImKH.T + K @ Rm @ K.T  # Joseph form: symmetric, PSD
        return GaussianBelief(m_new, P_new)

    def fuse(self, other: GaussianBelief) -> GaussianBelief:
        """Product-of-experts fusion of two beliefs about the same latent (cross-modal fusion).

        Equivalent to conditioning ``self`` on ``other`` treated as a direct Gaussian observation
        (``H = I``, ``R = other.cov``), so it reuses the exact Kalman update.
        """
        if other.dim != self._dim:
            raise ValueError(f"cannot fuse beliefs of dimension {self._dim} and {other.dim}")
        return self.update(np.eye(self._dim), other._mean, other._cov)

    def condition(self, indices: Any, values: Any) -> GaussianBelief:
        """Noiseless Gaussian conditioning: fix latent coordinates ``indices`` to ``values``.

        Returns the belief over the *remaining* coordinates. This is the exact ``R -> 0`` limit of
        an observation that reads off those coordinates.
        """
        obs = np.atleast_1d(np.asarray(indices, dtype=int))
        vals = np.atleast_1d(np.asarray(values, dtype=float))
        keep = np.array([i for i in range(self._dim) if i not in set(obs.tolist())], dtype=int)
        if keep.size == 0:
            raise ValueError("cannot condition on all coordinates (nothing left to infer)")
        Paa = self._cov[np.ix_(keep, keep)]
        Pab = self._cov[np.ix_(keep, obs)]
        Pbb = self._cov[np.ix_(obs, obs)]
        gain = np.linalg.solve(Pbb, Pab.T).T  # Pab Pbb⁻¹
        m_new = self._mean[keep] + gain @ (vals - self._mean[obs])
        P_new = Paa - gain @ Pab.T
        return GaussianBelief(m_new, P_new)

    def marginal(self, indices: Any) -> GaussianBelief:
        idx = np.atleast_1d(np.asarray(indices, dtype=int))
        return GaussianBelief(self._mean[idx], self._cov[np.ix_(idx, idx)])

    def __repr__(self) -> str:
        return f"GaussianBelief(dim={self._dim}, entropy={self.entropy():.3f} nats)"


def as_belief(obj: Any, node: Any = None) -> GaussianBelief:
    """Adapt any object exposing ``mean``/``cov`` (a ``FieldPosterior`` node, a fitted Gaussian, a
    ``ParameterPosterior``) into a :class:`GaussianBelief`.

    ``node`` is forwarded when the source is node-addressable (e.g. ``FieldPosterior.mean(node)`` /
    ``.cov(node)``); otherwise ``mean``/``cov`` are called with no argument.
    """

    def _call(attr: str) -> Any:
        fn = getattr(obj, attr, None)
        if fn is None:
            raise TypeError(f"{type(obj).__name__} has no {attr}() to build a belief from")
        if not callable(fn):
            return fn
        try:
            return fn(node) if node is not None else fn()
        except TypeError:
            return fn()

    return GaussianBelief(_call("mean"), _call("cov"))
