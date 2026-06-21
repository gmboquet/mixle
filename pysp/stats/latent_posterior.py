"""``LatentPosterior``: a first-class ``q(z | x)`` over a model's latent variables.

The spine of the sampling overhaul. Today each latent model handles its hidden variables implicitly
inside EM (the E-step returns raw responsibility arrays) and exposes no way to *sample* them, while
mean-field models scatter their variational parameters (LDA's per-document ``gamma``/per-token
``phi``, HMM's ``gamma``/``xi``, VMP factors) by hand. ``LatentPosterior`` makes ``q(z | x)`` a single
object -- *exact* for mixtures/HMMs, *mean-field* for LDA/VMP -- so the EM E-step, latent sampling,
and (for mean-field) the ELBO are all just methods on it:

    marginals()   -> the per-latent marginal responsibilities (the EM E-step output)
    sample(rng)   -> a draw  z ~ q(z | x)   (Gibbs / latent / posterior-predictive)
    mode()        -> the MAP latent configuration (argmax / Viterbi)
    entropy()     -> H[q]                    (the ELBO entropy term)

Mean-field realizations additionally provide ``expected_complete_ll(dist)`` / ``update(dist)`` /
``elbo(dist)`` (added with the LDA/VMP realizations); for exact posteriors those are not needed.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import RandomState


def _as_rng(rng: Any) -> RandomState:
    return rng if isinstance(rng, RandomState) else RandomState(rng)


class LatentPosterior(ABC):
    """The posterior ``q(z | x)`` over a model's latent variables (exact or mean-field)."""

    @abstractmethod
    def marginals(self) -> Any:
        """Per-latent marginal responsibilities -- the quantity the EM M-step consumes."""

    @abstractmethod
    def sample(self, rng: Any = None) -> Any:
        """Draw the latent variables ``z ~ q(z | x)`` (``rng`` is a seed, RandomState, or None)."""

    @abstractmethod
    def mode(self) -> Any:
        """The maximum-a-posteriori latent configuration."""

    @abstractmethod
    def entropy(self) -> Any:
        """The entropy ``H[q]`` (per latent / per observation)."""


class CategoricalLatentPosterior(LatentPosterior):
    """Independent categorical latents ``q(z) = prod_i Cat(z_i; r_i)``.

    The exact posterior for a finite mixture's component labels (and the per-token topic factor of an
    LDA document). ``responsibilities`` is the row-stochastic ``(N, K)`` matrix ``r_ik = q(z_i = k |
    x_i)``; ``support`` maps column ``k`` to its latent label (default ``0..K-1``).
    """

    def __init__(self, responsibilities: np.ndarray, support: Any = None) -> None:
        self.responsibilities = np.asarray(responsibilities, dtype=np.float64)
        if self.responsibilities.ndim != 2:
            raise ValueError("responsibilities must be a 2-D (N, K) matrix")
        self.n, self.k = self.responsibilities.shape
        self.support = np.arange(self.k) if support is None else np.asarray(list(support))

    def marginals(self) -> np.ndarray:
        """The ``(N, K)`` responsibility matrix."""
        return self.responsibilities

    def sample(self, rng: Any = None) -> np.ndarray:
        """Draw one latent label per observation; returns an ``(N,)`` array of support labels."""
        rng = _as_rng(rng)
        cdf = np.cumsum(self.responsibilities, axis=1)
        cdf[:, -1] = 1.0  # guard tiny round-off so every uniform draw lands in a bin
        u = rng.random_sample(self.n)[:, None]
        idx = (u < cdf).argmax(axis=1)
        return self.support[idx]

    def mode(self) -> np.ndarray:
        """The most-probable latent label per observation, ``(N,)``."""
        return self.support[np.argmax(self.responsibilities, axis=1)]

    def entropy(self) -> np.ndarray:
        """Per-observation entropy ``-sum_k r_ik log r_ik``, ``(N,)``."""
        r = self.responsibilities
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(r > 0.0, r * np.log(r), 0.0)
        return -terms.sum(axis=1)
