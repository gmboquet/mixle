"""IC-1 -- the shared `Posterior` protocol (frozen; work-plan Sec.5).

The single subsurface-posterior type both a core `mixle` distribution and a `mixle_pde` field posterior satisfy, so
the spine (E1-E11), the stochastic optimizer (H4), and the valuation (J2) all consume one structural type. `samples`
is a METHOD returning an ``(n, d)`` draw matrix (NOTE the plural -- `mixle_pde.latent.PosteriorField3D.sample` is
singular today; E1 adds the plural alias). `cov` returns a dense array for small problems or a `LinearOperator` for
survey-scale ones (never materialised -- work-plan Sec.C2). `derived_quantity` maps a pushforward ``fn`` over posterior
draws into a `DerivedQuantity` that carries the samples, a credible interval, AND the honesty flag `prior_dominated`
(work-plan A2): a driller-facing number is never emitted without it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse.linalg import LinearOperator


@runtime_checkable
class DerivedQuantity(Protocol):
    """A pushforward of the posterior through a functional: draws + interval + the prior-dominated honesty flag."""

    samples: np.ndarray  # (n,) or (n, k) draws of the derived quantity, physical units
    prior_dominated: bool  # True when the regulariser, not the data, sets the width (work-plan A2)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Central ``level`` interval (e.g. 0.9) of the derived quantity."""
        ...


@runtime_checkable
class Posterior(Protocol):
    """The frozen subsurface-posterior protocol. Signatures are fixed; only implementations vary."""

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` samples in physical units; shape ``(n, d)``."""
        ...

    @property
    def mean(self) -> np.ndarray:
        """Posterior mean, shape ``(d,)``, physical units."""
        ...

    @property
    def cov(self) -> np.ndarray | LinearOperator:
        """Covariance: a dense ``(d, d)`` array, or a matrix-free ``LinearOperator`` at survey scale."""
        ...

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-component central credible interval covering ``level`` mass; ``(lo, hi)`` each shape ``(d,)``."""
        ...

    def derived_quantity(
        self, fn: Callable[[np.ndarray], np.ndarray], n: int, rng: np.random.Generator
    ) -> DerivedQuantity:
        """Pushforward ``fn`` over ``n`` posterior draws into a `DerivedQuantity` (samples + CI + prior_dominated)."""
        ...
