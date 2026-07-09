"""Ask-tell Bayesian-optimization interface for mixle.doe (WS-E).

A small stateful optimizer object for the common human/experiment-in-the-loop workflow, where the
objective is expensive or physical and evaluated *outside* the loop:

    opt = BayesianOptimizer(bounds, acq="ei")
    for _ in range(n):
        x = opt.ask()          # next point(s) to evaluate
        y = run_experiment(x)  # ... done by the caller, however slow
        opt.tell(x, y)         # feed the result back
    opt.best                   # best (x, y) so far

It holds the observation history and delegates proposals to the functional API
(:mod:`mixle.doe.bayesopt`): the first ``n_init`` asks come from a space-filling Latin-hypercube
design (a GP needs data before it is useful), after which asks are GP-acquisition proposals
(``ask(q>1)`` returns a kriging-believer batch). Constrained and multi-objective problems keep their
functional drivers (``constrained_minimize`` / ``multi_minimize``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe._contracts import Acquisition
from mixle.doe.bayesopt import BayesOptResult, propose_batch, propose_next
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


class BayesianOptimizer:
    """Stateful ask-tell wrapper around the GP Bayesian-optimization loop.

    ``ask`` proposes the next point (or a batch), ``tell`` records evaluated observations, and
    ``best`` returns the incumbent. Minimizes by default; set ``maximize=True`` to maximize. The
    acquisition is selected by ``acq`` (``"ei"`` / ``"pi"`` / ``"ucb"`` or any registered name /
    callable) with per-acquisition parameters in ``acq_kwargs``.
    """

    def __init__(
        self,
        bounds: Bounds,
        *,
        acq: str | Acquisition = "ei",
        acq_kwargs: dict[str, Any] | None = None,
        maximize: bool = False,
        n_init: int | None = None,
        xi: float = 0.0,
        n_candidates: int = 512,
        fit_kwargs: dict[str, Any] | None = None,
        seed: int | RandomState | None = None,
    ) -> None:
        self.bounds = _as_bounds(bounds)
        self.dim = int(self.bounds.shape[0])
        self.acq = acq
        self.acq_kwargs = acq_kwargs
        self.maximize = bool(maximize)
        self.xi = float(xi)
        self.n_candidates = int(n_candidates)
        self.fit_kwargs = fit_kwargs
        self.rng = _as_rng(seed)
        self.n_init = (2 * self.dim + 1) if n_init is None else max(1, int(n_init))
        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._init_design: np.ndarray | None = None
        self._init_used = 0

    @property
    def x(self) -> np.ndarray:
        """Return the observed points as an ``(N, d)`` array."""
        return np.asarray(self._x, dtype=np.float64).reshape(-1, self.dim) if self._x else np.empty((0, self.dim))

    @property
    def y(self) -> np.ndarray:
        """Return the observed objective values as an ``(N,)`` array."""
        return np.asarray(self._y, dtype=np.float64)

    @property
    def n_observations(self) -> int:
        """Return the number of recorded observations."""
        return len(self._y)

    @property
    def best(self) -> BayesOptResult:
        """Return the incumbent (best observed point) as a :class:`BayesOptResult`."""
        if not self._y:
            raise ValueError("no observations yet; call tell(...) before best.")
        y = self.y
        idx = int(np.argmax(y) if self.maximize else np.argmin(y))
        return BayesOptResult(best_x=self.x[idx], best_y=float(y[idx]), x=self.x, y=y)

    def _next_init_point(self) -> np.ndarray:
        if self._init_design is None:
            self._init_design = latin_hypercube(self.bounds, self.n_init, self.rng)
        point = self._init_design[self._init_used % self.n_init]
        self._init_used += 1
        return np.asarray(point, dtype=np.float64)

    def ask(self, q: int = 1) -> np.ndarray:
        """Return the next point to evaluate as a ``(d,)`` array, or a ``(q, d)`` batch when ``q > 1``.

        The first ``n_init`` points come from a space-filling design; subsequent points are GP
        acquisition proposals (a kriging-believer batch when ``q > 1``).
        """
        if q < 1:
            raise ValueError("q must be positive.")
        points: list[np.ndarray] = []
        # Exhaust the space-filling initial design first (the GP needs data before it is useful).
        # Gated on self._init_used (points already DISPENSED), not self.n_observations (points
        # already TOLD): those two diverge in the parallel/async campaign this class explicitly
        # supports (ask() called several times before any tell()) -- gating on n_observations let a
        # later ask() re-enter this branch and re-dispense (via _init_used % n_init) an already-issued
        # init point as a duplicate, silently corrupting the space-filling design.
        while len(points) < q and (self._init_used + len(points)) < self.n_init:
            points.append(self._next_init_point())
        remaining = q - len(points)
        if remaining == 1:
            points.append(
                np.asarray(
                    propose_next(
                        self.x,
                        self.y,
                        self.bounds,
                        n_candidates=self.n_candidates,
                        seed=self.rng,
                        maximize=self.maximize,
                        xi=self.xi,
                        acq=self.acq,
                        acq_kwargs=self.acq_kwargs,
                        fit_kwargs=self.fit_kwargs,
                    ),
                    dtype=np.float64,
                )
            )
        elif remaining > 1:
            points.extend(
                np.asarray(
                    propose_batch(
                        self.x,
                        self.y,
                        self.bounds,
                        q=remaining,
                        n_candidates=self.n_candidates,
                        seed=self.rng,
                        maximize=self.maximize,
                        xi=self.xi,
                        acq=self.acq,
                        acq_kwargs=self.acq_kwargs,
                        fit_kwargs=self.fit_kwargs,
                    ),
                    dtype=np.float64,
                )
            )
        out = np.asarray(points, dtype=np.float64)
        return out[0] if q == 1 else out

    def tell(self, x: Any, y: Any) -> BayesianOptimizer:
        """Record one or more evaluated observations; returns ``self`` for chaining.

        ``x`` is a ``(d,)`` point or ``(m, d)`` batch and ``y`` the matching scalar or ``(m,)`` values.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64)).reshape(-1)
        if x.shape[1] != self.dim:
            raise ValueError(f"x has dimension {x.shape[1]}, expected {self.dim}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must describe the same number of observations.")
        for xi, yi in zip(x, y):
            self._x.append(np.asarray(xi, dtype=np.float64))
            self._y.append(float(yi))
        return self


__all__: Sequence[str] = ["BayesianOptimizer"]
