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

Every point ``ask()`` returns is tracked as *pending* until a matching :meth:`BayesianOptimizer.tell`
call resolves it -- see that method's docstring for the exact match / duplicate / unsolicited policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe._contracts import Acquisition
from mixle.doe.bayesopt import BayesOptResult, _fit_surrogate, _validate_prediction, propose_batch, propose_next
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, _require_exact_positive_int, latin_hypercube

# Absolute tolerance for accepting a tell()ed point that is nominally outside `bounds` by floating-
# point noise (matches the tolerance mixle.tests.doe_optimizer_test's own in-bounds helper uses).
_BOUNDS_ATOL = 1.0e-9


class BayesianOptimizer:
    """Stateful ask-tell wrapper around the GP Bayesian-optimization loop.

    ``ask`` proposes the next point (or a batch), ``tell`` records evaluated observations, and
    ``best`` returns the incumbent. Minimizes by default; set ``maximize=True`` to maximize. The
    acquisition is selected by ``acq`` (``"ei"`` / ``"pi"`` / ``"ucb"`` or any registered name /
    callable) with per-acquisition parameters in ``acq_kwargs``.

    ``ask()`` tracks every point it returns as *pending* (asked, not yet told) until a matching
    :meth:`tell` call resolves it. A second, overlapping ``ask()`` -- issued before an earlier one's
    points are told, as in a parallel/async evaluation campaign -- folds those still-pending points
    into the surrogate fit as fantasized (kriging-believer) observations, the same mechanism
    :func:`~mixle.doe.bayesopt.propose_batch` already uses to keep a single batch's own picks apart,
    so acquisition is steered away from proposing them again (MXR-080-0188).
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
        # A silently-clamped-to-1 n_init (the prior `max(1, int(n_init))`) hid a zero/negative/
        # fractional caller error behind a working-looking but wrong-sized initial design instead of
        # rejecting it. Route through the same validator every other count-like mixle.doe control
        # uses, so a bad n_init is rejected exactly like a bad n/trials/level count everywhere else
        # (MXR-080-0188).
        self.n_init = (2 * self.dim + 1) if n_init is None else _require_exact_positive_int(n_init, "n_init")
        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._init_design: np.ndarray | None = None
        self._init_used = 0
        self._pending: dict[int, np.ndarray] = {}  # proposal id -> point, asked but not yet told
        self._next_proposal_id = 0

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
    def pending(self) -> np.ndarray:
        """Return currently pending (asked, not yet told) points as a ``(P, d)`` array."""
        pts = list(self._pending.values())
        return np.asarray(pts, dtype=np.float64).reshape(-1, self.dim) if pts else np.empty((0, self.dim))

    @property
    def n_pending(self) -> int:
        """Return the number of currently pending (asked, not yet told) points."""
        return len(self._pending)

    @property
    def best(self) -> BayesOptResult:
        """Return the incumbent (best observed point) as a :class:`BayesOptResult`."""
        if not self._y:
            raise ValueError("no observations yet; call tell(...) before best.")
        y = self.y
        idx = int(np.argmax(y) if self.maximize else np.argmin(y))
        return BayesOptResult(best_x=self.x[idx], best_y=float(y[idx]), x=self.x, y=y)

    @staticmethod
    def _close(a: np.ndarray, b: np.ndarray) -> bool:
        """Match two points for tell()'s pending/duplicate lookup, tolerant of float round-tripping.

        Plain :func:`numpy.allclose` defaults: loose enough to absorb benign round-trip noise (e.g. a
        point serialized out to the caller's evaluation harness and back), tight enough that two
        independently-proposed points from a continuous search space colliding by chance is not a
        realistic concern.
        """
        return bool(np.allclose(a, b))

    def _peek_init_points(self, count: int) -> list[np.ndarray]:
        """Return the next ``count`` not-yet-dispensed initial-design points, without marking them used.

        Building the full batch (initial-design points plus any GP-acquired points beyond them)
        before committing anything to ``self._init_used`` / ``self._pending`` is what makes
        :meth:`ask` atomic: a batch that fails partway through (e.g. a GP proposal attempted with
        zero observations) leaves no side effect behind, so the call is always safe to retry instead
        of silently burning init-design points the caller never actually received (MXR-080-0188).
        """
        if count <= 0:
            return []
        if self._init_design is None:
            self._init_design = latin_hypercube(self.bounds, self.n_init, self.rng)
        start = self._init_used
        return [np.asarray(self._init_design[i], dtype=np.float64) for i in range(start, start + count)]

    def _fantasized_pending_xy(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(x, y)`` with every currently-pending point appended as a kriging-believer fantasy.

        Fits the surrogate on the real observations and predicts the posterior mean at each pending
        point, then treats that mean as if it were the told outcome -- exactly what
        :func:`~mixle.doe.bayesopt.propose_batch` already does to steer a single batch away from its
        own earlier picks, extended here across separate :meth:`ask` calls so a second, overlapping
        ``ask()`` before any ``tell()`` is steered away from points the first ``ask()`` already handed
        out (MXR-080-0188). Returns ``None`` (no augmentation) when nothing is pending, or when there
        are no real observations yet to fit a surrogate from -- in that case propose_next's /
        propose_batch's own zero-observations error is the right signal, not a fantasy built on no
        data.
        """
        if not self._pending or self.n_observations == 0:
            return None
        x, y = self.x, self.y
        gp = _fit_surrogate(x, y, None, self.fit_kwargs)
        pending = np.asarray(list(self._pending.values()), dtype=np.float64).reshape(-1, self.dim)
        fantasy_mean = gp.predict(x, y, pending, return_cov=False)
        fantasy_mean, _ = _validate_prediction(
            fantasy_mean, None, pending.shape[0], context="ask() (pending-fantasy augmentation)"
        )
        return np.vstack([x, pending]), np.concatenate([y, fantasy_mean])

    def _propose_gp_points(self, remaining: int) -> list[np.ndarray]:
        """Propose ``remaining`` GP-acquisition point(s), with pending points fantasized in."""
        augmented = self._fantasized_pending_xy()
        gp_x, gp_y = augmented if augmented is not None else (self.x, self.y)
        if remaining == 1:
            return [
                np.asarray(
                    propose_next(
                        gp_x,
                        gp_y,
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
            ]
        return list(
            np.asarray(
                propose_batch(
                    gp_x,
                    gp_y,
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

    def ask(self, q: int = 1) -> np.ndarray:
        """Return the next point to evaluate as a ``(d,)`` array, or a ``(q, d)`` batch when ``q > 1``.

        The first ``n_init`` points come from a space-filling design; subsequent points are GP
        acquisition proposals (a kriging-believer batch when ``q > 1``). Every returned point is
        tracked as pending until :meth:`tell` resolves it; a second, overlapping ``ask()`` before any
        ``tell()`` folds still-pending points into the GP fit as fantasized observations so it is
        steered away from proposing them again (see the class docstring, MXR-080-0188).

        This call is atomic: if it raises (e.g. a GP proposal attempted with zero observations), no
        internal state changes, so a failed call is always safe to retry.
        """
        if q < 1:
            raise ValueError("q must be positive.")
        # Exhaust the space-filling initial design first (the GP needs data before it is useful).
        # Gated on self._init_used (points already DISPENSED), not self.n_observations (points
        # already TOLD): those two diverge in the parallel/async campaign this class explicitly
        # supports (ask() called several times before any tell()) -- gating on n_observations let a
        # later ask() re-enter this branch and re-dispense (via _init_used % n_init) an already-issued
        # init point as a duplicate, silently corrupting the space-filling design.
        #
        # Gate solely on self._init_used, not `self._init_used + len(points)`: the old inline loop
        # double-counted every point it dispensed (once via _init_used's own increment inside
        # _next_init_point(), once via the local points list growing), so ask(5) with n_init=5 used
        # to stop after 3 points and fall through to GP batch acquisition with zero observations,
        # which fails (MXR-080-0187). _peek_init_points() below does not mutate _init_used at all --
        # see the atomicity note there and in this method's own docstring.
        n_from_init = min(q, max(0, self.n_init - self._init_used))
        init_points = self._peek_init_points(n_from_init)
        remaining = q - n_from_init
        gp_points = self._propose_gp_points(remaining) if remaining > 0 else []
        # Commit state only now that every point in the batch (init and GP alike) is actually in
        # hand: a GP proposal failure above (e.g. zero observations) leaves _init_used/_pending
        # untouched instead of silently burning init-design points the caller never received.
        self._init_used += len(init_points)
        points = init_points + gp_points
        for p in points:
            self._pending[self._next_proposal_id] = p
            self._next_proposal_id += 1
        out = np.asarray(points, dtype=np.float64)
        return out[0] if q == 1 else out

    def tell(self, x: Any, y: Any) -> BayesianOptimizer:
        """Record one or more evaluated observations; returns ``self`` for chaining.

        ``x`` is a ``(d,)`` point or ``(m, d)`` batch and ``y`` the matching scalar or ``(m,)``
        values. Every point must be finite and within ``bounds``, and every outcome must be finite --
        a NaN/Inf in either would otherwise poison every downstream GP fit and acquisition score
        silently. Every point must also match a currently *pending* proposal, i.e. one this same
        optimizer's own :meth:`ask` actually returned and that has not already been told:

        * a point that was never returned by ``ask()`` (an unsolicited observation) is rejected --
          this is almost always a caller bug (wrong array, a stale point from a different optimizer),
          so it is rejected by default rather than silently accepted;
        * a point that was already told (whether repeated within this same call or told in a
          previous one) is rejected as a duplicate -- ``tell()`` is write-once per point, there is no
          "update an existing observation" path.

        (MXR-080-0188.) Raises ``ValueError`` for any of the above. Validation covers the whole batch
        before anything is applied, so a batch that is only partly invalid never half-applies.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64)).reshape(-1)
        if x.shape[1] != self.dim:
            raise ValueError(f"x has dimension {x.shape[1]}, expected {self.dim}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must describe the same number of observations.")
        if not np.all(np.isfinite(x)):
            raise ValueError("tell(): x contains non-finite values (NaN/Inf).")
        if not np.all(np.isfinite(y)):
            raise ValueError("tell(): y contains non-finite values (NaN/Inf).")
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        if np.any(x < lo - _BOUNDS_ATOL) or np.any(x > hi + _BOUNDS_ATOL):
            raise ValueError("tell(): x is outside bounds.")
        # Resolve every point against the pending set before mutating anything, so a batch that is
        # only partly invalid (e.g. row 2 of 3 unsolicited) never half-applies (MXR-080-0188).
        matched_ids: list[int] = []
        claimed: set[int] = set()
        for xi in x:
            pid = next((cid for cid, p in self._pending.items() if cid not in claimed and self._close(p, xi)), None)
            if pid is not None:
                matched_ids.append(pid)
                claimed.add(pid)
                continue
            already_told = any(self._close(told, xi) for told in self._x)
            repeated_in_this_call = any(self._close(self._pending[cid], xi) for cid in claimed)
            if already_told or repeated_in_this_call:
                raise ValueError(
                    "tell(): duplicate observation -- x was already told, or repeated within this "
                    "same tell() call; tell() does not support re-telling or updating a point."
                )
            raise ValueError(
                "tell(): x does not match any pending ask() proposal; unsolicited observations are not accepted."
            )
        for pid, xi, yi in zip(matched_ids, x, y):
            del self._pending[pid]
            self._x.append(np.asarray(xi, dtype=np.float64))
            self._y.append(float(yi))
        return self


__all__: Sequence[str] = ["BayesianOptimizer"]
