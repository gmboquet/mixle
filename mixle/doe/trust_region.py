"""Trust-region Bayesian optimization (TuRBO, Eriksson et al. 2019) for higher-dimensional problems.

Global GP-BO degrades in high dimensions: one global surrogate is hard to fit and the acquisition is
over-exploratory. TuRBO instead keeps a **trust region** -- a box centered on the best point -- and runs
the BO only inside it, via Thompson sampling. The box side length grows after consecutive improvements
and shrinks after consecutive failures; when it collapses, the search restarts from a fresh design.
This local, self-tuning focus makes BO work for tens of dimensions where global EI stalls.

:func:`turbo_minimize` is the optimization loop (it calls your objective); :class:`TrustRegion` is the
expand/shrink state if you want to drive the loop yourself. Both fit the torch GP surrogate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe.batch import _safe_cholesky
from mixle.doe.bayesopt import _fit_surrogate, _require_finite_scalar
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, _require_exact_positive_int, latin_hypercube


@dataclass
class TrustRegion:
    """Expand/shrink state of a TuRBO trust region (side length in normalized ``[0, 1]^d`` coordinates).

    The length doubles after ``success_tol`` consecutive improving batches and halves after
    ``failure_tol`` consecutive non-improving ones; ``collapsed`` is True once it drops below
    ``length_min`` and the caller should restart. ``failure_tol`` defaults to the dimension (Eriksson 2019).

    ``__post_init__`` validates the complete region invariant (MXR-080-0196): ``dim`` is an exact
    positive integer; ``length``, ``length_min``, and ``length_max`` are finite with
    ``0 < length_min <= length <= length_max`` (so the region can never start collapsed, inverted, or
    unboundedly sized); ``success_tol`` is an exact positive integer and ``failure_tol`` an exact
    non-negative one (``0`` is the "default to ``dim``" sentinel resolved right below). Anything outside
    that invariant raises immediately instead of accepting geometry with no sensible search meaning --
    e.g. a ``length_max < length_min`` would let a "successful" expansion (``min(length * 2, length_max)``)
    silently shrink the region, and a non-finite/non-positive ``length`` would make every candidate box
    :func:`_tr_candidates` draws undefined or empty.
    """

    dim: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tol: int = 3
    failure_tol: int = 0  # 0 -> set to dim in __post_init__
    _success: int = 0
    _failure: int = 0

    def __post_init__(self) -> None:
        self.dim = _require_exact_positive_int(self.dim, "dim")
        self.success_tol = _require_exact_positive_int(self.success_tol, "success_tol")
        self.failure_tol = _require_exact_positive_int(self.failure_tol, "failure_tol", minimum=0)
        if self.failure_tol <= 0:
            self.failure_tol = max(4, self.dim)
        self.length_min = _require_finite_scalar(self.length_min, "length_min")
        if self.length_min <= 0.0:
            raise ValueError(f"length_min must be positive, got {self.length_min!r}.")
        self.length_max = _require_finite_scalar(self.length_max, "length_max")
        if self.length_max < self.length_min:
            raise ValueError(
                f"length_max must be >= length_min, got length_max={self.length_max!r}, length_min={self.length_min!r}."
            )
        self.length = _require_finite_scalar(self.length, "length")
        if not (self.length_min <= self.length <= self.length_max):
            raise ValueError(
                f"length must satisfy length_min <= length <= length_max, got length={self.length!r}, "
                f"length_min={self.length_min!r}, length_max={self.length_max!r}."
            )

    @property
    def collapsed(self) -> bool:
        """Return whether the trust-region length has shrunk below its usable minimum."""
        return self.length < self.length_min

    def update(self, improved: bool) -> None:
        """Record whether the latest batch improved the incumbent, and resize the region."""
        if improved:
            self._success += 1
            self._failure = 0
        else:
            self._failure += 1
            self._success = 0
        if self._success >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            self._success = 0
        if self._failure >= self.failure_tol:
            self.length /= 2.0
            self._failure = 0


def _tr_candidates(center: np.ndarray, length: float, n: int, rng: RandomState) -> np.ndarray:
    """``n`` candidates in the trust region (normalized), perturbing a random subset of dims per point.

    Each candidate perturbs each coordinate with probability ``min(1, 20/d)`` (else keeps the center
    value) -- the TuRBO trick that keeps the effective search low-dimensional in high ``d``.
    """
    d = center.size
    lb = np.clip(center - 0.5 * length, 0.0, 1.0)
    ub = np.clip(center + 0.5 * length, 0.0, 1.0)
    pert = lb[None, :] + (ub - lb)[None, :] * rng.random((n, d))
    prob = min(1.0, 20.0 / d)
    mask = rng.random((n, d)) < prob
    empty = ~mask.any(axis=1)
    if empty.any():
        mask[empty, rng.randint(0, d, int(empty.sum()))] = True
    return np.where(mask, pert, center[None, :])


def _thompson_batch(gp: Any, xn: np.ndarray, yn: np.ndarray, cand: np.ndarray, q: int, rng: RandomState) -> np.ndarray:
    """Pick ``q`` distinct trust-region candidates by Thompson sampling (joint GP posterior draws)."""
    if q > cand.shape[0]:
        # once every candidate index is chosen, the inner loop finds nothing left to pick and that
        # round silently contributes NOTHING to `picks` -- the caller would get fewer than q points
        # back with no error. Name the actual constraint instead.
        raise ValueError(f"_thompson_batch requires q <= cand.shape[0] (q={q}, candidates={cand.shape[0]}).")
    mean, cov = gp.predict(xn, yn, cand, return_cov=True)
    mean = np.asarray(mean, dtype=np.float64).ravel()
    chol = _safe_cholesky(np.atleast_2d(np.asarray(cov, dtype=np.float64)))
    picks: list[np.ndarray] = []
    chosen: set[int] = set()
    for _ in range(q):
        sample = mean + chol @ rng.standard_normal(mean.size)
        for idx in np.argsort(sample):
            if int(idx) not in chosen:
                chosen.add(int(idx))
                picks.append(cand[int(idx)])
                break
    return np.asarray(picks)


def turbo_minimize(
    objective: Callable[[np.ndarray], float],
    bounds: Bounds,
    *,
    n_init: int | None = None,
    max_evals: int = 100,
    batch_size: int = 1,
    maximize: bool = False,
    n_candidates: int | None = None,
    seed: int | RandomState | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize a black-box ``objective`` over ``bounds`` with TuRBO (trust-region BO).

    Starts from a Latin-hypercube design of ``n_init`` points (default ``2*d``), then repeatedly fits a
    GP on the normalized data, draws Thompson candidates inside the current trust region, evaluates the
    best ``batch_size`` of them, and resizes the region by success/failure. On collapse it restarts from
    a new design. Runs until ``max_evals`` objective calls.

    ``n_init`` (default ``2*d``), ``max_evals``, ``batch_size``, and ``n_candidates`` (default
    ``min(2000, 100*d)``) must each be exact positive integers -- none is silently truncated or coerced.
    ``max_evals >= n_init`` and ``1 <= batch_size <= n_candidates`` are both required up front: the first
    because the initial design alone spends ``n_init`` real evaluations before any GP step can begin, the
    second because a batch can never select more distinct candidates than the trust region generates for
    it, no matter how many times it retries (MXR-080-0196).

    Returns ``{'x', 'y', 'X', 'Y', 'n_restarts'}`` with the best point/value and the full history.
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    rng = _as_rng(seed)
    span = b[:, 1] - b[:, 0]
    # MXR-080-0196: every count/budget control below is an exact, validated positive integer -- none of
    # them silently truncates a fractional value or lets a negative one slip through. `n_init` and
    # `n_candidates` use `is not None` (not truthiness) for their "auto" sentinel, so an explicit 0 is
    # rejected rather than quietly reinterpreted as "use the default".
    n_init = _require_exact_positive_int(n_init, "n_init") if n_init is not None else 2 * d
    max_evals = _require_exact_positive_int(max_evals, "max_evals")
    if max_evals < n_init:
        # the initial Latin-hypercube design alone needs n_init real objective calls before any
        # GP-based step can even begin -- max_evals < n_init can't be honored (the function's own
        # contract is "runs until max_evals objective calls"), and evaluating the design anyway
        # would silently overshoot the caller's budget rather than raise.
        raise ValueError(f"turbo_minimize requires max_evals >= n_init (n_init={n_init}, max_evals={max_evals}).")
    n_cand = (
        _require_exact_positive_int(n_candidates, "n_candidates") if n_candidates is not None else min(2000, 100 * d)
    )
    batch_size = _require_exact_positive_int(batch_size, "batch_size")
    if batch_size > n_cand:
        # a batch selects `batch_size` DISTINCT candidates out of the `n_cand` generated per iteration
        # (_thompson_batch's own contract, checked every iteration) -- asking for more than that can
        # never be satisfied by any number of retries, shrinks, or restarts (n_cand is fixed for the
        # whole run), so this is a caller configuration error, not a model failure. Reject it here once,
        # up front, instead of letting it repeatedly raise inside the loop where it would be caught and
        # misattributed as "the model failed" (MXR-080-0196/0197).
        raise ValueError(
            f"turbo_minimize requires batch_size <= n_candidates (batch_size={batch_size}, "
            f"n_candidates={n_cand}); increase n_candidates or lower batch_size."
        )
    sign = -1.0 if maximize else 1.0  # always minimize sign*objective

    def to_unit(x):
        return (x - b[:, 0]) / span

    def evaluate(xn):
        return sign * float(objective(np.asarray(xn, dtype=np.float64)))

    x_all = latin_hypercube(b, n_init, rng)
    y_all = np.array([evaluate(p) for p in x_all], dtype=np.float64)
    tr = TrustRegion(dim=d)
    best = float(y_all.min())
    restarts = 0

    while y_all.shape[0] < max_evals:
        if tr.collapsed:
            restarts += 1
            # clamp the restart design to the REMAINING budget -- an unclamped n_init-point design
            # can overshoot max_evals by up to n_init real objective calls if the trust region
            # collapses near the end of the run (a real, common occurrence on hard landscapes).
            n_restart = min(n_init, max_evals - y_all.shape[0])
            xr = latin_hypercube(b, n_restart, rng)
            yr = np.array([evaluate(p) for p in xr], dtype=np.float64)
            x_all = np.vstack([x_all, xr])
            y_all = np.append(y_all, yr)
            tr = TrustRegion(dim=d)
            best = float(y_all.min())
            continue
        # Fit a LOCAL GP -- only the points near the trust-region centre. This is the TuRBO design (the
        # model is local) and it keeps the kernel matrix well-conditioned even after many evaluations
        # accumulate clustered points (a global fit goes singular on near-duplicates).
        xu_all = to_unit(x_all)
        center = to_unit(x_all[int(np.argmin(y_all))])
        dist = np.linalg.norm(xu_all - center, axis=1)
        floor = max(2 * d, 8)
        keep = np.where(dist <= tr.length)[0]
        if keep.size < floor:
            keep = np.argsort(dist)[:floor]
        if keep.size > 96:
            keep = keep[np.argsort(dist[keep])[:96]]
        xu, yloc = xu_all[keep], y_all[keep]
        ymean, ystd = float(yloc.mean()), float(yloc.std() or 1.0)
        yz = (yloc - ymean) / ystd
        cand = _tr_candidates(center, tr.length, n_cand, rng)
        q = min(batch_size, max(1, max_evals - y_all.shape[0]))
        try:
            gp = _fit_surrogate(xu, yz, None, fit_kwargs)
            picks_u = _thompson_batch(gp, xu, yz, cand, q, rng)
        except Exception:  # GP/Cholesky failure -> shrink the region and retry next iteration  # noqa: BLE001
            tr.update(False)
            continue
        improved = False
        for pu in picks_u:
            xn = b[:, 0] + pu * span
            yn = evaluate(xn)
            x_all = np.vstack([x_all, xn])
            y_all = np.append(y_all, yn)
            if yn < best - 1e-3 * abs(best):
                best = yn
                improved = True
        tr.update(improved)

    idx = int(np.argmin(y_all))
    return {
        "x": x_all[idx],
        "y": sign * float(y_all[idx]),
        "X": x_all,
        "Y": sign * y_all,
        "n_restarts": restarts,
    }


__all__ = ["TrustRegion", "turbo_minimize"]
