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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe.batch import _safe_cholesky
from mixle.doe.bayesopt import _fit_surrogate, _require_finite_scalar
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, _require_exact_positive_int, latin_hypercube


@dataclass(frozen=True)
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
    _success: int = field(default=0, init=False, repr=False)
    _failure: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        dim = _require_exact_positive_int(self.dim, "dim")
        success_tol = _require_exact_positive_int(self.success_tol, "success_tol")
        failure_tol = _require_exact_positive_int(self.failure_tol, "failure_tol", minimum=0)
        if failure_tol <= 0:
            failure_tol = max(4, dim)
        length_min = _require_finite_scalar(self.length_min, "length_min")
        if length_min <= 0.0:
            raise ValueError(f"length_min must be positive, got {length_min!r}.")
        length_max = _require_finite_scalar(self.length_max, "length_max")
        if length_max < length_min:
            raise ValueError(
                f"length_max must be >= length_min, got length_max={length_max!r}, length_min={length_min!r}."
            )
        length = _require_finite_scalar(self.length, "length")
        if not (length_min <= length <= length_max):
            raise ValueError(
                f"length must satisfy length_min <= length <= length_max, got length={length!r}, "
                f"length_min={length_min!r}, length_max={length_max!r}."
            )
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "success_tol", success_tol)
        object.__setattr__(self, "failure_tol", failure_tol)
        object.__setattr__(self, "length_min", length_min)
        object.__setattr__(self, "length_max", length_max)
        object.__setattr__(self, "length", length)

    @property
    def collapsed(self) -> bool:
        """Return whether the trust-region length has shrunk below its usable minimum."""
        return self.length < self.length_min

    def update(self, improved: bool) -> TrustRegion:
        """Return the next validated state after recording one batch outcome.

        States are immutable: callers must retain the returned instance. This prevents a region that
        passed construction validation from later being assigned invalid geometry.
        """
        if type(improved) is not bool:
            raise TypeError(f"improved must be a bool, got {type(improved).__name__}.")
        if self.collapsed:
            raise RuntimeError("A collapsed trust region must be restarted before it can be updated.")
        if improved:
            success = self._success + 1
            failure = 0
        else:
            failure = self._failure + 1
            success = 0
        length = self.length
        if success >= self.success_tol:
            length = min(length * 2.0, self.length_max)
            success = 0
        if failure >= self.failure_tol:
            length /= 2.0
            failure = 0

        # Construct from the still-valid pre-transition geometry, then install the derived state.
        # ``length`` may legitimately be below ``length_min`` only as the result of this transition;
        # public construction still rejects an initially collapsed region.
        result = TrustRegion(
            dim=self.dim,
            length=self.length,
            length_min=self.length_min,
            length_max=self.length_max,
            success_tol=self.success_tol,
            failure_tol=self.failure_tol,
        )
        object.__setattr__(result, "length", length)
        object.__setattr__(result, "_success", success)
        object.__setattr__(result, "_failure", failure)
        return result


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
    q = _require_exact_positive_int(q, "q")
    cand = np.asarray(cand, dtype=np.float64)
    if cand.ndim != 2 or cand.shape[0] == 0 or cand.shape[1] == 0:
        raise ValueError(f"cand must be a non-empty 2-D array, got shape {cand.shape}.")
    if not np.all(np.isfinite(cand)):
        raise ValueError("cand must contain only finite values.")
    if q > cand.shape[0]:
        # once every candidate index is chosen, the inner loop finds nothing left to pick and that
        # round silently contributes NOTHING to `picks` -- the caller would get fewer than q points
        # back with no error. Name the actual constraint instead.
        raise ValueError(f"_thompson_batch requires q <= cand.shape[0] (q={q}, candidates={cand.shape[0]}).")
    mean_raw, cov_raw = gp.predict(xn, yn, cand, return_cov=True)
    mean = np.asarray(mean_raw, dtype=np.float64)
    if mean.shape == (cand.shape[0], 1):
        mean = mean[:, 0]
    elif mean.shape != (cand.shape[0],):
        raise np.linalg.LinAlgError(
            f"GP posterior mean must have shape ({cand.shape[0]},) or ({cand.shape[0]}, 1), got {mean.shape}."
        )
    if not np.all(np.isfinite(mean)):
        raise np.linalg.LinAlgError("GP posterior mean must contain only finite values.")
    cov = np.asarray(cov_raw, dtype=np.float64)
    expected_cov_shape = (cand.shape[0], cand.shape[0])
    if cov.shape != expected_cov_shape:
        raise np.linalg.LinAlgError(f"GP posterior covariance must have shape {expected_cov_shape}, got {cov.shape}.")
    if not np.all(np.isfinite(cov)):
        raise np.linalg.LinAlgError("GP posterior covariance must contain only finite values.")
    scale = max(float(np.max(np.abs(cov))), np.finfo(np.float64).tiny)
    symmetry_tol = 64.0 * np.finfo(np.float64).eps * scale
    asymmetry = float(np.max(np.abs(cov - cov.T)))
    if asymmetry > symmetry_tol:
        raise np.linalg.LinAlgError(
            f"GP posterior covariance must be symmetric within {symmetry_tol:.3g}; "
            f"maximum asymmetry is {asymmetry:.3g}."
        )
    try:
        chol = _safe_cholesky(cov)
    except ValueError as exc:
        # _safe_cholesky raises plain ValueError for every one of its own failure modes (non-finite
        # covariance, not positive-semidefinite within tolerance, or PSD but still unfactorable after its
        # full jitter budget) -- each one IS a genuine numerical GP-posterior failure, the same family
        # turbo_minimize's loop already treats as "shrink the region and retry" (MXR-080-0197). Re-tag it
        # as LinAlgError here, at the one call site that KNOWS this particular ValueError means that, so
        # the loop's narrowed handler can catch it without also risking catching an unrelated ValueError
        # from somewhere else (e.g. this function's own q <= cand.shape[0] contract check above, which
        # must propagate as the programming-error signal it is, not be silently relabeled "model failure").
        raise np.linalg.LinAlgError(str(exc)) from exc
    picks: list[np.ndarray] = []
    chosen: set[int] = set()
    for _ in range(q):
        sample = mean + chol @ rng.standard_normal(mean.size)
        if not np.all(np.isfinite(sample)):
            raise np.linalg.LinAlgError("GP posterior draw produced non-finite values.")
        for idx in np.argsort(sample):
            if int(idx) not in chosen:
                chosen.add(int(idx))
                picks.append(cand[int(idx)])
                break
    if len(picks) != q or len(chosen) != q:
        raise np.linalg.LinAlgError(f"Thompson sampling produced {len(picks)} distinct picks; expected exactly {q}.")
    return np.asarray(picks, dtype=np.float64).reshape(q, cand.shape[1])


def _numerical_fit_error_types() -> tuple[type[BaseException], ...]:
    """The well-defined numerical failure types the TuRBO loop's GP fit/posterior step can raise.

    Mirrors :func:`mixle.doe.multifidelity._surrogate_fit_error_types` (MXR-080-0183) but covers this
    loop's full numerical surface: a singular or indefinite covariance during GP hyperparameter fitting
    (:func:`mixle.doe.bayesopt._fit_surrogate`) or posterior sampling (:func:`_thompson_batch`, which
    re-raises :func:`mixle.doe.batch._safe_cholesky`'s ``ValueError`` as ``np.linalg.LinAlgError`` at its
    call site specifically so it funnels through this same tuple) surfaces as ``numpy.linalg.LinAlgError``
    (a numpy-backed surrogate) or ``torch.linalg.LinAlgError`` (the default torch-backed one, out of
    ``torch.linalg.cholesky`` inside ``GaussianProcessRegressor.log_marginal_likelihood``/``predict``) --
    both a genuine "this data/posterior is ill-conditioned" failure, never a configuration or programming
    error. Anything else (a bad ``fit_kwargs`` optimizer name, a missing torch install, an unrelated bug)
    must propagate instead of being caught here and silently relabeled "model failure" (MXR-080-0197).
    Torch is imported lazily, matching the rest of this module's lazy torch usage, so merely importing
    this module never requires torch -- the tuple just narrows to numpy alone when torch is not installed.
    """
    errors: tuple[type[BaseException], ...] = (np.linalg.LinAlgError,)
    try:
        import torch
    except ImportError:
        return errors
    return (*errors, torch.linalg.LinAlgError)


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

    A well-defined numerical failure while fitting the local GP or sampling its joint posterior (a
    singular or indefinite covariance -- ``numpy``/``torch`` ``linalg.LinAlgError``) shrinks the trust
    region and retries next iteration, the same policy as an ordinary non-improving batch. Anything else
    (a configuration or programming bug -- a bad ``fit_kwargs`` optimizer name, a missing torch install,
    ...) propagates instead of being silently caught and relabeled "model failure" (MXR-080-0197).

    Returns ``{'x', 'y', 'X', 'Y', 'n_restarts', 'n_bayes_batches', 'n_fit_failures', 'last_fit_error',
    'degraded_to_random_search', 'stopped_reason'}``: the best point/value and the full evaluation
    history; how many trust-region collapses restarted from a fresh Latin-hypercube design
    (``n_restarts``); how many batches were genuine GP-guided Thompson proposals (``n_bayes_batches``)
    versus caught numerical failures (``n_fit_failures``, with the most recent one's diagnostic string in
    ``last_fit_error``, else ``None``); ``degraded_to_random_search`` is True iff *every* batch attempt
    failed numerically, meaning the entire result came from Latin-hypercube restarts and the GP-guided
    search never once ran; and ``stopped_reason`` is always ``"budget_exhausted"`` (this loop's only
    normal termination) for any call that returns rather than raises.
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
    if type(maximize) is not bool:
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}.")
    sign = -1.0 if maximize else 1.0  # always minimize sign*objective

    def to_unit(x):
        return (x - b[:, 0]) / span

    attempted_evals = 0
    failed_evaluations: list[dict[str, Any]] = []

    def evaluate(x: np.ndarray) -> float | None:
        nonlocal attempted_evals
        point = np.array(x, dtype=np.float64, copy=True)
        attempted_evals += 1
        raw_value = float(objective(point.copy()))
        if not np.isfinite(raw_value):
            failed_evaluations.append(
                {
                    "evaluation": attempted_evals,
                    "x": point,
                    "status": "nonfinite_observation",
                    "observation": raw_value,
                }
            )
            return None
        return sign * raw_value

    x_rows: list[np.ndarray] = []
    y_values: list[float] = []
    tr = TrustRegion(dim=d)
    restarts = 0
    n_bayes_batches = 0
    n_fit_failures = 0
    last_fit_error: str | None = None
    fit_error_types = _numerical_fit_error_types()

    def result(stopped_reason: str) -> dict[str, Any]:
        x_all = np.asarray(x_rows, dtype=np.float64).reshape(-1, d)
        y_all = np.asarray(y_values, dtype=np.float64)
        if y_all.size:
            idx = int(np.argmin(y_all))
            best_x: np.ndarray | None = x_all[idx].copy()
            best_y: float | None = sign * float(y_all[idx])
        else:
            best_x = None
            best_y = None
        return {
            "x": best_x,
            "y": best_y,
            "X": x_all,
            "Y": sign * y_all,
            "n_evaluations": attempted_evals,
            "failed_evaluations": tuple(failed_evaluations),
            "n_restarts": restarts,
            "n_bayes_batches": n_bayes_batches,
            "n_fit_failures": n_fit_failures,
            "last_fit_error": last_fit_error,
            "degraded_to_random_search": n_bayes_batches == 0,
            "stopped_reason": stopped_reason,
        }

    initial_design = latin_hypercube(b, n_init, rng)
    for point in initial_design:
        value = evaluate(point)
        if value is None:
            return result("objective_failed")
        x_rows.append(np.array(point, dtype=np.float64, copy=True))
        y_values.append(value)

    x_all = np.asarray(x_rows, dtype=np.float64)
    y_all = np.asarray(y_values, dtype=np.float64)
    best = float(y_all.min())

    while attempted_evals < max_evals:
        if tr.collapsed:
            restarts += 1
            # clamp the restart design to the REMAINING budget -- an unclamped n_init-point design
            # can overshoot max_evals by up to n_init real objective calls if the trust region
            # collapses near the end of the run (a real, common occurrence on hard landscapes).
            n_restart = min(n_init, max_evals - attempted_evals)
            xr = latin_hypercube(b, n_restart, rng)
            for point in xr:
                value = evaluate(point)
                if value is None:
                    return result("objective_failed")
                x_rows.append(np.array(point, dtype=np.float64, copy=True))
                y_values.append(value)
            x_all = np.asarray(x_rows, dtype=np.float64)
            y_all = np.asarray(y_values, dtype=np.float64)
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
        q = min(batch_size, max_evals - attempted_evals)
        try:
            gp = _fit_surrogate(xu, yz, None, fit_kwargs)
            picks_u = _thompson_batch(gp, xu, yz, cand, q, rng)
        except fit_error_types as exc:
            # A well-defined numerical failure (singular/indefinite GP covariance) -- shrink the region
            # and retry next iteration, the same policy as an ordinary non-improving batch. Anything
            # else (a configuration or programming bug) is NOT in fit_error_types and propagates instead
            # of being silently caught and relabeled "model failure" (MXR-080-0197).
            n_fit_failures += 1
            last_fit_error = f"{type(exc).__name__}: {exc}"
            tr = tr.update(False)
            continue
        n_bayes_batches += 1
        improved = False
        for pu in picks_u:
            xn = b[:, 0] + pu * span
            yn = evaluate(xn)
            if yn is None:
                return result("objective_failed")
            x_rows.append(np.array(xn, dtype=np.float64, copy=True))
            y_values.append(yn)
            if yn < best - 1e-3 * abs(best):
                best = yn
                improved = True
        x_all = np.asarray(x_rows, dtype=np.float64)
        y_all = np.asarray(y_values, dtype=np.float64)
        tr = tr.update(improved)

    return result("budget_exhausted")


__all__ = ["TrustRegion", "turbo_minimize"]
