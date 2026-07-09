"""``emulate`` -- a cheap forward surrogate for an expensive simulator, placed by acquisition under budget.

M1 (belief-driven interaction) and M3 (inversion-surrogate pair generation) both need to call a forward
map many times; when that map is a real simulator (a PDE solve, a Monte-Carlo physics run, ...) that cost
dominates. :func:`emulate` fits a GP surrogate of ``simulator`` over ``bounds`` from a *budget*-limited
number of true calls, places those calls where they most reduce the surrogate's own predictive variance
(active learning, not random sampling), and returns an :class:`Emulator` whose ``.predict`` gives a mean
and a calibrated standard deviation. Callers use the standard deviation directly: :meth:`Emulator.escalate_mask`
flags inputs where the surrogate is not to be trusted, so a caller (M1's planner, M3's pair generator) can
fall back to the true simulator exactly there instead of guessing a fixed re-run schedule.

**Discovery, and why nothing here is a new GP.** ``mixle.models.gaussian_process.GaussianProcessRegressor``
(the exact torch GP; ``mixle.doe`` already fits it as its default surrogate via
``mixle.doe.bayesopt._fit_surrogate``) is the only regression model this module touches --
``mixle.models.sparse_gaussian_process.SparseGaussianProcessRegressor`` (FITC) is for the ``n`` too large
for an exact GP to invert, which is not this card's regime (the whole point of a *budgeted* emulator is
that ``n`` stays small: a handful to a few hundred simulator calls). The placement logic is not new either:

* **Single fidelity** reuses :func:`mixle.doe.active.active_learning_design` verbatim -- ALC (Active
  Learning Cohn / IMSE), the integrated posterior-variance-reduction criterion, sequentially adds the
  candidate that most shrinks the surrogate's error *everywhere* over ``bounds``, which is exactly "cheap
  forward surrogate with uncertainty, placed by acquisition" for the single-fidelity case.
* **Multi-fidelity** reuses the GP-over-augmented-fidelity-coordinate construction from
  :func:`mixle.doe.multifidelity.multi_fidelity_minimize` (fit one GP on ``[x, s]``; pick the fidelity
  that buys the most target-variance reduction per unit cost) but swaps its *why pick this x* half: BOCA
  picks ``x`` by Expected Improvement (it is chasing an optimum), this module picks ``x`` by
  :func:`mixle.doe.active.alc_scores` at the target fidelity (it is chasing surrogate accuracy
  everywhere) -- active learning's ALC criterion transplanted onto multi-fidelity's cost-aware fidelity
  choice, not a new algorithm.

**Why not ``mixle.task.acquire``, despite the roadmap listing it as a dependency.** ``acquire()``'s
dispatch (:func:`mixle.task.acquire._proba_batch`) is built for models that emit a row-stochastic
``(n, k)`` categorical prediction over an already-materialized discrete pool -- text classifiers,
ensembles of them. A simulator here is a continuous, generally scalar-valued regression map over a
*bounded continuous domain*, not a finite pool of discrete items with class probabilities; forcing it
through ``acquire``'s ``predict_proba`` contract would mean discretizing the domain and inventing fake
class probabilities from a GP mean/std, which throws away exactly the calibrated uncertainty this module
exists to keep. The dependency is real in spirit, not in import: A5 established the "acquisition places
the next expensive call, active learning shape, not a hardcoded type" pattern in mixle.task; this module
is the continuous-domain sibling of it, reusing ``mixle.doe``'s acquisition machinery the same way A5
reuses ``mixle.epistemic.portfolio``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm

from mixle.doe.active import active_learning_design, alc_scores
from mixle.doe.bayesopt import _fit_surrogate
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube, random_design

__all__ = ["Emulator", "EmulatorReceipt", "emulate"]

_COVERAGE_Z = 1.0  # +/- 1 std -> the nominal two-sided coverage of a calibrated Gaussian error bar


@dataclass(frozen=True)
class EmulatorReceipt:
    """A measured, not asserted, report of an :class:`Emulator`'s own quality.

    ``held_out_rmse`` and ``coverage`` are computed against true-simulator calls that were *not* used
    to fit the surrogate (``n_holdout`` of them, carved out of ``budget`` before training starts).
    ``coverage`` is the empirical fraction of holdout points whose true value falls within the
    emulator's own ``mean +/- 1 std``; ``nominal_coverage`` is what that fraction should be if the
    error bars are calibrated (``~0.6827`` for a Gaussian posterior). ``cost_spent`` is the total
    simulator cost actually used (holdout + training; each single-fidelity call costs 1, each
    multi-fidelity call costs its fidelity's entry in ``costs``).
    """

    held_out_rmse: float
    coverage: float
    nominal_coverage: float
    n_holdout: int
    n_train: int
    cost_spent: float
    fidelities: tuple[float, ...] | None


class Emulator:
    """A fitted forward surrogate: ``.predict``, ``.escalate_mask``, ``.receipt``. Built by :func:`emulate`."""

    def __init__(
        self,
        gp: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        bounds: np.ndarray,
        target_fidelity: float | None,
        receipt: EmulatorReceipt,
    ) -> None:
        self._gp = gp
        self._x_train = x_train
        self._y_train = y_train
        self.bounds = bounds
        self._target_fidelity = target_fidelity
        self.receipt = receipt

    def _augmented(self, x: Any) -> np.ndarray:
        xs = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self._target_fidelity is not None:
            xs = np.column_stack([xs, np.full(xs.shape[0], self._target_fidelity)])
        return xs

    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)`` of the surrogate's posterior at ``x`` (always at the target fidelity)."""
        xs = self._augmented(x)
        mean, cov = self._gp.predict(self._x_train, self._y_train, xs, return_cov=True)
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return mean, std

    def escalate_mask(self, x: Any, tol: float) -> np.ndarray:
        """Return a boolean mask: ``True`` where the surrogate's std at ``x`` exceeds ``tol`` (escalate)."""
        _, std = self.predict(x)
        return std > float(tol)


def emulate(
    simulator: Callable[..., float],
    bounds: Bounds,
    *,
    budget: int,
    fidelities: Sequence[float] | None = None,
    costs: Sequence[float] | None = None,
    seed: int | RandomState | None = None,
    n_init: int | None = None,
    n_candidates: int = 256,
    n_reference: int = 128,
    holdout_frac: float = 0.2,
    method: str = "alc",
    fit_kwargs: dict[str, Any] | None = None,
) -> Emulator:
    """Fit a budget-limited GP surrogate of ``simulator`` over ``bounds``, placing calls by acquisition.

    ``simulator(x)`` (single fidelity) or ``simulator(x, s)`` (``fidelities`` given, ``s`` one of them)
    returns the true response at ``x``; ``budget`` is the total simulator *cost* available (single
    fidelity: 1 unit per call; multi-fidelity: ``costs`` per fidelity, default the fidelity value
    itself, mirroring :func:`mixle.doe.multifidelity.multi_fidelity_minimize`). A ``holdout_frac``
    slice of the budget is spent up front on Latin-hypercube points evaluated at the target (highest)
    fidelity and held out of training, purely to compute :class:`EmulatorReceipt`; the remainder trains
    the surrogate: single fidelity via :func:`mixle.doe.active.active_learning_design` (``method``
    ``"alc"`` or ``"alm"``; ``"random"`` places a plain Latin-hypercube design instead, for comparison),
    multi-fidelity via ALC-at-target-fidelity point choice plus BOCA-style cost-aware fidelity choice
    (see the module docstring). Returns a fitted :class:`Emulator`.
    """
    if int(budget) <= 0:
        raise ValueError("budget must be positive.")
    if method not in ("alc", "alm", "random"):
        raise ValueError("method must be 'alc', 'alm', or 'random'.")
    b = _as_bounds(bounds)
    d = b.shape[0]
    rng = _as_rng(seed)
    n_init = int(n_init) if n_init else 2 * d

    if fidelities is None:
        gp, x_train, y_train, x_hold, y_hold, cost_spent = _fit_single_fidelity(
            simulator,
            b,
            d,
            budget=int(budget),
            n_init=n_init,
            holdout_frac=holdout_frac,
            method=method,
            rng=rng,
            fit_kwargs=fit_kwargs,
        )
        target_fidelity = None
        fid_tuple = None
    else:
        gp, x_train, y_train, x_hold, y_hold, cost_spent, target_fidelity = _fit_multi_fidelity(
            simulator,
            b,
            d,
            budget=float(budget),
            fidelities=fidelities,
            costs=costs,
            n_init=n_init,
            holdout_frac=holdout_frac,
            n_candidates=n_candidates,
            n_reference=n_reference,
            rng=rng,
            fit_kwargs=fit_kwargs,
        )
        fid_tuple = tuple(float(s) for s in fidelities)

    receipt = _build_receipt(
        gp,
        x_train,
        y_train,
        x_hold,
        y_hold,
        target_fidelity=target_fidelity,
        cost_spent=cost_spent,
        fidelities=fid_tuple,
    )
    return Emulator(gp, x_train, y_train, b, target_fidelity, receipt)


def _fit_single_fidelity(
    simulator: Callable[[np.ndarray], float],
    b: np.ndarray,
    d: int,
    *,
    budget: int,
    n_init: int,
    holdout_frac: float,
    method: str,
    rng: RandomState,
    fit_kwargs: dict[str, Any] | None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n_holdout = max(d + 1, int(round(holdout_frac * budget)))
    if n_holdout >= budget:
        raise ValueError(f"budget={budget} is too small to reserve a holdout ({n_holdout} points) and train.")
    train_budget = budget - n_holdout
    if train_budget < n_init:
        raise ValueError(f"budget={budget} leaves only {train_budget} training calls, below n_init={n_init}.")

    x_hold = latin_hypercube(b, n_holdout, rng)
    y_hold = np.array([float(simulator(np.asarray(p, dtype=np.float64))) for p in x_hold], dtype=np.float64)

    if method == "random":
        x_train = random_design(b, train_budget, rng)
        y_train = np.array([float(simulator(p)) for p in x_train], dtype=np.float64)
    else:
        design = active_learning_design(
            simulator, b, n_init=n_init, max_evals=train_budget, method=method, seed=rng, fit_kwargs=fit_kwargs
        )
        x_train, y_train = design["X"], design["Y"]

    gp = _fit_surrogate(x_train, y_train, None, fit_kwargs)
    return gp, x_train, y_train, x_hold, y_hold, float(budget)


def _fit_multi_fidelity(
    simulator: Callable[[np.ndarray, float], float],
    b: np.ndarray,
    d: int,
    *,
    budget: float,
    fidelities: Sequence[float],
    costs: Sequence[float] | None,
    n_init: int,
    holdout_frac: float,
    n_candidates: int,
    n_reference: int,
    rng: RandomState,
    fit_kwargs: dict[str, Any] | None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    fids = np.asarray(fidelities, dtype=np.float64).ravel()
    if fids.size < 2:
        raise ValueError("fidelities must list at least two fidelity levels.")
    target = float(fids.max())
    cost_arr = fids if costs is None else np.asarray(costs, dtype=np.float64).ravel()
    cost_map = {float(s): float(c) for s, c in zip(fids, cost_arr)}
    if any(c <= 0.0 for c in cost_map.values()):
        raise ValueError(f"emulate requires every fidelity cost > 0, got {cost_map}")

    n_holdout = max(d + 1, int(round(holdout_frac * budget / cost_map[target])))
    hold_cost = n_holdout * cost_map[target]
    if hold_cost >= budget:
        raise ValueError(f"budget={budget} is too small to reserve a target-fidelity holdout costing {hold_cost}.")
    max_cost = budget - hold_cost

    x_hold = latin_hypercube(b, n_holdout, rng)
    y_hold = np.array([float(simulator(np.asarray(p, dtype=np.float64), target)) for p in x_hold], dtype=np.float64)

    rows: list[np.ndarray] = []
    y: list[float] = []
    for s in fids:  # seed every fidelity, same as multi_fidelity_minimize
        for xx in latin_hypercube(b, n_init, rng):
            rows.append(np.append(xx, s))
            y.append(float(simulator(np.asarray(xx, dtype=np.float64), float(s))))
    x_aug = np.asarray(rows, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    spent = float(sum(cost_map[float(s)] for s in x_aug[:, -1]))

    while spent < max_cost:
        try:
            gp = _fit_surrogate(x_aug, y_arr, None, fit_kwargs)
        except Exception:  # noqa: BLE001 -- GP fit can fail on ill-conditioned data; stop gracefully
            break
        cand = latin_hypercube(b, int(n_candidates), rng)
        ref = latin_hypercube(b, int(n_reference), rng)
        cand_t = np.column_stack([cand, np.full(cand.shape[0], target)])
        ref_t = np.column_stack([ref, np.full(ref.shape[0], target)])
        # x* by ALC (integrated variance reduction) at the target fidelity -- accuracy everywhere, not EI.
        alc = alc_scores(gp, x_aug, y_arr, cand_t, ref_t)
        xstar = cand[int(np.argmax(alc))]

        # which fidelity to actually spend: the one that most reduces target-fidelity variance per unit
        # cost (BOCA's fidelity choice, unchanged from mixle.doe.multifidelity.multi_fidelity_minimize).
        best_s, best_score = target, -np.inf
        for s in fids:
            pts = np.array([np.append(xstar, target), np.append(xstar, float(s))])
            _, c2 = gp.predict(x_aug, y_arr, pts, return_cov=True)
            c2 = np.atleast_2d(np.asarray(c2, dtype=np.float64))
            var_reduction = c2[0, 1] ** 2 / max(c2[1, 1], 1e-12)
            score = var_reduction / cost_map[float(s)]
            if score > best_score:
                best_score, best_s = score, float(s)

        yn = float(simulator(np.asarray(xstar, dtype=np.float64), best_s))
        x_aug = np.vstack([x_aug, np.append(xstar, best_s)])
        y_arr = np.append(y_arr, yn)
        spent += cost_map[best_s]

    gp = _fit_surrogate(x_aug, y_arr, None, fit_kwargs)
    return gp, x_aug, y_arr, x_hold, y_hold, spent + hold_cost, target


def _build_receipt(
    gp: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_hold: np.ndarray,
    y_hold: np.ndarray,
    *,
    target_fidelity: float | None,
    cost_spent: float,
    fidelities: tuple[float, ...] | None,
) -> EmulatorReceipt:
    xq = x_hold if target_fidelity is None else np.column_stack([x_hold, np.full(x_hold.shape[0], target_fidelity)])
    mean, cov = gp.predict(x_train, y_train, xq, return_cov=True)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    std = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    rmse = float(np.sqrt(np.mean((mean - y_hold) ** 2)))
    covered = np.abs(y_hold - mean) <= _COVERAGE_Z * std
    coverage = float(np.mean(covered))
    nominal_coverage = float(2.0 * norm.cdf(_COVERAGE_Z) - 1.0)
    return EmulatorReceipt(
        held_out_rmse=rmse,
        coverage=coverage,
        nominal_coverage=nominal_coverage,
        n_holdout=int(x_hold.shape[0]),
        n_train=int(x_train.shape[0]),
        cost_spent=float(cost_spent),
        fidelities=fidelities,
    )
