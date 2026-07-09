"""Gaussian-process Bayesian optimization over a bounded input space (WS-E).

A sequential model-based optimization loop: fit a GP surrogate to the observed points, score
Latin-hypercube candidates with an acquisition function, and evaluate the best candidate next.
Reuses :class:`mixle.models.gaussian_process.GaussianProcessRegressor` (torch) as the surrogate;
the acquisition functions themselves are torch-free numpy.

Acquisitions are looked up through a small registry (``register_acquisition`` / ``acq=`` name) --
the same "register, don't branch" pattern as the engines and encoded-data backends -- so a new
acquisition plugs in without editing the proposal loop. Built in: expected improvement (``"ei"``),
probability of improvement (``"pi"``), and the upper/lower confidence bound (``"ucb"``). Each takes
``(mean, std, best, *, maximize, **params)`` and returns a *merit* array that is maximized over the
candidate set. Batch proposals (``propose_batch``) use the kriging-believer heuristic: fantasize the
posterior mean at each pick, refit, and repeat, giving a spatially diverse batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import erfcx, ndtr

from mixle.doe._contracts import Acquisition, Surrogate
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


def expected_improvement(
    mean: Any, std: Any, best: float, xi: float = 0.0, *, maximize: bool = False, **_: Any
) -> np.ndarray:
    """Return the expected-improvement acquisition at points with surrogate ``mean`` and ``std``.

    For minimization the improvement over the incumbent ``best`` is ``best - mean - xi``; for
    maximization it is ``mean - best - xi``. ``xi >= 0`` trades exploration for exploitation.
    Points with zero predictive ``std`` get zero EI. Higher is better (maximized over candidates).
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    improve = (mean - best - xi) if maximize else (best - mean - xi)
    ei = np.zeros_like(std)
    pos = std > 1.0e-12
    z = np.zeros_like(std)
    z[pos] = improve[pos] / std[pos]
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei[pos] = improve[pos] * ndtr(z[pos]) + std[pos] * pdf[pos]
    return np.maximum(ei, 0.0)


def log_expected_improvement(
    mean: Any, std: Any, best: float, xi: float = 0.0, *, maximize: bool = False, **_: Any
) -> np.ndarray:
    """Return the log expected-improvement acquisition -- a numerically stable EI (Ament et al. 2023).

    Mathematically ``log(EI)``, but computed so it stays finite and informative deep in the
    no-improvement tail where ``EI`` itself underflows to 0 (and ``log EI`` to ``-inf``), keeping the
    optimizer's ordering and gradients usable. Same argmax as :func:`expected_improvement`; points with
    zero predictive ``std`` get ``-inf``. Higher is better. The ``z >= 0`` branch is the direct
    well-conditioned form; the ``z < 0`` tail uses the scaled complementary error function ``erfcx``
    (``Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt2)``), which is bounded there and so never under/overflows.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    improve = (mean - best - xi) if maximize else (best - mean - xi)
    out = np.full(std.shape, -np.inf, dtype=np.float64)
    pos = std > 1.0e-300
    z = improve[pos] / std[pos]
    log_h = np.empty_like(z)
    neg = z < 0.0
    zp = z[~neg]
    log_h[~neg] = np.log(zp * ndtr(zp) + np.exp(-0.5 * zp * zp) / np.sqrt(2.0 * np.pi))
    zn = z[neg]
    mills = np.sqrt(np.pi / 2.0) * erfcx(-zn / np.sqrt(2.0))
    log_h[neg] = -0.5 * zn * zn - 0.5 * np.log(2.0 * np.pi) + np.log1p(zn * mills)
    out[pos] = np.log(std[pos]) + log_h
    return out


def probability_of_improvement(
    mean: Any, std: Any, best: float, *, maximize: bool = False, xi: float = 0.0, **_: Any
) -> np.ndarray:
    """Return the probability-of-improvement acquisition.

    The probability that a candidate improves on the incumbent ``best`` by at least ``xi``:
    ``P(f < best - xi)`` for minimization, ``P(f > best + xi)`` for maximization. Where the
    predictive ``std`` is zero the improvement is deterministic (1.0 if it improves, else 0.0).
    Higher is better.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    improve = (mean - best - xi) if maximize else (best - mean - xi)
    pi = np.zeros_like(std)
    pos = std > 1.0e-12
    pi[pos] = ndtr(improve[pos] / std[pos])
    pi[~pos] = (improve[~pos] > 0.0).astype(np.float64)
    return pi


def upper_confidence_bound(
    mean: Any, std: Any, best: float = 0.0, *, maximize: bool = False, kappa: float = 1.96, **_: Any
) -> np.ndarray:
    """Return the confidence-bound acquisition (UCB for maximization, LCB for minimization).

    Returns a merit that is maximized over candidates: the optimistic bound ``mean + kappa * std``
    when maximizing, and ``kappa * std - mean`` when minimizing (so picking the largest merit
    selects the most promising low-objective point). ``kappa >= 0`` trades exploration for
    exploitation; ``best`` is ignored. Higher is better.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return (mean + kappa * std) if maximize else (kappa * std - mean)


def thompson_sampling(
    mean: Any, std: Any, best: float = 0.0, *, maximize: bool = False, rng: Any = None, **_: Any
) -> np.ndarray:
    """Thompson-sampling acquisition: one marginal posterior draw ``N(mean, std)`` per candidate.

    Returns a merit (maximized over candidates) equal to the drawn value when maximizing and its
    negation when minimizing, so the selected point is the optimum of the *sampled* objective. A
    randomized, exploration-aware acquisition -- repeated proposals explore competing optima in
    proportion to posterior probability, with no exploration knob to tune. This is the low-cost *marginal*
    variant (independent per-candidate draws, ignoring the GP's cross-candidate correlation); pass an
    ``rng`` (via ``acq_kwargs``) for reproducible proposals. ``best`` is ignored.
    """
    rng = rng if rng is not None else np.random.RandomState()
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    draw = mean + std * rng.standard_normal(mean.shape)
    return draw if maximize else -draw


# --- acquisition registry ("register, don't branch") --------------------------------------------
# An acquisition is ``fn(mean, std, best, *, maximize, **params) -> merit`` where ``merit`` is
# maximized over the candidate set. Built-ins are registered below; third parties register their own.
_ACQUISITIONS: dict[str, Acquisition] = {}


def register_acquisition(name: str, fn: Acquisition, aliases: tuple[str, ...] = ()) -> None:
    """Register an acquisition ``fn`` under ``name`` (and any ``aliases``).

    ``fn`` is called as ``fn(mean, std, best, *, maximize, **params)`` and must return a merit array
    that the proposal loop maximizes over candidates. This is the extension point for new
    acquisitions -- registering is all that is needed, no edits to ``propose_next``/``minimize``.
    """
    if not callable(fn):
        raise TypeError("acquisition must be callable.")
    _ACQUISITIONS[name.lower()] = fn
    for alias in aliases:
        _ACQUISITIONS[alias.lower()] = fn


def available_acquisitions() -> list[str]:
    """Return the sorted names (and aliases) of all registered acquisitions."""
    return sorted(_ACQUISITIONS)


def _get_acquisition(acq: str | Acquisition) -> Acquisition:
    if callable(acq):
        return acq
    fn = _ACQUISITIONS.get(str(acq).lower())
    if fn is None:
        raise ValueError("unknown acquisition %r; registered: %s" % (acq, ", ".join(available_acquisitions())))
    return fn


register_acquisition("expected_improvement", expected_improvement, aliases=("ei",))
register_acquisition("log_expected_improvement", log_expected_improvement, aliases=("logei", "log_ei"))
register_acquisition("probability_of_improvement", probability_of_improvement, aliases=("pi",))
register_acquisition("upper_confidence_bound", upper_confidence_bound, aliases=("ucb", "lcb", "confidence_bound", "cb"))
register_acquisition("thompson_sampling", thompson_sampling, aliases=("thompson", "ts"))


@dataclass(frozen=True)
class OptimizationResult:
    """Common outcome of a model-based optimization run: the full evaluation history.

    ``x`` is the ``(N, d)`` matrix of evaluated points and ``y`` the corresponding objective
    values (an ``(N,)`` vector for single-objective runs, an ``(N, M)`` matrix for multi-objective).
    Concrete result types extend this with their best-point / Pareto-front fields.
    """

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class BayesOptResult(OptimizationResult):
    """Outcome of a Bayesian-optimization run."""

    best_x: np.ndarray
    best_y: float


def _fit_surrogate(x: np.ndarray, y: np.ndarray, gp: Surrogate | None, fit_kwargs: dict[str, Any] | None) -> Surrogate:
    if gp is None:
        from mixle.models.gaussian_process import GaussianProcessRegressor

        # np.std([]) is nan (with a RuntimeWarning), and `nan or 1.0` evaluates to nan -- nan is
        # truthy, so the `or` fallback only ever caught the exact-zero-variance case, not the
        # empty-y case. An empty y is a real, documented path: BayesianOptimizer.ask(q) with
        # q > n_init before any tell() calls propose_batch with zero observations.
        std = float(np.std(y)) if y.size > 0 else 0.0
        scale = std if std > 0.0 else 1.0
        gp = GaussianProcessRegressor(lengthscale=1.0, amplitude=scale, noise=0.1 * scale + 1.0e-6)
    kwargs = {"out": None, **(fit_kwargs or {})}
    gp.fit(x, y, **kwargs)
    return gp


def _propose_one(
    x: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    rng: RandomState,
    *,
    maximize: bool,
    acq_fn: Acquisition,
    acq_kwargs: dict[str, Any],
    n_candidates: int,
    gp: Surrogate | None,
    fit_kwargs: dict[str, Any] | None,
) -> tuple[np.ndarray, float, Surrogate]:
    """Fit the surrogate, score Latin-hypercube candidates, return (best point, its merit, fitted gp)."""
    if int(n_candidates) <= 0:
        raise ValueError("n_candidates must be positive.")
    if y.size == 0:
        # np.min/np.max on an empty y crashes with an opaque "zero-size array" ValueError. This is a
        # real, reachable path: BayesianOptimizer.ask(q) with q > n_init before any tell() calls
        # propose_next/propose_batch with zero observations -- there is no incumbent to score
        # acquisition against yet, so name that clearly instead of a generic numpy crash.
        raise ValueError("cannot propose an acquisition-based point with zero observations; call tell() first.")
    gp = _fit_surrogate(x, y, gp, fit_kwargs)
    candidates = latin_hypercube(b, n_candidates, rng)
    mean, cov = gp.predict(x, y, candidates, return_cov=True)
    std = np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=np.float64)), 0.0, None))
    best = float(np.max(y)) if maximize else float(np.min(y))
    merit = np.asarray(
        acq_fn(np.asarray(mean, dtype=np.float64), std, best, maximize=maximize, **acq_kwargs), dtype=np.float64
    )
    idx = int(np.argmax(merit))
    return candidates[idx], float(merit[idx]), gp


def _kg_inner(a: np.ndarray, b: np.ndarray) -> float:
    """``E[max_i (a_i + b_i Z)] - max_i a_i`` for ``Z ~ N(0, 1)`` (Frazier 2009, Algorithm 1)."""
    order = np.lexsort((a, b))  # sort by slope b ascending, ties broken by intercept a
    a = a[order]
    b = b[order]
    keep = np.append(np.diff(b) > 0.0, True)  # equal-slope lines: keep only the highest intercept
    a = a[keep]
    b = b[keep]
    idx = [0]
    cross = [-np.inf]
    for i in range(1, len(b)):
        while True:
            j = idx[-1]
            c = (a[j] - a[i]) / (b[i] - b[j])
            if len(idx) > 1 and c <= cross[-1]:
                idx.pop()
                cross.pop()
            else:
                break
        idx.append(i)
        cross.append(c)
    a = a[idx]
    b = b[idx]
    c = np.array([*cross, np.inf])
    cl, cr = c[:-1], c[1:]
    pdf = np.exp(-0.5 * c * c) / np.sqrt(2.0 * np.pi)
    return float(np.sum(a * (ndtr(cr) - ndtr(cl)) + b * (pdf[:-1] - pdf[1:])) - a.max())


def knowledge_gradient(mean: Any, cov: Any, noise: float = 1.0e-6) -> np.ndarray:
    """Knowledge-gradient acquisition value of one observation at each candidate (Frazier et al. 2009).

    Given the Gaussian-process posterior ``mean`` and joint ``cov`` over a candidate set, returns, for
    each candidate ``x``, the expected increase in the best posterior mean after fantasizing one
    (noisy) observation there: ``KG(x) = E_y[max_x' mu_{n+1}(x')] - max_x' mu_n(x')``. Maximizing KG is
    the one-step Bayes-optimal, look-ahead choice (it values *information*, not just immediate
    improvement), so it explores where an observation would most change the believed optimum. Assumes a
    *maximization* objective; negate ``mean`` for minimization. Computed exactly via the piecewise-linear
    epigraph of the fantasized posterior means, and ``>= 0`` by construction.
    """
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    out = np.empty(mean.size)
    for x in range(mean.size):
        sigma_x = np.sqrt(max(cov[x, x] + noise, 1.0e-12))
        out[x] = _kg_inner(mean.copy(), cov[:, x] / sigma_x)
    return out


def propose_knowledge_gradient(
    x: Any,
    y: Any,
    bounds: Any,
    *,
    maximize: bool = False,
    n_candidates: int = 512,
    seed: int | None = None,
    gp: Surrogate | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    noise: float = 1.0e-6,
) -> np.ndarray:
    """Propose the next evaluation point by maximizing the knowledge gradient over a candidate set.

    Fits the GP surrogate to ``(x, y)``, draws ``n_candidates`` Latin-hypercube points, evaluates the
    joint posterior, and returns the candidate with the largest :func:`knowledge_gradient` -- the
    look-ahead Bayesian-optimization proposal. ``maximize`` selects the objective sense (the mean is
    negated for minimization).
    """
    if int(n_candidates) <= 0:
        raise ValueError("n_candidates must be positive.")
    x, y = _validate_xy(x, y)
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    gp = _fit_surrogate(x, y, gp, fit_kwargs)
    candidates = latin_hypercube(b, n_candidates, rng)
    mean, cov = gp.predict(x, y, candidates, return_cov=True)
    mean = np.asarray(mean, dtype=np.float64)
    signed_mean = mean if maximize else -mean  # KG is defined for maximization
    kg = knowledge_gradient(signed_mean, np.asarray(cov, dtype=np.float64), noise)
    return candidates[int(np.argmax(kg))]


def _validate_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations.")
    return x, y


def propose_next(
    x: Any,
    y: Any,
    bounds: Bounds,
    n_candidates: int = 512,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    gp: Surrogate | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    return_acquisition: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Propose the next point to evaluate by maximizing an acquisition function.

    Fits a GP to ``(x, y)``, scores ``n_candidates`` Latin-hypercube points by the ``acq`` acquisition
    (``"ei"`` / ``"pi"`` / ``"ucb"`` or any registered name / callable), and returns the best candidate
    (a ``(d,)`` array), optionally with its merit. ``xi`` is forwarded to acquisitions that use it
    (EI, PI); per-acquisition parameters such as ``kappa`` go in ``acq_kwargs``.
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    x, y = _validate_xy(x, y)
    acq_fn = _get_acquisition(acq)
    kw = {"xi": xi, **(acq_kwargs or {})}
    point, merit, _ = _propose_one(
        x,
        y,
        b,
        rng,
        maximize=maximize,
        acq_fn=acq_fn,
        acq_kwargs=kw,
        n_candidates=n_candidates,
        gp=gp,
        fit_kwargs=fit_kwargs,
    )
    if return_acquisition:
        return point, merit
    return point


def propose_batch(
    x: Any,
    y: Any,
    bounds: Bounds,
    q: int,
    n_candidates: int = 512,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Propose a batch of ``q`` points to evaluate together, via the kriging-believer heuristic.

    Each pick maximizes the acquisition; the GP posterior mean at the chosen point is then appended
    as a fantasized observation and the surrogate refit, so the next pick is steered away from it.
    Returns a ``(q, d)`` array. This needs no true objective evaluations between picks, so it suits
    parallel/asynchronous experiment campaigns.
    """
    if int(q) <= 0:
        raise ValueError("q must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    acq_fn = _get_acquisition(acq)
    kw = {"xi": xi, **(acq_kwargs or {})}
    picks = []
    for _ in range(int(q)):
        point, _, gp = _propose_one(
            xs,
            ys,
            b,
            rng,
            maximize=maximize,
            acq_fn=acq_fn,
            acq_kwargs=kw,
            n_candidates=n_candidates,
            gp=None,
            fit_kwargs=fit_kwargs,
        )
        picks.append(point)
        fantasy = np.asarray(gp.predict(xs, ys, point[None, :], return_cov=False), dtype=np.float64).reshape(-1)[0]
        xs = np.vstack([xs, point[None, :]])
        ys = np.append(ys, float(fantasy))
    return np.asarray(picks, dtype=np.float64)


def minimize(
    objective: Callable[[np.ndarray], float],
    bounds: Bounds,
    n_init: int = 5,
    n_iter: int = 15,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    n_candidates: int = 512,
    fit_kwargs: dict[str, Any] | None = None,
) -> BayesOptResult:
    """Run sequential GP Bayesian optimization of a scalar ``objective`` over ``bounds``.

    Seeds with an ``n_init``-point Latin-hypercube design, then runs ``n_iter`` acquisition-driven
    steps using ``acq`` (``"ei"`` by default; also ``"pi"`` / ``"ucb"`` or any registered acquisition).
    Minimizes by default; set ``maximize=True`` to maximize. ``objective`` takes a ``(d,)`` point and
    returns a float.
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    if n_init <= 0:
        raise ValueError("n_init must be positive.")

    x = latin_hypercube(b, n_init, rng)
    y = np.array([float(objective(np.asarray(row, dtype=np.float64))) for row in x], dtype=np.float64)

    for _ in range(int(n_iter)):
        nxt = propose_next(
            x,
            y,
            b,
            n_candidates=n_candidates,
            seed=rng,
            maximize=maximize,
            xi=xi,
            acq=acq,
            acq_kwargs=acq_kwargs,
            fit_kwargs=fit_kwargs,
        )
        nxt = np.asarray(nxt, dtype=np.float64)
        x = np.vstack([x, nxt[None, :]])
        y = np.append(y, float(objective(nxt)))

    best_idx = int(np.argmax(y)) if maximize else int(np.argmin(y))
    return BayesOptResult(best_x=x[best_idx], best_y=float(y[best_idx]), x=x, y=y)


__all__: Sequence[str] = [
    "Acquisition",
    "Surrogate",
    "expected_improvement",
    "log_expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound",
    "thompson_sampling",
    "knowledge_gradient",
    "propose_knowledge_gradient",
    "register_acquisition",
    "available_acquisitions",
    "propose_next",
    "propose_batch",
    "minimize",
    "OptimizationResult",
    "BayesOptResult",
]
