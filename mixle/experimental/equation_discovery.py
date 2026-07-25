"""P8 (experimental) -- the closed-loop scientist: equation discovery with an exact referee.

The integrative "AI scientist" loop, made honest and gradeable: an agent probes a world by choosing
*where* to observe its dynamics, fits a sparse combination of candidate operator terms
(SINDy-style sequentially-thresholded least squares -- symbolic regression over a term library),
and is graded against the *actual* governing operator: recovered-form match and coefficient error
versus experiment budget.

Because the true operator is known, discovery can be scored exactly -- the property the card
prizes ("the referee is exact"). And because identifying a high-order term (e.g. a cubic) needs
high-leverage observations at the extremes of state space, *choosing* the experiments beats random
probing at a fixed budget.

Scope: this is the in-repo, self-contained core on scalar dynamical-system worlds
(``dx/dt = f(x)``). The full P8 flagship runs this loop inside the mixle-pde PDE worlds (linear
diffusion -> advection-diffusion -> Burgers) over the ``register_dynamics_operator`` grammar; that
lives in the mixle-pde companion (Track N), not this repo.

Exploratory ``mixle.experimental`` code (P8 card).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LIBRARY_NAMES = ("1", "x", "x^2", "x^3")


def library_matrix(x: np.ndarray) -> np.ndarray:
    """Design matrix of candidate operator terms ``[1, x, x^2, x^3]``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a non-empty finite one-dimensional array.")
    return np.stack([np.ones_like(x), x, x**2, x**3], axis=1)


def _validated_true_coef(true_coef: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(true_coef, dtype=float)
    if coefficients.shape != (len(LIBRARY_NAMES),) or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"true_coef must contain {len(LIBRARY_NAMES)} finite library coefficients.")
    return coefficients


def _validated_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    return float(value)


def _validated_experiment_plan(budget: object, radius: object) -> tuple[int, float]:
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or int(budget) <= 0:
        raise ValueError("budget must be a positive integer.")
    if (
        isinstance(radius, (bool, np.bool_))
        or not np.isscalar(radius)
        or not np.isfinite(radius)
        or float(radius) <= 0.0
    ):
        raise ValueError("radius must be a finite positive scalar.")
    return int(budget), float(radius)


def observe(x: np.ndarray, true_coef: np.ndarray, *, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Measure the dynamics ``dx/dt = f(x) + noise`` at the probe points ``x``."""
    design = library_matrix(x)
    coefficients = _validated_true_coef(true_coef)
    noise = _validated_nonnegative(noise, "noise")
    if not hasattr(rng, "standard_normal"):
        raise TypeError("rng must provide standard_normal().")
    return design @ coefficients + noise * rng.standard_normal(design.shape[0])


def stlsq(design: np.ndarray, y: np.ndarray, *, threshold: float, iters: int = 12) -> np.ndarray:
    """Sequentially-thresholded least squares (the SINDy sparse-regression operator)."""
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    threshold = _validated_nonnegative(threshold, "threshold")
    if (
        design.ndim != 2
        or design.shape[0] == 0
        or design.shape[1] == 0
        or y.shape != (design.shape[0],)
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("design and y must be aligned, non-empty, and finite.")
    if isinstance(iters, (bool, np.bool_)) or not isinstance(iters, (int, np.integer)) or int(iters) <= 0:
        raise ValueError("iters must be a positive integer.")
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(int(iters)):
        small = np.abs(coef) < threshold
        coef[small] = 0.0
        big = ~small
        if not big.any():
            break
        coef = coef.copy()
        coef[big] = np.linalg.lstsq(design[:, big], y, rcond=None)[0]
    return coef


def recovered_form(coef: np.ndarray, *, tol: float = 1e-6) -> frozenset[int]:
    """The set of active term indices -- the recovered symbolic form."""
    coef = np.asarray(coef, dtype=float)
    tol = _validated_nonnegative(tol, "tol")
    if coef.ndim != 1 or not np.all(np.isfinite(coef)):
        raise ValueError("coef must be a finite one-dimensional vector.")
    return frozenset(int(i) for i in np.flatnonzero(np.abs(coef) > tol))


def active_experiments(budget: int, radius: float) -> np.ndarray:
    """High-leverage probe placement: Chebyshev-like nodes spanning the extremes of state space."""
    budget, radius = _validated_experiment_plan(budget, radius)
    k = np.arange(budget)
    return radius * np.cos(np.pi * (2 * k + 1) / (2 * budget))


def random_experiments(budget: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Passive baseline: probe points drawn uniformly over the state range."""
    budget, radius = _validated_experiment_plan(budget, radius)
    if not hasattr(rng, "uniform"):
        raise TypeError("rng must provide uniform().")
    return rng.uniform(-radius, radius, budget)


@dataclass
class DiscoveryReceipt:
    recovered_coef: np.ndarray
    recovered_terms: frozenset[int]
    true_terms: frozenset[int]
    form_match: bool
    coef_error: float  # L-inf error over the full library, including spurious discovered terms
    spurious_terms: frozenset[int]
    missing_terms: frozenset[int]


def discover(
    true_coef: np.ndarray, probes: np.ndarray, *, noise: float, threshold: float, seed: int
) -> DiscoveryReceipt:
    """Run one discovery experiment: observe at ``probes``, recover the operator, grade exactly."""
    true_coef = _validated_true_coef(true_coef)
    probes = np.asarray(probes, dtype=float)
    library_matrix(probes)  # validate the probe manifest before creating stochastic observations
    noise = _validated_nonnegative(noise, "noise")
    threshold = _validated_nonnegative(threshold, "threshold")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    rng = np.random.default_rng(seed)
    y = observe(probes, true_coef, noise=noise, rng=rng)
    coef = stlsq(library_matrix(probes), y, threshold=threshold)
    true_terms = recovered_form(true_coef)
    got = recovered_form(coef)
    form_match = got == true_terms
    coef_error = float(np.max(np.abs(coef - true_coef)))
    return DiscoveryReceipt(
        coef,
        got,
        true_terms,
        form_match,
        coef_error,
        spurious_terms=got - true_terms,
        missing_terms=true_terms - got,
    )


def discovery_rate(
    true_coef, *, strategy: str, budget: int, radius: float, noise: float, threshold: float, seeds
) -> float:
    """Fraction of seeds on which ``strategy`` recovers the exact operator form at the given budget."""
    if strategy not in {"active", "random"}:
        raise ValueError("strategy must be 'active' or 'random'.")
    budget, radius = _validated_experiment_plan(budget, radius)
    noise = _validated_nonnegative(noise, "noise")
    threshold = _validated_nonnegative(threshold, "threshold")
    true_coef = _validated_true_coef(true_coef)
    try:
        trial_seeds = tuple(seeds)
    except TypeError as exc:
        raise ValueError("seeds must be a non-empty iterable of integers.") from exc
    if not trial_seeds or any(
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) for seed in trial_seeds
    ):
        raise ValueError("seeds must be a non-empty iterable of integers.")

    hits = 0
    for seed in trial_seeds:
        seed = int(seed)
        rng = np.random.default_rng(10_000 + seed)
        probes = active_experiments(budget, radius) if strategy == "active" else random_experiments(budget, radius, rng)
        if discover(true_coef, probes, noise=noise, threshold=threshold, seed=seed).form_match:
            hits += 1
    return hits / len(trial_seeds)
