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
    return np.stack([np.ones_like(x), x, x**2, x**3], axis=1)


def observe(x: np.ndarray, true_coef: np.ndarray, *, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Measure the dynamics ``dx/dt = f(x) + noise`` at the probe points ``x``."""
    return library_matrix(x) @ np.asarray(true_coef, dtype=float) + noise * rng.standard_normal(len(x))


def stlsq(design: np.ndarray, y: np.ndarray, *, threshold: float, iters: int = 12) -> np.ndarray:
    """Sequentially-thresholded least squares (the SINDy sparse-regression operator)."""
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(iters):
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
    return frozenset(int(i) for i in np.flatnonzero(np.abs(coef) > tol))


def active_experiments(budget: int, radius: float) -> np.ndarray:
    """High-leverage probe placement: Chebyshev-like nodes spanning the extremes of state space."""
    k = np.arange(budget)
    return radius * np.cos(np.pi * (2 * k + 1) / (2 * budget))


def random_experiments(budget: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Passive baseline: probe points drawn uniformly over the state range."""
    return rng.uniform(-radius, radius, budget)


@dataclass
class DiscoveryReceipt:
    recovered_coef: np.ndarray
    recovered_terms: frozenset[int]
    true_terms: frozenset[int]
    form_match: bool
    coef_error: float  # L-inf error on the true active terms (only meaningful when form_match)


def discover(
    true_coef: np.ndarray, probes: np.ndarray, *, noise: float, threshold: float, seed: int
) -> DiscoveryReceipt:
    """Run one discovery experiment: observe at ``probes``, recover the operator, grade exactly."""
    rng = np.random.default_rng(seed)
    y = observe(probes, true_coef, noise=noise, rng=rng)
    coef = stlsq(library_matrix(probes), y, threshold=threshold)
    true_terms = recovered_form(np.asarray(true_coef, dtype=float))
    got = recovered_form(coef)
    form_match = got == true_terms
    active = list(true_terms)
    coef_error = float(np.max(np.abs(coef[active] - np.asarray(true_coef)[active]))) if active else 0.0
    return DiscoveryReceipt(coef, got, true_terms, form_match, coef_error)


def discovery_rate(
    true_coef, *, strategy: str, budget: int, radius: float, noise: float, threshold: float, seeds
) -> float:
    """Fraction of seeds on which ``strategy`` recovers the exact operator form at the given budget."""
    hits = 0
    for s in seeds:
        rng = np.random.default_rng(10_000 + s)
        probes = active_experiments(budget, radius) if strategy == "active" else random_experiments(budget, radius, rng)
        if discover(true_coef, probes, noise=noise, threshold=threshold, seed=s).form_match:
            hits += 1
    return hits / len(list(seeds))
