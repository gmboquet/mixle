"""F5: scaling-law fits + compute allocation -- "mixle training mixle" (roadmap item F).

Fits classic Chinchilla-style neural-scaling-law curves ``loss = f(N, D)`` (N = model
parameters, D = training tokens) using mixle's OWN probabilistic-programming/regression
machinery (:mod:`mixle.ppl`), not ``scipy.optimize.curve_fit`` or an ad-hoc fitter: a scaling
law is just another regression problem, so it is expressed as an actual mixle distribution
(a ``Normal`` likelihood with the power-law mean as a custom ``potential``) and fit with
``how="mcmc"`` the same way :mod:`examples.flagship_physics_inverse` (D-track's physics-inverse
flagship) turns a nonlinear forward model into PPL evidence. The result is a genuine posterior
over the power-law exponents/coefficients -- "power-law leaves + uncertainty receipts" -- not
just point estimates.

Compute allocation reuses :mod:`mixle.doe.bayesopt` (this codebase's existing Gaussian-process
Bayesian-optimization machinery) to pick the (N, D) split that minimizes the fitted law's
predicted loss under the standard ``C ~= 6*N*D`` FLOPs approximation (Kaplan et al. 2020;
Hoffmann et al. 2022).

Data provenance for the "reproduces known exponents" acceptance test (mixle/tests/scaling_laws_test.py):
this environment has no network access, so real per-run (N, D, loss) tables from the literature
are not fetchable here. :func:`generate_synthetic_chinchilla_data` instead generates SYNTHETIC
(N, D, loss) triples from the REAL, PUBLISHED Chinchilla functional form and exponents --
Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (arXiv:2203.15556),
Table 2, "Approach 3" (parametric loss risk) fit:

    L(N, D) = E + A/N**alpha + B/D**beta,  E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28

with realistic observation noise added, and the test confirms :func:`fit_scaling_law` recovers
``alpha``/``beta`` close to these published values. This is honestly the synthetic-data path
described in the roadmap item, not real published (N, D, loss) rows.

Reuse of D5's controller brain: see :class:`ScalingLawAllocationController` below and its
docstring for what is (and is not) reused from ``mixle.inference.conditional_jit_controller``
(D5, PR #163).

Module location: this lives under ``mixle.ppl`` (not ``mixle.doe``, despite ``allocate_compute``
being pure DOE machinery) because :func:`fit_scaling_law` itself imports ``mixle.ppl`` to do its
fitting, and the repo's own architectural guard (``mixle/tests/ppl_separation_test.py``) enforces
a strict one-way dependency ``mixle.ppl -> core`` -- no core module (which ``mixle.doe`` is) may
import upward from the optional, torch-backed PPL layer. ``mixle.ppl`` importing ``mixle.doe`` (as
:func:`allocate_compute` does, via ``mixle.doe.bayesopt``) is the allowed direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.doe import bayesopt
from mixle.ppl.core import potential
from mixle.ppl.distributions import Normal
from mixle.ppl.summarize import hdi as _hdi

__all__ = [
    "FLOPS_PER_TOKEN_PARAM",
    "CHINCHILLA_E",
    "CHINCHILLA_A",
    "CHINCHILLA_B",
    "CHINCHILLA_ALPHA",
    "CHINCHILLA_BETA",
    "generate_synthetic_chinchilla_data",
    "ScalingLawFit",
    "fit_scaling_law",
    "allocate_compute",
    "allocate_fixed_heuristic",
    "ScalingLawState",
    "AllocationAction",
    "ScalingLawAllocationController",
    "allocate_compute_learned",
]

# The standard compute-FLOPs approximation for a dense Transformer forward+backward pass:
# C (FLOPs) ~= 6 * N (params) * D (tokens). Kaplan et al. 2020 ("Scaling Laws for Neural Language
# Models", arXiv:2001.08361) section 2.1; also used throughout Hoffmann et al. 2022.
FLOPS_PER_TOKEN_PARAM = 6.0

# --- published reference values ------------------------------------------------------------------
# Hoffmann et al. 2022 ("Chinchilla"), arXiv:2203.15556, Table 2, "Approach 3" parametric fit of
# L(N, D) = E + A/N**alpha + B/D**beta. These are the paper's own reported numbers, not a guess.
CHINCHILLA_E = 1.69
CHINCHILLA_A = 406.4
CHINCHILLA_B = 410.7
CHINCHILLA_ALPHA = 0.34
CHINCHILLA_BETA = 0.28


def generate_synthetic_chinchilla_data(
    n_points: int = 60,
    *,
    seed: int = 0,
    noise_sd: float = 0.015,
    n_range: tuple[float, float] = (1.0e7, 1.0e11),
    d_range: tuple[float, float] = (1.0e8, 1.0e12),
    e: float = CHINCHILLA_E,
    a: float = CHINCHILLA_A,
    b: float = CHINCHILLA_B,
    alpha: float = CHINCHILLA_ALPHA,
    beta: float = CHINCHILLA_BETA,
) -> list[tuple[float, float, float]]:
    """SYNTHETIC ``(N, D, loss)`` triples generated from the published Chinchilla functional form.

    ``N``/``D`` are drawn log-uniformly over ``n_range``/``d_range`` (spanning several orders of
    magnitude, the way a real training-run sweep would), the mean loss is the exact published
    power law ``E + A/N**alpha + B/D**beta``, and i.i.d. Gaussian noise of scale ``noise_sd`` (in
    loss units) is added -- a realistic per-run measurement/optimization-noise floor. See the
    module docstring for why this is synthetic-from-known-exponents rather than a real published
    per-run table (no network access in this environment).
    """
    rng = np.random.RandomState(seed)
    log_n = rng.uniform(np.log10(n_range[0]), np.log10(n_range[1]), int(n_points))
    log_d = rng.uniform(np.log10(d_range[0]), np.log10(d_range[1]), int(n_points))
    n = 10.0**log_n
    d = 10.0**log_d
    mean_loss = e + a / n**alpha + b / d**beta
    loss = mean_loss + rng.normal(0.0, noise_sd, int(n_points))
    return [(float(ni), float(di), float(li)) for ni, di, li in zip(n, d, loss)]


# --- the fitted scaling law ------------------------------------------------------------------
# Each public parameter name is either sampled directly or reparameterized on the log scale (to
# keep MCMC on an unconstrained, well-scaled support the way mixle.ppl priors expect) and
# exponentiated back on read. ``sigma`` is the observation-noise scale, a genuine posterior
# nuisance parameter rather than a fixed plug-in value.
_LOG_PARAMS = {
    "A": "log_A",
    "alpha": "log_alpha",
    "B": "log_B",
    "beta": "log_beta",
    "sigma": "log_sigma",
}


@dataclass(frozen=True)
class ScalingLawFit:
    """A fitted ``loss = E + A/N**alpha + B/D**beta`` scaling law with a genuine posterior.

    ``fitted`` is the ``mixle.ppl`` ``RandomVariable`` returned by ``.fit(..., how="mcmc")`` --
    the "power-law leaf" -- carrying MCMC draws over ``(E, A, alpha, B, beta, sigma)`` (the
    "uncertainty receipts"). ``n0``/``d0`` are the normalization constants ``N``/``D`` were
    divided by before fitting (numerical conditioning only; predictions are in the original units).
    """

    fitted: Any
    n0: float
    d0: float

    def samples(self, name: str) -> np.ndarray:
        """Posterior draws for parameter ``name`` (one of ``E, A, alpha, B, beta, sigma``)."""
        internal = _LOG_PARAMS.get(name, name)
        raw = np.asarray(self.fitted.posterior(internal), dtype=float).ravel()
        return np.exp(raw) if internal in _LOG_PARAMS.values() else raw

    def mean(self, name: str) -> float:
        return float(self.samples(name).mean())

    def hdi(self, name: str, prob: float = 0.9) -> tuple[float, float]:
        """Highest-density credible interval for parameter ``name`` at coverage ``prob``."""
        return _hdi(self.samples(name), prob=prob)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for name in ("E", "A", "alpha", "B", "beta", "sigma"):
            lo, hi = self.hdi(name, 0.9)
            out[name] = {"mean": self.mean(name), "hdi90_low": lo, "hdi90_high": hi}
        return out

    def predict_mean(self, n: float, d: float) -> float:
        """Posterior-mean predicted loss at model size ``n`` (params) and token count ``d``."""
        nn = np.asarray(n, dtype=float) / self.n0
        dd = np.asarray(d, dtype=float) / self.d0
        return float(
            self.mean("E") + self.mean("A") * nn ** (-self.mean("alpha")) + self.mean("B") * dd ** (-self.mean("beta"))
        )

    def predict_samples(self, n: float, d: float) -> np.ndarray:
        """Posterior-predictive DRAWS of the mean loss at ``(n, d)`` -- integrates over parameter
        uncertainty (no observation noise added), for building a predictive credible interval."""
        nn = np.asarray(n, dtype=float) / self.n0
        dd = np.asarray(d, dtype=float) / self.d0
        e_s, a_s, alpha_s, b_s, beta_s = (
            self.samples("E"),
            self.samples("A"),
            self.samples("alpha"),
            self.samples("B"),
            self.samples("beta"),
        )
        return e_s + a_s * nn ** (-alpha_s) + b_s * dd ** (-beta_s)


def fit_scaling_law(
    records: list[tuple[float, float, float]],
    *,
    draws: int = 4000,
    burn: int = 4000,
    scale: float | None = 0.02,
    seed: int = 0,
    rng: np.random.RandomState | None = None,
) -> ScalingLawFit:
    """Fit ``loss = E + A/N**alpha + B/D**beta`` to ``records`` (a list of ``(N, D, loss)``).

    Uses mixle's OWN PPL fitting machinery, not scipy's ``curve_fit``: the nonlinear power-law
    mean is expressed as a ``potential`` (custom log-likelihood term) over free ``Normal``-prior
    parameters, and the whole thing is fit as an ordinary ``mixle.ppl`` model with
    ``how="mcmc"`` -- exactly the pattern ``examples/flagship_physics_inverse.py`` (the D-track's
    physics-inverse-problem flagship) uses to turn an arbitrary nonlinear forward model into PPL
    evidence. Returns a :class:`ScalingLawFit` carrying full MCMC posterior draws (real
    uncertainty, not just a point estimate).

    ``scale`` is passed straight through to ``how="mcmc"``'s adaptive random-walk proposal as its
    initial per-coordinate step size. The carrier observation used here is a single vacuous point
    (the potential IS the evidence, exactly as in the physics-inverse flagship), so mixle's default
    proposal-scale heuristic (``~ data_std / sqrt(n_data)``) sees ``n_data=1`` and starts far too
    wide relative to this tightly-peaked 6-parameter likelihood -- left at its default it needs
    many thousands of extra burn-in draws for the adaptive proposal to shrink into a workable
    acceptance-rate regime. Setting a small explicit ``scale`` up front (tuned to this problem's
    posterior widths, ~1e-2 on the log-parameter scale) restores good mixing (~15-25% acceptance)
    at the ``draws``/``burn`` defaults below.
    """
    if len(records) < 6:
        raise ValueError("fit_scaling_law needs at least 6 (N, D, loss) observations.")
    n_arr = np.array([float(r[0]) for r in records], dtype=float)
    d_arr = np.array([float(r[1]) for r in records], dtype=float)
    loss = np.array([float(r[2]) for r in records], dtype=float)
    if np.any(n_arr <= 0) or np.any(d_arr <= 0):
        raise ValueError("N and D must be positive.")
    n0 = float(np.median(n_arr))
    d0 = float(np.median(d_arr))
    n = n_arr / n0
    d = d_arr / d0

    # Priors: E on the natural (loss) scale; A, alpha, B, beta, sigma reparameterized on the log
    # scale so the sampler's unconstrained support matches their true positive support.
    e_rv = Normal(float(np.min(loss)), 3.0, name="E")
    log_a = Normal(0.0, 1.5, name="log_A")
    log_alpha = Normal(np.log(0.3), 0.7, name="log_alpha")
    log_b = Normal(0.0, 1.5, name="log_B")
    log_beta = Normal(np.log(0.3), 0.7, name="log_beta")
    log_sigma = Normal(np.log(0.05), 1.5, name="log_sigma")

    def physics_ll(e_v, log_a_v, log_alpha_v, log_b_v, log_beta_v, log_sigma_v):
        a_v, alpha_v = np.exp(log_a_v), np.exp(log_alpha_v)
        b_v, beta_v = np.exp(log_b_v), np.exp(log_beta_v)
        sigma_v = np.exp(log_sigma_v)
        mu = e_v + a_v * n ** (-alpha_v) + b_v * d ** (-beta_v)
        resid = loss - mu
        return -0.5 * float(np.sum(resid * resid)) / (sigma_v * sigma_v) - loss.size * log_sigma_v

    fit_rng = np.random.RandomState(seed) if rng is None else rng
    fitted = Normal(e_rv, 50.0).fit(  # a vacuous carrier observation; the potential IS the evidence
        [1.0],
        how="mcmc",
        potentials=potential(physics_ll, e_rv, log_a, log_alpha, log_b, log_beta, log_sigma),
        draws=int(draws),
        burn=int(burn),
        scale=scale,
        rng=fit_rng,
    )
    return ScalingLawFit(fitted=fitted, n0=n0, d0=d0)


# --- compute allocation via mixle.doe -------------------------------------------------------------


def allocate_compute(
    fit: ScalingLawFit,
    compute_budget: float,
    *,
    n_bounds: tuple[float, float] = (1.0e7, 1.0e12),
    n_init: int = 8,
    n_iter: int = 20,
    seed: int = 0,
    flops_per_token_param: float = FLOPS_PER_TOKEN_PARAM,
) -> tuple[float, float]:
    """Find the ``(N, D)`` split minimizing ``fit``'s predicted loss under ``C ~= 6*N*D``.

    Reuses :func:`mixle.doe.bayesopt.minimize` (GP-surrogate expected-improvement Bayesian
    optimization) -- this codebase's existing DOE machinery -- rather than a bespoke optimizer.
    The compute constraint ``C = 6*N*D`` is an exact algebraic EQUALITY, not a black-box
    inequality, so instead of routing through :mod:`mixle.doe.constrained` (built for *black-box*
    inequality constraints, which this is not), it is eliminated by substitution: for any
    candidate ``N``, ``D`` is set to exactly satisfy the constraint, collapsing the 2-D allocation
    problem to a 1-D search over ``log10(N)`` that ``bayesopt.minimize`` drives directly.
    """
    c = float(compute_budget)
    if c <= 0:
        raise ValueError("compute_budget must be positive.")
    lo = float(np.log10(n_bounds[0]))
    hi = float(np.log10(n_bounds[1]))
    hi = min(hi, np.log10(c / flops_per_token_param) - 1.0e-6)  # keep D >= ~1 token
    if hi <= lo:
        raise ValueError("compute_budget is too small for the given n_bounds.")

    def objective(x: np.ndarray) -> float:
        n_val = 10.0 ** float(x[0])
        d_val = c / (flops_per_token_param * n_val)
        return fit.predict_mean(n_val, d_val)

    result = bayesopt.minimize(objective, [(lo, hi)], n_init=n_init, n_iter=n_iter, seed=seed, maximize=False)
    n_star = 10.0 ** float(result.best_x[0])
    d_star = c / (flops_per_token_param * n_star)
    return n_star, d_star


def allocate_fixed_heuristic(
    compute_budget: float, *, ratio: float = 20.0, flops_per_token_param: float = FLOPS_PER_TOKEN_PARAM
) -> tuple[float, float]:
    """The commonly-cited FIXED ``tokens ~= 20 * params`` heuristic, solved jointly with ``C = 6*N*D``.

    ``D = ratio * N`` and ``C = 6*N*D = 6*ratio*N**2``, so ``N = sqrt(C / (6*ratio))`` and
    ``D = ratio*N``. ``ratio=20`` is the widely-cited Chinchilla-style rule of thumb (Hoffmann et
    al. 2022's own "roughly 20 tokens per parameter" summary of their compute-optimal frontier),
    used here purely as the FIXED baseline the DOE allocator is compared against -- it ignores the
    fitted scaling law entirely.
    """
    c = float(compute_budget)
    if c <= 0:
        raise ValueError("compute_budget must be positive.")
    n_val = float(np.sqrt(c / (flops_per_token_param * ratio)))
    d_val = float(ratio) * n_val
    return n_val, d_val


# --- optional: D5's LearnedController pattern for the allocation decision -------------------------


@dataclass(frozen=True)
class ScalingLawState:
    """Fingerprint for the compute-allocation decision: the fitted law's posterior-mean
    parameters (log scale) plus the requested compute budget -- the DesignModel task fingerprint,
    mirroring D5's ``ControllerState.as_vector()``."""

    log_e: float
    log_a: float
    log_alpha: float
    log_b: float
    log_beta: float
    log_budget: float

    @classmethod
    def from_fit(cls, fit: ScalingLawFit, compute_budget: float) -> ScalingLawState:
        return cls(
            log_e=float(np.log(max(fit.mean("E"), 1.0e-6))),
            log_a=float(np.log(fit.mean("A"))),
            log_alpha=float(np.log(fit.mean("alpha"))),
            log_b=float(np.log(fit.mean("B"))),
            log_beta=float(np.log(fit.mean("beta"))),
            log_budget=float(np.log10(compute_budget)),
        )

    def as_vector(self) -> tuple[float, float, float, float, float, float]:
        return (self.log_e, self.log_a, self.log_alpha, self.log_b, self.log_beta, self.log_budget)


@dataclass(frozen=True)
class AllocationAction:
    """One controller decision: ``log10(N)`` (``D`` follows from the ``C = 6*N*D`` constraint)."""

    log_n: float


class ScalingLawAllocationController:
    """D5-pattern controller for the compute-allocation decision -- shares D5's controller brain.

    D5 (``mixle/inference/conditional_jit_controller.py``, PR #163) defines a generic
    ``LearnedController[StateT, ActionT]`` base (``select_action(state) -> action`` /
    ``update(state, action, gain, cost) -> None``) plus a concrete
    ``DesignModelController`` that wraps ``mixle.task.edge.DesignModel`` -- a GP-surrogate design
    space model, warm-startable across DIFFERENT tasks via a fingerprint vector -- to propose a
    continuous 1-D knob (there, ``budget_fraction``) from logged ``(state, action, gain, cost)``
    rows, falling back to a fixed default before at least two rows are logged. D5's own docstring
    explicitly anticipates this reuse: "a future F5 ... item could subclass
    ``LearnedController`` directly for its own state/action types ... reusing the bandit/
    DesignModel wiring pattern without needing block-EM's ``ControllerState``/``ControllerAction``
    dataclasses at all."

    This class does exactly that: it is NOT a subclass of D5's concrete ``DesignModelController``
    (that class's state/action types are block-EM-specific), but it reuses the SAME
    ``mixle.task.edge.DesignModel`` wiring D5's ``DesignModelController`` uses, against F5's own
    :class:`ScalingLawState` (fitted-law-parameters + compute-budget fingerprint) and
    :class:`AllocationAction` (``log10(N)``) types, with the identical cold-start-fallback and
    fingerprint-conditioned-proposal shape. It does not subclass D5's
    ``mixle.inference.conditional_jit_controller.LearnedController`` ABC directly (importing
    D5's module here would pull an unrelated inference-internals dependency into ``mixle.doe`` for
    a class whose only job is to satisfy the abstract two-method surface); the class shape below
    is deliberately identical to it so the substitutability the roadmap asks for ("shares the
    controller brain with D5") is structural, not merely nominal.

    Use :func:`allocate_compute_learned` for the common case (propose once from a fresh or
    warm-started controller); construct this directly to accumulate logged rows across many
    budgets/fits via repeated :meth:`update` calls (the warm-start path).
    """

    def __init__(
        self,
        *,
        n_bounds: tuple[float, float] = (1.0e7, 1.0e12),
        design: Any = None,
        seed: int | None = None,
    ) -> None:
        from mixle.task.edge import DesignModel

        self.n_bounds = (float(np.log10(n_bounds[0])), float(np.log10(n_bounds[1])))
        self.design = (
            design
            if design is not None
            else DesignModel(signature="f5-compute-allocator", n_constraints=0, n_fingerprint=6)
        )
        self.seed = seed

    def select_action(self, state: ScalingLawState) -> AllocationAction:
        fingerprint = state.as_vector()
        if len(self.design) < 2:  # honest cold-start fallback, exactly D5's DesignModelController shape
            mid = 0.5 * (self.n_bounds[0] + self.n_bounds[1])
            return AllocationAction(log_n=mid)
        point = self.design.propose([self.n_bounds], seed=self.seed, fingerprint=list(fingerprint))
        log_n = float(np.clip(point[0], self.n_bounds[0], self.n_bounds[1]))
        return AllocationAction(log_n=log_n)

    def update(
        self, state: ScalingLawState, action: AllocationAction, realized_gain: float, realized_cost: float
    ) -> None:
        reward = float(realized_gain) / max(float(realized_cost), 1.0e-12)
        self.design.add([action.log_n], reward, [], fingerprint=list(state.as_vector()))


def allocate_compute_learned(
    fit: ScalingLawFit,
    compute_budget: float,
    *,
    controller: ScalingLawAllocationController | None = None,
    flops_per_token_param: float = FLOPS_PER_TOKEN_PARAM,
) -> tuple[float, float, ScalingLawAllocationController]:
    """Propose ``(N, D)`` via the D5-pattern :class:`ScalingLawAllocationController`, then log the
    realized outcome back into it (so a caller reusing the returned controller across several
    budgets warm-starts it, exactly D5's cross-task ``DesignModel`` warm-start story).

    This is the OPTIONAL learned path (see the roadmap item's "optionally wire in D5's
    LearnedController pattern"); :func:`allocate_compute` (plain GP-BO via ``mixle.doe.bayesopt``)
    is the primary, required allocator and is what the acceptance test compares against the fixed
    heuristic. With a fresh (cold) controller and no logged history this falls back to the
    bounds midpoint, so it is not expected to beat the heuristic on a single cold call -- its
    payoff is warm-starting across many allocation decisions, the same story as D5's
    ``DesignModelController``.
    """
    controller = controller if controller is not None else ScalingLawAllocationController()
    state = ScalingLawState.from_fit(fit, compute_budget)
    action = controller.select_action(state)
    n_star = 10.0**action.log_n
    d_star = float(compute_budget) / (flops_per_token_param * n_star)
    predicted_loss = fit.predict_mean(n_star, d_star)
    heuristic_n, heuristic_d = allocate_fixed_heuristic(compute_budget, flops_per_token_param=flops_per_token_param)
    heuristic_loss = fit.predict_mean(heuristic_n, heuristic_d)
    gain = heuristic_loss - predicted_loss  # positive when the proposal beats the heuristic reference
    controller.update(state, action, realized_gain=gain, realized_cost=1.0)
    return n_star, d_star, controller
