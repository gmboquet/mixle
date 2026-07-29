"""F5: scaling-law fits + compute allocation -- "mixle training mixle" (roadmap item F).

.. warning::

   **Experimental frontier-training prototype.** The curve fits and compute-allocation math are exact on
   the data you give them, but any extrapolation beyond the fitted regime is a research estimate, not a
   guarantee -- a scaling-law fit is only as trustworthy as its measured points. Not a production planner.

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

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
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
    "SyntheticScalingRecords",
    "generate_synthetic_chinchilla_data",
    "ScalingLawDiagnostics",
    "ScalingLawFit",
    "fit_scaling_law",
    "allocate_compute",
    "allocate_fixed_heuristic",
    "ScalingLawState",
    "AllocationAction",
    "AllocationProposal",
    "AllocationObservationReceipt",
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


def _exact_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an exact integer, got {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}.")
    return value


def _finite_scalar(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite numeric scalar, got {value!r}.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    if positive and value <= 0.0:
        raise ValueError(f"{label} must be strictly positive, got {value!r}.")
    if nonnegative and value < 0.0:
        raise ValueError(f"{label} must be non-negative, got {value!r}.")
    return value


def _positive_bounds(bounds: Any, label: str) -> tuple[float, float]:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{label} must be a (low, high) pair.")
    low = _finite_scalar(bounds[0], f"{label}[0]", positive=True)
    high = _finite_scalar(bounds[1], f"{label}[1]", positive=True)
    if not low < high:
        raise ValueError(f"{label} must satisfy 0 < low < high.")
    return low, high


class SyntheticScalingRecords(list):
    """List-compatible synthetic records with a copy-on-read generator configuration receipt."""

    __slots__ = ("_configuration",)

    def __init__(self, records, configuration):
        super().__init__(records)
        self._configuration = dict(configuration)

    @property
    def configuration(self) -> dict[str, Any]:
        return dict(self._configuration)


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
) -> SyntheticScalingRecords:
    """SYNTHETIC ``(N, D, loss)`` triples generated from the published Chinchilla functional form.

    ``N``/``D`` are drawn log-uniformly over ``n_range``/``d_range`` (spanning several orders of
    magnitude, the way a real training-run sweep would), the mean loss is the exact published
    power law ``E + A/N**alpha + B/D**beta``, and i.i.d. Gaussian noise of scale ``noise_sd`` (in
    loss units) is added -- a realistic per-run measurement/optimization-noise floor. See the
    module docstring for why this is synthetic-from-known-exponents rather than a real published
    per-run table (no network access in this environment).
    """
    n_points = _exact_integer(n_points, "n_points", minimum=1)
    seed = _exact_integer(seed, "seed")
    noise_sd = _finite_scalar(noise_sd, "noise_sd", nonnegative=True)
    n_range = _positive_bounds(n_range, "n_range")
    d_range = _positive_bounds(d_range, "d_range")
    e = _finite_scalar(e, "e", nonnegative=True)
    a = _finite_scalar(a, "a", positive=True)
    b = _finite_scalar(b, "b", positive=True)
    alpha = _finite_scalar(alpha, "alpha", positive=True)
    beta = _finite_scalar(beta, "beta", positive=True)
    configuration = {
        "n_points": n_points,
        "seed": seed,
        "noise_sd": noise_sd,
        "n_range": n_range,
        "d_range": d_range,
        "e": e,
        "a": a,
        "b": b,
        "alpha": alpha,
        "beta": beta,
    }
    rng = np.random.RandomState(seed)
    log_n = rng.uniform(np.log10(n_range[0]), np.log10(n_range[1]), n_points)
    log_d = rng.uniform(np.log10(d_range[0]), np.log10(d_range[1]), n_points)
    n = 10.0**log_n
    d = 10.0**log_d
    mean_loss = e + a / n**alpha + b / d**beta
    loss = mean_loss + rng.normal(0.0, noise_sd, n_points)
    records = [(float(ni), float(di), float(li)) for ni, di, li in zip(n, d, loss)]
    if not np.isfinite(np.asarray(records, dtype=float)).all():
        raise RuntimeError("synthetic scaling-law generation produced a non-finite record.")
    return SyntheticScalingRecords(records, configuration)


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
class ScalingLawDiagnostics:
    """Quality receipt required before a fitted law may drive an allocation."""

    usable: bool
    status: str
    acceptance_rate: float
    posterior_draws: int
    burn: int
    n_chains: int
    posterior_finite: bool
    caveats: tuple[str, ...] = ()


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
    diagnostics: ScalingLawDiagnostics | None = None

    def samples(self, name: str) -> np.ndarray:
        """Posterior draws for parameter ``name`` (one of ``E, A, alpha, B, beta, sigma``)."""
        if name not in {"E", "A", "alpha", "B", "beta", "sigma"}:
            raise ValueError(f"unknown scaling-law parameter {name!r}.")
        internal = _LOG_PARAMS.get(name, name)
        raw = np.asarray(self.fitted.posterior(internal), dtype=float).ravel()
        values = np.exp(raw) if internal in _LOG_PARAMS.values() else raw
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"posterior draws for {name} are empty or non-finite.")
        return values

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

    def require_usable(self) -> None:
        """Refuse scientific decisions from a fit that lacks a passing quality receipt."""
        if self.diagnostics is None or not self.diagnostics.usable:
            status = "missing" if self.diagnostics is None else self.diagnostics.status
            raise RuntimeError(f"scaling-law fit is not usable for allocation (diagnostic status: {status}).")

    def predict_mean(self, n: float, d: float) -> float:
        """Posterior expectation of mean loss, computed from correlated joint draws."""
        return float(np.mean(self.predict_samples(n, d)))

    def predict_samples(self, n: float, d: float) -> np.ndarray:
        """Posterior-predictive DRAWS of the mean loss at ``(n, d)`` -- integrates over parameter
        uncertainty (no observation noise added), for building a predictive credible interval."""
        n = _finite_scalar(n, "n", positive=True)
        d = _finite_scalar(d, "d", positive=True)
        n0 = _finite_scalar(self.n0, "fit.n0", positive=True)
        d0 = _finite_scalar(self.d0, "fit.d0", positive=True)
        nn = n / n0
        dd = d / d0
        e_s, a_s, alpha_s, b_s, beta_s = (
            self.samples("E"),
            self.samples("A"),
            self.samples("alpha"),
            self.samples("B"),
            self.samples("beta"),
        )
        sizes = {len(values) for values in (e_s, a_s, alpha_s, b_s, beta_s)}
        if len(sizes) != 1:
            raise RuntimeError("scaling-law posterior parameters do not have aligned joint draws.")
        result = e_s + a_s * nn ** (-alpha_s) + b_s * dd ** (-beta_s)
        if not np.isfinite(result).all():
            raise RuntimeError("scaling-law prediction produced non-finite posterior draws.")
        return result


def fit_scaling_law(
    records: Sequence[tuple[float, float, float]],
    *,
    draws: int = 4000,
    burn: int = 4000,
    scale: float | None = 0.02,
    seed: int | None = None,
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
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of (N, D, loss) triples.")
    if len(records) < 6:
        raise ValueError("fit_scaling_law needs at least 6 (N, D, loss) observations.")
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, (tuple, list, np.ndarray)) or len(record) != 3:
            raise ValueError(f"records[{index}] must be exactly one (N, D, loss) triple.")
        try:
            row = tuple(float(value) for value in record)
        except (TypeError, ValueError) as error:
            raise ValueError(f"records[{index}] must contain numeric N, D, and loss values.") from error
        if not np.isfinite(row).all():
            raise ValueError(f"records[{index}] must contain only finite values.")
        if row[0] <= 0.0 or row[1] <= 0.0 or row[2] < 0.0:
            raise ValueError(f"records[{index}] requires N > 0, D > 0, and loss >= 0.")
        rows.append(row)
    draws = _exact_integer(draws, "draws", minimum=20)
    burn = _exact_integer(burn, "burn", minimum=20)
    if scale is not None:
        scale = _finite_scalar(scale, "scale", positive=True)
    if seed is not None:
        seed = _exact_integer(seed, "seed")
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng to fit_scaling_law, not both.")
    if rng is not None and not isinstance(rng, np.random.RandomState):
        raise ValueError("rng must be a numpy.random.RandomState.")
    n_arr = np.array([row[0] for row in rows], dtype=float)
    d_arr = np.array([row[1] for row in rows], dtype=float)
    loss = np.array([row[2] for row in rows], dtype=float)
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

    carrier_sd = 50.0
    carrier_obs = 1.0

    def evidence_ll(e_v, log_a_v, log_alpha_v, log_b_v, log_beta_v, log_sigma_v):
        # The MCMC route needs a carrier observation to build its target, and the carrier
        # ``Normal(e_rv, carrier_sd)`` scored on ``[carrier_obs]`` contributes a REAL
        # ``N(carrier_obs | E, carrier_sd^2)`` likelihood factor (precision 1/carrier_sd^2)
        # pulling E toward the carrier point. Cancel its E-dependent term here so the carrier
        # is genuinely vacuous and the physics potential is the ONLY evidence.
        ll = physics_ll(e_v, log_a_v, log_alpha_v, log_b_v, log_beta_v, log_sigma_v)
        return ll + 0.5 * (carrier_obs - e_v) ** 2 / (carrier_sd * carrier_sd)

    fit_rng = np.random.RandomState(0 if seed is None else seed) if rng is None else rng
    fitted = Normal(e_rv, carrier_sd).fit(  # carrier observation; its E-factor is cancelled in evidence_ll
        [carrier_obs],
        how="mcmc",
        potentials=potential(evidence_ll, e_rv, log_a, log_alpha, log_b, log_beta, log_sigma),
        draws=draws,
        burn=burn,
        scale=scale,
        rng=fit_rng,
    )
    sample_columns = [
        np.asarray(fitted.posterior(name), dtype=float).ravel()
        for name in ("E", "log_A", "log_alpha", "log_B", "log_beta", "log_sigma")
    ]
    posterior_finite = bool(
        sample_columns and all(col.size == draws and np.isfinite(col).all() for col in sample_columns)
    )
    acceptance_rate = float(getattr(fitted.result, "acceptance_rate", float("nan")))
    n_chains = int(getattr(fitted.result, "n_chains", 1))
    if not posterior_finite:
        raise RuntimeError("scaling-law MCMC returned empty, misaligned, or non-finite posterior draws.")
    if not np.isfinite(acceptance_rate) or not 0.05 <= acceptance_rate <= 0.95:
        raise RuntimeError(
            "scaling-law MCMC failed its acceptance-rate quality gate "
            f"(required 0.05 <= rate <= 0.95, got {acceptance_rate!r})."
        )
    caveats = ("single_chain_no_between_chain_rhat",) if n_chains < 2 else ()
    diagnostics = ScalingLawDiagnostics(
        usable=True,
        status="usable_with_caveat" if caveats else "usable",
        acceptance_rate=acceptance_rate,
        posterior_draws=draws,
        burn=burn,
        n_chains=n_chains,
        posterior_finite=posterior_finite,
        caveats=caveats,
    )
    return ScalingLawFit(fitted=fitted, n0=n0, d0=d0, diagnostics=diagnostics)


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
    if not isinstance(fit, ScalingLawFit):
        raise ValueError("fit must be a ScalingLawFit.")
    fit.require_usable()
    c = _finite_scalar(compute_budget, "compute_budget", positive=True)
    flops_per_token_param = _finite_scalar(flops_per_token_param, "flops_per_token_param", positive=True)
    n_bounds = _positive_bounds(n_bounds, "n_bounds")
    n_init = _exact_integer(n_init, "n_init", minimum=1)
    n_iter = _exact_integer(n_iter, "n_iter", minimum=1)
    seed = _exact_integer(seed, "seed")
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
    if not np.isfinite((n_star, d_star)).all() or n_star <= 0.0 or d_star <= 0.0:
        raise RuntimeError("compute allocation produced a non-finite or non-positive design.")
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
    c = _finite_scalar(compute_budget, "compute_budget", positive=True)
    ratio = _finite_scalar(ratio, "ratio", positive=True)
    flops_per_token_param = _finite_scalar(flops_per_token_param, "flops_per_token_param", positive=True)
    n_val = float(np.sqrt(c / (flops_per_token_param * ratio)))
    d_val = ratio * n_val
    if not np.isfinite((n_val, d_val)).all() or n_val <= 0.0 or d_val <= 0.0:
        raise RuntimeError("fixed compute allocation produced a non-finite or non-positive design.")
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
        if not isinstance(fit, ScalingLawFit):
            raise ValueError("fit must be a ScalingLawFit.")
        fit.require_usable()
        compute_budget = _finite_scalar(compute_budget, "compute_budget", positive=True)
        means = {
            name: _finite_scalar(fit.mean(name), f"fit mean {name}", positive=True)
            for name in ("E", "A", "alpha", "B", "beta")
        }
        return cls(
            log_e=float(np.log(means["E"])),
            log_a=float(np.log(means["A"])),
            log_alpha=float(np.log(means["alpha"])),
            log_b=float(np.log(means["B"])),
            log_beta=float(np.log(means["beta"])),
            log_budget=float(np.log10(compute_budget)),
        )

    def as_vector(self) -> tuple[float, float, float, float, float, float]:
        return (self.log_e, self.log_a, self.log_alpha, self.log_b, self.log_beta, self.log_budget)


@dataclass(frozen=True)
class AllocationAction:
    """One controller decision: ``log10(N)`` (``D`` follows from the ``C = 6*N*D`` constraint)."""

    log_n: float


@dataclass(frozen=True)
class AllocationProposal:
    """Pending learned allocation; predicted quantities are not measured outcomes."""

    proposal_id: str
    state: ScalingLawState
    action: AllocationAction
    n_params: float
    n_tokens: float
    compute_budget: float
    flops_per_token_param: float
    predicted_loss: float
    heuristic_predicted_loss: float
    predicted_gain: float
    status: str = "pending_measurement"


@dataclass(frozen=True)
class AllocationObservationReceipt:
    """Auditable acceptance or rejection of one measured controller outcome."""

    accepted: bool
    proposal_id: str | None
    realized_gain: float | None
    realized_cost: float | None
    provenance: str | None
    reason: str
    submitted_gain: str
    submitted_cost: str


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

    Use :func:`allocate_compute_learned` to create a pending proposal, then call
    :meth:`record_outcome` with a measured gain, measured cost, and provenance. Only accepted
    measured outcomes enter the warm-start design history.
    """

    def __init__(
        self,
        *,
        n_bounds: tuple[float, float] = (1.0e7, 1.0e12),
        design: Any = None,
        seed: int | None = None,
    ) -> None:
        n_bounds = _positive_bounds(n_bounds, "n_bounds")
        if seed is not None:
            seed = _exact_integer(seed, "seed")
        self.n_bounds = (float(np.log10(n_bounds[0])), float(np.log10(n_bounds[1])))
        if design is None:
            from mixle.task.edge import DesignModel

            design = DesignModel(signature="f5-compute-allocator", n_constraints=0, n_fingerprint=6)
        self.design = design
        self.seed = seed
        self.pending: dict[str, AllocationProposal] = {}
        self.observation_receipts: list[AllocationObservationReceipt] = []
        self._proposal_counter = 0

    def select_action(self, state: ScalingLawState) -> AllocationAction:
        if not isinstance(state, ScalingLawState) or not np.isfinite(state.as_vector()).all():
            raise ValueError("state must be a finite ScalingLawState.")
        fingerprint = state.as_vector()
        if len(self.design) < 2:  # honest cold-start fallback, exactly D5's DesignModelController shape
            mid = 0.5 * (self.n_bounds[0] + self.n_bounds[1])
            return AllocationAction(log_n=mid)
        point = self.design.propose([self.n_bounds], seed=self.seed, fingerprint=list(fingerprint))
        log_n = float(np.clip(point[0], self.n_bounds[0], self.n_bounds[1]))
        return AllocationAction(log_n=log_n)

    def register_proposal(
        self,
        state: ScalingLawState,
        action: AllocationAction,
        *,
        n_params: float,
        n_tokens: float,
        compute_budget: float,
        flops_per_token_param: float,
        predicted_loss: float,
        heuristic_predicted_loss: float,
    ) -> AllocationProposal:
        self._proposal_counter += 1
        proposal_id = f"scaling-allocation-{self._proposal_counter:08d}"
        predicted_loss = _finite_scalar(predicted_loss, "predicted_loss")
        heuristic_predicted_loss = _finite_scalar(heuristic_predicted_loss, "heuristic_predicted_loss")
        proposal = AllocationProposal(
            proposal_id=proposal_id,
            state=state,
            action=action,
            n_params=_finite_scalar(n_params, "n_params", positive=True),
            n_tokens=_finite_scalar(n_tokens, "n_tokens", positive=True),
            compute_budget=_finite_scalar(compute_budget, "compute_budget", positive=True),
            flops_per_token_param=_finite_scalar(flops_per_token_param, "flops_per_token_param", positive=True),
            predicted_loss=predicted_loss,
            heuristic_predicted_loss=heuristic_predicted_loss,
            predicted_gain=heuristic_predicted_loss - predicted_loss,
        )
        self.pending[proposal_id] = proposal
        return proposal

    def update(
        self,
        state: ScalingLawState,
        action: AllocationAction,
        realized_gain: float,
        realized_cost: float,
        *,
        provenance: str,
        proposal_id: str | None = None,
    ) -> AllocationObservationReceipt:
        reason = "accepted"
        gain = cost = None
        provenance_value = provenance if isinstance(provenance, str) and provenance.strip() else None
        try:
            if not isinstance(state, ScalingLawState) or not np.isfinite(state.as_vector()).all():
                raise ValueError("state is not finite.")
            if not isinstance(action, AllocationAction) or not np.isfinite(action.log_n):
                raise ValueError("action is not finite.")
            gain = _finite_scalar(realized_gain, "realized_gain")
            cost = _finite_scalar(realized_cost, "realized_cost", positive=True)
            if provenance_value is None:
                raise ValueError("measured outcomes require non-empty provenance.")
        except ValueError as error:
            reason = str(error)
            receipt = AllocationObservationReceipt(
                accepted=False,
                proposal_id=proposal_id,
                realized_gain=gain,
                realized_cost=cost,
                provenance=provenance_value,
                reason=reason,
                submitted_gain=repr(realized_gain),
                submitted_cost=repr(realized_cost),
            )
            self.observation_receipts.append(receipt)
            return receipt

        reward = gain / cost
        self.design.add([action.log_n], reward, [], fingerprint=list(state.as_vector()))
        receipt = AllocationObservationReceipt(
            accepted=True,
            proposal_id=proposal_id,
            realized_gain=gain,
            realized_cost=cost,
            provenance=provenance_value,
            reason=reason,
            submitted_gain=repr(realized_gain),
            submitted_cost=repr(realized_cost),
        )
        self.observation_receipts.append(receipt)
        return receipt

    def record_outcome(
        self,
        proposal: AllocationProposal,
        *,
        realized_gain: float,
        realized_cost: float,
        provenance: str,
    ) -> AllocationObservationReceipt:
        """Attach one measured outcome to a pending proposal exactly once."""
        pending = self.pending.get(getattr(proposal, "proposal_id", None))
        if pending is None or pending != proposal:
            receipt = AllocationObservationReceipt(
                accepted=False,
                proposal_id=getattr(proposal, "proposal_id", None),
                realized_gain=None,
                realized_cost=None,
                provenance=provenance if isinstance(provenance, str) else None,
                reason="proposal is unknown, altered, or already resolved.",
                submitted_gain=repr(realized_gain),
                submitted_cost=repr(realized_cost),
            )
            self.observation_receipts.append(receipt)
            return receipt
        receipt = self.update(
            proposal.state,
            proposal.action,
            realized_gain,
            realized_cost,
            provenance=provenance,
            proposal_id=proposal.proposal_id,
        )
        if receipt.accepted:
            del self.pending[proposal.proposal_id]
        return receipt


def allocate_compute_learned(
    fit: ScalingLawFit,
    compute_budget: float,
    *,
    controller: ScalingLawAllocationController | None = None,
    flops_per_token_param: float = FLOPS_PER_TOKEN_PARAM,
) -> tuple[float, float, ScalingLawAllocationController, AllocationProposal]:
    """Propose ``(N, D)`` and return a pending receipt without fabricating an outcome.

    This is the OPTIONAL learned path (see the roadmap item's "optionally wire in D5's
    LearnedController pattern"); :func:`allocate_compute` (plain GP-BO via ``mixle.doe.bayesopt``)
    is the primary, required allocator and is what the acceptance test compares against the fixed
    heuristic. With a fresh (cold) controller and no logged history this falls back to the
    bounds midpoint, so it is not expected to beat the heuristic on a single cold call -- its
    payoff is warm-starting across many allocation decisions, the same story as D5's
    ``DesignModelController``.
    """
    if not isinstance(fit, ScalingLawFit):
        raise ValueError("fit must be a ScalingLawFit.")
    fit.require_usable()
    compute_budget = _finite_scalar(compute_budget, "compute_budget", positive=True)
    flops_per_token_param = _finite_scalar(flops_per_token_param, "flops_per_token_param", positive=True)
    if controller is not None and not isinstance(controller, ScalingLawAllocationController):
        raise ValueError("controller must be a ScalingLawAllocationController.")
    controller = controller if controller is not None else ScalingLawAllocationController()
    state = ScalingLawState.from_fit(fit, compute_budget)
    action = controller.select_action(state)
    n_star = 10.0**action.log_n
    d_star = compute_budget / (flops_per_token_param * n_star)
    predicted_loss = fit.predict_mean(n_star, d_star)
    heuristic_n, heuristic_d = allocate_fixed_heuristic(compute_budget, flops_per_token_param=flops_per_token_param)
    heuristic_loss = fit.predict_mean(heuristic_n, heuristic_d)
    proposal = controller.register_proposal(
        state,
        action,
        n_params=n_star,
        n_tokens=d_star,
        compute_budget=compute_budget,
        flops_per_token_param=flops_per_token_param,
        predicted_loss=predicted_loss,
        heuristic_predicted_loss=heuristic_loss,
    )
    return n_star, d_star, controller, proposal
