"""Teratogenic / reproductive / developmental risk: benchmark-dose (BMD/BMDL) analysis (K8, work-plan §7-K).

Fits a quantal log-logistic (or Hill) dose-response curve to a developmental-toxicity cohort
(``dose``, ``n_affected`` out of ``n_total`` per dose group) by maximum likelihood, following the
EPA BMDS convention: the benchmark dose (BMD) is the dose giving a specified benchmark response
(default 10% extra risk over background); the BMDL is a one-sided lower confidence bound on the BMD.
``rfd_exceedance`` divides the BMDL by an uncertainty factor to get a reference dose (RfD), then
pushes an exposure `Posterior` (IC-1) through the RfD threshold into an IC-8 `DerivedQuantity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize, stats

if TYPE_CHECKING:
    from mixle.reason.posterior_protocol import DerivedQuantity, Posterior

_MODELS = ("loglogistic", "hill")


@dataclass(frozen=True)
class _SampleDerivedQuantity:
    """A concrete IC-1 `DerivedQuantity`: a draw matrix + the honesty flag, CI by empirical quantile."""

    samples: np.ndarray
    prior_dominated: bool = False

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        alpha = (1.0 - level) / 2.0
        lo, hi = np.quantile(self.samples, [alpha, 1.0 - alpha])
        return float(lo), float(hi)


@dataclass
class BMDResult:
    """A fitted benchmark-dose analysis: the BMD, its lower confidence bound (BMDL), and fit metadata."""

    bmd: float
    bmdl: float
    bmr: float
    model: str
    dof: int
    _coef: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(2))


def _quantal_p(model: str, dose: np.ndarray, coef: np.ndarray) -> np.ndarray:
    dose = np.clip(dose, 1e-12, None)
    b, c = coef
    if model == "loglogistic":
        z = np.clip(-(b + c * np.log(dose)), -500, 500)
        return 1.0 / (1.0 + np.exp(z))
    if model == "hill":
        ec50 = np.exp(-b / c) if c != 0 else 1.0
        n = max(c, 1e-6)
        return dose**n / (ec50**n + dose**n)
    raise ValueError(f"unknown model {model!r}; expected one of {_MODELS}")


def _neg_log_likelihood(
    coef: np.ndarray, model: str, dose: np.ndarray, n_affected: np.ndarray, n_total: np.ndarray
) -> float:
    p = np.clip(_quantal_p(model, dose, coef), 1e-9, 1 - 1e-9)
    ll = n_affected * np.log(p) + (n_total - n_affected) * np.log(1 - p)
    return -float(np.sum(ll))


def _solve_bmd(model: str, coef: np.ndarray, background: float, bmr: float, risk: str, dose_hi: float) -> float:
    if risk == "extra":
        target = background + bmr * (1.0 - background)
    elif risk == "added":
        target = background + bmr
    else:
        raise ValueError(f"unknown risk convention {risk!r}; expected 'extra' or 'added'")
    target = min(target, 1.0 - 1e-9)

    def f(d: float) -> float:
        return _quantal_p(model, np.array([d]), coef)[0] - target

    lo, hi = 1e-9, dose_hi
    f_lo, f_hi = f(lo), f(hi)
    tries = 0
    while f_lo * f_hi > 0 and tries < 40:
        hi *= 2.0
        f_hi = f(hi)
        tries += 1
    if f_lo * f_hi > 0:
        return float(hi)
    return float(optimize.brentq(f, lo, hi))


def benchmark_dose(
    dose: np.ndarray,
    n_affected: np.ndarray,
    n_total: np.ndarray,
    *,
    bmr: float = 0.10,
    model: str = "loglogistic",
    risk: str = "extra",
    ci_level: float = 0.95,
) -> BMDResult:
    """Fit a quantal dose-response and report the benchmark dose (BMD) and its lower bound (BMDL).

    ``dose``/``n_affected``/``n_total`` are per-dose-group arrays (``n_affected <= n_total``).
    The curve is fit by maximum likelihood (DR-ALG K8); the BMD solves for the dose giving
    ``bmr`` extra (or added) risk over the fitted background rate; the BMDL is the one-sided
    ``ci_level`` lower confidence bound on the BMD by profile likelihood, falling back to the
    delta method if the profile search fails to bracket a root.
    """
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {_MODELS}")
    dose = np.asarray(dose, dtype=float)
    n_affected = np.asarray(n_affected, dtype=float)
    n_total = np.asarray(n_total, dtype=float)

    init = np.array([-1.0, 1.0])
    result = optimize.minimize(
        _neg_log_likelihood,
        init,
        args=(model, dose, n_affected, n_total),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
    )
    coef = result.x
    background = float(_quantal_p(model, np.array([dose.min() if dose.min() > 0 else 1e-9]), coef)[0])
    dose_hi = float(dose.max()) * 10.0
    bmd = _solve_bmd(model, coef, background, bmr, risk, dose_hi)

    nll_min = float(result.fun)
    chi2_1 = stats.chi2.ppf(2 * ci_level - 1, df=1)

    def nll_at_bmd(d: float) -> float:
        def obj(free_coef: np.ndarray) -> float:
            b_bg = float(_quantal_p(model, np.array([dose.min() if dose.min() > 0 else 1e-9]), free_coef)[0])
            try:
                implied = _solve_bmd(model, free_coef, b_bg, bmr, risk, dose_hi)
            except (FloatingPointError, OverflowError, RuntimeError, ValueError):
                return 1e12
            penalty = 1e6 * (implied - d) ** 2
            return _neg_log_likelihood(free_coef, model, dose, n_affected, n_total) + penalty

        r = optimize.minimize(obj, coef, method="Nelder-Mead", options={"maxiter": 2000})
        return float(r.fun) - nll_min

    try:
        lo_search, hi_search = 1e-9, bmd
        f_lo = nll_at_bmd(lo_search) - chi2_1 / 2.0
        f_hi = nll_at_bmd(hi_search) - chi2_1 / 2.0
        if f_lo * f_hi > 0:
            raise ValueError("no bracket")
        bmdl = float(optimize.brentq(lambda d: nll_at_bmd(d) - chi2_1 / 2.0, lo_search, hi_search, xtol=1e-6))
    except (FloatingPointError, OverflowError, RuntimeError, ValueError):
        eps = max(bmd * 1e-3, 1e-9)
        se_proxy = abs(_solve_bmd(model, coef, background, bmr + eps, risk, dose_hi) - bmd) / eps
        z = stats.norm.ppf(ci_level)
        bmdl = max(bmd - z * se_proxy * bmd, bmd * 0.01)

    bmdl = min(bmdl, bmd)
    dof = int(len(dose) - len(coef))
    return BMDResult(bmd=bmd, bmdl=bmdl, bmr=bmr, model=model, dof=dof, _coef=coef)


def _as_dose_samples(exposure: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(exposure, dtype=float)
    if arr.ndim == 0:
        return np.full(int(n), float(arr))
    if len(arr) == n:
        return arr
    idx = rng.integers(0, len(arr), size=n)
    return arr[idx]


def rfd_exceedance(
    exposure: Posterior | np.ndarray,
    bmd: BMDResult,
    *,
    uf: float = 100.0,
    n: int = 2000,
    rng: np.random.Generator | None = None,
) -> DerivedQuantity:
    """`P(exposure > RfD)` as an IC-8 `DerivedQuantity`, where `RfD = BMDL / uf` (EPA convention).

    ``exposure`` may be an IC-1 `Posterior` (the pushforward runs through its own
    ``derived_quantity`` so `prior_dominated` propagates), an array of exposure draws (resampled
    to ``n`` if its length differs), or a bare scalar (a degenerate point mass).
    """
    from mixle.reason.posterior_protocol import Posterior

    rng = rng if rng is not None else np.random.default_rng()
    rfd = bmd.bmdl / uf

    def fn(draws: np.ndarray) -> np.ndarray:
        return (draws > rfd).astype(float)

    if isinstance(exposure, Posterior):
        return exposure.derived_quantity(fn, n, rng)
    draws = _as_dose_samples(exposure, n, rng)
    return _SampleDerivedQuantity(samples=fn(draws), prior_dominated=False)
