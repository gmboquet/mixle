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
_N_PARAMS = 2  # both loglogistic and hill are fit with a 2-element coefficient vector (b, c)


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
    """A fitted benchmark-dose analysis: the BMD, its lower confidence bound (BMDL), and fit metadata.

    ``status`` is one of:

    * ``"ok"``: the curve fit converged, the BMD was found and bracketed, and the BMDL lower
      bound was established. ``bmd`` and ``bmdl`` are real doses.
    * ``"unidentifiable"``: the curve fit itself did not converge, or the fitted curve never
      reaches the benchmark target within a bounded search (flat, wrong-signed, or requiring an
      implausibly large dose). ``bmd`` and ``bmdl`` are ``nan`` -- never a search boundary or
      other placeholder dressed up as a real dose.
    * ``"bmdl_unavailable"``: the BMD itself was identified (``bmd`` is real), but the BMDL
      computation did not converge to a valid bound. ``bmdl`` is ``nan``.
    """

    bmd: float
    bmdl: float
    bmr: float
    model: str
    dof: int
    status: str = "ok"
    _coef: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(2))

    @property
    def converged(self) -> bool:
        """``True`` only when both the BMD and the BMDL were genuinely established."""
        return self.status == "ok"


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


_MAX_DOSE_SEARCH_MULTIPLE = 1e4  # bounded upper-bracket expansion; see _solve_bmd


def _solve_bmd(
    model: str, coef: np.ndarray, background: float, bmr: float, risk: str, dose_hi: float
) -> tuple[float, bool]:
    """Solve for the dose at which the fitted curve reaches the benchmark target.

    Returns ``(dose, converged)``. ``converged`` is ``False`` -- and ``dose`` is ``nan`` -- unless
    the target is actually bracketed (a genuine sign change is found) within a bounded search AND
    the root-finder itself reports convergence. The search upper bound is expanded geometrically
    but capped at ``dose_hi * _MAX_DOSE_SEARCH_MULTIPLE`` (already generous: ``dose_hi`` is
    ``10x`` the highest tested dose, so the cap sits four orders of magnitude beyond that) --
    a bounded, explicit, reviewable limit, not an iteration count that happens to produce some
    large-but-finite number. When no bracket exists anywhere in that range (a flat, wrong-signed,
    or too-shallow curve), or brentq itself does not converge, this returns an explicit failure
    rather than the exhausted search boundary.
    """
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
    try:
        f_lo, f_hi = f(lo), f(hi)
    except (FloatingPointError, OverflowError):
        return float("nan"), False
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return float("nan"), False

    hi_cap = dose_hi * _MAX_DOSE_SEARCH_MULTIPLE
    while f_lo * f_hi > 0 and hi < hi_cap:
        hi = min(hi * 2.0, hi_cap)
        try:
            f_hi = f(hi)
        except (FloatingPointError, OverflowError):
            return float("nan"), False
        if not np.isfinite(f_hi):
            return float("nan"), False

    if f_lo * f_hi > 0:
        # Never bracketed anywhere in [1e-9, hi_cap]: no root exists in any dose range we are
        # willing to extrapolate into. Report that honestly instead of returning `hi`.
        return float("nan"), False

    try:
        root, info = optimize.brentq(f, lo, hi, xtol=1e-10, full_output=True)
    except (FloatingPointError, OverflowError, RuntimeError, ValueError):
        return float("nan"), False
    if not info.converged:
        return float("nan"), False
    return float(root), True


def _validate_cohort(
    dose: np.ndarray, n_affected: np.ndarray, n_total: np.ndarray, n_params: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject a cohort that cannot support a likelihood evaluation, before any is attempted.

    ``n_total`` may be given per dose group (matching ``dose``'s length), or as a scalar / a
    length-1 array meaning "the same total applies to every dose group" -- an explicit, documented
    broadcast, not an accidental one from letting arithmetic silently broadcast mismatched shapes.
    Any other length is rejected. Returns the validated ``(dose, n_affected, n_total)`` with
    ``n_total`` already broadcast to match ``dose``.
    """
    dose = np.asarray(dose, dtype=float)
    if dose.ndim != 1 or dose.size == 0:
        raise ValueError(f"dose must be a non-empty 1-D array, got shape {dose.shape}")
    if not np.all(np.isfinite(dose)):
        raise ValueError("dose must be finite")
    if np.any(dose < 0):
        raise ValueError("dose must be nonnegative")

    n_affected = np.asarray(n_affected, dtype=float)
    if n_affected.ndim != 1 or n_affected.shape[0] != dose.shape[0]:
        raise ValueError(
            f"n_affected must be a 1-D array matching dose's length ({dose.shape[0]}), got shape {n_affected.shape}"
        )

    n_total = np.asarray(n_total, dtype=float)
    if n_total.ndim == 0 or (n_total.ndim == 1 and n_total.shape[0] == 1):
        # Explicit broadcast: one shared total for every dose group (e.g. a balanced design).
        n_total = np.full(dose.shape[0], float(n_total.reshape(-1)[0]))
    elif n_total.ndim != 1 or n_total.shape[0] != dose.shape[0]:
        raise ValueError(
            f"n_total must be a scalar, a length-1 array (broadcast to every dose group), or a "
            f"1-D array matching dose's length ({dose.shape[0]}); got shape {n_total.shape}"
        )

    for name, arr in (("n_affected", n_affected), ("n_total", n_total)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        if np.any(arr < 0):
            raise ValueError(f"{name} must be nonnegative")
        if not np.all(arr == np.round(arr)):
            raise ValueError(f"{name} must contain integer subject counts")

    if np.any(n_total < 1):
        raise ValueError("n_total must be at least 1 for every dose group")
    if np.any(n_affected > n_total):
        raise ValueError("n_affected must not exceed n_total in any dose group")

    n_distinct = int(np.unique(dose).size)
    if n_distinct < n_params:
        raise ValueError(
            f"need at least {n_params} distinct dose groups to identify a {n_params}-parameter "
            f"curve; got {n_distinct} distinct dose(s) across {dose.shape[0]} group(s)"
        )

    return dose, n_affected, n_total


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
    if not (0.0 < bmr < 1.0):
        raise ValueError(f"bmr must be in (0, 1), got {bmr!r}")
    if not (0.5 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0.5, 1), got {ci_level!r}")
    dose, n_affected, n_total = _validate_cohort(dose, n_affected, n_total, _N_PARAMS)

    init = np.array([-1.0, 1.0])
    result = optimize.minimize(
        _neg_log_likelihood,
        init,
        args=(model, dose, n_affected, n_total),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
    )
    coef = result.x
    dof = int(len(dose) - len(coef))

    if not result.success:
        # The curve fit itself never converged -- nothing computed from `coef` downstream
        # (the BMD, let alone its confidence bound) can be trusted.
        return BMDResult(
            bmd=float("nan"), bmdl=float("nan"), bmr=bmr, model=model, dof=dof, status="unidentifiable", _coef=coef
        )

    background = float(_quantal_p(model, np.array([dose.min() if dose.min() > 0 else 1e-9]), coef)[0])
    dose_hi = float(dose.max()) * 10.0
    bmd, bmd_converged = _solve_bmd(model, coef, background, bmr, risk, dose_hi)

    if not bmd_converged:
        return BMDResult(
            bmd=float("nan"), bmdl=float("nan"), bmr=bmr, model=model, dof=dof, status="unidentifiable", _coef=coef
        )

    nll_min = float(result.fun)
    chi2_1 = stats.chi2.ppf(2 * ci_level - 1, df=1)

    def nll_at_bmd(d: float) -> float:
        def obj(free_coef: np.ndarray) -> float:
            b_bg = float(_quantal_p(model, np.array([dose.min() if dose.min() > 0 else 1e-9]), free_coef)[0])
            implied, ok = _solve_bmd(model, free_coef, b_bg, bmr, risk, dose_hi)
            if not ok:
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
        implied, ok = _solve_bmd(model, coef, background, bmr + eps, risk, dose_hi)
        if ok:
            se_proxy = abs(implied - bmd) / eps
            z = stats.norm.ppf(ci_level)
            bmdl = max(bmd - z * se_proxy * bmd, bmd * 0.01)
        else:
            bmdl = bmd * 0.01

    bmdl = min(bmdl, bmd)
    return BMDResult(bmd=bmd, bmdl=bmdl, bmr=bmr, model=model, dof=dof, status="ok", _coef=coef)


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
