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
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize, stats

from mixle.analysis._evidence import require_delivered_draws
from mixle.analysis._interval import validated_level

if TYPE_CHECKING:
    from mixle.reason.posterior_protocol import DerivedQuantity, Posterior

_MODELS = ("loglogistic", "hill")
_N_PARAMS = 2  # both loglogistic and hill are fit with a 2-element coefficient vector (b, c)


@dataclass(frozen=True)
class _SampleDerivedQuantity:
    """A concrete IC-1 `DerivedQuantity`: a draw matrix + the honesty flag, CI by empirical quantile.

    Construction validates ``samples``: non-empty and finite (no NaN/Inf) -- defense-in-depth so
    invalid state can never flow downstream to a caller, even if some upstream pushforward fails to
    validate its own inputs (the same "samples-carrying result type" guard applied to
    ``carcinogenic_risk.RiskQuantity`` and ``health_risk._SampleDerivedQuantity``).
    """

    samples: np.ndarray
    prior_dominated: bool = False

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("_SampleDerivedQuantity.samples must be a non-empty one-dimensional vector.")
        if not np.isfinite(arr).all():
            raise ValueError("_SampleDerivedQuantity.samples must be finite (no NaN/Inf).")
        arr = arr.copy()
        arr.setflags(write=False)
        object.__setattr__(self, "samples", arr)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        level = validated_level(level)
        alpha = (1.0 - level) / 2.0
        lo, hi = np.quantile(self.samples, [alpha, 1.0 - alpha])
        return float(lo), float(hi)


@dataclass(frozen=True)
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

    ``bmd_se`` is the delta-method standard error of the BMD (``nan`` unless ``status ==
    "ok"``); ``bmdl = bmd - z * bmd_se`` clipped at 0, ``z`` the one-sided normal quantile for
    the requested confidence level.
    """

    bmd: float
    bmdl: float
    bmr: float
    model: str
    dof: int
    status: str = "ok"
    bmd_se: float = float("nan")
    _coef: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        def scalar(name: str, value: Any) -> float:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar.")
            return float(value)

        bmd = scalar("bmd", self.bmd)
        bmdl = scalar("bmdl", self.bmdl)
        bmr = scalar("bmr", self.bmr)
        bmd_se = scalar("bmd_se", self.bmd_se)
        if not 0.0 < bmr < 1.0 or not np.isfinite(bmr):
            raise ValueError("bmr must be finite and strictly between 0 and 1.")
        if self.model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}, got {self.model!r}.")
        if isinstance(self.dof, (bool, np.bool_)) or not isinstance(self.dof, Integral):
            raise TypeError("dof must be an exact non-Boolean integer.")
        if self.dof < 0:
            raise ValueError("dof must be nonnegative.")
        if self.status not in {"ok", "unidentifiable", "bmdl_unavailable"}:
            raise ValueError(f"unknown BMDResult status {self.status!r}.")

        if self.status == "ok":
            if not (np.isfinite(bmd) and bmd > 0.0):
                raise ValueError("an ok BMDResult requires a finite positive bmd.")
            if not (np.isfinite(bmdl) and 0.0 <= bmdl <= bmd):
                raise ValueError("an ok BMDResult requires finite 0 <= bmdl <= bmd.")
            if not (np.isfinite(bmd_se) and bmd_se >= 0.0):
                raise ValueError("an ok BMDResult requires a finite nonnegative bmd_se.")
        elif self.status == "unidentifiable":
            if not (np.isnan(bmd) and np.isnan(bmdl) and np.isnan(bmd_se)):
                raise ValueError("an unidentifiable BMDResult requires bmd, bmdl, and bmd_se to be NaN.")
        elif not (np.isfinite(bmd) and bmd > 0.0 and np.isnan(bmdl) and np.isnan(bmd_se)):
            raise ValueError("a bmdl_unavailable BMDResult requires a finite positive bmd and NaN bmdl/bmd_se.")

        coef = np.asarray(self._coef, dtype=float)
        if coef.shape != (_N_PARAMS,) or not np.all(np.isfinite(coef)):
            raise ValueError(f"_coef must be a finite {_N_PARAMS}-element coefficient vector.")
        coef = coef.copy()
        coef.setflags(write=False)
        object.__setattr__(self, "bmd", bmd)
        object.__setattr__(self, "bmdl", bmdl)
        object.__setattr__(self, "bmr", bmr)
        object.__setattr__(self, "bmd_se", bmd_se)
        object.__setattr__(self, "dof", int(self.dof))
        object.__setattr__(self, "_coef", coef)

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


def _benchmark_target(background: float, bmr: float, risk: str) -> float | None:
    """The response level the BMD is defined to reach, or ``None`` when it is not a probability.

    ``"extra"`` risk measures ``bmr`` as a fraction of the background-to-certainty headroom, so
    ``background + bmr * (1 - background)`` is below one for any ``background < 1``. ``"added"``
    risk adds ``bmr`` outright, so ``background + bmr`` can exceed one -- and the fitted background
    is estimated from the data, not chosen by the caller, so an ``added`` request that looked
    perfectly reasonable when it was made (say ``bmr=0.10``) becomes unattainable as soon as the
    curve fits a background above ``0.90``.

    That case used to be clipped to ``1 - 1e-9`` and solved anyway (MXR-080-1579), which reported a
    converged BMD for a benchmark response the caller never asked for: backgrounds of ``0.95``,
    ``0.99`` and ``0.999`` at ``bmr=0.10`` all returned the identical dose, because all three had
    been silently replaced by the same substituted target. Returning ``None`` instead lets the
    caller report the honest ``unidentifiable`` result rather than solving a different question.
    """
    if risk == "extra":
        target = background + bmr * (1.0 - background)
    elif risk == "added":
        target = background + bmr
    else:
        raise ValueError(f"unknown risk convention {risk!r}; expected 'extra' or 'added'")
    # `< 1` and not `<= 1`: a target of exactly one is only reached where the curve saturates, which
    # is an asymptote rather than a root, so it identifies no dose either.
    return float(target) if np.isfinite(target) and target < 1.0 else None


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

    Also returns an explicit failure when the requested benchmark response is not a probability at
    all -- see :func:`_benchmark_target`. Substituting a reachable target for an unreachable one
    (MXR-080-1579) answers a question the caller did not ask.
    """
    target = _benchmark_target(background, bmr, risk)
    if target is None:
        return float("nan"), False

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


def _bmd_gradient(
    model: str, coef: np.ndarray, dose_min_eff: float, bmr: float, risk: str, bmd: float
) -> np.ndarray | None:
    """``(d(BMD)/db, d(BMD)/dc)`` by implicit differentiation of the BMD-defining equation.

    The BMD solves ``F(d, b, c) = p(d; b, c) - target(b, c) = 0``, where ``target`` is itself a
    function of ``(b, c)`` through the fitted background rate ``p(dose_min_eff; b, c)``. By the
    implicit function theorem, ``d(BMD)/dtheta = -(dF/dtheta) / (dF/dd)`` for each parameter
    ``theta``. Every partial is a central finite difference of the same ``F`` -- this needs no
    per-model closed-form derivative, so it applies unchanged to both loglogistic and hill.

    Returns ``None`` (gradient unavailable) if any partial is non-finite, or the curve is locally
    flat in dose at the BMD (``dF/dd ~ 0``): dividing by a near-zero slope is ill-conditioned and
    would manufacture an arbitrarily large, meaningless gradient.
    """

    def target_of(b: float, c: float) -> float:
        bg = float(_quantal_p(model, np.array([dose_min_eff]), np.array([b, c]))[0])
        # NaN, not the old clip to `1 - 1e-9` (MXR-080-1579): if a finite-difference perturbation of
        # the coefficients pushes the benchmark target off the probability scale, this partial is
        # undefined, and the non-finite check below turns that into an unavailable gradient rather
        # than a derivative of a substituted target.
        target = _benchmark_target(bg, bmr, risk)
        return float("nan") if target is None else target

    def big_f(d: float, b: float, c: float) -> float:
        p_d = _quantal_p(model, np.array([d]), np.array([b, c]))[0]
        return float(p_d - target_of(b, c))

    b0, c0 = float(coef[0]), float(coef[1])
    h_d = max(abs(bmd) * 1e-4, 1e-9)
    h_b = max(abs(b0) * 1e-4, 1e-6)
    h_c = max(abs(c0) * 1e-4, 1e-6)

    dF_dd = (big_f(bmd + h_d, b0, c0) - big_f(bmd - h_d, b0, c0)) / (2.0 * h_d)
    dF_db = (big_f(bmd, b0 + h_b, c0) - big_f(bmd, b0 - h_b, c0)) / (2.0 * h_b)
    dF_dc = (big_f(bmd, b0, c0 + h_c) - big_f(bmd, b0, c0 - h_c)) / (2.0 * h_c)

    if not (np.isfinite(dF_dd) and np.isfinite(dF_db) and np.isfinite(dF_dc)):
        return None
    if abs(dF_dd) < 1e-10:
        return None

    grad = np.array([-dF_db / dF_dd, -dF_dc / dF_dd])
    return grad if np.all(np.isfinite(grad)) else None


def _observed_information_cov(
    model: str, coef: np.ndarray, dose: np.ndarray, n_affected: np.ndarray, n_total: np.ndarray
) -> np.ndarray | None:
    """Asymptotic covariance of the MLE via the inverse observed Fisher information.

    The Hessian of the negative log-likelihood at ``coef`` is estimated by central finite
    differences (Nelder-Mead, used to fit ``coef``, provides no Hessian of its own). Returns
    ``None`` if the Hessian is non-finite, singular, or not positive-semidefinite once inverted --
    the last case means the fit sits at a saddle rather than a genuine likelihood maximum, so the
    normal approximation the delta method relies on does not hold.
    """

    def nll(x: np.ndarray) -> float:
        return _neg_log_likelihood(x, model, dose, n_affected, n_total)

    n = len(coef)
    h = np.maximum(np.abs(coef) * 1e-4, 1e-5)
    f0 = nll(coef)
    if not np.isfinite(f0):
        return None

    hess = np.zeros((n, n))
    for i in range(n):
        step_i = np.zeros(n)
        step_i[i] = h[i]
        hess[i, i] = (nll(coef + step_i) - 2.0 * f0 + nll(coef - step_i)) / h[i] ** 2
        for j in range(i + 1, n):
            step_j = np.zeros(n)
            step_j[j] = h[j]
            cross = (
                nll(coef + step_i + step_j)
                - nll(coef + step_i - step_j)
                - nll(coef - step_i + step_j)
                + nll(coef - step_i - step_j)
            ) / (4.0 * h[i] * h[j])
            hess[i, j] = hess[j, i] = cross

    if not np.all(np.isfinite(hess)):
        return None
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(cov)):
        return None

    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-8 * max(1.0, float(np.max(np.abs(eigvals)))):
        return None
    return cov


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
    ``bmr`` extra (or added) risk over the fitted background rate. The BMDL is the one-sided
    ``ci_level`` lower confidence bound on the BMD by the delta method: the gradient of the BMD
    with respect to the fitted coefficients is obtained by implicit differentiation of the
    BMD-defining equation, the coefficient covariance by inverting the observed Fisher
    information at the MLE, and ``Var(BMD)`` by propagating one through the other
    (``grad @ Cov @ grad``); ``BMDL = BMD - z * SE(BMD)`` with ``z`` the one-sided normal
    quantile for ``ci_level``, clipped at 0 (dose cannot be negative). See ``BMDResult.status``
    for what happens when any step of that chain fails to converge.
    """
    if not isinstance(model, str) or model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {_MODELS}")
    if risk not in {"extra", "added"}:
        raise ValueError("risk must be either 'extra' or 'added'")
    if isinstance(bmr, (bool, np.bool_)) or not isinstance(bmr, Real):
        raise TypeError("bmr must be a real scalar probability")
    bmr = float(bmr)
    if not np.isfinite(bmr) or not 0.0 < bmr < 1.0:
        raise ValueError(f"bmr must be in (0, 1), got {bmr!r}")
    if isinstance(ci_level, (bool, np.bool_)) or not isinstance(ci_level, Real):
        raise TypeError("ci_level must be a real scalar probability")
    ci_level = float(ci_level)
    if not np.isfinite(ci_level) or not 0.5 < ci_level < 1.0:
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

    dose_min_eff = dose.min() if dose.min() > 0 else 1e-9
    background = float(_quantal_p(model, np.array([dose_min_eff]), coef)[0])
    dose_hi = float(dose.max()) * 10.0
    bmd, bmd_converged = _solve_bmd(model, coef, background, bmr, risk, dose_hi)

    if not bmd_converged:
        return BMDResult(
            bmd=float("nan"), bmdl=float("nan"), bmr=bmr, model=model, dof=dof, status="unidentifiable", _coef=coef
        )

    grad = _bmd_gradient(model, coef, dose_min_eff, bmr, risk, bmd)
    cov = _observed_information_cov(model, coef, dose, n_affected, n_total)

    bmd_se = float("nan")
    if grad is not None and cov is not None:
        var_bmd = float(grad @ cov @ grad)
        if np.isfinite(var_bmd) and var_bmd >= 0.0:
            bmd_se = float(np.sqrt(var_bmd))

    if np.isfinite(bmd_se):
        z = float(stats.norm.ppf(ci_level))
        # A one-sided normal-approximation bound against the natural dose >= 0 boundary: when
        # the delta-method interval would dip below 0, clipping at the boundary is the standard
        # treatment (not a fabricated number -- 0 is itself a valid, if uninformative, lower
        # bound whenever the data cannot statistically rule out a BMD near the origin).
        bmdl = max(bmd - z * bmd_se, 0.0)
        return BMDResult(bmd=bmd, bmdl=bmdl, bmr=bmr, model=model, dof=dof, status="ok", bmd_se=bmd_se, _coef=coef)

    return BMDResult(
        bmd=bmd, bmdl=float("nan"), bmr=bmr, model=model, dof=dof, status="bmdl_unavailable", bmd_se=bmd_se, _coef=coef
    )


def _as_dose_samples(exposure: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    if isinstance(exposure, (bool, np.bool_)):
        raise TypeError("exposure must be a real dose, not a Boolean value")
    arr = np.asarray(exposure, dtype=float)
    if arr.ndim == 0:
        if not np.isfinite(arr) or arr < 0:
            raise ValueError(f"exposure scalar must be finite and nonnegative, got {float(arr)!r}")
        return np.full(int(n), float(arr))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("exposure draws must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError("exposure draws must all be finite")
    if np.any(arr < 0):
        raise ValueError("exposure draws must all be nonnegative")
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

    ``uf`` must be finite and strictly positive, ``n`` a positive exact sample count, and
    ``exposure`` draws finite and nonnegative -- all rejected up front, before any division or
    resampling. ``bmd.bmdl`` must also be finite: a ``BMDResult`` with ``status != "ok"`` (see
    :class:`BMDResult`) has no real BMDL to divide by an uncertainty factor, and a NaN silently
    compared with ``draws > nan`` would quietly evaluate to all-``False`` -- a fabricated-looking
    "0% exceedance" result -- rather than surfacing the underlying unidentifiability. The
    finite/nonnegative exposure check applies uniformly to a `Posterior`'s own draws too, not only
    to a plain array/scalar: it lives inside ``fn`` below (the pushforward handed to
    ``Posterior.derived_quantity``), not in a separate top-level check on the raw ``exposure`` --
    otherwise a mis-specified exposure posterior could emit a negative or NaN draw that flowed
    straight through `draws > rfd` (silently ``False`` for NaN) into a confident-looking "not
    exceeding" result with no warning.
    """
    from mixle.reason.posterior_protocol import Posterior

    if isinstance(uf, (bool, np.bool_)) or not isinstance(uf, Real):
        raise TypeError(f"uf must be a finite positive scalar uncertainty factor, got {uf!r}")
    uf = float(uf)
    if not np.isfinite(uf) or uf <= 0:
        raise ValueError(f"uf must be a finite positive uncertainty factor, got {uf!r}")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
        raise ValueError(f"n must be a positive exact non-Boolean integer sample count, got {n!r}")
    n = int(n)
    if not isinstance(bmd, BMDResult):
        raise TypeError(f"bmd must be a validated BMDResult, got {type(bmd).__name__}")
    if bmd.status != "ok":
        raise ValueError(f"bmd status is {bmd.status!r}, not 'ok'; cannot compute an RfD without an identified BMDL")

    rng = rng if rng is not None else np.random.default_rng()
    rfd = bmd.bmdl / uf

    def fn(draws: np.ndarray) -> np.ndarray:
        # Validated here -- not only in `_as_dose_samples` -- so a `Posterior`'s own draws are
        # checked exactly like a plain array/scalar's: previously only the array/scalar path (via
        # `_as_dose_samples`) rejected a negative or non-finite exposure draw, while the `Posterior`
        # branch below handed `fn` straight to `exposure.derived_quantity`, unchecked.
        arr = np.asarray(draws, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"exposure posterior must produce exactly one scalar for each of {n} draws, got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("exposure draws must all be finite")
        if np.any(arr < 0):
            raise ValueError("exposure draws must all be nonnegative")
        return (arr > rfd).astype(float)

    if isinstance(exposure, Posterior):
        derived = exposure.derived_quantity(fn, n, rng)
        prior_dominated = getattr(derived, "prior_dominated", None)
        if not isinstance(prior_dominated, (bool, np.bool_)):
            raise TypeError("posterior-derived exposure result must carry a Boolean prior_dominated flag")
        # Exact posterior-delivery receipt (MXR-080-1900). `fn` already rejects an INPUT draw set that
        # is not exactly `n` long, but that only fires if `derived_quantity` actually calls it with
        # everything it intends to return: a `derived_quantity` that pushes forward `n` draws and then
        # subsamples, filters or caches its own OUTPUT delivered a shorter exceedance distribution,
        # and `_SampleDerivedQuantity` checks non-empty/1-D/finite but never the count. `P(exposure >
        # RfD)` read off a fraction of the requested draws is indistinguishable from the full answer.
        return _SampleDerivedQuantity(
            samples=require_delivered_draws(
                getattr(derived, "samples", None), n, what="the exposure posterior's derived quantity"
            ),
            prior_dominated=bool(prior_dominated),
        )
    draws = _as_dose_samples(exposure, n, rng)
    return _SampleDerivedQuantity(samples=fn(draws), prior_dominated=False)
