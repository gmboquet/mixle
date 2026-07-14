"""Carcinogenic-risk models: linear no-threshold slope-factor / unit-risk (EPA-IRIS convention).

The regulatory-toxicology answer to "how much extra cancer risk does this exposure add over a
lifetime": a chronic dose is pushed through a chemical-specific potency coefficient under the
linear no-threshold (LNT) low-dose assumption.

  * :class:`SlopeFactor` -- a chemical's potency: an oral cancer slope factor (per mg/kg-day) and/or
    an inhalation unit risk (per ug/m3), as published by EPA IRIS (or supplied by the caller/knowledge
    layer -- this module carries no slope-factor database, see Non-goals).
  * :func:`excess_lifetime_cancer_risk` -- ``risk = LADD * oral_csf`` (oral route) or
    ``risk = conc * inhalation_iur`` (inhalation route), the EPA-IRIS linear low-dose form; falls back
    to the exact ``1 - exp(-dose * slope)`` once the linear approximation would push risk above
    ~0.01. Exposure may be a scalar, a plain sample array, or an IC-1 ``Posterior`` -- in the last case
    the posterior's own :meth:`~mixle.reason.posterior_protocol.Posterior.derived_quantity` pushforward
    is used so the honesty flag ``prior_dominated`` propagates untouched from the exposure posterior.
  * :func:`radon_wlm_risk` -- the BEIR-VI radon working-level-month (WLM) coefficient,
    ``risk = wlm * risk_per_wlm``.

Both return an IC-8-style ``DerivedQuantity`` (samples + credible interval + ``prior_dominated``),
matching the construction used across the rest of ``mixle.analysis`` health/risk models. The concrete
carrier here is :class:`RiskQuantity`, which satisfies the frozen
:class:`mixle.reason.posterior_protocol.DerivedQuantity` protocol structurally (each health/risk module
in ``mixle.analysis`` mints its own small concrete carrier of the same shape rather than sharing one
class, so modules stay independent siblings -- see ``analysis/health_risk.py``'s equivalent).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle.reason.posterior_protocol import DerivedQuantity, Posterior

__all__ = ["SlopeFactor", "RiskQuantity", "excess_lifetime_cancer_risk", "radon_wlm_risk"]


@dataclass
class SlopeFactor:
    """A chemical's cancer potency, EPA-IRIS style.

    Attributes:
        oral_csf: oral cancer slope factor, (mg/kg-day)^-1. ``None`` if the chemical has no
            established oral potency.
        inhalation_iur: inhalation unit risk, (ug/m3)^-1. ``None`` if no established inhalation
            potency.
        sigma_log: log-scale standard deviation of a multiplicative log-normal uncertainty band
            around the point-estimate slope factor (0.0 = treat the slope factor as fixed).
        source: provenance tag for the potency values (default the EPA IRIS database, the
            regulatory-toxicology standard; caller-supplied values should override this).
    """

    oral_csf: float | None = None
    inhalation_iur: float | None = None
    sigma_log: float = 0.0
    source: str = "EPA-IRIS"


@dataclass
class RiskQuantity:
    """A pushforward risk distribution: draws + credible interval + the ``prior_dominated`` flag.

    Structurally satisfies the IC-1 ``DerivedQuantity`` protocol (``samples``, ``prior_dominated``,
    ``credible_interval``) and additionally exposes ``mean`` -- the point estimate callers actually
    read off first.
    """

    samples: np.ndarray
    prior_dominated: bool = False

    @property
    def mean(self) -> float:
        """Point estimate: the sample-mean excess risk."""
        return float(np.mean(self.samples))

    def credible_interval(self, level: float = 0.9) -> tuple[float, float]:
        """Central ``level`` credible interval of the risk samples (e.g. ``level=0.9`` -> 5%/95%)."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        alpha = (1.0 - level) / 2.0
        lo = float(np.quantile(self.samples, alpha))
        hi = float(np.quantile(self.samples, 1.0 - alpha))
        return lo, hi


def _lnt_risk(dose: np.ndarray, slope: np.ndarray | float) -> np.ndarray:
    """EPA-IRIS linear no-threshold form: ``dose * slope``, falling back to ``1 - exp(-dose*slope)``
    once the linear approximation would exceed ~0.01 (the point EPA guidance treats it as unsafe)."""
    product = dose * slope
    return np.where(product < 0.01, product, 1.0 - np.exp(-product))


def excess_lifetime_cancer_risk(
    exposure: Posterior | np.ndarray | float,
    sf: SlopeFactor,
    *,
    route: str = "oral",
    n: int = 2000,
    rng: np.random.Generator | None = None,
) -> DerivedQuantity:
    """Excess lifetime cancer risk under the linear no-threshold model (DR-ALG K7).

    ``route="oral"`` treats ``exposure`` as an LADD (lifetime average daily dose, mg/kg-day) and
    multiplies through :attr:`SlopeFactor.oral_csf`; ``route="inhalation"`` treats ``exposure`` as an
    air concentration (ug/m3) multiplied through :attr:`SlopeFactor.inhalation_iur`.

    Args:
        exposure: the lifetime-average dose/concentration. An IC-1 ``Posterior`` (its
            ``derived_quantity`` pushforward is used, so ``prior_dominated`` propagates from the
            exposure posterior), a plain array of exposure samples (already representing exposure
            uncertainty), or a single deterministic scalar.
        sf: the chemical's :class:`SlopeFactor`.
        route: ``"oral"`` or ``"inhalation"``.
        n: number of posterior draws to take when ``exposure`` is a ``Posterior``, or the number of
            slope-factor draws to take when ``exposure`` is a bare scalar and ``sf.sigma_log > 0``.
        rng: numpy random Generator (a fresh default one is created if omitted).

    Returns:
        A :class:`DerivedQuantity` of excess lifetime cancer risk (samples + CI + ``prior_dominated``).
    """
    if route not in ("oral", "inhalation"):
        raise ValueError(f"route must be 'oral' or 'inhalation', got {route!r}.")
    csf = sf.oral_csf if route == "oral" else sf.inhalation_iur
    if csf is None:
        raise ValueError(f"SlopeFactor has no {route} potency coefficient set.")
    rng = rng if rng is not None else np.random.default_rng()

    def _apply(draws: np.ndarray) -> np.ndarray:
        dose = np.atleast_1d(np.asarray(draws, dtype=float))
        if dose.ndim > 1:
            dose = dose.reshape(dose.shape[0], -1)[:, 0]
        if sf.sigma_log > 0:
            slope = csf * rng.lognormal(mean=0.0, sigma=sf.sigma_log, size=dose.shape)
        else:
            slope = csf
        return _lnt_risk(dose, slope)

    if isinstance(exposure, Posterior):
        dq = exposure.derived_quantity(_apply, n, rng)
        samples = np.atleast_1d(np.asarray(dq.samples, dtype=float))
        prior_dominated = bool(dq.prior_dominated)
    elif isinstance(exposure, np.ndarray):
        samples = _apply(exposure)
        prior_dominated = False
    else:
        dose_scalar = float(exposure)
        reps = n if sf.sigma_log > 0 else 1
        samples = _apply(np.full(reps, dose_scalar))
        prior_dominated = False

    return RiskQuantity(samples=samples, prior_dominated=prior_dominated)


def radon_wlm_risk(
    wlm: np.ndarray | float,
    *,
    risk_per_wlm: float = 5.38e-4,
    n: int = 2000,
    rng: np.random.Generator | None = None,
) -> DerivedQuantity:
    """Radon lung-cancer risk from cumulative working-level-months (BEIR-VI linear coefficient).

    ``risk = wlm * risk_per_wlm``, the BEIR-VI committee's linear excess-relative-risk coefficient
    per WLM of cumulative radon-progeny exposure (default ``5.38e-4`` per WLM).

    Args:
        wlm: cumulative working-level-months, a scalar or an array of samples (already representing
            exposure uncertainty).
        risk_per_wlm: BEIR-VI risk coefficient per WLM.
        n: unused when ``wlm`` is an array (kept for signature symmetry / future posterior support).
        rng: unused for the array/scalar paths (kept for signature symmetry).

    Returns:
        A :class:`DerivedQuantity` of radon-attributable lung-cancer risk.
    """
    # n / rng are accepted for signature symmetry with excess_lifetime_cancer_risk and to leave room
    # for a future posterior-valued wlm; the deterministic BEIR-VI formula needs neither.
    if isinstance(wlm, np.ndarray):
        samples = np.atleast_1d(np.asarray(wlm, dtype=float)) * risk_per_wlm
    else:
        samples = np.array([float(wlm) * risk_per_wlm])
    return RiskQuantity(samples=samples, prior_dominated=False)
