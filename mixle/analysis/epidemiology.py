"""Epidemiological cohort attribution: Cox PH + Aalen-Johansen (work-plan Sec.7-K, K9).

Given a cohort's covariates (one column marking the exposure of interest), event times, and censoring,
:func:`cohort_attribution` answers the two questions a health-risk sign-off actually needs: *how much*
does the exposure multiply the hazard (the hazard ratio, ``HR``), and *what fraction* of the hazard among
the exposed is attributable to it (``AF = (HR - 1) / HR``). Both come with an honest interval, not a bare
point estimate:

  * the hazard ratio and its Wald confidence interval fall straight out of the already-frozen
    :func:`mixle.inference.survival.cox_ph` partial-likelihood fit (``CoxResult.se``) -- no new survival
    math here, this module is a thin, health-domain-shaped wrapper around it (K9's non-goal list).
  * the attributable-fraction interval is a nonparametric bootstrap over cohort resamples, refitting
    ``cox_ph`` each draw; the bootstrap AF distribution is also handed back in ``provenance`` as an
    object satisfying the IC-1 ``DerivedQuantity`` shape (``samples`` + ``credible_interval`` +
    ``prior_dominated``), so it slots into the same decision-quantity plumbing (IC-8) the rest of the
    codebase uses for "never emit a number without its honesty flag."
  * ``competing=True`` additionally runs :func:`mixle.inference.survival.aalen_johansen` on the raw
    (multi-cause) event labels for cause-specific cumulative incidence -- the right absolute-risk curve
    when a competing cause (e.g. all-other-cause mortality) can remove a subject from the risk set before
    the outcome of interest can occur, which plain ``1 - KM`` overstates.
  * ``latency > 0`` encodes an exposure-to-effect lag as left-truncation: subjects contribute to the risk
    set (and to the bootstrap) only once they have survived past the latency period, via ``cox_ph``'s
    counting-process ``start`` argument. This is the standard occupational/environmental-epi device for
    "the exposure cannot have caused an event that happened before the biological lag had elapsed."

Every number in the returned :class:`CohortAttribution` traces back to one ``cox_ph`` fit (plus, when
``competing``, one ``aalen_johansen`` run) and one seeded RNG -- both recorded in ``provenance``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from mixle.inference.survival import aalen_johansen, cox_ph

__all__ = ["CohortAttribution", "cohort_attribution"]


class _AFDistribution:
    """The bootstrap attributable-fraction distribution, shaped to satisfy IC-1's ``DerivedQuantity``.

    Frequentist bootstrap over cohort resamples -- there is no prior/regulariser in a Cox partial-
    likelihood fit, so ``prior_dominated`` is always ``False``; the honesty flag is carried for
    interface uniformity with the rest of the decision-quantity surface (IC-8), not because it can ever
    trip here.
    """

    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        lo = np.nanquantile(self.samples, a)
        hi = np.nanquantile(self.samples, 1.0 - a)
        return lo, hi


@dataclass
class CohortAttribution:
    """Cox-PH hazard ratio + attributable fraction for one exposure, with intervals and provenance.

    Attributes:
        hazard_ratio: ``exp(coef[exposure_col])`` from the fitted Cox model.
        hr_ci: Wald ``(lo, hi)`` confidence interval on ``hazard_ratio`` (log scale, from ``CoxResult.se``).
        attributable_fraction: fraction of hazard among the *exposed* attributable to exposure,
            ``(hazard_ratio - 1) / hazard_ratio``.
        af_ci: bootstrap ``(lo, hi)`` confidence interval on ``attributable_fraction``.
        cif: cause-specific Aalen-Johansen cumulative incidence, ``{cause: array}`` (empty unless
            ``competing=True``).
        provenance: fit diagnostics, the RNG seed, and a bootstrap ``DerivedQuantity``-shaped AF summary.
    """

    hazard_ratio: float
    hr_ci: tuple[float, float]
    attributable_fraction: float
    af_ci: tuple[float, float]
    cif: dict[int, np.ndarray]
    provenance: dict


def _fit_lagged(x: np.ndarray, time: np.ndarray, event: np.ndarray, latency: float):
    """Fit ``cox_ph`` with an optional latency (left-truncation): subjects enter risk at ``latency``.

    Rows whose observed time does not exceed ``latency`` never survive into the truncated risk set (the
    exposure cannot yet have had an effect), so they are dropped rather than passed with a degenerate
    ``(start, stop]`` interval. Returns the fit plus the number of rows actually used.
    """
    if latency > 0:
        keep = time > latency
        x, time, event = x[keep], time[keep], event[keep]
        start = np.full(time.shape[0], latency)
    else:
        start = None
    return cox_ph(x, time, event, start=start, ties="efron"), time.shape[0]


def cohort_attribution(
    covariates: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    exposure_col: int = 0,
    competing: bool = False,
    latency: float = 0.0,
    n_boot: int = 1000,
    rng=None,
) -> CohortAttribution:
    """Attribute a cohort's hazard to one exposure via Cox PH, with an Aalen-Johansen competing-risks CIF.

    Args:
        covariates: ``(n, p)`` covariates; column ``exposure_col`` is the exposure of interest.
        time: ``(n,)`` event/censoring times.
        event: ``(n,)`` event indicator. Binary (``0``/``1``) when ``competing=False``; an integer cause
            label (``0`` = censored, ``1`` = the outcome of interest, ``2..K`` = competing causes) when
            ``competing=True``. Either way, the Cox fit is cause-specific for cause ``1``: competing-cause
            events are censored at their time for the hazard-ratio fit (only the CIF, when requested,
            reports the competing causes' own cumulative incidence).
        exposure_col: column of ``covariates`` treated as the exposure.
        competing: if True, also run `aalen_johansen` for cause-specific cumulative incidence.
        latency: exposure-to-effect lag (left-truncation, via `cox_ph`'s `start`); ``0`` disables it.
        n_boot: cohort-resample bootstrap draws for the attributable-fraction interval.
        rng: seed, `numpy.random.Generator`, or None.

    Returns:
        A :class:`CohortAttribution`.
    """
    cov_arr = np.asarray(covariates, dtype=float)
    x = cov_arr.reshape(-1, 1) if cov_arr.ndim == 1 else cov_arr
    t = np.asarray(time, dtype=float)
    e_raw = np.asarray(event, dtype=int)
    n = x.shape[0]
    rng = np.random.default_rng(rng)

    # Cause-specific event indicator for the hazard-ratio fit: only cause 1 counts as an event; true
    # censoring (0) AND competing causes (>=2) are both censored here, per the cause-specific-hazard
    # convention -- the CIF below is what reports the competing causes' own incidence.
    cox_event = (e_raw == 1).astype(float)

    fit, n_fit_rows = _fit_lagged(x, t, cox_event, latency)
    beta = float(fit.coef[exposure_col])
    se = float(fit.se[exposure_col])
    hazard_ratio = float(np.exp(beta))
    z = stats.norm.ppf(0.975)
    hr_ci = (float(np.exp(beta - z * se)), float(np.exp(beta + z * se)))

    attributable_fraction = (hazard_ratio - 1.0) / hazard_ratio

    exposed = x[:, exposure_col] > 0
    exposure_prevalence = float(exposed.mean())
    denom = 1.0 + exposure_prevalence * (hazard_ratio - 1.0)
    population_attributable_fraction = (
        float(exposure_prevalence * (hazard_ratio - 1.0) / denom) if denom != 0 else float("nan")
    )

    boot_af = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            fit_b, _ = _fit_lagged(x[idx], t[idx], cox_event[idx], latency)
            hr_b = float(np.exp(fit_b.coef[exposure_col]))
            if hr_b > 0:
                boot_af[b] = (hr_b - 1.0) / hr_b
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue  # a degenerate resample (e.g. no variation in the exposure column); skip it

    n_boot_valid = int(np.sum(np.isfinite(boot_af)))
    if n_boot_valid > 0:
        af_ci = (
            float(np.nanquantile(boot_af, 0.025)),
            float(np.nanquantile(boot_af, 0.975)),
        )
    else:
        af_ci = (float("nan"), float("nan"))

    cif: dict[int, np.ndarray] = {}
    aj: dict[str, Any] | None = None
    if competing:
        aj = aalen_johansen(t, e_raw)
        cif = aj["cif"]

    af_distribution = _AFDistribution(boot_af)
    seed_entropy = getattr(getattr(rng.bit_generator, "seed_seq", None), "entropy", None)
    provenance: dict[str, Any] = {
        "algorithm": "cox_ph+aalen_johansen" if competing else "cox_ph",
        "ties": "efron",
        "n": n,
        "n_fit_rows": n_fit_rows,
        "n_events": int(cox_event.sum()),
        "exposure_col": exposure_col,
        "exposure_prevalence": exposure_prevalence,
        "population_attributable_fraction": population_attributable_fraction,
        "latency": latency,
        "n_boot": n_boot,
        "n_boot_valid": n_boot_valid,
        "seed": seed_entropy,
        "coef": fit.coef.tolist(),
        "se": fit.se.tolist(),
        "concordance": fit.concordance,
        "loglik": fit.loglik,
        "n_iter": fit.n_iter,
        "competing": competing,
        "af_distribution": af_distribution,
    }
    if aj is not None:
        provenance["cif_time"] = aj["time"]
        provenance["overall_survival"] = aj["overall_survival"]

    return CohortAttribution(
        hazard_ratio=hazard_ratio,
        hr_ci=hr_ci,
        attributable_fraction=attributable_fraction,
        af_ci=af_ci,
        cif=cif,
        provenance=provenance,
    )
