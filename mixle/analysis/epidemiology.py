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
``competing``, one ``aalen_johansen`` run) and one RNG's full reproducible state -- both recorded in
``provenance``. ``provenance["rng_state"]`` is the bit-generator state (not just its construction seed):
a seed alone cannot replay the draws of a generator the caller had already advanced before passing it in,
and is unavailable at all for a bit generator built without a ``SeedSequence``.
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
    trip here. ``samples`` holds only draws that actually converged: failed resamples are dropped by the
    caller before construction (MXR-080-0089), never carried through as ``NaN``, so ``credible_interval``
    below can use a plain quantile instead of silently NaN-skipping values whose count it never checked.
    """

    def __init__(self, samples: np.ndarray):
        samples = np.asarray(samples, dtype=float)
        if samples.size == 0:
            raise ValueError("_AFDistribution.samples must be non-empty.")
        if not np.all(np.isfinite(samples)):
            raise ValueError(
                "_AFDistribution.samples must be finite; failed bootstrap draws must be filtered out "
                "before construction, not carried through as NaN."
            )
        self.samples = samples
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        lo = np.quantile(self.samples, a)
        hi = np.quantile(self.samples, 1.0 - a)
        return lo, hi


_MIN_BOOTSTRAP_DRAWS = 40
"""Absolute floor on valid bootstrap draws before :func:`cohort_attribution` will report an
attributable-fraction interval.

``af_ci`` and :class:`_AFDistribution` report the 2.5th/97.5th percentile of the bootstrap draws (a 95%
interval). With fewer than 40 draws, the 2.5th-percentile order statistic has less than one *expected*
draw beyond it (``40 * 0.025 = 1``): below this floor the tail quantile is extrapolated past the data
rather than interpolated within it, at any draw count -- not a reliability threshold that gets looser for
a smaller ``n_boot`` request.
"""

_MIN_BOOTSTRAP_FRACTION = 0.5
"""Minimum fraction of the *requested* ``n_boot`` draws that must converge, on top of
:data:`_MIN_BOOTSTRAP_DRAWS`.

Catches what the absolute floor alone cannot: e.g. ``n_boot=10_000`` with only 60 converged fits clears
the floor of 40, but a 99.4% per-draw failure rate is evidence about the cohort (near-complete separation
on the exposure column, most likely) -- not sampling noise a bigger draw count would average away.
"""


@dataclass
class _InsufficientBootstrapEvidence:
    """Typed placeholder for ``provenance["af_distribution"]`` when too few bootstrap draws converged to
    support an attributable-fraction interval (below :data:`_MIN_BOOTSTRAP_DRAWS` /
    :data:`_MIN_BOOTSTRAP_FRACTION`; MXR-080-0089) -- returned instead of an :class:`_AFDistribution`
    built from a handful of lucky draws, since a few successful draws carry no real information about
    sampling variability.

    Deliberately does *not* satisfy IC-1's ``DerivedQuantity`` protocol (no ``credible_interval`` method,
    no ``prior_dominated`` flag): a caller that treats ``af_distribution`` as a ``DerivedQuantity``
    without checking fails loudly (``AttributeError``, or a failed ``isinstance`` check) instead of
    silently plotting or quoting an interval that looks precise but rests on almost no evidence.
    """

    reason: str
    n_boot: int
    n_boot_valid: int
    samples: np.ndarray  # the few draws that did converge, kept for diagnostics only -- not an interval


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
        provenance: fit diagnostics, the RNG's full reproducible state (``rng_state``), and
            ``af_distribution`` -- an :class:`_AFDistribution` (IC-1 ``DerivedQuantity``-shaped) when the
            bootstrap produced adequate evidence, or an :class:`_InsufficientBootstrapEvidence` when it
            did not (see :data:`_MIN_BOOTSTRAP_DRAWS`).
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


_NON_COMPETING_EVENT_CODES = (0.0, 1.0)
"""The only valid ``event`` labels when ``competing=False``: ``0`` = censored, ``1`` = the outcome of
interest (see :func:`cohort_attribution`'s ``event`` parameter). ``competing=True`` widens this to any
non-negative integer (``2..K`` are competing causes), checked separately in :func:`_validate_cohort`
since ``K`` is open-ended rather than a fixed set."""


def _validate_cohort(
    x: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    *,
    exposure_col: int,
    latency: float,
    competing: bool,
) -> None:
    """Validate one cohort design -- shapes, finiteness, time, event codes, latency -- in a single pass
    before any Cox fit or bootstrap resample sees the data (MXR-080-0088).

    Every defect caught here used to either crash deep inside ``cox_ph`` with an error that pointed
    nowhere near the actual bad input (a length mismatch surfacing as a boolean-index-shape error three
    calls away), or worse, was silently absorbed: a fractional event code (e.g. ``1.5``) cast straight to
    ``int`` truncates toward whichever cause the fractional part happened to land on -- silently
    relabelling *which outcome a subject is recorded as having* -- and negative latency failed
    ``_fit_lagged``'s ``latency > 0`` check and fell through to "no latency" instead of being rejected.
    Bundling every check here and running it once, before fitting or the bootstrap loop starts, means a
    malformed cohort fails fast with one clear reason instead of corrupting a fit silently or crashing
    somewhere unrelated after some number of the ``n_boot`` resamples have already run.

    Args:
        x: ``(n, p)`` covariates, already reshaped to 2-D.
        t: ``(n,)`` event/censoring times.
        e: ``(n,)`` event codes, still ``float`` (not yet cast to ``int`` -- casting before this check is
            exactly the bug being closed).
        exposure_col: the column of ``x`` this call intends to treat as the exposure.
        latency: the requested left-truncation lag.
        competing: whether multi-cause event labels (``>1``) are in play.

    Raises:
        ValueError: on the first defect found (see the module-level docstring on validation ordering).
    """
    if x.ndim != 2:
        raise ValueError(f"covariates must be 1-D or 2-D, got {x.ndim}-D (shape {x.shape})")
    n = x.shape[0]
    if t.shape != (n,):
        raise ValueError(f"time must have shape ({n},) to match covariates' {n} row(s), got {t.shape}")
    if e.shape != (n,):
        raise ValueError(f"event must have shape ({n},) to match covariates' {n} row(s), got {e.shape}")
    if not 0 <= exposure_col < x.shape[1]:
        raise ValueError(f"exposure_col={exposure_col} is out of bounds for covariates with {x.shape[1]} column(s)")

    # Finiteness must be checked before any numeric comparison below: a NaN fails every ordering
    # comparison (`NaN >= 0` is False, `NaN != trunc(NaN)` is True), so a finiteness check placed after
    # a domain check would let NaN slip through some branches and get a misleading message from others.
    if not np.all(np.isfinite(x)):
        raise ValueError("covariates must be finite (no NaN or Inf).")
    if not np.all(np.isfinite(t)):
        raise ValueError("time must be finite (no NaN or Inf).")
    if not np.all(np.isfinite(e)):
        raise ValueError("event must be finite (no NaN or Inf).")

    if np.any(t < 0):
        raise ValueError("time must be non-negative.")

    if np.any(e != np.trunc(e)):
        raise ValueError(
            "event must be exact integer cause labels (fractional event codes are not supported); "
            "casting a fractional code to int would silently relabel it as a different cause."
        )
    if np.any(e < 0):
        raise ValueError("event codes must be non-negative (0 = censored, 1.. = event / competing causes).")
    if not competing and np.any(~np.isin(e, _NON_COMPETING_EVENT_CODES)):
        bad = sorted({float(v) for v in e[~np.isin(e, _NON_COMPETING_EVENT_CODES)]})
        raise ValueError(
            f"event must be binary {_NON_COMPETING_EVENT_CODES} when competing=False, got code(s) {bad}; "
            "pass competing=True for multi-cause event labels."
        )

    if not np.isfinite(latency):
        raise ValueError("latency must be finite.")
    if latency < 0:
        raise ValueError(
            f"latency must be non-negative, got {latency!r} (negative latency is not a valid exposure "
            "lag; it used to be silently treated as latency=0)."
        )


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
        covariates: ``(n, p)`` covariates (finite); column ``exposure_col`` is the exposure of interest.
        time: ``(n,)`` event/censoring times (finite, non-negative).
        event: ``(n,)`` event indicator. Binary (``0``/``1``) when ``competing=False``; an integer cause
            label (``0`` = censored, ``1`` = the outcome of interest, ``2..K`` = competing causes) when
            ``competing=True``. Either way, the Cox fit is cause-specific for cause ``1``: competing-cause
            events are censored at their time for the hazard-ratio fit (only the CIF, when requested,
            reports the competing causes' own cumulative incidence). Values must be exact integers --
            fractional codes are rejected outright, not truncated (a truncated ``1.5`` would silently
            relabel a subject from "the event of interest" to "censored").
        exposure_col: column of ``covariates`` treated as the exposure (must be a valid column index).
        competing: if True, also run `aalen_johansen` for cause-specific cumulative incidence.
        latency: exposure-to-effect lag (left-truncation, via `cox_ph`'s `start`); ``0`` disables it.
            Must be non-negative -- negative latency is rejected, not silently treated as ``0``.
        n_boot: cohort-resample bootstrap draws for the attributable-fraction interval. If fewer than
            :data:`_MIN_BOOTSTRAP_DRAWS` converge (or too small a fraction of ``n_boot``, see
            :data:`_MIN_BOOTSTRAP_FRACTION`), ``af_ci`` is ``(nan, nan)`` and
            ``provenance["af_distribution"]`` is an :class:`_InsufficientBootstrapEvidence` rather than a
            real interval.
        rng: seed, `numpy.random.Generator`, or None.

    Returns:
        A :class:`CohortAttribution`.

    Raises:
        ValueError: if the cohort is malformed -- see :func:`_validate_cohort` for the exact checks
            (mismatched array shapes, non-finite values, negative time, an out-of-vocabulary or
            fractional event code, negative latency, or an out-of-bounds ``exposure_col``).
    """
    cov_arr = np.asarray(covariates, dtype=float)
    x = cov_arr.reshape(-1, 1) if cov_arr.ndim == 1 else cov_arr
    t = np.asarray(time, dtype=float)
    e_float = np.asarray(event, dtype=float)
    _validate_cohort(x, t, e_float, exposure_col=exposure_col, latency=latency, competing=competing)
    e_raw = e_float.astype(np.int64)
    n = x.shape[0]
    rng = np.random.default_rng(rng)
    # Captured *before* any draw below: replaying this bit-generator state reproduces the exact bootstrap
    # resamples this call makes, regardless of whether `rng` arrived as an int seed, None, or an
    # already-advanced `Generator` the caller had used elsewhere first (MXR-080-0089). This replaces
    # `bit_generator.seed_seq.entropy`, which only ever reflects the generator's ORIGINAL construction
    # seed: `None` for a bit generator built without a `SeedSequence` (e.g. `Philox(key=...)`), and
    # silently the WRONG sequence to replay for a generator that had already been advanced before arriving
    # here -- entropy alone cannot tell the difference between "fresh" and "already consumed."
    rng_state = rng.bit_generator.state

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

    boot_af_valid: list[float] = []
    for _b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            fit_b, _ = _fit_lagged(x[idx], t[idx], cox_event[idx], latency)
            hr_b = float(np.exp(fit_b.coef[exposure_col]))
            if hr_b > 0:
                af_b = (hr_b - 1.0) / hr_b
                if np.isfinite(af_b):
                    boot_af_valid.append(af_b)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue  # a degenerate resample (e.g. no variation in the exposure column); skip it

    # Only draws that actually converged reach `boot_af` -- a failed draw is dropped outright, never
    # carried through as a NaN placeholder (MXR-080-0089): `_AFDistribution.samples` (below) must never
    # be NaN-contaminated, and `n_boot_valid` must count exactly what it says.
    boot_af = np.array(boot_af_valid, dtype=float)
    n_boot_valid = boot_af.shape[0]
    bootstrap_min_required = max(_MIN_BOOTSTRAP_DRAWS, int(np.ceil(_MIN_BOOTSTRAP_FRACTION * n_boot)))
    af_distribution: _AFDistribution | _InsufficientBootstrapEvidence
    if n_boot_valid >= bootstrap_min_required:
        af_ci = (
            float(np.quantile(boot_af, 0.025)),
            float(np.quantile(boot_af, 0.975)),
        )
        af_distribution = _AFDistribution(boot_af)
    else:
        # One (or a handful of) successful draws carries no information about sampling variability --
        # report this as a typed non-interval instead of a spuriously precise `af_ci` (MXR-080-0089).
        af_ci = (float("nan"), float("nan"))
        af_distribution = _InsufficientBootstrapEvidence(
            reason=(
                f"only {n_boot_valid} of {n_boot} bootstrap draws converged, below the "
                f"{bootstrap_min_required} required (max of {_MIN_BOOTSTRAP_DRAWS} absolute or "
                f"{_MIN_BOOTSTRAP_FRACTION:.0%} of n_boot) for a reliable attributable-fraction interval"
            ),
            n_boot=n_boot,
            n_boot_valid=n_boot_valid,
            samples=boot_af,
        )

    cif: dict[int, np.ndarray] = {}
    aj: dict[str, Any] | None = None
    if competing:
        aj = aalen_johansen(t, e_raw)
        cif = aj["cif"]

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
        "rng_state": rng_state,
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
