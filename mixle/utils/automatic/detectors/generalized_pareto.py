"""Generalized Pareto (peaks-over-threshold) continuous candidate -- heavy-tailed exceedances.

By the Pickands-Balkema-de Haan theorem the distribution of exceedances over a high threshold
converges to a generalized Pareto distribution (GPD). Its signature is a strictly-positive,
monotone-decreasing density with a *heavy* (Pareto) upper tail -- shape ``xi`` clearly above 0.

The ``xi -> 0`` limit of the GPD is the exponential, and the gamma family already covers that
(and the exponential itself); to avoid stealing exponential / gamma data the gate fires only when
a moment estimate of the tail index ``xi`` is unmistakably positive (a genuinely heavy tail). With
that gate, exponential and Gaussian data never reach this candidate.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# Minimum moment-estimated tail index for the GPD to even be considered. The exponential limit is
# xi = 0; requiring a clearly-positive xi keeps the candidate off exponential / gamma / Gaussian data.
_MIN_XI = 0.12


def _moment_fit(arr: np.ndarray):
    """Return ``(loc, scale, shape)`` from a fixed-threshold method-of-moments GPD gate, or None.

    The threshold ``loc`` is the data minimum (the peaks-over-threshold setup); ``scale`` and
    ``shape`` follow from the exceedance mean ``m`` and variance ``v`` in closed form
    (``xi = (1 - m^2/v)/2``, ``sigma = m (1 - xi)``), valid for ``xi < 1/2``.
    """
    loc = float(np.min(arr))
    y = arr - loc
    m = float(np.mean(y))
    v = float(np.var(y))
    if not (m > 0.0) or not (v > 0.0) or not math.isfinite(v):
        return None
    xi = 0.5 * (1.0 - (m * m) / v)
    scale = m * (1.0 - xi)
    if not (scale > 0.0) or not math.isfinite(scale):
        return None
    return loc, scale, xi


def _fit(arr: np.ndarray):
    """Return the exact MLE used by scoring, deployment, and diagnostics."""
    from scipy import stats

    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    try:
        loc = float(np.min(arr))
        shape, fitted_loc, scale = stats.genpareto.fit(arr, floc=loc)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(shape) or not math.isfinite(fitted_loc) or not math.isfinite(scale) or not scale > 0.0:
        return None
    return float(fitted_loc), float(scale), float(shape)


def _mle_matches_moments(moment_fit, mle_fit) -> bool:
    """Whether the scipy MLE fit is broadly consistent with the cheap moment estimate.

    ``_applies()``'s gate is based on ``_moment_fit``, a closed-form estimate that is NOT subject to
    the same magnitude-driven precision collapse the scipy MLE call in ``_fit()`` can hit: for
    peaks-over-threshold data at nanosecond-epoch magnitude, ``loc + exceedance`` computed in the
    CALLER's own float64 arithmetic -- before this detector ever runs -- already rounds most or all
    of the exceedance away (campaign nine, D-0209), so the moment estimate can report a plausible
    ``xi`` while the MLE, working from the same already-quantized raw array, lands on a nonsensical
    corner (scale near zero, shape far outside any plausible range). Neither estimator can recover
    information the caller's own arithmetic already destroyed, so this does not try to fix the fit
    itself -- it refuses to ADMIT a candidate whose two independent estimates disagree this badly,
    the same way ``_applies()`` already refuses a candidate whose moment estimate alone looks wrong.
    A genuine MLE refinement of a reasonable moment estimate stays within a few orders of magnitude
    on scale and a few units on shape; the reproduced collapse is 20+ orders of magnitude off on
    scale and 6+ units off on shape, nowhere close to this band.
    """
    if moment_fit is None or mle_fit is None:
        return False
    _, moment_scale, moment_xi = moment_fit
    _, mle_scale, mle_xi = mle_fit
    if not (math.isfinite(mle_scale) and math.isfinite(mle_xi)) or mle_scale <= 0.0:
        return False
    if mle_scale < moment_scale * 1e-6 or mle_scale > moment_scale * 1e6:
        return False
    return abs(mle_xi - moment_xi) <= 2.0


def _typical_adjacent_gap(arr: np.ndarray) -> float:
    """Median NONZERO gap between adjacent sorted values of ``arr``, or 0.0 if none exist.

    A direct, empirical measurement of how finely ``arr`` actually resolves distinct values --
    used by :func:`_magnitude_precision_is_suspect` in place of a statistical proxy (see that
    function's docstring for why a proxy like the moment-estimated scale is unsound here).
    """
    if arr.size < 2:
        return 0.0
    gaps = np.diff(np.sort(arr))
    gaps = gaps[gaps > 0.0]
    return float(np.median(gaps)) if gaps.size else 0.0


def _magnitude_precision_is_suspect(loc: float, typical_gap: float) -> bool:
    """Whether ``loc``'s own float64 grid is coarse enough, relative to the DATA'S OWN typical
    adjacent gap, that the scipy MLE call in ``_fit()`` plausibly hit the magnitude-driven collapse
    ``_mle_matches_moments`` exists to catch (campaign nine, D-0209).

    Deliberately compares against ``typical_gap`` -- the array's own measured spacing between
    adjacent distinct values (:func:`_typical_adjacent_gap`) -- rather than the moment-estimated
    scale an earlier version of this gate used. That earlier version compared ``half_ulp(loc)``
    against ``moment_scale``, but ``moment_scale`` can be small at ANY magnitude for a near-Dirac /
    heavily-tied sample (an ordinary, unrelated failure mode downstream's
    ``_degenerate_likelihood_spike`` in ``mixle/lifecycle.py`` already handles correctly) -- and
    since ``half_ulp(loc)`` grows with ``loc`` regardless of whether anything is actually being
    destroyed, that ratio eventually crosses ANY fixed threshold purely from ``loc`` growing, giving
    a false positive at ordinary real-world magnitudes (millisecond/microsecond-epoch timestamps,
    large IDs) the gate's own pinned test had not swept a range of offsets for. ``typical_gap``
    fixes this: an offset that has destroyed nothing leaves it unchanged (an integer-valued
    near-Dirac sample stays exactly spaced by >= 1 up to loc ~1e15, measured directly, regardless of
    how large ``loc`` is), while a genuinely magnitude-collapsed sample (peaks-over-threshold
    exceedances at nanosecond-epoch scale) shows its adjacent gaps compressed down toward
    ``half_ulp(loc)`` itself, since that is what the CALLER's own float64 addition can no longer
    distinguish. Measured: the near-Dirac fixture's ratio stays below 0.0625 through loc=1e15 (only
    reaching 1.0 -- total collapse, correctly caught -- at loc=1e18, an offset that magnitude alone
    would ALSO have flagged for genuinely continuous data); the reproduced GPD collapse sits at 0.5
    regardless of the exceedance scale tested (50, 300, 3000). The threshold below sits at the
    geometric midpoint of that gap in log-space, comfortable margin either way.
    """
    if loc == 0.0:
        # No offset at all: nothing for the caller's own arithmetic to have destroyed.
        return False
    if typical_gap <= 0.0:
        # Every value in the sample is (near-enough) identical despite a nonzero offset -- cannot
        # rule out magnitude as the cause, so err toward running the cross-check.
        return True
    return (0.5 * float(np.spacing(abs(loc)))) > 0.2 * typical_gap


def _boundary_point_mass_is_suspect(arr: np.ndarray, loc: float) -> bool:
    """Whether two or more observations sit at (or within a hair of) ``arr``'s own minimum ``loc``.

    A SECOND, independent cause of MLE degeneracy alongside magnitude (campaign nine, D-0209,
    round-2 review): the GPD density at ``x = loc`` is ``1/scale`` for ANY shape, so multiple
    observations tied at the fitted threshold can drive ``scale -> 0`` and the likelihood to
    ``+inf`` regardless of how well the sample's own magnitude is represented -- reproduces at
    ``loc`` offsets from 0 through at least 1e16. This is invisible to
    :func:`_magnitude_precision_is_suspect`: :func:`_typical_adjacent_gap` deliberately filters out
    exact ties (zero gaps) to measure the SPARSE tail's own resolution, so a large point mass at the
    threshold leaves it looking perfectly healthy. Checked separately here and ORed into the same
    cross-check trigger in :func:`_applies`, rather than folded into the magnitude gate, because the
    two causes are unrelated and conflating them risks re-breaking either one while fixing the other
    -- exactly what happened switching from the round-1 to the round-2 magnitude gate.
    """
    if arr.size < 2:
        return False
    tolerance = 8.0 * float(np.spacing(abs(loc))) if loc != 0.0 else 0.0
    return int(np.count_nonzero(arr - loc <= tolerance)) >= 2


def _applies(arr: np.ndarray) -> bool:
    # Positive support (threshold exceedances / tails); exclude non-positive data outright.
    if arr.size < 16 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return False
    moment_fit = _moment_fit(arr)
    if moment_fit is None:
        return False
    loc, _, xi = moment_fit
    # Only fire on an unmistakably heavy (Pareto) tail. Exponential / gamma / Gaussian samples
    # produce xi near 0 (or negative) and are screened out here, so the candidate cannot steal them.
    if xi < _MIN_XI:
        return False
    suspect = _magnitude_precision_is_suspect(loc, _typical_adjacent_gap(arr)) or _boundary_point_mass_is_suspect(
        arr, loc
    )
    if not suspect:
        return True
    # Cross-check the real fit before admitting this candidate: a downstream caller can query this
    # family directly or restrict the candidate pool, bypassing the frontier competition that would
    # otherwise usually let a better-scoring family replace a degenerate one.
    return _mle_matches_moments(moment_fit, _fit(arr))


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit(arr)
    if fit is None:
        return None
    loc, scale, shape = fit
    nll_nats = -float(np.mean(stats.genpareto.logpdf(arr, shape, loc=loc, scale=scale)))
    if not math.isfinite(nll_nats):
        return None
    # loc is pinned to a statistic selected from this same sample, so it is
    # charged alongside scale and shape rather than treated as external.
    return nll_nats / math.log(2.0) + _bic_penalty_bits(3, nobs)


class _FittedEstimator:
    """Estimator protocol that preserves the exact candidate selected by MLE."""

    def __init__(self, distribution):
        self.distribution = distribution
        self._delegate = distribution.estimator()

    def accumulator_factory(self):
        return self._delegate.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        return self.distribution

    def get_prior(self):
        return None

    def model_log_density(self, model):
        return 0.0


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import GeneralizedParetoDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    if fit is not None:
        loc, scale, xi = fit
    else:
        keys = [float(k) for k in vdict.keys() if isinstance(k, (int, float, np.integer, np.floating))]
        loc, scale, xi = (min(keys) if keys else 0.0), 1.0, 0.1
    distribution = GeneralizedParetoDistribution(scale=scale, shape=xi, loc=loc)
    return _FittedEstimator(distribution)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit(arr)
    if fit is None:
        return None
    loc, scale, shape = fit
    return stats.genpareto.cdf(arr, shape, loc=loc, scale=scale)


register(
    Detector(
        name="generalized_pareto",
        kind="continuous",
        applies=_applies,
        score=_score,
        factory=_factory,
        cdf=_cdf,
        n_params=3,
    )
)
