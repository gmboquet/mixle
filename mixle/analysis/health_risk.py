"""Dose-response and population health-risk models (K3, work-plan Workstream K).

Given an exposure/dose -- a bare number, an array of dose realisations, or an IC-1 `Posterior` over
a receptor field (K1/K2 transport output) -- these push it through a named dose-response curve into
an outcome-probability distribution, so downstream liability/constraint code (K6) always has a
distribution, never a bare point estimate:

  * :class:`DoseResponse` -- a named model (``loglinear`` / ``logit`` / ``hill`` /
    ``threshold_linear``) with caller-supplied ``params``; :meth:`DoseResponse.probability` maps a
    dose into an IC-1-shaped `DerivedQuantity` (samples + credible interval + the `prior_dominated`
    honesty flag). When ``dose`` is a `Posterior`, the pushforward runs through the posterior's own
    ``derived_quantity`` so the flag propagates correctly from the exposure uncertainty.
  * :func:`cumulative_exposure` -- trapezoidal time-integration of an exposure series, with an
    optional first-order biological-decay discount (older exposure counts less toward the current
    body burden) -- the intake feeding a chronic dose-response evaluation.
  * :func:`population_risk` -- aggregates per-receptor dose-response probabilities (one draw per
    posterior sample, or a single point evaluation for a bare array) into an expected-case-count
    `DerivedQuantity`.

This module supplies the dose-response *machinery*; it ships no regulatory/clinical dose-response
table -- ``params`` are always supplied by the caller or a knowledge lookup (see Non-goals).

K6 (this file's second addition, appended after K3) turns a K3/K4 risk distribution into the two
things J's objective/optimizer actually need -- a priced cost term and a hard feasibility screen:

  * :func:`health_liability` -- prices a case-count/exceedance-probability `DerivedQuantity` (e.g.
    `population_risk`'s output, or K4's `safety_risk_surface`) into a dollar-liability
    `DerivedQuantity` at ``cost_per_case`` per case, discounted -- the ``health_cost`` term J6's
    ``priced_liabilities``/``risk_adjusted_plan`` (``analysis/valuation.py``) sums alongside carbon
    (L6) and remediation (G-side) liabilities.
  * :func:`exposure_constraints` -- screens a list of candidate operating options against named
    occupational/community exposure ``limits``, marking each ``feasible`` (or not, naming the
    ``binding`` limit(s)) so H4's ``two_stage_stochastic_plan`` (``stochastic_opt.py``) only ever
    optimizes over the surviving, feasible candidate set.

K4 (this file's third addition) turns a G4 ground-deformation posterior into a spatial safety-risk
surface and a people-weighted incident probability:

  * :func:`safety_risk_surface` -- the spatial gradient (tilt) of a deformation field, pushed through
    the field's own IC-1 `Posterior.derived_quantity` pushforward so `P(tilt > gradient_limit)` per
    cell comes back as a `DerivedQuantity` (samples + credible interval + the `prior_dominated`
    honesty flag), not a point estimate. An optional static terrain `slope` adds to the
    deformation-induced tilt before the exceedance test -- a location that is already steep needs
    less differential settlement to become unsafe.
  * :func:`incident_probability` -- combines a per-cell hazard probability surface with a people-
    `exposure_map` (who is where) through a logistic link into a per-cell probability that a hazard
    cell becomes an actual incident (nobody is at risk on a steep-but-empty cell).

`safety_risk_surface` accepts either a raw `np.ndarray` deformation field (a deterministic,
already-inverted grid -- treated as a single degenerate draw with `prior_dominated=False`) or an IC-1
`Posterior` over deformation (the intended G4 case). The frozen `Posterior.derived_quantity` method
does the sampling and the honesty-flag bookkeeping; this module only supplies the gradient/exceedance
pushforward function, so the uncertainty accounting is always the posterior implementation's own (A2),
never re-derived here.

Non-goals: no deformation physics (`mixle_pde.poroelastic` owns the InSAR inversion itself), no
economic liability (`health_liability`, K6).

K5 (this file's fourth addition) turns a scalar monitoring series into a calibrated real-time
exceedance alert -- the sibling of :mod:`mixle.analysis.coverage` / :mod:`mixle.analysis.extreme` for
the health & safety pillar. It answers a monitoring-shift question -- "is this reading trending
toward, or past, a regulatory/occupational exposure limit, and can I trust the alert?" -- without
pretending a single noisy reading settles it:

  * :func:`exposure_exceedance_monitor` -- per-timestep ``P(exposure > limit)`` from a local predictive
    fit around each reading (the same *distribution-over-a-threshold* idea as IC-8's
    ``mixle_pde.decision_quantities.prob_exceed``, here applied to a scalar monitoring series rather
    than a spatial posterior field, since a live sensor stream is not itself an IC-1 ``Posterior``).
    The raw probability is then run through :func:`mixle.inference.conformal.split_conformal` against a
    held-out ``calib`` reference (known-safe, sub-limit history) so the alert threshold is
    distribution-free calibrated: under exchangeability with ``calib``, the empirical false-alarm rate
    is bounded by ``alpha``, not just "probably fine" from an untested normal-theory cutoff.

An alert firing (``ExceedanceReport.alerts.any()``) is the hook the mlops drift/retrain half (G7,
``mixle_mlops.drift_retrain``) watches for a re-check/retrain trigger -- that wiring is a *signal*
(read this array), not a code dependency: this module never imports anything from ``mixle_mlops``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from mixle.inference.conformal import split_conformal

if TYPE_CHECKING:
    from mixle.reason.posterior_protocol import DerivedQuantity, Posterior

# `Posterior`/`DerivedQuantity` are imported lazily inside the functions that need them at runtime
# (rather than at module level) so that merely importing `mixle.analysis` does not force-load
# `mixle.reason`'s package `__init__` -- `mixle.analysis.extreme` sits on `mixle.inference.risk`'s
# import path, which `mixle.stats.bayes.dirichlet` pulls in while it is itself mid-initialization;
# a module-level import here would close that into a real circular-import failure of `mixle.stats`.

DOSE_RESPONSE_MODELS = ("loglinear", "logit", "hill", "threshold_linear")

# `safety_risk_surface`'s signature is frozen by the K4 work order with no `n`/`rng` parameters, so the
# Monte-Carlo sample count and seed used to push an IC-1 posterior through the gradient/exceedance
# functional live here instead of on the call site. Fixed seed => repeated calls on the same posterior
# reproduce the same surface.
_MC_SAMPLES = 2000
_MC_SEED = 0


@dataclass
class _SampleDerivedQuantity:
    """A concrete IC-1 `DerivedQuantity`: a draw matrix + the honesty flag, CI by empirical quantile.

    The same "samples + quantile-based `credible_interval` + `prior_dominated`" shape used by the
    frozen IC-1 conformance stub and the H4 stochastic-plan tests -- the repo's established idiom for
    a concrete derived quantity, rather than a bespoke one per caller.
    """

    samples: np.ndarray
    prior_dominated: bool = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


@dataclass
class _DeterministicRisk:
    """A degenerate `DerivedQuantity` for a plain-`ndarray` (no-UQ) deformation input.

    Satisfies the IC-1 `DerivedQuantity` protocol with a single replicate: there is no posterior to
    sample from, so the exceedance is either 0 or 1 per cell and the credible interval collapses to a
    point. `prior_dominated` is always False -- there is no prior/regulariser in play, only a direct
    threshold test on the supplied field. `grid_shape` is extra (not part of IC-1) so a caller that
    knows it received an ndarray can reshape `samples` back into the original spatial layout.
    """

    samples: np.ndarray  # (1, n_cells), 0.0/1.0 exceedance indicator
    grid_shape: tuple[int, ...]
    prior_dominated: bool = field(default=False)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        point = self.samples[0]
        return point, point


def _dose_response_fn(model: str, params: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Return the elementwise dose -> outcome-probability map for the named ``model``."""
    if model == "loglinear":
        beta = float(params["beta"])
        return lambda d: 1.0 - np.exp(-beta * np.clip(np.asarray(d, dtype=float), 0.0, None))
    if model == "logit":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        return lambda d: 1.0 / (1.0 + np.exp(-(a * np.asarray(d, dtype=float) + b)))
    if model == "hill":
        emax = float(params.get("emax", 1.0))
        ec50 = float(params["ec50"])
        hill_n = float(params.get("n", 1.0))

        def _hill(d: np.ndarray) -> np.ndarray:
            x = np.clip(np.asarray(d, dtype=float), 0.0, None)
            xn = x**hill_n
            return emax * xn / (ec50**hill_n + xn)

        return _hill
    if model == "threshold_linear":
        slope = float(params["slope"])
        threshold = float(params.get("threshold", 0.0))
        return lambda d: np.clip(slope * (np.asarray(d, dtype=float) - threshold), 0.0, 1.0)
    raise ValueError(f"unknown dose-response model {model!r}; expected one of {DOSE_RESPONSE_MODELS}")


def _as_dose_samples(dose: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Coerce a bare-array/scalar ``dose`` into ``n`` dose draws (Posterior doses take a separate path).

    A scalar (or length-1 array) is a degenerate point mass, replicated ``n`` times. A length-``n``
    array is treated as an already-drawn ensemble. Any other length is an "array-with-UQ" sample set
    of a different size, resampled with replacement to ``n`` draws.
    """
    arr = np.atleast_1d(np.asarray(dose, dtype=float))
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.shape[0] == n:
        return arr
    idx = rng.integers(0, arr.shape[0], size=n)
    return arr[idx]


@dataclass
class DoseResponse:
    """A named dose-response model: ``model`` selects the functional form, ``params`` its coefficients.

    ``model in {"loglinear", "logit", "hill", "threshold_linear"}``:

      * ``loglinear``: ``P = 1 - exp(-beta * dose)`` (``params: {"beta"}``) -- the EPA-style linear
        low-dose cancer/chronic form.
      * ``logit``: ``P = sigmoid(a * dose + b)`` (``params: {"a", "b"}``, both optional).
      * ``hill``: ``P = emax * dose^n / (ec50^n + dose^n)`` (``params: {"ec50"}``, ``"emax"``/``"n"``
        optional) -- saturating receptor-occupancy form.
      * ``threshold_linear``: ``P = clip(slope * (dose - threshold), 0, 1)`` (``params: {"slope"}``,
        ``"threshold"`` optional) -- no response below ``threshold``.

    No regulatory dose-response table ships here -- ``params`` are supplied by the caller (see the
    module Non-goals).
    """

    model: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in DOSE_RESPONSE_MODELS:
            raise ValueError(f"unknown dose-response model {self.model!r}; expected one of {DOSE_RESPONSE_MODELS}")

    def response_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """The elementwise dose -> outcome-probability function for this model + params."""
        return _dose_response_fn(self.model, self.params)

    def probability(self, dose: Any, *, n: int = 2000, rng: np.random.Generator) -> DerivedQuantity:
        """Push ``dose`` through the dose-response model into an outcome-probability `DerivedQuantity`.

        ``dose`` may be an IC-1 `Posterior` over exposure (the pushforward runs through the
        posterior's own ``derived_quantity``, so `prior_dominated` propagates from the exposure
        uncertainty), an array of dose draws/ensemble members ("array-with-UQ", resampled to ``n``
        if its length differs), or a bare scalar (a degenerate point mass -- the returned quantity
        still carries a (trivial) credible interval).
        """
        from mixle.reason.posterior_protocol import Posterior

        fn = self.response_fn()
        if isinstance(dose, Posterior):
            return dose.derived_quantity(fn, n, rng)
        draws = _as_dose_samples(dose, n, rng)
        return _SampleDerivedQuantity(samples=fn(draws), prior_dominated=False)


def cumulative_exposure(series: np.ndarray, dt: float, *, decay: float = 0.0) -> float:
    """Time-integrated exposure (trapezoidal rule), with optional first-order biological decay.

    ``decay=0`` is the plain trapezoidal integral of ``series`` over its ``dt``-spaced timesteps
    (area under the exposure-rate curve). ``decay > 0`` discounts each sample toward the *final*
    timestep by ``exp(-decay * (t_end - t))`` before integrating -- the way a biological half-life
    would -- so a spike long ago contributes less to the current cumulative body burden than an
    equally large spike near the end of the series. Feeds a chronic dose-response evaluation (e.g.
    via :meth:`DoseResponse.probability`).
    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return float(x[0] * dt)
    if decay <= 0.0:
        return float(np.trapz(x, dx=dt))
    times = np.arange(x.size) * dt
    decayed = x * np.exp(-decay * (times[-1] - times))
    return float(np.trapz(decayed, dx=dt))


def population_risk(
    exposure: Posterior | np.ndarray, dr: DoseResponse, *, n: int, rng: np.random.Generator
) -> DerivedQuantity:
    """Aggregate per-receptor dose-response probabilities into an expected-case-count `DerivedQuantity`.

    ``exposure`` is a per-receptor dose: an IC-1 `Posterior` whose draws are ``(n, n_receptors)`` dose
    vectors (K1/K2 transport output propagated through UQ), or a plain ``(n_receptors,)`` array of
    point doses. Each posterior draw is pushed through ``dr``'s response function and summed over
    receptors, so the returned quantity carries the expected-case-count distribution (not just its
    mean); a bare array has no exposure uncertainty and yields a degenerate (constant) distribution.
    """
    from mixle.reason.posterior_protocol import Posterior

    fn = dr.response_fn()
    if isinstance(exposure, Posterior):
        return exposure.derived_quantity(lambda draws: fn(draws).sum(axis=-1), n, rng)
    arr = np.atleast_1d(np.asarray(exposure, dtype=float))
    expected_cases = float(np.sum(fn(arr)))
    return _SampleDerivedQuantity(samples=np.full(int(n), expected_cases), prior_dominated=False)


def health_liability(risk: DerivedQuantity, *, cost_per_case: float, discount: float = 0.0) -> DerivedQuantity:
    """Price a K3/K4 risk `DerivedQuantity` into an expected-liability `DerivedQuantity` (K6, work-plan §7-K).

    ``risk`` is any IC-1-shaped `DerivedQuantity` over an expected case count or exceedance
    probability (:func:`population_risk`'s output, or K4's ``safety_risk_surface``); every draw in
    ``risk.samples`` is multiplied by ``cost_per_case`` (dollars per case/incident) and divided by
    ``(1 + discount)`` -- a single-period present-value factor (``discount=0`` is undiscounted;
    multi-period accounting applies its own per-period factor before summing across periods, this
    function prices one period/one risk term at a time). Pricing does not change *how* uncertain the
    underlying risk is: the returned quantity keeps ``risk``'s sample count/shape and its
    `prior_dominated` flag unchanged, so a liability that is prior-dominated upstream is still
    honestly flagged as such downstream. Handed to J6's ``priced_liabilities``/``risk_adjusted_plan``
    (``analysis/valuation.py``) as the ``health_cost`` callable's output.
    """
    samples = np.asarray(risk.samples, dtype=float)
    factor = float(cost_per_case) / (1.0 + float(discount))
    prior_dominated = bool(getattr(risk, "prior_dominated", False))
    return _SampleDerivedQuantity(samples=samples * factor, prior_dominated=prior_dominated)


def exposure_constraints(options: list[dict], limits: dict[str, float]) -> list[dict]:
    """Screen candidate operating ``options`` against named exposure/exceedance ``limits`` (K6).

    ``limits`` maps an occupational/community exposure metric name (e.g. ``"silica_pm4"``, an
    8-hour TWA, or an exceedance probability) to the regulatory/policy limit for that metric.
    ``options`` is a list of plain dicts -- each one candidate operating configuration, carrying
    (among whatever other plan data the caller needs, e.g. block cost or grade) a value for zero or
    more of those metric keys.

    Returns a *new* list (the input dicts are never mutated), one entry per option, each the
    original option's key/value pairs plus:

    - ``"feasible"``: ``True`` iff the option breaches none of its limited metrics.
    - ``"binding"``: the sorted list of limit names actually breached (empty when feasible) --
      naming exactly which limit made the option infeasible.

    An option with no entry for a given limit key is not evaluated against that key (the metric was
    simply not modeled for that option), not treated as a violation. A caller filters the returned
    list down to the feasible options *before* handing the survivors' blocks to H4's
    ``two_stage_stochastic_plan`` (``stochastic_opt.py``) -- an infeasible option is dropped from the
    candidate set entirely, so the optimizer never has the chance to select it (see the K6 DoD).
    """
    annotated: list[dict] = []
    for option in options:
        binding = sorted(name for name, limit in limits.items() if name in option and option[name] > limit)
        out = dict(option)
        out["feasible"] = len(binding) == 0
        out["binding"] = binding
        annotated.append(out)
    return annotated


def _grid_shape_for(posterior: Posterior, slope: np.ndarray | None) -> tuple[int, ...]:
    """Infer the spatial grid shape backing a flat `(d,)` posterior mean.

    Prefers `slope`'s shape (the caller already knows the grid geometry whenever it supplies terrain
    slope), then an optional `grid_shape` attribute some posteriors may carry (additive -- not part of
    IC-1, but structural typing does not forbid extra attributes), then falls back to a square grid if
    `d` is a perfect square, else treats the field as a 1-D transect.
    """
    d = int(np.asarray(posterior.mean).shape[0])
    if slope is not None:
        shape = tuple(np.asarray(slope).shape)
        if int(np.prod(shape)) != d:
            raise ValueError(f"slope shape {shape} does not match deformation dimension {d}")
        return shape
    grid_shape = getattr(posterior, "grid_shape", None)
    if grid_shape is not None:
        shape = tuple(grid_shape)
        if int(np.prod(shape)) != d:
            raise ValueError(f"posterior.grid_shape {shape} does not match deformation dimension {d}")
        return shape
    side = int(round(np.sqrt(d)))
    if side * side == d:
        return (side, side)
    return (d,)


def _gradient_magnitude(grid: np.ndarray) -> np.ndarray:
    """Per-sample spatial-gradient magnitude of a `(n, *spatial_shape)` batch, one value per cell.

    `spatial_shape` may be 1-D (a transect) or 2-D+ (a true surface); the magnitude is the Euclidean
    norm of the per-axis finite-difference gradient (`np.gradient`), computed over the spatial axes
    only -- never across the leading Monte-Carlo/sample axis.
    """
    spatial_axes = tuple(range(1, grid.ndim))
    if not spatial_axes:
        return np.zeros_like(grid)
    grads = np.gradient(grid, axis=spatial_axes)
    if len(spatial_axes) == 1:
        grads = (grads,)
    return np.sqrt(sum(g**2 for g in grads))


def safety_risk_surface(
    deformation: Posterior | np.ndarray,
    *,
    gradient_limit: float,
    slope: np.ndarray | None = None,
) -> DerivedQuantity:
    """Map a deformation field into a per-cell `P(tilt > gradient_limit)` safety-risk surface.

    Args:
        deformation: an IC-1 `Posterior` over a flattened `(d,)` subsidence/deformation field (the
            G4 `poroelastic` InSAR-inversion case), or a plain `np.ndarray` spatial grid (any shape) of
            already-point-estimated deformation values.
        gradient_limit: the tilt/gradient magnitude above which a cell is considered geotechnically
            unsafe (e.g. an angular-distortion or differential-settlement limit).
        slope: an optional static terrain-slope field, same spatial shape as the deformation grid.
            When given, it is added to the deformation-induced gradient magnitude before the
            exceedance test: a cell that is already steep needs less additional differential movement
            to cross `gradient_limit`.

    Returns:
        A `DerivedQuantity` whose `samples` are the per-cell exceedance indicator (0.0/1.0) drawn over
        the posterior's Monte-Carlo replicates (or a single deterministic replicate for an `ndarray`
        input), flattened in the grid's row-major (C) order, together with a credible interval and the
        `prior_dominated` flag. The per-cell risk probability is `samples.mean(axis=0)`.
    """
    from mixle.reason.posterior_protocol import Posterior

    if isinstance(deformation, np.ndarray):
        grid = np.asarray(deformation, dtype=float)
        grid_shape = grid.shape
        tilt = _gradient_magnitude(grid[np.newaxis, ...])[0]
        if slope is not None:
            slope_arr = np.asarray(slope, dtype=float)
            if slope_arr.shape != grid_shape:
                raise ValueError(f"slope shape {slope_arr.shape} does not match deformation shape {grid_shape}")
            tilt = tilt + slope_arr
        exceed = (tilt > gradient_limit).astype(float).reshape(1, -1)
        return _DeterministicRisk(samples=exceed, grid_shape=grid_shape)

    if not isinstance(deformation, Posterior):
        raise TypeError("deformation must be an IC-1 Posterior or an np.ndarray field")

    grid_shape = _grid_shape_for(deformation, slope)
    slope_arr = None if slope is None else np.asarray(slope, dtype=float).reshape(grid_shape)

    def _pushforward(draws: np.ndarray) -> np.ndarray:
        n = draws.shape[0]
        grid = draws.reshape((n, *grid_shape))
        tilt = _gradient_magnitude(grid)
        if slope_arr is not None:
            tilt = tilt + slope_arr[np.newaxis, ...]
        exceed = (tilt > gradient_limit).astype(float)
        return exceed.reshape(n, -1)

    rng = np.random.default_rng(_MC_SEED)
    return deformation.derived_quantity(_pushforward, _MC_SAMPLES, rng)


def incident_probability(
    hazard: np.ndarray,
    exposure_map: np.ndarray,
    *,
    model: str = "logit",
) -> np.ndarray:
    """Combine a per-cell hazard probability surface with a people-`exposure_map` into incident risk.

    A hazard exceedance is only a safety *incident* if someone is exposed to it: an empty, unstable
    cell and a busy, unstable cell are not the same risk. `hazard` is expected in `[0, 1]` (e.g. the
    per-cell mean of `safety_risk_surface(...).samples`); `exposure_map` is a non-negative people-
    density/occupancy weight of the same shape.

    Args:
        hazard: per-cell hazard probability, shape `(*grid_shape,)`, values in `[0, 1]`.
        exposure_map: per-cell non-negative occupancy/exposure weight, same shape as `hazard`.
        model: `"logit"` (default) combines the hazard's own log-odds with `log1p(exposure_map)` so an
            unoccupied cell (`exposure_map == 0`) leaves the hazard probability unchanged and denser
            occupancy monotonically raises it; `"linear"` is the simple product `hazard * exposure_map`
            clipped to `[0, 1]`.

    Returns:
        Per-cell incident probability, same shape as `hazard`, values in `[0, 1]`.
    """
    hazard_arr = np.asarray(hazard, dtype=float)
    exposure_arr = np.asarray(exposure_map, dtype=float)
    if hazard_arr.shape != exposure_arr.shape:
        raise ValueError(f"hazard shape {hazard_arr.shape} does not match exposure_map shape {exposure_arr.shape}")
    if np.any(exposure_arr < 0):
        raise ValueError("exposure_map must be non-negative (a people-density/occupancy weight)")

    if model == "logit":
        eps = 1e-9
        p_hazard = np.clip(hazard_arr, eps, 1.0 - eps)
        logit_hazard = np.log(p_hazard / (1.0 - p_hazard))
        z = logit_hazard + np.log1p(exposure_arr)
        return 1.0 / (1.0 + np.exp(-z))
    if model == "linear":
        return np.clip(hazard_arr * exposure_arr, 0.0, 1.0)
    raise ValueError(f"unknown incident_probability model {model!r}; expected 'logit' or 'linear'")


# Causal look-back window for the local predictive fit. Not part of the public signature (the work
# order freezes ``exposure_exceedance_monitor``'s parameters); kept as an internal constant so the
# window can be retuned later without touching callers.
_LOCAL_WINDOW = 30
_MIN_LOCAL_HISTORY = 5
_MIN_SCALE = 1e-9


@dataclass
class ExceedanceReport:
    """Per-timestep exceedance call: which points alert, their raw probability, and the target rate.

    ``alerts`` is the boolean array the caller (or the mlops drift/retrain wiring) acts on; it is
    already conformal-calibrated -- do not re-threshold ``prob_exceed`` again downstream.
    """

    alerts: np.ndarray
    prob_exceed: np.ndarray
    false_alarm_target: float


def _causal_local_stats(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal (past-only) rolling mean/std per index, falling back to the global fit while warming up.

    ``x[t]`` never looks at itself or the future: the local predictive at ``t`` is built from
    ``x[max(0, t - window):t]``. Early indices (fewer than ``_MIN_LOCAL_HISTORY`` prior points) fall
    back to the series' global mean/std so the predictive is never built from a near-empty window.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    mean = np.empty(n)
    std = np.empty(n)
    global_mean = float(x.mean()) if n > 0 else 0.0
    global_std = max(float(x.std(ddof=1)), _MIN_SCALE) if n > 1 else _MIN_SCALE
    for t in range(n):
        hist = x[max(0, t - window) : t]
        if hist.shape[0] >= _MIN_LOCAL_HISTORY:
            mean[t] = hist.mean()
            std[t] = max(float(hist.std(ddof=1)), _MIN_SCALE)
        else:
            mean[t] = global_mean
            std[t] = global_std
    return mean, std


def exposure_exceedance_monitor(
    series: np.ndarray,
    limit: float,
    *,
    alpha: float = 0.05,
    calib: np.ndarray | None = None,
) -> ExceedanceReport:
    """Flag exceedance excursions in ``series`` against ``limit`` at a calibrated false-alarm rate.

    Args:
        series: ``(n,)`` monitoring readings (e.g. silica PM4 concentration over time).
        limit: the occupational/community exposure limit being monitored against.
        alpha: target false-alarm rate (``ExceedanceReport.false_alarm_target``); the empirical alert
            rate on exchangeable, non-exceeding data is bounded by this via conformal calibration.
        calib: ``(m,)`` held-out reference readings known to be exposure-compliant (sub-limit). When
            omitted, ``series`` calibrates itself -- a graceful degradation for callers with no
            separate holdout, at the cost of a slightly less independent calibration set.

    Returns:
        An :class:`ExceedanceReport`.

    Algorithm:
        1. Fit a causal local Gaussian predictive at every timestep of ``series`` (and, separately, of
           ``calib``) via :func:`_causal_local_stats`, then read off ``P(reading > limit)`` under that
           predictive with :func:`scipy.stats.norm.sf` -- the IC-8 ``prob_exceed`` idea (probability
           mass of a distribution above a threshold) applied pointwise instead of over a spatial
           posterior region.
        2. Calibrate the alert threshold: treat each calibration timestep's ``prob_exceed`` value as a
           one-sided conformal nonconformity score (:func:`mixle.inference.conformal.split_conformal`,
           ``side="upper"``, against a constant zero "prediction") on the *known-safe* ``calib`` set.
           The returned upper bound is the smallest cutoff such that, under exchangeability with
           ``calib``, at most an ``alpha`` fraction of non-exceeding timesteps would clear it --
           a distribution-free false-alarm-rate guarantee, not a normal-theory approximation.
        3. Alert wherever ``series``'s ``prob_exceed`` clears that calibrated threshold.
    """
    series = np.asarray(series, dtype=float)
    calib_arr = np.asarray(calib, dtype=float) if calib is not None else series

    mean, std = _causal_local_stats(series, _LOCAL_WINDOW)
    prob_exceed = stats.norm.sf(limit, loc=mean, scale=std)

    cal_mean, cal_std = _causal_local_stats(calib_arr, _LOCAL_WINDOW)
    prob_exceed_calib = stats.norm.sf(limit, loc=cal_mean, scale=cal_std)

    zero_calib_pred = np.zeros_like(prob_exceed_calib)
    zero_test_pred = np.zeros_like(prob_exceed)
    _, calibrated_upper = split_conformal(zero_calib_pred, prob_exceed_calib, zero_test_pred, alpha=alpha, side="upper")
    threshold = float(calibrated_upper[0]) if calibrated_upper.size else float("inf")

    alerts = prob_exceed > threshold
    return ExceedanceReport(alerts=alerts, prob_exceed=prob_exceed, false_alarm_target=alpha)


__all__ = [
    "DOSE_RESPONSE_MODELS",
    "DoseResponse",
    "cumulative_exposure",
    "population_risk",
    "health_liability",
    "exposure_constraints",
    "safety_risk_surface",
    "incident_probability",
    "ExceedanceReport",
    "exposure_exceedance_monitor",
]
