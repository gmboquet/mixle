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

  * :func:`exposure_exceedance_monitor` -- per-timestep ``P(exposure > limit)`` treating the OBSERVED
    reading itself as a noisy measurement of the true exposure level, scaled by a causal (strictly
    past-only) local predictive fit around it (the same *distribution-over-a-threshold* idea as IC-8's
    ``mixle_pde.decision_quantities.prob_exceed``, here applied to a scalar monitoring series rather
    than a spatial posterior field, since a live sensor stream is not itself an IC-1 ``Posterior``) --
    a sudden extreme reading enters its own score and can trigger at the timestep it happens, not only
    once later readings drag a lagging rolling mean toward it. The raw probability is then run through
    :func:`mixle.inference.conformal.split_conformal` against a held-out ``calib`` reference (known-safe,
    sub-limit history, drawn from a stable reference period and scored by the same past-only rule) so
    the alert threshold is distribution-free calibrated: under exchangeability with an adequately-sized
    ``calib``, the empirical false-alarm rate is bounded by ``alpha``, not just "probably fine" from an
    untested normal-theory cutoff. When ``calib`` is omitted or too small for ``alpha`` to have real
    resolution, the monitored series scores itself as a graceful degradation, but the report comes back
    explicitly ``calibrated=False`` rather than silently implying the bound holds.

An alert firing (``ExceedanceReport.alerts.any()``) is the hook the mlops drift/retrain half (G7,
``mixle_mlops.drift_retrain``) watches for a re-check/retrain trigger -- that wiring is a *signal*
(read this array), not a code dependency: this module never imports anything from ``mixle_mlops``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from mixle.analysis._interval import validated_level
from mixle.inference.conformal import split_conformal
from mixle.utils.exact import require_exact_bool

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


def _positive_draw_count(n: int, *, name: str = "n") -> int:
    """Return an exact positive draw count without Boolean or fractional coercion."""
    if isinstance(n, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-Boolean positive integer")
    try:
        count = operator.index(n)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-Boolean positive integer") from exc
    if count <= 0:
        raise ValueError(f"{name} must be positive, got {count}")
    return int(count)


def _validated_dose_values(dose: Any, *, name: str = "dose") -> np.ndarray:
    """Validate the universal physical dose domain before any model can clip or transform it."""
    arr = np.asarray(dose, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite (no NaN/Inf)")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _validated_probability_samples(samples: Any, *, n: int, name: str) -> np.ndarray:
    """Validate a scalar or receptor-vector probability draw ensemble with an explicit sample axis."""
    arr = np.asarray(samples, dtype=float)
    if arr.ndim not in (1, 2) or arr.shape[0] != n or any(size == 0 for size in arr.shape):
        raise ValueError(f"{name} must have shape (n,) or (n, n_receptors) with n={n}; got {arr.shape}")
    if not np.isfinite(arr).all() or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must contain only finite probabilities in [0, 1]")
    return arr


def _validated_prior_flag(quantity: Any, *, name: str) -> bool:
    flag = getattr(quantity, "prior_dominated", None)
    if not isinstance(flag, (bool, np.bool_)):
        raise ValueError(f"{name}.prior_dominated must be Boolean")
    return bool(flag)


def _require_exact_bool(value: Any, name: str) -> bool:
    """Require an actual Boolean for a policy or honesty flag -- no truthiness coercion.

    ``bool("false")`` is ``True``, so a flag read from serialized configuration text could invert the
    very policy it names: mark a result prior-dominated, enable ``treat_unmodeled_as_safe``, or
    declare a set of alerts conformal-calibrated when they are not (MXR-080-1588). These flags gate
    safety and honesty decisions, so a non-Boolean is a caller error rather than something to coerce.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an actual Boolean (True/False), got {value!r}")
    return bool(value)


@dataclass(frozen=True)
class _SampleDerivedQuantity:
    """A concrete IC-1 `DerivedQuantity`: a draw matrix + the honesty flag, CI by empirical quantile.

    The same "samples + quantile-based `credible_interval` + `prior_dominated`" shape used by the
    frozen IC-1 conformance stub and the H4 stochastic-plan tests -- the repo's established idiom for
    a concrete derived quantity, rather than a bespoke one per caller.

    Frozen, holding a validated read-only copy of the draws. The validation below was previously
    performed on a throwaway array while the field kept aliasing whatever the caller passed, so the
    checked object and the stored object were not the same one: a later in-place edit through the
    caller's handle put NaNs (or a different distribution entirely) behind a record that had already
    certified itself finite.
    """

    samples: np.ndarray
    prior_dominated: bool = False

    def __post_init__(self) -> None:
        # Generic defense-in-depth (MXR-080-0094/0098): this container is shared across semantically
        # different quantities (dose-response probabilities, expected case counts, dollar
        # liabilities), so only a finite/non-empty check is universal here -- the probability-specific
        # [0, 1] range gate lives at the dose-response pushforward itself (`_validated_response_fn`),
        # where the samples' meaning as a probability is actually known.
        arr = np.array(self.samples, dtype=float, copy=True)
        if arr.size == 0:
            raise ValueError("_SampleDerivedQuantity.samples must be non-empty.")
        if not np.isfinite(arr).all():
            raise ValueError("_SampleDerivedQuantity.samples must be finite (no NaN/Inf).")
        arr.setflags(write=False)
        object.__setattr__(self, "samples", arr)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Central ``level`` interval by empirical quantile (shared analysis contract; MXR-080-1580)."""
        a = (1.0 - validated_level(level)) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


@dataclass(frozen=True)
class _DeterministicRisk:
    """A degenerate `DerivedQuantity` for a plain-`ndarray` (no-UQ) deformation input.

    Satisfies the IC-1 `DerivedQuantity` protocol with a single replicate: there is no posterior to
    sample from, so the exceedance is either 0 or 1 per cell and the credible interval collapses to a
    point. `prior_dominated` is always False -- there is no prior/regulariser in play, only a direct
    threshold test on the supplied field. `grid_shape` is extra (not part of IC-1) so a caller that
    knows it received an ndarray can reshape `samples` back into the original spatial layout.

    Construction validates `samples`: non-empty and finite (no NaN/Inf) -- the same defense-in-depth
    guard `_SampleDerivedQuantity` above carries, so invalid state can never flow downstream even if
    some upstream pushforward (here, `safety_risk_surface`'s ndarray path) fails to validate its own
    inputs.
    """

    samples: np.ndarray  # (1, n_cells), 0.0/1.0 exceedance indicator
    grid_shape: tuple[int, ...]
    prior_dominated: bool = field(default=False)

    def __post_init__(self) -> None:
        arr = np.array(self.samples, dtype=float, copy=True)
        if arr.size == 0:
            raise ValueError("_DeterministicRisk.samples must be non-empty.")
        if not np.isfinite(arr).all():
            raise ValueError("_DeterministicRisk.samples must be finite (no NaN/Inf).")
        arr.setflags(write=False)
        object.__setattr__(self, "samples", arr)
        object.__setattr__(self, "grid_shape", tuple(int(d) for d in self.grid_shape))

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Degenerate interval: one replicate, so every level collapses onto the same point.

        ``level`` is still validated against the shared analysis contract even though the answer does
        not depend on it. Silently accepting `level=5.0` here -- while the sibling sample-based
        carriers raised on it -- is exactly how an out-of-range level went unnoticed until it reached
        a stochastic path (MXR-080-1580).
        """
        validated_level(level)
        point = self.samples[0]
        return point, point


def _dose_response_fn(model: str, params: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Return the elementwise dose -> outcome-probability map for the named ``model``.

    Every model's coefficients are validated against their own mathematically-required domain before
    the closure is built (MXR-080-0094): finite, plus whatever range keeps the model's output a valid
    probability for any non-negative dose. This is the first of two independent defenses against a
    non-probability output; the second is the finite-``[0, 1]`` gate `_validated_response_fn` wraps
    around the returned closure (see `DoseResponse.response_fn`), which also catches a bad *dose*
    value (e.g. non-finite input) that no amount of parameter validation alone can rule out.
    """
    if model == "loglinear":
        beta = float(params["beta"])
        # P = 1 - exp(-beta*dose): finite and in [0, 1] for any dose >= 0 iff beta is finite and >= 0
        # (beta < 0 makes the exponent positive, driving P negative for any dose > 0).
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError(f"loglinear dose-response requires a finite, non-negative 'beta'; got {beta!r}")
        return lambda d: 1.0 - np.exp(-beta * np.clip(np.asarray(d, dtype=float), 0.0, None))
    if model == "logit":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        # sigmoid(a*dose + b) is mathematically in (0, 1) for any finite logit, but a non-finite a/b
        # poisons the whole array with NaN (a finite dose times a NaN/inf coefficient is never finite).
        if not np.isfinite(a):
            raise ValueError(f"logit dose-response requires a finite 'a'; got {a!r}")
        if not np.isfinite(b):
            raise ValueError(f"logit dose-response requires a finite 'b'; got {b!r}")
        return lambda d: 1.0 / (1.0 + np.exp(-(a * np.asarray(d, dtype=float) + b)))
    if model == "hill":
        emax = float(params.get("emax", 1.0))
        ec50 = float(params["ec50"])
        hill_n = float(params.get("n", 1.0))
        # ec50 <= 0 makes the denominator ill-defined at dose == 0 (0/0) and, for non-integer hill_n,
        # a negative ec50 raised to a fractional power is complex, not a non-finite real -- silently
        # corrupting the output dtype rather than merely its range.
        if not np.isfinite(ec50) or ec50 <= 0.0:
            raise ValueError(f"hill dose-response requires a finite, positive 'ec50'; got {ec50!r}")
        # hill_n <= 0 makes dose**hill_n divide by zero at dose == 0 (hill_n < 0) or collapses the
        # curve to a dose-independent constant (hill_n == 0) -- neither is a valid Hill exponent.
        if not np.isfinite(hill_n) or hill_n <= 0.0:
            raise ValueError(f"hill dose-response requires a finite, positive 'n' (Hill exponent); got {hill_n!r}")
        # emax is the response ceiling as dose -> infinity; it must itself be a valid probability or
        # the curve exceeds [0, 1] (emax > 1) or goes negative (emax < 0) well before infinite dose.
        if not np.isfinite(emax) or emax < 0.0 or emax > 1.0:
            raise ValueError(f"hill dose-response requires 'emax' finite and in [0, 1]; got {emax!r}")

        def _hill(d: np.ndarray) -> np.ndarray:
            x = np.clip(np.asarray(d, dtype=float), 0.0, None)
            xn = x**hill_n
            return emax * xn / (ec50**hill_n + xn)

        return _hill
    if model == "threshold_linear":
        slope = float(params["slope"])
        threshold = float(params.get("threshold", 0.0))
        # The output clip(..., 0, 1) only sanitizes a finite argument -- clip() does not launder NaN
        # (any comparison with NaN is False, so a NaN slope/threshold passes straight through), and
        # slope == +/-inf can hit an inf*0 == NaN the moment dose == threshold exactly.
        if not np.isfinite(slope):
            raise ValueError(f"threshold_linear dose-response requires a finite 'slope'; got {slope!r}")
        if not np.isfinite(threshold):
            raise ValueError(f"threshold_linear dose-response requires a finite 'threshold'; got {threshold!r}")
        return lambda d: np.clip(slope * (np.asarray(d, dtype=float) - threshold), 0.0, 1.0)
    raise ValueError(f"unknown dose-response model {model!r}; expected one of {DOSE_RESPONSE_MODELS}")


def _validated_response_fn(fn: Callable[[np.ndarray], np.ndarray], model: str) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a dose-response elementwise function with the final finite-``[0, 1]`` output gate (MXR-080-0094).

    Per-model parameter validation in `_dose_response_fn` rejects bad coefficients before the closure
    is even built; this is the second, independent line of defense the finding demands: whatever the
    (already-validated) coefficients and whatever dose values a caller supplies, an array is only ever
    labeled an outcome-probability `DerivedQuantity` sample once it is confirmed finite and in
    ``[0, 1]`` here -- catching, in particular, a non-finite *dose* input that no amount of parameter
    validation alone could rule out.
    """

    def _wrapped(d: np.ndarray) -> np.ndarray:
        validated_dose = _validated_dose_values(d)
        out = np.asarray(fn(validated_dose), dtype=float)
        if not np.isfinite(out).all():
            raise ValueError(
                f"{model} dose-response pushforward produced non-finite output (NaN/Inf); refusing to "
                "label it an outcome-probability DerivedQuantity"
            )
        if np.any(out < 0.0) or np.any(out > 1.0):
            raise ValueError(
                f"{model} dose-response pushforward produced output outside [0, 1] "
                f"(min={float(out.min())!r}, max={float(out.max())!r}); refusing to label it an "
                "outcome-probability DerivedQuantity"
            )
        return out

    return _wrapped


def _as_dose_samples(dose: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Coerce a bare-array/scalar ``dose`` into ``n`` dose draws (Posterior doses take a separate path).

    A scalar (or length-1 array) is a degenerate point mass, replicated ``n`` times. A length-``n``
    array is treated as an already-drawn ensemble. Any other length is an "array-with-UQ" sample set
    of a different size, resampled with replacement to ``n`` draws.
    """
    raw = np.asarray(dose, dtype=float)
    if raw.ndim > 1:
        raise ValueError(
            f"bare dose must be a scalar or one-dimensional draw ensemble; got shape {raw.shape}. "
            "Use population_risk for an explicit receptor axis."
        )
    arr = np.atleast_1d(raw)
    _validated_dose_values(arr)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.shape[0] == n:
        return arr
    idx = rng.integers(0, arr.shape[0], size=n)
    return arr[idx]


@dataclass(frozen=True)
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

    Frozen, holding its own copy of ``params``. The eager domain validation below is only meaningful
    if the validated coefficients are the ones actually used: with a mutable record and a shared
    dict, a caller could construct a valid model and then swap ``model`` or edit ``params`` -- and
    every later ``response_fn()`` would rebuild the closure from coefficients that never passed the
    construction-time gate.
    """

    model: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in DOSE_RESPONSE_MODELS:
            raise ValueError(f"unknown dose-response model {self.model!r}; expected one of {DOSE_RESPONSE_MODELS}")
        object.__setattr__(self, "params", dict(self.params))
        # Eager parameter-domain validation (MXR-080-0094): fail at construction, not at first use --
        # `_dose_response_fn` raises a clear, model-specific error for an out-of-domain coefficient.
        _dose_response_fn(self.model, self.params)

    def response_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """The elementwise dose -> outcome-probability function for this model + params."""
        return _validated_response_fn(_dose_response_fn(self.model, self.params), self.model)

    def probability(self, dose: Any, *, n: int = 2000, rng: np.random.Generator) -> DerivedQuantity:
        """Push ``dose`` through the dose-response model into an outcome-probability `DerivedQuantity`.

        ``dose`` may be an IC-1 `Posterior` over exposure (the pushforward runs through the
        posterior's own ``derived_quantity``, so `prior_dominated` propagates from the exposure
        uncertainty), an array of dose draws/ensemble members ("array-with-UQ", resampled to ``n``
        if its length differs), or a bare scalar (a degenerate point mass -- the returned quantity
        still carries a (trivial) credible interval). Bare ensembles are one-dimensional. Posterior
        draws explicitly use shape ``(n,)`` for scalar dose or ``(n, n_receptors)`` for a receptor
        vector. Every dose must be finite and nonnegative, and ``n`` is an exact non-Boolean positive
        integer.
        """
        from mixle.reason.posterior_protocol import Posterior

        n = _positive_draw_count(n)
        fn = self.response_fn()
        if isinstance(dose, Posterior):

            def _pushforward(draws: np.ndarray) -> np.ndarray:
                dose_draws = _validated_dose_values(draws, name="posterior dose draws")
                if dose_draws.ndim not in (1, 2) or dose_draws.shape[0] != n:
                    raise ValueError(
                        "posterior dose draws must have shape (n,) or (n, n_receptors) "
                        f"with n={n}; got {dose_draws.shape}"
                    )
                return fn(dose_draws)

            quantity = dose.derived_quantity(_pushforward, n, rng)
            samples = _validated_probability_samples(
                getattr(quantity, "samples", None),
                n=n,
                name="posterior dose-response samples",
            )
            return _SampleDerivedQuantity(
                samples=np.array(samples, copy=True),
                prior_dominated=_validated_prior_flag(quantity, name="posterior dose-response result"),
            )
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

    ``series`` must be a one-dimensional finite nonnegative time series; multidimensional
    receptor/time data require an explicit aggregation before this scalar integration. ``dt`` must be
    finite and positive (MXR-080-0098: a zero or negative timestep makes the integral either
    trivially zero or sign-flipped, silently), and ``decay`` finite and non-negative (a negative
    "decay" would amplify, not discount, older readings without bound).
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"series must be a one-dimensional time series, got shape {x.shape}")
    if not np.isfinite(x).all() or np.any(x < 0.0):
        raise ValueError("series must be finite and nonnegative")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    decay = float(decay)
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError(f"decay must be finite and non-negative, got {decay!r}")
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return float(x[0] * dt)
    if decay <= 0.0:
        return float(np.trapezoid(x, dx=dt))
    times = np.arange(x.size) * dt
    decayed = x * np.exp(-decay * (times[-1] - times))
    return float(np.trapezoid(decayed, dx=dt))


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

    n = _positive_draw_count(n)
    fn = dr.response_fn()
    if isinstance(exposure, Posterior):

        def _aggregate(draws: np.ndarray) -> np.ndarray:
            dose_draws = _validated_dose_values(draws, name="posterior exposure draws")
            if dose_draws.ndim != 2 or dose_draws.shape[0] != n or dose_draws.shape[1] == 0:
                raise ValueError(
                    f"posterior exposure draws must have shape (n, n_receptors) with n={n}; got {dose_draws.shape}"
                )
            return fn(dose_draws).sum(axis=1)

        quantity = exposure.derived_quantity(_aggregate, n, rng)
        samples = np.asarray(getattr(quantity, "samples", None), dtype=float)
        if samples.shape != (n,) or not np.isfinite(samples).all() or np.any(samples < 0.0):
            raise ValueError(f"posterior population-risk samples must be finite nonnegative shape ({n},)")
        return _SampleDerivedQuantity(
            samples=np.array(samples, copy=True),
            prior_dominated=_validated_prior_flag(quantity, name="posterior population-risk result"),
        )
    arr = np.asarray(exposure, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"exposure must be a nonempty one-dimensional receptor vector, got shape {arr.shape}")
    expected_cases = float(np.sum(fn(arr)))
    return _SampleDerivedQuantity(samples=np.full(n, expected_cases), prior_dominated=False)


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

    ``risk.samples`` must be a nonempty finite nonnegative ``(n,)`` case-count series or
    ``(n, n_receptors)`` risk surface; negative harm is never interpreted as a benefit.
    ``cost_per_case`` must be finite and non-negative (a dollar price cannot be negative), and
    ``discount`` finite and strictly greater than ``-1.0`` (MXR-080-0098: the ``1 / (1 + discount)``
    present-value factor is only well-defined for a denominator that is positive; ``discount == -1``
    divides by zero, and anything below it silently flips the liability's sign).
    """
    cost_per_case = float(cost_per_case)
    if not np.isfinite(cost_per_case) or cost_per_case < 0.0:
        raise ValueError(f"cost_per_case must be finite and non-negative, got {cost_per_case!r}")
    discount = float(discount)
    if not np.isfinite(discount) or discount <= -1.0:
        raise ValueError(
            f"discount must be finite and > -1.0 (the 1/(1+discount) present-value factor requires a "
            f"positive denominator), got {discount!r}"
        )
    samples = np.asarray(risk.samples, dtype=float)
    if samples.ndim not in (1, 2) or any(size == 0 for size in samples.shape):
        raise ValueError("risk.samples must have shape (n,) or (n, n_receptors) and be nonempty")
    if not np.isfinite(samples).all() or np.any(samples < 0.0):
        raise ValueError("risk.samples must be finite and nonnegative")
    factor = cost_per_case / (1.0 + discount)
    with np.errstate(over="ignore", invalid="ignore"):
        liability_samples = samples * factor
    if not np.isfinite(liability_samples).all():
        raise ValueError("health liability samples overflowed")
    prior_dominated = _validated_prior_flag(risk, name="risk")
    return _SampleDerivedQuantity(samples=liability_samples, prior_dominated=prior_dominated)


def exposure_constraints(
    options: list[dict],
    limits: dict[str, float],
    *,
    treat_unmodeled_as_safe: bool = False,
) -> list[dict]:
    """Screen candidate operating ``options`` against named exposure/exceedance ``limits`` (K6).

    ``limits`` maps an occupational/community exposure metric name (e.g. ``"silica_pm4"``, an
    8-hour TWA, or an exceedance probability) to the regulatory/policy limit for that metric.
    ``options`` is a list of plain dicts -- each one candidate operating configuration, carrying
    (among whatever other plan data the caller needs, e.g. block cost or grade) a value for zero or
    more of those metric keys.

    Returns a *new* list (the input dicts are never mutated), one entry per option, each the
    original option's key/value pairs plus:

    - ``"status"``: one of ``"safe"``, ``"violating"``, or ``"unknown"``. ``"violating"`` means at
      least one limited metric the option DOES carry a finite value for breaches its (finite) limit.
      ``"unknown"`` means no limit was confirmed breached, but at least one limited metric could not
      be evaluated for this option -- missing from the option entirely, a non-finite (NaN/Inf)
      measurement, or a non-finite/missing limit -- "not modeled" and "safe" are never the same
      outcome here, because this is a hard occupational/community exposure screen, not a soft
      optimization preference (MXR-080-0095). A confirmed violation always dominates an unknown: one
      metric's missing data never erases a different metric's real breach.
    - ``"feasible"``: ``True`` iff ``status == "safe"`` -- unless ``treat_unmodeled_as_safe=True``, in
      which case ``status == "unknown"`` also counts as feasible (see below). Kept as a plain boolean
      for the existing contract H4's ``two_stage_stochastic_plan`` filtering relies on.
    - ``"binding"``: the sorted list of limit names actually confirmed breached (empty unless
      ``status == "violating"``) -- naming exactly which limit made the option infeasible.
    - ``"unmodeled"``: the sorted list of limited metric names that could not be evaluated for this
      option (empty unless ``status == "unknown"``) -- naming exactly which evidence was missing.

    This is a HARD safety screen, so it FAILS CLOSED by default: an option that breaches no limit it
    was evaluated against, but leaves one or more limits unevaluated, is ``"unknown"`` and therefore
    ``feasible=False`` -- unmodeled evidence is never silently treated as compliant, and a NaN
    measurement (which would otherwise compare False, and so silently pass, against ``> limit``) is
    treated the same as a missing one. Pass ``treat_unmodeled_as_safe=True`` to explicitly opt into
    the old permissive behavior; this is a deliberate policy override the caller must request, and it
    defaults to off. A caller filters the returned list down to the feasible options *before* handing
    the survivors' blocks to H4's ``two_stage_stochastic_plan`` (``stochastic_opt.py``) -- an
    infeasible (violating OR, by default, unknown) option is dropped from the candidate set entirely,
    so the optimizer never has the chance to select it (see the K6 DoD).
    """
    treat_unmodeled_as_safe = _require_exact_bool(treat_unmodeled_as_safe, "treat_unmodeled_as_safe")
    annotated: list[dict] = []
    for option in options:
        binding: list[str] = []
        unmodeled: list[str] = []
        for name, limit in limits.items():
            limit_val = float(limit)
            if not np.isfinite(limit_val):
                unmodeled.append(name)
                continue
            if name not in option:
                unmodeled.append(name)
                continue
            value = float(option[name])
            if not np.isfinite(value):
                unmodeled.append(name)
                continue
            if value > limit_val:
                binding.append(name)
        binding.sort()
        unmodeled.sort()

        if binding:
            status = "violating"
        elif unmodeled:
            status = "unknown"
        else:
            status = "safe"

        if status == "violating":
            feasible = False
        elif status == "unknown":
            feasible = require_exact_bool(treat_unmodeled_as_safe, "treat_unmodeled_as_safe")
        else:
            feasible = True

        out = dict(option)
        out["status"] = status
        out["feasible"] = feasible
        out["binding"] = binding
        out["unmodeled"] = unmodeled
        annotated.append(out)
    return annotated


def _grid_shape_for(posterior: Posterior, slope: np.ndarray | None) -> tuple[int, ...]:
    """Infer the spatial grid shape backing a flat `(d,)` posterior mean.

    Prefers `slope`'s shape (the caller already knows the grid geometry whenever it supplies terrain
    slope), then an optional `grid_shape` attribute some posteriors may carry (additive -- not part of
    IC-1, but structural typing does not forbid extra attributes), then falls back to a square grid if
    `d` is a perfect square, else treats the field as a 1-D transect.
    """
    mean = np.asarray(posterior.mean, dtype=float)
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError(f"posterior.mean must be a nonempty flattened one-dimensional field, got shape {mean.shape}")
    if not np.isfinite(mean).all():
        raise ValueError("posterior.mean must be finite")
    d = mean.size
    if slope is not None:
        shape = tuple(np.asarray(slope).shape)
        if not shape or any(size == 0 for size in shape):
            raise ValueError("slope must be a nonempty spatial field")
        if int(np.prod(shape)) != d:
            raise ValueError(f"slope shape {shape} does not match deformation dimension {d}")
        return shape
    grid_shape = getattr(posterior, "grid_shape", None)
    if grid_shape is not None:
        try:
            shape = tuple(_positive_draw_count(size, name="posterior.grid_shape entry") for size in grid_shape)
        except TypeError as exc:
            raise ValueError("posterior.grid_shape must be an iterable of positive integer dimensions") from exc
        if not shape:
            raise ValueError("posterior.grid_shape must contain at least one spatial dimension")
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
    spatial_axes = tuple(axis for axis in range(1, grid.ndim) if grid.shape[axis] > 1)
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

    ``gradient_limit`` must be finite and non-negative (a tilt *magnitude* threshold cannot itself be
    negative), and ``slope``/an ``ndarray`` ``deformation`` must be finite (MXR-080-0098): a NaN either
    way would otherwise compare False against ``> gradient_limit`` and silently mark an unevaluable
    cell as "not exceeding" instead of raising. Posterior evidence additionally requires a finite
    flattened mean and exactly ``(_MC_SAMPLES, n_cells)`` finite draws; the returned binary risk
    samples are validated independently before publication.
    """
    from mixle.reason.posterior_protocol import Posterior

    gradient_limit = float(gradient_limit)
    if not np.isfinite(gradient_limit) or gradient_limit < 0.0:
        raise ValueError(
            f"gradient_limit must be finite and non-negative (a tilt-magnitude threshold), got {gradient_limit!r}"
        )
    slope_arr_raw = None
    if slope is not None:
        slope_arr_raw = np.asarray(slope, dtype=float)
        if not np.isfinite(slope_arr_raw).all():
            raise ValueError("slope must be finite (no NaN/Inf)")

    if isinstance(deformation, np.ndarray):
        grid = np.asarray(deformation, dtype=float)
        if grid.ndim == 0 or grid.size == 0:
            raise ValueError("deformation ndarray must be a nonempty spatial field")
        if not np.isfinite(grid).all():
            raise ValueError("deformation ndarray must be finite (no NaN/Inf)")
        grid_shape = grid.shape
        tilt = _gradient_magnitude(grid[np.newaxis, ...])[0]
        if slope_arr_raw is not None:
            if slope_arr_raw.shape != grid_shape:
                raise ValueError(f"slope shape {slope_arr_raw.shape} does not match deformation shape {grid_shape}")
            tilt = tilt + slope_arr_raw
        exceed = (tilt > gradient_limit).astype(float).reshape(1, -1)
        return _DeterministicRisk(samples=exceed, grid_shape=grid_shape)

    if not isinstance(deformation, Posterior):
        raise TypeError("deformation must be an IC-1 Posterior or an np.ndarray field")

    grid_shape = _grid_shape_for(deformation, slope_arr_raw)
    n_cells = int(np.prod(grid_shape))
    slope_arr = None if slope_arr_raw is None else slope_arr_raw.reshape(grid_shape)

    def _pushforward(draws: np.ndarray) -> np.ndarray:
        draw_array = np.asarray(draws, dtype=float)
        expected_shape = (_MC_SAMPLES, n_cells)
        if draw_array.shape != expected_shape:
            raise ValueError(f"posterior deformation draws must have shape {expected_shape}, got {draw_array.shape}")
        if not np.isfinite(draw_array).all():
            raise ValueError("posterior deformation draws must be finite")
        grid = draw_array.reshape((_MC_SAMPLES, *grid_shape))
        tilt = _gradient_magnitude(grid)
        if slope_arr is not None:
            tilt = tilt + slope_arr[np.newaxis, ...]
        if not np.isfinite(tilt).all():
            raise ValueError("posterior deformation tilt must remain finite")
        exceed = (tilt > gradient_limit).astype(float)
        return exceed.reshape(_MC_SAMPLES, -1)

    rng = np.random.default_rng(_MC_SEED)
    quantity = deformation.derived_quantity(_pushforward, _MC_SAMPLES, rng)
    samples = np.asarray(getattr(quantity, "samples", None), dtype=float)
    expected_result_shape = (_MC_SAMPLES, n_cells)
    if samples.shape != expected_result_shape:
        raise ValueError(f"posterior safety-risk samples must have shape {expected_result_shape}, got {samples.shape}")
    if not np.isfinite(samples).all() or not np.all((samples == 0.0) | (samples == 1.0)):
        raise ValueError("posterior safety-risk samples must be finite binary exceedance indicators")
    return _SampleDerivedQuantity(
        samples=np.array(samples, copy=True),
        prior_dominated=_validated_prior_flag(quantity, name="posterior safety-risk result"),
    )


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

    ``exposure_map`` is on the *expected people present* scale: a cell's value is the mean occupancy
    of that cell, so ``0`` means nobody, ``1`` means about one person on average, and larger values are
    denser. Both models take that scale literally and return ``0`` for an unoccupied cell.

    Args:
        hazard: per-cell hazard probability, shape `(*grid_shape,)`, values in `[0, 1]`.
        exposure_map: per-cell non-negative expected-occupancy weight, same shape as `hazard`.
        model: `"logit"` (default) is a logistic link that multiplies the hazard's own odds by the
            probability that anybody is actually present, `1 - exp(-exposure_map)` (the Poisson
            occupancy probability for that mean): an unoccupied cell yields `0`, denser occupancy
            monotonically raises the result, and it saturates at the hazard probability itself.
            `"linear"` is the simple product `hazard * exposure_map` clipped to `[0, 1]`.

    Returns:
        Per-cell incident probability, same shape as `hazard`, values in `[0, 1]`.

    ``hazard`` outside ``[0, 1]`` is REJECTED, not clipped into a falsely-confident boundary
    probability (MXR-080-0098): the pre-fix ``"logit"`` path clipped any hazard value -- 5.0, -3.0,
    whatever -- straight into ``[eps, 1 - eps]`` before ever checking it was a valid probability to
    begin with. The narrow ``eps``-clip that remains below is purely for the logit transform's own
    numerical stability at the exact 0/1 boundary of an ALREADY-valid probability, not a substitute
    for validating the input range.

    The ``"logit"`` occupancy factor used to be ``log1p(exposure_map)`` added to the log-odds
    (MXR-080-1589). That has no zero element: ``log1p(0)`` is ``0``, so an empty cell added no
    log-odds at all and hazard ``0.8`` at zero occupancy returned incident probability ``0.8`` --
    flatly contradicting this function's own stated contract that an exceedance is only an incident
    when someone is exposed. It was also unbounded above, so a dense-enough cell reported MORE
    incidents than hazard exceedances. Multiplying the odds by an occupancy PROBABILITY fixes both:
    zero occupancy is the zero element, and the incident probability can never exceed the hazard
    probability it is conditioned on.
    """
    hazard_arr = np.asarray(hazard, dtype=float)
    exposure_arr = np.asarray(exposure_map, dtype=float)
    if hazard_arr.shape != exposure_arr.shape:
        raise ValueError(f"hazard shape {hazard_arr.shape} does not match exposure_map shape {exposure_arr.shape}")
    if not np.isfinite(hazard_arr).all():
        raise ValueError("hazard must be finite (no NaN/Inf)")
    if np.any(hazard_arr < 0.0) or np.any(hazard_arr > 1.0):
        raise ValueError(
            f"hazard must be a probability in [0, 1]; got range [{float(hazard_arr.min())!r}, "
            f"{float(hazard_arr.max())!r}]. incident_probability does not accept out-of-range hazard "
            "evidence -- fix the upstream hazard source rather than relying on boundary clipping."
        )
    if not np.isfinite(exposure_arr).all():
        raise ValueError("exposure_map must be finite (no NaN/Inf)")
    if np.any(exposure_arr < 0):
        raise ValueError("exposure_map must be non-negative (a people-density/occupancy weight)")

    if model == "logit":
        eps = 1e-9
        p_hazard = np.clip(hazard_arr, eps, 1.0 - eps)
        # P(at least one person present) for a cell whose mean occupancy is exposure_map: 0 when
        # nobody is expected, rising to 1 as the cell fills. Multiplying the hazard ODDS by it is
        # the additive-on-the-log-odds logistic link this model is named for, written in its
        # algebraic form so an unoccupied cell lands on exactly 0 without a log(0) intermediate.
        occupancy = -np.expm1(-exposure_arr)
        # denominator >= 1 - p_hazard >= eps, so this is division-safe for every valid input
        return p_hazard * occupancy / (1.0 - p_hazard + p_hazard * occupancy)
    if model == "linear":
        return np.clip(hazard_arr * exposure_arr, 0.0, 1.0)
    raise ValueError(f"unknown incident_probability model {model!r}; expected 'logit' or 'linear'")


# Causal look-back window for the local predictive fit. Not part of the public signature (the work
# order freezes ``exposure_exceedance_monitor``'s parameters); kept as an internal constant so the
# window can be retuned later without touching callers.
_LOCAL_WINDOW = 30
_MIN_LOCAL_HISTORY = 5
_MIN_SCALE = 1e-9

# MXR-080-0097: the smallest calibration size for which the finite-sample conformal quantile's own
# resolution, ``1 / (n + 1)``, sits at least ``_MIN_CALIB_MARGIN``-fold below the target ``alpha`` --
# so a calibrated threshold reflects the target false-alarm rate with real resolution, rather than
# merely being finite (``n ~ 1/alpha`` already gives a finite quantile -- see `_conformal_quantile` in
# ``mixle.inference.conformal`` -- but that quantile is then just the single loosest calibration score,
# which does not meaningfully approximate ``alpha``).
_MIN_CALIB_MARGIN = 5.0


def _min_adequate_calib_size(alpha: float) -> int:
    """Smallest calibration sample count (see ``_MIN_CALIB_MARGIN``) for a meaningful ``alpha`` bound."""
    return max(1, int(np.ceil(_MIN_CALIB_MARGIN / alpha)) - 1)


@dataclass(frozen=True)
class ExceedanceReport:
    """Per-timestep exceedance call: which points alert, their raw probability, and the target rate.

    ``alerts`` is the boolean array the caller (or the mlops drift/retrain wiring) acts on; when
    ``calibrated`` is True it is already conformal-calibrated -- do not re-threshold ``prob_exceed``
    again downstream.

    ``calibrated`` (MXR-080-0097) is False whenever the returned alerts do NOT carry the
    distribution-free false-alarm-rate guarantee: ``calib`` was omitted (the monitored series --
    including any anomalies it contains -- scored itself, which is neither independent of nor
    exchangeable with what it is calibrating), or the effective calibration set was too small to give
    ``false_alarm_target`` real resolution (see ``_min_adequate_calib_size``). The alerts are still
    computed on a best-effort basis in that case, but the caller must treat them as a heuristic, not a
    proven bound.

    ``warmed_up`` (MXR-080-0096) marks which timesteps had enough strictly-prior history to be scored
    at all; an unwarmed timestep always reports ``prob_exceed == 0.0`` and never alerts -- an explicit
    "not yet evaluated", never a confident "safe".

    The record is frozen and owns read-only copies of its three arrays. ``calibrated`` and
    ``warmed_up`` are the honesty flags that qualify ``alerts``; while the record was mutable,
    flipping ``calibrated`` to True or overwriting ``alerts`` in place upgraded a best-effort
    heuristic into an apparently-proven false-alarm bound with nothing else changing.

    Construction additionally enforces the record's own domain (MXR-080-1900), which it previously
    did not check at all -- ``prob_exceed`` of ``5.0``, ``-3.0`` or ``NaN``, a ``false_alarm_target``
    of ``7.5``, and a ``prob_exceed``/``warmed_up`` array of a different length than ``alerts`` all
    constructed silently, so the record could report an alert whose "probability" was not one, at a
    "target rate" that was not a rate, for timesteps that did not line up with the alerts:

    * ``prob_exceed`` finite and in ``[0, 1]``. It is a survival-function value by construction, so
      this refuses nothing the producer emits.
    * ``false_alarm_target`` finite and strictly in ``(0, 1)`` -- the same range
      :func:`exposure_exceedance_monitor` already requires of its ``alpha``, now also true of a
      directly constructed report.
    * ``prob_exceed`` the same length as ``alerts``, one probability per alert.
    * ``warmed_up`` either the same length as ``alerts`` or EMPTY. Empty is the field's own default
      and means "warm-up was not recorded"; refusing it would reject reports the package's own tests
      construct, and it is not the same claim as a per-timestep flag that disagrees with the alerts.
    """

    alerts: np.ndarray
    prob_exceed: np.ndarray
    false_alarm_target: float
    calibrated: bool = True
    warmed_up: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    def __post_init__(self) -> None:
        for name, dtype in (("alerts", bool), ("prob_exceed", float), ("warmed_up", bool)):
            owned = np.array(getattr(self, name), dtype=dtype, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        target = float(self.false_alarm_target)
        if not (np.isfinite(target) and 0.0 < target < 1.0):
            raise ValueError(
                f"ExceedanceReport.false_alarm_target must be a finite false-alarm RATE strictly in "
                f"(0, 1), got {self.false_alarm_target!r}"
            )
        object.__setattr__(self, "false_alarm_target", target)
        if not np.isfinite(self.prob_exceed).all() or np.any(self.prob_exceed < 0.0) or np.any(self.prob_exceed > 1.0):
            raise ValueError(
                "ExceedanceReport.prob_exceed must be finite probabilities in [0, 1]; refusing to label "
                f"{self.prob_exceed!r} an exceedance probability"
            )
        if self.prob_exceed.shape != self.alerts.shape:
            raise ValueError(
                f"ExceedanceReport.prob_exceed must carry one probability per alert; got shape "
                f"{self.prob_exceed.shape} against alerts {self.alerts.shape}"
            )
        if self.warmed_up.size and self.warmed_up.shape != self.alerts.shape:
            raise ValueError(
                f"ExceedanceReport.warmed_up must be empty (not recorded) or carry one flag per alert; "
                f"got shape {self.warmed_up.shape} against alerts {self.alerts.shape}"
            )
        object.__setattr__(self, "calibrated", _require_exact_bool(self.calibrated, "ExceedanceReport.calibrated"))


def _causal_local_scale(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal (strictly-past-only) rolling scale per index, plus which indices are warmed up.

    ``std[t]`` is fit ONLY from ``x[max(0, t - window):t]`` -- never ``x[t]`` itself, and never
    anything at or after ``t`` -- so it cannot leak future information into the predictive state at
    ``t``. ``warmed_up[t]`` is False whenever fewer than ``_MIN_LOCAL_HISTORY`` strictly-prior points
    are available (the earliest indices in any series); those entries carry a placeholder scale the
    caller must not use (:func:`exposure_exceedance_monitor` never scores an unwarmed index) rather
    than this function reaching past the causal window into a global -- and, for an early index,
    necessarily partly-future -- statistic the way the pre-fix implementation did (MXR-080-0096).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    std = np.full(n, _MIN_SCALE)
    warmed_up = np.zeros(n, dtype=bool)
    for t in range(n):
        hist = x[max(0, t - window) : t]
        if hist.shape[0] >= _MIN_LOCAL_HISTORY:
            std[t] = max(float(hist.std(ddof=1)), _MIN_SCALE)
            warmed_up[t] = True
    return std, warmed_up


def exposure_exceedance_monitor(
    series: np.ndarray,
    limit: float,
    *,
    alpha: float = 0.05,
    calib: np.ndarray | None = None,
) -> ExceedanceReport:
    """Flag exceedance excursions in ``series`` against ``limit`` at a calibrated false-alarm rate.

    Args:
        series: ``(n,)`` monitoring readings (e.g. silica PM4 concentration over time). Must be finite.
        limit: the occupational/community exposure limit being monitored against. Must be finite.
        alpha: target false-alarm rate (``ExceedanceReport.false_alarm_target``); the empirical alert
            rate on exchangeable, non-exceeding data is bounded by this via conformal calibration --
            but only when the returned ``calibrated`` is True (see ``calib`` below). Must be in
            ``(0, 1)``.
        calib: ``(m,)`` held-out reference readings known to be exposure-compliant (sub-limit), drawn
            from a stable, anomaly-free reference period and scored by the same past-only rule as
            ``series`` -- the temporal-exchangeability scheme this monitor's false-alarm-rate guarantee
            depends on. When omitted, ``series`` scores itself (a graceful degradation for callers with
            no separate holdout) -- but the monitored series, including any anomalies it contains, is
            neither independent of nor exchangeable with what it is calibrating, so the returned report
            is explicitly marked ``calibrated=False`` and the ``alpha`` bound is NOT guaranteed
            (MXR-080-0097). An explicitly-supplied ``calib`` that is too small to give ``alpha`` real
            resolution (fewer than roughly ``_min_adequate_calib_size(alpha)`` usable, warmed-up
            points) is likewise reported uncalibrated rather than silently accepted.

    Returns:
        An :class:`ExceedanceReport`. When ``calibrated`` is False the alerts are still a best-effort
        heuristic, not a proven distribution-free bound.

    Algorithm:
        1. At every timestep ``t`` of ``series`` (and, separately, of ``calib``), fit a causal
           (strictly past-only) local Gaussian SCALE ``std[t]`` from ``series[max(0, t-window):t]`` --
           never ``series[t]`` itself, never anything at or after ``t`` -- via
           :func:`_causal_local_scale`. Timesteps with too little prior history are not "warmed up"
           (``ExceedanceReport.warmed_up``) and are reported as an explicit not-yet-evaluated state
           (``prob_exceed=0.0``, never alerting) instead of reaching past the window into a global --
           and, for an early index, necessarily partly-future -- statistic (MXR-080-0096).
        2. Score ``prob_exceed[t]`` as the probability that the TRUE exposure level exceeds ``limit``,
           treating the OBSERVED ``series[t]`` itself as a noisy measurement of that level with scale
           ``std[t]``: ``P(X > limit)`` for ``X ~ Normal(series[t], std[t])``
           (:func:`scipy.stats.norm.sf`). Centering on the observation itself -- rather than on the
           local mean of *prior* readings alone, which is blind to what was actually just measured --
           is what lets a single sudden extreme reading enter its own score and trigger at the exact
           timestep it happens, instead of only once later readings drag a lagging rolling mean toward
           it (MXR-080-0096).
        3. Calibrate the alert threshold: treat each WARMED-UP calibration timestep's ``prob_exceed``
           (scored by the identical observed-value rule) as a one-sided conformal nonconformity score
           (:func:`mixle.inference.conformal.split_conformal`, ``side="upper"``, against a constant
           zero "prediction") on the known-safe ``calib`` set. The returned upper bound is the
           smallest cutoff such that, under exchangeability with ``calib``, at most an ``alpha``
           fraction of non-exceeding timesteps would clear it -- a distribution-free false-alarm-rate
           guarantee, not a normal-theory approximation, PROVIDED ``calib`` genuinely satisfies that
           exchangeability and is large enough (see ``calib`` above and ``calibrated``).
        4. Alert wherever a warmed-up ``series`` timestep's ``prob_exceed`` clears that threshold.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"series must be a 1-D (n,) monitoring array, got shape {series.shape}")
    if not np.isfinite(series).all():
        raise ValueError("series must be finite (no NaN/Inf)")
    limit = float(limit)
    if not np.isfinite(limit):
        raise ValueError(f"limit must be finite, got {limit!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    explicit_calib = calib is not None
    calib_arr = np.asarray(calib, dtype=float) if explicit_calib else series
    if calib_arr.ndim != 1:
        raise ValueError(f"calib must be a 1-D (m,) reference array, got shape {calib_arr.shape}")
    if not np.isfinite(calib_arr).all():
        raise ValueError("calib must be finite (no NaN/Inf)")

    std, warmed_up = _causal_local_scale(series, _LOCAL_WINDOW)
    prob_exceed = np.where(warmed_up, stats.norm.sf(limit, loc=series, scale=std), 0.0)

    cal_std, cal_warmed_up = _causal_local_scale(calib_arr, _LOCAL_WINDOW)
    # Only genuinely-scored (warmed-up) calibration points enter the reference distribution -- a
    # placeholder "not yet evaluated" score is not a real nonconformity score.
    prob_exceed_calib = stats.norm.sf(limit, loc=calib_arr, scale=cal_std)[cal_warmed_up]

    n_eff = int(prob_exceed_calib.shape[0])
    calibrated = explicit_calib and n_eff >= _min_adequate_calib_size(alpha)

    if n_eff == 0:
        threshold = float("inf")
    else:
        zero_calib_pred = np.zeros_like(prob_exceed_calib)
        zero_test_pred = np.zeros_like(prob_exceed)
        _, calibrated_upper = split_conformal(
            zero_calib_pred, prob_exceed_calib, zero_test_pred, alpha=alpha, side="upper"
        )
        threshold = float(calibrated_upper[0]) if calibrated_upper.size else float("inf")

    alerts = warmed_up & (prob_exceed > threshold)
    return ExceedanceReport(
        alerts=alerts,
        prob_exceed=prob_exceed,
        false_alarm_target=alpha,
        calibrated=calibrated,
        warmed_up=warmed_up,
    )


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
