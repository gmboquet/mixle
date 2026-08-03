"""Real options & decision-under-uncertainty (work-plan Sec.7-J, J3).

A risk-neutral NPV distribution (J2's :class:`~mixle.analysis.valuation.NPVDistribution`) answers "what
is the project worth if we commit now?" It does not answer the question a real decision-maker actually
faces: "should we commit now, or is it worth paying for the *option* to wait, expand, or walk away once
more is known?" Two tools close that gap:

  * :func:`real_option_value` -- prices the option embedded in the decision (defer / expand / abandon)
    with a binomial lattice on the NPV process itself, American-exercised at every step. NPV -- unlike a
    commodity price -- is a signed quantity that must be able to cross zero for "wait and see" to mean
    anything, so the lattice is *additive* (an arithmetic random walk around ``npv_dist.mean``, scaled by
    ``volatility`` as a fraction of the NPV's own magnitude) rather than the usual multiplicative GBM
    lattice: a strictly-positive geometric process started above zero can never reach the ``max(V, 0)``
    kink that gives the option its value. Volatility is what makes the option valuable: by Jensen's
    inequality, an American option on a driftless process is worth strictly more than the naive point NPV
    whenever there is dispersion wide enough to reach the kink, and it collapses to the naive
    ``max(NPV, 0)`` as volatility falls to zero (no dispersion, nothing to wait for). Each ``kind`` carries
    its own exercise economics: ``expand`` nets the capacity bonus against an ``expansion_cost``
    investment, so expanding is a genuine trade-off rather than a bonus every positive-NPV project gets
    for free, and ``abandon`` recovers an optional ``salvage_value`` instead of assuming a total write-off.
  * :func:`voi_dollars` -- the dollar value of a piece of information (e.g. a delineation drillhole): the
    expected value of the best decision *with* that information, minus the expected value of the best
    decision *without* it. A declared ``observation_model`` is required by default so this is a real
    expected value of sample information (EVSI). A Gaussian-dispersion HEURISTIC remains available only
    through the explicit ``drill_info["method"] = "variance_rescaling_heuristic"`` opt-in: a
    hypothetical drillhole's effect on the
    posterior is summarized as a fractional variance reduction (either supplied directly via
    ``drill_info["variance_reduction"]``, or -- when available -- via C8's
    ``mixle_pde.voi.expected_variance_reduction`` hook), and the pre-posterior simulation invents
    plausible future-posterior centers/spreads by RESCALING today's posterior draws rather than sampling a
    hypothetical observation from a real likelihood and conditioning on it -- exact only for a Gaussian,
    unimodal, unconstrained posterior, and flagged with a warning otherwise. For a genuine
    sample-then-condition preposterior, declare a linear-Gaussian :class:`GaussianObservationModel`
    (``observation_model=``): a hypothetical observation is then simulated from the latent's actual
    prior-predictive marginal and the posterior is conditioned on it analytically via closed-form
    conjugate-Gaussian updating -- the textbook EVSI construction, for that declared-likelihood case. A
    rational decision-maker never loses from more information *in expectation*, but a Monte Carlo estimate
    of that gain is noisy around zero when there is little or no information to gain; :func:`voi_estimate`
    reports the (unfloored) estimate together with its own standard error rather than clipping the noise
    away, which -- naively floored at zero -- was systematically biased upward rather than merely
    conservative (:class:`VoiEstimate`).

Repo-boundary note: J2's ``mixle.analysis.valuation`` module (``NPVDistribution`` / ``monte_carlo_npv``)
has since landed and is imported normally below; C8's ``mixle_pde`` VOI hooks have not, so
``voi_dollars`` still best-effort imports that hook and falls back to a self-contained
posterior-refinement simulation over the frozen IC-1 :class:`~mixle.reason.posterior_protocol.Posterior`
when it is unavailable. ``real_option_value``'s ``npv_dist`` parameter stays duck-typed against
anything exposing a ``.mean`` -- the hard import below is for the type annotation and documentation
value only (previously a ``TYPE_CHECKING``-only forward reference, which left the name unresolvable to
``typing.get_type_hints()`` -- a real, if rarely exercised, runtime introspection break); no
``isinstance`` check was added, so any duck-typed object still works exactly as before.
"""

from __future__ import annotations

import copy
import operator
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mixle.analysis._evidence import require_delivered_draws
from mixle.analysis.valuation import NPVDistribution
from mixle.reason.posterior_protocol import Posterior

__all__ = [
    "OptionValue",
    "real_option_value",
    "GaussianObservationModel",
    "VoiEstimate",
    "voi_estimate",
    "voi_dollars",
    "VoiStoppingDecision",
    "voi_stopping_decision",
]

_KINDS = ("defer", "expand", "abandon")


class OptionValue(NamedTuple):
    """The priced option, alongside the exercise policy that generated it.

    ``value`` is the total value of holding the option (project value + optionality). ``exercise_boundary``
    is the per-time-step critical underlying value at which immediate exercise first becomes optimal
    (``nan`` at any step where it is never optimal to exercise). ``premium_over_npv`` is ``value`` minus
    the naive ``npv_dist.mean`` -- the dollar amount the flexibility to defer/expand/abandon is worth
    over just committing to the point-estimate NPV today.
    """

    value: float
    exercise_boundary: np.ndarray
    premium_over_npv: float


def _require_finite(value: Any, name: str, *, context: str = "real_option_value") -> float:
    """Reject NaN/+-inf, which silently poison downstream arithmetic rather than raising (MXR-080-0110):
    e.g. ``float("nan") < 0`` is ``False``, so a naive ``< 0`` guard lets a NaN control straight through
    to contaminate the whole lattice, and ``rate``/``expand_fraction`` have no such guard at all today."""
    fvalue = float(value)
    if not np.isfinite(fvalue):
        raise ValueError(f"{context}: {name} must be finite, got {value!r}")
    return fvalue


def _require_finite_int(value: Any, name: str, *, context: str = "real_option_value") -> int:
    """Require an actual non-Boolean integral scalar without coercion or truncation."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: {name} must be an exact integer and non-Boolean, got {value!r}")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{context}: {name} must be an exact integer and non-Boolean, got {value!r}") from exc
    return int(integer)


def _require_sample_count(value: Any, name: str, *, minimum: int = 1) -> int:
    count = _require_finite_int(value, name, context="voi_dollars")
    if count < minimum:
        qualifier = f">= {minimum}"
        raise ValueError(f"voi_dollars: {name} must be {qualifier}, got {count}")
    return count


def _require_finite_array(value: Any, name: str, *, context: str = "real_option_value") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context}: {name} is not representable as finite float64")
    return array


def _intrinsic(
    v: np.ndarray,
    kind: str,
    expand_fraction: float,
    expansion_cost: float = 0.0,
    salvage_value: float = 0.0,
) -> np.ndarray:
    """Exercise payoff at underlying value(s) ``v`` for the given option ``kind``."""
    if kind == "defer":
        # Option to wait, then invest only if the project is worth it: invest for max(V, 0), else walk.
        # V is already an NPV (net of the initial investment), so no separate exercise cost applies here.
        return np.maximum(v, 0.0)
    if kind == "abandon":
        # Already committed; option to abandon for `salvage_value` (default 0) rather than ride the NPV.
        return np.maximum(v, salvage_value)
    if kind == "expand":
        # Option to scale up capacity by `expand_fraction`, net of the expansion's own investment cost
        # (MXR-080-0110: without this, the bonus was free, so a positive project always "expanded" for
        # no cost and there was no actual exercise decision to price). Exercise only when the bonus net
        # of that cost is positive; otherwise just run the base (unexpanded) project for `v`.
        expansion_gain = np.maximum(expand_fraction * np.maximum(v, 0.0) - expansion_cost, 0.0)
        return v + expansion_gain
    raise ValueError(f"real_option_value: kind must be one of {_KINDS}, got {kind!r}")


def _exercise_is_economic(
    v: np.ndarray,
    kind: str,
    expand_fraction: float,
    expansion_cost: float,
    salvage_value: float,
) -> np.ndarray:
    """Whether immediate exercise means taking the named option action, not merely running the base."""
    if kind == "defer":
        return v > 0.0
    if kind == "abandon":
        return v < salvage_value
    if expand_fraction == 0.0:
        return np.zeros(v.shape, dtype=bool)
    return expand_fraction * np.maximum(v, 0.0) > expansion_cost


def _terminal_boundary(
    kind: str,
    expand_fraction: float,
    expansion_cost: float,
    salvage_value: float,
) -> float:
    if kind == "defer":
        return 0.0
    if kind == "abandon":
        return salvage_value
    if expand_fraction == 0.0:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        boundary = expansion_cost / expand_fraction
    return _require_finite(boundary, "terminal expansion boundary")


def real_option_value(
    npv_dist: NPVDistribution,
    *,
    volatility: float,
    horizon: int,
    kind: str = "defer",
    rate: float,
    n_steps: int | None = None,
    expand_fraction: float = 0.3,
    expansion_cost: float = 0.0,
    salvage_value: float = 0.0,
) -> OptionValue:
    """Price the defer/expand/abandon option on a project via an additive binomial lattice on NPV.

    The NPV process is modeled as a driftless arithmetic random walk started at ``npv_dist.mean``: each
    of ``n_steps`` (default: ``max(horizon, 1)``) steps up to ``horizon`` moves the value by
    ``+-h`` with ``h = volatility * scale * sqrt(dt)``, ``scale = abs(npv_dist.mean)`` (or ``1.0`` if the
    mean is exactly zero, so ``volatility`` is never degenerate) -- i.e. ``volatility`` is a *fractional*
    per-sqrt-period dispersion relative to the project's own scale, matching how the parameter is usually
    quoted (e.g. ``0.3`` ~ "30% dispersion"), while still letting the walk go negative the way a signed
    NPV must be able to. The risk-neutral measure is taken driftless (an NPV is already the discounted
    expectation of a self-financing project, so there is no further risk-neutral drift to add); ``rate``
    only discounts the continuation value, which is what makes immediate exercise optimal once volatility
    vanishes. At every node the holder may exercise (payoff via :func:`_intrinsic`, depending on ``kind``)
    or continue holding the option; the American value is the larger of the two, by backward induction
    from the horizon.

    Args:
        npv_dist: anything exposing ``.mean`` (a float) -- typically J2's ``NPVDistribution``. Only the
            mean is used; ``real_option_value`` prices the *option on top of* the point estimate, not a
            re-derivation of the distribution itself. Must be finite.
        volatility: fractional (per-sqrt-period) dispersion of NPV around its mean. Must be finite and
            ``>= 0``; ``0`` means no dispersion and the option collapses to immediate exercise of
            ``kind``'s own payoff on ``npv_dist.mean`` (:func:`_intrinsic`) -- ``max(npv_dist.mean, 0)``
            for ``"defer"``, ``max(npv_dist.mean, salvage_value)`` for ``"abandon"``, ``npv_dist.mean +
            max(expand_fraction * max(npv_dist.mean, 0) - expansion_cost, 0)`` for ``"expand"``.
        horizon: number of periods over which the option may be exercised. ``0`` means "decide now". Must
            be a non-Boolean integral scalar ``>= 0``.
        kind: one of ``"defer"``, ``"expand"``, ``"abandon"``.
        rate: per-period discount rate applied to the continuation value. Must be finite.
        n_steps: lattice steps (defaults to ``max(horizon, 1)``, i.e. one step per period). Must be a
            non-Boolean positive integral scalar when given explicitly.
        expand_fraction: for ``kind="expand"``, the fractional capacity bonus applied to a positive
            underlying value when the expansion is exercised. Must be finite and nonnegative.
        expansion_cost: for ``kind="expand"``, the investment cost of actually exercising the expansion,
            netted against ``expand_fraction``'s capacity bonus -- without it, a positive-NPV project
            always "expands" for free and there is no real exercise decision being priced (see
            :func:`_intrinsic`). Ignored for other kinds. Must be finite and nonnegative; default ``0.0`` (no cost,
            matching this function's behavior before ``expansion_cost`` existed).
        salvage_value: for ``kind="abandon"``, the value recovered by abandoning the project instead of
            riding out its NPV, replacing the previous hardcoded assumption of a total write-off. Ignored
            for other kinds. Must be finite; default ``0.0`` (total write-off, matching this function's
            behavior before ``salvage_value`` existed).

    Returns:
        An :class:`OptionValue`.
    """
    if kind not in _KINDS:
        raise ValueError(f"real_option_value: kind must be one of {_KINDS}, got {kind!r}")
    volatility = _require_finite(volatility, "volatility")
    if volatility < 0.0:
        raise ValueError("real_option_value: volatility must be non-negative")
    horizon = _require_finite_int(horizon, "horizon")
    if horizon < 0:
        raise ValueError("real_option_value: horizon must be non-negative")
    if n_steps is not None:
        n_steps = _require_finite_int(n_steps, "n_steps")
        if n_steps <= 0:
            # n_steps=0 with horizon>0 divides by zero below; n_steps<0 builds an EMPTY lattice and then
            # indexes boundary[n] == boundary[-1] on it, raising a confusing IndexError three lines later.
            # One clear error at the boundary instead of two different internal crashes downstream.
            raise ValueError(f"real_option_value: n_steps must be a positive integer, got {n_steps}")
    rate = _require_finite(rate, "rate")
    expand_fraction = _require_finite(expand_fraction, "expand_fraction")
    expansion_cost = _require_finite(expansion_cost, "expansion_cost")
    salvage_value = _require_finite(salvage_value, "salvage_value")
    if expand_fraction < 0.0:
        raise ValueError("real_option_value: expand_fraction must be nonnegative")
    if expansion_cost < 0.0:
        raise ValueError("real_option_value: expansion_cost must be nonnegative")

    npv_mean = _require_finite(npv_dist.mean, "npv_dist.mean")
    n = n_steps if n_steps is not None else max(horizon, 1)
    dt = float(horizon) / n if horizon > 0 else 0.0
    scale = abs(npv_mean) if npv_mean != 0.0 else 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        h = volatility * scale * float(np.sqrt(dt))
    h = _require_finite(h, "lattice step size")

    if h == 0.0:
        # No dispersion (or no time to disperse in): waiting has no upside, and discounting only makes
        # it worse, so the optimal policy is immediate exercise -- the SAME payoff _intrinsic gives
        # every node of the lattice below, not a hardcoded max(npv_mean, 0.0) (which happens to match
        # _intrinsic for "defer"/"abandon" but silently ignores "expand"'s expand_fraction bonus).
        intrinsic = _require_finite_array(
            _intrinsic(np.array([npv_mean]), kind, expand_fraction, expansion_cost, salvage_value),
            "immediate exercise value",
        )
        value = float(intrinsic[0])
        boundary = np.full(n + 1, np.nan)
        boundary[n] = _terminal_boundary(kind, expand_fraction, expansion_cost, salvage_value)
        premium = _require_finite(value - npv_mean, "premium_over_npv")
        return OptionValue(value=value, exercise_boundary=boundary, premium_over_npv=premium)

    with np.errstate(over="ignore", invalid="ignore"):
        disc = float(np.exp(-rate * dt))
    disc = _require_finite(disc, "discount factor")
    boundary = np.full(n + 1, np.nan)

    j = np.arange(n + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        v = npv_mean + (2 * j - n) * h  # j up-moves, (n - j) down-moves out of n steps
    v = _require_finite_array(v, "terminal lattice values")
    option = _require_finite_array(
        _intrinsic(v, kind, expand_fraction, expansion_cost, salvage_value),
        "terminal option values",
    )
    boundary[n] = _terminal_boundary(kind, expand_fraction, expansion_cost, salvage_value)

    for step in range(n - 1, -1, -1):
        j = np.arange(step + 1)
        with np.errstate(over="ignore", invalid="ignore"):
            v = npv_mean + (2 * j - step) * h
            continuation = disc * 0.5 * (option[1 : step + 2] + option[0 : step + 1])
        v = _require_finite_array(v, f"lattice values at step {step}")
        continuation = _require_finite_array(continuation, f"continuation values at step {step}")
        intrinsic = _require_finite_array(
            _intrinsic(v, kind, expand_fraction, expansion_cost, salvage_value),
            f"intrinsic values at step {step}",
        )
        choose_intrinsic = intrinsic > continuation
        exercise = choose_intrinsic & _exercise_is_economic(v, kind, expand_fraction, expansion_cost, salvage_value)
        option = np.where(choose_intrinsic, intrinsic, continuation)
        if np.any(exercise):
            exercised_v = v[exercise]
            boundary[step] = float(np.min(exercised_v)) if kind != "abandon" else float(np.max(exercised_v))

    value = float(option[0])
    premium = _require_finite(value - npv_mean, "premium_over_npv")
    return OptionValue(value=value, exercise_boundary=boundary, premium_over_npv=premium)


def _variance_reduction(posterior: Posterior, drill_info: dict) -> float:
    """Fraction of posterior variance a hypothetical drillhole is expected to remove, in ``[0, 1)``.

    Best-effort: if C8's ``mixle_pde.voi.expected_variance_reduction`` hook is importable and
    ``drill_info`` supplies the geometry/forward-operator it needs, use it. Otherwise fall back to a
    directly-supplied ``drill_info["variance_reduction"]`` (default ``0.5`` -- a generic "meaningfully
    informative" delineation hole).
    """
    candidate_geometry = drill_info.get("candidate_geometry")
    forward_op = drill_info.get("forward_op")
    if candidate_geometry is not None and forward_op is not None:
        try:
            from mixle_pde.voi import expected_variance_reduction  # C8 hook; mixle_pde is a soft dependency
        except ImportError:
            expected_variance_reduction = None
        if expected_variance_reduction is not None:
            reduction = float(
                expected_variance_reduction(
                    posterior,
                    candidate_geometry,
                    forward_op,
                    region=drill_info.get("region"),
                    cell_volumes=drill_info.get("cell_volumes"),
                )
            )
            return _validate_variance_reduction(reduction)
    return _validate_variance_reduction(float(drill_info.get("variance_reduction", 0.5)))


def _validate_variance_reduction(reduction: float) -> float:
    """Validate a declared fractional variance reduction is in ``[0, 1)`` without rewriting it."""
    if not np.isfinite(reduction):
        raise ValueError(f"voi_dollars: variance_reduction must be finite, got {reduction!r}")
    if not 0.0 <= reduction < 1.0:
        raise ValueError(f"voi_dollars: variance_reduction must be in [0, 1), got {reduction!r}")
    return reduction


class VoiEstimate(NamedTuple):
    """A Monte Carlo value-of-information estimate, reported honestly alongside its own sampling noise
    (MXR-080-0108: independent no-info/with-info draws made a should-be-zero difference noisy, and
    flooring that noise at zero upward-biased it; see :func:`voi_estimate`'s docstring for the fix).

    ``value`` is the point estimate, ``E[value | info] - E[value | no info]`` -- NOT floored at zero (see
    :func:`voi_estimate`'s docstring for why flooring a noisy estimate at zero systematically biases it
    upward, rather than merely rounding it). ``standard_error`` is the *paired-difference* Monte Carlo
    standard error of ``value`` across the ``n_outer`` hypothetical-information replicates (paired because
    each replicate's with-info and no-info evaluations share their underlying random draw -- see
    :func:`voi_estimate`). ``ci_low``/``ci_high`` are an approximate two-sided 95% confidence interval
    (``value +- 1.96 * standard_error``; a normal approximation, not exact for small ``n_outer``). When
    ``ci_low <= 0 <= ci_high``, ``value`` is statistically indistinguishable from "no additional value" --
    a more honest read than a bare point estimate that happens to be (noisily) positive or negative.

    ``method`` names which computation produced the estimate: ``"variance_rescaling_heuristic"`` (an
    explicit opt-in Gaussian-dispersion approximation, exact only when the posterior is
    close to Gaussian, unimodal, and unconstrained) or ``"gaussian_conjugate_evsi"`` (a genuine
    sample-then-condition preposterior computation under a declared linear-Gaussian ``observation_model``).
    """

    value: float
    standard_error: float
    ci_low: float
    ci_high: float
    method: str


_VOI_CI95_Z = 1.959963984540054  # two-sided 95% normal-approximation multiplier


def _voi_estimate_from_paired_diffs(diffs: np.ndarray, method: str) -> VoiEstimate:
    n = int(diffs.shape[0])
    if n < 2 or not np.all(np.isfinite(diffs)):
        raise ValueError("voi_dollars: paired decision differences must contain at least two finite replicates")
    value = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    ci_low, ci_high = value - _VOI_CI95_Z * se, value + _VOI_CI95_Z * se
    _require_finite_array([value, se, ci_low, ci_high], "VOI estimate", context="voi_dollars")
    return VoiEstimate(value=value, standard_error=se, ci_low=ci_low, ci_high=ci_high, method=method)


class GaussianObservationModel(NamedTuple):
    """A declared linear-Gaussian observation model: ``y = obs_matrix @ theta + noise``, ``noise ~
    N(0, obs_cov)``, for a latent ``theta`` distributed as the current posterior's ``(mean, cov)``.

    MXR-080-0109: the opt-in heuristic invents future-posterior centers/spreads by rescaling draws from
    the CURRENT posterior; it never samples an observation from a likelihood/forward model and conditions
    on it, so it is not a real Bayesian preposterior for non-Gaussian, multimodal, or constrained beliefs.
    This is the simplest well-defined experiment likelihood for which the preposterior -- the distribution
    of what TOMORROW's posterior could look like, before the observation is actually taken -- has a closed
    form: conjugate-Gaussian updating. Passing this to :func:`voi_dollars` / :func:`voi_estimate` (as
    ``observation_model=``) is what makes them compute a genuine expected value of sample information
    (EVSI): simulate a hypothetical ``y`` from theta's actual prior-predictive marginal, analytically
    condition the CURRENT posterior on that ``y`` to get a real future posterior, decide against it, and
    average over many simulated ``y`` -- rather than the opt-in heuristic's rescaling of today's draws
    (see :func:`voi_estimate`'s docstring for why that heuristic is not a real preposterior in general).

    ``obs_matrix`` (``H``, shape ``(k, d)``) and ``obs_cov`` (``R``, shape ``(k, k)``, symmetric positive
    definite) together say: a real experiment would return a ``k``-dimensional noisy linear readout of the
    ``d``-dimensional latent quantity the posterior is over (``d`` = ``posterior.mean``'s length). This
    path requires ``posterior.mean`` / ``posterior.cov`` to fully characterize the belief -- i.e. a
    genuinely Gaussian posterior -- since that is what makes the conjugate update exact; a dense
    ``posterior.cov`` is required (a matrix-free ``LinearOperator`` covariance is not supported here).
    ``obs_matrix = 0`` (or ``obs_cov`` large) declares a literally uninformative experiment -- VOI comes
    back exactly zero for that case, not just approximately (see :func:`voi_estimate`'s docstring).
    """

    obs_matrix: np.ndarray
    obs_cov: np.ndarray


_GAUSSIAN_SKEW_WARN = 0.75
_GAUSSIAN_EXCESS_KURTOSIS_WARN = 1.5


def _warn_if_not_gaussian_like(posterior: Posterior, rng: np.random.Generator, *, n_probe: int = 512) -> None:
    """Cheap empirical screen for the Gaussian/unimodal/unconstrained regime both VOI paths lean on.

    Not a proof: a finite probe sample can miss subtle non-Gaussianity, or (rarely) flag a legitimate
    Gaussian draw on an unlucky sample. It is a fast, honest diagnostic so a caller silently outside the
    assumed regime (multimodal, heavily skewed, bounded/constrained posteriors) gets a warning instead of
    a silently over-claimed textbook VOI number: both the opt-in variance-rescaling heuristic and the
    Gaussian-conjugate ``observation_model`` path assume the posterior's mean/cov fully characterize its
    shape, which is only true for a genuinely Gaussian belief. Thresholds are generous (roughly 7 standard
    errors at ``n_probe=512`` for true Gaussian data) to keep the false-positive rate low.

    Those thresholds are calibrated to ``n_probe``, so a posterior delivering fewer draws than requested
    makes the screen quietly weaker than that claim (MXR-080-1900). This now reports the short delivery
    instead of running a screen at an unknown resolution and then staying silent -- but it WARNS rather
    than raising, unlike the estimator paths below: this is a diagnostic, and the Gaussian-conjugate path
    never otherwise samples the posterior, so raising here would newly reject callers whose actual
    computation does not depend on a draw count at all.
    """
    diagnostic_rng = copy.deepcopy(rng)
    probe = np.asarray(posterior.samples(n_probe, diagnostic_rng), dtype=np.float64)
    if probe.ndim == 0 or probe.shape[0] != n_probe:
        delivered = probe.shape[0] if probe.ndim > 0 else 1
        warnings.warn(
            f"voi_dollars/voi_estimate: the Gaussian-regime screen requested {n_probe} probe draws and the "
            f"posterior delivered {delivered}, so the screen was NOT run. Draw no conclusion either way "
            "about whether this posterior is Gaussian-like: the skew/excess-kurtosis thresholds are "
            "calibrated to the requested probe size and mean nothing at another.",
            UserWarning,
            stacklevel=3,
        )
        return
    if probe.ndim == 1:
        probe = probe[:, None]
    sd = probe.std(axis=0, ddof=1)
    valid = sd > 1e-12
    if not np.any(valid):
        return
    centered = probe[:, valid] - probe[:, valid].mean(axis=0, keepdims=True)
    skew = np.mean(centered**3, axis=0) / sd[valid] ** 3
    excess_kurtosis = np.mean(centered**4, axis=0) / sd[valid] ** 4 - 3.0
    if np.any(np.abs(skew) > _GAUSSIAN_SKEW_WARN) or np.any(np.abs(excess_kurtosis) > _GAUSSIAN_EXCESS_KURTOSIS_WARN):
        warnings.warn(
            "voi_dollars/voi_estimate: posterior samples look non-Gaussian (skewness or excess kurtosis "
            "beyond this heuristic screen's threshold on a fresh probe sample). Both the default "
            "variance-rescaling heuristic and the Gaussian-conjugate observation_model path assume a "
            "Gaussian, unimodal, unconstrained posterior; treat this VOI estimate as approximate, or "
            "supply a non-Gaussian-aware decision procedure instead.",
            UserWarning,
            stacklevel=3,
        )


def _require_dense_finite_matrix(value: Any, name: str) -> np.ndarray:
    if isinstance(value, LinearOperator):
        raise TypeError(
            f"voi_dollars: {name} must be a dense array for the observation_model path (got a matrix-free "
            "LinearOperator) -- survey-scale field posteriors aren't supported by the closed-form "
            "Gaussian-conjugate path; use the explicitly opted-in variance-rescaling heuristic instead."
        )
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"voi_dollars: {name} must be finite, got values containing NaN/inf")
    return arr


def _require_covariance(
    value: Any,
    name: str,
    size: int,
    *,
    positive_definite: bool,
) -> np.ndarray:
    arr = _require_dense_finite_matrix(value, name)
    if arr.shape != (size, size):
        raise ValueError(f"voi_dollars: {name} must have shape ({size}, {size}), got {arr.shape}")
    scale = max(float(np.max(np.abs(arr), initial=0.0)), 1.0)
    tolerance = 1e-10 * scale
    if not np.allclose(arr, arr.T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"voi_dollars: {name} must be symmetric")
    symmetric = 0.5 * (arr + arr.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if positive_definite:
        if float(eigenvalues.min(initial=np.inf)) <= tolerance:
            raise ValueError(f"voi_dollars: {name} must be positive-definite")
    elif float(eigenvalues.min(initial=np.inf)) < -tolerance:
        raise ValueError(f"voi_dollars: {name} must be positive-semidefinite")
    return symmetric


def _safe_cholesky(mat: np.ndarray, *, name: str) -> np.ndarray:
    """Cholesky factor of a symmetric PSD matrix, with a small jitter retry for numerical near-singularity."""
    try:
        return np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        pass
    scale = float(np.trace(mat)) / max(mat.shape[0], 1)
    jitter = 1e-10 * (scale if scale > 0.0 else 1.0)
    try:
        return np.linalg.cholesky(mat + jitter * np.eye(mat.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"voi_dollars: {name} is not (numerically) symmetric positive-definite, even after a small "
            "jitter -- the Gaussian-conjugate observation_model path requires a genuine covariance."
        ) from exc


def _require_decision_value(value: Any, *, replicate: int, side: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"voi_dollars: decision_fn returned a non-scalar value for replicate {replicate} ({side})"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"voi_dollars: decision_fn returned a non-finite value for replicate {replicate} ({side})")
    return result


def _voi_estimate_gaussian_conjugate(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    observation_model: GaussianObservationModel,
    *,
    rng: np.random.Generator,
    n_outer: int,
    n_inner: int,
) -> VoiEstimate:
    """The genuine sample-then-condition EVSI path: see :class:`GaussianObservationModel`'s docstring."""
    method = "gaussian_conjugate_evsi"
    mean = np.asarray(posterior.mean, dtype=np.float64)
    if mean.ndim != 1 or mean.size == 0 or not np.all(np.isfinite(mean)):
        raise ValueError("voi_dollars: posterior.mean must be a finite non-empty one-dimensional array")
    d = mean.shape[0]
    cov = _require_covariance(posterior.cov, "posterior.cov", d, positive_definite=False)

    obs_matrix = _require_dense_finite_matrix(observation_model.obs_matrix, "observation_model.obs_matrix")
    if obs_matrix.ndim != 2 or obs_matrix.shape[1] != d:
        raise ValueError(f"voi_dollars: observation_model.obs_matrix must have shape (k, {d}), got {obs_matrix.shape}")
    k = obs_matrix.shape[0]
    if k == 0:
        raise ValueError("voi_dollars: observation_model.obs_matrix must contain at least one observation row")
    obs_cov = _require_covariance(
        observation_model.obs_cov,
        "observation_model.obs_cov",
        k,
        positive_definite=True,
    )

    _warn_if_not_gaussian_like(posterior, rng)

    # Prior-predictive covariance of the hypothetical observation, the Kalman gain, and the post-observation
    # covariance -- all independent of the observed VALUE, a standard closed-form property of conjugate-
    # Gaussian updating that lets these be computed once rather than per simulated observation.
    mean_y = obs_matrix @ mean
    cov_y = _require_covariance(
        obs_matrix @ cov @ obs_matrix.T + obs_cov,
        "the prior-predictive observation covariance",
        k,
        positive_definite=True,
    )
    gain = np.linalg.solve(cov_y, obs_matrix @ cov).T  # (d, k); solved rather than inverting cov_y directly
    post_cov = _require_covariance(
        cov - gain @ obs_matrix @ cov,
        "the post-observation covariance",
        d,
        positive_definite=False,
    )

    l_prior = _safe_cholesky(cov, name="posterior.cov")
    l_post = _safe_cholesky(post_cov, name="the post-observation covariance")
    l_y = _safe_cholesky(cov_y, name="the prior-predictive observation covariance")

    y_draws = mean_y + rng.standard_normal((n_outer, k)) @ l_y.T

    diffs = np.empty(n_outer, dtype=np.float64)
    for i in range(n_outer):
        # Shared innovations z: at obs_matrix = 0 (or obs_cov -> large), gain = 0 and post_cov = cov
        # EXACTLY (both independent of y_draws[i]), so with_info_samples == no_info_samples bit-for-bit
        # and diffs[i] == 0.0 exactly -- the same common-random-numbers guarantee as the heuristic path.
        z = rng.standard_normal((n_inner, d))
        no_info_samples = mean + z @ l_prior.T
        post_mean_i = mean + gain @ (y_draws[i] - mean_y)
        with_info_samples = post_mean_i + z @ l_post.T
        with_info = _require_decision_value(decision_fn(with_info_samples), replicate=i, side="with information")
        no_info = _require_decision_value(decision_fn(no_info_samples), replicate=i, side="without information")
        diffs[i] = with_info - no_info

    return _voi_estimate_from_paired_diffs(diffs, method)


def voi_estimate(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
    observation_model: GaussianObservationModel | None = None,
) -> VoiEstimate:
    """Value of information, honestly: the point estimate together with its own Monte Carlo uncertainty.

    ``decision_fn`` maps a set of posterior draws (an ``(n, d)`` array, physical units) to the dollar
    value of the *single* best decision made using that belief state -- e.g. a risk-neutral go/no-go
    choice is ``max(samples.mean(), 0)``, not an average of a per-draw payoff: the latter implicitly
    assumes the realization is already known, which is exactly the perfect-information case this function
    is pricing the *gap* to, not the belief itself.

    Two methods are available:

    * ``observation_model=None`` together with explicit
      ``drill_info["method"] == "variance_rescaling_heuristic"``: a pre-posterior
      Monte Carlo built on the law of total variance, NOT real Bayesian conditioning on a simulated
      observation. Today's posterior variance splits into "where the post-drill posterior will be
      centered" (unknown until the drillhole is actually put in) and "how much spread remains once it
      lands there." If the drillhole is expected to remove a fraction ``r`` of variance
      (:func:`_variance_reduction`), then for ``n_outer`` hypothetical drill outcomes a center is drawn
      with standard deviation scaled by ``sqrt(r)`` around today's posterior mean (how much the belief
      could plausibly shift), and a *shared* base draw (see "Common random numbers" below) is re-centered
      there and rescaled by ``sqrt(1 - r)`` for the remaining spread. This INVENTS future-belief "centers"
      by rescaling *today's* draws rather than drawing them from an actual forward/likelihood model, so it
      is accurate only when the posterior is (close to) Gaussian, unimodal, and unconstrained -- outside
      that regime it is flagged with a :class:`UserWarning` (see :func:`_warn_if_not_gaussian_like`), not
      silently trusted.
    * ``observation_model=`` a :class:`GaussianObservationModel` (``method="gaussian_conjugate_evsi"``): a
      genuine sample-then-condition preposterior. A hypothetical observation ``y`` is simulated from the
      latent's real prior-predictive marginal (``y = H @ theta + noise`` for ``theta ~ N(posterior.mean,
      posterior.cov)``), the CURRENT posterior is analytically conditioned on that ``y`` via closed-form
      conjugate-Gaussian updating to get a genuine future posterior, and the decision is re-made against
      it -- averaged over many simulated ``y`` for ``E[value | info]``. This is the textbook EVSI
      construction for the declared-likelihood case, but it still requires the PRIOR (``posterior.mean`` /
      ``posterior.cov``) to be genuinely Gaussian for the conjugate update to be exact -- also screened by
      the same warning.

    Common random numbers (MXR-080-0108): each of the ``n_outer`` replicates draws ONE base/innovation
    sample and uses it for BOTH the no-info decision and the with-info (refined) decision for that
    replicate, differing only by a deterministic transform (the heuristic's re-centering/rescaling, or the
    conjugate update's affine map) -- not independent draws for the two sides. This matters when there is
    truly no information to gain (``r = 0`` for the heuristic; ``obs_matrix = 0`` or ``obs_cov``
    arbitrarily large for the declared-likelihood path): with independent draws, the no-info and with-info
    values would be two separately noisy estimates of the *same* quantity, so their difference would be
    noise centered at zero; flooring that noise at zero (the previous behavior) kept only its positive
    half, upward-biasing the reported VOI (a standard-normal fake posterior with a go/no-go decision
    spuriously returned positive VOI at zero information for most tested seeds). With a shared draw, the
    with-info value is bit-for-bit identical to the no-info value whenever there is truly no information
    (checked directly for the heuristic via an analytic short-circuit below, not just left to fall out of
    the Monte Carlo -- and provably exact by construction for the declared-likelihood path too), and the
    two stay correlated -- hence lower-variance in their difference -- away from that boundary as well.

    Args:
        posterior: the current (pre-drill) belief, satisfying IC-1's ``Posterior`` protocol.
        decision_fn: belief state (``(n, d)`` draws) -> expected dollar value of the best decision.
        drill_info: describes the hypothetical drillhole. Recognized keys: ``method`` (must explicitly be
            ``"variance_rescaling_heuristic"`` when no ``observation_model`` is supplied),
            ``variance_reduction`` (direct
            fractional variance reduction, default ``0.5``), or ``candidate_geometry`` + ``forward_op``
            (+ optional ``region`` / ``cell_volumes``) to route through the C8 VOI hook when available;
            ``n_outer_samples`` / ``n_inner_samples`` override the Monte Carlo sample counts. Ignored
            (aside from the sample-count overrides) when ``observation_model`` is given.
        rng: seeded random generator for reproducibility.
        n_outer: number of hypothetical-information replicates (overridden by ``drill_info``).
        n_inner: posterior draws per replicate (overridden by ``drill_info``).
        observation_model: when given, a declared :class:`GaussianObservationModel` routes the computation
            through the genuine ``"gaussian_conjugate_evsi"`` path instead of the opt-in heuristic; see
            above. Requires a dense (non-``LinearOperator``) ``posterior.cov``.

    Returns:
        A :class:`VoiEstimate`: the (unfloored) point estimate plus its Monte Carlo uncertainty.
    """
    n_outer = _require_sample_count(drill_info.get("n_outer_samples", n_outer), "n_outer_samples", minimum=2)
    n_inner = _require_sample_count(drill_info.get("n_inner_samples", n_inner), "n_inner_samples")

    if observation_model is not None:
        return _voi_estimate_gaussian_conjugate(
            posterior, decision_fn, observation_model, rng=rng, n_outer=n_outer, n_inner=n_inner
        )

    method = drill_info.get("method")
    if method != "variance_rescaling_heuristic":
        raise ValueError(
            "voi_dollars: a declared observation_model is required for Bayesian EVSI; "
            "the approximation is available only by explicitly setting "
            "drill_info['method']='variance_rescaling_heuristic'"
        )
    reduction = _variance_reduction(posterior, drill_info)
    if reduction == 0.0:
        # Analytic equality at exactly zero reduction, not just an empirically-near-zero estimate: by
        # this heuristic's own construction (inner_scale = sqrt(1 - r) = 1, center_scale = sqrt(r) = 0
        # below), the with-info belief IS today's belief, so the two decisions coincide exactly -- no
        # Monte Carlo needed (and, since no regime approximation is actually being exercised in this
        # trivial case, no need for the Gaussian-regime warning either), and none of its sampling noise
        # to worry about flooring.
        return VoiEstimate(value=0.0, standard_error=0.0, ci_low=0.0, ci_high=0.0, method=method)

    _warn_if_not_gaussian_like(posterior, rng)

    center_scale = float(np.sqrt(reduction))
    inner_scale = float(np.sqrt(1.0 - reduction))
    mean = np.asarray(posterior.mean, dtype=np.float64)

    # Exact posterior-delivery receipts on BOTH sampling axes (MXR-080-1900), the same check
    # `valuation._draw_grade_per_period` already applies to `monte_carlo_npv` (MXR-080-0118).
    # The outer draw used to fail only incidentally -- `centers[i]` eventually raised `IndexError`,
    # naming no cause -- and the INNER draw did not fail at all: `decision_fn(base)` reduces however
    # many draws it is handed to a single float, and `centers[i] + inner_scale * (base - mean)`
    # broadcasts cleanly at any row count, so a thinned/filtered posterior produced a VOI in dollars
    # computed on a fraction of the requested inner evidence and reported it as the full estimate.
    # `VoiEstimate` carries no draw count, so nothing downstream could have detected it either.
    outer_draws = require_delivered_draws(
        posterior.samples(n_outer, rng), n_outer, what="voi_estimate: posterior.samples(n_outer, rng)"
    )
    centers = mean + center_scale * (outer_draws - mean)
    diffs = np.empty(n_outer, dtype=np.float64)
    for i in range(n_outer):
        # shared by both sides of this replicate -- see docstring
        base = require_delivered_draws(
            posterior.samples(n_inner, rng),
            n_inner,
            what=f"voi_estimate: posterior.samples(n_inner, rng) at replicate {i}",
        )
        no_info = _require_decision_value(decision_fn(base), replicate=i, side="without information")
        refined = centers[i] + inner_scale * (base - mean)
        with_info = _require_decision_value(decision_fn(refined), replicate=i, side="with information")
        diffs[i] = with_info - no_info

    return _voi_estimate_from_paired_diffs(diffs, method)


def voi_dollars(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
    observation_model: GaussianObservationModel | None = None,
) -> float:
    """The value-of-information point estimate, in dollars: ``E[value | info] - E[value | no info]``.

    A thin convenience wrapper around :func:`voi_estimate` for callers that only want the point estimate;
    see its docstring for the full method (the opt-in heuristic vs. the genuine ``observation_model``
    EVSI path, common random numbers, the zero-information analytic equality, and what ``method`` the
    estimate used) and for how to also get the Monte Carlo standard error / CI.

    Args:
        posterior, decision_fn, drill_info, rng, n_outer, n_inner, observation_model: forwarded to
            :func:`voi_estimate` unchanged.

    Returns:
        The value-of-information point estimate, in dollars. NOT floored at zero -- see
        :func:`voi_estimate`'s docstring for why a floor is dishonest, not merely conservative; this can
        be a small negative number when the true VOI is at or near zero, which is the honest reflection
        of Monte Carlo noise, not a defect. Use :func:`voi_estimate` if you need to tell that case apart
        from a real, larger difference.
    """
    return voi_estimate(
        posterior,
        decision_fn,
        drill_info,
        rng=rng,
        n_outer=n_outer,
        n_inner=n_inner,
        observation_model=observation_model,
    ).value


class VoiStoppingDecision(NamedTuple):
    """A real decision-theoretic answer to "should we sample again?": compare the value of one more
    sample against its cost, rather than an arbitrary uncertainty threshold picked by hand.

    ``standard_error`` is the underlying :class:`VoiEstimate`'s Monte Carlo standard error, carried
    through so a caller can tell a close-to-the-cost ``net_value`` (statistical noise) apart from a real,
    larger gap -- the same honesty :func:`voi_estimate` reports at the VOI level (MXR-080-0108/0110).
    """

    voi_dollars: float
    sample_cost: float
    net_value: float
    keep_sampling: bool
    standard_error: float


def voi_stopping_decision(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    sample_cost: float,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
    observation_model: GaussianObservationModel | None = None,
) -> VoiStoppingDecision:
    """Should the next sample (drillhole, monitoring well, survey station, ...) actually be taken?

    A real, principled alternative to picking an uncertainty threshold by hand and telling an LLM
    loop-controller to stop when it's cleared (the pattern used, and flagged as a real gap, in
    ``experiments/adaptive-groundwater-monitoring`` and ``experiments/adaptive-gravity-survey-design``):
    sample again iff :func:`voi_dollars` -- the actual expected dollar value the next sample would add
    to the decision -- exceeds what that sample costs. As uncertainty tightens, ``voi_dollars`` shrinks
    toward zero (there's less left to learn that would change the decision), so this converges to a
    stopping rule on its own without any separately-chosen threshold; the only free parameter is
    ``sample_cost``, which is an actual real-world number (drilling/survey cost) rather than an
    arbitrary uncertainty width.

    Args:
        posterior, decision_fn, drill_info, rng, n_outer, n_inner, observation_model: forwarded to
            :func:`voi_estimate` unchanged -- see its docstring for what each means.
        sample_cost: the real dollar cost of taking the next sample. Must be finite (MXR-080-0110: an
            unvalidated NaN cost previously compared as ``False`` against everything, silently forcing
            ``keep_sampling=False`` regardless of the actual VOI).

    Returns:
        A :class:`VoiStoppingDecision`: the computed VOI (with its standard error), the cost it was
        compared against, their difference, and whether sampling should continue (``voi_dollars >
        sample_cost``).
    """
    sample_cost = _require_finite(sample_cost, "sample_cost", context="voi_stopping_decision")
    est = voi_estimate(
        posterior,
        decision_fn,
        drill_info,
        rng=rng,
        n_outer=n_outer,
        n_inner=n_inner,
        observation_model=observation_model,
    )
    return VoiStoppingDecision(
        voi_dollars=est.value,
        sample_cost=sample_cost,
        net_value=est.value - sample_cost,
        keep_sampling=est.value > sample_cost,
        standard_error=est.standard_error,
    )
