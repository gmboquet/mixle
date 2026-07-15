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
    ``max(NPV, 0)`` as volatility falls to zero (no dispersion, nothing to wait for).
  * :func:`voi_dollars` -- the dollar value of a piece of information (e.g. a delineation drillhole)
    computed the textbook way: the expected value of the best decision *with* that information, minus
    the expected value of the best decision *without* it. A hypothetical drillhole's effect on the
    posterior is summarized as a fractional variance reduction (either supplied directly via
    ``drill_info["variance_reduction"]``, or -- when available -- via C8's
    ``mixle_pde.voi.expected_variance_reduction`` hook); the pre-posterior simulation splits the prior
    variance, by the law of total variance, into "where the future posterior will be centered" and "how
    much uncertainty remains once it lands there." A rational decision-maker never loses from more
    information, so the estimate is floored at zero.

Repo-boundary note (see this task's PR description): as of this PR neither J2's
``mixle.analysis.valuation`` module (``NPVDistribution`` / ``monte_carlo_npv``) nor C8's ``mixle_pde``
VOI hooks had landed on ``release/0.8.0`` -- J2 and C8 are this task's stated dependencies but are
themselves still in flight. Consistent with how J4's ``valuation.py`` already handled the same kind of
repo-boundary gap (see that module's docstring), this module treats both as soft dependencies: the
``npv_dist`` parameter is duck-typed against anything exposing a ``.mean`` (the ``"NPVDistribution"``
annotation is a forward reference, exactly as written in the frozen work order, never imported at
runtime), and ``voi_dollars`` best-effort imports the C8 hook and falls back to a self-contained
posterior-refinement simulation over the frozen IC-1 :class:`~mixle.reason.posterior_protocol.Posterior`
when it is unavailable. Nothing here blocks on either dependency merging first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from mixle.reason.posterior_protocol import Posterior

if TYPE_CHECKING:
    from mixle.analysis.valuation import NPVDistribution

__all__ = ["OptionValue", "real_option_value", "voi_dollars", "VoiStoppingDecision", "voi_stopping_decision"]

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


def _intrinsic(v: np.ndarray, kind: str, expand_fraction: float) -> np.ndarray:
    """Exercise payoff at underlying value(s) ``v`` for the given option ``kind``."""
    if kind == "defer":
        # Option to wait, then invest only if the project is worth it: invest for max(V, 0), else walk.
        return np.maximum(v, 0.0)
    if kind == "abandon":
        # Already committed; option to abandon for salvage (assumed 0) rather than ride a negative NPV.
        return np.maximum(v, 0.0)
    if kind == "expand":
        # Option to scale up capacity by `expand_fraction` whenever doing so is profitable.
        return v + expand_fraction * np.maximum(v, 0.0)
    raise ValueError(f"real_option_value: kind must be one of {_KINDS}, got {kind!r}")


def real_option_value(
    npv_dist: NPVDistribution,
    *,
    volatility: float,
    horizon: int,
    kind: str = "defer",
    rate: float,
    n_steps: int | None = None,
    expand_fraction: float = 0.3,
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
            re-derivation of the distribution itself.
        volatility: fractional (per-sqrt-period) dispersion of NPV around its mean. Must be ``>= 0``;
            ``0`` means no dispersion and the option collapses to the naive ``max(npv_dist.mean, 0)``.
        horizon: number of periods over which the option may be exercised. ``0`` means "decide now".
        kind: one of ``"defer"``, ``"expand"``, ``"abandon"``.
        rate: per-period discount rate applied to the continuation value.
        n_steps: lattice steps (defaults to ``max(horizon, 1)``, i.e. one step per period).
        expand_fraction: for ``kind="expand"``, the fractional capacity bonus applied to a positive
            underlying value when the expansion is exercised.

    Returns:
        An :class:`OptionValue`.
    """
    if kind not in _KINDS:
        raise ValueError(f"real_option_value: kind must be one of {_KINDS}, got {kind!r}")
    if volatility < 0.0:
        raise ValueError("real_option_value: volatility must be non-negative")
    if horizon < 0:
        raise ValueError("real_option_value: horizon must be non-negative")

    npv_mean = float(npv_dist.mean)
    n = int(n_steps) if n_steps is not None else max(int(horizon), 1)
    dt = float(horizon) / n if horizon > 0 else 0.0
    scale = abs(npv_mean) if npv_mean != 0.0 else 1.0
    h = volatility * scale * float(np.sqrt(dt))

    if h == 0.0:
        # No dispersion (or no time to disperse in): waiting has no upside, and discounting only makes it
        # worse, so the optimal policy is "exercise now if positive, else never" -- the naive NPV floor.
        value = max(npv_mean, 0.0)
        boundary = np.full(n + 1, np.nan)
        return OptionValue(value=value, exercise_boundary=boundary, premium_over_npv=value - npv_mean)

    disc = float(np.exp(-rate * dt))
    boundary = np.full(n + 1, np.nan)

    j = np.arange(n + 1)
    v = npv_mean + (2 * j - n) * h  # j up-moves, (n - j) down-moves out of n steps
    option = _intrinsic(v, kind, expand_fraction)
    boundary[n] = 0.0  # at maturity, exercise iff the underlying is non-negative (kind-independent here)

    for step in range(n - 1, -1, -1):
        j = np.arange(step + 1)
        v = npv_mean + (2 * j - step) * h
        continuation = disc * 0.5 * (option[1 : step + 2] + option[0 : step + 1])
        intrinsic = _intrinsic(v, kind, expand_fraction)
        exercise = intrinsic > continuation
        option = np.where(exercise, intrinsic, continuation)
        if np.any(exercise):
            exercised_v = v[exercise]
            boundary[step] = float(np.min(exercised_v)) if kind != "abandon" else float(np.max(exercised_v))

    value = float(option[0])
    return OptionValue(value=value, exercise_boundary=boundary, premium_over_npv=value - npv_mean)


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
            return min(max(reduction, 0.0), 1.0 - 1e-9)
    return min(max(float(drill_info.get("variance_reduction", 0.5)), 0.0), 1.0 - 1e-9)


def voi_dollars(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
) -> float:
    """Value of information, in dollars: ``E[value | drill info] - E[value | no info]``.

    ``decision_fn`` maps a set of posterior draws (an ``(n, d)`` array, physical units) to the dollar
    value of the *single* best decision made using that belief state -- e.g. a risk-neutral go/no-go
    choice is ``max(samples.mean(), 0)``, not an average of a per-draw payoff: the latter implicitly
    assumes the realization is already known, which is exactly the perfect-information case this function
    is pricing the *gap* to, not the belief itself. The no-info value is ``decision_fn`` applied to
    today's posterior draws.

    The with-info value is a pre-posterior Monte Carlo built on the law of total variance: today's
    posterior variance splits into "where the post-drill posterior will be centered" (unknown until the
    drillhole is actually put in) and "how much spread remains once it lands there." If the drillhole is
    expected to remove a fraction ``r`` of variance (:func:`_variance_reduction`), then for ``n_outer``
    hypothetical drill outcomes: a center is drawn with standard deviation scaled by ``sqrt(r)`` around
    today's posterior mean (how much the belief could plausibly shift), and for each center, ``n_inner``
    refined-posterior draws are formed with the remaining spread scaled by ``sqrt(1 - r)``. Re-deciding
    with ``decision_fn`` on each refined belief and averaging over the (unknown, before drilling) outcome
    gives ``E[value | drill info]``; as ``r -> 0`` this construction degenerates back to exactly today's
    posterior (no information, no premium), and as ``r -> 1`` the center draws recover the full prior
    spread while the refined posterior collapses to a point (perfect information).

    A perfectly rational decision-maker is never made worse off by more information, so the result is
    floored at ``0.0``.

    Args:
        posterior: the current (pre-drill) belief, satisfying IC-1's ``Posterior`` protocol.
        decision_fn: belief state (``(n, d)`` draws) -> expected dollar value of the best decision.
        drill_info: describes the hypothetical drillhole. Recognized keys: ``variance_reduction`` (direct
            fractional variance reduction, default ``0.5``), or ``candidate_geometry`` + ``forward_op``
            (+ optional ``region`` / ``cell_volumes``) to route through the C8 VOI hook when available;
            ``n_outer_samples`` / ``n_inner_samples`` override the Monte Carlo sample counts.
        rng: seeded random generator for reproducibility.

    Returns:
        The value of information in dollars, ``>= 0``.
    """
    n_outer = int(drill_info.get("n_outer_samples", n_outer))
    n_inner = int(drill_info.get("n_inner_samples", n_inner))

    value_no_info = float(decision_fn(posterior.samples(n_inner, rng)))

    reduction = _variance_reduction(posterior, drill_info)
    center_scale = float(np.sqrt(reduction))
    inner_scale = float(np.sqrt(1.0 - reduction))
    mean = np.asarray(posterior.mean, dtype=np.float64)

    centers = mean + center_scale * (posterior.samples(n_outer, rng) - mean)
    values_with_info = np.empty(n_outer, dtype=np.float64)
    for i in range(n_outer):
        base = posterior.samples(n_inner, rng)
        refined = centers[i] + inner_scale * (base - mean)
        values_with_info[i] = decision_fn(refined)
    value_with_info = float(np.mean(values_with_info))

    return max(value_with_info - value_no_info, 0.0)


class VoiStoppingDecision(NamedTuple):
    """A real decision-theoretic answer to "should we sample again?": compare the value of one more
    sample against its cost, rather than an arbitrary uncertainty threshold picked by hand."""

    voi_dollars: float
    sample_cost: float
    net_value: float
    keep_sampling: bool


def voi_stopping_decision(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    sample_cost: float,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
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
        posterior, decision_fn, drill_info, rng, n_outer, n_inner: forwarded to :func:`voi_dollars`
            unchanged -- see its docstring for what each means.
        sample_cost: the real dollar cost of taking the next sample.

    Returns:
        A :class:`VoiStoppingDecision`: the computed VOI, the cost it was compared against, their
        difference, and whether sampling should continue (``voi_dollars > sample_cost``).
    """
    voi = voi_dollars(posterior, decision_fn, drill_info, rng=rng, n_outer=n_outer, n_inner=n_inner)
    return VoiStoppingDecision(
        voi_dollars=voi,
        sample_cost=float(sample_cost),
        net_value=voi - float(sample_cost),
        keep_sampling=voi > float(sample_cost),
    )
