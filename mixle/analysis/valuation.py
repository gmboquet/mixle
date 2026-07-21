"""Mine economics: parametric cost curves and capex/opex roll-up (work-plan Sec.7-J, J4).

J's objective function needs a `$/t` cost that is a function of *where* the ore is (depth), *what* it
is (grade), and *how fast* it is mined (throughput) before J2 can turn price paths + posterior grade
draws into an NPV distribution, and before H's block-level optimizers (`mixle.stochastic_opt`,
`mixle.relations`) have a `block_cost` to subtract from revenue:

  * :func:`cost_curve` -- parametric mining + processing cost in `$/t`, monotone increasing in haul/
    pumping depth, complexity-adjusted by grade, and shaped like the classic economies-of-scale curve
    in throughput: cheapest at the plant's design capacity, more expensive both under- and
    over-utilized.
  * :func:`capex_opex` -- rolls a period-by-period mine plan (tonnage, depth, grade, throughput, plus
    any lumpy capital spend) up into total capital and total operating cost, via :func:`cost_curve`.

This module is created by J4 (Wave 1) and extended here by J2 (Wave 2) with :func:`monte_carlo_npv` /
`NPVDistribution`, the risk-neutral expected-DCF distribution H4's objective is priced against:

  * :func:`monte_carlo_npv` -- Monte-Carlo DCF over ``posterior.samples(n, rng) x price scenarios``
    (DR-ALG J2): grade draws come straight off an IC-1 `Posterior` (frozen `mixle.reason
    .posterior_protocol.Posterior`), price scenarios come from J1's ``PriceForecast.paths`` (or any
    array-like of per-period price paths), and per-period tonnage/capex come off ``schedule``. Returns
    the full `NPVDistribution` (mean, P10/P50/P90, and a grade-vs-price variance decomposition) --
    a distribution, never a single point estimate.

Repo-boundary note (see the PR body for the full explanation): as of this PR, J1
(`mixle.inference.price_forecast.forecast_price`, whose `PriceForecast.paths` this task's own Algorithm
text names as the price-scenario source) and A5 (`mixle_pde.decision_quantities`, IC-8's calibrated
net-pay/tonnage quantities) had not landed. `monte_carlo_npv` therefore consumes ``price_paths`` and
``schedule`` structurally (an ``(n_paths, n_periods)`` array-like and a tonnage/capex-per-period
mapping or array-like, respectively) rather than importing either concrete module by name -- exactly
what its frozen Public API signature already commits to (``price_paths: Any``, ``schedule: Any``). A
real `PriceForecast.paths` / A5 decision-quantity slots in without any change to this function once
those tasks land, since both already produce array-likes of that shape.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from mixle.reason.posterior_protocol import Posterior

__all__ = ["NPVDistribution", "capex_opex", "cost_curve", "monte_carlo_npv"]

# Default parameters, used for any key the caller's `params` dict omits. Chosen to be dimensionally
# sane toy defaults ($/t and $/t-per-metre in the low single digits), not a claim about any real mine.
_DEFAULTS: dict[str, float] = {
    "base_cost": 0.0,  # $/t floor: cost at zero depth, reference grade, design-capacity throughput
    "haul_cost_per_m": 0.0,  # $/t per metre of depth: haulage + dewatering/pumping, linear in depth
    "grade_complexity_coef": 0.0,  # $/t, scales the 1/grade metallurgical-complexity penalty
    "throughput_scale_coef": 0.0,  # $/t, scales the (Q/Q* - 1)^2 economies-of-scale penalty
    "design_capacity": 1.0,  # Q*: throughput at which the economies-of-scale term is zero
    "capex_fixed": 0.0,  # $, one-off development/construction capital independent of tonnage
    "capex_per_tonne": 0.0,  # $/t, sustaining capital that scales with total tonnage mined
}


def _param(params: dict, key: str) -> float:
    return float(params[key]) if key in params else _DEFAULTS[key]


def cost_curve(depth: Any, grade: Any, throughput: Any, *, params: dict) -> np.ndarray:
    """Parametric mining + processing cost in `$/t`, as a function of depth, grade, and throughput.

    ``depth``, ``grade``, and ``throughput`` are broadcastable array-likes (one entry per block or per
    scheduling period; scalars broadcast against the others). ``params`` recognizes (all optional,
    defaulting to zero/no-effect):

    - ``base_cost``: `$/t` floor cost.
    - ``haul_cost_per_m``: `$/t` per metre of depth -- haulage and pumping/dewatering cost, modeled as
      linear in depth, so the curve is strictly increasing in ``depth`` whenever this is positive.
    - ``grade_complexity_coef``: `$/t` scale of a ``1 / grade`` metallurgical-complexity penalty --
      lower-grade ore needs proportionally more material handled and processed per unit of recovered
      metal, so this term falls as grade rises.
    - ``throughput_scale_coef`` / ``design_capacity``: the plant has one throughput, ``design_capacity``
      (``Q*``), at which fixed costs are spread most efficiently; cost rises quadratically away from
      it in *either* direction -- ``throughput_scale_coef * ((Q - Q*) / Q*) ** 2`` -- capturing both
      under-utilized fixed-cost drag below ``Q*`` and overtime/expediting/accelerated-wear cost above
      it (the classic "decreasing then rising past design capacity" U-shaped average-cost curve).

    Returns the elementwise `$/t` cost, broadcast to the common shape of the three inputs.
    """
    d = np.asarray(depth, dtype=np.float64)
    g = np.asarray(grade, dtype=np.float64)
    q = np.asarray(throughput, dtype=np.float64)

    if np.any(g <= 0.0):
        raise ValueError("cost_curve: grade must be strictly positive (used as a 1/grade complexity term)")
    q_star = _param(params, "design_capacity")
    if q_star <= 0.0:
        raise ValueError("cost_curve: params['design_capacity'] must be strictly positive")
    if np.any(q <= 0.0):
        raise ValueError("cost_curve: throughput must be strictly positive")

    base = _param(params, "base_cost")
    haul = _param(params, "haul_cost_per_m") * d
    complexity = _param(params, "grade_complexity_coef") / g
    scale = _param(params, "throughput_scale_coef") * ((q - q_star) / q_star) ** 2

    return base + haul + complexity + scale


def _plan_get(plan: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off ``plan``, whether it is a mapping (dict) or an attribute-bearing object."""
    if isinstance(plan, dict):
        return plan.get(key, default)
    return getattr(plan, key, default)


def capex_opex(plan: Any, *, params: dict) -> tuple[float, float]:
    """Roll a mine plan's tonnage/depth/grade/throughput profile up into (total capex, total opex).

    ``plan`` is a mapping or attribute-bearing object exposing, per scheduling period:

    - ``tonnage``: array-like, tonnes mined/processed each period (required).
    - ``depth``, ``grade``, ``throughput``: array-likes (or scalars, broadcast against ``tonnage``)
      fed to :func:`cost_curve` to get each period's `$/t`.
    - ``capex_schedule`` (optional): array-like of lumpy capital spend per period (e.g. pre-strip,
      plant construction, fleet purchases); summed into total capex on top of the params below.

    ``params`` is passed through to :func:`cost_curve` for the opex side, plus two capex-only keys:
    ``capex_fixed`` (one-off, tonnage-independent capital) and ``capex_per_tonne`` (sustaining capital
    that scales with total tonnage mined over the plan).

    Total opex is ``sum(tonnage * cost_curve(depth, grade, throughput, params=params))``; total capex is
    ``capex_fixed + capex_per_tonne * sum(tonnage) + sum(capex_schedule)``. Returns ``(capex, opex)``,
    both plain floats -- the totals :func:`monte_carlo_npv` (J2) discounts into a DCF, and the same
    `$/t` curve this function calls is what feeds `block_cost` for H's optimizers.
    """
    tonnage = np.asarray(_plan_get(plan, "tonnage"), dtype=np.float64)
    depth = _plan_get(plan, "depth")
    grade = _plan_get(plan, "grade")
    throughput = _plan_get(plan, "throughput")

    per_period_cost = cost_curve(depth, grade, throughput, params=params)
    opex_total = float(np.sum(tonnage * per_period_cost))

    total_tonnage = float(np.sum(tonnage))
    capex_total = _param(params, "capex_fixed") + _param(params, "capex_per_tonne") * total_tonnage
    capex_schedule = _plan_get(plan, "capex_schedule", None)
    if capex_schedule is not None:
        capex_total += float(np.sum(np.asarray(capex_schedule, dtype=np.float64)))

    return capex_total, opex_total


class NPVDistribution(NamedTuple):
    """A Monte-Carlo DCF outcome: the full NPV sample, not just a point estimate.

    ``samples`` is the length-``n`` array of per-draw NPVs (real dollars, one entry per Monte-Carlo
    trial) that :func:`value_at_risk` / :func:`conditional_value_at_risk` (J5) consume directly.
    ``mean``/``p10``/``p50``/``p90`` are the usual project-finance summary of that distribution.
    ``sensitivity`` decomposes the NPV variance into the grade- and price-uncertainty contributions
    (each factor's share of variance when the *other* factor is frozen at its mean; see
    :func:`monte_carlo_npv`), keyed ``"grade"`` / ``"price"`` (fraction of total variance, in
    ``[0, 1]``) plus the raw ``"grade_variance"`` / ``"price_variance"`` / ``"total_variance"``.
    """

    samples: np.ndarray
    mean: float
    p10: float
    p50: float
    p90: float
    sensitivity: dict


def _unpack_schedule(schedule: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read per-period ``(tonnage, capex)`` off ``schedule``.

    ``schedule`` is a mapping or attribute-bearing object with a required ``tonnage`` (recoverable
    tonnage per period, before any grade scaling) and an optional ``capex`` (lumpy capital spend per
    period; defaults to zero). A bare array-like is accepted too and treated as ``tonnage`` with no
    capex, so a plain per-period tonnage vector is enough for a project with all cost carried in
    ``cost_model``.
    """
    if isinstance(schedule, dict):
        tonnage = np.asarray(schedule["tonnage"], dtype=np.float64)
        capex = schedule.get("capex")
    else:
        tonnage_val = getattr(schedule, "tonnage", None)
        if tonnage_val is not None:
            tonnage = np.asarray(tonnage_val, dtype=np.float64)
            capex = getattr(schedule, "capex", None)
        else:
            tonnage = np.asarray(schedule, dtype=np.float64)
            capex = None

    tonnage = np.atleast_1d(tonnage)
    capex_arr = np.zeros_like(tonnage) if capex is None else np.atleast_1d(np.asarray(capex, dtype=np.float64))
    if capex_arr.shape != tonnage.shape:
        raise ValueError("monte_carlo_npv: schedule 'capex' must have the same shape as 'tonnage' (per period)")
    return tonnage, capex_arr


def _grade_per_period(grade: np.ndarray, n_periods: int, *, what: str) -> np.ndarray:
    """Broadcast a ``(n_draws, d)`` (or length-``d``) grade array onto ``n_periods`` periods.

    ``d == 1`` is one grade draw for the project's whole life (a single-deposit head grade), broadcast
    unchanged across every period; ``d == n_periods`` is one draw per scheduling period. Any other ``d``
    is a genuine mismatch between the posterior's dimensionality and the schedule's period count.
    """
    g = np.atleast_1d(grade)
    if g.ndim == 1:
        d = g.shape[0]
        if d == 1:
            return np.broadcast_to(g, (n_periods,)).astype(np.float64, copy=True)
        if d == n_periods:
            return g.astype(np.float64, copy=False)
        raise ValueError(
            f"monte_carlo_npv: {what} has {d} grade dimension(s) but schedule has {n_periods} period(s); "
            f"expected 1 (a single project-life grade draw) or {n_periods} (one grade draw per period)"
        )
    n_draws, d = g.shape
    if d == 1:
        return np.broadcast_to(g, (n_draws, n_periods)).astype(np.float64, copy=True)
    if d == n_periods:
        return g.astype(np.float64, copy=False)
    raise ValueError(
        f"monte_carlo_npv: {what} has {d} grade dimension(s) but schedule has {n_periods} period(s); "
        f"expected 1 (a single project-life grade draw) or {n_periods} (one grade draw per period)"
    )


def _align_price_paths(price_paths: Any, n: int, n_periods: int, rng: np.random.Generator) -> np.ndarray:
    """Coerce ``price_paths`` to exactly ``(n, n_periods)`` -- accepting EITHER orientation of a J1
    :class:`~mixle.inference.price_forecast.PriceForecast`.

    A scenario-major ``(m, n_periods)`` matrix (one row per price path, one column per period) is used
    as-is; ``mixle.inference.price_forecast.PriceForecast.paths`` is documented and produced as
    ``(n_periods, m)`` (time-major, mirroring how ``forecast_price`` builds it one horizon step at a
    time) -- passing ``pf.paths`` straight in used to raise, or (worse) silently score the wrong axis
    as "period" whenever ``m`` happened to equal ``n_periods``. Detected here from ``n_periods``
    (known independently, from ``schedule``) and transposed automatically; only genuinely ambiguous
    when ``m == n_periods`` too, where a square matrix is accepted as scenario-major -- its existing,
    tested behavior -- since no shape-only check can disambiguate a square matrix.

    Resamples with replacement to ``n`` rows when ``m != n`` (the "align" step of DR-ALG J2); a
    ``(m,)`` vector is treated as ``m`` single-period draws when ``n_periods == 1``, or as one
    deterministic ``n_periods``-long path shared by every draw otherwise.
    """
    prices = np.asarray(price_paths, dtype=np.float64)
    if prices.ndim == 1:
        prices = prices[:, None] if n_periods == 1 else prices[None, :]
    if prices.ndim == 2 and prices.shape[1] != n_periods and prices.shape[0] == n_periods:
        prices = prices.T  # PriceForecast.paths orientation: (n_periods, m) -> (m, n_periods)
    if prices.ndim != 2 or prices.shape[1] != n_periods:
        raise ValueError(
            f"monte_carlo_npv: price_paths must be shaped (m, {n_periods}) (one row per scenario, one "
            f"column per period) or its transpose ({n_periods}, m) (PriceForecast.paths); got {prices.shape}"
        )
    m = prices.shape[0]
    if m == n:
        return prices
    idx = rng.integers(0, m, size=n)
    return prices[idx]


def _cost_model_accepts_tonnage(cost_model: Callable) -> bool:
    try:
        return len(inspect.signature(cost_model).parameters) >= 2
    except (TypeError, ValueError):
        # a builtin/C callable or anything else signature() can't introspect: assume the
        # single-argument form rather than risk invoking `cost_model` twice on unrelated errors.
        return False


def _call_cost_model(cost_model: Callable, t: int, tonnage_t: float, *, accepts_tonnage: bool) -> float:
    """Call ``cost_model`` as ``cost_model(t, tonnage_t)`` if it takes two args, else ``cost_model(t)``.

    ``cost_model`` is opaque to this module (DR-ALG J2 writes it simply as ``opex_t(cost_model)``); a
    caller closing over :func:`cost_curve` (J4) typically needs the period's tonnage too. Arity is
    determined once up front via ``accepts_tonnage`` (see :func:`_cost_model_accepts_tonnage`) rather
    than by catching ``TypeError`` from the call itself -- a ``TypeError`` raised *inside*
    ``cost_model(t, tonnage_t)`` for an unrelated reason used to be misread as an arity mismatch and
    silently retried as ``cost_model(t)``, invoking ``cost_model`` a second time and masking the real
    error.
    """
    if accepts_tonnage:
        return float(cost_model(t, tonnage_t))
    return float(cost_model(t))


def _npv_samples(
    grade_per_period: np.ndarray,
    price_per_period: np.ndarray,
    tonnage: np.ndarray,
    opex: np.ndarray,
    capex: np.ndarray,
    discount: np.ndarray,
) -> np.ndarray:
    """``sum_t (tonnage_t * grade_{i,t} * price_{i,t} - opex_t - capex_t) / (1 + r) ** t``, vectorized."""
    cashflow = tonnage[None, :] * grade_per_period * price_per_period - opex[None, :] - capex[None, :]
    return cashflow @ discount


def monte_carlo_npv(
    posterior: Posterior,
    price_paths: Any,
    cost_model: Callable,
    schedule: Any,
    *,
    discount_rate: float,
    n: int = 10000,
    rng: np.random.Generator,
) -> NPVDistribution:
    """Monte-Carlo discounted-cash-flow NPV distribution (DR-ALG J2).

    Draws ``n`` grade realizations off the IC-1 ``posterior`` and pairs them, draw for draw, with ``n``
    (resampled/aligned as needed) price scenarios from ``price_paths``; for each draw and each of the
    ``schedule``'s periods, ``cashflow_t = tonnage_t * grade_t * price_t - opex_t - capex_t``, discounted
    at ``discount_rate`` (period ``0`` undiscounted, period ``t`` divided by ``(1 + discount_rate) ** t``)
    and summed into one ``NPV`` per draw:

    - ``posterior``: an IC-1 `Posterior` (frozen `mixle.reason.posterior_protocol.Posterior`); its
      ``.samples(n, rng)`` are the grade draws. A single project-life grade (``d == 1``) broadcasts
      across every period; a per-period posterior (``d == len(schedule)``) is used period by period.
    - ``price_paths``: a scenario-major ``(m, n_periods)`` array-like (one row per scenario), OR a J1
      ``PriceForecast.paths`` passed directly -- ``(n_periods, m)``, time-major -- detected and
      transposed automatically (see :func:`_align_price_paths`). Resampled with replacement to ``n``
      rows when ``m != n``.
    - ``cost_model``: called per period as ``cost_model(t, tonnage_t)`` (falling back to
      ``cost_model(t)``) to get that period's deterministic ``opex_t``.
    - ``schedule``: per-period ``tonnage`` (required) and ``capex`` (optional, default zero); see
      :func:`_unpack_schedule`. ``len(schedule)``'s tonnage vector fixes ``n_periods``.
    - ``discount_rate``: the DCF discount rate per period.
    - ``n`` / ``rng``: Monte-Carlo draw count and the shared `numpy.random.Generator`.

    Returns an `NPVDistribution` with the raw ``samples``, ``mean``/``p10``/``p50``/``p90``, and a
    ``sensitivity`` dict decomposing NPV variance into grade vs. price contributions: each factor's
    share of variance with the *other* factor frozen at its mean (posterior mean for grade, per-period
    mean price path for price) — not a full ANOVA decomposition, but the frozen-factor sensitivity the
    task's Algorithm calls for.
    """
    if n <= 0:
        raise ValueError("monte_carlo_npv: n must be a positive integer")

    tonnage, capex = _unpack_schedule(schedule)
    n_periods = tonnage.shape[0]

    grade_draws = np.asarray(posterior.samples(n, rng), dtype=np.float64)
    if grade_draws.ndim == 1:
        # IC-1 promises shape (n, d); a conforming posterior that squeezes a d == 1 draw matrix down to
        # (n,) is unambiguous here (we requested exactly n draws), unlike the mean/param case below.
        grade_draws = grade_draws.reshape(n, 1)
    grade_per_period = _grade_per_period(grade_draws, n_periods, what="posterior.samples(n, rng)")

    price_per_period = _align_price_paths(price_paths, n, n_periods, rng)

    accepts_tonnage = _cost_model_accepts_tonnage(cost_model)
    opex = np.array(
        [_call_cost_model(cost_model, t, float(tonnage[t]), accepts_tonnage=accepts_tonnage) for t in range(n_periods)]
    )
    discount = 1.0 / (1.0 + float(discount_rate)) ** np.arange(n_periods, dtype=np.float64)

    npv = _npv_samples(grade_per_period, price_per_period, tonnage, opex, capex, discount)

    mean = float(np.mean(npv))
    p10, p50, p90 = (float(q) for q in np.quantile(npv, [0.1, 0.5, 0.9]))

    # Sensitivity: freeze one factor at its mean, vary the other, compare the resulting variance to the
    # joint distribution's total variance.
    mean_price_per_period = np.broadcast_to(price_per_period.mean(axis=0), (n, n_periods))
    npv_grade_only = _npv_samples(grade_per_period, mean_price_per_period, tonnage, opex, capex, discount)

    mean_grade_per_period = _grade_per_period(np.atleast_1d(posterior.mean), n_periods, what="posterior.mean")
    mean_grade_broadcast = np.broadcast_to(mean_grade_per_period, (n, n_periods))
    npv_price_only = _npv_samples(mean_grade_broadcast, price_per_period, tonnage, opex, capex, discount)

    total_variance = float(np.var(npv))
    grade_variance = float(np.var(npv_grade_only))
    price_variance = float(np.var(npv_price_only))
    sensitivity = {
        "grade": grade_variance / total_variance if total_variance > 0.0 else 0.0,
        "price": price_variance / total_variance if total_variance > 0.0 else 0.0,
        "grade_variance": grade_variance,
        "price_variance": price_variance,
        "total_variance": total_variance,
    }

    return NPVDistribution(samples=npv, mean=mean, p10=p10, p50=p50, p90=p90, sensitivity=sensitivity)
