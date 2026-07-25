"""Parametric extraction-cost curves and Monte-Carlo project valuation (work-plan Sec.7-J, J4).

A general project-economics toolkit: a parametric per-unit cost as a function of depth, grade
(concentration of the recovered constituent), and throughput; a capex/opex roll-up over a
multi-period plan; and a Monte-Carlo discounted-cash-flow valuation over posterior grade draws and
price-path scenarios. This module's worked instantiation is mine economics -- J's objective function
needs a `$/t` cost that is a function of *where* the ore is (depth), *what* it is (grade), and *how
fast* it is mined (throughput) before J2 can turn price paths + posterior grade draws into an NPV
distribution, and before H's block-level optimizers (`mixle.stochastic_opt`, `mixle.relations`) have
a `block_cost` to subtract from revenue -- but the cost curve and DCF machinery apply to any
extraction/production project with a depth-, grade-, and throughput-sensitive cost structure:

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


# Coefficient/capital params that are $/t or $ scales of an additive cost or capital term: negative
# here lets one term of the cost curve (or the capex roll-up) go negative and, added into an otherwise
# non-negative total, silently turn part of the modeled spend into modeled income with no error
# (MXR-080-0119). `design_capacity` is deliberately excluded -- it is a physical throughput scale, not
# a cost coefficient, and is already required strictly positive (a divisor) below.
_NONNEGATIVE_PARAMS = frozenset(
    {
        "base_cost",
        "haul_cost_per_m",
        "grade_complexity_coef",
        "throughput_scale_coef",
        "capex_fixed",
        "capex_per_tonne",
    }
)


def _param(params: dict, key: str, *, context: str = "cost_curve") -> float:
    value = float(params[key]) if key in params else _DEFAULTS[key]
    if not np.isfinite(value):
        raise ValueError(f"{context}: params[{key!r}] must be finite, got {value!r}")
    if key in _NONNEGATIVE_PARAMS and value < 0.0:
        raise ValueError(f"{context}: params[{key!r}] must be non-negative, got {value!r}")
    return value


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

    ``depth`` must be finite and non-negative (metres below surface/datum; there is no such thing as
    negative haul depth); ``grade`` and ``throughput`` must be finite and strictly positive (both are
    also divisors elsewhere in the mine-planning stack, and a zero or negative ore grade or throughput
    is not a physical plan). Every numeric ``params`` entry must be finite, and the cost-coefficient
    entries (``base_cost``, ``haul_cost_per_m``, ``grade_complexity_coef``, ``throughput_scale_coef``)
    must be non-negative -- a negative coefficient would let one additive term of the curve go negative
    and silently discount an otherwise non-negative total cost into a subsidy (MXR-080-0119). Raises
    ``ValueError`` if any of these are violated.

    Returns the elementwise `$/t` cost, broadcast to the common shape of the three inputs.
    """
    d = np.asarray(depth, dtype=np.float64)
    g = np.asarray(grade, dtype=np.float64)
    q = np.asarray(throughput, dtype=np.float64)

    if not np.all(np.isfinite(d)):
        raise ValueError("cost_curve: depth must be finite")
    if np.any(d < 0.0):
        raise ValueError("cost_curve: depth must be non-negative")
    if not np.all(np.isfinite(g)):
        raise ValueError("cost_curve: grade must be finite")
    if np.any(g <= 0.0):
        raise ValueError("cost_curve: grade must be strictly positive (used as a 1/grade complexity term)")
    if not np.all(np.isfinite(q)):
        raise ValueError("cost_curve: throughput must be finite")
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


def _require_scalar_or_period_shape(value: Any, period_shape: tuple[int, ...], name: str) -> np.ndarray:
    """Coerce ``value`` to a float array that is either a true scalar (0-d; broadcasts against any
    period shape) or has EXACTLY ``period_shape`` (the plan's ``tonnage`` shape).

    :func:`cost_curve` broadcasts its three inputs against each other, which is the right behavior when
    it is called standalone -- but inside :func:`capex_opex`'s roll-up, a value that is merely
    numpy-broadcast-*compatible* with ``tonnage`` without sharing its shape (e.g. ``depth`` shaped
    ``(n_periods, 1)`` instead of ``(n_periods,)``) silently balloons ``cost_curve``'s output into an
    ``(n_periods, n_periods)`` cross-product matrix instead of raising -- summed against ``tonnage``,
    that multiplies each period's tonnage against every *other* period's cost too, not just its own
    (MXR-080-0119). Requiring an exact shape match (or a genuine scalar) closes that off; see
    ``test_capex_opex_rejects_broadcast_compatible_but_mismatched_period_shape``.
    """
    if value is None:
        raise ValueError(f"capex_opex: plan is missing required field {name!r}")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != () and arr.shape != period_shape:
        raise ValueError(
            f"capex_opex: plan {name!r} has shape {arr.shape}, but 'tonnage' has shape {period_shape}; "
            f"{name!r} must be a scalar or match 'tonnage' exactly, one entry per period (a shape that "
            "is merely numpy-broadcast-compatible but not identical is rejected -- it can silently "
            "multiply cost across periods instead of pairing each period with its own cost)"
        )
    return arr


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

    ``tonnage`` must be finite and non-negative and have at least one period; ``depth``/``grade``/
    ``throughput`` must each be a scalar or share ``tonnage``'s exact shape (see
    :func:`_require_scalar_or_period_shape`); ``capex_schedule``, if given, must be finite and
    non-negative. A negative tonnage or a negative cost/capital term would otherwise turn modeled
    spending into modeled revenue with no error (MXR-080-0119); raises ``ValueError`` if violated.
    """
    tonnage = np.atleast_1d(np.asarray(_plan_get(plan, "tonnage"), dtype=np.float64))
    if tonnage.shape[0] == 0:
        raise ValueError("capex_opex: plan 'tonnage' must have at least one period")
    if not np.all(np.isfinite(tonnage)):
        raise ValueError("capex_opex: plan 'tonnage' must be finite")
    if np.any(tonnage < 0.0):
        raise ValueError("capex_opex: plan 'tonnage' must be non-negative")

    depth = _require_scalar_or_period_shape(_plan_get(plan, "depth"), tonnage.shape, "depth")
    grade = _require_scalar_or_period_shape(_plan_get(plan, "grade"), tonnage.shape, "grade")
    throughput = _require_scalar_or_period_shape(_plan_get(plan, "throughput"), tonnage.shape, "throughput")

    per_period_cost = cost_curve(depth, grade, throughput, params=params)
    opex_total = float(np.sum(tonnage * per_period_cost))

    total_tonnage = float(np.sum(tonnage))
    capex_total = _param(params, "capex_fixed", context="capex_opex")
    capex_total += _param(params, "capex_per_tonne", context="capex_opex") * total_tonnage
    capex_schedule = _plan_get(plan, "capex_schedule", None)
    if capex_schedule is not None:
        capex_arr = np.asarray(capex_schedule, dtype=np.float64)
        if not np.all(np.isfinite(capex_arr)):
            raise ValueError("capex_opex: plan 'capex_schedule' must be finite")
        if np.any(capex_arr < 0.0):
            raise ValueError("capex_opex: plan 'capex_schedule' must be non-negative")
        capex_total += float(np.sum(capex_arr))

    return capex_total, opex_total


class _NPVDistributionFields(NamedTuple):
    """Field layout for :class:`NPVDistribution` -- kept as a plain, unvalidated `NamedTuple` base
    because ``typing.NamedTuple``'s class-creation machinery prohibits overriding ``__new__``/
    ``__init__`` directly on a class that inherits from ``NamedTuple`` itself; a genuine subclass
    (:class:`NPVDistribution` below) is the standard way around that to add construction-time
    validation while keeping every existing ``NPVDistribution(...)`` call site (keyword or
    positional) unchanged.
    """

    samples: np.ndarray
    mean: float
    p10: float
    p50: float
    p90: float
    sensitivity: dict


class NPVDistribution(_NPVDistributionFields):
    """A Monte-Carlo DCF outcome: the full NPV sample, not just a point estimate.

    ``samples`` is the length-``n`` array of per-draw NPVs (real dollars, one entry per Monte-Carlo
    trial) that :func:`value_at_risk` / :func:`conditional_value_at_risk` (J5) consume directly.
    ``mean``/``p10``/``p50``/``p90`` are the usual project-finance summary of that distribution.
    ``sensitivity`` decomposes the NPV variance into the grade- and price-uncertainty contributions, as
    Sobol first-order variance-based sensitivity indices (see :func:`monte_carlo_npv` and
    :func:`_sobol_first_order_share`): keyed ``"grade"`` / ``"price"`` (each factor's fraction of total
    variance explained *on its own*, mathematically in ``[0, 1]`` and not required to sum to ``1`` --
    NPV is multiplicative in grade and price, so some variance is a grade/price interaction effect
    attributed to neither) plus the raw ``"grade_variance"`` / ``"price_variance"`` / ``"total_variance"``.

    Construction validates ``samples``: non-empty and finite (no NaN/Inf) -- defense-in-depth so
    invalid state can never flow downstream to a caller even if some upstream computation (here,
    :func:`monte_carlo_npv`) fails to validate its own inputs, the same guard applied to
    ``carcinogenic_risk.RiskQuantity`` and the ``_SampleDerivedQuantity`` carriers elsewhere in
    ``mixle.analysis``.
    """

    __slots__ = ()

    def __new__(
        cls,
        samples: np.ndarray,
        mean: float,
        p10: float,
        p50: float,
        p90: float,
        sensitivity: dict,
    ) -> NPVDistribution:
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            raise ValueError("NPVDistribution.samples must be non-empty.")
        if not np.isfinite(arr).all():
            raise ValueError("NPVDistribution.samples must be finite (no NaN/Inf).")
        return super().__new__(cls, samples, mean, p10, p50, p90, sensitivity)


def _unpack_schedule(schedule: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read per-period ``(tonnage, capex)`` off ``schedule``.

    ``schedule`` is a mapping or attribute-bearing object with a required ``tonnage`` (recoverable
    tonnage per period, before any grade scaling) and an optional ``capex`` (lumpy capital spend per
    period; defaults to zero). A bare array-like is accepted too and treated as ``tonnage`` with no
    capex, so a plain per-period tonnage vector is enough for a project with all cost carried in
    ``cost_model``.

    ``tonnage`` must have at least one period and, together with ``capex``, must be finite and
    non-negative (MXR-080-0118/0119 economic-domain contract). Raises ``ValueError`` if violated.
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
    if tonnage.shape[0] == 0:
        raise ValueError("monte_carlo_npv: schedule 'tonnage' must have at least one period")
    if not np.all(np.isfinite(tonnage)):
        raise ValueError("monte_carlo_npv: schedule 'tonnage' must be finite")
    if np.any(tonnage < 0.0):
        raise ValueError("monte_carlo_npv: schedule 'tonnage' must be non-negative")

    capex_arr = np.zeros_like(tonnage) if capex is None else np.atleast_1d(np.asarray(capex, dtype=np.float64))
    if capex_arr.shape != tonnage.shape:
        raise ValueError("monte_carlo_npv: schedule 'capex' must have the same shape as 'tonnage' (per period)")
    if not np.all(np.isfinite(capex_arr)):
        raise ValueError("monte_carlo_npv: schedule 'capex' must be finite")
    if np.any(capex_arr < 0.0):
        raise ValueError("monte_carlo_npv: schedule 'capex' must be non-negative")
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


def _draw_grade_per_period(
    posterior: Posterior, n: int, n_periods: int, rng: np.random.Generator, *, what: str = "posterior.samples(n, rng)"
) -> np.ndarray:
    """Draw exactly ``n`` grade realizations off ``posterior``, validate them, and broadcast per period.

    IC-1 promises ``posterior.samples(n, rng)`` returns exactly ``n`` draws, shape ``(n, d)``. That
    count was never actually checked: a posterior that returns fewer rows (most dangerously exactly
    one) than requested used to pass straight through to :func:`_npv_samples`, where a ``(1, n_periods)``
    grade array silently numpy-broadcasts against the ``(n, n_periods)`` price array -- fabricating ``n``
    "independent" NPV draws that all in fact share the single real grade draw (MXR-080-0118). Checking
    ``.shape[0] == n`` up front, before any broadcasting can hide the shortfall, closes that off.
    """
    grade_draws = np.asarray(posterior.samples(n, rng), dtype=np.float64)
    if grade_draws.ndim == 0 or grade_draws.shape[0] != n:
        got = grade_draws.shape[0] if grade_draws.ndim > 0 else 1
        raise ValueError(f"monte_carlo_npv: {what} returned {got} draw(s), expected exactly n={n}")
    if grade_draws.ndim == 1:
        # A conforming posterior that squeezes a d == 1 draw matrix down to (n,) is unambiguous here
        # (we just confirmed exactly n draws), unlike the mean/param case elsewhere in this module.
        grade_draws = grade_draws.reshape(n, 1)
    if not np.all(np.isfinite(grade_draws)):
        raise ValueError(f"monte_carlo_npv: {what} must be finite")
    if np.any(grade_draws < 0.0):
        raise ValueError(f"monte_carlo_npv: {what} (grade) must be non-negative")
    return _grade_per_period(grade_draws, n_periods, what=what)


def _coerce_price_scenario_pool(price_paths: Any, n_periods: int) -> np.ndarray:
    """Coerce ``price_paths`` to a validated ``(m, n_periods)`` scenario pool -- shape/orientation and
    economic-domain checks only, no resampling; see :func:`_align_price_paths`.

    Accepts EITHER orientation of a J1 :class:`~mixle.inference.price_forecast.PriceForecast`: a
    scenario-major ``(m, n_periods)`` matrix (one row per price path, one column per period) is used
    as-is; ``PriceForecast.paths`` is documented and produced as ``(n_periods, m)`` (time-major,
    mirroring how ``forecast_price`` builds it one horizon step at a time) -- passing ``pf.paths``
    straight in used to raise, or (worse) silently score the wrong axis as "period" whenever ``m``
    happened to equal ``n_periods``. Detected here from ``n_periods`` (known independently, from
    ``schedule``) and transposed automatically; only genuinely ambiguous when ``m == n_periods`` too,
    where a square matrix is accepted as scenario-major -- its existing, tested behavior -- since no
    shape-only check can disambiguate a square matrix.
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
    if prices.shape[0] == 0:
        raise ValueError("monte_carlo_npv: price_paths must contain at least one scenario")
    if not np.all(np.isfinite(prices)):
        raise ValueError("monte_carlo_npv: price_paths must be finite")
    if np.any(prices < 0.0):
        raise ValueError("monte_carlo_npv: price_paths must be non-negative")
    return prices


def _align_price_paths(price_paths: Any, n: int, n_periods: int, rng: np.random.Generator) -> np.ndarray:
    """Coerce ``price_paths`` to exactly ``(n, n_periods)``: :func:`_coerce_price_scenario_pool` plus
    resampling with replacement to ``n`` rows when ``m != n`` (the "align" step of DR-ALG J2); a
    ``(m,)`` vector is treated as ``m`` single-period draws when ``n_periods == 1``, or as one
    deterministic ``n_periods``-long path shared by every draw otherwise.
    """
    prices = _coerce_price_scenario_pool(price_paths, n_periods)
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


def _sobol_first_order_share(
    y_a: np.ndarray, y_b: np.ndarray, y_ab_i: np.ndarray, total_variance: float
) -> tuple[float, float]:
    """Jansen (1999) / Saltelli (2010) first-order Sobol sensitivity index estimator for one factor.

    ``y_a = f(A)`` and ``y_b = f(B)`` are model outputs on two INDEPENDENT full joint samples (every
    factor independently drawn in both); ``y_ab_i = f(A_B^(i))`` is the output on the hybrid sample
    that takes factor ``i`` from ``B`` and every other factor from ``A``. Returns
    ``(variance_i, share_i)``:

    - ``variance_i = mean(y_b * (y_ab_i - y_a))``, a consistent estimator of ``Var(E[Y | X_i])`` (the
      part of ``Y``'s variance explained by ``X_i`` alone, averaged over the other factors). ``y_b``
      and ``y_ab_i`` share ONLY factor ``i`` (both take it from ``B``); every other factor is an
      independent draw between them (one keeps ``A``'s realization, the other keeps ``B``'s) -- that
      shared-``X_i``-independent-elsewhere construction is exactly what isolates factor ``i``'s own
      contribution, regardless of how ``A``'s columns happen to be paired internally.
    - ``share_i = variance_i / total_variance`` (0 if ``total_variance`` is 0): the first-order Sobol
      index proper, ``Var(E[Y | X_i]) / Var(Y)``. For independent inputs this is mathematically
      guaranteed in ``[0, 1]`` (a variance decomposition can only explain a fraction of the total). The
      finite-``n`` Monte Carlo ESTIMATE of it can stray slightly outside that range from ordinary
      sampling noise (well documented for Sobol estimators generally, and unavoidable at small ``n``);
      both outputs are clipped to their population-guaranteed range (``variance_i >= 0``, ``share_i in
      [0, 1]``) to absorb exactly that noise.

    This replaces a previous "freeze the other factor at its mean, compare that variance to the actual
    paired joint sample's variance" ratio, which had NO such guarantee: an adversarially (or merely
    negatively) paired joint sample can have its variance driven arbitrarily close to zero by
    cancellation while each frozen-factor variance stays large, sending the ratio to the millions
    (MXR-080-0117). The clip here guards against small-``n`` estimator noise around a bound that
    provably holds in the population; it is not concealing an unbounded failure mode the way clipping
    the old ratio would have been.
    """
    variance_i = max(float(np.mean(y_b * (y_ab_i - y_a))), 0.0)
    share_i = min(variance_i / total_variance, 1.0) if total_variance > 0.0 else 0.0
    return variance_i, share_i


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
    - ``discount_rate``: the DCF discount rate per period. Must be finite and strictly greater than
      ``-1`` -- ``discount_rate <= -1`` makes the per-period discount factor
      ``1 / (1 + discount_rate) ** t`` divide by zero (at exactly ``-1``) or alternate sign every other
      period (below ``-1``), neither of which is a coherent discount rate.
    - ``n`` / ``rng``: Monte-Carlo draw count and the shared `numpy.random.Generator`. ``n`` must be a
      positive ``int`` (not merely a positive-valued float); ``posterior.samples(n, rng)`` is required
      to return exactly ``n`` draws, checked explicitly rather than trusted -- a posterior that returns
      fewer (e.g. a single row) used to silently numpy-broadcast across every price scenario, fabricating
      ``n`` "independent" draws that all shared one real grade sample (MXR-080-0118).

    Every numeric input -- the grade draws, ``price_paths``, ``schedule``'s ``tonnage``/``capex``, and
    ``cost_model``'s returned ``opex`` -- must be finite and non-negative; raises ``ValueError`` if any
    of these, ``n``, or ``discount_rate`` are violated, before any DCF arithmetic runs.

    Returns an `NPVDistribution` with the raw ``samples``, ``mean``/``p10``/``p50``/``p90``, and a
    ``sensitivity`` dict decomposing NPV variance into grade vs. price contributions as Sobol
    first-order variance-based sensitivity indices (Saltelli/Jansen two-independent-sample estimator;
    see :func:`_sobol_first_order_share`) -- each factor's own share of the total NPV variance,
    estimated from a SECOND joint sample independent of the one ``samples``/``mean``/``p10``/``p50``/
    ``p90`` are built from, so the indices stay meaningful (population-bounded in ``[0, 1]``) even when
    a caller's ``price_paths`` happen to be paired against the posterior's draws in a way that makes
    the actual joint sample's own variance small (MXR-080-0117). Not a full ANOVA decomposition (grade
    and price's interaction/second-order effect, if any, is not itself reported), but each reported
    share is individually a real, correctly-estimated first-order Sobol index.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"monte_carlo_npv: n must be a positive integer, got {n!r}")
    if not np.isfinite(discount_rate):
        raise ValueError(f"monte_carlo_npv: discount_rate must be finite, got {discount_rate!r}")
    if discount_rate <= -1.0:
        raise ValueError(
            f"monte_carlo_npv: discount_rate must be > -1, got {discount_rate!r} -- at or below -1 the "
            "per-period discount factor 1 / (1 + discount_rate) ** t divides by zero or alternates sign"
        )

    tonnage, capex = _unpack_schedule(schedule)
    n_periods = tonnage.shape[0]

    grade_per_period = _draw_grade_per_period(posterior, n, n_periods, rng)

    price_per_period = _align_price_paths(price_paths, n, n_periods, rng)

    accepts_tonnage = _cost_model_accepts_tonnage(cost_model)
    opex = np.array(
        [_call_cost_model(cost_model, t, float(tonnage[t]), accepts_tonnage=accepts_tonnage) for t in range(n_periods)]
    )
    if not np.all(np.isfinite(opex)):
        raise ValueError("monte_carlo_npv: cost_model must return finite opex")
    if np.any(opex < 0.0):
        raise ValueError("monte_carlo_npv: cost_model must return non-negative opex")
    discount = 1.0 / (1.0 + float(discount_rate)) ** np.arange(n_periods, dtype=np.float64)

    npv = _npv_samples(grade_per_period, price_per_period, tonnage, opex, capex, discount)

    mean = float(np.mean(npv))
    p10, p50, p90 = (float(q) for q in np.quantile(npv, [0.1, 0.5, 0.9]))

    # Sensitivity: Sobol first-order variance-based indices for grade and price, via the standard
    # Saltelli/Jansen two-independent-sample estimator (see _sobol_first_order_share). Draw a SECOND,
    # fully independent joint sample B (independent grade AND independent price -- regardless of how
    # A's own grade_per_period/price_per_period happen to be paired), plus the two single-factor swaps.
    grade_b_per_period = _draw_grade_per_period(
        posterior, n, n_periods, rng, what="posterior.samples(n, rng) (sensitivity resample)"
    )
    price_pool = _coerce_price_scenario_pool(price_paths, n_periods)
    price_b_per_period = price_pool[rng.integers(0, price_pool.shape[0], size=n)]

    y_a = npv  # f(A): the actual paired joint sample, already computed above
    y_b = _npv_samples(grade_b_per_period, price_b_per_period, tonnage, opex, capex, discount)  # f(B)
    y_ab_grade = _npv_samples(
        grade_b_per_period, price_per_period, tonnage, opex, capex, discount
    )  # grade from B, price from A
    y_ab_price = _npv_samples(
        grade_per_period, price_b_per_period, tonnage, opex, capex, discount
    )  # price from B, grade from A

    total_variance = float(np.var(y_a))
    grade_variance, grade_share = _sobol_first_order_share(y_a, y_b, y_ab_grade, total_variance)
    price_variance, price_share = _sobol_first_order_share(y_a, y_b, y_ab_price, total_variance)
    sensitivity = {
        "grade": grade_share,
        "price": price_share,
        "grade_variance": grade_variance,
        "price_variance": price_variance,
        "total_variance": total_variance,
    }

    return NPVDistribution(samples=npv, mean=mean, p10=p10, p50=p50, p90=p90, sensitivity=sensitivity)
