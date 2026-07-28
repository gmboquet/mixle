"""Scope 1/2/3 GHG (carbon) accounting for an operation's production activity, plus the L6 climate
objective + risk terms folded into J's objective and H's optimizer (work-plan Sec.7-L).

Maps a production ``activity`` schedule -- direct fuel combustion and blasting, purchased-grid
electricity draw, and upstream reagents / downstream haulage -- onto CO2e emissions via
GHG-Protocol-style :class:`EmissionFactors`.

  * :func:`emissions_footprint` -- Scope 1 (direct combustion/blasting) + Scope 2 (purchased
    electricity) + Scope 3 (upstream reagents / downstream transport) totals, with an optional
    Monte-Carlo 90% credible interval when per-factor uncertainties (:attr:`EmissionFactors.sigma`)
    are supplied, and a content-addressed ``activity_content_hash`` in the returned
    :class:`Footprint`'s provenance so every number traces back to the exact activity schedule that
    produced it.
  * :func:`transition_risk` -- prices a :class:`Footprint` against a set of carbon-price/policy
    scenario paths and subtracts the resulting per-scenario carbon cost from a J2 ``npv_samples``
    distribution, returning an IC-1 `DerivedQuantity` that carries the carbon-adjusted NPV samples
    (uncertainty-aware, not just a point estimate) plus a mean-value scenario ranking.
  * :func:`climate_terms` -- folds a :class:`Footprint` and an optional water budget into the two
    numbers J6's objective (``analysis/valuation.py``) and H4's optimizer (``stochastic_opt.py``)
    need: a priced carbon cost, and a hard water-feasibility flag (an explicit ``None``/unknown state,
    not a permissive default, when water evidence is missing or non-finite), plus a shortfall
    time-fraction downside term -- a single-trajectory duration statistic, not a probability.

Emission factors are always supplied by the caller (or an upstream knowledge store) -- this module
vendors no lifecycle-inventory database; see the work-plan Non-goals for L1.

Repo-boundary note (L6): L2's `WaterBudget` (a dataclass in the separate `mixle-pde` repo) is not a
dependency of mixle core and is never imported here; ``water`` duck-types against the local
:class:`WaterBudget` structural stand-in below (any object exposing a ``.shortfall_m3`` float and,
optionally, ``.storage``/``.provenance``/``.demand_m3`` satisfies it), so this module has no runtime
dependency on `mixle-pde`. The stand-in is a real name in THIS module's namespace -- unlike the bare
forward-reference string ``"WaterBudget | None"`` this replaced, which named an import that never
happens and left ``climate_terms``'s annotation unresolvable to ``typing.get_type_hints()``.
"""

from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.analysis._interval import validated_level
from mixle.data.hashing import _canonical
from mixle.reason.posterior_protocol import DerivedQuantity

__all__ = [
    "EmissionFactors",
    "Footprint",
    "emissions_footprint",
    "TransitionRiskResult",
    "transition_risk",
    "WaterBudget",
    "climate_terms",
]


@runtime_checkable
class WaterBudget(Protocol):
    """Structural stand-in for L2's `WaterBudget` (mixle-pde) -- see the module's repo-boundary note.

    Never constructed here; a real ``WaterBudget`` (or anything shaped like one) is passed in and
    duck-typed via ``getattr(water, ..., default)`` throughout this module, so a caller-supplied
    object satisfying only ``shortfall_m3`` (the one required field) still works -- the optional
    fields default to absent, matching the "or None if absent" fallbacks already documented on
    :func:`_water_demand_m3` and :func:`_shortfall_time_fraction`.
    """

    shortfall_m3: float
    storage: Any = None
    provenance: dict | None = None
    demand_m3: float | None = None


_VALID_SCOPES = (1, 2, 3)


def _finite_real_scalar(value: Any, *, name: str, nonnegative: bool = False) -> float:
    """Coerce one declared real scalar without accepting Boolean, text, or array axes."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-Boolean finite real scalar")
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(arr)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return scalar


@dataclass
class EmissionFactors:
    """Per-activity-key CO2e emission factors, one dict per GHG-Protocol scope.

    ``scope1``/``scope2``/``scope3`` map an activity key (e.g. ``"diesel_L"``, ``"grid_kWh"``,
    ``"explosives_kg"``, ``"transport_t_km"``) to a CO2e factor expressed per unit of that activity
    (e.g. kg CO2e per litre of diesel). A key absent from a scope's dict simply does not contribute to
    that scope. ``sigma`` optionally gives the standard deviation of each *factor* (not the activity
    quantity itself), keyed the same way across all three scopes, for Monte-Carlo uncertainty
    propagation in :func:`emissions_footprint`; a key with no entry in ``sigma`` is treated as exactly
    known (std 0).
    """

    scope1: dict[str, float]
    scope2: dict[str, float]
    scope3: dict[str, float]
    sigma: dict[str, float] | None = None


@dataclass(frozen=True)
class Footprint:
    """A Scope 1/2/3 CO2e footprint with an optional 90% credible interval and full provenance.

    ``scope1``/``scope2``/``scope3``/``total`` are in the same physical CO2e units as the emission
    factors (typically kg CO2e / tCO2e). ``ci`` is the ``(lo, hi)`` 90% Monte-Carlo interval on
    ``total`` when factor uncertainties were propagated, else ``None`` (the default -- a caller that
    only needs :func:`climate_terms`' point costs can construct one without ``ci``/``provenance``).
    ``provenance`` identifies *both* inputs to the ``activity x factors`` product, plus how the
    interval was produced: ``factor_source``; the 64-hex ``activity_content_hash`` and
    ``factor_content_hash`` (sha256 of the canonical activity / factor-model encodings, the same
    hashing convention IC-2 uses for field artifacts); ``factor_keys`` (the union of activity keys
    the factor model prices, for a readable diff when the two hashes disagree); the ``scopes``
    actually included in ``total``; and ``draws`` / ``uncertainty_propagated`` describing the
    Monte-Carlo configuration behind ``ci``. Hashing the activity alone was not enough to
    distinguish two runs: the same schedule costed against a supplier-specific factor set and
    against a national average is a different footprint with different provenance.

    The record is frozen: a reported footprint is evidence, and rebinding ``total`` or ``scope2``
    after construction would leave the numbers, the interval and the provenance describing three
    different computations. ``ci`` is normalized to a ``(lo, hi)`` tuple and ``provenance`` to a
    dict this record owns, so a later edit to the caller's own dict cannot retroactively rewrite
    what a footprint claims it was computed from.
    """

    scope1: float
    scope2: float
    scope3: float
    total: float
    ci: tuple[float, float] | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ci is not None:
            lo, hi = self.ci
            object.__setattr__(self, "ci", (float(lo), float(hi)))
        object.__setattr__(self, "provenance", dict(self.provenance))


def _scope_dict(factors: EmissionFactors, scope: int) -> dict[str, float]:
    if scope == 1:
        return factors.scope1
    if scope == 2:
        return factors.scope2
    if scope == 3:
        return factors.scope3
    raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")


def _scope_total(activity: dict[str, float], scope_factors: dict[str, float]) -> float:
    """Sum ``factor * activity[key]`` over the keys the scope's factor dict knows about."""
    products: list[float] = []
    for key, factor in scope_factors.items():
        product = float(factor) * float(activity.get(key, 0.0))
        if not np.isfinite(product):
            raise ValueError(f"emissions_footprint: factor * activity overflowed for key {key!r}")
        products.append(product)
    try:
        total = math.fsum(products)
    except OverflowError as exc:
        raise ValueError("emissions_footprint: scope total overflowed") from exc
    if not np.isfinite(total):
        raise ValueError("emissions_footprint: scope total must be finite")
    return total


def _activity_content_hash(activity: dict[str, float]) -> str:
    """sha256 hex digest of the canonical byte encoding of ``activity`` (IC-2 hashing convention:
    a deterministic, key-order-independent encoding of the record so the same activity numbers
    always hash the same, and any change to a key or a value changes the hash)."""
    return hashlib.sha256(_canonical(dict(activity))).hexdigest()


def _factor_content_hash(factors: EmissionFactors) -> str:
    """sha256 hex digest of the canonical encoding of the complete factor model.

    A footprint is ``activity x factors``: hashing only the activity left the other half of the
    computation untraced, so two runs against different factor models -- a supplier-specific set
    versus a national average, or a factor table revised between reporting periods -- produced
    totals that differ by a factor of several and provenance dicts that were byte-identical. All
    four fields participate (``sigma`` included: it sets the credible interval, so a change to it
    changes the reported footprint even when the point total is untouched).
    """
    payload = {
        "scope1": dict(factors.scope1),
        "scope2": dict(factors.scope2),
        "scope3": dict(factors.scope3),
        "sigma": dict(factors.sigma) if factors.sigma else {},
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _validate_unique_scopes(scopes: tuple[int, ...]) -> None:
    """Reject a ``scopes`` tuple that names the same GHG-Protocol scope more than once.

    A duplicate scope is a caller error, not a legitimate "sum it twice" input -- Scope 1/2/3 are
    fixed accounting categories, so the same scope appearing twice can only arise by mistake (e.g.
    programmatically concatenating two scope lists). Rejecting outright, rather than silently
    deduping, matters because the point-total path and the Monte-Carlo path below disagree on what a
    duplicate would mean if let through: the dict-keyed point total collapses it (counted once)
    while a loop over the raw tuple counts it once per occurrence, so a silent dedupe in one path and
    not the other is exactly how the reported point total and the reported credible interval used to
    end up describing two different footprints.
    """
    seen: set[int] = set()
    duplicates: set[int] = set()
    for s in scopes:
        if s in seen:
            duplicates.add(s)
        seen.add(s)
    if duplicates:
        raise ValueError(
            f"scopes must not contain duplicate entries, got {scopes!r} (duplicated: {sorted(duplicates)})"
        )


def _validate_activity_values(activity: dict[str, float]) -> None:
    """Every activity quantity must be a finite, nonnegative physical amount (litres, kWh, kg, t*km, ...)."""
    for key, value in activity.items():
        v = float(value)
        if not np.isfinite(v):
            raise ValueError(f"activity[{key!r}] must be finite, got {value!r}")
        if v < 0.0:
            raise ValueError(f"activity[{key!r}] must be nonnegative, got {value!r}")


def _validate_factor_values(factors: EmissionFactors) -> None:
    """Emission factors must be finite (sign is not constrained -- a documented carbon-negative
    feedstock factor is a legitimate finite negative number). ``sigma`` entries are standard
    deviations of a factor and must additionally be finite and nonnegative: a negative sigma is not
    a smaller uncertainty, it is a physically meaningless input that must be rejected rather than
    silently treated as exact (``std <= 0`` in the Monte-Carlo sampler below already means "no
    resampling", so a negative sigma was silently collapsing to a fixed, zero-uncertainty factor).
    """
    for label, scope_dict in (("scope1", factors.scope1), ("scope2", factors.scope2), ("scope3", factors.scope3)):
        for key, value in scope_dict.items():
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(f"factors.{label}[{key!r}] must be finite, got {value!r}")
    if factors.sigma:
        for key, value in factors.sigma.items():
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(f"factors.sigma[{key!r}] must be finite, got {value!r}")
            if v < 0.0:
                raise ValueError(f"factors.sigma[{key!r}] must be nonnegative, got {value!r}")


def _validate_draw_count(n: int) -> int:
    """Return an exact non-Boolean, nonnegative Monte Carlo draw count."""
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("emissions_footprint: n must be a non-Boolean nonnegative integer")
    try:
        count = operator.index(n)
    except TypeError as exc:
        raise ValueError("emissions_footprint: n must be a non-Boolean nonnegative integer") from exc
    if count < 0:
        raise ValueError("emissions_footprint: n must be nonnegative; use n=0 for a point estimate")
    return int(count)


def emissions_footprint(
    activity: dict[str, float],
    factors: EmissionFactors,
    *,
    scopes: tuple[int, ...] = (1, 2, 3),
    n: int = 0,
    rng: np.random.Generator | None = None,
) -> Footprint:
    """Scope 1/2/3 CO2e footprint of a production ``activity`` schedule.

    Each ``activity`` key (e.g. ``diesel_L``, ``grid_kWh``, ``explosives_kg``, ``transport_t_km``) is
    priced by the corresponding factor in whichever of ``factors.scope1/scope2/scope3`` includes that
    key: Scope 1 is direct combustion/blasting, Scope 2 is purchased electricity, Scope 3 is upstream
    reagents plus downstream transport. ``scopes`` selects which of the three scopes are actually
    included in the returned footprint (a scope not requested reports ``0.0`` and does not contribute
    to ``total`` -- e.g. ``scopes=(1, 2)`` for a Scope-1/2-only disclosure).

    If ``n > 0`` and ``factors.sigma`` is supplied, each priced factor is treated as
    ``Normal(mean=factor, std=sigma.get(key, 0.0))`` and resampled ``n`` times (factors with no
    ``sigma`` entry stay fixed); the resulting distribution of ``total`` yields a 90% credible interval
    in ``ci``. Without both a positive ``n`` and a non-empty ``sigma``, ``ci`` is ``None`` -- the point
    total is still returned, just without an uncertainty band.

    ``provenance`` always fingerprints both factors of the ``activity x factors`` product --
    ``activity_content_hash`` AND ``factor_content_hash`` (64-hex sha256 over the canonical
    encodings, IC-2's hashing convention) -- alongside ``factor_keys``, ``scopes``, ``draws`` and
    ``uncertainty_propagated``. A downstream carbon-cost or transition-risk term (L3/L6) can
    therefore be traced back to the exact activity schedule *and* the exact factor table it was
    priced with; recording only the activity meant a footprint restated against a revised or
    supplier-specific factor set was indistinguishable from the original by its provenance alone.

    ``scopes`` must not contain duplicate entries (each GHG-Protocol scope may appear at most once) --
    a duplicate raises rather than being silently summed once on the point-total path and once per
    occurrence on the Monte-Carlo path, which previously left the point estimate and the credible
    interval describing two different footprints. ``activity`` values must be finite and nonnegative;
    ``factors`` (``scope1``/``scope2``/``scope3``) values must be finite; ``factors.sigma`` values,
    when supplied, must be finite and nonnegative -- a negative sigma is rejected rather than silently
    treated as an exactly-known (zero-uncertainty) factor. ``n`` must be an exact non-Boolean
    nonnegative integer; zero explicitly requests the point estimate without an interval. All of
    this is validated before either the point-total or the Monte-Carlo path runs.
    """
    n = _validate_draw_count(n)
    for s in scopes:
        if s not in _VALID_SCOPES:
            raise ValueError(f"scopes must be a subset of {_VALID_SCOPES}, got {scopes!r}")
    _validate_unique_scopes(scopes)
    _validate_activity_values(activity)
    _validate_factor_values(factors)

    scope_values = {s: (_scope_total(activity, _scope_dict(factors, s)) if s in scopes else 0.0) for s in _VALID_SCOPES}
    try:
        total = math.fsum(scope_values.values())
    except OverflowError as exc:
        raise ValueError("emissions_footprint: total footprint overflowed") from exc
    if not np.isfinite(total):
        raise ValueError("emissions_footprint: total footprint must be finite")

    ci: tuple[float, float] | None = None
    if n > 0 and factors.sigma:
        gen = np.random.default_rng() if rng is None else rng
        totals = np.zeros(n)
        for s in scopes:
            for key, mean in _scope_dict(factors, s).items():
                qty = activity.get(key, 0.0)
                std = float(factors.sigma.get(key, 0.0))
                draws = gen.normal(mean, std, size=n) if std > 0 else np.full(n, mean)
                with np.errstate(over="ignore", invalid="ignore"):
                    contributions = draws * qty
                    totals += contributions
                if not np.isfinite(contributions).all() or not np.isfinite(totals).all():
                    raise ValueError("emissions_footprint: Monte Carlo totals must remain finite")
        lo, hi = np.quantile(totals, [0.05, 0.95])
        ci = (float(lo), float(hi))

    provenance = {
        "factor_source": "caller_supplied",
        "activity_content_hash": _activity_content_hash(activity),
        "factor_content_hash": _factor_content_hash(factors),
        "factor_keys": tuple(sorted(set(factors.scope1) | set(factors.scope2) | set(factors.scope3))),
        "scopes": tuple(scopes),
        "draws": n,
        "uncertainty_propagated": ci is not None,
    }

    return Footprint(
        scope1=scope_values[1],
        scope2=scope_values[2],
        scope3=scope_values[3],
        total=total,
        ci=ci,
        provenance=provenance,
    )


@dataclass(frozen=True)
class TransitionRiskResult:
    """The carbon-adjusted NPV distribution across carbon-price/policy scenarios (L3).

    Satisfies the frozen ``mixle.reason.posterior_protocol.DerivedQuantity`` structural protocol --
    ``samples``, ``prior_dominated``, ``credible_interval`` -- so a carbon-adjusted value can flow
    anywhere a `DerivedQuantity` is expected (J5 tail risk, J2 re-valuation). ``samples`` is shaped
    ``(n, k)``: the ``n`` baseline ``npv_samples`` draws, each re-priced under every one of the ``k``
    carbon-price scenarios (one column per scenario) -- the re-ranking below stays uncertainty-aware
    rather than collapsing straight to a point estimate. ``prior_dominated`` is always ``False``: there
    is no prior/regulariser here, the distribution's width is set entirely by ``npv_samples``.

    Beyond the protocol, ``scenario_mean`` (per-scenario mean carbon-adjusted NPV), ``ranking``
    (scenario indices sorted best -> worst by ``scenario_mean``), and ``carbon_cost`` (the
    priced-and-discounted carbon cost subtracted from each scenario) carry the scenario-level
    comparison :func:`transition_risk` exists to produce.

    Construction validates ``samples``: non-empty and finite (no NaN/Inf) -- defense-in-depth so
    invalid state can never flow downstream to a caller even if some upstream pushforward (here,
    :func:`transition_risk`) fails to validate its own inputs, the same guard applied to
    ``carcinogenic_risk.RiskQuantity`` and the ``_SampleDerivedQuantity`` carriers elsewhere in
    ``mixle.analysis``.

    The record is frozen and owns its arrays (validated copies, write-locked), so ``ranking`` and
    ``scenario_mean`` cannot drift out of agreement with the ``samples`` they were derived from:
    an unfrozen record let a caller reorder the ranking or overwrite a scenario mean while every
    summary method kept reporting off the original draws.
    """

    samples: np.ndarray
    prior_dominated: bool
    scenario_mean: np.ndarray
    ranking: tuple[int, ...]
    carbon_cost: np.ndarray
    provenance: dict

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("TransitionRiskResult.samples must be a non-empty 2-D (n, k) array.")
        if not np.isfinite(arr).all():
            raise ValueError("TransitionRiskResult.samples must be finite (no NaN/Inf).")
        means = np.asarray(self.scenario_mean, dtype=float)
        costs = np.asarray(self.carbon_cost, dtype=float)
        n_scenarios = arr.shape[1]
        if means.shape != (n_scenarios,) or not np.isfinite(means).all():
            raise ValueError("TransitionRiskResult.scenario_mean must be one finite value per scenario.")
        if costs.shape != (n_scenarios,) or not np.isfinite(costs).all():
            raise ValueError("TransitionRiskResult.carbon_cost must be one finite value per scenario.")
        if isinstance(self.prior_dominated, (bool, np.bool_)) is False:
            raise ValueError("TransitionRiskResult.prior_dominated must be Boolean.")
        if (
            len(self.ranking) != n_scenarios
            or any(isinstance(i, (bool, np.bool_)) for i in self.ranking)
            or sorted(self.ranking) != list(range(n_scenarios))
        ):
            raise ValueError("TransitionRiskResult.ranking must be a permutation of scenario indices.")
        for name, source in (("samples", arr), ("scenario_mean", means), ("carbon_cost", costs)):
            owned = np.array(source, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "ranking", tuple(int(i) for i in self.ranking))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-scenario central ``level`` interval of the carbon-adjusted NPV, each shape ``(k,)``.

        ``level`` goes through the shared ``mixle.analysis`` interval contract (MXR-080-1580).
        """
        level = validated_level(level)
        alpha = (1.0 - level) / 2.0
        lo = np.quantile(self.samples, alpha, axis=0)
        hi = np.quantile(self.samples, 1.0 - alpha, axis=0)
        return lo, hi


def _coerce_price_paths(carbon_price_paths: np.ndarray) -> np.ndarray:
    """Coerce ``carbon_price_paths`` to a ``(k, t)`` scenario matrix (one row per scenario).

    A 1-D ``(k,)`` array is a flat carbon price per scenario with no explicit period axis (each
    scenario has a single "period"); a 2-D ``(k, t)`` array is one price path per scenario, ``t``
    periods each -- the same "one row per scenario" convention `monte_carlo_npv` (J2) uses for
    ``price_paths``.
    """
    prices = np.asarray(carbon_price_paths, dtype=np.float64)
    if prices.ndim == 1:
        return prices[:, None]
    if prices.ndim == 2:
        return prices
    raise ValueError(f"transition_risk: carbon_price_paths must be 1-D (k,) or 2-D (k, t); got shape {prices.shape}")


def _validate_scenario_prices(prices: np.ndarray) -> None:
    """``prices`` is the already-coerced ``(k, t)`` scenario matrix from :func:`_coerce_price_paths`.

    Rejects an empty scenario set, an empty period axis, and any non-finite or negative price -- a
    carbon price is a $/tCO2e cost and has no documented meaning as negative or non-finite here.
    """
    n_scenarios, n_periods = prices.shape
    if n_scenarios == 0:
        raise ValueError("transition_risk: carbon_price_paths must have at least one scenario, got 0")
    if n_periods == 0:
        raise ValueError("transition_risk: carbon_price_paths must have at least one period, got 0")
    if not np.all(np.isfinite(prices)):
        raise ValueError("transition_risk: carbon_price_paths must be finite")
    if np.any(prices < 0.0):
        raise ValueError("transition_risk: carbon_price_paths must be nonnegative ($/tCO2e)")


def _validate_discount_weights(weights: np.ndarray) -> None:
    """Discount weights (e.g. ``1 / (1 + r) ** t``) must be finite and nonnegative."""
    if not np.all(np.isfinite(weights)):
        raise ValueError("transition_risk: discount must be finite")
    if np.any(weights < 0.0):
        raise ValueError("transition_risk: discount must be nonnegative")


def _validate_npv_samples(npv: np.ndarray) -> None:
    """``npv`` must be the 1-D ``(n,)`` baseline NPV distribution the docstring documents -- a single
    draw set shared across every carbon-price scenario, matching J2's `NPVDistribution.samples`
    contract. A 2-D (or higher) array is REJECTED rather than silently flattened: a caller passing a
    meaningfully-shaped ``(n_scenarios, n_draws)`` matrix (e.g. mistakenly expecting one NPV draw set
    per scenario) used to have it collapsed by ``.reshape(-1)`` into one long, structure-free pool of
    draws with no error -- exactly the kind of silent shape mistake this guards against.
    """
    if npv.ndim != 1:
        raise ValueError(
            "transition_risk: npv_samples must be 1-D (n,) -- a single baseline NPV distribution shared "
            f"across every carbon-price scenario (see the NPVDistribution.samples contract); got shape "
            f"{npv.shape}. It is not silently flattened, since that would collapse any real per-scenario "
            "structure into one long, meaningless pool of draws."
        )
    if npv.size == 0:
        raise ValueError("transition_risk: npv_samples must be nonempty")
    if not np.all(np.isfinite(npv)):
        raise ValueError("transition_risk: npv_samples must be finite")


def _discounted_carbon_costs(footprint_total: float, prices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute each nonnegative weighted price path without unchecked NumPy overflow."""
    costs = np.empty(prices.shape[0], dtype=np.float64)
    for scenario_index, price_path in enumerate(prices):
        terms: list[float] = []
        for price, weight in zip(price_path, weights, strict=True):
            term = float(price) * float(weight)
            if not np.isfinite(term):
                raise ValueError("transition_risk: discounted carbon price overflowed")
            terms.append(term)
        try:
            weighted_price = math.fsum(terms)
        except OverflowError as exc:
            raise ValueError("transition_risk: discounted carbon price overflowed") from exc
        cost = footprint_total * weighted_price
        if not np.isfinite(cost):
            raise ValueError("transition_risk: carbon cost overflowed")
        costs[scenario_index] = cost
    return costs


def _stable_column_means(values: np.ndarray) -> np.ndarray:
    """Mean each column without overflowing the sum of otherwise finite values."""
    divisor = values.shape[0]
    means = np.array(
        [math.fsum(float(value) / divisor for value in values[:, column]) for column in range(values.shape[1])],
        dtype=np.float64,
    )
    if not np.isfinite(means).all():
        raise ValueError("transition_risk: scenario means must remain finite")
    return means


def transition_risk(
    footprint: Footprint,
    carbon_price_paths: np.ndarray,
    *,
    npv_samples: np.ndarray,
    discount: np.ndarray | None = None,
) -> DerivedQuantity:
    """Carbon-adjusted NPV distribution + scenario ranking under a set of carbon-price paths (L3).

    For each of the ``k`` scenarios in ``carbon_price_paths`` (a ``(k,)`` flat price or a ``(k, t)``
    per-period path -- see :func:`_coerce_price_paths`), the priced carbon cost is
    ``footprint.total * sum_t(price[t] * discount[t])`` (``discount`` defaults to all-ones, i.e. no
    discounting, when omitted -- pass period discount factors, e.g. ``1 / (1 + r) ** t``, to match a
    J2 ``monte_carlo_npv`` DCF). That per-scenario carbon cost is subtracted from every draw of the
    baseline ``npv_samples`` (a J2 `NPVDistribution.samples`-shaped ``(n,)`` array), yielding an
    ``(n, k)`` carbon-adjusted value distribution: one re-priced NPV distribution per scenario, still
    carrying the original valuation uncertainty.

    Scenarios are ranked by mean carbon-adjusted NPV (``scenario_mean``, descending: best scenario
    first) so a high-carbon-price/policy scenario reliably re-ranks below a low-price one, with the
    gap between any two scenarios scaling linearly in ``footprint.total`` -- a bigger footprint pays
    proportionally more carbon cost under the same price paths. The returned
    :class:`TransitionRiskResult` satisfies IC-1's `DerivedQuantity` protocol so the re-ranking is
    always inspectable with a credible interval, not just a point estimate; it feeds J5 tail risk and
    J2 re-valuation directly.

    Validated before any ranking happens: ``carbon_price_paths`` must have at least one scenario and
    at least one period, with every price finite and nonnegative; ``discount``, when supplied, must be
    finite and nonnegative; ``footprint.total`` must be finite; and ``npv_samples`` must be a
    non-empty, finite, one-dimensional ``(n,)`` array (a multi-dimensional array is rejected, not
    silently flattened). Letting any of these through used to allow NaN scenario means and a
    meaningless ranking to construct successfully, failing only later when a caller requested a
    credible interval.
    """
    footprint_total = _finite_real_scalar(footprint.total, name="transition_risk: footprint.total")

    prices = _coerce_price_paths(carbon_price_paths)
    _validate_scenario_prices(prices)
    n_scenarios, n_periods = prices.shape

    if discount is None:
        weights = np.ones(n_periods, dtype=np.float64)
    else:
        weights = np.asarray(discount, dtype=np.float64)
        if weights.shape != (n_periods,):
            raise ValueError(
                f"transition_risk: discount must have shape ({n_periods},) to match carbon_price_paths' "
                f"period axis; got {weights.shape}"
            )
        _validate_discount_weights(weights)

    npv = np.asarray(npv_samples, dtype=np.float64)
    _validate_npv_samples(npv)

    carbon_cost = _discounted_carbon_costs(footprint_total, prices, weights)
    with np.errstate(over="ignore", invalid="ignore"):
        adjusted = npv[:, None] - carbon_cost[None, :]  # (n, k)
    if not np.isfinite(adjusted).all():
        raise ValueError("transition_risk: carbon-adjusted NPV samples must remain finite")

    scenario_mean = _stable_column_means(adjusted)
    ranking = tuple(int(i) for i in np.argsort(-scenario_mean))

    provenance = {
        "footprint_activity_hash": footprint.provenance.get("activity_content_hash"),
        "n_scenarios": n_scenarios,
        "n_periods": n_periods,
        "discounted": discount is not None,
        "carbon_cost": [float(c) for c in carbon_cost],
    }

    return TransitionRiskResult(
        samples=adjusted,
        prior_dominated=False,
        scenario_mean=scenario_mean,
        ranking=ranking,
        carbon_cost=carbon_cost,
        provenance=provenance,
    )


def _water_demand_m3(water: Any) -> float | None:
    """Best-effort scalar water demand off an L2 `WaterBudget`-shaped object, or ``None`` if absent.

    Checks a direct ``.demand_m3`` attribute first (a convenience some callers may attach), then falls
    back to ``water.provenance["demand_m3"]`` -- `water_balance`'s own ``demand_m3`` argument (L2) is
    exactly the kind of input the provenance dict is documented to retain.
    """
    demand = getattr(water, "demand_m3", None)
    if demand is not None:
        return float(demand)
    provenance = getattr(water, "provenance", None)
    if isinstance(provenance, dict) and "demand_m3" in provenance:
        return float(provenance["demand_m3"])
    return None


def _shortfall_time_fraction(water: Any, shortfall_m3: float) -> float | None:
    """Fraction of ONE deterministic water-budget trajectory's time steps sitting at a binding (zero)
    storage -- a duration/occupancy statistic, NOT a probability: it summarizes a single
    already-simulated trajectory and carries no information about uncertainty. A genuine probability
    of shortfall requires a defined ENSEMBLE of trajectories under different realizations (e.g.
    resampled inflow/demand) and the fraction of members that hit shortfall -- see
    :func:`_shortfall_probability_over_ensemble` for that computation, given an explicit ensemble; this
    module has no water-uncertainty model of its own to draw one from (see the module's repo-boundary
    note).

    Uses the `WaterBudget.storage` per-step array L2's algorithm always populates (one entry per routed
    time step) when available. Falls back to a 0/1 point read of ``shortfall_m3`` when no ``storage``
    trajectory is available (e.g. a caller-supplied summary rather than a full `WaterBudget`). Returns
    ``None`` -- an explicit "not evaluable", rather than a silently wrong number -- when the available
    evidence is unusable: a storage trajectory containing non-finite entries, or a ``shortfall_m3``
    that is non-finite or negative with no trajectory to fall back on. A shortfall is an unmet volume
    in m3 and cannot be negative; reading one as ``0.0`` ("never short") would report a definite
    all-clear off malformed evidence.
    """
    storage = getattr(water, "storage", None)
    if storage is not None:
        arr = np.asarray(storage, dtype=np.float64)
        if arr.size:
            if not np.all(np.isfinite(arr)):
                return None
            return float(np.mean(arr <= 0.0))
    if not np.isfinite(shortfall_m3) or shortfall_m3 < 0.0:
        return None
    return 1.0 if shortfall_m3 > 0.0 else 0.0


def _shortfall_probability_over_ensemble(trajectories: np.ndarray) -> float:
    """Genuine cross-realization probability of shortfall, given an EXPLICIT ensemble of independent
    storage trajectories (one row per realization -- e.g. resampled inflow/demand scenarios from an
    upstream uncertainty model).

    Unlike :func:`_shortfall_time_fraction` (a duration statistic over ONE deterministic trajectory),
    this treats each row as a distinct possible future and returns the empirical fraction of
    realizations whose trajectory hits a binding (zero) storage at any point -- a real probability over
    uncertainty, not over time. This module has no water-uncertainty model of its own (see the module's
    repo-boundary note), so the ensemble must be supplied by the caller; a single 1-D trajectory is not
    a valid ensemble (use :func:`_shortfall_time_fraction` for that).

    ``trajectories`` is a ``(n_members, n_steps)`` array, one storage path per ensemble member.
    """
    arr = np.asarray(trajectories, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "_shortfall_probability_over_ensemble: trajectories must be a 2-D (n_members, n_steps) "
            f"array, one storage path per realization; got shape {arr.shape}. A single 1-D trajectory "
            "is a duration statistic, not a probability -- see _shortfall_time_fraction."
        )
    n_members, n_steps = arr.shape
    if n_members == 0 or n_steps == 0:
        raise ValueError(
            f"_shortfall_probability_over_ensemble: trajectories must be nonempty in both axes; got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("_shortfall_probability_over_ensemble: trajectories must be finite")
    member_hits_shortfall = np.any(arr <= 0.0, axis=1)  # (n_members,)
    return float(np.mean(member_hits_shortfall))


def _kleene_or(a: bool | None, b: bool | None) -> bool | None:
    """Three-valued OR: a confirmed ``True`` dominates (a real risk signal from one check cannot be
    erased by another check's unknown), then Unknown, then confirmed ``False`` -- so a missing or
    non-finite check never silently downgrades a confirmed positive, and never rounds up to a false
    "feasible" either."""
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def climate_terms(
    footprint: Footprint,
    water: WaterBudget | None,  # duck-typed against the local structural stand-in; see module docstring
    *,
    carbon_price: float,
    water_limit_m3: float | None = None,
) -> dict:
    """Fold an emissions footprint and a water budget into J6's carbon cost + H4's water constraint.

    Returns ``{"carbon_cost": float, "water_feasible": bool | None, "water_binding": bool | None,
    "shortfall_time_fraction": float | None}``:

    - ``carbon_cost = footprint.total * carbon_price`` -- the priced carbon term J6's objective
      (`analysis/valuation.py`) subtracts alongside `capex_opex`'s cost roll-up. Always a definite
      float; it does not depend on water evidence.
    - ``water_binding`` is set when the L2 `WaterBudget.shortfall_m3 > 0` (the routed water balance
      already ran dry at some point over the plan) OR when a caller-supplied ``water_limit_m3`` is given
      and the budget's demand exceeds it; ``water_feasible`` is its negation -- a hard constraint H4
      (`stochastic_opt.py`) folds into a per-option cost (derating or dropping the option) rather than
      into the mean-grade objective. Either can come back ``None`` (unknown) instead of a definite
      ``bool`` -- see below.
    - ``shortfall_time_fraction`` is the fraction of the water budget's own per-step storage trajectory
      sitting at a binding zero (:func:`_shortfall_time_fraction`) -- a deterministic duration/occupancy
      statistic over ONE trajectory, reported independently of ``water_limit_m3`` even for options that
      are feasible on average but fragile. This is NOT a probability: it carries no uncertainty
      information. A genuine cross-realization probability requires an explicit ensemble of
      trajectories (:func:`_shortfall_probability_over_ensemble`), which is not available from a single
      ``water`` object.

    Missing or non-finite water evidence is an explicit UNKNOWN state, never a permissive pass:

    - When ``water`` is ``None`` (no water budget available for this option -- e.g. an inland option
      with no catchment exposure), ``water_feasible``, ``water_binding``, and
      ``shortfall_time_fraction`` are all ``None``: climate risk for that option is carbon-only, but
      the water terms are reported as "not evaluated", not silently "known feasible".
    - When ``shortfall_m3``, the water-limit demand, or ``water_limit_m3`` itself is non-finite
      (NaN/inf) **or negative**, that check contributes ``None`` (unknown) rather than silently
      comparing and passing as non-binding. All three are physical volumes in m3 and cannot be
      negative under the documented contract, so a negative one is malformed evidence rather than
      evidence of comfortable headroom (or, for a negative demand/limit, of a violation). The two
      binding checks combine via three-valued (Kleene) OR: a confirmed binding signal from either
      check always wins, an unknown never gets rounded down to "feasible".
    """
    footprint_total = _finite_real_scalar(footprint.total, name="climate_terms: footprint.total")
    price = _finite_real_scalar(carbon_price, name="climate_terms: carbon_price", nonnegative=True)
    carbon_cost = footprint_total * price
    if not np.isfinite(carbon_cost):
        raise ValueError("climate_terms: carbon cost overflowed")

    if water is None:
        return {
            "carbon_cost": carbon_cost,
            "water_feasible": None,
            "water_binding": None,
            "shortfall_time_fraction": None,
        }

    raw_shortfall = getattr(water, "shortfall_m3", 0.0)
    shortfall_m3 = float(raw_shortfall) if raw_shortfall is not None else 0.0
    # Shortfall, demand and the configured limit are physical volumes in m3 and cannot be negative
    # under the documented contract. A negative one is malformed evidence, not evidence of anything:
    # treating it as a finite number let a negative shortfall read as "comfortably non-binding"
    # (feasible) and a negative demand or limit manufacture a binding violation out of nothing. Both
    # directions are wrong, so a negative volume joins non-finite as UNKNOWN and rides the same
    # Kleene path -- never rounded down to feasible.
    shortfall_known = np.isfinite(shortfall_m3) and shortfall_m3 >= 0.0
    shortfall_binding: bool | None = (shortfall_m3 > 0.0) if shortfall_known else None

    demand_binding: bool | None = False  # no water_limit_m3 configured: this check is inapplicable, not unknown
    if water_limit_m3 is not None:
        limit = float(water_limit_m3)
        demand = _water_demand_m3(water)
        if demand is None:
            demand_binding = None  # a limit was configured but there is no demand evidence to check it against
        elif not (np.isfinite(limit) and np.isfinite(demand)) or limit < 0.0 or demand < 0.0:
            demand_binding = None
        else:
            demand_binding = demand > limit

    water_binding = _kleene_or(shortfall_binding, demand_binding)
    water_feasible = None if water_binding is None else (not water_binding)

    return {
        "carbon_cost": carbon_cost,
        "water_feasible": water_feasible,
        "water_binding": water_binding,
        "shortfall_time_fraction": _shortfall_time_fraction(water, shortfall_m3),
    }
