"""Scope 1/2/3 GHG (carbon) accounting for an operation's production activity.

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

Emission factors are always supplied by the caller (or an upstream knowledge store) -- this module
vendors no lifecycle-inventory database; see the work-plan Non-goals for L1. A full risk-adjusted
mining objective that also folds in water constraints is out of scope here too -- see
:mod:`mixle.analysis.emissions`'s ``climate_terms`` (L6), which builds on :class:`Footprint` and
:func:`transition_risk` alongside L2's water balance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mixle.data.hashing import _canonical
from mixle.reason.posterior_protocol import DerivedQuantity

_VALID_SCOPES = (1, 2, 3)


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


@dataclass
class Footprint:
    """A Scope 1/2/3 CO2e footprint with an optional 90% credible interval and full provenance.

    ``scope1``/``scope2``/``scope3``/``total`` are in the same physical CO2e units as the emission
    factors (typically kg CO2e). ``ci`` is the ``(lo, hi)`` 90% Monte-Carlo interval on ``total`` when
    factor uncertainties were propagated, else ``None``. ``provenance`` carries ``factor_source``,
    the 64-hex ``activity_content_hash`` (sha256 of the canonical activity encoding, the same hashing
    convention IC-2 uses for field artifacts), and the ``scopes`` actually included in ``total``.
    """

    scope1: float
    scope2: float
    scope3: float
    total: float
    ci: tuple[float, float] | None
    provenance: dict


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
    return float(sum(factor * activity.get(key, 0.0) for key, factor in scope_factors.items()))


def _activity_content_hash(activity: dict[str, float]) -> str:
    """sha256 hex digest of the canonical byte encoding of ``activity`` (IC-2 hashing convention:
    a deterministic, key-order-independent encoding of the record so the same activity numbers
    always hash the same, and any change to a key or a value changes the hash)."""
    return hashlib.sha256(_canonical(dict(activity))).hexdigest()


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

    ``provenance`` always carries ``activity_content_hash`` (a 64-hex sha256 fingerprint of the
    activity dict, IC-2's hashing convention) so a downstream carbon-cost or transition-risk term
    (L3/L6) can always be traced back to the exact activity schedule it was computed from.
    """
    for s in scopes:
        if s not in _VALID_SCOPES:
            raise ValueError(f"scopes must be a subset of {_VALID_SCOPES}, got {scopes!r}")

    scope_values = {s: (_scope_total(activity, _scope_dict(factors, s)) if s in scopes else 0.0) for s in _VALID_SCOPES}
    total = float(sum(scope_values.values()))

    ci: tuple[float, float] | None = None
    if n > 0 and factors.sigma:
        gen = np.random.default_rng() if rng is None else rng
        totals = np.zeros(n)
        for s in scopes:
            for key, mean in _scope_dict(factors, s).items():
                qty = activity.get(key, 0.0)
                std = float(factors.sigma.get(key, 0.0))
                draws = gen.normal(mean, std, size=n) if std > 0 else np.full(n, mean)
                totals += draws * qty
        lo, hi = np.quantile(totals, [0.05, 0.95])
        ci = (float(lo), float(hi))

    provenance = {
        "factor_source": "caller_supplied",
        "activity_content_hash": _activity_content_hash(activity),
        "scopes": tuple(scopes),
    }

    return Footprint(
        scope1=scope_values[1],
        scope2=scope_values[2],
        scope3=scope_values[3],
        total=total,
        ci=ci,
        provenance=provenance,
    )


@dataclass
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
    """

    samples: np.ndarray
    prior_dominated: bool
    scenario_mean: np.ndarray
    ranking: list[int]
    carbon_cost: np.ndarray
    provenance: dict

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-scenario central ``level`` interval of the carbon-adjusted NPV, each shape ``(k,)``."""
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
    """
    prices = _coerce_price_paths(carbon_price_paths)
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

    carbon_cost = footprint.total * (prices * weights[None, :]).sum(axis=1)  # (k,)

    npv = np.asarray(npv_samples, dtype=np.float64).reshape(-1)  # (n,)
    adjusted = npv[:, None] - carbon_cost[None, :]  # (n, k)

    scenario_mean = adjusted.mean(axis=0)
    ranking = [int(i) for i in np.argsort(-scenario_mean)]

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
