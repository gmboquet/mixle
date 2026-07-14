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

Emission factors are always supplied by the caller (or an upstream knowledge store) -- this module
vendors no lifecycle-inventory database; see the work-plan Non-goals for L1. Carbon *pricing* (turning
a footprint into a cost/NPV term) is out of scope here too -- see :mod:`mixle.analysis.emissions`'s
``transition_risk`` (L3) and ``climate_terms`` (L6), which build on :class:`Footprint`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from mixle.data.hashing import _canonical

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
