"""IC-10 tool/model catalog -> verified :class:`~mixle.task.router.Router` wiring (M3-core).

Every routable capability -- a physics forward, an economic model, a climate projection, an
external domain model (IC-7) -- registers with the same uniform shape (:class:`CatalogEntry`, IC-10)
so a decomposer/router never special-cases a tool family. :func:`build_catalog_router` turns a flat
list of entries into one calibrated :class:`~mixle.task.router.Router` tier per entry (cheapest first),
falling through to a frontier/teacher callable exactly the way :class:`~mixle.task.cascade.Cascade`
already escalates uncertain calls.

Local convention (not part of the frozen IC-10 shape, documented here because M3 owns the wiring):
``CatalogEntry.schema`` is a plain ``dict[str, Any]`` (IC-10 imposes no further shape on it), so a
registrant MAY stash two well-known keys on it: ``"output"`` (the JSON-schema of what this
entry produces, used for schema-compatibility matching in :mod:`mixle.task.knowledge_routing`) and
``"invoke"`` (a ``Callable[[dict], Any]`` that actually runs the tool/model given a request dict).
Both are required by :func:`build_catalog_router`: the router returns a verified output, never an
entry id standing in for an answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from mixle.task.calibrate import ESCALATE
from mixle.task.router import Router

__all__ = ["CatalogCalibration", "CatalogEntry", "build_catalog_router"]


def _finite_real(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class CatalogEntry:
    """A routable capability: identity, its JSON `schema`, owning subsystem, `cost`, prior
    `reliability`, and an optional verifier (the IC-10 tuple).

    IC-10 (``notes/exec/contracts.md``) freezes ``verifier`` as ``str | None`` -- the *name* of the
    IC-6 verifier kind that gates this entry's output. M3's own algorithm (step 5: "attach each
    verifier; verified item ids resolve gaps") needs to actually CALL a verifier, not just look one up
    by name, so this field is typed ``Any | None`` here and accepts either an IC-10-style string tag
    or a live IC-6 ``Verifier``-shaped object (anything exposing ``.verify(claim, context)``) -- a
    thin, additive shim over the frozen contract, not a rename of it. Field names/order/defaults are
    otherwise identical to the frozen stub.
    """

    id: str
    schema: dict[str, Any]
    owner: str  # "physics" | "economic" | "climate" | "external" | ...
    cost: float = 0.0
    reliability: float = 1.0
    verifier: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("catalog entry id must be a non-empty string")
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("catalog entry owner must be a non-empty string")
        if not isinstance(self.schema, dict):
            raise TypeError("catalog entry schema must be a dictionary")
        cost = _finite_real(self.cost, "catalog entry cost", minimum=0.0, maximum=float("inf"))
        reliability = _finite_real(self.reliability, "catalog entry reliability", minimum=0.0, maximum=1.0)
        if (
            self.verifier is not None
            and not isinstance(self.verifier, str)
            and not callable(getattr(self.verifier, "verify", None))
        ):
            raise TypeError("catalog entry verifier must be a verifier name, an object exposing verify(), or None")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "reliability", reliability)


@dataclass(frozen=True)
class CatalogCalibration:
    """Held-out verifier outcomes used to gate a catalog capability."""

    verified: int
    trials: int
    confidence: float = 0.95

    def __post_init__(self) -> None:
        trials = _exact_int(self.trials, "calibration trials", minimum=1)
        verified = _exact_int(self.verified, "verified calibration outcomes", minimum=0)
        if verified > trials:
            raise ValueError("verified calibration outcomes cannot exceed trials")
        confidence = _finite_real(self.confidence, "calibration confidence", minimum=0.0, maximum=1.0)
        if confidence in (0.0, 1.0):
            raise ValueError("calibration confidence must lie strictly between 0 and 1")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "verified", verified)
        object.__setattr__(self, "confidence", confidence)

    @property
    def observed_reliability(self) -> float:
        return self.verified / self.trials

    @property
    def lower_bound(self) -> float:
        """One-sided Clopper-Pearson lower confidence bound."""
        if self.verified == 0:
            return 0.0
        return float(
            beta_distribution.ppf(
                1.0 - self.confidence,
                self.verified,
                self.trials - self.verified + 1,
            )
        )


_JSON_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})


def _validate_output_schema(schema: Any, *, path: str = "output") -> None:
    if not isinstance(schema, dict):
        raise TypeError(f"{path} schema must be a dictionary")
    kind = schema.get("type")
    if kind not in _JSON_TYPES:
        raise ValueError(f"{path} schema must declare one JSON type from {sorted(_JSON_TYPES)}")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{path}.enum must be a non-empty list")
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict):
            raise TypeError(f"{path}.properties must be a dictionary")
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) or not name for name in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError(f"{path}.required must contain unique non-empty strings")
        if not set(required) <= set(properties):
            raise ValueError(f"{path}.required names must be declared in properties")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{path}.properties keys must be non-empty strings")
            _validate_output_schema(child, path=f"{path}.properties.{name}")
    elif kind == "array":
        if "items" not in schema:
            raise ValueError(f"{path} array schema must declare items")
        _validate_output_schema(schema["items"], path=f"{path}.items")


def _matches_output_schema(value: Any, schema: dict[str, Any]) -> bool:
    if "enum" in schema and value not in schema["enum"]:
        return False
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            return False
        if any(name not in value for name in schema.get("required", [])):
            return False
        return all(
            name not in value or _matches_output_schema(value[name], child)
            for name, child in schema.get("properties", {}).items()
        )
    if kind == "array":
        return isinstance(value, list) and all(_matches_output_schema(item, schema["items"]) for item in value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and np.isfinite(float(value))
    if kind == "integer":
        return not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral)
    if kind == "boolean":
        return isinstance(value, (bool, np.bool_))
    return value is None


def _verdict_passed(verdict: Any) -> bool:
    if isinstance(verdict, dict):
        passed = verdict.get("passed")
        low_power = verdict.get("low_power", False)
    else:
        passed = getattr(verdict, "passed", None)
        low_power = getattr(verdict, "low_power", False)
    if not isinstance(passed, (bool, np.bool_)) or not isinstance(low_power, (bool, np.bool_)):
        raise TypeError("verifier verdict must expose boolean passed and low_power fields")
    return bool(passed) and not bool(low_power)


class _VerifiedToolGate:
    """Execute, schema-check, and verify one calibrated catalog capability."""

    def __init__(
        self,
        entry: CatalogEntry,
        *,
        calibration: CatalogCalibration,
        min_verified_reliability: float,
    ) -> None:
        self.entry = entry
        self.calibration = calibration
        self.min_verified_reliability = min_verified_reliability
        self.invoke = entry.schema["invoke"]
        self.output_schema = entry.schema["output"]

    def decide(self, x: Any) -> Any:
        if not isinstance(x, dict) or x.get("domain") != self.entry.owner:
            return ESCALATE
        if self.calibration.lower_bound < self.min_verified_reliability:
            return ESCALATE
        output = self.invoke(x)
        if not _matches_output_schema(output, self.output_schema):
            raise ValueError(f"catalog entry {self.entry.id!r} returned output outside its declared schema")
        verdict = self.entry.verifier.verify(
            claim={"payload": output},
            context={"request": x, "entry": self.entry.id},
        )
        return output if _verdict_passed(verdict) else ESCALATE


def build_catalog_router(
    catalog: list[CatalogEntry],
    teacher: Any,
    *,
    teacher_cost: float,
    calibration: Mapping[str, CatalogCalibration],
    min_verified_reliability: float,
) -> Router:
    """Build executable, schema-checked, verified tiers and one explicitly-priced frontier.

    ``calibration`` supplies held-out verifier outcomes for every entry. A local tier is eligible only
    when its one-sided reliability lower bound clears ``min_verified_reliability``; the entry's
    caller-supplied ``reliability`` prior is documentary and never authorizes an answer.
    """
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("catalog must be a non-empty list")
    if not callable(teacher):
        raise TypeError("teacher must be callable")
    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must map catalog ids to CatalogCalibration values")
    threshold = _finite_real(
        min_verified_reliability,
        "min_verified_reliability",
        minimum=0.0,
        maximum=1.0,
    )
    frontier_cost = _finite_real(teacher_cost, "teacher_cost", minimum=0.0, maximum=float("inf"))
    entries: list[CatalogEntry] = []
    ids: set[str] = set()
    for i, entry in enumerate(catalog):
        if not isinstance(entry, CatalogEntry):
            raise TypeError(f"catalog item {i} must be a CatalogEntry")
        if entry.id == "frontier" or entry.id in ids:
            raise ValueError(f"duplicate or reserved catalog id {entry.id!r}")
        ids.add(entry.id)
        if entry.cost <= 0.0:
            raise ValueError(f"catalog entry {entry.id!r} must declare a positive per-request cost")
        invoke = entry.schema.get("invoke")
        output_schema = entry.schema.get("output")
        if not callable(invoke):
            raise TypeError(f"catalog entry {entry.id!r} must declare callable schema['invoke']")
        _validate_output_schema(output_schema, path=f"{entry.id}.output")
        if not callable(getattr(entry.verifier, "verify", None)):
            raise TypeError(f"catalog entry {entry.id!r} must provide an executable verifier")
        evidence = calibration.get(entry.id)
        if not isinstance(evidence, CatalogCalibration):
            raise ValueError(f"catalog entry {entry.id!r} is missing a CatalogCalibration")
        entries.append(entry)
    extra_calibration = set(calibration) - ids
    if extra_calibration:
        raise ValueError(f"calibration contains unknown catalog ids: {sorted(extra_calibration)!r}")
    max_local_cost = max(entry.cost for entry in entries)
    if frontier_cost < max_local_cost:
        raise ValueError("teacher_cost must be at least the most expensive catalog tier")

    ordered = sorted(entries, key=lambda entry: (entry.cost, entry.id))
    tiers: list[tuple[str, Any, float]] = [
        (
            entry.id,
            _VerifiedToolGate(
                entry,
                calibration=calibration[entry.id],
                min_verified_reliability=threshold,
            ),
            entry.cost,
        )
        for entry in ordered
    ]
    tiers.append(("frontier", teacher, frontier_cost))
    return Router(tiers)
