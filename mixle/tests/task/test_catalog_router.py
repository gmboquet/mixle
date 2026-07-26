"""M3 DoD: IC-10 catalog entries wire into one calibrated Router tier per entry, ascending cost,
teacher/frontier last -- and a low-reliability entry escalates rather than answering."""

from __future__ import annotations

import dataclasses

import pytest

from mixle.task.catalog_router import CatalogCalibration, CatalogEntry, build_catalog_router


class _StubVerifier:
    def verify(self, claim, context):
        return {"passed": True, "score": 1.0, "reasons": [], "kind": "exact"}


def _output(owner):
    def invoke(request):
        return {"source": owner, "domain": request["domain"]}

    return {
        "invoke": invoke,
        "output": {
            "type": "object",
            "required": ["source", "domain"],
            "properties": {
                "source": {"type": "string"},
                "domain": {"type": "string"},
            },
        },
    }


def _catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            id="physics_survey",
            schema=_output("physics"),
            owner="physics",
            cost=0.2,
            reliability=0.9,
            verifier=_StubVerifier(),
        ),
        CatalogEntry(
            id="economic_model",
            schema=_output("economic"),
            owner="economic",
            cost=0.1,
            reliability=0.85,
            verifier=_StubVerifier(),
        ),
        CatalogEntry(
            id="climate_projection",
            schema=_output("climate"),
            owner="climate",
            cost=0.3,
            reliability=0.99,
            verifier=_StubVerifier(),
        ),
    ]


def _teacher(texts):
    return [{"domain": "frontier", "for": (t.get("domain") if isinstance(t, dict) else t)} for t in texts]


def _calibration():
    return {
        "physics_survey": CatalogCalibration(verified=100, trials=100),
        "economic_model": CatalogCalibration(verified=100, trials=100),
        "climate_projection": CatalogCalibration(verified=2, trials=10),
    }


def _router(catalog=None, calibration=None):
    return build_catalog_router(
        _catalog() if catalog is None else catalog,
        _teacher,
        teacher_cost=1.0,
        calibration=_calibration() if calibration is None else calibration,
        min_verified_reliability=0.8,
    )


def test_entry_fields_match_ic10_names_and_order():
    names = [f.name for f in dataclasses.fields(CatalogEntry)]
    assert names == ["id", "schema", "owner", "cost", "reliability", "verifier"]


def test_entry_is_immutable():
    entry = CatalogEntry(id="x", schema={}, owner="external")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.cost = 2.0


def test_one_tier_per_entry_ascending_cost_teacher_last():
    router = _router()
    assert len(router.tiers) == len(_catalog()) + 1

    costs = [cost for _, _, cost in router.tiers[:-1]]
    assert costs == sorted(costs)
    assert [name for name, _, _ in router.tiers[:-1]] == ["economic_model", "physics_survey", "climate_projection"]

    name, model, cost = router.tiers[-1]
    assert name == "frontier"
    assert model is _teacher
    assert cost == 1.0


def test_matching_calibrated_entry_returns_verified_output_without_escalating():
    router = _router()
    label = router({"domain": "physics"})
    assert label == {"source": "physics", "domain": "physics"}

    tier_by_name = {t["tier"]: t["answered"] for t in router.report()["tiers"]}
    assert tier_by_name["physics_survey"] == 1
    assert tier_by_name["economic_model"] == 0
    assert tier_by_name["climate_projection"] == 0


def test_uncalibrated_entry_escalates_despite_a_high_prior():
    router = _router()
    result = router({"domain": "climate"})
    assert result == {"domain": "frontier", "for": "climate"}
    assert router.report()["harvested_labels"] == 1


def test_missing_domain_never_matches_every_entry():
    router = _router()
    assert router({"question": "ambiguous"}) == {"domain": "frontier", "for": None}
    assert router.report()["harvested_labels"] == 1


def test_schema_mismatch_is_recorded_as_degraded_and_escalates():
    catalog = _catalog()
    bad = catalog[0]
    catalog[0] = CatalogEntry(
        id=bad.id,
        schema={**bad.schema, "invoke": lambda _request: {"source": 3, "domain": "physics"}},
        owner=bad.owner,
        cost=bad.cost,
        reliability=bad.reliability,
        verifier=bad.verifier,
    )
    router = _router(catalog=catalog)
    assert router({"domain": "physics"}) == {"domain": "frontier", "for": "physics"}
    assert router.stats.degraded
    assert "outside its declared schema" in router.stats.degraded[0].reason


def test_router_requires_explicit_cost_calibration_schema_and_unique_ids():
    with pytest.raises(TypeError):
        build_catalog_router(
            _catalog(),
            _teacher,
            teacher_cost=1.0,
            calibration=_calibration(),
            min_verified_reliability=None,
        )
    with pytest.raises(ValueError, match="missing a CatalogCalibration"):
        _router(calibration={})
    with pytest.raises(ValueError, match="teacher_cost"):
        build_catalog_router(
            _catalog(),
            _teacher,
            teacher_cost=0.05,
            calibration=_calibration(),
            min_verified_reliability=0.8,
        )
    duplicate = [_catalog()[0], _catalog()[0]]
    with pytest.raises(ValueError, match="duplicate"):
        build_catalog_router(
            duplicate,
            _teacher,
            teacher_cost=1.0,
            calibration={"physics_survey": CatalogCalibration(100, 100)},
            min_verified_reliability=0.8,
        )
    malformed = [
        CatalogEntry(
            id="bad",
            schema={"invoke": lambda _request: {}, "output": {"properties": {}}},
            owner="physics",
            cost=0.1,
            reliability=0.5,
            verifier=_StubVerifier(),
        )
    ]
    with pytest.raises(ValueError, match="JSON type"):
        build_catalog_router(
            malformed,
            _teacher,
            teacher_cost=1.0,
            calibration={"bad": CatalogCalibration(100, 100)},
            min_verified_reliability=0.8,
        )


def test_entry_and_calibration_reject_nonfinite_or_impossible_values():
    with pytest.raises(ValueError):
        CatalogEntry(id="x", schema={}, owner="x", cost=float("nan"))
    with pytest.raises(ValueError):
        CatalogEntry(id="x", schema={}, owner="x", reliability=1.1)
    with pytest.raises(ValueError):
        CatalogCalibration(verified=2, trials=1)
