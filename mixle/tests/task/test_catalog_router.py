"""M3 DoD: IC-10 catalog entries wire into one calibrated Router tier per entry, ascending cost,
teacher/frontier last -- and a low-reliability entry escalates rather than answering."""

from __future__ import annotations

import dataclasses

import pytest

from mixle.task.catalog_router import CatalogEntry, build_catalog_router


class _StubVerifier:
    def verify(self, claim, context):
        return {"passed": True, "score": 1.0, "reasons": [], "kind": "exact"}


def _catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            id="physics_survey", schema={}, owner="physics", cost=0.2, reliability=0.9, verifier=_StubVerifier()
        ),
        CatalogEntry(
            id="economic_model", schema={}, owner="economic", cost=0.1, reliability=0.85, verifier=_StubVerifier()
        ),
        CatalogEntry(
            id="climate_projection", schema={}, owner="climate", cost=0.3, reliability=0.2, verifier=_StubVerifier()
        ),
    ]


def _teacher(texts):
    return [{"domain": "frontier", "for": (t.get("domain") if isinstance(t, dict) else t)} for t in texts]


def test_entry_fields_match_ic10_names_and_order():
    names = [f.name for f in dataclasses.fields(CatalogEntry)]
    assert names == ["id", "schema", "owner", "cost", "reliability", "verifier"]


def test_entry_is_immutable():
    entry = CatalogEntry(id="x", schema={}, owner="external")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.cost = 2.0


def test_one_tier_per_entry_ascending_cost_teacher_last():
    router = build_catalog_router(_catalog(), _teacher)
    assert len(router.tiers) == len(_catalog()) + 1

    costs = [cost for _, _, cost in router.tiers[:-1]]
    assert costs == sorted(costs)
    assert [name for name, _, _ in router.tiers[:-1]] == ["economic_model", "physics_survey", "climate_projection"]

    name, model, cost = router.tiers[-1]
    assert name == "frontier"
    assert model is _teacher
    assert cost > max(costs)


def test_matching_reliable_entry_answers_without_escalating():
    router = build_catalog_router(_catalog(), _teacher)
    label = router({"domain": "physics"})
    assert label == "physics_survey"

    tier_by_name = {t["tier"]: t["answered"] for t in router.report()["tiers"]}
    assert tier_by_name["physics_survey"] == 1
    assert tier_by_name["economic_model"] == 0
    assert tier_by_name["climate_projection"] == 0


def test_low_reliability_entry_escalates_to_frontier():
    router = build_catalog_router(_catalog(), _teacher)
    result = router({"domain": "climate"})
    # climate_projection's reliability (0.2) is below the adapter's gate -- it must escalate rather
    # than answer, and the frontier teacher gets the whole request (batched, per Router.__call__).
    assert result == {"domain": "frontier", "for": "climate"}
    assert router.report()["harvested_labels"] == 1
