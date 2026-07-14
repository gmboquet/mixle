"""M3 DoD: a compound query decomposes into typed prerequisite gaps across >=3 catalog domains,
resolves seeded-answerable gaps with real item ids, leaves unmatched/unverified gaps open and
explicit, and never lets a tool's free text become a canonical delta item."""

from __future__ import annotations

from mixle.task.catalog_router import CatalogEntry
from mixle.task.knowledge_routing import research_proposal_to_gap, route_task
from mixle.task.task_decomposition import init_decomposition_proposer

QUESTION = "What is the projected copper tonnage under baseline economics and RCP4.5 climate?"


class _PassVerifier:
    def verify(self, claim, context):
        return {"passed": True, "score": 1.0, "reasons": ["matches expected structure"], "kind": "physical"}


class _FailVerifier:
    def verify(self, claim, context):
        return {"passed": False, "score": 0.0, "reasons": ["implausible free text"], "kind": "physical"}


def _physics_tool(gap):
    return {"tonnage_mt": 12.5, "units": "Mt"}


def _economic_tool(gap):
    return {"npv_usd": 4.2e7}


def _climate_tool(gap):
    return {"precip_delta_pct": -8.0}


def _rumor_tool(gap):
    return "probably fine, trust me"  # free-form prose -- must never become canonical


def _catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            id="physics_survey",
            schema={"invoke": _physics_tool, "output": {"properties": {"tonnage_mt": {"type": "number"}}}},
            owner="physics",
            cost=0.2,
            reliability=0.9,
            verifier=_PassVerifier(),
        ),
        CatalogEntry(
            id="economic_model",
            schema={"invoke": _economic_tool, "output": {"properties": {"npv_usd": {"type": "number"}}}},
            owner="economic",
            cost=0.1,
            reliability=0.85,
            verifier=_PassVerifier(),
        ),
        CatalogEntry(
            id="climate_projection",
            schema={"invoke": _climate_tool, "output": {"properties": {"precip_delta_pct": {"type": "number"}}}},
            owner="climate",
            cost=0.3,
            reliability=0.8,
            verifier=_PassVerifier(),
        ),
        CatalogEntry(
            id="rumor_mill",
            schema={"invoke": _rumor_tool},
            owner="rumor",
            cost=0.05,
            reliability=0.99,
            verifier=_FailVerifier(),
        ),
    ]


def _proposer():
    # Seeded/heuristic decomposition (pre-M5, per the M3 non-goals): a repeated seed corpus so the
    # fitted Markov chain reliably proposes the same ordered sub-task domains for this question.
    seed_decompositions = [["physics", "economic", "climate"]] * 20
    return init_decomposition_proposer(seed_decompositions)


def test_compound_query_routes_at_least_three_catalog_ids_and_creates_typed_gaps():
    result = route_task(QUESTION, _catalog(), proposer=_proposer(), budget=8)

    assert len(result.answer["catalog_ids_considered"]) >= 3
    assert {"physics_survey", "economic_model", "climate_projection"} <= set(result.answer["catalog_ids_considered"])

    add_gaps = result.delta["add_gaps"]
    assert len(add_gaps) >= 3
    for gap in add_gaps:
        assert gap["required_schema"].get("domain")  # typed, not a hidden prose intermediate
        assert gap["acceptance_criteria"]

    resolved_ids = set(result.answer["resolved_gap_ids"])
    assert len(resolved_ids) >= 3
    assert result.delta["add_items"]
    for item in result.delta["add_items"]:
        assert item["content_hash"]
        assert isinstance(item["payload"], dict)


def test_seeded_bundle_item_resolves_its_gap_without_a_tool_call():
    seeded_item = {
        "id": "geo-report-1",
        "kind": "artifact",
        "modality": "structured",
        "schema_uri": "mixle://schema/typed-table/1",
        "content_hash": "a" * 64,
        "payload": {"assay": "cu"},
        "metadata": {"domain": "geology"},
    }
    seed_gap = {
        "id": "gap-seed-geology",
        "question": "what is the historical assay grade",
        "required_schema": {"type": "object", "domain": "geology"},
        "acceptance_criteria": ["a verified geology item resolves this gap"],
        "status": "open",
        "priority": 50,
        "owner": None,
        "attempts": [],
        "resolved_by_item_ids": [],
    }
    bundle = {"id": "bundle-1", "revision": 3, "items": [seeded_item], "gaps": [seed_gap]}

    result = route_task(QUESTION, _catalog(), proposer=_proposer(), budget=8, bundle=bundle)

    assert "gap-seed-geology" in result.answer["resolved_gap_ids"]
    seed_updates = [u for u in result.delta["gap_updates"] if u["gap_id"] == "gap-seed-geology"]
    assert seed_updates and seed_updates[0]["resolved_by_item_ids"] == ["geo-report-1"]
    # the pre-existing bundle item resolved it directly -- no new item was manufactured for it
    assert not any("gap-seed-geology" in item["id"] for item in result.delta["add_items"])


def test_unmatched_and_unverified_gaps_remain_open_and_free_text_never_becomes_a_delta():
    bundle = {
        "gaps": [
            {
                "id": "gap-seed-hydrology",
                "question": "what is the water table depth",
                "required_schema": {"type": "object", "domain": "hydrology"},
                "acceptance_criteria": ["a verified hydrology item resolves this gap"],
                "status": "open",
                "priority": 50,
                "owner": None,
                "attempts": [],
                "resolved_by_item_ids": [],
            },
            {
                "id": "gap-seed-rumor",
                "question": "any word on the permit?",
                "required_schema": {"type": "object", "domain": "rumor"},
                "acceptance_criteria": ["a verified rumor item resolves this gap"],
                "status": "open",
                "priority": 50,
                "owner": None,
                "attempts": [],
                "resolved_by_item_ids": [],
            },
        ]
    }

    result = route_task(QUESTION, _catalog(), proposer=_proposer(), budget=8, bundle=bundle)

    unresolved_ids = {g["id"] for g in result.remaining_gaps}
    assert {"gap-seed-hydrology", "gap-seed-rumor"} <= unresolved_ids

    hydrology_gap = next(g for g in result.remaining_gaps if g["id"] == "gap-seed-hydrology")
    assert hydrology_gap["attempts"][-1]["status"] == "no_matching_tool"  # no catalog entry for this domain

    rumor_gap = next(g for g in result.remaining_gaps if g["id"] == "gap-seed-rumor")
    assert rumor_gap["attempts"][-1]["status"] == "failed"  # tool ran, but its verifier rejected the output

    # the rumor tool's free-form prose result never became a canonical item/delta entry
    assert not any(item.get("metadata", {}).get("domain") == "rumor" for item in result.delta["add_items"])
    assert not any(isinstance(item["payload"], str) for item in result.delta["add_items"])


def test_research_proposal_to_gap_maps_into_the_frozen_gap_shape():
    from mixle.scientist import ResearchProposal

    proposal = ResearchProposal(
        question="how permeable is the tailings facility foundation?",
        missing="foundation permeability",
        nearest_knowledge=[{"score": 0.4, "text": "nearby borehole log"}],
        options=[{"how": "run a falling-head permeability test", "cost": 500.0}],
    )
    gap = research_proposal_to_gap(proposal, gap_id="gap-from-proposal-1")

    assert gap["id"] == "gap-from-proposal-1"
    assert gap["question"] == proposal.question
    assert gap["required_schema"]["description"] == "foundation permeability"
    assert gap["acceptance_criteria"] == ["run a falling-head permeability test"]
    assert gap["status"] == "open"
    assert gap["resolved_by_item_ids"] == []
