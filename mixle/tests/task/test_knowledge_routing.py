"""M3 DoD: a compound query decomposes into typed prerequisite gaps across >=3 catalog domains,
resolves seeded-answerable gaps with real item ids, leaves unmatched/unverified gaps open and
explicit, and never lets a tool's free text become a canonical delta item."""

from __future__ import annotations

import numpy as np

from mixle.inference.calibration_gate import CalibrationVerifier, posterior_predictive_calibration
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


def test_a_gap_can_opt_into_seeing_another_gaps_real_resolved_result():
    """Fix for a real, documented limitation (found and worked around across several
    experiments/ demos by bridging through separate route_task calls): `invoke(gap)` had no way to
    see another gap's already-resolved result within the SAME route_task call. A tool that declares
    a `known_items` parameter now gets the real accumulated items -- not a value this test
    precomputes and hands over regardless of whether the system actually resolved the upstream gap."""
    physics_ref = {}

    def _physics_tool_recorded(gap):
        result = {"tonnage_mt": 12.5, "units": "Mt"}
        physics_ref.update(result)
        return result

    def _dependent_economic_tool(gap, known_items):
        physics_items = [i for i in known_items if i.get("metadata", {}).get("domain") == "physics"]
        assert physics_items, "known_items must contain the already-resolved physics item"
        tonnage = physics_items[0]["payload"]["tonnage_mt"]
        assert tonnage == physics_ref["tonnage_mt"]  # the REAL value the physics tool actually returned
        return {"npv_usd": tonnage * 1_000_000.0}

    catalog = [
        CatalogEntry(
            id="physics_survey",
            schema={"invoke": _physics_tool_recorded, "output": {}},
            owner="physics",
            cost=0.1,
            reliability=0.9,
            verifier=_PassVerifier(),
        ),
        CatalogEntry(
            id="dependent_economic_model",
            schema={"invoke": _dependent_economic_tool, "output": {}},
            owner="economic",
            cost=0.1,
            reliability=0.9,
            verifier=_PassVerifier(),
        ),
    ]
    seed = [["physics", "economic"]] * 20
    proposer = init_decomposition_proposer(seed)
    result = route_task("dependent question", catalog, proposer=proposer, budget=5)

    assert result.answer["resolved_gap_ids"] == ["gap-0-physics", "gap-1-economic"]
    economic_item = next(i for i in result.delta["add_items"] if i["metadata"]["domain"] == "economic")
    assert economic_item["payload"]["npv_usd"] == 12.5 * 1_000_000.0


def test_a_one_arg_invoke_is_completely_unaffected():
    """Every catalog entry written before this fix uses `invoke(gap)` -- must keep working exactly
    as before, with no known_items argument forced on it."""

    def _plain_tool(gap):
        return {"tonnage_mt": 7.0}

    catalog = [
        CatalogEntry(
            id="physics_survey",
            schema={"invoke": _plain_tool, "output": {}},
            owner="physics",
            cost=0.1,
            reliability=0.9,
            verifier=_PassVerifier(),
        )
    ]
    proposer = init_decomposition_proposer([["physics"]] * 20)
    result = route_task("plain question", catalog, proposer=proposer, budget=3)
    assert result.answer["resolved_gap_ids"] == ["gap-0-physics"]


def test_a_transient_tool_failure_is_retried_in_place_not_desynced_onto_later_gaps():
    """Regression test for a gap-skipping desync: `_RoutingWorld._cursor` used to advance on every
    `step()` call (including a call that raised), while `_sequential_plan` picked the next gap index
    from `len(history)`. `orchestrate`'s re-plan-on-failure path calls `world.step()` twice for one
    failed turn but appends to `history` only once, so the two counters fell out of sync: the failed
    gap was silently skipped forever, the gap after it got processed twice, and the last gap in the
    sequence was never attempted because `world.done` fired one call early.

    Four domains, ordered physics/economic/climate/geology by the seeded proposer. The economic tool
    raises on its first call and would succeed on a second -- a realistic transient failure -- so a
    correct implementation must retry that SAME gap (not skip to climate) and still go on to resolve
    climate and geology afterward."""
    economic_calls = {"n": 0}

    def _physics_tool(gap):
        return {"tonnage_mt": 12.5}

    def _economic_tool(gap):
        economic_calls["n"] += 1
        if economic_calls["n"] == 1:
            raise RuntimeError("transient upstream failure")
        return {"npv_usd": 4.2e7}

    def _climate_tool(gap):
        return {"precip_delta_pct": -8.0}

    def _geology_tool(gap):
        return {"assay_grade": 1.1}

    catalog = [
        CatalogEntry(
            id="physics_survey",
            schema={"invoke": _physics_tool, "output": {}},
            owner="physics",
            cost=0.1,
            reliability=0.9,
            verifier=_PassVerifier(),
        ),
        CatalogEntry(
            id="economic_model",
            schema={"invoke": _economic_tool, "output": {}},
            owner="economic",
            cost=0.1,
            reliability=0.9,
            verifier=_PassVerifier(),
        ),
        CatalogEntry(
            id="climate_projection",
            schema={"invoke": _climate_tool, "output": {}},
            owner="climate",
            cost=0.1,
            reliability=0.9,
            verifier=_PassVerifier(),
        ),
        CatalogEntry(
            id="geology_survey",
            schema={"invoke": _geology_tool, "output": {}},
            owner="geology",
            cost=0.1,
            reliability=0.9,
            verifier=_PassVerifier(),
        ),
    ]
    seed = [["physics", "economic", "climate", "geology"]] * 20
    proposer = init_decomposition_proposer(seed)

    result = route_task("needs physics, economic, climate, and geology evidence", catalog, proposer=proposer, budget=10)

    # the economic tool was actually retried (called twice), not silently abandoned after one failure
    assert economic_calls["n"] == 2

    # every gap resolved -- none silently skipped, none left unattempted because `done` fired early
    assert result.answer["unresolved_gap_ids"] == []
    assert result.answer["resolved_gap_ids"] == [
        "gap-0-physics",
        "gap-1-economic",
        "gap-2-climate",
        "gap-3-geology",
    ]
    assert set(result.answer["catalog_ids_considered"]) == {
        "physics_survey",
        "economic_model",
        "climate_projection",
        "geology_survey",
    }

    # each gap was resolved exactly once -- no gap got double-processed as a side effect of the desync
    resolved_gap_ids_in_updates = [u["gap_id"] for u in result.delta["gap_updates"] if u["status"] == "resolved"]
    assert sorted(resolved_gap_ids_in_updates) == sorted(set(resolved_gap_ids_in_updates))
    assert len(resolved_gap_ids_in_updates) == 4

    # exactly one add_items entry per gap -- no gap produced a duplicate item
    domains_produced = [item["metadata"]["domain"] for item in result.delta["add_items"]]
    assert sorted(domains_produced) == ["climate", "economic", "geology", "physics"]


# --- a low-power calibration verdict must not resolve a knowledge gap ---
#
# CalibrationVerdict.passed reads True for a low_power result BY DESIGN (a check that had no
# statistical power to catch a problem should not itself block on that account alone -- see
# mixle.inference.calibration_gate.CalibrationVerdict's docstring); CalibrationVerdict.low_power
# distinguishes "undetectable with this little data" from a real, well-powered pass. route_task must
# honor that distinction: a bare duck-typed `.passed` is not sufficient resolving evidence for an
# epistemic knowledge gap if the same verdict also reports `low_power=True`.


def _predictive_ensemble(truth_sd: float, ensemble_sd: float, *, k: int, m: int = 500, seed: int = 0):
    """Held-out truths drawn N(0, truth_sd^2); a predictive ensemble centered at 0 with ensemble_sd.
    Mirrors mixle/tests/calibration_gate_test.py's fixture of the same name."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, truth_sd, size=k)
    ensemble = rng.normal(0.0, ensemble_sd, size=(k, m))
    return ensemble, y


def _low_power_calibration_tool(gap):
    # k=6, ensemble_sd=0.3 vs truth_sd=1.0: deliberately overconfident, but so few held-out points
    # that even gross miscalibration is within the finite-sample null noise -- the exact fixture
    # calibration_gate_test.py::test_tiny_holdout_is_flagged_low_power_not_false_alarmed uses.
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3, k=6)
    return {"ensemble": ensemble.tolist(), "held_out_y": y.tolist()}


def _well_powered_calibration_tool(gap):
    # k=400, ensemble_sd == truth_sd: genuinely calibrated with ample power to have failed.
    ensemble, y = _predictive_ensemble(truth_sd=1.0, ensemble_sd=1.0, k=400)
    return {"ensemble": ensemble.tolist(), "held_out_y": y.tolist()}


def test_low_power_calibration_verdict_does_not_resolve_the_knowledge_gap():
    """The reported bug, reproduced end to end: a low-power (undetectable-with-this-data)
    calibration result must not close out a knowledge gap. Confirm the fixture really is in the
    low-power-but-nominally-passed regime at the calibration_gate level, then confirm route_task
    leaves the gap open rather than treating that bare pass as resolving evidence."""
    direct_verdict = posterior_predictive_calibration(*_predictive_ensemble(truth_sd=1.0, ensemble_sd=0.3, k=6))
    assert direct_verdict.calibration_status == "low_power"
    assert direct_verdict.passed  # by design -- see CalibrationVerdict's docstring
    assert direct_verdict.low_power

    catalog = [
        CatalogEntry(
            id="calibration_probe",
            schema={"invoke": _low_power_calibration_tool, "output": {}},
            owner="physics",
            cost=0.1,
            reliability=0.9,
            verifier=CalibrationVerifier(),
        )
    ]
    proposer = init_decomposition_proposer([["physics"]] * 20)
    result = route_task("is this posterior calibrated?", catalog, proposer=proposer, budget=3)

    assert result.answer["resolved_gap_ids"] == []
    assert result.answer["unresolved_gap_ids"] == ["gap-0-physics"]
    assert result.delta["add_items"] == []  # no item was promoted to canonical evidence
    assert result.delta["gap_updates"] == []

    gap = next(g for g in result.remaining_gaps if g["id"] == "gap-0-physics")
    assert gap["status"] == "open"
    assert gap["resolved_by_item_ids"] == []
    assert gap["attempts"][-1]["status"] == "low_power"  # distinct from a genuine "failed" rejection


def test_well_powered_calibration_verdict_still_resolves_the_knowledge_gap():
    """Companion to the low-power regression above: the fix must not over-correct into blocking a
    genuine, well-powered calibration pass from resolving its gap."""
    catalog = [
        CatalogEntry(
            id="calibration_probe",
            schema={"invoke": _well_powered_calibration_tool, "output": {}},
            owner="physics",
            cost=0.1,
            reliability=0.9,
            verifier=CalibrationVerifier(),
        )
    ]
    proposer = init_decomposition_proposer([["physics"]] * 20)
    result = route_task("is this posterior calibrated?", catalog, proposer=proposer, budget=3)

    assert result.answer["resolved_gap_ids"] == ["gap-0-physics"]
    assert result.answer["unresolved_gap_ids"] == []
    assert result.delta["add_items"]
    assert result.delta["gap_updates"][0]["status"] == "resolved"
