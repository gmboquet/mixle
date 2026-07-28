"""N2: critical-habitat & listed-species constraints into production (IC-9, IC-12, IC-1)."""

from __future__ import annotations

import numpy as np
import pytest

knowledge_contracts = pytest.importorskip(
    "mixle_knowledge.contracts",
    reason="cross-project habitat contracts require the optional mixle-knowledge package",
)
_N2_SYMBOLS = ("CriticalHabitatDesignation", "ListedSpecies", "SourceRef", "SpatialBounds")
_n2_missing = [s for s in _N2_SYMBOLS if not hasattr(knowledge_contracts, s)]
if _n2_missing:
    pytest.skip(f"installed mixle-knowledge lacks {', '.join(_n2_missing)}", allow_module_level=True)
CriticalHabitatDesignation = knowledge_contracts.CriticalHabitatDesignation
ListedSpecies = knowledge_contracts.ListedSpecies
SourceRef = knowledge_contracts.SourceRef
SpatialBounds = knowledge_contracts.SpatialBounds

# Imported from the top-level package, not the submodule: MXR-080-0093 found that
# critical_habitat_exclusion/apply_habitat_constraints were fully implemented but unreachable via
# `mixle.analysis`, exactly because nothing exercised that import path. Importing from `mixle.analysis`
# here, matching covariance_shrinkage_test.py/coverage_estimation_test.py/extreme_value_test.py/
# kde_test.py/kriging_test.py/rank_aggregation_test.py's convention, means a future regression on either
# name's export breaks collection of this whole file instead of silently going unnoticed again.
from mixle.analysis import apply_habitat_constraints, critical_habitat_exclusion
from mixle.analysis.sdm import HabitatModel
from mixle.relations import min_cost_flow

_SPECIES_ID = "gopherus-agassizii"


def _habitat_model(
    mean_targets: np.ndarray, *, prior_dominated: bool = False, species_id: str = _SPECIES_ID
) -> HabitatModel:
    """A trivial HabitatModel whose fitted intensity field is exactly ``mean_targets``: an identity
    design matrix makes ``mean = exp(design @ beta) = exp(beta)``, so ``beta = log(mean_targets)``."""
    k = mean_targets.shape[0]
    return HabitatModel(
        beta=np.log(mean_targets),
        beta_cov=np.eye(k) * 1.0e-8,
        design=np.eye(k),
        cell_area=np.ones(k),
        species_id=species_id,
        prior_dominated=prior_dominated,
    )


def _listed_species(*, critical_habitat: bool, species_id: str = _SPECIES_ID) -> ListedSpecies:
    return ListedSpecies(
        species_id=species_id,
        scientific_name="Gopherus agassizii",
        listing_status="ESA_threatened",
        jurisdiction="US-FWS",
        critical_habitat=critical_habitat,
        source=SourceRef(uri="mixle://document/fed-register-2011-16862#page=1"),
    )


def _reference_network() -> dict[str, np.ndarray]:
    """A 4-node reference network: 0 = source, 3 = sink; two parallel paths from 0 to 3, a cheap one
    through node 1 and an expensive detour through node 2 -- the same fixture
    test_apply_habitat_constraints_removes_exactly_enclosed_blocks_and_raises_cost uses, factored out for
    the MXR-080-0092 validation tests below, which need a fresh network dict per case."""
    quantity = 10.0
    cap = np.array(
        [
            [0.0, quantity, quantity, 0.0],
            [0.0, 0.0, 0.0, quantity],
            [0.0, 0.0, 0.0, quantity],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    cost = np.array(
        [
            [0.0, 1.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    supply = np.array([quantity, 0.0, 0.0, -quantity])
    return {"cap": cap, "cost": cost, "supply": supply}


def test_critical_habitat_exclusion_flags_exactly_the_high_suitability_block():
    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]))
    listed = [_listed_species(critical_habitat=True)]

    mask = critical_habitat_exclusion(habitat, listed, suitability_cut=1.0)

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.array([False, True, False, False]))
    # provenance: the excluded mask traces back to a listed-species record with a citation
    assert listed[0].critical_habitat is True
    assert listed[0].source.uri.startswith("mixle://document/")


def test_critical_habitat_exclusion_ignores_species_without_critical_habitat():
    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]))
    listed = [_listed_species(critical_habitat=False)]

    mask = critical_habitat_exclusion(habitat, listed, suitability_cut=1.0)

    assert not mask.any()


def test_critical_habitat_exclusion_buffers_conservatively():
    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1, 0.1]))
    listed = [_listed_species(critical_habitat=True)]

    mask = critical_habitat_exclusion(habitat, listed, suitability_cut=1.0, buffer_cells=1)

    np.testing.assert_array_equal(mask, np.array([True, True, True, False, False]))


def test_critical_habitat_exclusion_is_conservative_when_prior_dominated():
    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]), prior_dominated=True)
    listed = [_listed_species(critical_habitat=True)]

    mask = critical_habitat_exclusion(habitat, listed, suitability_cut=1.0)

    assert mask.all()  # not enough data to clear any block -- exclude everything, not nothing


def test_apply_habitat_constraints_removes_exactly_enclosed_blocks_and_raises_cost():
    # A 4-node reference network: 0 = source, 3 = sink; two parallel paths from 0 to 3, a cheap one
    # through node 1 (the critical-habitat block) and an expensive detour through node 2.
    quantity = 10.0
    cap = np.array(
        [
            [0.0, quantity, quantity, 0.0],
            [0.0, 0.0, 0.0, quantity],
            [0.0, 0.0, 0.0, quantity],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    cost = np.array(
        [
            [0.0, 1.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    supply = np.array([quantity, 0.0, 0.0, -quantity])
    network = {"cap": cap, "cost": cost, "supply": supply}

    baseline = min_cost_flow(cap, cost, supply)
    assert baseline.value == pytest.approx(2.0 * quantity)  # entirely routed through the cheap node-1 path

    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]))  # node 1 is the high-suitability block
    listed = [_listed_species(critical_habitat=True)]
    mask = critical_habitat_exclusion(habitat, listed, suitability_cut=1.0)

    payload = apply_habitat_constraints(network, mask)

    assert payload["forbidden_nodes"] == [1]  # exactly the one enclosed block, no more, no less
    assert np.all(payload["cap"][1, :] == 0.0)
    assert np.all(payload["cap"][:, 1] == 0.0)
    assert payload["supply"][1] == 0.0

    constrained = min_cost_flow(payload["cap"], payload["cost"], payload["supply"])

    assert constrained.value > baseline.value  # strictly higher cost ...
    assert constrained.value == pytest.approx(10.0 * quantity)  # ... forced onto the detour via node 2
    assert constrained.flow[2, 3] == pytest.approx(quantity)  # ... still feasible: full demand routed


# MXR-080-0092: network['block_nodes'] used to be cast with a bare `dtype=int` before any validation, so
# a fractional id silently truncated to the wrong node, a negative id silently exercised NumPy's
# from-the-end indexing instead of raising, and an out-of-range id raised a bare, internal IndexError only
# after an earlier, in-range id in the *same* call had already been applied to the local cap copy.
# network['supply'] and network['cap']/network['cost'] were not validated for shape or finiteness either.
# Every case below reproduces one of those failure modes against the fixed apply_habitat_constraints
# and confirms it is now rejected up front, before any exclusion is applied.


def test_apply_habitat_constraints_rejects_fractional_block_node_id():
    network = _reference_network()
    network["block_nodes"] = np.array([0.0, 2.7, 2.0, 3.0])  # block 1 claims node "2.7"
    mask = np.array([False, True, False, False])

    with pytest.raises(ValueError, match="exact integers"):
        apply_habitat_constraints(network, mask)


def test_apply_habitat_constraints_rejects_negative_block_node_id():
    network = _reference_network()
    network["block_nodes"] = np.array([0, -1, 2, 3])  # block 1 claims node "-1"
    mask = np.array([False, True, False, False])

    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        apply_habitat_constraints(network, mask)


def test_apply_habitat_constraints_rejects_out_of_range_block_node_id_before_any_mutation():
    network = _reference_network()
    # block 0 -> node 1 (in-range, would have been applied first under the old sorted-loop behavior);
    # block 1 -> node 1000 (out of range for this 4-node network).
    network["block_nodes"] = np.array([1, 1000, 2, 3])
    mask = np.array([True, True, False, False])
    cap_before = np.array(network["cap"], copy=True)

    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        apply_habitat_constraints(network, mask)

    # atomic: the caller's own cap array is completely untouched by the rejected call.
    np.testing.assert_array_equal(network["cap"], cap_before)


def test_apply_habitat_constraints_rejects_duplicate_block_node_ids():
    network = _reference_network()
    network["block_nodes"] = np.array([1, 1, 2, 3])  # blocks 0 and 1 both alias node 1
    mask = np.array([True, True, False, False])

    with pytest.raises(ValueError, match="duplicate"):
        apply_habitat_constraints(network, mask)


def test_apply_habitat_constraints_rejects_wrong_length_supply():
    network = _reference_network()
    network["supply"] = np.array([10.0, 0.0, -10.0])  # length 3; this network has 4 nodes
    mask = np.array([False, True, False, False])

    with pytest.raises(ValueError, match="length-4"):
        apply_habitat_constraints(network, mask)


def test_apply_habitat_constraints_rejects_non_finite_cap():
    network = _reference_network()
    network["cap"][0, 1] = np.nan
    mask = np.array([False, True, False, False])

    with pytest.raises(ValueError, match="finite"):
        apply_habitat_constraints(network, mask)


def test_apply_habitat_constraints_rejects_non_finite_cost():
    network = _reference_network()
    network["cost"][0, 1] = np.inf
    mask = np.array([False, True, False, False])

    with pytest.raises(ValueError, match="finite"):
        apply_habitat_constraints(network, mask)


def test_apply_habitat_constraints_rejects_negative_cap():
    network = _reference_network()
    network["cap"][0, 1] = -5.0
    mask = np.array([False, True, False, False])

    with pytest.raises(ValueError, match="nonnegative"):
        apply_habitat_constraints(network, mask)


def test_apply_habitat_constraints_accepts_exact_integer_valued_float_block_node_ids():
    # negative control: exact-integer-valued floats (2.0, not 2.7) are legitimate, and legitimate
    # unique in-range ids still correctly exclude exactly the intended node.
    network = _reference_network()
    network["block_nodes"] = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.array([False, True, False, False])

    payload = apply_habitat_constraints(network, mask)

    assert payload["forbidden_nodes"] == [1]
    assert np.all(payload["cap"][1, :] == 0.0)
    assert np.all(payload["cap"][:, 1] == 0.0)


def test_critical_habitat_designation_and_species_carry_source_provenance():
    designation = CriticalHabitatDesignation(
        species_id="gopherus-agassizii",
        bounds=SpatialBounds(crs="EPSG:32611", min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0),
        buffer_m=100.0,
        source=SourceRef(uri="mixle://document/fed-register-2011-16862#page=9"),
    )
    listed = _listed_species(critical_habitat=True)

    assert designation.species_id == listed.species_id
    assert designation.source.uri and listed.source.uri


# --------------------------------------------------------------------------------------------------
# MXR-080-1591: any listed record with a truthy critical-habitat flag caused the one supplied
# HabitatModel's mask to be applied, but the model carried no species identity and the function never
# compared one. A field fitted for species A could impose species B's statutory exclusion, and several
# listed species all collapsed onto species A's same field.
# --------------------------------------------------------------------------------------------------
_OTHER_SPECIES_ID = "athene-cunicularia"


def test_critical_habitat_exclusion_rejects_a_model_fitted_for_a_different_species():
    """Audit repro: species B's legal exclusion must not be imposed from species A's fitted field."""
    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]), species_id=_SPECIES_ID)
    listed = [_listed_species(critical_habitat=True, species_id=_OTHER_SPECIES_ID)]

    with pytest.raises(ValueError, match=_OTHER_SPECIES_ID):
        critical_habitat_exclusion(habitat, listed, suitability_cut=1.0)


def test_critical_habitat_exclusion_rejects_several_species_collapsing_onto_one_field():
    """Two listed species with one supplied field: the second has no model of its own and must not
    silently reuse the first species' range."""
    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]), species_id=_SPECIES_ID)
    listed = [
        _listed_species(critical_habitat=True, species_id=_SPECIES_ID),
        _listed_species(critical_habitat=True, species_id=_OTHER_SPECIES_ID),
    ]

    with pytest.raises(ValueError, match=_OTHER_SPECIES_ID):
        critical_habitat_exclusion(habitat, listed, suitability_cut=1.0)


def test_critical_habitat_exclusion_unions_only_matched_species_fields():
    """An explicit species-to-model mapping unions each species' OWN range, and nothing else."""
    tortoise = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]), species_id=_SPECIES_ID)
    owl = _habitat_model(np.array([0.1, 0.1, 5.0, 0.1]), species_id=_OTHER_SPECIES_ID)
    models = {_SPECIES_ID: tortoise, _OTHER_SPECIES_ID: owl}

    both = critical_habitat_exclusion(
        models,
        [
            _listed_species(critical_habitat=True, species_id=_SPECIES_ID),
            _listed_species(critical_habitat=True, species_id=_OTHER_SPECIES_ID),
        ],
        suitability_cut=1.0,
    )
    np.testing.assert_array_equal(both, np.array([False, True, True, False]))

    # only the owl is listed with critical habitat: the tortoise's block must NOT be excluded
    owl_only = critical_habitat_exclusion(
        models,
        [
            _listed_species(critical_habitat=False, species_id=_SPECIES_ID),
            _listed_species(critical_habitat=True, species_id=_OTHER_SPECIES_ID),
        ],
        suitability_cut=1.0,
    )
    np.testing.assert_array_equal(owl_only, np.array([False, False, True, False]))


def test_critical_habitat_exclusion_rejects_a_mapping_key_disagreeing_with_the_model():
    tortoise = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]), species_id=_SPECIES_ID)
    with pytest.raises(ValueError, match="does not match"):
        critical_habitat_exclusion(
            {_OTHER_SPECIES_ID: tortoise},
            [_listed_species(critical_habitat=True, species_id=_OTHER_SPECIES_ID)],
            suitability_cut=1.0,
        )


def test_critical_habitat_exclusion_matched_species_negative_control():
    """Negative control: the ordinary matched single-model case must be completely unaffected."""
    habitat = _habitat_model(np.array([0.1, 5.0, 0.1, 0.1]), species_id=_SPECIES_ID)
    mask = critical_habitat_exclusion(
        habitat, [_listed_species(critical_habitat=True, species_id=_SPECIES_ID)], suitability_cut=1.0
    )
    np.testing.assert_array_equal(mask, np.array([False, True, False, False]))
