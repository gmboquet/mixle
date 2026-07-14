"""N2: critical-habitat & listed-species constraints into production (IC-9, IC-12, IC-1)."""

from __future__ import annotations

import numpy as np
import pytest
from mixle_knowledge.contracts import CriticalHabitatDesignation, ListedSpecies, SourceRef, SpatialBounds

from mixle.analysis.habitat_constraints import apply_habitat_constraints, critical_habitat_exclusion
from mixle.analysis.sdm import HabitatModel
from mixle.relations import min_cost_flow


def _habitat_model(mean_targets: np.ndarray, *, prior_dominated: bool = False) -> HabitatModel:
    """A trivial HabitatModel whose fitted intensity field is exactly ``mean_targets``: an identity
    design matrix makes ``mean = exp(design @ beta) = exp(beta)``, so ``beta = log(mean_targets)``."""
    k = mean_targets.shape[0]
    return HabitatModel(
        beta=np.log(mean_targets),
        beta_cov=np.eye(k) * 1.0e-8,
        design=np.eye(k),
        cell_area=np.ones(k),
        prior_dominated=prior_dominated,
    )


def _listed_species(*, critical_habitat: bool) -> ListedSpecies:
    return ListedSpecies(
        species_id="gopherus-agassizii",
        scientific_name="Gopherus agassizii",
        listing_status="ESA_threatened",
        jurisdiction="US-FWS",
        critical_habitat=critical_habitat,
        source=SourceRef(uri="mixle://document/fed-register-2011-16862#page=1"),
    )


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
