"""Dependency-free validation tests for critical-habitat network constraints."""

from types import SimpleNamespace

import numpy as np
import pytest

from mixle.analysis.habitat_constraints import apply_habitat_constraints, critical_habitat_exclusion
from mixle.analysis.sdm import HabitatModel

_SPECIES = "spotted-owl"


def _habitat(mean: np.ndarray) -> HabitatModel:
    # `species_id` is required: critical_habitat_exclusion matches each listed record to its OWN
    # fitted field by id and refuses to borrow another species' range (MXR-080-1591), so a model
    # without an identity cannot participate in a statutory exclusion at all.
    return HabitatModel(
        beta=np.log(mean),
        beta_cov=np.eye(mean.size) * 1.0e-8,
        design=np.eye(mean.size),
        cell_area=np.ones(mean.size),
        species_id=_SPECIES,
    )


def _listed() -> list[SimpleNamespace]:
    # `critical_habitat` must be an actual Boolean, not merely truthy (MXR-080-1588).
    return [SimpleNamespace(critical_habitat=True, species_id=_SPECIES)]


def _network() -> dict[str, np.ndarray]:
    return {
        "cap": np.ones((4, 4), dtype=float),
        "cost": np.zeros((4, 4), dtype=float),
        "supply": np.zeros(4, dtype=float),
    }


@pytest.mark.parametrize("buffer_cells", [-1, 1.5, True])
def test_exclusion_rejects_invalid_buffer_radius(buffer_cells):
    with pytest.raises(ValueError, match="buffer_cells"):
        critical_habitat_exclusion(
            _habitat(np.array([0.1, 2.0])),
            _listed(),
            suitability_cut=1.0,
            buffer_cells=buffer_cells,
        )


@pytest.mark.parametrize("suitability_cut", [-1.0, np.nan, np.inf, True, np.array([1.0])])
def test_exclusion_rejects_invalid_suitability_cut(suitability_cut):
    with pytest.raises(ValueError, match="suitability_cut"):
        critical_habitat_exclusion(_habitat(np.array([0.1, 2.0])), _listed(), suitability_cut=suitability_cut)


def test_exclusion_requires_one_dimensional_habitat_mean():
    malformed = SimpleNamespace(mean=np.ones((2, 1)), species_id=_SPECIES)
    with pytest.raises(ValueError, match="habitat.mean"):
        critical_habitat_exclusion(malformed, _listed(), suitability_cut=1.0)


def test_exclusion_returns_and_buffers_the_declared_boolean_cell_mask():
    result = critical_habitat_exclusion(
        _habitat(np.array([0.1, 2.0, 0.1])),
        _listed(),
        suitability_cut=1.0,
        buffer_cells=1,
    )
    np.testing.assert_array_equal(result, np.array([True, True, True]))


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array(True),
        np.array([True]),
        np.array([0, 1]),
    ],
)
def test_exclusion_requires_exact_boolean_result_shape(bad_mask):
    habitat = _habitat(np.array([0.1, 2.0]))
    habitat.critical_habitat_mask = lambda _cut: bad_mask
    with pytest.raises(ValueError, match="Boolean|shape"):
        critical_habitat_exclusion(habitat, _listed(), suitability_cut=1.0)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([0.2, np.nan, 0.0, 1.0]),
        np.array(True),
        np.array([[False, True, False, False]]),
    ],
)
def test_apply_rejects_nonbinary_or_nonvector_mask(mask):
    with pytest.raises(ValueError, match="exclusion_mask"):
        apply_habitat_constraints(_network(), mask)


def test_apply_accepts_an_exact_binary_mask():
    result = apply_habitat_constraints(_network(), np.array([0, 1, 0, 0]))
    assert result["forbidden_nodes"] == [1]


@pytest.mark.parametrize(
    "block_nodes",
    [
        np.array([True, False, True, False]),
        np.array([0, True, 2, 3], dtype=object),
    ],
)
def test_apply_rejects_boolean_node_identities(block_nodes):
    network = _network()
    network["block_nodes"] = block_nodes
    with pytest.raises(ValueError, match="Boolean"):
        apply_habitat_constraints(network, np.array([False, True, False, False]))
