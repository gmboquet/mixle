"""Slow acceptance sweep over the public distribution fixture catalog."""

import pytest

from mixle.experimental.typed_runtime import compile_update_graph
from mixle.tests.sampler_seed_test import _stats_public_distribution_catalog

pytestmark = [pytest.mark.experimental]


def test_every_public_distribution_fixture_compiles_without_execution():
    catalog = _stats_public_distribution_catalog()
    compiled = {name: compile_update_graph(model) for name, model in catalog.items()}

    assert compiled.keys() == catalog.keys()
    assert all(graph.nodes for graph in compiled.values())
    assert sum(len(graph.nodes) for graph in compiled.values()) >= len(catalog)
