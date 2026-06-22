"""The concern-oriented namespace structure resolves and re-exports faithfully (no behavior change)."""

import importlib


def test_object_namespaces_alias_the_families():
    import pysp

    gauss = importlib.import_module("pysp.stats.leaf.gaussian").GaussianDistribution
    assert pysp.dist.GaussianDistribution is gauss  # pysp.dist aliases pysp.stats
    assert pysp.process.HawkesProcessDistribution.__name__ == "HawkesProcessDistribution"
    assert pysp.graph.MarkovChainDistribution.__name__ == "MarkovChainDistribution"
    assert "GaussianDistribution" in pysp.dist.__all__


def test_concern_namespaces_gather_each_concern():
    import pysp

    assert callable(pysp.enumeration.supports_enumeration) and hasattr(pysp.enumeration, "Enumerable")
    assert callable(pysp.sampling.sample) and hasattr(pysp.sampling, "LatentPosterior")
    assert callable(pysp.inference.conjugate_posterior) and callable(pysp.inference.optimize)
    assert callable(pysp.ops.quantize)


def test_contracts_gathers_every_contract_in_one_import():
    from pysp.contracts import (  # eager: cast + capabilities  # lazy: subsystem roles (resolved via __getattr__)
        ComputeEngine,
        Conditionable,
        Distribution,
        Enumerable,
        ForwardOperator,
        Relation,
        Surrogate,
    )

    for c in (Distribution, Enumerable, Conditionable, Relation, ComputeEngine, ForwardOperator, Surrogate):
        assert isinstance(c, type)
    # the subsystem roles really come from their home modules
    assert Relation is importlib.import_module("pysp.relations").Relation
    assert Surrogate is importlib.import_module("pysp.doe._contracts").Surrogate


def test_pysp_dir_advertises_the_namespaces():
    import pysp

    for ns in ("dist", "process", "graph", "enumeration", "sampling", "inference", "ops", "contracts"):
        assert ns in dir(pysp)
