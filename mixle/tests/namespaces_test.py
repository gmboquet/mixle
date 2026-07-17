"""The concern-oriented namespace structure resolves and re-exports faithfully (no behavior change)."""

import importlib
from unittest.mock import patch

import pytest


def test_object_namespaces_alias_the_families():
    import mixle

    gauss = importlib.import_module("mixle.stats.univariate.continuous.gaussian").GaussianDistribution
    assert mixle.dist.GaussianDistribution is gauss  # mixle.dist aliases mixle.stats
    assert mixle.process.HawkesProcessDistribution.__name__ == "HawkesProcessDistribution"
    # a Markov chain is a distribution, not a graph — it lives in the mixle.dist umbrella
    assert mixle.dist.MarkovChainDistribution.__name__ == "MarkovChainDistribution"
    assert not hasattr(mixle, "graph")  # mixle.graph was dropped (minimal namespaces)
    assert "GaussianDistribution" in mixle.dist.__all__
    # generic / applied models (GPs, neural nets, forests) are their own object namespace
    assert mixle.models.GaussianProcessRegressor.__name__ == "GaussianProcessRegressor"
    assert mixle.models.RandomForestEstimator.__name__ == "RandomForestEstimator"


def test_concern_namespaces_gather_each_concern():
    import mixle

    assert callable(mixle.enumeration.supports_enumeration) and hasattr(mixle.enumeration, "Enumerable")
    assert callable(mixle.enumeration.density_rank)
    assert callable(mixle.inference.conjugate_posterior) and callable(mixle.inference.optimize)
    assert callable(mixle.ppl.loo_stack)
    assert callable(mixle.ops.quantize)
    parallel = importlib.import_module("mixle.utils.parallel")
    assert callable(parallel.plan) and callable(parallel.encoded_data)
    assert hasattr(parallel, "Resources")
    utils = importlib.import_module("mixle.utils")
    assert callable(utils.analyze_structure)
    assert callable(utils.htsne)


def test_contracts_gathers_every_contract_in_one_import():
    from mixle.contracts import (  # eager: cast + capabilities  # lazy: subsystem roles (resolved via __getattr__)
        ComputeEngine,
        Conditionable,
        Distribution,
        Enumerable,
        Relation,
        Surrogate,
    )

    for c in (Distribution, Enumerable, Conditionable, Relation, ComputeEngine, Surrogate):
        assert isinstance(c, type)
    # the subsystem roles really come from their home modules
    assert Relation is importlib.import_module("mixle.relations").Relation
    assert Surrogate is importlib.import_module("mixle.doe._contracts").Surrogate


def test_pysp_dir_advertises_the_namespaces():
    import mixle

    for ns in ("dist", "process", "models", "enumeration", "inference", "ops", "contracts"):
        assert ns in dir(mixle)


def test_lazy_import_only_translates_a_missing_requested_module():
    import mixle

    requested = ModuleNotFoundError("missing requested module", name="mixle.not_present")
    with patch("importlib.import_module", side_effect=requested):
        with pytest.raises(AttributeError):
            mixle.__getattr__("not_present")

    nested = ModuleNotFoundError("missing nested dependency", name="required_dependency")
    with patch("importlib.import_module", side_effect=nested):
        with pytest.raises(ModuleNotFoundError, match="nested dependency"):
            mixle.__getattr__("broken")
