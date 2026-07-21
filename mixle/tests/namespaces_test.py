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

    for ns in ("dist", "process", "models", "enumeration", "inference", "ops", "contracts", "stats", "utils"):
        assert ns in dir(mixle)


def test_dir_advertises_stats_and_utils_before_any_attribute_access():
    # __dir__ used to union globals() with only _NAMESPACES -- "stats"/"utils" are declared in
    # __all__ but aren't in _NAMESPACES (resolved via __getattr__ like every other lazy submodule), so
    # dir(mixle) omitted both until something ELSE had already triggered their lazy import as a side
    # effect (which auto-binds them into globals()). A subprocess is needed to see the state truly
    # before first access -- in-process, some earlier test has almost certainly already touched
    # mixle.stats/mixle.utils, which would mask the bug.
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import mixle; print('stats' in dir(mixle)); print('utils' in dir(mixle))"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"], result.stdout


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
