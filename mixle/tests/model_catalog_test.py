"""Package-level model catalog compatibility and maturity policy."""

from __future__ import annotations

import importlib

import mixle.models as models


def test_public_model_module_tiers_are_disjoint_and_importable():
    supported = set(models.SUPPORTED_MODEL_MODULES)
    experimental = set(models.EXPERIMENTAL_MODEL_MODULES)
    assert supported
    assert experimental
    assert supported.isdisjoint(experimental)
    for module_name in sorted(supported | experimental):
        imported = importlib.import_module(f"mixle.models.{module_name}")
        assert imported is not None


def test_major_experimental_surfaces_are_deliberate_package_exports():
    expected = {
        "CoarsenResult",
        "CoarseningMetrics",
        "CompressedModel",
        "CompressionReceipt",
        "EvalReport",
        "SparseGaussianProcessRegressor",
        "SelectiveRecomputePolicy",
        "CompressedAdam",
        "NeuralOptimizerPlan",
        "EMATeacher",
        "propagate_moments",
        "evaluate_checkpoint",
        "compress",
        "coarsen",
    }
    assert expected <= set(models.__all__)
    for name in expected:
        assert getattr(models, name) is not None


def test_every_catalog_export_is_bound():
    missing = [name for name in models.__all__ if not hasattr(models, name)]
    assert missing == []
