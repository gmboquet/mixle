"""Regression tests for the reviewed ``mixle.inference`` package boundary."""

from __future__ import annotations

import importlib

import mixle.inference as inference


def test_public_manifest_is_unique_complete_and_resolvable() -> None:
    manifest = inference.public_api_manifest()
    names = [entry.name for entry in manifest]

    assert names == inference.__all__
    assert len(names) == len(set(names))
    assert {entry.status for entry in manifest} == {"stable", "experimental"}
    assert all(entry.source.startswith("mixle.") for entry in manifest)
    for name in names:
        assert getattr(inference, name) is not None


def test_public_manifest_covers_primary_objective_and_mcmc_apis() -> None:
    names = set(inference.__all__)
    assert {
        "ObjectiveFitResult",
        "ObjectiveParameterSet",
        "fit_objective",
        "fit_parameter_objective",
        "variational_projection",
        "MCMCResult",
        "metropolis_hastings",
        "sample_conjugate_posterior",
        "sample_parameter_posterior",
    } <= names


def test_condition_function_has_a_collision_free_package_alias() -> None:
    condition_function = inference.condition_model
    condition_module = importlib.import_module("mixle.inference.condition")

    assert condition_function is condition_module.condition
    assert inference.condition is condition_module
    assert inference.condition_model is condition_function
