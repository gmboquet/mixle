"""Focused release-contract probes for global sensitivity estimators."""

from __future__ import annotations

import numpy as np
import pytest

import mixle.doe.sensitivity as sensitivity_module
from mixle.doe.sensitivity import dgsm, fast_indices, sobol_indices


def _linear(x):
    return x[:, 0] + 2.0 * x[:, 1]


def test_model_output_preserves_leading_sample_axis_before_normalization():
    with pytest.raises(ValueError, match="leading sample axis"):
        sobol_indices(lambda x: _linear(x)[None, :], [(0.0, 1.0)] * 2, n=8, n_bootstrap=4)


def test_singleton_scalar_output_axis_is_explicitly_supported():
    result = sobol_indices(lambda x: _linear(x)[:, None], [(0.0, 1.0)] * 2, n=16, n_bootstrap=4)
    assert result["S1"].shape == (2,)


def test_one_resample_cannot_claim_a_bootstrap_standard_error():
    with pytest.raises(ValueError, match="n_bootstrap"):
        sobol_indices(_linear, [(0.0, 1.0)] * 2, n=8, n_bootstrap=1)


@pytest.mark.parametrize(
    "estimator,kwargs,match",
    [
        (sobol_indices, {"n": 16, "n_bootstrap": 4}, "variance|S1|ST"),
        (fast_indices, {"n": 64, "harmonics": 2, "n_bootstrap": 4}, "variance|spectrum"),
        (dgsm, {"n": 16}, "derivative|nu"),
    ],
)
def test_finite_model_outputs_cannot_overflow_into_sensitivity_evidence(estimator, kwargs, match):
    def extreme(x):
        return 1e308 * x[:, 0]

    with pytest.raises(ValueError, match=match):
        estimator(extreme, [(0.0, 1.0)], **kwargs)


@pytest.mark.parametrize("rel_step", [0.0, -1e-4, np.nan, True, np.array([1e-4])])
def test_dgsm_requires_a_finite_positive_scalar_step(rel_step):
    with pytest.raises(ValueError, match="rel_step"):
        dgsm(_linear, [(0.0, 1.0)] * 2, n=8, rel_step=rel_step)


def test_fast_preserves_raw_estimate_and_reports_uncertainty():
    result = fast_indices(
        lambda x: x[:, 0],
        [(0.0, 1.0)] * 2,
        n=128,
        harmonics=3,
        seed=0,
        n_bootstrap=20,
    )
    assert result["S1"][1] < 0.0
    assert result["S1_clipped"][1] == 0.0
    assert np.isfinite(result["S1_standard_error"]).all()
    assert (result["S1_standard_error"] >= 0.0).all()
    assert (result["S1_ci_low"] <= result["S1_ci_high"]).all()
    assert result["uncertainty_method"] == "circular_block_bootstrap"


def test_sobol_sampler_discloses_method_and_does_not_swallow_internal_failures(monkeypatch):
    result = sobol_indices(_linear, [(0.0, 1.0)] * 2, n=8, n_bootstrap=4)
    assert result["sampling_method"] == "scrambled_sobol"

    def broken_qmc(*args, **kwargs):
        raise RuntimeError("QMC implementation failure")

    monkeypatch.setattr(sensitivity_module, "_qmc_unit", broken_qmc)
    with pytest.raises(RuntimeError, match="QMC implementation failure"):
        sobol_indices(_linear, [(0.0, 1.0)] * 2, n=8, n_bootstrap=4)
