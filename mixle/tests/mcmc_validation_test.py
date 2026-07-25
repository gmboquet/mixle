"""Fail-closed validation and ordered diagnostic contracts for MCMC/SMC drivers."""

from unittest.mock import patch

import numpy as np
import pytest

from mixle.inference.mcmc import (
    MCMCResult,
    RandomWalkProposal,
    affine_invariant_ensemble,
    dense_mass_hmc,
    hamiltonian_monte_carlo,
    metropolis_hastings,
    nuts,
    particle_filter,
    reflective_hmc,
)


def _log_normal(value):
    x = np.asarray(value, dtype=float)
    return float(-0.5 * np.sum(x * x))


def _grad_log_normal(value):
    return -np.asarray(value, dtype=float)


@pytest.mark.parametrize("name,value", [("num_samples", -1), ("burn_in", -1), ("thin", 0), ("thin", 1.5)])
def test_ensemble_rejects_invalid_counts(name, value):
    kwargs = {"num_samples": 1, "burn_in": 0, "thin": 1}
    kwargs[name] = value
    with pytest.raises(ValueError):
        affine_invariant_ensemble(_log_normal, np.zeros((6, 2)), **kwargs)


def test_ensemble_enforces_documented_walker_bound_and_finite_geometry():
    with pytest.raises(ValueError, match=r"W >= 2\*d \+ 2"):
        affine_invariant_ensemble(_log_normal, np.zeros((4, 2)), num_samples=1)
    bad = np.zeros((6, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        affine_invariant_ensemble(_log_normal, bad, num_samples=1)


def test_all_core_drivers_reject_non_integral_or_nonfinite_controls_before_execution():
    proposal = RandomWalkProposal(scale=1.0)
    with pytest.raises(ValueError):
        metropolis_hastings(_log_normal, 0.0, proposal, num_samples=1.5)
    with pytest.raises(ValueError):
        hamiltonian_monte_carlo(_log_normal, _grad_log_normal, 0.0, 1, np.nan, 2)
    with pytest.raises(ValueError):
        reflective_hmc(_log_normal, _grad_log_normal, 0.0, -1.0, 1.0, 1, 0.1, 1, burn_in=-1)
    with pytest.raises(ValueError):
        reflective_hmc(_log_normal, _grad_log_normal, 0.0, -np.inf, 1.0, 1, 0.1, 1)
    with pytest.raises(ValueError):
        dense_mass_hmc(_log_normal, _grad_log_normal, 0.0, 1, 0.1, 2.5)
    with pytest.raises(ValueError):
        nuts(_log_normal, _grad_log_normal, 0.0, num_samples=1, target_accept=np.nan)
    with pytest.raises(ValueError):
        nuts(_log_normal, _grad_log_normal, 0.0, num_samples=1, max_tree_depth=0)


def test_particle_filter_rejects_impossible_evidence_instead_of_returning_nan_state():
    particles = np.zeros((4, 1))
    with pytest.raises(ValueError, match="zero probability"):
        particle_filter(
            [1],
            lambda state, rng: state,
            lambda state, observation: np.full(len(state), -np.inf),
            particles,
            rng=np.random.RandomState(0),
        )


def test_particle_filter_validates_particle_and_likelihood_shapes():
    with pytest.raises(ValueError, match=r"shape \(N, d\)"):
        particle_filter([], lambda state, rng: state, lambda state, obs: [], np.asarray([]))
    with pytest.raises(ValueError, match="log_likelihood"):
        particle_filter(
            [1],
            lambda state, rng: state,
            lambda state, observation: np.zeros(len(state) + 1),
            np.zeros((4, 1)),
        )


def test_dense_hmc_retains_accept_reject_decisions_in_transition_order():
    ordered = np.asarray([True, False, True], dtype=bool)
    warm = (np.empty((0, 1)), np.empty(0, dtype=bool))
    sampled = (np.asarray([[0.0], [1.0], [0.5]]), ordered)
    with patch("mixle.inference.mcmc.samplers._dense_hmc_run", side_effect=(warm, sampled)):
        result = dense_mass_hmc(_log_normal, _grad_log_normal, 0.0, 3, 0.1, 2, warmup=0)
    np.testing.assert_array_equal(result.accepted, ordered)


def test_mcmc_result_rejects_incomplete_or_nonfinite_artifacts():
    with pytest.raises(ValueError, match="one finite value"):
        MCMCResult(samples=[0.0], log_probs=np.asarray([]), accepted=np.asarray([True]))
    with pytest.raises(ValueError, match="one finite value"):
        MCMCResult(samples=[0.0], log_probs=np.asarray([np.nan]), accepted=np.asarray([True]))
    with pytest.raises(ValueError, match="transition label count"):
        MCMCResult(
            samples=[0.0],
            log_probs=np.asarray([0.0]),
            accepted=np.asarray([True]),
            transition_labels=(),
        )
