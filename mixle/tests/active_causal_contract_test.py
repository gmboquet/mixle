"""Contract regressions for the active-causal experimental benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.active_causal import (
    LinearGaussianSCM,
    StructurePosterior,
    active_discovery,
    expected_information_gain,
    markov_equivalent_triple,
)


def test_markov_equivalent_candidates_have_the_same_observational_law() -> None:
    candidates = markov_equivalent_triple(weight=0.6, noise=1.3)
    covariances = [candidate.observational_covariance() for candidate in candidates]
    for covariance in covariances[1:]:
        np.testing.assert_allclose(covariance, covariances[0], atol=1e-12, rtol=1e-12)

    observations = np.random.default_rng(8).multivariate_normal(np.zeros(3), covariances[0], size=20)
    likelihoods = [candidate.log_likelihood(observations) for candidate in candidates]
    np.testing.assert_allclose(likelihoods, likelihoods[0], atol=1e-11, rtol=1e-12)

    posterior = StructurePosterior(candidates)
    posterior.update(observations, None)
    np.testing.assert_allclose(posterior.probs, np.full(3, 1.0 / 3.0), atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_nodes": 0, "parents": {}},
        {"n_nodes": 2, "parents": {0: [2]}},
        {"n_nodes": 2, "parents": {0: [0]}},
        {"n_nodes": 2, "parents": {0: [1], 1: [0]}},
        {"n_nodes": 2, "parents": {}, "noise": 0.0},
        {"n_nodes": 2, "parents": {}, "noise_scales": (1.0, -1.0)},
        {"n_nodes": 2, "parents": {1: [0]}, "edge_weights": {(1, 0): 0.4}},
    ],
)
def test_scm_rejects_invalid_graph_and_distribution_contracts(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LinearGaussianSCM("invalid", **kwargs)


def test_scm_validates_data_and_interventions() -> None:
    scm = LinearGaussianSCM("chain", 2, {1: [0]})
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        scm.simulate(0, rng)
    with pytest.raises(ValueError):
        scm.simulate(2, rng, (2, 1.0))
    with pytest.raises(ValueError):
        scm.simulate(2, rng, (0, np.inf))
    with pytest.raises(ValueError):
        scm.log_likelihood(np.zeros((2, 3)))

    samples = scm.simulate(3, rng, (0, 2.0))
    assert np.isfinite(scm.log_likelihood(samples, (0, 2.0)))
    samples[0, 0] = 1.0
    assert scm.log_likelihood(samples, (0, 2.0)) == -np.inf


def test_discovery_controls_fail_closed() -> None:
    candidates = markov_equivalent_triple()
    with pytest.raises(ValueError):
        active_discovery(candidates[0], candidates, strategy="typo")
    with pytest.raises(ValueError):
        active_discovery(candidates[0], candidates, threshold=0.0)
    with pytest.raises(ValueError):
        active_discovery(candidates[0], candidates, max_experiments=0)
    with pytest.raises(ValueError):
        expected_information_gain(
            StructurePosterior(candidates),
            None,
            n_batch=0,
            rng=np.random.default_rng(0),
        )
