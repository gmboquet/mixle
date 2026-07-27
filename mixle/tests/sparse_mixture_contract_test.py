"""Adversarial contracts for certified sparse-mixture operations."""

import numpy as np
import pytest

from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.latent.sparse_mixture import (
    collapse_gaussian_mixture,
    collapse_identical,
    sparse_mixture_score,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


def _mixture():
    return MixtureDistribution(
        [GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 2.0)],
        [0.4, 0.6],
    )


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, np.nan])
def test_sparse_score_requires_a_positive_exact_component_limit(limit):
    with pytest.raises((TypeError, ValueError)):
        sparse_mixture_score(_mixture(), 0.0, limit)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, np.nan])
def test_gaussian_collapse_requires_a_positive_exact_component_limit(limit):
    with pytest.raises((TypeError, ValueError)):
        collapse_gaussian_mixture(_mixture(), limit)


def test_display_text_collision_does_not_merge_distinct_laws(monkeypatch):
    monkeypatch.setattr(GaussianDistribution, "__str__", lambda self: "same display text")
    mixture = _mixture()

    collapsed = collapse_identical(mixture)

    assert len(collapsed.components) == 2
    for value in (-2.0, 0.0, 2.0, 5.0):
        assert collapsed.log_density(value) == pytest.approx(mixture.log_density(value))


def test_canonical_family_and_state_identity_still_merges_exact_duplicates():
    mixture = MixtureDistribution(
        [
            GaussianDistribution(0.0, 1.0),
            GaussianDistribution(0.0, 1.0),
            GaussianDistribution(2.0, 1.0),
        ],
        [0.2, 0.3, 0.5],
    )

    collapsed = collapse_identical(mixture)

    assert len(collapsed.components) == 2
    assert collapsed.w == pytest.approx([0.5, 0.5])
    for value in (-2.0, 0.0, 2.0, 5.0):
        assert collapsed.log_density(value) == pytest.approx(mixture.log_density(value))


def test_gaussian_collapse_removes_zero_weight_components_before_merging():
    mixture = MixtureDistribution(
        [
            GaussianDistribution(0.0, 1.0),
            GaussianDistribution(100.0, 1.0),
            GaussianDistribution(-100.0, 1.0),
        ],
        [1.0, 0.0, 0.0],
    )

    collapsed = collapse_gaussian_mixture(mixture, 1)

    assert len(collapsed.components) == 1
    assert collapsed.w == pytest.approx([1.0])
    assert collapsed.components[0].mu == 0.0


def test_sparse_score_rejects_invalid_weight_geometry():
    class InvalidMixture:
        components = [GaussianDistribution(0.0, 1.0)]
        log_w = np.array([0.0, -np.inf])

    with pytest.raises(ValueError, match="one entry per component"):
        sparse_mixture_score(InvalidMixture(), 0.0, 1)
