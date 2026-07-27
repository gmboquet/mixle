"""Compatibility manifest for the focused multivariate public namespace."""

import mixle.stats as stats
import mixle.stats.multivariate as multivariate


def test_multivariate_manifest_is_unique_and_resolvable() -> None:
    assert len(multivariate.__all__) == len(set(multivariate.__all__))
    missing = [name for name in multivariate.__all__ if not hasattr(multivariate, name)]
    assert missing == []


def test_consolidated_namespace_resolves_every_multivariate_symbol() -> None:
    missing = [name for name in multivariate.__all__ if not hasattr(stats, name)]
    assert missing == []
    for name in multivariate.__all__:
        assert getattr(stats, name) is getattr(multivariate, name)


def test_curated_stats_surface_includes_user_facing_multivariate_capabilities() -> None:
    required = {
        "closure",
        "clr",
        "clr_inv",
        "ilr",
        "ilr_inv",
        "ilr_basis",
        "AitchisonNormalDistribution",
        "AitchisonNormalEstimator",
        "MultinomialEncodedData",
        "DirichletMultinomialFitReceipt",
        "DirichletMultinomialResourceError",
        "PairCandidateEvidence",
        "PairSelectionReceipt",
        "VinePairFitError",
    }
    assert required <= set(stats.__all__)


def test_framework_plumbing_is_importable_but_not_in_curated_star_surface() -> None:
    for name in (
        "MultinomialAccumulator",
        "IntegerMultinomialAccumulatorFactory",
        "MultivariateGaussianDataEncoder",
        "GaussianCopulaSampler",
    ):
        assert hasattr(stats, name)
        assert name not in stats.__all__
