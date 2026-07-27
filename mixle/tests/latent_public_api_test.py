"""Compatibility manifest for the focused latent-variable namespace."""

import mixle.stats as stats
import mixle.stats.latent as latent


def test_latent_manifest_is_unique_complete_and_resolvable():
    manifest = latent.latent_api_manifest()
    names = [entry.name for entry in manifest]

    assert names == latent.__all__
    assert len(names) == len(set(names))
    assert {entry.status for entry in manifest} == {"stable"}
    assert all(entry.source.startswith("mixle.stats.latent") for entry in manifest)
    assert all(entry.lazy == (entry.source != latent.__name__) for entry in manifest)
    assert all(getattr(latent, name) is not None for name in names)


def test_consolidated_namespace_resolves_every_latent_symbol():
    missing = [name for name in latent.__all__ if not hasattr(stats, name)]

    assert missing == []
    for name in latent.__all__:
        assert getattr(stats, name) is getattr(latent, name)


def test_curated_stats_surface_includes_latent_capability_families():
    required = {
        "LatentAPIEntry",
        "latent_api_manifest",
        "GaussianMixtureDistribution",
        "HierarchicalNormalDistribution",
        "PCFGTerminationCertificate",
        "DeterminizedSequenceDistribution",
        "LabeledLDADistribution",
        "LDAOptimizationDiagnostics",
        "LookbackHiddenMarkovModelDistribution",
        "ProbabilisticCircuitDistribution",
        "QuantizedHMMFitDiagnostics",
        "SemiSupervisedHiddenMarkovModelDistribution",
        "SparseScore",
        "collapse_identical",
        "PhaseSchedule",
        "TransitionOperator",
        "fit_chunked",
    }

    assert required <= set(stats.__all__)


def test_framework_plumbing_is_importable_but_not_in_curated_star_surface():
    for name in (
        "GaussianMixtureAccumulatorFactory",
        "HeterogeneousPCFGSampler",
        "IOHMMDataEncoder",
        "LabeledLDAAccumulator",
        "VariationalEmbeddingAttentionAccumulatorFactory",
    ):
        assert hasattr(stats, name)
        assert name not in stats.__all__
