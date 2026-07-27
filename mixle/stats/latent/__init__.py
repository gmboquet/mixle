"""Public latent-variable probability-model namespace.

The focused namespace exposes fitted families, estimators, samplers, enumerators,
encoders, accumulators, operators, schedules, diagnostics, and high-level helpers.
Exports are lazy so importing :mod:`mixle.stats.latent` does not initialize every
optional numerical backend. The curated non-plumbing subset is also available from
:mod:`mixle.stats`; integration roles remain directly importable from either package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LatentAPIEntry:
    """One immutable entry in the reviewed latent-package export manifest."""

    name: str
    source: str
    status: Literal["stable", "provisional"]
    lazy: bool = True


_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "mixle.stats.latent.chained_attention": (
        "ChainedAttentionDistribution",
        "ChainedAttentionSampler",
        "ChainedAttentionAccumulator",
        "ChainedAttentionAccumulatorFactory",
        "ChainedAttentionEstimator",
        "ChainedAttentionDataEncoder",
    ),
    "mixle.stats.latent.dirac_length": (
        "DiracLengthMixtureDistribution",
        "DiracLengthMixtureEnumerator",
        "DiracLengthMixtureSampler",
        "DiracLengthMixtureAccumulator",
        "DiracLengthMixtureAccumulatorFactory",
        "DiracLengthMixtureEstimator",
        "DiracLengthMixtureDataEncoder",
    ),
    "mixle.stats.latent.effective_sample": (
        "EffectiveSampleReceipt",
        "validated_positive_integer",
        "validated_observation_weight",
        "validated_observation_weights",
        "validated_count_array",
        "validated_statistic_tuple",
        "validate_effective_sample_mass",
        "validated_weighted_responsibilities",
    ),
    "mixle.stats.latent.gated_mixture": (
        "GateOptimizationReceipt",
        "GateBufferReceipt",
        "SoftmaxGate",
        "GatedMixtureDistribution",
        "GatedMixtureSampler",
        "GatedMixtureDataEncoder",
        "GatedMixtureAccumulator",
        "GatedMixtureAccumulatorFactory",
        "GatedMixtureEstimator",
    ),
    "mixle.stats.latent.gaussian_mixture": (
        "GaussianMixtureDistribution",
        "GaussianMixtureSampler",
        "GaussianMixtureAccumulator",
        "GaussianMixtureAccumulatorFactory",
        "GaussianMixtureEstimator",
        "GaussianMixtureDataEncoder",
    ),
    "mixle.stats.latent.heterogeneous_mixture": (
        "HeterogeneousMixtureDistribution",
        "HeterogeneousMixtureEnumerator",
        "HeterogeneousMixtureSampler",
        "HeterogeneousMixtureAccumulator",
        "HeterogeneousMixtureAccumulatorFactory",
        "HeterogeneousMixtureEstimator",
        "HeterogeneousMixtureDataEncoder",
    ),
    "mixle.stats.latent.heterogeneous_pcfg": (
        "PCFGTerminationCertificate",
        "HeterogeneousPCFGDistribution",
        "HeterogeneousPCFGEnumerator",
        "HeterogeneousPCFGSampler",
        "HeterogeneousPCFGAccumulator",
        "HeterogeneousPCFGAccumulatorFactory",
        "HeterogeneousPCFGEstimator",
        "InducedHeterogeneousPCFGEstimator",
        "HeterogeneousPCFGDataEncoder",
        "HeterogeneousPCFGFisherView",
    ),
    "mixle.stats.latent.hidden_association": (
        "HiddenAssociationDistribution",
        "HiddenAssociationEnumerator",
        "HiddenAssociationSampler",
        "HiddenAssociationAccumulator",
        "HiddenAssociationAccumulatorFactory",
        "HiddenAssociationEstimator",
        "HiddenAssociationDataEncoder",
    ),
    "mixle.stats.latent.hidden_markov": (
        "hmm_dirichlet_default_prior",
        "terminal_forward_loglik",
        "terminal_forward_backward",
        "HiddenMarkovModelDistribution",
        "HiddenMarkovModelEnumerator",
        "HiddenMarkovSampler",
        "HiddenMarkovAccumulator",
        "HiddenMarkovAccumulatorFactory",
        "HiddenMarkovEstimator",
        "HiddenMarkovDataEncoder",
        "HiddenMarkovModelSampler",
        "HiddenMarkovModelAccumulator",
        "HiddenMarkovModelAccumulatorFactory",
        "HiddenMarkovModelEstimator",
        "HiddenMarkovModelDataEncoder",
        "HiddenMarkovFisherView",
    ),
    "mixle.stats.latent.hierarchical": (
        "HierarchicalNormalDistribution",
        "HierarchicalNormalEstimator",
        "HierarchicalNormalFitDiagnostics",
    ),
    "mixle.stats.latent.hierarchical_mixture": (
        "HierarchicalMixtureDistribution",
        "HierarchicalMixtureEnumerator",
        "HierarchicalMixtureSampler",
        "HierarchicalMixtureAccumulator",
        "HierarchicalMixtureAccumulatorFactory",
        "HierarchicalMixtureEstimator",
        "HierarchicalMixtureDataEncoder",
    ),
    "mixle.stats.latent.hmm_determinize": (
        "determinize_quantized_terminal",
        "determinize_terminal_hmm",
        "DeterminizedSequenceDistribution",
        "DeterminizedSampler",
        "DeterminizedDataEncoder",
        "DeterminizedEnumerator",
    ),
    "mixle.stats.latent.indian_buffet_process": (
        "IndianBuffetProcessFisherView",
        "IndianBuffetProcessDistribution",
        "IndianBuffetProcessEnumerator",
        "IndianBuffetProcessSampler",
        "IndianBuffetProcessAccumulator",
        "IndianBuffetProcessAccumulatorFactory",
        "IndianBuffetProcessEstimator",
        "IndianBuffetProcessDataEncoder",
    ),
    "mixle.stats.latent.integer_hidden_association": (
        "IntegerHiddenAssociationDistribution",
        "IntegerHiddenAssociationEnumerator",
        "IntegerHiddenAssociationSampler",
        "IntegerHiddenAssociationAccumulator",
        "IntegerHiddenAssociationAccumulatorFactory",
        "IntegerHiddenAssociationEstimator",
        "IntegerHiddenAssociationDataEncoder",
    ),
    "mixle.stats.latent.integer_probabilistic_latent_semantic_indexing": (
        "IntegerPLSIEncodedData",
        "IntegerProbabilisticLatentSemanticIndexingDistribution",
        "IntegerProbabilisticLatentSemanticIndexingEnumerator",
        "IntegerProbabilisticLatentSemanticIndexingSampler",
        "IntegerProbabilisticLatentSemanticIndexingAccumulator",
        "IntegerProbabilisticLatentSemanticIndexingAccumulatorFactory",
        "IntegerProbabilisticLatentSemanticIndexingEstimator",
        "IntegerProbabilisticLatentSemanticIndexingDataEncoder",
        "multinomial_bag_stream",
        "bag_stream",
    ),
    "mixle.stats.latent.joint_mixture": (
        "JointMixtureDistribution",
        "JointMixtureEnumerator",
        "JointMixtureSampler",
        "JointMixtureAccumulator",
        "JointMixtureAccumulatorFactory",
        "JointMixtureEstimator",
        "JointMixtureDataEncoder",
        "JointMixtureFisherView",
    ),
    "mixle.stats.latent.labeled_lda": (
        "LabeledLDADistribution",
        "LabeledLDASampler",
        "LabeledLDALabelSetStats",
        "LabeledLDAAccumulator",
        "LabeledLDAAccumulatorFactory",
        "LabeledLDAEstimator",
        "LabeledLDADataEncoder",
        "doc_label_sets",
        "coupled_alpha_objective",
        "coupled_alpha_gradient",
        "update_alpha_coupled",
    ),
    "mixle.stats.latent.lda": (
        "LDAOptimizationDiagnostics",
        "LDAConvergenceError",
        "LDADistribution",
        "LDASampler",
        "LDAAccumulator",
        "LDAAccumulatorFactory",
        "LDAEstimator",
        "LDADataEncoder",
    ),
    "mixle.stats.latent.lookback_hidden_markov_model": (
        "LookbackHiddenMarkovModelDistribution",
        "LookbackHiddenMarkovModelSampler",
        "LookbackHiddenMarkovModelAccumulator",
        "LookbackHiddenMarkovModelAccumulatorFactory",
        "LookbackHiddenMarkovModelEstimator",
        "LookbackHiddenMarkovModelDataEncoder",
        "LookbackTerminalDataEncoder",
    ),
    "mixle.stats.latent.markov_stopping": (
        "HiddenMarkovNonterminationError",
        "validated_state_ids",
        "validated_terminal_states",
        "validated_terminal_values",
        "validate_terminal_reachability",
        "validated_terminal_step_cap",
        "require_terminal_reached",
    ),
    "mixle.stats.latent.mixture": (
        "mixture_prior",
        "MixtureDistribution",
        "MixtureEnumerator",
        "MixtureSampler",
        "MixtureAccumulator",
        "MixtureAccumulatorFactory",
        "MixtureEstimator",
        "MixtureDataEncoder",
        "MixtureFisherView",
    ),
    "mixle.stats.latent.probabilistic_circuit": (
        "leaf",
        "prod",
        "summ",
        "ProbabilisticCircuitDistribution",
        "ProbabilisticCircuitEncoder",
        "ProbabilisticCircuitSampler",
        "ProbabilisticCircuitAccumulator",
        "ProbabilisticCircuitAccumulatorFactory",
        "ProbabilisticCircuitEstimator",
    ),
    "mixle.stats.latent.probabilistic_pca": (
        "ProbabilisticPCADistribution",
        "ProbabilisticPCASampler",
        "ProbabilisticPCAAccumulator",
        "ProbabilisticPCAAccumulatorFactory",
        "ProbabilisticPCAEstimator",
        "ProbabilisticPCADataEncoder",
    ),
    "mixle.stats.latent.quantized_hidden_markov_model": (
        "QuantizedHMMFitDiagnostics",
        "QuantizedHMMOptimizationError",
        "QuantizedHiddenMarkovModelDistribution",
        "QuantizedHiddenMarkovModelEnumerator",
        "QuantizedHiddenMarkovEstimator",
        "QuantizedHiddenMarkovModelEstimator",
    ),
    "mixle.stats.latent.responsibility_attention": (
        "ResponsibilityAttentionDistribution",
        "ResponsibilityAttentionSampler",
        "ResponsibilityAttentionAccumulator",
        "ResponsibilityAttentionAccumulatorFactory",
        "ResponsibilityAttentionEstimator",
        "ResponsibilityAttentionDataEncoder",
        "sequence_to_triples",
    ),
    "mixle.stats.latent.scheduled_hidden_markov_model": (
        "PhaseSchedule",
        "Homogeneous",
        "ByPosition",
        "ByRelativePosition",
        "ByLength",
        "ScheduledHiddenMarkovModelDistribution",
        "ScheduledHMMSampler",
        "ScheduledHMMDataEncoder",
        "ScheduledHMMAccumulator",
        "ScheduledHMMAccumulatorFactory",
        "ScheduledHMMEstimator",
    ),
    "mixle.stats.latent.segmental_hidden_markov_model": (
        "SegmentalHiddenMarkovDistribution",
        "SegmentalHiddenMarkovModelDistribution",
        "SegmentalHiddenMarkovSampler",
        "SegmentalHiddenMarkovAccumulator",
        "SegmentalHiddenMarkovAccumulatorFactory",
        "SegmentalHiddenMarkovEstimator",
        "SegmentalHiddenMarkovDataEncoder",
        "SegmentalHiddenMarkovModelSampler",
        "SegmentalHiddenMarkovModelAccumulator",
        "SegmentalHiddenMarkovModelAccumulatorFactory",
        "SegmentalHiddenMarkovModelEstimator",
        "SegmentalHiddenMarkovModelDataEncoder",
    ),
    "mixle.stats.latent.semi_supervised_hidden_markov_model": (
        "SemiSupervisedHiddenMarkovModelDistribution",
        "SemiSupervisedHiddenMarkovSampler",
        "SemiSupervisedHiddenMarkovAccumulator",
        "SemiSupervisedHiddenMarkovAccumulatorFactory",
        "SemiSupervisedHiddenMarkovEstimator",
        "SemiSupervisedHiddenMarkovDataEncoder",
        "SemiSupervisedHiddenMarkovModelSampler",
        "SemiSupervisedHiddenMarkovModelAccumulator",
        "SemiSupervisedHiddenMarkovModelAccumulatorFactory",
        "SemiSupervisedHiddenMarkovModelEstimator",
        "SemiSupervisedHiddenMarkovModelDataEncoder",
    ),
    "mixle.stats.latent.semi_supervised_mixture": (
        "SemiSupervisedMixtureDistribution",
        "SemiSupervisedMixtureSampler",
        "SemiSupervisedMixtureAccumulator",
        "SemiSupervisedMixtureAccumulatorFactory",
        "SemiSupervisedMixtureEstimator",
        "SemiSupervisedMixtureDataEncoder",
    ),
    "mixle.stats.latent.sparse_mixture": (
        "SparseScore",
        "log_density_sup",
        "sparse_mixture_score",
        "collapse_identical",
        "collapse_gaussian_mixture",
    ),
    "mixle.stats.latent.structured_hmm": (
        "TransitionOperator",
        "HMMFitDiagnostics",
        "HMMFitResult",
        "DenseTransition",
        "LowRankTransition",
        "SparseTransition",
        "BlockDiagonalTransition",
        "KroneckerTransition",
        "FinalStateEnumeration",
        "StructuredHMM",
        "StructuredHMMDataEncoder",
        "StructuredHMMAccumulator",
        "StructuredHMMAccumulatorFactory",
        "StructuredHMMEstimator",
        "InputOutputHMM",
        "IOHMMDataEncoder",
        "IOHMMAccumulator",
        "IOHMMAccumulatorFactory",
        "IOHMMEstimator",
        "ExplicitDurationHMM",
        "EDHMMDataEncoder",
        "EDHMMAccumulator",
        "EDHMMAccumulatorFactory",
        "EDHMMEstimator",
        "sticky_transition",
        "dirichlet_transition",
        "kron_initial",
        "left_to_right_edges",
        "banded_edges",
        "chunked_state_posteriors",
        "fit_chunked",
        "stationary_initial",
        "jit_forward_loglik",
    ),
    "mixle.stats.latent.tree_hidden_markov_model": (
        "TreeHiddenMarkovModelDistribution",
        "TreeHiddenMarkovSampler",
        "TreeHiddenMarkovAccumulator",
        "TreeHiddenMarkovAccumulatorFactory",
        "TreeHiddenMarkovEstimator",
        "TreeHiddenMarkovDataEncoder",
        "TreeHiddenMarkovModelSampler",
        "TreeHiddenMarkovModelAccumulator",
        "TreeHiddenMarkovModelAccumulatorFactory",
        "TreeHiddenMarkovModelEstimator",
        "TreeHiddenMarkovModelDataEncoder",
        "find_level",
        "level_state_prob",
    ),
    "mixle.stats.latent.variational_embedding_attention": (
        "VariationalEmbeddingAttentionDistribution",
        "VariationalEmbeddingAttentionSampler",
        "VariationalEmbeddingAttentionAccumulator",
        "VariationalEmbeddingAttentionAccumulatorFactory",
        "VariationalEmbeddingAttentionEstimator",
        "VariationalEmbeddingAttentionDataEncoder",
    ),
    "mixle.stats.latent.variational_multihop_attention": (
        "VariationalMultiHopAttentionDistribution",
        "VariationalMultiHopAttentionSampler",
        "VariationalMultiHopAttentionAccumulator",
        "VariationalMultiHopAttentionAccumulatorFactory",
        "VariationalMultiHopAttentionEstimator",
        "VariationalMultiHopAttentionDataEncoder",
    ),
}

_EXPORT_PAIRS = [(name, module) for module, names in _MODULE_EXPORTS.items() for name in names]
_DUPLICATES = sorted({name for name, _module in _EXPORT_PAIRS if sum(pair[0] == name for pair in _EXPORT_PAIRS) > 1})
if _DUPLICATES:
    raise RuntimeError(f"duplicate latent exports: {_DUPLICATES}")
_EXPORTS = dict(_EXPORT_PAIRS)

__all__ = ["LatentAPIEntry", "latent_api_manifest", *_EXPORTS]

_PUBLIC_API_MANIFEST = (
    LatentAPIEntry("LatentAPIEntry", __name__, "stable", lazy=False),
    LatentAPIEntry("latent_api_manifest", __name__, "stable", lazy=False),
    *(LatentAPIEntry(name, module, "stable") for name, module in _EXPORT_PAIRS),
)


def latent_api_manifest() -> tuple[LatentAPIEntry, ...]:
    """Return the immutable, ordered latent-package export manifest."""
    return _PUBLIC_API_MANIFEST


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = None
    for attribute, path in _EXPORTS.items():
        if path == module_path:
            exported = getattr(module, attribute)
            globals()[attribute] = exported
            if attribute == name:
                value = exported
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
