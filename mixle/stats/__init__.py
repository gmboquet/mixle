"""Public statistical modeling surface for Mixle.

This package re-exports the core distribution, sampler, estimator, accumulator,
encoder, compute, and serialization APIs used by higher-level inference
surfaces. Prefer these imports for release-facing examples unless a narrative
guide points to a narrower implementation module.
"""

from __future__ import annotations

__all__ = [
    # Bayesian (conjugate/variational) families folded in from the former mixle.bstats
    "conjugate_posterior",
    "ConjugatePosterior",
    "mixture_conjugate_posterior",
    "MixtureConjugatePosterior",
    "mixture_prior",
    "DictDirichletDistribution",
    "DictDirichletSampler",
    "SymmetricDirichletDistribution",
    "SymmetricDirichletSampler",
    "NormalGammaDistribution",
    "NormalGammaSampler",
    "NormalWishartDistribution",
    "NormalWishartSampler",
    "MultivariateNormalGammaDistribution",
    "MultivariateNormalGammaSampler",
    "DirichletProcessMixtureDistribution",
    "DirichletProcessMixtureEstimator",
    "DirichletProcessMixtureSampler",
    "HierarchicalDirichletProcessMixtureDistribution",
    "HierarchicalDirichletProcessMixtureEstimator",
    "HierarchicalDirichletProcessMixtureSampler",
    "PitmanYorProcessDistribution",
    "PitmanYorProcessSampler",
    "PitmanYorProcessEstimator",
    "PitmanYorProcessDataEncoder",
    "sample",
    "seq_encode",
    "seq_log_density",
    "seq_log_density_sum",
    "log_density",
    "density",
    "load_models",
    "dump_models",
    "DistributionEnumerator",
    "EnumerationError",
    "KeyValidationError",
    "validate_estimator_keys",
    "encoded_nbytes",
    "scale_suff_stat",
    "EncodedData",
    "ResidentEncodedPayload",
    "as_encoded_data",
    "move_encoded_payload",
    "DistributionCapabilities",
    "capabilities_for",
    "numpy_only_distribution_types",
    "register_capabilities",
    "registered_capability_types",
    "supported_engines",
    "DistributionDeclaration",
    "ExponentialFamilySpec",
    "ParameterSpec",
    "StatisticSpec",
    "BackendScoringError",
    "backend_log_density_sum",
    "backend_seq_component_log_density",
    "backend_seq_log_density",
    "declaration_issues",
    "declaration_for",
    "declared_distribution_types",
    "generated_log_density_diagnostics",
    "generated_log_density",
    "generated_stacked_available",
    "generated_stacked_log_density",
    "generated_stacked_params",
    "generated_stacked_preferred",
    "generated_stacked_strategy",
    "generated_sufficient_statistics",
    "generated_sufficient_statistics_available",
    "generated_numba_log_density",
    "generated_numba_log_density_available",
    "generated_numba_stacked_log_density",
    "generated_numba_stacked_available",
    "generated_stacked_sufficient_statistics",
    "generated_stacked_sufficient_statistics_available",
    "register_declaration",
    "statistic_layout_issues",
    "validate_declaration",
    "validate_statistic_layout",
    "RecordDistribution",
    "RecordEstimator",
    "RecordSampler",
    "RecordDataEncoder",
    "DictRecordDistribution",
    "DictRecordEstimator",
    "DictRecordSampler",
    "DictRecordDataEncoder",
    "field",
    "record",
    "record_estimator",
    "EngineNotSupportedError",
    "Kernel",
    "KernelFactory",
    "GenericKernel",
    "GenericKernelFactory",
    "NumbaKernel",
    "GeneratedNumbaKernel",
    "NumbaKernelFactory",
    "GeneratedNumbaKernelFactory",
    "StackedComponentParams",
    "StackedMixtureResidentStats",
    "StackedMixtureShardEstimate",
    "StackedMixtureKernel",
    "StackedMixtureKernelFactory",
    "stacked_component_log_density",
    "stacked_component_params",
    "stacked_component_strategy",
    "estimate_component_shard_value",
    "tie_component_shard_values",
    "register_kernel_factory",
    "kernel_for",
    "BernoulliDistribution",
    "BernoulliSampler",
    "BernoulliEstimator",
    "BernoulliDataEncoder",
    "BernoulliEnumerator",
    "BetaDistribution",
    "BetaSampler",
    "BetaEstimator",
    "BetaDataEncoder",
    "LaplaceDistribution",
    "LaplaceSampler",
    "LaplaceEstimator",
    "LaplaceDataEncoder",
    "LogisticDistribution",
    "LogisticSampler",
    "LogisticEstimator",
    "LogisticDataEncoder",
    "LogSeriesDistribution",
    "LogSeriesSampler",
    "LogSeriesEstimator",
    "LogSeriesDataEncoder",
    "BinomialDistribution",
    "BetaBinomialDistribution",
    "BetaBinomialSampler",
    "BetaBinomialEstimator",
    "BetaBinomialDataEncoder",
    "BinomialSampler",
    "BinomialEstimator",
    "BinomialDataEncoder",
    "BinomialEnumerator",
    "CategoricalDistribution",
    "ChineseRestaurantProcessDistribution",
    "ChineseRestaurantProcessSampler",
    "ChineseRestaurantProcessEstimator",
    "ChineseRestaurantProcessDataEncoder",
    "CategoricalSampler",
    "CategoricalEstimator",
    "CategoricalDataEncoder",
    "CategoricalEnumerator",
    "MultinomialDistribution",
    "MultinomialSampler",
    "MultinomialEstimator",
    "MultinomialDataEncoder",
    "MultinomialEnumerator",
    "CompositeDistribution",
    "CompositeEstimator",
    "CompositeSampler",
    "CompositeDataEncoder",
    "CompositeEnumerator",
    "ConditionalDistribution",
    "ConditionalDistributionSampler",
    "ConditionalDistributionEstimator",
    "ConditionalDistributionDataEncoder",
    "ConditionalDistributionEnumerator",
    "ConditionalEstimator",
    "ConditionalAccumulator",
    "ConditionalAccumulatorFactory",
    "ConditionalDataEncoder",
    "ConditionalEnumerator",
    "ChowLiuTreeDistribution",
    "ChowLiuTreeEstimator",
    "ChowLiuTreeSampler",
    "ChowLiuTreeDataEncoder",
    "ChowLiuTreeEnumerator",
    "DiracLengthMixtureDistribution",
    "DiracLengthMixtureSampler",
    "DiracLengthMixtureEstimator",
    "DiracLengthMixtureEnumerator",
    "DirichletDistribution",
    "DirichletSampler",
    "DirichletEstimator",
    "DirichletDataEncoder",
    "DiagonalGaussianDistribution",
    "DiagonalGaussianSampler",
    "DistributionSampler",
    "DiagonalGaussianEstimator",
    "DiagonalGaussianDataEncoder",
    "ExponentialDistribution",
    "ExponentialSampler",
    "ExponentialEstimator",
    "ExponentialDataEncoder",
    "GammaDistribution",
    "GammaSampler",
    "GammaEstimator",
    "GammaDataEncoder",
    "GaussianDistribution",
    "GaussianSampler",
    "GaussianEstimator",
    "GaussianDataEncoder",
    "InverseGammaDistribution",
    "InverseGammaSampler",
    "InverseGammaEstimator",
    "InverseGammaDataEncoder",
    "InverseGaussianDistribution",
    "InverseGaussianSampler",
    "InverseGaussianEstimator",
    "InverseGaussianDataEncoder",
    "GeometricDistribution",
    "GeometricSampler",
    "GeometricEstimator",
    "GeometricDataEncoder",
    "GeometricEnumerator",
    "GumbelDistribution",
    "GumbelSampler",
    "GumbelEstimator",
    "GumbelDataEncoder",
    "HalfNormalDistribution",
    "HalfNormalSampler",
    "HalfNormalEstimator",
    "HalfNormalDataEncoder",
    "NegativeBinomialDistribution",
    "NegativeBinomialSampler",
    "NegativeBinomialEstimator",
    "NegativeBinomialDataEncoder",
    "NegativeBinomialEnumerator",
    "ParetoDistribution",
    "ParetoSampler",
    "ParetoEstimator",
    "ParetoDataEncoder",
    "RayleighDistribution",
    "RayleighSampler",
    "RayleighEstimator",
    "RayleighDataEncoder",
    "StudentTDistribution",
    "StudentTSampler",
    "StudentTEstimator",
    "StudentTDataEncoder",
    "UniformDistribution",
    "UniformSampler",
    "UniformEstimator",
    "UniformDataEncoder",
    "VonMisesDistribution",
    "WrappedCauchyDistribution",
    "WrappedCauchySampler",
    "WrappedCauchyEstimator",
    "WrappedCauchyDataEncoder",
    "ProjectedNormalDistribution",
    "ProjectedNormalSampler",
    "ProjectedNormalEstimator",
    "ProjectedNormalDataEncoder",
    "WrappedNormalDistribution",
    "WrappedNormalSampler",
    "WrappedNormalEstimator",
    "WrappedNormalDataEncoder",
    "GeneralizedGaussianDistribution",
    "GeneralizedGaussianSampler",
    "GeneralizedGaussianEstimator",
    "GeneralizedGaussianDataEncoder",
    "NakagamiDistribution",
    "NakagamiSampler",
    "NakagamiEstimator",
    "NakagamiDataEncoder",
    "RicianDistribution",
    "RicianSampler",
    "RicianEstimator",
    "RicianDataEncoder",
    "VonMisesSampler",
    "VonMisesEstimator",
    "VonMisesDataEncoder",
    "WeibullDistribution",
    "GeneralizedParetoDistribution",
    "GeneralizedParetoSampler",
    "GeneralizedParetoEstimator",
    "GeneralizedParetoDataEncoder",
    "GeneralizedExtremeValueDistribution",
    "GeneralizedExtremeValueSampler",
    "GeneralizedExtremeValueEstimator",
    "GeneralizedExtremeValueDataEncoder",
    "WeibullSampler",
    "WeibullEstimator",
    "WeibullDataEncoder",
    "GatedMixtureDistribution",
    "GatedMixtureSampler",
    "GatedMixtureEstimator",
    "GatedMixtureAccumulator",
    "GatedMixtureAccumulatorFactory",
    "GatedMixtureDataEncoder",
    "SoftmaxGate",
    "HeterogeneousMixtureDistribution",
    "HeterogeneousMixtureSampler",
    "HeterogeneousMixtureEstimator",
    "HeterogeneousMixtureDataEncoder",
    "HeterogeneousMixtureEnumerator",
    "HeterogeneousPCFGDistribution",
    "HeterogeneousPCFGSampler",
    "HeterogeneousPCFGEstimator",
    "InducedHeterogeneousPCFGEstimator",
    "HeterogeneousPCFGDataEncoder",
    "HeterogeneousPCFGEnumerator",
    "HiddenAssociationDistribution",
    "HiddenAssociationSampler",
    "HiddenAssociationEstimator",
    "StructuredHMM",
    "StructuredHMMEstimator",
    "StructuredHMMDataEncoder",
    "TransitionOperator",
    "DenseTransition",
    "LowRankTransition",
    "BlockDiagonalTransition",
    "KroneckerTransition",
    "SparseTransition",
    "InputOutputHMM",
    "ExplicitDurationHMM",
    "jit_forward_loglik",
    "stationary_initial",
    "sticky_transition",
    "dirichlet_transition",
    "kron_initial",
    "left_to_right_edges",
    "banded_edges",
    "HiddenAssociationDataEncoder",
    "HiddenMarkovModelDistribution",
    "HiddenMarkovSampler",
    "HiddenMarkovEstimator",
    "HiddenMarkovDataEncoder",
    "HiddenMarkovModelEnumerator",
    "HiddenMarkovModelSampler",
    "HiddenMarkovModelEstimator",
    "HiddenMarkovModelDataEncoder",
    "HiddenMarkovModelAccumulator",
    "HiddenMarkovModelAccumulatorFactory",
    "QuantizedHiddenMarkovModelDistribution",
    "QuantizedHiddenMarkovEstimator",
    "QuantizedHiddenMarkovModelEnumerator",
    "QuantizedHiddenMarkovModelEstimator",
    "HierarchicalMixtureDistribution",
    "HierarchicalMixtureSampler",
    "HierarchicalMixtureEstimator",
    "HierarchicalMixtureDataEncoder",
    "HierarchicalMixtureEnumerator",
    "IndianBuffetProcessDistribution",
    "IndianBuffetProcessSampler",
    "IndianBuffetProcessEstimator",
    "IndianBuffetProcessDataEncoder",
    "IntegerChowLiuTreeDistribution",
    "IntegerChowLiuTreeEstimator",
    "IntegerChowLiuTreeSampler",
    "IntegerChowLiuTreeDataEncoder",
    "IntegerChowLiuTreeEnumerator",
    "IgnoredDistribution",
    "IgnoredSampler",
    "IgnoredEstimator",
    "IgnoredDataEncoder",
    "IntegerBernoulliEditDistribution",
    "IntegerBernoulliEditSampler",
    "IntegerBernoulliEditEstimator",
    "IntegerBernoulliEditDataEncoder",
    "IntegerBernoulliEditEnumerator",
    "IntegerStepBernoulliEditDistribution",
    "IntegerStepBernoulliEditSampler",
    "IntegerStepBernoulliEditEstimator",
    "IntegerStepBernoulliEditDataEncoder",
    "IntegerStepBernoulliEditEnumerator",
    "IntegerHiddenAssociationDistribution",
    "IntegerHiddenAssociationEstimator",
    "IntegerHiddenAssociationSampler",
    "IntegerHiddenAssociationDataEncoder",
    "IntegerMarkovChainDistribution",
    "IntegerMarkovChainSampler",
    "IntegerMarkovChainEstimator",
    "IntegerMarkovChainDataEncoder",
    "IntegerMarkovChainEnumerator",
    "IntegerProbabilisticLatentSemanticIndexingDistribution",
    "IntegerProbabilisticLatentSemanticIndexingSampler",
    "IntegerProbabilisticLatentSemanticIndexingEstimator",
    "IntegerProbabilisticLatentSemanticIndexingDataEncoder",
    "IntegerUniformSpikeDistribution",
    "IntegerUniformSpikeEstimator",
    "IntegerUniformSpikeSampler",
    "IntegerUniformSpikeDataEncoder",
    "IntegerUniformSpikeEnumerator",
    "IntegerMultinomialDistribution",
    "DirichletMultinomialDistribution",
    "DirichletMultinomialSampler",
    "DirichletMultinomialEstimator",
    "DirichletMultinomialDataEncoder",
    "IntegerMultinomialSampler",
    "IntegerMultinomialEstimator",
    "IntegerMultinomialDataEncoder",
    "IntegerMultinomialEnumerator",
    "IntegerCategoricalDistribution",
    "IntegerCategoricalSampler",
    "IntegerCategoricalEstimator",
    "IntegerCategoricalDataEncoder",
    "IntegerCategoricalEnumerator",
    "IntegerBernoulliSetDistribution",
    "IntegerBernoulliSetSampler",
    "IntegerBernoulliSetEstimator",
    "IntegerBernoulliSetDataEncoder",
    "IntegerBernoulliSetEnumerator",
    "JointMixtureDistribution",
    "JointMixtureSampler",
    "JointMixtureEstimator",
    "JointMixtureDataEncoder",
    "JointMixtureEnumerator",
    "LogGaussianDistribution",
    "LogGaussianSampler",
    "LogGaussianEstimator",
    "LogGaussianDataEncoder",
    "MarkovChainDistribution",
    "MarkovChainSampler",
    "MarkovChainEstimator",
    "MarkovChainDataEncoder",
    "MarkovChainEnumerator",
    "ProbabilisticPCADistribution",
    "ProbabilisticPCASampler",
    "ProbabilisticPCAEstimator",
    "ProbabilisticPCADataEncoder",
    "MixtureDistribution",
    # responsibility attention: EM-able mixture-over-positions attention head
    "ResponsibilityAttentionDistribution",
    "ResponsibilityAttentionEstimator",
    "ResponsibilityAttentionSampler",
    "ResponsibilityAttentionAccumulator",
    "ResponsibilityAttentionAccumulatorFactory",
    "ResponsibilityAttentionDataEncoder",
    "sequence_to_triples",
    # variational-EM attention with tied latent embeddings
    "VariationalEmbeddingAttentionDistribution",
    "VariationalEmbeddingAttentionEstimator",
    "VariationalEmbeddingAttentionSampler",
    "VariationalEmbeddingAttentionAccumulator",
    "VariationalEmbeddingAttentionAccumulatorFactory",
    "VariationalEmbeddingAttentionDataEncoder",
    # chained (multi-hop) attention -- L-hop stack via forward-backward
    "ChainedAttentionDistribution",
    "ChainedAttentionEstimator",
    "ChainedAttentionSampler",
    "ChainedAttentionAccumulator",
    "ChainedAttentionAccumulatorFactory",
    "ChainedAttentionDataEncoder",
    # variational multi-hop attention (2-hop chain over tied latent embeddings, annealed)
    "VariationalMultiHopAttentionDistribution",
    "VariationalMultiHopAttentionEstimator",
    "VariationalMultiHopAttentionSampler",
    "VariationalMultiHopAttentionAccumulator",
    "VariationalMultiHopAttentionAccumulatorFactory",
    "VariationalMultiHopAttentionDataEncoder",
    "Posterior",
    "LatentPosterior",
    "CategoricalLatentPosterior",
    "MarkovChainLatentPosterior",
    "MeanFieldLDAPosterior",
    "MixtureSampler",
    "MixtureEstimator",
    "MixtureDataEncoder",
    "MixtureEnumerator",
    "MultivariateGaussianDistribution",
    "MultivariateGaussianEstimator",
    # sampling completeness / richness / diversity from frequency counts
    "MultivariateGaussianSampler",
    "MultivariateGaussianDataEncoder",
    "NullDistribution",
    "NullSampler",
    "NullEstimator",
    "NullDataEncoder",
    "NullEnumerator",
    "MISSING",
    "marginalized",
    "composite_with_missing",
    "OptionalDistribution",
    "OptionalSampler",
    "OptionalEstimator",
    "OptionalDataEncoder",
    "OptionalEnumerator",
    "PoissonDistribution",
    "PoissonSampler",
    "PoissonEstimator",
    "PoissonDataEncoder",
    "PoissonEnumerator",
    "PointMassDistribution",
    "PointMassSampler",
    "PointMassEstimator",
    "PointMassDataEncoder",
    "PointMassEnumerator",
    "SelectDistribution",
    "SelectEstimator",
    "TypeDispatch",
    "SelectEnumerator",
    "SequenceDistribution",
    "SequenceSampler",
    "SequenceEstimator",
    "SequenceDataEncoder",
    "SequenceEnumerator",
    "ScheduledHiddenMarkovModelDistribution",
    "ScheduledHMMEstimator",
    "Homogeneous",
    "ByPosition",
    "ByRelativePosition",
    "ByLength",
    "SegmentalHiddenMarkovModelDistribution",
    "SegmentalHiddenMarkovDistribution",
    "SegmentalHiddenMarkovSampler",
    "SegmentalHiddenMarkovEstimator",
    "SegmentalHiddenMarkovDataEncoder",
    "SegmentalHiddenMarkovModelSampler",
    "SegmentalHiddenMarkovModelEstimator",
    "SegmentalHiddenMarkovModelDataEncoder",
    "BernoulliSetDistribution",
    "BernoulliSetSampler",
    "BernoulliSetEstimator",
    "BernoulliSetDataEncoder",
    "BernoulliSetEnumerator",
    "SparseMarkovAssociationDistribution",
    "SparseMarkovAssociationSampler",
    "SparseMarkovAssociationEstimator",
    "SparseMarkovAssociationDataEncoder",
    "SpearmanRankingDistribution",
    "SpearmanRankingSampler",
    "SpearmanRankingEstimator",
    "SpearmanRankingDataEncoder",
    "SpearmanRankingEnumerator",
    "KnowledgeGraphDistribution",
    "KnowledgeGraphEstimator",
    "KnowledgeGraphSampler",
    "KnowledgeGraphEnsemble",
    "fit_knowledge_graph_ensemble",
    "KnowledgeGraphDataEncoder",
    "PlackettLuceDistribution",
    "PlackettLuceSampler",
    "PlackettLuceEstimator",
    "PlackettLuceDataEncoder",
    "PlackettLuceEnumerator",
    "MallowsDistribution",
    "MallowsSampler",
    "MallowsEstimator",
    "MallowsDataEncoder",
    "MallowsEnumerator",
    "GeneralizedMallowsDistribution",
    "GeneralizedMallowsSampler",
    "GeneralizedMallowsEstimator",
    "GeneralizedMallowsAccumulator",
    "GeneralizedMallowsAccumulatorFactory",
    "GeneralizedMallowsDataEncoder",
    "GeneralizedMallowsModelDistribution",
    "GeneralizedMallowsModelSampler",
    "GeneralizedMallowsModelEstimator",
    "GeneralizedMallowsModelAccumulator",
    "GeneralizedMallowsModelAccumulatorFactory",
    "GeneralizedMallowsModelDataEncoder",
    "BradleyTerryDistribution",
    "BradleyTerrySampler",
    "BradleyTerryEstimator",
    "BradleyTerryAccumulator",
    "BradleyTerryAccumulatorFactory",
    "BradleyTerryDataEncoder",
    "LowRankPermutationDistribution",
    "LowRankPermutationSampler",
    "LowRankPermutationEstimator",
    "LowRankPermutationAccumulator",
    "LowRankPermutationAccumulatorFactory",
    "LowRankPermutationDataEncoder",
    "ThurstoneDistribution",
    "ThurstoneSampler",
    "ThurstoneEstimator",
    "ThurstoneAccumulator",
    "ThurstoneAccumulatorFactory",
    "ThurstoneDataEncoder",
    "ThurstoneMostellerDistribution",
    "ThurstoneMostellerSampler",
    "ThurstoneMostellerEstimator",
    "PairWinAccumulator",
    "PairWinAccumulatorFactory",
    "PairDataEncoder",
    "DavidsonDistribution",
    "DavidsonEstimator",
    "RaoKupperDistribution",
    "RaoKupperEstimator",
    "EwensDistribution",
    "EwensSampler",
    "EwensEstimator",
    "EwensAccumulator",
    "EwensAccumulatorFactory",
    "EwensDataEncoder",
    "SpanningTreeDistribution",
    "SpanningTreeSampler",
    "SpanningTreeEstimator",
    "SpanningTreeDataEncoder",
    "SpanningTreeEnumerator",
    "MatchingDistribution",
    "MatchingSampler",
    "MatchingEstimator",
    "MatchingDataEncoder",
    "MatchingEnumerator",
    "RandomDotProductGraphDistribution",
    "RandomDotProductGraphSampler",
    "RandomDotProductGraphEstimator",
    "SemiSupervisedMixtureDistribution",
    "SemiSupervisedMixtureSampler",
    "SemiSupervisedMixtureEstimator",
    "SemiSupervisedMixtureDataEncoder",
    "TreeHiddenMarkovModelDistribution",
    "TreeHiddenMarkovSampler",
    "TreeHiddenMarkovEstimator",
    "TreeHiddenMarkovModelSampler",
    "TreeHiddenMarkovModelEstimator",
    "TransformDistribution",
    "TransformSampler",
    "TransformEstimator",
    "TransformDataEncoder",
    "TransformEnumerator",
    "FiniteStochasticTransformDistribution",
    "FiniteStochasticTransformSampler",
    "FiniteStochasticTransformEstimator",
    "FiniteStochasticTransformDataEncoder",
    "FiniteStochasticTransformEnumerator",
    "ZeroInflatedDistribution",
    "ZeroInflatedSampler",
    "ZeroInflatedEstimator",
    "ZeroInflatedDataEncoder",
    "HurdleDistribution",
    "HurdleSampler",
    "HurdleEstimator",
    "HurdleDataEncoder",
    "SurvivalDistribution",
    "SurvivalSampler",
    "SurvivalEstimator",
    "SurvivalDataEncoder",
    "TruncatedDistribution",
    "TruncatedSampler",
    "TruncatedEstimator",
    "TruncatedDataEncoder",
    "TruncatedEnumerator",
    "CensoredDistribution",
    "CensoredSampler",
    "CensoredEstimator",
    "CensoredAccumulator",
    "CensoredAccumulatorFactory",
    "CensoredDataEncoder",
    "CopulaDistribution",
    "CopulaSampler",
    "CopulaEstimator",
    "CopulaAccumulator",
    "CopulaAccumulatorFactory",
    "CopulaDataEncoder",
    "ExponentiallyModifiedGaussianDistribution",
    "ExponentiallyModifiedGaussianSampler",
    "ExponentiallyModifiedGaussianEstimator",
    "ExponentiallyModifiedGaussianDataEncoder",
    "SkellamDistribution",
    "SkellamSampler",
    "SkellamEstimator",
    "SkellamDataEncoder",
    "SkewNormalDistribution",
    "SkewNormalSampler",
    "SkewNormalEstimator",
    "SkewNormalDataEncoder",
    "TweedieDistribution",
    "TweedieSampler",
    "TweedieEstimator",
    "TweedieDataEncoder",
    "BirthDeathSamplingDistribution",
    "BirthDeathSamplingSampler",
    "BirthDeathSamplingEstimator",
    "BirthDeathSamplingDataEncoder",
    "ContinuousTimeMarkovChainDistribution",
    "ContinuousTimeMarkovChainSampler",
    "ContinuousTimeMarkovChainEstimator",
    "ContinuousTimeMarkovChainDataEncoder",
    "InhomogeneousPoissonProcessDistribution",
    "InhomogeneousPoissonProcessSampler",
    "InhomogeneousPoissonProcessEstimator",
    "InhomogeneousPoissonProcessDataEncoder",
    "RenewalProcessDistribution",
    "RenewalProcessSampler",
    "RenewalProcessEstimator",
    "RenewalProcessDataEncoder",
    "HawkesProcessDistribution",
    "MultivariateHawkesProcessDistribution",
    "MultivariateHawkesProcessSampler",
    "MultivariateHawkesProcessEstimator",
    "MultivariateHawkesProcessDataEncoder",
    "HawkesProcessSampler",
    "HawkesProcessEstimator",
    "HawkesProcessDataEncoder",
    "ExponentialTiltedDistribution",
    "ExponentialTiltedSampler",
    "ExponentialTiltedEstimator",
    "ExponentialTiltedDataEncoder",
    "ExponentialTiltedEnumerator",
    "register_exponential_tilt",
    "registered_tilt_families",
    "IdentityTransform",
    "AffineTransform",
    "ExpTransform",
    "LogTransform",
    "LogitTransform",
    "LDADistribution",
    "LDASampler",
    "LDAEstimator",
    "LDADataEncoder",
    "VonMisesFisherDistribution",
    "GaussianCopulaDistribution",
    "GaussianCopulaSampler",
    "GaussianCopulaEstimator",
    "GaussianCopulaDataEncoder",
    "ClaytonCopulaDistribution",
    "ClaytonCopulaSampler",
    "ClaytonCopulaEstimator",
    "FrankCopulaDistribution",
    "FrankCopulaSampler",
    "FrankCopulaEstimator",
    "StudentTCopulaDistribution",
    "StudentTCopulaSampler",
    "StudentTCopulaEstimator",
    "GumbelCopulaDistribution",
    "GumbelCopulaSampler",
    "GumbelCopulaEstimator",
    "CVineCopulaDistribution",
    "CVineCopulaSampler",
    "CVineCopulaEstimator",
    "DVineCopulaDistribution",
    "DVineCopulaSampler",
    "DVineCopulaEstimator",
    "RVineCopulaDistribution",
    "RVineCopulaSampler",
    "RVineCopulaEstimator",
    "GaussianPairCopula",
    "ClaytonPairCopula",
    "FrankPairCopula",
    "GumbelPairCopula",
    "StudentTPairCopula",
    "IndependencePairCopula",
    "MatrixNormalDistribution",
    "MatrixNormalSampler",
    "MatrixNormalEstimator",
    "MatrixNormalDataEncoder",
    "WatsonDistribution",
    "WatsonSampler",
    "WatsonEstimator",
    "WatsonDataEncoder",
    "WishartDistribution",
    "WishartSampler",
    "WishartEstimator",
    "WishartDataEncoder",
    "LKJDistribution",
    "LKJSampler",
    "LKJEstimator",
    "LKJDataEncoder",
    "KentDistribution",
    "KentSampler",
    "KentEstimator",
    "KentDataEncoder",
    "BinghamDistribution",
    "BinghamSampler",
    "BinghamEstimator",
    "BinghamDataEncoder",
    "InverseWishartDistribution",
    "InverseWishartSampler",
    "InverseWishartEstimator",
    "InverseWishartDataEncoder",
    "VonMisesFisherSampler",
    "VonMisesFisherEstimator",
    "VonMisesFisherDataEncoder",
    "MultivariateStudentTDistribution",
    "MultivariateStudentTSampler",
    "MultivariateStudentTEstimator",
    "MultivariateStudentTDataEncoder",
    "WeightedDistribution",
    "WeightedDataEncoder",
    "WeightedEstimator",
    "ErdosRenyiGraphDistribution",
    "ErdosRenyiGraphSampler",
    "ErdosRenyiGraphAccumulator",
    "ErdosRenyiGraphAccumulatorFactory",
    "ErdosRenyiGraphEstimator",
    "TemporalGraphGrammarDistribution",
    "TemporalGraphGrammarSampler",
    "TemporalGraphGrammarEstimator",
    "TemporalGraphGrammarAccumulator",
    "TemporalGraphGrammarAccumulatorFactory",
    "CommonNeighbourMotif",
    "LabeledTemporalGraphGrammarDistribution",
    "LabeledTemporalGraphGrammarSampler",
    "LabeledTemporalGraphGrammarEstimator",
    "LabeledTemporalGraphGrammarAccumulator",
    "LabeledTemporalGraphGrammarAccumulatorFactory",
    "HomophilyTemporalGraphGrammarDistribution",
    "HomophilyTemporalGraphGrammarSampler",
    "HomophilyTemporalGraphGrammarEstimator",
    "HomophilyTemporalGraphGrammarAccumulator",
    "HomophilyTemporalGraphGrammarAccumulatorFactory",
    "ChurningTemporalGraphGrammarDistribution",
    "ChurningTemporalGraphGrammarSampler",
    "ChurningTemporalGraphGrammarEstimator",
    "ChurningTemporalGraphGrammarAccumulator",
    "ChurningTemporalGraphGrammarAccumulatorFactory",
    "LatentTemporalGraphGrammarDistribution",
    "LatentTemporalGraphGrammarSampler",
    "LatentTemporalGraphGrammarEstimator",
    "LatentTemporalGraphGrammarAccumulator",
    "LatentTemporalGraphGrammarAccumulatorFactory",
    "LatentAttributedTemporalGraphGrammarDistribution",
    "LatentAttributedTemporalGraphGrammarSampler",
    "LatentAttributedTemporalGraphGrammarEstimator",
    "LatentAttributedTemporalGraphGrammarAccumulator",
    "LatentAttributedTemporalGraphGrammarAccumulatorFactory",
    "LatentChurningTemporalGraphGrammarDistribution",
    "LatentChurningTemporalGraphGrammarSampler",
    "LatentChurningTemporalGraphGrammarEstimator",
    "LatentChurningTemporalGraphGrammarAccumulator",
    "LatentChurningTemporalGraphGrammarAccumulatorFactory",
    "StochasticBlockGraphDistribution",
    "StochasticBlockGraphSampler",
    "StochasticBlockGraphAccumulator",
    "StochasticBlockGraphAccumulatorFactory",
    "StochasticBlockGraphEstimator",
]

# --- public surface curation -------------------------------------------------------------------------
# The list above enumerates every imported name, but ~half is per-family plumbing a model author never
# constructs: samplers (use ``dist.sampler()``), data encoders, and EM accumulators/factories. Those stay
# importable for advanced/internal use, but they are demoted from the blessed public surface (``from
# mixle.stats import *``, tooling that honors ``__all__``, docs) so users see distributions, estimators, the
# combinators, enumerators, and helpers -- not the EM/encoding plumbing. The filter is by construction, so
# a new family's sampler/encoder/accumulator is demoted automatically. (The kernel/codegen and backend
# scoring layers stay public for now: they are reached via ``from mixle.stats import *`` in places.)
_INTERNAL_SUFFIXES = ("Sampler", "DataEncoder", "Accumulator", "AccumulatorFactory")
__all__ = [_n for _n in __all__ if not _n.endswith(_INTERNAL_SUFFIXES)]

### Abstract Classes
### Generic Distributions
### Discrete base distributions
from mixle.stats.bayes.conjugate import (
    ConjugatePosterior,
    MixtureConjugatePosterior,
    conjugate_posterior,
    mixture_conjugate_posterior,
)
from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution, DictDirichletSampler
from mixle.stats.bayes.dirichlet import (
    DirichletDataEncoder,
    DirichletDistribution,
    DirichletEstimator,
    DirichletSampler,
)
from mixle.stats.bayes.dirichlet_process_mixture import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
    DirichletProcessMixtureSampler,
)
from mixle.stats.bayes.hierarchical_dirichlet_process_mixture import (
    HierarchicalDirichletProcessMixtureDistribution,
    HierarchicalDirichletProcessMixtureEstimator,
    HierarchicalDirichletProcessMixtureSampler,
)
from mixle.stats.bayes.multivariate_normal_gamma import (
    MultivariateNormalGammaDistribution,
    MultivariateNormalGammaSampler,
)
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution, NormalGammaSampler
from mixle.stats.bayes.normal_wishart import NormalWishartDistribution, NormalWishartSampler
from mixle.stats.bayes.pitman_yor import (
    PitmanYorProcessDataEncoder,
    PitmanYorProcessDistribution,
    PitmanYorProcessEstimator,
    PitmanYorProcessSampler,
)
from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution, SymmetricDirichletSampler

### combinators distributions
from mixle.stats.combinator.censored import (
    CensoredAccumulator,
    CensoredAccumulatorFactory,
    CensoredDataEncoder,
    CensoredDistribution,
    CensoredEstimator,
    CensoredSampler,
)
from mixle.stats.combinator.composite import (
    CompositeDataEncoder,
    CompositeDistribution,
    CompositeEnumerator,
    CompositeEstimator,
    CompositeSampler,
)
from mixle.stats.combinator.conditional import (
    ConditionalAccumulator,
    ConditionalAccumulatorFactory,
    ConditionalDataEncoder,
    ConditionalDistribution,
    ConditionalDistributionDataEncoder,
    ConditionalDistributionEnumerator,
    ConditionalDistributionEstimator,
    ConditionalDistributionSampler,
    ConditionalEnumerator,
    ConditionalEstimator,
)
from mixle.stats.combinator.copula import (
    CopulaAccumulator,
    CopulaAccumulatorFactory,
    CopulaDataEncoder,
    CopulaDistribution,
    CopulaEstimator,
    CopulaSampler,
)
from mixle.stats.combinator.exponential_tilt import (
    ExponentialTiltedDataEncoder,
    ExponentialTiltedDistribution,
    ExponentialTiltedEnumerator,
    ExponentialTiltedEstimator,
    ExponentialTiltedSampler,
    register_exponential_tilt,
    registered_tilt_families,
)
from mixle.stats.combinator.finite_stochastic_transform import (
    FiniteStochasticTransformDataEncoder,
    FiniteStochasticTransformDistribution,
    FiniteStochasticTransformEnumerator,
    FiniteStochasticTransformEstimator,
    FiniteStochasticTransformSampler,
)
from mixle.stats.combinator.hurdle import (
    HurdleDataEncoder,
    HurdleDistribution,
    HurdleEstimator,
    HurdleSampler,
)
from mixle.stats.combinator.ignored import IgnoredDataEncoder, IgnoredDistribution, IgnoredEstimator, IgnoredSampler
from mixle.stats.combinator.null_dist import (
    NullDataEncoder,
    NullDistribution,
    NullEnumerator,
    NullEstimator,
    NullSampler,
)
from mixle.stats.combinator.optional import (
    OptionalDataEncoder,
    OptionalDistribution,
    OptionalEnumerator,
    OptionalEstimator,
    OptionalSampler,
)
from mixle.stats.combinator.record import (
    DictRecordDataEncoder,
    DictRecordDistribution,
    DictRecordEstimator,
    DictRecordSampler,
    RecordDataEncoder,
    RecordDistribution,
    RecordEstimator,
    RecordSampler,
    field,
    record,
    record_estimator,
)
from mixle.stats.combinator.select import SelectDistribution, SelectEnumerator, SelectEstimator, TypeDispatch
from mixle.stats.combinator.sequence import (
    SequenceDataEncoder,
    SequenceDistribution,
    SequenceEnumerator,
    SequenceEstimator,
    SequenceSampler,
)
from mixle.stats.combinator.survival import (
    SurvivalDataEncoder,
    SurvivalDistribution,
    SurvivalEstimator,
    SurvivalSampler,
)
from mixle.stats.combinator.transform import (
    AffineTransform,
    ExpTransform,
    IdentityTransform,
    LogitTransform,
    LogTransform,
    TransformDataEncoder,
    TransformDistribution,
    TransformEnumerator,
    TransformEstimator,
    TransformSampler,
)
from mixle.stats.combinator.truncated import (
    TruncatedDataEncoder,
    TruncatedDistribution,
    TruncatedEnumerator,
    TruncatedEstimator,
    TruncatedSampler,
)
from mixle.stats.combinator.weighted import WeightedDataEncoder, WeightedDistribution, WeightedEstimator
from mixle.stats.combinator.zero_inflated import (
    ZeroInflatedDataEncoder,
    ZeroInflatedDistribution,
    ZeroInflatedEstimator,
    ZeroInflatedSampler,
)
from mixle.stats.compute.backend import (
    BackendScoringError,
    backend_log_density_sum,
    backend_seq_component_log_density,
    backend_seq_log_density,
)
from mixle.stats.compute.capabilities import (
    DistributionCapabilities,
    capabilities_for,
    numpy_only_distribution_types,
    register_capabilities,
    registered_capability_types,
    supported_engines,
)
from mixle.stats.compute.declarations import (
    DistributionDeclaration,
    ExponentialFamilySpec,
    ParameterSpec,
    StatisticSpec,
    declaration_for,
    declaration_issues,
    declared_distribution_types,
    generated_log_density,
    generated_log_density_diagnostics,
    generated_numba_log_density,
    generated_numba_log_density_available,
    generated_numba_stacked_available,
    generated_numba_stacked_log_density,
    generated_stacked_available,
    generated_stacked_log_density,
    generated_stacked_params,
    generated_stacked_preferred,
    generated_stacked_strategy,
    generated_stacked_sufficient_statistics,
    generated_stacked_sufficient_statistics_available,
    generated_sufficient_statistics,
    generated_sufficient_statistics_available,
    register_declaration,
    statistic_layout_issues,
    validate_declaration,
    validate_statistic_layout,
)
from mixle.stats.compute.encoded import EncodedData, ResidentEncodedPayload, as_encoded_data, move_encoded_payload
from mixle.stats.compute.kernel import (
    EngineNotSupportedError,
    GeneratedNumbaKernel,
    GeneratedNumbaKernelFactory,
    GenericKernel,
    GenericKernelFactory,
    Kernel,
    KernelFactory,
    NumbaKernel,
    NumbaKernelFactory,
    kernel_for,
    register_kernel_factory,
)

# Base contracts re-exported as accessible attributes (e.g. isinstance checks against
# mixle.stats.SequenceEncodableProbabilityDistribution / mixle.stats.ParameterEstimator), but
# intentionally NOT in __all__: the distribution catalog scanners (star-import / sampler-seed /
# serialization-registry) must stay concrete-distributions-only.
from mixle.stats.compute.pdist import (  # noqa: F401
    DensitySemantics,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    KeyValidationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    encoded_nbytes,
    scale_suff_stat,
    validate_estimator_keys,
)
from mixle.stats.compute.posterior import (
    CategoricalLatentPosterior,
    LatentPosterior,
    MarkovChainLatentPosterior,
    MeanFieldLDAPosterior,
    Posterior,
)
from mixle.stats.compute.sampling_api import sample
from mixle.stats.compute.stacked import (
    StackedComponentParams,
    StackedMixtureKernel,
    StackedMixtureKernelFactory,
    StackedMixtureResidentStats,
    StackedMixtureShardEstimate,
    estimate_component_shard_value,
    stacked_component_log_density,
    stacked_component_params,
    stacked_component_strategy,
    tie_component_shard_values,
)
from mixle.stats.directional.bingham import (
    BinghamDataEncoder,
    BinghamDistribution,
    BinghamEstimator,
    BinghamSampler,
)
from mixle.stats.directional.kent import (
    KentDataEncoder,
    KentDistribution,
    KentEstimator,
    KentSampler,
)
from mixle.stats.directional.projected_normal import (
    ProjectedNormalDataEncoder,
    ProjectedNormalDistribution,
    ProjectedNormalEstimator,
    ProjectedNormalSampler,
)
from mixle.stats.directional.von_mises import (
    VonMisesDataEncoder,
    VonMisesDistribution,
    VonMisesEstimator,
    VonMisesSampler,
)
from mixle.stats.directional.von_mises_fisher import (
    VonMisesFisherDataEncoder,
    VonMisesFisherDistribution,
    VonMisesFisherEstimator,
    VonMisesFisherSampler,
)
from mixle.stats.directional.watson import (
    WatsonDataEncoder,
    WatsonDistribution,
    WatsonEstimator,
    WatsonSampler,
)
from mixle.stats.directional.wrapped_cauchy import (
    WrappedCauchyDataEncoder,
    WrappedCauchyDistribution,
    WrappedCauchyEstimator,
    WrappedCauchySampler,
)
from mixle.stats.directional.wrapped_normal import (
    WrappedNormalDataEncoder,
    WrappedNormalDistribution,
    WrappedNormalEstimator,
    WrappedNormalSampler,
)
from mixle.stats.graphs.erdos_renyi_graph import (
    ErdosRenyiGraphAccumulator,
    ErdosRenyiGraphAccumulatorFactory,
    ErdosRenyiGraphDistribution,
    ErdosRenyiGraphEstimator,
    ErdosRenyiGraphSampler,
)
from mixle.stats.graphs.knowledge_graph import (
    KnowledgeGraphDataEncoder,
    KnowledgeGraphDistribution,
    KnowledgeGraphEnsemble,
    KnowledgeGraphEstimator,
    KnowledgeGraphSampler,
    fit_knowledge_graph_ensemble,
)
from mixle.stats.graphs.random_dot_product_graph import (
    RandomDotProductGraphDistribution,
    RandomDotProductGraphEstimator,
    RandomDotProductGraphSampler,
)
from mixle.stats.graphs.stochastic_block_graph import (
    StochasticBlockGraphAccumulator,
    StochasticBlockGraphAccumulatorFactory,
    StochasticBlockGraphDistribution,
    StochasticBlockGraphEstimator,
    StochasticBlockGraphSampler,
)
from mixle.stats.graphs.temporal_graph_grammar import (
    ChurningTemporalGraphGrammarAccumulator,
    ChurningTemporalGraphGrammarAccumulatorFactory,
    ChurningTemporalGraphGrammarDistribution,
    ChurningTemporalGraphGrammarEstimator,
    ChurningTemporalGraphGrammarSampler,
    CommonNeighbourMotif,
    HomophilyTemporalGraphGrammarAccumulator,
    HomophilyTemporalGraphGrammarAccumulatorFactory,
    HomophilyTemporalGraphGrammarDistribution,
    HomophilyTemporalGraphGrammarEstimator,
    HomophilyTemporalGraphGrammarSampler,
    LabeledTemporalGraphGrammarAccumulator,
    LabeledTemporalGraphGrammarAccumulatorFactory,
    LabeledTemporalGraphGrammarDistribution,
    LabeledTemporalGraphGrammarEstimator,
    LabeledTemporalGraphGrammarSampler,
    LatentAttributedTemporalGraphGrammarAccumulator,
    LatentAttributedTemporalGraphGrammarAccumulatorFactory,
    LatentAttributedTemporalGraphGrammarDistribution,
    LatentAttributedTemporalGraphGrammarEstimator,
    LatentAttributedTemporalGraphGrammarSampler,
    LatentChurningTemporalGraphGrammarAccumulator,
    LatentChurningTemporalGraphGrammarAccumulatorFactory,
    LatentChurningTemporalGraphGrammarDistribution,
    LatentChurningTemporalGraphGrammarEstimator,
    LatentChurningTemporalGraphGrammarSampler,
    LatentTemporalGraphGrammarAccumulator,
    LatentTemporalGraphGrammarAccumulatorFactory,
    LatentTemporalGraphGrammarDistribution,
    LatentTemporalGraphGrammarEstimator,
    LatentTemporalGraphGrammarSampler,
    TemporalGraphGrammarAccumulator,
    TemporalGraphGrammarAccumulatorFactory,
    TemporalGraphGrammarDistribution,
    TemporalGraphGrammarEstimator,
    TemporalGraphGrammarSampler,
)
from mixle.stats.latent.chained_attention import (
    ChainedAttentionAccumulator,
    ChainedAttentionAccumulatorFactory,
    ChainedAttentionDataEncoder,
    ChainedAttentionDistribution,
    ChainedAttentionEstimator,
    ChainedAttentionSampler,
)
from mixle.stats.latent.dirac_length import (
    DiracLengthMixtureDistribution,
    DiracLengthMixtureEnumerator,
    DiracLengthMixtureEstimator,
    DiracLengthMixtureSampler,
)
from mixle.stats.latent.gated_mixture import (
    GatedMixtureAccumulator,
    GatedMixtureAccumulatorFactory,
    GatedMixtureDataEncoder,
    GatedMixtureDistribution,
    GatedMixtureEstimator,
    GatedMixtureSampler,
    SoftmaxGate,
)
from mixle.stats.latent.heterogeneous_mixture import (
    HeterogeneousMixtureDataEncoder,
    HeterogeneousMixtureDistribution,
    HeterogeneousMixtureEnumerator,
    HeterogeneousMixtureEstimator,
    HeterogeneousMixtureSampler,
)
from mixle.stats.latent.heterogeneous_pcfg import (
    HeterogeneousPCFGDataEncoder,
    HeterogeneousPCFGDistribution,
    HeterogeneousPCFGEnumerator,
    HeterogeneousPCFGEstimator,
    HeterogeneousPCFGSampler,
    InducedHeterogeneousPCFGEstimator,
)
from mixle.stats.latent.hidden_association import (
    HiddenAssociationDataEncoder,
    HiddenAssociationDistribution,
    HiddenAssociationEstimator,
    HiddenAssociationSampler,
)

### Reduced Generic Distributions
from mixle.stats.latent.hierarchical_mixture import (
    HierarchicalMixtureDataEncoder,
    HierarchicalMixtureDistribution,
    HierarchicalMixtureEnumerator,
    HierarchicalMixtureEstimator,
    HierarchicalMixtureSampler,
)
from mixle.stats.latent.indian_buffet_process import (
    IndianBuffetProcessDataEncoder,
    IndianBuffetProcessDistribution,
    IndianBuffetProcessEstimator,
    IndianBuffetProcessSampler,
)
from mixle.stats.latent.joint_mixture import (
    JointMixtureDataEncoder,
    JointMixtureDistribution,
    JointMixtureEnumerator,
    JointMixtureEstimator,
    JointMixtureSampler,
)
from mixle.stats.latent.mixture import (
    MixtureDataEncoder,
    MixtureDistribution,
    MixtureEnumerator,
    MixtureEstimator,
    MixtureSampler,
    mixture_prior,
)
from mixle.stats.latent.probabilistic_pca import (
    ProbabilisticPCADataEncoder,
    ProbabilisticPCADistribution,
    ProbabilisticPCAEstimator,
    ProbabilisticPCASampler,
)
from mixle.stats.latent.responsibility_attention import (
    ResponsibilityAttentionAccumulator,
    ResponsibilityAttentionAccumulatorFactory,
    ResponsibilityAttentionDataEncoder,
    ResponsibilityAttentionDistribution,
    ResponsibilityAttentionEstimator,
    ResponsibilityAttentionSampler,
    sequence_to_triples,
)
from mixle.stats.latent.scheduled_hidden_markov_model import (
    ByLength,
    ByPosition,
    ByRelativePosition,
    Homogeneous,
    ScheduledHiddenMarkovModelDistribution,
    ScheduledHMMEstimator,
)
from mixle.stats.latent.segmental_hidden_markov_model import (
    SegmentalHiddenMarkovDataEncoder,
    SegmentalHiddenMarkovDistribution,
    SegmentalHiddenMarkovEstimator,
    SegmentalHiddenMarkovModelDataEncoder,
    SegmentalHiddenMarkovModelDistribution,
    SegmentalHiddenMarkovModelEstimator,
    SegmentalHiddenMarkovModelSampler,
    SegmentalHiddenMarkovSampler,
)
from mixle.stats.latent.semi_supervised_mixture import (
    SemiSupervisedMixtureDataEncoder,
    SemiSupervisedMixtureDistribution,
    SemiSupervisedMixtureEstimator,
    SemiSupervisedMixtureSampler,
)
from mixle.stats.latent.structured_hmm import (
    BlockDiagonalTransition,
    DenseTransition,
    ExplicitDurationHMM,
    InputOutputHMM,
    KroneckerTransition,
    LowRankTransition,
    SparseTransition,
    StructuredHMM,
    StructuredHMMDataEncoder,
    StructuredHMMEstimator,
    TransitionOperator,
    banded_edges,
    dirichlet_transition,
    jit_forward_loglik,
    kron_initial,
    left_to_right_edges,
    stationary_initial,
    sticky_transition,
)
from mixle.stats.latent.variational_embedding_attention import (
    VariationalEmbeddingAttentionAccumulator,
    VariationalEmbeddingAttentionAccumulatorFactory,
    VariationalEmbeddingAttentionDataEncoder,
    VariationalEmbeddingAttentionDistribution,
    VariationalEmbeddingAttentionEstimator,
    VariationalEmbeddingAttentionSampler,
)
from mixle.stats.latent.variational_multihop_attention import (
    VariationalMultiHopAttentionAccumulator,
    VariationalMultiHopAttentionAccumulatorFactory,
    VariationalMultiHopAttentionDataEncoder,
    VariationalMultiHopAttentionDistribution,
    VariationalMultiHopAttentionEstimator,
    VariationalMultiHopAttentionSampler,
)
from mixle.stats.matrix.inverse_wishart import (
    InverseWishartDataEncoder,
    InverseWishartDistribution,
    InverseWishartEstimator,
    InverseWishartSampler,
)
from mixle.stats.matrix.lkj import (
    LKJDataEncoder,
    LKJDistribution,
    LKJEstimator,
    LKJSampler,
)
from mixle.stats.matrix.matrix_normal import (
    MatrixNormalDataEncoder,
    MatrixNormalDistribution,
    MatrixNormalEstimator,
    MatrixNormalSampler,
)
from mixle.stats.matrix.wishart import (
    WishartDataEncoder,
    WishartDistribution,
    WishartEstimator,
    WishartSampler,
)
from mixle.stats.missing import MISSING, composite_with_missing, marginalized
from mixle.stats.multivariate.categorical_multinomial import (
    MultinomialDataEncoder,
    MultinomialDistribution,
    MultinomialEnumerator,
    MultinomialEstimator,
    MultinomialSampler,
)
from mixle.stats.multivariate.clayton_copula import (
    ClaytonCopulaDistribution,
    ClaytonCopulaEstimator,
    ClaytonCopulaSampler,
)
from mixle.stats.multivariate.diagonal_gaussian import (
    DiagonalGaussianDataEncoder,
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    DiagonalGaussianSampler,
)
from mixle.stats.multivariate.dirichlet_multinomial import (
    DirichletMultinomialDataEncoder,
    DirichletMultinomialDistribution,
    DirichletMultinomialEstimator,
    DirichletMultinomialSampler,
)
from mixle.stats.multivariate.frank_copula import (
    FrankCopulaDistribution,
    FrankCopulaEstimator,
    FrankCopulaSampler,
)
from mixle.stats.multivariate.gaussian_copula import (
    GaussianCopulaDataEncoder,
    GaussianCopulaDistribution,
    GaussianCopulaEstimator,
    GaussianCopulaSampler,
)
from mixle.stats.multivariate.gumbel_copula import (
    GumbelCopulaDistribution,
    GumbelCopulaEstimator,
    GumbelCopulaSampler,
)
from mixle.stats.multivariate.integer_multinomial import (
    IntegerMultinomialDataEncoder,
    IntegerMultinomialDistribution,
    IntegerMultinomialEnumerator,
    IntegerMultinomialEstimator,
    IntegerMultinomialSampler,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
    MultivariateGaussianSampler,
)
from mixle.stats.multivariate.multivariate_student_t import (
    MultivariateStudentTDataEncoder,
    MultivariateStudentTDistribution,
    MultivariateStudentTEstimator,
    MultivariateStudentTSampler,
)
from mixle.stats.multivariate.rvine_copula import (
    RVineCopulaDistribution,
    RVineCopulaEstimator,
    RVineCopulaSampler,
)
from mixle.stats.multivariate.student_t_copula import (
    StudentTCopulaDistribution,
    StudentTCopulaEstimator,
    StudentTCopulaSampler,
)
from mixle.stats.multivariate.vine_copula import (
    ClaytonPairCopula,
    CVineCopulaDistribution,
    CVineCopulaEstimator,
    CVineCopulaSampler,
    DVineCopulaDistribution,
    DVineCopulaEstimator,
    DVineCopulaSampler,
    FrankPairCopula,
    GaussianPairCopula,
    GumbelPairCopula,
    IndependencePairCopula,
    StudentTPairCopula,
)
from mixle.stats.processes.birth_death import (
    BirthDeathSamplingDataEncoder,
    BirthDeathSamplingDistribution,
    BirthDeathSamplingEstimator,
    BirthDeathSamplingSampler,
)
from mixle.stats.processes.chinese_restaurant_process import (
    ChineseRestaurantProcessDataEncoder,
    ChineseRestaurantProcessDistribution,
    ChineseRestaurantProcessEstimator,
    ChineseRestaurantProcessSampler,
)
from mixle.stats.processes.ctmc import (
    ContinuousTimeMarkovChainDataEncoder,
    ContinuousTimeMarkovChainDistribution,
    ContinuousTimeMarkovChainEstimator,
    ContinuousTimeMarkovChainSampler,
)
from mixle.stats.processes.hawkes_process import (
    HawkesProcessDataEncoder,
    HawkesProcessDistribution,
    HawkesProcessEstimator,
    HawkesProcessSampler,
)
from mixle.stats.processes.inhomogeneous_poisson import (
    InhomogeneousPoissonProcessDataEncoder,
    InhomogeneousPoissonProcessDistribution,
    InhomogeneousPoissonProcessEstimator,
    InhomogeneousPoissonProcessSampler,
)
from mixle.stats.processes.multivariate_hawkes import (
    MultivariateHawkesProcessDataEncoder,
    MultivariateHawkesProcessDistribution,
    MultivariateHawkesProcessEstimator,
    MultivariateHawkesProcessSampler,
)
from mixle.stats.processes.renewal_process import (
    RenewalProcessDataEncoder,
    RenewalProcessDistribution,
    RenewalProcessEstimator,
    RenewalProcessSampler,
)
from mixle.stats.rankings.bradley_terry import (
    BradleyTerryAccumulator,
    BradleyTerryAccumulatorFactory,
    BradleyTerryDataEncoder,
    BradleyTerryDistribution,
    BradleyTerryEstimator,
    BradleyTerrySampler,
)
from mixle.stats.rankings.ewens import (
    EwensAccumulator,
    EwensAccumulatorFactory,
    EwensDataEncoder,
    EwensDistribution,
    EwensEstimator,
    EwensSampler,
)
from mixle.stats.rankings.generalized_mallows import (
    GeneralizedMallowsAccumulator,
    GeneralizedMallowsAccumulatorFactory,
    GeneralizedMallowsDataEncoder,
    GeneralizedMallowsDistribution,
    GeneralizedMallowsEstimator,
    GeneralizedMallowsSampler,
)
from mixle.stats.rankings.generalized_mallows_model import (
    GeneralizedMallowsModelAccumulator,
    GeneralizedMallowsModelAccumulatorFactory,
    GeneralizedMallowsModelDataEncoder,
    GeneralizedMallowsModelDistribution,
    GeneralizedMallowsModelEstimator,
    GeneralizedMallowsModelSampler,
)
from mixle.stats.rankings.low_rank_permutation import (
    LowRankPermutationAccumulator,
    LowRankPermutationAccumulatorFactory,
    LowRankPermutationDataEncoder,
    LowRankPermutationDistribution,
    LowRankPermutationEstimator,
    LowRankPermutationSampler,
)
from mixle.stats.rankings.mallows import (
    MallowsDataEncoder,
    MallowsDistribution,
    MallowsEnumerator,
    MallowsEstimator,
    MallowsSampler,
)
from mixle.stats.rankings.matching import (
    MatchingDataEncoder,
    MatchingDistribution,
    MatchingEnumerator,
    MatchingEstimator,
    MatchingSampler,
)
from mixle.stats.rankings.paired_comparison import (
    DavidsonDistribution,
    DavidsonEstimator,
    PairDataEncoder,
    PairWinAccumulator,
    PairWinAccumulatorFactory,
    RaoKupperDistribution,
    RaoKupperEstimator,
    ThurstoneMostellerDistribution,
    ThurstoneMostellerEstimator,
    ThurstoneMostellerSampler,
)
from mixle.stats.rankings.plackett_luce import (
    PlackettLuceDataEncoder,
    PlackettLuceDistribution,
    PlackettLuceEnumerator,
    PlackettLuceEstimator,
    PlackettLuceSampler,
)
from mixle.stats.rankings.spearman_rho import (
    SpearmanRankingDataEncoder,
    SpearmanRankingDistribution,
    SpearmanRankingEnumerator,
    SpearmanRankingEstimator,
    SpearmanRankingSampler,
)
from mixle.stats.rankings.thurstone import (
    ThurstoneAccumulator,
    ThurstoneAccumulatorFactory,
    ThurstoneDataEncoder,
    ThurstoneDistribution,
    ThurstoneEstimator,
    ThurstoneSampler,
)
from mixle.stats.sequences.integer_markov_chain import (
    IntegerMarkovChainDataEncoder,
    IntegerMarkovChainDistribution,
    IntegerMarkovChainEnumerator,
    IntegerMarkovChainEstimator,
    IntegerMarkovChainSampler,
)
from mixle.stats.sequences.markov_chain import (
    MarkovChainDataEncoder,
    MarkovChainDistribution,
    MarkovChainEnumerator,
    MarkovChainEstimator,
    MarkovChainSampler,
)
from mixle.stats.sequences.sparse_markov_transform import (
    SparseMarkovAssociationDataEncoder,
    SparseMarkovAssociationDistribution,
    SparseMarkovAssociationEstimator,
    SparseMarkovAssociationSampler,
)
from mixle.stats.sets.bernoulli_set import (
    BernoulliSetDataEncoder,
    BernoulliSetDistribution,
    BernoulliSetEnumerator,
    BernoulliSetEstimator,
    BernoulliSetSampler,
)
from mixle.stats.sets.integer_bernoulli_edit import (
    IntegerBernoulliEditDataEncoder,
    IntegerBernoulliEditDistribution,
    IntegerBernoulliEditEnumerator,
    IntegerBernoulliEditEstimator,
    IntegerBernoulliEditSampler,
)
from mixle.stats.sets.integer_bernoulli_set import (
    IntegerBernoulliSetDataEncoder,
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEnumerator,
    IntegerBernoulliSetEstimator,
    IntegerBernoulliSetSampler,
)
from mixle.stats.sets.integer_step_bernoulli_edit import (
    IntegerStepBernoulliEditDataEncoder,
    IntegerStepBernoulliEditDistribution,
    IntegerStepBernoulliEditEnumerator,
    IntegerStepBernoulliEditEstimator,
    IntegerStepBernoulliEditSampler,
)
from mixle.stats.trees.chow_liu_tree import (
    ChowLiuTreeDataEncoder,
    ChowLiuTreeDistribution,
    ChowLiuTreeEnumerator,
    ChowLiuTreeEstimator,
    ChowLiuTreeSampler,
)
from mixle.stats.trees.integer_chow_liu_tree import (
    IntegerChowLiuTreeDataEncoder,
    IntegerChowLiuTreeDistribution,
    IntegerChowLiuTreeEnumerator,
    IntegerChowLiuTreeEstimator,
    IntegerChowLiuTreeSampler,
)
from mixle.stats.trees.spanning_tree import (
    SpanningTreeDataEncoder,
    SpanningTreeDistribution,
    SpanningTreeEnumerator,
    SpanningTreeEstimator,
    SpanningTreeSampler,
)

### Continuous base distributions
from mixle.stats.univariate.continuous.beta import BetaDataEncoder, BetaDistribution, BetaEstimator, BetaSampler
from mixle.stats.univariate.continuous.exgaussian import (
    ExponentiallyModifiedGaussianDataEncoder,
    ExponentiallyModifiedGaussianDistribution,
    ExponentiallyModifiedGaussianEstimator,
    ExponentiallyModifiedGaussianSampler,
)
from mixle.stats.univariate.continuous.exponential import (
    ExponentialDataEncoder,
    ExponentialDistribution,
    ExponentialEstimator,
    ExponentialSampler,
)
from mixle.stats.univariate.continuous.gamma import GammaDataEncoder, GammaDistribution, GammaEstimator, GammaSampler
from mixle.stats.univariate.continuous.gaussian import (
    GaussianDataEncoder,
    GaussianDistribution,
    GaussianEstimator,
    GaussianSampler,
)
from mixle.stats.univariate.continuous.generalized_extreme_value import (
    GeneralizedExtremeValueDataEncoder,
    GeneralizedExtremeValueDistribution,
    GeneralizedExtremeValueEstimator,
    GeneralizedExtremeValueSampler,
)
from mixle.stats.univariate.continuous.generalized_gaussian import (
    GeneralizedGaussianDataEncoder,
    GeneralizedGaussianDistribution,
    GeneralizedGaussianEstimator,
    GeneralizedGaussianSampler,
)
from mixle.stats.univariate.continuous.generalized_pareto import (
    GeneralizedParetoDataEncoder,
    GeneralizedParetoDistribution,
    GeneralizedParetoEstimator,
    GeneralizedParetoSampler,
)
from mixle.stats.univariate.continuous.gumbel import (
    GumbelDataEncoder,
    GumbelDistribution,
    GumbelEstimator,
    GumbelSampler,
)
from mixle.stats.univariate.continuous.half_normal import (
    HalfNormalDataEncoder,
    HalfNormalDistribution,
    HalfNormalEstimator,
    HalfNormalSampler,
)
from mixle.stats.univariate.continuous.inverse_gamma import (
    InverseGammaDataEncoder,
    InverseGammaDistribution,
    InverseGammaEstimator,
    InverseGammaSampler,
)
from mixle.stats.univariate.continuous.inverse_gaussian import (
    InverseGaussianDataEncoder,
    InverseGaussianDistribution,
    InverseGaussianEstimator,
    InverseGaussianSampler,
)
from mixle.stats.univariate.continuous.laplace import (
    LaplaceDataEncoder,
    LaplaceDistribution,
    LaplaceEstimator,
    LaplaceSampler,
)
from mixle.stats.univariate.continuous.log_gaussian import (
    LogGaussianDataEncoder,
    LogGaussianDistribution,
    LogGaussianEstimator,
    LogGaussianSampler,
)
from mixle.stats.univariate.continuous.logistic import (
    LogisticDataEncoder,
    LogisticDistribution,
    LogisticEstimator,
    LogisticSampler,
)
from mixle.stats.univariate.continuous.nakagami import (
    NakagamiDataEncoder,
    NakagamiDistribution,
    NakagamiEstimator,
    NakagamiSampler,
)
from mixle.stats.univariate.continuous.pareto import (
    ParetoDataEncoder,
    ParetoDistribution,
    ParetoEstimator,
    ParetoSampler,
)
from mixle.stats.univariate.continuous.rayleigh import (
    RayleighDataEncoder,
    RayleighDistribution,
    RayleighEstimator,
    RayleighSampler,
)
from mixle.stats.univariate.continuous.rician import (
    RicianDataEncoder,
    RicianDistribution,
    RicianEstimator,
    RicianSampler,
)
from mixle.stats.univariate.continuous.skew_normal import (
    SkewNormalDataEncoder,
    SkewNormalDistribution,
    SkewNormalEstimator,
    SkewNormalSampler,
)
from mixle.stats.univariate.continuous.student_t import (
    StudentTDataEncoder,
    StudentTDistribution,
    StudentTEstimator,
    StudentTSampler,
)
from mixle.stats.univariate.continuous.tweedie import (
    TweedieDataEncoder,
    TweedieDistribution,
    TweedieEstimator,
    TweedieSampler,
)
from mixle.stats.univariate.continuous.uniform import (
    UniformDataEncoder,
    UniformDistribution,
    UniformEstimator,
    UniformSampler,
)
from mixle.stats.univariate.continuous.weibull import (
    WeibullDataEncoder,
    WeibullDistribution,
    WeibullEstimator,
    WeibullSampler,
)
from mixle.stats.univariate.discrete.bernoulli import (
    BernoulliDataEncoder,
    BernoulliDistribution,
    BernoulliEnumerator,
    BernoulliEstimator,
    BernoulliSampler,
)
from mixle.stats.univariate.discrete.beta_binomial import (
    BetaBinomialDataEncoder,
    BetaBinomialDistribution,
    BetaBinomialEstimator,
    BetaBinomialSampler,
)
from mixle.stats.univariate.discrete.binomial import (
    BinomialDataEncoder,
    BinomialDistribution,
    BinomialEnumerator,
    BinomialEstimator,
    BinomialSampler,
)
from mixle.stats.univariate.discrete.categorical import (
    CategoricalDataEncoder,
    CategoricalDistribution,
    CategoricalEnumerator,
    CategoricalEstimator,
    CategoricalSampler,
)
from mixle.stats.univariate.discrete.geometric import (
    GeometricDataEncoder,
    GeometricDistribution,
    GeometricEnumerator,
    GeometricEstimator,
    GeometricSampler,
)
from mixle.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDataEncoder,
    IntegerCategoricalDistribution,
    IntegerCategoricalEnumerator,
    IntegerCategoricalEstimator,
    IntegerCategoricalSampler,
)
from mixle.stats.univariate.discrete.integer_uniform_spike import (
    IntegerUniformSpikeDataEncoder,
    IntegerUniformSpikeDistribution,
    IntegerUniformSpikeEnumerator,
    IntegerUniformSpikeEstimator,
    IntegerUniformSpikeSampler,
)
from mixle.stats.univariate.discrete.logseries import (
    LogSeriesDataEncoder,
    LogSeriesDistribution,
    LogSeriesEstimator,
    LogSeriesSampler,
)
from mixle.stats.univariate.discrete.negative_binomial import (
    NegativeBinomialDataEncoder,
    NegativeBinomialDistribution,
    NegativeBinomialEnumerator,
    NegativeBinomialEstimator,
    NegativeBinomialSampler,
)
from mixle.stats.univariate.discrete.point_mass import (
    PointMassDataEncoder,
    PointMassDistribution,
    PointMassEnumerator,
    PointMassEstimator,
    PointMassSampler,
)
from mixle.stats.univariate.discrete.poisson import (
    PoissonDataEncoder,
    PoissonDistribution,
    PoissonEnumerator,
    PoissonEstimator,
    PoissonSampler,
)
from mixle.stats.univariate.discrete.skellam import (
    SkellamDataEncoder,
    SkellamDistribution,
    SkellamEstimator,
    SkellamSampler,
)


def _register_builtin_compute_metadata() -> None:
    numba_caps = DistributionCapabilities(engine_ready=("numpy",), kernel_status="numba_adapter")
    backend_caps = DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")
    legacy_numpy_caps = DistributionCapabilities(engine_ready=("numpy",), kernel_status="legacy_numpy")
    for dist_type in (
        GaussianDistribution,
        LogGaussianDistribution,
        GammaDistribution,
        InverseGammaDistribution,
        InverseGaussianDistribution,
        GumbelDistribution,
        HalfNormalDistribution,
        BernoulliDistribution,
        StudentTDistribution,
        LogisticDistribution,
        WeibullDistribution,
        RayleighDistribution,
        ParetoDistribution,
        UniformDistribution,
        IntegerCategoricalDistribution,
        PoissonDistribution,
        ExponentialDistribution,
        GeometricDistribution,
        BinomialDistribution,
        NegativeBinomialDistribution,
        DiagonalGaussianDistribution,
        BetaDistribution,
        LaplaceDistribution,
        MultivariateGaussianDistribution,
        IntegerBernoulliEditDistribution,
        IntegerStepBernoulliEditDistribution,
        SpearmanRankingDistribution,
        VonMisesFisherDistribution,
        MultivariateStudentTDistribution,
        VonMisesDistribution,
        LogSeriesDistribution,
    ):
        register_capabilities(dist_type, backend_caps)

    for dist_type in (
        CategoricalDistribution,
        CompositeDistribution,
        SequenceDistribution,
        OptionalDistribution,
        IgnoredDistribution,
        MixtureDistribution,
    ):
        register_capabilities(dist_type, numba_caps)

    for dist_type in (
        SegmentalHiddenMarkovModelDistribution,
        HeterogeneousMixtureDistribution,
        HierarchicalMixtureDistribution,
        JointMixtureDistribution,
        SemiSupervisedMixtureDistribution,
        DiracLengthMixtureDistribution,
        HiddenAssociationDistribution,
    ):
        register_capabilities(dist_type, legacy_numpy_caps)

    # The HMM/PLSI/LDA distributions live in heavy numba modules that are imported
    # lazily (see _LAZY_NAMES / __getattr__ below). Their legacy_numpy capabilities
    # are registered the first time one of those modules is accessed, which always
    # precedes first use of the distribution. See _register_lazy_module_capabilities.

    numpy_only_reasons = {}
    for dist_type, reason in numpy_only_reasons.items():
        register_capabilities(
            dist_type,
            DistributionCapabilities(engine_ready=("numpy",), kernel_status="numpy_only", numpy_only_reason=reason),
        )

    for dist_type in (
        GaussianDistribution,
        PoissonDistribution,
        ExponentialDistribution,
        BernoulliDistribution,
        CategoricalDistribution,
        GammaDistribution,
        InverseGammaDistribution,
        InverseGaussianDistribution,
        GumbelDistribution,
        HalfNormalDistribution,
        LogGaussianDistribution,
        BinomialDistribution,
        NegativeBinomialDistribution,
        GeometricDistribution,
        DiagonalGaussianDistribution,
        StudentTDistribution,
        LogisticDistribution,
        WeibullDistribution,
        RayleighDistribution,
        ParetoDistribution,
        UniformDistribution,
        IntegerCategoricalDistribution,
        BetaDistribution,
        DirichletDistribution,
        LaplaceDistribution,
        MultivariateGaussianDistribution,
        NullDistribution,
        PointMassDistribution,
        BernoulliSetDistribution,
        IndianBuffetProcessDistribution,
        IntegerUniformSpikeDistribution,
        IntegerBernoulliSetDistribution,
        SpearmanRankingDistribution,
        VonMisesFisherDistribution,
        MultivariateStudentTDistribution,
        VonMisesDistribution,
        LogSeriesDistribution,
    ):
        register_declaration(dist_type.compute_declaration())

    register_kernel_factory(MixtureDistribution, StackedMixtureKernelFactory())


_register_builtin_compute_metadata()


# ---------------------------------------------------------------------------
# Lazy submodule loading (PEP 562).
#
# A handful of latent modules pull in numba and decorate cache=True kernels at
# import time, which costs ~1s even when no HMM/PLSI/LDA model is used. We map
# their public names here and import the module only on first attribute access.
# The names still resolve via `from mixle.stats import X` and `import *` (they
# remain in __all__), and the module's legacy_numpy compute capabilities are
# registered on that first access -- always before the distribution is used.
# ---------------------------------------------------------------------------
_LAZY_NAMES: dict[str, str] = {
    # hidden_markov
    "HiddenMarkovDataEncoder": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovEstimator": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovModelAccumulator": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovModelAccumulatorFactory": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovModelDataEncoder": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovModelDistribution": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovModelEnumerator": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovModelEstimator": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovModelSampler": "mixle.stats.latent.hidden_markov",
    "HiddenMarkovSampler": "mixle.stats.latent.hidden_markov",
    # quantized_hmm (imports hidden_markov at module top)
    "QuantizedHiddenMarkovEstimator": "mixle.stats.latent.quantized_hidden_markov_model",
    "QuantizedHiddenMarkovModelDistribution": "mixle.stats.latent.quantized_hidden_markov_model",
    "QuantizedHiddenMarkovModelEnumerator": "mixle.stats.latent.quantized_hidden_markov_model",
    "QuantizedHiddenMarkovModelEstimator": "mixle.stats.latent.quantized_hidden_markov_model",
    # tree_hmm
    "TreeHiddenMarkovEstimator": "mixle.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovModelDistribution": "mixle.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovModelEstimator": "mixle.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovModelSampler": "mixle.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovSampler": "mixle.stats.latent.tree_hidden_markov_model",
    # int_plsi
    "IntegerProbabilisticLatentSemanticIndexingDataEncoder": "mixle.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerProbabilisticLatentSemanticIndexingDistribution": "mixle.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerProbabilisticLatentSemanticIndexingEstimator": "mixle.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerProbabilisticLatentSemanticIndexingSampler": "mixle.stats.latent.integer_probabilistic_latent_semantic_indexing",
    # int_hidden_association (imports int_plsi + numba)
    "IntegerHiddenAssociationDataEncoder": "mixle.stats.latent.integer_hidden_association",
    "IntegerHiddenAssociationDistribution": "mixle.stats.latent.integer_hidden_association",
    "IntegerHiddenAssociationEstimator": "mixle.stats.latent.integer_hidden_association",
    "IntegerHiddenAssociationSampler": "mixle.stats.latent.integer_hidden_association",
    # lda
    "LDADataEncoder": "mixle.stats.latent.lda",
    "LDADistribution": "mixle.stats.latent.lda",
    "LDAEstimator": "mixle.stats.latent.lda",
    "LDASampler": "mixle.stats.latent.lda",
}

# Distribution classes (by attribute name) whose legacy_numpy capabilities must be
# registered when their lazy module is first loaded -- previously registered eagerly
# in _register_builtin_compute_metadata.
_LAZY_MODULE_CAP_NAMES: dict[str, tuple[str, ...]] = {
    "mixle.stats.latent.hidden_markov": ("HiddenMarkovModelDistribution",),
    "mixle.stats.latent.quantized_hidden_markov_model": ("QuantizedHiddenMarkovModelDistribution",),
    "mixle.stats.latent.tree_hidden_markov_model": ("TreeHiddenMarkovModelDistribution",),
    "mixle.stats.latent.integer_hidden_association": ("IntegerHiddenAssociationDistribution",),
    "mixle.stats.latent.integer_probabilistic_latent_semantic_indexing": (
        "IntegerProbabilisticLatentSemanticIndexingDistribution",
    ),
    "mixle.stats.latent.lda": ("LDADistribution",),
}

_LAZY_CAPS_REGISTERED: set[str] = set()


def _register_lazy_module_capabilities(module_path: str, module) -> None:
    """Register the legacy_numpy capabilities for a heavy module on first load."""
    if module_path in _LAZY_CAPS_REGISTERED:
        return
    _LAZY_CAPS_REGISTERED.add(module_path)
    caps = DistributionCapabilities(engine_ready=("numpy",), kernel_status="legacy_numpy")
    for cls_name in _LAZY_MODULE_CAP_NAMES.get(module_path, ()):
        register_capabilities(getattr(module, cls_name), caps)


def __getattr__(name: str):
    module_path = _LAZY_NAMES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    _register_lazy_module_capabilities(module_path, module)
    # Bind every public name this module owns into this namespace so subsequent
    # accesses skip __getattr__ entirely.
    value = None
    for attr, path in _LAZY_NAMES.items():
        if path == module_path:
            obj = getattr(module, attr)
            globals()[attr] = obj
            if attr == name:
                value = obj
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NAMES))


def load_models(x: str):
    """Reconstruct a model or collection of models from dump_models() JSON."""
    from mixle.utils.serialization import from_json

    return from_json(x)


def dump_models(x) -> str:
    """Serialize a stats model or collection of models to safe strict JSON."""
    from mixle.utils.serialization import to_json

    return to_json(x)


# Vectorized sequence-driver API — implementations live in mixle.stats.compute.sequence so the
# inference machinery can import them without importing this package. Re-exported here unchanged.
from mixle.stats.compute.sequence import (  # noqa: E402
    density,
    log_density,
    seq_encode,
    seq_log_density,
    seq_log_density_sum,
)
