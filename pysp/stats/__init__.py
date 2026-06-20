"""Load SequenceEncodableProbabilityDistribution, DistributionSampler, ParameterEstimator,
and DataSequenceEncoder objects for the distributions in pyps.stats. This module also loads functions used to
estimate Distributions from data sets.
"""

from __future__ import annotations

__all__ = [
    # Bayesian (conjugate/variational) families folded in from the former pysp.bstats
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
    "initialize",
    "estimate",
    "seq_encode",
    "seq_log_density",
    "seq_log_density_sum",
    "log_density",
    "density",
    "seq_estimate",
    "seq_initialize",
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
    "BinomialSampler",
    "BinomialEstimator",
    "BinomialDataEncoder",
    "BinomialEnumerator",
    "CategoricalDistribution",
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
    "VonMisesSampler",
    "VonMisesEstimator",
    "VonMisesDataEncoder",
    "WeibullDistribution",
    "WeibullSampler",
    "WeibullEstimator",
    "WeibullDataEncoder",
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
    "ICLTreeDistribution",
    "ICLTreeEstimator",
    "ICLTreeSampler",
    "ICLTreeDataEncoder",
    "ICLTreeEnumerator",
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
    "IntegerPLSIDistribution",
    "IntegerPLSISampler",
    "IntegerPLSIEstimator",
    "IntegerPLSIDataEncoder",
    "IntegerUniformSpikeDistribution",
    "IntegerUniformSpikeEstimator",
    "IntegerUniformSpikeSampler",
    "IntegerUniformSpikeDataEncoder",
    "IntegerUniformSpikeEnumerator",
    "IntegerMultinomialDistribution",
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
    "MixtureSampler",
    "MixtureEstimator",
    "MixtureDataEncoder",
    "MixtureEnumerator",
    "MultivariateGaussianDistribution",
    "MultivariateGaussianEstimator",
    "MultivariateGaussianSampler",
    "MultivariateGaussianDataEncoder",
    "NullDistribution",
    "NullSampler",
    "NullEstimator",
    "NullDataEncoder",
    "NullEnumerator",
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
    "SelectEnumerator",
    "SequenceDistribution",
    "SequenceSampler",
    "SequenceEstimator",
    "SequenceDataEncoder",
    "SequenceEnumerator",
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
    "ExponentiallyModifiedGaussianDistribution",
    "ExponentiallyModifiedGaussianSampler",
    "ExponentiallyModifiedGaussianEstimator",
    "ExponentiallyModifiedGaussianDataEncoder",
    "SkellamDistribution",
    "SkellamSampler",
    "SkellamEstimator",
    "SkellamDataEncoder",
    "TweedieDistribution",
    "TweedieSampler",
    "TweedieEstimator",
    "TweedieDataEncoder",
    "BirthDeathSamplingDistribution",
    "BirthDeathSamplingSampler",
    "BirthDeathSamplingEstimator",
    "BirthDeathSamplingDataEncoder",
    "InhomogeneousPoissonProcessDistribution",
    "InhomogeneousPoissonProcessSampler",
    "InhomogeneousPoissonProcessEstimator",
    "InhomogeneousPoissonProcessDataEncoder",
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
    "StochasticBlockGraphDistribution",
    "StochasticBlockGraphSampler",
    "StochasticBlockGraphAccumulator",
    "StochasticBlockGraphAccumulatorFactory",
    "StochasticBlockGraphEstimator",
]

### Abstract Classes
from pysp.stats.bayes.catdirichlet import DictDirichletDistribution, DictDirichletSampler
from pysp.stats.bayes.dirichlet import DirichletDataEncoder, DirichletDistribution, DirichletEstimator, DirichletSampler
from pysp.stats.bayes.dpm import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
    DirichletProcessMixtureSampler,
)
from pysp.stats.bayes.hdpm import (
    HierarchicalDirichletProcessMixtureDistribution,
    HierarchicalDirichletProcessMixtureEstimator,
    HierarchicalDirichletProcessMixtureSampler,
)
from pysp.stats.bayes.mvngamma import MultivariateNormalGammaDistribution, MultivariateNormalGammaSampler
from pysp.stats.bayes.normgamma import NormalGammaDistribution, NormalGammaSampler
from pysp.stats.bayes.normwishart import NormalWishartDistribution, NormalWishartSampler
from pysp.stats.bayes.pitman_yor import (
    PitmanYorProcessDataEncoder,
    PitmanYorProcessDistribution,
    PitmanYorProcessEstimator,
    PitmanYorProcessSampler,
)
from pysp.stats.bayes.symdirichlet import SymmetricDirichletDistribution, SymmetricDirichletSampler

### combinators distributions
from pysp.stats.combinator.censored import (
    CensoredAccumulator,
    CensoredAccumulatorFactory,
    CensoredDataEncoder,
    CensoredDistribution,
    CensoredEstimator,
    CensoredSampler,
)
from pysp.stats.combinator.composite import (
    CompositeDataEncoder,
    CompositeDistribution,
    CompositeEnumerator,
    CompositeEstimator,
    CompositeSampler,
)
from pysp.stats.combinator.conditional import (
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
from pysp.stats.combinator.exponential_tilt import (
    ExponentialTiltedDataEncoder,
    ExponentialTiltedDistribution,
    ExponentialTiltedEnumerator,
    ExponentialTiltedEstimator,
    ExponentialTiltedSampler,
    register_exponential_tilt,
    registered_tilt_families,
)
from pysp.stats.combinator.finite_stochastic_transform import (
    FiniteStochasticTransformDataEncoder,
    FiniteStochasticTransformDistribution,
    FiniteStochasticTransformEnumerator,
    FiniteStochasticTransformEstimator,
    FiniteStochasticTransformSampler,
)
from pysp.stats.combinator.ignored import IgnoredDataEncoder, IgnoredDistribution, IgnoredEstimator, IgnoredSampler
from pysp.stats.combinator.null_dist import (
    NullDataEncoder,
    NullDistribution,
    NullEnumerator,
    NullEstimator,
    NullSampler,
)
from pysp.stats.combinator.optional import (
    OptionalDataEncoder,
    OptionalDistribution,
    OptionalEnumerator,
    OptionalEstimator,
    OptionalSampler,
)
from pysp.stats.combinator.record import (
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
from pysp.stats.combinator.select import SelectDistribution, SelectEnumerator, SelectEstimator
from pysp.stats.combinator.sequence import (
    SequenceDataEncoder,
    SequenceDistribution,
    SequenceEnumerator,
    SequenceEstimator,
    SequenceSampler,
)
from pysp.stats.combinator.transform import (
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
from pysp.stats.combinator.truncated import (
    TruncatedDataEncoder,
    TruncatedDistribution,
    TruncatedEnumerator,
    TruncatedEstimator,
    TruncatedSampler,
)
from pysp.stats.combinator.weighted import WeightedDataEncoder, WeightedDistribution, WeightedEstimator
from pysp.stats.compute.backend import (
    BackendScoringError,
    backend_log_density_sum,
    backend_seq_component_log_density,
    backend_seq_log_density,
)
from pysp.stats.compute.capabilities import (
    DistributionCapabilities,
    capabilities_for,
    numpy_only_distribution_types,
    register_capabilities,
    registered_capability_types,
    supported_engines,
)
from pysp.stats.compute.declarations import (
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
from pysp.stats.compute.encoded import EncodedData, ResidentEncodedPayload, as_encoded_data, move_encoded_payload
from pysp.stats.compute.kernel import (
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
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
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
from pysp.stats.compute.stacked import (
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
from pysp.stats.graph.chow_liu_tree import (
    ChowLiuTreeDataEncoder,
    ChowLiuTreeDistribution,
    ChowLiuTreeEnumerator,
    ChowLiuTreeEstimator,
    ChowLiuTreeSampler,
)
from pysp.stats.graph.erdos_renyi_graph import (
    ErdosRenyiGraphAccumulator,
    ErdosRenyiGraphAccumulatorFactory,
    ErdosRenyiGraphDistribution,
    ErdosRenyiGraphEstimator,
    ErdosRenyiGraphSampler,
)
from pysp.stats.graph.icltree import (
    ICLTreeDataEncoder,
    ICLTreeDistribution,
    ICLTreeEnumerator,
    ICLTreeEstimator,
    ICLTreeSampler,
)
from pysp.stats.graph.int_markovchain import (
    IntegerMarkovChainDataEncoder,
    IntegerMarkovChainDistribution,
    IntegerMarkovChainEnumerator,
    IntegerMarkovChainEstimator,
    IntegerMarkovChainSampler,
)
from pysp.stats.graph.knowledge_graph import (
    KnowledgeGraphDataEncoder,
    KnowledgeGraphDistribution,
    KnowledgeGraphEnsemble,
    KnowledgeGraphEstimator,
    KnowledgeGraphSampler,
    fit_knowledge_graph_ensemble,
)
from pysp.stats.graph.mallows import (
    MallowsDataEncoder,
    MallowsDistribution,
    MallowsEnumerator,
    MallowsEstimator,
    MallowsSampler,
)
from pysp.stats.graph.markov_chain import (
    MarkovChainDataEncoder,
    MarkovChainDistribution,
    MarkovChainEnumerator,
    MarkovChainEstimator,
    MarkovChainSampler,
)
from pysp.stats.graph.matching import (
    MatchingDataEncoder,
    MatchingDistribution,
    MatchingEnumerator,
    MatchingEstimator,
    MatchingSampler,
)
from pysp.stats.graph.plackett_luce import (
    PlackettLuceDataEncoder,
    PlackettLuceDistribution,
    PlackettLuceEnumerator,
    PlackettLuceEstimator,
    PlackettLuceSampler,
)
from pysp.stats.graph.rdpg import (
    RandomDotProductGraphDistribution,
    RandomDotProductGraphEstimator,
    RandomDotProductGraphSampler,
)
from pysp.stats.graph.spanning_tree import (
    SpanningTreeDataEncoder,
    SpanningTreeDistribution,
    SpanningTreeEnumerator,
    SpanningTreeEstimator,
    SpanningTreeSampler,
)
from pysp.stats.graph.sparse_markov_transform import (
    SparseMarkovAssociationDataEncoder,
    SparseMarkovAssociationDistribution,
    SparseMarkovAssociationEstimator,
    SparseMarkovAssociationSampler,
)
from pysp.stats.graph.spearman_rho import (
    SpearmanRankingDataEncoder,
    SpearmanRankingDistribution,
    SpearmanRankingEnumerator,
    SpearmanRankingEstimator,
    SpearmanRankingSampler,
)
from pysp.stats.graph.stochastic_block_graph import (
    StochasticBlockGraphAccumulator,
    StochasticBlockGraphAccumulatorFactory,
    StochasticBlockGraphDistribution,
    StochasticBlockGraphEstimator,
    StochasticBlockGraphSampler,
)
from pysp.stats.latent.dirac_length import (
    DiracLengthMixtureDistribution,
    DiracLengthMixtureEnumerator,
    DiracLengthMixtureEstimator,
    DiracLengthMixtureSampler,
)
from pysp.stats.latent.heterogeneous_mixture import (
    HeterogeneousMixtureDataEncoder,
    HeterogeneousMixtureDistribution,
    HeterogeneousMixtureEnumerator,
    HeterogeneousMixtureEstimator,
    HeterogeneousMixtureSampler,
)
from pysp.stats.latent.heterogeneous_pcfg import (
    HeterogeneousPCFGDataEncoder,
    HeterogeneousPCFGDistribution,
    HeterogeneousPCFGEnumerator,
    HeterogeneousPCFGEstimator,
    HeterogeneousPCFGSampler,
    InducedHeterogeneousPCFGEstimator,
)
from pysp.stats.latent.hidden_association import (
    HiddenAssociationDataEncoder,
    HiddenAssociationDistribution,
    HiddenAssociationEstimator,
    HiddenAssociationSampler,
)

### Reduced Generic Distributions
from pysp.stats.latent.hmixture import (
    HierarchicalMixtureDataEncoder,
    HierarchicalMixtureDistribution,
    HierarchicalMixtureEnumerator,
    HierarchicalMixtureEstimator,
    HierarchicalMixtureSampler,
)
from pysp.stats.latent.ibp import (
    IndianBuffetProcessDataEncoder,
    IndianBuffetProcessDistribution,
    IndianBuffetProcessEstimator,
    IndianBuffetProcessSampler,
)
from pysp.stats.latent.jmixture import (
    JointMixtureDataEncoder,
    JointMixtureDistribution,
    JointMixtureEnumerator,
    JointMixtureEstimator,
    JointMixtureSampler,
)

### Generic Distributions
from pysp.stats.latent.mixture import (
    MixtureDataEncoder,
    MixtureDistribution,
    MixtureEnumerator,
    MixtureEstimator,
    MixtureSampler,
    mixture_prior,
)
from pysp.stats.latent.ppca import (
    ProbabilisticPCADataEncoder,
    ProbabilisticPCADistribution,
    ProbabilisticPCAEstimator,
    ProbabilisticPCASampler,
)
from pysp.stats.latent.segmental_hmm import (
    SegmentalHiddenMarkovDataEncoder,
    SegmentalHiddenMarkovDistribution,
    SegmentalHiddenMarkovEstimator,
    SegmentalHiddenMarkovModelDataEncoder,
    SegmentalHiddenMarkovModelDistribution,
    SegmentalHiddenMarkovModelEstimator,
    SegmentalHiddenMarkovModelSampler,
    SegmentalHiddenMarkovSampler,
)
from pysp.stats.latent.ss_mixture import (
    SemiSupervisedMixtureDataEncoder,
    SemiSupervisedMixtureDistribution,
    SemiSupervisedMixtureEstimator,
    SemiSupervisedMixtureSampler,
)

### Discrete base distributions
from pysp.stats.leaf.bernoulli import (
    BernoulliDataEncoder,
    BernoulliDistribution,
    BernoulliEnumerator,
    BernoulliEstimator,
    BernoulliSampler,
)

### Continuous base distributions
from pysp.stats.leaf.beta import BetaDataEncoder, BetaDistribution, BetaEstimator, BetaSampler
from pysp.stats.leaf.binomial import (
    BinomialDataEncoder,
    BinomialDistribution,
    BinomialEnumerator,
    BinomialEstimator,
    BinomialSampler,
)
from pysp.stats.leaf.birth_death import (
    BirthDeathSamplingDataEncoder,
    BirthDeathSamplingDistribution,
    BirthDeathSamplingEstimator,
    BirthDeathSamplingSampler,
)
from pysp.stats.leaf.cat_multinomial import (
    MultinomialDataEncoder,
    MultinomialDistribution,
    MultinomialEnumerator,
    MultinomialEstimator,
    MultinomialSampler,
)
from pysp.stats.leaf.categorical import (
    CategoricalDataEncoder,
    CategoricalDistribution,
    CategoricalEnumerator,
    CategoricalEstimator,
    CategoricalSampler,
)
from pysp.stats.leaf.exgaussian import (
    ExponentiallyModifiedGaussianDataEncoder,
    ExponentiallyModifiedGaussianDistribution,
    ExponentiallyModifiedGaussianEstimator,
    ExponentiallyModifiedGaussianSampler,
)
from pysp.stats.leaf.exponential import (
    ExponentialDataEncoder,
    ExponentialDistribution,
    ExponentialEstimator,
    ExponentialSampler,
)
from pysp.stats.leaf.gamma import GammaDataEncoder, GammaDistribution, GammaEstimator, GammaSampler
from pysp.stats.leaf.gaussian import GaussianDataEncoder, GaussianDistribution, GaussianEstimator, GaussianSampler
from pysp.stats.leaf.geometric import (
    GeometricDataEncoder,
    GeometricDistribution,
    GeometricEnumerator,
    GeometricEstimator,
    GeometricSampler,
)
from pysp.stats.leaf.gumbel import (
    GumbelDataEncoder,
    GumbelDistribution,
    GumbelEstimator,
    GumbelSampler,
)
from pysp.stats.leaf.half_normal import (
    HalfNormalDataEncoder,
    HalfNormalDistribution,
    HalfNormalEstimator,
    HalfNormalSampler,
)
from pysp.stats.leaf.inhomogeneous_poisson import (
    InhomogeneousPoissonProcessDataEncoder,
    InhomogeneousPoissonProcessDistribution,
    InhomogeneousPoissonProcessEstimator,
    InhomogeneousPoissonProcessSampler,
)
from pysp.stats.leaf.int_multinomial import (
    IntegerMultinomialDataEncoder,
    IntegerMultinomialDistribution,
    IntegerMultinomialEnumerator,
    IntegerMultinomialEstimator,
    IntegerMultinomialSampler,
)
from pysp.stats.leaf.int_range import (
    IntegerCategoricalDataEncoder,
    IntegerCategoricalDistribution,
    IntegerCategoricalEnumerator,
    IntegerCategoricalEstimator,
    IntegerCategoricalSampler,
)
from pysp.stats.leaf.int_spike import (
    IntegerUniformSpikeDataEncoder,
    IntegerUniformSpikeDistribution,
    IntegerUniformSpikeEnumerator,
    IntegerUniformSpikeEstimator,
    IntegerUniformSpikeSampler,
)
from pysp.stats.leaf.inverse_gamma import (
    InverseGammaDataEncoder,
    InverseGammaDistribution,
    InverseGammaEstimator,
    InverseGammaSampler,
)
from pysp.stats.leaf.inverse_gaussian import (
    InverseGaussianDataEncoder,
    InverseGaussianDistribution,
    InverseGaussianEstimator,
    InverseGaussianSampler,
)
from pysp.stats.leaf.laplace import LaplaceDataEncoder, LaplaceDistribution, LaplaceEstimator, LaplaceSampler
from pysp.stats.leaf.log_gaussian import (
    LogGaussianDataEncoder,
    LogGaussianDistribution,
    LogGaussianEstimator,
    LogGaussianSampler,
)
from pysp.stats.leaf.logistic import LogisticDataEncoder, LogisticDistribution, LogisticEstimator, LogisticSampler
from pysp.stats.leaf.logseries import (
    LogSeriesDataEncoder,
    LogSeriesDistribution,
    LogSeriesEstimator,
    LogSeriesSampler,
)
from pysp.stats.leaf.negative_binomial import (
    NegativeBinomialDataEncoder,
    NegativeBinomialDistribution,
    NegativeBinomialEnumerator,
    NegativeBinomialEstimator,
    NegativeBinomialSampler,
)
from pysp.stats.leaf.pareto import ParetoDataEncoder, ParetoDistribution, ParetoEstimator, ParetoSampler
from pysp.stats.leaf.point_mass import (
    PointMassDataEncoder,
    PointMassDistribution,
    PointMassEnumerator,
    PointMassEstimator,
    PointMassSampler,
)
from pysp.stats.leaf.poisson import (
    PoissonDataEncoder,
    PoissonDistribution,
    PoissonEnumerator,
    PoissonEstimator,
    PoissonSampler,
)
from pysp.stats.leaf.rayleigh import RayleighDataEncoder, RayleighDistribution, RayleighEstimator, RayleighSampler
from pysp.stats.leaf.skellam import (
    SkellamDataEncoder,
    SkellamDistribution,
    SkellamEstimator,
    SkellamSampler,
)
from pysp.stats.leaf.student_t import StudentTDataEncoder, StudentTDistribution, StudentTEstimator, StudentTSampler
from pysp.stats.leaf.tweedie import (
    TweedieDataEncoder,
    TweedieDistribution,
    TweedieEstimator,
    TweedieSampler,
)
from pysp.stats.leaf.uniform import UniformDataEncoder, UniformDistribution, UniformEstimator, UniformSampler
from pysp.stats.leaf.von_mises import (
    VonMisesDataEncoder,
    VonMisesDistribution,
    VonMisesEstimator,
    VonMisesSampler,
)
from pysp.stats.leaf.weibull import WeibullDataEncoder, WeibullDistribution, WeibullEstimator, WeibullSampler
from pysp.stats.multivariate.dmvn import (
    DiagonalGaussianDataEncoder,
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    DiagonalGaussianSampler,
)
from pysp.stats.multivariate.mvn import (
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
    MultivariateGaussianSampler,
)
from pysp.stats.multivariate.mvt import (
    MultivariateStudentTDataEncoder,
    MultivariateStudentTDistribution,
    MultivariateStudentTEstimator,
    MultivariateStudentTSampler,
)
from pysp.stats.multivariate.vmf import (
    VonMisesFisherDataEncoder,
    VonMisesFisherDistribution,
    VonMisesFisherEstimator,
    VonMisesFisherSampler,
)
from pysp.stats.sets.int_edit_setdist import (
    IntegerBernoulliEditDataEncoder,
    IntegerBernoulliEditDistribution,
    IntegerBernoulliEditEnumerator,
    IntegerBernoulliEditEstimator,
    IntegerBernoulliEditSampler,
)
from pysp.stats.sets.int_edit_stepsetdist import (
    IntegerStepBernoulliEditDataEncoder,
    IntegerStepBernoulliEditDistribution,
    IntegerStepBernoulliEditEnumerator,
    IntegerStepBernoulliEditEstimator,
    IntegerStepBernoulliEditSampler,
)
from pysp.stats.sets.int_setdist import (
    IntegerBernoulliSetDataEncoder,
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEnumerator,
    IntegerBernoulliSetEstimator,
    IntegerBernoulliSetSampler,
)
from pysp.stats.sets.setdist import (
    BernoulliSetDataEncoder,
    BernoulliSetDistribution,
    BernoulliSetEnumerator,
    BernoulliSetEstimator,
    BernoulliSetSampler,
)
from pysp.utils.optional_deps import RDD_TYPES, pyspark


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
# The names still resolve via `from pysp.stats import X` and `import *` (they
# remain in __all__), and the module's legacy_numpy compute capabilities are
# registered on that first access -- always before the distribution is used.
# ---------------------------------------------------------------------------
_LAZY_NAMES: dict[str, str] = {
    # hidden_markov
    "HiddenMarkovDataEncoder": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovEstimator": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovModelAccumulator": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovModelAccumulatorFactory": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovModelDataEncoder": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovModelDistribution": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovModelEnumerator": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovModelEstimator": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovModelSampler": "pysp.stats.latent.hidden_markov",
    "HiddenMarkovSampler": "pysp.stats.latent.hidden_markov",
    # quantized_hmm (imports hidden_markov at module top)
    "QuantizedHiddenMarkovEstimator": "pysp.stats.latent.quantized_hmm",
    "QuantizedHiddenMarkovModelDistribution": "pysp.stats.latent.quantized_hmm",
    "QuantizedHiddenMarkovModelEnumerator": "pysp.stats.latent.quantized_hmm",
    "QuantizedHiddenMarkovModelEstimator": "pysp.stats.latent.quantized_hmm",
    # tree_hmm
    "TreeHiddenMarkovEstimator": "pysp.stats.latent.tree_hmm",
    "TreeHiddenMarkovModelDistribution": "pysp.stats.latent.tree_hmm",
    "TreeHiddenMarkovModelEstimator": "pysp.stats.latent.tree_hmm",
    "TreeHiddenMarkovModelSampler": "pysp.stats.latent.tree_hmm",
    "TreeHiddenMarkovSampler": "pysp.stats.latent.tree_hmm",
    # int_plsi
    "IntegerPLSIDataEncoder": "pysp.stats.latent.int_plsi",
    "IntegerPLSIDistribution": "pysp.stats.latent.int_plsi",
    "IntegerPLSIEstimator": "pysp.stats.latent.int_plsi",
    "IntegerPLSISampler": "pysp.stats.latent.int_plsi",
    # int_hidden_association (imports int_plsi + numba)
    "IntegerHiddenAssociationDataEncoder": "pysp.stats.latent.int_hidden_association",
    "IntegerHiddenAssociationDistribution": "pysp.stats.latent.int_hidden_association",
    "IntegerHiddenAssociationEstimator": "pysp.stats.latent.int_hidden_association",
    "IntegerHiddenAssociationSampler": "pysp.stats.latent.int_hidden_association",
    # lda
    "LDADataEncoder": "pysp.stats.latent.lda",
    "LDADistribution": "pysp.stats.latent.lda",
    "LDAEstimator": "pysp.stats.latent.lda",
    "LDASampler": "pysp.stats.latent.lda",
}

# Distribution classes (by attribute name) whose legacy_numpy capabilities must be
# registered when their lazy module is first loaded -- previously registered eagerly
# in _register_builtin_compute_metadata.
_LAZY_MODULE_CAP_NAMES: dict[str, tuple[str, ...]] = {
    "pysp.stats.latent.hidden_markov": ("HiddenMarkovModelDistribution",),
    "pysp.stats.latent.quantized_hmm": ("QuantizedHiddenMarkovModelDistribution",),
    "pysp.stats.latent.tree_hmm": ("TreeHiddenMarkovModelDistribution",),
    "pysp.stats.latent.int_hidden_association": ("IntegerHiddenAssociationDistribution",),
    "pysp.stats.latent.int_plsi": ("IntegerPLSIDistribution",),
    "pysp.stats.latent.lda": ("LDADistribution",),
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


### imports
import pickle
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")
T_D = TypeVar("T_D", bound=SequenceEncodableProbabilityDistribution)


def load_models(x: str):
    """Reconstruct a model or collection of models from dump_models() JSON."""
    from pysp.utils.serialization import from_json

    return from_json(x)


def dump_models(x) -> str:
    """Serialize a stats model or collection of models to safe strict JSON."""
    from pysp.utils.serialization import to_json

    return to_json(x)


def initialize(
    data: Sequence[T] | pyspark.rdd.RDD, estimator: ParameterEstimator, rng: np.random.RandomState, p: float = 0.1
) -> SequenceEncodableProbabilityDistribution:
    """Randomly initialize a model corresponding to ParameterEstimator for iid observations data.

    Note: ParameterEstimator must be of data type T, matching the input data.

    This function sequentially iterates over the entire data set 'data', repeatedly calling initialize() method
    of the SequenceEncodableStatisticAccumulator object created from 'estimator'. Data points are weighted 0 or 1 with
    probability p.

    Seq_initialize() is much more efficient, and should produce the same initialized model for the same data sets.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Set of iid observations compatible with 'estimator'.
        estimator (ParameterEstimator): ParameterEstimator object for desired model to be estimated from data.
        rng (RandomState): RandomState object for setting seed.
        p (float): Proportion of data to randomly sample for initializing model.

    Returns:
        SequenceEncodableProbabilityDistribution object consistent with 'estimator'.

    """
    validate_estimator_keys(estimator)

    if isinstance(data, RDD_TYPES):
        factory = estimator.accumulator_factory()
        sc = data.context

        num_partitions = data.getNumPartitions()
        seeds = rng.randint(2**31, size=num_partitions)

        estimator_broadcast = sc.broadcast(estimator)
        seeds_broadcast = sc.broadcast(seeds)

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            rng_loc = np.random.RandomState(seeds_broadcast.value[split_index])
            rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))

            for x in itr:
                w = rng_w.binomial(n=1, p=p)  # partition-local rng; the driver's rng is identical on every split
                counts_for_split += w
                accumulator_for_split.initialize(x, w, rng_loc)

            return iter([(counts_for_split, accumulator_for_split.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobs_for_split, stats_for_split in temp.collect():
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2**31))

        for i, x in enumerate(idata):
            w = rng_w.binomial(n=1, p=p)
            nobs += w
            accumulator.initialize(x, w, rng)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())


def estimate(
    data: Sequence[T] | pyspark.rdd.RDD,
    estimator: ParameterEstimator,
    prev_estimate: SequenceEncodableProbabilityDistribution | None = None,
) -> SequenceEncodableProbabilityDistribution:
    """Perform E-step in EM algorithm by iterating over all observations in 'data'.

    Arg estimator must be consistent with prev_estimate. That is, prev_estimate must be an estimate that could be
    obtained from estimator.

    Data must type consistent with estimator and prev_estimate.

    Returns the next iteration of EM algorithm by iterating over each observation of data. See seq_estimate() for
    a more computationally efficient implementation.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Sequence of iid observations of data type consistent with
            'estimator' and/or 'prev_estimate'.
        estimator (ParameterEstimator): Model to be estimated from 'data'.
        prev_estimate (Optional[SequenceEncodableProbabilityDistribution]): Previous estimate of EM algorithm. Must
            be included for distributions that require initialization.

    Returns:
        SequenceEncodableProbabilityDistribution object.

    """
    validate_estimator_keys(estimator)

    # accumulators distinguish estimate-free updates with `estimate is None`;
    # substituting a NullDistribution here would defeat those guards
    if isinstance(prev_estimate, NullDistribution):
        prev_estimate = None

    if isinstance(data, RDD_TYPES):
        sc = data.context
        factory = estimator.accumulator_factory()
        estimator_broadcast = sc.broadcast(estimator)

        temp_estimate = pickle.dumps(prev_estimate, protocol=0)
        temp_estimate_b = sc.broadcast(temp_estimate)

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            loc_prev_estimate = pickle.loads(temp_estimate_b.value)

            for x in itr:
                counts_for_split = counts_for_split + 1.0
                accumulator_for_split.update(x, 1.0, estimate=loc_prev_estimate)

            return iter([(counts_for_split, accumulator_for_split.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobs_for_split, stats_for_split in temp.collect():
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for x in idata:
            nobs += 1.0
            accumulator.update(x, 1.0, estimate=prev_estimate)

        return estimator.estimate(nobs, accumulator.value())


def seq_encode(
    data: Sequence[T] | pyspark.rdd.RDD,
    encoder: DataSequenceEncoder | None = None,
    estimator: ParameterEstimator | None = None,
    model: SequenceEncodableProbabilityDistribution | None = None,
    num_chunks: int = 1,
    chunk_size: int | None = None,
) -> pyspark.rdd.RDD | list[tuple[int, Any]]:
    """Sequence encode a sequence of iid observations from a distribution corresponding to 'encoder'.

    Takes data of type Union[Sequence[T], pyspark.rdd.RDD], where the data type of the DataSequenceEncoder object's
    corresponding distribution is type T.

    If not RDD, returns a List[Tuple[int, T1]], with each list entry being a tuple containing the number of observations
    in the sequence (chunk_size), and an encoded sequence of the observations having type T1. The list has length
    num_chunks.

    RDD version with receive the Tuple of chunk_size and encoded data of type T1 for each corresponding node.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Sequence of iid observations of data type consistent with
            'encoder'.
        encoder (Optional[DataSequenceEncoder]): A DataSequenceEncoder object for sequence encoding iid sequences.
        estimator (Optional[ParameterEstimator]): An estimator to create DataSequenceEncoder from.
        model (Optional[SequenceEncodableProbabilityDistribution]): A distribution to create DataSequenceEncoder from.
        num_chunks (int): Number of chunks to split the data into. Useful for distributed data sets.
        chunk_size (Optional[int]): Approximate size of chunks to determine num_chunks above.

    Returns:
        Sequence encoded data for use with 'seq_' functions.

    """
    # tolerate a model or estimator passed positionally in the encoder slot
    if isinstance(encoder, SequenceEncodableProbabilityDistribution):
        model, encoder = encoder, None
    elif isinstance(encoder, ParameterEstimator):
        estimator, encoder = encoder, None

    if encoder is None:
        if model is not None:
            encoder = model.dist_to_encoder()
        elif estimator is not None:
            encoder = estimator.accumulator_factory().make().acc_to_encoder()
        else:
            raise Exception("At least one arg: encoder, estimator, or dist must be passed.")

    if isinstance(data, RDD_TYPES):
        sc = data.context
        temp_encoder = pickle.dumps(encoder, protocol=0)
        encoder_broadcast = sc.broadcast(temp_encoder)

        enc_data = (
            data.glom()
            .map(lambda x: list(x))
            .map(lambda x: (len(x), pickle.loads(encoder_broadcast.value).seq_encode(x)))
        )

        return enc_data

    else:
        sz = len(data)
        if chunk_size is not None:
            num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
        else:
            num_chunks_loc = num_chunks

        rv = []
        for i in range(num_chunks_loc):
            data_loc = [data[i] for i in range(i, sz, num_chunks_loc)]
            enc_data = encoder.seq_encode(data_loc)
            rv.append((len(data_loc), enc_data))

        return rv


def seq_log_density_sum(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD, estimate: SequenceEncodableProbabilityDistribution
) -> tuple[float, float]:
    """Vectorized evaluation of the sum of log_density values for a given SequenceEncodableProbabilityDistribution
        over encoded data.

    Returns a Tuple containing the sum of all observations in enc_data, and the sum of the log_density evaluated at all
    encoded data observations in enc_data. This is a fully vectorized evaluation.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimate (SequenceEncodableProbabilityDistribution): Distribution to use for log_density evaluations. Must
            be consistent with enc_data.

    Returns:
        Tuple of sum of total obs, and sum of log_density of estimate at all encoded data observations.

    """
    if hasattr(enc_data, "pysp_seq_log_density_sum"):
        # parallel-backend handle (pysp.utils.parallel.multiprocessing / parallel_mpi)
        return enc_data.pysp_seq_log_density_sum(estimate)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        estimate_broadcast = sc.broadcast(pickle.dumps(estimate, protocol=0))

        def acc(itr):

            rv = 0.0
            cnt = 0.0
            estimate_loc = pickle.loads(estimate_broadcast.value)
            for sz, x in itr:
                rv += estimate_loc.seq_log_density(x).sum()
                cnt += sz

            return [(cnt, rv)]

        return enc_data.mapPartitions(acc).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))

    else:
        return sum([u[0] for u in enc_data]), sum([estimate.seq_log_density(u[1]).sum() for u in enc_data])


def seq_log_density(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD,
    estimate: Sequence[SequenceEncodableProbabilityDistribution] | SequenceEncodableProbabilityDistribution,
) -> list[np.ndarray]:
    """Vectorized evaluation of 'estimate' log-density for each observation in enc_data.

    If 'estimate' is input as a List of numpy arrays. Each list entry corresponds to the seq_log_density calls of all
    the encoded data for each List entry of estimate.

    If 'estimate' is a single SequenceEncodableProbabilityDistribution instance. The log_density of every observation
    in the 'enc_data' data set is returned as a list.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimate (SequenceEncodableProbabilityDistribution): Distribution to use for log_density evaluations. Must
            be consistent with enc_data.

    Returns:
        List[np.ndarray[float]] or List[float] depending on input.

    """
    is_list = issubclass(type(estimate), Sequence)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        temp_estimate = pickle.dumps(estimate, protocol=0)
        estimate_broadcast = sc.broadcast(temp_estimate)

        def acc(itr):
            loc_estimate = pickle.loads(estimate_broadcast.value)
            if is_list:
                return [np.asarray([ee.seq_log_density(x) for ee in loc_estimate]) for sz, x in itr]
            else:
                return [loc_estimate.seq_log_density(x) for sz, x in itr]

        return enc_data.mapPartitions(acc).collect()

    else:
        if is_list:
            return [np.asarray([ee.seq_log_density(u[1]) for ee in estimate]) for u in enc_data]
        else:
            return [estimate.seq_log_density(u[1]) for u in enc_data]


def log_density(
    data: Sequence[T] | pyspark.rdd.RDD,
    model: SequenceEncodableProbabilityDistribution,
) -> np.ndarray:
    """Per-observation log-density of 'model' over raw (unencoded) 'data'.

    Convenience wrapper that encodes 'data' with the model's own encoder, evaluates the vectorized
    seq_log_density, and returns a single flat numpy array aligned to the input order -- the common need that
    otherwise requires the seq_encode / seq_log_density / np.concatenate boilerplate. For a distributed RDD the
    densities are collected to the driver in partition order.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Raw iid observations of data type consistent with 'model'.
        model (SequenceEncodableProbabilityDistribution): Distribution to score the observations under.

    Returns:
        np.ndarray of per-observation log-densities.

    """
    # num_chunks=1 keeps the result aligned to the input order (multi-chunk encoding interleaves observations)
    enc_data = seq_encode(data, model=model, num_chunks=1)
    parts = seq_log_density(enc_data, model)
    return np.concatenate([np.atleast_1d(np.asarray(p, dtype=float)) for p in parts])


def density(
    data: Sequence[T] | pyspark.rdd.RDD,
    model: SequenceEncodableProbabilityDistribution,
) -> np.ndarray:
    """Per-observation density of 'model' over raw (unencoded) 'data'.

    Exponentiated companion to log_density(); returns a flat numpy array of densities aligned to the input order.

    Args:
        data (Union[Sequence[T], pyspark.rdd.RDD]): Raw iid observations of data type consistent with 'model'.
        model (SequenceEncodableProbabilityDistribution): Distribution to score the observations under.

    Returns:
        np.ndarray of per-observation densities.

    """
    return np.exp(log_density(data, model))


def seq_estimate(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD, estimator: ParameterEstimator, prev_estimate: T_D
) -> T_D:
    """Perform vectorized E-step in EM algorithm for encoded sequence of observations in 'enc_data'.

    Arg estimator must be consistent with prev_estimate. That is, prev_estimate must be an estimate that could be
    obtained from estimator.

    Arg enc_data must type consistent with estimator and prev_estimate (result of seq_encode() call).

    Returns the next iteration of EM algorithm with vectorized calls to "seq_update()" of the corresponding
    SequenceEncodableStatsiticAccumulator objects.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimator (ParameterEstimator): Model to be estimated from 'enc_data'.
        prev_estimate (SequenceEncodableProbabilityDistribution): Previous estimate of EM algorithm.

    Returns:
        SequenceEncodableProbabilityDistribution object.

    """
    validate_estimator_keys(estimator)

    if hasattr(enc_data, "pysp_seq_estimate"):
        # parallel-backend handle (pysp.utils.parallel.multiprocessing / parallel_mpi)
        return enc_data.pysp_seq_estimate(estimator, prev_estimate)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context

        estimator_broadcast = sc.broadcast(estimator)
        estimate_broadcast = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            local_estimate = pickle.loads(estimate_broadcast.value)

            for sz, x in itr:
                counts_for_split = counts_for_split + sz
                accumulator_for_split.seq_update(x, np.ones(sz), local_estimate)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)

            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = 0.0
        accumulator = estimator.accumulator_factory().make()

        for stuff in temp.collect():
            nobs_for_split, stats_for_split = pickle.loads(stuff)
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        estimate_broadcast.destroy()
        estimator_broadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return estimator.estimate(nobs, accumulator.value())

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, np.ones(sz), prev_estimate)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(None, accumulator.value())


def seq_initialize(
    enc_data: list[tuple[int, T]] | pyspark.rdd.RDD,
    estimator: ParameterEstimator,
    rng: np.random.RandomState,
    p: float = 0.1,
) -> SequenceEncodableProbabilityDistribution:
    """Vectorized initialization of a model corresponding to ParameterEstimator for encoded sequences of iid data
        observations.

    Arg enc_data must type consistent with estimator (result of seq_encode() call).
    Arg estimator must be of data type consistent with encoded sequence data type in 'enc_data'.

    Vectorized initialization of SequenceEncodableProbabilityDistribution corresponding to 'estimator' from enc_data.
    Observations in the encoded sequence enc_data are kept with probability p.

    This functions relies on calls to SequenceEncodableStatisticAccumulator.seq_initialize(), which is a vectorized
    initialization of the SequenceEncodableStatisticAccumulator object.

    This method should produce the same initialized model as a call to initialize() if the data sets are the same.

    Args:
        enc_data (Union[List[Tuple[int, T]], 'pyspark.rdd.RDD']): Sequence encoded data of format matching output of
            seq_encode() function.
        estimator (ParameterEstimator): Model to be estimated from 'enc_data'.
        rng (RandomState): RandomState object for setting seed.
        p (float): Proportion of data to randomly sample for initializing model.

    Returns:
        SequenceEncodableProbabilityDistribution object consistent with 'estimator'.

    """
    validate_estimator_keys(estimator)

    if hasattr(enc_data, "pysp_seq_initialize"):
        # parallel-backend handle (pysp.utils.parallel.multiprocessing / parallel_mpi)
        return enc_data.pysp_seq_initialize(estimator, rng, p)

    if isinstance(enc_data, RDD_TYPES):
        sc = enc_data.context
        num_partitions = enc_data.getNumPartitions()
        seeds = rng.randint(2**31, size=num_partitions)

        estimator_broadcast = sc.broadcast(estimator)
        seeds_broadcast = sc.broadcast(pickle.dumps(seeds, protocol=0))

        def acc(split_index, itr):
            accumulator_for_split = estimator_broadcast.value.accumulator_factory().make()
            counts_for_split = 0.0
            rng_loc = np.random.RandomState(seeds_broadcast.value[split_index])
            rng_loc_w = np.random.RandomState(seed=rng_loc.randint(2**31))

            for sz, x in itr:
                w = np.zeros(sz, dtype=float)
                w_1 = rng_loc_w.rand(sz) <= p
                w[w_1] = 1.0

                counts_for_split += np.sum(w)
                accumulator_for_split.seq_initialize(x, w, rng_loc)

            rv = pickle.dumps((counts_for_split, accumulator_for_split.value()), protocol=0)
            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimator_broadcast.value.accumulator_factory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))

            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = 0.0
        accumulator = estimator.accumulator_factory().make()

        for stuff in temp.collect():
            nobs_for_split, stats_for_split = pickle.loads(stuff)
            nobs = nobs + nobs_for_split
            accumulator.combine(stats_for_split)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        seeds_broadcast.destroy()
        estimator_broadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return estimator.estimate(nobs, accumulator.value())

    else:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        rng_w = np.random.RandomState(seed=rng.randint(2**31 - 1))

        for sz, enc_x in enc_data:
            w = rng_w.binomial(n=1, p=p, size=sz).astype(dtype=np.float64)
            accumulator.seq_initialize(enc_x, w, rng)
            nobs += float(w.sum())  # count the kept (weight-1) observations, matching the RDD/non-seq paths

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())
