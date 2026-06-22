"""Load SequenceEncodableProbabilityDistribution, DistributionSampler, ParameterEstimator,
and DataSequenceEncoder objects for the distributions in pyps.stats. This module also loads functions used to
estimate Distributions from data sets.
"""

from __future__ import annotations

__all__ = [
    # Bayesian (conjugate/variational) families folded in from the former pysp.bstats
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
    "LedoitWolfEstimator",
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
    "StochasticBlockGraphDistribution",
    "StochasticBlockGraphSampler",
    "StochasticBlockGraphAccumulator",
    "StochasticBlockGraphAccumulatorFactory",
    "StochasticBlockGraphEstimator",
]

### Abstract Classes
### Generic Distributions
from pysp.sampling.latent_posterior import (
    CategoricalLatentPosterior,
    LatentPosterior,
    MarkovChainLatentPosterior,
    MeanFieldLDAPosterior,
)
from pysp.sampling.sampling_api import sample

### Discrete base distributions
from pysp.stats.base.bernoulli import (
    BernoulliDataEncoder,
    BernoulliDistribution,
    BernoulliEnumerator,
    BernoulliEstimator,
    BernoulliSampler,
)

### Continuous base distributions
from pysp.stats.base.beta import BetaDataEncoder, BetaDistribution, BetaEstimator, BetaSampler
from pysp.stats.base.beta_binomial import (
    BetaBinomialDataEncoder,
    BetaBinomialDistribution,
    BetaBinomialEstimator,
    BetaBinomialSampler,
)
from pysp.stats.base.binomial import (
    BinomialDataEncoder,
    BinomialDistribution,
    BinomialEnumerator,
    BinomialEstimator,
    BinomialSampler,
)
from pysp.stats.base.birth_death import (
    BirthDeathSamplingDataEncoder,
    BirthDeathSamplingDistribution,
    BirthDeathSamplingEstimator,
    BirthDeathSamplingSampler,
)
from pysp.stats.base.categorical import (
    CategoricalDataEncoder,
    CategoricalDistribution,
    CategoricalEnumerator,
    CategoricalEstimator,
    CategoricalSampler,
)
from pysp.stats.base.categorical_multinomial import (
    MultinomialDataEncoder,
    MultinomialDistribution,
    MultinomialEnumerator,
    MultinomialEstimator,
    MultinomialSampler,
)
from pysp.stats.base.chinese_restaurant_process import (
    ChineseRestaurantProcessDataEncoder,
    ChineseRestaurantProcessDistribution,
    ChineseRestaurantProcessEstimator,
    ChineseRestaurantProcessSampler,
)
from pysp.stats.base.dirichlet_multinomial import (
    DirichletMultinomialDataEncoder,
    DirichletMultinomialDistribution,
    DirichletMultinomialEstimator,
    DirichletMultinomialSampler,
)
from pysp.stats.base.exgaussian import (
    ExponentiallyModifiedGaussianDataEncoder,
    ExponentiallyModifiedGaussianDistribution,
    ExponentiallyModifiedGaussianEstimator,
    ExponentiallyModifiedGaussianSampler,
)
from pysp.stats.base.exponential import (
    ExponentialDataEncoder,
    ExponentialDistribution,
    ExponentialEstimator,
    ExponentialSampler,
)
from pysp.stats.base.gamma import GammaDataEncoder, GammaDistribution, GammaEstimator, GammaSampler
from pysp.stats.base.gaussian import GaussianDataEncoder, GaussianDistribution, GaussianEstimator, GaussianSampler
from pysp.stats.base.generalized_extreme_value import (
    GeneralizedExtremeValueDataEncoder,
    GeneralizedExtremeValueDistribution,
    GeneralizedExtremeValueEstimator,
    GeneralizedExtremeValueSampler,
)
from pysp.stats.base.generalized_pareto import (
    GeneralizedParetoDataEncoder,
    GeneralizedParetoDistribution,
    GeneralizedParetoEstimator,
    GeneralizedParetoSampler,
)
from pysp.stats.base.geometric import (
    GeometricDataEncoder,
    GeometricDistribution,
    GeometricEnumerator,
    GeometricEstimator,
    GeometricSampler,
)
from pysp.stats.base.gumbel import (
    GumbelDataEncoder,
    GumbelDistribution,
    GumbelEstimator,
    GumbelSampler,
)
from pysp.stats.base.half_normal import (
    HalfNormalDataEncoder,
    HalfNormalDistribution,
    HalfNormalEstimator,
    HalfNormalSampler,
)
from pysp.stats.base.hawkes_process import (
    HawkesProcessDataEncoder,
    HawkesProcessDistribution,
    HawkesProcessEstimator,
    HawkesProcessSampler,
)
from pysp.stats.base.inhomogeneous_poisson import (
    InhomogeneousPoissonProcessDataEncoder,
    InhomogeneousPoissonProcessDistribution,
    InhomogeneousPoissonProcessEstimator,
    InhomogeneousPoissonProcessSampler,
)
from pysp.stats.base.integer_categorical import (
    IntegerCategoricalDataEncoder,
    IntegerCategoricalDistribution,
    IntegerCategoricalEnumerator,
    IntegerCategoricalEstimator,
    IntegerCategoricalSampler,
)
from pysp.stats.base.integer_multinomial import (
    IntegerMultinomialDataEncoder,
    IntegerMultinomialDistribution,
    IntegerMultinomialEnumerator,
    IntegerMultinomialEstimator,
    IntegerMultinomialSampler,
)
from pysp.stats.base.integer_uniform_spike import (
    IntegerUniformSpikeDataEncoder,
    IntegerUniformSpikeDistribution,
    IntegerUniformSpikeEnumerator,
    IntegerUniformSpikeEstimator,
    IntegerUniformSpikeSampler,
)
from pysp.stats.base.inverse_gamma import (
    InverseGammaDataEncoder,
    InverseGammaDistribution,
    InverseGammaEstimator,
    InverseGammaSampler,
)
from pysp.stats.base.inverse_gaussian import (
    InverseGaussianDataEncoder,
    InverseGaussianDistribution,
    InverseGaussianEstimator,
    InverseGaussianSampler,
)
from pysp.stats.base.laplace import LaplaceDataEncoder, LaplaceDistribution, LaplaceEstimator, LaplaceSampler
from pysp.stats.base.log_gaussian import (
    LogGaussianDataEncoder,
    LogGaussianDistribution,
    LogGaussianEstimator,
    LogGaussianSampler,
)
from pysp.stats.base.logistic import LogisticDataEncoder, LogisticDistribution, LogisticEstimator, LogisticSampler
from pysp.stats.base.logseries import (
    LogSeriesDataEncoder,
    LogSeriesDistribution,
    LogSeriesEstimator,
    LogSeriesSampler,
)
from pysp.stats.base.multivariate_hawkes import (
    MultivariateHawkesProcessDataEncoder,
    MultivariateHawkesProcessDistribution,
    MultivariateHawkesProcessEstimator,
    MultivariateHawkesProcessSampler,
)
from pysp.stats.base.negative_binomial import (
    NegativeBinomialDataEncoder,
    NegativeBinomialDistribution,
    NegativeBinomialEnumerator,
    NegativeBinomialEstimator,
    NegativeBinomialSampler,
)
from pysp.stats.base.pareto import ParetoDataEncoder, ParetoDistribution, ParetoEstimator, ParetoSampler
from pysp.stats.base.point_mass import (
    PointMassDataEncoder,
    PointMassDistribution,
    PointMassEnumerator,
    PointMassEstimator,
    PointMassSampler,
)
from pysp.stats.base.poisson import (
    PoissonDataEncoder,
    PoissonDistribution,
    PoissonEnumerator,
    PoissonEstimator,
    PoissonSampler,
)
from pysp.stats.base.rayleigh import RayleighDataEncoder, RayleighDistribution, RayleighEstimator, RayleighSampler
from pysp.stats.base.renewal_process import (
    RenewalProcessDataEncoder,
    RenewalProcessDistribution,
    RenewalProcessEstimator,
    RenewalProcessSampler,
)
from pysp.stats.base.skellam import (
    SkellamDataEncoder,
    SkellamDistribution,
    SkellamEstimator,
    SkellamSampler,
)
from pysp.stats.base.skew_normal import (
    SkewNormalDataEncoder,
    SkewNormalDistribution,
    SkewNormalEstimator,
    SkewNormalSampler,
)
from pysp.stats.base.student_t import StudentTDataEncoder, StudentTDistribution, StudentTEstimator, StudentTSampler
from pysp.stats.base.tweedie import (
    TweedieDataEncoder,
    TweedieDistribution,
    TweedieEstimator,
    TweedieSampler,
)
from pysp.stats.base.uniform import UniformDataEncoder, UniformDistribution, UniformEstimator, UniformSampler
from pysp.stats.base.von_mises import (
    VonMisesDataEncoder,
    VonMisesDistribution,
    VonMisesEstimator,
    VonMisesSampler,
)
from pysp.stats.base.weibull import WeibullDataEncoder, WeibullDistribution, WeibullEstimator, WeibullSampler
from pysp.stats.base.wrapped_cauchy import (
    WrappedCauchyDataEncoder,
    WrappedCauchyDistribution,
    WrappedCauchyEstimator,
    WrappedCauchySampler,
)
from pysp.stats.bayes.conjugate import (
    ConjugatePosterior,
    MixtureConjugatePosterior,
    conjugate_posterior,
    mixture_conjugate_posterior,
)
from pysp.stats.bayes.dict_dirichlet import DictDirichletDistribution, DictDirichletSampler
from pysp.stats.bayes.dirichlet import DirichletDataEncoder, DirichletDistribution, DirichletEstimator, DirichletSampler
from pysp.stats.bayes.dirichlet_process_mixture import (
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
    DirichletProcessMixtureSampler,
)
from pysp.stats.bayes.hierarchical_dirichlet_process_mixture import (
    HierarchicalDirichletProcessMixtureDistribution,
    HierarchicalDirichletProcessMixtureEstimator,
    HierarchicalDirichletProcessMixtureSampler,
)
from pysp.stats.bayes.multivariate_normal_gamma import (
    MultivariateNormalGammaDistribution,
    MultivariateNormalGammaSampler,
)
from pysp.stats.bayes.normal_gamma import NormalGammaDistribution, NormalGammaSampler
from pysp.stats.bayes.normal_wishart import NormalWishartDistribution, NormalWishartSampler
from pysp.stats.bayes.pitman_yor import (
    PitmanYorProcessDataEncoder,
    PitmanYorProcessDistribution,
    PitmanYorProcessEstimator,
    PitmanYorProcessSampler,
)
from pysp.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution, SymmetricDirichletSampler

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
from pysp.stats.combinator.hurdle import (
    HurdleDataEncoder,
    HurdleDistribution,
    HurdleEstimator,
    HurdleSampler,
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
from pysp.stats.combinator.survival import (
    SurvivalDataEncoder,
    SurvivalDistribution,
    SurvivalEstimator,
    SurvivalSampler,
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
from pysp.stats.combinator.zero_inflated import (
    ZeroInflatedDataEncoder,
    ZeroInflatedDistribution,
    ZeroInflatedEstimator,
    ZeroInflatedSampler,
)
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

# Base contracts re-exported as accessible attributes (e.g. isinstance checks against
# pysp.stats.SequenceEncodableProbabilityDistribution / pysp.stats.ParameterEstimator), but
# intentionally NOT in __all__: the distribution catalog scanners (star-import / sampler-seed /
# serialization-registry) must stay concrete-distributions-only.
from pysp.stats.compute.pdist import (  # noqa: F401
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

# Backward-compatible ICLTree* aliases (redundant import alias marks an intentional re-export).
from pysp.stats.graph.integer_chow_liu_tree import ICLTreeDataEncoder as ICLTreeDataEncoder  # noqa: F401
from pysp.stats.graph.integer_chow_liu_tree import ICLTreeDistribution as ICLTreeDistribution  # noqa: F401
from pysp.stats.graph.integer_chow_liu_tree import ICLTreeEnumerator as ICLTreeEnumerator  # noqa: F401
from pysp.stats.graph.integer_chow_liu_tree import ICLTreeEstimator as ICLTreeEstimator  # noqa: F401
from pysp.stats.graph.integer_chow_liu_tree import ICLTreeSampler as ICLTreeSampler  # noqa: F401
from pysp.stats.graph.integer_chow_liu_tree import (
    IntegerChowLiuTreeDataEncoder,
    IntegerChowLiuTreeDistribution,
    IntegerChowLiuTreeEnumerator,
    IntegerChowLiuTreeEstimator,
    IntegerChowLiuTreeSampler,
)
from pysp.stats.graph.integer_markov_chain import (
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
from pysp.stats.graph.random_dot_product_graph import (
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
from pysp.stats.latent.hierarchical_mixture import (
    HierarchicalMixtureDataEncoder,
    HierarchicalMixtureDistribution,
    HierarchicalMixtureEnumerator,
    HierarchicalMixtureEstimator,
    HierarchicalMixtureSampler,
)
from pysp.stats.latent.indian_buffet_process import (
    IndianBuffetProcessDataEncoder,
    IndianBuffetProcessDistribution,
    IndianBuffetProcessEstimator,
    IndianBuffetProcessSampler,
)
from pysp.stats.latent.joint_mixture import (
    JointMixtureDataEncoder,
    JointMixtureDistribution,
    JointMixtureEnumerator,
    JointMixtureEstimator,
    JointMixtureSampler,
)
from pysp.stats.latent.mixture import (
    MixtureDataEncoder,
    MixtureDistribution,
    MixtureEnumerator,
    MixtureEstimator,
    MixtureSampler,
    mixture_prior,
)
from pysp.stats.latent.probabilistic_pca import (
    ProbabilisticPCADataEncoder,
    ProbabilisticPCADistribution,
    ProbabilisticPCAEstimator,
    ProbabilisticPCASampler,
)
from pysp.stats.latent.segmental_hidden_markov_model import (
    SegmentalHiddenMarkovDataEncoder,
    SegmentalHiddenMarkovDistribution,
    SegmentalHiddenMarkovEstimator,
    SegmentalHiddenMarkovModelDataEncoder,
    SegmentalHiddenMarkovModelDistribution,
    SegmentalHiddenMarkovModelEstimator,
    SegmentalHiddenMarkovModelSampler,
    SegmentalHiddenMarkovSampler,
)
from pysp.stats.latent.semi_supervised_mixture import (
    SemiSupervisedMixtureDataEncoder,
    SemiSupervisedMixtureDistribution,
    SemiSupervisedMixtureEstimator,
    SemiSupervisedMixtureSampler,
)
from pysp.stats.multivariate.covariance_shrinkage import LedoitWolfEstimator
from pysp.stats.multivariate.diagonal_gaussian import (
    DiagonalGaussianDataEncoder,
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    DiagonalGaussianSampler,
)
from pysp.stats.multivariate.gaussian_copula import (
    GaussianCopulaDataEncoder,
    GaussianCopulaDistribution,
    GaussianCopulaEstimator,
    GaussianCopulaSampler,
)
from pysp.stats.multivariate.inverse_wishart import (
    InverseWishartDataEncoder,
    InverseWishartDistribution,
    InverseWishartEstimator,
    InverseWishartSampler,
)
from pysp.stats.multivariate.matrix_normal import (
    MatrixNormalDataEncoder,
    MatrixNormalDistribution,
    MatrixNormalEstimator,
    MatrixNormalSampler,
)
from pysp.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
    MultivariateGaussianSampler,
)
from pysp.stats.multivariate.multivariate_student_t import (
    MultivariateStudentTDataEncoder,
    MultivariateStudentTDistribution,
    MultivariateStudentTEstimator,
    MultivariateStudentTSampler,
)
from pysp.stats.multivariate.von_mises_fisher import (
    VonMisesFisherDataEncoder,
    VonMisesFisherDistribution,
    VonMisesFisherEstimator,
    VonMisesFisherSampler,
)
from pysp.stats.multivariate.watson import (
    WatsonDataEncoder,
    WatsonDistribution,
    WatsonEstimator,
    WatsonSampler,
)
from pysp.stats.multivariate.wishart import (
    WishartDataEncoder,
    WishartDistribution,
    WishartEstimator,
    WishartSampler,
)
from pysp.stats.sets.bernoulli_set import (
    BernoulliSetDataEncoder,
    BernoulliSetDistribution,
    BernoulliSetEnumerator,
    BernoulliSetEstimator,
    BernoulliSetSampler,
)
from pysp.stats.sets.integer_bernoulli_edit import (
    IntegerBernoulliEditDataEncoder,
    IntegerBernoulliEditDistribution,
    IntegerBernoulliEditEnumerator,
    IntegerBernoulliEditEstimator,
    IntegerBernoulliEditSampler,
)
from pysp.stats.sets.integer_bernoulli_set import (
    IntegerBernoulliSetDataEncoder,
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEnumerator,
    IntegerBernoulliSetEstimator,
    IntegerBernoulliSetSampler,
)
from pysp.stats.sets.integer_step_bernoulli_edit import (
    IntegerStepBernoulliEditDataEncoder,
    IntegerStepBernoulliEditDistribution,
    IntegerStepBernoulliEditEnumerator,
    IntegerStepBernoulliEditEstimator,
    IntegerStepBernoulliEditSampler,
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
    "QuantizedHiddenMarkovEstimator": "pysp.stats.latent.quantized_hidden_markov_model",
    "QuantizedHiddenMarkovModelDistribution": "pysp.stats.latent.quantized_hidden_markov_model",
    "QuantizedHiddenMarkovModelEnumerator": "pysp.stats.latent.quantized_hidden_markov_model",
    "QuantizedHiddenMarkovModelEstimator": "pysp.stats.latent.quantized_hidden_markov_model",
    # tree_hmm
    "TreeHiddenMarkovEstimator": "pysp.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovModelDistribution": "pysp.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovModelEstimator": "pysp.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovModelSampler": "pysp.stats.latent.tree_hidden_markov_model",
    "TreeHiddenMarkovSampler": "pysp.stats.latent.tree_hidden_markov_model",
    # int_plsi
    "IntegerProbabilisticLatentSemanticIndexingDataEncoder": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerProbabilisticLatentSemanticIndexingDistribution": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerProbabilisticLatentSemanticIndexingEstimator": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerProbabilisticLatentSemanticIndexingSampler": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    # int_plsi backward-compatible aliases (former IntegerPLSI* names)
    "IntegerPLSIDataEncoder": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerPLSIDistribution": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerPLSIEstimator": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    "IntegerPLSISampler": "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing",
    # int_hidden_association (imports int_plsi + numba)
    "IntegerHiddenAssociationDataEncoder": "pysp.stats.latent.integer_hidden_association",
    "IntegerHiddenAssociationDistribution": "pysp.stats.latent.integer_hidden_association",
    "IntegerHiddenAssociationEstimator": "pysp.stats.latent.integer_hidden_association",
    "IntegerHiddenAssociationSampler": "pysp.stats.latent.integer_hidden_association",
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
    "pysp.stats.latent.quantized_hidden_markov_model": ("QuantizedHiddenMarkovModelDistribution",),
    "pysp.stats.latent.tree_hidden_markov_model": ("TreeHiddenMarkovModelDistribution",),
    "pysp.stats.latent.integer_hidden_association": ("IntegerHiddenAssociationDistribution",),
    "pysp.stats.latent.integer_probabilistic_latent_semantic_indexing": (
        "IntegerProbabilisticLatentSemanticIndexingDistribution",
    ),
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


def load_models(x: str):
    """Reconstruct a model or collection of models from dump_models() JSON."""
    from pysp.utils.serialization import from_json

    return from_json(x)


def dump_models(x) -> str:
    """Serialize a stats model or collection of models to safe strict JSON."""
    from pysp.utils.serialization import to_json

    return to_json(x)


# Vectorized sequence-driver API — implementations live in pysp.stats.compute.sequence so the
# inference machinery can import them without importing this package. Re-exported here unchanged.
from pysp.stats.compute.sequence import (  # noqa: E402
    density,
    log_density,
    seq_encode,
    seq_log_density,
    seq_log_density_sum,
)
