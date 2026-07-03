"""Applied models -- richer, domain-specialized families that plug into the same contract as ``mixle.stats``.

Where ``mixle.stats`` holds the *elementary* distributions (a Gaussian, a Poisson, a categorical), this package
holds the models that are **more than one elementary density**: a neural network, a Gaussian process, a random
forest, a knowledge graph, a grammar, a decision process, a causal skeleton. Each is exposed through the same
five-piece ``Distribution``/``Estimator``/``Accumulator``/``Sampler``/``Encoder`` contract (or, for the
supervised/decision/causal ones, a small task-appropriate surface), so it composes with the stats core -- a
neural leaf drops into a ``CompositeDistribution``, a GP into a mixture, and so on.

The right mental model is a small catalog of applied model *families* (``mixle/models/README.md`` maps every
module to one):

  * **neural & deep** -- neural nets, transformers/LMs, embeddings, and their training utilities;
  * **non-parametric** -- Gaussian processes and random forests as ``p(y | x)`` leaves;
  * **relational / structured** -- knowledge graphs, random graphs, grammars;
  * **latent-variable** -- Bayesian-nonparametric mixtures (Dirichlet process);
  * **decision & control** -- partially observable Markov decision processes;
  * **causal discovery** -- constraint-based structure learning.

(The imports below stay alphabetical -- the ruff import sorter enforces that -- so use the families above, not
import order, as the map.)

These surfaces vary in maturity (see the Project status table in the top-level README); treat them as specialist
adapters composable with the stable stats spine, not as the spine itself.
"""

from mixle.models.continual import ewc, fisher_diagonal, snapshot
from mixle.models.dependence import (
    CausalSkeleton,
    ConditionalIndependenceResult,
    PartiallyDirectedGraph,
    discrete_conditional_mutual_information,
    gaussian_conditional_independence,
    gaussian_partial_correlation,
    learn_pc_skeleton,
    orient_v_structures,
)
from mixle.models.dirichlet_process_mixture import (
    TruncatedDirichletProcessMixtureFitResult,
    TruncatedDirichletProcessMixtureModel,
    expected_log_stick_weights,
    fit_truncated_dpm,
    mean_stick_weights,
    sample_crp_assignments,
    stick_breaking_weights,
)
from mixle.models.dpo_leaf import DPOLeaf
from mixle.models.embedding import CategoricalEmbedding
from mixle.models.gaussian_process import GaussianProcessRegressor
from mixle.models.grammar import (
    GrammarLearningResult,
    PCFGParseNode,
    fit_induced_pcfg,
    grammar_rule_table,
    pcfg_log_likelihood,
    viterbi_parse,
)
from mixle.models.knowledge_graph import KnowledgeGraphFitResult, TransEKnowledgeGraphModel
from mixle.models.language_model import LM
from mixle.models.mixture_density import NeuralConditionalDensity, build_conditional_flow, build_mdn
from mixle.models.neural import (
    CategoricalClassificationNeuralNetwork,
    GaussianRegressionNeuralNetwork,
    PoissonRegressionNeuralNetwork,
    make_mlp,
)
from mixle.models.neural_density import NeuralDensity, build_coupling_flow, build_maf, build_vae
from mixle.models.neural_leaf import NeuralLeaf
from mixle.models.partially_observable_markov_decision_process import (
    PartiallyObservableMarkovDecisionProcessFilterResult,
    PartiallyObservableMarkovDecisionProcessFitResult,
    PartiallyObservableMarkovDecisionProcessModel,
    baum_welch_pomdp,
)
from mixle.models.random_forest import (
    RandomForestConditional,
    RandomForestEstimator,
)
from mixle.models.random_graph import (
    ErdosRenyiGraphModel,
    HardEMResult,
    StochasticBlockGraphModel,
    fit_erdos_renyi_mle,
    fit_stochastic_block_mle,
    hard_em_stochastic_block_model,
)
from mixle.models.softmax_leaf import SoftmaxNeuralLeaf
from mixle.models.streaming_transformer_leaf import (
    StreamingTransformerLeaf,
    TransformerLMEstimator,
    stream_fit,
)
from mixle.models.train_search import (
    TrainingSearchResult,
    TrainingSpace,
    extrapolate_learning_curve,
    lm_train_fn,
    tune_training,
)
from mixle.models.transformer import build_causal_lm

__all__ = [
    "LM",
    "CategoricalEmbedding",
    "CausalSkeleton",
    "CategoricalClassificationNeuralNetwork",
    "ConditionalIndependenceResult",
    "DPOLeaf",
    "ErdosRenyiGraphModel",
    "GaussianProcessRegressor",
    "GaussianRegressionNeuralNetwork",
    "NeuralConditionalDensity",
    "NeuralDensity",
    "NeuralLeaf",
    "SoftmaxNeuralLeaf",
    "StreamingTransformerLeaf",
    "TrainingSearchResult",
    "TrainingSpace",
    "TransformerLMEstimator",
    "build_causal_lm",
    "ewc",
    "extrapolate_learning_curve",
    "fisher_diagonal",
    "lm_train_fn",
    "snapshot",
    "tune_training",
    "stream_fit",
    "GrammarLearningResult",
    "HardEMResult",
    "KnowledgeGraphFitResult",
    "PartiallyObservableMarkovDecisionProcessFilterResult",
    "PartiallyObservableMarkovDecisionProcessFitResult",
    "PartiallyObservableMarkovDecisionProcessModel",
    "PartiallyDirectedGraph",
    "PCFGParseNode",
    "PoissonRegressionNeuralNetwork",
    "RandomForestConditional",
    "RandomForestEstimator",
    "StochasticBlockGraphModel",
    "TransEKnowledgeGraphModel",
    "TruncatedDirichletProcessMixtureFitResult",
    "TruncatedDirichletProcessMixtureModel",
    "baum_welch_pomdp",
    "discrete_conditional_mutual_information",
    "expected_log_stick_weights",
    "fit_erdos_renyi_mle",
    "fit_stochastic_block_mle",
    "fit_truncated_dpm",
    "fit_induced_pcfg",
    "gaussian_conditional_independence",
    "gaussian_partial_correlation",
    "grammar_rule_table",
    "hard_em_stochastic_block_model",
    "learn_pc_skeleton",
    "build_conditional_flow",
    "build_coupling_flow",
    "build_maf",
    "build_mdn",
    "build_vae",
    "make_mlp",
    "mean_stick_weights",
    "orient_v_structures",
    "pcfg_log_likelihood",
    "sample_crp_assignments",
    "stick_breaking_weights",
    "viterbi_parse",
]
