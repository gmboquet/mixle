Model Families
==============

``mixle.models`` is the applied-model namespace above the core distribution
families. Use it when a specialized model family is genuinely part of the
problem: a neural likelihood leaf, a Gaussian-process surrogate, a
random-forest conditional, a graph model, an induced grammar, a
knowledge-graph embedding, a POMDP, a truncated DPM helper, or a training-loop
utility.

For ordinary distribution modeling, ``mixle.stats`` and ``mixle.inference``
remain the primary surface. ``mixle.models`` is the applied-model layer for
specialized families that need additional training, optional dependencies, or
domain-specific conventions.

The design goal is still compositional. Where practical, helpers expose the
same estimator, distribution, sampler, scoring, or fit-result conventions used
by the rest of Mixle. That makes it possible for a Gaussian process, a random
forest conditional, a Transformer leaf, or a learned grammar to participate in
a larger heterogeneous model instead of living in a separate terminal pipeline.

Because this namespace is deliberately broader than ``mixle.stats``, promotion
requires explicit evidence. Record optional dependencies, device placement,
random seeds, training settings, and held-out behavior for every applied model
that will be served or reused by another package.

Maturity Summary
----------------

.. list-table::
   :header-rows: 1

   * - Area
     - Public surface
     - Maturity
     - Use when
   * - Neural and language leaves
     - ``LM``, ``TransformerLMEstimator``, ``StreamingTransformer``,
       ``DPOModel``, ``CategoricalEmbedding``, ``NeuralDensity``,
       ``NeuralConditionalDensity``, ``VAE``, ``Flow``, ``MAF``,
       ``DiscreteAR``
     - Incubating
     - A neural likelihood is one channel inside a larger model, or you are
       experimenting with compact local language-model helpers.
   * - Gaussian processes
     - ``GaussianProcessRegressor``, sparse GP helpers
     - Usable, evolving
     - You need a smooth surrogate or uncertainty-aware response surface.
   * - Random-forest conditionals
     - ``RandomForestEstimator``, ``RandomForestConditional``
     - Usable, evolving
     - You want a nonlinear conditional leaf ``p(y | x)`` that can be scored
       as part of a broader model.
   * - Random graphs
     - ``ErdosRenyiGraphModel``, ``StochasticBlockGraphModel``
     - Usable for small graph workflows
     - The observation is a graph or block structure is the quantity of
       interest.
   * - DPMs, grammars, knowledge graphs, POMDPs
     - ``fit_truncated_dpm``, ``fit_induced_pcfg``,
       ``TransEKnowledgeGraphModel``, ``PartiallyObservableMarkovDecisionProcessModel``
     - Specialized, validation required
     - You need a specialized family and are prepared to validate the result
       against held-out data or task loss.
   * - Dependence discovery and training search
     - ``learn_pc_skeleton``, ``orient_v_structures``, ``TrainingSpace``,
       ``tune_training``, ``ewc``
     - Specialized, validation required
     - You want proposals, diagnostics, or experiment support rather than
       a final model contract.

Choosing a Family
-----------------

.. list-table::
   :header-rows: 1

   * - Data pattern
     - Public surface
     - Typical role
   * - Text, token streams, or sequence context
     - ``LM``, ``TransformerLMEstimator``, ``StreamingTransformer``
     - Incubating neural likelihood leaf inside a hybrid model.
   * - Preference triples
     - ``DPOModel``
     - Preference-optimized leaf over chosen/rejected responses.
   * - Smooth regression with uncertainty
     - ``GaussianProcessRegressor``
     - Smooth surrogate or uncertainty-aware response surface.
   * - Tabular conditional prediction
     - ``RandomForestEstimator``, ``RandomForestConditional``
     - Nonlinear conditional leaf ``p(y | x)``.
   * - Flexible neural density
     - ``NeuralDensity``, ``NeuralConditionalDensity``, ``VAE``, ``Flow``,
       ``MAF``, ``DiscreteAR``, ``build_mdn``, ``build_conditional_flow``
     - Torch-backed density leaf for exact, bounded, mixture, or conditional
       neural likelihoods.
   * - Unknown clustering structure
     - ``fit_truncated_dpm``, ``TruncatedDirichletProcessMixtureModel``
     - Variational truncated Dirichlet-process mixture helper.
   * - Conditional independence structure
     - ``learn_pc_skeleton``, ``orient_v_structures``
     - Propose a dependency skeleton before modeling records.
   * - Symbolic or structured sequences
     - ``fit_induced_pcfg``, ``viterbi_parse``
     - Learn an induced heterogeneous PCFG.
   * - Knowledge graph triples
     - ``TransEKnowledgeGraphModel``
     - Entity/relation embedding model.
   * - Graph-valued observations
     - ``ErdosRenyiGraphModel``, ``StochasticBlockGraphModel``
     - Random graph likelihoods and block structure.
   * - Controlled latent dynamics
     - ``PartiallyObservableMarkovDecisionProcessModel``, ``baum_welch_pomdp``
     - Hidden state filtering with action-conditioned transitions.
   * - Hyperparameter and training policy
     - ``TrainingSpace``, ``tune_training``, ``ewc``
     - Experiment support for neural components.

Neural and Language Leaves
--------------------------

For Transformer and LLM-centered modeling, start with :doc:`neural-llm`.
Treat these objects as optional adapters around Torch-backed models. They are
useful when a neural likelihood has to compose with classical fields, but they
carry more dependency, reproducibility, and training-state risk than the core
stats families.

That guide covers:

* ``LM`` for a compact causal language model with direct ``fit``, ``nll``, and
  ``generate`` methods.
* ``TransformerLMEstimator`` for fitting ``(context, next_token)`` observations
  as a Mixle estimator-compatible leaf.
* ``StreamingTransformer`` when you already have a Torch module and want it
  to participate in Mixle's accumulation and scoring contract.
* ``DPOModel`` for direct preference optimization.
* ``CategoricalEmbedding`` for tying token embeddings across leaves.
* ``NeuralDensity`` and ``NeuralConditionalDensity`` for Torch modules that
  expose explicit log-density methods.
* ``VAE``, ``Flow``, ``MAF``, and ``DiscreteAR`` for constructible neural
  density families that can be dropped into estimator trees without manually
  building and wrapping a Torch module.
* ``build_mdn`` for multimodal conditional density and
  ``build_conditional_flow`` for exact conditional normalizing flows.

Neural Builder Inventory
------------------------

.. list-table::
   :header-rows: 1

   * - Import
     - Role
   * - ``make_mlp``
     - Build a compact MLP body for neural helpers and experiments.
   * - ``CategoricalClassificationNeuralNetwork``
     - Categorical classifier helper.
   * - ``GaussianRegressionNeuralNetwork``
     - Gaussian regression helper.
   * - ``PoissonRegressionNeuralNetwork``
     - Count regression helper.
   * - ``NeuralGaussian`` / ``NeuralCategorical``
     - Torch-backed likelihood leaves for regression and classification.
   * - ``VAE`` / ``Flow`` / ``MAF`` / ``DiscreteAR``
     - Constructible neural-density distribution families.
   * - ``build_causal_lm``
     - Low-level causal language-model module builder used by ``LM``.
   * - ``stream_fit``
     - Streaming fit helper for Transformer leaves.

The model-family view is that these objects are likelihood components. They can
be used alone, but their strategic value is larger: they can be children of
mixtures, HMM emissions, record fields, density gates, or task-distillation
cascades.

Serialization support has been broadened for neural leaves, direct LMs,
streaming Transformer leaves, DPO leaves, and neural-density leaves. Prefer the
documented ``save``/``load`` or ``to_dict``/``to_json`` routes for artifacts,
and still keep a held-out behavioral check around restored neural models.

Artifact Standard
-----------------

Applied model artifacts should be treated as reproducible assets, not only as
fitted Python objects. A release-quality artifact includes:

* the public class and constructor settings;
* the training data fingerprint or dataset version;
* optional dependency versions and device choice;
* a validation score on held-out examples;
* calibration or uncertainty evidence when the model exposes probabilities;
* a reload check that rescored or predicted on at least one representative
  example;
* a note about unsupported capabilities, such as unconditional sampling for
  purely conditional leaves.

This standard matters most for optional or rapidly evolving helpers. A model
family may require additional validation for a particular domain, but that
status must be visible in the artifact and in the promotion record.

Gaussian Processes
------------------

``GaussianProcessRegressor`` is an exact GP regression model with stationary
kernels and Gaussian observation noise. It supports RBF, Matern-3/2, Matern-5/2,
and related stationary kernels through the ``kernel=`` argument.

Use a GP when you need a smooth response surface, uncertainty over predictions,
or sample-efficient modeling of an expensive scientific or operational signal.
For large datasets or production use, validate runtime, numerical conditioning,
and calibration on the target problem.

.. code-block:: python

   from mixle.models import GaussianProcessRegressor

   gp = GaussianProcessRegressor(kernel="matern52", lengthscale=1.0, noise=0.05)
   gp.fit(x_train, y_train, steps=200)
   mean, cov = gp.predict(x_train, y_train, x_query, return_cov=True)

GPs are a natural partner for design-of-experiments workflows in :doc:`doe`,
Bayesian optimization in :doc:`evolution`, and calibrated prediction in
:doc:`uncertainty`.

Random Forest Conditionals
--------------------------

``RandomForestEstimator`` fits a native NumPy random forest as a conditional
distribution. Observations are ``(x, y)`` pairs, and the fitted
``RandomForestConditional`` scores ``log p(y | x)``.

This is a pragmatic conditional leaf, not a substitute for a full feature
engineering and model-governance stack. Use held-out proper scores and
calibration checks before embedding it in a larger decision system.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.models import RandomForestEstimator

   rows = [
       ([0.2, 1.0, 3.5], "approve"),
       ([1.7, 0.4, 2.2], "review"),
   ]

   model = optimize(rows, RandomForestEstimator(task="classification"), max_its=1)
   score = model.log_density(([0.4, 0.9, 3.1], "approve"))

Because the result is a probability distribution over targets conditional on
features, it can be embedded into a broader model where other fields are
generative, latent, neural, or calibrated.

Dirichlet-Process Mixtures
--------------------------

``fit_truncated_dpm`` fits a truncated Dirichlet-process mixture by variational
updates. Component M-steps are delegated to ordinary Mixle estimators, so the
component family can be Gaussian, categorical, composite, or another compatible
distribution family.

.. code-block:: python

   from mixle.models import fit_truncated_dpm
   from mixle.stats import GaussianDistribution, GaussianEstimator

   initial = [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)]
   result = fit_truncated_dpm(
       data,
       initial_components=initial,
       component_estimator=GaussianEstimator(),
       alpha=1.0,
   )

   model = result.model

Use this when the number of clusters is uncertain but you still want a finite,
inspectable fitted object. Treat truncation level, initialization, and
held-out likelihood as part of the model specification.

Low-level DPM helpers include ``stick_breaking_weights``,
``mean_stick_weights``, ``expected_log_stick_weights``, and
``sample_crp_assignments``. ``TruncatedDirichletProcessMixtureFitResult``
records the fitted model and variational diagnostics.

Dependence Discovery
--------------------

The dependence helpers expose conditional-independence tests and PC-style
structure discovery:

* ``ConditionalIndependenceResult`` records the result of an independence test.
* ``gaussian_partial_correlation`` estimates partial correlation.
* ``gaussian_conditional_independence`` tests conditional independence for
  continuous Gaussian-like data.
* ``discrete_conditional_mutual_information`` measures discrete conditional
  dependence.
* ``learn_pc_skeleton`` returns a ``CausalSkeleton``.
* ``orient_v_structures`` turns a skeleton into a ``PartiallyDirectedGraph``.

These functions are not a substitute for domain assumptions. They are a way to
turn a wide heterogeneous table into a candidate dependency structure that can
then guide a record model, graphical model, or PPL specification.

.. code-block:: python

   from mixle.models import learn_pc_skeleton, orient_v_structures

   skeleton = learn_pc_skeleton(table, alpha=0.01, method="gaussian")
   graph = orient_v_structures(skeleton)

For production use, treat discovered structure as a proposal. Hold out data,
compare alternatives with proper scores, and keep a record of rejected edges.

Grammars and Structured Sequences
---------------------------------

``fit_induced_pcfg`` learns a heterogeneous probabilistic context-free grammar
from sequences. Terminal emissions are delegated to ordinary Mixle estimators,
which lets a grammar cover mixed token types rather than assuming a single
flat vocabulary.

After fitting:

* ``pcfg_log_likelihood`` scores sequences under the grammar.
* ``viterbi_parse`` returns the most likely parse tree.
* ``grammar_rule_table`` exposes learned rule probabilities.
* ``GrammarLearningResult`` and ``PCFGParseNode`` carry fitted grammar and
  parse-tree metadata.

This is useful when a sequence has latent compositional structure: commands,
event traces, symbolic scientific strings, semi-structured logs, or mixed
categorical/numeric segments.

Use induced grammars as candidate structure. Inspect learned rules and compare
against simpler sequence models before treating a parse as meaningful.

Knowledge Graphs
----------------

``TransEKnowledgeGraphModel`` models triples by embedding entities and
relations into a shared space. ``KnowledgeGraphFitResult`` records fitted model
metadata and training diagnostics.

Use it when the core object is relational rather than tabular: ``(head,
relation, tail)`` triples, entity completion, relation plausibility, and
knowledge-graph-derived features for larger models.

This helper is best understood as an embedding baseline for relational data,
not as a complete knowledge-graph platform.

Random Graphs
-------------

The random graph helpers cover both homogeneous and block-structured graphs:

* ``ErdosRenyiGraphModel`` and ``fit_erdos_renyi_mle`` for one global edge
  probability.
* ``StochasticBlockGraphModel`` and ``fit_stochastic_block_mle`` for block
  membership and block-pair edge probabilities.
* ``hard_em_stochastic_block_model`` when block assignments are unknown.
* ``HardEMResult`` for hard-EM diagnostics and assignments.

These are useful for graph-valued observations, network monitoring, and
extracting block assignments as latent features for another model.

POMDPs
------

``PartiallyObservableMarkovDecisionProcessModel`` is the action-conditioned
counterpart to an HMM. It tracks hidden state beliefs when transitions depend
on actions and observations are noisy. ``baum_welch_pomdp`` fits the model from
sequences with expectation-maximization.

``PartiallyObservableMarkovDecisionProcessFitResult`` records fitted
parameters and training diagnostics. ``PartiallyObservableMarkovDecisionProcessFilterResult``
records filtered beliefs for a sequence.

Use POMDPs when sequences include interventions, decisions, or controls:
support tickets with actions, robot trajectories, treatment histories, or
interactive agents.

The current surface is appropriate for scoped modeling studies where the
sequence, action, and observation spaces are well controlled. Larger decision
systems still need explicit simulator, policy, and evaluation infrastructure
around the fitted POMDP.

Training Search and Continual Learning
--------------------------------------

The neural training utilities are deliberately small:

* ``TrainingSpace`` describes tunable training parameters.
* ``tune_training`` evaluates a training function over that space.
* ``lm_train_fn`` adapts LM training into the search surface.
* ``extrapolate_learning_curve`` estimates future loss from observed steps.
* ``snapshot``, ``fisher_diagonal``, and ``ewc`` support elastic weight
  consolidation for continual learning.
* ``TrainingSearchResult`` records search outcomes.

Compatibility note: ``NeuralLeaf``, ``SoftmaxNeuralLeaf``,
``StreamingTransformerLeaf``, and ``DPOLeaf`` remain importable aliases for the
preferred names ``NeuralGaussian``, ``NeuralCategorical``,
``StreamingTransformer``, and ``DPOModel``.

For broader model-level search and anti-regression gates, use :doc:`evolution`.
For task-level local model selection and LLM teachers, use
:doc:`task-distillation`.

Treat these as focused helpers for model development and regression checks.
They are intentionally not a complete training platform.

Compositional Practice
----------------------

The safest way to use ``mixle.models`` is to keep each family's role explicit:

* Use neural leaves for high-dimensional context, and document the modeling
  assumptions that are not visible from the fitted weights.
* Use GP and forest conditionals when a target is conditional on observed
  features.
* Use dependence discovery to propose structure, then verify the structure
  with held-out likelihood or decision loss.
* Use random graphs, grammars, and POMDPs when the observation itself carries
  structure that a flat table would destroy.
* Promote a candidate model only after it improves a proper score, calibration,
  or decision objective.

That is the forward direction for Mixle model families: specialist model
classes that still compose under one inference, scoring, and uncertainty
interface.
