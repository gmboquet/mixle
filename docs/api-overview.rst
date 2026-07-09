API Overview
============

The generated reference under :doc:`api/modules` documents the broad public
module surface. This page is the human map for common imports and workflows.

Use this page and the narrative guides as the stable public entry points. The
generated reference includes public modules and selected implementation modules
so release reviewers can audit what the branch ships. Names with a leading
underscore, generated kernel modules, and low-level helper modules are not
public compatibility promises unless a guide page or package ``__init__``
explicitly promotes them.

Public Surface Contract
-----------------------

For public examples and release-facing applications, prefer imports from these
front doors:

* ``mixle.inference`` for fitting, creation, certification, simulation,
  synthesis, uncertainty dispatch, and production wrappers;
* ``mixle.stats`` for distributions, estimators, combinators, mixtures, HMMs,
  Bayesian families, and structured probability models;
* ``mixle.ppl`` for expression-style model declarations that lower back to the
  ordinary stats and inference surfaces;
* ``mixle.doe`` for design generators, Bayesian optimization, active learning,
  sensitivity analysis, propagation, calibration, and distillation selectors;
* ``mixle.task`` for teacher/student distillation and calibrated local task
  replacement; and
* ``mixle.reason`` / ``mixle.substrate`` for local reasoning, evidence, and
  uncertainty workflows.

Import directly from implementation modules when you need a specific class or
when a narrative page points there. Treat generated API pages as reference and
audit material; the narrative pages define the supported workflow semantics,
missing-data behavior, route evidence, and release expectations.

Fit a model
-----------

.. list-table::
   :header-rows: 1

   * - Task
     - Import
     - Notes
   * - Fit an estimator or prototype distribution
     - ``from mixle.inference import optimize``
     - Main EM/MLE entry point.
   * - Initialize or estimate directly
     - ``from mixle.inference import initialize, estimate``
     - Useful when you want explicit control over one E/M pass.
   * - Try multiple random starts
     - ``from mixle.inference import best_of``
     - Common for mixtures and HMMs.
   * - Stream updates
     - ``from mixle.inference import StreamingEstimator``
     - Online or mini-batch estimation.
   * - Create a certified artifact
     - ``from mixle.inference import create``
     - Fits a model and attaches certificate, optional calibration, UQ, and
       provenance.
   * - Simulate from a fitted model
     - ``from mixle.inference import simulate``
     - Packages a model as a baseline/scenario simulator.
   * - Build a verified synthetic dataset
     - ``from mixle.inference import synthesize``
     - Draws inputs, optionally labels them, and keeps rows that verify.
   * - Record and replay a fit
     - ``from mixle.inference import record_fit, verify_reproducible``
     - Stores and checks data/parameter fingerprints.
   * - Certify and place estimation blocks
     - ``from mixle.inference import certify, plan_placement``
     - Reports estimation guarantees and local/pool placement.
   * - Use gradient MAP/MLE
     - ``from mixle.inference.gradient_fit import fit_map, fit_mle``
     - For differentiable parameter objectives.

Build distributions and estimators
----------------------------------

.. list-table::
   :header-rows: 1

   * - Family group
     - Common imports
   * - Scalar families
     - ``GaussianDistribution``, ``PoissonDistribution``, ``CategoricalDistribution``
   * - Estimators
     - ``GaussianEstimator``, ``PoissonEstimator``, ``CategoricalEstimator``
   * - Records and tuples
     - ``CompositeDistribution``, ``CompositeEstimator``, ``RecordDistribution``
   * - Sequences
     - ``SequenceDistribution``, ``SequenceEstimator``
   * - Latent models
     - ``MixtureDistribution``, ``MixtureEstimator``, ``HiddenMarkovModelDistribution``
   * - Bayesian families
     - ``mixle.stats.bayes`` and ``mixle.inference.priors``

Most high-level distribution symbols are re-exported from ``mixle.stats``:

.. code-block:: python

   from mixle.stats import GaussianEstimator, MixtureEstimator, SequenceEstimator

Use the implementation submodules when you want a narrower import or source
location, for example ``mixle.stats.univariate.continuous.gaussian``.

For the full narrative catalog, use :doc:`stats-univariate`,
:doc:`stats-structured`, and :doc:`stats-latent-bayes`.

Use neural and language-model leaves
------------------------------------

The symbols in this section live in ``mixle.models``, an incubating applied
helper namespace. Use them when a neural likelihood really belongs inside a
larger model. For ordinary distribution work, prefer ``mixle.stats`` first.

.. list-table::
   :header-rows: 1

   * - Task
     - Import
     - Notes
   * - Fit a small causal LM directly
     - ``from mixle.models import LM``
     - ``LM.fit`` trains token sequences; ``generate`` and ``nll`` query it.
   * - Put a Transformer inside a distribution
     - ``from mixle.models import StreamingTransformer``
     - Wraps a Torch module as an estimator-compatible neural leaf.
   * - Use a ready LM estimator
     - ``from mixle.models import TransformerLMEstimator``
     - Fits ``(context, next_token)`` observations as a generative leaf.
   * - Tie token embeddings across experts
     - ``from mixle.models import CategoricalEmbedding``
     - Reuse one embedding in several LM estimators.
   * - Preference optimization
     - ``from mixle.models import DPOModel``
     - A DPO-trained preference leaf over ``(x, chosen, rejected)`` triples.
   * - Neural Gaussian/categorical leaves
     - ``from mixle.models import NeuralGaussian, NeuralCategorical``
     - Conditional Torch-backed regression and classification likelihoods.
   * - Unconditional neural density
     - ``from mixle.models import NeuralDensity, build_maf, build_coupling_flow``
     - Wrap exact-density Torch modules as Mixle leaves.
   * - Constructible neural density families
     - ``from mixle.models import VAE, Flow, MAF, DiscreteAR``
     - Use common neural-density families directly as distribution objects.
   * - Conditional neural density
     - ``from mixle.models import NeuralConditionalDensity, build_mdn, build_conditional_flow``
     - Model ``p(y | x)`` with an MDN or exact conditional flow.
   * - Energy-based density
     - ``from mixle.models import EnergyModel, build_energy_net``
     - Approximate normalized density from NCE-trained energy functions.
   * - Autoregressive categorical density
     - ``from mixle.models import build_autoregressive_categorical``
     - Exact neural density over discrete vectors.
   * - Conditional autoregressive categorical density
     - ``from mixle.models import build_conditional_autoregressive_categorical``
     - Exact neural ``p(y | x)`` over discrete target vectors.

Use other model families
------------------------

These helpers also live in ``mixle.models``. They share Mixle conventions where
practical, but they do not all have the same maturity as the core distribution
families.

.. list-table::
   :header-rows: 1

   * - Task
     - Import
     - Notes
   * - Gaussian-process regression
     - ``from mixle.models import GaussianProcessRegressor``
     - Exact GP regression with stationary kernels and predictive uncertainty.
   * - Random forest conditional leaf
     - ``from mixle.models import RandomForestEstimator``
     - Fits ``p(y | x)`` as a Mixle-compatible conditional distribution.
   * - Truncated Dirichlet-process mixture
     - ``from mixle.models import fit_truncated_dpm``
     - Variational finite truncation with ordinary Mixle component estimators.
   * - Dependence discovery
     - ``from mixle.models import learn_pc_skeleton, orient_v_structures``
     - Conditional-independence structure discovery for tabular data.
   * - Induced grammars
     - ``from mixle.models import fit_induced_pcfg, viterbi_parse``
     - Heterogeneous PCFG learning and parse extraction.
   * - Knowledge graphs
     - ``from mixle.models import TransEKnowledgeGraphModel``
     - Embedding model for entity-relation triples.
   * - Random graphs
     - ``from mixle.models import ErdosRenyiGraphModel, StochasticBlockGraphModel``
     - Graph-valued likelihoods and block structure.
   * - POMDPs
     - ``from mixle.models import PartiallyObservableMarkovDecisionProcessModel``
     - Action-conditioned hidden-state model.

Represent heterogeneous inputs
------------------------------

.. code-block:: python

   from mixle.represent import (
       ByteSegmenter,
       FeatureEmbedding,
       HeterogeneousEncoder,
       VectorQuantizer,
       WindowSegmenter,
   )
   from mixle.represent.posterior import PosteriorRetriever

Use ``mixle.represent`` when the front end of the model must handle multiple
modalities. Segmenters cut raw data into units, embeddings map units into a
shared vector space, and ``VectorQuantizer`` optionally learns a discrete
codebook in that space. ``PosteriorRetriever`` uses a fitted mixture's
posterior affinity to retrieve or rerank heterogeneous records by what the
model believes is similar.

For deterministic image and signal baselines:

.. code-block:: python

   from mixle.represent.modality import image_features, signal_features, vectorize

Design and distill tasks
------------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Import
     - Notes
   * - Recommend a generative model from data
     - ``from mixle.task import recommend_model``
     - Returns an estimator plus confidence gaps and dependency hints.
   * - Let an LLM propose a model spec
     - ``from mixle.task import design_model``
     - Builds only allowlisted specs and fit-validates before trusting them.
   * - Distill a teacher into a local model
     - ``from mixle.task import distill``
     - Teacher can be a slow model, human-facing function, or LLM labeler.
   * - Distill a generative text student
     - ``from mixle.task import distill_text_generative``
     - Fits per-class token models so the student exposes label posteriors and
       text evidence.
   * - Replace a classification or routing function
     - ``from mixle.task import solve``
     - Trains a calibrated local student and escalates uncertain cases.
   * - Replace a numeric scoring or pricing function
     - ``from mixle.task import solve_regression``
     - Uses split-conformal intervals and answers locally only when the
       calibrated width meets ``tol``.
   * - Replace a multi-label tagger
     - ``from mixle.task import solve_multilabel``
     - Decides each label as present or absent and escalates if any label is
       ambiguous.
   * - Replace a dict-valued enrichment function
     - ``from mixle.task import solve_structured``
     - Splits a stable output schema into calibrated categorical and numeric
       field solvers, then escalates if any field is uncertain.
   * - Actively choose LLM labels
     - ``from mixle.task import active_distill``
     - Queries the teacher on the most informative pool items.
   * - Calibrate and cascade
     - ``from mixle.task import CalibratedTaskModel, Cascade``
     - Local answer when reliable; escalate to the teacher otherwise.
   * - Route and score deployed students
     - ``from mixle.task import Router, scorecard``
     - Measure escalation, local agreement, cost, and route behavior.
   * - Distill extraction
     - ``from mixle.task import llm_extractor, distill_extractor``
     - Turns LLM field extraction into a local sequence tagger.
   * - Distill tool calling
     - ``from mixle.task import ToolSpec, distill_tool_caller``
     - Local tool selector plus per-tool argument extractors.
   * - Distill planning
     - ``from mixle.task import distill_planner, sft_planner``
     - Stepwise next-tool planner or trace-SFT generative planner.
   * - Train from agent history
     - ``from mixle.task import harvest_agent_traces``
     - Build deterministic teachers from stored tool-use traces.
   * - Distill one Torch module into another
     - ``from mixle.task.distill_methods import response_distill, hint_distill``
     - Classic KD, feature matching, attention transfer, relational KD, and
       sequence-level distillation.

Build a local reasoning application
-----------------------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Import
     - Notes
   * - Store and retrieve typed knowledge
     - ``from mixle.substrate import Substrate, retrieve``
     - Local scoped store for documents, records, artifacts, traces, and
       context packets.
   * - Ask over evidence and skills
     - ``from mixle.substrate import Reasoner, investigate``
     - Fires retrieve/compute/simulate/create/delegate actions under a budget.
   * - Package a model as a capability
     - ``from mixle.inference import skill, SkillRegistry``
     - Named callable with provenance and inherited certificate metadata.
   * - Check answer factuality
     - ``from mixle.substrate import check_factuality``
     - Claim-level support from substrate evidence.
   * - Apply ontology constraints
     - ``from mixle.reason.ontology import Ontology``
     - Typed relation constraints and graph-fact auditing.
   * - Submit local-or-pool work
     - ``from mixle.pool import PoolJob, submit``
     - Budgeted job abstraction with local fallback.
   * - Record decision telemetry
     - ``from mixle.telemetry import Telemetry, record``
     - JSONL events for routing, placement, reasoning, pool jobs, and drift.

Quantify LLM and reasoning uncertainty
--------------------------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Import
     - Notes
   * - Semantic-entropy LLM UQ
     - ``from mixle.reason import LLMUncertainty``
     - Wraps any ``generate(prompt) -> str`` callable.
   * - Claim-level reliability
     - ``from mixle.reason import sentence_claims, information_corroborator``
     - Checks whether response claims recur across independent samples.
   * - Cross-modal evidence fusion
     - ``from mixle.reason import Latent, Evidence, reason``
     - Exact linear-Gaussian latent assimilation with attribution.
   * - Epistemic/aleatoric splits
     - ``from mixle.inference.uncertainty import decompose_entropy``
     - Used by LLM and scientific reasoning surfaces.

Transform distributions
-----------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Import
     - Notes
   * - Quantize a distribution
     - ``from mixle.ops import quantize``
     - Turns a continuous distribution into finite support for enumeration.
   * - Condition or marginalize
     - ``from mixle.ops import condition, marginalize``
     - Requires the model to expose the relevant capability.
   * - Build a latent mixture
     - ``from mixle.ops import mixture``
     - Convenience constructor for weighted mixtures.
   * - Project into a simpler family
     - ``from mixle.ops import project``
     - Sample-based forward-KL projection into a fittable target family.
   * - Collapse or reduce Gaussian mixtures exactly
     - ``from mixle.inference import collapse_mixture, reduce_mixture``
     - Closed-form moment projection and Runnalls mixture reduction.
   * - Merge parameter estimates by Fisher information
     - ``from mixle.inference import fisher_merge``
     - Precision-weighted parameter merge for Laplace/Fisher summaries.
   * - Pool experts
     - ``from mixle.ops import product_of_experts``
     - Exact for supported tractable families such as shared categoricals and Gaussians.

Solve structured relations
--------------------------

.. code-block:: python

   from mixle.relations import Assignment, EditDistance, ShortestPath, ViterbiPath

Relations enumerate feasible structured objects in objective order. Use them
for k-best assignments, paths, edit neighborhoods, spanning trees, constrained
subsets, and graph decisions.

Model temporal processes
------------------------

.. code-block:: python

   from mixle.process import (
       HawkesProcessDistribution,
       InhomogeneousPoissonProcessDistribution,
       RenewalProcessDistribution,
   )
   from mixle.stats.processes.ctmc import ContinuousTimeMarkovChainDistribution

The process namespace collects event-time, renewal, self-exciting,
birth-death, CTMC, and random-partition families.

Analyze diagnostics and data structure
--------------------------------------

.. code-block:: python

   from mixle.analysis import gpd_fit, kde, ordinary_kriging, chao1, borda_count

``mixle.analysis`` contains applied routines for extreme values, KDE, coverage,
kriging, rank aggregation, spatial mixtures, max-stable processes, and
covariance shrinkage.

Improve and search models
-------------------------

.. code-block:: python

   from mixle.evolve import improve, search, Space, Real, Integer, Categorical, nll_objective

``mixle.evolve`` provides anti-regression improvement loops, typed search
spaces, objective builders, operator registries, promotion verdicts, and
evolution ledgers.

Use the broader inference toolkit
---------------------------------

.. code-block:: python

   from mixle.inference import (
       collapse_mixture,
       laplace_posterior,
       log_score,
       reliability_curve,
       select_best,
       split_conformal,
       vuong_test,
   )

Use :doc:`inference-toolkit` for scoring rules, calibration, conformal
prediction, cross-validation, model comparison, multiple testing, regression,
survival, resampling, robust covariance, posterior helpers, closed-form
projection/compression, verifier-based selection, and decision utilities.

Serve task models
-----------------

.. code-block:: python

   from mixle.task import (
       DeviceSpec,
       Router,
       quantize_mlp,
       scorecard,
       solve,
       solve_multilabel,
       solve_regression,
       solve_structured,
   )

Use :doc:`task-serving` for one-call task replacement, multi-tier routing,
numeric, multi-label, and structured-output task replacement, edge-device
search, quantized students, scorecards, and harnesses for replacing legacy
extractors, alert rules, and matchers.

Build reasoning systems
-----------------------

.. code-block:: python

   from mixle.reason import reason_discrete, GraphLLM, CrossModalStore, CrossModalModel

Use :doc:`reasoning-systems` for finite-hypothesis reasoning, graph-producing
LLMs, cross-modal retrieval, evidence acquisition, amortized encoders, and
learned multimodal latent models.

Inspect capabilities
--------------------

.. code-block:: python

   import mixle

   mixle.describe(model)
   mixle.capabilities(model)
   mixle.supports(model, mixle.capability.Enumerable)

The capability layer is the right way to ask what an object can do. It is more
stable than checking concrete classes. See :doc:`capabilities-contracts` for
the full behavior catalog.

Use the lifecycle facade
------------------------

.. code-block:: python

   import mixle

   m = mixle.propose(rows, fit=True)
   print(m.evaluate(holdout))
   print(m.explain())

``mixle.Model`` and ``mixle.propose`` provide a high-level lifecycle around
proposal, fitting, evaluation, posterior queries, distillation, deployment, and
explanation. See :doc:`lifecycle`.

Understand automatic modeling internals
---------------------------------------

.. code-block:: python

   from mixle.utils.automatic import analyze_structure, get_estimator

   profile = analyze_structure(rows)
   for line in profile.explain():
       print(line)

Use :doc:`automatic-modeling-internals` when you need to inspect the profiling
objects, model-family score gaps, dependency hints, validation notes, and
factory functions behind ``recommend_model`` and ``get_estimator``.

Use the PPL surface
-------------------

.. code-block:: python

   from mixle.ppl import Normal, Poisson, Mix, Markov, Field, Group, free

PPL constructors build symbolic random variables. Calling ``.fit(...)`` lowers
them to ordinary ``mixle.stats`` distributions and estimators.

Enumerate structured supports
-----------------------------

.. code-block:: python

   from mixle.enumeration import top_k, density_rank, supports_enumeration

``top_k`` and ``supports_enumeration`` are the usual first calls. Use
``density_rank`` when you need rank or cumulative-mass information for a value.

Run on engines and backends
---------------------------

.. code-block:: python

   from mixle.engines import TorchEngine, NumpyEngine, SymbolicEngine
   from mixle.inference import optimize

   optimize(data, estimator, engine=TorchEngine(device="cuda"))
   optimize(data, estimator, backend="mp", num_workers=4)

``engine=`` controls array math and devices. ``backend=`` controls where encoded
data is folded: local process, multiprocessing, Spark, Dask, MPI, and related
adapters.

Use lower-level compute surfaces
--------------------------------

.. code-block:: python

   from mixle.stats.compute.sequence import seq_encode, seq_log_density_sum
   from mixle.stats.compute.kernel import kernel_for

The compute layer contains the distribution contracts, encoded-data helpers,
sequence drivers, declaration metadata, generated kernels, backend scoring, and
stacked mixture paths that support the public APIs. See :doc:`compute-layer`.

Use utility and parallel helpers
--------------------------------

.. code-block:: python

   from mixle.utils.serialization import to_json, from_json
   from mixle.utils.parallel import Resources, encoded_data, plan

Use :doc:`utilities-and-parallelism` for safe serialization, optional
dependency gates, metrics, HVIS helpers, encoded-data backends, resource
planning, and model-parallel estimators.

Work with data sources
----------------------

.. code-block:: python

   from mixle.data import Schema, Field, Real, Text, check_dataset, dataset_hash

The data layer is optional. Plain Python sequences remain accepted by the
encoder contract.

Generated Reference Scope
-------------------------

The Sphinx API reference is regenerated from the package tree with
``make -C docs apidoc``. It should be treated as an inspection surface, not as
the recommended import map. For application code:

* prefer package-level exports such as ``mixle.stats``, ``mixle.inference``,
  ``mixle.task``, ``mixle.doe``, and ``mixle.ppl`` when they exist;
* use implementation submodules when a guide names them directly or when you
  need a specific advanced surface; and
* avoid depending on private modules solely because they appear in generated
  autodoc output.

When a new module is added to the package, regenerate the API pages and include
the changed ``docs/api/*.rst`` source files in the documentation PR. The built
HTML under ``docs/_build`` remains local build output and should not be
committed.
