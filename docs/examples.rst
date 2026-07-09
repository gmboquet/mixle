Examples
========

Runnable examples live in ``examples/``. They are plain Python scripts, so the
fastest way to learn a workflow is to run the script under the same environment
you use for tests.

For a shorter, receipt-focused index of the newer applied adapter and
multimodal-pretraining workflows, see :doc:`examples_gallery`.

Start with the base-install examples. Move to Torch, task, or real-data workflows
only after the core distribution and inference path is clear.

Examples are documentation assets, not release gates by themselves. When an
example backs a public claim, record its execution status in
:doc:`example-execution-manifest` and note any skipped optional dependency.

Recommended First Runs
----------------------

.. list-table::
   :header-rows: 1

   * - Script
     - What it demonstrates
     - Needs
   * - ``gallery_univariate_example.py``
     - Scalar continuous and discrete families.
     - Base install
   * - ``gallery_combinators_example.py``
     - Records, tuples, sequences, optional fields, and transforms.
     - Base install
   * - ``gallery_structured_example.py``
     - Mixtures, HMMs, LDA, and latent-variable models.
     - Base install
   * - ``auto_example.py``
     - Automatic estimator selection for mixed Python records.
     - Base install
   * - ``structured_hmm_example.py``
     - Low-rank, sparse, Kronecker, duration, terminal, and input-output HMMs.
     - Base install
   * - ``production_example.py``
     - Provenance, registry, serving, drift, and checkpoints.
     - Base install

Run them directly:

.. code-block:: sh

   python examples/gallery_univariate_example.py
   python examples/gallery_combinators_example.py
   python examples/gallery_structured_example.py
   python examples/auto_example.py
   python examples/structured_hmm_example.py
   python examples/production_example.py

Choose by Workflow
------------------

.. list-table::
   :header-rows: 1

   * - Workflow
     - Start with
     - Related guides
   * - Heterogeneous records
     - ``auto_example.py``, ``gallery_combinators_example.py``
     - :doc:`quickstart`, :doc:`tutorials/heterogeneous-records`
   * - Mixtures and latent state models
     - ``gallery_structured_example.py``, ``structured_hmm_example.py``
     - :doc:`hmms-latent`, :doc:`stats-latent-bayes`
   * - Mixture compression and projection
     - ``mixture_reduction_benchmark.py``,
       ``project_neural_to_structured.py``
     - :doc:`inference-toolkit`, :doc:`operations`
   * - Distribution family discovery
     - ``gallery_univariate_example.py``, ``gallery_multivariate_example.py``,
       ``gallery_directional_example.py``
     - :doc:`stats-univariate`, :doc:`stats-structured`
   * - Graphs, rankings, trees, and sets
     - ``gallery_graphs_example.py``, ``gallery_rankings_example.py``
     - :doc:`stats-structured`, :doc:`relations`
   * - Temporal and point processes
     - ``gallery_processes_example.py``
     - :doc:`processes`
   * - Probabilistic programming
     - ``ppl_example.py``
     - :doc:`ppl`, :doc:`tutorials/ppl-mixture`
   * - Exact support traversal
     - ``enumeration_example.py``, ``enumeration_showcase_example.py``
     - :doc:`enumeration`, :doc:`tutorials/enumeration-ranking`
   * - Scaling and engines
     - ``scaling_example.py``, ``engine_benchmark_example.py``
     - :doc:`engines`, :doc:`utilities-and-parallelism`
   * - Neural or representation workflows
     - ``shared_embedding_example.py``, ``heterogeneous_representation_example.py``,
       ``cross_modal_fit_receipt.py``
     - :doc:`neural-llm`, :doc:`representation`
   * - Local reasoning ecosystem
     - ``frontier_ecosystem_demo.py``, ``reasoner_investigation_demo.py``,
       ``flagship_triage_app.py``, ``flagship_kg_agent.py``
     - :doc:`reasoning-ecosystem`, :doc:`reasoning-systems`
   * - Local scientist and edge distillation
     - ``laptop_scientist.py``, ``foundation_to_edge.py``,
       ``vision_edge_distillation/``
     - :doc:`reasoning-ecosystem`, :doc:`task-distillation`
   * - Scientific inverse problems
     - ``flagship_physics_inverse.py``, ``skeptic_challenge_example.py``
     - :doc:`inference`, :doc:`ppl`, :doc:`uncertainty`
   * - LLM/task replacement
     - ``task_distill_example.py``, ``task_llm_active_example.py``,
       ``task_cascade_economics_example.py``
     - :doc:`task-distillation`, :doc:`task-serving`
   * - Extraction and agent-style task behavior
     - ``task_extraction_example.py``, ``win_demo_example.py``
     - :doc:`task-serving`, :doc:`agentic-task-distillation`
   * - Real-data task workflow
     - ``real_receipt_banking77.py``
     - :doc:`task-serving`, :doc:`maturity`

Dependency Notes
----------------

Most examples use only the base package dependencies. Examples that train
neural students, neural leaves, representation models, or neural-density
teachers generally need ``mixle[torch]``. The real-data Banking77 receipt
additionally needs the Hugging Face ``datasets`` package and downloads the
dataset on first run.
The ``laptop_scientist.py`` and ``foundation_to_edge.py`` workflows need the
``scientist`` extra and local Hugging Face model weights where noted by the
scripts.

.. code-block:: sh

   pip install -e .
   pip install -e ".[torch]"
   pip install -e ".[scientist]"
   pip install datasets

Use ``mixle[all]`` only when you deliberately want the full optional surface.
For most examples, a smaller extra is easier to debug.

Execution Standard
------------------

When adding or updating an example:

* keep it runnable from the repository root;
* print a compact result that confirms the intended behavior;
* avoid private data, external credentials, and hidden services;
* mark optional dependencies in this page or in the script header;
* keep long benchmarks outside the documentation example set;
* update :doc:`example-execution-manifest` when the example is part of release
  evidence.

Examples that require downloaded datasets or large local weights should degrade
clearly when those resources are missing.

Complete Inventory
------------------

.. list-table::
   :header-rows: 1

   * - Script
     - Category
     - Purpose
   * - ``auto_example.py``
     - Automatic inference
     - Infer an estimator shape for mixed Python records.
   * - ``cross_modal_fit_receipt.py``
     - Cross-modal inference
     - Fit a heterogeneous Bayesian network over categorical, image-vector,
       signal-vector, and continuous fields.
   * - ``doe_example.py``
     - Design of experiments
     - Latin hypercube designs, Bayesian optimization, and sensitivity.
   * - ``engine_benchmark_example.py``
     - Engines
     - Compare NumPy and Torch engine paths across workloads with explicit
       tolerances.
   * - ``enumeration_example.py``
     - Enumeration
     - Exact top-k support traversal.
   * - ``enumeration_showcase_example.py``
     - Enumeration
     - Broader ranking, seek, and structured-support examples.
   * - ``extensibility_seams_example.py``
     - Extension
     - Show where new families and backends attach.
   * - ``flagship_kg_agent.py``
     - Reasoning ecosystem
     - Ontology-constrained graph facts, KG completion, and cited KG-RAG.
   * - ``flagship_physics_inverse.py``
     - Scientific inference
     - Bayesian inverse problem with coverage-oriented uncertainty checks.
   * - ``flagship_triage_app.py``
     - Reasoning ecosystem
     - Support triage over substrate, skills, pool-style jobs, monitoring, and
       grounded answering.
   * - ``foundation_to_edge.py``
     - Edge distillation
     - Distill a foundation-model capability into a smaller local artifact and
       report retained accuracy.
   * - ``frontier_ecosystem_demo.py``
     - Local reasoning ecosystem
     - End-to-end tour of substrate, creation, simulation, skills, reasoning,
       telemetry, and governance surfaces.
   * - ``gallery_combinators_example.py``
     - Distribution gallery
     - Composite, record, sequence, optional, and transformed families.
   * - ``gallery_directional_example.py``
     - Distribution gallery
     - Circular and spherical directional families.
   * - ``gallery_graphs_example.py``
     - Distribution gallery
     - Graph and tree distributions.
   * - ``gallery_multivariate_example.py``
     - Distribution gallery
     - Vector and matrix families.
   * - ``gallery_processes_example.py``
     - Distribution gallery
     - Temporal and point-process families.
   * - ``gallery_rankings_example.py``
     - Distribution gallery
     - Ranking and permutation families.
   * - ``gallery_structured_example.py``
     - Distribution gallery
     - Mixtures, HMMs, LDA, and latent models.
   * - ``gallery_univariate_example.py``
     - Distribution gallery
     - Scalar continuous and discrete families.
   * - ``heterogeneous_correctness_example.py``
     - Validation
     - Correctness checks across heterogeneous components.
   * - ``heterogeneous_representation_example.py``
     - Representation
     - Segmenters, embeddings, and shared heterogeneous encoders.
   * - ``hidden_association_example.py``
     - Latent models
     - Association models with hidden structure.
   * - ``hierarchical_mixture_example.py``
     - Latent models
     - Hierarchical mixture variants.
   * - ``joint_mixture_example.py``
     - Latent models
     - Joint mixture variants.
   * - ``laptop_scientist.py``
     - Local scientist
     - Optional assembled workflow for cached encoders, local answering, and
       verified scientific responses.
   * - ``latent_variable_models_example.py``
     - Latent models
     - Latent families beyond the first HMM path.
   * - ``lookback_hmm_example.py``
     - Latent models
     - HMMs with longer history.
   * - ``mixture_reduction_benchmark.py``
     - Projection/compression
     - Compare closed-form Gaussian-mixture reduction with sample-and-refit
       projection.
   * - ``project_neural_to_structured.py``
     - Projection/compression
     - Project a trained neural density onto a structured Gaussian-mixture
       student and measure size, latency, and likelihood tradeoffs.
   * - ``ppl_example.py``
     - PPL
     - ``free`` parameters, mixtures, sequences, and moments.
   * - ``production_example.py``
     - Production
     - Provenance, registry, serving, drift, and checkpoints.
   * - ``real_receipt_banking77.py``
     - Real-data task workflow
     - Banking77 intent classification through the ``solve`` loop.
   * - ``reasoner_investigation_demo.py``
     - Reasoning ecosystem
     - Evidence acquisition over retrieve, compute, simulate, and delegate
       actions with abstention.
   * - ``scaling_example.py``
     - Parallelism
     - Same ``optimize`` call on local and multiprocessing backends.
   * - ``semi_supervised_mixture_example.py``
     - Latent models
     - Partially labeled mixture fitting.
   * - ``shared_embedding_example.py``
     - Neural leaf
     - Mixture of language-model experts with one shared embedding.
   * - ``skeptic_challenge_example.py``
     - Verification workflow
     - Stress-check claims and examples against explicit evidence.
   * - ``structure_learning_example.py``
     - Applied models
     - Dependency and structure learning before modeling.
   * - ``structured_hmm_example.py``
     - HMMs
     - Structured transition variants and duration behavior.
   * - ``structured_leaves_example.py``
     - Structured models
     - Structured emissions and leaves.
   * - ``task_cascade_economics_example.py``
     - Task workflow
     - Cascade cost accounting, harvesting, and retraining.
   * - ``task_distill_example.py``
     - Task workflow
     - Distill a slow teacher into a local callable artifact.
   * - ``task_extraction_example.py``
     - Task workflow
     - Distill LLM field extraction into a local sequence tagger.
   * - ``task_llm_active_example.py``
     - Task workflow
     - LLM teacher, active labeling, local student, and calibrated cascade.
   * - ``vision_edge_distillation/``
     - Edge distillation
     - Train and verify a compact vision student from cached or reproduced
       foundation-model features.
   * - ``win_demo_example.py``
     - End-to-end workflow
     - Replace a ticket router and invoice extractor with calibrated models.

Benchmark and distributed-stress harnesses that were previously under
``examples/`` have been moved to gitignored benchmark areas. The tracked
examples page now focuses on scripts designed to be read and run as
documentation.

Representative Source
---------------------

Structured HMMs:

.. literalinclude:: ../examples/structured_hmm_example.py
   :language: python
   :caption: examples/structured_hmm_example.py

Active LLM distillation and cascade:

.. literalinclude:: ../examples/task_llm_active_example.py
   :language: python
   :caption: examples/task_llm_active_example.py
