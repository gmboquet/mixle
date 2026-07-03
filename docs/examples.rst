Examples
========

Runnable examples live in ``examples/``. They are plain Python scripts, so the
fastest way to learn a workflow is to run the script under the same environment
you use for tests.

Start with the base-install examples. Move to Torch, task, or real-data workflows
only after the core distribution and inference path is clear.

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

Choose By Workflow
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
     - ``shared_embedding_example.py``, ``heterogeneous_representation_example.py``
     - :doc:`neural-llm`, :doc:`representation`
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

.. code-block:: sh

   pip install -e .
   pip install -e ".[torch]"
   pip install datasets

Use ``mixle[all]`` only when you deliberately want the full optional surface.
For most examples, a smaller extra is easier to debug.

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
   * - ``doe_example.py``
     - Design of experiments
     - Latin hypercube designs, Bayesian optimization, and sensitivity.
   * - ``engine_benchmark_example.py``
     - Engines
     - Compare NumPy and Torch engine paths honestly across workloads.
   * - ``enumeration_example.py``
     - Enumeration
     - Exact top-k support traversal.
   * - ``enumeration_showcase_example.py``
     - Enumeration
     - Broader ranking, seek, and structured-support examples.
   * - ``extensibility_seams_example.py``
     - Extension
     - Show where new families and backends attach.
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
   * - ``scaling_example.py``
     - Parallelism
     - Same ``optimize`` call on local and multiprocessing backends.
   * - ``semi_supervised_mixture_example.py``
     - Latent models
     - Partially labeled mixture fitting.
   * - ``shared_embedding_example.py``
     - Neural leaf
     - Mixture of language-model experts with one shared embedding.
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
   * - ``win_demo_example.py``
     - End-to-end workflow
     - Replace a ticket router and invoice extractor with calibrated models.

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
