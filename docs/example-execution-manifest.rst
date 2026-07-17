Example Execution Manifest
==========================

This page is the release-facing execution manifest for scripts shipped under
``examples/``. The examples guide describes what each script teaches; this
manifest records what must be executed, skipped, or marked blocked before a
public release.

Current Inventory
-----------------

The core package currently ships 57 Python example scripts:

.. list-table::
   :header-rows: 1

   * - Group
     - Examples
     - Release tier
   * - Base distribution galleries
     - ``gallery_*``, ``auto_example.py``, ``ppl_example.py``
     - smoke/validation
   * - Latent and structured models
     - HMM, mixture, association, and structure-learning examples
     - validation
   * - DOE, enumeration, production, and scaling
     - DOE, enumeration, registry/provenance, and backend examples
     - validation/manual
   * - Task and distillation workflows
     - task distillation, active labeling, extraction, cascade economics
     - validation/manual
   * - Reasoning, cross-modal, and scientist workflows
     - frontier ecosystem, KG agent, scientist, physics inverse, receipts
     - manual unless dependencies are provisioned
   * - Vision edge distillation
     - ``examples/vision_edge_distillation``
     - manual/GPU or cached-feature validation

Release Execution Status
------------------------

The 23 base-install ("Execute.") examples were re-run on 2026-07-17 against the
built ``mixle-0.8.0.dev0`` wheel, installed into a bare venv (``pip install
dist/mixle-0.8.0.dev0-py3-none-any.whl``, no extras, no editable install, no
``PYTHONPATH``, Python 3.12), each with a 90s budget:

* **19 passed**: ``enumeration_example.py``, ``extensibility_seams_example.py``,
  ``gallery_combinators_example.py``, ``gallery_directional_example.py``,
  ``gallery_graphs_example.py``, ``gallery_multivariate_example.py``,
  ``gallery_processes_example.py``, ``gallery_rankings_example.py``,
  ``gallery_structured_example.py``, ``gallery_univariate_example.py``,
  ``heterogeneous_correctness_example.py``, ``hidden_association_example.py``,
  ``joint_mixture_example.py``, ``latent_variable_models_example.py``,
  ``ppl_example.py``, ``semi_supervised_mixture_example.py``,
  ``structure_learning_example.py``, ``structured_hmm_example.py``,
  ``structured_leaves_example.py``.
* **2 reclassified to blocked** (unchanged from the prior pass; see the
  Inventory table below): ``skeptic_challenge_example.py`` (needs
  ``scikit-learn``, not a mixle dependency) and ``win_demo_example.py`` (needs
  ``torch``; its distillation path has no classical fallback).
* **2 exceeded the 90s budget**: ``hierarchical_mixture_example.py`` and
  ``lookback_hmm_example.py``. Recorded here as ``timed_out`` per this run's
  evidence, not asserted as a release-blocking regression -- this execution
  host was under extreme concurrent load for the whole pass (``uptime`` load
  averages of ~140-175 against 10 cores, from unrelated concurrent work), which
  inflates every wall-clock measurement in this run. Re-run individually with a
  15-minute allowance to tell "slow under load" from "hung":

  * ``lookback_hmm_example.py`` completed cleanly in 86s on the re-run --
    essentially at the 90s line, consistent with host-load variance rather
    than a regression. It calls ``optimize(..., max_its=1000, delta=None)``;
    ``delta=None`` disables early-stopping, so it always runs the full 1000 EM
    iterations regardless of convergence, by design -- a fixed amount of work
    that was already close to the budget before this pass.
  * ``hierarchical_mixture_example.py`` did **not** complete even with the
    15-minute allowance (10x the declared budget), while confirmed still
    actively computing throughout, not deadlocked (steadily increasing CPU
    time, never crashed or errored). This is a bigger gap than host load alone
    plausibly explains. Its fit path
    (``HierarchicalMixtureEstimatorAccumulator``) was rewritten four days
    before this pass by the EM-monotonicity fix in #435 (2026-07-13, see
    ``mixle/stats/latent/hierarchical_mixture.py``), which plausibly changed
    how many of its ``max_its=10000`` iterations are needed to satisfy the
    default convergence ``delta``. Treat this one as the more likely of the
    two to be a real behavior change and re-verify on an unloaded runner
    before re-budgeting or investigating further.

Separately, the two real-data flagship examples added 2026-07-11 (F10.1/F10.2;
not part of the base-install set above -- both need network access, and the
Adult flagship additionally needs the optional ``datasets`` package) were run
against the same wheel with their dependencies met:

* ``flagship_temporal_sunspots.py`` -- **passed**. Fetches the public monthly
  sunspot series over the network; no extra package is required for the fit
  itself (the ``hmmlearn`` comparison degrades to ``None`` if absent, but
  ``hmmlearn`` was installed for this run). Held-out mean log-likelihood per
  observation: mixle -2.0762 vs. hmmlearn -2.0776 -- independent-baseline
  agreement is the receipt.
* ``flagship_heterogeneous_adult.py`` -- **passed**. Downloads UCI Adult via
  the optional ``datasets`` package (``pip install datasets``; not a core
  dependency or always-installed extra). Mean log-density: train -10.719,
  held-out -10.839 (held-out close to train -- generalized, not memorized).

The remaining examples (task/DOE/reasoning/vision workflows, plus the other
three flagships) were not executed in this pass -- they need optional extras,
external datasets, model weights, or services per their Inventory entries
below, none of which are provisioned in this environment.

Execution status should be recorded as evidence, not inferred from import
success or from an earlier notebook run. If an example writes an artifact, the
artifact path and any cleanup policy should be captured with the status.

Every example must be recorded as one of:

``passed``
    The script completed from a clean install with the documented command.

``failed``
    The script exited non-zero. Record the first meaningful exception.

``timed_out``
    The script exceeded the declared runtime budget.

``blocked``
    A required dataset, credential, GPU, model weight, external service, or
    optional dependency was unavailable.

``skipped``
    The release configuration intentionally omitted the script with a recorded
    reason.

Do not merge ``blocked`` and ``skipped``. A blocked example needs an external
prerequisite; a skipped example was deliberately left out of the release gate.
That distinction matters when deciding whether documentation can claim the
workflow is healthy.

Minimum Release Run
-------------------

The minimum release run should include:

* all base-install examples listed in :doc:`examples`;
* every example referenced by README files or Sphinx pages;
* every example touching public APIs changed in the current release scope;
* task-distillation examples because the current release scope includes task
  and DOE-distillation surfaces;
* DOE examples because the current release scope includes pool-based DOE for
  distillation and cross-modal training; and
* vision or reasoning examples only when their optional dependencies are
  installed, otherwise mark them blocked with the missing prerequisite.

Example command shape:

.. code-block:: console

   python examples/gallery_univariate_example.py
   python examples/task_distill_example.py
   python examples/doe_example.py

When release validation uses a timeout wrapper, record the timeout and whether
the script is expected to be short, long, or manual.

Timeouts should be chosen before execution. A script that times out under the
declared budget should be recorded as ``timed_out`` even if it might finish
eventually on a warmer machine.

Inventory
---------

.. list-table::
   :header-rows: 1

   * - Path
     - Expected status before release
   * - ``examples/auto_example.py``
     - Execute or record failure.
   * - ``examples/calibrated_report_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/cross_modal_fit_receipt.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/doe_example.py``
     - Execute for DOE coverage.
   * - ``examples/engine_benchmark_example.py``
     - Manual/benchmark or bounded smoke run.
   * - ``examples/enumeration_example.py``
     - Execute.
   * - ``examples/enumeration_showcase_example.py``
     - Execute or classify as long-running.
   * - ``examples/extensibility_seams_example.py``
     - Execute.
   * - ``examples/flagship_heterogeneous_adult.py``
     - Execute with network access and the optional ``datasets`` package
       (downloads UCI Adult, not a core dependency); intended as an F10.4
       release gate.
   * - ``examples/flagship_kg_agent.py``
     - Manual unless KG/RAG prerequisites are provisioned.
   * - ``examples/flagship_physics_inverse.py``
     - Execute or mark blocked on PDE/scientific dependencies.
   * - ``examples/flagship_temporal_sunspots.py``
     - Execute with network access (fetches the public sunspot series);
       ``hmmlearn`` comparison is optional and degrades to ``None`` if
       absent; intended as an F10.4 release gate.
   * - ``examples/flagship_triage_app.py``
     - Manual unless local reasoning prerequisites are provisioned.
   * - ``examples/frontier_family_showcase.py``
     - Manual/integration.
   * - ``examples/geoscience_inversion_report.py``
     - Execute or mark blocked on scientific dependencies.
   * - ``examples/foundation_to_edge.py``
     - Manual unless model weights are provisioned.
   * - ``examples/frontier_ecosystem_demo.py``
     - Manual/integration.
   * - ``examples/gallery_combinators_example.py``
     - Execute.
   * - ``examples/gallery_directional_example.py``
     - Execute.
   * - ``examples/gallery_graphs_example.py``
     - Execute.
   * - ``examples/gallery_multivariate_example.py``
     - Execute.
   * - ``examples/gallery_processes_example.py``
     - Execute.
   * - ``examples/gallery_rankings_example.py``
     - Execute.
   * - ``examples/gallery_structured_example.py``
     - Execute.
   * - ``examples/gallery_univariate_example.py``
     - Execute.
   * - ``examples/heterogeneous_correctness_example.py``
     - Execute.
   * - ``examples/heterogeneous_representation_example.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/hidden_association_example.py``
     - Execute.
   * - ``examples/hierarchical_mixture_example.py``
     - Execute.
   * - ``examples/joint_mixture_example.py``
     - Execute.
   * - ``examples/label_economics_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/laptop_scientist.py``
     - Manual unless local model weights are provisioned.
   * - ``examples/latent_variable_models_example.py``
     - Execute.
   * - ``examples/lookback_hmm_example.py``
     - Execute.
   * - ``examples/mixture_reduction_benchmark.py``
     - Manual/benchmark or bounded smoke run.
   * - ``examples/multimodal_stage1_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/peft_lora_grad_leaf.py``
     - Execute with optional-dependency status recorded (needs ``peft``,
       not a mixle dependency).
   * - ``examples/ppl_example.py``
     - Execute.
   * - ``examples/production_example.py``
     - Execute with artifact-output path recorded.
   * - ``examples/project_neural_to_structured.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/real_receipt_banking77.py``
     - Manual or blocked unless dataset download is permitted.
   * - ``examples/reasoner_investigation_demo.py``
     - Manual/integration.
   * - ``examples/scaling_example.py``
     - Execute or classify by backend availability.
   * - ``examples/semi_supervised_mixture_example.py``
     - Execute.
   * - ``examples/shared_embedding_example.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/skeptic_challenge_example.py``
     - Blocked on ``scikit-learn`` (Act 1's sklearn-baseline comparison
       imports it directly; not a mixle dependency or extra, install
       separately to run).
   * - ``examples/structure_learning_example.py``
     - Execute.
   * - ``examples/structured_hmm_example.py``
     - Execute.
   * - ``examples/structured_leaves_example.py``
     - Execute.
   * - ``examples/task_cascade_economics_example.py``
     - Execute for task coverage.
   * - ``examples/task_distill_example.py``
     - Execute for task coverage.
   * - ``examples/task_extraction_example.py``
     - Execute for task coverage.
   * - ``examples/task_llm_active_example.py``
     - Execute or mark blocked on teacher/provider requirements.
   * - ``examples/vision_edge_distillation/distill_clip_features.py``
     - Manual or blocked unless cached features/model weights are present.
   * - ``examples/vision_edge_distillation/verify_on_laptop.py``
     - Execute when the distilled artifact exists; otherwise blocked.
   * - ``examples/vlm_trust_receipts_demo.py``
     - Execute with optional-dependency status recorded.
   * - ``examples/win_demo_example.py``
     - Blocked on ``torch`` (``solve()``'s MLP distillation path
       (``mixle.task.distill._fit_mlp``) always needs torch, no classical
       fallback; ``pip install mixle[torch]``).

Evidence to Record
------------------

For each executed example, record:

* command;
* package version or wheel filename installed;
* optional extras installed;
* timeout;
* status;
* first error or output artifact path; and
* whether network or external data was used.

For blocked or skipped examples, record:

* the missing prerequisite or release-scope reason;
* whether the example is referenced by a public guide;
* the owner of the follow-up decision; and
* the condition that would move it back into the release gate.

The manifest should make it clear which examples are evidence for the release
and which examples remain illustrative but unexecuted.
