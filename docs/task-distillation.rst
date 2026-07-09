Task Distillation
=================

``mixle.task`` is for application tasks where the expensive thing is not
fitting a density, but asking a teacher. The teacher might be a frontier LLM, a
hosted endpoint, a human-reviewed service, or a slow rule system. Mixle turns
that teacher into a small local model and gives you the machinery to decide
when the local model is allowed to answer.

The serving loop is:

.. code-block:: text

   unlabeled pool -> teacher labels -> local student
        ^                              |
        |                              v
   harvested escalations <- calibrated cascade <- traffic

For single-label tasks, the local model answers only when its calibrated label
set is a singleton. For multi-label tasks, ``solve_multilabel`` decides each
tag separately and escalates if any tag is ambiguous. For numeric tasks,
``solve_regression`` answers only when its split-conformal interval is narrow
enough for the caller's tolerance. For dict-valued tasks,
``solve_structured`` composes per-field calibrated solvers and escalates if any
field is uncertain. These routes keep the same safety shape: local when
calibrated, teacher when uncertain.

Why This Exists
---------------

If you call the LLM for every request, quality may be high but cost and latency
stay high. If you serve a local classifier without calibration, a confident
softmax can still be wrong or out-of-distribution. ``mixle.task`` makes the
middle path concrete:

* spend teacher calls on useful labels;
* train a local student;
* calibrate answer sets with conformal prediction;
* escalate ambiguous or OOD inputs;
* track realized dollars saved;
* harvest escalations as targeted labels for the next training round.

Wrap an LLM Teacher
-------------------

``CallableLLM`` adapts any callable. ``llm_labeler`` constrains an LLM-like
generator to a label set.

.. code-block:: python

   from mixle.task import CallableLLM, llm_labeler

   def generate(prompt, system=None):
       # In production this can call an OpenAI-compatible endpoint.
       return "spam" if "free prize" in prompt.lower() else "ham"

   teacher = llm_labeler(
       CallableLLM(generate),
       ["spam", "ham"],
       instruction="Classify the email as spam or ham.",
   )

For hosted or local servers with an OpenAI-compatible API, use
``OpenAICompatLLM``.

Teacher Contract
----------------

The teacher is part of the dataset, not an interchangeable implementation detail. For
every distillation run, record:

* the teacher function, endpoint, or human-review process;
* the prompt or instruction used to constrain the teacher;
* the allowed label set, schema, tool list, or numeric output contract;
* retry and parsing behavior for malformed teacher responses;
* the date or model version when a hosted teacher is used;
* examples rejected by validation before they reached the student.

This is the provenance that lets a future run distinguish model improvement
from a silent change in what the teacher meant.

Numeric Teachers
----------------

Some teachers return numbers rather than labels: pricing functions, risk
scores, sizing rules, simulation surrogates, and scoring services. Use
``solve_regression`` for that shape.

.. code-block:: python

   from mixle.task import solve_regression

   solution = solve_regression(price, historical_items, tol=5.0, alpha=0.1)
   value = solution(new_item)
   yhat, lo, hi = solution.interval(new_item)

Calibration is split conformal over absolute residuals. The local model only
answers when the calibrated width ``qhat`` is at most ``tol``; otherwise it
calls the original teacher and harvests the pair for a later improvement
round. See :doc:`task-serving` for the production contract.

Multi-Label Teachers
--------------------

Use ``solve_multilabel`` when the teacher returns a set of tags or flags.

.. code-block:: python

   from mixle.task import solve_multilabel

   def flags(transaction):
       out = []
       if transaction["amount"] > 400:
           out.append("high-value")
       if transaction["region"] == "eu":
           out.append("eu-rules")
       return out

   solution = solve_multilabel(flags, historical_transactions, alpha=0.1)
   tags = solution(new_transaction)

Calibration is per-label. A tag can be confidently present, confidently
absent, or ambiguous. The whole request escalates if any tag is ambiguous, so a
locally returned set is made only of decided labels rather than guessed labels.
This is useful for compliance flags, routing tags, alert annotations, and
document categories where several labels may be true at once.

Structured Output Teachers
--------------------------

Use ``solve_structured`` when the teacher returns a stable dictionary.

.. code-block:: python

   from mixle.task import solve_structured

   def enrich(ticket):
       return {
           "route": "finance" if ticket["amount"] > 10_000 else "ops",
           "priority": "high" if ticket["age_hours"] > 24 else "normal",
           "reserve": ticket["amount"] * 0.15,
       }

   solution = solve_structured(
       enrich,
       historical_tickets,
       tol={"reserve": 25.0},
       alpha=0.1,
   )
   output = solution(new_ticket)

Categorical fields become calibrated label solvers. Numeric fields become
calibrated regressors and require a tolerance. The structured solution answers
locally only when every field answers locally, so one uncertain field escalates
the whole dictionary. Use this for enrichers, triagers, quote builders, and
metadata-producing services where field coherence matters.

Distill a Student
-----------------

``distill`` asks the teacher for labels and trains a local model:

.. code-block:: python

   from mixle.task import distill

   student = distill(
       teacher,
       train_texts,
       n=4,
       dim=512,
       hidden=[64],
       epochs=250,
       seed=0,
       task="spam vs ham",
   )

The result is a :class:`mixle.task.TaskModel`. It can be saved, loaded in a
fresh process, and called as a function.

.. code-block:: python

   student.save("spam_student")

   from mixle.task import TaskModel

   local = TaskModel.load("spam_student")
   print(local("free prize click now"))

Torch Representation Distillation
---------------------------------

The label-distillation path above asks a teacher for outputs and trains a
small task artifact. ``mixle.task.distill_methods`` covers classic
Torch-to-Torch knowledge distillation when you already have a trained teacher
module and an untrained student module.

.. code-block:: python

   from mixle.task.distill_methods import response_distill

   result = response_distill(
       student,
       teacher,
       x_train,
       y_train,
       temperature=4.0,
       alpha=0.9,
       epochs=300,
       seed=0,
   )

   print(result.metric, result.before, result.after, result.improved)

Available methods include:

``response_distill``
    Hinton-style soft-target response distillation, optionally mixed with hard
    labels.

``multi_teacher_distill``
    Soft-target distillation from an averaged or weighted teacher ensemble.

``hint_distill``
    FitNets-style feature matching through intermediate-layer hooks.

``attention_transfer``
    Spatial attention-map transfer between teacher and student layers.

``relational_distill``
    Batch-relationship distillation through distances and angles in feature
    space.

``sequence_level_distill``
    Sequence-level distillation for small language-model students.

These methods return ``DistillResult`` records with before/after fidelity
numbers and a training-loss history. They require Torch and are not task
``Solution`` objects; use them to train or compress modules before wrapping the
result in a Mixle model, skill, or service boundary.

Generative Text Students
------------------------

``distill_text_generative`` trains a small generative text classifier instead
of a discriminative hashed-feature student. The teacher still supplies labels,
but the student fits one token model per class plus class priors.

.. code-block:: python

   from mixle.task import distill_text_generative

   student = distill_text_generative(
       teacher,
       train_texts,
       labels=["spam", "ham"],
       min_count=2,
       task="spam vs ham",
   )
   label = student("free prize click now")

Use this when the local model should own both ``P(label | text)`` and a
typicality score for the text itself. The adapter is ``GenerativeTextIO``:
``proba_batch`` computes class posteriors from class-conditional token
likelihoods, and ``log_evidence`` reports length-normalized ``log p(text)``
for density-style checks. ``distill_text_generative_from_labels`` is the same
training core when labels have already been collected.

Tune the Recipe
---------------

``tune_recipe`` uses ``mixle.doe`` to search for lower-cost student settings
that still match the teacher well.

.. code-block:: python

   from mixle.task import tune_recipe

   tuned = tune_recipe(
       teacher,
       train_texts,
       validation_texts,
       n_init=4,
       n_iter=6,
       cost_weight=0.5,
       seed=0,
   )
   print(tuned.recipe, tuned.agreement, tuned.cost)

Use this when local training cost matters or when you want a principled small
model before deployment.

DOE-Guided Label Batches
------------------------

``mixle.doe`` can choose which examples deserve teacher calls before a
distillation round. This is useful when a pool spans several tasks, modalities,
or acquisition costs.

.. code-block:: python

   from mixle.doe import distillation_design

   design = distillation_design(
       embeddings,
       n=40,
       task_labels=task_names,
       modalities=modalities,
       uncertainty=student_uncertainty,
       cost=teacher_cost,
       seed=0,
   )

   selected = [unlabeled_pool[i] for i in design.indices]
   labels = [teacher(x) for x in selected]

Use :doc:`doe` for multi-task and cross-modal selectors. Use this page for the
teacher wrapper, student artifact, calibration, cascade, and harvest/retrain
loop after the design has chosen the batch.

For a multi-task pool, keep the design record next to the label record. A
release artifact should be able to answer three questions: which rows were
eligible, which rows were selected, and which task or modality coverage target
caused the selection. ``DistillationDesign.indices`` gives the chosen rows,
``task_counts`` and ``modality_counts`` describe the selected batch, and
``metadata`` records the requested targets and scoring weights.

Do not collapse missing modalities into ordinary numeric values. The
cross-modal selector treats a non-finite row in one modality as "this view is
missing" for eligibility, while finite ranking vectors such as uncertainty and
cost remain required. Preserve that distinction in notebooks and manifests.

Active Labeling
---------------

``active_distill`` spends the label budget on examples the current student most
needs.

.. code-block:: python

   from mixle.task import active_distill

   active = active_distill(
       teacher,
       unlabeled_pool,
       budget=60,
       seed_size=20,
       rounds=4,
       acquisition="margin",
       recipe={"n": 4, "dim": 512, "hidden": [64], "epochs": 200, "lr": 1e-2},
   )

Compare ``acquisition="margin"`` with ``"random"`` to quantify how much active
labeling helped on your pool.

Active labeling should keep its acquisition trace. Store the selected example
ids, acquisition scores, round number, teacher output, and student version that
requested each label. Those records explain why the labeled set is not an iid
sample and make later calibration or audit work much easier.

Calibrate Answer Sets
---------------------

Raw softmax confidence is not a guarantee. ``CalibratedTaskModel`` learns a
conformal threshold from held-out teacher labels. Its decision rule is:

.. list-table::
   :header-rows: 1

   * - Conformal set
     - Decision
   * - one label
     - answer locally
   * - empty set
     - escalate
   * - multiple labels
     - escalate
   * - one label but density gate says OOD
     - escalate

.. code-block:: python

   from mixle.task import CalibratedTaskModel

   calibrated = CalibratedTaskModel(active.model, alpha=0.1).calibrate(
       calibration_texts,
       teacher(calibration_texts),
   )

``alpha=0.1`` targets 90% marginal coverage for the conformal label sets on
exchangeable data.

Add an OOD Density Gate
-----------------------

A peaked softmax does not prove that an input is near the training
distribution. ``DensityGate`` adds a generative check over features.

.. code-block:: python

   from mixle.task import DensityGate, HashedNGram

   gate = DensityGate(HashedNGram(n=3, dim=48, seed=1)).fit(
       train_texts,
       n_components=3,
       seed=0,
   )
   calibrated = CalibratedTaskModel(active.model, alpha=0.1, density_gate=gate).calibrate(
       calibration_texts,
       teacher(calibration_texts),
   )

Serve a Cascade
---------------

``Cascade`` is the deployed object: call the local model when calibrated, call
the teacher when not, and record the economics.

.. code-block:: python

   from mixle.task import Cascade, CostModel

   cascade = Cascade(
       calibrated,
       teacher,
       cost=CostModel(c_frontier=0.01, c_local=0.00001),
   )

   predictions = cascade.serve(requests)
   report = cascade.report()
   print(report["realized_escalation_rate"])
   print(report["savings_vs_frontier"])

``report`` is based on actual served traffic, not a projection.

Harvest and Retrain
-------------------

Every escalation is an example the local model could not safely answer, and the
teacher has supplied the answer. Harvest those labels:

.. code-block:: python

   hard_texts, hard_labels = cascade.harvested()

Add them to the next distillation run. This closes the loop: the cascade should
reduce escalation cost as it sees the cases it previously escalated.

Promotion Gates
---------------

Do not promote a newly distilled student only because it trained successfully.
Use a held-out or traffic-shadow scorecard and require:

* no regression in agreement with the teacher on representative traffic;
* acceptable calibration-set coverage at the chosen ``alpha``;
* escalation behavior that is explainable by ambiguity or OOD gates;
* segment checks for rare labels, expensive teachers, and high-risk request
  classes;
* a reload check for the saved ``TaskModel`` or ``Solution`` artifact.

If a new student improves average agreement but loses a rare label, keep the
older route or escalate that segment until enough labels have been collected.

Capability Profiles and Regression Guards
-----------------------------------------

Agreement with the teacher is necessary but not sufficient. A student can
match a clean holdout set while failing under typos, harmless whitespace
changes, missing fields, or rare task tags. Use capability profiles to record
that behavior explicitly.

.. code-block:: python

   from mixle.task import (
       CapabilitySuite,
       capture_profile,
       case_jitter_invariance,
       keyboard_typo_corruption,
       whitespace_invariance,
   )

   suite = CapabilitySuite(
       corruptions={"typo_05": keyboard_typo_corruption(0.05, seed=0)},
       invariances={
           "case": case_jitter_invariance,
           "space": whitespace_invariance,
       },
       probes=["", "refund order A-102", "unseen jargon"],
   )
   profile = capture_profile(student, teacher, heldout_texts, suite)

The profile reports clean agreement, corruption agreement, invariance
violation rates for both student and teacher, fixed-probe predictions, and
abstention rates when the model exposes ``decide`` or ``batch_decide``. It
does not return a single aggregate score; the release gate should state which
profile fields matter for the task.

For extraction students, use ``extractive_capture_profile``. It measures
field-level F1 against fixed teacher extractions and records schema validity
instead of treating an entire dict as one exact-match label.

Use disagreement and collapse checks when iterating:

* ``fit_disagreement_gate`` models where the current student differs from the
  teacher, which is useful for targeted labeling and routing;
* ``collapse_monitor`` tracks whether iterative improvement lost score or
  diversity;
* segment scorecards should cover rare labels, high-cost teachers, and
  examples selected by DOE because they were uncertain or under-covered.

These reports should travel with the artifact. They explain what changed
between training rounds and prevent a better average score from hiding a
weaker operational behavior.

Extraction Tasks
----------------

The same teacher/student pattern works for structured extraction. The LLM emits
fields; the student learns a local sequence tagger.

.. code-block:: python

   from mixle.task import CallableLLM, distill_extractor, llm_extractor

   fields = ["id", "amount", "date", "vendor"]
   teacher = llm_extractor(CallableLLM(generate), fields)
   extractor = distill_extractor(teacher, invoice_lines, fields, epochs=150)

   print(extractor("INV-1234 Acme charged $19.95 on 2026-07-01"))

Agentic Tasks
-------------

When the teacher emits tool calls or multi-step plans rather than labels, use
:doc:`agentic-task-distillation`. That guide covers ``ToolSpec``,
``distill_tool_caller``, ``distill_planner``, ``sft_planner``,
``GenerativePlanner``, and ``harvest_agent_traces``.

Run the Examples
----------------

These scripts are examples, not release certification. Treat a passing example
as a smoke check and keep task-specific scorecards for any real deployment.

.. code-block:: sh

   python examples/task_distill_example.py
   python examples/task_llm_active_example.py
   python examples/task_cascade_economics_example.py
   python examples/task_extraction_example.py

API Map
-------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - ``TaskModel``
     - durable local model artifact
   * - ``CallableLLM`` / ``OpenAICompatLLM``
     - teacher adapters
   * - ``llm_labeler``
     - constrained text-to-label teacher
   * - ``distill`` / ``distill_from_labels``
     - train local classifiers
   * - ``distill_text_generative`` / ``distill_text_generative_from_labels``
     - train generative text students with class-conditional token models
   * - ``solve`` / ``solve_regression`` / ``solve_multilabel`` /
       ``solve_structured``
     - one-call replacement for label, numeric, multi-label, and dict-valued
       task functions
   * - ``active_distill``
     - spend label budget on informative examples
   * - ``CalibratedTaskModel``
     - conformal label sets and answer/escalate decisions
   * - ``DensityGate``
     - OOD escalation based on generative density
   * - ``Cascade``
     - serving wrapper with escalation, spend, and harvest
   * - ``CostModel``
     - per-request economics and route planning
   * - ``llm_extractor`` / ``distill_extractor``
     - teacher/student extraction pipeline

Detailed Task Inventory
-----------------------

.. list-table::
   :header-rows: 1

   * - Area
     - Imports
     - Notes
   * - Student payloads
     - ``TextClassifierIO``, ``RecordClassifierIO``, ``StructuredClassifierIO``,
       ``HashedNGram``, ``HashedRecord``
     - Feature adapters and payload classes used by ``TaskModel``.
   * - Adapter registry
     - ``register_adapter``, ``adapter_from_spec``
     - Add a new student adapter type.
   * - Label and record distillation
     - ``distill_records``, ``distill_records_from_labels``,
       ``distill_structured``, ``distill_structured_from_labels``
     - Use when the input is structured rather than plain text.
   * - Generative text students
     - ``GenerativeTextIO``, ``distill_text_generative``,
       ``distill_text_generative_from_labels``
     - Per-class token models with posterior probabilities and text evidence.
   * - One-call task replacement
     - ``Solution``, ``RegressionSolution``, ``MultiLabelSolution``,
       ``StructuredSolution``
     - Calibrated answer-or-escalate wrappers for common task shapes.
   * - Active learning internals
     - ``ActiveResult``, ``acquisition_scores``
     - Inspect active-labeling rounds and scoring.
   * - Capability profiles and disagreement
     - ``CapabilitySuite``, ``capture_profile``, ``extractive_capture_profile``,
       ``fit_disagreement_gate``, ``collapse_monitor``
     - Behavioral checks for corruptions, invariances, schema validity,
       disagreement, and iterative collapse.
   * - Recipe tuning
     - ``RecipeSpace``, ``TuneResult``
     - DOE-backed search over student settings.
   * - Extraction internals
     - ``ExtractionIO``, ``tokenize``, ``extraction_f1``
     - Sequence-tagger IO, tokenization, and evaluation.
   * - Model recommendation
     - ``ModelRecommendation``, ``FieldChoice``, ``DesignedModel``,
       ``spec_to_estimator``
     - Recommendation and LLM-designed model records.
   * - Serving records
     - ``CascadeStats``, ``RouterStats``, ``Scorecard``
     - Operational reports and evaluation summaries.
   * - Artifacts
     - ``get_builder``, ``get_arrays_builder``, ``load_harvested``
     - Builder lookup and harvested escalation IO.
   * - Harnesses
     - ``ExtractorHarness``, ``MatcherHarness``
     - Replacement wrappers for legacy extraction and matching code.
   * - Edge search
     - ``EdgeDistillResult``, ``task_fingerprint``, ``FINGERPRINT_KEYS``
     - Edge search outputs and task-shape features.
   * - LLM utilities
     - ``pick_label``
     - Low-level label choice helper used by LLM teachers.

More Task Surfaces
------------------

This page covers the main distillation and cascade loop. See
:doc:`task-serving` for the production-facing task surfaces: ``solve``,
``solve_regression``, ``solve_multilabel``, ``solve_structured``, ``Router``,
``route_stack``, edge distillation, quantized students, LNS structured
classifiers, replacement harnesses, scorecards, artifact builders, and route
economics.
