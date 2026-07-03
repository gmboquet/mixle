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

``tune_recipe`` uses ``mixle.doe`` to search for cheaper student settings that
still match the teacher well.

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

A classifier cannot know that an input is far from the training distribution
just because its softmax is peaked. ``DensityGate`` adds a generative check over
features.

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
teacher just labeled it. Harvest those labels:

.. code-block:: python

   hard_texts, hard_labels = cascade.harvested()

Add them to the next distillation run. This closes the loop: the cascade should
get cheaper as it sees the cases it previously escalated.

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
